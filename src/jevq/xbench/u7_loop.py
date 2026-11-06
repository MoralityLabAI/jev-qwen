"""SPEC-U7 part L: a compact looped transformer for Campsite repair, trained from scratch.

Each grid cell is a token. The inputs are what the verifier and the puzzle show: whether the cell
holds a tree, the candidate's symbol there (T, X, C, or none for a ragged/missing cell), and the
row and column tent targets. Attention gets a learned per-head bias by relation bucket (self, one of
the 8 neighbours, same row, same column, other), so nothing depends on absolute position and the
same weights run on any grid size. The model predicts tent / no tent for every non-tree cell.

Arms (same width, same per-block readout head, same steps):
  LOOP  one 2-layer block, weight-tied, iterated T times (T_train 16), the input re-injected each
        iteration, supervised after every iteration;
  DEEP  16 untied 2-layer blocks, supervised after every block (compute-matched at T 16);
  FLAT  one 2-layer block applied once (parameter-matched).

Outputs: results/u7/data/*, results/u7/runs/<arm>-s<seed>/{ckpt.pt,train_log.jsonl,arm.json},
results/u7/eval/<set>/<arm>-s<seed>-T<t>.jsonl.
"""

from __future__ import annotations

import json
import math
import random
import time
from functools import lru_cache
from pathlib import Path

from ..config import ROOT
from . import u4_campsite as u4
from . import u5_campsite as u5

OUT = ROOT / "results" / "u7"
TRAIN_SEED, L7_SEED = 20261107, 20261108
N_TRAIN_PUZZLES, TRAIN_SHIFT = 8000, 1600       # std puzzles (3x5..6x6, 20% of them 6x6)
L7_SIZES = ((8, 26), (10, 26), (12, 26), (14, 26))  # puzzles planted per size; 15 items per type kept
L7_PER_TYPE = 15
MAX_SOLUTIONS = 4
REAL_UPSAMPLE = 10
SYMBOLS = {"T": 0, "X": 1, "C": 2}               # 3 = none (missing / ragged)
N_BUCKETS = 12
ARMS = ("LOOP", "DEEP", "FLAT")
T_TRAIN = 16
T_EVAL = (1, 2, 4, 8, 16, 32, 64, 128)
STEPS, BATCH, LR, WARMUP = 20000, 128, 1e-3, 500
D_MODEL, N_HEADS, D_FF = 128, 4, 512


# ---------------------------------------------------------------------------- data


def _held_out_hashes() -> set:
    """Every puzzle any registered U-series test touches, plus the U7 large set."""
    hashes = {t.hash for p in ("test", "u5test", "u5std") for t, _ in u4.puzzles(p)}
    hashes |= {t.hash for t, _ in u5.large_puzzles()}
    return hashes


def encode(item, solutions=None) -> dict:
    """Flatten one (puzzle, candidate) item to per-cell integer features."""
    task = item["task"]
    n, m = len(task["grid"]), len(task["grid"][0])
    cand = item["candidate"] if isinstance(item["candidate"], list) else []
    tree, sym = [], []
    for r in range(n):
        for c in range(m):
            tree.append(int(task["grid"][r][c] == "T"))
            row = cand[r] if r < len(cand) and isinstance(cand[r], list) else []
            sym.append(SYMBOLS.get(row[c], 3) if c < len(row) else 3)
    out = {"id": item["item_id"], "n": n, "m": m, "tree": tree, "sym": sym,
           "rows": list(task["row_constraints"]), "cols": list(task["col_constraints"])}
    if solutions is not None:
        out["sols"] = [[int(g[r][c] == "C") for r in range(n) for c in range(m)] for g in solutions]
    return out


def _solutions(task, gold) -> list:
    sols, _ = u4.camp().solve_candidates(task, max_candidates=MAX_SOLUTIONS)
    return sols or [gold]


def _items_for_puzzles(pool: str, puzzles) -> list[dict]:
    """The eight U4 candidate types plus the zero-tent copy (solve from nothing) for every puzzle."""
    verify = u4.camp().verify_candidate
    items = []
    for task, gold in puzzles:
        sols = _solutions(task, gold)
        for kind in u4.TYPES + ("zero_tent",):
            if kind == "zero_tent":
                cand = [list(row) for row in task.grid]
            else:
                cand = u4.make_candidate(task, gold, kind, random.Random(f"u7:{pool}:{task.task_id}:{kind}"))
                if cand is None or verify(task, cand)["official_pass"] != (kind == "correct"):
                    continue
            item = {"item_id": f"u7.{pool}.{task.task_id}.{kind}", "kind": kind, "task": task.runtime_payload(),
                    "candidate": cand, "gold": gold}
            items.append(encode(item, sols) | {"kind": kind})
    return items


def build_train() -> dict:
    """Train: fresh std puzzles (held-out hashes dropped) x 9 candidates, plus the SPEC-U6 realtrain
    proposals (x REAL_UPSAMPLE). Val: the U4 val pool's items and the realval proposals."""
    from . import u6_campsite as u6

    held = _held_out_hashes()
    tasks = [t for t in u4.camp().generate_unseen_tasks(N_TRAIN_PUZZLES, seed=TRAIN_SEED, include_size_shift=TRAIN_SHIFT)
             if t.hash not in held]
    puzzles = [(t, u4.camp().solve_candidates(t, max_candidates=1)[0][0]) for t in tasks]
    train = _items_for_puzzles("train", puzzles)
    real = [encode(i, _solutions(u4.task_of(i), i["gold"])) | {"kind": "real"} for i in u6.items("realtrain") if i["kind"] == "std"]
    val = [encode(i, _solutions(u4.task_of(i), i["gold"])) | {"kind": i["kind"]} for i in u4.build_items("val")]
    val += [encode(i, _solutions(u4.task_of(i), i["gold"])) | {"kind": "real"} for i in u6.items("realval") if i["kind"] == "std"]
    data = {"train": train, "real": real, "val": val, "n_puzzles": len(puzzles), "dropped_held_out": N_TRAIN_PUZZLES - len(tasks)}
    path = OUT / "data"
    path.mkdir(parents=True, exist_ok=True)
    (path / "train.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def plant_reverse(n: int, rng: random.Random, task_id: str):
    """(task, planted solution) for an n x n puzzle, built tents-first: non-touching tents at random,
    then one tree on a free side of each. Used above 10x10, where hermes-lite's tree-first
    backtracking planter did not finish (15 min for 26 puzzles at 12x12). Checked with the verifier."""
    camp = u4.camp()
    target = int(n * n * 0.2)
    while True:
        cells = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(cells)
        tents = []
        for r, c in cells:
            if all(max(abs(r - a), abs(c - b)) > 1 for a, b in tents):
                tents.append((r, c))
                if len(tents) == target:
                    break
        if len(tents) < target:
            continue
        taken, trees = set(tents), []
        ok = True
        for r, c in tents:
            free = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= r + dr < n and 0 <= c + dc < n and (r + dr, c + dc) not in taken]
            if not free:
                ok = False
                break
            tree = rng.choice(free)
            taken.add(tree)
            trees.append(tree)
        if not ok:
            continue
        grid = [["X"] * n for _ in range(n)]
        for r, c in trees:
            grid[r][c] = "T"
        solution = [row[:] for row in grid]
        for r, c in tents:
            solution[r][c] = "C"
        rows = [sum(x == "C" for x in row) for row in solution]
        cols = [sum(solution[r][c] == "C" for r in range(n)) for c in range(n)]
        task = camp.CampsiteTask.from_payload({"task_id": task_id, "grid": grid, "row_constraints": rows, "col_constraints": cols})
        if camp.verify_candidate(task, solution)["official_pass"]:
            return task, solution


@lru_cache(maxsize=None)
def l7_items() -> tuple:
    """Large synthetic set: 8x8 and 10x10 (hermes-lite planting) and 12x12 and 14x14 (tents-first
    planting), the eight U4 candidate types, 15 items each."""
    exclude = frozenset(_held_out_hashes())
    puzzles = list(u5.plant_large(L7_SEED, L7_SIZES[:2], "u7l7", exclude))
    rng = random.Random(L7_SEED)
    for size, count in L7_SIZES[2:]:
        seen = set(exclude)
        made = 0
        while made < count:
            task, gold = plant_reverse(size, rng, f"u7l7_{size}x{size}_{made:03d}")
            if task.hash in seen:
                continue
            seen.add(task.hash)
            puzzles.append((task, gold))
            made += 1
    verify = u4.camp().verify_candidate
    items = []
    for size, _ in L7_SIZES:
        sized = [(t, g) for t, g in puzzles if len(t.grid) == size]
        for kind in u4.TYPES:
            taken = 0
            for task, gold in sized:
                if taken >= L7_PER_TYPE:
                    break
                cand = u4.make_candidate(task, gold, kind, random.Random(f"u7-l7:{task.task_id}:{kind}"))
                if cand is None or verify(task, cand)["official_pass"] != (kind == "correct"):
                    continue
                items.append({"item_id": f"u7.l7.{task.task_id}.{kind}", "pool": "l7", "kind": f"{size}x{size}", "type": kind,
                              "task": task.runtime_payload(), "candidate": cand, "gold": gold})
                taken += 1
            if taken < L7_PER_TYPE:
                raise RuntimeError(f"only {taken} l7 items of type {kind} at {size}x{size}")
    return tuple(items)


u5.EXTRA_SETS["l7"] = (l7_items, OUT)


def eval_items(set_name: str) -> list[dict]:
    return list(l7_items()) if set_name == "l7" else u5.items_for(set_name)


def load_data() -> dict:
    return json.loads((OUT / "data" / "train.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- model


def buckets(n: int, m: int):
    import torch

    r = torch.arange(n * m) // m
    c = torch.arange(n * m) % m
    dr = r[None, :] - r[:, None]
    dc = c[None, :] - c[:, None]
    out = torch.full((n * m, n * m), 11, dtype=torch.long)
    out[dc == 0] = 10
    out[dr == 0] = 9
    near = (dr.abs() <= 1) & (dc.abs() <= 1)
    offset = (dr + 1) * 3 + (dc + 1)               # 0..8, 4 = self
    neighbour = torch.where(offset > 4, offset, offset + 1)  # 1..8 for the eight neighbours
    out[near] = neighbour[near]
    out[(dr == 0) & (dc == 0)] = 0
    return out


def batch_tensors(examples: list[dict], device):
    """Pad a list of encoded examples to one batch: features, buckets, mask, solution targets."""
    import torch

    L = max(e["n"] * e["m"] for e in examples)
    B = len(examples)
    tree = torch.zeros(B, L, dtype=torch.long)
    sym = torch.full((B, L), 3, dtype=torch.long)
    rowt = torch.zeros(B, L, dtype=torch.long)
    colt = torch.zeros(B, L, dtype=torch.long)
    bucket = torch.full((B, L, L), 11, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    S = max((len(e.get("sols", [[0]])) for e in examples), default=1)
    sols = torch.zeros(B, S, L, dtype=torch.long)
    sol_mask = torch.zeros(B, S, dtype=torch.bool)
    for b, e in enumerate(examples):
        k = e["n"] * e["m"]
        tree[b, :k] = torch.tensor(e["tree"])
        sym[b, :k] = torch.tensor(e["sym"])
        rowt[b, :k] = torch.tensor([min(e["rows"][i // e["m"]], 15) for i in range(k)])
        colt[b, :k] = torch.tensor([min(e["cols"][i % e["m"]], 15) for i in range(k)])
        bucket[b, :k, :k] = buckets(e["n"], e["m"])
        mask[b, :k] = True
        for s, sol in enumerate(e.get("sols", [])):
            sols[b, s, :k] = torch.tensor(sol)
            sol_mask[b, s] = True
    return {k: v.to(device) for k, v in dict(tree=tree, sym=sym, rowt=rowt, colt=colt, bucket=bucket, mask=mask,
                                                 sols=sols, sol_mask=sol_mask).items()}


def make_model(arm: str):
    import torch
    from torch import nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
            self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
            self.proj = nn.Linear(D_MODEL, D_MODEL)
            self.bias = nn.Parameter(torch.zeros(N_HEADS, N_BUCKETS))
            self.ff = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Linear(D_FF, D_MODEL))

        def forward(self, h, bucket, mask):
            B, L, _ = h.shape
            q, k, v = self.qkv(self.ln1(h)).view(B, L, 3, N_HEADS, D_MODEL // N_HEADS).permute(2, 0, 3, 1, 4)
            scores = q @ k.transpose(-1, -2) / math.sqrt(D_MODEL // N_HEADS)
            scores = scores + self.bias[:, bucket].permute(1, 0, 2, 3)
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
            h = h + self.proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D_MODEL))
            return h + self.ff(self.ln2(h))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

        def forward(self, h, bucket, mask):
            for layer in self.layers:
                h = layer(h, bucket, mask)
            return h

    class CampsiteNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e_tree, self.e_sym = nn.Embedding(2, D_MODEL), nn.Embedding(4, D_MODEL)
            self.e_row, self.e_col = nn.Embedding(16, D_MODEL), nn.Embedding(16, D_MODEL)
            self.blocks = nn.ModuleList([Block() for _ in range(16 if arm == "DEEP" else 1)])
            self.ln = nn.LayerNorm(D_MODEL)
            self.head = nn.Linear(D_MODEL, 2)

        def forward(self, x, iterations: int):
            """Logits after every block application: a list of [B, L, 2]."""
            e = self.e_tree(x["tree"]) + self.e_sym(x["sym"]) + self.e_row(x["rowt"]) + self.e_col(x["colt"])
            h, outs = e, []
            steps = len(self.blocks) if arm == "DEEP" else (1 if arm == "FLAT" else iterations)
            for t in range(steps):
                block = self.blocks[t] if arm == "DEEP" else self.blocks[0]
                h = block(h + e if t else h, x["bucket"], x["mask"])
                outs.append(self.head(self.ln(h)))
            return outs

    return CampsiteNet()


def loss_fn(outs, x):
    """Mean over supervised steps of the per-example min (over solutions) cell cross-entropy on non-tree cells."""
    import torch

    cells = (x["mask"] & (x["tree"] == 0)).float()
    total = 0.0
    for logits in outs:
        logp = logits.float().log_softmax(-1)                                   # B, L, 2
        per_sol = []
        for s in range(x["sols"].shape[1]):
            ce = -logp.gather(-1, x["sols"][:, s, :].unsqueeze(-1)).squeeze(-1)  # B, L
            per_sol.append((ce * cells).sum(-1) / cells.sum(-1).clamp(min=1))
        per_sol = torch.stack(per_sol, 1).masked_fill(~x["sol_mask"], float("inf"))
        total = total + per_sol.min(1).values.mean()
    return total / len(outs)


def decode(logits, x, examples) -> list:
    """Grid per example from one step's logits: trees kept, tent where predicted on non-tree cells."""
    pred = logits.argmax(-1).cpu()
    grids = []
    for b, e in enumerate(examples):
        n, m = e["n"], e["m"]
        grids.append([["T" if e["tree"][r * m + c] else ("C" if int(pred[b, r * m + c]) == 1 else "X") for c in range(m)] for r in range(n)])
    return grids


# ---------------------------------------------------------------------------- training


def run_dir(arm: str, seed: int) -> Path:
    return OUT / "runs" / f"{arm}-s{seed}"


def train(arm: str, seed: int, steps: int = STEPS, device: str | None = None) -> dict:
    """Train one arm; resumes from ckpt.pt. Same data order and steps for every arm at a seed."""
    import torch

    torch.set_num_threads(4)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data()
    pool = data["train"] + data["real"] * REAL_UPSAMPLE
    out = run_dir(arm, seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = make_model(arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    start, ckpt = 0, out / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"]
    rng = random.Random(f"u7-order:{seed}")
    order = [rng.randrange(len(pool)) for _ in range(steps * BATCH)]
    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    t0 = time.perf_counter()
    model.train()
    for step in range(start, steps):
        lr = LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = [pool[i] for i in order[step * BATCH:(step + 1) * BATCH]]
        x = batch_tensors(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            outs = model(x, T_TRAIN)
        loss = loss_fn(outs, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 200 == 0 or step + 1 == steps:
            log.write(json.dumps({"step": step + 1, "loss": float(loss), "lr": lr, "seconds": time.perf_counter() - t0}) + "\n")
            log.flush()
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, ckpt)
    log.close()
    info = {"arm": arm, "seed": seed, "params": sum(p.numel() for p in model.parameters()), "steps": steps,
            "train_examples": len(data["train"]), "real_examples": len(data["real"]), "real_upsample": REAL_UPSAMPLE}
    (out / "arm.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ---------------------------------------------------------------------------- evaluation


def evaluate(arm: str, seed: int, sets=("u4test", "proposals", "l7", "u5test"), device: str | None = None) -> None:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(arm).to(device)
    state = torch.load(run_dir(arm, seed) / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    iterations = T_EVAL if arm == "LOOP" else (16 if arm == "DEEP" else 1,)
    for set_name in sets:
        items = eval_items(set_name)
        enc = [encode(i) for i in items]
        rows = {t: [] for t in iterations}
        for start in range(0, len(items), 64):
            chunk, chunk_items = enc[start:start + 64], items[start:start + 64]
            x = batch_tensors(chunk, device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                outs = model(x, max(iterations))
            for t in iterations:
                for item, grid in zip(chunk_items, decode(outs[t - 1] if arm == "LOOP" else outs[-1], x, chunk)):
                    rows[t].append(u5.repair_row(item, grid) | {"type": item.get("type", item["kind"])})
        dest = OUT / "eval" / set_name
        dest.mkdir(parents=True, exist_ok=True)
        for t, r in rows.items():
            (dest / f"{arm}-s{seed}-T{t}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in r), encoding="utf-8")


def run_cpu() -> None:
    """Build the training data and the L7 set; run the repair modules and the CSP solver on L7."""
    data = build_train()
    items = list(l7_items())
    u5.compute_projections("l7")
    camp = u4.camp()
    rows = {"identity": [], "c_repair": [], "dual_repair": [], "CSP-resolve": []}
    proj = u5.projections("l7")
    for item in items:
        rows["identity"].append(u5.repair_row(item, item["candidate"]) | {"type": item["type"]})
        for module in ("c_repair", "dual_repair"):
            p = proj[item["item_id"]][module]
            rows[module].append(u5.repair_row(item, p["grid"], status=p["status"], seconds=p["seconds"]) | {"type": item["type"]})
        t0 = time.perf_counter()
        solved, meta = camp.solve_candidates(u4.task_of(item), max_candidates=1)
        rows["CSP-resolve"].append(u5.repair_row(item, solved[0] if solved else None, seconds=time.perf_counter() - t0,
                                                 nodes=meta["nodes"]) | {"type": item["type"]})
    dest = OUT / "eval" / "l7"
    dest.mkdir(parents=True, exist_ok=True)
    for name, r in rows.items():
        (dest / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in r), encoding="utf-8")
    print(json.dumps({"train_examples": len(data["train"]), "real": len(data["real"]), "val": len(data["val"]),
                      "puzzles": data["n_puzzles"], "dropped_held_out": data["dropped_held_out"], "l7_items": len(items)}, indent=2))
