"""SPEC-U4: Campsite repair head-to-head, scored by hermes-lite's official verifier.

Puzzles, verifier and CSP solver: hermes-lite `agent.intellect3_logic`. Repair modules: Hermes-Skills
`c_only_projection` / `dual_signature_projection`, executed unchanged from the committed source.
Part D (deciders choose commit / c_repair / dual_repair / reject) writes results/u4/decide/<record>.jsonl;
Part G (repairers output a grid) writes results/u4/repair/<record>.jsonl.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import types
from functools import lru_cache

from ..config import ROOT
from .foreign import HERMES_LITE, HERMES_SKILLS, git_show, import_from
from .jrunner import ChoiceItem, CotItem

OUT = ROOT / "results" / "u4"
POOLS = {"test": (90, 18, 20261008), "train": (400, 80, 20261009), "val": (80, 16, 20261010),  # puzzles, 6x6, seed
         # SPEC-U5 pools: a fresh synthetic test set and the standard-size puzzles for model proposals.
         "u5test": (110, 22, 20261011), "u5std": (120, 24, 20261012)}
TEST_POOLS = ("test", "u5test")  # 60 items per type, seeded order
TEST_PER_TYPE = 60
TYPES = ("correct", "drop_tent", "extra_tent", "move_in_row", "move_any", "swap_rect", "tree_mutation", "bad_shape")
ACTIONS = ("commit", "c_repair", "dual_repair", "reject")
GATES = ("syntax_valid", "shape_match", "trees_unchanged", "row_counts_match", "col_counts_match", "no_tent_touching",
         "perfect_tree_matching")
REPAIR_SCRIPT = "research/scripts/run_intellect3_camp_gate_micro_env.py"
LETTERS = "ABCD"
MAX_NEW_TOKENS = 120


@lru_cache(maxsize=None)
def camp():
    return import_from(HERMES_LITE, "agent.intellect3_logic", subdir="src")


@lru_cache(maxsize=None)
def skills():
    """Hermes-Skills' micro-env script at HEAD, executed as a module (stdlib imports only)."""
    source = git_show(HERMES_SKILLS, REPAIR_SCRIPT)
    module = types.ModuleType("hermes_skills_camp_gate_micro_env")
    exec(compile(source, f"Hermes-Skills HEAD:{REPAIR_SCRIPT}", "exec"), module.__dict__)  # noqa: S102
    return module


def source_hash() -> str:
    return hashlib.sha256(git_show(HERMES_SKILLS, REPAIR_SCRIPT).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------- items


def _copy(grid):
    return [list(row) for row in grid]


def _cells(grid, symbol: str):
    return [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == symbol]


def _near_tree(task_grid, r: int, c: int) -> bool:
    n, m = len(task_grid), len(task_grid[0])
    return any(0 <= r + dr < n and 0 <= c + dc < m and task_grid[r + dr][c + dc] == "T" for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def make_candidate(task, gold, kind: str, rng: random.Random):
    g = _copy(gold)
    n, m = len(g), len(g[0])
    tents = _cells(g, "C")
    free = [(r, c) for r, c in _cells(g, "X") if _near_tree(task.grid, r, c)]
    if kind == "correct":
        return g
    if kind == "drop_tent":
        r, c = rng.choice(tents)
        g[r][c] = "X"
    elif kind == "extra_tent":
        if not free:
            return None
        r, c = rng.choice(free)
        g[r][c] = "C"
    elif kind == "move_in_row":
        r, c = rng.choice(tents)
        options = [(r, cc) for cc in range(m) if g[r][cc] == "X"]
        if not options:
            return None
        rr, cc = rng.choice(options)
        g[r][c], g[rr][cc] = "X", "C"
    elif kind == "move_any":
        r, c = rng.choice(tents)
        options = [(rr, cc) for rr, cc in free if rr != r and cc != c]
        if not options:
            return None
        rr, cc = rng.choice(options)
        g[r][c], g[rr][cc] = "X", "C"
    elif kind == "swap_rect":
        pairs = [(a, b) for a, b in itertools.combinations(tents, 2)
                 if a[0] != b[0] and a[1] != b[1] and g[a[0]][b[1]] == "X" and g[b[0]][a[1]] == "X"]
        if not pairs:
            return None
        (r1, c1), (r2, c2) = rng.choice(pairs)
        g[r1][c1], g[r2][c2], g[r1][c2], g[r2][c1] = "X", "X", "C", "C"
    elif kind == "tree_mutation":
        r, c = rng.choice(_cells(g, "T"))
        g[r][c] = "X"
    elif kind == "bad_shape":
        r = rng.randrange(n)
        g[r] = g[r][:-1]
    else:
        raise ValueError(kind)
    return g


def _check_expected(task, gold) -> None:
    """The projections read only the expected grid's shape and T/C counts: they must equal what the puzzle shows."""
    s = skills()
    if s.row_signature(gold, "C") != task.row_constraints or s.col_signature(gold, "C") != task.col_constraints:
        raise RuntimeError(f"{task.task_id}: gold C counts differ from the constraints")
    if _cells(gold, "T") != _cells(task.grid, "T"):
        raise RuntimeError(f"{task.task_id}: gold trees differ from the puzzle")


@lru_cache(maxsize=None)
def puzzles(pool: str):
    """The pool's puzzles with their CSP gold grids. Train and val drop any puzzle that is also in an
    earlier pool (test, then train), so the three pools are disjoint by puzzle hash."""
    count, shift, seed = POOLS[pool]
    earlier = list(POOLS)[: list(POOLS).index(pool)]
    taken = {task.hash for p in earlier for task, _ in puzzles(p)}
    out = []
    for task in camp().generate_unseen_tasks(count, seed=seed, include_size_shift=shift):
        if task.hash in taken:
            continue
        gold = camp().solve_candidates(task, max_candidates=1)[0][0]
        _check_expected(task, gold)
        out.append((task, gold))
    return out


@lru_cache(maxsize=None)
def build_items(pool: str) -> tuple:
    """Items as dicts (tuple for caching). Test: the first 60 eligible puzzles per type in a seeded order."""
    tasks = list(puzzles(pool))
    test_pool = pool in TEST_POOLS
    if test_pool:
        random.Random("u4-test-order" if pool == "test" else f"{pool}-order").shuffle(tasks)
    verify = camp().verify_candidate
    items = []
    for kind in TYPES:
        taken = 0
        for task, gold in tasks:
            if test_pool and taken >= TEST_PER_TYPE:
                break
            cand = make_candidate(task, gold, kind, random.Random(f"u4-items:{pool}:{task.task_id}:{kind}"))
            if cand is None:
                continue
            passes = verify(task, cand)["official_pass"]
            if passes != (kind == "correct"):
                continue
            items.append({"item_id": f"u4.{pool}.{kind}.{taken:03d}", "pool": pool, "kind": kind, "task": task.runtime_payload(),
                          "candidate": cand, "gold": gold})
            taken += 1
        if test_pool and taken < TEST_PER_TYPE:
            raise RuntimeError(f"only {taken} test items of type {kind}")
    return tuple(items)


def items_hash(pool: str = "test") -> str:
    return hashlib.sha256(json.dumps(list(build_items(pool)), sort_keys=True).encode("utf-8")).hexdigest()[:16]


def task_of(item):
    return camp().CampsiteTask.from_payload(item["task"])


def verify(item, grid) -> dict:
    return camp().verify_candidate(task_of(item), grid)


def rectangular(item, grid) -> bool:
    n, m = len(item["task"]["grid"]), len(item["task"]["grid"][0])
    return isinstance(grid, list) and len(grid) == n and all(isinstance(row, list) and len(row) == m for row in grid)


# ---------------------------------------------------------------------------- repair modules and outcomes


def project(item, module: str):
    """A Hermes-Skills projection, or None. Ragged grids count as 'no projection', matching its parse_grid gate."""
    if not rectangular(item, item["candidate"]):
        return None
    fn = skills().c_only_projection if module == "c_repair" else skills().dual_signature_projection
    return fn(item["candidate"], item["gold"])


def committed(item, action: str):
    if action == "reject":
        return None
    if action == "commit":
        return item["candidate"]
    out = project(item, action)
    return out if out is not None else item["candidate"]


def outcome(item, action: str, **extra) -> dict:
    grid = committed(item, action)
    passed = grid is not None and verify(item, grid)["official_pass"]
    return {"item_id": item["item_id"], "kind": item["kind"], "action": action, "success": passed,
            "unsafe": grid is not None and not passed, "rejected": grid is None, **extra}


def best_action(item) -> str:
    """Menu ceiling order: commit if the candidate passes, else c_repair, else dual_repair, else reject."""
    for action in ACTIONS[:3]:
        if outcome(item, action)["success"]:
            return action
    return "reject"


def hamming(a, b):
    if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return None
    return sum(x != y for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def repair_row(item, grid, **extra) -> dict:
    passed = grid is not None and verify(item, grid)["official_pass"]
    return {"item_id": item["item_id"], "kind": item["kind"], "success": passed, "parsed": grid is not None,
            "edits_to_candidate": hamming(grid, item["candidate"]), "edits_to_gold": hamming(grid, item["gold"]), **extra}


def signatures(item) -> tuple[bool, bool]:
    """Hermes-Skills' T and C signature passes for the candidate (False when it is not rectangular)."""
    if not rectangular(item, item["candidate"]):
        return False, False
    s = skills()
    return s.signature_pass(item["candidate"], item["gold"], "T"), s.signature_pass(item["candidate"], item["gold"], "C")


def flow_policy(item, name: str) -> str:
    t_ok, c_ok = signatures(item)
    if name == "c_repair_if_c_fail":
        return "commit" if c_ok else "c_repair"
    if name == "dual_repair_if_any_sig_fail":
        return "commit" if (t_ok and c_ok) else "dual_repair"
    if name == "always-commit":
        return "commit"
    raise ValueError(name)


# ---------------------------------------------------------------------------- Decision-TRM


def features(item) -> list[float]:
    v = verify(item, item["candidate"])
    vec = [float(v["gates"][g]) for g in GATES] + [float(v["official_pass"])]
    task = item["task"]
    n, m = len(task["grid"]), len(task["grid"][0])
    cand = item["candidate"]
    t_ok, c_ok = signatures(item)
    if rectangular(item, cand):
        rows = [sum(x == "C" for x in row) for row in cand]
        cols = [sum(cand[r][c] == "C" for r in range(n)) for c in range(m)]
        row_off = sum(a != b for a, b in zip(rows, task["row_constraints"])) / n
        col_off = sum(a != b for a, b in zip(cols, task["col_constraints"])) / m
        tent_diff = sum(rows) - sum(task["row_constraints"])
        tree_diff = len(_cells(cand, "T")) - len(_cells(task["grid"], "T"))
        shape = 1.0
    else:
        row_off = col_off = 1.0
        tent_diff = tree_diff = 0
        shape = 0.0
    clip = lambda x: max(-1.0, min(1.0, x / 3.0))  # noqa: E731
    return vec + [shape, row_off, col_off, clip(tent_diff), clip(tree_diff), float(t_ok), float(c_ok), n / 6.0, m / 6.0]


def train_decision_trm(train, val, seed: int = 0, hidden: int = 64, steps: int = 4, label=None):
    """`label(item)` gives the training action; default: the menu-ceiling order with live projections."""
    import torch

    label = label or best_action
    from torch import nn

    class DecisionTRM(nn.Module):
        """hermes-lite TinyRecursivePolicy's structure with a 4-way action head."""

        def __init__(self, dim: int):
            super().__init__()
            self.input_projection = nn.Linear(dim, hidden)
            self.recurrent = nn.Linear(hidden * 2, hidden)
            self.action_head = nn.Linear(hidden, len(ACTIONS))

        def forward(self, x):
            encoded = torch.tanh(self.input_projection(x))
            h = torch.zeros_like(encoded)
            for _ in range(steps):
                h = torch.tanh(self.recurrent(torch.cat([encoded, h], dim=-1)))
            return self.action_head(h)

    def tensors(items):
        return (torch.tensor([features(i) for i in items], dtype=torch.float32),
                torch.tensor([ACTIONS.index(label(i)) for i in items]))

    xt, yt = tensors(train)
    xv, yv = tensors(val)
    best = None
    for n_steps in (50, 200, 500, 1000):
        torch.manual_seed(seed)
        model = DecisionTRM(xt.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(n_steps):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(xt), yt).backward()
            optimizer.step()
        with torch.no_grad():
            acc = (model(xv).argmax(-1) == yv).float().mean().item()
        if best is None or acc > best[0]:
            best = (acc, n_steps, model)
    acc, n_steps, model = best
    model.eval()

    def predict(item) -> tuple[str, list[float]]:
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor([features(item)], dtype=torch.float32))[0], -1).tolist()
        return ACTIONS[max(range(len(probs)), key=probs.__getitem__)], probs

    return predict, {"params_total": sum(p.numel() for p in model.parameters()), "steps_selected": n_steps, "val_accuracy": acc}


# ---------------------------------------------------------------------------- Jev prompts

RULES = (
    "Campsite rules: every tree (T) stays where it is; each tree gets exactly one tent (C) directly above, below, "
    "left or right of it; tents never touch each other, not even diagonally; each row and each column holds the "
    "stated number of tents. X is an empty cell."
)
DECIDE_INSTRUCTION = (
    RULES + " A candidate grid has been checked by the verifier. Choose one action:\n"
    "commit: accept the candidate as it is.\n"
    "c_repair: move, add or remove tents with the fewest cell edits so that every row and column tent count "
    "matches; the trees are kept as they are in the candidate.\n"
    "dual_repair: like c_repair, and also restore the number of trees in every row and column with the fewest edits.\n"
    "reject: commit nothing.\n"
    "Choose the action that ends with a committed grid that passes every rule; if no action can, reject. "
    "Answer with the option letter."
)
REPAIR_INSTRUCTION = (
    RULES + " A candidate grid has been checked by the verifier. Repair it so that it passes every rule, changing as "
    "few cells as possible. Write only the repaired grid, one row per line, cells separated by spaces."
)


def grid_text(grid) -> str:
    return "\n".join(" ".join(row) for row in grid)


def case_text(item) -> str:
    task = item["task"]
    v = verify(item, item["candidate"])
    report = "all checks pass" if v["official_pass"] else "failed " + ", ".join(v["failed_gates"])
    counts = ""
    if "row_counts" in v["details"]:
        counts = (f"; candidate tents per row {' '.join(map(str, v['details']['row_counts']))}, "
                  f"per column {' '.join(map(str, v['details']['col_counts']))}")
    return "\n".join([
        "Puzzle:", grid_text(task["grid"]),
        f"Row tents: {' '.join(map(str, task['row_constraints']))}",
        f"Column tents: {' '.join(map(str, task['col_constraints']))}",
        "Candidate:", grid_text(item["candidate"]),
        f"Verifier: {report}{counts}",
    ])


def _options(item_id: str) -> list[str]:
    options = list(ACTIONS)
    random.Random(f"u4-options:{item_id}").shuffle(options)
    return options


def _decide_block(item) -> tuple[list[str], list[str]]:
    options = _options(item["item_id"])
    return [case_text(item), "Options:"] + [f"{l}) {a}" for l, a in zip(LETTERS, options)], options


@lru_cache(maxsize=None)
def decide_shots() -> tuple:
    """One train item per best action, of the type that shows it most plainly."""
    plain = {"commit": "correct", "c_repair": "drop_tent", "dual_repair": "tree_mutation", "reject": "swap_rect"}
    train = build_items("train")
    shots = []
    for action, kind in plain.items():
        shots.append(next(i for i in train if i["kind"] == kind and best_action(i) == action))
    random.Random("u4-decide-shots").shuffle(shots)
    return tuple(shots)


def decide_prompt(item) -> tuple[str, list[str]]:
    lines = [DECIDE_INSTRUCTION, ""]
    for shot in decide_shots():
        block, options = _decide_block(shot)
        lines += block + [f"Answer: {LETTERS[options.index(best_action(shot))]}", ""]
    block, options = _decide_block(item)
    return "\n".join(lines + block + ["Answer:"]), options


def decide_items(items) -> list[ChoiceItem]:
    out = []
    for item in items:
        prompt, options = decide_prompt(item)
        out.append(ChoiceItem(item["item_id"], prompt, [" " + l for l in LETTERS], options, "", {"kind": item["kind"]}))
    return out


@lru_cache(maxsize=None)
def repair_shots() -> tuple:
    train = build_items("train")
    return tuple(next(i for i in train if i["kind"] == kind) for kind in ("drop_tent", "tree_mutation", "move_any"))


def repair_prompt(item) -> str:
    lines = [REPAIR_INSTRUCTION, ""]
    for shot in repair_shots():
        lines += [case_text(shot), "Repaired:", grid_text(shot["gold"]), ""]
    return "\n".join(lines + [case_text(item), "Repaired:"]) + "\n"


def parse_grid(text: str, n_rows: int):
    rows = []
    for line in text.split("\n"):
        if not line.strip():
            if rows:
                break
            continue
        cells = line.split()
        if any(cell not in ("T", "X", "C") for cell in cells):
            break
        rows.append(cells)
        if len(rows) == n_rows:
            break
    return rows or None


def repair_items(items) -> list[CotItem]:
    out = []
    for item in items:
        n = len(item["task"]["grid"])
        out.append(CotItem(item["item_id"], repair_prompt(item), "", (lambda text, n=n: json.dumps(parse_grid(text, n))),
                           (lambda text, n=n: sum(1 for line in text.split("\n")[:-1] if line.strip()) >= n), {"kind": item["kind"]}))
    return out


# ---------------------------------------------------------------------------- runs


def _write(part: str, record: str, rows: list[dict], info: dict) -> None:
    out = OUT / part
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{record}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / f"{record}.json").write_text(json.dumps({"record": record, **info}, indent=2), encoding="utf-8")


def run_cpu() -> None:
    test = list(build_items("test"))
    base = {"items_hash": items_hash(), "repair_source_sha256_16": source_hash()}
    csp = {i["item_id"]: camp().solve_candidates(task_of(i), max_candidates=1)[0][0] for i in test}
    _write("repair", "identity", [repair_row(i, i["candidate"]) for i in test], {**base, "neural": False})
    for module in ("c_repair", "dual_repair"):
        _write("repair", module, [repair_row(i, project(i, module), projected=project(i, module) is not None) for i in test],
               {**base, "neural": False, "source": f"Hermes-Skills HEAD:{REPAIR_SCRIPT}"})
    _write("repair", "CSP-resolve", [repair_row(i, csp[i["item_id"]]) for i in test], {**base, "neural": False, "ceiling": True})
    for name in ("always-commit", "c_repair_if_c_fail", "dual_repair_if_any_sig_fail"):
        _write("decide", name, [outcome(i, flow_policy(i, name)) for i in test], {**base, "neural": False})
    _write("decide", "menu-ceiling", [outcome(i, best_action(i)) for i in test], {**base, "neural": False, "ceiling": True})
    _write("decide", "CSP-resolve", [{"item_id": i["item_id"], "kind": i["kind"], "action": "resolve", "success": verify(i, csp[i["item_id"]])["official_pass"],
                                      "unsafe": not verify(i, csp[i["item_id"]])["official_pass"], "rejected": False} for i in test],
           {**base, "neural": False, "ceiling": True})
    train, val = list(build_items("train")), list(build_items("val"))
    for seed in (0, 1, 2):
        predict, info = train_decision_trm(train, val, seed=seed)
        rows = []
        for i in test:
            action, probs = predict(i)
            rows.append(outcome(i, action, probs=probs))
        _write("decide", "Decision-TRM" if seed == 0 else f"Decision-TRM-s{seed}", rows,
               {**base, "neural": True, "seed": seed, "train_items": len(train), "val_items": len(val), **info})
    print(json.dumps({"test_items": len(test), "train_items": len(train), "val_items": len(val), **base}, indent=2))


def run_j(arm_id: str, seed: int = 0, iterations: int | None = None) -> None:
    from ..looped import LoopSpec
    from .arms import LOOP_SPAN, seeded
    from .jrunner import load_j_bundle, run_choice, run_cot

    arm = seeded(arm_id, seed)
    record = arm.arm_id + (f"-r{iterations}" if iterations else "")
    loop = LoopSpec(*LOOP_SPAN, iterations) if iterations else None
    bundle = load_j_bundle(arm.adapter)
    test = list(build_items("test"))
    by_id = {i["item_id"]: i for i in test}
    info = {"neural": True, "adapter": arm.adapter, "loop_iterations": iterations, "base_arm": arm_id, "seed": seed, "items_hash": items_hash()}
    rows = []
    for r in run_choice(bundle, decide_items(test), loop=loop):
        rows.append(outcome(by_id[r["item_id"]], r["pred"], probs=r["probs"], options=r["options"], label_mass=r["label_mass"],
                            latency_s=r["latency_s"]))
    _write("decide", record, rows, info)
    if loop is None:
        rows = []
        for r in run_cot(bundle, repair_items(test), max_new_tokens=MAX_NEW_TOKENS):
            rows.append(repair_row(by_id[r["item_id"]], json.loads(r["pred"]), raw_text=r["raw_text"], emitted_tokens=r["emitted_tokens"],
                                   hit_token_budget=r["hit_token_budget"], latency_s=r["latency_s"]))
        _write("repair", record, rows, info)
