"""S1: the RMP task bundle (SPEC section 3).

Items are the first `per_depth` base examples (no minimal-pair siblings) per depth, in shard
order, from `validation` (depths 1-8; sudoku4 1-3) and `ood_stress` (depths 9-12). J-arms see a
text rendering of each item's `semantic` mapping; RMP arms see the shard's own tokens.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .foreign import LOOPED_TRANSFORMERS, RMP_ARTIFACTS, import_from
from .jrunner import ChoiceItem, CotItem

DATASETS = LOOPED_TRANSFORMERS / "experiments" / "recurrence_monitorability_v0" / "datasets"
FAMILIES = ("pointer_chase", "gated_pointer_chase", "masked_pointer_chase", "reachability", "sudoku4")
CORE4 = ("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4")
NODE = "ABCDEFGHIJKLMNOP"
SPECIAL = "X"  # REFUSE (gated) / ABSTAIN (masked)
N_SHOTS = 3

# Token constants of the RMP bundle (research_gym/interp/tasks.py, architecture_discovery/tasks.py).
POINTER_TARGET_BASE, SUDOKU_TARGET_BASE = 800, 928
REFUSE_TOKEN, ABSTAIN_TOKEN, REACH_NO, REACH_YES = 1100, 1101, 1104, 1105

RMP_CELLS = {
    # arm id: (artifact path relative to rmp_v0, RMP arm name)
    "LOOP-T-ds": ("artifacts/LOOP-T-core4-ds-s1/checkpoints/C3.pt", "LOOP-T"),
    "LOOP-T": ("snapshots/LOOP-T-core4-s1-final.pt", "LOOP-T"),
    "FF-U": ("snapshots/FF-U-core4-s1-final.pt", "FF-U"),
    "FF-U-ds": ("snapshots/FF-U-core4-ds-s1-final.pt", "FF-U"),
}


# ---------------------------------------------------------------------------- items


def shard_paths(family: str) -> list[Path]:
    paths = [DATASETS / "validation" / f"{family}.jsonl"]
    ood = DATASETS / "ood_stress" / f"{family}.jsonl"
    if ood.exists():
        paths.append(ood)
    return paths


def load_items(family: str, per_depth: int = 25) -> list[dict]:
    """Deterministic subset: first `per_depth` base rows per depth, validation then ood_stress."""
    taken: dict[int, int] = defaultdict(int)
    items = []
    for path in shard_paths(family):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["sibling"] is not None:
                    continue
                depth = int(row["difficulty"])
                if taken[depth] >= per_depth:
                    continue
                taken[depth] += 1
                items.append(row)
    return items


def gold_value(family: str, target_token: int) -> str:
    if family == "reachability":
        return {REACH_YES: "yes", REACH_NO: "no"}[target_token]
    if family == "sudoku4":
        return str(target_token - SUDOKU_TARGET_BASE)
    if target_token in (REFUSE_TOKEN, ABSTAIN_TOKEN):
        return SPECIAL
    return NODE[target_token - POINTER_TARGET_BASE]


def options_for(family: str) -> tuple[list[str], list[str]]:
    """(semantic options, single-token labels) for a family."""
    if family == "reachability":
        return ["yes", "no"], [" yes", " no"]
    if family == "sudoku4":
        return ["1", "2", "3", "4"], [" A", " B", " C", " D"]
    nodes = list(NODE) + ([SPECIAL] if family in ("gated_pointer_chase", "masked_pointer_chase") else [])
    return nodes, [" " + n for n in nodes]


def option_tokens(family: str) -> list[int]:
    """RMP vocabulary tokens aligned with `options_for(family)[0]`."""
    if family == "reachability":
        return [REACH_YES, REACH_NO]
    if family == "sudoku4":
        return [SUDOKU_TARGET_BASE + d for d in range(1, 5)]
    tokens = [POINTER_TARGET_BASE + i for i in range(16)]
    if family == "gated_pointer_chase":
        tokens.append(REFUSE_TOKEN)
    if family == "masked_pointer_chase":
        tokens.append(ABSTAIN_TOKEN)
    return tokens


# ---------------------------------------------------------------------------- text rendering

INSTRUCTIONS = {
    "pointer_chase": "Each node points to one node. Follow the pointers and answer with the node letter.",
    "gated_pointer_chase": (
        "Each node points to one node. A pointer may carry a lock (1, 2 or 3); you may follow a locked "
        "pointer only if you hold that grant. Answer with the node letter you reach, or X if the walk "
        "must refuse."
    ),
    "masked_pointer_chase": (
        "Each node points to one node; '?' means the pointer is unknown. Answer with the node letter you "
        "reach, or X if the walk needs an unknown pointer."
    ),
    "reachability": "Each node has up to two outgoing edges. Answer yes or no.",
    "sudoku4": (
        "Fill the 4x4 grid so that every row, column and 2x2 box holds 1, 2, 3 and 4 once. "
        "Choose the digit for the asked cell."
    ),
}


def render_question(family: str, semantic: dict) -> str:
    if family in ("pointer_chase", "gated_pointer_chase", "masked_pointer_chase"):
        parts = []
        for source, dest in enumerate(semantic["mapping"]):
            text = f"{NODE[source]}->{'?' if dest < 0 else NODE[dest]}"
            if family == "gated_pointer_chase" and semantic["locks"][source]:
                text += f" (lock {semantic['locks'][source]})"
            parts.append(text)
        lines = ["Map: " + ", ".join(parts) + "."]
        if family == "gated_pointer_chase":
            grants = ", ".join(str(g) for g in semantic["grants"]) or "none"
            lines.append(f"Grants held: {grants}.")
        lines.append(
            f"Start at {NODE[semantic['start']]} and follow the pointer {semantic['hops']} times. Where do you end?"
        )
        return "\n".join(lines)
    if family == "reachability":
        edges = [f"{NODE[node]}->{NODE[dest]}" for node, row in enumerate(semantic["edges"]) for dest in row if dest >= 0]
        return (
            "Edges: " + ", ".join(edges) + ".\n"
            f"Can you reach {NODE[semantic['goal']]} from {NODE[semantic['start']]} in at most "
            f"{semantic['horizon']} steps?"
        )
    if family == "sudoku4":
        puzzle = semantic["puzzle"]
        rows = [" ".join(str(v) if v else "." for v in puzzle[r * 4 : r * 4 + 4]) for r in range(4)]
        row, col = divmod(semantic["query"], 4)
        return "Grid:\n" + "\n".join(rows) + f"\nWhich digit goes in row {row + 1}, column {col + 1}?"
    raise ValueError(f"unknown family {family!r}")


def _choice_block(family: str, semantic: dict) -> list[str]:
    lines = [f"Q: {render_question(family, semantic)}"]
    if family == "sudoku4":
        lines += ["Options:", "A) 1", "B) 2", "C) 3", "D) 4"]
    return lines


def _answer_label(family: str, gold: str) -> str:
    options, labels = options_for(family)
    return labels[options.index(gold)].strip()


def render_choice_prompt(family: str, shots: list[dict], semantic: dict) -> str:
    lines = [INSTRUCTIONS[family], ""]
    for shot in shots:
        lines += _choice_block(family, shot["semantic"])
        lines += [f"Answer: {_answer_label(family, gold_value(family, shot['target_token']))}", ""]
    lines += _choice_block(family, semantic) + ["Answer:"]
    return "\n".join(lines)


def rationale(family: str, row: dict) -> str:
    """One-line worked solution from the item's ground-truth trace."""
    semantic, trace = row["semantic"], row["trace"]
    if family == "pointer_chase":
        return "->".join(NODE[n] for n in trace["cur_node"]) + "."
    if family == "gated_pointer_chase":
        path, locks, grants = trace["cur_node"], semantic["locks"], set(semantic["grants"])
        steps = []
        for node in path[:-1]:
            lock = locks[node]
            if lock and lock not in grants:
                steps.append(f"{NODE[node]} (lock {lock}, not held): refuse")
                return ", ".join(steps) + "."
            steps.append(f"{NODE[node]}" + (f" (lock {lock}, held)" if lock else ""))
        return ", ".join(steps) + f", end at {NODE[path[-1]]}."
    if family == "masked_pointer_chase":
        path = [n for n in trace["cur_node"] if n is not None]
        if gold_value(family, row["target_token"]) == SPECIAL:
            return "->".join(NODE[n] for n in path) + "->?: unknown, abstain."
        return "->".join(NODE[n] for n in path) + "."
    if family == "reachability":
        parts = [f"step {i}: {', '.join(NODE[n] for n in layer) or 'none'}" for i, layer in enumerate(trace["frontier"]) if i]
        found = gold_value(family, row["target_token"]) == "yes"
        return "; ".join(parts) + f"; {NODE[semantic['goal']]} {'reached' if found else 'not reached'}."
    if family == "sudoku4":
        history = trace["candidates"]
        single = {1: 1, 2: 2, 4: 3, 8: 4}
        parts = []
        for r in range(1, len(history)):
            new = [
                f"r{c // 4 + 1}c{c % 4 + 1}={single[history[r][c]]}"
                for c in range(16)
                if history[r][c] in single and history[r - 1][c] not in single
            ]
            if new:
                parts.append(f"round {r}: " + ", ".join(new))
            if history[r][semantic["query"]] in single:
                break
        return "; ".join(parts) + "."
    raise ValueError(family)


def render_cot_prompt(family: str, shots: list[dict], row: dict) -> str:
    lines = [INSTRUCTIONS[family] + " Write one line of reasoning, then the answer.", ""]
    for shot in shots:
        lines += [f"Q: {render_question(family, shot['semantic'])}", f"Reasoning: {rationale(family, shot)}"]
        lines += [f"A: {_cot_answer(family, gold_value(family, shot['target_token']))}", ""]
    lines += [f"Q: {render_question(family, row['semantic'])}", "Reasoning:"]
    return "\n".join(lines)


def _cot_answer(family: str, gold: str) -> str:
    return gold  # digits, yes/no, node letters and X are written as themselves


def train_shots(family: str, n: int = N_SHOTS, start_index: int = 0) -> list[dict]:
    """Shots from the RMP *train* hash region via the bundle's own generator (never eval rows)."""
    bundle = import_from(LOOPED_TRANSFORMERS, "research_gym.interp.bundle")
    config = rmp_bundle_config()
    shots, index = [], start_index
    depths_seen = set()
    while len(shots) < n:
        example = bundle.train_example(family, index, config).to_dict()
        index += 1
        # Spread the shots over depths, so the format is not tied to one difficulty.
        if example["difficulty"] in depths_seen and len(depths_seen) < 3:
            continue
        depths_seen.add(example["difficulty"])
        shots.append(example)
    return shots


def rmp_bundle_config():
    bundle = import_from(LOOPED_TRANSFORMERS, "research_gym.interp.bundle")
    with open(LOOPED_TRANSFORMERS / "configs" / "rmp_task_bundle_v0.json", "r", encoding="utf-8") as fh:
        return bundle.BundleConfig.from_mapping(json.load(fh))


def _meta(family: str, row: dict) -> dict:
    return {"family": family, "depth": int(row["difficulty"]), "ood": int(row["difficulty"]) > 8, "example_id": row["example_id"]}


def choice_items(family: str, per_depth: int = 25, shots: list[dict] | None = None) -> list[ChoiceItem]:
    shots = shots if shots is not None else train_shots(family)
    options, labels = options_for(family)
    out = []
    for row in load_items(family, per_depth):
        gold = gold_value(family, row["target_token"])
        meta = _meta(family, row)
        if family in ("gated_pointer_chase", "masked_pointer_chase"):
            meta["abstain_gold"] = gold == SPECIAL
        out.append(ChoiceItem(row["example_id"], render_choice_prompt(family, shots, row["semantic"]), labels, options, gold, meta))
    return out


def parse_cot(text: str) -> str:
    from ..tasks.base import cot_answer

    answer = cot_answer(text).strip().rstrip(".").strip()
    return answer.split()[0] if answer else ""


def cot_items(family: str, per_depth: int = 10, shots: list[dict] | None = None) -> list[CotItem]:
    from ..tasks.base import cot_done

    shots = shots if shots is not None else train_shots(family)
    out = []
    for row in load_items(family, per_depth):
        gold = gold_value(family, row["target_token"])
        meta = _meta(family, row)
        out.append(CotItem(row["example_id"], render_cot_prompt(family, shots, row), gold, parse_cot, cot_done, meta))
    return out


# ---------------------------------------------------------------------------- native RMP arms


def load_rmp_model(arm_id: str):
    import torch

    models = import_from(LOOPED_TRANSFORMERS, "research_gym.interp.models")
    relative, arm = RMP_CELLS[arm_id]
    path = RMP_ARTIFACTS / relative
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = models.build_model(arm, beta=1.0)
    model.load_state_dict(saved["model"] if "model" in saved else saved)
    model.eval()
    step = (saved.get("state") or {}).get("step") if isinstance(saved, dict) else None
    return model, path, step


def native_rows(arm_id: str, family: str, per_depth: int = 25, visits: int | None = None, batch: int = 200) -> tuple[list[dict], dict]:
    """Rows for an RMP arm; `iteration_preds` hold the per-visit lens prediction (options only)."""
    import time

    import torch

    model, path, step = load_rmp_model(arm_id)
    items = load_items(family, per_depth)
    options = options_for(family)[0]
    tokens = option_tokens(family)
    token_index = torch.tensor(tokens)
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            ids = torch.tensor([r["tokens"] for r in chunk], dtype=torch.long)
            t0 = time.perf_counter()
            output = model(ids, depth_visits=visits)
            elapsed = (time.perf_counter() - t0) / len(chunk)
            visit_logits = [model.lens(state[:, -1]) for state in output.visit_outputs]  # [visits] x [batch, vocab]
            final = visit_logits[-1]
            for i, row in enumerate(chunk):
                probs = torch.softmax(final[i, token_index], dim=-1).tolist()
                pred = options[max(range(len(probs)), key=probs.__getitem__)]
                exact = int(final[i].argmax()) == int(row["target_token"])
                gold = gold_value(family, row["target_token"])
                out = {
                    "item_id": row["example_id"],
                    "gold": gold,
                    "pred": pred,
                    "correct": pred == gold,
                    "exact_full_vocab": exact,
                    "options": options,
                    "probs": probs,
                    "passes": 1,
                    "block_iterations": len(visit_logits),
                    "emitted_tokens": 0,
                    "latency_s": elapsed,
                    "iteration_preds": [
                        options[int(v[i, token_index].argmax())] for v in visit_logits
                    ],
                    **_meta(family, row),
                }
                if family in ("gated_pointer_chase", "masked_pointer_chase"):
                    out["abstain_gold"] = gold == SPECIAL
                    out["abstain_pred"] = pred == SPECIAL
                rows.append(out)
    info = {
        "checkpoint": str(path),
        "checkpoint_step": step,
        "params_total": sum(p.numel() for p in model.parameters()),
        "visits": len(visit_logits),
    }
    return rows, info
