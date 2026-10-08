"""SPEC-U6 part T: train on real model proposals. Real J-V0 / Bonsai-8B proposals on training puzzles
(the U4 train and val pools plus new 8x8 / 10x10 puzzles) become training items for J-real (J-u4
continued) and a Decision-TRM-real; both are tested on the SPEC-U5 proposal set, which stays held out.

Outputs: results/u6/proposals/<split>/<proposer>.jsonl, results/u6/projections/real{train,val}.jsonl,
results/u6/{decide,repair}/<set>/<record>.jsonl.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache

from ..config import ROOT
from . import u4_campsite as u4
from . import u5_campsite as u5

OUT = ROOT / "results" / "u6"
LARGE = {"realtrain": (20261014, ((8, 100), (10, 100))), "realval": (20261015, ((8, 20), (10, 20)))}
STD_POOL = {"realtrain": "train", "realval": "val"}
VAL_LIMIT = 100


@lru_cache(maxsize=None)
def puzzles(split: str) -> tuple:
    """Standard puzzles from the U4 pool of the same role, plus freshly planted large ones (disjoint
    from the U5 large puzzles and from each other)."""
    seed, sizes = LARGE[split]
    exclude = {t.hash for t, _ in u5.large_puzzles()}
    if split == "realval":
        exclude |= {t.hash for t, _ in puzzles("realtrain")}
    large = u5.plant_large(seed, sizes, f"{split}_large", frozenset(exclude))
    return tuple(list(u4.puzzles(STD_POOL[split])) + list(large))


def proposal_path(split: str, proposer: str):
    return OUT / "proposals" / split / f"{proposer}.jsonl"


def run_proposer(proposer: str) -> None:
    for split in ("realtrain", "realval"):
        u5.run_proposer(proposer, puzzles(split), proposal_path(split, proposer))


@lru_cache(maxsize=None)
def items(split: str) -> tuple:
    by_id = {task.task_id: (task, gold) for task, gold in puzzles(split)}
    out = []
    for proposer in u5.PROPOSERS:
        for row in map(json.loads, proposal_path(split, proposer).read_text(encoding="utf-8").splitlines()):
            task, gold = by_id[row["puzzle_id"]]
            item = {"item_id": f"u6.{split}.{proposer}.{task.task_id}", "pool": split, "kind": u5.band(task), "proposer": proposer,
                    "task": task.runtime_payload(), "candidate": row["grid"] or [], "gold": gold}
            item["proposal_pass"] = u4.verify(item, item["candidate"])["official_pass"]
            out.append(item)
    return tuple(out)


for _split in ("realtrain", "realval"):
    u5.EXTRA_SETS[_split] = ((lambda s=_split: items(s)), OUT)


def label(split: str | None = None):
    """Menu-ceiling label from the cached projections of the item's own split (its `pool`)."""
    return lambda item: u5.best_action(split or item["pool"], item)


def training_rows(split: str, limit: int | None = None):
    """(decide ChoiceItems, repair (prompt, target) pairs) from real proposals; labels use the cached projections."""
    rows = list(items(split))
    if limit is not None:
        rows = random.Random(f"u6-val-subset:{split}").sample(rows, min(limit, len(rows)))
    decide = u5.decide_items_zero(rows, gold_fn=label(split))
    repair = [(u5.repair_prompt_zero(i), u4.grid_text(i["gold"]) + "\n\n") for i in rows]
    return decide, repair


def _write(part: str, set_name: str, record: str, rows: list[dict], info: dict) -> None:
    out = OUT / part / set_name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{record}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / f"{record}.json").write_text(json.dumps({"record": record, "set": set_name, **info}, indent=2), encoding="utf-8")


def _projections_complete(split: str) -> bool:
    path = u5.projection_path(split)
    if not path.exists():
        return False
    return {r["item_id"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} == {i["item_id"] for i in items(split)}


def run_cpu() -> None:
    for split in ("realtrain", "realval"):
        if not _projections_complete(split):  # deterministic, so a complete cache from an earlier attempt is reused
            u5.compute_projections(split)
    train, val = list(items("realtrain")), list(items("realval"))
    test = u5.items_for("proposals")
    summary = {"train_items": len(train), "val_items": len(val),
               "train_labels": {a: sum(label()(i) == a for i in train) for a in u4.ACTIONS}}
    for seed in (0, 1, 2):
        predict, info = u4.train_decision_trm(train, val, seed=seed, label=label())
        rows = []
        for item in test:
            action, probs = predict(item)
            rows.append(u5.outcome("proposals", item, action, probs=probs))
        _write("decide", "proposals", "Decision-TRM-real" if seed == 0 else f"Decision-TRM-real-s{seed}", rows,
               {"neural": True, "seed": seed, "trained_on": "real proposals on train puzzles (SPEC-U6)", **info})
    (OUT / "cpu_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_j(arm_id: str = "J-real", seed: int = 0) -> None:
    """J-real on the held-out U5 proposals (decide + write) and the U4 test set (decide + write)."""
    from .arms import seeded
    from .jrunner import load_j_bundle, run_choice, run_cot

    arm = seeded(arm_id, seed)
    bundle = load_j_bundle(arm.adapter)
    info = {"neural": True, "adapter": arm.adapter, "base_arm": arm_id, "seed": seed, "prompt": "zero-shot (training format)"}
    for set_name in ("proposals", "u4test"):
        test = u5.items_for(set_name)
        by_id = {i["item_id"]: i for i in test}
        rows = []
        for r in run_choice(bundle, u5.decide_items_zero(test)):
            item = by_id[r["item_id"]]
            extra = {"probs": r["probs"], "options": r["options"], "label_mass": r["label_mass"]}
            rows.append(u4.outcome(item, r["pred"], **extra) if set_name == "u4test" else u5.outcome(set_name, item, r["pred"], **extra))
        _write("decide", set_name, arm.arm_id, rows, info)
        budget = max(2 * len(i["task"]["grid"]) * len(i["task"]["grid"][0]) + 16 for i in test)
        rows = []
        for r in run_cot(bundle, u5.repair_items_zero(test), max_new_tokens=budget):
            rows.append(u5.repair_row(by_id[r["item_id"]], json.loads(r["pred"]), raw_text=r["raw_text"], emitted_tokens=r["emitted_tokens"],
                                      latency_s=r["latency_s"]))
        _write("repair", set_name, arm.arm_id, rows, info)
