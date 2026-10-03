"""S3: RLM long-context control suite, eval split (24 tasks) (SPEC sections 3-4).

J-arms read the full transcript and choose among the task's candidates in one pass (chunked
prefill when needed). Two executions are scored: the raw proposal (J-only) and the proposal
through the suite's own `typed_execute` membrane (J->LDT), mirroring trained_trm_only and
trained_trm_ldt_fixed. Recorded arms are imported from evaluation_records.jsonl.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .foreign import LOOPED_TRANSFORMERS
from .jrunner import ChoiceItem

TASKS = LOOPED_TRANSFORMERS / "data" / "benchmarks" / "rlm_hybrid_long_context_tasks_v1.json"
RECORDS = LOOPED_TRANSFORMERS / "experiments" / "rlm_trm_ldt_hybrid_neighborhood_v1_1" / "campaign" / "evaluation_records.jsonl"
LETTERS = "ABCDEFGHIJKLMNOP"


def load_tasks(split: str = "eval") -> list[dict]:
    with open(TASKS, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [t for t in data["tasks"] if t["split"] == split]


def ldt_action(task: dict) -> str:
    """The suite's LDT fallback: highest LDT score among exact_allowed candidates (ties by name)."""
    allowed = [c for c in task["candidates"] if c in task["exact_allowed"]] or list(task["candidates"])
    return max(allowed, key=lambda c: (task["ldt_scores"].get(c, 0.0), c))


def typed_execute(task: dict, proposal: str | None) -> tuple[str, bool]:
    """Same rule as research_gym.benchmarks.rlm_hybrid_neighborhood.typed_execute."""
    if proposal is None or proposal not in task["candidates"] or proposal not in task["exact_allowed"]:
        return ldt_action(task), True
    return proposal, False


INSTRUCTION = (
    "Below is a long operations transcript. Most lines are noise. Read the lines that matter and choose "
    "the single best action for the final decision point. Answer with the option letter."
)


def choice_items(split: str = "eval") -> list[ChoiceItem]:
    items = []
    for task in load_tasks(split):
        options = list(task["candidates"])
        lines = [INSTRUCTION, "", "Transcript:", task["transcript"], "", "Options:"]
        lines += [f"{LETTERS[i]}) {option}" for i, option in enumerate(options)]
        lines.append("Answer:")
        meta = {"family": task["family"], "depth": None, "task": task}
        items.append(ChoiceItem(task["task_id"], "\n".join(lines), [" " + LETTERS[i] for i in range(len(options))], options, task["optimal_action"], meta))
    return items


def score_rows(rows: list[dict], membrane: bool) -> list[dict]:
    """Rescore J rows as executed actions; drops the embedded task from the stored row."""
    out = []
    for row in rows:
        task = row.pop("task")
        executed, fallback = typed_execute(task, row["pred"]) if membrane else (row["pred"], False)
        new = dict(row)
        new.update(
            {
                "proposal": row["pred"],
                "pred": executed,
                "correct": executed == task["optimal_action"],
                "fallback": fallback,
                "utility": task["utilities"][executed],
                "unsafe": executed not in task["exact_allowed"],
            }
        )
        out.append(new)
        row["task"] = task
    return out


def recorded_arms() -> dict[str, list[dict]]:
    """Recorded rows per architecture, as unified item rows (seed-replicates kept separate)."""
    tasks = {t["task_id"]: t for t in load_tasks("eval")}
    by: dict[str, list[dict]] = defaultdict(list)
    with open(RECORDS, "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by[r["architecture_id"]].append(
                {
                    "item_id": f"{r['task_id']}@{r['replicate_seed']}",
                    "gold": r["optimal_action"],
                    "pred": r["executed_action"],
                    "proposal": r["proposal_action"],
                    "correct": bool(r["correct"]),
                    "unsafe": bool(r["unsafe"]),
                    "utility": r["utility"],
                    "fallback": bool(r["fallback"]),
                    "family": r["task_family"],
                    "depth": None,
                    "options": None,
                    "probs": None,
                    "seed": r["replicate_seed"],
                }
            )
    missing = {tid for rows in by.values() for tid in [x["item_id"].split("@")[0] for x in rows]} - set(tasks)
    if missing:
        raise RuntimeError(f"recorded rows reference unknown tasks: {sorted(missing)[:3]}")
    return dict(by)
