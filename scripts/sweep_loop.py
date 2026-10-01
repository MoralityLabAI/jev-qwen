"""Zero-shot recurrence sweep: one run per span, every iteration count read from it.

    python scripts/sweep_loop.py --sweep configs/sweeps/loop_span.yaml

Each span is run once at `max_iters` with the tail lens on. The lens after iteration r is
exactly the model's output had the loop stopped at r (tests/test_looped.py asserts this), so
one run yields accuracy and state statistics for every n_iters from 1 to max_iters. Row
n_iters=1 of every span is the vanilla model and must agree across spans.
"""

import argparse
import json
import math
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from jevq.config import ROOT, deep_merge, load_yaml, resolve, resolve_path
from jevq.harness import run
from jevq.modeling import load_bundle
from jevq.records import write_json

STATS = ("rel_delta_last", "cos_prev_last", "cos_first_last", "cos_prev2_last", "norm_last", "lens_entropy", "lens_kl_prev")


def read_jsonl(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def per_iteration(run_dir, max_iters: int) -> list[dict]:
    """One row per n_iters: early-exit accuracy, calibration and state statistics."""
    examples = {r["example_id"]: r for r in read_jsonl(run_dir / "examples.jsonl") if r["readout"] == "choice"}
    steps = [s for s in read_jsonl(run_dir / "steps.jsonl") if s["readout"] == "choice" and s["kind"] == "span_iter"]
    first = {s["example_id"]: s["lens_watch_probs"] for s in steps if s["iter"] == 0}

    table = []
    for it in range(max_iters):
        rows = [s for s in steps if s["iter"] == it]
        correct, p_correct, shift, flips, unsafe = [], [], [], [], []
        for step in rows:
            example = examples[step["example_id"]]
            probs = step["lens_watch_probs"]
            pred = max(range(len(probs)), key=probs.__getitem__)
            base = first[step["example_id"]]
            correct.append(pred == example["answer_index"])
            p_correct.append(probs[example["answer_index"]])
            shift.append(sum(abs(a - b) for a, b in zip(probs, base)) / 2)
            flips.append(pred != max(range(len(base)), key=base.__getitem__))
            if example["task_class"] == "control" and example["meta"]["truth"] == "DENY":
                unsafe.append(example["options"][pred] == "ALLOW")
        row = {
            "n_iters": it + 1,
            "n": len(rows),
            "accuracy": mean(correct),
            "nll": mean(-math.log(max(p, 1e-12)) for p in p_correct),
            "tv_shift_vs_vanilla": mean(shift),
            "pred_changed_vs_vanilla": mean(flips),
            "unsafe_allow_rate": mean(unsafe),
        }
        row.update({key: mean(s.get(key) for s in rows) for key in STATS})
        table.append(row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default="configs/sweeps/loop_span.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--only", metavar="START,END", help="run just this span (one queue step per span)")
    args = parser.parse_args()
    sweep = load_yaml(resolve_path(args.sweep))
    max_iters = sweep["max_iters"]
    if args.only:
        only = [int(part) for part in args.only.split(",")]
        if only not in sweep["spans"]:
            raise SystemExit(f"span {only} is not in {args.sweep}")
        sweep["spans"] = [only]

    base = resolve("configs/variants/v0_driver_identity.yaml", sweep["suite"], args.set)
    base = deep_merge(base, {"readout": {"modes": ["choice"]}, "instrument": {"enabled": True, "lens_mode": "tail"}})
    bundle = load_bundle(base["model"])

    spans = []
    for start, end in sweep["spans"]:
        name = f"v2_loop_s{start}e{end}_r{max_iters}"
        cfg = deep_merge(base, {"variant": {"name": name, "loop": {"start": start, "end": end, "n_iters": max_iters}}})
        record = run(cfg, bundle=bundle)
        table = per_iteration(resolve_path(cfg["paths"]["results"]) / record["run_id"], max_iters)
        final = record["scores"]["choice"]["overall"]["accuracy"]
        if abs(table[-1]["accuracy"] - final) > 1e-9:
            raise SystemExit(f"{name}: lens accuracy {table[-1]['accuracy']} != run accuracy {final}")
        spans.append({"start": start, "end": end, "run_id": record["run_id"], "by_n_iters": table})
        for row in table:
            print(
                f"[{start},{end}) n_iters={row['n_iters']} acc={row['accuracy']:.3f} "
                f"shift={row['tv_shift_vs_vanilla']:.3f} rel_delta={row['rel_delta_last']:.3f} "
                f"cos_prev={row['cos_prev_last']:.3f} norm={row['norm_last']:.1f}",
                flush=True,
            )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "_".join(f"s{start}e{end}" for start, end in sweep["spans"])
    out = ROOT / "results" / "sweeps" / f"{stamp}_loop_span_{label}.json"
    write_json(out, {"sweep": sweep, "spans": spans})
    print(f"sweep: {out}")


if __name__ == "__main__":
    main()
