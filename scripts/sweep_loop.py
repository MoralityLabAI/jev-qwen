"""Zero-shot recurrence sweep over (span, n_iters); loads the model once.

    python scripts/sweep_loop.py --sweep configs/sweeps/loop_span.yaml

Each cell is a normal harness run with its own record and steps.jsonl. The sweep file
written to results/sweeps/ indexes them and tabulates accuracy plus the final-iteration
state statistics, which is the first look at whether extra hidden computation moves the
state or just settles.
"""

import argparse
import json
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from jevq.config import ROOT, deep_merge, load_yaml, resolve, resolve_path
from jevq.harness import run
from jevq.modeling import load_bundle
from jevq.records import write_json


def last_iter_stats(run_dir, n_iters: int) -> dict:
    """Mean over examples of the statistics of the final span iteration (None where undefined)."""
    keys = ("rel_delta_last", "cos_prev_last", "cos_first_last", "norm_last", "lens_entropy", "lens_kl_prev")
    values: dict[str, list[float]] = {key: [] for key in keys}
    steps = run_dir / "steps.jsonl"
    if not steps.exists():
        return {}
    with open(steps, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["kind"] == "span_iter" and row["iter"] == n_iters - 1:
                for key in keys:
                    if row.get(key) is not None:
                        values[key].append(row[key])
    return {f"final_iter_{key}": (sum(v) / len(v) if v else None) for key, v in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default="configs/sweeps/loop_span.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    args = parser.parse_args()
    sweep = load_yaml(resolve_path(args.sweep))

    base = resolve("configs/variants/v0_driver_identity.yaml", sweep["suite"], args.set)
    base = deep_merge(base, {"readout": {"modes": sweep["readout_modes"]}})
    bundle = load_bundle(base["model"])

    cells = []
    for start, end in sweep["spans"]:
        for n_iters in sweep["n_iters"]:
            name = f"v2_loop_s{start}e{end}_r{n_iters}"
            cfg = deep_merge(
                base, {"variant": {"name": name, "loop": {"start": start, "end": end, "n_iters": n_iters}}}
            )
            record = run(cfg, bundle=bundle)
            run_dir = resolve_path(cfg["paths"]["results"]) / record["run_id"]
            cell = {"run_id": record["run_id"], "start": start, "end": end, "n_iters": n_iters}
            for readout, scores in record["scores"].items():
                cell[f"{readout}_accuracy"] = scores["overall"]["accuracy"]
                if scores["control"]:
                    cell[f"{readout}_unsafe_allow_rate"] = scores["control"]["unsafe_allow_rate"]
            cell.update(last_iter_stats(run_dir, n_iters))
            cells.append(cell)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / "sweeps" / f"{stamp}_loop_span.json"
    write_json(out, {"sweep": sweep, "cells": cells})
    print(f"sweep index: {out}")


if __name__ == "__main__":
    main()
