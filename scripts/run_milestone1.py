"""Milestone 1 in one process: one model load, then the checks and the three smoke runs.

    python scripts/run_milestone1.py

Order: load -> driver-vs-stock checks on real weights -> V0 (stock) -> V0 through the
schedule driver (must predict the same) -> V2 zero-shot loop. Loading once keeps the time
on the shared GPU short. Stops at the first failed gate.
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401
from check_model_load import run_checks
from jevq.config import ROOT, resolve, resolve_path
from jevq.harness import run
from jevq.modeling import load_bundle
from jevq.records import write_json

VARIANTS = ["v0_baseline", "v0_driver_identity", "v2_loop_mid_r2"]


def predictions(record: dict, results_dir) -> list:
    path = results_dir / record["run_id"] / "examples.jsonl"
    with open(path, "r", encoding="utf-8") as fh:
        return [(row["example_id"], row["readout"], row["pred"]) for row in map(json.loads, fh)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="evals/suites/smoke.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    args = parser.parse_args()

    cfgs = {name: resolve(f"configs/variants/{name}.yaml", args.suite, args.set) for name in VARIANTS}
    first = cfgs[VARIANTS[0]]
    results_dir = resolve_path(first["paths"]["results"])

    t0 = time.perf_counter()
    bundle = load_bundle(first["model"])
    report = run_checks(bundle, time.perf_counter() - t0)
    if not report["driver_matches_stock"]:
        raise SystemExit("FAIL: schedule driver disagrees with the stock forward on real weights")

    records = {name: run(cfgs[name], bundle=bundle) for name in VARIANTS}

    stock = predictions(records["v0_baseline"], results_dir)
    identity = predictions(records["v0_driver_identity"], results_dir)
    mismatches = [a for a, b in zip(stock, identity) if a != b]
    summary = {
        "model_load": {k: v for k, v in report.items() if k not in ("hardware", "layer_types")},
        "identity_prediction_mismatches": len(mismatches),
        "identity_mismatch_examples": mismatches[:10],
        "runs": {
            name: {
                "run_id": record["run_id"],
                **{
                    readout: {
                        "overall": scores["overall"],
                        "accuracy_by_task": {t: s["accuracy"] for t, s in scores["by_task"].items()},
                        "accuracy_by_difficulty": {d: s["accuracy"] for d, s in scores["by_difficulty"].items()},
                        "control": scores["control"],
                    }
                    for readout, scores in record["scores"].items()
                },
            }
            for name, record in records.items()
        },
    }
    out = ROOT / "results" / "milestone1_summary.json"
    write_json(out, summary)
    print(f"identity-driver prediction mismatches vs stock: {len(mismatches)} of {len(stock)}")
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
