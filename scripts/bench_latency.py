"""Paired latency comparison of variants on a GPU that other jobs also use.

    python scripts/bench_latency.py --bench configs/sweeps/latency.yaml

Runs made minutes apart are not comparable here: the same computation was measured at 41 ms
and 66 ms within one session. This script instead times every variant on the same example
back to back, rotates the order each round, and reports the ratio to the reference variant
per (example, round) pair, so drift in background load cancels to first order. No
instrumentation is attached. It also records how many other processes were on the GPU.
"""

import argparse
import statistics
import subprocess
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from jevq.config import ROOT, load_yaml, resolve, resolve_path
from jevq.flops import FlopModel
from jevq.harness import label_token_ids, make_variant, run_example
from jevq.modeling import load_bundle
from jevq.records import git_state, write_json
from jevq.tasks import build_task


def other_gpu_processes() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return max(len(out.stdout.split()) - 1, 0)  # minus this process


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", default="configs/sweeps/latency.yaml")
    args = parser.parse_args()
    bench = load_yaml(resolve_path(args.bench))
    readout = bench["readout"]

    cfgs = [resolve(path, bench["suite"]) for path in bench["variants"]]
    bundle = load_bundle(cfgs[0]["model"])
    num_layers = bundle.text_model.config.num_hidden_layers
    variants = [make_variant(cfg["variant"], num_layers) for cfg in cfgs]
    fm = FlopModel.from_text_model(bundle.text_model, bundle.lm_head)
    label_ids = label_token_ids(bundle.tokenizer)

    suite = cfgs[0]["suite"]
    items = []
    for task in suite["tasks"]:
        shots, tests = build_task(
            task, cfgs[0]["seed"], suite["difficulties"], suite["n_per_difficulty"], cfgs[0]["readout"]["n_shots"]
        )
        step = max(len(tests) // bench["per_task"], 1)
        items += [(shots, example) for example in tests[::step][: bench["per_task"]]]

    def timed(index: int, shots, example) -> dict:
        return run_example(bundle, variants[index], fm, readout, shots, example, cfgs[index], label_ids, None)

    for index in range(len(variants)):  # warm-up: first calls pay one-off allocation costs
        timed(index, *items[0])

    before = other_gpu_processes()
    samples = {v.name: [] for v in variants}
    ratios = {v.name: [] for v in variants[1:]}
    for round_index in range(bench["repeats"]):
        for item_index, (shots, example) in enumerate(items):
            order = list(range(len(variants)))
            shift = (round_index + item_index) % len(order)
            order = order[shift:] + order[:shift]
            latency = {index: timed(index, shots, example)["latency_s"] for index in order}
            for index, variant in enumerate(variants):
                samples[variant.name].append(latency[index])
                if index > 0:
                    ratios[variant.name].append(latency[index] / latency[0])
    after = other_gpu_processes()

    def quantiles(values: list[float]) -> dict:
        cuts = statistics.quantiles(values, n=10)
        return {"median": statistics.median(values), "p10": cuts[0], "p90": cuts[-1]}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "bench": bench,
        "model": bundle.info,
        "n_examples": len(items),
        "other_gpu_processes_before": before,
        "other_gpu_processes_after": after,
        "reference": variants[0].name,
        "latency_s": {name: quantiles(values) for name, values in samples.items()},
        "paired_ratio_to_reference": {name: quantiles(values) for name, values in ratios.items()},
        "layer_applications": {
            v.name: num_layers + (v.loop.span * (v.loop.n_iters - 1) if v.loop else 0) for v in variants
        },
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / "latency" / f"{stamp}_{readout}.json"
    write_json(out, report)
    for name, stats in report["latency_s"].items():
        ratio = report["paired_ratio_to_reference"].get(name)
        suffix = f"  ratio to {variants[0].name}: {ratio['median']:.3f} [{ratio['p10']:.3f}, {ratio['p90']:.3f}]" if ratio else ""
        print(f"{name:<22} median {stats['median'] * 1000:.0f} ms{suffix}")
    print(f"other GPU processes before/after: {before}/{after}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
