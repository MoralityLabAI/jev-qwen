"""Paired latency for H1: `choice` vs `generate_cot` of the same adapted model (EXPERIMENT.md section 8).

    python scripts/bench_h1_latency.py --variant configs/variants/v1_lora_s0.yaml --per-task 5 --repeats 3

For each example both readouts run back to back in alternating order; the statistic is the median
of the per-example ratio latency(generate_cot) / latency(choice). H1's efficiency bound: >= 5x.
Examples come from the 400-example dev prefix used for the reasoning-trace runs.
"""

import argparse
import statistics
import subprocess
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from jevq.config import ROOT, resolve
from jevq.flops import FlopModel
from jevq.harness import label_token_ids, make_variant, run_example
from jevq.modeling import attach_adapter, load_bundle
from jevq.records import git_state, write_json
from jevq.tasks import build_task
from jevq.xbench.harness_paths import adapter_for


def other_gpu_processes() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return max(len(out.stdout.split()) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="configs/variants/v1_lora_s0.yaml")
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    cfg = resolve(args.variant, "evals/suites/dev.yaml", ["suite.n_per_difficulty=10"])
    bundle = load_bundle(cfg["model"])
    adapter = adapter_for(cfg)
    if adapter:
        bundle = attach_adapter(bundle, adapter)
    variant = make_variant(cfg["variant"], bundle.text_model.config.num_hidden_layers)
    fm = FlopModel.from_text_model(bundle.text_model, bundle.lm_head)
    label_ids = label_token_ids(bundle.tokenizer)
    suite = cfg["suite"]
    items = []
    for task in suite["tasks"]:
        shots, tests = build_task(task, cfg["seed"], suite["difficulties"], suite["n_per_difficulty"], cfg["readout"]["n_shots"])
        step = max(len(tests) // args.per_task, 1)
        items += [(shots, e) for e in tests[::step][: args.per_task]]

    run_example(bundle, variant, fm, "choice", *items[0], cfg, label_ids, None)  # warm-up
    run_example(bundle, variant, fm, "generate_cot", *items[0], cfg, label_ids, None)
    before = other_gpu_processes()
    ratios, choice_s, cot_s, tokens = [], [], [], []
    for round_index in range(args.repeats):
        for index, (shots, example) in enumerate(items):
            order = ("choice", "generate_cot") if (round_index + index) % 2 == 0 else ("generate_cot", "choice")
            times = {}
            for readout in order:
                row = run_example(bundle, variant, fm, readout, shots, example, cfg, label_ids, None)
                times[readout] = row["latency_s"]
                if readout == "generate_cot":
                    tokens.append(row["output_tokens"])
            ratios.append(times["generate_cot"] / times["choice"])
            choice_s.append(times["choice"])
            cot_s.append(times["generate_cot"])
    after = other_gpu_processes()

    def q(values):
        cuts = statistics.quantiles(values, n=10)
        return {"median": statistics.median(values), "p10": cuts[0], "p90": cuts[-1]}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "variant": cfg["variant"],
        "adapter": str(adapter) if adapter else None,
        "n_examples": len(items),
        "repeats": args.repeats,
        "ratio_cot_over_choice": q(ratios),
        "choice_latency_s": q(choice_s),
        "cot_latency_s": q(cot_s),
        "cot_tokens_mean": sum(tokens) / len(tokens),
        "h1_efficiency_bound": 5.0,
        "h1_efficiency_met": statistics.median(ratios) >= 5.0,
        "other_gpu_processes_before_after": [before, after],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json(ROOT / "results" / "latency" / f"{stamp}_h1_{cfg['variant']['name']}.json", report)
    print(report)


if __name__ == "__main__":
    main()
