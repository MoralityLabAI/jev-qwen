"""Run one variant over one suite.

    python scripts/run_eval.py --variant configs/variants/v0_baseline.yaml --suite evals/suites/smoke.yaml
    python scripts/run_eval.py --variant configs/variants/v2_loop_mid_r2.yaml --set variant.loop.n_iters=3
"""

import argparse

import _bootstrap  # noqa: F401
from jevq.config import resolve
from jevq.harness import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, help="variant YAML, relative to the repo root")
    parser.add_argument("--suite", default="evals/suites/smoke.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    args = parser.parse_args()

    record = run(resolve(args.variant, args.suite, args.set))
    for readout, scores in record["scores"].items():
        overall = scores["overall"]
        print(
            f"{readout:<8} acc={overall['accuracy']:.3f} "
            f"out_tok={overall['output_tokens_mean']:.1f} "
            f"lat_p50={overall['latency_s_p50'] * 1000:.0f}ms "
            f"layer_apps={overall['layer_applications_mean']:.0f}"
        )


if __name__ == "__main__":
    main()
