"""Print a one-line summary per xbench record.

    python scripts/xbench_table.py [suite ...]
"""

import json
import sys

import _bootstrap  # noqa: F401
from jevq.xbench.records import RESULTS


def fmt(value, digits=3):
    return "-" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def main() -> None:
    suites = sys.argv[1:] or sorted(p.name for p in RESULTS.iterdir() if p.is_dir())
    for suite in suites:
        print(f"== {suite}")
        for arm_dir in sorted((RESULTS / suite).iterdir()):
            record = json.load(open(arm_dir / "record.json", encoding="utf-8"))
            if "not_applicable" in record:
                print(f"  {arm_dir.name:<28} N/A: {record['not_applicable']}")
                continue
            m = record["metrics"]
            arm = record["arm"]
            parts = [f"n={m.get('n', record['n_items'])}"]
            if "accuracy" in m:
                parts.append(f"acc={fmt(m['accuracy'])}")
            if m.get("unsafe", {}).get("n"):
                parts.append(f"unsafe={fmt(m['unsafe']['rate'])} ({m['unsafe']['k']}/{m['unsafe']['n']})")
            if m.get("over_refusal", {}).get("n"):
                parts.append(f"overref={fmt(m['over_refusal']['rate'])} ({m['over_refusal']['k']}/{m['over_refusal']['n']})")
            if "utility_mean" in m:
                parts.append(f"utility={fmt(m['utility_mean'])}")
            if "half_life" in m:
                parts.append(f"t1/2={m['half_life']} flipped={m.get('flipped_by_turn10')} excluded={m['excluded_wrong_at_turn0']}")
            if m.get("calibration"):
                parts.append(f"ece={fmt(m['calibration']['ece'])}")
            if m.get("episodes"):
                parts.append(f"attack_success={fmt(m['episodes']['attack_success_rate'])} benign_done={fmt(m['episodes']['benign_useful_completion_rate'])}")
            parts.append(f"params={arm.get('params_total')}")
            parts.append(record["claim_label"])
            print(f"  {arm_dir.name:<28} " + "  ".join(parts))


if __name__ == "__main__":
    main()
