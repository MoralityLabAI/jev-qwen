"""xbench runner (docs/xbench/SPEC.md).

    python scripts/xbench.py cpu                      # every non-J arm: RMP, scripts, ports, recorded, controls
    python scripts/xbench.py j --arm J-V1 --suites s1,s2,s7
    python scripts/xbench.py import-s4 --map J-V0=<run_dir> J-cot-V0=<run_dir> ...

Records go to results/xbench/<suite>/<arm>/{record.json,items.jsonl}.
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from jevq.xbench import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cpu = sub.add_parser("cpu")
    cpu.add_argument("--only", default="", help="comma list of: s1,s2,s3,s5,s6,s7,na")
    j = sub.add_parser("j")
    j.add_argument("--arm", required=True)
    j.add_argument("--suites", default="s1,s2,s4,s5,s6,s7")
    probe = sub.add_parser("probe-s3")
    probe.add_argument("--arm", default="J-V0")
    imp = sub.add_parser("import-s4")
    imp.add_argument("--map", nargs="+", required=True, metavar="ARM=RUN_DIR")
    args = parser.parse_args()

    if args.command == "cpu":
        only = set(filter(None, args.only.split(","))) or {"s1", "s2", "s3", "s5", "s6", "s7", "na"}
        if "s1" in only:
            for arm_id in run.s1_rmp.RMP_CELLS:
                print(f"[xbench] {arm_id} on s1", flush=True)
                run.run_s1_native(arm_id)
            run.run_s1_script()
        if "s2" in only:
            run.run_s2_nonj()
        if "s3" in only:
            run.run_s3_recorded()
        if "s5" in only:
            run.run_s5_controls()
        if "s6" in only:
            run.run_s6_script()
            run.run_s6_contract_controls()
        if "s7" in only:
            run.run_s7_native()
        if "na" in only:
            run.run_not_applicable()
    elif args.command == "j":
        run.run_j(args.arm, [s.strip() for s in args.suites.split(",") if s.strip()])
    elif args.command == "probe-s3":
        run.probe_s3(args.arm)
    elif args.command == "import-s4":
        mapping = dict(item.split("=", 1) for item in args.map)
        run.run_s4_recorded({k: Path(v) for k, v in mapping.items()})


if __name__ == "__main__":
    main()
