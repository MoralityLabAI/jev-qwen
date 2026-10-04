"""xbench runner (docs/xbench/SPEC.md).

    python scripts/xbench.py cpu                      # every non-J arm: RMP, scripts, ports, recorded, controls
    python scripts/xbench.py j --arm J-V1 --suites s1,s2,s7
    python scripts/xbench.py import-s4 --map J-V0=<run_dir> J-cot-V0=<run_dir> ...
    python scripts/xbench.py coverage [--fill-na]       # registered (arm, suite) cells without a record

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
    j.add_argument("--seed", type=int, default=0, help="training seed of the adapter (0 = the registered arm)")
    bon = sub.add_parser("bonsai")
    bon.add_argument("--suites", default="s1,s2,s4,s5,s6,s7")
    adapt = sub.add_parser("adaptive")
    adapt.add_argument("--gates", default="J-V1,J-V0,J-V1c,J-V2b-r2,Bonsai-8B")
    adapt.add_argument("--max-targets", type=int)
    adapt.add_argument("--force", action="store_true", help="rerun gates that already have an s6a record")
    probe = sub.add_parser("probe-s3")
    probe.add_argument("--arm", default="J-V0")
    imp = sub.add_parser("import-s4")
    imp.add_argument("--map", nargs="+", required=True, metavar="ARM=RUN_DIR")
    cov = sub.add_parser("coverage")
    cov.add_argument("--fill-na", action="store_true", help="write not-applicable records for cells the SPEC defines no adapter for")
    args = parser.parse_args()

    if args.command == "cpu":
        only = set(filter(None, args.only.split(","))) or {"s1", "s2", "s3", "s4", "s5", "s6", "s7", "na"}
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
        if "s4" in only:
            run.run_s4_script()
        if "na" in only:
            run.run_not_applicable()
    elif args.command == "j":
        run.run_j(args.arm, [s.strip() for s in args.suites.split(",") if s.strip()], seed=args.seed)
    elif args.command == "bonsai":
        from jevq.xbench.run_bonsai import run_bonsai

        run_bonsai([x.strip() for x in args.suites.split(",") if x.strip()])
    elif args.command == "adaptive":
        from jevq.xbench.run_bonsai import run_s6_adaptive

        run_s6_adaptive([x.strip() for x in args.gates.split(",") if x.strip()], args.max_targets, force=args.force)
    elif args.command == "probe-s3":
        run.probe_s3(args.arm)
    elif args.command == "import-s4":
        mapping = dict(item.split("=", 1) for item in args.map)
        run.run_s4_recorded({k: Path(v) for k, v in mapping.items()})
    elif args.command == "coverage":
        from jevq.xbench import coverage

        if args.fill_na:
            for arm, suite, reason in coverage.fill_structural():
                print(f"[coverage] wrote N/A {suite} {arm}: {reason}")
        cells = coverage.coverage()
        missing = sorted(cell for cell, (state, _) in cells.items() if state == "missing")
        counts = {state: sum(v[0] == state for v in cells.values()) for state in ("live", "na", "missing")}
        print(f"[coverage] {len(cells)} cells: {counts}")
        for arm, suite in missing:
            print(f"  missing: {arm} on {suite}")


if __name__ == "__main__":
    main()
