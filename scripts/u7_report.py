"""SPEC-U7 report: the compact looped reasoner on Campsite (part L) and J-mix (part M).

    python scripts/u7_report.py

Reads results/u7 (and the earlier U4-U6 / xbench / U2 records it compares with); writes reports/ubench/u7.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import u2_report as u2r
import u6_report as u6r
import ubench_report as ubr
from jevq.config import ROOT
from jevq.xbench import u2_joint
from jevq.xbench.stats import holm, wilson
from xbench_report import load_records

R = ROOT / "results"
OUT = ROOT / "reports" / "ubench"
SEEDS = (0, 1, 2)
T_EVAL = (1, 2, 4, 8, 16, 32, 64, 128)
SETS = ("u4test", "u5test", "proposals", "l7")
BANDS = {"u4test": None, "u5test": None, "proposals": ("std", "8x8", "10x10"), "l7": ("8x8", "10x10", "12x12", "14x14")}


def sfx(seed: int) -> str:
    return "" if seed == 0 else f"-s{seed}"


def rows(path: Path) -> dict | None:
    if not path.exists():
        return None
    return {r["item_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def arm_paths(arm: str, set_name: str, t: int | None = None) -> list[tuple[int, Path]]:
    """(seed, path) for every instance of an arm's written-repair records on a set."""
    if arm in ("LOOP", "DEEP", "FLAT"):
        t = t or {"LOOP": 16, "DEEP": 16, "FLAT": 1}[arm]
        return [(s, R / "u7" / "eval" / set_name / f"{arm}-s{s}-T{t}.jsonl") for s in SEEDS]
    if arm in ("J-u4", "J-real", "J-mix"):
        base = {"J-u4": R / "u5" / "repair", "J-real": R / "u6" / "repair", "J-mix": R / "u7" / "repair"}[arm]
        return [(s, base / set_name / f"{arm}{sfx(s)}.jsonl") for s in SEEDS]
    base = {"u4test": R / "u4" / "repair", "proposals": R / "u5" / "repair" / "proposals", "l7": R / "u7" / "eval" / "l7"}.get(set_name)
    return [(0, base / f"{arm}.jsonl")] if base else []


def instances(arm: str, set_name: str, t: int | None = None) -> list[tuple[int, dict]]:
    out = []
    for seed, path in arm_paths(arm, set_name, t):
        r = rows(path)
        if r is not None:
            out.append((seed, r))
    return out


def rate(inst, pred=lambda r: True) -> tuple[int, int]:
    rs = [r for _, res in inst for r in res.values() if pred(r)]
    return sum(bool(r["success"]) for r in rs), len(rs)


def fmt(k: int, n: int, ci: bool = False) -> str:
    if not n:
        return "-"
    if not ci:
        return f"{k / n:.3f}"
    low, high = wilson(k, n)
    return f"{k / n:.3f} [{low:.2f}, {high:.2f}]"


L_TESTS = {
    "L1": (("LOOP", "u4test", 16), ("J-u4", "u4test", None), "two", "LOOP (T 16) vs J-u4, written repair (U4 test)"),
    "L2": (("LOOP", "proposals", 16), ("J-real", "proposals", None), "two", "LOOP (T 16) vs J-real, written repair (real proposals)"),
    "L3": (("LOOP", "l7", 64), ("DEEP", "l7", None), "one", "LOOP (T 64) beats DEEP on the large set"),
    "L4": (("LOOP", "l7", 64), ("LOOP", "l7", 16), "one", "test-time iterations help past the trained depth (large set)"),
}


def tests(spec: dict, decided_when) -> list[str]:
    lines = ["| Test | Claim | Result | Status |", "|---|---|---|---|"]
    results, ps = {}, {}
    for tid, (a, b, sided, claim) in spec.items():
        ia, ib = instances(*a), instances(*b)
        if not ia or not ib:
            results[tid] = None
            continue
        res = u2r.pooled(ia, ib)
        results[tid] = (res, fmt(*rate(ia)), fmt(*rate(ib)), len(ia), len(ib))
        ps[tid] = res["p_one_sided"] if sided == "one" else res["p_two_sided"]
    decided = all(v is not None for v in results.values()) and decided_when()
    adjusted = holm(ps) if decided else {}
    for tid, (a, b, sided, claim) in spec.items():
        if results[tid] is None:
            lines.append(f"| {tid} | {claim} | - | pending |")
            continue
        res, ra, rb, na, nb = results[tid]
        h = f", Holm p = {adjusted[tid]:.3g}" if tid in adjusted else ""
        lines.append(f"| {tid} | {claim} | {ra} vs {rb}; a-only {res['a_only']} vs b-only {res['b_only']} ({na} x {nb}), "
                     f"p = {ps[tid]:.3g}{h} | {'decided' if decided else 'interim'} |")
    return lines


def part_l() -> list[str]:
    lines = ["## Part L: compact looped transformer", "", "### Registered tests", ""]
    lines += tests(L_TESTS, lambda: all(len(instances(a, "l7", 64 if a == "LOOP" else None)) == 3 for a in ("LOOP", "DEEP")))
    for set_name in SETS:
        bands = BANDS[set_name]
        head = ["All"] + list(bands or [])
        lines += ["", f"### {set_name}: repaired grid passes the verifier", "",
                  "| Arm | Instances | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
        arms = [("LOOP", t) for t in T_EVAL] + [("DEEP", None), ("FLAT", None)]
        arms += [(a, None) for a in ("J-u4", "J-real", "J-mix", "identity", "c_repair", "dual_repair", "CSP-resolve")]
        for arm, t in arms:
            inst = instances(arm, set_name, t)
            if not inst:
                continue
            cells = [fmt(*rate(inst), ci=True)] + [fmt(*rate(inst, lambda r, b=b: r["kind"] == b)) for b in (bands or [])]
            label = f"LOOP T={t}" if arm == "LOOP" else arm
            lines.append(f"| {label} | {len(inst)} | " + " | ".join(cells) + " |")
    lines += ["", "### l7 by candidate type (LOOP T 64 / DEEP / FLAT)", "", "| Type | LOOP T=16 | LOOP T=64 | DEEP | FLAT | c_repair | CSP |",
              "|---|---|---|---|---|---|---|"]
    for typ in ("correct", "drop_tent", "extra_tent", "move_in_row", "move_any", "swap_rect", "tree_mutation", "bad_shape"):
        cells = []
        for arm, t in (("LOOP", 16), ("LOOP", 64), ("DEEP", None), ("FLAT", None), ("c_repair", None), ("CSP-resolve", None)):
            cells.append(fmt(*rate(instances(arm, "l7", t), lambda r, typ=typ: r.get("type") == typ)))
        lines.append(f"| {typ} | " + " | ".join(cells) + " |")
    infos = []
    for arm in ("LOOP", "DEEP", "FLAT"):
        p = R / "u7" / "runs" / f"{arm}-s0" / "arm.json"
        if p.exists():
            infos.append(json.loads(p.read_text(encoding="utf-8")))
    if infos:
        lines += ["", "Arms: " + "; ".join(f"{i['arm']} {i['params']:,} parameters" for i in infos) + "."]
    return lines


def mix_pairs(suite: str):
    records = load_records()
    subset = ubr.s1_core_trained if suite == "s1" else None
    return ubr.seed_pairs(records, suite, "J-mix", "J-multi", subset)


def u2_pairs():
    items = u2_joint.build_items()
    pairs, missing = [], []
    for seed in SEEDS:
        a = u2r.outcomes(items, f"J-mix{sfx(seed)}", f"J-mix{sfx(seed)}")
        b = u2r.outcomes(items, f"J-multi{sfx(seed)}", f"J-multi{sfx(seed)}")
        if a is None or b is None:
            missing.append(seed)
            continue
        conv = lambda res: {k: {"correct": v["success"]} for k, v in res.items()}  # noqa: E731
        pairs.append((conv(a), conv(b), sorted(a)))
    return pairs, missing


M_TESTS = {
    "M6": (("J-mix", "proposals", None), ("J-real", "proposals", None), "two", "J-mix vs J-real, written repair (real proposals)"),
    "M7": (("J-mix", "u4test", None), ("J-u4", "u4test", None), "two", "J-mix vs J-u4, written repair (U4 test)"),
}


def part_m() -> list[str]:
    lines = ["## Part M: one Jev for everything (J-mix)", "",
             f"M1-M5: non-inferior iff the lower end of the two-sided {u6r.F_LEVEL:.0%} item-bootstrap interval of the mean paired "
             f"difference (J-mix - J-multi) is above {u6r.MARGIN}.", "",
             "| Test | Suite | J-mix | J-multi | Difference [interval] | Seeds | Verdict |", "|---|---|---|---|---|---|---|"]
    cases = {"M1": ("S1 core4, depths 1-8", lambda: mix_pairs("s1")), "M2": ("S2", lambda: mix_pairs("s2")),
             "M3": ("S4", lambda: mix_pairs("s4")), "M4": ("S7p", lambda: mix_pairs("s7p")), "M5": ("U2 pipeline C", u2_pairs)}
    for tid, (name, get) in cases.items():
        pairs, missing = get()
        if not pairs:
            lines.append(f"| {tid} | {name} | - | - | - | 0 | pending |")
            continue
        res = u6r.noninferiority(pairs)
        n = sum(len(ids) for *_, ids in pairs)
        acc_a = sum(bool(a[i]["correct"]) for a, _, ids in pairs for i in ids) / n
        acc_b = sum(bool(b[i]["correct"]) for _, b, ids in pairs for i in ids) / n
        verdict = ("non-inferior" if res["noninferior"] else "**not non-inferior**") if not missing else "interim"
        lines.append(f"| {tid} | {name} | {acc_a:.3f} | {acc_b:.3f} | {res['estimate']:+.3f} [{res['low']:+.3f}, {res['high']:+.3f}] | "
                     f"{res['seeds']} | {verdict} |")
    lines += ["", "### Written repair (Holm over M6-M7)", ""]
    lines += tests(M_TESTS, lambda: len(instances("J-mix", "proposals")) == 3)
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = ["# SPEC-U7 report: compact looped reasoner on Campsite, and J-mix", "",
            "Generated by `scripts/u7_report.py`. Registration: `docs/ubench/SPEC-U7.md`.", ""]
    (OUT / "u7.md").write_text("\n".join(text + part_l() + [""] + part_m()) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'u7.md'}")


if __name__ == "__main__":
    main()
