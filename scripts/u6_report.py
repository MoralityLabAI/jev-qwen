"""SPEC-U6 report: training on real proposals (part T) and what the Campsite training cost the Jev on
its earlier suites (part F).

    python scripts/u6_report.py

Reads results/u6, results/u5, results/u4, results/xbench and results/u2; writes reports/ubench/u6.md.
"""

from __future__ import annotations

import json
import random

import _bootstrap  # noqa: F401
import u2_report as u2r
import ubench_report as ubr
from jevq.config import ROOT
from jevq.xbench import u2_joint
from jevq.xbench.stats import holm, wilson
from xbench_report import load_records

U4_RUNS, U5_RUNS, U6_RUNS = (ROOT / "results" / p for p in ("u4", "u5", "u6"))
OUT = ROOT / "reports" / "ubench"
SEEDS = (0, 1, 2)
BANDS = ("std", "8x8", "10x10")
MARGIN = -0.02
F_LEVEL = 1 - 0.05 / 5  # Bonferroni over the five forgetting suites: two-sided 99% intervals
SEEDED = {"J-real": "J-real{s}", "J-u4": "J-u4{s}", "Decision-TRM-real": "Decision-TRM-real{s}", "Decision-TRM": "Decision-TRM{s}"}
T_TESTS = {
    "T1": (("repair", "J-real"), ("repair", "c_repair"), "two", "J-real written repair vs c_repair (U5 proposals)"),
    "T2": (("repair", "J-real"), ("repair", "J-u4"), "one", "training on real proposals improves the Jev's written repair (U5 proposals)"),
    "T3": (("decide", "J-real"), ("decide", "Decision-TRM-real"), "two", "J-real decider vs Decision-TRM-real (U5 proposals)"),
}


def suffix(seed: int) -> str:
    return "" if seed == 0 else f"-s{seed}"


def path_for(part: str, set_name: str, record: str):
    if record.startswith(("J-real", "Decision-TRM-real")):
        return U6_RUNS / part / set_name / f"{record}.jsonl"
    if set_name == "u4test" and not record.startswith("J-u4"):
        return U4_RUNS / part / f"{record}.jsonl"
    return U5_RUNS / part / set_name / f"{record}.jsonl"


def instances(part: str, set_name: str, arm: str) -> list[tuple[int, dict]]:
    pattern = SEEDED.get(arm, arm)
    out = []
    for seed in (SEEDS if "{s}" in pattern else (0,)):
        path = path_for(part, set_name, pattern.format(s=suffix(seed)))
        if path.exists():
            out.append((seed, {r["item_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}))
    return out


def rate(inst, key: str, pred=lambda r: True) -> tuple[int, int]:
    rows = [r for _, res in inst for r in res.values() if pred(r)]
    return sum(bool(r.get(key)) for r in rows), len(rows)


def fmt(k: int, n: int, ci: bool = False) -> str:
    if not n:
        return "-"
    if not ci:
        return f"{k / n:.3f}"
    low, high = wilson(k, n)
    return f"{k / n:.3f} [{low:.2f}, {high:.2f}]"


def noninferiority(pairs, margin: float = MARGIN, level: float = F_LEVEL, resamples: int = 10_000, seed: int = 0) -> dict:
    """ubench_report.noninferiority with a chosen two-sided level: mean paired difference (a - b) over seeds,
    item-bootstrap percentile interval; non-inferior iff its lower end is above `margin`."""
    ids = pairs[0][2]
    if any(p[2] != ids for p in pairs):
        raise ValueError("seeds must share one item set")
    diffs = [[int(bool(a[i]["correct"])) - int(bool(b[i]["correct"])) for i in ids] for a, b, _ in pairs]
    per_item = [sum(d[k] for d in diffs) / len(diffs) for k in range(len(ids))]
    rng = random.Random(seed)
    n = len(per_item)
    boots = sorted(sum(per_item[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    tail = (1 - level) / 2
    low, high = boots[int(tail * resamples)], boots[min(resamples - 1, int((1 - tail) * resamples))]
    return {"estimate": sum(per_item) / n, "low": low, "high": high, "noninferior": low > margin, "seeds": len(pairs), "n_items": n}


def u2_pairs() -> tuple[list, list]:
    items = u2_joint.build_items()
    pairs, missing = [], []
    for seed in SEEDS:
        sfx = suffix(seed)
        a = u2r.outcomes(items, f"J-u4{sfx}", f"J-u4{sfx}")
        b = u2r.outcomes(items, f"J-multi{sfx}", f"J-multi{sfx}")
        if a is None or b is None:
            missing.append(seed)
            continue
        conv = lambda res: {k: {"correct": v["success"]} for k, v in res.items()}  # noqa: E731
        pairs.append((conv(a), conv(b), sorted(a)))
    return pairs, missing


def forgetting_section() -> list[str]:
    records = load_records()
    cases = {
        "F1": ("S1 core4, depths 1-8", lambda: ubr.seed_pairs(records, "s1", "J-u4", "J-multi", ubr.s1_core_trained)),
        "F2": ("S2 commit/veto", lambda: ubr.seed_pairs(records, "s2", "J-u4", "J-multi")),
        "F3": ("S4 dev suite", lambda: ubr.seed_pairs(records, "s4", "J-u4", "J-multi")),
        "F4": ("S7p skill routing (shuffled shortlist)", lambda: ubr.seed_pairs(records, "s7p", "J-u4", "J-multi")),
        "F5": ("U2 end to end, Jev routes and gates (pipeline C)", u2_pairs),
    }
    lines = ["## Part F: what the Campsite training cost (J-u4 vs J-multi, same seed)", "",
             f"Non-inferior iff the lower end of the two-sided {F_LEVEL:.0%} item-bootstrap interval of the mean paired difference "
             f"(J-u4 - J-multi) is above {MARGIN}. The level is Bonferroni-adjusted over the five suites.", "",
             "| Test | Suite | J-u4 | J-multi | Difference [interval] | Seeds | Verdict |", "|---|---|---|---|---|---|---|"]
    for tid, (name, get) in cases.items():
        pairs, missing = get()
        if not pairs:
            lines.append(f"| {tid} | {name} | - | - | - | 0 | pending |")
            continue
        res = noninferiority(pairs)
        acc_a = sum(bool(a[i]["correct"]) for a, _, ids in pairs for i in ids) / sum(len(ids) for *_, ids in pairs)
        acc_b = sum(bool(b[i]["correct"]) for _, b, ids in pairs for i in ids) / sum(len(ids) for *_, ids in pairs)
        status = ("non-inferior" if res["noninferior"] else "**not non-inferior**") if not missing else "interim"
        lines.append(f"| {tid} | {name} | {acc_a:.3f} | {acc_b:.3f} | {res['estimate']:+.3f} [{res['low']:+.3f}, {res['high']:+.3f}] | "
                     f"{res['seeds']} | {status} |")
    return lines


def t_section() -> list[str]:
    lines = ["## Part T: training on real proposals", "", "### Registered tests (U5 proposal set, 474 items, held out)", "",
             "| Test | Claim | Result | Status |", "|---|---|---|---|"]
    results, ps = {}, {}
    for tid, ((pa, a), (pb, b), sided, claim) in T_TESTS.items():
        ia, ib = instances(pa, "proposals", a), instances(pb, "proposals", b)
        if not ia or not ib:
            results[tid] = None
            continue
        res = u2r.pooled(ia, ib)
        results[tid] = (res, len(ia), len(ib), fmt(*rate(ia, "success")), fmt(*rate(ib, "success")))
        ps[tid] = res["p_one_sided"] if sided == "one" else res["p_two_sided"]
    decided = all(v is not None for v in results.values()) and len(instances("repair", "proposals", "J-real")) == 3 \
        and len(instances("decide", "proposals", "Decision-TRM-real")) == 3
    adjusted = holm(ps) if decided else {}
    for tid, ((pa, a), (pb, b), sided, claim) in T_TESTS.items():
        if results[tid] is None:
            lines.append(f"| {tid} | {claim} | - | pending |")
            continue
        res, na, nb, ra, rb = results[tid]
        holm_text = f", Holm p = {adjusted[tid]:.3g}" if tid in adjusted else ""
        lines.append(f"| {tid} | {claim} | {ra} vs {rb}; {a}-only {res['a_only']} vs {b}-only {res['b_only']} ({na} x {nb}), "
                     f"p = {ps[tid]:.3g}{holm_text} | {'decided' if decided else 'interim'} |")

    band = {b: (lambda r, b=b: r["kind"] == b) for b in BANDS}
    lines += ["", "### Repairers on the U5 proposals", "", "| Arm | Instances | Success | " + " | ".join(BANDS) + " |", "|---|---|---|" + "---|" * len(BANDS)]
    for arm in ("J-real", "J-u4", "c_repair", "dual_repair", "identity", "CSP-resolve"):
        inst = instances("repair", "proposals", arm)
        if inst:
            lines.append(f"| {arm} | {len(inst)} | {fmt(*rate(inst, 'success'), ci=True)} | " + " | ".join(fmt(*rate(inst, "success", band[b])) for b in BANDS) + " |")
    lines += ["", "### Deciders on the U5 proposals", "", "| Arm | Instances | Success | Unsafe | Rejected | " +
              " | ".join(f"{b} success / unsafe" for b in BANDS) + " |", "|---|---|---|---|---|" + "---|" * len(BANDS)]
    for arm in ("J-real", "J-u4", "Decision-TRM-real", "Decision-TRM", "dual_repair_if_any_sig_fail+plain",
                "dual_repair_if_any_sig_fail+verify-reject", "menu-ceiling"):
        inst = instances("decide", "proposals", arm)
        if inst:
            cells = [f"{fmt(*rate(inst, 'success', band[b]))} / {fmt(*rate(inst, 'unsafe', band[b]))}" for b in BANDS]
            lines.append(f"| {arm} | {len(inst)} | {fmt(*rate(inst, 'success'), ci=True)} | {fmt(*rate(inst, 'unsafe'))} | "
                         f"{fmt(*rate(inst, 'rejected'))} | " + " | ".join(cells) + " |")
    lines += ["", "### Synthetic skill kept? (U4 test set)", "", "| Arm | Decide success | Decide unsafe | Written repair success |", "|---|---|---|---|"]
    for arm in ("J-real", "J-u4"):
        d, r = instances("decide", "u4test", arm), instances("repair", "u4test", arm)
        if d or r:
            lines.append(f"| {arm} | {fmt(*rate(d, 'success'))} | {fmt(*rate(d, 'unsafe'))} | {fmt(*rate(r, 'success'))} |")
    summary = U6_RUNS / "cpu_summary.json"
    if summary.exists():
        lines += ["", f"Training proposals: {json.loads(summary.read_text(encoding='utf-8'))}"]
    return lines


def render() -> str:
    lines = ["# SPEC-U6 report: training on real proposals, and the cost of the Campsite training", "",
             "Generated by `scripts/u6_report.py`. Registration: `docs/ubench/SPEC-U6.md`.", ""]
    return "\n".join(lines + t_section() + [""] + forgetting_section()) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u6.md").write_text(render(), encoding="utf-8")
    print(f"wrote {OUT / 'u6.md'}")


if __name__ == "__main__":
    main()
