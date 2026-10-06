"""SPEC-U1 report: the registered tests (docs/ubench/SPEC.md section 5) and descriptives.

    python scripts/ubench_report.py

Writes reports/ubench/u1.md. Tests whose seeds are not all present are marked pending; the
registered tests are decided only on seeds 0-2.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from jevq.config import ROOT
from jevq.xbench.stats import holm
from xbench_report import REPLICATE, ADDENDUM_ARMS, ci, cost, fmt, load_records, readiness, reliability

OUT = ROOT / "reports" / "ubench"
SEEDS = (0, 1, 2)
ORDER_INVARIANT_S7 = ("lexical-router", "SkillRouter-TRM")


def arm_id(base: str, seed: int, iteration: int | None = None) -> str:
    """Record id of a seeded arm: J-multi, J-multi-s1, J-multi-loop-s2-r2, ..."""
    return f"{base}{'' if seed == 0 else f'-s{seed}'}{'' if iteration is None else f'-r{iteration}'}"


def s1_core_trained(items: dict) -> list[str]:
    """S1 core4 families (masked_pointer_chase is J-only), depths 1-8."""
    return sorted(i for i, r in items.items() if r.get("family") != "masked_pointer_chase" and r.get("depth") is not None and r["depth"] <= 8)


def s1_core_heldout(items: dict) -> list[str]:
    return sorted(i for i, r in items.items() if r.get("family") != "masked_pointer_chase" and r.get("depth") is not None and r["depth"] > 8)


def binom_one_sided(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def pooled(pairs: list[tuple[dict, dict, list[str]]]) -> dict:
    """Discordant (seed, item) pairs pooled over seeds: a-only vs b-only, one-sided exact binomial."""
    a_only = sum(bool(a[i]["correct"]) and not b[i]["correct"] for a, b, ids in pairs for i in ids)
    b_only = sum(bool(b[i]["correct"]) and not a[i]["correct"] for a, b, ids in pairs for i in ids)
    return {"a_only": a_only, "b_only": b_only, "p_one_sided": binom_one_sided(a_only, a_only + b_only), "seeds": len(pairs)}


def noninferiority(pairs: list[tuple[dict, dict, list[str]]], margin: float = -0.02, resamples: int = 10_000, seed: int = 0) -> dict:
    """Mean over seeds of the paired accuracy difference (a - b) and its item-bootstrap percentile
    interval; the item set is shared by every seed, so one resample of item ids serves all seeds.
    Non-inferior iff the lower end of the two-sided 95% interval is above `margin`."""
    ids = pairs[0][2]
    if any(p[2] != ids for p in pairs):
        raise ValueError("seeds must share one item set")
    diffs = [[int(bool(a[i]["correct"])) - int(bool(b[i]["correct"])) for i in ids] for a, b, _ in pairs]
    per_item = [sum(d[k] for d in diffs) / len(diffs) for k in range(len(ids))]
    estimate = sum(per_item) / len(per_item)
    rng = random.Random(seed)
    n = len(per_item)
    boots = sorted(sum(per_item[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low, high = boots[int(0.025 * resamples)], boots[min(resamples - 1, int(0.975 * resamples))]
    return {"estimate": estimate, "low": low, "high": high, "margin": margin, "noninferior": low > margin, "seeds": len(pairs)}


def items_of(records: dict, suite: str, arm: str) -> dict | None:
    record = records.get(suite, {}).get(arm)
    if not record or "not_applicable" in record or not record.get("_items"):
        return None
    return {r["item_id"]: r for r in record["_items"]}


def seed_pairs(records, suite, a_base, b_base, subset=None, a_iter=None, b_iter=None, b_seeded=True):
    """Per-seed (a, b, ids) for every seed where both records exist; ids restricted by `subset`."""
    pairs, missing = [], []
    for seed in SEEDS:
        a = items_of(records, suite, arm_id(a_base, seed, a_iter))
        b = items_of(records, suite, arm_id(b_base, seed if b_seeded else 0, b_iter))
        if a is None or b is None:
            missing.append(seed)
            continue
        ids = subset(a) if subset else sorted(a)
        pairs.append((a, b, [i for i in ids if i in b]))
    return pairs, missing


def registered_tests(records) -> dict:
    t = {}
    t["T1"] = seed_pairs(records, "s1", "J-multi", "J-V1", s1_core_trained)
    t["T2"] = seed_pairs(records, "s7p", "J-multi", "J-V1")
    t["T3"] = seed_pairs(records, "s2", "J-multi", "J-V0", b_seeded=False)
    t["H"] = seed_pairs(records, "s4", "J-multi", "J-V1")
    t["S"] = seed_pairs(records, "s1", "J-multi", "J-V1-rmp", s1_core_trained)
    t["L"] = seed_pairs(records, "s1", "J-multi-loop", "J-multi", s1_core_trained, a_iter=2)
    out = {}
    for key, (pairs, missing) in t.items():
        if not pairs:
            out[key] = {"status": "pending", "missing_seeds": missing}
            continue
        result = noninferiority(pairs) if key in ("H", "S") else pooled(pairs)
        out[key] = {**result, "status": "decided" if not missing else "interim", "missing_seeds": missing}
    decided = {k: out[k]["p_one_sided"] for k in ("T1", "T2", "T3") if out[k].get("status") == "decided"}
    if len(decided) == 3:
        for k, p in holm(decided).items():
            out[k]["p_holm"] = p
    return out


def s7_pool(records) -> dict:
    """SPEC-U1 section 5: S7p records for text arms, S7 records for order-invariant arms."""
    pool = {a: r for a, r in records.get("s7p", {}).items() if not REPLICATE.match(a) and a not in ADDENDUM_ARMS}
    pool.update({a: records["s7"][a] for a in ORDER_INVARIANT_S7 if a in records.get("s7", {})})
    return pool


def reliability_pools(records) -> dict[str, dict]:
    """v1 registered arms (seed 0) plus the U1 seed-0 arms, per utility suite."""
    u1 = {"J-multi", "J-multi-loop-r1", "J-multi-loop-r2", "J-multi-loop-r3", "J-multi-LDT"}
    pools = {}
    for suite in ("s1", "s2", "s3", "s4"):
        pools[suite] = {a: r for a, r in records.get(suite, {}).items()
                        if (a in u1) or not (REPLICATE.match(a) or a in ADDENDUM_ARMS or a.startswith("J-multi"))}
    pools["s7"] = {a: r for a, r in s7_pool(records).items()}
    return pools


def acc_line(records, suite, arm, subset=None) -> str:
    items = items_of(records, suite, arm)
    if items is None:
        return "-"
    ids = subset(items) if subset else sorted(items)
    k = sum(bool(items[i]["correct"]) for i in ids)
    return f"{k / len(ids):.3f} {ci(k, len(ids))}"


def render(records) -> str:
    tests = registered_tests(records)
    lines = ["# SPEC-U1 report", "",
             "Generated by `scripts/ubench_report.py`. Registration: `docs/ubench/SPEC.md` (cac47df). "
             "Pooled = discordant (seed, item) pairs over seeds 0-2, one-sided exact binomial. Non-inferiority = mean "
             "paired difference over seeds, item bootstrap (10,000, seed 0), lower end of the two-sided 95% interval "
             "> -0.02 (interpretation recorded in notes/009 before the three-seed outcome).", "",
             "## Registered tests", "",
             "| Test | Claim | Seeds | Result | Status |", "|---|---|---|---|---|"]
    claims = {"T1": "J-multi > J-V1, S1 core4 d1-8", "T2": "J-multi > J-V1, S7p", "T3": "J-multi > J-V0, S2",
              "H": "J-multi non-inferior to J-V1, S4", "S": "J-multi non-inferior to J-V1-rmp, S1 core4 d1-8",
              "L": "J-multi-loop-r2 > J-multi, S1 core4 d1-8"}
    for key, claim in claims.items():
        r = tests[key]
        if r["status"] == "pending":
            lines.append(f"| {key} | {claim} | none | - | pending (missing seeds {r['missing_seeds']}) |")
            continue
        if key in ("H", "S"):
            res = f"diff {r['estimate']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]; {'non-inferior' if r['noninferior'] else 'not shown non-inferior'}"
        else:
            holm_p = f", Holm p = {r['p_holm']:.3g}" if "p_holm" in r else ""
            res = f"{r['a_only']} vs {r['b_only']}, p = {r['p_one_sided']:.3g}{holm_p}"
        status = r["status"] + (f" (missing seeds {r['missing_seeds']})" if r["missing_seeds"] else "")
        lines.append(f"| {key} | {claim} | {r['seeds']} | {res} | {status} |")

    lines += ["", "## Accuracy by seed [95% Wilson]", "",
              "| Suite | Arm | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|---|"]
    rows = [("s1 core4 d1-8", "s1", "J-multi", None, s1_core_trained), ("s1 core4 d1-8", "s1", "J-multi-loop", 2, s1_core_trained),
            ("s1 core4 d1-8", "s1", "J-V1", None, s1_core_trained), ("s1 core4 d1-8", "s1", "J-V1-rmp", None, s1_core_trained),
            ("s1 core4 d9-12", "s1", "J-multi", None, s1_core_heldout), ("s1 core4 d9-12", "s1", "J-multi-loop", 2, s1_core_heldout),
            ("s1 core4 d9-12", "s1", "J-V1-rmp", None, s1_core_heldout),
            ("s4", "s4", "J-multi", None, None), ("s4", "s4", "J-multi-loop", 2, None), ("s4", "s4", "J-V1", None, None),
            ("s2", "s2", "J-multi", None, None), ("s2", "s2", "J-multi-loop", 2, None),
            ("s7p", "s7p", "J-multi", None, None), ("s7p", "s7p", "J-multi-loop", 2, None), ("s7p", "s7p", "J-V1", None, None)]
    for label, suite, base, it, subset in rows:
        cells = [acc_line(records, suite, arm_id(base, s, it), subset) for s in SEEDS]
        lines.append(f"| {label} | {base}{'' if it is None else f'-r{it}'} | " + " | ".join(cells) + " |")
    lines += ["", "Single-seed comparators: J-V0 S1 d1-8 " + acc_line(records, "s1", "J-V0", s1_core_trained)
              + "; LOOP-T-ds S1 d1-8 " + acc_line(records, "s1", "LOOP-T-ds", s1_core_trained)
              + ", d9-12 " + acc_line(records, "s1", "LOOP-T-ds", s1_core_heldout)
              + "; J-cot-V0 d9-12 " + acc_line(records, "s1", "J-cot-V0", s1_core_heldout)
              + "; J-V0 S2 " + acc_line(records, "s2", "J-V0") + "; J-V0 S7p " + acc_line(records, "s7p", "J-V0") + "."]

    lines += ["", "## Calibration (ECE / NLL) by seed", "", "| Suite | Arm | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|---|"]
    for suite in ("s1", "s2", "s4", "s7p"):
        for base, it in (("J-multi", None), ("J-multi-loop", 2), ("J-V1", None)):
            cells = []
            for s in SEEDS:
                rec = records.get(suite, {}).get(arm_id(base, s, it))
                cal = (rec or {}).get("metrics", {}).get("calibration") if rec and "not_applicable" not in rec else None
                cells.append(f"{cal['ece']:.3f} / {cal['nll']:.3f}" if cal else "-")
            lines.append(f"| {suite} | {base}{'' if it is None else f'-r{it}'} | " + " | ".join(cells) + " |")

    lines += ["", "## Readiness of J-multi-loop (pointer chasing, v1 rule, item bootstrap)", ""]
    for s in SEEDS:
        rec = records.get("s1", {}).get(arm_id("J-multi-loop", s, 3))
        if not rec or "not_applicable" in rec:
            lines.append(f"- seed {s}: pending")
            continue
        info = readiness(rec).get("pointer_chase", {})
        interval = info.get("rho_interval")
        lines.append(f"- seed {s}: ready at {info.get('ready') or 'never'}; rho = {fmt(info.get('spearman'))} "
                     f"({'[%.2f, %.2f]' % interval if interval else 'undefined'}); depth-indexed "
                     f"{'yes' if info.get('depth_indexed') else 'no'} (in {info.get('share_depth_indexed', 0):.0%} of resamples)")

    lines += ["", "## S3 (seed 0, zero-shot, descriptive)", "", "| Arm | Accuracy [95% CI] | Utility | Unsafe |", "|---|---|---|---|"]
    for arm in ("J-multi", "J-multi-LDT", "J-V1", "J-V1-LDT", "J-V0", "J-V0-LDT", "LDT", "ControlTRM-LDT"):
        rec = records.get("s3", {}).get(arm)
        if not rec or "not_applicable" in rec:
            continue
        m = rec["metrics"]
        k = round(m["accuracy"] * m["n"])
        lines.append(f"| {arm} | {m['accuracy']:.3f} {ci(k, m['n'])} | {fmt(m.get('utility_mean'))} | {m['unsafe']['k']}/{m['unsafe']['n']} |")

    lines += ["", "## Reliability (v1 rule; pool = v1 registered arms + U1 seed-0 arms; S7 uses S7p for text arms)", "",
              "| Suite | Best arm | Reliable J arms (all pool) | Reliable J arms (neural pool) | J-multi: R1 / R2 / underpowered |", "|---|---|---|---|---|"]
    for suite, pool in reliability_pools(records).items():
        if not pool:
            continue
        rel_all, rel_neural = reliability(suite, pool, neural_only=False), reliability(suite, pool, neural_only=True)
        if not rel_all:
            continue
        best = next(iter(rel_all.values()))["best"]
        j_all = [a for a, v in rel_all.items() if a.startswith("J-") and v["reliable"]]
        j_neural = [a for a, v in rel_neural.items() if a.startswith("J-") and v["reliable"]]
        jm = rel_all.get("J-multi", {})
        jm_text = f"{jm.get('r1')} / {jm.get('r2')} / {jm.get('underpowered')}" if jm else "-"
        lines.append(f"| {suite} | {best} | {', '.join(j_all) or 'none'} | {', '.join(j_neural) or 'none'} | {jm_text} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    records = load_records()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u1.md").write_text(render(records), encoding="utf-8")
    print(f"wrote {OUT / 'u1.md'}")


if __name__ == "__main__":
    sys.exit(main())
