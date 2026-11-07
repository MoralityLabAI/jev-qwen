"""SPEC-U8 report: small Jevs (docs/ubench/SPEC-U8.md section 4) and descriptives.

    python scripts/u8_report.py

Writes reports/ubench/u8.md. A test is decided only when every seed of its arms is present;
until then it is reported as interim on the seeds that are.
"""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401
from jevq.xbench.stats import ece, holm
from ubench_report import OUT, SEEDS, arm_id, binom_one_sided, items_of, noninferiority, s1_core_heldout, s1_core_trained
from xbench_report import ci, load_records

POOL = ("s1", "s2", "s4", "s7p")
SUITES = ("s1", "s2", "s4", "s7", "s7p")
UNSEEDED = ("J-V0", "J-V0-0.8B", "J-V0-2B")
ARMS = ("J-V0-0.8B", "J-V0-2B", "J-V0", "J-multi-0.8B", "J-multi-2B", "J-multi", "J-V1c", "J-brier-0.8B", "J-rlcd-0.8B", "J-rlvr-0.8B")
RESAMPLES = 10_000


def pooled_items(records, base: str, seed: int) -> dict | None:
    """P for one arm at one seed: S1 core4 depths 1-8, S2, S4, S7p, keyed `suite:item_id`."""
    out = {}
    for suite in POOL:
        items = items_of(records, suite, base if base in UNSEEDED else arm_id(base, seed))
        if items is None:
            return None
        ids = s1_core_trained(items) if suite == "s1" else sorted(items)
        out.update({f"{suite}:{i}": items[i] for i in ids})
    return out


def seeds_of(records, base: str) -> dict[int, dict]:
    return {s: p for s in ((0,) if base in UNSEEDED else SEEDS) if (p := pooled_items(records, base, s)) is not None}


def paired(records, a: str, b: str) -> tuple[list, list[int]]:
    """(a, b, shared ids) per seed of `a`; an unseeded `b` pairs with every seed."""
    sa, sb = seeds_of(records, a), seeds_of(records, b)
    pairs, missing = [], []
    for seed in SEEDS:
        a_items, b_items = sa.get(seed), sb.get(0 if b in UNSEEDED else seed)
        if a_items is None or b_items is None:
            missing.append(seed)
            continue
        pairs.append((a_items, b_items, sorted(set(a_items) & set(b_items))))
    if pairs:
        ids = sorted(set.intersection(*(set(p[2]) for p in pairs)))
        pairs = [(x, y, ids) for x, y, _ in pairs]
    return pairs, missing


def superiority(pairs) -> dict:
    a_only = sum(bool(a[i]["correct"]) and not b[i]["correct"] for a, b, ids in pairs for i in ids)
    b_only = sum(bool(b[i]["correct"]) and not a[i]["correct"] for a, b, ids in pairs for i in ids)
    return {"a_only": a_only, "b_only": b_only, "p_one_sided": binom_one_sided(a_only, a_only + b_only)}


def _conf_ok(items: dict, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([max(items[i]["probs"]) for i in ids]), np.array([bool(items[i]["correct"]) for i in ids], dtype=float)


def _weighted_ece(conf: np.ndarray, ok: np.ndarray, w: np.ndarray, bins: int = 10) -> float:
    """stats.ece with integer weights (bootstrap multiplicities); bins (lo, hi], 0 in the first."""
    b = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    n = np.bincount(b, weights=w, minlength=bins)
    gap = np.bincount(b, weights=w * (ok - conf), minlength=bins)
    return float(np.abs(gap).sum() / n.sum())


def ece_pooled(seed_items: list[dict], ids: list[str]) -> float:
    conf, ok = zip(*(_conf_ok(items, ids) for items in seed_items))
    return float(ece(list(np.concatenate(conf)), list(np.concatenate(ok).astype(bool))))


def ece_diff_bootstrap(records, a: str, b: str) -> dict:
    """ECE(a) - ECE(b) over P with seeds pooled; item bootstrap, one resample of ids for every seed."""
    pairs, missing = paired(records, a, b)
    if not pairs:
        return {"status": "pending", "missing_seeds": missing}
    ids = pairs[0][2]
    arrays = {k: [_conf_ok(p[k], ids) for p in pairs] for k in (0, 1)}
    conf = {k: np.concatenate([c for c, _ in arrays[k]]) for k in arrays}
    ok = {k: np.concatenate([o for _, o in arrays[k]]) for k in arrays}
    n, s = len(ids), len(pairs)
    point = _weighted_ece(conf[0], ok[0], np.ones(n * s)) - _weighted_ece(conf[1], ok[1], np.ones(n * s))
    rng = np.random.default_rng(0)
    boots = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        w = np.tile(np.bincount(rng.integers(0, n, n), minlength=n).astype(float), s)
        boots[r] = _weighted_ece(conf[0], ok[0], w) - _weighted_ece(conf[1], ok[1], w)
    return {"estimate": point, "p05": float(np.percentile(boots, 5)), "p95": float(np.percentile(boots, 95)),
            "seeds": s, "status": "decided" if not missing else "interim", "missing_seeds": missing}


def registered(records) -> dict:
    out = {}
    for key, a, b in (("Z1", "J-multi-2B", "J-multi"), ("Z2", "J-multi-0.8B", "J-multi"), ("R2", "J-rlcd-0.8B", "J-brier-0.8B")):
        pairs, missing = paired(records, a, b)
        out[key] = ({**noninferiority(pairs), "status": "decided" if not missing else "interim", "missing_seeds": missing}
                    if pairs else {"status": "pending", "missing_seeds": missing})
    for key, a, b in (("Z3", "J-multi-0.8B", "J-V0-0.8B"), ("R1", "J-rlcd-0.8B", "J-V0-0.8B")):
        pairs, missing = paired(records, a, b)
        out[key] = ({**superiority(pairs), "seeds": len(pairs), "status": "decided" if not missing else "interim", "missing_seeds": missing}
                    if pairs else {"status": "pending", "missing_seeds": missing})
    out["R3"] = ece_diff_bootstrap(records, "J-rlvr-0.8B", "J-rlcd-0.8B")
    out["R4"] = ece_diff_bootstrap(records, "J-rlcd-0.8B", "J-brier-0.8B")
    decided = {k: out[k]["p_one_sided"] for k in ("Z3", "R1") if out[k].get("status") == "decided"}
    if len(decided) == 2:
        for k, p in holm(decided).items():
            out[k]["p_holm"] = p
    return out


def verdict(key: str, r: dict) -> str:
    if r.get("status") == "pending":
        return "pending"
    if key in ("Z1", "Z2", "R2"):
        held = r["noninferior"]
    elif key in ("Z3", "R1"):
        held = r.get("p_holm", r["p_one_sided"]) < 0.05
    elif key == "R3":
        held = r["p05"] > 0
    else:
        held = r["p95"] < 0.02
    return ("holds" if held else "does not hold") + ("" if r["status"] == "decided" else f" (interim, seeds missing {r['missing_seeds']})")


def detail(key: str, r: dict) -> str:
    if r.get("status") == "pending":
        return f"missing seeds {r['missing_seeds']}"
    if key in ("Z1", "Z2", "R2"):
        return f"diff {r['estimate']:+.3f}, 95% [{r['low']:+.3f}, {r['high']:+.3f}], {r['seeds']} seeds"
    if key in ("Z3", "R1"):
        holm_p = f", Holm {r['p_holm']:.2g}" if "p_holm" in r else ""
        return f"{r['a_only']} vs {r['b_only']} discordant, p {r['p_one_sided']:.2g}{holm_p}, {r['seeds']} seeds"
    return f"ECE diff {r['estimate']:+.3f}, 5-95% [{r['p05']:+.3f}, {r['p95']:+.3f}], {r['seeds']} seeds"


CLAIMS = {
    "Z1": "J-multi-2B non-inferior to J-multi (4B) on P",
    "Z2": "J-multi-0.8B non-inferior to J-multi (4B) on P",
    "Z3": "J-multi-0.8B > J-V0-0.8B on P",
    "R1": "J-rlcd-0.8B > J-V0-0.8B on P",
    "R2": "J-rlcd-0.8B non-inferior to J-brier-0.8B on P",
    "R3": "ECE(J-rlvr-0.8B) > ECE(J-rlcd-0.8B) on P",
    "R4": "ECE(J-rlcd-0.8B) - ECE(J-brier-0.8B) < 0.02 on P",
}


def suite_cell(records, suite: str, base: str, subset=None) -> str:
    vals = []
    for seed in ((0,) if base in UNSEEDED else SEEDS):
        items = items_of(records, suite, base if base in UNSEEDED else arm_id(base, seed))
        if items is None:
            continue
        ids = subset(items) if subset else sorted(items)
        vals.append((sum(bool(items[i]["correct"]) for i in ids), len(ids), ece_pooled([items], ids)))
    if not vals:
        return "-"
    k, n = sum(v[0] for v in vals), sum(v[1] for v in vals)
    return f"{k / n:.3f} {ci(k, n)} / {sum(v[2] for v in vals) / len(vals):.3f} ({len(vals)})"


def render(records) -> str:
    tests = registered(records)
    lines = ["# SPEC-U8 report: small Jevs", "", "Generated by `scripts/u8_report.py` from `results/xbench`. "
             "Spec: `docs/ubench/SPEC-U8.md`.", "", "## Registered tests", "", "| ID | Claim | Verdict | Detail |", "|---|---|---|---|"]
    lines += [f"| {k} | {CLAIMS[k]} | {verdict(k, r)} | {detail(k, r)} |" for k, r in tests.items()]
    lines += ["", "## Pooled set P (S1 core4 d1-8, S2, S4, S7p)", "", "| Arm | Seeds | Accuracy | ECE (seeds pooled) |", "|---|---|---|---|"]
    for base in ARMS:
        per_seed = seeds_of(records, base)
        if not per_seed:
            lines.append(f"| {base} | 0 | - | - |")
            continue
        ids = sorted(set.intersection(*(set(p) for p in per_seed.values())))
        k = sum(bool(p[i]["correct"]) for p in per_seed.values() for i in ids)
        n = len(ids) * len(per_seed)
        lines.append(f"| {base} | {len(per_seed)} | {k / n:.3f} {ci(k, n)} | {ece_pooled(list(per_seed.values()), ids):.3f} |")
    lines += ["", "## By suite: accuracy [Wilson 95%] / mean ECE over seeds (seeds)", "",
              "| Arm | " + " | ".join(SUITES) + " | S1 depths 9+ |", "|---|" + "---|" * (len(SUITES) + 1)]
    for base in ARMS:
        cells = [suite_cell(records, s, base, s1_core_trained if s == "s1" else None) for s in SUITES]
        lines.append(f"| {base} | " + " | ".join(cells) + f" | {suite_cell(records, 's1', base, s1_core_heldout)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    records = load_records()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u8.md").write_text(render(records), encoding="utf-8")
    print(f"wrote {OUT / 'u8.md'}")


if __name__ == "__main__":
    sys.exit(main())
