"""SPEC-U3 report: repair-rudder replication (part R) and the TRM router's abstain path (part A).

    python scripts/u3_report.py

Reads results/u3/rudder/<record>/<protocol>.jsonl, results/u2/<arm>/route.jsonl and
results/u3/s7/<arm>.jsonl; writes reports/ubench/u3.md.
"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import u2_report as u2r
from jevq.config import ROOT
from jevq.xbench import u2_joint as u
from jevq.xbench import u3_rudder as r
from jevq.xbench.stats import holm

RUDDER = ROOT / "results" / "u3" / "rudder"
S7 = ROOT / "results" / "u3" / "s7"
OUT = ROOT / "reports" / "ubench"
SEEDS = (0, 1, 2)

# arm -> (record pattern, description); "{s}" is the seed suffix ("" for seed 0).
RUDDER_ARMS = {
    "J-V0": ("J-V0", "generic small Qwen (Qwen3.5-4B-Base)"),
    "J-multi": ("J-multi{s}", "Jev (J-multi)"),
    "J-multi-loop": ("J-multi-loop{s}-r2", "looped Jev, 2 passes"),
    "Repair-TRM": ("Repair-TRM{s}", "planned repair TRM (TinyTRM, two heads)"),
    "kNN": ("kNN", "k = 5 on the TRM's features"),
    "lookup": ("lookup", "train majority per (failure label, after-arm keyword)"),
    "Qwen2.5-3B": ("Qwen2.5-3B", "published, Qwen2.5-3B-Instruct Q4"),
    "Qwen3.5-9B": ("Qwen3.5-9B", "published, Qwen3.5-9B Q4_K_M"),
    "Qwen3.5-27B": ("Qwen3.5-27B", "published, Qwen3.5-27B Q4_K_M"),
    "MeTTa-validator": ("MeTTa-validator", "MeTTa rules, reads the outcome bucket (ceiling)"),
}
PROTOCOL_ORDER = ("raw", "retrieval", "retrieval-published", "action-space", "static-gate", "features", "lookup", "validator")
RUDDER_TESTS = {
    "R1": (("J-multi", "retrieval"), ("J-V0", "retrieval"), "one", "Jev beats the generic small Qwen (leak-free retrieval)"),
    "R2": (("J-multi", "retrieval"), ("Repair-TRM", "features"), "two", "Jev vs the repair TRM (two-sided)"),
    "R3": (("J-multi", "retrieval-published"), ("Qwen3.5-27B", "retrieval-published"), "two",
           "Jev vs published Qwen3.5-27B, same leaky protocol (two-sided)"),
}
PIPELINES_A = {
    "B": ("SkillRouter-TRM", "TRM-cv", "TRM router picks, TRM gate decides (U2)"),
    "B-abstain": ("SkillRouter-TRM-abstain", "TRM-cv", "TRM router + fitted abstain head picks, TRM gate decides"),
    "B-threshold": ("SkillRouter-TRM-threshold", "TRM-cv", "TRM router + top-probability threshold picks, TRM gate decides"),
    "H": ("J-multi{s}", "TRM-cv", "Jev picks, TRM gate decides (U2)"),
}
A_TESTS = {
    "A1": ("B-abstain", "B", "one", "the abstain path improves the TRM pipeline"),
    "A2": ("B-abstain", "H", "two", "TRM router with abstain vs Jev router, same TRM gate (two-sided)"),
}


def suffix(seed: int) -> str:
    return "" if seed == 0 else f"-s{seed}"


def load_protocol(record: str, protocol: str) -> dict[str, dict] | None:
    path = RUDDER / record / f"{protocol}.jsonl"
    if not path.exists():
        return None
    return {row["key"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def rudder_instances(arm: str, protocol: str) -> list[tuple[int, dict]]:
    pattern, _ = RUDDER_ARMS[arm]
    seeds = SEEDS if "{s}" in pattern else (0,)
    out = []
    for seed in seeds:
        rows = load_protocol(pattern.format(s=suffix(seed)), protocol)
        if rows is not None:
            out.append((seed, rows))
    return out


def restrict(instances, splits) -> list[tuple[int, dict]]:
    return [(seed, {k: v for k, v in rows.items() if v["split"] in splits}) for seed, rows in instances]


def as_success(instances) -> list[tuple[int, dict]]:
    return [(seed, {k: {"success": v["joint_ok"]} for k, v in rows.items()}) for seed, rows in instances]


def frac(instances, field: str) -> tuple[int, int]:
    k = sum(bool(v[field]) for _, rows in instances for v in rows.values())
    n = sum(len(rows) for _, rows in instances)
    return k, n


def cell(instances, field: str) -> str:
    k, n = frac(instances, field)
    return f"{k / n:.3f}" if n else "-"


def render_rudder() -> list[str]:
    lines = ["## Part R: repair rudder", "",
             "88 rows: val_seen 34, holdout_seen 36, holdout_unseen_family 18. Registered tests use the 54 holdout rows. "
             "Rates pool every available seed. Any arm that picks from the six train repair actions is capped at "
             "70/88 joint (36/54 on the holdout rows): the unseen-family actions are not in that list.", "",
             "### Registered tests (joint accuracy, 54 holdout rows)", "", "| Test | Claim | Result | Status |", "|---|---|---|---|"]
    results, ps = {}, {}
    for tid, (a, b, sided, claim) in RUDDER_TESTS.items():
        ia, ib = rudder_instances(*a), rudder_instances(*b)
        if not ia or not ib:
            results[tid] = None
            continue
        res = u2r.pooled(as_success(restrict(ia, r.TEST_SPLITS)), as_success(restrict(ib, r.TEST_SPLITS)))
        results[tid] = (res, len(ia), len(ib))
        ps[tid] = res["p_one_sided"] if sided == "one" else res["p_two_sided"]
    decided = all(v is not None for v in results.values()) and len(rudder_instances("J-multi", "retrieval")) == 3 \
        and len(rudder_instances("Repair-TRM", "features")) == 3
    adjusted = holm(ps) if decided else {}
    for tid, (a, b, sided, claim) in RUDDER_TESTS.items():
        if results[tid] is None:
            lines.append(f"| {tid} | {claim} | - | pending |")
            continue
        res, na, nb = results[tid]
        p = ps[tid]
        holm_text = f", Holm p = {adjusted[tid]:.3g}" if tid in adjusted else ""
        status = "decided" if decided else "interim"
        lines.append(f"| {tid} | {claim} | {a[0]}-only {res['a_only']} vs {b[0]}-only {res['b_only']} "
                     f"({na} x {nb} instances), p = {p:.3g}{holm_text} | {status} |")

    lines += ["", "### Every arm and protocol", "",
              "| Arm | Protocol | Instances | Joint (88) | Repair (88) | Commit (88) | Unsafe (88) | Joint (54 holdout) | "
              "Joint val_seen | Joint holdout_seen | Joint unseen family |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, (_, desc) in RUDDER_ARMS.items():
        for protocol in PROTOCOL_ORDER:
            inst = rudder_instances(arm, protocol)
            if not inst:
                continue
            per_split = [cell(restrict(inst, (s,)), "joint_ok") for s in r.EVAL_SPLITS]
            lines.append(f"| {arm} ({desc}) | {protocol} | {len(inst)} | {cell(inst, 'joint_ok')} | {cell(inst, 'repair_ok')} | "
                         f"{cell(inst, 'action_ok')} | {cell(inst, 'unsafe')} | {cell(restrict(inst, r.TEST_SPLITS), 'joint_ok')} | "
                         + " | ".join(per_split) + " |")
    checks = RUDDER / "checks.json"
    if checks.exists():
        lines += ["", "### Port checks", "", "Share of published rows reproduced by this repo's ports:", ""]
        lines += [f"- {name}: {value:.3f}" for name, value in json.loads(checks.read_text(encoding="utf-8")).items()]
    return lines


def render_abstain(items) -> list[str]:
    saved = dict(u2r.PIPELINES)
    u2r.PIPELINES.update(PIPELINES_A)
    try:
        inst = {name: u2r.instances(items, name) for name in PIPELINES_A}
    finally:
        u2r.PIPELINES.clear()
        u2r.PIPELINES.update(saved)
    lines = ["## Part A: abstain path for the TRM router", "", "U2's 600 items (360 joint, 150 confusable, 90 negative), scored as in SPEC-U2.", "",
             "### Registered tests (end-to-end success)", "", "| Test | Claim | Result | Status |", "|---|---|---|---|"]
    results, ps = {}, {}
    for tid, (a, b, sided, claim) in A_TESTS.items():
        if not inst[a] or not inst[b]:
            results[tid] = None
            continue
        res = u2r.pooled(inst[a], inst[b])
        results[tid] = res
        ps[tid] = res["p_one_sided"] if sided == "one" else res["p_two_sided"]
    decided = all(v is not None for v in results.values()) and len(inst["H"]) == 3
    adjusted = holm(ps) if decided else {}
    for tid, (a, b, sided, claim) in A_TESTS.items():
        res = results[tid]
        if res is None:
            lines.append(f"| {tid} | {claim} | - | pending |")
            continue
        holm_text = f", Holm p = {adjusted[tid]:.3g}" if tid in adjusted else ""
        lines.append(f"| {tid} | {claim} | {a}-only {res['a_only']} vs {b}-only {res['b_only']}, p = {ps[tid]:.3g}{holm_text} | "
                     f"{'decided' if decided else 'interim'} |")
    lines += ["", "### Pipelines", "", "| Pipeline | Router -> gate | Instances | End-to-end success | Unsafe | Route ok (joint) | "
              "Route ok (confusable) | Route ok (negative) |", "|---|---|---|---|---|---|---|---|"]
    for name, (_, _, desc) in PIPELINES_A.items():
        res = inst[name]
        if not res:
            lines.append(f"| {name} | {desc} | 0 | pending | | | | |")
            continue
        cells = [u2r.fmt_rate(*u2r.rate(res, "success")), u2r.fmt_rate(*u2r.rate(res, "unsafe"))]
        cells += [u2r.fmt_rate(*u2r.rate(res, "route_ok", {kind})) for kind in ("joint", "confusable", "negative")]
        lines.append(f"| {name} | {desc} | {len(res)} | " + " | ".join(cells) + " |")
    lines += ["", "### Registered S7 held cases (128: 105 positive, 23 negative)", "", "| Router | Positives pass | Negatives pass | Abstained |",
              "|---|---|---|---|"]
    for arm in ("SkillRouter-TRM-abstain", "SkillRouter-TRM-threshold"):
        path = S7 / f"{arm}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        pos = [x for x in rows if x["kind"] == "positive"]
        neg = [x for x in rows if x["kind"] == "negative"]
        abst = sum(x["pred"] == "ABSTAIN" for x in rows)
        lines.append(f"| {arm} | {sum(x['correct'] for x in pos)}/{len(pos)} | {sum(x['correct'] for x in neg)}/{len(neg)} | {abst} |")
    arm_json = ROOT / "results" / "u2" / "SkillRouter-TRM-abstain" / "arm.json"
    if arm_json.exists():
        info = json.loads(arm_json.read_text(encoding="utf-8"))
        lines += ["", f"Abstain-head fit: {json.dumps(info.get('abstain_fit'))}. Threshold tau = {info.get('threshold')}."]
    return lines


def render(items) -> str:
    lines = ["# SPEC-U3 report: repair rudder and the TRM router's abstain path", "",
             "Generated by `scripts/u3_report.py`. Registration: `docs/ubench/SPEC-U3.md`.", ""]
    lines += render_rudder() + [""] + render_abstain(items)
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u3.md").write_text(render(u.build_items()), encoding="utf-8")
    print(f"wrote {OUT / 'u3.md'}")


if __name__ == "__main__":
    main()
