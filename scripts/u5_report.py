"""SPEC-U5 report: reject paths (part R), the Jev trained on the U4 train items (part J), and real
model proposals on standard and larger puzzles (part P).

    python scripts/u5_report.py

Reads results/u5/... and, for the U4 test set's earlier arms, results/u4/...; writes reports/ubench/u5.md.
"""

from __future__ import annotations

import json
import statistics

import _bootstrap  # noqa: F401
import u2_report as u2r
from jevq.config import ROOT
from jevq.xbench import u5_campsite as u5
from jevq.xbench.stats import holm, wilson

U4_RUNS = ROOT / "results" / "u4"
U5_RUNS = ROOT / "results" / "u5"
OUT = ROOT / "reports" / "ubench"
SEEDS = (0, 1, 2)
BANDS = ("std", "8x8", "10x10")

# (part, set, arm) -> record pattern; "{s}" is the seed suffix.
SEEDED = {"J-u4": "J-u4{s}", "Decision-TRM": "Decision-TRM{s}", "J-multi": "J-multi{s}"}
TESTS = {
    "J1": (("decide", "u4test", "J-u4"), ("decide", "u4test", "Decision-TRM"), "two", "trained Jev decider vs Decision-TRM (U4 test)"),
    "J2": (("repair", "u4test", "J-u4"), ("repair", "u4test", "dual_repair"), "two", "trained Jev-written repair vs dual_repair (U4 test)"),
    "J3": (("decide", "u4test", "J-u4"), ("decide", "u4test", "J-multi"), "one", "training on U4 items improves the Jev decider (U4 test)"),
    "P1": (("decide", "proposals", "J-u4"), ("decide", "proposals", "Decision-TRM"), "two", "trained Jev decider vs Decision-TRM (real proposals)"),
    "P2": (("repair", "proposals", "J-u4"), ("repair", "proposals", "c_repair"), "two", "trained Jev-written repair vs c_repair (real proposals)"),
}


def suffix(seed: int) -> str:
    return "" if seed == 0 else f"-s{seed}"


def path_for(part: str, set_name: str, record: str):
    if set_name == "u4test" and not record.startswith("J-u4"):
        return U4_RUNS / part / f"{record}.jsonl"  # SPEC-U4's own records on its test set
    return U5_RUNS / part / set_name / f"{record}.jsonl"


def instances(part: str, set_name: str, arm: str) -> list[tuple[int, dict]]:
    pattern = SEEDED.get(arm, arm)
    out = []
    for seed in (SEEDS if "{s}" in pattern else (0,)):
        path = path_for(part, set_name, pattern.format(s=suffix(seed)))
        if path.exists():
            out.append((seed, {r["item_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}))
    return out


def select(inst, pred) -> list[tuple[int, dict]]:
    return [(s, {k: v for k, v in rows.items() if pred(v)}) for s, rows in inst]


def rate(inst, key: str) -> tuple[int, int]:
    rows = [r for _, res in inst for r in res.values()]
    return sum(bool(r.get(key)) for r in rows), len(rows)


def fmt(k: int, n: int, ci: bool = False) -> str:
    if not n:
        return "-"
    if not ci:
        return f"{k / n:.3f}"
    low, high = wilson(k, n)
    return f"{k / n:.3f} [{low:.2f}, {high:.2f}]"


def tests_section() -> list[str]:
    lines = ["## Registered tests", "", "| Test | Claim | Result | Status |", "|---|---|---|---|"]
    results, ps = {}, {}
    for tid, (a, b, sided, claim) in TESTS.items():
        ia, ib = instances(*a), instances(*b)
        if not ia or not ib:
            results[tid] = None
            continue
        res = u2r.pooled(ia, ib)
        results[tid] = (res, len(ia), len(ib))
        ps[tid] = res["p_one_sided"] if sided == "one" else res["p_two_sided"]
    decided = all(v is not None for v in results.values()) and all(len(instances("decide", s, "J-u4")) == 3 for s in ("u4test", "proposals"))
    adjusted = holm(ps) if decided else {}
    for tid, (a, b, sided, claim) in TESTS.items():
        if results[tid] is None:
            lines.append(f"| {tid} | {claim} | - | pending |")
            continue
        res, na, nb = results[tid]
        holm_text = f", Holm p = {adjusted[tid]:.3g}" if tid in adjusted else ""
        ra, rb = fmt(*rate(instances(*a), "success")), fmt(*rate(instances(*b), "success"))
        lines.append(f"| {tid} | {claim} | {ra} vs {rb}; {a[2]}-only {res['a_only']} vs {b[2]}-only {res['b_only']} ({na} x {nb}), "
                     f"p = {ps[tid]:.3g}{holm_text} | {'decided' if decided else 'interim'} |")
    return lines


def decide_table(set_name: str, arms: list[str], groups: dict) -> list[str]:
    head = " | ".join(f"{g}: success / unsafe" for g in groups)
    lines = [f"| Arm | Instances | Success | Unsafe | Rejected | {head} |", "|---|---|---|---|---|" + "---|" * len(groups)]
    for arm in arms:
        inst = instances("decide", set_name, arm)
        if not inst:
            continue
        cells = []
        for pred in groups.values():
            sub = select(inst, pred)
            cells.append(f"{fmt(*rate(sub, 'success'))} / {fmt(*rate(sub, 'unsafe'))}")
        lines.append(f"| {arm} | {len(inst)} | {fmt(*rate(inst, 'success'), ci=True)} | {fmt(*rate(inst, 'unsafe'))} | "
                     f"{fmt(*rate(inst, 'rejected'))} | " + " | ".join(cells) + " |")
    return lines


def repair_table(set_name: str, arms: list[str], groups: dict) -> list[str]:
    lines = [f"| Arm | Instances | Success | " + " | ".join(groups) + " | Median seconds | Timeouts / skipped |",
             "|---|---|---|" + "---|" * len(groups) + "---|---|"]
    for arm in arms:
        inst = instances("repair", set_name, arm)
        if not inst:
            continue
        rows = [r for _, res in inst for r in res.values()]
        secs = [r.get("seconds") if r.get("seconds") is not None else r.get("latency_s") for r in rows]
        secs = [s for s in secs if s is not None]
        status = f"{sum(r.get('status') == 'timeout' for r in rows)} / {sum(r.get('status') == 'skipped' for r in rows)}"
        cells = [fmt(*rate(select(inst, pred), "success")) for pred in groups.values()]
        median = f"{statistics.median(secs):.3f}" if secs else "-"
        lines.append(f"| {arm} | {len(inst)} | {fmt(*rate(inst, 'success'), ci=True)} | " + " | ".join(cells) + f" | {median} | {status} |")
    return lines


def render() -> str:
    lines = ["# SPEC-U5 report: reject paths, the U4-trained Jev, real proposals", "",
             "Generated by `scripts/u5_report.py`. Registration: `docs/ubench/SPEC-U5.md`.", ""]
    lines += tests_section()

    policies = [f"{p}+{m}" for p in u5.POLICIES for m in u5.REJECT_MODES]
    deciders = policies + ["Decision-TRM", "J-u4", "menu-ceiling"]
    lines += ["", "## Part R: reject paths", "",
              "noop-reject never commits an unchanged candidate that the verifier has already failed (no extra verifier call); "
              "verify-reject commits only a grid that passes the verifier. Neither can lower success: both only withhold commits "
              "that would fail.", "", "### Fresh synthetic test set (U5 test, 480 items)", ""]
    lines += decide_table("u5test", deciders, {"correct": lambda r: r["kind"] == "correct",
                                               "unfixable types": lambda r: r["kind"] in ("swap_rect", "bad_shape")})
    by_band = {b: (lambda r, b=b: r["kind"] == b) for b in BANDS}
    lines += ["", "### Real proposals, by puzzle size", ""] + decide_table("proposals", deciders, by_band)

    lines += ["", "## Part J: the Jev trained on the U4 train items", "", "### U4 test set (480; earlier arms from SPEC-U4)", ""]
    lines += decide_table("u4test", ["J-u4", "J-multi", "Decision-TRM", "menu-ceiling"], {"all": lambda r: True})
    lines += [""] + repair_table("u4test", ["J-u4", "dual_repair", "c_repair", "J-multi", "identity"], {"all": lambda r: True})

    lines += ["", "## Part P: real proposals", ""]
    props = [json.loads(x) for p in u5.PROPOSERS if (U5_RUNS / "proposals" / f"{p}.jsonl").exists()
             for x in (U5_RUNS / "proposals" / f"{p}.jsonl").read_text(encoding="utf-8").splitlines()]
    inst = instances("repair", "proposals", "identity")
    if inst:
        rows = list(inst[0][1].values())
        lines += ["Raw proposal quality (the proposal passes the verifier as written):", "",
                  "| Proposer | " + " | ".join(BANDS) + " | Parsed as a full grid |", "|---|" + "---|" * len(BANDS) + "---|"]
        for proposer in u5.PROPOSERS:
            cells = []
            for b in BANDS:
                sub = [r for r in rows if r["proposer"] == proposer and r["kind"] == b]
                cells.append(fmt(sum(r["success"] for r in sub), len(sub)))
            sub = [r for r in rows if r["proposer"] == proposer]
            cells.append(fmt(sum(r["edits_to_candidate"] is not None for r in sub), len(sub)))
            lines.append(f"| {proposer} | " + " | ".join(cells) + " |")
    lines += ["", "Repairers on the proposals:", ""]
    lines += repair_table("proposals", ["identity", "c_repair", "dual_repair", "J-u4", "CSP-resolve"], by_band)
    lines += ["", f"Proposal rows: {len(props)}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u5.md").write_text(render(), encoding="utf-8")
    print(f"wrote {OUT / 'u5.md'}")


if __name__ == "__main__":
    main()
