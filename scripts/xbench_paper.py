"""Write the paper's xbench tables and plots from the records (registered arms only).

    python scripts/xbench_paper.py

Writes paper/generated/{s1_table,gate_table,halflife_table,pareto}.tex; main.tex \\input-s them,
so refreshing after a queue finishes is this command plus a rebuild. Uses the same loading and
addenda split as scripts/xbench_report.py.
"""

from __future__ import annotations

import math

import _bootstrap  # noqa: F401
from jevq.config import ROOT
from xbench_report import cost, load_records, reliability, split_addenda, unsafe_rate

OUT = ROOT / "paper" / "generated"


def tex(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def num(x, d=3) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def params_text(record: dict) -> str:
    p = cost(record)[0]
    if math.isnan(p):
        return "?"
    if p == 0:
        return "0"
    for unit, scale in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if p >= scale:
            return f"{p / scale:.{2 if unit == 'B' else 1}f}{unit}".replace(".0k", "k")
    return f"{p:.0f}"


def live(records: dict) -> dict:
    return {a: r for a, r in records.items() if "not_applicable" not in r and "accuracy" in r.get("metrics", {})}


def s1_table(registered) -> str:
    records = live(registered.get("s1", {}))
    rel_all = reliability("s1", records, neural_only=False)
    rel_neural = reliability("s1", records, neural_only=True)
    rows = []
    for arm, record in records.items():
        # Core4 only: masked_pointer_chase is J-only (SPEC section 3), so every row shares one item set.
        items = [r for r in record.get("_items") or [] if r.get("family") != "masked_pointer_chase"]
        idr = [r["correct"] for r in items if r.get("depth") is not None and r["depth"] <= 8]
        ood = [r["correct"] for r in items if r.get("depth") is not None and r["depth"] > 8]
        id_acc = sum(idr) / len(idr) if idr else None
        ood_acc = sum(ood) / len(ood) if ood else None
        flag = ("all" if rel_all.get(arm, {}).get("reliable") else "neural" if rel_neural.get(arm, {}).get("reliable") else "--")
        rows.append((-(id_acc or 0), arm, params_text(record), num(cost(record)[1], 1), num(id_acc), num(ood_acc),
                     num(sum(r["correct"] for r in items) / len(items) if items else None), flag))
    lines = [r"\begin{tabular}{lrrrrrl}", r"\toprule",
             r"Arm & Params & Passes & Depth 1--8 & Depth 9--12 & All & Reliable \\", r"\midrule"]
    lines += [f"{tex(arm)} & {p} & {ps} & {a} & {b} & {c} & {f} \\\\" for _, arm, p, ps, a, b, c, f in sorted(rows)]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def gate_table(registered) -> str:
    """One row per arm with any gate record: S2 commit/veto, S4 auth_gate, S5 monitor episodes."""
    arms = sorted({a for s in ("s2", "s5") for a in live(registered.get(s, {}))})
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{S2 commit/veto} & \multicolumn{3}{c}{S5 monitor} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Arm & Acc. & Unsafe & Over-ref. & Attack succ. & False blocks & ECE \\", r"\midrule"]
    for arm in arms:
        cells = []
        for suite in ("s2", "s5"):
            record = live(registered.get(suite, {})).get(arm)
            if record is None:
                cells += ["--"] * (3 if suite == "s2" else 2)
                continue
            m = record["metrics"]
            k, n = unsafe_rate(record, suite)
            o = [r for r in record["_items"] if r.get("over_refusal") is not None]
            ok = sum(bool(r["over_refusal"]) for r in o)
            if suite == "s2":
                cells += [num(m["accuracy"]), f"{k}/{n}", f"{ok}/{len(o)}"]
            else:
                ep = m.get("episodes") or {}
                cells += [num(ep.get("attack_success_rate"), 2), f"{ok}/{len(o)}"]
        ece = (live(registered.get("s5", {})).get(arm, {}).get("metrics", {}).get("calibration") or {}).get("ece")
        lines.append(f"{tex(arm)} & " + " & ".join(cells) + f" & {num(ece)} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def halflife_table(registered) -> str:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Ladder & Gate & Targets & Excluded & Flipped by 10 & Half-life \\", r"\midrule"]
    names = {"s6": "scripted, S4", "s6c": "scripted, S5", "s6a": "adaptive, S4"}
    for suite in ("s6", "s6c", "s6a"):
        for arm, record in sorted(registered.get(suite, {}).items()):
            m = record.get("metrics", {})
            if "survival" not in m:
                continue
            hl = m["half_life"] if m["half_life"] is not None else "$>10$"
            lines.append(f"{names[suite]} & {tex(arm)} & {m['n_targets']} & {m['excluded_wrong_at_turn0']} & "
                         f"{m.get('flipped_by_turn10', '--')} & {hl} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


# Arm families for the frontier plot: (label, mark style). Order is the legend order.
FAMILIES = [
    ("script / rule (0 params)", "mark=square*, black"),
    ("TRM gate (6k--17k)", "mark=triangle*, orange!90!black, mark size=2.6pt"),
    ("looped decoder (0.46--1.8M)", "mark=diamond*, green!50!black, mark size=2.8pt"),
    ("4B single pass", "mark=*, blue!80!black"),
    ("4B reasoning", "mark=o, blue!80!black, thick"),
    ("8B LLM, 1-bit", "mark=x, red!80!black, thick, mark size=3pt"),
]


def family(arm: str, record: dict) -> int:
    if arm.startswith("J-cot"):
        return 4
    if arm.startswith("J-"):
        return 3
    if arm.startswith("Bonsai"):
        return 5
    if arm.startswith(("LOOP-T", "FF-U")):
        return 2
    if "TRM" in arm:
        return 1
    return 0 if not record["arm"].get("neural") else 1


def pareto(registered, suites=("s1", "s4", "s7")) -> str:
    """Accuracy against log10(resident params x passes), one panel per suite in a row, one mark per
    arm family and a shared legend (per-point labels overlap where the 4B arms cluster)."""
    panels = []
    for index, suite in enumerate(suites):
        records = live(registered.get(suite, {}))
        points: dict[int, list[str]] = {}
        for arm, record in sorted(records.items()):
            params, passes = cost(record)
            if math.isnan(params):
                continue
            x = math.log10(max(params, 1) * passes) if params else 0.0
            points.setdefault(family(arm, record), []).append(f"({x:.2f},{record['metrics']['accuracy']:.3f})")
        opts = f"title={{{suite.upper()}}}" + (r", ylabel={accuracy}, legend to name=paretolegend, legend columns=3,"
                                                r" legend style={font=\footnotesize, draw=none, /tikz/every even column/.append style={column sep=0.8em}}"
                                                if index == 0 else "")
        panel = [rf"\nextgroupplot[{opts}]"]
        if index == 0:  # legend images independent of which families have points in this panel
            for label, style in FAMILIES:
                panel += [rf"\addlegendimage{{only marks, {style}}}", rf"\addlegendentry{{{label}}}"]
        for fam, coords in sorted(points.items()):
            panel.append(rf"\addplot[only marks, {FAMILIES[fam][1]}, forget plot] coordinates {{" + " ".join(coords) + "};")
        panels += panel
    return "\n".join([
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=" + str(len(suites)) + r" by 1, horizontal sep=0.9cm},"
        r" width=0.37\linewidth, height=4.4cm, xmin=-0.7, xmax=12, ymin=0, ymax=1.05, grid=major,"
        r" xlabel={$\log_{10}$(params $\times$ passes)}, tick label style={font=\scriptsize},"
        r" label style={font=\small}, title style={font=\small}]",
        *panels,
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        "",
        r"\smallskip\ref{paretolegend}",
        "",
    ])


def u1_table(records) -> str:
    """SPEC-U1 table: accuracy averaged over the seeds present (count in brackets)."""
    from ubench_report import SEEDS, arm_id, items_of, s1_core_heldout, s1_core_trained

    columns = [("s1", s1_core_trained, "S1 d1--8"), ("s1", s1_core_heldout, "S1 d9--12"), ("s2", None, "S2"),
               ("s4", None, "S4"), ("s7p", None, "S7 (shuffled)")]
    arms = [("J-V0", None, False, "V0 (base)"), ("J-V1", None, True, "V1 (decision suite)"),
            ("J-V1-rmp", None, True, "V1-rmp (S1 rows)"), ("J-multi", None, True, "multi"),
            ("J-multi-loop", 2, True, "multi-loop, 2 passes")]
    lines = [r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule",
             "Arm & " + " & ".join(c[2] for c in columns) + r" \\", r"\midrule"]
    for base, iteration, seeded, label in arms:
        cells = []
        for suite, subset, _ in columns:
            accs = []
            for seed in (SEEDS if seeded else (0,)):
                items = items_of(records, suite, arm_id(base, seed, iteration))
                if items is None:
                    continue
                ids = subset(items) if subset else sorted(items)
                accs.append(sum(bool(items[i]["correct"]) for i in ids) / len(ids))
            cells.append(f"{sum(accs) / len(accs):.3f}" + (f" ({len(accs)})" if seeded else "") if accs else "--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main() -> None:
    all_records = load_records()
    registered, _ = split_addenda(all_records)
    OUT.mkdir(parents=True, exist_ok=True)
    header = "% Generated by scripts/xbench_paper.py from results/xbench; do not edit.\n"
    for name, body in (("s1_table", s1_table(registered)), ("gate_table", gate_table(registered)),
                       ("halflife_table", halflife_table(registered)), ("pareto", pareto(registered)),
                       ("u1_table", u1_table(all_records))):
        (OUT / f"{name}.tex").write_text(header + body, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
