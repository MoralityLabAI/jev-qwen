"""SPEC-U2 runs: every router on every routing item, every gate on every joint item's gate state.
Pipelines are composed afterwards per item (scripts/u2_report.py).

Outputs: results/u2/<arm>/{route,gate}.jsonl and arm.json. Text routers see the lexical top-5
shortlist (hermes-lite's, no gold injection) in a per-item shuffled order plus ABSTAIN, with the
S7p few-shot rows; text gates see the S2 near-miss prompt and shots.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ROOT
from . import s2_commit_veto as s2
from . import s7_routing as s7
from . import u2_joint as u
from .jrunner import ChoiceItem

OUT = ROOT / "results" / "u2"


def route_items(items: list[dict]) -> list[ChoiceItem]:
    primary = s7.contracts()
    shots = [s7._train_case(r, primary, f"shot:{r['query']}") for r in s7._shot_rows()]
    out = []
    for item in items:
        route_query, cands = s7.shortlist(item["query"], primary)
        cands = s7.shuffled(item["item_id"], cands)
        options = [str(c["name"]) for c in cands] + [s7.ABSTAIN]
        labels = [" " + s7.LETTERS[i] for i in range(len(options))]
        out.append(ChoiceItem(item["item_id"], s7._prompt(shots, route_query, cands), labels, options,
                              item["gold_skill"] or f"not:{item['forbidden']}", {"kind": item["kind"], "variant": item["variant"]}))
    return out


def gate_items(items: list[dict]) -> list[ChoiceItem]:
    shots = s2.pick_shots(s2.load("near_miss")["train"])
    return [ChoiceItem(i["item_id"], s2.choice_prompt(shots, {"state": i["gate_state"]}), s2.LABELS, s2.OPTIONS, i["gold_gate"],
                       {"kind": "joint"}) for i in items if i["kind"] == "joint"]


def write(arm: str, route: list[dict] | None, gate: list[dict] | None, info: dict) -> Path:
    out = OUT / arm
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("route", route), ("gate", gate)):
        if rows is not None:
            (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "arm.json").write_text(json.dumps({"arm": arm, **info}, indent=2), encoding="utf-8")
    return out


def _slim(row: dict) -> dict:
    keep = ("item_id", "gold", "pred", "probs", "options", "label_mass", "latency_s", "kind", "variant", "raw_text")
    return {k: row[k] for k in keep if k in row}


def run_cpu() -> None:
    """Lexical and TRM routers (hermes-lite native selection); script, kNN and TRM-cv gates."""
    items = u.build_items()
    m = s7.mesh()
    primary, ram, trm = s7.contracts(), s7.load_ram(), s7.load_trm()
    for arm, native in (("lexical-router", "lexical_typed"), ("SkillRouter-TRM", "trm_typed")):
        rows = []
        for item in items:
            selected, _, _ = m._arm_selection(native, str(item["query"]), primary, ram, trm)
            rows.append({"item_id": item["item_id"], "pred": str(selected["name"]), "kind": item["kind"], "variant": item["variant"]})
        write(arm, rows, None, {"neural": arm != "lexical-router", "native_arm": native})
    data = s2.load("near_miss")
    predict, info = s2.train_trm_cv(data["train"], data["val_seen"])
    joint = [i for i in items if i["kind"] == "joint"]
    gates = {
        "script": lambda st: (s2.script_gate(st), None),
        "kNN-critic": lambda st: s2.knn_predict(data["train"], st),
        "TRM-cv": predict,
    }
    for arm, fn in gates.items():
        rows = []
        for item in joint:
            pred, probs = fn(item["gate_state"])
            rows.append({"item_id": item["item_id"], "gold": item["gold_gate"], "pred": pred, "probs": probs})
        write(arm, None, rows, {"neural": arm == "TRM-cv", "trained_on": "S2 near-miss train (val_seen for selection)" if arm != "script" else None,
                                **({"trm_cv": info} if arm == "TRM-cv" else {})})


def run_j(arm_id: str, seed: int = 0, iterations: int | None = None) -> None:
    from ..looped import LoopSpec
    from .arms import LOOP_SPAN, seeded
    from .jrunner import load_j_bundle, run_choice

    arm = seeded(arm_id, seed)
    record_id = arm.arm_id + (f"-r{iterations}" if iterations else "")
    loop = LoopSpec(*LOOP_SPAN, iterations) if iterations else None
    bundle = load_j_bundle(arm.adapter)
    items = u.build_items()
    route = [_slim(r) for r in run_choice(bundle, route_items(items), loop=loop)]
    gate = [_slim(r) for r in run_choice(bundle, gate_items(items), loop=loop)]
    write(record_id, route, gate, {"neural": True, "adapter": arm.adapter, "loop_iterations": iterations, "base_arm": arm_id, "seed": seed})


def run_bonsai() -> None:
    from . import bonsai

    items = u.build_items()
    with bonsai.BonsaiServer() as server:
        rows = [_slim(r) for r in bonsai.run_choice(server, route_items(items))]
    write("Bonsai-8B", rows, None, {"neural": True, "model": "Bonsai-8B-Q1_0 (llama-server, raw completion)"})
