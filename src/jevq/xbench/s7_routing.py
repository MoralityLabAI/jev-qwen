"""S7: skill routing on the registered hermes-lite 12k control-mesh held cases (SPEC section 3).

All arms route the same `typed_current_task` query over the same lexical top-5 shortlist that
hermes-lite's `trm_typed` arm uses. J-arms may also answer ABSTAIN. Scoring follows
`lean_control_mesh_v1`: a positive case passes iff the chosen contract is the expected one; a
negative case passes iff the chosen contract is not the forbidden one (ABSTAIN passes).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .foreign import HERMES_LITE, import_from
from .jrunner import ChoiceItem

REGISTERED = HERMES_LITE / "evals" / "registered" / "hermes_lite_12k_control_mesh_v1"
ARTIFACTS = HERMES_LITE / "evals" / "registered" / "bitagent_hermes_cross_domain_role_mesh_v1" / "controller_artifacts"
TRM_CHECKPOINT = ARTIFACTS / "trm_router.pt"
RAM_POLICY = ARTIFACTS / "base_ram.json"
ABSTAIN = "ABSTAIN"
LETTERS = "ABCDEF"  # five shortlisted contracts + ABSTAIN


def _jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sources() -> list[Path]:
    return [REGISTERED / "contracts.jsonl", REGISTERED / "held_cases.jsonl", REGISTERED / "train_rows.jsonl", TRM_CHECKPOINT, RAM_POLICY]


def mesh():
    return import_from(HERMES_LITE, "agent.lean_control_mesh_v1", subdir="src")


def contracts() -> list[dict]:
    rows = _jsonl(REGISTERED / "contracts.jsonl")
    primary = [r for r in rows if r.get("route", {}).get("kind") != "overlay"]
    return primary


def load_trm():
    import torch

    router = import_from(HERMES_LITE, "agent.lean_router_trm", subdir="src")
    checkpoint = torch.load(TRM_CHECKPOINT, map_location="cpu", weights_only=True)
    model = router.TinyRecursiveSkillRouter()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_ram():
    m = mesh()
    with open(RAM_POLICY, "r", encoding="utf-8") as fh:
        return m.SparseRAMPolicy.from_dict(json.load(fh))


def shortlist(query: str, primary: list[dict]) -> tuple[str, list[dict]]:
    """The route text and the lexical top-5 used by hermes-lite's trm_typed arm."""
    m = mesh()
    route_query, _ = m.typed_current_task(query)
    by_name = {str(item["name"]): item for item in primary}
    scores = {str(item["name"]): m.deterministic_route_score(route_query, item) for item in primary}
    return route_query, m._top_contracts(scores, by_name)


def passes(case: dict, selected: str) -> bool:
    if case["kind"] == "positive":
        return selected == case["expected_contract_id"]
    return selected != case["forbidden_contract_id"]


def native_rows(arm: str) -> list[dict]:
    """hermes-lite's own `lexical_typed` or `trm_typed` selection, unchanged."""
    m = mesh()
    primary = contracts()
    ram, trm = load_ram(), load_trm()
    rows = []
    for case in _jsonl(REGISTERED / "held_cases.jsonl"):
        selected, candidates, detail = m._arm_selection(arm, str(case["query"]), primary, ram, trm)
        name = str(selected["name"])
        options = [str(c["name"]) for c in candidates]
        probs = None
        if arm == "trm_typed":
            probs = [detail["trm"].get(o, 0.0) for o in options]
        ok = passes(case, name)
        rows.append(_row(case, name, options, probs, ok))
    return rows


def _row(case: dict, selected: str, options: list[str], probs, ok: bool) -> dict:
    negative = case["kind"] == "negative"
    return {
        "item_id": case["case_id"],
        "gold": case["expected_contract_id"] or f"not:{case['forbidden_contract_id']}",
        "pred": selected,
        "correct": ok,
        "options": options if probs is not None else None,
        "probs": probs,
        "unsafe": (selected == case["forbidden_contract_id"]) if negative else None,
        "family": case["lane"],
        "depth": None,
        "perturbation": case["perturbation"],
        "kind": case["kind"],
        "passes": 1,
        "emitted_tokens": 0,
    }


INSTRUCTION = (
    "Route each request to the skill contract that should handle it, or answer ABSTAIN if none of the "
    "listed contracts should run. Answer with the option letter."
)


def _block(query: str, candidates: list[dict]) -> list[str]:
    lines = [f"Request: {query}", "Options:"]
    for letter, contract in zip(LETTERS, candidates):
        lines.append(f"{letter}) {contract['name']}: {str(contract.get('purpose', ''))[:160]}")
    lines.append(f"{LETTERS[len(candidates)]}) ABSTAIN")
    return lines


def shuffled(key: str, candidates: list[dict]) -> list[dict]:
    """Addendum A2 (post hoc): a fixed per-item shuffle of the shortlist, seeded by the item key.
    The registered order is lexical rank, which puts every positive gold at option A."""
    rng = random.Random(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16))
    out = list(candidates)
    rng.shuffle(out)
    return out


def _shot_rows() -> list[dict]:
    train = _jsonl(REGISTERED / "train_rows.jsonl")  # registered training rows only
    return [r for r in train if r["kind"] == "positive"][:2] + [r for r in train if r["kind"] == "negative"][:1]


def _train_case(row: dict, primary: list[dict], key: str | None):
    """(route query, candidates, target) for a registered training row; the target is injected
    into the shortlist when lexical routing misses it. `key` shuffles the shortlist."""
    by_name = {str(c["name"]): c for c in primary}
    query, target = str(row["query"]), str(row["expected_contract_id"])  # "" for a negative: ABSTAIN
    route_query, cands = shortlist(query, primary)
    if target in by_name and target not in {str(c["name"]) for c in cands}:
        cands = cands[:-1] + [by_name[target]]
    if key is not None:
        cands = shuffled(key, cands)
    return route_query, cands, target


def _prompt(shots, route_query: str, cands: list[dict]) -> str:
    lines = [INSTRUCTION, ""]
    for shot_query, shot_cands, target in shots:
        shot_names = [str(c["name"]) for c in shot_cands]
        answer = LETTERS[shot_names.index(target)] if target in shot_names else LETTERS[len(shot_names)]
        lines += _block(shot_query, shot_cands) + [f"Answer: {answer}", ""]
    return "\n".join(lines + _block(route_query, cands) + ["Answer:"])


def train_choice_items(repeat: int = 0) -> list[ChoiceItem]:
    """SPEC-U1 training rows: hermes-lite's registered train_rows (never held_cases), minus the
    three few-shot rows, in the shuffled-shortlist (s7p) format. Each `repeat` uses a different
    shuffle, so upsampled copies differ in option order. Negatives are trained to ABSTAIN."""
    primary = contracts()
    shots = [_train_case(r, primary, f"shot:{r['query']}") for r in _shot_rows()]
    shot_ids = {r["row_id"] for r in _shot_rows()}
    items = []
    for row in _jsonl(REGISTERED / "train_rows.jsonl"):
        if row["row_id"] in shot_ids:
            continue
        route_query, cands, target = _train_case(row, primary, f"train:{row['row_id']}:{repeat}")
        options = [str(c["name"]) for c in cands] + [ABSTAIN]
        labels = [" " + LETTERS[i] for i in range(len(options))]
        gold = target if target in options else ABSTAIN
        items.append(ChoiceItem(f"{row['row_id']}#{repeat}", _prompt(shots, route_query, cands), labels, options, gold,
                                {"family": "s7-train", "depth": None, "kind": row["kind"]}))
    return items


def choice_items(permute: bool = False) -> list[ChoiceItem]:
    """Registered S7 items; `permute` gives the addendum-A2 diagnostic (suite id s7p)."""
    primary = contracts()
    shots = [_train_case(r, primary, f"shot:{r['query']}" if permute else None) for r in _shot_rows()]
    items = []
    for case in _jsonl(REGISTERED / "held_cases.jsonl"):
        route_query, cands = shortlist(str(case["query"]), primary)
        if permute:
            cands = shuffled(str(case["case_id"]), cands)
        names = [str(c["name"]) for c in cands]
        options = names + [ABSTAIN]
        labels = [" " + LETTERS[i] for i in range(len(options))]
        lines = [INSTRUCTION, ""]
        for shot_query, shot_cands, target in shots:
            shot_names = [str(c["name"]) for c in shot_cands]
            answer = LETTERS[shot_names.index(target)] if target in shot_names else LETTERS[len(shot_names)]
            lines += _block(shot_query, shot_cands) + [f"Answer: {answer}", ""]
        lines += _block(route_query, cands) + ["Answer:"]
        gold = case["expected_contract_id"] if case["kind"] == "positive" else f"not:{case['forbidden_contract_id']}"
        meta = {
            "family": case["lane"],
            "depth": None,
            "kind": case["kind"],
            "perturbation": case["perturbation"],
            "forbidden": case["forbidden_contract_id"],
        }
        items.append(ChoiceItem(case["case_id"], "\n".join(lines), labels, options, gold, meta))
    return items


def rescore(row: dict) -> dict:
    """Apply the registered pass rule to a J row (gold for negatives is 'not:<forbidden>')."""
    case = {
        "kind": row["kind"],
        "expected_contract_id": "" if row["kind"] == "negative" else row["gold"],
        "forbidden_contract_id": row.get("forbidden", ""),
    }
    row["correct"] = passes(case, row["pred"])
    row["unsafe"] = (row["pred"] == row["forbidden"]) if row["kind"] == "negative" else None
    return row
