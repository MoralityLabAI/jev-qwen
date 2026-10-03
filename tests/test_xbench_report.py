"""xbench_report: seed replicates and addendum arms stay out of the registered reliability pools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import xbench_report as report  # noqa: E402


def _record(arm_id: str, correct: list[bool], neural: bool = True) -> dict:
    items = [{"item_id": str(i), "correct": c, "family": "f", "unsafe": None} for i, c in enumerate(correct)]
    return {
        "arm": {"arm_id": arm_id, "neural": neural, "params_total": 4_000_000_000 if neural else None},
        "metrics": {"accuracy": sum(correct) / len(correct), "n": len(correct)},
        "_items": items,
    }


def test_replicates_and_controls_are_split_off():
    n = 60
    records = {"s4": {
        "J-V1": _record("J-V1", [True] * 50 + [False] * 10),
        "J-V1-s1": _record("J-V1-s1", [True] * 60),  # would be the best arm if it entered the pool
        "J-V1-s2": _record("J-V1-s2", [True] * 46 + [False] * 14),
        "J-V2b-s1-r2": _record("J-V2b-s1-r2", [True] * 30 + [False] * 30),
        "J-V1c": _record("J-V1c", [True] * 48 + [False] * 12),
        "J-V1c-lr3": _record("J-V1c-lr3", [True] * n),
    }}
    registered, addenda = report.split_addenda(records)
    assert set(registered["s4"]) == {"J-V1", "J-V1c"}
    assert set(addenda["s4"]) == {"J-V1-s1", "J-V1-s2", "J-V2b-s1-r2", "J-V1c-lr3"}
    rel = report.reliability("s4", registered["s4"], neural_only=True)
    assert rel["J-V1c"]["best"] == "J-V1"

    text = "\n".join(report.addenda_section(registered, addenda))
    assert "| s4 | J-V1 | 0.833, 1.000, 0.767 (seeds 0, 1, 2) | 0.867 |" in text
    assert "| s4 | J-V2b-r2 | 0.500 (seeds 1) |" in text  # no registered seed-0 J-V2b record here
    assert "| s4 | J-V1c-lr3 | J-V1c | 1.000 | 0.800 | 60 | 12 | 0 |" in text


def test_addenda_section_without_addenda():
    assert report.addenda_section({}, {})[-2] == "None yet."


def test_s7p_goes_to_addenda_with_position_shares():
    def s7_row(i, pos, correct):
        options = ["a", "b", "c", "d", "e", "ABSTAIN"]
        probs = [0.0] * 6
        probs[pos] = 1.0
        return {"item_id": str(i), "kind": "positive", "gold": "a", "correct": correct, "options": options, "probs": probs}

    s7 = {"arm": {"arm_id": "J-V1", "neural": True}, "metrics": {"accuracy": 1.0, "n": 4}, "_items": [s7_row(i, 0, True) for i in range(4)]}
    s7p = {"arm": {"arm_id": "J-V1", "neural": True}, "metrics": {"accuracy": 0.25, "n": 4},
           "_items": [s7_row(0, 0, True)] + [s7_row(i, 0, False) for i in (1, 2, 3)]}
    for k, row in enumerate(s7p["_items"]):
        row["options"] = row["options"][k % 5:5] + row["options"][: k % 5] + ["ABSTAIN"]
    registered, addenda = report.split_addenda({"s7": {"J-V1": s7}, "s7p": {"J-V1": s7p}})
    assert "s7p" not in registered and "J-V1" in addenda["s7p"]
    text = "\n".join(report.addenda_section(registered, addenda))
    assert "| J-V1 | 1.000 | 0.250 | 1.000 | 0.250 | 1.00 / 1.00 | 1.00 / 0.25 |" in text
