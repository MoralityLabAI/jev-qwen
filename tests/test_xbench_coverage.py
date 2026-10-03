"""xbench coverage: the registered matrix, structural not-applicable cells, and fill-in."""

import json

from jevq.xbench import coverage
from jevq.xbench.records import write_not_applicable


def test_structural_reasons_never_cover_registered_runs():
    assert coverage.structural_reason("LOOP-T", "s1") is None
    assert "RMP token vocabulary" in coverage.structural_reason("LOOP-T", "s4")
    assert coverage.structural_reason("J-V1", "s3") is None  # registered: queued, not N/A
    assert coverage.structural_reason("J-V2b", "s3") == "Not registered for this suite (SPEC section 4)."
    assert coverage.structural_reason("J-cot-V1", "s2") == coverage.COT_NA
    assert coverage.structural_reason("J-cot-V1", "s6") is None
    assert coverage.structural_reason("CH-q0..q4", "s6") is None
    assert coverage.structural_reason("Bonsai-8B", "s1") is None


def test_fill_structural_and_any_records(tmp_path):
    write_not_applicable("any", "RMP-COT", "never built", root=tmp_path)
    live = tmp_path / "s1" / "LOOP-T"
    live.mkdir(parents=True)
    (live / "record.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    before = coverage.coverage(tmp_path)
    assert before[("RMP-COT", "s5")] == ("na", "never built")
    assert before[("LOOP-T", "s1")] == ("live", "s1/LOOP-T")
    assert before[("LOOP-T", "s2")][0] == "missing"

    written = coverage.fill_structural(tmp_path)
    after = coverage.coverage(tmp_path)
    assert after[("LOOP-T", "s2")][0] == "na"
    assert all((tmp_path / "s4" / cid / "record.json").exists() for cid in coverage.CH_IDS)
    assert ("J-V1", "s1") not in {(a, s) for a, s, _ in written}
    still_missing = {cell for cell, (state, _) in after.items() if state == "missing"}
    assert all(coverage.structural_reason(*cell) is None for cell in still_missing)
    assert ("J-V1", "s1") in still_missing and ("Bonsai-8B", "s5") in still_missing
