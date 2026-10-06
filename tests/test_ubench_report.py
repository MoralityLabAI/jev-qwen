"""ubench_report: seed naming, pooled one-sided tests, item-bootstrap non-inferiority, pending and
decided states with Holm, and the S7 pool."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ubench_report as u  # noqa: E402


def items(correct, prefix="i", **extra):
    return {f"{prefix}{k}": {"item_id": f"{prefix}{k}", "correct": c, **extra} for k, c in enumerate(correct)}


def record(arm, correct, **extra):
    rows = list(items(correct, **extra).values())
    return {"arm": {"arm_id": arm, "neural": True, "params_total": 1}, "metrics": {"accuracy": sum(correct) / len(correct), "n": len(correct)}, "_items": rows}


def test_arm_ids():
    assert u.arm_id("J-multi", 0) == "J-multi"
    assert u.arm_id("J-multi", 2) == "J-multi-s2"
    assert u.arm_id("J-multi-loop", 1, 2) == "J-multi-loop-s1-r2"
    assert u.arm_id("J-multi-loop", 0, 3) == "J-multi-loop-r3"


def test_binomial_and_pooled():
    assert u.binom_one_sided(0, 0) == 1.0
    assert u.binom_one_sided(3, 3) == pytest.approx(0.125)
    assert u.binom_one_sided(14, 14) == pytest.approx(2 ** -14)
    a, b = items([1, 1, 0, 1]), items([0, 1, 0, 0])
    r = u.pooled([(a, b, sorted(a)), (a, b, sorted(a))])
    assert (r["a_only"], r["b_only"], r["seeds"]) == (4, 0, 2) and r["p_one_sided"] == pytest.approx(1 / 16)


def test_noninferiority_bootstrap():
    same = items([1, 0] * 50)
    r = u.noninferiority([(same, same, sorted(same))], resamples=500)
    assert r["estimate"] == 0 and r["low"] == 0 and r["noninferior"]
    worse = items([0] * 10 + [1] * 90)
    better = items([1] * 100)
    r = u.noninferiority([(worse, better, sorted(worse))], resamples=500)
    assert r["estimate"] == pytest.approx(-0.10) and r["high"] < -0.02 and not r["noninferior"]
    with pytest.raises(ValueError):
        u.noninferiority([(same, same, sorted(same)), (same, same, sorted(same)[:-1])])


def test_registered_tests_pending_interim_decided_and_holm():
    s1 = dict(family="pointer_chase", depth=1)
    good, bad = [1] * 20, [0] * 20
    records = {"s1": {}, "s2": {}, "s4": {}, "s7p": {}}
    assert u.registered_tests(records)["T1"]["status"] == "pending"
    for seed in (0, 1, 2):
        records["s1"][u.arm_id("J-multi", seed)] = record("J-multi", good, **s1)
        records["s1"][u.arm_id("J-V1", seed)] = record("J-V1", bad, **s1)
        records["s7p"][u.arm_id("J-multi", seed)] = record("J-multi", good)
        records["s7p"][u.arm_id("J-V1", seed)] = record("J-V1", bad)
        records["s2"][u.arm_id("J-multi", seed)] = record("J-multi", good)
    records["s2"]["J-V0"] = record("J-V0", bad)
    out = u.registered_tests(records)
    assert out["T1"]["status"] == "decided" and out["T1"]["a_only"] == 60
    assert all("p_holm" in out[k] for k in ("T1", "T2", "T3"))
    assert out["L"]["status"] == "pending"
    del records["s7p"]["J-V1-s2"]
    out = u.registered_tests(records)
    assert out["T2"]["status"] == "interim" and out["T2"]["missing_seeds"] == [2] and "p_holm" not in out["T1"]


def test_s7_pool_uses_s7p_for_text_arms_and_s7_for_routers():
    records = {"s7": {"lexical-router": "lex", "SkillRouter-TRM": "trm", "J-V1": "registered-order"},
               "s7p": {"J-V1": "shuffled", "J-V1-s1": "replicate", "J-multi": "u1"}}
    pool = u.s7_pool(records)
    assert pool == {"J-V1": "shuffled", "J-multi": "u1", "lexical-router": "lex", "SkillRouter-TRM": "trm"}
