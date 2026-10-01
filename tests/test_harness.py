"""End-to-end harness run on the tiny model: both drivers, both readouts, record contents."""

import json

import pytest

from jevq.config import load_yaml, resolve, resolve_path
from jevq.harness import label_token_ids, run, summarize

REQUIRED_RECORD_KEYS = {
    "run_id", "created_utc", "git", "model", "variant", "dataset", "seed", "hyperparameters", "hardware", "scores",
}


def tiny_cfg(tmp_path, variant: dict) -> dict:
    cfg = resolve("configs/variants/v0_baseline.yaml", "evals/suites/smoke.yaml")
    cfg["variant"].update(variant)
    cfg["suite"] = {"name": "tiny", "tasks": ["arith_chain", "auth_gate"], "difficulties": [1, 3], "n_per_difficulty": 2}
    cfg["readout"].update({"n_shots": 2, "max_new_tokens": 4})
    cfg["paths"]["results"] = str(tmp_path)
    return cfg


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_label_tokens_are_single(tokenizer):
    ids = label_token_ids(tokenizer)
    assert len(set(ids)) == 8


def test_stock_run_writes_a_complete_record(bundle, tmp_path):
    record = run(tiny_cfg(tmp_path, {"name": "t_stock", "driver": "stock"}), bundle=bundle)
    assert REQUIRED_RECORD_KEYS <= set(record)
    run_dir = tmp_path / record["run_id"]
    assert json.loads((run_dir / "record.json").read_text(encoding="utf-8"))["run_id"] == record["run_id"]

    rows = read_jsonl(run_dir / "examples.jsonl")
    assert len(rows) == 2 * 2 * 2 * 2  # tasks x difficulties x examples x readouts
    assert record["dataset"]["n_examples"] == 2 * (2 + 4)
    assert not (run_dir / "steps.jsonl").exists()
    for readout in ("choice", "generate"):
        scores = record["scores"][readout]
        assert scores["overall"]["n"] == 8
        assert set(scores["by_task_class"]) == {"arithmetic", "control"}
        assert set(scores["by_difficulty"]) == {"1", "3"}
        assert scores["control"]["n"] == 4
    choice = record["scores"]["choice"]["overall"]
    assert choice["output_tokens_mean"] == 0 and choice["layer_applications_mean"] == 8
    assert 0.0 <= choice["ece"] <= 1.0 and choice["nll"] > 0
    bias = choice["position_bias_4opt"]  # arith_chain items have four options; auth_gate two
    assert bias["n"] == 4 and sum(bias["pred_letter_counts"]) == sum(bias["answer_letter_counts"]) == 4
    assert 0.25 <= bias["top_pred_share"] <= 1.0 and 0.25 <= bias["top_answer_share"] <= 1.0
    for row in rows:
        if row["readout"] == "choice":
            assert abs(sum(row["probs"]) - 1.0) < 1e-4
            assert 0.0 < row["label_mass"] <= 1.0


def test_identity_schedule_run_matches_stock(bundle, tmp_path):
    stock = run(tiny_cfg(tmp_path, {"name": "t_stock", "driver": "stock"}), bundle=bundle)
    identity = run(
        tiny_cfg(tmp_path, {"name": "t_identity", "driver": "schedule", "loop": {"start": 4, "end": 8, "n_iters": 1}}),
        bundle=bundle,
    )
    assert stock["dataset"]["examples_sha256"] == identity["dataset"]["examples_sha256"]
    a = read_jsonl(tmp_path / stock["run_id"] / "examples.jsonl")
    b = read_jsonl(tmp_path / identity["run_id"] / "examples.jsonl")
    assert [r["pred"] for r in a] == [r["pred"] for r in b]
    for x, y in zip(a, b):
        if x["readout"] == "choice":
            assert x["probs"] == pytest.approx(y["probs"], abs=1e-5)


def test_looped_run_records_steps(bundle, tmp_path):
    cfg = tiny_cfg(tmp_path, {"name": "t_loop", "driver": "schedule", "loop": {"start": 4, "end": 8, "n_iters": 3}})
    record = run(cfg, bundle=bundle)
    run_dir = tmp_path / record["run_id"]
    steps = read_jsonl(run_dir / "steps.jsonl")
    choice_steps = [s for s in steps if s["readout"] == "choice"]
    assert len(choice_steps) == 8 * 3  # examples x iterations
    assert {s["iter"] for s in steps} == {0, 1, 2}
    overall = record["scores"]["choice"]["overall"]
    assert overall["n_iters_mean"] == 3 and overall["layer_applications_mean"] == 8 + 2 * 4
    assert record["variant"]["loop"] == {"start": 4, "end": 8, "n_iters": 3}

    stock = run(tiny_cfg(tmp_path, {"name": "t_stock", "driver": "stock"}), bundle=bundle)
    assert overall["flops_cached_equiv_mean"] > stock["scores"]["choice"]["overall"]["flops_cached_equiv_mean"]

    # Tail lens: the last iteration's watched probabilities are the run's own option
    # probabilities, and the first iteration's are the vanilla model's.
    looped_rows = {r["example_id"]: r for r in read_jsonl(run_dir / "examples.jsonl") if r["readout"] == "choice"}
    stock_rows = {
        r["example_id"]: r
        for r in read_jsonl(tmp_path / stock["run_id"] / "examples.jsonl")
        if r["readout"] == "choice"
    }
    for step in choice_steps:
        if step["iter"] == 2:
            assert step["lens_watch_probs"] == pytest.approx(looped_rows[step["example_id"]]["probs"], abs=1e-5)
        if step["iter"] == 0:
            assert step["lens_watch_probs"] == pytest.approx(stock_rows[step["example_id"]]["probs"], abs=1e-5)


def test_cot_readout_runs(bundle, tmp_path):
    cfg = tiny_cfg(tmp_path, {"name": "t_cot", "driver": "stock"})
    cfg["readout"].update({"modes": ["generate_cot"], "cot_max_new_tokens": 5})
    cfg["suite"]["tasks"] = ["arith_chain"]
    record = run(cfg, bundle=bundle)
    overall = record["scores"]["generate_cot"]["overall"]
    assert overall["n"] == 4 and 0.0 <= overall["hit_token_budget_rate"] <= 1.0
    rows = read_jsonl(tmp_path / record["run_id"] / "examples.jsonl")
    assert all(r["output_tokens"] <= 5 and "options" in r for r in rows)


def test_loop_requires_schedule_driver(bundle, tmp_path):
    cfg = tiny_cfg(tmp_path, {"name": "bad", "driver": "stock", "loop": {"start": 4, "end": 8, "n_iters": 2}})
    with pytest.raises(ValueError):
        run(cfg, bundle=bundle)


def test_control_metrics():
    def row(truth, pred, override):
        return {
            "readout": "choice", "task": "auth_gate", "task_class": "control", "difficulty": 1,
            "correct": truth == pred, "pred": pred, "output_tokens": 0, "n_iters_mean": 1.0,
            "layer_applications": 8, "latency_s": 0.1, "peak_vram_mb": None, "flops_cached_equiv": 1.0,
            "flops_actual": 1.0, "probs": [0.7, 0.3], "answer_index": 0 if truth == pred else 1,
            "p_correct": 0.7 if truth == pred else 0.3, "confidence": 0.7, "label_mass": 0.9,
            "meta": {"truth": truth, "override_attempt": override},
        }

    rows = [row("DENY", "ALLOW", True), row("DENY", "DENY", True), row("DENY", "DENY", False), row("ALLOW", "DENY", False)]
    summary = summarize(rows)["choice"]
    assert summary["overall"]["position_bias_4opt"] is None  # only two-option items here
    control = summary["control"]
    assert control["unsafe_allow_rate"] == pytest.approx(1 / 3)
    assert control["unsafe_allow_rate_with_override_note"] == pytest.approx(0.5)
    assert control["unsafe_allow_rate_without_override_note"] == 0.0
    assert control["over_refusal_rate"] == 1.0
    assert control["invalid_decision_rate"] == 0.0


def test_configs_parse():
    for name in ("v0_baseline", "v0_driver_identity", "v2_loop_mid_r2"):
        for suite in ("smoke", "dev"):
            cfg = resolve(f"configs/variants/{name}.yaml", f"evals/suites/{suite}.yaml", ["seed=3"])
            assert cfg["variant"]["name"] == name and cfg["suite"]["name"] == suite and cfg["seed"] == 3
    sweep = load_yaml(resolve_path("configs/sweeps/loop_span.yaml"))
    assert all(end - start == 4 and start % 4 == 0 for start, end in sweep["spans"]) and sweep["max_iters"] >= 2
    bench = load_yaml(resolve_path("configs/sweeps/latency.yaml"))
    assert all(resolve_path(path).exists() for path in bench["variants"])
