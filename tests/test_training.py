"""Training path on the tiny random Qwen3.5 (CPU): data, targets, losses, optimisation, resume,
the loop driver, and loading the adapter back into the harness."""

import copy
import json

import pytest
import torch

from jevq.config import resolve
from jevq.harness import label_token_ids, run
from jevq.modeling import attach_adapter, make_bundle
from jevq.tasks.base import CHOICE_INSTRUCTION, COT_INSTRUCTION, GENERATE_INSTRUCTION, LABELS
from jevq.training import FORMATS, brier, build_items, encode, eval_questions, load_train_config, lr_at, train


def data_cfg(**overrides) -> dict:
    cfg = load_train_config("configs/train/v1_lora.yaml")["data"]
    cfg.update({"tasks": ["arith_chain", "auth_gate"], "n_per_difficulty": 6, "n_val_per_difficulty": 2})
    cfg.update(overrides)
    return cfg


def tiny_train_cfg(**overrides) -> dict:
    cfg = load_train_config("configs/train/v1_lora.yaml")
    # Three validation questions per task, so the round-robin covers every format.
    cfg["data"] = data_cfg(difficulties=[1], n_per_difficulty=4, n_val_per_difficulty=3)
    cfg["lora"]["r"] = 4
    cfg["lora"]["dropout"] = 0.1  # non-zero, so resume must restore the RNG to match exactly
    cfg["optim"].update({"lr": 5e-3, "warmup_steps": 1, "grad_accum": 2, "epochs": 2})
    cfg.update({"eval_every": 2, "save_every": 2, "gradient_checkpointing": True})
    cfg.update(overrides)
    return cfg


def fresh_bundle(tiny_model, tokenizer):
    return make_bundle(copy.deepcopy(tiny_model), tokenizer, {"id": "tiny-random-qwen3_5"})


def adapter_tensors(path) -> dict:
    from safetensors.torch import load_file

    return load_file(path / "adapter" / "adapter_model.safetensors")


# ------------------------------------------------------------------ data


def test_train_data_is_disjoint_from_evaluation_and_held_out_difficulties():
    cfg = data_cfg(tasks=None, n_per_difficulty=20, n_val_per_difficulty=3)
    train_items, val_items, info = build_items(cfg, seed=0, n_shots=4)
    held_out = eval_questions(0, cfg["exclude_suites"], 4)
    questions = [i.example.question for i in train_items + val_items]
    assert len(questions) == len(set(questions))
    assert not set(questions) & held_out
    assert {i.example.difficulty for i in train_items + val_items} == {1, 2, 3}
    assert len(train_items) == 8 * 3 * 20 and len(val_items) == 8 * 3 * 3
    assert set(info["formats"]) == set(FORMATS) and max(info["formats"].values()) - min(info["formats"].values()) <= 1
    for task in {i.example.task for i in train_items}:  # and every task sees every format
        assert {i.fmt for i in train_items if i.example.task == task} == set(FORMATS)
    assert info["n_eval_questions_excluded"] == len(held_out) > 1000

    again, _, info2 = build_items(cfg, seed=0, n_shots=4)
    assert [i.example.question for i in again] == [i.example.question for i in train_items]
    assert info2["train_sha256"] == info["train_sha256"]
    other, _, _ = build_items(cfg, seed=1, n_shots=4)
    assert [i.example.question for i in other] != [i.example.question for i in train_items]


def test_targets_match_inference_conditions(tokenizer):
    label_ids = label_token_ids(tokenizer)
    items, _, _ = build_items(data_cfg(), seed=0, n_shots=4)
    seen = set()
    for item in items:
        enc = encode(tokenizer, item, label_ids)
        seen.add(enc.fmt)
        instruction = {"choice": CHOICE_INSTRUCTION, "generate": GENERATE_INSTRUCTION, "generate_cot": COT_INSTRUCTION}
        assert tokenizer.decode(enc.input_ids[: enc.n_prompt]).startswith(instruction[enc.fmt])
        assert enc.input_ids[enc.n_prompt :] == enc.targets
        target_text = tokenizer.decode(enc.targets)
        if enc.fmt == "choice":
            assert enc.targets == [label_ids[item.example.answer_index]]
            assert target_text == " " + LABELS[item.example.answer_index]
            assert tokenizer.decode(enc.input_ids[: enc.n_prompt]).endswith("Answer:")
        elif enc.fmt == "generate":
            assert target_text == f" {item.example.answer}\n"
            assert tokenizer.decode(enc.input_ids[: enc.n_prompt]).endswith("\nA:")
        else:
            assert target_text == f" {item.example.meta['rationale']}\nA: {item.example.answer}\n"
            assert tokenizer.decode(enc.input_ids[: enc.n_prompt]).endswith("\nReasoning:")
    assert seen == set(FORMATS)


def test_brier():
    logits = torch.tensor([2.0, 0.0, -1.0])
    p = torch.softmax(logits, dim=-1)
    expected = (p[0] - 1) ** 2 + p[1] ** 2 + p[2] ** 2
    assert torch.isclose(brier(logits, 0), expected)
    assert brier(torch.tensor([50.0, -50.0]), 0).item() < 1e-6
    assert abs(brier(torch.tensor([-50.0, 50.0]), 0).item() - 2.0) < 1e-6


def test_lr_schedule():
    cfg = {"lr": 1.0, "warmup_steps": 4, "min_lr_ratio": 0.1}
    assert lr_at(0, 100, cfg) == 0.25 and lr_at(3, 100, cfg) == 1.0
    assert lr_at(4, 100, cfg) == pytest.approx(1.0)
    assert lr_at(99, 100, cfg) == pytest.approx(0.1, abs=1e-3)


def test_train_config_inheritance():
    v1 = load_train_config("configs/train/v1_lora.yaml")
    v1c = load_train_config("configs/train/v1c_lora.yaml")
    assert v1c["loss"]["choice"] == "brier" and v1["loss"]["choice"] == "ce"
    v1c_minus = {k: v for k, v in v1c.items() if k not in ("name", "loss")}
    v1_minus = {k: v for k, v in v1.items() if k not in ("name", "loss")}
    assert v1c_minus == v1_minus  # one-factor difference


# ------------------------------------------------------------------ optimisation


@pytest.mark.parametrize("choice_loss", ["ce", "brier"])
def test_training_lowers_the_loss_and_writes_an_adapter(tiny_model, tokenizer, tmp_path, choice_loss):
    cfg = tiny_train_cfg(loss={"choice": choice_loss})
    cfg["optim"]["epochs"] = 6
    record = train(cfg, bundle=fresh_bundle(tiny_model, tokenizer), out_dir=tmp_path)
    log = [json.loads(line) for line in open(tmp_path / "train_log.jsonl", encoding="utf-8")]
    assert len(log) == record["steps"] == 6 * 8 // 2
    first, last = sum(r["loss"] for r in log[:3]) / 3, sum(r["loss"] for r in log[-3:]) / 3
    assert last < first * 0.8, (first, last)
    assert "val_choice_accuracy" in log[-1] and "val_ce_generate" in log[-1]
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert not (tmp_path / "last").exists()
    assert record["model"]["trainable_params"] > 0
    assert record["data"]["n_train"] == 8 and record["config"]["loss"]["choice"] == choice_loss


def test_resume_reproduces_an_uninterrupted_run(tiny_model, tokenizer, tmp_path):
    cfg = tiny_train_cfg()
    train(cfg, bundle=fresh_bundle(tiny_model, tokenizer), out_dir=tmp_path / "straight")

    out = tmp_path / "interrupted"
    stopped = train(cfg, bundle=fresh_bundle(tiny_model, tokenizer), out_dir=out, stop_after=3)
    assert stopped == {"stopped_at": 3} and (out / "last" / "state.pt").exists()  # saved at step 2
    train(cfg, bundle=fresh_bundle(tiny_model, tokenizer), out_dir=out)

    a, b = adapter_tensors(tmp_path / "straight"), adapter_tensors(out)
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), key

    changed = copy.deepcopy(cfg)
    changed["optim"]["lr"] = 1e-3
    out2 = tmp_path / "mismatch"
    train(cfg, bundle=fresh_bundle(tiny_model, tokenizer), out_dir=out2, stop_after=3)
    with pytest.raises(SystemExit, match="different config"):
        train(changed, bundle=fresh_bundle(tiny_model, tokenizer), out_dir=out2)


def test_training_through_the_loop_driver(tiny_model, tokenizer, tmp_path):
    cfg = tiny_train_cfg(
        variant="configs/variants/v2_loop_mid_r2.yaml",
        run_overrides=["variant.loop.start=4", "variant.loop.end=8", "variant.loop.n_iters=2"],
    )
    record = train(cfg, bundle=fresh_bundle(tiny_model, tokenizer), out_dir=tmp_path)
    assert record["variant"]["loop"] == {"start": 4, "end": 8, "n_iters": 2}
    weights = adapter_tensors(tmp_path)
    assert any("layers.5." in key and "lora_B" in key and weights[key].abs().sum() > 0 for key in weights)


# ------------------------------------------------------------------ evaluation with the adapter


def test_harness_evaluates_a_trained_adapter(tiny_model, tokenizer, tmp_path):
    trained = fresh_bundle(tiny_model, tokenizer)
    train(tiny_train_cfg(), bundle=trained, out_dir=tmp_path / "ckpt")
    adapter_dir = tmp_path / "ckpt" / "adapter"

    loaded = attach_adapter(fresh_bundle(tiny_model, tokenizer), adapter_dir)
    assert loaded.info["adapter"] == str(adapter_dir)
    assert loaded.info["adapter_train"]["name"] == "v1_lora"

    ids = torch.randint(1, 100, (1, 20))
    with torch.no_grad():
        trained.model.eval()
        expected = trained.model(input_ids=ids, use_cache=False).logits
        got = loaded.model(input_ids=ids, use_cache=False).logits
        base = tiny_model(input_ids=ids, use_cache=False).logits
    assert torch.allclose(expected, got, atol=1e-5)
    assert not torch.allclose(base, got, atol=1e-4)

    cfg = resolve("configs/variants/v1_lora_s0.yaml", "evals/suites/smoke.yaml")
    cfg["variant"]["adapter"] = str(adapter_dir)
    cfg["suite"] = {"name": "tiny", "tasks": ["auth_gate"], "difficulties": [1], "n_per_difficulty": 2}
    cfg["readout"].update({"n_shots": 2, "max_new_tokens": 3, "modes": ["choice", "generate"]})
    cfg["paths"]["results"] = str(tmp_path / "results")
    record = run(cfg, bundle=loaded)
    assert record["model"]["adapter"] == str(adapter_dir)
    assert record["model"]["adapter_train"]["train_sha256"]

    with pytest.raises(ValueError, match="does not match"):
        run(cfg, bundle=fresh_bundle(tiny_model, tokenizer))
