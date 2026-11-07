"""SPEC-U8: bandit losses, small-base arms, training configs."""

import torch

from jevq.training import bandit_loss, brier, build_items, load_train_config
from jevq.xbench import arms


def _mean_grad(fn, logits: torch.Tensor, draws: int) -> torch.Tensor:
    total = torch.zeros_like(logits)
    for _ in range(draws):
        x = logits.clone().requires_grad_(True)
        fn(x).backward()
        total += x.grad
    return total / draws


def test_rlcd_expected_gradient_is_the_brier_gradient():
    torch.manual_seed(0)
    logits, answer = torch.tensor([0.3, -0.4, 1.1, 0.0]), 1
    x = logits.clone().requires_grad_(True)
    brier(x, answer).backward()
    mc = _mean_grad(lambda z: bandit_loss(z, answer, "rlcd"), logits, 40_000)
    assert torch.allclose(mc, x.grad, atol=4e-3), (mc, x.grad)


def test_rlvr_expected_gradient_is_minus_twice_the_accuracy_gradient():
    torch.manual_seed(1)
    logits, answer = torch.tensor([0.3, -0.4, 1.1, 0.0]), 1
    x = logits.clone().requires_grad_(True)
    (-2.0 * torch.softmax(x, -1)[answer]).backward()
    mc = _mean_grad(lambda z: bandit_loss(z, answer, "rlvr"), logits, 40_000)
    assert torch.allclose(mc, x.grad, atol=4e-3), (mc, x.grad)


def test_small_base_arms():
    cfg = arms.model_cfg_for(arms.J_ARMS["J-rlcd-0.8B"])
    assert cfg["id"] == "Qwen/Qwen3.5-0.8B-Base" and cfg["revision"].startswith("dc7cdfe2")
    assert cfg["dtype"] == "bfloat16"
    assert arms.model_cfg_for(arms.J_ARMS["J-multi"]) is None
    s2 = arms.seeded("J-multi-2B", 2)
    assert s2.adapter == "u8_multi2b_s2/adapter" and s2.base == "2B" and s2.arm_id == "J-multi-2B-s2"
    assert set(arms.U8_ARMS) <= set(arms.U1_ARMS)


def test_u8_arms_stay_out_of_the_v1_pools():
    import sys

    sys.path.insert(0, "scripts")
    from xbench_report import U_SERIES_PREFIXES

    for arm in arms.U8_ARMS:
        for record in (arm, f"{arm}-s1"):
            assert record.startswith(U_SERIES_PREFIXES)
    for v1 in ("J-V0", "J-V1", "J-V1c", "J-V2b-r2", "J-V1-s1"):
        assert not v1.startswith(U_SERIES_PREFIXES)


def test_u8_configs():
    rlcd = load_train_config("configs/train/u8_rlcd08.yaml")
    assert rlcd["name"] == "u8_rlcd08" and rlcd["loss"]["choice"] == "rlcd"
    assert "model.id=Qwen/Qwen3.5-0.8B-Base" in rlcd["run_overrides"]
    assert rlcd["data"]["sources"][0]["formats"] == ["choice"]
    assert [s["kind"] for s in rlcd["data"]["sources"]] == ["dev", "rmp", "s2", "s7"]
    multi, multi08 = load_train_config("configs/train/multi_lora.yaml"), load_train_config("configs/train/u8_multi08.yaml")
    assert {k: v for k, v in multi08.items() if k not in ("name", "run_overrides")} == {k: v for k, v in multi.items() if k != "name"}
    assert load_train_config("configs/train/u8_rlvr08.yaml")["loss"]["choice"] == "rlvr"


def test_choice_only_rows_are_j_multis_rows():
    multi = load_train_config("configs/train/multi_lora.yaml")["data"]["sources"][0]
    choice = load_train_config("configs/train/u8_brier08.yaml")["data"]["sources"][0]
    a, _, _ = build_items(multi, 0, 4)
    b, _, _ = build_items(choice, 0, 4)
    assert [i.example.id for i in a] == [i.example.id for i in b]
    assert {i.fmt for i in b} == {"choice"} and len(b) == 2400
