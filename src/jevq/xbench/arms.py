"""J-arm registry (SPEC section 2). Adapter paths are relative to paths.checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

from ..looped import LoopSpec

LOOP_SPAN = (12, 16)

# SPEC-U8 smaller bases of the same family (Qwen3_5, hybrid DeltaNet, full attention every 4th
# layer), pinned at the Hub commits read 2026-11-07. None = configs/base.yaml (Qwen3.5-4B-Base).
BASES = {
    "0.8B": {"id": "Qwen/Qwen3.5-0.8B-Base", "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"},
    "2B": {"id": "Qwen/Qwen3.5-2B-Base", "revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"},
}
U8_SUITES = ("s1", "s2", "s4", "s7")  # s7p runs with s7


@dataclass(frozen=True)
class JArm:
    arm_id: str
    adapter: str | None
    loop_iters: int | None = None  # None = stock driver; N = loop over LOOP_SPAN, recorded per iteration
    readout: str = "choice"  # choice | cot
    trained_on: str | None = None
    suites: tuple[str, ...] = ("s1", "s2", "s3", "s4", "s5", "s6", "s7")
    base: str | None = None  # key of BASES; None = the 4B base

    @property
    def loop(self) -> LoopSpec | None:
        return LoopSpec(*LOOP_SPAN, self.loop_iters) if self.loop_iters else None


def model_cfg_for(arm: JArm) -> dict | None:
    """configs/base.yaml's model block with the arm's base swapped in; None keeps the 4B base."""
    if arm.base is None:
        return None
    from ..config import load_yaml, resolve_path

    model = dict(load_yaml(resolve_path("configs/base.yaml"))["model"])
    model.update(BASES[arm.base])
    return model


J_ARMS = {
    arm.arm_id: arm
    for arm in (
        JArm("J-V0", None),
        JArm("J-V1", "v1_lora_s0/adapter", trained_on="jev-qwen train split (3 formats)"),
        JArm("J-V1c", "v1c_lora_s0/adapter", trained_on="jev-qwen train split (3 formats), Brier on choice"),
        # Learning-rate control for V1c (configs/train/v1c_lr3.yaml); S4 only.
        JArm("J-V1c-lr3", "v1c_lr3_s0/adapter", trained_on="as J-V1c with a 3x learning rate", suites=("s4",)),
        # One run at 3 iterations; the tail lens gives the exact r1 and r2 outputs (instrument.py).
        JArm("J-V2b", "v2b_lora_s0/adapter", loop_iters=3, trained_on="jev-qwen train split, loop 12-15 x2",
             suites=("s1", "s2", "s4", "s5", "s6", "s7")),
        JArm("J-cot-V0", None, readout="cot", suites=("s1", "s2", "s4", "s6")),
        JArm("J-cot-V1", "v1_lora_s0/adapter", readout="cot", trained_on="jev-qwen train split (3 formats)",
             suites=("s1", "s2", "s4", "s6")),
        JArm("J-V1-rmp", "v1_rmp_s0/adapter", trained_on="2,400 RMP train-region rows", suites=("s1",)),
        JArm("J-V2b-rmp", "v2b_rmp_s0/adapter", loop_iters=3, trained_on="2,400 RMP train-region rows, loop 12-15 x2",
             suites=("s1",)),
        # SPEC-U1 (docs/ubench/SPEC.md): one adapter on the utility suites' training splits.
        # Not SPEC v1 arms: kept out of the v1 reliability pools and coverage matrix.
        JArm("J-multi", "multi_lora_s0/adapter", trained_on="S4 + S1 + S2 + S7 training splits (SPEC-U1)",
             suites=("s1", "s2", "s3", "s4", "s7")),
        JArm("J-multi-loop", "multi_loop_s0/adapter", loop_iters=3,
             trained_on="S4 + S1 + S2 + S7 training splits, loop 12-15 x2 (SPEC-U1)", suites=("s1", "s2", "s4", "s7")),
        # SPEC-U5 part J: J-multi continued on the SPEC-U4 train items (Campsite decide + repair).
        JArm("J-u4", "u4_lora_s0/adapter", trained_on="J-multi + SPEC-U4 train items (SPEC-U5)", suites=("s1", "s2", "s4", "s7")),
        # SPEC-U6 part T: J-u4 continued on real model proposals for the U4 train/val puzzles.
        JArm("J-real", "real_lora_s0/adapter", trained_on="J-u4 + real J-V0/Bonsai proposals on train puzzles (SPEC-U6)", suites=()),
        # SPEC-U7 part M: one adapter from the base model on every training source of the thread.
        JArm("J-mix", "mix_lora_s0/adapter", trained_on="J-multi sources + U4 items + U6 realtrain proposals (SPEC-U7)",
             suites=("s1", "s2", "s4", "s7")),
        # SPEC-U8: small Jevs. Scale arms use J-multi's recipe unchanged; the R arms use its rows in
        # choice format only, with full-information Brier or bandit feedback (RLCD, RLVR).
        JArm("J-V0-0.8B", None, base="0.8B", suites=U8_SUITES),
        JArm("J-V0-2B", None, base="2B", suites=U8_SUITES),
        JArm("J-multi-0.8B", "u8_multi08_s0/adapter", base="0.8B", trained_on="J-multi's recipe on Qwen3.5-0.8B-Base (SPEC-U8)",
             suites=U8_SUITES),
        JArm("J-multi-2B", "u8_multi2b_s0/adapter", base="2B", trained_on="J-multi's recipe on Qwen3.5-2B-Base (SPEC-U8)",
             suites=U8_SUITES),
        JArm("J-brier-0.8B", "u8_brier08_s0/adapter", base="0.8B", trained_on="J-multi's rows, choice only, Brier (SPEC-U8)",
             suites=U8_SUITES),
        JArm("J-rlcd-0.8B", "u8_rlcd08_s0/adapter", base="0.8B", trained_on="J-multi's rows, choice only, bandit r = c - p_a (SPEC-U8)",
             suites=U8_SUITES),
        JArm("J-rlvr-0.8B", "u8_rlvr08_s0/adapter", base="0.8B", trained_on="J-multi's rows, choice only, bandit r = c (SPEC-U8)",
             suites=U8_SUITES),
    )
}
U8_ARMS = ("J-V0-0.8B", "J-V0-2B", "J-multi-0.8B", "J-multi-2B", "J-brier-0.8B", "J-rlcd-0.8B", "J-rlvr-0.8B")
U1_ARMS = ("J-multi", "J-multi-loop", "J-u4", "J-real", "J-mix") + U8_ARMS  # outside the v1 pools and coverage matrix


def derived_arm_ids(arm: JArm) -> list[str]:
    """Record ids a run produces: looped arms yield one record per iteration count."""
    if arm.loop_iters:
        return [f"{arm.arm_id}-r{n}" for n in range(1, arm.loop_iters + 1)]
    return [arm.arm_id]


def seeded(arm_id: str, seed: int) -> JArm:
    """The same arm trained with another seed: adapter `<name>_s<seed>`, id `<arm>-s<seed>`."""
    from dataclasses import replace

    arm = J_ARMS[arm_id]
    if seed == 0:
        return arm
    if not arm.adapter or "_s0/" not in arm.adapter:
        raise ValueError(f"{arm_id} has no seed-0 adapter to replicate")
    return replace(arm, arm_id=f"{arm_id}-s{seed}", adapter=arm.adapter.replace("_s0/", f"_s{seed}/"),
                   trained_on=f"{arm.trained_on} (seed {seed})")
