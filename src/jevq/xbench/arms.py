"""J-arm registry (SPEC section 2). Adapter paths are relative to paths.checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

from ..looped import LoopSpec

LOOP_SPAN = (12, 16)


@dataclass(frozen=True)
class JArm:
    arm_id: str
    adapter: str | None
    loop_iters: int | None = None  # None = stock driver; N = loop over LOOP_SPAN, recorded per iteration
    readout: str = "choice"  # choice | cot
    trained_on: str | None = None
    suites: tuple[str, ...] = ("s1", "s2", "s3", "s4", "s5", "s6", "s7")

    @property
    def loop(self) -> LoopSpec | None:
        return LoopSpec(*LOOP_SPAN, self.loop_iters) if self.loop_iters else None


J_ARMS = {
    arm.arm_id: arm
    for arm in (
        JArm("J-V0", None),
        JArm("J-V1", "v1_lora_s0/adapter", trained_on="jev-qwen train split (3 formats)"),
        JArm("J-V1c", "v1c_lora_s0/adapter", trained_on="jev-qwen train split (3 formats), Brier on choice"),
        # One run at 3 iterations; the tail lens gives the exact r1 and r2 outputs (instrument.py).
        JArm("J-V2b", "v2b_lora_s0/adapter", loop_iters=3, trained_on="jev-qwen train split, loop 12-15 x2",
             suites=("s1", "s2", "s4", "s5", "s6", "s7")),
        JArm("J-cot-V0", None, readout="cot", suites=("s1", "s2", "s4", "s6")),
        JArm("J-cot-V1", "v1_lora_s0/adapter", readout="cot", trained_on="jev-qwen train split (3 formats)",
             suites=("s1", "s2", "s4", "s6")),
        JArm("J-V1-rmp", "v1_rmp_s0/adapter", trained_on="2,400 RMP train-region rows", suites=("s1",)),
        JArm("J-V2b-rmp", "v2b_rmp_s0/adapter", loop_iters=3, trained_on="2,400 RMP train-region rows, loop 12-15 x2",
             suites=("s1",)),
    )
}


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
