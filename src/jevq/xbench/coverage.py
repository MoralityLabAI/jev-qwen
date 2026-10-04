"""The registered (arm, suite) matrix and which cells have a record (SPEC sections 2-4).

A cell is covered by a live record, a not-applicable record in the suite, or one in `any/`
(arms that cannot run at all). `structural_reason` gives the not-applicable reason for cells
the SPEC's adapters never define; cells an arm is registered to run are never filled with one.
"""

from __future__ import annotations

import json
from pathlib import Path

from .arms import J_ARMS, derived_arm_ids
from .records import RESULTS

SUITES = ("s1", "s2", "s3", "s4", "s5", "s6", "s7")
SUITE_DIRS = {"s6": ("s6", "s6c", "s6a")}  # S6 = scripted ladder on S4 or S5 targets, or adaptive

RMP_ARMS = ("LOOP-T-ds", "LOOP-T", "FF-U", "FF-U-ds")
CH_IDS = ("CH-q0-none", "CH-q1-action-gate", "CH-q2-trajectory-budget", "CH-q3-delegation-guard",
          "CH-q3-claims-ignored", "CH-q4-provenance-membrane")

# SPEC arm name -> record ids that can cover it (any one suffices).
ARMS: dict[str, tuple[str, ...]] = {
    **{a.arm_id: (a.arm_id, *derived_arm_ids(a)) for a in J_ARMS.values() if a.arm_id != "J-V1c-lr3"},
    **{a: (a,) for a in RMP_ARMS},
    "ControlTRM": ("ControlTRM", "ControlTRM-LDT"),
    "SkillRouter-TRM": ("SkillRouter-TRM",),
    "TRM-cv": ("TRM-cv",),
    "kNN-critic": ("kNN-critic",),
    "script": ("script",),
    "LDT": ("LDT",),
    "CH-q0..q4": CH_IDS,
    "lexical-router": ("lexical-router",),
    "Bonsai-8B": ("Bonsai-8B",),
    "RLM-API": ("RLM-API",),
    # Registered as not run (SPEC section 2).
    "Qwen2.5-3B-Q4": ("Qwen2.5-3B-Q4",),
    "commit-veto-LoRA-TRM": ("commit-veto-LoRA-TRM",),
    "TinyRecursivePolicy": ("TinyRecursivePolicy",),
    "RMP-COT": ("RMP-COT",),
    "MeTTa-gate": ("MeTTa-gate",),  # named in RQ-C, not in SPEC section 2 (see run_not_applicable)
}

# Arms whose adapter exists for one suite only (SPEC section 4), with the reason for the others.
SINGLE_SUITE = {
    **{a: ("s1", "RMP decoders read the RMP token vocabulary only.") for a in RMP_ARMS},
    "J-V1-rmp": ("s1", "S1-only arm: trained on RMP train-region rows (SPEC section 2)."),
    "J-V2b-rmp": ("s1", "S1-only arm: trained on RMP train-region rows (SPEC section 2)."),
    "ControlTRM": ("s3", "ControlTRM consumes S3 public features only."),
    "LDT": ("s3", "LDT is a recorded S3 arm; it has no adapter for other suites."),
    "RLM-API": ("s3", "Paid-API arm: recorded S3 results only; never re-called."),
    "SkillRouter-TRM": ("s7", "Router features are contract-route features (S7 only)."),
    "lexical-router": ("s7", "The lexical router scores contract routes (S7 only)."),
    "TRM-cv": ("s2", "TRM-cv consumes S2 state features only."),
    "kNN-critic": ("s2", "kNN-critic consumes S2 state features only."),
}

COT_NA = "No registered worked-solution format for this suite (SPEC section 4 lists cot only where traces exist)."


def structural_reason(arm: str, suite: str) -> str | None:
    """Why the SPEC defines no adapter for this cell; None if the arm is registered to run it."""
    if arm in SINGLE_SUITE:
        home, reason = SINGLE_SUITE[arm]
        return None if suite == home else reason
    if arm == "CH-q0..q4":
        return None if suite in ("s5", "s6") else "Control-Harness controls gate S5 contract actions only."
    if arm == "script":
        return {"s3": "No script gate is registered for S3; LDT is its non-neural arm.",
                "s5": "S5's script gates are the CH-q0..q4 controls.",
                "s7": "S7's script gate is the lexical router."}.get(suite)
    if arm in J_ARMS:
        j = J_ARMS[arm]
        if j.readout == "cot" and suite in ("s2", "s3", "s5", "s7"):
            return COT_NA
        if suite not in j.suites:
            return "Not registered for this suite (SPEC section 4)."
    return None


def _cell(arm: str, suite: str, root: Path) -> tuple[str, str | None]:
    """('live' | 'na' | 'missing', detail) for one cell."""
    for record_id in ARMS[arm]:
        if (root / "any" / record_id / "record.json").exists():
            return "na", json.loads((root / "any" / record_id / "record.json").read_text(encoding="utf-8"))["not_applicable"]
        for directory in SUITE_DIRS.get(suite, (suite,)):
            path = root / directory / record_id / "record.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if "not_applicable" in record:
                    return "na", record["not_applicable"]
                return "live", f"{directory}/{record_id}"
    return "missing", None


def coverage(root: Path = RESULTS) -> dict[tuple[str, str], tuple[str, str | None]]:
    return {(arm, suite): _cell(arm, suite, root) for arm in ARMS for suite in SUITES}


def fill_structural(root: Path = RESULTS) -> list[tuple[str, str, str]]:
    """Write a not-applicable record for every missing cell that has a structural reason."""
    from .records import write_not_applicable

    written = []
    for (arm, suite), (state, _) in coverage(root).items():
        reason = structural_reason(arm, suite) if state == "missing" else None
        if reason:
            # Group arms (the CH controls) get one record per member, others one under their base id.
            for record_id in ARMS[arm] if arm == "CH-q0..q4" else ARMS[arm][:1]:
                write_not_applicable(suite, record_id, reason, root=root)
            written.append((arm, suite, reason))
    return written
