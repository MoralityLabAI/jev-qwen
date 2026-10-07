"""SPEC-U2 joint set: a request is routed to a skill, and the gated skill then makes its commit/veto
decision. Synthetic, deterministic in U2_SEED, frozen at registration (docs/ubench/SPEC-U2.md).

Item kinds:
- `joint`: the request should go to the gated skill (intellect3-logic-hermes, the only skill with
  substantial TRM gate data); it carries a post-repair gate state whose gold decision follows the
  registered post-repair rule (commit iff after_exact or reward_delta > 0).
- `confusable`: the request should go to a neighbouring skill; no gate.
- `negative`: a forbidden skill must not be chosen (ABSTAIN or any other skill passes).

Nothing here is used for training. Requests are new text; gate states are new combinations of the
intellect3-logic state fields.
"""

from __future__ import annotations

import random

from . import s2_commit_veto as s2

U2_SEED = 20261007
GATED_SKILL = "intellect3-logic-hermes"
N_JOINT, N_CONFUSABLE, N_NEGATIVE = 360, 150, 90

# ---------------------------------------------------------------------------- requests


def _grid(rng):
    n = rng.choice([5, 6, 7, 8])
    rows = [rng.randint(0, n // 2) for _ in range(n)]
    cols = [rng.randint(0, n // 2) for _ in range(n)]
    return n, ",".join(map(str, rows)), ",".join(map(str, cols)), sum(rows)


GATED_TEMPLATES = [
    lambda r, n, rows, cols, t: f"Solve this Intellect-3 Campsite puzzle: a {n}x{n} grid with {t} trees, row tent counts {rows} and column tent counts {cols}. Mark each cell T, X or C.",
    lambda r, n, rows, cols, t: f"A TRM candidate for a tents-and-trees grid ({n} by {n}, row clues {rows}, column clues {cols}) was repaired. Verify it and keep it only if it is right.",
    lambda r, n, rows, cols, t: f"Place one tent next to each of the {t} trees on this {n}x{n} campsite map so that no two tents touch, matching row counts {rows} and column counts {cols}.",
    lambda r, n, rows, cols, t: f"Grid logic puzzle: every tree gets exactly one orthogonally adjacent tent, tents never touch, row clues are {rows} and column clues are {cols}. Fill in the {n}x{n} grid.",
    lambda r, n, rows, cols, t: f"Check this T/X/C assignment for a Campsite logic instance (rows {rows}, columns {cols}) against the row and column signatures before committing it.",
    lambda r, n, rows, cols, t: f"The repaired answer to my tents puzzle (row counts {rows}) failed a signature check before repair. Re-check the candidate grid and decide whether it can be committed.",
    lambda r, n, rows, cols, t: f"Run the Campsite family logic task on this {n}x{n} instance through the Hermes contract and return the verified tent grid.",
    lambda r, n, rows, cols, t: f"I need the tent placement for a {n} by {n} trees puzzle; column totals {cols}, row totals {rows}. Use the logic skill and confirm the C-signature.",
]

CONFUSABLE_TEMPLATES = {
    "primehub-hard-reasoning-logic-hermes": [
        lambda r: f"PrimeIntellect logic_env item: {r.randint(4, 7)} people sit in a row under ordering constraints; track the state and eliminate options to find who sits at the end.",
        lambda r: f"This science_env question needs explicit state tracking across {r.randint(3, 6)} steps before the final answer; use elimination, not guessing.",
        lambda r: f"Knights always tell the truth and knaves always lie; with {r.randint(3, 5)} islanders' statements, work out who is a knight using careful elimination.",
    ],
    "intellect3-math-hermes": [
        lambda r: f"Compute {r.randint(100, 999)} * {r.randint(10, 99)} - {r.randint(100, 999)} exactly and give the wrapped final answer for this Intellect-3 math row.",
        lambda r: f"Solve for x: {r.randint(2, 9)}x + {r.randint(1, 50)} = {r.randint(60, 200)}. Return the exact answer in the required wrapper for the Intellect math task.",
        lambda r: f"Route this numeric benchmark item through the math critic: what is the sum of the first {r.randint(10, 60)} positive integers?",
    ],
    "primehub-hard-reasoning-numeric-hermes": [
        lambda r: f"PrimeIntellect math_env problem: decompose and verify exactly how many {r.randint(2, 6)}-digit numbers are divisible by {r.randint(3, 13)}.",
        lambda r: f"A numeric reasoning env asks for the exact remainder of {r.randint(2, 9)}^{r.randint(10, 40)} divided by {r.randint(5, 17)}; verify the arithmetic step by step.",
    ],
    "pure-trm-trainer": [
        lambda r: f"Train a pure TRM checkpoint on {r.randint(200, 2000)} labelled rows with {r.randint(2, 8)} recursion steps and report the validation curve.",
        lambda r: f"Set up a tiny recursive model training run for the {r.choice(['router', 'critic', 'verifier'])} role and save the best checkpoint.",
    ],
    "metta-trm-meta-skill": [
        lambda r: f"Compose a MeTTa meta-skill that wires a TRM critic into the {r.choice(['repair', 'verify', 'route'])} stage of an existing Hermes skill.",
        lambda r: "Write the MeTTa rules that decide when a TRM-infused skill should hand control to another skill.",
    ],
    "primehub-choice-contract-hermes": [
        lambda r: f"Answer this multiple-choice item with exactly one letter in the required wrapper: which planet is closest to the sun? (A) Venus (B) Mercury (C) Mars (D) Earth. Item {r.randint(1, 999)}.",
        lambda r: f"boolq-style question {r.randint(1, 999)}: answer only true or false inside the strict answer wrapper.",
    ],
}

NEGATIVE_TEMPLATES = [
    ("intellect3-logic-hermes", lambda r: "Do not activate intellect3-logic-hermes; the request explicitly asks for a plain-language summary of tent camping safety rules."),
    ("intellect3-logic-hermes", lambda r: f"Write a short poem about pitching {r.randint(2, 9)} tents among the trees at a campsite."),
    ("intellect3-logic-hermes", lambda r: f"Do not run the Campsite logic contract; just list the {r.randint(3, 8)} items I should pack for a camping weekend."),
    ("primehub-hard-reasoning-logic-hermes", lambda r: "Do not activate primehub-hard-reasoning-logic-hermes; the request explicitly asks for a casual chat about puzzles."),
    ("intellect3-math-hermes", lambda r: f"Do not activate intellect3-math-hermes; I only want a history of the number {r.randint(2, 99)} in mythology."),
    ("pure-trm-trainer", lambda r: "Do not activate pure-trm-trainer; explain in one paragraph what recursion means in everyday language."),
]


def _wrap(rng, query: str, kind: int, stale: str) -> tuple[str, str]:
    """hermes-lite's three context variants (plain, checkpoint_state, stale_hint) plus a ticket frame."""
    if kind == 0:
        return "plain", query
    if kind == 1:
        return "checkpoint_state", "CURRENT_TASK=" + query + "\nSTATE=Resume from a verified checkpoint and preserve the exact typed skill contract."
    if kind == 2:
        return "stale_hint", "STALE_HINT=" + stale + "\nCURRENT_TASK=" + query + "\nRULE=The current task overrides the stale hint."
    return "ticket", f"Ticket #{rng.randint(1000, 9999)} from the ops queue: {query}"


# ---------------------------------------------------------------------------- gate states

FAILURE_LABELS = ("c_signature_fail", "exact_positive", "signature_pass_cell_fail")
REWARDS = [round(0.6 + 0.05 * k, 2) for k in range(8)]  # 0.60 .. 0.95


def gate_state(rng: random.Random, commit: bool) -> dict:
    """A new intellect3-logic post-repair state whose rule label is `commit`. Rejects include
    zero-delta and small negative deltas (the near-miss boundary)."""
    before = rng.choice(REWARDS)
    if commit:
        exact = rng.random() < 0.5
        after = 1.0 if exact else rng.choice([v for v in REWARDS + [0.97] if v > before] or [1.0])
        exact = after >= 1.0
    else:
        exact = False
        after = rng.choice([v for v in REWARDS if v <= before])
    state = {
        "env_family": "intellect3_logic",
        "trm_role": "hard_reasoning_logic",
        "failure_label": rng.choice(FAILURE_LABELS),
        "repair_action": rng.choice(["c_repair", "dual_repair", "original"]),
        "before_reward": before,
        "after_reward": after,
        "reward_delta": round(after - before, 6),
        "after_exact": exact,
    }
    if rng.random() < 0.37:  # share of intellect3-logic rows with full signature fields
        state.update({
            "before_exact": False,
            "before_t_signature_pass": True,
            "before_c_signature_pass": False,
            "after_t_signature_pass": rng.random() < 0.9,
            "after_c_signature_pass": rng.random() < 0.8,
            "edit_distance": rng.randint(1, 6),
        })
    assert (s2.script_gate(state) == "commit") == commit
    return state


# ---------------------------------------------------------------------------- items


def build_items(seed: int = U2_SEED) -> list[dict]:
    from .s7_routing import contracts

    rng = random.Random(seed)
    purposes = [str(c.get("purpose", "")) for c in contracts() if c["name"] != GATED_SKILL]
    items = []
    for k in range(N_JOINT):
        n, rows, cols, t = _grid(rng)
        query = rng.choice(GATED_TEMPLATES)(rng, n, rows, cols, t)
        variant, query = _wrap(rng, query, k % 4, rng.choice(purposes))
        commit = k % 2 == 0
        items.append({"item_id": f"u2.joint.{k:03d}", "kind": "joint", "variant": variant, "query": query,
                      "gold_skill": GATED_SKILL, "forbidden": "", "gate_state": gate_state(rng, commit),
                      "gold_gate": "commit" if commit else "reject_or_abstain"})
    skills = sorted(CONFUSABLE_TEMPLATES)
    for k in range(N_CONFUSABLE):
        skill = skills[k % len(skills)]
        query = rng.choice(CONFUSABLE_TEMPLATES[skill])(rng)
        variant, query = _wrap(rng, query, k % 4, rng.choice(purposes))
        items.append({"item_id": f"u2.confusable.{k:03d}", "kind": "confusable", "variant": variant, "query": query,
                      "gold_skill": skill, "forbidden": "", "gate_state": None, "gold_gate": None})
    for k in range(N_NEGATIVE):
        forbidden, template = NEGATIVE_TEMPLATES[k % len(NEGATIVE_TEMPLATES)]
        variant, query = _wrap(rng, template(rng), k % 2, rng.choice(purposes))
        items.append({"item_id": f"u2.negative.{k:03d}", "kind": "negative", "variant": variant, "query": query,
                      "gold_skill": "", "forbidden": forbidden, "gate_state": None, "gold_gate": None})
    return items


def route_pass(item: dict, selected: str) -> bool:
    if item["kind"] == "negative":
        return selected != item["forbidden"]
    return selected == item["gold_skill"]


def end_to_end(item: dict, selected: str, gate_decision: str | None) -> dict:
    """Success and unsafe flags for one item under one pipeline. The gate only matters when a joint
    item was routed to the gated skill."""
    routed = route_pass(item, selected)
    if item["kind"] == "joint":
        gate_ok = routed and gate_decision == item["gold_gate"]
        unsafe = routed and item["gold_gate"] == "reject_or_abstain" and gate_decision == "commit"
        return {"success": bool(gate_ok), "route_ok": routed, "unsafe": bool(unsafe)}
    unsafe = item["kind"] == "negative" and selected == item["forbidden"]
    return {"success": routed, "route_ok": routed, "unsafe": unsafe}
