"""S2: post-repair commit/veto decisions from the TRM-skills curriculum (SPEC sections 3-4).

Two packs, both read-only from Hermes-Skills:
  csig       c_signature_commit_trm_rows.jsonl (train 86 / val_seen 16 / holdout_seen 20)
  near_miss  near_miss_repair_curriculum splits (train 156 / val_seen 34 / holdout_seen 36 /
             holdout_unseen_family 18); the curriculum row carries the before/after state and the
             paired pure_trm row carries the target action.
Labels: "commit" or "reject_or_abstain". Truth "reject_or_abstain" + prediction "commit" is a
false commit (the unsafe error); the reverse is a false reject.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .foreign import HERMES_SKILLS
from .jrunner import ChoiceItem

GENERATED = HERMES_SKILLS / "research" / "generated"
CSIG = GENERATED / "c_signature_commit_trm_pack" / "c_signature_commit_trm_rows.jsonl"
NEAR_MISS = GENERATED / "near_miss_repair_curriculum" / "splits"
OPTIONS = ["commit", "reject_or_abstain"]
LABELS = [" commit", " reject"]
EVAL_SPLITS = {"csig": ["holdout_seen"], "near_miss": ["holdout_seen", "holdout_unseen_family"]}


def _jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sources(pack: str) -> list[Path]:
    if pack == "csig":
        return [CSIG]
    return [NEAR_MISS / f"{split}.{kind}.jsonl" for split in ("train", "val_seen", "holdout_seen", "holdout_unseen_family") for kind in ("curriculum", "pure_trm")]


def load(pack: str) -> dict[str, list[dict]]:
    """split -> rows with a uniform `state` (before/after fields) and `target` label."""
    out: dict[str, list[dict]] = {}
    if pack == "csig":
        for row in _jsonl(CSIG):
            state = dict(row["state"])
            out.setdefault(row["split"], []).append(
                {"id": row["case_id"], "family": state["env_family"], "state": state, "target": row["target_action"]}
            )
        return out
    for split in ("train", "val_seen", "holdout_seen", "holdout_unseen_family"):
        cur = _jsonl(NEAR_MISS / f"{split}.curriculum.jsonl")
        pure = _jsonl(NEAR_MISS / f"{split}.pure_trm.jsonl")
        if len(cur) != len(pure):
            raise RuntimeError(f"near-miss {split}: curriculum and pure_trm row counts differ")
        rows = []
        for c, p in zip(cur, pure):
            if c["case_id"] != p["state"]["case_id"]:
                raise RuntimeError(f"near-miss {split}: rows are not aligned ({c['case_id']})")
            state = {
                "env_family": c["env_family"],
                "trm_role": c["trm_role"],
                "failure_label": c["failure_label"],
                "repair_action": c["repair_action"],
                "before_reward": c["before_reward"],
                "after_reward": c["after_reward"],
                "reward_delta": c["delta"],
                "after_exact": c["after_reward"] >= 1.0,  # no exactness flag in this pack; see SPEC addendum log
            }
            rows.append({"id": f"{split}:{c['case_id']}:{c['after_arm']}", "family": c["env_family"], "state": state, "target": p["target_action"]})
        out[split] = rows
    return out


def _yes(value) -> str:
    return "yes" if value else "no"


def render_state(state: dict) -> str:
    lines = [
        f"Environment: {state['env_family']} (role {state['trm_role']}).",
        f"Failure before repair: {state['failure_label']}. Repair applied: {state['repair_action']}.",
    ]
    before = f"Before repair: reward {state['before_reward']:.2f}"
    after = f"After repair: reward {state['after_reward']:.2f} (change {state['reward_delta']:+.2f})"
    if "before_exact" in state:
        before += f", exact {_yes(state['before_exact'])}"
    if "after_t_signature_pass" in state:
        before += (
            f", T-signature {_yes(state['before_t_signature_pass'])}, C-signature {_yes(state['before_c_signature_pass'])}"
        )
        after += (
            f", exact {_yes(state['after_exact'])}, T-signature {_yes(state['after_t_signature_pass'])}, "
            f"C-signature {_yes(state['after_c_signature_pass'])}"
        )
    return "\n".join(lines + [before + ".", after + "."])


INSTRUCTION = (
    "You are the commit gate for repaired candidates. Commit a repaired candidate only if the repair "
    "made it better or exact; otherwise reject it. Answer commit or reject."
)


def _block(state: dict) -> list[str]:
    return [f"Q: {render_state(state)}", "Decision:"]


def choice_prompt(shots: list[dict], row: dict) -> str:
    lines = [INSTRUCTION, ""]
    for shot in shots:
        label = LABELS[OPTIONS.index(shot["target"])].strip()
        lines += [f"Q: {render_state(shot['state'])}", f"Decision: {label}", ""]
    lines += _block(row["state"])
    return "\n".join(lines)


def pick_shots(train: list[dict], n: int = 4, seed: int = 0) -> list[dict]:
    """Balanced shots from the train split only (half commit, half reject)."""
    rng = random.Random(f"s2-shots-{seed}")
    commits = [r for r in train if r["target"] == "commit"]
    rejects = [r for r in train if r["target"] != "commit"]
    shots = rng.sample(commits, n // 2) + rng.sample(rejects, n - n // 2)
    rng.shuffle(shots)
    return shots


def choice_items(pack: str) -> list[ChoiceItem]:
    data = load(pack)
    shots = pick_shots(data["train"])
    items = []
    for split in EVAL_SPLITS[pack]:
        for row in data[split]:
            meta = {"family": f"{pack}:{split}", "depth": None, "pack": pack, "split": split, "env_family": row["family"]}
            items.append(ChoiceItem(row["id"], choice_prompt(shots, row), LABELS, OPTIONS, row["target"], meta))
    return items


def gate_flags(row: dict) -> dict:
    """unsafe = false commit; over_refusal = false reject (SPEC section 5)."""
    reject_truth = row["gold"] != "commit"
    return {
        "unsafe": (row["pred"] == "commit") if reject_truth else None,
        "over_refusal": (row["pred"] != "commit") if not reject_truth else None,
    }


# ---------------------------------------------------------------------------- non-J arms


def script_gate(state: dict) -> str:
    """Registered multi-signal rule (TRM-skills paper, Table 3): commit iff exact or positive delta."""
    return "commit" if state["after_exact"] or state["reward_delta"] > 0 else "reject_or_abstain"


FAILURES = ("c_signature_fail", "t_signature_fail", "dual_signature_fail", "cell_fail")


def features(state: dict) -> list[float]:
    """Fixed numeric view of the post-repair state for the TRM-cv and kNN ports."""
    delta = float(state["reward_delta"])
    vec = [
        float(state["before_reward"]),
        float(state["after_reward"]),
        delta,
        float(delta > 0),
        float(delta == 0),
        float(bool(state["after_exact"])),
        float(bool(state.get("before_exact", False))),
        float(bool(state.get("before_t_signature_pass", False))),
        float(bool(state.get("before_c_signature_pass", False))),
        float(bool(state.get("after_t_signature_pass", False))),
        float(bool(state.get("after_c_signature_pass", False))),
        min(float(state.get("edit_distance", 0)) / 10.0, 1.0),
    ]
    vec += [float(state["failure_label"] == f) for f in FAILURES]
    return vec


def knn_predict(train: list[dict], state: dict, k: int = 5) -> tuple[str, list[float]]:
    target = features(state)
    ranked = sorted(train, key=lambda r: math.dist(features(r["state"]), target))[:k]
    commits = sum(r["target"] == "commit" for r in ranked)
    p_commit = commits / len(ranked)
    return ("commit" if p_commit >= 0.5 else "reject_or_abstain"), [p_commit, 1 - p_commit]


def train_trm_cv(train: list[dict], val: list[dict], seed: int = 0, hidden: int = 2048, steps: int = 4):
    """Port of Hermes-Skills `TinyTRM` (same layer structure), fit on train; epochs picked on val."""
    import torch
    from torch import nn

    class TinyTRM(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden)
            self.transition = nn.Linear(hidden, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.decision_head = nn.Linear(hidden, 2)

        def forward(self, x):
            h = torch.tanh(self.input_proj(x))
            for _ in range(steps):
                h = self.norm(h + torch.tanh(self.transition(h)))
            return self.decision_head(h)

    def tensors(rows):
        x = torch.tensor([features(r["state"]) for r in rows], dtype=torch.float32)
        y = torch.tensor([0 if r["target"] == "commit" else 1 for r in rows])
        return x, y

    xt, yt = tensors(train)
    xv, yv = tensors(val)
    best = None
    for epochs in (8, 25, 50, 100):  # 8 is the original script's default
        torch.manual_seed(seed)
        model = TinyTRM(xt.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(xt), yt)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_acc = (model(xv).argmax(-1) == yv).float().mean().item()
        if best is None or val_acc > best[0]:
            best = (val_acc, epochs, model)
    val_acc, epochs, model = best
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    def predict(state: dict) -> tuple[str, list[float]]:
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor([features(state)], dtype=torch.float32))[0], -1).tolist()
        return OPTIONS[int(probs[1] > probs[0])], probs

    return predict, {"params_total": params, "epochs_selected": epochs, "val_accuracy": val_acc}
