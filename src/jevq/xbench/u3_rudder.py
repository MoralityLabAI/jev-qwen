"""SPEC-U3 part R: the Hermes-Skills repair rudder on the near-miss curriculum's 88 non-train rows.

A rudder sees the pre-repair state of a TRM-infused skill's candidate and chooses the repair action
and the commit action. The published runner is Hermes-Skills
`research/scripts/run_3b_repair_training_rudder_benchmark.py`; its state rendering, retrieval and
MeTTa rules are ported here unchanged (functions marked "port"). Outputs:
results/u3/rudder/<record>/<protocol>.jsonl, one row per eval row in a shared schema.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..config import ROOT
from . import s2_commit_veto as s2
from .foreign import HERMES_SKILLS, git_show
from .jrunner import ChoiceItem

OUT = ROOT / "results" / "u3" / "rudder"
SPLITS = s2.NEAR_MISS
EVAL_SPLITS = ("val_seen", "holdout_seen", "holdout_unseen_family")
TEST_SPLITS = ("holdout_seen", "holdout_unseen_family")
SHOTS = 4
LETTERS = "ABCDEFGH"
PROTOCOLS = ("raw", "retrieval", "retrieval-published", "action-space", "static-gate")
ARTIFACTS = "research/studies/2026-04-22-metta-trm-hermes-pipeline/artifacts"
# model -> [(rows file under Hermes-Skills HEAD, {published arm: protocol})]
RECEIPTS = {
    "Qwen2.5-3B": [
        (f"{ARTIFACTS}/local_3b_repair_training_rudder_benchmark/local_3b_repair_training_rudder.rows.jsonl",
         {"raw_3b_rudder": "raw", "repair_training_rudder": "retrieval-published"}),
        (f"{ARTIFACTS}/local_3b_metta_action_space_rudder_benchmark/local_3b_repair_training_rudder.rows.jsonl",
         {"metta_action_space_rudder": "action-space", "metta_static_gate_rudder": "static-gate",
          "metta_validator_gate": "validator"}),
    ],
    "Qwen3.5-9B": [
        (f"{ARTIFACTS}/remote_9b_repair_training_rudder_20260502T203509Z/remote_repair_training_rudder.rows.jsonl",
         {"raw_3b_rudder": "raw", "repair_training_rudder": "retrieval-published",
          "metta_action_space_rudder": "action-space", "metta_static_gate_rudder": "static-gate"}),
    ],
    "Qwen3.5-27B": [
        (f"{ARTIFACTS}/remote_27b_repair_training_rudder_20260502T204314Z/remote_repair_training_rudder.rows.jsonl",
         {"raw_3b_rudder": "raw", "repair_training_rudder": "retrieval-published",
          "metta_action_space_rudder": "action-space", "metta_static_gate_rudder": "static-gate"}),
    ],
}


def _jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def key(row: dict) -> str:
    return f"{row['eval_split']}:{row['state']['case_id']}"


def load_rows() -> dict[str, list[dict]]:
    """split -> pure_trm rows, each tagged with `eval_split`."""
    out = {}
    for split in ("train",) + EVAL_SPLITS:
        out[split] = [dict(r, eval_split=split) for r in _jsonl(SPLITS / f"{split}.pure_trm.jsonl")]
    return out


def eval_rows(rows: dict[str, list[dict]] | None = None) -> list[dict]:
    rows = rows or load_rows()
    return [r for split in EVAL_SPLITS for r in rows[split]]


def allowed_actions(train: list[dict]) -> list[str]:
    """port: the published allowed list is every repair action seen in train."""
    return sorted({str(r.get("action") or "") for r in train if str(r.get("action") or "")})


# ---------------------------------------------------------------------------- ports


def row_text(row: dict, *, include_answer: bool) -> str:
    """port of `row_text`: the published state rendering."""
    state = row.get("state") or {}
    tool_map = {str(t.get("name")): str(t.get("result")) for t in (row.get("tools") or []) if isinstance(t, dict)}
    fields = {
        "env_family": state.get("env_family"),
        "trm_role": state.get("trm_role"),
        "before_arm": state.get("before_arm"),
        "after_arm": state.get("after_arm"),
        "before_reward": state.get("before_reward"),
        "failure_label": state.get("failure_label"),
        "candidate_excerpt": state.get("candidate_excerpt"),
        "route_gate": tool_map.get("route_gate"),
        "validate_gate": tool_map.get("validate_gate"),
    }
    if include_answer:
        fields["repair_gate"] = tool_map.get("repair_gate")
        fields["bucket"] = row.get("bucket")
        fields["target_repair_action"] = row.get("action")
        fields["target_commit_action"] = row.get("target_action")
    return json.dumps(fields, ensure_ascii=True, separators=(",", ":"))


def _tokens(parts) -> Counter:
    text = " ".join(str(p or "").lower() for p in parts)
    return Counter(re.findall(r"[a-z0-9_]+", text))


def _pre_repair_parts(row: dict) -> list:
    st = row.get("state") or {}
    return [st.get("env_family"), st.get("trm_role"), st.get("before_arm"), st.get("after_arm"), st.get("failure_label"),
            st.get("candidate_excerpt")]


def published_retrieve(train: list[dict], query: dict, shots: int = SHOTS) -> list[dict]:
    """port of `retrieve_examples`. Leaky: the query's bucket, action and target_action enter the match."""
    def toks(row):
        return _tokens(_pre_repair_parts(row) + [row.get("bucket"), row.get("action"), row.get("target_action")])

    q = toks(query)
    qs = query.get("state") or {}
    scored = []
    for row in train:
        rt = toks(row)
        st = row.get("state") or {}
        overlap = sum(min(q[t], rt[t]) for t in q)
        score = overlap + (3.0 if st.get("trm_role") == qs.get("trm_role") else 0.0)
        score += 5.0 if st.get("failure_label") == qs.get("failure_label") else 0.0
        score += 1.0 if row.get("bucket") == query.get("bucket") else 0.0
        scored.append((score, str(st.get("case_id") or ""), row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored[:shots]]


def clean_retrieve(train: list[dict], query: dict, shots: int = SHOTS) -> list[dict]:
    """The published retrieval restricted to pre-repair fields on both sides, without the bucket bonus."""
    q = _tokens(_pre_repair_parts(query))
    qs = query.get("state") or {}
    scored = []
    for row in train:
        rt = _tokens(_pre_repair_parts(row))
        st = row.get("state") or {}
        overlap = sum(min(q[t], rt[t]) for t in q)
        score = overlap + (3.0 if st.get("trm_role") == qs.get("trm_role") else 0.0)
        score += 5.0 if st.get("failure_label") == qs.get("failure_label") else 0.0
        scored.append((score, str(st.get("case_id") or ""), row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored[:shots]]


def metta_repair_action(row: dict) -> str:
    """port of `metta_repair_action`: the hand-written MeTTa action-space rule."""
    state = row.get("state") or {}
    env_family = str(state.get("env_family") or "")
    case_id = str(state.get("case_id") or "")
    failure_label = str(state.get("failure_label") or "")
    after_arm = str(state.get("after_arm") or "")
    if env_family == "intellect3_logic":
        if failure_label in {"exact_positive", "signature_pass_cell_fail"}:
            return "original"
        if failure_label == "c_signature_fail":
            return "dual_repair" if "dual_repair" in after_arm else "c_repair"
        return "original"
    if env_family == "intellect3_camp_gate":
        return "camp_signature_min_edit_projection"
    if env_family == "ascii_tree_deep":
        if case_id == "package_tree_deep":
            return "['ascii_tree_deep_canonical_commit']"
        return "node_list_to_canonical_tree"
    if env_family == "ifeval_contract_subset":
        return "['ifeval_contract_subset_canonical_commit']"
    if env_family in {"pydantic_hard_schema", "safety_abstain_router"}:
        return "canonical_commit"
    if env_family == "tool_contract_router":
        return "intent_schema_arg_repair"
    if env_family == "choice_contract":
        return "choice_token_extract"
    if env_family == "hard_reasoning_numeric":
        return "boxed_choice_extract"
    return "original"


def metta_static_target_action(row: dict) -> str | None:
    """port of `metta_static_target_action`. Reads `bucket`, which is the repair outcome."""
    state = row.get("state") or {}
    failure_label = str(state.get("failure_label") or "")
    case_id = str(state.get("case_id") or "")
    bucket = str(row.get("bucket") or "")
    if bucket in {"repair_success", "partial_repair_improvement"}:
        return "commit"
    if failure_label in {"exact_positive", "exact", "exact_grid", "exact_json", "exact_tree"}:
        return "commit"
    if failure_label == "c_signature_fail" and bucket == "repair_failure_or_no_gain":
        return "reject_or_abstain"
    if failure_label == "signature_pass_cell_fail":
        return "reject_or_abstain"
    if case_id.endswith(":none") or case_id.endswith(":weak_surface"):
        return "reject_or_abstain"
    return None


def metta_validator_target_action(row: dict) -> str:
    """port of `metta_validator_target_action`: the post-repair ceiling (reads `bucket`)."""
    return "commit" if str(row.get("bucket") or "") in {"repair_success", "partial_repair_improvement", "exact_positive"} else "reject_or_abstain"


# ---------------------------------------------------------------------------- scoring


def score(row: dict, pred_repair: str, pred_action: str, **extra) -> dict:
    repair_ok = pred_repair == row["action"]
    action_ok = pred_action == row["target_action"]
    return {
        "key": key(row), "split": row["eval_split"], "case_id": row["state"]["case_id"],
        "failure_label": row["state"]["failure_label"], "bucket": row["bucket"],
        "target_repair": row["action"], "target_action": row["target_action"],
        "pred_repair": pred_repair, "pred_action": pred_action,
        "repair_ok": repair_ok, "action_ok": action_ok, "joint_ok": repair_ok and action_ok,
        "unsafe": pred_action == "commit" and row["target_action"] != "commit",
        **extra,
    }


def write(record: str, protocol: str, rows: list[dict], info: dict | None = None) -> Path:
    out = OUT / record
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{protocol}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if info is not None:
        (out / "arm.json").write_text(json.dumps({"record": record, **info}, indent=2), encoding="utf-8")
    return path


def static_gate(rows: list[dict], action_space: dict[str, dict]) -> list[dict]:
    """static-gate protocol: the MeTTa static rule where it fires, else the arm's action-space answer."""
    out = []
    for row in rows:
        static = metta_static_target_action(row)
        pred = static if static is not None else action_space[key(row)]["pred_action"]
        out.append(score(row, metta_repair_action(row), pred, static_rule=static is not None))
    return out


# ---------------------------------------------------------------------------- Jev prompts

REPAIR_INSTRUCTION = (
    "You are the repair rudder for a TRM-infused Hermes skill. For each state, choose the repair action "
    "to apply before the commit gate. Do not solve the original task. Answer with the option letter."
)
TARGET_INSTRUCTION = (
    "You are the commit gate for a TRM-infused Hermes skill. Given the state and the repair action, "
    "decide whether the repaired candidate should be committed or rejected. Answer commit or reject."
)


def _shuffled(row_key: str, actions: list[str]) -> list[str]:
    rng = random.Random(f"u3-repair-options:{row_key}")
    out = list(actions)
    rng.shuffle(out)
    return out


def examples_for(protocol: str, train: list[dict], row: dict) -> list[dict]:
    if protocol == "retrieval":
        return clean_retrieve(train, row)
    if protocol == "retrieval-published":
        return published_retrieve(train, row)
    return []


def repair_item(row: dict, actions: list[str], examples: list[dict]) -> ChoiceItem:
    options = _shuffled(key(row), actions)
    letters = LETTERS[: len(options)]
    lines = [REPAIR_INSTRUCTION, "", "Repair actions:"] + [f"{l}) {a}" for l, a in zip(letters, options)] + [""]
    for ex in examples:
        lines += [f"State: {row_text(ex, include_answer=False)}", f"Repair: {letters[options.index(ex['action'])]}", ""]
    lines += [f"State: {row_text(row, include_answer=False)}", "Repair:"]
    return ChoiceItem(key(row), "\n".join(lines), [" " + l for l in letters], options, row["action"],
                      {"split": row["eval_split"], "question": "repair"})


def target_item(row: dict, repair: str, examples: list[dict]) -> ChoiceItem:
    lines = [TARGET_INSTRUCTION, ""]
    for ex in examples:
        label = s2.LABELS[s2.OPTIONS.index(ex["target_action"])].strip()
        lines += [f"State: {row_text(ex, include_answer=False)}", f"Repair: {ex['action']}", f"Decision: {label}", ""]
    lines += [f"State: {row_text(row, include_answer=False)}", f"Repair: {repair}", "Decision:"]
    return ChoiceItem(key(row), "\n".join(lines), s2.LABELS, s2.OPTIONS, row["target_action"],
                      {"split": row["eval_split"], "question": "target"})


def run_j(arm_id: str, seed: int = 0, iterations: int | None = None) -> None:
    from ..looped import LoopSpec
    from .arms import LOOP_SPAN, seeded
    from .jrunner import load_j_bundle, run_choice

    arm = seeded(arm_id, seed)
    record = arm.arm_id + (f"-r{iterations}" if iterations else "")
    loop = LoopSpec(*LOOP_SPAN, iterations) if iterations else None
    bundle = load_j_bundle(arm.adapter)
    data = load_rows()
    rows, train, actions = eval_rows(data), data["train"], allowed_actions(data["train"])
    info = {"neural": True, "adapter": arm.adapter, "loop_iterations": iterations, "base_arm": arm_id, "seed": seed}
    action_space = None
    for protocol in ("raw", "retrieval", "retrieval-published", "action-space"):
        examples = {key(r): examples_for(protocol, train, r) for r in rows}
        if protocol == "action-space":
            chosen = {key(r): (metta_repair_action(r), None) for r in rows}
        else:
            out = run_choice(bundle, [repair_item(r, actions, examples[key(r)]) for r in rows], loop=loop)
            chosen = {o["item_id"]: (o["pred"], o) for o in out}
        tgt = {o["item_id"]: o for o in run_choice(bundle, [target_item(r, chosen[key(r)][0], examples[key(r)]) for r in rows], loop=loop)}
        result = []
        for r in rows:
            rep, rep_row = chosen[key(r)]
            t = tgt[key(r)]
            extra = {"action_probs": t["probs"], "action_label_mass": t["label_mass"],
                     "retrieved": [e["state"]["case_id"] for e in examples[key(r)]]}
            if rep_row is not None:
                extra.update(repair_options=rep_row["options"], repair_probs=rep_row["probs"], repair_label_mass=rep_row["label_mass"])
            result.append(score(r, rep, t["pred"], **extra))
        write(record, protocol, result, info)
        if protocol == "action-space":
            action_space = {x["key"]: x for x in result}
    write(record, "static-gate", static_gate(rows, action_space))


# ---------------------------------------------------------------------------- CPU arms

AFTER_KEYWORDS = ("dual_repair", "c_repair", "original")


def after_keyword(row: dict) -> str:
    after = str(row["state"].get("after_arm") or "")
    return next((k for k in AFTER_KEYWORDS if k in after), "other")


def vocabularies(train: list[dict]) -> dict[str, list[str]]:
    return {k: sorted({str(r["state"].get(k) or "") for r in train}) for k in ("env_family", "trm_role", "before_arm", "failure_label")}


def features(row: dict, vocab: dict[str, list[str]]) -> list[float]:
    """Pre-repair features only: one-hots over the train vocabulary plus an unknown slot."""
    st = row["state"]
    vec = []
    for name, values in vocab.items():
        v = str(st.get(name) or "")
        vec += [float(v == x) for x in values] + [float(v not in values)]
    kw = after_keyword(row)
    vec += [float(kw == k) for k in AFTER_KEYWORDS + ("other",)]
    vec.append(float(st.get("before_reward") or 0.0))
    return vec


def lookup_predictor(train: list[dict]):
    cells: dict[tuple, Counter] = defaultdict(Counter)
    for r in train:
        cells[(r["state"]["failure_label"], after_keyword(r))][(r["action"], r["target_action"])] += 1
    overall = Counter((r["action"], r["target_action"]) for r in train).most_common(1)[0][0]

    def predict(row: dict) -> tuple[str, str]:
        cell = cells.get((row["state"]["failure_label"], after_keyword(row)))
        return cell.most_common(1)[0][0] if cell else overall

    return predict


def knn_predictor(train: list[dict], k: int = 5):
    vocab = vocabularies(train)
    feats = [(features(r, vocab), r) for r in train]

    def predict(row: dict) -> tuple[str, str]:
        target = features(row, vocab)
        near = [r for _, r in sorted(feats, key=lambda fr: math.dist(fr[0], target))[:k]]
        repair = Counter(r["action"] for r in near).most_common(1)[0][0]
        commits = sum(r["target_action"] == "commit" for r in near)
        return repair, ("commit" if commits * 2 > len(near) else "reject_or_abstain")

    return predict


def train_repair_trm(train: list[dict], val: list[dict], seed: int = 0, hidden: int = 2048, steps: int = 4):
    """TinyTRM (TRM-cv's structure) with a repair head and a commit head; epochs picked on val joint accuracy."""
    import torch
    from torch import nn

    vocab = vocabularies(train)
    actions = allowed_actions(train)

    class RepairTRM(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden)
            self.transition = nn.Linear(hidden, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.repair_head = nn.Linear(hidden, len(actions))
            self.decision_head = nn.Linear(hidden, 2)

        def forward(self, x):
            h = torch.tanh(self.input_proj(x))
            for _ in range(steps):
                h = self.norm(h + torch.tanh(self.transition(h)))
            return self.repair_head(h), self.decision_head(h)

    def tensors(rows):
        x = torch.tensor([features(r, vocab) for r in rows], dtype=torch.float32)
        return x, torch.tensor([actions.index(r["action"]) for r in rows]), torch.tensor([s2.OPTIONS.index(r["target_action"]) for r in rows])

    xt, rt, at = tensors(train)
    xv, rv, av = tensors(val)
    best = None
    for epochs in (8, 25, 50, 100):
        torch.manual_seed(seed)
        model = RepairTRM(xt.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        for _ in range(epochs):
            optimizer.zero_grad()
            rl, al = model(xt)
            loss = nn.functional.cross_entropy(rl, rt) + nn.functional.cross_entropy(al, at)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            rl, al = model(xv)
            joint = ((rl.argmax(-1) == rv) & (al.argmax(-1) == av)).float().mean().item()
        if best is None or joint > best[0]:
            best = (joint, epochs, model)
    val_joint, epochs, model = best
    model.eval()

    def predict(row: dict) -> tuple[str, str]:
        with torch.no_grad():
            rl, al = model(torch.tensor([features(row, vocab)], dtype=torch.float32))
        return actions[int(rl[0].argmax())], s2.OPTIONS[int(al[0].argmax())]

    return predict, {"params_total": sum(p.numel() for p in model.parameters()), "epochs_selected": epochs, "val_joint": val_joint}


def import_receipts() -> dict:
    """The published 3B / 9B / 27B rows, rescored in the shared schema. Returns reproduction checks."""
    data = load_rows()
    by_key = {key(r): r for r in eval_rows(data)}
    checks = {}
    for model, files in RECEIPTS.items():
        for path, arms in files:
            published = [json.loads(x) for x in git_show(HERMES_SKILLS, path).splitlines() if x.strip()]
            for arm, protocol in arms.items():
                rows = [p for p in published if p["arm"] == arm]
                out = []
                for p in rows:
                    row = by_key[f"{p['eval_split']}:{p['case_id']}"]
                    out.append(score(row, p["predicted_repair_action"], p["predicted_target_action"],
                                     retrieved=p.get("retrieved_case_ids"), published_joint=p["joint_correct"]))
                if len(out) != len(by_key) or any(o["joint_ok"] != bool(o["published_joint"]) for o in out):
                    raise RuntimeError(f"{model} {arm}: rescoring does not reproduce the published rows")
                write(model, protocol, out, {"neural": True, "source": f"Hermes-Skills HEAD:{path}", "published_arm": arm})
                if protocol == "retrieval-published":
                    ported = {k: [e["state"]["case_id"] for e in published_retrieve(data["train"], by_key[k])] for k in by_key}
                    checks[f"{model} retrieval port"] = sum(o["retrieved"] == ported[o["key"]] for o in out) / len(out)
                if protocol in ("action-space", "static-gate", "validator"):
                    checks[f"{model} {protocol} MeTTa repair port"] = sum(o["pred_repair"] == metta_repair_action(by_key[o["key"]]) for o in out) / len(out)
    return checks


def run_cpu() -> None:
    data = load_rows()
    rows, train = eval_rows(data), data["train"]
    checks = import_receipts()
    write("MeTTa-validator", "validator", [score(r, metta_repair_action(r), metta_validator_target_action(r)) for r in rows],
          {"neural": False, "reads": "bucket (the repair outcome)"})
    write("lookup", "lookup", [score(r, *lookup_predictor(train)(r)) for r in rows], {"neural": False, "trained_on": "near-miss train"})
    knn = knn_predictor(train)
    write("kNN", "features", [score(r, *knn(r)) for r in rows], {"neural": False, "k": 5, "trained_on": "near-miss train"})
    for seed in (0, 1, 2):
        predict, info = train_repair_trm(train, data["val_seen"], seed=seed)
        record = "Repair-TRM" if seed == 0 else f"Repair-TRM-s{seed}"
        write(record, "features", [score(r, *predict(r)) for r in rows],
              {"neural": True, "seed": seed, "trained_on": "near-miss train (val_seen for epoch selection)", **info})
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
