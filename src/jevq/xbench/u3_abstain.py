"""SPEC-U3 part A: an abstain path for hermes-lite's TRM skill router.

`TinyRecursiveSkillRouter` has an `abstain_head` that its training never fits and `trm_typed` never
reads. Here only that head is fitted (the registered score path is frozen), on hermes-lite's registered
S7 train rows plus each contract's positive examples. A threshold on the router's top probability is
the descriptive baseline. Outputs: results/u2/<router>/route.jsonl (U2 items, composed with the U2 gates
by scripts/u3_report.py) and results/u3/s7/<router>.jsonl (registered S7 held cases).
"""

from __future__ import annotations

import json

from ..config import ROOT
from . import s7_routing as s7
from . import u2_joint as u
from .u2_run import write as write_u2

OUT = ROOT / "results" / "u3"
ABSTAIN_ARM = "SkillRouter-TRM-abstain"
THRESHOLD_ARM = "SkillRouter-TRM-threshold"
STEPS = 500
LR = 1e-2


def training_rows() -> list[tuple[str, int]]:
    """(query, abstain label): S7 train rows (negatives -> 1) and contract positive examples (-> 0)."""
    rows = [(str(r["query"]), int(r["kind"] == "negative")) for r in s7._jsonl(s7.REGISTERED / "train_rows.jsonl")]
    for contract in s7.contracts():
        rows += [(str(q), 0) for q in contract.get("route", {}).get("positive_examples", [])]
    return rows


def _features(route_query: str, candidates: list[dict]):
    import torch

    contracts = import_contracts()
    return torch.tensor([[contracts.route_feature_vector(route_query, c, contracts.deterministic_route_score(route_query, c))
                          for c in candidates]], dtype=torch.float32)


def import_contracts():
    from .foreign import HERMES_LITE, import_from

    return import_from(HERMES_LITE, "agent.lean_contracts", subdir="src")


def view(query: str, primary: list[dict], model) -> dict:
    """What the TRM router sees and says for one request: shortlist, its choice, top prob, abstain prob."""
    import torch

    route_query, candidates = s7.shortlist(query, primary)
    x = _features(route_query, candidates)
    with torch.no_grad():
        logits, abstain = model(x)
    probs = torch.softmax(logits[0, : len(candidates)], dim=-1).tolist()
    best = max(range(len(probs)), key=probs.__getitem__)
    return {"choice": str(candidates[best]["name"]), "top_prob": probs[best], "abstain_prob": torch.sigmoid(abstain[0]).item(),
            "options": [str(c["name"]) for c in candidates], "probs": probs}


def fit_abstain_head(model, rows: list[tuple[str, int]], primary: list[dict]) -> dict:
    """Full-batch logistic fit of `abstain_head` only; deterministic from the checkpoint's head weights."""
    import torch

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.abstain_head.parameters():
        p.requires_grad_(True)
    xs, ys = [], []
    for query, label in rows:
        route_query, candidates = s7.shortlist(query, primary)
        xs.append(_features(route_query, candidates))
        ys.append(float(label))
    # Every shortlist has five candidates (25 primaries), so the rows stack into one batch.
    x, y = torch.cat(xs), torch.tensor(ys)
    optimizer = torch.optim.AdamW(model.abstain_head.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    first = None
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = loss_fn(model(x)[1], y)
        first = first if first is not None else loss.item()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(x)[1]) > 0.5).float() == y).float().mean().item()
    return {"rows": len(rows), "negatives": int(sum(ys)), "loss_first": first, "loss_last": loss.item(), "train_accuracy": acc,
            "steps": STEPS, "lr": LR}


def pick_threshold(views: list[dict], labels: list[int]) -> float:
    """tau maximising abstain accuracy (abstain iff top_prob < tau); ties go to the smaller tau."""
    tops = sorted({v["top_prob"] for v in views})
    grid = [0.0] + [(a + b) / 2 for a, b in zip(tops, tops[1:])] + [1.0 + 1e-9]
    best = max(grid, key=lambda t: (sum((v["top_prob"] < t) == bool(y) for v, y in zip(views, labels)), -t))
    return best


def run() -> None:
    import torch

    primary = s7.contracts()
    model = s7.load_trm()
    rows = training_rows()
    fit = fit_abstain_head(model, rows, primary)
    train_views = [view(q, primary, model) for q, _ in rows]
    tau = pick_threshold(train_views, [y for _, y in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "abstain").mkdir(exist_ok=True)
    torch.save(model.abstain_head.state_dict(), OUT / "abstain" / "abstain_head.pt")

    def decide(v: dict, arm: str) -> str:
        abstain = v["abstain_prob"] > 0.5 if arm == ABSTAIN_ARM else v["top_prob"] < tau
        return s7.ABSTAIN if abstain else v["choice"]

    m = s7.mesh()
    ram = s7.load_ram()
    items = u.build_items()
    held = s7._jsonl(s7.REGISTERED / "held_cases.jsonl")
    for arm in (ABSTAIN_ARM, THRESHOLD_ARM):
        out = []
        for item in items:
            v = view(str(item["query"]), primary, model)
            native, _, _ = m._arm_selection("trm_typed", str(item["query"]), primary, ram, model)
            if str(native["name"]) != v["choice"]:
                raise RuntimeError(f"{item['item_id']}: TRM choice differs from hermes-lite trm_typed")
            out.append({"item_id": item["item_id"], "pred": decide(v, arm), "kind": item["kind"], "variant": item["variant"],
                        "trm_choice": v["choice"], "top_prob": v["top_prob"], "abstain_prob": v["abstain_prob"]})
        write_u2(arm, out, None, {"neural": True, "base": "SkillRouter-TRM (registered trm_router.pt)", "spec": "SPEC-U3 part A",
                                  "abstain_fit": fit, "threshold": tau})
        s7_rows = []
        for case in held:
            v = view(str(case["query"]), primary, model)
            pred = decide(v, arm)
            s7_rows.append({"item_id": case["case_id"], "kind": case["kind"], "perturbation": case["perturbation"], "pred": pred,
                            "correct": s7.passes(case, pred), "trm_choice": v["choice"], "abstain_prob": v["abstain_prob"], "top_prob": v["top_prob"]})
        (OUT / "s7").mkdir(exist_ok=True)
        (OUT / "s7" / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s7_rows), encoding="utf-8")
    print(json.dumps({"fit": fit, "threshold": tau}, indent=2))
