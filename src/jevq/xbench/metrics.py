"""Per-record metrics over unified item rows (SPEC section 5)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

from .stats import ece, wilson


def _rate(rows: Sequence[dict], key: str) -> dict:
    flagged = [r for r in rows if r.get(key) is not None]
    hits = sum(bool(r[key]) for r in flagged)
    low, high = wilson(hits, len(flagged))
    return {"rate": hits / len(flagged) if flagged else None, "n": len(flagged), "k": hits, "wilson95": [low, high]}


def summarize(rows: Sequence[dict], by: Sequence[str] = ("family", "depth")) -> dict:
    """Accuracy, gate error rates, calibration and cost for a list of item rows."""
    n = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    out: dict = {
        "n": n,
        "accuracy": correct / n if n else None,
        "accuracy_wilson95": list(wilson(correct, n)),
        "unsafe": _rate(rows, "unsafe"),
        "over_refusal": _rate(rows, "over_refusal"),
        "passes_mean": _mean(r.get("passes") for r in rows),
        "emitted_tokens_mean": _mean(r.get("emitted_tokens") for r in rows),
        "latency_s_median": _median(r.get("latency_s") for r in rows),
    }
    with_probs = [r for r in rows if r.get("probs")]
    if with_probs:
        conf = [max(r["probs"]) for r in with_probs]
        ok = [bool(r["correct"]) for r in with_probs]
        gold = [r["options"].index(r["gold"]) if r["gold"] in r["options"] else None for r in with_probs]
        scored = [(r, g) for r, g in zip(with_probs, gold) if g is not None]
        out["calibration"] = {
            "n": len(with_probs),
            "ece": ece(conf, ok),
            "nll": _mean(-math.log(max(r["probs"][g], 1e-12)) for r, g in scored),
            "brier": _mean(sum((p - (i == g)) ** 2 for i, p in enumerate(r["probs"])) for r, g in scored),
            "confidence_mean": sum(conf) / len(conf),
        }
        out["position_bias"] = position_bias(with_probs)
    abstain = [r for r in rows if r.get("abstain_gold") is not None]
    if abstain:
        tp = sum(r["abstain_gold"] and r["abstain_pred"] for r in abstain)
        pred = sum(bool(r["abstain_pred"]) for r in abstain)
        gold = sum(bool(r["abstain_gold"]) for r in abstain)
        out["abstention"] = {
            "precision": tp / pred if pred else None,
            "recall": tp / gold if gold else None,
            "n_gold": gold,
            "n_pred": pred,
        }
    for key in by:
        groups: dict = defaultdict(list)
        for r in rows:
            if r.get(key) is not None:
                groups[str(r[key])].append(r)
        if groups:
            out[f"accuracy_by_{key}"] = {
                k: {"n": len(v), "accuracy": sum(bool(r["correct"]) for r in v) / len(v)}
                for k, v in sorted(groups.items(), key=lambda kv: _sort_key(kv[0]))
            }
    return out


def position_bias(rows: Sequence[dict]) -> dict | None:
    """Most-predicted option-position share vs gold, over the most common option count."""
    if not rows:
        return None
    size = Counter(len(r["options"]) for r in rows).most_common(1)[0][0]
    same = [r for r in rows if len(r["options"]) == size]
    pred = Counter(max(range(size), key=r["probs"].__getitem__) for r in same)
    gold = Counter(r["options"].index(r["gold"]) for r in same if r["gold"] in r["options"])
    return {
        "n_options": size,
        "n": len(same),
        "top_pred_share": max(pred.values()) / len(same),
        "top_gold_share": max(gold.values()) / len(same) if gold else None,
    }


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _median(values) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _sort_key(value: str):
    return (0, int(value)) if value.lstrip("-").isdigit() else (1, value)
