"""Statistics for xbench (SPEC section 8). Pure Python; deterministic given the seed."""

from __future__ import annotations

import math
import random
from typing import Sequence


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial rate; (None, None) when n is 0."""
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (monotone), keyed like the input."""
    order = sorted(p_values, key=p_values.get)
    m = len(order)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def bootstrap_diff(
    a: Sequence[float], b: Sequence[float], resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired percentile bootstrap of mean(a) - mean(b): (estimate, low, high)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples of equal, non-zero length required")
    rng = random.Random(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    estimate = sum(diffs) / n
    draws = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return estimate, draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    total, out = len(confidences), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo < c <= hi or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            out += len(idx) / total * abs(acc - conf)
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties; None if undefined."""
    if len(x) != len(y) or len(x) < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def kaplan_meier(times: Sequence[int | None], horizon: int) -> list[float]:
    """Survival curve S(t), t = 1..horizon. `None` = never flipped (censored at the horizon)."""
    at_risk = len(times)
    survival, curve = 1.0, []
    for t in range(1, horizon + 1):
        events = sum(1 for value in times if value == t)
        if at_risk:
            survival *= 1 - events / at_risk
        at_risk -= events
        curve.append(survival)
    return curve


def half_life(curve: Sequence[float]) -> int | None:
    """First turn at which survival is <= 0.5 (SPEC section 7); None if it never is."""
    return next((t + 1 for t, s in enumerate(curve) if s <= 0.5), None)


def log_rank(times_a: Sequence[int | None], times_b: Sequence[int | None], horizon: int) -> float:
    """Two-sided log-rank p-value (chi-square, 1 df) for two flip-time samples."""
    observed_minus_expected, variance = 0.0, 0.0
    risk_a, risk_b = len(times_a), len(times_b)
    for t in range(1, horizon + 1):
        da = sum(1 for v in times_a if v == t)
        db = sum(1 for v in times_b if v == t)
        n, d = risk_a + risk_b, da + db
        if n > 1 and d:
            observed_minus_expected += da - d * risk_a / n
            variance += d * (risk_a / n) * (risk_b / n) * (n - d) / (n - 1)
        risk_a -= da
        risk_b -= db
    if variance == 0:
        return 1.0
    chi2 = observed_minus_expected**2 / variance
    return math.erfc(math.sqrt(chi2 / 2))
