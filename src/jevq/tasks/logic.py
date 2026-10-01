"""Compact logic tasks."""

from __future__ import annotations

import random

from .base import pick_options

NAMES = ["Ann", "Bob", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jon", "Kim", "Lou"]


def gen_order_chain(rng: random.Random, difficulty: int):
    """Transitive ordering over difficulty+2 people, premises given in shuffled order."""
    people = rng.sample(NAMES, difficulty + 2)  # tallest first
    premises = [f"{people[i]} is taller than {people[i + 1]}." for i in range(len(people) - 1)]
    rng.shuffle(premises)
    kind = rng.choice(["tallest", "shortest"])
    answer = people[0] if kind == "tallest" else people[-1]

    others = [p for p in people if p != answer]
    rng.shuffle(others)
    padding = [n for n in NAMES if n not in people]
    rng.shuffle(padding)
    options = pick_options(rng, answer, others + padding)
    question = f"{' '.join(premises)} Who is the {kind}?"
    return question, answer, options, {"chain_length": len(people)}


def _bool_expr(rng: random.Random, depth: int) -> tuple[str, bool]:
    if depth == 0:
        value = rng.choice([True, False])
        return str(value), value
    op = rng.choice(["and", "or", "not"])
    if op == "not":
        text, value = _bool_expr(rng, depth - 1)
        return (f"not {text}" if depth == 1 else f"not ({text})"), not value
    left_text, left = _bool_expr(rng, depth - 1)
    right_text, right = _bool_expr(rng, rng.randint(0, depth - 1))
    if depth > 1:
        left_text = f"({left_text})"
    if " " in right_text:
        right_text = f"({right_text})"
    value = (left and right) if op == "and" else (left or right)
    return f"{left_text} {op} {right_text}", value


def gen_bool_eval(rng: random.Random, difficulty: int):
    """Evaluate a boolean expression of nesting depth difficulty+1."""
    text, value = _bool_expr(rng, difficulty + 1)
    answer = str(value)
    options = pick_options(rng, answer, [str(not value)], k=2)
    return f"Evaluate: {text}", answer, options, {"depth": difficulty + 1, "expr": text}
