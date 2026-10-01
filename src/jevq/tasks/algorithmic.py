"""Arithmetic / algorithmic tasks. Difficulty = number of sequential operations."""

from __future__ import annotations

import random

from .base import pick_options


def gen_arith_chain(rng: random.Random, difficulty: int):
    """Fully parenthesised left-nested integer expression with difficulty+1 operations."""
    value = rng.randint(2, 9)
    expr = str(value)
    operand = 0
    trace = []
    for _ in range(difficulty + 1):
        op = rng.choice("+-*")
        operand = rng.randint(2, 4) if op == "*" else rng.randint(2, 9)
        expr = f"({expr} {op} {operand})"
        result = {"+": value + operand, "-": value - operand, "*": value * operand}[op]
        trace.append(f"{value} {op} {operand} = {result}.")
        value = result
    expr = expr[1:-1]  # drop the outermost parentheses

    offsets = [1, -1, 2, -2, 10, -10, operand, -operand]
    rng.shuffle(offsets)
    answer = str(value)
    options = pick_options(rng, answer, [str(value + off) for off in offsets])
    meta = {"n_ops": difficulty + 1, "expr": expr, "rationale": " ".join(trace)}
    return f"Compute {expr}.", answer, options, meta


def gen_var_trace(rng: random.Random, difficulty: int):
    """Track three integer variables through difficulty+1 updates; report one of them."""
    names = ["x", "y", "z"]
    values = {name: rng.randint(1, 9) for name in names}
    initial = dict(values)
    steps, touched, trace = [], [], []
    for _ in range(difficulty + 1):
        kind = rng.choice(["swap", "add", "copy", "inc"])
        a, b = rng.sample(names, 2)
        if kind == "swap":
            values[a], values[b] = values[b], values[a]
            steps.append(f"Swap {a} and {b}.")
            touched += [a, b]
        elif kind == "add":
            values[b] += values[a]
            steps.append(f"Add {a} to {b}.")
            touched.append(b)
        elif kind == "copy":
            values[a] = values[b]
            steps.append(f"Set {a} to {b}.")
            touched.append(a)
        else:
            amount = rng.randint(1, 5)
            values[a] += amount
            steps.append(f"Increase {a} by {amount}.")
            touched.append(a)
        state = ", ".join(f"{name} = {values[name]}" for name in names)
        trace.append(f"After step {len(steps)}: {state}.")

    target = rng.choice(touched)
    answer = str(values[target])
    setup = ", ".join(f"{name} = {initial[name]}" for name in names)
    question = f"{setup}. {' '.join(steps)} What is {target}?"
    distractors = [str(values[n]) for n in names if n != target] + [str(initial[target])]
    distractors += [str(values[target] + off) for off in (1, -1, 2, -2, 3)]
    options = pick_options(rng, answer, distractors)
    return question, answer, options, {"n_steps": difficulty + 1, "rationale": " ".join(trace)}
