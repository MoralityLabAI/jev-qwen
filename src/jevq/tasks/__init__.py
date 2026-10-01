"""Task registry and deterministic example construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .actions import gen_tool_select
from .algorithmic import gen_arith_chain, gen_var_trace
from .base import Example, make_rng
from .control import gen_auth_gate
from .logic import gen_bool_eval, gen_order_chain
from .multihop import gen_relation_hops
from .planning import gen_graph_hops

# Bump when any generator changes, so records from different task versions are not compared.
GENERATOR_VERSION = 1


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_class: str
    generate: Callable


TASKS = {
    spec.name: spec
    for spec in [
        TaskSpec("arith_chain", "arithmetic", gen_arith_chain),
        TaskSpec("var_trace", "arithmetic", gen_var_trace),
        TaskSpec("order_chain", "logic", gen_order_chain),
        TaskSpec("bool_eval", "logic", gen_bool_eval),
        TaskSpec("tool_select", "tool_call", gen_tool_select),
        TaskSpec("graph_hops", "planning", gen_graph_hops),
        TaskSpec("relation_hops", "multihop", gen_relation_hops),
        TaskSpec("auth_gate", "control", gen_auth_gate),
    ]
}


def _generate(task: str, seed: int, split: str, difficulty: int, count: int, exclude: set[str]) -> list[Example]:
    spec = TASKS[task]
    rng = make_rng(seed, task, split, difficulty)
    examples: list[Example] = []
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError(f"{task} difficulty {difficulty}: cannot produce {count} distinct questions")
        question, answer, options, meta = spec.generate(rng, difficulty)
        if question in exclude:
            continue
        exclude.add(question)
        examples.append(
            Example(
                id=f"{task}-{split}-d{difficulty}-{len(examples)}",
                task=task,
                task_class=spec.task_class,
                difficulty=difficulty,
                question=question,
                answer=answer,
                options=options,
                meta=meta,
            )
        )
    return examples


def build_task(task: str, seed: int, difficulties: list[int], n_per_difficulty: int, n_shots: int):
    """Returns (shots, test_examples). Shots cycle through the difficulties and never repeat a test question."""
    seen: set[str] = set()
    shot_difficulties = [difficulties[i % len(difficulties)] for i in range(n_shots)]
    pools = {
        d: _generate(task, seed, "shot", d, shot_difficulties.count(d), seen) for d in dict.fromkeys(shot_difficulties)
    }
    shots = [pools[d].pop(0) for d in shot_difficulties]
    tests: list[Example] = []
    for difficulty in difficulties:
        tests += _generate(task, seed, "test", difficulty, n_per_difficulty, seen)
    return shots, tests
