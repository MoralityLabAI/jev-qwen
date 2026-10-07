"""Subprocess body for time-capped Hermes-Skills projections (SPEC-U5). Imports nothing heavy, so a
spawned worker starts quickly; the caller passes the committed projection source as text."""

from __future__ import annotations

import time
import types


def run(source: str, function: str, candidate, expected, queue) -> None:
    module = types.ModuleType("hermes_skills_camp_gate_micro_env")
    exec(compile(source, "Hermes-Skills HEAD projection source", "exec"), module.__dict__)  # noqa: S102
    started = time.perf_counter()
    out = getattr(module, function)(candidate, expected)
    queue.put((out, time.perf_counter() - started))
