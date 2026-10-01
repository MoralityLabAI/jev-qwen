"""Shared task plumbing: the Example type, prompt rendering, answer normalisation.

Qwen3.5-4B-Base is a pretrained-only model with no chat template, so every prompt
is a plain few-shot completion. The few-shot block is identical for every example
of a task, which makes it a shared prefix with one short question branch per example.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

LABELS = "ABCDEFGH"

GENERATE_INSTRUCTION = "Answer each question with only the final answer."
CHOICE_INSTRUCTION = "Choose the correct option for each question. Answer with the letter only."
COT_INSTRUCTION = "Answer each question. Write one line of reasoning, then the final answer."


@dataclass
class Example:
    id: str
    task: str
    task_class: str
    difficulty: int
    question: str
    answer: str
    options: list[str]  # contains `answer`; order is already shuffled
    meta: dict = field(default_factory=dict)  # always has "rationale": a one-line worked solution

    @property
    def answer_index(self) -> int:
        return self.options.index(self.answer)


def make_rng(seed: int, task: str, split: str, difficulty: int = 0) -> random.Random:
    # String seeds hash deterministically across processes, unlike hash().
    return random.Random(f"{seed}|{task}|{split}|{difficulty}")


def pick_options(rng: random.Random, answer: str, distractors: list[str], k: int = 4) -> list[str]:
    """`answer` plus up to k-1 distinct distractors, shuffled."""
    options = [answer]
    for candidate in distractors:
        if candidate not in options:
            options.append(candidate)
        if len(options) == k:
            break
    rng.shuffle(options)
    return options


def render_generate(shots: list[Example], example: Example) -> str:
    lines = [GENERATE_INSTRUCTION, ""]
    for shot in shots:
        lines += [f"Q: {shot.question}", f"A: {shot.answer}", ""]
    lines += [f"Q: {example.question}", "A:"]
    return "\n".join(lines)


def _choice_block(example: Example) -> list[str]:
    lines = [f"Q: {example.question}", "Options:"]
    lines += [f"{LABELS[i]}) {option}" for i, option in enumerate(example.options)]
    return lines


def render_choice(shots: list[Example], example: Example) -> str:
    lines = [CHOICE_INSTRUCTION, ""]
    for shot in shots:
        lines += _choice_block(shot) + [f"Answer: {LABELS[shot.answer_index]}", ""]
    lines += _choice_block(example) + ["Answer:"]
    return "\n".join(lines)


def render_cot(shots: list[Example], example: Example) -> str:
    """Reasoning-trace prompt: each shot shows a one-line worked solution before its answer."""
    lines = [COT_INSTRUCTION, ""]
    for shot in shots:
        lines += [f"Q: {shot.question}", f"Reasoning: {shot.meta['rationale']}", f"A: {shot.answer}", ""]
    lines += [f"Q: {example.question}", "Reasoning:"]
    return "\n".join(lines)


def normalize(text: str) -> str:
    return text.strip().strip(".").strip().lower()


def first_answer_line(generated: str) -> str:
    """The first non-empty line of a completion: the model's answer to the open question."""
    for line in generated.splitlines():
        if line.strip():
            return line.strip()
    return ""


def direct_done(generated: str) -> bool:
    """`generate` readout: stop once a newline follows the first non-blank text."""
    return "\n" in generated.lstrip()


_COT_ANSWER = re.compile(r"(?:^|\n)A:[ \t]*([^\n]*)")


def cot_answer(generated: str) -> str:
    """`generate_cot` readout: the text on the first `A:` line, or '' if the model never answered."""
    match = _COT_ANSWER.search(generated)
    return match.group(1).strip() if match else ""


def cot_done(generated: str) -> bool:
    """Stop once the `A:` line is complete, or the model has moved on to a new question."""
    return re.search(r"(?:^|\n)A:[^\n]*\n", generated) is not None or "\nQ:" in generated
