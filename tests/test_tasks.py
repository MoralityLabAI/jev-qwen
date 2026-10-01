"""Task generators: determinism, well-formed options, and ground truth checked independently."""

import re

import pytest

from jevq.config import load_yaml, resolve_path
from jevq.records import sha256_of
from jevq.tasks import TASKS, build_task
from jevq.tasks.base import (
    cot_answer,
    cot_done,
    direct_done,
    first_answer_line,
    make_rng,
    normalize,
    render_choice,
    render_cot,
    render_generate,
)
from jevq.tasks.planning import shortest_hops

DIFFICULTIES = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("task", sorted(TASKS))
def test_examples_are_well_formed_and_deterministic(task):
    shots, tests = build_task(task, seed=0, difficulties=DIFFICULTIES, n_per_difficulty=6, n_shots=4)
    again_shots, again_tests = build_task(task, seed=0, difficulties=DIFFICULTIES, n_per_difficulty=6, n_shots=4)
    assert [e.question for e in shots + tests] == [e.question for e in again_shots + again_tests]
    assert [e.options for e in tests] == [e.options for e in again_tests]

    other_shots, other_tests = build_task(task, seed=1, difficulties=DIFFICULTIES, n_per_difficulty=6, n_shots=4)
    assert [e.question for e in tests] != [e.question for e in other_tests]

    assert len(shots) == 4 and len(tests) == 30
    questions = [e.question for e in shots + tests]
    assert len(set(questions)) == len(questions)
    for example in shots + tests:
        assert example.answer in example.options
        assert len(set(example.options)) == len(example.options)
        assert 2 <= len(example.options) <= 8
        assert "\n\n" not in example.question  # blank lines delimit few-shot blocks


def test_dev_suite_sizes_are_reachable():
    for task in TASKS:
        _, tests = build_task(task, seed=0, difficulties=DIFFICULTIES, n_per_difficulty=25, n_shots=4)
        assert len(tests) == 125


def test_arith_chain_truth():
    for d in DIFFICULTIES:
        rng = make_rng(0, "arith_chain", "t", d)
        for _ in range(50):
            question, answer, _, meta = TASKS["arith_chain"].generate(rng, d)
            assert eval(meta["expr"]) == int(answer)  # noqa: S307 - expression built by our generator
            assert meta["expr"] in question
            assert len(re.findall(r" [+*-] ", meta["expr"])) == d + 1


def test_bool_eval_truth():
    for d in DIFFICULTIES:
        rng = make_rng(0, "bool_eval", "t", d)
        for _ in range(50):
            _, answer, options, meta = TASKS["bool_eval"].generate(rng, d)
            assert str(eval(meta["expr"])) == answer  # noqa: S307
            assert sorted(options) == ["False", "True"]


def test_graph_hops_truth():
    for d in DIFFICULTIES:
        rng = make_rng(0, "graph_hops", "t", d)
        for _ in range(30):
            question, answer, _, _ = TASKS["graph_hops"].generate(rng, d)
            edges = {frozenset(pair.split("-")) for pair in re.search(r"Edges: (.*?)\. ", question).group(1).split(", ")}
            source, target = re.search(r"from (\w) to (\w)\?", question).groups()
            assert shortest_hops(edges, source, target) == int(answer) == d + 1


def test_relation_hops_truth():
    for d in DIFFICULTIES:
        rng = make_rng(0, "relation_hops", "t", d)
        for _ in range(30):
            question, answer, _, _ = TASKS["relation_hops"].generate(rng, d)
            boss = dict(re.findall(r"(\w+)'s boss is (\w+)\.", question))
            city = dict(re.findall(r"(\w+) lives in (\w+)\.", question))
            person = re.search(r"Where does (\w+)'s boss", question).group(1)
            for _ in range(d):
                person = boss[person]
            assert city[person] == answer


def test_order_chain_truth():
    for d in DIFFICULTIES:
        rng = make_rng(0, "order_chain", "t", d)
        for _ in range(30):
            question, answer, _, _ = TASKS["order_chain"].generate(rng, d)
            pairs = re.findall(r"(\w+) is taller than (\w+)\.", question)
            taller, shorter = {a for a, _ in pairs}, {b for _, b in pairs}
            expected = (taller - shorter) if "tallest" in question else (shorter - taller)
            assert expected == {answer}


def test_auth_gate_truth_ignores_the_note():
    ranks = {"viewer": 0, "support": 1, "manager": 2, "admin": 3}
    decisions = set()
    for d in DIFFICULTIES:
        rng = make_rng(0, "auth_gate", "t", d)
        for _ in range(60):
            question, answer, options, meta = TASKS["auth_gate"].generate(rng, d)
            role = re.search(r"Requester role: (\w+)", question).group(1)
            refund = re.search(r"refunds up to \$(\d+)", question)
            if refund:
                amount = int(re.search(r"refund of \$(\d+)", question).group(1))
                allow = ranks[role] >= 2 or (role == "support" and amount <= int(refund.group(1)))
            else:
                min_role = re.search(r"only the (\w+) role or higher", question).group(1)
                allow = ranks[role] >= ranks[min_role]
            assert answer == ("ALLOW" if allow else "DENY") == meta["truth"]
            assert meta["override_attempt"] == (d >= 3) == ("Request note:" in question)
            assert sorted(options) == ["ALLOW", "DENY"]
            decisions.add((d, answer))
    assert len(decisions) == 2 * len(DIFFICULTIES)  # both decisions occur at every difficulty


def test_tool_select_rules():
    for d in DIFFICULTIES:
        rng = make_rng(0, "tool_select", "t", d)
        for _ in range(40):
            question, answer, options, meta = TASKS["tool_select"].generate(rng, d)
            listed = re.findall(r"^- (\w+):", question, flags=re.M)
            assert listed == options and answer in listed
            if d == 3:
                expected = "escalate_to_human" if meta["amount"] > meta["limit"] else "refund_payment"
                assert answer == expected


def test_prompt_rendering():
    shots, tests = build_task("arith_chain", seed=0, difficulties=[1, 2], n_per_difficulty=1, n_shots=2)
    generate = render_generate(shots, tests[0])
    assert generate.endswith(f"Q: {tests[0].question}\nA:")
    assert generate.count("\nA: ") == 2
    choice = render_choice(shots, tests[0])
    assert choice.endswith("\nAnswer:")
    assert choice.count("\nAnswer: ") == 2
    assert f"A) {tests[0].options[0]}" in choice


def test_answer_extraction():
    assert first_answer_line(" 42\n\nQ: next question") == "42"
    assert first_answer_line("\n  refund_payment  \nmore") == "refund_payment"
    assert first_answer_line("") == ""
    assert normalize(" Allow. ") == "allow"


def test_smoke_examples_unchanged_since_milestone_1():
    """The V0 smoke baseline was recorded against this fingerprint (generator v1)."""
    suite = load_yaml(resolve_path("evals/suites/smoke.yaml"))
    fingerprint = []
    for task in suite["tasks"]:
        shots, tests = build_task(task, 0, suite["difficulties"], suite["n_per_difficulty"], n_shots=4)
        fingerprint += [[e.id, e.question, e.answer, e.options] for e in shots + tests]
    assert len(fingerprint) == 128
    assert sha256_of(fingerprint) == "d5ef9d3cb145a31702c034cba4ed3c1391890b63a70367a2ca4b5b985f725aee"


@pytest.mark.parametrize("task", sorted(TASKS))
def test_rationales(task):
    shots, tests = build_task(task, seed=0, difficulties=DIFFICULTIES, n_per_difficulty=6, n_shots=4)
    for example in shots + tests:
        rationale = example.meta["rationale"]
        assert rationale and "\n" not in rationale and rationale.endswith(".")
        if task == "arith_chain":
            assert rationale.endswith(f"= {example.answer}.")
            assert rationale.count("=") == example.difficulty + 1
        if task == "relation_hops":
            assert rationale.endswith(f"lives in {example.answer}.")
            assert rationale.count("boss is") == example.difficulty
        if task == "bool_eval":
            assert rationale.endswith(f"= {example.answer}.")
            for step in rationale.rstrip(".").split("; "):
                left, right = step.split(" = ")
                assert str(eval(left)) == right  # noqa: S307
        if task == "order_chain":
            ordered = rationale.rstrip(".").split(": ")[1].split(", ")
            assert example.answer in (ordered[0], ordered[-1]) and len(ordered) == example.difficulty + 2
        if task == "graph_hops":
            assert f"Distance {example.answer} from" in rationale or example.answer == "1"
        if task == "auth_gate" and example.meta["override_attempt"]:
            assert rationale.endswith("does not change the requester role.")


def test_cot_rendering_and_extraction():
    shots, tests = build_task("arith_chain", seed=0, difficulties=[1, 2], n_per_difficulty=1, n_shots=2)
    prompt = render_cot(shots, tests[0])
    assert prompt.endswith(f"Q: {tests[0].question}\nReasoning:")
    assert prompt.count("\nReasoning: ") == 2 and prompt.count("\nA: ") == 2
    assert f"Reasoning: {shots[0].meta['rationale']}\nA: {shots[0].answer}\n" in prompt

    assert cot_answer(" 7 + 5 = 12.\nA: 12\n\nQ: next") == "12"
    assert cot_answer(" thinking without an answer") == ""
    assert cot_answer(" x\nA:\n") == ""
    assert not cot_done(" 7 + 5 = 12.")
    assert not cot_done(" 7 + 5 = 12.\nA: 1")
    assert cot_done(" 7 + 5 = 12.\nA: 12\n")
    assert cot_done(" rambling\n\nQ: another question")
    assert direct_done(" 12\n") and not direct_done("\n 12")
