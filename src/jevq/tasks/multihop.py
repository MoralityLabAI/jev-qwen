"""Short answer, several inferential steps: follow a relation chain, then read an attribute."""

from __future__ import annotations

import random

from .base import pick_options
from .logic import NAMES

CITIES = ["Rome", "Oslo", "Lima", "Cairo", "Tokyo", "Perth", "Quito", "Delhi", "Hanoi", "Turin"]


def gen_relation_hops(rng: random.Random, difficulty: int):
    """`difficulty` boss-of hops, then one city lookup. Facts are shuffled and include distractors."""
    hops = difficulty
    people = rng.sample(NAMES, hops + 3)
    chain, extras = people[: hops + 1], people[hops + 1 :]
    city = dict(zip(people, rng.sample(CITIES, len(people))))

    facts = [f"{chain[i]}'s boss is {chain[i + 1]}." for i in range(hops)]
    facts.append(f"{extras[0]}'s boss is {extras[1]}.")
    facts += [f"{person} lives in {city[person]}." for person in people]
    rng.shuffle(facts)

    answer = city[chain[-1]]
    subject = f"{chain[0]}'s boss" + "'s boss" * (hops - 1)
    question = f"{' '.join(facts)} Where does {subject} live?"
    distractors = [city[p] for p in people if p != chain[-1]]
    rng.shuffle(distractors)
    # The start of the chain is the most tempting wrong answer; always offer it.
    options = pick_options(rng, answer, [city[chain[0]]] + distractors)
    return question, answer, options, {"hops": hops}
