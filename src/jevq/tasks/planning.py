"""Short planning: fewest edges between two nodes of a small undirected graph."""

from __future__ import annotations

import random
from collections import deque

from .base import pick_options

# Letters that do not collide with the option labels A-H.
NODE_NAMES = list("JKLMNPQRSTUVWXYZ")


def shortest_hops(edges: set[frozenset], source: str, target: str) -> int | None:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        a, b = tuple(edge)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen, queue = {source}, deque([(source, 0)])
    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def gen_graph_hops(rng: random.Random, difficulty: int):
    """Shortest path of exactly difficulty+1 edges, hidden among extra nodes and edges."""
    hops = difficulty + 1
    nodes = rng.sample(NODE_NAMES, hops + 4)
    backbone, extras = nodes[: hops + 1], nodes[hops + 1 :]
    source, target = backbone[0], backbone[-1]
    edges = {frozenset(pair) for pair in zip(backbone, backbone[1:])}

    for extra in extras:  # hang each extra node off the graph built so far
        anchor = rng.choice([n for n in nodes if n != extra and any(n in e for e in edges)])
        edges.add(frozenset((extra, anchor)))
    for _ in range(hops + 2):  # extra edges, rejected if they create a shortcut
        edge = frozenset(rng.sample(nodes, 2))
        if edge in edges:
            continue
        if shortest_hops(edges | {edge}, source, target) == hops:
            edges.add(edge)

    # Sets iterate in hash order, which varies between processes; sort before drawing from rng.
    rendered = [f"{a}-{b}" if rng.random() < 0.5 else f"{b}-{a}" for a, b in sorted(sorted(e) for e in edges)]
    rng.shuffle(rendered)

    answer = str(hops)
    distractors = [str(hops + 1), str(hops - 1), str(hops + 2), str(hops + 3)]
    rng.shuffle(distractors)
    options = pick_options(rng, answer, [d for d in distractors if d != "0"])
    question = (
        f"Edges: {', '.join(rendered)}. "
        f"What is the fewest number of edges on a path from {source} to {target}?"
    )
    layers = [
        f"{dist} from {source}: {', '.join(sorted(n for n in nodes if shortest_hops(edges, source, n) == dist))}."
        for dist in range(1, hops + 1)
    ]
    rationale = "Nodes at distance " + " Distance ".join(layers)
    return question, answer, options, {"hops": hops, "n_edges": len(edges), "rationale": rationale}
