"""SPEC-U7 part L: relation buckets, encoding, the three arms' shapes and parameter matching, the
min-over-solutions loss, decoding, and that a looped model can fit a few examples (CPU)."""

import pytest
import torch

from jevq.xbench.foreign import HERMES_LITE, HERMES_SKILLS

needs_both = pytest.mark.skipif(not (HERMES_LITE.exists() and HERMES_SKILLS.exists()), reason="hermes-lite / Hermes-Skills absent")


def test_buckets_are_size_agnostic_relations():
    from jevq.xbench import u7_loop as u7

    b = u7.buckets(3, 4)
    centre = 1 * 4 + 1
    assert b[centre, centre] == 0
    neighbours = {int(b[centre, (1 + dr) * 4 + (1 + dc)]) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}
    assert neighbours == set(range(1, 9))
    assert b[centre, 1 * 4 + 3] == 9 and b[0, 2 * 4 + 0] == 10 and b[0, 2 * 4 + 3] == 11
    # The same relation gets the same bucket on a larger grid.
    big = u7.buckets(10, 10)
    assert big[11, 12] == b[centre, centre + 1] and big[0, 99] == 11


def test_encode_marks_ragged_cells_none():
    from jevq.xbench import u7_loop as u7

    item = {"item_id": "x", "task": {"grid": [["T", "X"], ["X", "X"]], "row_constraints": [1, 0], "col_constraints": [0, 1]},
            "candidate": [["T", "C"], ["X"]]}
    e = u7.encode(item, [[["T", "C"], ["X", "X"]]])
    assert e["tree"] == [1, 0, 0, 0] and e["sym"] == [0, 2, 1, 3] and e["sols"] == [[0, 1, 0, 0]]
    assert u7.encode(dict(item, candidate=[]))["sym"] == [3, 3, 3, 3]


def _toy(n=3, m=3, seed=0):
    import random

    rng = random.Random(seed)
    tree = [int(rng.random() < 0.3) for _ in range(n * m)]
    sol = [0 if t else int(rng.random() < 0.3) for t in tree]
    return {"id": f"t{seed}", "n": n, "m": m, "tree": tree, "sym": [1] * (n * m), "rows": [1] * n, "cols": [1] * m,
            "sols": [sol, [0 if t else 1 - s for t, s in zip(tree, sol)]]}


def test_arms_shapes_params_and_loss():
    from jevq.xbench import u7_loop as u7

    loop, flat, deep = (u7.make_model(a) for a in ("LOOP", "FLAT", "DEEP"))
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(loop) == count(flat) and count(deep) > 10 * count(loop)
    x = u7.batch_tensors([_toy(3, 3, 0), _toy(4, 5, 1)], "cpu")
    assert len(loop(x, 5)) == 5 and len(flat(x, 5)) == 1 and len(deep(x, 5)) == 16
    outs = loop(x, 3)
    assert outs[-1].shape == (2, 20, 2)
    # min over solutions: perfect logits for solution 0 give a loss far below either solution's average.
    sol0 = x["sols"][:, 0, :]
    perfect = [torch.nn.functional.one_hot(sol0, 2).float() * 20.0]
    assert u7.loss_fn(perfect, x) < 1e-3
    grids = u7.decode(perfect[0], x, [_toy(3, 3, 0), _toy(4, 5, 1)])
    assert len(grids[1]) == 4 and len(grids[1][0]) == 5


def test_loop_fits_a_few_examples():
    from jevq.xbench import u7_loop as u7

    torch.manual_seed(0)
    model = u7.make_model("LOOP")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    toys = []
    for s in range(4):  # learnable from the inputs (no absolute positions): the candidate already marks the tents
        t = _toy(3, 4, s)
        t["sym"] = [0 if tr else (2 if v else 1) for tr, v in zip(t["tree"], t["sols"][0])]
        t["sols"] = t["sols"][:1]
        toys.append(t)
    x = u7.batch_tensors(toys, "cpu")
    first = None
    for _ in range(40):
        loss = u7.loss_fn(model(x, 4), x)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first * 0.5
