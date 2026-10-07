"""SPEC-U5: reject paths for the Campsite flow policies (part R), the Jev trained on the U4 train
items (part J), and repairs of real model proposals on standard and larger puzzles (part P).

Builds on `u4_campsite` (same verifier, generator, projections and prompts). Projections are computed
once per item set and cached in results/u5/projections/<set>.jsonl; dual_repair is time-capped above
6x6 and not run on 10x10 (design probe: over 60 s there). Records: results/u5/{decide,repair}/<set>/<record>.jsonl.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from functools import lru_cache

from ..config import ROOT
from . import u4_campsite as u4
from .jrunner import ChoiceItem, CotItem

OUT = ROOT / "results" / "u5"
LARGE_SEED = 20261013
LARGE = ((8, 60), (10, 60))  # size, puzzles
PROJECTION_CAP_S = 60
DUAL_SKIP_ABOVE = 64  # cells: dual_repair is not run on 10x10 (> 60 s per call in the design probe)
PROPOSERS = ("J-V0", "Bonsai-8B")
SETS = ("u4test", "u5test", "proposals")
POLICIES = ("c_repair_if_c_fail", "dual_repair_if_any_sig_fail")
REJECT_MODES = ("plain", "noop-reject", "verify-reject")


# ---------------------------------------------------------------------------- puzzles and items


@lru_cache(maxsize=None)
def large_puzzles() -> tuple:
    """8x8 and 10x10 puzzles from hermes-lite's own planting and CSP routines (its generator stops at 6x6)."""
    camp = u4.camp()
    rng = random.Random(LARGE_SEED)
    out, seen = [], set()
    for n, count in LARGE:
        made = 0
        while made < count:
            base = [["X"] * n for _ in range(n)]
            cells = [(r, c) for r in range(n) for c in range(n)]
            rng.shuffle(cells)
            for r, c in cells[: int(n * n * 0.2)]:
                base[r][c] = "T"
            planted = camp._assign_one_tent_per_tree(base, rng)
            if planted is None:
                continue
            rows = [sum(x == "C" for x in row) for row in planted]
            cols = [sum(planted[r][c] == "C" for r in range(n)) for c in range(n)]
            task = camp.CampsiteTask.from_payload({"task_id": f"large_{n}x{n}_{made:03d}", "grid": base, "row_constraints": rows,
                                                   "col_constraints": cols})
            if task.hash in seen:
                continue
            solved, _ = camp.solve_candidates(task, max_candidates=1)
            if not solved:
                continue
            u4._check_expected(task, solved[0])
            seen.add(task.hash)
            out.append((task, solved[0]))
            made += 1
    return tuple(out)


def band(task) -> str:
    n = len(task.grid)
    return "std" if n <= 6 else f"{n}x{n}"


@lru_cache(maxsize=None)
def proposal_puzzles() -> tuple:
    return tuple(list(u4.puzzles("u5std")) + list(large_puzzles()))


PROPOSE_INSTRUCTION = (
    u4.RULES + " Solve the puzzle: write the completed grid with one tent (C) for every tree, one row per line, "
    "cells separated by spaces."
)


@lru_cache(maxsize=None)
def propose_shots() -> tuple:
    train = [i for i in u4.build_items("train") if i["kind"] == "correct"]
    return tuple(train[:3])


def puzzle_text(task_payload) -> str:
    return "\n".join(["Puzzle:", u4.grid_text(task_payload["grid"]),
                      f"Row tents: {' '.join(map(str, task_payload['row_constraints']))}",
                      f"Column tents: {' '.join(map(str, task_payload['col_constraints']))}"])


def propose_prompt(task) -> str:
    lines = [PROPOSE_INSTRUCTION, ""]
    for shot in propose_shots():
        lines += [puzzle_text(shot["task"]), "Solution:", u4.grid_text(shot["gold"]), ""]
    return "\n".join(lines + [puzzle_text(task.runtime_payload()), "Solution:"]) + "\n"


def proposal_budget(task) -> int:
    return 2 * len(task.grid) * len(task.grid[0]) + 16


def run_proposer(proposer: str) -> None:
    puzzles = proposal_puzzles()
    rows = []
    if proposer == "J-V0":
        from .jrunner import load_j_bundle, run_cot

        bundle = load_j_bundle(None)
        items = [CotItem(task.task_id, propose_prompt(task), "", (lambda text, n=len(task.grid): json.dumps(u4.parse_grid(text, n))),
                         (lambda text, n=len(task.grid): sum(1 for line in text.split("\n")[:-1] if line.strip()) >= n), {})
                 for task, _ in puzzles]
        for r in run_cot(bundle, items, max_new_tokens=max(proposal_budget(t) for t, _ in puzzles)):
            rows.append({"puzzle_id": r["item_id"], "grid": json.loads(r["pred"]), "raw_text": r["raw_text"],
                         "emitted_tokens": r["emitted_tokens"], "latency_s": r["latency_s"]})
    elif proposer == "Bonsai-8B":
        from . import bonsai

        with bonsai.BonsaiServer() as server:
            for task, _ in puzzles:
                t0 = time.perf_counter()
                out = server.complete(propose_prompt(task), proposal_budget(task), n_probs=0)
                text = out.get("content", "")
                rows.append({"puzzle_id": task.task_id, "grid": u4.parse_grid(text, len(task.grid)), "raw_text": text,
                             "emitted_tokens": out.get("tokens_predicted"), "latency_s": time.perf_counter() - t0})
    else:
        raise ValueError(proposer)
    path = OUT / "proposals"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{proposer}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@lru_cache(maxsize=None)
def proposal_items() -> tuple:
    """One item per (proposer, puzzle): the proposal is the candidate (an unparseable one is an empty grid)."""
    by_id = {task.task_id: (task, gold) for task, gold in proposal_puzzles()}
    items = []
    for proposer in PROPOSERS:
        path = OUT / "proposals" / f"{proposer}.jsonl"
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            task, gold = by_id[row["puzzle_id"]]
            cand = row["grid"] or []
            item = {"item_id": f"u5.prop.{proposer}.{task.task_id}", "pool": "proposals", "kind": band(task), "proposer": proposer,
                    "task": task.runtime_payload(), "candidate": cand, "gold": gold}
            item["proposal_pass"] = u4.verify(item, cand)["official_pass"]
            items.append(item)
    return tuple(items)


def items_for(set_name: str) -> list[dict]:
    return {"u4test": lambda: list(u4.build_items("test")), "u5test": lambda: list(u4.build_items("u5test")),
            "proposals": lambda: list(proposal_items())}[set_name]()


# ---------------------------------------------------------------------------- projections (cached, capped)


def _cells(item) -> int:
    return len(item["task"]["grid"]) * len(item["task"]["grid"][0])


def _capped(item, function: str, source: str, cap: float) -> dict:
    return _capped_many([("one", item, function)], source, cap)["one"]


def _capped_many(jobs: list[tuple[str, dict, str]], source: str, cap: float, workers: int = 6) -> dict:
    """Run (key, item, function) projection jobs in up to `workers` processes, each killed after `cap` s."""
    import queue as queue_mod

    from . import projection_worker

    ctx = mp.get_context("spawn")
    results, pending, running = {}, list(jobs), {}
    while pending or running:
        while pending and len(running) < workers:
            key, item, function = pending.pop(0)
            q = ctx.Queue()
            proc = ctx.Process(target=projection_worker.run, args=(source, function, item["candidate"], item["gold"], q))
            proc.start()
            running[key] = (proc, q, time.monotonic())
        for key, (proc, q, started) in list(running.items()):
            try:
                out, seconds = q.get_nowait()
                proc.join()
                results[key] = {"status": "ok" if out is not None else "none", "grid": out, "seconds": seconds}
            except queue_mod.Empty:
                if time.monotonic() - started > cap:
                    proc.terminate()
                    proc.join()
                    results[key] = {"status": "timeout", "grid": None, "seconds": cap}
                elif not proc.is_alive():
                    time.sleep(0.2)
                    if q.empty():
                        results[key] = {"status": "error", "grid": None, "seconds": time.monotonic() - started}
                    continue
                else:
                    continue
            del running[key]
        time.sleep(0.05)
    return results


def compute_projections(set_name: str) -> dict:
    items = items_for(set_name)
    source = u4.git_show(u4.HERMES_SKILLS, u4.REPAIR_SCRIPT)
    rows, capped = [], []
    for item in items:
        row = {"item_id": item["item_id"]}
        for module, function in (("c_repair", "c_only_projection"), ("dual_repair", "dual_signature_projection")):
            if not u4.rectangular(item, item["candidate"]):
                row[module] = {"status": "none", "grid": None, "seconds": 0.0}
            elif module == "dual_repair" and _cells(item) > DUAL_SKIP_ABOVE:
                row[module] = {"status": "skipped", "grid": None, "seconds": None}
            elif module == "dual_repair" and _cells(item) > 36:
                capped.append(((item["item_id"], module), item, function))
            else:
                t0 = time.perf_counter()
                out = getattr(u4.skills(), function)(item["candidate"], item["gold"])
                row[module] = {"status": "ok" if out is not None else "none", "grid": out, "seconds": time.perf_counter() - t0}
        rows.append(row)
    done = _capped_many(capped, source, PROJECTION_CAP_S)
    for row in rows:
        for module in ("c_repair", "dual_repair"):
            if module not in row:
                row[module] = done[(row["item_id"], module)]
    path = OUT / "projections"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{set_name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return {r["item_id"]: r for r in rows}


@lru_cache(maxsize=None)
def projections(set_name: str) -> dict:
    path = OUT / "projections" / f"{set_name}.jsonl"
    return {r["item_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def committed(set_name: str, item, action: str):
    if action == "reject":
        return None
    if action == "commit":
        return item["candidate"]
    grid = projections(set_name)[item["item_id"]][action]["grid"]
    return grid if grid is not None else item["candidate"]


def outcome(set_name: str, item, action: str, **extra) -> dict:
    grid = committed(set_name, item, action)
    passed = grid is not None and u4.verify(item, grid)["official_pass"]
    return {"item_id": item["item_id"], "kind": item["kind"], "proposer": item.get("proposer"), "action": action, "success": passed,
            "unsafe": grid is not None and not passed, "rejected": grid is None, **extra}


def best_action(set_name: str, item) -> str:
    for action in u4.ACTIONS[:3]:
        if outcome(set_name, item, action)["success"]:
            return action
    return "reject"


def with_reject(set_name: str, item, action: str, mode: str) -> str:
    """Part R. noop-reject: never commit the unchanged candidate once the verifier has failed it (no extra
    verifier call). verify-reject: commit only a grid that passes the verifier (one post-repair check)."""
    if mode == "plain" or action == "reject":
        return action
    if mode == "verify-reject":
        return action if outcome(set_name, item, action)["success"] else "reject"
    pre_pass = u4.verify(item, item["candidate"])["official_pass"]
    grid = committed(set_name, item, action)
    return "reject" if (not pre_pass and grid == item["candidate"]) else action


# ---------------------------------------------------------------------------- Jev trained on U4 (part J)


def decide_prompt_zero(item) -> tuple[str, list[str]]:
    block, options = u4._decide_block(item)
    return "\n".join([u4.DECIDE_INSTRUCTION, ""] + block + ["Answer:"]), options


def repair_prompt_zero(item) -> str:
    return "\n".join([u4.REPAIR_INSTRUCTION, "", u4.case_text(item), "Repaired:"]) + "\n"


def decide_items_zero(items, gold_fn=None) -> list[ChoiceItem]:
    out = []
    for item in items:
        prompt, options = decide_prompt_zero(item)
        out.append(ChoiceItem(item["item_id"], prompt, [" " + l for l in u4.LETTERS], options, gold_fn(item) if gold_fn else "",
                              {"kind": item["kind"]}))
    return out


def repair_items_zero(items) -> list[CotItem]:
    return [CotItem(item["item_id"], repair_prompt_zero(item), "", (lambda text, n=len(item["task"]["grid"]): json.dumps(u4.parse_grid(text, n))),
                    (lambda text, n=len(item["task"]["grid"]): sum(1 for line in text.split("\n")[:-1] if line.strip()) >= n),
                    {"kind": item["kind"]}) for item in items]


def training_rows(split: str, limit: int | None = None):
    """(decide ChoiceItems, repair (prompt, target) pairs) from a U4 pool: train, or a fixed val subset."""
    items = list(u4.build_items(split))
    if limit is not None:
        items = random.Random(f"u5-val-subset:{split}").sample(items, limit)
    decide = decide_items_zero(items, gold_fn=u4.best_action)
    repair = [(repair_prompt_zero(i), u4.grid_text(i["gold"]) + "\n\n") for i in items]
    return decide, repair


def encode_generate(tokenizer, prompt: str, target: str):
    from ..training import Encoded

    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    return Encoded(prompt_ids + target_ids, len(prompt_ids), target_ids, "generate", 0, -1)


# ---------------------------------------------------------------------------- runs


def _write(part: str, set_name: str, record: str, rows: list[dict], info: dict) -> None:
    out = OUT / part / set_name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{record}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / f"{record}.json").write_text(json.dumps({"record": record, "set": set_name, **info}, indent=2), encoding="utf-8")


def repair_row(item, grid, **extra) -> dict:
    return {**u4.repair_row(item, grid, **extra), "proposer": item.get("proposer")}


def run_cpu() -> None:
    camp = u4.camp()
    train, val = list(u4.build_items("train")), list(u4.build_items("val"))
    trms = [u4.train_decision_trm(train, val, seed=s) for s in (0, 1, 2)]
    for set_name in ("u5test", "proposals"):
        items = items_for(set_name)
        proj = compute_projections(set_name)
        info = {"neural": False}
        for policy in POLICIES:
            for mode in REJECT_MODES:
                rows = [outcome(set_name, i, with_reject(set_name, i, u4.flow_policy(i, policy), mode)) for i in items]
                _write("decide", set_name, f"{policy}+{mode}", rows, info)
        _write("decide", set_name, "menu-ceiling", [outcome(set_name, i, best_action(set_name, i)) for i in items], {**info, "ceiling": True})
        for seed, (predict, trm_info) in enumerate(trms):
            rows = []
            for i in items:
                action, probs = predict(i)
                rows.append(outcome(set_name, i, action, probs=probs))
            _write("decide", set_name, "Decision-TRM" if seed == 0 else f"Decision-TRM-s{seed}", rows,
                   {"neural": True, "seed": seed, "recipe": "SPEC-U4 Decision-TRM, retrained", **trm_info})
        csp_rows = []
        for i in items:
            t0 = time.perf_counter()
            solved, meta = camp.solve_candidates(u4.task_of(i), max_candidates=1)
            csp_rows.append(repair_row(i, solved[0] if solved else None, seconds=time.perf_counter() - t0, nodes=meta["nodes"],
                                       truncated=meta["truncated"]))
        _write("repair", set_name, "CSP-resolve", csp_rows, {"neural": False, "ceiling": True})
        _write("repair", set_name, "identity", [repair_row(i, i["candidate"]) for i in items], info)
        for module in ("c_repair", "dual_repair"):
            _write("repair", set_name, module, [repair_row(i, proj[i["item_id"]][module]["grid"], status=proj[i["item_id"]][module]["status"],
                                                           seconds=proj[i["item_id"]][module]["seconds"]) for i in items], info)
    print(json.dumps({"u5test_hash": u4.items_hash("u5test"), "proposal_items": len(proposal_items())}, indent=2))


def run_j(arm_id: str, seed: int = 0) -> None:
    """J-u4 (or another arm) on the U4 test set, the U5 test set and the proposals, in the zero-shot format."""
    from .arms import seeded
    from .jrunner import load_j_bundle, run_choice, run_cot

    arm = seeded(arm_id, seed)
    bundle = load_j_bundle(arm.adapter)
    info = {"neural": True, "adapter": arm.adapter, "base_arm": arm_id, "seed": seed, "prompt": "zero-shot (training format)"}
    for set_name in SETS:
        items = items_for(set_name)
        by_id = {i["item_id"]: i for i in items}
        u4_set = set_name == "u4test"
        rows = []
        for r in run_choice(bundle, decide_items_zero(items)):
            item = by_id[r["item_id"]]
            extra = {"probs": r["probs"], "options": r["options"], "label_mass": r["label_mass"]}
            rows.append(u4.outcome(item, r["pred"], **extra) if u4_set else outcome(set_name, item, r["pred"], **extra))
        _write("decide", set_name, arm.arm_id, rows, info)
        if set_name == "u5test":
            continue  # generation is registered on the U4 test set and the proposals only
        rows = []
        budget = max(2 * _cells(i) + 16 for i in items)
        for r in run_cot(bundle, repair_items_zero(items), max_new_tokens=budget):
            rows.append(repair_row(by_id[r["item_id"]], json.loads(r["pred"]), raw_text=r["raw_text"], emitted_tokens=r["emitted_tokens"],
                                   latency_s=r["latency_s"]))
        _write("repair", set_name, arm.arm_id, rows, info)
