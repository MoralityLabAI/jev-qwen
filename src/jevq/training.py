"""Parameter-efficient training (V1, V1c, and later V2b through the loop driver).

Data: a `train` and a `val` split drawn from the same seeded task generators as the
benchmarks, difficulties 1-3 only, with every question of the evaluation suites excluded.
Each question is rendered in exactly one readout format (round-robin), with the same few-shot
prefix the harness uses at evaluation, so the adapted model is evaluated in the format it
was trained in.

Targets are encoded separately from the prompt, as at inference, where the prompt tokens are
fixed and the model produces the continuation:

    choice        " B"                              (one label token)
    generate      " 42\n\n"
    generate_cot  " <worked solution>\nA: 42\n\n"

The trailing blank line matches the few-shot blocks token for token (checked against the
real tokenizer in notes/006).

Losses, averaged per sequence so every example weighs the same whatever its length:

    ce     token cross-entropy over the full vocabulary (standard SFT; V1)
    brier  on choice items only, the Brier score of the distribution over the valid option
           labels (V1c); other formats keep cross-entropy, so V1 and V1c differ in one factor
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import deep_merge, load_yaml, resolve, resolve_path
from .harness import label_token_ids, make_variant
from .hardware import describe
from .looped import forward_hidden
from .modeling import ModelBundle, load_bundle, make_bundle
from .records import git_state, sha256_of, write_json
from .tasks import GENERATOR_VERSION, build_task, generate_split
from .tasks.base import LABELS, Example, render_choice, render_cot, render_generate

FORMATS = ("choice", "generate", "generate_cot")


# ---------------------------------------------------------------- config


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Training config with optional `inherit:` (another training config) applied first."""
    from .config import apply_override

    raw = load_yaml(resolve_path(path))
    if "inherit" in raw:
        parent = load_train_config(raw.pop("inherit"))
        raw = deep_merge(parent, raw)
    for assignment in overrides or []:
        apply_override(raw, assignment)
    return raw


# ---------------------------------------------------------------- data


@dataclass
class Item:
    example: Example
    fmt: str
    shots: list[Example]


def eval_questions(seed: int, suite_paths: list[str], n_shots: int) -> set[str]:
    """Every question (shots and tests) of the evaluation suites: never trained on."""
    seen: set[str] = set()
    for path in suite_paths:
        suite = load_yaml(resolve_path(path))
        for task in suite["tasks"]:
            shots, tests = build_task(task, seed, suite["difficulties"], suite["n_per_difficulty"], n_shots)
            seen |= {e.question for e in shots + tests}
    return seen


def build_items(data_cfg: dict, seed: int, n_shots: int) -> tuple[list[Item], list[Item], dict]:
    """(train items, val items, data description). Deterministic in `seed`."""
    suite = load_yaml(resolve_path(data_cfg["prompt_suite"]))
    excluded = eval_questions(seed, data_cfg["exclude_suites"], n_shots)
    n_excluded = len(excluded)
    formats = data_cfg["formats"]
    train: list[Item] = []
    val: list[Item] = []
    tasks = data_cfg.get("tasks") or suite["tasks"]
    # One rotation per split across all (task, difficulty) cells, so the format totals balance.
    rotation = {"train": 0, "val": 0}
    for task in tasks:
        shots, _ = build_task(task, seed, suite["difficulties"], suite["n_per_difficulty"], n_shots)
        for difficulty in data_cfg["difficulties"]:
            for split, count, out in (
                ("train", data_cfg["n_per_difficulty"], train),
                ("val", data_cfg["n_val_per_difficulty"], val),
            ):
                for example in generate_split(task, seed, split, difficulty, count, excluded):
                    out.append(Item(example, formats[rotation[split] % len(formats)], shots))
                    rotation[split] += 1
    random.Random(f"{seed}|train-order").shuffle(train)
    description = {
        "prompt_suite": data_cfg["prompt_suite"],
        "exclude_suites": data_cfg["exclude_suites"],
        "n_eval_questions_excluded": n_excluded,
        "tasks": tasks,
        "difficulties": data_cfg["difficulties"],
        "n_train": len(train),
        "n_val": len(val),
        "formats": {fmt: sum(item.fmt == fmt for item in train) for fmt in formats},
        "generator_version": GENERATOR_VERSION,
        "train_sha256": sha256_of([[i.example.id, i.example.question, i.fmt] for i in train]),
        "val_sha256": sha256_of([[i.example.id, i.example.question, i.fmt] for i in val]),
    }
    return train, val, description


@dataclass
class Encoded:
    input_ids: list[int]
    n_prompt: int
    targets: list[int]
    fmt: str
    n_options: int
    answer_index: int


def encode(tokenizer, item: Item, label_ids: list[int]) -> Encoded:
    example, fmt = item.example, item.fmt
    if fmt == "choice":
        prompt, target_ids = render_choice(item.shots, example), [label_ids[example.answer_index]]
    else:
        # Targets end with a blank line because every few-shot answer does: Qwen's tokenizer
        # makes "\n\n" one token, and a lone "\n" there is a token the model rarely emits.
        if fmt == "generate":
            prompt, target = render_generate(item.shots, example), f" {example.answer}\n\n"
        elif fmt == "generate_cot":
            prompt = render_cot(item.shots, example)
            target = f" {example.meta['rationale']}\nA: {example.answer}\n\n"
        else:
            raise ValueError(f"unknown format {fmt!r}")
        target_ids = tokenizer.encode(target, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt)
    return Encoded(
        input_ids=prompt_ids + target_ids,
        n_prompt=len(prompt_ids),
        targets=target_ids,
        fmt=fmt,
        n_options=len(example.options),
        answer_index=example.answer_index,
    )


# ---------------------------------------------------------------- loss


def brier(option_logits: torch.Tensor, answer_index: int) -> torch.Tensor:
    """Brier score of softmax(option_logits) against the one-hot answer; 0 is perfect, 2 the worst."""
    probs = torch.softmax(option_logits.float(), dim=-1)
    target = torch.zeros_like(probs)
    target[answer_index] = 1.0
    return ((probs - target) ** 2).sum()


def sequence_loss(bundle: ModelBundle, loop, enc: Encoded, choice_loss: str, label_ids: list[int]) -> tuple:
    """(loss, choice_correct or None) for one encoded sequence."""
    ids = torch.tensor([enc.input_ids], device=bundle.device)
    hidden = forward_hidden(bundle.text_model, input_ids=ids, loop=loop).hidden
    # Position t predicts token t+1: the targets are predicted from n_prompt-1 onwards.
    positions = torch.arange(enc.n_prompt - 1, len(enc.input_ids) - 1, device=bundle.device)
    logits = bundle.lm_head(hidden[0, positions]).float()
    targets = torch.tensor(enc.targets, device=bundle.device)
    correct = None
    if enc.fmt == "choice":
        option_logits = logits[0, torch.tensor(label_ids[: enc.n_options], device=bundle.device)]
        correct = bool(option_logits.argmax() == enc.answer_index)
        if choice_loss == "brier":
            return brier(option_logits, enc.answer_index), correct
        if choice_loss != "ce":
            raise ValueError(f"unknown choice loss {choice_loss!r}")
    return F.cross_entropy(logits, targets), correct


# ---------------------------------------------------------------- model


def attach_lora(bundle: ModelBundle, lora_cfg: dict, seed: int) -> ModelBundle:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)  # LoRA A is randomly initialised
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        layers_to_transform=lora_cfg.get("layers_to_transform"),
        bias="none",
    )
    model = get_peft_model(bundle.model, config)
    info = dict(bundle.info)
    info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return make_bundle(model, bundle.tokenizer, info)


def lr_at(step: int, total: int, optim_cfg: dict) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio x lr."""
    peak, warmup = optim_cfg["lr"], optim_cfg["warmup_steps"]
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    floor = optim_cfg["min_lr_ratio"]
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


# ---------------------------------------------------------------- evaluation during training


@torch.no_grad()
def validate(bundle: ModelBundle, loop, encoded: list[Encoded], label_ids: list[int]) -> dict:
    was_training = bundle.model.training
    bundle.model.eval()
    losses: dict[str, list[float]] = {fmt: [] for fmt in FORMATS}
    correct: list[bool] = []
    for enc in encoded:
        loss, ok = sequence_loss(bundle, loop, enc, "ce", label_ids)
        losses[enc.fmt].append(loss.item())
        if ok is not None:
            correct.append(ok)
    if was_training:
        bundle.model.train()
    out = {f"val_ce_{fmt}": sum(v) / len(v) for fmt, v in losses.items() if v}
    out["val_choice_accuracy"] = sum(correct) / len(correct) if correct else None
    return out


# ---------------------------------------------------------------- training loop


def train(cfg: dict, bundle: ModelBundle | None = None, out_dir: Path | None = None, stop_after: int | None = None) -> dict:
    """Train one adapter; returns the training record. Resumes from `<out>/resume.pt` if present.

    `cfg["run_overrides"]` (optional) are `key=value` overrides applied to the resolved run config,
    e.g. `variant.loop.n_iters=3` for V2b. `stop_after` ends the process-equivalent early, as if
    it had been killed after that many optimizer steps (tests use it to exercise resume).
    """
    started = time.perf_counter()
    seed = cfg["seed"]
    run_cfg = resolve(cfg["variant"], cfg["data"]["prompt_suite"], cfg.get("run_overrides"))
    out_dir = Path(out_dir) if out_dir else resolve_path(run_cfg["paths"]["checkpoints"]) / f"{cfg['name']}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_of({k: v for k, v in cfg.items() if k not in ("max_steps", "paths")})

    if bundle is None:
        bundle = load_bundle(run_cfg["model"])
    num_layers = bundle.text_model.config.num_hidden_layers
    variant = make_variant(run_cfg["variant"], num_layers)
    label_ids = label_token_ids(bundle.tokenizer)

    train_items, val_items, data_info = build_items(cfg["data"], seed, run_cfg["readout"]["n_shots"])
    train_enc = [encode(bundle.tokenizer, item, label_ids) for item in train_items]
    val_enc = [encode(bundle.tokenizer, item, label_ids) for item in val_items]
    data_info["max_train_tokens"] = max(len(e.input_ids) for e in train_enc)

    bundle = attach_lora(bundle, cfg["lora"], seed)
    if cfg["gradient_checkpointing"]:
        bundle.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(params, lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])

    accum = optim_cfg["grad_accum"]
    total_steps = (len(train_enc) * optim_cfg["epochs"]) // accum
    if cfg.get("max_steps"):
        total_steps = min(total_steps, cfg["max_steps"])
    order = [i for epoch in range(optim_cfg["epochs"]) for i in _epoch_order(len(train_enc), seed, epoch)]

    step, log_rows = 0, []
    resume_file = out_dir / RESUME_FILE
    if resume_file.exists():
        state = torch.load(resume_file, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise SystemExit(f"{resume_file} was written by a different config; move it away to start over")
        _load_adapter_weights(bundle.model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        step, log_rows = state["step"], state["log"]
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng"])
        print(f"[train] resumed at optimizer step {step}/{total_steps}", flush=True)
    else:
        torch.manual_seed(seed)

    cuda = bundle.device.type == "cuda"
    bundle.model.train()
    choice_loss = cfg["loss"]["choice"]
    while step < total_steps:
        t0 = time.perf_counter()
        lr = lr_at(step, total_steps, optim_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = [train_enc[i] for i in order[step * accum : (step + 1) * accum]]
        losses: dict[str, list[float]] = {}
        correct: list[bool] = []
        for enc in batch:
            loss, ok = sequence_loss(bundle, variant.loop, enc, choice_loss, label_ids)
            (loss / accum).backward()
            losses.setdefault(enc.fmt, []).append(loss.item())
            if ok is not None:
                correct.append(ok)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, optim_cfg["max_grad_norm"]).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        row = {
            "step": step,
            "lr": lr,
            "grad_norm": grad_norm,
            "seconds": time.perf_counter() - t0,
            "loss": sum(sum(v) for v in losses.values()) / len(batch),
            **{f"loss_{fmt}": sum(v) / len(v) for fmt, v in losses.items()},
            "train_choice_accuracy": sum(correct) / len(correct) if correct else None,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        }
        if step % cfg["eval_every"] == 0 or step == total_steps:
            row.update(validate(bundle, variant.loop, val_enc, label_ids))
        log_rows.append(row)
        print(
            f"[train] step {step}/{total_steps} loss={row['loss']:.4f} lr={lr:.2e} "
            f"{row['seconds']:.1f}s" + (f" val_acc={row['val_choice_accuracy']:.3f}" if "val_choice_accuracy" in row else ""),
            flush=True,
        )
        if step % cfg["save_every"] == 0 and step < total_steps:
            _save_state(bundle.model, optimizer, step, log_rows, fingerprint, resume_file)
        if stop_after is not None and step >= stop_after:
            return {"stopped_at": step}

    bundle.model.save_pretrained(out_dir / "adapter")
    record = {
        "name": cfg["name"],
        "seed": seed,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_state(),
        "config": cfg,
        "config_sha256": fingerprint,
        "variant": run_cfg["variant"],
        "model": bundle.info,
        "data": data_info,
        "steps": total_steps,
        "final": log_rows[-1] if log_rows else None,
        "hardware": describe(),
        "wall_time_s": time.perf_counter() - started,
        "adapter": "adapter",
    }
    write_json(out_dir / "train_record.json", record)
    with open(out_dir / "train_log.jsonl", "w", encoding="utf-8") as fh:
        for row in log_rows:
            fh.write(json.dumps(row) + "\n")
    if resume_file.exists():  # finished: a later run with this name must start fresh, not resume
        _retry(resume_file.unlink)
    print(f"[train] adapter: {out_dir / 'adapter'}", flush=True)
    return record


def _epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    # Epoch 0 keeps the (already shuffled) build order; later epochs reshuffle.
    order = list(range(n))
    if epoch:
        random.Random(f"{seed}|epoch|{epoch}").shuffle(order)
    return order


RESUME_FILE = "resume.pt"


def _retry(fn, attempts: int = 10, wait_s: float = 3.0):
    """Run a filesystem call, retrying while something else holds the file.

    Sync clients and antivirus scanners briefly lock new files on Windows; the first M3 run
    died on exactly that (notes/006).
    """
    for attempt in range(attempts):
        try:
            return fn()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait_s)


def _adapter_state(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def _load_adapter_weights(model, saved: dict) -> None:
    params = dict(model.named_parameters())
    missing = [k for k in saved if k not in params]
    if missing:
        raise RuntimeError(f"resume state has unknown parameters, e.g. {missing[:3]}")
    with torch.no_grad():
        for name, value in saved.items():
            params[name].copy_(value)


def _save_state(model, optimizer, step: int, log_rows: list, fingerprint: str, resume_file: Path) -> None:
    """Everything needed to resume, in one file that replaces the previous one atomically.

    One file means there is never a moment with weights from one step and optimizer state from
    another, and no directory ever has to be deleted (which a sync client can refuse).
    """
    tmp = resume_file.with_name(resume_file.name + ".tmp")
    state = {
        "step": step,
        "adapter": _adapter_state(model),
        "optimizer": optimizer.state_dict(),
        "log": log_rows,
        "fingerprint": fingerprint,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    }
    _retry(lambda: torch.save(state, tmp))
    _retry(lambda: os.replace(tmp, resume_file))
