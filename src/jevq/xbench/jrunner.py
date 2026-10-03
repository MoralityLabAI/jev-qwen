"""Scoring for the Jev-style arms (J-*) on any xbench suite.

Suites hand over `ChoiceItem`s (one forward pass, softmax over the item's single-token labels)
or `CotItem`s (greedy decoding, answer parsed from the text). The model is a `ModelBundle`,
optionally with an adapter attached and a loop for the J-V2b arms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from ..config import load_yaml, resolve_path
from ..instrument import StepRecorder
from ..looped import LoopSpec, forward_hidden
from ..modeling import ModelBundle, attach_adapter, load_bundle


@dataclass
class ChoiceItem:
    item_id: str
    prompt: str
    labels: list[str]  # single-token label strings with their leading space, aligned with options
    options: list[str]  # semantic values, e.g. "K", "REFUSE", "commit"
    gold: str
    meta: dict = field(default_factory=dict)


@dataclass
class CotItem:
    item_id: str
    prompt: str
    gold: str
    parse: Callable[[str], str]  # completion text -> semantic answer ("" if none)
    done: Callable[[str], bool]  # completion text -> stop decoding
    meta: dict = field(default_factory=dict)


def label_ids(tokenizer, labels: list[str]) -> list[int]:
    ids = []
    for label in labels:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1:
            raise RuntimeError(f"label {label!r} is not a single token: {tokens}")
        ids.append(tokens[0])
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"labels {labels} map to duplicate tokens")
    return ids


def resolve_adapter(adapter: str) -> Path:
    """Arm adapter paths are relative to paths.checkpoints (outside OneDrive), like variant files."""
    path = Path(adapter).expanduser()
    if path.is_absolute():
        return path
    base = load_yaml(resolve_path("configs/base.yaml"))
    return resolve_path(base["paths"]["checkpoints"]) / path


def load_j_bundle(adapter: str | None = None, model_cfg: dict | None = None) -> ModelBundle:
    if adapter:
        path = resolve_adapter(adapter)
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"adapter not found: {path}")
    model_cfg = model_cfg or load_yaml(resolve_path("configs/base.yaml"))["model"]
    bundle = load_bundle(model_cfg)
    if adapter:
        bundle = attach_adapter(bundle, path)
    return bundle


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _last_logits(bundle: ModelBundle, ids: torch.Tensor, loop: LoopSpec | None, recorder, chunk: int | None):
    if loop is not None:
        recorder and recorder.begin_forward(gen_step=0)
        result = forward_hidden(bundle.text_model, input_ids=ids, loop=loop, recorder=recorder)
        return bundle.lm_head(result.hidden[:, -1])[0]
    if chunk is None or ids.shape[1] <= chunk:
        return bundle.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]
    # Chunked prefill through the cache: same function, bounded attention memory (SPEC section 4).
    from transformers import DynamicCache

    cache = DynamicCache(config=bundle.text_model.config)
    logits = None
    for start in range(0, ids.shape[1], chunk):
        out = bundle.model(
            input_ids=ids[:, start : start + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        cache, logits = out.past_key_values, out.logits[0, -1]
    return logits


@torch.no_grad()
def run_choice(
    bundle: ModelBundle,
    items: list[ChoiceItem],
    *,
    loop: LoopSpec | None = None,
    record_iterations: bool = False,
    chunk: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """One row per item in the unified schema; with `record_iterations`, also the tail-lens option
    probabilities after every block iteration (exact early-exit outputs, see instrument.py)."""
    rows = []
    device = bundle.device
    lens = (lambda h: bundle.lm_head(bundle.text_model.norm(h))) if record_iterations else None
    for index, item in enumerate(items):
        ids_list = label_ids(bundle.tokenizer, item.labels)
        ids = torch.tensor([bundle.tokenizer.encode(item.prompt)], device=device)
        recorder = None
        if record_iterations and loop is not None:
            recorder = StepRecorder(lens, lens_mode="tail", watch_ids=ids_list)
        _sync(device)
        t0 = time.perf_counter()
        logits = _last_logits(bundle, ids, loop, recorder, chunk).float()
        _sync(device)
        latency = time.perf_counter() - t0
        option_logits = logits[torch.tensor(ids_list, device=logits.device)]
        probs = torch.softmax(option_logits, dim=-1).tolist()
        pred_index = max(range(len(probs)), key=probs.__getitem__)
        row = {
            "item_id": item.item_id,
            "gold": item.gold,
            "pred": item.options[pred_index],
            "correct": item.options[pred_index] == item.gold,
            "options": item.options,
            "probs": probs,
            "label_mass": torch.softmax(logits, dim=-1)[torch.tensor(ids_list, device=logits.device)].sum().item(),
            "passes": 1,
            "block_iterations": loop.n_iters if loop else None,
            "emitted_tokens": 0,
            "prompt_tokens": ids.shape[1],
            "latency_s": latency,
            **item.meta,
        }
        if recorder is not None:
            row["iteration_probs"] = [r["lens_watch_probs"] for r in recorder.rows if r["kind"] == "span_iter"]
            row["iteration_preds"] = [
                item.options[max(range(len(p)), key=p.__getitem__)] for p in row["iteration_probs"]
            ]
        rows.append(row)
        if progress:
            progress(index + 1, len(items))
    return rows


@torch.no_grad()
def run_cot(bundle: ModelBundle, items: list[CotItem], *, max_new_tokens: int, progress=None) -> list[dict]:
    """Greedy decoding with the stock (cached) driver; one row per item."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    tokenizer = bundle.tokenizer
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    rows = []
    for index, item in enumerate(items):
        ids = torch.tensor([tokenizer.encode(item.prompt)], device=bundle.device)
        n_prompt = ids.shape[1]

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                done = item.done(tokenizer.decode(input_ids[0, n_prompt:].tolist()))
                return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)

        _sync(bundle.device)
        t0 = time.perf_counter()
        out = bundle.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([Stop()]),
            pad_token_id=pad if pad is not None else (eos if eos is not None else 0),
        )
        _sync(bundle.device)
        new_ids = out[0, n_prompt:].tolist()
        text = tokenizer.decode(new_ids)
        pred = item.parse(text)
        rows.append(
            {
                "item_id": item.item_id,
                "gold": item.gold,
                "pred": pred,
                "correct": pred == item.gold,
                "options": None,
                "probs": None,
                "passes": len(new_ids),
                "emitted_tokens": len(new_ids),
                "hit_token_budget": len(new_ids) >= max_new_tokens,
                "prompt_tokens": n_prompt,
                "latency_s": time.perf_counter() - t0,
                "raw_text": text,
                **item.meta,
            }
        )
        if progress:
            progress(index + 1, len(items))
    return rows
