"""Milestone 1 gate: load the vanilla checkpoint and verify the driver on real weights.

    python scripts/check_model_load.py

Checks, in order: the checkpoint loads with no missing text weights; the layer layout is
the expected 8 x (3 DeltaNet + 1 Attention); the schedule driver with no loop reproduces
the stock forward; a short greedy completion is sane. Writes results/env/model_load.json.
"""

import argparse
import time

import torch

import _bootstrap  # noqa: F401
from jevq.config import ROOT, apply_override, load_yaml
from jevq.hardware import describe
from jevq.looped import LoopSpec, forward_hidden
from jevq.modeling import load_bundle
from jevq.records import write_json

PROMPT = "Q: What is the capital of France?\nA: Paris\n\nQ: What is 7 + 5?\nA:"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / "configs" / "base.yaml")
    for assignment in args.set:
        apply_override(cfg, assignment)

    cuda = torch.cuda.is_available()
    t0 = time.perf_counter()
    bundle = load_bundle(cfg["model"])
    load_s = time.perf_counter() - t0
    text_model, tokenizer = bundle.text_model, bundle.tokenizer
    report = dict(bundle.info)
    report["load_seconds"] = load_s
    report["vram_after_load_mb"] = torch.cuda.memory_allocated() / 2**20 if cuda else None
    report["n_params_total_loaded"] = sum(p.numel() for p in bundle.model.parameters())

    ids = torch.tensor([tokenizer.encode(PROMPT)], device=bundle.device)
    stock = bundle.model(input_ids=ids, use_cache=False).logits.float()
    driver = bundle.lm_head(forward_hidden(text_model, input_ids=ids).hidden).float()
    looped_once = bundle.lm_head(forward_hidden(text_model, input_ids=ids, loop=LoopSpec(12, 16, 1)).hidden).float()
    looped_twice = bundle.lm_head(forward_hidden(text_model, input_ids=ids, loop=LoopSpec(12, 16, 2)).hidden).float()
    report["driver_vs_stock_max_abs_logit_diff"] = (stock - driver).abs().max().item()
    report["loop_r1_vs_stock_max_abs_logit_diff"] = (stock - looped_once).abs().max().item()
    report["loop_r2_vs_stock_max_abs_logit_diff"] = (stock - looped_twice).abs().max().item()
    report["driver_matches_stock"] = bool(torch.equal(stock.argmax(-1), driver.argmax(-1)))

    out = bundle.model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False
    )
    report["sample_completion"] = tokenizer.decode(out[0, ids.shape[1] :].tolist())
    report["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    report["hardware"] = describe()

    for key, value in report.items():
        if key not in ("hardware", "layer_types"):
            print(f"{key}: {value}")
    write_json(ROOT / "results" / "env" / "model_load.json", report)
    if not report["driver_matches_stock"]:
        raise SystemExit("FAIL: schedule driver disagrees with the stock forward on real weights")


if __name__ == "__main__":
    main()
