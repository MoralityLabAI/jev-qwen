"""Model loading. The vanilla checkpoint is loaded unmodified; variants wrap it at call time."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ModelBundle:
    model: torch.nn.Module  # the HF causal-LM object, weights untouched
    tokenizer: object
    text_model: torch.nn.Module  # the Qwen3_5TextModel inside `model`
    lm_head: torch.nn.Module
    device: torch.device
    info: dict = field(default_factory=dict)


def find_text_model(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the decoder stack regardless of which wrapper class was loaded."""
    for module in model.modules():
        if all(hasattr(module, name) for name in ("layers", "embed_tokens", "rotary_emb", "norm")):
            return module
    raise RuntimeError("no Qwen3.5 text model (layers/embed_tokens/rotary_emb/norm) found inside the loaded model")


def make_bundle(model: torch.nn.Module, tokenizer, info: dict | None = None) -> ModelBundle:
    text_model = find_text_model(model)
    device = next(model.parameters()).device
    info = dict(info or {})
    info["n_params_text_stack"] = sum(p.numel() for p in text_model.parameters())
    info["num_layers"] = text_model.config.num_hidden_layers
    info["layer_types"] = list(text_model.config.layer_types)
    return ModelBundle(
        model=model, tokenizer=tokenizer, text_model=text_model, lm_head=model.lm_head, device=device, info=info
    )


def load_model(model_cfg: dict) -> tuple[torch.nn.Module, dict]:
    """Load the text-only causal LM (vision tower and MTP head are dropped by the HF class)."""
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    kwargs: dict = {"dtype": dtype, "revision": model_cfg.get("revision")}
    quantized = bool(model_cfg.get("load_in_4bit"))
    if quantized:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}

    model, loading = AutoModelForCausalLM.from_pretrained(model_cfg["id"], output_loading_info=True, **kwargs)
    # A missing key means some weights were randomly initialised: the baseline would be invalid.
    missing = [key for key in loading.get("missing_keys", []) if key != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} keys for the text model, e.g. {missing[:5]}")
    if not quantized:
        model.to(model_cfg.get("device", "cuda"))
    model.eval()

    info = {
        "id": model_cfg["id"],
        "revision_requested": model_cfg.get("revision"),
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "class": type(model).__name__,
        "dtype": str(dtype).replace("torch.", ""),
        "load_in_4bit": quantized,
        "unexpected_keys_dropped": len(loading.get("unexpected_keys", [])),
    }
    return model, info


def load_bundle(model_cfg: dict) -> ModelBundle:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_cfg.get("revision"))
    model, info = load_model(model_cfg)
    return make_bundle(model, tokenizer, info)
