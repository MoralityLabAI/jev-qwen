"""The loader path used for the 4B checkpoint, exercised on a saved tiny model (CPU, offline)."""

import pytest
import torch

from jevq.looped import forward_hidden
from jevq.modeling import find_text_model, load_model, make_bundle


def test_load_model_roundtrip(tiny_model, tokenizer, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    model, info = load_model({"id": str(tmp_path), "dtype": "float32", "device": "cpu"})

    assert info["class"] == "Qwen3_5ForCausalLM" and info["dtype"] == "float32"
    assert not model.training
    assert find_text_model(model) is model.model

    ids = torch.randint(1, 100, (1, 12))
    with torch.no_grad():
        expected = tiny_model(input_ids=ids, use_cache=False).logits
        loaded = model.lm_head(forward_hidden(model.model, input_ids=ids).hidden)
    assert torch.allclose(expected, loaded, atol=1e-6)

    bundle = make_bundle(model, tokenizer, info)
    assert bundle.info["num_layers"] == 8
    assert bundle.info["layer_types"] == ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + [
        "full_attention"
    ]
    assert bundle.device.type == "cpu"


def test_load_model_rejects_missing_weights(tiny_model, tmp_path):
    from safetensors.torch import load_file, save_file

    tiny_model.save_pretrained(tmp_path)
    weights = load_file(tmp_path / "model.safetensors")
    dropped = {k: v for k, v in weights.items() if "layers.3.mlp" not in k}
    assert len(dropped) < len(weights)
    save_file(dropped, tmp_path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="missing"):
        load_model({"id": str(tmp_path), "dtype": "float32", "device": "cpu"})
