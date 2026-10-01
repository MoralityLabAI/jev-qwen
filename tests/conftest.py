"""Test fixtures: a tiny random-weight Qwen3.5 text model and a character tokenizer.

Nothing here downloads anything or touches the GPU. The tiny model has the real
architecture (2 x [3 Gated DeltaNet + 1 Gated Attention]) at toy width, so the
driver is exercised against the same HF code path the 4B checkpoint uses.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevq.modeling import make_bundle  # noqa: E402
from jevq.tasks.base import LABELS  # noqa: E402


class FakeTokenizer:
    """One token per character, except ' A'..' H' which are single tokens like in a BPE vocab."""

    pad_token_id = 0
    eos_token_id = None

    def __init__(self):
        chars = ["\n"] + [chr(c) for c in range(32, 127)]
        self.char_to_id = {ch: i + 1 for i, ch in enumerate(chars)}
        self.label_to_id = {f" {label}": len(chars) + 1 + i for i, label in enumerate(LABELS)}
        self.id_to_text = {i: ch for ch, i in self.char_to_id.items()}
        self.id_to_text.update({i: text for text, i in self.label_to_id.items()})
        self.id_to_text[0] = ""
        self.vocab_size = len(chars) + 1 + len(LABELS)

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            pair = text[i : i + 2]
            # ' A' is a label token only when the letter stands alone, e.g. after 'Answer:'.
            if pair in self.label_to_id and (i + 2 == len(text) or not text[i + 2].isalnum()):
                ids.append(self.label_to_id[pair])
                i += 2
            else:
                ids.append(self.char_to_id[text[i]])
                i += 1
        return ids

    def decode(self, ids, **kwargs):
        return "".join(self.id_to_text.get(int(i), "?") for i in ids)


@pytest.fixture(scope="session")
def tokenizer():
    return FakeTokenizer()


@pytest.fixture(scope="session")
def tiny_model(tokenizer):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=tokenizer.vocab_size + 8,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=4096,
        pad_token_id=0,
    )
    model = Qwen3_5ForCausalLM(config).eval()
    # The default init zeroes or shrinks enough that random logits are near-ties; widen them.
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:
                param.normal_(0.0, 0.08)
    return model


@pytest.fixture(scope="session")
def bundle(tiny_model, tokenizer):
    return make_bundle(tiny_model, tokenizer, {"id": "tiny-random-qwen3_5"})
