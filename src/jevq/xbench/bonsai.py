"""Bonsai-8B on the local PrismML llama-server (SPEC sections 2 and 4).

The server is started on a free loopback port with the pins used in Control-Harness (alias, no
LoRA adapters, model file sha256) and stopped afterwards. Prompts are the same rendered text the
J-arms receive, sent to the raw `/completion` endpoint at temperature 0. When the first generated
token's top-k probabilities include the option labels, they are renormalised into option
probabilities; otherwise probabilities are None.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

LLAMA_DIR = Path(r"C:\Users\patri\Documents\Codex\BitAgent-gym\llama-prism\bin")
MODEL = Path(r"C:\Users\patri\Documents\Codex\BitAgent-gym\prime-artifacts\Bonsai-8B-Q1_0.gguf")
ALIAS = "Bonsai-8B-Q1_0"
SHA256_PREFIX = "284a335a"  # pinned in Control-Harness (memory: control-harness-live-actor-matrix)
SEED = 20261001


def model_sha256() -> str:
    digest = hashlib.sha256()
    with open(MODEL, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BonsaiServer:
    def __init__(self, ctx: int = 8192, gpu_layers: int = 99):
        exe = LLAMA_DIR / "llama-server.exe"
        if not exe.exists():
            raise FileNotFoundError(f"{exe} is missing; if Norton quarantined it, the user must restore it")
        sha = model_sha256()
        if not sha.startswith(SHA256_PREFIX):
            raise RuntimeError(f"Bonsai model sha256 {sha[:12]} does not match the pin {SHA256_PREFIX}")
        self.sha256 = sha
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.process = subprocess.Popen(
            [str(exe), "-m", str(MODEL), "-ngl", str(gpu_layers), "-c", str(ctx), "-np", "1", "--no-webui", "--alias", ALIAS,
             "--reasoning", "off", "--host", "127.0.0.1", "--port", str(self.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(LLAMA_DIR),
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                models = self._get("/v1/models")
                break
            except OSError:
                if self.process.poll() is not None:
                    raise RuntimeError("llama-server exited during startup")
                time.sleep(2)
        else:
            self.stop()
            raise TimeoutError("llama-server did not start")
        names = [m.get("id") for m in models.get("data", [])]
        if names != [ALIAS]:
            self.stop()
            raise RuntimeError(f"server serves {names}, expected exactly [{ALIAS}]")
        try:
            adapters = self._get("/lora-adapters")
        except OSError:
            adapters = []
        if adapters:
            self.stop()
            raise RuntimeError("a LoRA adapter is loaded; the pin requires none")

    def _get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=30) as response:
            return json.loads(response.read())

    def complete(self, prompt: str, n_predict: int, n_probs: int = 20) -> dict:
        body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": SEED,
                           "n_probs": n_probs, "cache_prompt": False}).encode()
        request = urllib.request.Request(self.url + "/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())

    def chat(self, prompt: str, max_tokens: int) -> str:
        """One user turn through the chat template (addendum A5: the attacker role follows
        instructions; the raw endpoint made it continue the prompt instead)."""
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                           "temperature": 0.0, "top_k": 1, "seed": SEED, "cache_prompt": False}).encode()
        request = urllib.request.Request(self.url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            out = json.loads(response.read())
        return out["choices"][0]["message"].get("content") or ""

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def _first_token_probs(result: dict) -> dict[str, float]:
    probs = result.get("completion_probabilities") or []
    if not probs:
        return {}
    first = probs[0]
    entries = first.get("top_logprobs") or first.get("probs") or []
    out = {}
    for entry in entries:
        token = entry.get("token") or entry.get("tok_str") or ""
        if "logprob" in entry:
            import math

            out[token] = math.exp(entry["logprob"])
        elif "prob" in entry:
            out[token] = entry["prob"]
    return out


def parse_choice(text: str, labels: list[str]) -> int | None:
    """Index of the label the completion starts with (labels carry a leading space)."""
    stripped = text.strip()
    for index, label in enumerate(labels):
        word = label.strip()
        if stripped == word or stripped.startswith(word + "\n") or stripped.startswith(word + " ") or stripped.startswith(word + ".") or stripped.startswith(word + ")"):
            return index
    return None


def run_choice(server: BonsaiServer, items) -> list[dict]:
    rows = []
    for item in items:
        t0 = time.perf_counter()
        result = server.complete(item.prompt, n_predict=4)
        text = result.get("content", "")
        index = parse_choice(text, item.labels)
        top = _first_token_probs(result)
        label_mass = [top.get(label, top.get(label.strip(), 0.0)) for label in item.labels]
        probs = [p / sum(label_mass) for p in label_mass] if sum(label_mass) > 0 else None
        pred = item.options[index] if index is not None else ""
        rows.append({
            "item_id": item.item_id, "gold": item.gold, "pred": pred, "correct": pred == item.gold,
            "options": item.options if probs else None, "probs": probs, "parsed": index is not None,
            "passes": len(result.get("completion_probabilities") or []) or 1, "emitted_tokens": result.get("tokens_predicted"),
            "latency_s": time.perf_counter() - t0, "raw_text": text, **item.meta,
        })
    return rows


def generate(server: BonsaiServer, prompt: str, n_predict: int = 96) -> str:
    return server.complete(prompt, n_predict=n_predict, n_probs=0).get("content", "")


def generate_chat(server: BonsaiServer, prompt: str, max_tokens: int = 96) -> str:
    return server.chat(prompt, max_tokens)
