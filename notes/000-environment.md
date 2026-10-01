# 000 - Environment (inspected 2026-10-01)

## Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, 16303 MiB, driver 572.76, CUDA 12.8 runtime |
| iGPU | AMD Radeon 880M (hybrid graphics laptop) |
| CPU | AMD Ryzen AI 9 365, 10 cores / 20 threads |
| RAM | 31.1 GB |
| Disk | C: 664 GB free |
| OS | Windows 11 Home 10.0.26200, no WSL installed |

## Software found

- System Python 3.12.10 (`py -3.12`) with no ML packages.
- No conda, uv, ollama or `git` on PATH. GitHub Desktop's bundled git is at
  `%LOCALAPPDATA%\GitHubDesktop\app-3.6.6\resources\app\git\cmd\git.exe`; `jevq.records`
  falls back to it for the commit hash.
- Two existing venvs belonging to other projects:
  - `C:\Users\patri\.venvs\bluebeam`: torch 2.11.0+cu128, transformers 5.17.0, peft 0.21.1,
    accelerate 1.15.0, bitsandbytes 0.50.2. **transformers 5.17.0 contains the Qwen3.5
    implementation** (`models/qwen3_5`), which is what was inspected for the intervention point.
  - `C:\Users\patri\Documents\Codex\BitAgent-gym\.venv`: torch 2.9.1+cu128, transformers 4.57.6
    (too old for Qwen3.5).
- No Qwen3.5 weights anywhere on disk. Local HF caches hold only `Qwen/Qwen3-0.6B` and
  `prism-ml/Bonsai-8B-unpacked` (under the BitAgent-gym runs directory).

`requirements.txt` pins the versions verified in the bluebeam venv. This project's own venv was
created later the same day at `C:\Users\patri\.venvs\jev-qwen` (same versions; all tests pass
in it). The checkpoint is in the default HF cache, `C:\Users\patri\.cache\huggingface\hub`,
commit `1001bb4d826a52d1f399e183466143f4da7b741b`, 9.33 GB.

## Contention at inspection time

The GPU was not free. Other agent jobs were cycling on it during the whole session:

- 09:20: 15.9 / 16.3 GB used, 100% utilisation. `train.py` (BitAgent gym LoRA run) and
  `bench_gen.py` (BlueBeam), both launched from the Codex runtime Python.
- 09:45: a different Python process holding ~7-14 GB VRAM and 15 GB of private memory.
- System commit charge 61-64 GB of a 67 GB limit.

Consequence: the 4B model was not loaded in this session. Nothing of ours has touched the GPU.

## Gotchas found the hard way

1. **Never set `CUDA_VISIBLE_DEVICES=-1` on this machine.** It makes torch load and then unload
   the NVIDIA hybrid-graphics DLL (`nvdxgdmal64.dll`) while one of its threads is still alive;
   the process then dies with an access violation at a random later point (Windows Application
   log: faulting module `nvdxgdmal64.dll_unloaded`). It looked like random crashes in unrelated
   Python code. Tests keep the tiny model on CPU without hiding the GPU.
2. `torch.cuda.get_device_name()` creates a CUDA context. `jevq.hardware` takes the GPU name
   from `nvidia-smi` instead so describing the machine costs no VRAM.
3. The repo lives under OneDrive. `checkpoints/` and `results/` are synced unless excluded;
   keep only adapters there, and keep the venv and the HF cache outside OneDrive (defaults:
   `%USERPROFILE%\.venvs\jev-qwen`, `%USERPROFILE%\.cache\huggingface`).
4. **Norton Antivirus re-signs HTTPS traffic** ("Norton Web/Mail Shield Root"). That root is in
   the Windows trust store but not in certifi, so `huggingface_hub` fails with
   `CERTIFICATE_VERIFY_FAILED` while pip works. `scripts/_bootstrap.py` injects `truststore` so
   requests verify against the Windows store. Certificate verification is never disabled.
5. **System commit is often nearly exhausted** (3-5 GB free of 67 GB even with free RAM),
   because of the other jobs. `load_model` therefore streams weights straight to the device with
   a single-device map instead of building a CPU copy first.
6. The HF cache cannot use symlinks here (no Developer Mode), so files are stored as copies.
7. The reference PyTorch fallbacks for the DeltaNet kernels are used because
   `flash-linear-attention` and `causal_conv1d` are not installed. Transformers says they are
   correct but slower. Latency numbers are therefore for the fallback path; say so when quoting them.
