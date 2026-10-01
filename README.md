# jev-qwen

Experiments on whether Qwen3.5-4B-Base can be adapted into a Jev-style decision model: short
typed answers from few forward passes, with optional recurrent or latent hidden computation in
place of emitted reasoning tokens. The research design is in [EXPERIMENT.md](EXPERIMENT.md).

## Layout

| Path | Contents |
|------|----------|
| `EXPERIMENT.md` | research question, Jev facts vs. hypotheses, variants, plans, falsification criteria |
| `configs/` | `base.yaml`, `variants/*.yaml` (one per architecture variant), `sweeps/*.yaml` |
| `evals/suites/` | benchmark suite definitions (`smoke`, `dev`) |
| `src/jevq/` | `looped.py` (layer-schedule driver), `instrument.py`, `harness.py`, `flops.py`, `modeling.py`, `records.py`, `tasks/` |
| `scripts/` | `run_eval.py`, `sweep_loop.py`, `check_model_load.py`, `inspect_env.py`, `setup_env.ps1` |
| `tests/` | driver equivalence, task ground truth, harness end-to-end (tiny random model, CPU) |
| `data/` | reserved for generated training splits and teacher traces |
| `checkpoints/` | adapters only |
| `results/` | one directory per run: `record.json`, `examples.jsonl`, `steps.jsonl` |
| `notes/` | environment, sources, implementation inspection, compromises, milestone logs |

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

Creates `%USERPROFILE%\.venvs\jev-qwen` (outside OneDrive) and installs `requirements.txt`.
Downloads about 3 GB of wheels. The model itself (about 10 GB) is fetched from Hugging Face on
first load into the default HF cache.

## Use

```powershell
$py = "$env:USERPROFILE\.venvs\jev-qwen\Scripts\python.exe"
& $py -m pytest tests -q                       # no downloads, no GPU
& $py scripts\download_model.py                # fetch the checkpoint (9.3 GB), no GPU
& $py scripts\wait_for_gpu.py --min-free-mib 11000   # block until the shared GPU is free
& $py scripts\check_model_load.py              # load weights, verify the driver on them
& $py scripts\run_eval.py --variant configs\variants\v0_baseline.yaml --suite evals\suites\smoke.yaml
& $py scripts\run_eval.py --variant configs\variants\v2_loop_mid_r2.yaml --set variant.loop.n_iters=3
& $py scripts\sweep_loop.py --sweep configs\sweeps\loop_span.yaml
& $py scripts\bench_latency.py --bench configs\sweeps\latency.yaml   # paired timing on a shared GPU
& $py scripts\run_queue.py --queue configs\queues\m2_probe.yaml      # resumable job queue, waits for VRAM
```

Do not set `CUDA_VISIBLE_DEVICES=-1` on this machine (see `notes/000-environment.md`).

## Run records

Every run writes `results/<run_id>/record.json` with the git commit, model id and resolved
revision, variant, dataset fingerprint, seed, the fully resolved config, hardware, and scores.
`examples.jsonl` has one row per example and readout; `steps.jsonl` has the per-iteration
instrumentation for schedule-driver runs.
