"""Environment and hardware description for run records."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_smi() -> dict | None:
    """GPU state from the driver, including memory held by other processes."""
    query = "name,memory.total,memory.used,utilization.gpu,driver_version"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, total, used, util, driver = [part.strip() for part in out.stdout.strip().splitlines()[0].split(",")]
    return {
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_pct": int(util),
        "driver_version": driver,
    }


def describe(include_torch: bool = True) -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: _version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors", "tokenizers")
        },
        "gpu": _nvidia_smi(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 2**30, 1)
        info["ram_available_gb"] = round(vm.available / 2**30, 1)
    except ImportError:
        pass
    if include_torch:
        import torch

        # The device name comes from nvidia-smi above; asking torch for it would create a CUDA
        # context (VRAM) just to describe the machine.
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
    return info
