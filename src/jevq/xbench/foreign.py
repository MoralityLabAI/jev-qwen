"""Read-only access to the other Morality Lab repositories (SPEC section 9).

Imports never write bytecode into a foreign tree, and every path is checked before use.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

LOOPED_TRANSFORMERS = Path(r"C:\Users\patri\OneDrive\Documents\GitHub\LoopedTransformers")
RMP_ARTIFACTS = Path(r"C:\Users\patri\Documents\Codex\rmp_v0")
HERMES_SKILLS = Path(r"C:\Users\patri\Documents\GitHub\MoralityLabAI\Hermes-Skills")
HERMES_LITE = Path(r"C:\Users\patri\Documents\Codex\hermes-lite")
CONTROL_HARNESS = Path(r"C:\Users\patri\Documents\GitHub\MoralityLabAI\Control-Harness")


@contextlib.contextmanager
def _no_bytecode():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def import_from(root: Path, module: str, *, subdir: str | None = None):
    """Import `module` with `root` (or root/subdir) first on sys.path, without writing .pyc files."""
    base = root / subdir if subdir else root
    if not base.is_dir():
        raise FileNotFoundError(f"foreign repository path missing: {base}")
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    with _no_bytecode():
        return importlib.import_module(module)


def available(path: Path) -> bool:
    return path.exists()
