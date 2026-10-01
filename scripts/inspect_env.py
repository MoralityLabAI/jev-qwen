"""Print and save the hardware / software environment (results/env/environment.json)."""

import json

import _bootstrap  # noqa: F401
from jevq.config import ROOT
from jevq.hardware import describe
from jevq.records import write_json


def main() -> None:
    try:
        info = describe(include_torch=True)
    except ImportError:
        info = describe(include_torch=False)
        info["torch_import_error"] = True
    print(json.dumps(info, indent=2))
    write_json(ROOT / "results" / "env" / "environment.json", info)


if __name__ == "__main__":
    main()
