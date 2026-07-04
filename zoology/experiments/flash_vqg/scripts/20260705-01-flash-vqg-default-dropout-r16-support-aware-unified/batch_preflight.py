#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = (
    SCRIPT_DIR.parent
    / "20260703-04-flash-vqg-safe-limiter-readk2-readk4-screen/safe_limiter_batch_preflight.py"
)
WRAPPER = SCRIPT_DIR / "r16_support_aware_unified.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SRC = _load(SOURCE, "read_support_write_confidence_batch_preflight_base")
WRAP = _load(WRAPPER, "r16_support_aware_unified_for_batch_preflight")


def main() -> int:
    WRAP._patch_identity()
    WRAP._patch_support()
    SRC.SRC = WRAP
    return SRC.main()


if __name__ == "__main__":
    raise SystemExit(main())
