#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "zoology").is_dir():
            return candidate
    raise RuntimeError(f"无法从 {start} 定位仓库根目录.")


ROOT = find_repo_root(Path(__file__).resolve())
CANONICAL_RUNNER = (
    ROOT
    / "zoology/experiments/flash_vqg/scripts/20260521-longer-mqar-canonical/longer_mqar_eval_runner.py"
)


def load_canonical_runner():
    spec = importlib.util.spec_from_file_location("longer_mqar_official_core_runner", CANONICAL_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 canonical runner: {CANONICAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_phase5(module) -> None:
    artifact_dir = ROOT / "docs/artifacts/gdn-flash-fairness-20260526/phase5-longer-mqar"
    module.ARTIFACT_DIR = artifact_dir
    module.LEDGER_PATH = artifact_dir / "longer-mqar-phase5-detail.csv"
    module.SUMMARY_PATH = artifact_dir / "longer-mqar-phase5-summary.csv"
    module.STATUS_CSV_PATH = artifact_dir / "status.csv"
    module.TMP_ROOT = ROOT / "tmp/20260526-gdn-flash-fairness-phase5-longer-mqar"
    module.BATCH_ID = "20260526-gdn-flash-fairness-phase5-longer-mqar"
    module.EVAL_SCOPE = "longer_mqar_eval_only_gdn_flash_fairness_phase5_20260526"
    module.CORE_FLASH_TARGETS = set()
    module.CORE_GDN_TARGETS = {
        ("gdnxk-h4-ek8-ev4-usegate0", "123"),
        ("gdn-banked-k-h2-b4-k256-v64-sharedv-usegate0", "123"),
        ("gdn-banked-k-h2-b4-k256-v64-sharedv-usegate0", "124"),
        ("gdn-banked-k-h2-b4-k256-v64-sharedv-usegate0", "125"),
        ("gdn-banked-k-h2-b4-k256-v64-sharedv-usegate0", "126"),
    }
    module.EXPECTED_CORE_SOURCE_COUNT = len(module.CORE_GDN_TARGETS)
    for field in ("started_at_utc", "ended_at_utc", "status"):
        if field not in module.EXTRA_DETAIL_FIELDS:
            module.EXTRA_DETAIL_FIELDS.append(field)

    original_run_eval_once = module.run_eval_once

    def run_eval_once_with_timing(**kwargs):
        started = datetime.now(timezone.utc)
        result, log_path = original_run_eval_once(**kwargs)
        ended = datetime.now(timezone.utc)
        result.setdefault("started_at_utc", started.isoformat())
        result.setdefault("ended_at_utc", ended.isoformat())
        result.setdefault("wall_clock_sec", (ended - started).total_seconds())
        return result, log_path

    original_result_to_row_extra = module.result_to_row_extra

    def result_to_row_extra_with_timing(result: dict, log_path: str) -> dict:
        extra = original_result_to_row_extra(result, log_path)
        ended = result.get("ended_at_utc")
        started = result.get("started_at_utc")
        if not ended:
            ended = datetime.now(timezone.utc).isoformat()
        if not started:
            try:
                started_dt = datetime.fromisoformat(str(ended)) - timedelta(seconds=float(result.get("wall_clock_sec") or 0))
                started = started_dt.isoformat()
            except Exception:
                started = ""
        extra["started_at_utc"] = started
        extra["ended_at_utc"] = ended
        extra["status"] = result.get("status", "")
        return extra

    module.run_eval_once = run_eval_once_with_timing
    module.result_to_row_extra = result_to_row_extra_with_timing

    def is_phase5_official_source(row: dict[str, str]) -> bool:
        if row.get("train_batch_size") != module.OFFICIAL_CORE_BATCH_PROFILE["source_train_batch_size"]:
            return False
        if row.get("eval_batch_size") != module.OFFICIAL_CORE_BATCH_PROFILE["source_eval_batch_size"]:
            return False
        if row.get("gradient_accumulation_steps") != module.OFFICIAL_CORE_BATCH_PROFILE["source_gradient_accumulation_steps"]:
            return False
        if row.get("effective_train_batch_size") != module.OFFICIAL_CORE_BATCH_PROFILE["source_effective_train_batch_size"]:
            return False
        if row.get("batch_accum_profile") != module.OFFICIAL_CORE_BATCH_PROFILE["source_batch_accum_profile"]:
            return False
        if row.get("dtype_policy") != "float32":
            return False
        if row.get("outer_model_dtype") not in {"", "float32"}:
            return False
        return row.get("official_scope") in {
            "b64_ga4_fp32_official",
            "gdn_flash_fairness_phase4_banked_k_multiseed",
        }

    module.is_b64_ga4_fp32_official = is_phase5_official_source


def main() -> int:
    module = load_canonical_runner()
    configure_phase5(module)
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
