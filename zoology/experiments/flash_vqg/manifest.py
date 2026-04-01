import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from zoology.config import TrainConfig


MANIFEST_ENV_VAR = "FLASH_VQG_MANIFEST_PATH"
MANIFEST_SCHEMA_VERSION = 3
CHECKPOINT_LOCAL_FIELDS = (
    "checkpoint_run_dir",
    "best_checkpoint",
    "last_checkpoint",
    "train_config_json",
)
EVAL_SOURCE_FIELDS = (
    "checkpoint_launch_id",
    "checkpoint_run_id",
    "best_checkpoint",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


@contextmanager
def _locked_manifest(manifest_path: Path) -> Iterator[dict[str, Any]]:
    import fcntl

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read().strip()
        data = json.loads(raw) if raw else {}
        yield data
        f.seek(0)
        f.truncate()
        json.dump(_to_jsonable(data), f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def manifest_path_from_env() -> Path | None:
    raw = os.environ.get(MANIFEST_ENV_VAR)
    if not raw:
        return None
    return Path(raw).resolve()


def checkpoint_local_paths_from_config(config: TrainConfig) -> dict[str, str | None]:
    if config.launch_id is None or not config.checkpoint.enabled:
        return {field: None for field in CHECKPOINT_LOCAL_FIELDS}

    run_dir = (Path(config.checkpoint.root_dir) / config.launch_id / config.run_id).resolve()
    return {
        "checkpoint_run_dir": str(run_dir),
        "best_checkpoint": str((run_dir / "best.pt").resolve()),
        "last_checkpoint": str((run_dir / "last.pt").resolve()),
        "train_config_json": str((run_dir / "train_config.json").resolve()),
    }


def initialize_manifest(
    *,
    manifest_path: Path,
    launch_id: str,
    sweep_id: str | None,
    logger_backend: str,
    project: str | None,
    entity: str | None,
    run_ids: list[str],
    launch_config_file: Path,
    eval_sources: dict[str, dict[str, Any]] | None = None,
    eval_task: str | None = None,
):
    eval_sources = eval_sources or {}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "launch_id": launch_id,
        "sweep_id": sweep_id,
        "eval_task": eval_task,
        "logger_backend": logger_backend,
        "project": project,
        "entity": entity,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "launch_config_file": str(launch_config_file.resolve()),
        "runs": [
            {
                "run_id": run_id,
                "eval_task": eval_task,
                "status": "planned",
                "error": None,
                "updated_at_utc": _utc_now(),
                "swanlab": {
                    "project": project,
                    "entity": entity,
                    "experiment_id": None,
                    "run_url": None,
                },
                "local": {
                    "run_dir": None,
                    "backup_file": None,
                    "config_file": None,
                    "metadata_file": None,
                    "log_file": None,
                    "checkpoint_run_dir": None,
                    "best_checkpoint": None,
                    "last_checkpoint": None,
                    "train_config_json": None,
                },
                "eval_source": {
                    field: eval_sources.get(run_id, {}).get(field)
                    for field in EVAL_SOURCE_FIELDS
                },
            }
            for run_id in run_ids
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_manifest_for_run(
    *,
    config: TrainConfig,
    logger_summary: dict[str, Any] | None,
    status: str,
    error: str | None = None,
    manifest_path: Path | None = None,
    eval_source: dict[str, Any] | None = None,
):
    resolved_path = manifest_path or manifest_path_from_env()
    if resolved_path is None or config.launch_id is None:
        return

    with _locked_manifest(resolved_path) as manifest:
        runs = manifest.setdefault("runs", [])
        target = None
        for run in runs:
            if run.get("run_id") == config.run_id:
                target = run
                break
        if target is None:
            target = {
                "run_id": config.run_id,
                "swanlab": {},
                "local": {},
                "eval_source": {},
            }
            runs.append(target)

        target["status"] = status
        target["error"] = error
        target["updated_at_utc"] = _utc_now()

        swanlab_payload = target.setdefault("swanlab", {})
        local_payload = target.setdefault("local", {})
        eval_source_payload = target.setdefault("eval_source", {})
        local_payload.update(checkpoint_local_paths_from_config(config))
        if eval_source:
            eval_source_payload.update(
                {field: eval_source.get(field) for field in EVAL_SOURCE_FIELDS}
            )
        if logger_summary:
            swanlab_payload.update(
                {
                    "project": logger_summary.get("project"),
                    "entity": logger_summary.get("entity"),
                    "experiment_id": logger_summary.get("run_id"),
                    "run_url": logger_summary.get("run_url"),
                }
            )
            local_payload.update(
                {
                    "run_dir": logger_summary.get("run_dir"),
                    "backup_file": logger_summary.get("backup_file"),
                    "config_file": logger_summary.get("config_file"),
                    "metadata_file": logger_summary.get("metadata_file"),
                }
            )

        manifest["updated_at_utc"] = _utc_now()
