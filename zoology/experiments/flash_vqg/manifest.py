import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from zoology.config import TrainConfig


MANIFEST_ENV_VAR = "FLASH_VQG_MANIFEST_PATH"
MANIFEST_SCHEMA_VERSION = 4
GENERATED_ROOT = Path(__file__).resolve().parent / "generated"
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


def generated_launch_dir(launch_id: str) -> Path:
    return GENERATED_ROOT / str(launch_id)


def manifest_path_for_launch(launch_id: str) -> Path:
    return generated_launch_dir(launch_id) / "manifest.json"


def load_manifest(launch_id: str) -> dict[str, Any]:
    path = manifest_path_for_launch(launch_id)
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_entry(manifest: dict[str, Any], run_id: str) -> dict[str, Any]:
    for run in manifest.get("runs", []):
        if run.get("run_id") == run_id:
            return run
    raise ValueError(f"launch_id={manifest.get('launch_id')} 中未找到 run_id={run_id}.")


def resolve_best_checkpoint_from_manifest(manifest: dict[str, Any], run_id: str) -> Path:
    run_entry = _resolve_run_entry(manifest, run_id)
    local_info = run_entry.get("local") or {}
    checkpoint_path_raw = local_info.get("best_checkpoint")
    if not checkpoint_path_raw:
        raise ValueError(
            "manifest 中缺少 `local.best_checkpoint`. 该 launch 可能是旧 manifest, "
            "请使用扩展后的新 run 重新生成 manifest."
        )
    checkpoint_path = Path(checkpoint_path_raw)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"manifest 记录的 checkpoint 不存在: {checkpoint_path}")
    return checkpoint_path


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


def _find_flash_vqg_kwargs(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if node.get("name") == "zoology.mixers.flash_vqg.FlashVQGMixer":
            kwargs = node.get("kwargs")
            return kwargs if isinstance(kwargs, dict) else {}
        for value in node.values():
            found = _find_flash_vqg_kwargs(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_flash_vqg_kwargs(item)
            if found is not None:
                return found
    return None


def config_summary_from_config(config: TrainConfig) -> dict[str, Any]:
    config_dict = config.model_dump() if hasattr(config, "model_dump") else config.dict()
    flash_kwargs = _find_flash_vqg_kwargs(config_dict.get("model") or {}) or {}
    return {
        "experiment_part": flash_kwargs.get("experiment_part"),
        "experiment_mode": flash_kwargs.get("experiment_mode"),
        "fox_remote_formula": flash_kwargs.get("fox_remote_formula"),
        "fox_remote_read_topk": flash_kwargs.get("fox_remote_read_topk"),
        "fox_clr_selector_mode": flash_kwargs.get("fox_clr_selector_mode"),
        "fox_clr_merge_mode": flash_kwargs.get("fox_clr_merge_mode"),
        "fox_clr_gate_mode": flash_kwargs.get("fox_clr_gate_mode"),
        "fox_clr_lambda_remote": flash_kwargs.get("fox_clr_lambda_remote"),
        "fox_clr_gate_init_bias": flash_kwargs.get("fox_clr_gate_init_bias"),
        "fox_clr_residual_update_mode": flash_kwargs.get("fox_clr_residual_update_mode"),
        "fox_clr_residual_forget_mode": flash_kwargs.get("fox_clr_residual_forget_mode"),
        "fox_clr_state_write_topk": flash_kwargs.get("fox_clr_state_write_topk"),
        "fox_clr_delta_target_mode": flash_kwargs.get("fox_clr_delta_target_mode"),
        "fox_gd_residual_rank": flash_kwargs.get("fox_gd_residual_rank"),
        "fox_gd_residual_write_topk": flash_kwargs.get("fox_gd_residual_write_topk"),
        "fox_gd_residual_builder": flash_kwargs.get("fox_gd_residual_builder"),
        "fox_gd_residual_pack_mode": flash_kwargs.get("fox_gd_residual_pack_mode"),
        "fox_gd_residual_chunk_size": flash_kwargs.get("fox_gd_residual_chunk_size"),
        "fox_gd_residual_lambda_init": flash_kwargs.get("fox_gd_residual_lambda_init"),
        "fox_dense_teacher_mode": flash_kwargs.get("fox_dense_teacher_mode"),
        "fox_dense_teacher_loss_mode": flash_kwargs.get("fox_dense_teacher_loss_mode"),
        "fox_dense_teacher_layer_idx": flash_kwargs.get("fox_dense_teacher_layer_idx"),
        "fox_dense_teacher_lambda": flash_kwargs.get("fox_dense_teacher_lambda"),
        "fox_dense_teacher_tau_teacher": flash_kwargs.get("fox_dense_teacher_tau_teacher"),
        "fox_dense_teacher_row_weight_mode": flash_kwargs.get("fox_dense_teacher_row_weight_mode"),
        "fox_dense_teacher_warmup_steps": flash_kwargs.get("fox_dense_teacher_warmup_steps"),
        "init_checkpoint_path": config_dict.get("init_checkpoint_path"),
        "init_checkpoint_source_launch_id": config_dict.get("init_checkpoint_source_launch_id"),
        "init_checkpoint_source_run_id": config_dict.get("init_checkpoint_source_run_id"),
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
                "config_summary": None,
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
        target["config_summary"] = config_summary_from_config(config)

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
