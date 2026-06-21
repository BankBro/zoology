from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
FLASH_VQG_ROOT = Path("/home/lyj/mnt/project/Flash-VQG")
GD_BUILDER_PATH = (
    SCRIPT_DIR.parent / "20260425-gd-residual-v1-mqar" / "config_builder.py"
)
METRICS_WHITE_LIST_FILE = (
    SCRIPT_DIR.parent / "20260425-gd-residual-v1-mqar" / "metrics.yaml"
)
ARTIFACT_DIR = REPO_ROOT / "docs" / "artifacts" / "20260603-gd-init-transplant"
LARGE_ARTIFACT_DIR = REPO_ROOT / "checkpoints" / "20260603-gd-init-transplant"
SNAPSHOT_DIR = LARGE_ARTIFACT_DIR / "init_snapshots"
INIT_CHECKPOINT_DIR = LARGE_ARTIFACT_DIR / "init_checkpoints"
GENERATED_ROOT = REPO_ROOT / "zoology" / "experiments" / "flash_vqg" / "generated"
RESULTS_ROOT = REPO_ROOT / "zoology" / "analysis" / "flash_vqg" / "results"
PYTHON_BIN = os.environ.get("PYTHON_BIN", "/home/lyj/miniconda3/envs/flash-vqg/bin/python")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoology.checkpoints import serialize_train_config
from zoology.config import TrainConfig
from zoology.experiments.flash_vqg.manifest import MANIFEST_ENV_VAR, initialize_manifest
from zoology.model import ContinuousInputModel, LanguageModel
from zoology.utils import set_determinism


@dataclass(frozen=True)
class TargetSpec:
    target: str
    seed: int
    num_codebook_vectors: int
    gd_rank: int
    label: str


TARGET_SPECS: dict[str, TargetSpec] = {
    "cb64-r16-s124": TargetSpec("cb64-r16-s124", 124, 64, 16, "cb64-r16-boundary"),
    "cb64-r16-s125": TargetSpec("cb64-r16-s125", 125, 64, 16, "cb64-r16-good"),
    "cb256-r4-s123": TargetSpec("cb256-r4-s123", 123, 256, 4, "cb256-r4-historical-good"),
    "cb256-r4-s124": TargetSpec("cb256-r4-s124", 124, 256, 4, "cb256-r4-bad"),
    "cb256-r4-s125": TargetSpec("cb256-r4-s125", 125, 256, 4, "cb256-r4-boundary"),
}


FLASH_KEY_MARKERS = (
    ".sequence_mixer.attn.",
    ".sequence_mixer.mixer.attn.",
)


INIT_PATH_NOTES = [
    {
        "component": "zoology FlashVQGMixer wrapper",
        "path": str(REPO_ROOT / "zoology" / "mixers" / "flash_vqg.py"),
        "details": "构造 FlashVQGConfig, 传递 codebook, addr, beta/lambda, gd_residual_v1 参数.",
    },
    {
        "component": "FlashVQGAttention projections",
        "path": str(FLASH_VQG_ROOT / "src" / "flash_vqg" / "nn" / "attn.py"),
        "details": "qkvg_proj, res_proj, quantizer, fox_gd_residual_addr_proj, beta/lambda projection 初始化.",
    },
    {
        "component": "codebook initialization",
        "path": str(FLASH_VQG_ROOT / "src" / "flash_vqg" / "nn" / "vq.py"),
        "details": "LearnableVQ/RoutingVQ 创建 codebook, 调用 make_codebook_initializer.",
    },
    {
        "component": "codebook RNG strategy",
        "path": str(FLASH_VQG_ROOT / "src" / "flash_vqg" / "nn" / "vq_init.py"),
        "details": "global/local_burn/local_noburn 随机初始化和 scale/bootstrap 策略.",
    },
    {
        "component": "model custom init",
        "path": str(FLASH_VQG_ROOT / "src" / "flash_vqg" / "nn" / "model.py"),
        "details": "gd_residual_beta/lambda custom init, 普通 Linear/Embedding 初始化策略.",
    },
    {
        "component": "gd_residual config builder",
        "path": str(GD_BUILDER_PATH),
        "details": "构造 gd_residual_v1 MQAR TrainConfig, seed/data_seed/codebook/addr/beta/lambda 参数.",
    },
    {
        "component": "training entry",
        "path": str(REPO_ROOT / "zoology" / "train.py"),
        "details": "set_determinism(seed, deterministic=TORCH_DETERMINISTIC==1), LanguageModel 构造, init_checkpoint_path strict 加载.",
    },
]


def ensure_artifact_dirs() -> None:
    for path in (ARTIFACT_DIR, SNAPSHOT_DIR, INIT_CHECKPOINT_DIR, GENERATED_ROOT, RESULTS_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_config(config: TrainConfig) -> TrainConfig:
    return config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)


def build_builder_args(
    target: str,
    *,
    max_epochs: int = 4,
    train_batch_size: int = 64,
    eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    validations_per_epoch: int = 2,
    data_seed: int = 123,
    run_id: str | None = None,
    launch_id_prefix: str = "flash-vqg-20260603-gd-init-transplant",
    experiment_mode: str | None = None,
) -> argparse.Namespace:
    if target not in TARGET_SPECS:
        raise ValueError(f"未知 target: {target}. 可选: {sorted(TARGET_SPECS)}")
    spec = TARGET_SPECS[target]
    target_tag = target.replace("-", "_")
    return argparse.Namespace(
        backend="torch",
        logger_backend="swanlab",
        dmodels="128",
        learning_rates="1e-3",
        seed_values=str(spec.seed),
        data_seed=int(data_seed),
        num_codebook_vectors=str(spec.num_codebook_vectors),
        metrics_white_list=None,
        metrics_white_list_file=str(METRICS_WHITE_LIST_FILE),
        launch_id_prefix=launch_id_prefix,
        train_batch_order="global_shuffle",
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values="2",
        fox_remote_formula="gd_residual_v1",
        fox_gd_residual_rank=int(spec.gd_rank),
        fox_gd_residual_write_topk=4,
        fox_gd_residual_builder="grouped_chunk_torch_ref",
        fox_gd_residual_pack_mode="semivec_ref",
        fox_gd_residual_chunk_size=64,
        fox_gd_residual_mu_min_count=0.1,
        fox_gd_residual_addr_eps=1e-6,
        fox_gd_residual_den_eps=1e-6,
        fox_gd_residual_rho_eps=1e-12,
        fox_gd_residual_addr_init_rng_mode="global",
        fox_gd_residual_addr_init_seed=None,
        fox_gd_residual_beta_init=0.5,
        fox_gd_residual_beta_cap=None,
        fox_gd_residual_beta_cap_final=None,
        fox_gd_residual_beta_cap_release_start_train_steps=0,
        fox_gd_residual_beta_cap_release_end_train_steps=0,
        fox_gd_residual_beta_cap_eval_policy="final",
        fox_gd_residual_beta_control_mode="hard_cap",
        fox_gd_residual_beta_sigmoid_temp=1.0,
        fox_gd_residual_beta_low=None,
        fox_gd_residual_beta_high=None,
        fox_gd_residual_beta_low_final=None,
        fox_gd_residual_beta_high_final=None,
        fox_gd_residual_beta_band_release_start_train_steps=0,
        fox_gd_residual_beta_band_release_end_train_steps=0,
        fox_gd_residual_beta_band_eval_policy="final",
        fox_gd_residual_beta_band_schedule="smoothstep",
        fox_gd_residual_lambda_init=0.05,
        fox_gd_residual_lambda_floor=0.0,
        fox_gd_residual_write_strength_mode="renorm_topk",
        fox_gd_residual_write_strength_cap=None,
        fox_gd_residual_write_strength_cap_mode="hard",
        fox_gd_residual_write_strength_cap_until_train_steps=0,
        fox_gd_residual_write_strength_cap_final=None,
        fox_gd_residual_write_strength_cap_release_start_train_steps=0,
        fox_gd_residual_write_strength_cap_release_end_train_steps=0,
        fox_gd_residual_write_strength_cap_eval_policy="final",
        fox_gd_residual_write_budget=None,
        fox_gd_residual_write_budget_final=None,
        fox_gd_residual_write_budget_release_start_train_steps=0,
        fox_gd_residual_write_budget_release_end_train_steps=0,
        fox_gd_residual_write_budget_eval_policy="final",
        fox_gd_residual_write_budget_schedule="smoothstep",
        fox_gd_residual_write_total_cap=None,
        fox_gd_residual_write_total_cap_final=None,
        fox_gd_residual_write_total_cap_release_start_train_steps=0,
        fox_gd_residual_write_total_cap_release_end_train_steps=0,
        fox_gd_residual_write_total_cap_eval_policy="final",
        fox_gd_residual_write_total_cap_schedule="smoothstep",
        fox_gd_residual_write_q_alpha=1.0,
        fox_gd_residual_m_norm_cap=None,
        fox_gd_residual_update_norm_cap=None,
        fox_gd_residual_norm_with_gain=False,
        fox_gd_residual_use_separate_addr_codebook=False,
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=0.25,
        codebook_init_rng_mode="global",
        codebook_init_seed=None,
        vq_topk=4,
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        train_batch_size=int(train_batch_size),
        eval_batch_size=int(eval_batch_size),
        validations_per_epoch=int(validations_per_epoch),
        disable_early_stopping="true",
        cache_dir="./data/flash_vqg",
        project="flash_vqg_gd_init_transplant",
        entity="scu-mclab",
        max_epochs=int(max_epochs),
        run_id=run_id,
        experiment_mode=experiment_mode or f"gd_init_transplant_{target_tag}",
    )


def build_config(
    target: str,
    *,
    max_epochs: int = 4,
    train_batch_size: int = 64,
    eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    validations_per_epoch: int = 2,
    data_seed: int = 123,
    run_id: str | None = None,
    launch_id_prefix: str = "flash-vqg-20260603-gd-init-transplant",
    experiment_mode: str | None = None,
    smoke_data: bool = False,
) -> TrainConfig:
    builder = load_module_from_path(GD_BUILDER_PATH, "gd_residual_v1_mqar_config_builder")
    args = build_builder_args(
        target,
        max_epochs=max_epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        validations_per_epoch=validations_per_epoch,
        data_seed=data_seed,
        run_id=run_id,
        launch_id_prefix=launch_id_prefix,
        experiment_mode=experiment_mode,
    )
    if smoke_data:
        configs = builder.build_gd_residual_v1_smoke_configs(args)
    else:
        configs = builder.build_gd_residual_v1_train_configs(args)
    if len(configs) != 1:
        raise RuntimeError(f"{target} 应生成一个 config, 实际为 {len(configs)}")
    return configs[0]


def build_model(config: TrainConfig):
    if config.input_type == "continuous":
        return ContinuousInputModel(config.model)
    return LanguageModel(config.model)


def initialized_model_and_state(config: TrainConfig):
    if os.environ.get("TORCH_DETERMINISTIC") == "1":
        raise RuntimeError("本实验禁止启用 TORCH_DETERMINISTIC=1.")
    set_determinism(int(config.seed), deterministic=False)
    model = build_model(config)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return model, state


def is_flash_key(key: str) -> bool:
    return any(marker in key for marker in FLASH_KEY_MARKERS)


def filter_state_dict(state: dict[str, torch.Tensor], scope: str) -> dict[str, torch.Tensor]:
    if scope == "full_model":
        return {key: value.detach().cpu().clone() for key, value in state.items()}
    if scope == "flash_only":
        return {key: value.detach().cpu().clone() for key, value in state.items() if is_flash_key(key)}
    if scope == "non_flash_only":
        return {key: value.detach().cpu().clone() for key, value in state.items() if not is_flash_key(key)}
    raise ValueError(f"未知 snapshot scope: {scope}")


def sha256_tensor_dict(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def snapshot_path(target: str, scope: str) -> Path:
    return SNAPSHOT_DIR / f"{target}-{scope}.pt"


def load_state_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 不是 dict payload.")
    return payload


def extract_payload_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("state_dict")
    if state is None:
        state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError("payload 缺少 state_dict/model_state_dict.")
    return state


def save_snapshot(
    *,
    target: str,
    scope: str,
    config: TrainConfig,
    state: dict[str, torch.Tensor],
) -> Path:
    ensure_artifact_dirs()
    subset = filter_state_dict(state, scope)
    path = snapshot_path(target, scope)
    payload = {
        "kind": "flash_vqg_init_snapshot",
        "created_at_utc": utc_now_iso(),
        "target": target,
        "target_spec": TARGET_SPECS[target].__dict__,
        "scope": scope,
        "state_dict": subset,
        "model_state_dict": subset if scope == "full_model" else None,
        "state_keys": sorted(subset),
        "num_tensors": len(subset),
        "num_elements": int(sum(t.numel() for t in subset.values())),
        "sha256": sha256_tensor_dict(subset),
        "torch_deterministic_env": os.environ.get("TORCH_DETERMINISTIC", "0"),
        "config": serialize_train_config(config),
    }
    torch.save(payload, path)
    return path


def transplant_checkpoint_path(name: str) -> Path:
    return INIT_CHECKPOINT_DIR / name / "best.pt"


def make_transplant_checkpoint(
    *,
    name: str,
    recipient_target: str,
    donor_snapshot_path: str | Path,
    overlay_scope: str,
    max_epochs: int = 4,
    run_id: str | None = None,
) -> tuple[Path, TrainConfig, dict[str, Any]]:
    ensure_artifact_dirs()
    donor_payload = load_state_payload(donor_snapshot_path)
    donor_state = extract_payload_state(donor_payload)
    recipient_run_id = run_id or name
    config = build_config(
        recipient_target,
        max_epochs=max_epochs,
        run_id=recipient_run_id,
        experiment_mode=f"gd_init_transplant_{name}",
    )
    _, recipient_state = initialized_model_and_state(config)
    if overlay_scope not in {"full_model", "flash_only", "non_flash_only"}:
        raise ValueError(f"未知 overlay_scope: {overlay_scope}")
    overlay_state = filter_state_dict(donor_state, overlay_scope)
    if not overlay_state:
        raise RuntimeError(f"{donor_snapshot_path} 在 scope={overlay_scope} 下没有可覆盖参数.")

    shape_errors: list[str] = []
    missing_keys: list[str] = []
    for key, value in overlay_state.items():
        if key not in recipient_state:
            missing_keys.append(key)
            continue
        if tuple(recipient_state[key].shape) != tuple(value.shape):
            shape_errors.append(
                f"{key}: donor={tuple(value.shape)} recipient={tuple(recipient_state[key].shape)}"
            )
            continue
        recipient_state[key] = value.detach().cpu().clone()

    if missing_keys or shape_errors:
        details = {
            "missing_keys": missing_keys[:20],
            "num_missing_keys": len(missing_keys),
            "shape_errors": shape_errors[:20],
            "num_shape_errors": len(shape_errors),
        }
        raise RuntimeError(f"transplant key/shape mismatch: {json.dumps(details, ensure_ascii=False)}")

    checkpoint_path = transplant_checkpoint_path(name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "flash_vqg_init_transplant_checkpoint",
        "created_at_utc": utc_now_iso(),
        "name": name,
        "recipient_target": recipient_target,
        "recipient_spec": TARGET_SPECS[recipient_target].__dict__,
        "recipient_seed": int(config.seed),
        "donor_snapshot_path": str(Path(donor_snapshot_path).resolve()),
        "donor_target": donor_payload.get("target"),
        "donor_scope": donor_payload.get("scope"),
        "overlay_scope": overlay_scope,
        "overlay_num_tensors": len(overlay_state),
        "overlay_num_elements": int(sum(t.numel() for t in overlay_state.values())),
        "overlay_sha256": sha256_tensor_dict(overlay_state),
        "full_model_sha256": sha256_tensor_dict(recipient_state),
        "torch_deterministic_env": os.environ.get("TORCH_DETERMINISTIC", "0"),
    }
    payload = {
        "model_state_dict": recipient_state,
        "epoch": -1,
        "metrics": {"init/transplant_overlay_tensors": float(len(overlay_state))},
        "run_id": config.run_id,
        "launch_id": None,
        "sweep_id": config.sweep_id,
        "model_name": config.model.name,
        "init_transplant_metadata": metadata,
    }
    torch.save(payload, checkpoint_path)
    metadata_path = checkpoint_path.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return checkpoint_path, config, metadata


def apply_init_checkpoint(config: TrainConfig, init_checkpoint_path: Path, *, source_name: str) -> TrainConfig:
    updated = copy_config(config)
    updated.init_checkpoint_path = str(init_checkpoint_path.resolve())
    updated.init_checkpoint_source_launch_id = "init-transplant-snapshot"
    updated.init_checkpoint_source_run_id = source_name
    updated.init_checkpoint_strict = True
    return updated


def render_generated_config(builder_args: list[dict[str, Any]]) -> str:
    payload = json.dumps(builder_args, ensure_ascii=False, indent=2)
    return (
        "from __future__ import annotations\n\n"
        "import importlib.util\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        f"_SCRIPT_DIR = Path({str(SCRIPT_DIR)!r})\n"
        "sys.path.insert(0, str(_SCRIPT_DIR))\n"
        "_SCRIPT_PATH = _SCRIPT_DIR / 'launch_configs_builder.py'\n"
        "_SPEC = importlib.util.spec_from_file_location('gd_init_transplant_launch_builder', _SCRIPT_PATH)\n"
        "_MODULE = importlib.util.module_from_spec(_SPEC)\n"
        "assert _SPEC is not None and _SPEC.loader is not None\n"
        "_SPEC.loader.exec_module(_MODULE)\n"
        f"_RUN_SPECS = json.loads({payload!r})\n"
        "configs = _MODULE.build_configs_from_specs(_RUN_SPECS)\n"
    )


def write_generated_launch(
    *,
    launch_id: str,
    run_specs: list[dict[str, Any]],
    logger_backend: str = "swanlab",
    project: str = "flash_vqg_gd_init_transplant",
    entity: str = "scu-mclab",
) -> Path:
    ensure_artifact_dirs()
    generated_dir = GENERATED_ROOT / launch_id
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_dir / "launch_configs.py"
    generated_path.write_text(render_generated_config(run_specs), encoding="utf-8")
    manifest_path = generated_dir / "manifest.json"
    initialize_manifest(
        manifest_path=manifest_path,
        launch_id=launch_id,
        sweep_id=launch_id.rsplit("-", maxsplit=1)[0],
        logger_backend=logger_backend,
        project=project,
        entity=entity,
        run_ids=[str(item["run_id"]) for item in run_specs],
        launch_config_file=generated_path,
    )
    return generated_path


def run_launch(
    *,
    generated_path: Path,
    launch_id: str,
    gpus: str,
    parallelize: bool = False,
) -> subprocess.CompletedProcess:
    manifest_path = generated_path.parent / "manifest.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env[MANIFEST_ENV_VAR] = str(manifest_path.resolve())
    if env.get("TORCH_DETERMINISTIC") == "1":
        raise RuntimeError("本实验禁止启用 TORCH_DETERMINISTIC=1.")
    if parallelize:
        cmd = [
            PYTHON_BIN,
            str(SCRIPT_DIR / "local_parallel_launch.py"),
            "--launch-config",
            str(generated_path),
            "--launch-id",
            launch_id,
            "--manifest-path",
            str(manifest_path.resolve()),
            "--gpus",
            str(gpus),
        ]
        return subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    cmd = [
        PYTHON_BIN,
        "-m",
        "zoology.launch",
        str(generated_path),
        "--launch-id",
        launch_id,
        "--gpus",
        str(gpus),
    ]
    return subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    import csv

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_init_path_notes() -> Path:
    ensure_artifact_dirs()
    path = ARTIFACT_DIR / "init-path-notes.json"
    write_json(path, {"created_at_utc": utc_now_iso(), "items": INIT_PATH_NOTES})
    return path


def parse_targets(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("targets 不能为空.")
    for value in values:
        if value not in TARGET_SPECS:
            raise ValueError(f"未知 target={value}. 可选: {sorted(TARGET_SPECS)}")
    return values


def parse_scopes(raw: str) -> list[str]:
    valid = {"full_model", "flash_only", "non_flash_only"}
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("scopes 不能为空.")
    for value in values:
        if value not in valid:
            raise ValueError(f"未知 scope={value}. 可选: {sorted(valid)}")
    return values


def print_path(path: Path) -> None:
    print(str(path.resolve()), flush=True)


def fail_if_outside_allowed(path: Path) -> None:
    resolved = path.resolve()
    allowed = [REPO_ROOT.resolve(), FLASH_VQG_ROOT.resolve()]
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise RuntimeError(f"拒绝操作允许目录之外的路径: {resolved}")


def validate_scope_boundaries() -> None:
    fail_if_outside_allowed(REPO_ROOT)
    fail_if_outside_allowed(FLASH_VQG_ROOT)
    fail_if_outside_allowed(SCRIPT_DIR)
    fail_if_outside_allowed(ARTIFACT_DIR)
    fail_if_outside_allowed(LARGE_ARTIFACT_DIR)


__all__ = [
    "ARTIFACT_DIR",
    "GENERATED_ROOT",
    "INIT_CHECKPOINT_DIR",
    "INIT_PATH_NOTES",
    "LARGE_ARTIFACT_DIR",
    "RESULTS_ROOT",
    "SCRIPT_DIR",
    "SNAPSHOT_DIR",
    "TARGET_SPECS",
    "apply_init_checkpoint",
    "build_config",
    "ensure_artifact_dirs",
    "extract_payload_state",
    "filter_state_dict",
    "initialized_model_and_state",
    "is_flash_key",
    "make_transplant_checkpoint",
    "parse_scopes",
    "parse_targets",
    "print_path",
    "run_launch",
    "save_snapshot",
    "sha256_tensor_dict",
    "snapshot_path",
    "utc_now_iso",
    "validate_scope_boundaries",
    "write_csv",
    "write_generated_launch",
    "write_init_path_notes",
    "write_json",
]
