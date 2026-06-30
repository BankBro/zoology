#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("ZOOLOGY_REPO_ROOT", "/home/lyj/mnt/project/zoology")).resolve()
PRIOR_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/s124_fixed_r4_4ep_confirm.py"
)
PRIOR_INIT_CHECKPOINT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260630-03-flash-vqg-s124-fixed-r4-4ep-confirm/outputs/canonical-init/cb64r16-s124-init.pt"
)
PRIOR_INIT_META = PRIOR_INIT_CHECKPOINT.with_suffix(".meta.json")
EXPERIMENT_ID = "20260630-04-flash-vqg-default-dropout-fixed-r4-1ep-screen"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
EXPECTED_INIT_STATE_SHA256 = "2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0"
TARGETS = ("fixed-r4",)
DEFAULT_MAX_EPOCHS = 1
DEFAULT_EMBED_DROPOUT = 0.1
DEFAULT_RESID_DROPOUT = 0.0
DEFAULT_DROP_PATH = 0.0


def _load_prior_module():
    spec = importlib.util.spec_from_file_location("s124_fixed_r4_4ep_prior", PRIOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prior script: {PRIOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRIOR = _load_prior_module()
BASE = PRIOR.BASE
ORIGINAL_BASE_ARGS = PRIOR.ORIGINAL_BASE_ARGS
ORIGINAL_BUILD_CONFIG = BASE.build_config

VARIANTS = {
    "fixed-r4": BASE.VARIANTS["fixed-r4"],
}


def _read_expected_init_hash() -> str:
    env_value = os.environ.get("EXPECTED_INIT_STATE_SHA256")
    if env_value:
        return env_value
    if PRIOR_INIT_META.exists():
        payload = json.loads(PRIOR_INIT_META.read_text(encoding="utf-8"))
        return str(payload.get("model_state_sha256", EXPECTED_INIT_STATE_SHA256))
    return EXPECTED_INIT_STATE_SHA256


def _patch_base() -> None:
    BASE.SCRIPT_DIR = SCRIPT_DIR
    BASE.EXPERIMENT_ID = EXPERIMENT_ID
    BASE.ARTIFACT_DIR = ARTIFACT_DIR
    BASE.TARGETS = TARGETS
    BASE.VARIANTS = VARIANTS
    BASE.DEFAULT_INIT_CHECKPOINT = PRIOR_INIT_CHECKPOINT
    BASE.EXPECTED_INIT_STATE_SHA256 = _read_expected_init_hash()
    BASE.DEFAULT_MAX_EPOCHS = DEFAULT_MAX_EPOCHS
    BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS = BASE.EXPECTED_STEPS_PER_EPOCH * DEFAULT_MAX_EPOCHS

    def _base_args(
        *,
        target: str,
        machine_name: str,
        variant: str,
        logger_backend: str,
        trace_output_dir: Path,
        max_epochs: int,
        max_train_steps: int | None,
        max_validation_batches: int | None,
    ):
        args = ORIGINAL_BASE_ARGS(
            target=target,
            machine_name=machine_name,
            variant=variant,
            logger_backend=logger_backend,
            trace_output_dir=trace_output_dir,
            max_epochs=max_epochs,
            max_train_steps=max_train_steps,
            max_validation_batches=max_validation_batches,
        )
        args.seed_values = "124"
        args.project = "flash_vqg_default_dropout_fixed_r4_1ep_screen"
        args.experiment_mode = f"{EXPERIMENT_ID}_{variant}_s124_d123_b64ga4_{machine_name}"
        args.run_id = f"{EXPERIMENT_ID}-{variant}-s124-d123-b64ga4-{machine_name}"
        args.launch_id_prefix = f"fvqg-{EXPERIMENT_ID}-{machine_name}-{target}"
        return args

    def build_config(
        *,
        target: str,
        machine_name: str,
        variant: str,
        logger_backend: str,
        trace_output_dir: Path,
        max_epochs: int,
        max_train_steps: int | None,
        max_validation_batches: int | None,
    ):
        config = ORIGINAL_BUILD_CONFIG(
            target=target,
            machine_name=machine_name,
            variant=variant,
            logger_backend=logger_backend,
            trace_output_dir=trace_output_dir,
            max_epochs=max_epochs,
            max_train_steps=max_train_steps,
            max_validation_batches=max_validation_batches,
        )
        config.model.embed_dropout = DEFAULT_EMBED_DROPOUT
        config.model.resid_dropout = DEFAULT_RESID_DROPOUT
        config.model.drop_path = DEFAULT_DROP_PATH
        return config

    BASE._base_args = _base_args
    BASE.build_config = build_config


_patch_base()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=BASE._json_default)
        + "\n",
        encoding="utf-8",
    )


def run_preflight(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    config = BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/preflight-traces" / args.machine_name / args.target,
        max_epochs=args.max_epochs,
        max_train_steps=None,
        max_validation_batches=None,
    )
    train_loader, _ = BASE.prepare_data(config.data)
    train_batches = len(train_loader)
    accum = int(config.gradient_accumulation_steps)
    optim_steps_per_epoch = (train_batches + accum - 1) // accum
    max_epochs = int(config.max_epochs)
    total_optimizer_steps = optim_steps_per_epoch * max_epochs
    flash_settings = BASE._flash_vqg_settings(config)
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "machine_name": args.machine_name,
        "target": args.target,
        "variant": args.variant,
        "variant_spec": BASE._variant_config(args.variant),
        "env": BASE.env_snapshot(args.machine_name),
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "train_batches": train_batches,
        "gradient_accumulation_steps": accum,
        "max_epochs": max_epochs,
        "num_optimizer_steps_per_epoch": optim_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "expected_optimizer_steps_per_epoch": BASE.EXPECTED_STEPS_PER_EPOCH,
        "expected_total_optimizer_steps": BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS,
        "cache_dir": config.data.cache_dir,
        "embed_dropout": config.model.embed_dropout,
        "resid_dropout": config.model.resid_dropout,
        "drop_path": config.model.drop_path,
        **flash_settings,
    }
    result["passed"] = (
        train_batches == 2815
        and accum == 4
        and max_epochs == DEFAULT_MAX_EPOCHS
        and optim_steps_per_epoch == BASE.EXPECTED_STEPS_PER_EPOCH
        and total_optimizer_steps == BASE.EXPECTED_TOTAL_OPTIMIZER_STEPS
        and abs(float(config.model.embed_dropout) - DEFAULT_EMBED_DROPOUT) < 1e-12
        and abs(float(config.model.resid_dropout) - DEFAULT_RESID_DROPOUT) < 1e-12
        and abs(float(config.model.drop_path) - DEFAULT_DROP_PATH) < 1e-12
        and flash_settings["num_codebook_vectors"] == BASE.EXPECTED_NUM_CODEBOOK_VECTORS
        and BASE._variant_settings_match(flash_settings, args.variant)
    )
    if args.output_json:
        save_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=BASE._json_default))
    return 0 if result["passed"] else 1


def run_config_summary(args: argparse.Namespace) -> int:
    config = BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/config-traces" / args.machine_name / args.target,
        max_epochs=DEFAULT_MAX_EPOCHS,
        max_train_steps=None,
        max_validation_batches=None,
    )
    payload = BASE.serialize_train_config(config)
    flash_settings = BASE._flash_vqg_settings(config)
    if args.output_json:
        save_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "launch_id": config.launch_id,
                "embed_dropout": config.model.embed_dropout,
                "resid_dropout": config.model.resid_dropout,
                "drop_path": config.model.drop_path,
                "max_epochs": config.max_epochs,
                **flash_settings,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=BASE._json_default,
        )
    )
    return 0


def run_train(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("torch.cuda is unavailable inside this container.")
    config = BASE.build_config(
        target=args.target,
        machine_name=args.machine_name,
        variant=args.variant,
        logger_backend=args.logger_backend,
        trace_output_dir=args.trace_output_dir,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_validation_batches=args.max_validation_batches,
    )
    config.init_checkpoint_path = str(args.init_checkpoint)
    config.init_checkpoint_strict = True
    config.init_checkpoint_source_launch_id = "canonical-init-2080ti"
    config.init_checkpoint_source_run_id = "initlock-cb64r16-default-s124-r1-d123-b64ga4-2080ti"
    if args.output_config_json:
        save_json(args.output_config_json, BASE.serialize_train_config(config))
    result = BASE.train(config)
    if args.output_result_json:
        save_json(
            args.output_result_json,
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "machine_name": args.machine_name,
                "target": args.target,
                "variant": args.variant,
                "variant_spec": BASE._variant_config(args.variant),
                "init_checkpoint": str(args.init_checkpoint),
                "train_result": result,
                "env": BASE.env_snapshot(args.machine_name),
            },
        )
    return 0


def run_collect(args: argparse.Namespace) -> int:
    code = BASE.run_collect(args)
    readme = (
        f"# {EXPERIMENT_ID}\n\n"
        "本 artifact 收尾 `seed=124` default-dropout fixed-r4 1 epoch screen. "
        "本轮是 diagnostic / exploratory screen, 不写 official MQAR ledger.\n\n"
        "共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, "
        "train-time `read_topk=4`, canonical MQAR cache, seed124 canonical init, "
        "`max_epochs=1`, `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`.\n\n"
        "## 文件\n\n"
        "- `run-summary.csv`: per-run final/best metrics.\n"
        "- `variant-summary.csv`: fixed-r4 的 2080ti/3090 成对结果.\n"
        "- `cross-machine-comparison.csv`: fixed-r4 的 1024x256 cross-machine gap.\n"
        "- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.\n"
        "- `queue-summary.csv`: queue 状态.\n"
        "- `invalid-runs.csv`: failed/interrupted/pending run.\n"
        "- `source-manifest.csv`: mirrored raw evidence 路径和 sha256.\n"
        "- `metadata.json`: 收尾元数据.\n"
        "\n"
        "注: `best_*` 是日志观测到的 best validation metric, 不是 saved-best checkpoint 复评. "
        "本轮只回答 default dropout 下 1 epoch 是否值得继续 4 epoch confirm。\n"
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cache-hash")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=BASE.run_cache_hash)

    p = sub.add_parser("verify-init")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--checkpoint", type=Path, default=PRIOR_INIT_CHECKPOINT)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=BASE.run_verify_init)

    p = sub.add_parser("preflight")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_preflight)

    p = sub.add_parser("config-summary")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=run_config_summary)

    p = sub.add_parser("train")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--variant", choices=TARGETS, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--trace-output-dir", type=Path, required=True)
    p.add_argument("--output-config-json", type=Path)
    p.add_argument("--output-result-json", type=Path)
    p.add_argument("--logger-backend", choices=["none", "swanlab", "wandb"], default="none")
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--max-train-steps", type=int)
    p.add_argument("--max-validation-batches", type=int)
    p.set_defaults(func=run_train)

    p = sub.add_parser("collect")
    p.add_argument("--outputs-dir", type=Path, default=SCRIPT_DIR / "outputs")
    p.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    p.set_defaults(func=run_collect)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    BASE.EXPECTED_INIT_STATE_SHA256 = _read_expected_init_hash()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
