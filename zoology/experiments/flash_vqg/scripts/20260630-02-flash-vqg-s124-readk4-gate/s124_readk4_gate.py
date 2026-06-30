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
BASE_SCRIPT = (
    REPO_ROOT
    / "zoology/experiments/flash_vqg/scripts/20260630-01-flash-vqg-train-read-topk-screen/train_read_topk_screen.py"
)
EXPERIMENT_ID = "20260630-02-flash-vqg-s124-readk4-gate"
ARTIFACT_DIR = REPO_ROOT / "docs/artifacts" / EXPERIMENT_ID
DEFAULT_INIT_CHECKPOINT = SCRIPT_DIR / "outputs/canonical-init/cb64r16-s124-init.pt"
DEFAULT_INIT_META = SCRIPT_DIR / "outputs/canonical-init/cb64r16-s124-init.meta.json"
TARGETS = ("fixed-r2-baseline", "fixed-r4")


def _load_base_module():
    spec = importlib.util.spec_from_file_location("train_read_topk_screen_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
ORIGINAL_BASE_ARGS = BASE._base_args

VARIANTS = {
    "fixed-r2-baseline": BASE.VARIANTS["fixed-r2-baseline"],
    "fixed-r4": BASE.VARIANTS["fixed-r4"],
}


def _read_expected_init_hash() -> str:
    env_value = os.environ.get("EXPECTED_INIT_STATE_SHA256")
    if env_value:
        return env_value
    if DEFAULT_INIT_META.exists():
        payload = json.loads(DEFAULT_INIT_META.read_text(encoding="utf-8"))
        return str(payload.get("model_state_sha256", ""))
    return ""


def _patch_base() -> None:
    BASE.SCRIPT_DIR = SCRIPT_DIR
    BASE.EXPERIMENT_ID = EXPERIMENT_ID
    BASE.ARTIFACT_DIR = ARTIFACT_DIR
    BASE.TARGETS = TARGETS
    BASE.VARIANTS = VARIANTS
    BASE.DEFAULT_INIT_CHECKPOINT = DEFAULT_INIT_CHECKPOINT
    BASE.EXPECTED_INIT_STATE_SHA256 = _read_expected_init_hash()

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
        args.project = "flash_vqg_s124_readk4_gate"
        args.experiment_mode = f"{EXPERIMENT_ID}_{variant}_s124_d123_b64ga4_{machine_name}"
        args.run_id = f"{EXPERIMENT_ID}-{variant}-s124-d123-b64ga4-{machine_name}"
        args.launch_id_prefix = f"fvqg-{EXPERIMENT_ID}-{machine_name}-{target}"
        return args

    BASE._base_args = _base_args


_patch_base()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=BASE._json_default)
        + "\n",
        encoding="utf-8",
    )


def make_init(args: argparse.Namespace) -> int:
    config = BASE.build_config(
        target="fixed-r2-baseline",
        machine_name=args.machine_name,
        variant="fixed-r2-baseline",
        logger_backend="none",
        trace_output_dir=SCRIPT_DIR / "outputs/init-traces" / args.machine_name,
        max_epochs=1,
        max_train_steps=None,
        max_validation_batches=None,
    )
    BASE.set_determinism(config.seed, deterministic=False)
    model = BASE.LanguageModel(config.model)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    state_hash, per_tensor = BASE._state_dict_hashes(state)
    payload = {
        "model_state_dict": state,
        "model_state_sha256": state_hash,
        "per_tensor_sha256": per_tensor,
        "config": BASE.serialize_train_config(config),
        "env": BASE.env_snapshot(args.machine_name),
        "run_id": config.run_id,
        "launch_id": config.launch_id,
        "experiment_id": EXPERIMENT_ID,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    meta = {
        "checkpoint": str(args.output),
        "model_state_sha256": state_hash,
        "num_tensors": len(per_tensor),
        "env": payload["env"],
        "run_id": config.run_id,
        "launch_id": config.launch_id,
    }
    if args.meta_json:
        save_json(args.meta_json, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, default=BASE._json_default))
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
        "本 artifact 收尾 `seed=124` fixed read_topk gate 1 epoch screen. "
        "本轮是 diagnostic screen, 不写 official MQAR ledger.\n\n"
        "共同配置: `seed=124`, `data_seed=123`, `cb64-r16`, `write_topk=4`, "
        "no-dropout, canonical MQAR cache, seed124 canonical init, 1 epoch. "
        "变量只有 train-time read_topk: fixed 2 vs fixed 4.\n\n"
        "## 文件\n\n"
        "- `run-summary.csv`: per-run final metrics.\n"
        "- `variant-summary.csv`: 每个 variant 的 2080ti/3090 成对结果.\n"
        "- `cross-machine-comparison.csv`: 每个 variant 的 1024x256 cross-machine gap.\n"
        "- `cache-init-preflight-summary.csv`: cache content hash 与 init state hash preflight.\n"
        "- `queue-summary.csv`: queue 状态.\n"
        "- `invalid-runs.csv`: failed/interrupted/pending run.\n"
        "- `source-manifest.csv`: raw evidence 路径和 sha256.\n"
        "- `metadata.json`: 收尾元数据.\n"
        "\n"
        "注: `best_*` 是日志观测到的 best validation metric, 不是 saved-best checkpoint 复评.\n"
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "README.md").write_text(readme, encoding="utf-8")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-init")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--output", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--meta-json", type=Path, default=DEFAULT_INIT_META)
    p.set_defaults(func=make_init)

    p = sub.add_parser("cache-hash")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=BASE.run_cache_hash)

    p = sub.add_parser("verify-init")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=BASE.run_verify_init)

    p = sub.add_parser("preflight")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--max-epochs", type=int, default=BASE.DEFAULT_MAX_EPOCHS)
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=BASE.run_preflight)

    p = sub.add_parser("config-summary")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--variant", choices=TARGETS, default=TARGETS[0])
    p.add_argument("--output-json", type=Path)
    p.set_defaults(func=BASE.run_config_summary)

    p = sub.add_parser("train")
    p.add_argument("--machine-name", required=True)
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--variant", choices=TARGETS, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--trace-output-dir", type=Path, required=True)
    p.add_argument("--output-config-json", type=Path)
    p.add_argument("--output-result-json", type=Path)
    p.add_argument("--logger-backend", choices=["none", "swanlab", "wandb"], default="none")
    p.add_argument("--max-epochs", type=int, default=BASE.DEFAULT_MAX_EPOCHS)
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
