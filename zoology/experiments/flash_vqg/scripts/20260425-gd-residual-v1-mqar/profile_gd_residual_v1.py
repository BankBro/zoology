from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs
from zoology.model import LanguageModel
from zoology.utils import set_determinism


GD_RESIDUAL_METRICS = [
    "train/loss",
    "attn/gd_residual_inject_ratio",
    "attn/gd_residual_lambda_mean",
    "attn/gd_residual_write_strength_mean",
    "attn/gd_residual_m_norm_mean",
    "attn/gd_residual_m_norm_max",
    "attn/gd_residual_mu_valid_ratio",
]


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _parse_read_topk(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"dense", "none", "null"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError("remote read topk must be a positive integer or dense.")
    return parsed


def _read_topk_tag(value: int | None) -> str:
    return "dense" if value is None else f"top{int(value)}"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _collect_model_scalar_metrics(model: nn.Module) -> dict[str, float]:
    scalar_metrics: dict[str, float] = {}
    for module in model.modules():
        getter = getattr(module, "get_scalar_metrics", None)
        if getter is None:
            continue
        module_metrics = getter()
        if not module_metrics:
            continue
        for key, value in module_metrics.items():
            scalar_metrics[str(key)] = float(value)
    return scalar_metrics


def _build_config(args: argparse.Namespace, read_topk: int | None):
    configs = build_configs(
        sweep_id="gd-residual-v1-profile",
        flash_backend="torch",
        logger_backend="none",
        include_gdn=False,
        block_len=int(args.block_len),
        local_num_blocks=2,
        dmodels=[int(args.d_model)],
        learning_rates=[float(args.learning_rate)],
        if_remote_enabled=True,
        train_batch_order="global_shuffle",
        seed_values=[int(args.seed)],
        data_seed=int(args.seed),
        num_codebook_vectors_values=[int(args.num_codebook_vectors)],
        fox_remote_path_backend="torch",
        fox_remote_read_topk_values=[read_topk],
        fox_remote_formula="gd_residual_v1",
        fox_gd_residual_rank=int(args.rank),
        fox_gd_residual_write_topk=int(args.write_topk),
        fox_gd_residual_builder=str(args.builder),
        fox_gd_residual_pack_mode=str(args.pack_mode),
        fox_gd_residual_chunk_size=int(args.chunk_size),
        fox_gd_residual_beta_init=0.5,
        fox_gd_residual_lambda_init=0.05,
        vq_score_mode="codebook_dot",
        vq_weight_mode="dense_softmax",
        vq_update_mode="grad",
        vq_softmax_tau=float(args.vq_softmax_tau),
        vq_topk=max(int(args.write_topk), 4),
        gradient_accumulation_steps=1,
        train_batch_size=int(args.batch_size),
        eval_batch_size=1,
        cache_dir="./data/flash_vqg",
        metrics_white_list=GD_RESIDUAL_METRICS,
    )
    if len(configs) != 1:
        raise RuntimeError(f"Expected one profiling config, got {len(configs)}.")
    config = configs[0]
    config.checkpoint.enabled = False
    config.run_id = (
        f"gd-residual-v1-profile-rread-{_read_topk_tag(read_topk)}"
        f"-r{int(args.rank)}-wk{int(args.write_topk)}-b{int(args.batch_size)}"
        f"-t{int(args.seq_len)}"
    )
    return config


def _profiler_context(enabled: bool, output_dir: Path):
    if not enabled:
        return nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir / "trace")),
    )


def _write_profiler_tables(prof, output_dir: Path, has_cuda: bool) -> dict[str, str]:
    if prof is None:
        return {}
    sort_keys = ["cpu_time_total", "self_cpu_time_total"]
    if has_cuda:
        sort_keys.extend(
            [
                "self_cuda_time_total",
                "cuda_time_total",
                "cuda_memory_usage",
                "self_cuda_memory_usage",
            ]
        )
    else:
        sort_keys.append("self_cpu_memory_usage")
    outputs: dict[str, str] = {}
    averages = prof.key_averages()
    for sort_key in sort_keys:
        table = averages.table(sort_by=sort_key, row_limit=100)
        path = output_dir / f"profiler_{sort_key}.txt"
        path.write_text(table, encoding="utf-8")
        outputs[sort_key] = str(path)
    return outputs


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    read_topk = _parse_read_topk(args.remote_read_topk)
    seq_len = int(args.seq_len)
    block_len = int(args.block_len)
    if seq_len <= 0 or seq_len % block_len != 0:
        raise ValueError("PROFILE_SEQ_LEN must be positive and divisible by block_len.")
    gd_diagnostics_enabled = bool(args.enable_gd_diagnostics)
    os.environ["FOX_GD_RESIDUAL_PROFILE_DIAGNOSTICS"] = "1" if gd_diagnostics_enabled else "0"

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(int(args.seed), deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _build_config(args, read_topk)
    model = LanguageModel(config.model).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = int(args.batch_size)
    vocab_size = int(config.model.vocab_size)
    input_ids = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records: list[dict[str, Any]] = []
    prof_enabled = bool(args.enable_torch_profiler)
    with _profiler_context(prof_enabled, output_dir) as prof:
        for microbatch_idx in range(int(args.microbatches)):
            _sync(device)
            total_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            forward_start = time.perf_counter()
            logits = model(input_ids)
            loss = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())
            _sync(device)
            forward_sec = time.perf_counter() - forward_start

            backward_start = time.perf_counter()
            loss.backward()
            _sync(device)
            backward_sec = time.perf_counter() - backward_start

            optimizer_start = time.perf_counter()
            optimizer.step()
            _sync(device)
            optimizer_sec = time.perf_counter() - optimizer_start

            metrics_start = time.perf_counter()
            scalar_metrics = {
                key: value
                for key, value in _collect_model_scalar_metrics(model).items()
                if "gd_residual" in key
            }
            _sync(device)
            metrics_sec = time.perf_counter() - metrics_start

            total_sec = time.perf_counter() - total_start
            record = {
                "microbatch": microbatch_idx,
                "forward_sec": forward_sec,
                "backward_sec": backward_sec,
                "optimizer_sec": optimizer_sec,
                "metrics_collect_sec": metrics_sec,
                "microbatch_sec": total_sec,
                "loss": float(loss.detach().cpu().item()),
                "metrics": scalar_metrics,
            }
            records.append(record)
            if prof is not None:
                prof.step()

    has_cuda = device.type == "cuda"
    memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if has_cuda else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if has_cuda else None,
    }
    profiler_tables = _write_profiler_tables(
        prof if prof_enabled else None,
        output_dir=output_dir,
        has_cuda=has_cuda,
    )
    summary = {
        "run_id": config.run_id,
        "device": str(device),
        "profile": {
            "seq_len": seq_len,
            "microbatches": int(args.microbatches),
            "batch_size": batch_size,
            "read_topk": read_topk,
            "rank": int(args.rank),
            "write_topk": int(args.write_topk),
            "vq_softmax_tau": float(args.vq_softmax_tau),
            "builder": str(args.builder),
            "pack_mode": str(args.pack_mode),
            "chunk_size": int(args.chunk_size),
            "gd_diagnostics": gd_diagnostics_enabled,
        },
        "memory": memory,
        "records": records,
        "profiler_tables": profiler_tables,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile gd_residual_v1 MQAR microbatches.")
    parser.add_argument("--remote-read-topk", default=os.environ.get("FOX_REMOTE_READ_TOPK", "2"))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("FOX_GD_RESIDUAL_RANK", "16")))
    parser.add_argument(
        "--write-topk",
        type=int,
        default=int(os.environ.get("FOX_GD_RESIDUAL_WRITE_TOPK", "4")),
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("TRAIN_BATCH_SIZE", "16")))
    parser.add_argument("--seq-len", type=int, default=int(os.environ.get("PROFILE_SEQ_LEN", "128")))
    parser.add_argument(
        "--microbatches",
        type=int,
        default=int(os.environ.get("PROFILE_MICROBATCHES", "3")),
    )
    parser.add_argument(
        "--enable-torch-profiler",
        action="store_true",
        default=_env_bool("PROFILE_ENABLE_TORCH_PROFILER", "0"),
    )
    parser.add_argument(
        "--enable-gd-diagnostics",
        action="store_true",
        default=_env_bool("PROFILE_ENABLE_GD_DIAGNOSTICS", "0"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "PROFILE_OUTPUT_DIR",
            "tmp/20260425-gd-residual-v1-profile",
        ),
    )
    parser.add_argument("--d-model", type=int, default=int(os.environ.get("DMODEL", "128")))
    parser.add_argument(
        "--num-codebook-vectors",
        type=int,
        default=int(os.environ.get("NUM_CODEBOOK_VECTORS", "128")),
    )
    parser.add_argument(
        "--vq-softmax-tau",
        type=float,
        default=float(os.environ.get("VQ_SOFTMAX_TAU", "0.25")),
    )
    parser.add_argument("--block-len", type=int, default=int(os.environ.get("BLOCK_LEN", "32")))
    parser.add_argument(
        "--builder",
        choices=["token_step_ref", "grouped_chunk_torch_ref"],
        default=os.environ.get("FOX_GD_RESIDUAL_BUILDER", "grouped_chunk_torch_ref"),
    )
    parser.add_argument(
        "--pack-mode",
        choices=["loop_ref", "semivec_ref"],
        default=os.environ.get("FOX_GD_RESIDUAL_PACK_MODE", "semivec_ref"),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("FOX_GD_RESIDUAL_CHUNK_SIZE", "64")),
    )
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-3")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED_VALUES", "123").split(",")[0]))
    return parser.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())
