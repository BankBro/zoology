#!/usr/bin/env python3
"""
Flash-VQG CLR v1 显存 OOM smoke test.

测试 12 个组合 (den × rank × remat), 判断哪些配置能在当前 VRAM
下跑通正式配方的短程训练 + 评估. 支持双 GPU 并行.

用法:
    # 单 GPU
    python -m zoology.experiments.flash_vqg.scripts.smoke_clr_oom_grid --gpu 0

    # 双 GPU 并行 (12 组合按 rank 分成两半, 各 6 个)
    python -m zoology.experiments.flash_vqg.scripts.smoke_clr_oom_grid --gpu 0,1

=============================================================================
模型/数据超参对齐说明 (对齐 run_flash_vqg_e0.sh 正式配方)
=============================================================================

以下参数与 e0 正式配方完全一致, CLR v1 仅覆盖 fox_remote_formula 和扫描维度:

  d_model            = 128
  num_heads           = 2          # build_configs 硬编码, k=v=64
  block_len           = 32         # e0: --block-len 32
  local_num_blocks    = 2          # CLI 默认
  num_codebook_vectors = 128       # DEFAULT_NUM_CODEBOOK_VECTORS_MAP[128]
  n_layers            = 2          # add_flash_vqg 默认
  sequence_mixer      = Hybrid([BaseConv, FlashVQG])
                                   # L0=BaseConv, L1=FlashVQG (仅 1 层注意力)
  state_mixer         = Identity   # 无 MLP
  vocab_size          = 8192
  use_time_mixing     = kv_shift
  vq_score_mode       = l2
  vq_weight_mode      = one-hot
  vq_update_mode      = ema
  if_value_silu       = True
  output_gate         = swish + RMSNorm
  fox_if_local_use_vq_k = False
  codebook_beta       = 0.25

CLR v1 强制覆盖 (相对 e0 legacy):
  flash_backend             = torch    # CLR 要求, e0 用 accel
  fox_remote_formula        = clr_v1   # e0 用 legacy
  fox_state_build_backend   = torch    # CLR 不支持 triton
  fox_remote_path_backend   = torch    # CLR 不支持 triton
  vq_use_triton_shortcodes  = False    # 跟随 flash_backend=torch

数据配方 (_build_data_config):
  train segments: 5 段
    seq_len  64, 100k examples, 4 kv_pairs
    seq_len 128,  20k examples, 8 kv_pairs
    seq_len 256,  20k examples, 16 kv_pairs
    seq_len 256,  20k examples, 32 kv_pairs
    seq_len 256,  20k examples, 64 kv_pairs
  test segments: 8 段, seq_len 最大 1024
  batch_size = (256 train, 32 eval)

扫描维度:
  fox_clr_rank             ∈ {2, 4, 8}
  fox_clr_use_den_residual ∈ {False, True}   (den0, den1)
  fox_clr_remat_mode       ∈ {off, post_phase1}

Smoke 协议:
  每个组合: 2 batches × 5 train segments = 10 train steps (含 backward)
          + 1 batch  × 8 test segments  = 8 eval steps  (torch.no_grad)
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

# ---------------------------------------------------------------------------
# 扫描维度
# ---------------------------------------------------------------------------
RANKS = [2, 4, 8]
DEN_VALUES = [False, True]       # fox_clr_use_den_residual
REMAT_VALUES = ["off", "post_phase1"]
DEFAULT_TRAIN_BATCH_SIZE = 256
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1

# ---------------------------------------------------------------------------
# Smoke 协议
# ---------------------------------------------------------------------------
TRAIN_BATCHES_PER_SEGMENT = 2    # 每个 train segment 取前 N 个 batch
EVAL_BATCHES_PER_SEGMENT = 1     # 每个 eval segment 取前 N 个 batch


def _combo_tag(rank: int, den: bool, remat: str) -> str:
    return f"r{rank}-den{int(den)}-remat-{'on' if remat == 'post_phase1' else 'off'}"


def _build_one_config(
    rank: int,
    den: bool,
    remat: str,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
):
    """用 build_configs 生成单个 TrainConfig, 对齐 e0 正式配方."""
    from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs

    configs = build_configs(
        sweep_id="smoke-clr-oom",
        flash_backend="torch",
        logger_backend="none",
        include_gdn=False,
        block_len=32,
        local_num_blocks=2,
        dmodels=[128],
        learning_rates=[1e-3],
        if_remote_enabled=True,
        num_codebook_vectors_values=[128],
        fox_remote_path_backend="torch",
        fox_remote_formula="clr_v1",
        fox_clr_rank=rank,
        fox_clr_use_den_residual=den,
        fox_clr_remat_mode=remat,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        train_batch_order="global_shuffle",
        cache_dir="./data/flash_vqg",
        wandb_project="smoke",
        wandb_entity="smoke",
        max_epochs=1,
        metrics_white_list=["train/loss"],
    )
    if len(configs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 config, got {len(configs)} "
            f"(rank={rank}, den={den}, remat={remat})"
        )
    return configs[0]


def _run_one_combo(
    rank: int,
    den: bool,
    remat: str,
    device: torch.device,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
) -> dict:
    """跑单个组合的 smoke, 返回结果 dict."""
    from zoology.data.utils import prepare_data
    from zoology.model import LanguageModel

    tag = _combo_tag(rank, den, remat)
    result = {
        "tag": tag,
        "rank": rank,
        "den": int(den),
        "remat": "on" if remat == "post_phase1" else "off",
        "status": "unknown",
        "oom_phase": None,
        "train_steps": 0,
        "eval_steps": 0,
        "peak_mb": 0,
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "effective_train_batch_size": int(train_batch_size) * int(gradient_accumulation_steps),
        "error": None,
    }

    # --- 构建 ---
    try:
        config = _build_one_config(
            rank,
            den,
            remat,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        model = LanguageModel(config.model).to(device)
        train_dl, test_dl = prepare_data(config.data)
    except Exception as exc:
        result["status"] = "build_error"
        result["error"] = str(exc)
        return result

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    torch.cuda.reset_peak_memory_stats(device)

    # --- Train ---
    try:
        model.train()
        steps_done = 0
        accum_steps = int(config.gradient_accumulation_steps)
        num_batches = len(train_dl)
        remainder = num_batches % accum_steps
        partial_start = num_batches - remainder if remainder > 0 else num_batches
        optimizer.zero_grad()

        # train_dl 的 batch 按 segment 顺序排列, 同一 segment 的 batch 连续
        # 这里用 segment→batch 的方式, 每个 segment 取前 N 个 batch
        seg_counts: dict[int, int] = {}
        for step_idx, (inputs, targets, slices) in enumerate(train_dl):
            seg_idx = slices[0].get("mqar_case", steps_done)
            seg_counts.setdefault(seg_idx, 0)
            if seg_counts[seg_idx] >= TRAIN_BATCHES_PER_SEGMENT:
                continue
            seg_counts[seg_idx] += 1

            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())

            aux = []
            model.apply(
                lambda m: aux.append(m.get_auxiliary_loss())
                if hasattr(m, "get_auxiliary_loss")
                else None
            )
            if aux:
                loss = loss + sum(aux)

            effective_accum = remainder if step_idx >= partial_start and remainder > 0 else accum_steps
            (loss / effective_accum).backward()

            is_accum_boundary = (step_idx + 1) % accum_steps == 0
            is_last_batch = (step_idx + 1) == num_batches
            if is_accum_boundary or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
            steps_done += 1

            # 所有 segment 都取够了就停
            if all(v >= TRAIN_BATCHES_PER_SEGMENT for v in seg_counts.values()):
                if len(seg_counts) >= 5:
                    break

        result["train_steps"] = steps_done
    except torch.cuda.OutOfMemoryError:
        result["status"] = "oom"
        result["oom_phase"] = "train"
        result["peak_mb"] = torch.cuda.max_memory_allocated(device) // (1 << 20)
        _cleanup(model, optimizer)
        return result
    except Exception as exc:
        result["status"] = "train_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _cleanup(model, optimizer)
        return result

    # --- Eval ---
    try:
        model.eval()
        eval_steps = 0
        seg_counts_eval: dict[int, int] = {}
        with torch.no_grad():
            for inputs, targets, slices in test_dl:
                seg_idx = slices[0].get("mqar_case", eval_steps)
                seg_counts_eval.setdefault(seg_idx, 0)
                if seg_counts_eval[seg_idx] >= EVAL_BATCHES_PER_SEGMENT:
                    continue
                seg_counts_eval[seg_idx] += 1

                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                _ = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())
                eval_steps += 1

                if all(v >= EVAL_BATCHES_PER_SEGMENT for v in seg_counts_eval.values()):
                    if len(seg_counts_eval) >= 8:
                        break

        result["eval_steps"] = eval_steps
    except torch.cuda.OutOfMemoryError:
        result["status"] = "oom"
        result["oom_phase"] = "eval"
        result["peak_mb"] = torch.cuda.max_memory_allocated(device) // (1 << 20)
        _cleanup(model, optimizer)
        return result
    except Exception as exc:
        result["status"] = "eval_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _cleanup(model, optimizer)
        return result

    result["status"] = "pass"
    result["peak_mb"] = torch.cuda.max_memory_allocated(device) // (1 << 20)
    _cleanup(model, optimizer)
    return result


def _cleanup(model, optimizer):
    """释放模型和优化器, 强制 GC + CUDA cache 清理."""
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()


def _run_combos_on_gpu(
    gpu_id: int,
    combos: list[tuple[int, bool, str]],
    *,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
) -> list[dict]:
    """在指定 GPU 上串行跑一组 combos, 返回结果列表."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem_mb = torch.cuda.get_device_properties(device).total_memory // (1 << 20)
    print(f"[GPU {gpu_id}] {gpu_name} ({gpu_mem_mb} MB), {len(combos)} combos", flush=True)

    results = []
    for i, (rank, den, remat) in enumerate(combos, 1):
        tag = _combo_tag(rank, den, remat)
        print(f"[GPU {gpu_id}] [{i}/{len(combos)}] {tag}", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

        t0 = time.time()
        result = _run_one_combo(
            rank,
            den,
            remat,
            device,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        elapsed = time.time() - t0

        status_icon = "+" if result["status"] == "pass" else "X"
        status_detail = result["status"]
        if result["oom_phase"]:
            status_detail += f" ({result['oom_phase']})"
        if result["error"]:
            status_detail += f" [{result['error'][:60]}]"

        print(
            f"[GPU {gpu_id}]   {status_icon} {status_detail}  |  "
            f"train={result['train_steps']} eval={result['eval_steps']}  |  "
            f"peak={result['peak_mb']} MB  |  {elapsed:.1f}s",
            flush=True,
        )
        results.append(result)

    return results


def _worker(
    gpu_id: int,
    combos: list[tuple[int, bool, str]],
    result_file: str,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
):
    """子进程入口: 在 gpu_id 上跑 combos, 结果写 JSON 文件."""
    results = _run_combos_on_gpu(
        gpu_id,
        combos,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    for r in results:
        r["gpu"] = gpu_id
    with open(result_file, "w") as f:
        json.dump(results, f)


def _print_summary(results: list[dict]):
    print()
    print("=" * 120)
    print("Summary")
    print("=" * 120)
    header = (
        f"{'combo':<28} {'gpu':>3}  {'status':<16} {'phase':<8} {'peak_MB':>8}  "
        f"{'tbs':>5} {'ebs':>5} {'ga':>4} {'eff':>6}  {'train':>5}  {'eval':>4}"
    )
    print(header)
    print("-" * 120)
    for r in sorted(results, key=lambda x: (x["rank"], x["den"], x["remat"])):
        phase = r.get("oom_phase") or "-"
        gpu = r.get("gpu", "?")
        print(
            f"{r['tag']:<28} {gpu:>3}  {r['status']:<16} {phase:<8} {r['peak_mb']:>8}  "
            f"{r['train_batch_size']:>5} {r['eval_batch_size']:>5} "
            f"{r['gradient_accumulation_steps']:>4} {r['effective_train_batch_size']:>6}  "
            f"{r['train_steps']:>5}  {r['eval_steps']:>4}"
        )

    pass_count = sum(1 for r in results if r["status"] == "pass")
    oom_count = sum(1 for r in results if r["status"] == "oom")
    err_count = sum(1 for r in results if r["status"] not in ("pass", "oom"))
    print("-" * 120)
    print(f"pass={pass_count}  oom={oom_count}  error={err_count}  total={len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Flash-VQG CLR v1 OOM smoke test")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU id(s), 逗号分隔. 例: 0  或  0,1",
    )
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpu.split(",")]
    all_combos = list(product(RANKS, DEN_VALUES, REMAT_VALUES))

    print(f"GPUs: {gpu_ids}")
    print(f"Combos: {len(all_combos)}")
    print(
        f"Train batch: {args.train_batch_size}, Eval batch: {args.eval_batch_size}, "
        f"GA: {args.gradient_accumulation_steps}, "
        f"Effective train batch: {args.train_batch_size * args.gradient_accumulation_steps}"
    )
    print()

    if len(gpu_ids) == 1:
        # 单 GPU: 直接在主进程跑
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
        results = _run_combos_on_gpu(
            gpu_ids[0],
            all_combos,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        for r in results:
            r["gpu"] = gpu_ids[0]
        _print_summary(results)
        return

    # 多 GPU: 按 rank 交错分配, 让每张卡负载均衡 (小 rank 快, 大 rank 慢)
    chunks: list[list[tuple[int, bool, str]]] = [[] for _ in gpu_ids]
    for i, combo in enumerate(all_combos):
        chunks[i % len(gpu_ids)].append(combo)

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="smoke_clr_")
    result_files = [os.path.join(tmp_dir, f"gpu{gid}.json") for gid in gpu_ids]

    processes = []
    for gid, chunk, rf in zip(gpu_ids, chunks, result_files):
        p = mp.Process(
            target=_worker,
            args=(
                gid,
                chunk,
                rf,
                args.train_batch_size,
                args.eval_batch_size,
                args.gradient_accumulation_steps,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 汇总
    all_results = []
    for rf in result_files:
        if os.path.exists(rf):
            with open(rf) as f:
                all_results.extend(json.load(f))
            os.remove(rf)
    os.rmdir(tmp_dir)

    _print_summary(all_results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
