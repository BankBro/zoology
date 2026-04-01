#!/usr/bin/env python3
"""
对比 remat=on 与 remat=off 的端到端训练效果.

用小 batch_size (16) 让两个模式都能跑进显存, 对比 loss 和 accuracy 曲线.
选 r4-den1 作为代表性 combo.

用法:
    # 两张卡各跑一个模式
    python -m zoology.experiments.flash_vqg.scripts.compare_remat_training \
        --mode off --gpu 0 --out /tmp/remat_off.jsonl &
    python -m zoology.experiments.flash_vqg.scripts.compare_remat_training \
        --mode on --gpu 1 --out /tmp/remat_on.jsonl &
    wait
    python -m zoology.experiments.flash_vqg.scripts.compare_remat_training --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 实验参数
# ---------------------------------------------------------------------------
RANK = 4
DEN = True           # fox_clr_use_den_residual
MAX_EPOCHS = 8
LR = 1e-3
SEED = 123
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1


def _build_config(
    remat_mode: str,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
):
    from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs

    configs = build_configs(
        sweep_id="remat-cmp",
        flash_backend="torch",
        logger_backend="none",
        include_gdn=False,
        block_len=32,
        local_num_blocks=2,
        dmodels=[128],
        learning_rates=[LR],
        if_remote_enabled=True,
        num_codebook_vectors_values=[128],
        fox_remote_path_backend="torch",
        fox_remote_formula="clr_v1",
        fox_clr_rank=RANK,
        fox_clr_use_den_residual=DEN,
        fox_clr_remat_mode=remat_mode,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        train_batch_order="global_shuffle",
        data_seed=SEED,
        cache_dir="./data/flash_vqg",
        wandb_project="smoke",
        wandb_entity="smoke",
        max_epochs=MAX_EPOCHS,
        metrics_white_list=["train/loss"],
    )
    assert len(configs) == 1
    return configs[0]


def _train(
    mode_label: str,
    remat_mode: str,
    device: torch.device,
    out_path: Path,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
):
    from zoology.data.utils import prepare_data
    from zoology.model import LanguageModel
    from zoology.utils import set_determinism

    set_determinism(SEED)

    config = _build_config(
        remat_mode,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    model = LanguageModel(config.model).to(device)
    train_dl, test_dl = prepare_data(config.data)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    accum_steps = int(config.gradient_accumulation_steps)

    log_records = []

    for epoch in range(MAX_EPOCHS):
        # --- train ---
        model.train()
        train_losses = []
        num_batches = len(train_dl)
        remainder = num_batches % accum_steps
        partial_start = num_batches - remainder if remainder > 0 else num_batches
        optimizer.zero_grad()
        for step_idx, (inputs, targets, slices) in enumerate(tqdm(
            train_dl, desc=f"[{mode_label}] Train {epoch}", leave=False
        )):
            inputs, targets = inputs.to(device), targets.to(device)

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

            train_losses.append(loss.item())
            effective_accum = remainder if step_idx >= partial_start and remainder > 0 else accum_steps
            (loss / effective_accum).backward()

            is_accum_boundary = (step_idx + 1) % accum_steps == 0
            is_last_batch = (step_idx + 1) == num_batches
            if is_accum_boundary or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # --- eval ---
        model.eval()
        eval_correct = 0
        eval_total = 0
        eval_losses = []
        with torch.no_grad():
            for inputs, targets, slices in tqdm(
                test_dl, desc=f"[{mode_label}] Eval {epoch}", leave=False
            ):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())
                eval_losses.append(loss.item())
                preds = logits.argmax(dim=-1)
                mask = targets != -100
                eval_correct += (preds[mask] == targets[mask]).sum().item()
                eval_total += mask.sum().item()

        train_loss = float(np.mean(train_losses))
        eval_loss = float(np.mean(eval_losses))
        eval_acc = eval_correct / max(eval_total, 1)

        record = {
            "mode": mode_label,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "eval_loss": round(eval_loss, 6),
            "eval_acc": round(eval_acc, 6),
            "train_batch_size": int(train_batch_size),
            "eval_batch_size": int(eval_batch_size),
            "gradient_accumulation_steps": int(accum_steps),
            "effective_train_batch_size": int(train_batch_size) * int(accum_steps),
        }
        log_records.append(record)
        print(
            f"[{mode_label}] epoch={epoch}  "
            f"train_loss={train_loss:.4f}  "
            f"eval_loss={eval_loss:.4f}  "
            f"eval_acc={eval_acc:.4f}"
        )

    # 写出
    with open(out_path, "w") as f:
        for r in log_records:
            f.write(json.dumps(r) + "\n")
    print(f"[{mode_label}] Results saved to {out_path}")


def _print_summary():
    off_path = Path("/tmp/remat_off.jsonl")
    on_path = Path("/tmp/remat_on.jsonl")

    for p in [off_path, on_path]:
        if not p.exists():
            print(f"Missing: {p}")
            sys.exit(1)

    def load(path):
        records = [json.loads(line) for line in path.read_text().strip().split("\n")]
        return {r["epoch"]: r for r in records}

    off = load(off_path)
    on = load(on_path)

    epochs = sorted(set(off.keys()) & set(on.keys()))

    print()
    print(f"{'epoch':>5}  {'train_loss_off':>14} {'train_loss_on':>14} {'diff':>10}  "
          f"{'eval_acc_off':>12} {'eval_acc_on':>12} {'diff':>10}")
    print("-" * 85)
    for e in epochs:
        tl_off, tl_on = off[e]["train_loss"], on[e]["train_loss"]
        ea_off, ea_on = off[e]["eval_acc"], on[e]["eval_acc"]
        print(
            f"{e:>5}  {tl_off:>14.6f} {tl_on:>14.6f} {tl_on - tl_off:>+10.6f}  "
            f"{ea_off:>12.6f} {ea_on:>12.6f} {ea_on - ea_off:>+10.6f}"
        )

    final = epochs[-1]
    print()
    print(f"Final epoch {final}:")
    print(f"  remat=off  eval_acc={off[final]['eval_acc']:.4f}")
    print(f"  remat=on   eval_acc={on[final]['eval_acc']:.4f}")
    print(f"  diff       {on[final]['eval_acc'] - off[final]['eval_acc']:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["off", "on"], help="remat mode to run")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--summary", action="store_true", help="只打印两边结果对比")
    args = parser.parse_args()

    if args.summary:
        _print_summary()
        return

    if args.mode is None:
        parser.error("--mode is required (or use --summary)")

    remat_mode = "post_phase1" if args.mode == "on" else "off"
    mode_label = f"remat-{args.mode}"
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    out_path = Path(args.out or f"/tmp/remat_{args.mode}.jsonl")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"Mode: {mode_label}, rank={RANK}, den={int(DEN)}, epochs={MAX_EPOCHS}, "
        f"train_batch={args.train_batch_size}, eval_batch={args.eval_batch_size}, "
        f"ga={args.gradient_accumulation_steps}, "
        f"effective_train_batch={args.train_batch_size * args.gradient_accumulation_steps}"
    )
    print()

    _train(
        mode_label,
        remat_mode,
        device,
        out_path,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
