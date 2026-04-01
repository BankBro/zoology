#!/usr/bin/env python3
"""
验证 remat=on 与 remat=off 在数值上等价.

方法: 缩小 batch size 让 remat=off 也能跑进显存, 然后在完全相同的输入和
初始权重下各跑一个 forward+backward, 比较:
  1. loss 值是否一致
  2. 所有参数梯度是否一致 (最大绝对误差 + 相对误差)

用法:
    python -m zoology.experiments.flash_vqg.scripts.verify_remat_equivalence [--gpu 0]
"""

from __future__ import annotations

import argparse
import copy
import gc

import torch
import torch.nn as nn
from einops import rearrange


# 用小 batch 让 remat=off 能跑, 同时覆盖所有 rank/den 组合
SMALL_BATCH_SIZE = 16
SEQ_LEN = 256       # 最大 train seq_len
VOCAB_SIZE = 8192

COMBOS = [
    # (rank, den, label)
    (2, False, "r2-den0"),
    (2, True,  "r2-den1"),
    (4, False, "r4-den0"),
    (4, True,  "r4-den1"),
    (8, False, "r8-den0"),
    (8, True,  "r8-den1"),
]


def _build_model(rank: int, den: bool, remat: str):
    """构建 LanguageModel, 对齐 e0 配方 (除 batch_size 外)."""
    from zoology.config import ModelConfig, ModuleConfig

    flash_vqg_mixer = {
        "name": "zoology.mixers.flash_vqg.FlashVQGMixer",
        "kwargs": {
            "vocab_size": VOCAB_SIZE,
            "num_heads": 2,
            "key_dim": 64,
            "value_dim": 64,
            "num_codebook_vectors": 128,
            "block_len": 32,
            "local_num_blocks": 2,
            "if_remote_enabled": True,
            "attn_backend": "flash",
            "attn_cfg": {},
            "vq_use_triton_shortcodes": False,
            "fox_state_build_backend": "torch",
            "fox_remote_path_backend": "torch",
            "fox_remote_read_topk": None,
            "fox_remote_formula": "clr_v1",
            "fox_clr_rank": rank,
            "fox_clr_use_den_residual": den,
            "fox_clr_remat_mode": remat,
            "use_time_mixing": "kv_shift",
            "vq_score_mode": "l2",
            "vq_weight_mode": "one-hot",
            "vq_update_mode": "ema",
            "if_value_silu": True,
            "if_output_gate_use_rmsnorm": True,
            "output_gate_activation": "swish",
            "fox_if_local_use_vq_k": False,
            "codebook_beta": 0.25,
            "enable_layer_metrics": False,
        },
    }
    conv_mixer = {
        "name": "zoology.mixers.base_conv.BaseConv",
        "kwargs": {"l_max": SEQ_LEN},
    }
    model_config = ModelConfig(
        block_type="TransformerBlock",
        d_model=128,
        n_layers=2,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=0,
        name="flash_vqg",
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, flash_vqg_mixer]},
        ),
    )

    from zoology.model import LanguageModel
    return LanguageModel(model_config)


def _run_one_step(model, inputs, targets, device):
    """跑一步 forward+backward, 返回 (loss_value, grad_dict)."""
    model.train()
    model.zero_grad()

    logits = model(inputs)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(rearrange(logits, "... c -> (...) c"), targets.flatten())

    # auxiliary loss (VQ commitment)
    aux = []
    model.apply(
        lambda m: aux.append(m.get_auxiliary_loss())
        if hasattr(m, "get_auxiliary_loss")
        else None
    )
    if aux:
        loss = loss + sum(aux)

    loss.backward()

    loss_val = loss.detach().item()
    grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return loss_val, grads


def _verify_one_combo(rank: int, den: bool, label: str, device: torch.device):
    """对比一个 rank/den 组合在 remat=off 和 remat=on 下的数值."""
    # 1. 构建 remat=off 模型
    model_off = _build_model(rank, den, "off").to(device)

    # 2. 深拷贝权重给 remat=on 模型 (保证初始权重完全一致)
    model_on = _build_model(rank, den, "post_phase1").to(device)
    model_on.load_state_dict(model_off.state_dict())

    # 3. 生成确定性输入
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    inputs = torch.randint(0, VOCAB_SIZE, (SMALL_BATCH_SIZE, SEQ_LEN), generator=gen)
    targets = torch.randint(0, VOCAB_SIZE, (SMALL_BATCH_SIZE, SEQ_LEN), generator=gen)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # 4. 各跑一步
    loss_off, grads_off = _run_one_step(model_off, inputs, targets, device)
    loss_on, grads_on = _run_one_step(model_on, inputs, targets, device)

    # 5. 对比
    loss_diff = abs(loss_off - loss_on)

    max_abs_err = 0.0
    max_rel_err = 0.0
    worst_param = ""
    mismatched_params = []

    for name in grads_off:
        if name not in grads_on:
            mismatched_params.append(f"{name}: missing in remat=on")
            continue
        g_off = grads_off[name]
        g_on = grads_on[name]
        abs_err = (g_off - g_on).abs().max().item()
        denom = g_off.abs().max().item()
        rel_err = abs_err / max(denom, 1e-12)

        if abs_err > max_abs_err:
            max_abs_err = abs_err
            max_rel_err = rel_err
            worst_param = name

        # 标记不可忽略的差异
        if rel_err > 1e-3:
            mismatched_params.append(f"{name}: abs={abs_err:.2e} rel={rel_err:.2e}")

    for name in grads_on:
        if name not in grads_off:
            mismatched_params.append(f"{name}: missing in remat=off")

    # 清理
    del model_off, model_on, inputs, targets
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "loss_off": loss_off,
        "loss_on": loss_on,
        "loss_diff": loss_diff,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "worst_param": worst_param,
        "mismatched_params": mismatched_params,
        "num_params_checked": len(grads_off),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify remat=on vs remat=off numerical equivalence"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="启用 torch deterministic 模式 (排除 scatter_add 等非确定性算子的影响)",
    )
    args = parser.parse_args()

    if args.deterministic:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        print("** Deterministic mode ON **")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}")
    print(f"Batch size: {SMALL_BATCH_SIZE}, Seq len: {SEQ_LEN}")
    print()

    results = []
    for rank, den, label in COMBOS:
        print(f"Testing {label} ...", end=" ", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        r = _verify_one_combo(rank, den, label, device)
        results.append(r)

        ok = len(r["mismatched_params"]) == 0
        icon = "OK" if ok else "MISMATCH"
        print(
            f"{icon}  loss_diff={r['loss_diff']:.2e}  "
            f"max_grad_abs_err={r['max_abs_err']:.2e}  "
            f"max_grad_rel_err={r['max_rel_err']:.2e}  "
            f"({r['worst_param']})"
        )
        if r["mismatched_params"]:
            for msg in r["mismatched_params"]:
                print(f"  !! {msg}")

    # 汇总
    print()
    print("=" * 80)
    all_ok = all(len(r["mismatched_params"]) == 0 for r in results)
    max_loss_diff = max(r["loss_diff"] for r in results)
    max_grad_err = max(r["max_abs_err"] for r in results)

    print(f"All combos equivalent (rel_err < 1e-3): {'YES' if all_ok else 'NO'}")
    print(f"Max loss diff across combos:  {max_loss_diff:.2e}")
    print(f"Max grad abs err across combos: {max_grad_err:.2e}")

    if all_ok:
        print()
        print("Conclusion: remat=on is numerically equivalent to remat=off.")
        print("Safe to use remat=on for all CLR v1 experiments.")


if __name__ == "__main__":
    main()
