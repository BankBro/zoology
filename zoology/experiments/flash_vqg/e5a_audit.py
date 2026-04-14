from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from flash_vqg.nn.attn_fox import FOX_CLR_REMOTE_CODE_CHUNK, compute_boundary_c_e_from_c_end


E5A_AUDIT_LAYER_IDX = 1
E5A_MODES = ("student", "write", "time", "query", "ref")
E5A_CASES = ("512x128", "1024x256")
E5A_EPS = 1e-12


def _nanmean_tensor(values: torch.Tensor, dim: int) -> torch.Tensor:
    valid = torch.isfinite(values)
    numer = torch.where(valid, values, torch.zeros_like(values)).sum(dim=dim)
    denom = valid.to(values.dtype).sum(dim=dim)
    out = numer / denom.clamp_min(1.0)
    return torch.where(denom > 0, out, torch.full_like(out, float("nan")))


def _resolve_runtime_get(runtime, key: str, default=None):
    if runtime is None:
        return default
    if isinstance(runtime, dict):
        return runtime.get(key, default)
    return getattr(runtime, key, default)


@contextmanager
def _audit_runtime_scope(module, runtime):
    module.set_audit_runtime(runtime)
    try:
        yield
    finally:
        module.clear_audit_runtime()


def _find_audit_flash_mixer(model):
    matches: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "FlashVQGMixer":
            continue
        attn = getattr(module, "attn", None)
        if attn is None:
            continue
        if getattr(attn, "layer_idx", None) == E5A_AUDIT_LAYER_IDX:
            matches.append((name, module))
    if len(matches) != 1:
        raise ValueError(
            f"E5A 需要唯一一个 layer_idx={E5A_AUDIT_LAYER_IDX} 的 FlashVQGMixer, "
            f"当前找到 {len(matches)} 个."
        )
    return matches[0]


def _assert_supported_mainline(module) -> dict[str, Any]:
    attn = module.attn
    cfg = getattr(attn, "config", None)
    if cfg is None:
        raise ValueError("目标 FlashVQGMixer 缺少 attn.config, 无法执行 E5A.")

    checks = {
        "if_remote_enabled": bool(getattr(cfg, "if_remote_enabled", False)),
        "fox_remote_formula": str(getattr(cfg, "fox_remote_formula", "")).lower(),
        "vq_score_mode": str(getattr(cfg, "vq_score_mode", "")).lower(),
        "vq_weight_mode": str(getattr(cfg, "vq_weight_mode", "")).lower(),
        "vq_softmax_tau": float(getattr(cfg, "vq_softmax_tau", float("nan"))),
        "fox_remote_read_topk": getattr(cfg, "fox_remote_read_topk", None),
        "fox_clr_selector_mode": str(getattr(cfg, "fox_clr_selector_mode", "")).lower(),
        "fox_clr_merge_mode": str(getattr(cfg, "fox_clr_merge_mode", "")).lower(),
        "fox_clr_remat_mode": str(getattr(cfg, "fox_clr_remat_mode", "")).lower(),
    }
    expected = {
        "if_remote_enabled": True,
        "fox_remote_formula": "clr_v1",
        "vq_score_mode": "codebook_dot",
        "vq_weight_mode": "dense_softmax",
        "vq_softmax_tau": 0.25,
        "fox_remote_read_topk": 2,
        "fox_clr_selector_mode": "den_aware",
        "fox_clr_merge_mode": "shared_den",
        "fox_clr_remat_mode": "off",
    }
    for key, expected_value in expected.items():
        current_value = checks[key]
        if isinstance(expected_value, float):
            if not np.isclose(current_value, expected_value):
                raise ValueError(f"E5A 仅支持 {key}={expected_value}, 当前收到 {current_value}.")
            continue
        if current_value != expected_value:
            raise ValueError(f"E5A 仅支持 {key}={expected_value}, 当前收到 {current_value}.")
    return checks


def _flatten_case_name(slices: list[dict[str, Any]]) -> list[str]:
    case_names: list[str] = []
    for slice_info in slices:
        case_name = slice_info.get("mqar_case")
        if case_name is None:
            input_seq_len = slice_info.get("input_seq_len")
            num_kv_pairs = slice_info.get("num_kv_pairs")
            case_name = f"{input_seq_len}x{num_kv_pairs}"
        case_names.append(str(case_name))
    return case_names


def _top2_from_scores(
    scores: torch.Tensor,
    *,
    fallback_top2: torch.Tensor,
    hist_has_remote: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_codes = int(scores.size(-1))
    top_k = min(2, num_codes)
    topv, topi = torch.topk(scores, k=top_k, dim=-1)
    if top_k < 2:
        topv = torch.cat([topv, torch.zeros_like(topv[..., :1])], dim=-1)
        topi = torch.cat([topi, topi.new_full(topi[..., :1].shape, -1)], dim=-1)
    valid_primary = hist_has_remote & (topv[..., 0] > E5A_EPS)
    top2 = torch.where(valid_primary.unsqueeze(-1), topi, fallback_top2)
    valid_second = valid_primary & (topv[..., 1] > E5A_EPS)
    margin = torch.full(topv[..., 0].shape, float("nan"), dtype=torch.float32, device=scores.device)
    margin = torch.where(
        valid_second,
        torch.log(topv[..., 0].clamp_min(E5A_EPS)).float()
        - torch.log(topv[..., 1].clamp_min(E5A_EPS)).float(),
        margin,
    )
    return top2.to(torch.int64), margin


def _compute_overlap(a_codes: torch.Tensor, b_codes: torch.Tensor) -> torch.Tensor:
    a = a_codes.unsqueeze(-1)
    b = b_codes.unsqueeze(-2)
    valid = (a_codes >= 0).unsqueeze(-1) & (b_codes >= 0).unsqueeze(-2)
    matched = ((a == b) & valid).any(dim=-1).to(torch.float32).sum(dim=-1)
    return matched / 2.0


def _gather_scores(scores: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    safe_codes = codes.clamp_min(0)
    gathered = torch.gather(scores, dim=-1, index=safe_codes)
    return torch.where(codes >= 0, gathered, torch.zeros_like(gathered))


def _compute_prefix_scores(write_weights: torch.Tensor, c_all: torch.Tensor, hist_end: torch.Tensor):
    B, H, N, L, S = write_weights.shape
    T = N * L
    hist_end_by_row = hist_end.repeat_interleave(L)
    gather_idx = hist_end_by_row.clamp_min(1) - 1
    hist_mask = hist_end_by_row.view(1, 1, T, 1) > 0

    w_flat = write_weights.reshape(B, H, T, S).float()
    prefix_write = torch.cumsum(w_flat, dim=2)
    write_scores = prefix_write.index_select(dim=2, index=gather_idx)
    write_scores = torch.where(hist_mask, write_scores, torch.zeros_like(write_scores))

    c_flat = c_all.reshape(B, H, T).float()
    prefix_time = torch.cumsum(w_flat * torch.exp(-c_flat).unsqueeze(-1), dim=2)
    time_scores = prefix_time.index_select(dim=2, index=gather_idx)
    time_scores = torch.where(hist_mask, time_scores, torch.zeros_like(time_scores))
    return write_scores.reshape(B, H, N, L, S), time_scores.reshape(B, H, N, L, S)


def _compute_query_ref_scores(payload: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    Q_blk = payload["Q_blk"].float()
    K_blk = payload["K_blk"].float()
    write_weights = payload["write_weights"].float()
    c_all = payload["c_all"].reshape(Q_blk.size(0), Q_blk.size(1), -1).float()
    hist_end = payload["hist_end"].to(torch.int64)

    B, H, N, L, K = Q_blk.shape
    S = write_weights.size(-1)
    T = N * L
    K_flat = K_blk.reshape(B, H, T, K)
    W_flat = write_weights.reshape(B, H, T, S)
    query_scores = torch.zeros((B, H, N, L, S), dtype=torch.float32, device=Q_blk.device)
    ref_scores = torch.zeros_like(query_scores)

    for block_idx in range(N):
        hist = int(hist_end[block_idx].item())
        if hist <= 0:
            continue
        q_block = Q_blk[:, :, block_idx].float()
        k_hist = K_flat[:, :, :hist].float()
        w_hist = W_flat[:, :, :hist].float()
        qk = torch.einsum("bhlk,bhtk->bhlt", q_block, k_hist)

        q_shift = qk.max(dim=-1, keepdim=True).values
        query_scores[:, :, block_idx] = torch.einsum(
            "bhlt,bhts->bhls",
            torch.exp(qk - q_shift),
            w_hist,
        )

        c_hist = c_all[:, :, :hist].float().unsqueeze(-2)
        ref_logits = qk - c_hist
        ref_shift = ref_logits.max(dim=-1, keepdim=True).values
        ref_scores[:, :, block_idx] = torch.einsum(
            "bhlt,bhts->bhls",
            torch.exp(ref_logits - ref_shift),
            w_hist,
        )

    return query_scores, ref_scores


def _recompute_remote_outputs(
    payload: dict[str, torch.Tensor],
    override_top2_codes: torch.Tensor,
    cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    Q_blk = payload["Q_blk"].float()
    codebook = payload["codebook"].float()
    clr_basis = payload["clr_basis"].float()
    g_remote = payload["g_remote"].float()
    R_remote = payload["R_remote"].float()
    L_remote = payload["L_remote"].float()
    a_remote = payload["a_remote"].float()
    c_q = payload["c_q"].float()
    c_end = payload["c_end"].float()
    local_max = payload["local_max"].float()

    B, H, N, L, _ = Q_blk.shape
    V = g_remote.size(-1)
    num_codes = int(codebook.size(1))
    clr_rank = int(clr_basis.size(-1))
    code_chunk_size = min(FOX_CLR_REMOTE_CODE_CHUNK, num_codes)
    if code_chunk_size <= 0:
        raise ValueError(f"无效 num_codes={num_codes}.")
    if override_top2_codes.shape != (B, H, N, L, 2):
        raise ValueError(
            "override_top2_codes 形状不匹配, 期望 "
            f"{(B, H, N, L, 2)}, 收到 {tuple(override_top2_codes.shape)}."
        )

    idx_remote = (payload["hist_end"] // L).to(torch.int64)
    c_e = compute_boundary_c_e_from_c_end(c_end, idx_remote).float()
    p_bias = (c_q - c_e.unsqueeze(-1)).unsqueeze(-1)
    use_den_residual = bool(getattr(cfg, "fox_clr_use_den_residual", True))
    row_valid = torch.ones((B, H, N, L), dtype=torch.bool, device=Q_blk.device)
    row_valid_bhnl1 = row_valid.unsqueeze(-1)
    neg_large = Q_blk.new_full((), -1e30).float()
    remote_max = torch.full((B, H, N, L), -1e30, dtype=torch.float32, device=Q_blk.device)

    for s_start in range(0, num_codes, code_chunk_size):
        s_end = min(s_start + code_chunk_size, num_codes)
        codebook_chunk = codebook[:, s_start:s_end]
        L_chunk = L_remote[:, :, :, s_start:s_end].float()
        S_far_base_chunk = torch.einsum("bhnlk,hsk->bhnls", Q_blk, codebook_chunk) + p_bias
        den_raw_chunk = L_chunk.unsqueeze(-2)
        alpha_chunk = None
        if clr_rank > 0:
            clr_basis_chunk = clr_basis[:, s_start:s_end]
            alpha_chunk = torch.einsum("bhnlk,hskr->bhnlsr", Q_blk, clr_basis_chunk)
        if use_den_residual:
            a_chunk = a_remote[:, :, :, s_start:s_end, :].float()
            den_raw_chunk = den_raw_chunk + torch.einsum("bhnsr,bhnlsr->bhnls", a_chunk, alpha_chunk)
        den_eff_chunk = den_raw_chunk.clamp_min(0.0)
        code_valid_chunk = den_eff_chunk > 1e-30
        safe_den_chunk = torch.where(code_valid_chunk, den_eff_chunk, torch.ones_like(den_eff_chunk))
        remote_logits_chunk = S_far_base_chunk + torch.log(safe_den_chunk)
        remote_logits_chunk = torch.where(row_valid_bhnl1 & code_valid_chunk, remote_logits_chunk, neg_large)
        remote_max = torch.maximum(remote_max, remote_logits_chunk.max(dim=-1).values)

    m_shared = torch.maximum(remote_max, local_max)
    Num_far = torch.zeros((B, H, N, L, V), dtype=torch.float32, device=Q_blk.device)
    Den_far = torch.zeros((B, H, N, L), dtype=torch.float32, device=Q_blk.device)

    for s_start in range(0, num_codes, code_chunk_size):
        s_end = min(s_start + code_chunk_size, num_codes)
        codebook_chunk = codebook[:, s_start:s_end]
        g_chunk = g_remote[:, :, :, s_start:s_end, :].float()
        L_chunk = L_remote[:, :, :, s_start:s_end].float()

        S_far_base_chunk = torch.einsum("bhnlk,hsk->bhnls", Q_blk, codebook_chunk) + p_bias
        den_eff_chunk = L_chunk.unsqueeze(-2)
        alpha_chunk = None
        a_chunk = None
        if clr_rank > 0:
            clr_basis_chunk = clr_basis[:, s_start:s_end]
            alpha_chunk = torch.einsum("bhnlk,hskr->bhnlsr", Q_blk, clr_basis_chunk)
        if use_den_residual:
            a_chunk = a_remote[:, :, :, s_start:s_end, :].float()
            den_eff_chunk = den_eff_chunk + torch.einsum("bhnsr,bhnlsr->bhnls", a_chunk, alpha_chunk)
        den_eff_chunk = den_eff_chunk.clamp_min(0.0)
        code_valid_chunk = den_eff_chunk > 1e-30
        valid_mask_chunk = row_valid_bhnl1 & code_valid_chunk
        weight_logits_chunk = torch.where(
            valid_mask_chunk,
            S_far_base_chunk - m_shared.unsqueeze(-1),
            neg_large,
        )
        weight_chunk = torch.exp(weight_logits_chunk)
        chunk_code_idx = torch.arange(
            s_start,
            s_end,
            device=Q_blk.device,
            dtype=torch.int64,
        ).view(1, 1, 1, 1, -1)
        selected = override_top2_codes.unsqueeze(-1)
        keep_mask_chunk = valid_mask_chunk & (
            (selected >= 0) & (selected == chunk_code_idx)
        ).any(dim=-2)
        apply_weight_chunk = weight_chunk * keep_mask_chunk.to(weight_chunk.dtype)

        Num_far = Num_far + torch.einsum("bhnls,bhnsv->bhnlv", apply_weight_chunk, g_chunk)
        Den_far = Den_far + torch.einsum("bhnls,bhns->bhnl", apply_weight_chunk, L_chunk)
        if clr_rank > 0:
            R_chunk = R_remote[:, :, :, s_start:s_end, :, :].float()
            for r_idx in range(clr_rank):
                weight_alpha_chunk = apply_weight_chunk * alpha_chunk[..., r_idx]
                Num_far = Num_far + torch.einsum("bhnls,bhnsv->bhnlv", weight_alpha_chunk, R_chunk[..., r_idx])
                if use_den_residual:
                    Den_far = Den_far + torch.einsum("bhnls,bhns->bhnl", weight_alpha_chunk, a_chunk[..., r_idx])

    o_remote = Num_far / Den_far.clamp_min(1e-9).unsqueeze(-1)
    h_layer = (Num_far + payload["Num_loc"].float()) / (
        (Den_far + payload["Den_loc"].float()).clamp_min(1e-9).unsqueeze(-1)
    )
    return o_remote, h_layer


def _compute_mode_payload(payload: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    student_top2 = payload["student_top2"].to(torch.int64)
    hist_has_remote = payload["hist_end"].view(1, 1, -1, 1) > 0
    hist_has_remote = hist_has_remote.expand(student_top2.size(0), student_top2.size(1), -1, student_top2.size(3))

    write_scores, time_scores = _compute_prefix_scores(
        write_weights=payload["write_weights"],
        c_all=payload["c_all"],
        hist_end=payload["hist_end"].to(torch.int64),
    )
    query_scores, ref_scores = _compute_query_ref_scores(payload)

    write_top2, write_margin = _top2_from_scores(
        write_scores,
        fallback_top2=student_top2,
        hist_has_remote=hist_has_remote,
    )
    time_top2, time_margin = _top2_from_scores(
        time_scores,
        fallback_top2=student_top2,
        hist_has_remote=hist_has_remote,
    )
    query_top2, query_margin = _top2_from_scores(
        query_scores,
        fallback_top2=student_top2,
        hist_has_remote=hist_has_remote,
    )
    ref_top2, ref_margin = _top2_from_scores(
        ref_scores,
        fallback_top2=student_top2,
        hist_has_remote=hist_has_remote,
    )

    student_valid = payload["valid_remote_code_count"] >= 2
    student_margin = torch.where(
        student_valid,
        payload["student_top1_top2_margin"].float(),
        torch.full_like(payload["student_top1_top2_margin"].float(), float("nan")),
    )

    return {
        "student": {
            "top2_codes": student_top2,
            "margin": student_margin,
        },
        "write": {
            "top2_codes": write_top2,
            "margin": write_margin,
        },
        "time": {
            "top2_codes": time_top2,
            "margin": time_margin,
        },
        "query": {
            "top2_codes": query_top2,
            "margin": query_margin,
        },
        "ref": {
            "top2_codes": ref_top2,
            "margin": ref_margin,
            "ref_scores": ref_scores.float(),
        },
    }


def _collect_row_metrics(
    *,
    payload: dict[str, torch.Tensor],
    mode_payloads: dict[str, dict[str, torch.Tensor]],
    cfg,
) -> dict[str, dict[str, torch.Tensor]]:
    ref_scores = mode_payloads["ref"]["ref_scores"]
    ref_top2 = mode_payloads["ref"]["top2_codes"]
    student_top2 = mode_payloads["student"]["top2_codes"]
    o_remote_ref, h_layer_ref = _recompute_remote_outputs(payload, ref_top2, cfg)

    results: dict[str, dict[str, torch.Tensor]] = {}
    for mode_name, info in mode_payloads.items():
        top2_codes = info["top2_codes"]
        margin = info["margin"]
        overlap_student_head = _compute_overlap(top2_codes, student_top2)
        overlap_ref_head = _compute_overlap(top2_codes, ref_top2)
        ref_num = _gather_scores(ref_scores, top2_codes).sum(dim=-1)
        ref_den = _gather_scores(ref_scores, ref_top2).sum(dim=-1).clamp_min(E5A_EPS)
        ref_mass_head = ref_num / ref_den

        o_remote_mode, h_layer_mode = _recompute_remote_outputs(payload, top2_codes, cfg)
        remote_dist_row = torch.sqrt(((o_remote_mode - o_remote_ref).pow(2)).mean(dim=(1, 4)))
        final_dist_row = torch.sqrt(((h_layer_mode - h_layer_ref).pow(2)).mean(dim=(1, 4)))

        results[mode_name] = {
            "top2_codes": top2_codes,
            "margin_head": margin.float(),
            "overlap_student_head": overlap_student_head.float(),
            "overlap_ref_head": overlap_ref_head.float(),
            "ref_mass_head": ref_mass_head.float(),
            "remote_dist_row": remote_dist_row.float(),
            "final_dist_row": final_dist_row.float(),
            "overlap_student_row": overlap_student_head.float().mean(dim=1),
            "overlap_ref_row": overlap_ref_head.float().mean(dim=1),
            "ref_mass_row": ref_mass_head.float().mean(dim=1),
            "margin_row": _nanmean_tensor(margin.float(), dim=1),
        }
    return results


@dataclass
class _MetricBucket:
    total: float = 0.0
    count: int = 0

    def add_tensor(self, values: torch.Tensor):
        if values.numel() == 0:
            return
        finite = torch.isfinite(values)
        if not finite.any():
            return
        total = torch.where(finite, values, torch.zeros_like(values)).sum().item()
        count = int(finite.to(torch.int64).sum().item())
        self.total += float(total)
        self.count += count

    def mean(self) -> float:
        if self.count == 0:
            return float("nan")
        return float(self.total / self.count)


class _AggregateStore:
    def __init__(self):
        self.example_acc: dict[tuple[str, str, int], tuple[float, int]] = {}
        self.metric_buckets: dict[tuple[str, str], dict[str, _MetricBucket]] = defaultdict(
            lambda: defaultdict(_MetricBucket)
        )

    def _add_metric(self, mode: str, case_name: str, metric_name: str, values: torch.Tensor):
        self.metric_buckets[(mode, case_name)][metric_name].add_tensor(values)
        self.metric_buckets[(mode, "__total__")][metric_name].add_tensor(values)

    def update(
        self,
        *,
        mode_name: str,
        case_names: list[str],
        example_ids: torch.Tensor,
        task_mask: torch.Tensor,
        audit_mask: torch.Tensor,
        preds: torch.Tensor,
        preds_student: torch.Tensor,
        targets: torch.Tensor,
        row_metrics: dict[str, torch.Tensor],
    ) -> None:
        B, T = preds.shape
        task_mask_flat = task_mask.reshape(B, T)
        audit_mask_flat = audit_mask.reshape(B, T)
        preds_flat = preds.reshape(B, T)
        preds_student_flat = preds_student.reshape(B, T)
        targets_flat = targets.reshape(B, T)
        for batch_idx in range(B):
            case_name = case_names[batch_idx]
            example_id = int(example_ids[batch_idx].item())
            sample_task = task_mask_flat[batch_idx]
            task_count = int(sample_task.to(torch.int64).sum().item())
            if task_count == 0:
                continue
            correct = (preds_flat[batch_idx] == targets_flat[batch_idx])[sample_task].to(torch.float32)
            self.example_acc[(mode_name, case_name, example_id)] = (float(correct.sum().item()), task_count)

            pred_changed = (preds_flat[batch_idx] != preds_student_flat[batch_idx])[sample_task].to(torch.float32)
            self._add_metric(mode_name, case_name, "pred_changed_vs_student", pred_changed)

            sample_audit = audit_mask_flat[batch_idx]
            audit_values = {
                "overlap_vs_student": row_metrics["overlap_student_row"].reshape(B, T)[batch_idx][sample_audit],
                "overlap_vs_ref": row_metrics["overlap_ref_row"].reshape(B, T)[batch_idx][sample_audit],
                "ref_mass_capture": row_metrics["ref_mass_row"].reshape(B, T)[batch_idx][sample_audit],
                "top1_top2_margin": row_metrics["margin_row"].reshape(B, T)[batch_idx][sample_audit],
                "remote_dist_to_ref": row_metrics["remote_dist_row"].reshape(B, T)[batch_idx][sample_audit],
                "final_dist_to_ref": row_metrics["final_dist_row"].reshape(B, T)[batch_idx][sample_audit],
            }
            for metric_name, values in audit_values.items():
                self._add_metric(mode_name, case_name, metric_name, values.float())

    def finalize(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        example_rows = []
        for (mode, case_name, example_id), (correct_sum, row_count) in self.example_acc.items():
            example_rows.append(
                {
                    "mode": mode,
                    "mqar_case": case_name,
                    "example_idx": example_id,
                    "accuracy": correct_sum / max(row_count, 1),
                }
            )
        example_df = pd.DataFrame(example_rows)
        if example_df.empty:
            raise RuntimeError("E5A 没有收集到任何 example accuracy 记录.")

        acc_total = (
            example_df.groupby("mode", as_index=False)["accuracy"].mean().rename(columns={"accuracy": "acc_total"})
        )
        acc_case = (
            example_df.groupby(["mode", "mqar_case"], as_index=False)["accuracy"]
            .mean()
            .rename(columns={"accuracy": "acc_case"})
        )

        case_rows: list[dict[str, Any]] = []
        aggregate_rows: list[dict[str, Any]] = []
        for mode in E5A_MODES:
            total_metrics = {"mode": mode}
            total_row = acc_total.loc[acc_total["mode"] == mode]
            total_metrics["acc_total"] = float(total_row["acc_total"].iloc[0]) if not total_row.empty else float("nan")
            for case_name in E5A_CASES:
                case_row = {"mode": mode, "mqar_case": case_name}
                case_acc = acc_case.loc[(acc_case["mode"] == mode) & (acc_case["mqar_case"] == case_name)]
                case_row["acc_case"] = float(case_acc["acc_case"].iloc[0]) if not case_acc.empty else float("nan")
                bucket_map = self.metric_buckets.get((mode, case_name), {})
                for metric_name in (
                    "overlap_vs_student",
                    "overlap_vs_ref",
                    "ref_mass_capture",
                    "top1_top2_margin",
                    "remote_dist_to_ref",
                    "final_dist_to_ref",
                    "pred_changed_vs_student",
                ):
                    case_row[metric_name] = bucket_map.get(metric_name, _MetricBucket()).mean()
                case_rows.append(case_row)
                total_metrics[f"acc_{case_name}"] = case_row["acc_case"]

            total_bucket = self.metric_buckets.get((mode, "__total__"), {})
            for metric_name in (
                "overlap_vs_student",
                "overlap_vs_ref",
                "ref_mass_capture",
                "top1_top2_margin",
                "remote_dist_to_ref",
                "final_dist_to_ref",
                "pred_changed_vs_student",
            ):
                total_metrics[metric_name] = total_bucket.get(metric_name, _MetricBucket()).mean()
            aggregate_rows.append(total_metrics)

        return pd.DataFrame(aggregate_rows), pd.DataFrame(case_rows)


class _RowAuditWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None

    def write(
        self,
        *,
        mode_name: str,
        case_names: list[str],
        example_ids: torch.Tensor,
        targets: torch.Tensor,
        preds: torch.Tensor,
        preds_student: torch.Tensor,
        task_mask: torch.Tensor,
        audit_mask: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        student_top2: torch.Tensor,
        mode_top2: torch.Tensor,
        ref_top2: torch.Tensor,
    ) -> None:
        B, T = preds.shape
        H = student_top2.size(1)
        N = student_top2.size(2)
        L = student_top2.size(3)
        task_mask_blk = task_mask.reshape(B, N, L)
        task_pos = task_mask_blk.nonzero(as_tuple=False)
        if task_pos.numel() == 0:
            return

        device = task_pos.device
        num_task = int(task_pos.size(0))
        b_idx = task_pos[:, 0].repeat_interleave(H)
        n_idx = task_pos[:, 1].repeat_interleave(H)
        l_idx = task_pos[:, 2].repeat_interleave(H)
        h_idx = torch.arange(H, device=device, dtype=torch.int64).repeat(num_task)
        row_idx = n_idx * L + l_idx

        pred_values = preds[b_idx, row_idx]
        pred_student_values = preds_student[b_idx, row_idx]
        target_values = targets[b_idx, row_idx]
        audit_values = audit_mask.reshape(B, T)[b_idx, row_idx]
        remote_dist = metrics["remote_dist_row"].reshape(B, T)[b_idx, row_idx]
        final_dist = metrics["final_dist_row"].reshape(B, T)[b_idx, row_idx]
        pred_changed = (pred_values != pred_student_values).to(torch.float32)

        payload = {
            "mode": np.array([mode_name] * int(b_idx.numel()), dtype=object),
            "mqar_case": np.array([case_names[int(i)] for i in b_idx.cpu().tolist()], dtype=object),
            "example_idx": example_ids[b_idx].cpu().numpy().astype(np.int64),
            "head_idx": h_idx.cpu().numpy().astype(np.int16),
            "row_idx": row_idx.cpu().numpy().astype(np.int32),
            "target": target_values.cpu().numpy().astype(np.int64),
            "pred": pred_values.cpu().numpy().astype(np.int64),
            "pred_student": pred_student_values.cpu().numpy().astype(np.int64),
            "is_task_row": np.ones(int(b_idx.numel()), dtype=bool),
            "is_audit_row": audit_values.cpu().numpy().astype(bool),
            "student_top2_0": student_top2[b_idx, h_idx, n_idx, l_idx, 0].cpu().numpy().astype(np.int64),
            "student_top2_1": student_top2[b_idx, h_idx, n_idx, l_idx, 1].cpu().numpy().astype(np.int64),
            "mode_top2_0": mode_top2[b_idx, h_idx, n_idx, l_idx, 0].cpu().numpy().astype(np.int64),
            "mode_top2_1": mode_top2[b_idx, h_idx, n_idx, l_idx, 1].cpu().numpy().astype(np.int64),
            "ref_top2_0": ref_top2[b_idx, h_idx, n_idx, l_idx, 0].cpu().numpy().astype(np.int64),
            "ref_top2_1": ref_top2[b_idx, h_idx, n_idx, l_idx, 1].cpu().numpy().astype(np.int64),
            "overlap_vs_student": metrics["overlap_student_head"][b_idx, h_idx, n_idx, l_idx].cpu().numpy().astype(np.float32),
            "overlap_vs_ref": metrics["overlap_ref_head"][b_idx, h_idx, n_idx, l_idx].cpu().numpy().astype(np.float32),
            "ref_mass_capture": metrics["ref_mass_head"][b_idx, h_idx, n_idx, l_idx].cpu().numpy().astype(np.float32),
            "top1_top2_margin": metrics["margin_head"][b_idx, h_idx, n_idx, l_idx].cpu().numpy().astype(np.float32),
            "remote_dist_to_ref": remote_dist.cpu().numpy().astype(np.float32),
            "final_dist_to_ref": final_dist.cpu().numpy().astype(np.float32),
            "pred_changed_vs_student": pred_changed.cpu().numpy().astype(np.float32),
        }
        table = pa.Table.from_pydict(payload)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _collect_student_payload(model, mixer, inputs: torch.Tensor):
    payloads: list[dict[str, Any]] = []

    def sink(payload: dict[str, Any]) -> None:
        payloads.append(payload)

    runtime = {
        "enabled": True,
        "capture_tensors": True,
        "mode_name": "student",
        "override_top2_codes": None,
        "capture_sink": sink,
    }
    with torch.inference_mode(), _audit_runtime_scope(mixer, runtime):
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
    if len(payloads) != 1:
        raise RuntimeError(f"E5A student forward 期望捕获 1 份 payload, 当前收到 {len(payloads)} 份.")
    return preds, payloads[0]


def _collect_mode_preds(model, mixer, inputs: torch.Tensor, mode_name: str, top2_codes: torch.Tensor):
    runtime = {
        "enabled": True,
        "capture_tensors": False,
        "mode_name": mode_name,
        "override_top2_codes": top2_codes,
        "capture_sink": None,
    }
    with torch.inference_mode(), _audit_runtime_scope(mixer, runtime):
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
    return preds


def _build_final_ruling(aggregate_df: pd.DataFrame, aggregate_by_case_df: pd.DataFrame) -> dict[str, Any]:
    student = aggregate_df.loc[aggregate_df["mode"] == "student"].iloc[0]
    ref = aggregate_df.loc[aggregate_df["mode"] == "ref"].iloc[0]
    ref_case = aggregate_by_case_df.loc[aggregate_by_case_df["mode"] == "ref"].set_index("mqar_case")
    student_case = aggregate_by_case_df.loc[aggregate_by_case_df["mode"] == "student"].set_index("mqar_case")
    case_gains = {
        case_name: float(ref_case.loc[case_name, "acc_case"] - student_case.loc[case_name, "acc_case"])
        for case_name in E5A_CASES
    }
    avg_gain = float(np.mean(list(case_gains.values())))
    student_gap_flag = bool(
        (float(student["overlap_vs_ref"]) <= 0.60) or (float(student["ref_mass_capture"]) <= 0.75)
    )
    student_close_flag = bool(
        (float(student["overlap_vs_ref"]) >= 0.80) and (float(student["ref_mass_capture"]) >= 0.90)
    )
    ref_gain_flag = bool(
        avg_gain >= 0.01
        and min(case_gains.values()) >= 0.005
        and max(0.0, -(min(case_gains.values()))) <= 0.003
    )
    ref_limited_flag = bool(avg_gain < 0.003 and max(case_gains.values()) < 0.005)

    if student_gap_flag and ref_gain_flag:
        final_ruling = "selection_bottleneck"
    elif student_close_flag and ref_limited_flag:
        final_ruling = "in_code_approx_bottleneck"
    else:
        final_ruling = "mixed_or_inconclusive"
    return {
        "student_gap_flag": student_gap_flag,
        "ref_gain_flag": ref_gain_flag,
        "student_close_flag": student_close_flag,
        "ref_limited_flag": ref_limited_flag,
        "avg_acc_gain_ref_minus_student": avg_gain,
        "acc_gain_by_case": case_gains,
        "final_ruling": final_ruling,
        "ref_acc_total": float(ref["acc_total"]),
        "student_acc_total": float(student["acc_total"]),
    }


def run_e5a_audit(
    *,
    bundle: dict[str, Any],
    test_dataloader: DataLoader,
    checkpoint_launch_id: str,
    checkpoint_run_id: str,
    checkpoint_path: Path,
    eval_launch_id: str,
    eval_run_id: str,
    output_dir: Path,
    logger=None,
) -> dict[str, Any]:
    model = bundle["model"]
    _, mixer = _find_audit_flash_mixer(model)
    config_summary = _assert_supported_mainline(mixer)

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_store = _AggregateStore()
    row_writer = _RowAuditWriter(output_dir / "row_audit.parquet")

    example_offset = 0
    try:
        for inputs, targets, slices in test_dataloader:
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(inputs.device)
            batch_size = int(inputs.size(0))
            case_names = _flatten_case_name(slices)
            example_ids = torch.arange(example_offset, example_offset + batch_size, device=inputs.device, dtype=torch.int64)

            preds_student, payload = _collect_student_payload(model, mixer, inputs)
            payload = {str(k): v for k, v in payload.items() if isinstance(v, torch.Tensor)}

            mode_payloads = _compute_mode_payload(payload)
            preds_by_mode = {"student": preds_student}
            for mode_name in E5A_MODES[1:]:
                preds_by_mode[mode_name] = _collect_mode_preds(
                    model=model,
                    mixer=mixer,
                    inputs=inputs,
                    mode_name=mode_name,
                    top2_codes=mode_payloads[mode_name]["top2_codes"],
                )

            row_metrics_by_mode = _collect_row_metrics(payload=payload, mode_payloads=mode_payloads, cfg=mixer.attn.config)

            B, T = targets.shape
            N = payload["Q_blk"].size(2)
            L = payload["Q_blk"].size(3)
            task_mask = (targets != -100).reshape(B, N, L)
            audit_mask = task_mask & (payload["hist_end"].view(1, N, 1) > 0)

            for mode_name in E5A_MODES:
                preds = preds_by_mode[mode_name]
                metrics = row_metrics_by_mode[mode_name]
                aggregate_store.update(
                    mode_name=mode_name,
                    case_names=case_names,
                    example_ids=example_ids,
                    task_mask=task_mask,
                    audit_mask=audit_mask,
                    preds=preds,
                    preds_student=preds_student,
                    targets=targets,
                    row_metrics=metrics,
                )
                row_writer.write(
                    mode_name=mode_name,
                    case_names=case_names,
                    example_ids=example_ids,
                    targets=targets,
                    preds=preds,
                    preds_student=preds_student,
                    task_mask=task_mask,
                    audit_mask=audit_mask,
                    metrics=metrics,
                    student_top2=mode_payloads["student"]["top2_codes"],
                    mode_top2=mode_payloads[mode_name]["top2_codes"],
                    ref_top2=mode_payloads["ref"]["top2_codes"],
                )

            example_offset += batch_size

        aggregate_df, aggregate_by_case_df = aggregate_store.finalize()
        aggregate_df.insert(0, "eval_run_id", eval_run_id)
        aggregate_df.insert(0, "eval_launch_id", eval_launch_id)
        aggregate_df.insert(0, "checkpoint_run_id", checkpoint_run_id)
        aggregate_df.insert(0, "checkpoint_launch_id", checkpoint_launch_id)
        aggregate_df["audit_layer_idx"] = E5A_AUDIT_LAYER_IDX

        aggregate_by_case_df.insert(0, "eval_run_id", eval_run_id)
        aggregate_by_case_df.insert(0, "eval_launch_id", eval_launch_id)
        aggregate_by_case_df.insert(0, "checkpoint_run_id", checkpoint_run_id)
        aggregate_by_case_df.insert(0, "checkpoint_launch_id", checkpoint_launch_id)
        aggregate_by_case_df["audit_layer_idx"] = E5A_AUDIT_LAYER_IDX

        aggregate_df.to_csv(output_dir / "aggregate.csv", index=False)
        aggregate_by_case_df.to_csv(output_dir / "aggregate_by_case.csv", index=False)

        final_ruling = _build_final_ruling(aggregate_df, aggregate_by_case_df)
        summary = {
            "checkpoint_launch_id": checkpoint_launch_id,
            "checkpoint_run_id": checkpoint_run_id,
            "best_checkpoint": str(checkpoint_path),
            "eval_launch_id": eval_launch_id,
            "eval_run_id": eval_run_id,
            "audit_layer_idx": E5A_AUDIT_LAYER_IDX,
            "cases": list(E5A_CASES),
            "modes": list(E5A_MODES),
            "output_dir": str(output_dir.resolve()),
            "row_audit_path": str((output_dir / "row_audit.parquet").resolve()),
            "config_summary": config_summary,
            "aggregate": json.loads(aggregate_df.to_json(orient="records")),
            "aggregate_by_case": json.loads(aggregate_by_case_df.to_json(orient="records")),
            "final_ruling": final_ruling,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if logger is not None:
            log_metrics = {}
            for _, row in aggregate_df.iterrows():
                mode_name = str(row["mode"])
                log_metrics[f"e5a/{mode_name}/acc_total"] = float(row["acc_total"])
                log_metrics[f"e5a/{mode_name}/overlap_vs_ref"] = float(row["overlap_vs_ref"])
                log_metrics[f"e5a/{mode_name}/ref_mass_capture"] = float(row["ref_mass_capture"])
            logger.log(log_metrics, step=0)

        return summary
    finally:
        row_writer.close()
