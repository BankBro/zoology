from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


E5A_SCRIPT_PATH = Path(__file__).with_name("e5a_audit.py")
E5B_AUDIT_LAYER_IDX = 1
E5B_CASES = ("512x128", "1024x256")
E5B_QUANTILES = (("q50", 0.50), ("q75", 0.75), ("q90", 0.90))
E5B_BASE_VARIANTS = (
    "student",
    "dense_value_full",
    "dense_value_row_all",
    "dense_value_conf_q50",
    "dense_value_conf_q75",
    "dense_value_conf_q90",
)
E5B_AGG_METRICS = (
    "override_row_coverage",
    "override_head_coverage",
    "correct_to_wrong",
    "wrong_to_correct",
    "net_fix",
    "pred_changed_vs_student",
    "final_margin_mean",
    "final_margin_changed_mean",
    "student_final_margin_changed_mean",
)
E5B_EXPECTED_FULL_ACC_TOTAL = {
    "dense-t025-s123-d123": 0.895859,
    "dense-t025-s124-d123": 0.915402,
}
E5B_EPS = 1e-12
_E5A_MODULE = None


def _load_e5a_module():
    global _E5A_MODULE
    if _E5A_MODULE is not None:
        return _E5A_MODULE

    cached = sys.modules.get("flash_vqg_e5a_audit")
    if cached is not None:
        _E5A_MODULE = cached
        return _E5A_MODULE

    module_name = "flash_vqg_e5a_audit_e5b_shared"
    if not E5A_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"未找到 E5A 审计脚本: {E5A_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(module_name, E5A_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {E5A_SCRIPT_PATH} 加载 E5A 审计模块.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _E5A_MODULE = module
    return _E5A_MODULE


def _tensor_payload_only(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {str(k): v for k, v in payload.items() if isinstance(v, torch.Tensor)}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    e5a = _load_e5a_module()
    masked = torch.where(mask, values, torch.full_like(values, float("nan")))
    return e5a._nanmean_tensor(masked, dim=dim)


def _compute_preds_and_margin(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    preds = logits.argmax(dim=-1)
    top_k = min(2, int(logits.size(-1)))
    if top_k < 2:
        margin = torch.full(preds.shape, float("nan"), dtype=torch.float32, device=logits.device)
    else:
        topv = torch.topk(logits, k=top_k, dim=-1).values
        margin = (topv[..., 0] - topv[..., 1]).float()
    return preds, margin


def _collect_student_forward(model, mixer, inputs: torch.Tensor):
    e5a = _load_e5a_module()
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
    with torch.inference_mode(), e5a._audit_runtime_scope(mixer, runtime):
        logits = model(inputs)
    preds, final_margin = _compute_preds_and_margin(logits)
    if len(payloads) != 1:
        raise RuntimeError(f"E5B student forward 期望捕获 1 份 payload, 当前收到 {len(payloads)} 份.")
    return logits, preds, final_margin, payloads[0]


def _collect_override_forward(
    model,
    mixer,
    inputs: torch.Tensor,
    *,
    variant_name: str,
    override_top2_codes: torch.Tensor,
):
    e5a = _load_e5a_module()
    runtime = {
        "enabled": True,
        "capture_tensors": False,
        "mode_name": variant_name,
        "override_top2_codes": override_top2_codes,
        "capture_sink": None,
    }
    with torch.inference_mode(), e5a._audit_runtime_scope(mixer, runtime):
        logits = model(inputs)
    preds, final_margin = _compute_preds_and_margin(logits)
    return logits, preds, final_margin


def _compute_dense_value_metadata(
    *,
    payload: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    e5a = _load_e5a_module()
    student_top2 = payload["student_top2"].to(torch.int64)
    hist_has_remote = payload["hist_end"].view(1, 1, -1, 1) > 0
    hist_has_remote = hist_has_remote.expand(
        student_top2.size(0),
        student_top2.size(1),
        -1,
        student_top2.size(3),
    )

    _, dense_value_scores = e5a._compute_dense_teacher_scores(payload)
    dense_value_top2, dense_value_margin = e5a._top2_from_scores(
        dense_value_scores,
        fallback_top2=student_top2,
        hist_has_remote=hist_has_remote,
    )

    overlap_vs_student_head = e5a._compute_overlap(dense_value_top2, student_top2)
    disagreement_head = hist_has_remote & (overlap_vs_student_head < 1.0)
    teacher_adv_head = (
        e5a._gather_scores(dense_value_scores, dense_value_top2).sum(dim=-1)
        - e5a._gather_scores(dense_value_scores, student_top2).sum(dim=-1)
    ).float()
    teacher_margin_head = dense_value_margin.float()
    teacher_adv_row = _masked_mean(teacher_adv_head, disagreement_head, dim=1)
    teacher_margin_row = _masked_mean(teacher_margin_head, disagreement_head, dim=1)
    row_has_disagreement = disagreement_head.any(dim=1)

    B = targets.size(0)
    N = payload["Q_blk"].size(2)
    L = payload["Q_blk"].size(3)
    task_mask = (targets != -100).reshape(B, N, L)
    audit_mask = task_mask & hist_has_remote[:, 0]
    return {
        "student_top2": student_top2,
        "teacher_top2": dense_value_top2.to(torch.int64),
        "dense_value_scores": dense_value_scores.float(),
        "hist_has_remote": hist_has_remote,
        "task_mask": task_mask,
        "audit_mask": audit_mask,
        "overlap_vs_student_head": overlap_vs_student_head.float(),
        "disagreement_head": disagreement_head,
        "row_has_disagreement": row_has_disagreement,
        "teacher_adv_head": teacher_adv_head,
        "teacher_adv_row": teacher_adv_row.float(),
        "teacher_margin_head": teacher_margin_head.float(),
        "teacher_margin_row": teacher_margin_row.float(),
    }


class _QuantileCollector:
    def __init__(self):
        self.values_by_case: dict[str, list[float]] = defaultdict(list)

    def update(
        self,
        *,
        case_names: list[str],
        task_mask: torch.Tensor,
        row_has_disagreement: torch.Tensor,
        teacher_adv_row: torch.Tensor,
    ) -> None:
        batch_size = len(case_names)
        for batch_idx in range(batch_size):
            case_name = str(case_names[batch_idx])
            valid_mask = (
                task_mask[batch_idx]
                & row_has_disagreement[batch_idx]
                & torch.isfinite(teacher_adv_row[batch_idx])
            )
            if not valid_mask.any():
                continue
            values = teacher_adv_row[batch_idx][valid_mask].detach().cpu().numpy().astype(np.float32)
            self.values_by_case[case_name].extend(values.tolist())

    def finalize(self) -> dict[str, dict[str, float | int]]:
        thresholds: dict[str, dict[str, float | int]] = {}
        for case_name in E5B_CASES:
            values = np.asarray(self.values_by_case.get(case_name, []), dtype=np.float32)
            case_summary: dict[str, float | int] = {"count": int(values.size)}
            if values.size == 0:
                for quantile_name, _ in E5B_QUANTILES:
                    case_summary[quantile_name] = float("inf")
            else:
                for quantile_name, quantile_value in E5B_QUANTILES:
                    case_summary[quantile_name] = float(np.quantile(values, quantile_value))
            thresholds[case_name] = case_summary
        return thresholds


def _quantile_threshold_tensor(
    case_names: list[str],
    quantile_thresholds: dict[str, dict[str, float | int]],
    *,
    quantile_name: str,
    device: torch.device,
) -> torch.Tensor:
    values = [
        float(quantile_thresholds.get(str(case_name), {}).get(quantile_name, float("inf")))
        for case_name in case_names
    ]
    return torch.tensor(values, dtype=torch.float32, device=device).view(-1, 1, 1)


@dataclass
class _VariantPayload:
    name: str
    top2_codes: torch.Tensor
    override_applied_head: torch.Tensor
    row_override_applied: torch.Tensor


def _build_variant_payloads(
    *,
    metadata: dict[str, torch.Tensor],
    case_names: list[str],
    quantile_thresholds: dict[str, dict[str, float | int]],
) -> list[_VariantPayload]:
    student_top2 = metadata["student_top2"]
    teacher_top2 = metadata["teacher_top2"]
    disagreement_head = metadata["disagreement_head"]
    row_has_disagreement = metadata["row_has_disagreement"]
    teacher_adv_row = metadata["teacher_adv_row"]

    B, H, _, _, _ = student_top2.shape
    zero_mask = torch.zeros_like(disagreement_head, dtype=torch.bool)

    def make_variant(name: str, override_applied_head: torch.Tensor) -> _VariantPayload:
        override_applied_head = override_applied_head.to(torch.bool)
        top2_codes = torch.where(
            override_applied_head.unsqueeze(-1),
            teacher_top2,
            student_top2,
        )
        return _VariantPayload(
            name=name,
            top2_codes=top2_codes.to(torch.int64),
            override_applied_head=override_applied_head,
            row_override_applied=override_applied_head.any(dim=1),
        )

    variants = [make_variant("student", zero_mask)]
    variants.append(make_variant("dense_value_full", disagreement_head))

    row_all_head_mask = row_has_disagreement.unsqueeze(1).expand(-1, H, -1, -1).clone()
    variants.append(make_variant("dense_value_row_all", row_all_head_mask))

    for quantile_name, _ in E5B_QUANTILES:
        threshold = _quantile_threshold_tensor(
            case_names,
            quantile_thresholds,
            quantile_name=quantile_name,
            device=teacher_adv_row.device,
        )
        row_mask = (
            row_has_disagreement
            & torch.isfinite(teacher_adv_row)
            & (teacher_adv_row >= threshold)
        )
        head_mask = row_mask.unsqueeze(1).expand(-1, H, -1, -1).clone()
        variants.append(make_variant(f"dense_value_conf_{quantile_name}", head_mask))

    for head_idx in range(H):
        head_mask = torch.zeros_like(disagreement_head, dtype=torch.bool)
        head_mask[:, head_idx] = disagreement_head[:, head_idx]
        variants.append(make_variant(f"dense_value_head_h{head_idx}", head_mask))

    return variants


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
    def __init__(self, variant_names: list[str]):
        self.variant_names = list(variant_names)
        self.example_acc: dict[tuple[str, str, int], tuple[float, int]] = {}
        self.metric_buckets: dict[tuple[str, str], dict[str, _MetricBucket]] = defaultdict(
            lambda: defaultdict(_MetricBucket)
        )

    def _add_metric(self, variant_name: str, case_name: str, metric_name: str, values: torch.Tensor):
        values = values.float()
        self.metric_buckets[(variant_name, case_name)][metric_name].add_tensor(values)
        self.metric_buckets[(variant_name, "__total__")][metric_name].add_tensor(values)

    def update(
        self,
        *,
        variant_name: str,
        case_names: list[str],
        example_ids: torch.Tensor,
        task_mask: torch.Tensor,
        preds: torch.Tensor,
        preds_student: torch.Tensor,
        targets: torch.Tensor,
        final_margin: torch.Tensor,
        final_margin_student: torch.Tensor,
        row_override_applied: torch.Tensor,
        override_applied_head: torch.Tensor,
    ) -> None:
        B, T = preds.shape
        task_mask_flat = task_mask.reshape(B, T)
        preds_flat = preds.reshape(B, T)
        preds_student_flat = preds_student.reshape(B, T)
        targets_flat = targets.reshape(B, T)
        final_margin_flat = final_margin.reshape(B, T)
        final_margin_student_flat = final_margin_student.reshape(B, T)
        row_override_flat = row_override_applied.reshape(B, T)

        task_head_mask = task_mask.unsqueeze(1).expand_as(override_applied_head)
        for batch_idx in range(B):
            case_name = str(case_names[batch_idx])
            example_id = int(example_ids[batch_idx].item())
            sample_task = task_mask_flat[batch_idx]
            task_count = int(sample_task.to(torch.int64).sum().item())
            if task_count == 0:
                continue

            pred_values = preds_flat[batch_idx]
            pred_student_values = preds_student_flat[batch_idx]
            target_values = targets_flat[batch_idx]

            correct = (pred_values == target_values)[sample_task].to(torch.float32)
            self.example_acc[(variant_name, case_name, example_id)] = (float(correct.sum().item()), task_count)

            pred_changed = (pred_values != pred_student_values)[sample_task]
            correct_to_wrong = (
                (pred_student_values == target_values) & (pred_values != target_values)
            )[sample_task]
            wrong_to_correct = (
                (pred_student_values != target_values) & (pred_values == target_values)
            )[sample_task]

            self._add_metric(
                variant_name,
                case_name,
                "override_row_coverage",
                row_override_flat[batch_idx][sample_task].to(torch.float32),
            )
            self._add_metric(
                variant_name,
                case_name,
                "override_head_coverage",
                override_applied_head[batch_idx][task_head_mask[batch_idx]].to(torch.float32),
            )
            self._add_metric(
                variant_name,
                case_name,
                "correct_to_wrong",
                correct_to_wrong.to(torch.float32),
            )
            self._add_metric(
                variant_name,
                case_name,
                "wrong_to_correct",
                wrong_to_correct.to(torch.float32),
            )
            self._add_metric(
                variant_name,
                case_name,
                "net_fix",
                wrong_to_correct.to(torch.float32) - correct_to_wrong.to(torch.float32),
            )
            self._add_metric(
                variant_name,
                case_name,
                "pred_changed_vs_student",
                pred_changed.to(torch.float32),
            )
            self._add_metric(
                variant_name,
                case_name,
                "final_margin_mean",
                final_margin_flat[batch_idx][sample_task].float(),
            )

            changed_task_mask = sample_task & (pred_values != pred_student_values)
            self._add_metric(
                variant_name,
                case_name,
                "final_margin_changed_mean",
                final_margin_flat[batch_idx][changed_task_mask].float(),
            )
            self._add_metric(
                variant_name,
                case_name,
                "student_final_margin_changed_mean",
                final_margin_student_flat[batch_idx][changed_task_mask].float(),
            )

    def finalize(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        example_rows = []
        for (variant_name, case_name, example_id), (correct_sum, row_count) in self.example_acc.items():
            example_rows.append(
                {
                    "variant": variant_name,
                    "mqar_case": case_name,
                    "example_idx": example_id,
                    "accuracy": correct_sum / max(row_count, 1),
                }
            )
        example_df = pd.DataFrame(example_rows)
        if example_df.empty:
            raise RuntimeError("E5B 没有收集到任何 example accuracy 记录.")

        acc_total = (
            example_df.groupby("variant", as_index=False)["accuracy"]
            .mean()
            .rename(columns={"accuracy": "acc_total"})
        )
        acc_case = (
            example_df.groupby(["variant", "mqar_case"], as_index=False)["accuracy"]
            .mean()
            .rename(columns={"accuracy": "acc_case"})
        )

        aggregate_rows: list[dict[str, Any]] = []
        case_rows: list[dict[str, Any]] = []
        for variant_name in self.variant_names:
            total_row = {"variant": variant_name}
            total_acc_row = acc_total.loc[acc_total["variant"] == variant_name]
            total_row["acc_total"] = (
                float(total_acc_row["acc_total"].iloc[0]) if not total_acc_row.empty else float("nan")
            )
            for case_name in E5B_CASES:
                case_acc_row = acc_case.loc[
                    (acc_case["variant"] == variant_name) & (acc_case["mqar_case"] == case_name)
                ]
                case_acc = float(case_acc_row["acc_case"].iloc[0]) if not case_acc_row.empty else float("nan")
                total_row[f"acc_{case_name}"] = case_acc
            total_bucket = self.metric_buckets.get((variant_name, "__total__"), {})
            for metric_name in E5B_AGG_METRICS:
                total_row[metric_name] = total_bucket.get(metric_name, _MetricBucket()).mean()
            aggregate_rows.append(total_row)

            for case_name in E5B_CASES:
                case_row = {"variant": variant_name, "mqar_case": case_name}
                case_acc_row = acc_case.loc[
                    (acc_case["variant"] == variant_name) & (acc_case["mqar_case"] == case_name)
                ]
                case_acc = float(case_acc_row["acc_case"].iloc[0]) if not case_acc_row.empty else float("nan")
                case_row["acc_total"] = case_acc
                for case_name_col in E5B_CASES:
                    case_row[f"acc_{case_name_col}"] = case_acc if case_name_col == case_name else float("nan")
                case_bucket = self.metric_buckets.get((variant_name, case_name), {})
                for metric_name in E5B_AGG_METRICS:
                    case_row[metric_name] = case_bucket.get(metric_name, _MetricBucket()).mean()
                case_rows.append(case_row)

        return pd.DataFrame(aggregate_rows), pd.DataFrame(case_rows)


class _ParquetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer: Any | None = None

    def write_payload(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        first_key = next(iter(payload))
        if len(payload[first_key]) == 0:
            return
        e5a = _load_e5a_module()
        pa, pq = e5a._load_pyarrow()
        table = pa.Table.from_pydict(payload)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _derive_changed_type(
    pred_values: torch.Tensor,
    pred_student_values: torch.Tensor,
    target_values: torch.Tensor,
) -> np.ndarray:
    pred_np = pred_values.cpu().numpy().astype(np.int64)
    student_np = pred_student_values.cpu().numpy().astype(np.int64)
    target_np = target_values.cpu().numpy().astype(np.int64)

    changed_type = np.full(pred_np.shape[0], "wrong_to_wrong", dtype=object)
    unchanged = pred_np == student_np
    correct_to_wrong = (student_np == target_np) & (pred_np != target_np)
    wrong_to_correct = (student_np != target_np) & (pred_np == target_np)

    changed_type[unchanged] = "unchanged"
    changed_type[correct_to_wrong] = "correct_to_wrong"
    changed_type[wrong_to_correct] = "wrong_to_correct"
    return changed_type


class _RowPartialAuditWriter(_ParquetWriter):
    def write(
        self,
        *,
        variant_name: str,
        case_names: list[str],
        example_ids: torch.Tensor,
        targets: torch.Tensor,
        preds: torch.Tensor,
        preds_student: torch.Tensor,
        final_margin: torch.Tensor,
        final_margin_student: torch.Tensor,
        metadata: dict[str, torch.Tensor],
        variant: _VariantPayload,
    ) -> None:
        task_mask = metadata["task_mask"]
        task_pos = task_mask.nonzero(as_tuple=False)
        if task_pos.numel() == 0:
            return

        B, N, L = task_mask.shape
        del B
        b_idx = task_pos[:, 0]
        n_idx = task_pos[:, 1]
        l_idx = task_pos[:, 2]
        row_idx = n_idx * L + l_idx
        pred_values = preds[b_idx, row_idx]
        pred_student_values = preds_student[b_idx, row_idx]
        target_values = targets[b_idx, row_idx]
        num_overridden_heads = variant.override_applied_head.to(torch.int64).sum(dim=1)

        payload = {
            "variant": np.array([variant_name] * int(b_idx.numel()), dtype=object),
            "mqar_case": np.array([case_names[int(i)] for i in b_idx.cpu().tolist()], dtype=object),
            "example_idx": example_ids[b_idx].cpu().numpy().astype(np.int64),
            "row_idx": row_idx.cpu().numpy().astype(np.int32),
            "target": target_values.cpu().numpy().astype(np.int64),
            "pred": pred_values.cpu().numpy().astype(np.int64),
            "pred_student": pred_student_values.cpu().numpy().astype(np.int64),
            "is_task_row": np.ones(int(b_idx.numel()), dtype=bool),
            "is_audit_row": metadata["audit_mask"][b_idx, n_idx, l_idx].cpu().numpy().astype(bool),
            "row_has_disagreement": metadata["row_has_disagreement"][b_idx, n_idx, l_idx]
            .cpu()
            .numpy()
            .astype(bool),
            "row_override_applied": variant.row_override_applied[b_idx, n_idx, l_idx].cpu().numpy().astype(bool),
            "num_overridden_heads": num_overridden_heads[b_idx, n_idx, l_idx].cpu().numpy().astype(np.int16),
            "teacher_adv_row": metadata["teacher_adv_row"][b_idx, n_idx, l_idx].cpu().numpy().astype(np.float32),
            "teacher_margin_row": metadata["teacher_margin_row"][b_idx, n_idx, l_idx]
            .cpu()
            .numpy()
            .astype(np.float32),
            "final_margin": final_margin[b_idx, row_idx].cpu().numpy().astype(np.float32),
            "final_margin_student": final_margin_student[b_idx, row_idx].cpu().numpy().astype(np.float32),
            "changed_type": _derive_changed_type(pred_values, pred_student_values, target_values),
        }
        self.write_payload(payload)


class _HeadPartialAuditWriter(_ParquetWriter):
    def write(
        self,
        *,
        variant_name: str,
        case_names: list[str],
        example_ids: torch.Tensor,
        metadata: dict[str, torch.Tensor],
        variant: _VariantPayload,
    ) -> None:
        task_mask = metadata["task_mask"]
        task_pos = task_mask.nonzero(as_tuple=False)
        if task_pos.numel() == 0:
            return

        H = metadata["student_top2"].size(1)
        L = task_mask.size(2)
        num_task = int(task_pos.size(0))
        device = task_pos.device

        b_idx = task_pos[:, 0].repeat_interleave(H)
        n_idx = task_pos[:, 1].repeat_interleave(H)
        l_idx = task_pos[:, 2].repeat_interleave(H)
        h_idx = torch.arange(H, device=device, dtype=torch.int64).repeat(num_task)
        row_idx = n_idx * L + l_idx

        payload = {
            "variant": np.array([variant_name] * int(b_idx.numel()), dtype=object),
            "mqar_case": np.array([case_names[int(i)] for i in b_idx.cpu().tolist()], dtype=object),
            "example_idx": example_ids[b_idx].cpu().numpy().astype(np.int64),
            "row_idx": row_idx.cpu().numpy().astype(np.int32),
            "head_idx": h_idx.cpu().numpy().astype(np.int16),
            "override_applied_head": variant.override_applied_head[b_idx, h_idx, n_idx, l_idx]
            .cpu()
            .numpy()
            .astype(bool),
            "teacher_adv_head": metadata["teacher_adv_head"][b_idx, h_idx, n_idx, l_idx]
            .cpu()
            .numpy()
            .astype(np.float32),
            "teacher_margin_head": metadata["teacher_margin_head"][b_idx, h_idx, n_idx, l_idx]
            .cpu()
            .numpy()
            .astype(np.float32),
            "overlap_vs_student": metadata["overlap_vs_student_head"][b_idx, h_idx, n_idx, l_idx]
            .cpu()
            .numpy()
            .astype(np.float32),
            "teacher_top2_0": metadata["teacher_top2"][b_idx, h_idx, n_idx, l_idx, 0]
            .cpu()
            .numpy()
            .astype(np.int64),
            "teacher_top2_1": metadata["teacher_top2"][b_idx, h_idx, n_idx, l_idx, 1]
            .cpu()
            .numpy()
            .astype(np.int64),
            "student_top2_0": metadata["student_top2"][b_idx, h_idx, n_idx, l_idx, 0]
            .cpu()
            .numpy()
            .astype(np.int64),
            "student_top2_1": metadata["student_top2"][b_idx, h_idx, n_idx, l_idx, 1]
            .cpu()
            .numpy()
            .astype(np.int64),
        }
        self.write_payload(payload)


def _select_best_variant(aggregate_df: pd.DataFrame) -> str:
    ranked = aggregate_df.copy()
    ranked = ranked.sort_values(
        by=["acc_total", "acc_1024x256", "override_row_coverage"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["variant"])


def _build_run_conclusion(aggregate_df: pd.DataFrame) -> dict[str, Any]:
    df = aggregate_df.set_index("variant")
    student = df.loc["student"]
    full = df.loc["dense_value_full"]
    partial_df = aggregate_df.loc[
        ~aggregate_df["variant"].isin(["student", "dense_value_full"])
    ].copy()
    head_df = partial_df.loc[partial_df["variant"].str.startswith("dense_value_head_h")]
    frontier_df = partial_df.loc[partial_df["variant"].str.startswith("dense_value_conf_q")]

    positive_mask = (
        (partial_df["acc_total"] > float(student["acc_total"]))
        & (partial_df["acc_1024x256"] > float(student["acc_1024x256"]))
        & (partial_df["net_fix"] > 0)
    )
    head_positive_mask = (
        (head_df["acc_total"] > float(student["acc_total"]))
        & (head_df["acc_1024x256"] > float(student["acc_1024x256"]))
        & (head_df["net_fix"] > 0)
    )
    coverage_sensitive = False
    if not frontier_df.empty:
        coverage_sensitive = bool(
            frontier_df["acc_total"].max() > float(full["acc_total"])
            or frontier_df["net_fix"].max() > float(full["net_fix"])
        )

    return {
        "best_variant": _select_best_variant(aggregate_df),
        "positive_partial_signal": bool(positive_mask.any()),
        "positive_partial_variants": partial_df.loc[positive_mask, "variant"].tolist(),
        "head_localizable": bool(head_positive_mask.any()),
        "head_positive_variants": head_df.loc[head_positive_mask, "variant"].tolist(),
        "coverage_sensitive": coverage_sensitive,
        "prefer_exp4_or_exp5": "exp5" if bool(positive_mask.any()) else "exp4",
        "dense_value_row_all_beats_full": bool(
            float(df.loc["dense_value_row_all", "acc_total"]) > float(full["acc_total"])
        ),
    }


def _build_coverage_summary(aggregate_df: pd.DataFrame) -> list[dict[str, Any]]:
    summary_cols = [
        "variant",
        "acc_total",
        "acc_1024x256",
        "override_row_coverage",
        "override_head_coverage",
        "net_fix",
        "pred_changed_vs_student",
    ]
    return json.loads(aggregate_df.loc[:, summary_cols].to_json(orient="records"))


def _validate_dense_value_full(
    *,
    aggregate_df: pd.DataFrame,
    checkpoint_run_id: str,
) -> dict[str, Any]:
    expected = E5B_EXPECTED_FULL_ACC_TOTAL.get(checkpoint_run_id)
    if expected is None:
        return {
            "checkpoint_run_id": checkpoint_run_id,
            "expected_acc_total": None,
            "actual_acc_total": None,
            "passed": None,
        }
    row = aggregate_df.loc[aggregate_df["variant"] == "dense_value_full"]
    if row.empty:
        raise RuntimeError("E5B 缺少 dense_value_full 行, 无法执行一致性校验.")
    actual = float(row["acc_total"].iloc[0])
    passed = bool(np.isclose(actual, expected, atol=5e-6))
    return {
        "checkpoint_run_id": checkpoint_run_id,
        "expected_acc_total": float(expected),
        "actual_acc_total": actual,
        "passed": passed,
    }


def run_e5b_partial_override(
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
    e5a = _load_e5a_module()
    model = bundle["model"]
    _, mixer = e5a._find_audit_flash_mixer(model)
    config_summary = e5a._assert_supported_mainline(mixer)
    if int(E5B_AUDIT_LAYER_IDX) != int(e5a.E5A_AUDIT_LAYER_IDX):
        raise RuntimeError(
            f"E5B audit layer 与 E5A 不一致: {E5B_AUDIT_LAYER_IDX} vs {e5a.E5A_AUDIT_LAYER_IDX}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    quantile_collector = _QuantileCollector()
    for inputs, targets, slices in test_dataloader:
        inputs = inputs.to(next(model.parameters()).device)
        targets = targets.to(inputs.device)
        case_names = e5a._flatten_case_name(slices)

        _, _, _, payload = _collect_student_forward(model, mixer, inputs)
        payload = _tensor_payload_only(payload)
        metadata = _compute_dense_value_metadata(payload=payload, targets=targets)
        quantile_collector.update(
            case_names=case_names,
            task_mask=metadata["task_mask"],
            row_has_disagreement=metadata["row_has_disagreement"],
            teacher_adv_row=metadata["teacher_adv_row"],
        )

    quantile_thresholds = quantile_collector.finalize()
    variant_names: list[str] | None = None
    aggregate_store: _AggregateStore | None = None
    row_writer = _RowPartialAuditWriter(output_dir / "row_partial_audit.parquet")
    head_writer = _HeadPartialAuditWriter(output_dir / "head_partial_audit.parquet")

    example_offset = 0
    try:
        for inputs, targets, slices in test_dataloader:
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(inputs.device)
            batch_size = int(inputs.size(0))
            case_names = e5a._flatten_case_name(slices)
            example_ids = torch.arange(
                example_offset,
                example_offset + batch_size,
                device=inputs.device,
                dtype=torch.int64,
            )

            _, preds_student, final_margin_student, payload = _collect_student_forward(model, mixer, inputs)
            payload = _tensor_payload_only(payload)
            metadata = _compute_dense_value_metadata(payload=payload, targets=targets)
            variants = _build_variant_payloads(
                metadata=metadata,
                case_names=case_names,
                quantile_thresholds=quantile_thresholds,
            )
            if variant_names is None:
                variant_names = [variant.name for variant in variants]
                aggregate_store = _AggregateStore(variant_names)

            for variant in variants:
                if variant.name == "student":
                    preds = preds_student
                    final_margin = final_margin_student
                else:
                    _, preds, final_margin = _collect_override_forward(
                        model,
                        mixer,
                        inputs,
                        variant_name=variant.name,
                        override_top2_codes=variant.top2_codes,
                    )

                assert aggregate_store is not None
                aggregate_store.update(
                    variant_name=variant.name,
                    case_names=case_names,
                    example_ids=example_ids,
                    task_mask=metadata["task_mask"],
                    preds=preds,
                    preds_student=preds_student,
                    targets=targets,
                    final_margin=final_margin,
                    final_margin_student=final_margin_student,
                    row_override_applied=variant.row_override_applied,
                    override_applied_head=variant.override_applied_head,
                )
                row_writer.write(
                    variant_name=variant.name,
                    case_names=case_names,
                    example_ids=example_ids,
                    targets=targets,
                    preds=preds,
                    preds_student=preds_student,
                    final_margin=final_margin,
                    final_margin_student=final_margin_student,
                    metadata=metadata,
                    variant=variant,
                )
                head_writer.write(
                    variant_name=variant.name,
                    case_names=case_names,
                    example_ids=example_ids,
                    metadata=metadata,
                    variant=variant,
                )

            example_offset += batch_size

        if aggregate_store is None or variant_names is None:
            raise RuntimeError("E5B 没有成功初始化任何 variant.")

        aggregate_df, aggregate_by_case_df = aggregate_store.finalize()
        aggregate_df.insert(0, "eval_run_id", eval_run_id)
        aggregate_df.insert(0, "eval_launch_id", eval_launch_id)
        aggregate_df.insert(0, "checkpoint_run_id", checkpoint_run_id)
        aggregate_df.insert(0, "checkpoint_launch_id", checkpoint_launch_id)
        aggregate_df["audit_layer_idx"] = E5B_AUDIT_LAYER_IDX

        aggregate_by_case_df.insert(0, "eval_run_id", eval_run_id)
        aggregate_by_case_df.insert(0, "eval_launch_id", eval_launch_id)
        aggregate_by_case_df.insert(0, "checkpoint_run_id", checkpoint_run_id)
        aggregate_by_case_df.insert(0, "checkpoint_launch_id", checkpoint_launch_id)
        aggregate_by_case_df["audit_layer_idx"] = E5B_AUDIT_LAYER_IDX

        aggregate_df.to_csv(output_dir / "aggregate.csv", index=False)
        aggregate_by_case_df.to_csv(output_dir / "aggregate_by_case.csv", index=False)

        consistency_check = _validate_dense_value_full(
            aggregate_df=aggregate_df,
            checkpoint_run_id=checkpoint_run_id,
        )
        final_ruling = _build_run_conclusion(aggregate_df)
        summary = {
            "checkpoint_launch_id": checkpoint_launch_id,
            "checkpoint_run_id": checkpoint_run_id,
            "best_checkpoint": str(checkpoint_path),
            "eval_launch_id": eval_launch_id,
            "eval_run_id": eval_run_id,
            "audit_layer_idx": E5B_AUDIT_LAYER_IDX,
            "cases": list(E5B_CASES),
            "variants": list(variant_names),
            "output_dir": str(output_dir.resolve()),
            "row_partial_audit_path": str((output_dir / "row_partial_audit.parquet").resolve()),
            "head_partial_audit_path": str((output_dir / "head_partial_audit.parquet").resolve()),
            "config_summary": config_summary,
            "quantile_thresholds": quantile_thresholds,
            "coverage_summary": _build_coverage_summary(aggregate_df),
            "aggregate": json.loads(aggregate_df.to_json(orient="records")),
            "aggregate_by_case": json.loads(aggregate_by_case_df.to_json(orient="records")),
            "consistency_check": consistency_check,
            "final_ruling": final_ruling,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if logger is not None:
            log_metrics = {}
            for _, row in aggregate_df.iterrows():
                variant_name = str(row["variant"])
                log_metrics[f"e5b/{variant_name}/acc_total"] = float(row["acc_total"])
                log_metrics[f"e5b/{variant_name}/net_fix"] = float(row["net_fix"])
                log_metrics[f"e5b/{variant_name}/override_row_coverage"] = float(row["override_row_coverage"])
            logger.log(log_metrics, step=0)

        if consistency_check["passed"] is False:
            raise RuntimeError(
                "E5B dense_value_full 与阶段 A dense_value 不一致: "
                f"expected={consistency_check['expected_acc_total']}, "
                f"actual={consistency_check['actual_acc_total']}."
            )

        return summary
    finally:
        row_writer.close()
        head_writer.close()
