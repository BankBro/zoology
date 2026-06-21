from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import torch

from experiment_lib import (
    ARTIFACT_DIR,
    build_config,
    ensure_artifact_dirs,
    extract_payload_state,
    initialized_model_and_state,
    load_state_payload,
    parse_targets,
    snapshot_path,
    validate_scope_boundaries,
    write_csv,
    write_json,
)


def _float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().cpu().item())
    return float(value)


def _summary(prefix: str, values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    quantiles = torch.quantile(
        values,
        torch.tensor([0.05, 0.50, 0.95], dtype=torch.float32, device=values.device),
    )
    return {
        f"{prefix}_mean": _float(values.mean()),
        f"{prefix}_std": _float(values.std(unbiased=False)),
        f"{prefix}_min": _float(values.min()),
        f"{prefix}_p05": _float(quantiles[0]),
        f"{prefix}_p50": _float(quantiles[1]),
        f"{prefix}_p95": _float(quantiles[2]),
        f"{prefix}_max": _float(values.max()),
    }


def _offdiag_mask(n: int, device: torch.device) -> torch.Tensor:
    return ~torch.eye(n, dtype=torch.bool, device=device)


def _pairwise_cos_rows(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    x = torch.nn.functional.normalize(x, dim=-1, eps=1e-8)
    sim = x @ x.transpose(-1, -2)
    n = sim.size(-1)
    if n <= 1:
        return torch.empty((0,), dtype=torch.float32, device=sim.device)
    return sim[..., _offdiag_mask(n, sim.device)]


def _pairwise_l2_rows(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    dist = torch.cdist(x, x, p=2)
    n = dist.size(-1)
    if n <= 1:
        return torch.empty((0,), dtype=torch.float32, device=dist.device)
    return dist[..., _offdiag_mask(n, dist.device)]


def _nearest_offdiag(values: torch.Tensor, *, larger_is_nearer: bool) -> torch.Tensor:
    values = values.detach().float()
    n = values.size(-1)
    if n <= 1:
        return torch.empty((0,), dtype=torch.float32, device=values.device)
    masked = values.clone()
    eye = torch.eye(n, dtype=torch.bool, device=values.device)
    if larger_is_nearer:
        masked[..., eye] = -float("inf")
        return masked.max(dim=-1).values
    masked[..., eye] = float("inf")
    return masked.min(dim=-1).values


def _safe_frob_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.detach().float().reshape(a.shape[0], -1)
    b_flat = b.detach().float().reshape(b.shape[0], -1)
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1, eps=1e-8)


def _layer_from_key(key: str) -> int:
    match = re.search(r"backbone\.layers\.(\d+)\.", key)
    if match is None:
        return -1
    return int(match.group(1))


def _base_prefix_from_codebook_key(key: str) -> str:
    return key[: -len("quantizer.codebook")]


def _sigmoid_stats(prefix: str, tensor: torch.Tensor | None) -> dict[str, float]:
    if tensor is None:
        return {}
    return _summary(prefix, torch.sigmoid(tensor.detach().float()))


def _codebook_row_metrics(codebook: torch.Tensor) -> dict[str, float]:
    c = codebook.detach().float()
    h, s, d = c.shape
    row: dict[str, float] = {
        "num_heads": float(h),
        "num_codebook_vectors": float(s),
        "key_dim": float(d),
    }
    norms = c.norm(dim=-1)
    row.update(_summary("code_norm", norms))
    row.update(_summary("code_rms", torch.sqrt((c * c).mean(dim=-1).clamp_min(1e-12))))
    sims = torch.nn.functional.normalize(c, dim=-1, eps=1e-8) @ torch.nn.functional.normalize(
        c, dim=-1, eps=1e-8
    ).transpose(-1, -2)
    dists = torch.cdist(c, c, p=2)
    row.update(_summary("code_pair_cos_offdiag", _pairwise_cos_rows(c)))
    row.update(_summary("code_nearest_cos", _nearest_offdiag(sims, larger_is_nearer=True)))
    row.update(_summary("code_pair_l2_offdiag", _pairwise_l2_rows(c)))
    row.update(_summary("code_nearest_l2", _nearest_offdiag(dists, larger_is_nearer=False)))
    if h > 1:
        head_flat = c.reshape(h, -1)
        row.update(_summary("head_code_cos_offdiag", _pairwise_cos_rows(head_flat)))
    return row


def _projection_metrics(
    *,
    codebook: torch.Tensor,
    qkvg_weight: torch.Tensor | None,
    res_proj_weight: torch.Tensor | None,
) -> dict[str, float]:
    if qkvg_weight is None:
        return {}
    c = codebook.detach().float()
    w = qkvg_weight.detach().float()
    h, _, d_k = c.shape
    q_ch = h * d_k
    if res_proj_weight is not None:
        v_ch = int(res_proj_weight.shape[1])
    else:
        v_ch = q_ch
    if w.shape[0] < 2 * q_ch + v_ch:
        return {"projection_parse_error": 1.0}
    q = w[:q_ch].reshape(h, d_k, -1)
    k = w[q_ch : 2 * q_ch].reshape(h, d_k, -1)
    v = w[2 * q_ch : 2 * q_ch + v_ch].reshape(h, v_ch // h, -1)
    row: dict[str, float] = {
        "projection_parse_error": 0.0,
        "q_proj_rms": _float(torch.sqrt((q * q).mean())),
        "k_proj_rms": _float(torch.sqrt((k * k).mean())),
        "v_proj_rms": _float(torch.sqrt((v * v).mean())),
    }
    if res_proj_weight is not None:
        row["o_proj_rms"] = _float(torch.sqrt((res_proj_weight.detach().float() ** 2).mean()))

    c_centered = c - c.mean(dim=1, keepdim=True)
    c_cov = torch.einsum("hsd,hse->hde", c_centered, c_centered) / max(int(c.size(1)) - 1, 1)
    k_gram = torch.einsum("hdi,hei->hde", k, k) / max(int(k.size(-1)), 1)
    q_gram = torch.einsum("hdi,hei->hde", q, q) / max(int(q.size(-1)), 1)
    row.update(_summary("proj_code_cov_cos_k", _safe_frob_cos(c_cov, k_gram)))
    row.update(_summary("proj_code_cov_cos_q", _safe_frob_cos(c_cov, q_gram)))
    row.update(_summary("qk_proj_cov_cos", _safe_frob_cos(q_gram, k_gram)))
    if h > 1:
        row.update(_summary("head_k_proj_cos_offdiag", _pairwise_cos_rows(k.reshape(h, -1))))
        row.update(_summary("head_q_proj_cos_offdiag", _pairwise_cos_rows(q.reshape(h, -1))))
    return row


def _addr_metrics(*, codebook: torch.Tensor, addr_proj: torch.Tensor | None) -> dict[str, float]:
    if addr_proj is None:
        return {}
    c = codebook.detach().float()
    a = addr_proj.detach().float()
    row: dict[str, float] = {}
    row.update(_summary("addr_col_cos_abs_offdiag", _pairwise_cos_rows(a.transpose(1, 2)).abs()))
    svals = torch.linalg.svdvals(a)
    row.update(_summary("addr_singular", svals))
    cond = svals[..., 0] / svals[..., -1].clamp_min(1e-12)
    row.update(_summary("addr_condition", cond))
    coords = torch.einsum("hsd,hdr->hsr", c, a)
    row.update(_summary("addr_coord_pair_l2_offdiag", _pairwise_l2_rows(coords)))
    coord_dists = torch.cdist(coords, coords, p=2)
    row.update(_summary("addr_coord_nearest_l2", _nearest_offdiag(coord_dists, larger_is_nearer=False)))
    row.update(_summary("addr_coord_pair_cos_offdiag", _pairwise_cos_rows(coords)))
    if a.size(0) > 1:
        row.update(_summary("head_addr_proj_cos_offdiag", _pairwise_cos_rows(a.reshape(a.size(0), -1))))
    return row


def audit_snapshot(snapshot: Path) -> list[dict[str, Any]]:
    payload = load_state_payload(snapshot)
    state = extract_payload_state(payload)
    rows: list[dict[str, Any]] = []
    for key in sorted(state):
        if not key.endswith("quantizer.codebook"):
            continue
        codebook = state[key]
        prefix = _base_prefix_from_codebook_key(key)
        row: dict[str, Any] = {
            "snapshot_path": str(snapshot.resolve()),
            "target": payload.get("target"),
            "scope": payload.get("scope"),
            "layer": _layer_from_key(key),
            "codebook_key": key,
            "snapshot_sha256": payload.get("sha256"),
        }
        row.update(_codebook_row_metrics(codebook))
        row.update(
            _projection_metrics(
                codebook=codebook,
                qkvg_weight=state.get(prefix + "qkvg_proj.weight"),
                res_proj_weight=state.get(prefix + "res_proj.weight"),
            )
        )
        row.update(_addr_metrics(codebook=codebook, addr_proj=state.get(prefix + "fox_gd_residual_addr_proj")))
        row.update(_sigmoid_stats("beta_bias_sigmoid", state.get(prefix + "fox_gd_residual_beta_proj.bias")))
        row.update(_sigmoid_stats("lambda_bias_sigmoid", state.get(prefix + "fox_gd_residual_lambda_proj.bias")))
        rows.append(row)
    return rows


def _collect_module_metrics(model) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, module in model.named_modules():
        getter = getattr(module, "get_scalar_metrics", None)
        if getter is None:
            continue
        values = getter()
        if not values:
            continue
        safe_name = name.replace(".", "/")
        for key, value in values.items():
            try:
                metrics[f"{safe_name}:{key}"] = float(value)
            except Exception:
                continue
    return metrics


def _candidate_metrics_for_layer(layer, hidden_states: torch.Tensor, residual: torch.Tensor | None) -> dict[str, float]:
    flash_mixer = None
    for _, module in layer.sequence_mixer.named_modules():
        if hasattr(module, "attn"):
            flash_mixer = module
            break
    if flash_mixer is None:
        return {}
    dropped = layer.drop_path1(layer.dropout1(hidden_states))
    residual_next = dropped + residual if residual is not None else dropped
    normed = layer.norm1(residual_next.to(dtype=layer.norm1.weight.dtype))
    attn = flash_mixer.attn
    _, k, _, _ = attn.project_qkv(normed)
    quantizer = attn.quantizer
    if hasattr(quantizer, "_compute_scores"):
        scores = quantizer._compute_scores(k)
        if hasattr(quantizer, "_compute_weights"):
            weights = quantizer._compute_weights(scores)
        else:
            weights = torch.nn.functional.softmax(scores.float(), dim=-1).to(scores.dtype)
    else:
        codebook = quantizer.get_codebook().to(device=k.device, dtype=k.dtype)
        scores = -(
            torch.sum(k**2, dim=-1, keepdim=True)
            - 2.0 * torch.einsum("bhld,hsd->bhls", k, codebook)
            + torch.sum(codebook**2, dim=-1).unsqueeze(0).unsqueeze(2)
        )
        weights = torch.nn.functional.softmax(scores.float(), dim=-1).to(scores.dtype)
    top_k = min(4, int(scores.size(-1)))
    top_vals = torch.topk(scores.float(), k=min(2, int(scores.size(-1))), dim=-1).values
    if top_vals.size(-1) >= 2:
        margin = top_vals[..., 0] - top_vals[..., 1]
    else:
        margin = top_vals[..., 0]
    weight_top = torch.topk(weights.float(), k=top_k, dim=-1).values
    counts = weights.float().sum(dim=(0, 2))
    probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    active_ratio = (counts > (counts.mean(dim=-1, keepdim=True) * 0.1)).float().mean(dim=-1)
    row: dict[str, float] = {}
    row.update(_summary("candidate_score_margin_top1_top2", margin))
    row.update(_summary("candidate_weight_top1", weight_top[..., 0]))
    row.update(_summary("candidate_weight_top4_mass", weight_top.sum(dim=-1)))
    row.update(_summary("write_count", counts))
    row.update(_summary("write_entropy_per_head", entropy))
    row.update(_summary("write_active_ratio_per_head", active_ratio))
    return row


def run_probe(snapshot: Path, *, device: str) -> list[dict[str, Any]]:
    payload = load_state_payload(snapshot)
    target = str(payload.get("target"))
    if not target or target == "None":
        return []
    state = extract_payload_state(payload)
    config = build_config(
        target,
        max_epochs=1,
        run_id=f"audit-probe-{target}",
        experiment_mode=f"gd_init_geometry_probe_{target.replace('-', '_')}",
        smoke_data=True,
    )
    model, _ = initialized_model_and_state(config)
    incompatible = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    from zoology.data.utils import prepare_data

    train_loader, _ = prepare_data(config.data)
    inputs, _, slices = next(iter(train_loader))
    inputs = inputs.to(device)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        hidden_states = model.backbone.embeddings(inputs)
        residual = None
        for layer_idx, layer in enumerate(model.backbone.layers):
            candidate_row = _candidate_metrics_for_layer(layer, hidden_states, residual)
            hidden_states, residual = layer(hidden_states, residual)
            module_metrics = _collect_module_metrics(model)
            row: dict[str, Any] = {
                "snapshot_path": str(snapshot.resolve()),
                "target": target,
                "scope": payload.get("scope"),
                "layer": layer_idx,
                "probe_device": device,
                "probe_case": slices[0].get("mqar_case") if slices else None,
                "missing_keys": len(incompatible.missing_keys),
                "unexpected_keys": len(incompatible.unexpected_keys),
            }
            row.update(candidate_row)
            for key, value in module_metrics.items():
                if key.startswith(f"backbone/layers/{layer_idx}/sequence_mixer"):
                    metric_name = key.split(":", maxsplit=1)[1]
                    if metric_name.startswith("vq/") or metric_name.startswith("attn/gd_residual"):
                        row[f"forward_{metric_name}"] = value
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 Flash-VQG gd_residual_v1 初始化几何.")
    parser.add_argument("--targets", default="cb64-r16-s124,cb64-r16-s125,cb256-r4-s123,cb256-r4-s124")
    parser.add_argument("--scope", default="full_model")
    parser.add_argument("--snapshot", action="append", default=[])
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    validate_scope_boundaries()
    ensure_artifact_dirs()
    snapshots = [Path(item) for item in args.snapshot]
    if not snapshots:
        snapshots = [snapshot_path(target, args.scope) for target in parse_targets(args.targets)]

    audit_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if not snapshot.exists():
            raise FileNotFoundError(f"snapshot 不存在: {snapshot}")
        audit_rows.extend(audit_snapshot(snapshot))
        if args.probe:
            probe_rows.extend(run_probe(snapshot, device=str(args.device)))

    audit_path = ARTIFACT_DIR / "init-geometry-audit.csv"
    write_csv(audit_path, audit_rows)
    probe_path = ARTIFACT_DIR / "init-geometry-probe.csv"
    if args.probe:
        write_csv(probe_path, probe_rows)
    write_json(
        ARTIFACT_DIR / "audit-status.json",
        {
            "status": "audit_completed",
            "audit_path": str(audit_path.resolve()),
            "probe_path": str(probe_path.resolve()) if args.probe else None,
            "num_audit_rows": len(audit_rows),
            "num_probe_rows": len(probe_rows),
        },
    )
    print(f"audit: {audit_path.resolve()}", flush=True)
    if args.probe:
        print(f"probe: {probe_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
