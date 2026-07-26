#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LOCAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(LOCAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPT_DIR))

from common import (  # noqa: E402
    EXPERIMENT_ID,
    GDN_KERNEL_DTYPE,
    LONGER_SHAPES,
    REPO_ROOT,
    atomic_write_json,
    load_json,
    output_root,
    sha256_file,
    stable_json_sha256,
    utc_now,
)
from coordinator import (  # noqa: E402
    RELATIVE_OUTPUT,
    REMOTE_CONTAINER,
    REMOTE_HOST,
    REMOTE_PROJECT,
    remote_read,
)


ARTIFACT_DIR = REPO_ROOT / "docs" / "artifacts" / EXPERIMENT_ID
REPORT_PATH = REPO_ROOT / "docs" / f"{EXPERIMENT_ID}-report.md"
EVIDENCE_PATHS = (
    Path("preflight.json"),
    Path("status.json"),
    Path("formal-detail.json"),
    Path("gates/training-smoke.json"),
    Path("gates/batch-profile.json"),
    Path("gates/eval-smoke.json"),
    Path("gates/legacy-canary.json"),
    Path("gates/LOCAL_SMOKE_PASSED.json"),
    Path("gates/GLOBAL_FORMAL_GATE.json"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    temporary.replace(path)


def load_machine_details() -> dict[str, list[dict[str, Any]]]:
    local_path = output_root("2080ti") / "formal-detail.json"
    local = load_json(local_path)
    remote = remote_read(Path("formal-detail.json"))
    if remote is None:
        raise RuntimeError("Remote formal detail is unavailable.")
    return {"2080ti": local, "3090": remote}


def evaluation_result_metadata() -> dict[str, dict[str, Any]]:
    fields = (
        "event_id",
        "started_at_utc",
        "ended_at_utc",
        "eval_batch_size",
        "dataset_num_examples",
        "query_count",
        "wall_clock_sec",
        "peak_allocated_mib",
        "result_path",
    )
    metadata: dict[str, dict[str, Any]] = {}
    local_root = output_root("2080ti") / "evaluation"
    for result_path in sorted(local_root.glob("formal*/*/result.json")):
        payload = load_json(result_path)
        row = {key: payload.get(key) for key in fields}
        row["result_path"] = str(result_path.resolve())
        metadata[payload["event_id"]] = row

    remote_root = REMOTE_PROJECT / RELATIVE_OUTPUT / "evaluation"
    code = f"""import json
from pathlib import Path
root = Path({str(remote_root)!r})
fields = {fields!r}
rows = []
for path in sorted(root.glob('formal*/*/result.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    row = {{key: payload.get(key) for key in fields}}
    row['result_path'] = str(path)
    rows.append(row)
print(json.dumps(rows, ensure_ascii=False))
"""
    remote_command = shlex.join(
        [
            "docker",
            "exec",
            "-u",
            "lyj",
            REMOTE_CONTAINER,
            "/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python",
            "-c",
            code,
        ]
    )
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            REMOTE_HOST,
            remote_command,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not collect remote eval metadata: {result.stderr}")
    for row in json.loads(result.stdout):
        metadata[row["event_id"]] = row
    return metadata


def flatten(details: dict[str, list[dict[str, Any]]]):
    training = []
    evaluation = []
    for machine, rows in details.items():
        for row in rows:
            result = row["training_result"]
            training.append(
                {
                    "machine": machine,
                    "descriptor_id": row["descriptor_id"],
                    "model": result["model"],
                    "seed": result["seed"],
                    "data_seed": 123,
                    "train_precision": result["train_precision"],
                    "status": result["status"],
                    "configured_max_epochs": 4,
                    "final_epoch": result["last_checkpoint"]["epoch"],
                    "train_batch_size": 64,
                    "eval_batch_size": 16,
                    "gradient_accumulation_steps": 4,
                    "effective_train_batch_size": 256,
                    "batch_accum_profile": "b64_ga4",
                    "validations_per_epoch": 4,
                    "wall_clock_sec": result["wall_clock_sec"],
                    "started_at_utc": result["started_at_utc"],
                    "ended_at_utc": result["ended_at_utc"],
                    "gdn_kernel_dtype": result["gdn_kernel_dtype"],
                    "grad_scaler_skips": result["resume_audit"][
                        "grad_scaler_skips"
                    ],
                    "model_state_dtypes": ";".join(
                        result["resume_audit"]["model_state_dtypes"]
                    ),
                    "optimizer_state_dtypes": ";".join(
                        result["resume_audit"]["optimizer_state_dtypes"]
                    ),
                    "optimizer_step_wall_sec_p50": result["telemetry"].get(
                        "optimizer_step_wall_sec_p50"
                    ),
                    "optimizer_step_wall_sec_p90": result["telemetry"].get(
                        "optimizer_step_wall_sec_p90"
                    ),
                    "peak_allocated_mib": result["telemetry"].get(
                        "peak_allocated_mib"
                    ),
                    "peak_reserved_mib": result["telemetry"].get(
                        "peak_reserved_mib"
                    ),
                    "normalized_config_sha256": result[
                        "normalized_config_sha256"
                    ],
                    "resolved_config_path": result["resolved_config_path"],
                    "resolved_config_sha256": result["resolved_config_sha256"],
                    "valid_loss": result["last_checkpoint"]["metrics"][
                        "valid/loss"
                    ],
                    "valid_accuracy": result["last_checkpoint"]["metrics"][
                        "valid/accuracy"
                    ],
                    "last_checkpoint_path": result["last_checkpoint"]["path"],
                    "last_checkpoint_sha256": result["last_checkpoint"][
                        "file_sha256"
                    ],
                    "last_checkpoint_model_state_sha256": result[
                        "last_checkpoint"
                    ]["model_state_sha256"],
                    "best_checkpoint_path": result["best_checkpoint"]["path"],
                    "best_checkpoint_sha256": result["best_checkpoint"][
                        "file_sha256"
                    ],
                    "best_checkpoint_model_state_sha256": result[
                        "best_checkpoint"
                    ]["model_state_sha256"],
                }
            )
            for event in row["evaluation"]:
                evaluation.append({"source_machine": machine, **event})
    return training, evaluation


def validate_counts(training: list[dict[str, Any]], evaluation: list[dict[str, Any]]):
    if len(training) != 30:
        raise RuntimeError(f"Expected 30 training rows, got {len(training)}.")
    expected = {"2080ti": 12 * 2 * 2 * 13, "3090": 18 * 2 * 3 * 13}
    observed = defaultdict(int)
    for row in evaluation:
        observed[row["source_machine"]] += 1
    if dict(observed) != expected:
        raise RuntimeError(
            f"Unexpected logical eval counts: expected={expected}, observed={dict(observed)}"
        )
    descriptor_ids = {row["descriptor_id"] for row in training}
    if len(descriptor_ids) != 30 or any(
        row["status"] != "completed" for row in training
    ):
        raise RuntimeError("Training rows are incomplete or not unique.")
    unexpected_eval_status = {
        row["status"]
        for row in evaluation
        if row["status"] not in {"completed", "deduplicated"}
    }
    if unexpected_eval_status:
        raise RuntimeError(
            f"Unexpected formal evaluation statuses: {unexpected_eval_status}"
        )
    if any(int(row["grad_scaler_skips"]) > 2 for row in training):
        raise RuntimeError("A formal run exceeded the registered scaler-skip limit.")
    if any(
        row["model_state_dtypes"] != "torch.float32"
        or row["optimizer_state_dtypes"] != "torch.float32"
        for row in training
    ):
        raise RuntimeError("Master model or optimizer state escaped FP32.")
    for row in training:
        if row["model"] == "gdn" and row["gdn_kernel_dtype"] != GDN_KERNEL_DTYPE[
            row["train_precision"]
        ]:
            raise RuntimeError(f"Unexpected GDN kernel dtype: {row}")
    dataset_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in evaluation:
        dataset_hashes[(row["shape"], int(row["num_examples"]))].add(
            row["dataset_hash"]
        )
    if len(dataset_hashes) != 13 or any(
        len(hashes) != 1 for hashes in dataset_hashes.values()
    ):
        raise RuntimeError(f"Dataset identity mismatch: {dataset_hashes}")


def aggregate(evaluation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in evaluation:
        key = (
            row["source_machine"],
            row["model"],
            row["checkpoint_role"],
            row["train_precision"],
            row["eval_precision"],
            row["shape"],
            int(row["num_examples"]),
        )
        buckets[key].append(float(row["accuracy"]))
    rows = []
    for key, values in sorted(buckets.items()):
        (
            machine,
            model,
            role,
            train_precision,
            eval_precision,
            shape,
            num_examples,
        ) = key
        rows.append(
            {
                "machine": machine,
                "model": model,
                "checkpoint_role": role,
                "train_precision": train_precision,
                "eval_precision": eval_precision,
                "shape": shape,
                "num_examples": num_examples,
                "n_seeds": len(values),
                "accuracy_mean": statistics.mean(values),
                "accuracy_population_std": statistics.pstdev(values),
            }
        )
    return rows


def make_figure(summary: list[dict[str, Any]], role: str) -> None:
    import matplotlib.pyplot as plt

    longer_order = [f"{seq}x{kv}" for seq, kv in LONGER_SHAPES]
    colors = {"flash": "#0072B2", "gdn": "#D55E00"}
    line_styles = {"fp32": "-", "fp16": "--", "bf16": ":"}
    markers = {"fp32": "o", "fp16": "s", "bf16": "^"}
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    for panel, (axis, machine) in enumerate(zip(axes, ("2080ti", "3090"))):
        rows = [
            row
            for row in summary
            if row["machine"] == machine
            and row["checkpoint_role"] == role
            and row["train_precision"] == row["eval_precision"]
            and int(row["num_examples"]) == 500
        ]
        groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            groups[(row["model"], row["train_precision"])][row["shape"]] = row
        for (model, precision), by_shape in sorted(groups.items()):
            means = [float(by_shape[shape]["accuracy_mean"]) for shape in longer_order]
            stds = [
                float(by_shape[shape]["accuracy_population_std"])
                for shape in longer_order
            ]
            x = list(range(len(longer_order)))
            axis.plot(
                x,
                means,
                color=colors[model],
                linestyle=line_styles[precision],
                marker=markers[precision],
                markersize=3.5,
                linewidth=1.4,
                label=f"{model.upper()} {precision.upper()}",
            )
            axis.fill_between(
                x,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=colors[model],
                alpha=0.12,
                linewidth=0,
            )
        machine_label = "2080 Ti" if machine == "2080ti" else "RTX 3090"
        axis.set_title(
            f"{chr(65 + panel)}  {machine_label}", loc="left", fontweight="bold"
        )
        axis.set_xticks(range(len(longer_order)), longer_order, rotation=25, ha="right")
        axis.set_xlabel("Sequence length × key–value pairs")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Per-example MQAR accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.4,
    )
    fig.suptitle(
        f"Matching train/eval precision — {role} checkpoint",
        y=0.985,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.72))
    figure_dir = ARTIFACT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"matching-precision-{role}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def mirror_evidence() -> list[dict[str, Any]]:
    rows = []
    for machine in ("2080ti", "3090"):
        for relative_path in EVIDENCE_PATHS:
            if machine == "2080ti":
                source_path = output_root(machine) / relative_path
                if not source_path.exists():
                    raise RuntimeError(f"Missing local evidence: {source_path}")
                payload = load_json(source_path)
            else:
                source_path = REMOTE_PROJECT / RELATIVE_OUTPUT / relative_path
                payload = remote_read(relative_path)
                if payload is None:
                    raise RuntimeError(f"Missing remote evidence: {source_path}")
            mirror_path = ARTIFACT_DIR / "machines" / machine / relative_path
            atomic_write_json(mirror_path, payload)
            source_hash = stable_json_sha256(payload)
            mirror_hash = stable_json_sha256(load_json(mirror_path))
            if source_hash != mirror_hash:
                raise RuntimeError(
                    f"Evidence mirror hash mismatch: {source_path} -> {mirror_path}"
                )
            rows.append(
                {
                    "artifact_type": "json_evidence",
                    "machine": machine,
                    "descriptor_id": "",
                    "model": "",
                    "seed": "",
                    "train_precision": "",
                    "checkpoint_role": "",
                    "source_path": str(source_path),
                    "mirror_path": str(mirror_path.relative_to(REPO_ROOT)),
                    "sha256": source_hash,
                    "mirror_sha256": mirror_hash,
                    "hash_kind": "stable_json_sha256",
                    "mirror_status": "mirrored_and_verified",
                }
            )
    return rows


def remote_read_absolute_json(path: Path) -> dict[str, Any]:
    remote_command = shlex.join(
        [
            "docker",
            "exec",
            "-u",
            "lyj",
            REMOTE_CONTAINER,
            "cat",
            str(path),
        ]
    )
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            REMOTE_HOST,
            remote_command,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"Could not read remote JSON {path}: {result.stderr}")
    return json.loads(result.stdout)


def mirror_resolved_configs(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in training:
        source_path = Path(row["resolved_config_path"])
        payload = (
            load_json(source_path)
            if row["machine"] == "2080ti"
            else remote_read_absolute_json(source_path)
        )
        mirror_path = (
            ARTIFACT_DIR
            / "machines"
            / row["machine"]
            / "resolved-configs"
            / f"{row['descriptor_id']}.json"
        )
        atomic_write_json(mirror_path, payload)
        mirror_hash = sha256_file(mirror_path)
        source_hash = row["resolved_config_sha256"]
        if mirror_hash != source_hash:
            raise RuntimeError(
                f"Resolved config mirror hash mismatch: {source_path} -> {mirror_path}"
            )
        rows.append(
            {
                "artifact_type": "resolved_config",
                "machine": row["machine"],
                "descriptor_id": row["descriptor_id"],
                "model": row["model"],
                "seed": row["seed"],
                "train_precision": row["train_precision"],
                "checkpoint_role": "",
                "source_path": str(source_path),
                "mirror_path": str(mirror_path.relative_to(REPO_ROOT)),
                "sha256": source_hash,
                "mirror_sha256": mirror_hash,
                "hash_kind": "file_sha256",
                "mirror_status": "mirrored_and_verified",
            }
        )
    return rows


def source_manifest(
    training: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for row in training:
        for role in ("last", "best"):
            rows.append(
                {
                    "artifact_type": "checkpoint",
                    "machine": row["machine"],
                    "descriptor_id": row["descriptor_id"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "train_precision": row["train_precision"],
                    "checkpoint_role": role,
                    "source_path": row[f"{role}_checkpoint_path"],
                    "mirror_path": "",
                    "sha256": row[f"{role}_checkpoint_sha256"],
                    "mirror_sha256": "",
                    "hash_kind": "file_sha256",
                    "mirror_status": "large_raw_retained_on_source_machine",
                }
            )
    return rows + evidence_rows


def training_ledger(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preflights = {
        machine: load_json(output_root(machine) / "preflight.json")
        if machine == "2080ti"
        else remote_read(Path("preflight.json"))
        for machine in ("2080ti", "3090")
    }
    rows = []
    for row in training:
        environment = preflights[row["machine"]]["environment"]
        is_flash = row["model"] == "flash"
        outer_dtype = {
            "fp32": "float32",
            "fp16": "amp_float16",
            "bf16": "amp_bfloat16",
        }[row["train_precision"]]
        rows.append(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "summary_scope": "epoch4_final_only",
                "comparison_scope": "matching_dtype_precision_profile",
                "run_type": "precision_profile_formal",
                "replicate_id": f"seed{row['seed']}",
                "machine": row["machine"],
                "gpu": environment["cuda_visible_devices"],
                "gpu_name": environment["gpu_name"],
                "gpu_compute_capability": ".".join(
                    str(value) for value in environment["gpu_capability"]
                ),
                "run_id": row["descriptor_id"],
                "model_family": "flash_vqg" if is_flash else "gated_delta_net",
                "config": "baseline-r16-joint"
                if is_flash
                else "gdnxk-h2-ek4-ev4-usegate0",
                "num_codebook_vectors": 256 if is_flash else "",
                "rank": 16 if is_flash else "",
                "num_heads": "" if is_flash else 2,
                "expand_k": "" if is_flash else 4,
                "expand_v": "" if is_flash else 4,
                "seed": row["seed"],
                "data_seed": row["data_seed"],
                "train_batch_size": row["train_batch_size"],
                "eval_batch_size": row["eval_batch_size"],
                "gradient_accumulation_steps": row[
                    "gradient_accumulation_steps"
                ],
                "effective_train_batch_size": row[
                    "effective_train_batch_size"
                ],
                "batch_accum_profile": row["batch_accum_profile"],
                "configured_max_epochs": row["configured_max_epochs"],
                "final_epoch": row["final_epoch"],
                "validations_per_epoch": row["validations_per_epoch"],
                "early_stopping_disabled": True,
                "status": row["status"],
                "started_at_utc": row["started_at_utc"],
                "ended_at_utc": row["ended_at_utc"],
                "wall_clock_sec": row["wall_clock_sec"],
                "dtype_policy": row["train_precision"],
                "outer_model_dtype": outer_dtype,
                "master_weight_dtype": row["model_state_dtypes"],
                "optimizer_state_dtype": row["optimizer_state_dtypes"],
                "kernel_input_dtype": "float32_boundary"
                if is_flash
                else row["gdn_kernel_dtype"],
                "actual_kernel_dtype": "float32"
                if is_flash
                else row["gdn_kernel_dtype"],
                "dtype_comparison_scope": "dtype_probe_matching_only",
                "grad_scaler_skips": row["grad_scaler_skips"],
                "peak_allocated_mib": row["peak_allocated_mib"],
                "peak_reserved_mib": row["peak_reserved_mib"],
                "valid_loss": row["valid_loss"],
                "valid_accuracy": row["valid_accuracy"],
                "normalized_config_sha256": row["normalized_config_sha256"],
                "resolved_config_path": row["resolved_config_path"],
                "resolved_config_sha256": row["resolved_config_sha256"],
                "last_checkpoint_path": row["last_checkpoint_path"],
                "last_checkpoint_sha256": row["last_checkpoint_sha256"],
                "best_checkpoint_path": row["best_checkpoint_path"],
                "best_checkpoint_sha256": row["best_checkpoint_sha256"],
                "zoology_commit": environment["zoology_commit"],
                "flash_commit": environment["flash_commit"],
                "official_scope": "independent_precision_profile",
                "metadata_verification_level": "verified_artifact_and_gate",
            }
        )
    return rows


def longer_mqar_ledger(
    training: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    result_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    training_lookup = {
        (row["machine"], row["model"], int(row["seed"]), row["train_precision"]): row
        for row in training
    }
    preflights = {
        machine: load_json(output_root(machine) / "preflight.json")
        if machine == "2080ti"
        else remote_read(Path("preflight.json"))
        for machine in ("2080ti", "3090")
    }
    longer_shapes = {f"{seq}x{kv}" for seq, kv in LONGER_SHAPES}
    rows = []
    for event in evaluation:
        if int(event["num_examples"]) != 500 or event["shape"] not in longer_shapes:
            continue
        source = training_lookup[
            (
                event["source_machine"],
                event["model"],
                int(event["seed"]),
                event["train_precision"],
            )
        ]
        metadata = result_metadata.get(event["event_id"])
        if metadata is None:
            raise RuntimeError(f"Missing raw result metadata: {event['event_id']}")
        environment = preflights[event["source_machine"]]["environment"]
        input_seq_len, num_kv_pairs = (
            int(value) for value in event["shape"].split("x")
        )
        role = event["checkpoint_role"]
        rows.append(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "eval_event_id": event["event_id"],
                "logical_source_id": event["logical_source_id"],
                "run_type": "longer_mqar_precision_profile",
                "eval_scope": "matching_dtype"
                if event["train_precision"] == event["eval_precision"]
                else "off_diagonal_mechanism_probe",
                "eval_status": "completed",
                "physical_execution_status": event["status"],
                "started_at_utc": metadata["started_at_utc"],
                "ended_at_utc": metadata["ended_at_utc"],
                "wall_clock_sec": metadata.get("wall_clock_sec")
                or event.get("wall_clock_sec"),
                "machine": event["source_machine"],
                "gpu": environment["cuda_visible_devices"],
                "gpu_name": environment["gpu_name"],
                "gpu_compute_capability": ".".join(
                    str(value) for value in environment["gpu_capability"]
                ),
                "source_model_family": "flash_vqg"
                if event["model"] == "flash"
                else "gated_delta_net",
                "source_run_id": source["descriptor_id"],
                "source_role": role,
                "source_seed": event["seed"],
                "source_data_seed": source["data_seed"],
                "source_train_precision": event["train_precision"],
                "source_batch_accum_profile": source["batch_accum_profile"],
                "source_train_batch_size": source["train_batch_size"],
                "source_eval_batch_size": source["eval_batch_size"],
                "source_gradient_accumulation_steps": source[
                    "gradient_accumulation_steps"
                ],
                "source_effective_train_batch_size": source[
                    "effective_train_batch_size"
                ],
                "source_configured_max_epochs": source[
                    "configured_max_epochs"
                ],
                "source_final_epoch": source["final_epoch"],
                "source_checkpoint_path": source[f"{role}_checkpoint_path"],
                "source_checkpoint_sha256": source[
                    f"{role}_checkpoint_sha256"
                ],
                "source_checkpoint_model_state_sha256": source[
                    f"{role}_checkpoint_model_state_sha256"
                ],
                "eval_precision": event["eval_precision"],
                "eval_kernel_dtype": "float32"
                if event["model"] == "flash"
                else GDN_KERNEL_DTYPE[event["eval_precision"]],
                "eval_batch_size": metadata["eval_batch_size"],
                "eval_data_seed": 123,
                "dataset_policy": "generated_seeded",
                "dataset_hash": event["dataset_hash"],
                "input_seq_len": input_seq_len,
                "num_kv_pairs": num_kv_pairs,
                "num_examples": event["num_examples"],
                "query_count": metadata["query_count"],
                "loss": event["loss"],
                "accuracy": event["accuracy"],
                "peak_allocated_mib": event.get("peak_allocated_mib")
                or metadata["peak_allocated_mib"],
                "raw_result_path": metadata["result_path"],
                "zoology_commit": environment["zoology_commit"],
                "flash_commit": environment["flash_commit"],
                "official_scope": "independent_precision_profile",
            }
        )
    expected = 12 * 2 * 2 * 5 + 18 * 2 * 3 * 5
    if len(rows) != expected or any(
        not row["started_at_utc"] or not row["ended_at_utc"] for row in rows
    ):
        raise RuntimeError(
            f"Incomplete longer-MQAR canonical ledger: {len(rows)} != {expected}"
        )
    return rows


def matching_table(summary: list[dict[str, Any]], role: str) -> str:
    longer_order = [f"{seq}x{kv}" for seq, kv in LONGER_SHAPES]
    precision_order = {"fp32": 0, "fp16": 1, "bf16": 2}
    rows = [
        row
        for row in summary
        if row["checkpoint_role"] == role
        and row["train_precision"] == row["eval_precision"]
        and int(row["num_examples"]) == 500
        and row["shape"] in longer_order
    ]
    lookup = {
        (
            row["machine"],
            row["model"],
            row["train_precision"],
            row["shape"],
        ): row
        for row in rows
    }
    groups = sorted(
        {(row["machine"], row["model"], row["train_precision"]) for row in rows},
        key=lambda value: (
            ("2080ti", "3090").index(value[0]),
            ("flash", "gdn").index(value[1]),
            precision_order[value[2]],
        ),
    )
    lines = [
        "| GPU | 模型 | 精度 | " + " | ".join(longer_order) + " |",
        "|---|---|---|" + "---:|" * len(longer_order),
    ]
    for machine, model, precision in groups:
        values = []
        for shape in longer_order:
            row = lookup[(machine, model, precision, shape)]
            values.append(
                f"{float(row['accuracy_mean']):.4f} ± "
                f"{float(row['accuracy_population_std']):.4f}"
            )
        machine_label = "2080 Ti" if machine == "2080ti" else "RTX 3090"
        lines.append(
            f"| {machine_label} | {model.upper()} | {precision.upper()} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines)


def grouped_training(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in training:
        buckets[(row["machine"], row["model"], row["train_precision"])].append(
            row
        )
    rows = []
    for (machine, model, precision), values in sorted(buckets.items()):
        rows.append(
            {
                "machine": machine,
                "model": model,
                "precision": precision,
                "n": len(values),
                "wall_minutes": statistics.mean(
                    float(row["wall_clock_sec"]) for row in values
                )
                / 60,
                "step_p50": statistics.mean(
                    float(row["optimizer_step_wall_sec_p50"]) for row in values
                ),
                "peak_allocated_mib": statistics.mean(
                    float(row["peak_allocated_mib"]) for row in values
                ),
                "peak_reserved_mib": statistics.mean(
                    float(row["peak_reserved_mib"]) for row in values
                ),
                "grad_scaler_skips": sum(
                    int(row["grad_scaler_skips"]) for row in values
                ),
            }
        )
    return rows


def training_table(training: list[dict[str, Any]]) -> str:
    rows = grouped_training(training)
    lines = [
        "| GPU | 模型 | 训练精度 | run | wall time, min | step p50, s | peak allocated, MiB | peak reserved, MiB | scaler skips |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        machine_label = "2080 Ti" if row["machine"] == "2080ti" else "RTX 3090"
        lines.append(
            f"| {machine_label} | {row['model'].upper()} | {row['precision'].upper()} "
            f"| {row['n']} | {row['wall_minutes']:.2f} | {row['step_p50']:.3f} "
            f"| {row['peak_allocated_mib']:.0f} | {row['peak_reserved_mib']:.0f} "
            f"| {row['grad_scaler_skips']} |"
        )
    return "\n".join(lines)


def analysis_facts(
    training: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> dict[str, Any]:
    longer_order = [f"{seq}x{kv}" for seq, kv in LONGER_SHAPES]
    last_matching = {
        (
            row["machine"],
            row["model"],
            row["train_precision"],
            row["shape"],
        ): float(row["accuracy_mean"])
        for row in summary
        if row["checkpoint_role"] == "last"
        and row["train_precision"] == row["eval_precision"]
        and int(row["num_examples"]) == 500
        and row["shape"] in longer_order
    }

    def deltas(machine: str, model: str, low: str) -> list[float]:
        return [
            last_matching[(machine, model, low, shape)]
            - last_matching[(machine, model, "fp32", shape)]
            for shape in longer_order
        ]

    gdn_low_deltas = deltas("2080ti", "gdn", "fp16")
    for precision in ("fp16", "bf16"):
        gdn_low_deltas.extend(deltas("3090", "gdn", precision))

    eval_buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in evaluation:
        key = (
            row["source_machine"],
            row["model"],
            int(row["seed"]),
            row["train_precision"],
            row["checkpoint_role"],
            row["shape"],
            int(row["num_examples"]),
        )
        eval_buckets[key].append(float(row["accuracy"]))
    max_eval_dtype_range = max(max(values) - min(values) for values in eval_buckets.values())

    last_rows = {
        (
            row["source_machine"],
            int(row["seed"]),
            row["train_precision"],
            row["shape"],
            row["model"],
        ): float(row["accuracy"])
        for row in evaluation
        if row["checkpoint_role"] == "last"
        and row["train_precision"] == row["eval_precision"]
        and int(row["num_examples"]) == 500
        and row["shape"] in longer_order
    }
    paired_total = 0
    paired_flash_wins = 0
    endpoint_total = 0
    endpoint_flash_wins = 0
    for machine, precisions in (
        ("2080ti", ("fp32", "fp16")),
        ("3090", ("fp32", "fp16", "bf16")),
    ):
        for precision in precisions:
            for seed in (123, 124, 125):
                for shape in longer_order:
                    flash = last_rows[(machine, seed, precision, shape, "flash")]
                    gdn = last_rows[(machine, seed, precision, shape, "gdn")]
                    if shape == longer_order[0]:
                        endpoint_total += 1
                        endpoint_flash_wins += int(flash > gdn)
                    else:
                        paired_total += 1
                        paired_flash_wins += int(flash > gdn)

    training_groups = {
        (row["machine"], row["model"], row["precision"]): row
        for row in grouped_training(training)
    }

    def efficiency_ratio(machine: str, model: str, precision: str, key: str):
        return (
            training_groups[(machine, model, precision)][key]
            / training_groups[(machine, model, "fp32")][key]
        )

    return {
        "flash_2080_fp16": deltas("2080ti", "flash", "fp16"),
        "flash_3090_fp16": deltas("3090", "flash", "fp16"),
        "flash_3090_bf16": deltas("3090", "flash", "bf16"),
        "gdn_max_abs_delta": max(abs(value) for value in gdn_low_deltas),
        "max_eval_dtype_range": max_eval_dtype_range,
        "paired_total": paired_total,
        "paired_flash_wins": paired_flash_wins,
        "endpoint_total": endpoint_total,
        "endpoint_flash_wins": endpoint_flash_wins,
        "flash_2080_fp16_wall_ratio": efficiency_ratio(
            "2080ti", "flash", "fp16", "wall_minutes"
        ),
        "flash_3090_fp16_wall_ratio": efficiency_ratio(
            "3090", "flash", "fp16", "wall_minutes"
        ),
        "flash_3090_bf16_wall_ratio": efficiency_ratio(
            "3090", "flash", "bf16", "wall_minutes"
        ),
        "flash_low_alloc_ratio": efficiency_ratio(
            "3090", "flash", "fp16", "peak_allocated_mib"
        ),
        "gdn_2080_fp16_wall_ratio": efficiency_ratio(
            "2080ti", "gdn", "fp16", "wall_minutes"
        ),
        "gdn_3090_fp16_wall_ratio": efficiency_ratio(
            "3090", "gdn", "fp16", "wall_minutes"
        ),
        "gdn_3090_bf16_wall_ratio": efficiency_ratio(
            "3090", "gdn", "bf16", "wall_minutes"
        ),
        "gdn_low_alloc_ratio": efficiency_ratio(
            "3090", "gdn", "fp16", "peak_allocated_mib"
        ),
        "scaler_skips": sum(int(row["grad_scaler_skips"]) for row in training),
    }


def _range_text(values: list[float]) -> str:
    return f"{min(values):+.4f} 至 {max(values):+.4f}"


def write_report(
    training: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
):
    completed = sum(row["status"] == "completed" for row in training)
    physical = sum(row["status"] == "completed" for row in evaluation)
    deduplicated = sum(row["status"] == "deduplicated" for row in evaluation)
    facts = analysis_facts(training, evaluation, summary)
    global_gate = load_json(output_root("2080ti") / "gates/GLOBAL_FORMAL_GATE.json")
    report = f"""# MQAR 低精度与长度泛化实验报告

## 1. 结果概览

`{EXPERIMENT_ID}` 已完成 {completed}/30 个正式训练 run 和 {len(evaluation)} 个逻辑 checkpoint-eval 事件, 其中 {physical} 个为物理执行, {deduplicated} 个因 best/last state hash 相同而可审计去重. 2080 Ti 完成 12/12 run, RTX 3090 完成 18/18 run. 全部结果均在双机 train/validation/eval smoke, controlled resume, 全量 batch capacity, batch invariance, legacy canary 和 global commit/cache gate 通过后生成.

主结论是: 低精度训练在两种模型上均可稳定完成, 且不改变 Flash 在四个真正外推 slice 上相对 GDN 的长度泛化优势. 但 Flash 对训练机器和低精度训练轨迹比 GDN 更敏感; 不能把 2080 Ti 与 RTX 3090 合并为 `n=6`, 也不能把单次低精度差异解释为精度本身的确定性增益.

## 2. 实验口径

RTX 2080 Ti 比较 FP32 与 AMP-FP16, RTX 3090 比较 FP32, AMP-FP16 与 AMP-BF16. 每个模型和 dtype 使用 seeds `123,124,125`, 固定 B64, GA4 和 4 epochs. Flash-VQG 仅在 grouped update 与 selected-read Triton core 外建立 FP32 boundary; GDN 使用与实验 dtype 匹配的 FLA kernel dtype.

主结果使用 matching train/eval dtype. Off-diagonal 网格只用于机制分析. 两张 GPU 分别计算 3 seeds 的 mean 与 population std, 不合并为 `n=6`.

全局 gate 绑定 Zoology `{global_gate['zoology_commit'][:12]}`, Flash-VQG `{global_gate['flash_commit'][:12]}` 和 cache `{global_gate['cache_content_sha256']}`. 两机 13 个 `shape x num_examples` 数据身份各自只有一个 dataset hash.

## 3. Matching dtype 主结果

下表是 last checkpoint 的 500-example longer-MQAR accuracy, 格式为 `mean ± population SD`, 每行 `n=3` seeds.

{matching_table(summary, 'last')}

关键观察:

- GDN 对训练精度近乎不敏感. 所有低精度相对 FP32 的五个 slice 均值变化绝对值不超过 `{facts['gdn_max_abs_delta']:.6f}`.
- Flash 在 2080 Ti 上使用 FP16 后, last accuracy 相对 FP32 的五个 slice 变化为 `{_range_text(facts['flash_2080_fp16'])}`. 在 RTX 3090 上, FP16 变化为 `{_range_text(facts['flash_3090_fp16'])}`, BF16 变化为 `{_range_text(facts['flash_3090_bf16'])}`. 方向随机器改变, 说明主要是训练轨迹和 GPU 数值路径敏感性, 不是统一的“低精度提升”或“低精度退化”.
- 在排除训练端点 `1024x256` 后的四个外推 slice 上, Flash 在 `{facts['paired_flash_wins']}/{facts['paired_total']}` 个 `GPU x matching dtype x seed x shape` 配对中高于 GDN. 在 `1024x256` 训练端点, Flash 仅在 `{facts['endpoint_flash_wins']}/{facts['endpoint_total']}` 个配对中高于 GDN, 因而端点不支持 Flash 优于 GDN.
- 对固定训练 checkpoint 只改变 eval dtype, accuracy 的全网格最大跨度为 `{facts['max_eval_dtype_range']:.6f}`. 这远小于 Flash 的主要训练精度和跨 GPU 差异, 说明外围 FP32 boundary 与低精度 evaluator 本身没有造成主要质量漂移.

Best checkpoint 图支持相同的定性结论. 2080 Ti 上 best 选择可缓解部分 Flash last 波动; RTX 3090 上多数 best/last state 相同并被物理去重. 完整 best 数值见汇总 CSV.

## 4. 训练效率与数值审计

{training_table(training)}

- Flash-FP16 的平均 wall time 相对 FP32 为 2080 Ti `{facts['flash_2080_fp16_wall_ratio']:.3f}x`, RTX 3090 `{facts['flash_3090_fp16_wall_ratio']:.3f}x`; RTX 3090 Flash-BF16 为 `{facts['flash_3090_bf16_wall_ratio']:.3f}x`. Flash 低精度 peak allocated memory 约为 FP32 的 `{facts['flash_low_alloc_ratio']:.3f}x`.
- GDN 低精度加速更明显: 2080 Ti FP16 为 `{facts['gdn_2080_fp16_wall_ratio']:.3f}x`, RTX 3090 FP16 为 `{facts['gdn_3090_fp16_wall_ratio']:.3f}x`, BF16 为 `{facts['gdn_3090_bf16_wall_ratio']:.3f}x`. GDN 低精度 peak allocated memory 约为 FP32 的 `{facts['gdn_low_alloc_ratio']:.3f}x`.
- 30 个 run 的 model master weights 和 optimizer state 均保持 FP32. GDN kernel dtype 分别严格为 `float32`, `float16`, `bfloat16`. 全实验只记录 `{facts['scaler_skips']}` 次 FP16 GradScaler skip, 位于 `3090-flash-s125-fp16`, 未超过预注册的每 run 上限 2, 该 run 最终正常完成 epoch 4 且指标有限.

## 5. 审计与证据

- 2080 Ti gate: 52/52 capacity profiles, 52/52 batch invariance, 312/312 eval smoke, 26/26 canary, 16/16 standard accuracy audit.
- RTX 3090 gate: 78/78 capacity profiles, 78/78 batch invariance, 702/702 eval smoke, 26/26 canary, 16/16 standard accuracy audit.
- 两机 `8190x2047` smoke 均实际执行 controlled interrupt 并从 batch cursor 恢复完成.
- 18 个 preflight/status/formal-detail/gate JSON 与30个resolved training config已镜像到artifact, 全部通过source/mirror SHA256一致性校验. 60个checkpoint大文件保留在source machine原路径, file SHA256记录于source manifest.
- 30 条正式训练记录写入独立 canonical training ledger; 780 条正式 longer-MQAR 逻辑评估写入 canonical eval ledger, 包含 source/eval dtype, 开始结束时间, wall time, GPU, batch, dataset/checkpoint hash 和物理去重状态.
- 本实验不覆盖历史 FP32 canonical ledger; 它是独立 precision profile. Matching dtype 为 official 主比较口径, off-diagonal 仅用于机制分析.

## 6. 产物

- Last 图: [matching-precision-last.pdf](artifacts/{EXPERIMENT_ID}/figures/matching-precision-last.pdf).
- Best 图: [matching-precision-best.pdf](artifacts/{EXPERIMENT_ID}/figures/matching-precision-best.pdf).
- 正式明细: [final.csv](artifacts/{EXPERIMENT_ID}/final.csv).
- 汇总: [precision-grid-summary.csv](artifacts/{EXPERIMENT_ID}/combined/precision-grid-summary.csv).
- 训练 ledger: [canonical-training-ledger.csv](artifacts/{EXPERIMENT_ID}/canonical-training-ledger.csv).
- Longer-MQAR ledger: [canonical-longer-mqar-ledger.csv](artifacts/{EXPERIMENT_ID}/canonical-longer-mqar-ledger.csv).
- Source manifest: [source-manifest.csv](artifacts/{EXPERIMENT_ID}/source-manifest.csv).
"""
    temporary = REPORT_PATH.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(REPORT_PATH)


def collect() -> dict[str, Any]:
    details = load_machine_details()
    training, evaluation = flatten(details)
    validate_counts(training, evaluation)
    summary = aggregate(evaluation)
    result_metadata = evaluation_result_metadata()
    training_ledger_rows = training_ledger(training)
    longer_ledger_rows = longer_mqar_ledger(
        training, evaluation, result_metadata
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    gate_evidence_rows = mirror_evidence()
    resolved_config_rows = mirror_resolved_configs(training)
    evidence_rows = gate_evidence_rows + resolved_config_rows
    write_csv(ARTIFACT_DIR / "training.csv", training)
    write_csv(ARTIFACT_DIR / "final.csv", evaluation)
    write_csv(
        ARTIFACT_DIR / "source-manifest.csv",
        source_manifest(training, evidence_rows),
    )
    write_csv(
        ARTIFACT_DIR / "combined" / "precision-grid-summary.csv",
        summary,
    )
    write_csv(
        ARTIFACT_DIR / "canonical-training-ledger.csv",
        training_ledger_rows,
    )
    write_csv(
        ARTIFACT_DIR / "canonical-longer-mqar-ledger.csv",
        longer_ledger_rows,
    )
    global_gate = load_json(
        output_root("2080ti") / "gates/GLOBAL_FORMAL_GATE.json"
    )
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "training_rows": len(training),
        "logical_evaluation_rows": len(evaluation),
        "physical_evaluation_rows": sum(
            row["status"] == "completed" for row in evaluation
        ),
        "deduplicated_evaluation_rows": sum(
            row["status"] == "deduplicated" for row in evaluation
        ),
        "mirrored_json_evidence_rows": len(evidence_rows),
        "mirrored_gate_status_rows": len(gate_evidence_rows),
        "mirrored_resolved_config_rows": len(resolved_config_rows),
        "checkpoint_manifest_rows": len(training) * 2,
        "canonical_training_ledger_rows": len(training_ledger_rows),
        "canonical_longer_mqar_ledger_rows": len(longer_ledger_rows),
        "statistics": "mean and population standard deviation over three seeds per machine",
        "gpu_pooling": "disabled",
        "zoology_commit": global_gate["zoology_commit"],
        "flash_commit": global_gate["flash_commit"],
        "cache_content_sha256": global_gate["cache_content_sha256"],
        "global_gate_binding_sha256": global_gate["binding_sha256"],
        "collected_at_utc": utc_now(),
    }
    atomic_write_json(ARTIFACT_DIR / "metadata.json", metadata)
    make_figure(summary, "last")
    make_figure(summary, "best")
    write_report(training, evaluation, summary, evidence_rows)
    return metadata


if __name__ == "__main__":
    print(json.dumps(collect(), ensure_ascii=False, indent=2))
