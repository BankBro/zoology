from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiment_lib import (
    ARTIFACT_DIR,
    TARGET_SPECS,
    build_config,
    ensure_artifact_dirs,
    initialized_model_and_state,
    make_transplant_checkpoint,
    run_launch,
    save_snapshot,
    snapshot_path,
    validate_scope_boundaries,
    write_csv,
    write_generated_launch,
    write_json,
)


def _launch_id(mode: str, matrix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S")
    return f"flash-vqg-20260603-gd-init-transplant-{mode}-{matrix}-{stamp}"


def _normal(run_id: str, target: str, *, max_epochs: int, smoke_data: bool) -> dict[str, Any]:
    return {
        "kind": "normal",
        "run_id": run_id,
        "target": target,
        "max_epochs": max_epochs,
        "smoke_data": smoke_data,
        "experiment_mode": run_id,
    }


def _full_rerun(run_id: str, target: str, *, max_epochs: int, smoke_data: bool) -> dict[str, Any]:
    return {
        "kind": "full_model_rerun",
        "run_id": run_id,
        "target": target,
        "max_epochs": max_epochs,
        "smoke_data": smoke_data,
        "experiment_mode": run_id,
        "init_checkpoint_path": str(snapshot_path(target, "full_model").resolve()),
        "init_source_name": f"{target}-full_model",
        "donor_target": target,
        "overlay_scope": "full_model",
    }


def _transplant(
    run_id: str,
    *,
    donor_target: str,
    recipient_target: str,
    overlay_scope: str,
    max_epochs: int,
    smoke_data: bool,
) -> dict[str, Any]:
    donor_snapshot = snapshot_path(donor_target, overlay_scope)
    checkpoint_path, _, metadata = make_transplant_checkpoint(
        name=run_id,
        recipient_target=recipient_target,
        donor_snapshot_path=donor_snapshot,
        overlay_scope=overlay_scope,
        max_epochs=max_epochs,
        run_id=run_id,
    )
    return {
        "kind": f"{overlay_scope}_transplant",
        "run_id": run_id,
        "target": recipient_target,
        "max_epochs": max_epochs,
        "smoke_data": smoke_data,
        "experiment_mode": run_id,
        "init_checkpoint_path": str(checkpoint_path.resolve()),
        "init_source_name": run_id,
        "donor_target": donor_target,
        "recipient_target": recipient_target,
        "overlay_scope": overlay_scope,
        "overlay_sha256": metadata["overlay_sha256"],
        "full_model_sha256": metadata["full_model_sha256"],
    }


def _required_snapshots(matrix: str) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {
        "cb64-r16-s124": {"full_model", "flash_only"},
        "cb64-r16-s125": {"full_model", "flash_only", "non_flash_only"},
        "cb256-r4-s123": {"full_model", "flash_only"},
        "cb256-r4-s124": {"full_model", "flash_only"},
    }
    if matrix == "extended":
        required.setdefault("cb256-r4-s125", set()).update({"full_model", "flash_only"})
    return required


def _ensure_snapshots(required: dict[str, set[str]], *, max_epochs: int) -> None:
    for target, scopes in required.items():
        missing = [scope for scope in sorted(scopes) if not snapshot_path(target, scope).exists()]
        if not missing:
            continue
        config = build_config(target, max_epochs=max_epochs, run_id=f"snapshot-{target}")
        _, state = initialized_model_and_state(config)
        for scope in missing:
            path = save_snapshot(target=target, scope=scope, config=config, state=state)
            print(f"created missing snapshot {target} {scope}: {path}", flush=True)


def build_matrix(*, matrix: str, max_epochs: int, smoke_data: bool) -> list[dict[str, Any]]:
    if matrix not in {"core", "extended"}:
        raise ValueError("matrix 只支持 core 或 extended.")
    specs: list[dict[str, Any]] = [
        _normal("normal-cb64-r16-s124", "cb64-r16-s124", max_epochs=max_epochs, smoke_data=smoke_data),
        _normal("normal-cb64-r16-s125", "cb64-r16-s125", max_epochs=max_epochs, smoke_data=smoke_data),
        _full_rerun("fullrerun-cb64-r16-s125", "cb64-r16-s125", max_epochs=max_epochs, smoke_data=smoke_data),
        _transplant(
            "flashdonor-cb64-r16-s125-to-s124",
            donor_target="cb64-r16-s125",
            recipient_target="cb64-r16-s124",
            overlay_scope="flash_only",
            max_epochs=max_epochs,
            smoke_data=smoke_data,
        ),
        _transplant(
            "flashdonor-cb64-r16-s124-to-s125",
            donor_target="cb64-r16-s124",
            recipient_target="cb64-r16-s125",
            overlay_scope="flash_only",
            max_epochs=max_epochs,
            smoke_data=smoke_data,
        ),
        _transplant(
            "nonflashdonor-cb64-r16-s125-to-s124",
            donor_target="cb64-r16-s125",
            recipient_target="cb64-r16-s124",
            overlay_scope="non_flash_only",
            max_epochs=max_epochs,
            smoke_data=smoke_data,
        ),
        _normal("normal-cb256-r4-s123", "cb256-r4-s123", max_epochs=max_epochs, smoke_data=smoke_data),
        _normal("normal-cb256-r4-s124", "cb256-r4-s124", max_epochs=max_epochs, smoke_data=smoke_data),
        _transplant(
            "flashdonor-cb256-r4-s123-to-s124",
            donor_target="cb256-r4-s123",
            recipient_target="cb256-r4-s124",
            overlay_scope="flash_only",
            max_epochs=max_epochs,
            smoke_data=smoke_data,
        ),
    ]
    if matrix == "extended":
        specs.extend(
            [
                _normal("normal-cb256-r4-s125", "cb256-r4-s125", max_epochs=max_epochs, smoke_data=smoke_data),
                _full_rerun("fullrerun-cb256-r4-s123", "cb256-r4-s123", max_epochs=max_epochs, smoke_data=smoke_data),
                _transplant(
                    "flashdonor-cb256-r4-s124-to-s123",
                    donor_target="cb256-r4-s124",
                    recipient_target="cb256-r4-s123",
                    overlay_scope="flash_only",
                    max_epochs=max_epochs,
                    smoke_data=smoke_data,
                ),
            ]
        )
    return specs


def _matrix_rows(run_specs: list[dict[str, Any]], *, launch_id: str, mode: str, matrix: str) -> list[dict[str, Any]]:
    rows = []
    for item in run_specs:
        target = str(item["target"])
        spec = TARGET_SPECS[target]
        rows.append(
            {
                "launch_id": launch_id,
                "mode": mode,
                "matrix": matrix,
                "run_id": item["run_id"],
                "kind": item["kind"],
                "target": target,
                "target_label": spec.label,
                "seed": spec.seed,
                "num_codebook_vectors": spec.num_codebook_vectors,
                "gd_rank": spec.gd_rank,
                "donor_target": item.get("donor_target"),
                "recipient_target": item.get("recipient_target"),
                "overlay_scope": item.get("overlay_scope"),
                "init_checkpoint_path": item.get("init_checkpoint_path"),
                "max_epochs": item.get("max_epochs"),
                "smoke_data": item.get("smoke_data"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 init-transplant 最小矩阵配置.")
    parser.add_argument("--mode", choices=["early", "train"], default="early")
    parser.add_argument("--matrix", choices=["core", "extended"], default="core")
    parser.add_argument("--launch-id", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--parallelize", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    validate_scope_boundaries()
    ensure_artifact_dirs()
    max_epochs = 1 if args.mode == "early" else 4
    smoke_data = args.mode == "early"
    required = _required_snapshots(args.matrix)
    _ensure_snapshots(required, max_epochs=max_epochs)
    run_specs = build_matrix(matrix=args.matrix, max_epochs=max_epochs, smoke_data=smoke_data)
    launch_id = args.launch_id or _launch_id(args.mode, args.matrix)
    generated_path = write_generated_launch(launch_id=launch_id, run_specs=run_specs)

    run_specs_path = generated_path.parent / "run_specs.json"
    write_json(run_specs_path, {"launch_id": launch_id, "mode": args.mode, "matrix": args.matrix, "runs": run_specs})
    matrix_path = ARTIFACT_DIR / f"{args.mode}-{args.matrix}-matrix.csv"
    write_csv(matrix_path, _matrix_rows(run_specs, launch_id=launch_id, mode=args.mode, matrix=args.matrix))
    write_json(
        ARTIFACT_DIR / f"{args.mode}-{args.matrix}-status.json",
        {
            "status": "generated",
            "launch_id": launch_id,
            "generated_path": str(generated_path.resolve()),
            "run_specs_path": str(run_specs_path.resolve()),
            "matrix_path": str(matrix_path.resolve()),
            "num_runs": len(run_specs),
            "launch_requested": bool(args.launch),
        },
    )
    print(json.dumps({"launch_id": launch_id, "generated_path": str(generated_path.resolve()), "num_runs": len(run_specs)}, ensure_ascii=False))
    if args.launch:
        run_launch(
            generated_path=generated_path,
            launch_id=launch_id,
            gpus=str(args.gpus),
            parallelize=bool(args.parallelize),
        )


if __name__ == "__main__":
    main()
