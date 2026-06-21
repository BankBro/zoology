from __future__ import annotations

import argparse
from pathlib import Path

from experiment_lib import (
    ARTIFACT_DIR,
    TARGET_SPECS,
    build_config,
    ensure_artifact_dirs,
    initialized_model_and_state,
    parse_scopes,
    parse_targets,
    save_snapshot,
    validate_scope_boundaries,
    write_csv,
    write_init_path_notes,
    write_json,
)


DEFAULT_TARGETS = "cb64-r16-s124,cb64-r16-s125,cb256-r4-s123,cb256-r4-s124,cb256-r4-s125"
DEFAULT_SCOPES = "full_model,flash_only,non_flash_only"


def main() -> None:
    parser = argparse.ArgumentParser(description="保存 Flash-VQG gd_residual_v1 初始化快照.")
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--scopes", default=DEFAULT_SCOPES)
    parser.add_argument("--max-epochs", type=int, default=4)
    args = parser.parse_args()

    validate_scope_boundaries()
    ensure_artifact_dirs()
    write_init_path_notes()

    rows: list[dict[str, object]] = []
    for target in parse_targets(args.targets):
        config = build_config(target, max_epochs=int(args.max_epochs), run_id=f"snapshot-{target}")
        _, state = initialized_model_and_state(config)
        spec = TARGET_SPECS[target]
        for scope in parse_scopes(args.scopes):
            path = save_snapshot(target=target, scope=scope, config=config, state=state)
            rows.append(
                {
                    "target": target,
                    "label": spec.label,
                    "seed": spec.seed,
                    "num_codebook_vectors": spec.num_codebook_vectors,
                    "gd_rank": spec.gd_rank,
                    "scope": scope,
                    "snapshot_path": str(path.resolve()),
                }
            )
            print(f"saved {target} {scope}: {path}", flush=True)

    manifest_path = ARTIFACT_DIR / "init-transplant-source-manifest.csv"
    write_csv(manifest_path, rows)
    write_json(
        ARTIFACT_DIR / "snapshot-status.json",
        {
            "status": "snapshots_saved",
            "manifest_path": str(manifest_path.resolve()),
            "num_rows": len(rows),
        },
    )
    print(f"manifest: {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
