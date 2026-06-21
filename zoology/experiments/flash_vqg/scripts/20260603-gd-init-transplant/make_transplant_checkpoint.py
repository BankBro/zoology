from __future__ import annotations

import argparse
import json

from experiment_lib import (
    ARTIFACT_DIR,
    ensure_artifact_dirs,
    make_transplant_checkpoint,
    validate_scope_boundaries,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成可 strict 加载的 init-transplant checkpoint.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--recipient-target", required=True)
    parser.add_argument("--donor-snapshot", required=True)
    parser.add_argument(
        "--overlay-scope",
        required=True,
        choices=["full_model", "flash_only", "non_flash_only"],
    )
    parser.add_argument("--max-epochs", type=int, default=4)
    args = parser.parse_args()

    validate_scope_boundaries()
    ensure_artifact_dirs()
    checkpoint_path, config, metadata = make_transplant_checkpoint(
        name=args.name,
        recipient_target=args.recipient_target,
        donor_snapshot_path=args.donor_snapshot,
        overlay_scope=args.overlay_scope,
        max_epochs=int(args.max_epochs),
        run_id=args.name,
    )
    write_json(
        ARTIFACT_DIR / "last-transplant-checkpoint.json",
        {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "run_id": config.run_id,
            "metadata": metadata,
        },
    )
    print(json.dumps({"checkpoint_path": str(checkpoint_path.resolve()), "run_id": config.run_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
