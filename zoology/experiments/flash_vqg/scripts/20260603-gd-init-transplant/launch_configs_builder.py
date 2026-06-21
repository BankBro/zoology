from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment_lib import apply_init_checkpoint, build_config


def build_configs_from_specs(run_specs: list[dict[str, Any]]):
    configs = []
    for item in run_specs:
        target = str(item["target"])
        run_id = str(item["run_id"])
        max_epochs = int(item.get("max_epochs", 4))
        smoke_data = bool(item.get("smoke_data", False))
        config = build_config(
            target,
            max_epochs=max_epochs,
            train_batch_size=int(item.get("train_batch_size", 64)),
            eval_batch_size=int(item.get("eval_batch_size", 16)),
            gradient_accumulation_steps=int(item.get("gradient_accumulation_steps", 4)),
            validations_per_epoch=int(item.get("validations_per_epoch", 2)),
            run_id=run_id,
            experiment_mode=str(item.get("experiment_mode", run_id)),
            smoke_data=smoke_data,
        )
        init_checkpoint_path = item.get("init_checkpoint_path")
        if init_checkpoint_path:
            config = apply_init_checkpoint(
                config,
                Path(str(init_checkpoint_path)),
                source_name=str(item.get("init_source_name", run_id)),
            )
        configs.append(config)
    return configs
