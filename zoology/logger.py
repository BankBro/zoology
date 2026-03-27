from __future__ import annotations

from typing import Any, Protocol

from torch.nn import Module

from zoology.config import TrainConfig


LoggerSummary = dict[str, Any]


class LoggerProtocol(Protocol):
    def log_config(self, config: TrainConfig):
        ...

    def log_model(self, model: Module, config: TrainConfig):
        ...

    def log(self, metrics: dict, *, step: int | None = None):
        ...

    def finish(self):
        ...

    def get_summary(self) -> LoggerSummary:
        ...


def _build_summary(
    *,
    backend: str,
    enabled: bool,
    project: str | None = None,
    entity: str | None = None,
    run_id: str | None = None,
    run_name: str | None = None,
    run_url: str | None = None,
    run_dir: str | None = None,
    backup_file: str | None = None,
    config_file: str | None = None,
    metadata_file: str | None = None,
) -> LoggerSummary:
    return {
        "backend": backend,
        "enabled": enabled,
        "project": project,
        "entity": entity,
        "run_id": run_id,
        "run_name": run_name,
        "run_url": run_url,
        "run_dir": run_dir,
        "backup_file": backup_file,
        "config_file": config_file,
        "metadata_file": metadata_file,
    }


def _model_summary_metrics(model: Module, config: TrainConfig) -> dict[str, Any]:
    max_seq_len = max(c.input_seq_len for c in config.data.test_configs)
    return {
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_size": model.state_size(sequence_length=max_seq_len),
    }


class WandbLogger:
    backend = "wandb"

    def __init__(self, config: TrainConfig):
        self.config = config
        self.no_logger = False
        self.run = None
        self._wandb = None

        if config.logger.project_name is None or config.logger.entity is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return

        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "backend=wandb requires the `wandb` package to be installed."
            ) from e

        self._wandb = wandb
        self.run = wandb.init(
            name=config.run_id,
            entity=config.logger.entity,
            project=config.logger.project_name,
        )

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        self.run.config.update(config.model_dump(), allow_val_change=True)

    def log_model(self, model: Module, config: TrainConfig):
        if self.no_logger:
            return
        self._wandb.log(_model_summary_metrics(model, config))
        self._wandb.watch(model)

    def log(self, metrics: dict, *, step: int | None = None):
        if self.no_logger:
            return
        self._wandb.log(metrics, step=step)

    def finish(self):
        if self.no_logger or self.run is None:
            return
        self.run.finish()

    def get_summary(self) -> LoggerSummary:
        if self.no_logger or self.run is None:
            return _build_summary(
                backend=self.backend,
                enabled=False,
                entity=self.config.logger.entity,
                project=self.config.logger.project_name,
            )

        project = None
        project_attr = getattr(self.run, "project", None)
        if callable(project_attr):
            try:
                project = project_attr()
            except Exception:
                project = None
        elif isinstance(project_attr, str):
            project = project_attr

        return _build_summary(
            backend=self.backend,
            enabled=True,
            project=project,
            entity=getattr(self.run, "entity", None),
            run_id=getattr(self.run, "id", None),
            run_name=getattr(self.run, "name", None),
            run_url=getattr(self.run, "url", None),
        )


class SwanLabLogger:
    backend = "swanlab"

    def __init__(self, config: TrainConfig):
        self.config = config
        self.run = None
        self._swanlab = None

        if config.logger.project_name is None or config.logger.entity is None:
            raise ValueError("backend=swanlab requires both logger.project_name and logger.entity.")

        try:
            import swanlab
        except ImportError as e:
            raise ImportError(
                "backend=swanlab requires the `swanlab` package to be installed."
            ) from e

        self._swanlab = swanlab
        self.run = swanlab.init(
            project=config.logger.project_name,
            workspace=config.logger.entity,
            experiment_name=config.run_id,
        )

    def log_config(self, config: TrainConfig):
        self._swanlab.config.update(config.model_dump())

    def log_model(self, model: Module, config: TrainConfig):
        self._swanlab.log(_model_summary_metrics(model, config))

    def log(self, metrics: dict, *, step: int | None = None):
        self._swanlab.log(metrics, step=step)

    def finish(self):
        if self.run is None:
            return
        self._swanlab.finish()

    def get_summary(self) -> LoggerSummary:
        if self.run is None:
            return _build_summary(
                backend=self.backend,
                enabled=False,
                project=self.config.logger.project_name,
                entity=self.config.logger.entity,
            )

        public = getattr(self.run, "public", None)
        cloud = getattr(public, "cloud", None) if public is not None else None
        project = getattr(public, "project_name", None)
        run_id = getattr(public, "run_id", None) or getattr(self.run, "id", None)
        run_name = getattr(cloud, "experiment_name", None) or self.config.run_id
        run_url = getattr(cloud, "experiment_url", None)
        run_dir = getattr(public, "run_dir", None)
        backup_file = getattr(public, "backup_file", None)
        config_file = None if run_dir is None else f"{run_dir}/files/config.yaml"
        metadata_file = None if run_dir is None else f"{run_dir}/files/swanlab-metadata.json"
        return _build_summary(
            backend=self.backend,
            enabled=True,
            project=project or self.config.logger.project_name,
            entity=self.config.logger.entity,
            run_id=run_id,
            run_name=run_name,
            run_url=run_url,
            run_dir=run_dir,
            backup_file=backup_file,
            config_file=config_file,
            metadata_file=metadata_file,
        )


class NoOpLogger:
    backend = "none"

    def log_config(self, config: TrainConfig):
        return

    def log_model(self, model: Module, config: TrainConfig):
        return

    def log(self, metrics: dict, *, step: int | None = None):
        return

    def finish(self):
        return

    def get_summary(self) -> LoggerSummary:
        return _build_summary(backend=self.backend, enabled=False)


def build_logger(config: TrainConfig) -> LoggerProtocol:
    backend = config.logger.backend
    if backend == "wandb":
        return WandbLogger(config)
    if backend == "swanlab":
        return SwanLabLogger(config)
    if backend == "none":
        return NoOpLogger()
    raise ValueError(f"Unsupported logger backend: {backend}")
