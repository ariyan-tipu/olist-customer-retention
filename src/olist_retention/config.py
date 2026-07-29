"""Typed configuration loaded from conf/pipeline.yaml.

Every knob the pipeline exposes lives in one YAML file and is parsed into
frozen dataclasses, so a bad value fails loudly at start-up rather than
halfway through a Spark job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("conf") / "pipeline.yaml"


def _as_date(value: Any, name: str) -> date:
    """Coerce a YAML value to a date, accepting ISO strings defensively."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"{name} must be an ISO date, got {value!r}")


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    staging_dir: Path
    features_dir: Path
    outputs_dir: Path
    kaggle_handle: str


@dataclass(frozen=True)
class SparkConfig:
    master: str
    driver_memory: str
    shuffle_partitions: int
    app_name: str

    def __post_init__(self) -> None:
        if self.shuffle_partitions < 1:
            raise ValueError("shuffle_partitions must be positive")


@dataclass(frozen=True)
class LabellingConfig:
    train_cutoff: date
    test_cutoff: date
    horizon_days: int

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.test_cutoff <= self.train_cutoff:
            raise ValueError("test_cutoff must be after train_cutoff")


@dataclass(frozen=True)
class SurvivalConfig:
    observation_end: date


@dataclass(frozen=True)
class ModellingConfig:
    seed: int
    cv_folds: int
    model_dir: Path

    def __post_init__(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")


@dataclass(frozen=True)
class ABTestConfig:
    alpha: float
    target_power: float
    relative_uplifts: tuple[float, ...]
    simulations: int
    simulation_seed: int

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < self.target_power < 1:
            raise ValueError("target_power must be in (0, 1)")


@dataclass(frozen=True)
class ServiceConfig:
    model_path: Path


@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration object handed around the pipeline."""

    root: Path
    data: DataConfig
    spark: SparkConfig
    labelling: LabellingConfig
    survival: SurvivalConfig
    modelling: ModellingConfig
    ab_test: ABTestConfig
    service: ServiceConfig
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path | None = None, root: Path | None = None) -> PipelineConfig:
        """Load configuration, resolving relative paths against the repo root.

        Args:
            config_path: YAML file to read. Defaults to conf/pipeline.yaml
                under the root.
            root: Repository root used to resolve relative paths. Defaults
                to the parent of the config file's directory.
        """
        if config_path is None:
            base = root if root is not None else Path.cwd()
            config_path = base / DEFAULT_CONFIG_PATH
        config_path = config_path.resolve()
        if root is None:
            root = config_path.parent.parent
        root = root.resolve()

        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        def path_of(section: dict[str, Any], key: str) -> Path:
            return (root / Path(section[key])).resolve()

        data = raw["data"]
        labelling = raw["labelling"]
        modelling = raw["modelling"]
        ab_test = raw["ab_test"]

        return cls(
            root=root,
            data=DataConfig(
                raw_dir=path_of(data, "raw_dir"),
                staging_dir=path_of(data, "staging_dir"),
                features_dir=path_of(data, "features_dir"),
                outputs_dir=path_of(data, "outputs_dir"),
                kaggle_handle=str(data["kaggle_handle"]),
            ),
            spark=SparkConfig(
                master=str(raw["spark"]["master"]),
                driver_memory=str(raw["spark"]["driver_memory"]),
                shuffle_partitions=int(raw["spark"]["shuffle_partitions"]),
                app_name=str(raw["spark"]["app_name"]),
            ),
            labelling=LabellingConfig(
                train_cutoff=_as_date(labelling["train_cutoff"], "train_cutoff"),
                test_cutoff=_as_date(labelling["test_cutoff"], "test_cutoff"),
                horizon_days=int(labelling["horizon_days"]),
            ),
            survival=SurvivalConfig(
                observation_end=_as_date(raw["survival"]["observation_end"], "observation_end"),
            ),
            modelling=ModellingConfig(
                seed=int(modelling["seed"]),
                cv_folds=int(modelling["cv_folds"]),
                model_dir=path_of(modelling, "model_dir"),
            ),
            ab_test=ABTestConfig(
                alpha=float(ab_test["alpha"]),
                target_power=float(ab_test["target_power"]),
                relative_uplifts=tuple(float(u) for u in ab_test["relative_uplifts"]),
                simulations=int(ab_test["simulations"]),
                simulation_seed=int(ab_test["simulation_seed"]),
            ),
            service=ServiceConfig(
                model_path=path_of(raw["service"], "model_path"),
            ),
        )
