"""Configuration loading and validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from olist_retention.config import ABTestConfig, LabellingConfig, PipelineConfig

REPO_ROOT = Path(__file__).parents[1]


def test_default_config_loads_and_is_typed():
    config = PipelineConfig.load(config_path=REPO_ROOT / "conf" / "pipeline.yaml")

    assert config.labelling.train_cutoff == date(2018, 1, 1)
    assert config.labelling.test_cutoff == date(2018, 4, 1)
    assert config.labelling.horizon_days == 180
    assert config.survival.observation_end == date(2018, 10, 17)
    assert config.spark.shuffle_partitions == 16
    assert config.data.raw_dir.is_absolute()
    assert config.service.model_path.name == "churn_model.joblib"


def test_cutoff_ordering_is_enforced():
    with pytest.raises(ValueError, match="test_cutoff"):
        LabellingConfig(
            train_cutoff=date(2018, 4, 1), test_cutoff=date(2018, 1, 1), horizon_days=180
        )


def test_alpha_bounds_are_enforced():
    with pytest.raises(ValueError, match="alpha"):
        ABTestConfig(
            alpha=1.5,
            target_power=0.8,
            relative_uplifts=(0.1,),
            simulations=100,
            simulation_seed=1,
        )
