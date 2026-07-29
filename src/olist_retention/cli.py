"""Command line entry points for every stage of the pipeline.

Heavy dependencies are imported inside the handlers, so ``--help`` is
instant and the scoring container (which has no Spark) can still use the
parts it needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from olist_retention.config import DEFAULT_CONFIG_PATH, PipelineConfig
from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = PipelineConfig.load(config_path=args.config)
    handler = {
        "download": _run_download,
        "synthetic": _run_synthetic,
        "pipeline": _run_pipeline,
        "train": _run_train,
        "power": _run_power,
        "serve": _run_serve,
    }[args.command]
    return handler(args, config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="olist-retention",
        description="Customer retention pipeline on the Olist dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the pipeline YAML (default: conf/pipeline.yaml)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("download", help="Fetch the Olist dataset from Kaggle")

    synthetic = commands.add_parser(
        "synthetic", help="Generate the synthetic fixture dataset instead of downloading"
    )
    synthetic.add_argument("--dest", type=Path, default=None, help="Target directory")
    synthetic.add_argument("--customers", type=int, default=800)
    synthetic.add_argument("--seed", type=int, default=7)

    pipeline = commands.add_parser(
        "pipeline", help="Stage raw data, build both feature snapshots and the survival export"
    )
    pipeline.add_argument(
        "--skip-validation", action="store_true", help="Skip the raw data validation report"
    )

    commands.add_parser("train", help="Train, compare and persist the churn models")

    power = commands.add_parser("power", help="Run the A/B test power analysis")
    power.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Baseline repeat rate; defaults to the measured rate from the test snapshot",
    )

    serve = commands.add_parser("serve", help="Run the scoring API locally")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------


def _run_download(args: argparse.Namespace, config: PipelineConfig) -> int:
    from olist_retention.data.download import DatasetDownloader

    DatasetDownloader(config.data.kaggle_handle, config.data.raw_dir).fetch()
    return 0


def _run_synthetic(args: argparse.Namespace, config: PipelineConfig) -> int:
    from olist_retention.data.synthetic import SyntheticOlist, SyntheticParams

    dest = args.dest or config.data.raw_dir
    params = SyntheticParams(n_customers=args.customers, seed=args.seed)
    SyntheticOlist(params).write_csvs(dest)
    logger.info("Wrote synthetic dataset (%d customers) to %s", args.customers, dest)
    return 0


def _run_pipeline(args: argparse.Namespace, config: PipelineConfig) -> int:
    from olist_retention.pipeline.features import FeatureBuilder
    from olist_retention.pipeline.session import build_spark_session
    from olist_retention.pipeline.staging import RawToStaging
    from olist_retention.pipeline.survival import SurvivalDatasetBuilder

    spark = build_spark_session(config.spark)
    try:
        RawToStaging(spark, config.data).run(validate=not args.skip_validation)
        for cutoff in (config.labelling.train_cutoff, config.labelling.test_cutoff):
            FeatureBuilder(spark, config.data, cutoff, config.labelling.horizon_days).run()
        SurvivalDatasetBuilder(spark, config.data, config.survival.observation_end).run()
    finally:
        spark.stop()
    return 0


def _run_train(args: argparse.Namespace, config: PipelineConfig) -> int:
    from olist_retention.models.train import ChurnModelTrainer

    trainer = ChurnModelTrainer(
        modelling=config.modelling,
        features_dir=config.data.features_dir,
        outputs_dir=config.data.outputs_dir,
        horizon_days=config.labelling.horizon_days,
    )
    result = trainer.run(config.labelling.train_cutoff, config.labelling.test_cutoff)
    logger.info("Winner: %s (metrics at %s)", result.winner, result.metrics_path)
    return 0


def _run_power(args: argparse.Namespace, config: PipelineConfig) -> int:
    from olist_retention.experiments.power import TwoProportionPower

    baseline = args.baseline
    if baseline is None:
        summary_path = (
            config.data.outputs_dir
            / f"features_summary_{config.labelling.test_cutoff.isoformat()}.json"
        )
        if not summary_path.is_file():
            logger.error(
                "No measured baseline available at %s; run the pipeline first or pass --baseline",
                summary_path,
            )
            return 1
        baseline = json.loads(summary_path.read_text(encoding="utf-8"))["base_rate"]
        logger.info("Using measured baseline repeat rate %.4f%%", 100 * baseline)

    analysis = TwoProportionPower(config.ab_test.alpha, config.ab_test.target_power)
    scenarios = analysis.run_grid(
        baseline_rate=baseline,
        relative_uplifts=config.ab_test.relative_uplifts,
        replications=config.ab_test.simulations,
        seed=config.ab_test.simulation_seed,
    )
    analysis.write_report(scenarios, config.data.outputs_dir / "ab_test")
    return 0


def _run_serve(args: argparse.Namespace, config: PipelineConfig) -> int:
    import uvicorn

    from olist_retention.service.app import create_app

    app = create_app(model_path=str(config.service.model_path))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
