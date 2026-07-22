
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import SyntheticBenchmarkDataset
from .config import PUDCGPConfig
from .gp_evaluation import cross_validate_gp_models
from .mean_baselines import run_mean_targets


@dataclass(frozen=True, slots=True)
class BenchmarkPredictionMetrics:

    method_name: str
    mean_rmse_by_outcome: Mapping[str, float]
    wasserstein_rmse_by_outcome: Mapping[str, float]
    normalized_mean_rmse: float
    normalized_wasserstein_rmse: float | None


def evaluate_benchmark_predictions(
    dataset: SyntheticBenchmarkDataset,
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
    n_folds: int = 5,
) -> Mapping[str, BenchmarkPredictionMetrics]:

    validation = cross_validate_gp_models(dataset.runs, config, n_folds=n_folds)
    outcome_names, observed_means = run_mean_targets(dataset.runs)
    outcome_scales = {
        mechanism.outcome: mechanism.baseline_scale
        for mechanism in contract.mechanisms
    }
    mean_rmse = {
        outcome: validation.mean_gp.metrics[outcome].rmse
        for outcome in outcome_names
    }
    results = {
        "mean_gp": BenchmarkPredictionMetrics(
            method_name="mean_gp",
            mean_rmse_by_outcome=mean_rmse,
            wasserstein_rmse_by_outcome={},
            normalized_mean_rmse=float(
                np.mean(
                    [
                        mean_rmse[outcome] / outcome_scales[outcome]
                        for outcome in outcome_names
                    ]
                )
            ),
            normalized_wasserstein_rmse=None,
        )
    }
    weights = _trapezoid_weights(np.asarray(contract.quantile_grid, dtype=float))
    for method_name, validation_name in (
        ("distribution_gp_no_pu", "distribution_gp"),
        ("pu_dcgp", "pu_dcgp"),
    ):
        distribution = validation.distribution_models[validation_name]
        method_mean_rmse = {}
        wasserstein_rmse = {}
        for outcome_index, outcome in enumerate(outcome_names):
            predicted_means = distribution.quantile_predictions[outcome] @ weights
            method_mean_rmse[outcome] = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            observed_means[:, outcome_index] - predicted_means
                        )
                    )
                )
            )
            wasserstein_rmse[outcome] = distribution.metrics[
                outcome
            ].wasserstein_rmse
        results[method_name] = BenchmarkPredictionMetrics(
            method_name=method_name,
            mean_rmse_by_outcome=method_mean_rmse,
            wasserstein_rmse_by_outcome=wasserstein_rmse,
            normalized_mean_rmse=float(
                np.mean(
                    [
                        method_mean_rmse[outcome] / outcome_scales[outcome]
                        for outcome in outcome_names
                    ]
                )
            ),
            normalized_wasserstein_rmse=float(
                np.mean(
                    [
                        wasserstein_rmse[outcome] / outcome_scales[outcome]
                        for outcome in outcome_names
                    ]
                )
            ),
        )
    pu = results["pu_dcgp"]
    results["support_gated_pu_dcgp"] = BenchmarkPredictionMetrics(
        method_name="support_gated_pu_dcgp",
        mean_rmse_by_outcome=pu.mean_rmse_by_outcome,
        wasserstein_rmse_by_outcome=pu.wasserstein_rmse_by_outcome,
        normalized_mean_rmse=pu.normalized_mean_rmse,
        normalized_wasserstein_rmse=pu.normalized_wasserstein_rmse,
    )
    return results


def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    differences = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights / weights.sum()
