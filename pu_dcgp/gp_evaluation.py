"""Grouped out-of-setting evaluation for the Gaussian-process layer."""

from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import FloatArray, PreparedData, RunBatch
from .data_source import subset_run_batch
from .distribution_encoder import BootstrapWassersteinFPCAEncoder
from .gaussian_process import (
    GaussianProcessDistributionModel,
    GaussianProcessMeanModel,
)
from .mean_baselines import (
    MeanBaselineCVResult,
    grouped_setting_folds,
    outcome_metrics,
    run_mean_targets,
)


@dataclass(frozen=True, slots=True)
class DistributionMetrics:
    quantile_mae: float
    wasserstein_rmse: float


@dataclass(frozen=True, slots=True)
class DistributionCVResult:
    model_name: str
    quantile_predictions: dict[str, FloatArray]
    quantile_variances: dict[str, FloatArray]
    metrics: dict[str, DistributionMetrics]


@dataclass(frozen=True, slots=True)
class GPValidationResult:
    fold_ids: np.ndarray
    mean_gp: MeanBaselineCVResult
    distribution_models: dict[str, DistributionCVResult]


def cross_validate_gp_models(
    runs: RunBatch,
    config: PUDCGPConfig,
    n_folds: int = 5,
) -> GPValidationResult:
    """Evaluate mean GP, score GP, and PU-DCGP on identical setting-grouped folds."""

    folds = grouped_setting_folds(runs.treatment_values, n_folds)
    all_indices = np.arange(len(runs.run_ids))
    fold_ids = np.empty(len(runs.run_ids), dtype=int)
    outcome_names, mean_targets = run_mean_targets(runs)
    mean_predictions = np.empty_like(mean_targets)
    mean_variances = np.empty_like(mean_targets)
    quantile_grid = np.asarray(config.quantile_grid, dtype=float)
    true_quantiles = {
        outcome: np.empty((len(runs.run_ids), len(quantile_grid)))
        for outcome in outcome_names
    }
    model_names = (
        "distribution_fold_mean",
        "distribution_gp",
        "pu_dcgp",
    )
    quantile_predictions = {
        model_name: {
            outcome: np.empty_like(true_quantiles[outcome])
            for outcome in outcome_names
        }
        for model_name in model_names
    }
    quantile_variances = {
        model_name: {
            outcome: np.zeros_like(true_quantiles[outcome])
            for outcome in outcome_names
        }
        for model_name in model_names
    }

    for fold_index, test_indices in enumerate(folds):
        train_indices = np.setdiff1d(all_indices, test_indices)
        fold_ids[test_indices] = fold_index
        train_runs = subset_run_batch(runs, train_indices)
        test_runs = subset_run_batch(runs, test_indices)

        mean_model = GaussianProcessMeanModel(config)
        mean_model.fit(train_runs)
        fold_means, fold_variances = mean_model.predict(
            test_runs.treatment_values,
            test_runs.context_values,
        )
        mean_predictions[test_indices] = fold_means
        mean_variances[test_indices] = fold_variances

        encoder = BootstrapWassersteinFPCAEncoder(config)
        train_representation = encoder.fit_transform(train_runs)
        train_quantiles = encoder.empirical_quantiles(train_runs)
        test_quantiles = encoder.empirical_quantiles(test_runs)
        for outcome in outcome_names:
            true_quantiles[outcome][test_indices] = test_quantiles[outcome]
            quantile_predictions["distribution_fold_mean"][outcome][
                test_indices
            ] = train_quantiles[outcome].mean(axis=0)

        prepared = PreparedData(
            runs=train_runs,
            distributions=train_representation,
        )
        for model_name, use_uncertainty in (
            ("distribution_gp", False),
            ("pu_dcgp", True),
        ):
            model = GaussianProcessDistributionModel(config, use_uncertainty)
            model.fit(prepared)
            score_prediction = model.predict(
                test_runs.treatment_values,
                test_runs.context_values,
            )
            distribution_prediction = encoder.inverse_transform(score_prediction)
            for outcome in outcome_names:
                quantile_predictions[model_name][outcome][test_indices] = (
                    distribution_prediction.means[outcome]
                )
                quantile_variances[model_name][outcome][test_indices] = (
                    distribution_prediction.variances[outcome]
                )

    mean_result = MeanBaselineCVResult(
        model_name="mean_gp",
        outcome_names=outcome_names,
        predictions=mean_predictions,
        fold_ids=fold_ids.copy(),
        metrics=outcome_metrics(outcome_names, mean_targets, mean_predictions),
    )
    weights = _trapezoid_weights(quantile_grid)
    distribution_results = {
        model_name: DistributionCVResult(
            model_name=model_name,
            quantile_predictions=predictions,
            quantile_variances=quantile_variances[model_name],
            metrics={
                outcome: _distribution_metrics(
                    true_quantiles[outcome],
                    predictions[outcome],
                    weights,
                )
                for outcome in outcome_names
            },
        )
        for model_name, predictions in quantile_predictions.items()
    }
    return GPValidationResult(
        fold_ids=fold_ids,
        mean_gp=mean_result,
        distribution_models=distribution_results,
    )


def _distribution_metrics(
    observed: FloatArray,
    predicted: FloatArray,
    weights: FloatArray,
) -> DistributionMetrics:
    errors = observed - predicted
    return DistributionMetrics(
        quantile_mae=float(
            np.average(np.abs(errors), axis=1, weights=weights).mean()
        ),
        wasserstein_rmse=float(
            np.sqrt(np.average(np.square(errors), axis=1, weights=weights).mean())
        ),
    )


def _trapezoid_weights(grid: FloatArray) -> FloatArray:
    differences = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights / weights.sum()
