"""Predictive audit for execution-order drift within replicated settings."""

from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import RunBatch
from .gp_evaluation import DistributionMetrics
from .mean_baselines import OutcomeMetrics, outcome_metrics, run_mean_targets


@dataclass(frozen=True, slots=True)
class TemporalDriftAudit:
    """Leave-one-run-out comparison of constant and shared-slope predictors."""

    scope: str
    runs: int
    repeated_settings: int
    median_slope_per_10_runs: dict[str, float]
    constant_mean_metrics: dict[str, OutcomeMetrics]
    drift_mean_metrics: dict[str, OutcomeMetrics]
    constant_distribution_metrics: dict[str, DistributionMetrics]
    drift_distribution_metrics: dict[str, DistributionMetrics]


def audit_shared_linear_drift(
    runs: RunBatch,
    config: PUDCGPConfig,
    selected_indices: np.ndarray,
    scope: str,
) -> TemporalDriftAudit:
    """Test one shared within-setting slope using leave-one-run-out prediction."""

    selected = np.asarray(selected_indices, dtype=int)
    setting_rows: dict[tuple[float, ...], list[int]] = {}
    for index in selected:
        setting_rows.setdefault(tuple(runs.treatment_values[index]), []).append(
            int(index)
        )
    repeated_rows = {
        setting: indices
        for setting, indices in setting_rows.items()
        if len(indices) >= 2
    }
    eligible_indices = np.asarray(
        sorted(index for indices in repeated_rows.values() for index in indices),
        dtype=int,
    )

    outcome_names, mean_targets = run_mean_targets(runs)
    quantile_grid = np.asarray(config.quantile_grid, dtype=float)
    quantile_curves = {
        outcome: np.vstack(
            [
                np.quantile(values, quantile_grid, method="linear")
                for values in samples
            ]
        )
        for outcome, samples in runs.particle_samples.items()
    }
    n_rows = len(eligible_indices)
    constant_mean_predictions = np.empty((n_rows, len(outcome_names)))
    drift_mean_predictions = np.empty_like(constant_mean_predictions)
    slopes = np.empty_like(constant_mean_predictions)
    constant_distribution_predictions = {
        outcome: np.empty((n_rows, len(quantile_grid)))
        for outcome in outcome_names
    }
    drift_distribution_predictions = {
        outcome: np.empty_like(constant_distribution_predictions[outcome])
        for outcome in outcome_names
    }
    execution_order = np.asarray(runs.context_values[:, 0], dtype=float)

    for prediction_row, test_index in enumerate(eligible_indices):
        train_indices = eligible_indices[eligible_indices != test_index]
        training_groups = _group_indices(runs, train_indices)
        target_setting = tuple(runs.treatment_values[test_index])
        sibling_indices = np.asarray(training_groups[target_setting], dtype=int)
        order_offset = execution_order[test_index] - execution_order[
            sibling_indices
        ].mean()

        mean_slope = _fixed_effect_slope(
            execution_order,
            mean_targets,
            training_groups,
        )
        constant_mean_predictions[prediction_row] = mean_targets[
            sibling_indices
        ].mean(axis=0)
        drift_mean_predictions[prediction_row] = (
            constant_mean_predictions[prediction_row] + mean_slope * order_offset
        )
        slopes[prediction_row] = mean_slope

        for outcome in outcome_names:
            quantile_slope = _fixed_effect_slope(
                execution_order,
                quantile_curves[outcome],
                training_groups,
            )
            constant_distribution_predictions[outcome][prediction_row] = (
                quantile_curves[outcome][sibling_indices].mean(axis=0)
            )
            drift_distribution_predictions[outcome][prediction_row] = (
                constant_distribution_predictions[outcome][prediction_row]
                + quantile_slope * order_offset
            )

    observed_means = mean_targets[eligible_indices]
    weights = _trapezoid_weights(quantile_grid)
    return TemporalDriftAudit(
        scope=scope,
        runs=n_rows,
        repeated_settings=len(repeated_rows),
        median_slope_per_10_runs={
            outcome: float(np.median(slopes[:, outcome_index]) * 10)
            for outcome_index, outcome in enumerate(outcome_names)
        },
        constant_mean_metrics=outcome_metrics(
            outcome_names,
            observed_means,
            constant_mean_predictions,
        ),
        drift_mean_metrics=outcome_metrics(
            outcome_names,
            observed_means,
            drift_mean_predictions,
        ),
        constant_distribution_metrics={
            outcome: _distribution_metrics(
                quantile_curves[outcome][eligible_indices],
                constant_distribution_predictions[outcome],
                weights,
            )
            for outcome in outcome_names
        },
        drift_distribution_metrics={
            outcome: _distribution_metrics(
                quantile_curves[outcome][eligible_indices],
                drift_distribution_predictions[outcome],
                weights,
            )
            for outcome in outcome_names
        },
    )


def _group_indices(
    runs: RunBatch,
    indices: np.ndarray,
) -> dict[tuple[float, ...], list[int]]:
    groups: dict[tuple[float, ...], list[int]] = {}
    for index in indices:
        groups.setdefault(tuple(runs.treatment_values[index]), []).append(int(index))
    return groups


def _fixed_effect_slope(
    execution_order: np.ndarray,
    values: np.ndarray,
    groups: dict[tuple[float, ...], list[int]],
) -> np.ndarray:
    centered_orders = []
    centered_values = []
    for indices in groups.values():
        group_indices = np.asarray(indices, dtype=int)
        centered_orders.append(
            execution_order[group_indices] - execution_order[group_indices].mean()
        )
        centered_values.append(
            values[group_indices] - values[group_indices].mean(axis=0)
        )
    order_residuals = np.concatenate(centered_orders)
    value_residuals = np.vstack(centered_values)
    return (order_residuals[:, None] * value_residuals).sum(axis=0) / (
        order_residuals @ order_residuals
    )


def _distribution_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
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


def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    differences = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights / weights.sum()
