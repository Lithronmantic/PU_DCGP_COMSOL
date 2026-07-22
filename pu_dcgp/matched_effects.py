
from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import FloatArray, RunBatch
from .estimands import DOEEstimand
from .mean_baselines import run_mean_targets


@dataclass(frozen=True, slots=True)
class MatchedStratumEffect:

    fixed_treatments: tuple[tuple[str, float], ...]
    mean_differences: dict[str, float]
    quantile_differences: dict[str, FloatArray]
    wasserstein_distances: dict[str, float]


@dataclass(frozen=True, slots=True)
class AggregateOutcomeEffect:

    mean_difference: float
    median_quantile_difference: float
    quantile_difference: FloatArray
    wasserstein_norm: float
    mean_stratum_wasserstein: float
    leave_one_out_mean_differences: FloatArray
    leave_one_out_min: float
    leave_one_out_max: float
    leave_one_out_sign_stable: bool


@dataclass(frozen=True, slots=True)
class MatchedDistributionEffectResult:

    estimand: DOEEstimand
    quantile_grid: FloatArray
    strata: tuple[MatchedStratumEffect, ...]
    aggregate_effects: dict[str, AggregateOutcomeEffect]


def estimate_matched_distribution_effects(
    runs: RunBatch,
    config: PUDCGPConfig,
    estimand: DOEEstimand,
) -> MatchedDistributionEffectResult:

    treatment_index = runs.treatment_names.index(estimand.treatment_name)
    other_indices = [
        index
        for index in range(len(runs.treatment_names))
        if index != treatment_index
    ]
    grouped_rows: dict[tuple[float, ...], list[int]] = {}
    for row_index, treatment_row in enumerate(runs.treatment_values):
        grouped_rows.setdefault(tuple(treatment_row[other_indices]), []).append(
            row_index
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
    weights = _trapezoid_weights(quantile_grid)
    strata = []
    for fixed_values, row_indices in sorted(grouped_rows.items()):
        rows = np.asarray(row_indices, dtype=int)
        treatment_values = runs.treatment_values[rows, treatment_index]
        reference_rows = rows[treatment_values == estimand.reference_value]
        intervention_rows = rows[treatment_values == estimand.intervention_value]
        if len(reference_rows) == 0 or len(intervention_rows) == 0:
            continue
        mean_differences = {}
        distribution_differences = {}
        wasserstein_distances = {}
        for outcome_index, outcome in enumerate(outcome_names):
            mean_differences[outcome] = float(
                mean_targets[intervention_rows, outcome_index].mean()
                - mean_targets[reference_rows, outcome_index].mean()
            )
            quantile_difference = (
                quantile_curves[outcome][intervention_rows].mean(axis=0)
                - quantile_curves[outcome][reference_rows].mean(axis=0)
            )
            distribution_differences[outcome] = quantile_difference
            wasserstein_distances[outcome] = float(
                np.sqrt(np.average(np.square(quantile_difference), weights=weights))
            )
        strata.append(
            MatchedStratumEffect(
                fixed_treatments=tuple(
                    (runs.treatment_names[index], float(value))
                    for index, value in zip(other_indices, fixed_values)
                ),
                mean_differences=mean_differences,
                quantile_differences=distribution_differences,
                wasserstein_distances=wasserstein_distances,
            )
        )

    if len(strata) < 2:
        raise ValueError("At least two matched strata are required")
    median_index = int(np.argmin(np.abs(quantile_grid - 0.5)))
    aggregate_effects = {}
    for outcome in outcome_names:
        stratum_means = np.asarray(
            [stratum.mean_differences[outcome] for stratum in strata],
            dtype=float,
        )
        stratum_quantiles = np.vstack(
            [stratum.quantile_differences[outcome] for stratum in strata]
        )
        aggregate_quantiles = stratum_quantiles.mean(axis=0)
        aggregate_mean = float(stratum_means.mean())
        leave_one_out = (stratum_means.sum() - stratum_means) / (
            len(stratum_means) - 1
        )
        sign_stable = bool(
            np.all(leave_one_out > 0)
            if aggregate_mean > 0
            else np.all(leave_one_out < 0)
            if aggregate_mean < 0
            else False
        )
        aggregate_effects[outcome] = AggregateOutcomeEffect(
            mean_difference=aggregate_mean,
            median_quantile_difference=float(aggregate_quantiles[median_index]),
            quantile_difference=aggregate_quantiles,
            wasserstein_norm=float(
                np.sqrt(np.average(np.square(aggregate_quantiles), weights=weights))
            ),
            mean_stratum_wasserstein=float(
                np.mean(
                    [stratum.wasserstein_distances[outcome] for stratum in strata]
                )
            ),
            leave_one_out_mean_differences=leave_one_out,
            leave_one_out_min=float(leave_one_out.min()),
            leave_one_out_max=float(leave_one_out.max()),
            leave_one_out_sign_stable=sign_stable,
        )
    return MatchedDistributionEffectResult(
        estimand=estimand,
        quantile_grid=quantile_grid,
        strata=tuple(strata),
        aggregate_effects=aggregate_effects,
    )


def _trapezoid_weights(grid: FloatArray) -> FloatArray:
    differences = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights / weights.sum()
