"""Table-sequence sensitivity for matched A-group DOE effects."""

from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import FloatArray, RunBatch
from .estimands import DOEEstimand
from .matched_effects import estimate_matched_distribution_effects
from .mean_baselines import run_mean_targets


@dataclass(frozen=True, slots=True)
class SequenceAdjustedOutcomeEffect:
    """Unadjusted and table-sequence-adjusted effects for one outcome."""

    unadjusted_mean_effect: float
    adjusted_mean_effect: float
    sequence_slope_per_10_runs: float
    mean_sign_retained: bool
    relative_magnitude: float
    adjusted_quantile_effect: FloatArray
    adjusted_wasserstein_norm: float


@dataclass(frozen=True, slots=True)
class SequenceSensitivityResult:
    """Equal-stratum fixed-effect sensitivity for one DOE estimand."""

    estimand: DOEEstimand
    matched_strata: int
    matched_runs: int
    treatment_sequence_correlation: float
    design_condition_number: float
    outcome_effects: dict[str, SequenceAdjustedOutcomeEffect]


def estimate_sequence_adjusted_effects(
    runs: RunBatch,
    config: PUDCGPConfig,
    estimand: DOEEstimand,
) -> SequenceSensitivityResult:
    """Fit stratum fixed effects with a within-stratum table-sequence term."""

    matched = estimate_matched_distribution_effects(runs, config, estimand)
    strata = _matched_row_indices(runs, estimand)
    selected_rows = np.concatenate(strata)

    treatment_index = runs.treatment_names.index(estimand.treatment_name)
    treatment = (
        runs.treatment_values[selected_rows, treatment_index]
        == estimand.intervention_value
    ).astype(float)
    sequence_per_10_runs = runs.context_values[selected_rows, 0] / 10.0
    weights = np.concatenate(
        [np.full(len(rows), 1.0 / len(rows)) for rows in strata]
    )
    treatment_centered = _within_stratum_center(treatment, strata)
    sequence_centered = _within_stratum_center(sequence_per_10_runs, strata)
    design = np.column_stack((treatment_centered, sequence_centered))
    weighted_design = np.sqrt(weights)[:, None] * design

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
    integration_weights = _trapezoid_weights(quantile_grid)

    outcome_effects = {}
    for outcome_index, outcome in enumerate(outcome_names):
        centered_means = _within_stratum_center(
            mean_targets[selected_rows, outcome_index], strata
        )
        mean_coefficients = _weighted_coefficients(
            weighted_design, centered_means, weights
        )
        centered_quantiles = _within_stratum_center(
            quantile_curves[outcome][selected_rows], strata
        )
        quantile_coefficients = _weighted_coefficients(
            weighted_design, centered_quantiles, weights
        )
        adjusted_quantile_effect = quantile_coefficients[0]
        unadjusted_mean = matched.aggregate_effects[outcome].mean_difference
        adjusted_mean = float(mean_coefficients[0])
        outcome_effects[outcome] = SequenceAdjustedOutcomeEffect(
            unadjusted_mean_effect=unadjusted_mean,
            adjusted_mean_effect=adjusted_mean,
            sequence_slope_per_10_runs=float(mean_coefficients[1]),
            mean_sign_retained=bool(unadjusted_mean * adjusted_mean > 0),
            relative_magnitude=float(abs(adjusted_mean / unadjusted_mean)),
            adjusted_quantile_effect=adjusted_quantile_effect,
            adjusted_wasserstein_norm=float(
                np.sqrt(
                    np.average(
                        np.square(adjusted_quantile_effect),
                        weights=integration_weights,
                    )
                )
            ),
        )

    return SequenceSensitivityResult(
        estimand=estimand,
        matched_strata=len(strata),
        matched_runs=len(selected_rows),
        treatment_sequence_correlation=_weighted_correlation(
            treatment_centered, sequence_centered, weights
        ),
        design_condition_number=float(np.linalg.cond(weighted_design)),
        outcome_effects=outcome_effects,
    )


def _matched_row_indices(
    runs: RunBatch, estimand: DOEEstimand
) -> tuple[np.ndarray, ...]:
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

    strata = []
    for _, row_indices in sorted(grouped_rows.items()):
        rows = np.asarray(row_indices, dtype=int)
        values = runs.treatment_values[rows, treatment_index]
        selected = rows[
            (values == estimand.reference_value)
            | (values == estimand.intervention_value)
        ]
        selected_values = runs.treatment_values[selected, treatment_index]
        if (
            np.any(selected_values == estimand.reference_value)
            and np.any(selected_values == estimand.intervention_value)
        ):
            strata.append(selected)
    if len(strata) < 2:
        raise ValueError("At least two matched strata are required")
    return tuple(strata)


def _within_stratum_center(
    values: FloatArray, strata: tuple[np.ndarray, ...]
) -> FloatArray:
    centered = []
    offset = 0
    for rows in strata:
        block = values[offset : offset + len(rows)]
        centered.append(block - block.mean(axis=0))
        offset += len(rows)
    return np.concatenate(centered, axis=0)


def _weighted_coefficients(
    weighted_design: FloatArray, outcome: FloatArray, weights: FloatArray
) -> FloatArray:
    weighted_outcome = np.sqrt(weights).reshape(
        (-1,) + (1,) * (outcome.ndim - 1)
    ) * outcome
    return np.linalg.lstsq(weighted_design, weighted_outcome, rcond=None)[0]


def _weighted_correlation(
    left: FloatArray, right: FloatArray, weights: FloatArray
) -> float:
    covariance = np.sum(weights * left * right)
    left_variance = np.sum(weights * np.square(left))
    right_variance = np.sum(weights * np.square(right))
    return float(covariance / np.sqrt(left_variance * right_variance))


def _trapezoid_weights(grid: FloatArray) -> FloatArray:
    differences = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights / weights.sum()
