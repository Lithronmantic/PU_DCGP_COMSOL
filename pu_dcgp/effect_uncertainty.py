"""Hierarchical uncertainty for matched A-group distributional effects."""

from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import FloatArray, RunBatch
from .estimands import DOEEstimand
from .matched_effects import estimate_matched_distribution_effects
from .mean_baselines import run_mean_targets


@dataclass(frozen=True, slots=True)
class MeanEffectInterval:
    """Hierarchical-bootstrap uncertainty for one mean effect."""

    point_estimate: float
    bootstrap_mean: float
    bootstrap_standard_error: float
    lower_bound: float
    upper_bound: float
    same_sign_probability: float
    interval_excludes_zero: bool
    bootstrap_effects: FloatArray


@dataclass(frozen=True, slots=True)
class MatchedMeanUncertaintyResult:
    """Mean-effect intervals for one frozen DOE estimand."""

    estimand: DOEEstimand
    matched_strata: int
    bootstrap_replicates: int
    interval_level: float
    outcome_intervals: dict[str, MeanEffectInterval]


@dataclass(frozen=True, slots=True)
class QuantileEffectBand:
    """Pointwise and simultaneous uncertainty across one quantile grid."""

    quantile_grid: FloatArray
    point_effect: FloatArray
    bootstrap_mean: FloatArray
    pointwise_lower_bound: FloatArray
    pointwise_upper_bound: FloatArray
    simultaneous_lower_bound: FloatArray
    simultaneous_upper_bound: FloatArray
    simultaneous_critical_value: float
    same_sign_probability: FloatArray
    pointwise_excludes_zero: tuple[bool, ...]
    simultaneous_excludes_zero_anywhere: bool
    simultaneous_excludes_zero_everywhere: bool
    bootstrap_effects: FloatArray


@dataclass(frozen=True, slots=True)
class MatchedDistributionUncertaintyResult:
    """Three-level uncertainty for one estimand and distributional outcome."""

    estimand: DOEEstimand
    outcome: str
    matched_strata: int
    bootstrap_replicates: int
    interval_level: float
    mean_interval: MeanEffectInterval
    quantile_band: QuantileEffectBand


def bootstrap_matched_mean_effects(
    runs: RunBatch,
    config: PUDCGPConfig,
    estimand: DOEEstimand,
) -> MatchedMeanUncertaintyResult:
    """Resample matched strata and runs within each contrast arm."""

    matched = estimate_matched_distribution_effects(runs, config, estimand)
    strata = _matched_arm_rows(runs, estimand)
    outcome_names, mean_targets = run_mean_targets(runs)
    generator = np.random.default_rng(config.random_seed)
    bootstrap_effects = np.empty(
        (config.effect_bootstrap_replicates, len(outcome_names)), dtype=float
    )

    for replicate in range(config.effect_bootstrap_replicates):
        sampled_strata = generator.integers(0, len(strata), size=len(strata))
        stratum_effects = []
        for stratum_index in sampled_strata:
            reference_rows, intervention_rows = strata[stratum_index]
            sampled_reference = generator.choice(
                reference_rows, size=len(reference_rows), replace=True
            )
            sampled_intervention = generator.choice(
                intervention_rows, size=len(intervention_rows), replace=True
            )
            stratum_effects.append(
                mean_targets[sampled_intervention].mean(axis=0)
                - mean_targets[sampled_reference].mean(axis=0)
            )
        bootstrap_effects[replicate] = np.vstack(stratum_effects).mean(axis=0)

    outcome_intervals = {}
    for outcome_index, outcome in enumerate(outcome_names):
        effects = bootstrap_effects[:, outcome_index]
        point_estimate = matched.aggregate_effects[outcome].mean_difference
        outcome_intervals[outcome] = _mean_effect_interval(
            point_estimate,
            effects,
            config.effect_interval_level,
        )

    return MatchedMeanUncertaintyResult(
        estimand=estimand,
        matched_strata=len(strata),
        bootstrap_replicates=config.effect_bootstrap_replicates,
        interval_level=config.effect_interval_level,
        outcome_intervals=outcome_intervals,
    )


def bootstrap_matched_distribution_effect(
    runs: RunBatch,
    config: PUDCGPConfig,
    estimand: DOEEstimand,
    outcome: str,
) -> MatchedDistributionUncertaintyResult:
    """Resample strata, runs, and particles for one distributional effect."""

    matched = estimate_matched_distribution_effects(runs, config, estimand)
    strata = _matched_arm_rows(runs, estimand)
    quantile_grid = np.asarray(config.quantile_grid, dtype=float)
    matched_rows = np.unique(
        np.concatenate(
            [rows for stratum in strata for rows in stratum]
        )
    )
    local_index = {
        int(row): index for index, row in enumerate(matched_rows)
    }
    local_strata = tuple(
        (
            np.asarray([local_index[int(row)] for row in reference], dtype=int),
            np.asarray([local_index[int(row)] for row in intervention], dtype=int),
        )
        for reference, intervention in strata
    )
    generator = np.random.default_rng(config.random_seed)
    particle_means, particle_quantiles = _particle_bootstrap_summaries(
        runs,
        outcome,
        matched_rows,
        quantile_grid,
        config.effect_bootstrap_replicates,
        generator,
    )

    bootstrap_means = np.empty(config.effect_bootstrap_replicates, dtype=float)
    bootstrap_quantiles = np.empty(
        (config.effect_bootstrap_replicates, len(quantile_grid)), dtype=float
    )
    for replicate in range(config.effect_bootstrap_replicates):
        sampled_strata = generator.integers(
            0, len(local_strata), size=len(local_strata)
        )
        stratum_means = []
        stratum_quantiles = []
        for stratum_index in sampled_strata:
            reference_rows, intervention_rows = local_strata[stratum_index]
            sampled_reference = generator.choice(
                reference_rows, size=len(reference_rows), replace=True
            )
            sampled_intervention = generator.choice(
                intervention_rows, size=len(intervention_rows), replace=True
            )
            stratum_means.append(
                particle_means[replicate, sampled_intervention].mean()
                - particle_means[replicate, sampled_reference].mean()
            )
            stratum_quantiles.append(
                particle_quantiles[
                    replicate, sampled_intervention
                ].mean(axis=0)
                - particle_quantiles[
                    replicate, sampled_reference
                ].mean(axis=0)
            )
        bootstrap_means[replicate] = np.mean(stratum_means)
        bootstrap_quantiles[replicate] = np.vstack(stratum_quantiles).mean(
            axis=0
        )

    point_mean = matched.aggregate_effects[outcome].mean_difference
    point_quantiles = matched.aggregate_effects[outcome].quantile_difference
    tail_probability = (1.0 - config.effect_interval_level) / 2.0
    pointwise_lower, pointwise_upper = np.quantile(
        bootstrap_quantiles,
        [tail_probability, 1.0 - tail_probability],
        axis=0,
        method="linear",
    )
    maximum_deviations = np.max(
        np.abs(bootstrap_quantiles - point_quantiles), axis=1
    )
    critical_value = float(
        np.quantile(
            maximum_deviations,
            config.effect_interval_level,
            method="linear",
        )
    )
    simultaneous_lower = point_quantiles - critical_value
    simultaneous_upper = point_quantiles + critical_value
    pointwise_excludes_zero = tuple(
        bool(lower > 0 or upper < 0)
        for lower, upper in zip(pointwise_lower, pointwise_upper)
    )
    simultaneous_excludes = (
        (simultaneous_lower > 0) | (simultaneous_upper < 0)
    )

    return MatchedDistributionUncertaintyResult(
        estimand=estimand,
        outcome=outcome,
        matched_strata=len(strata),
        bootstrap_replicates=config.effect_bootstrap_replicates,
        interval_level=config.effect_interval_level,
        mean_interval=_mean_effect_interval(
            point_mean,
            bootstrap_means,
            config.effect_interval_level,
        ),
        quantile_band=QuantileEffectBand(
            quantile_grid=quantile_grid,
            point_effect=point_quantiles,
            bootstrap_mean=bootstrap_quantiles.mean(axis=0),
            pointwise_lower_bound=pointwise_lower,
            pointwise_upper_bound=pointwise_upper,
            simultaneous_lower_bound=simultaneous_lower,
            simultaneous_upper_bound=simultaneous_upper,
            simultaneous_critical_value=critical_value,
            same_sign_probability=np.mean(
                bootstrap_quantiles * point_quantiles > 0, axis=0
            ),
            pointwise_excludes_zero=pointwise_excludes_zero,
            simultaneous_excludes_zero_anywhere=bool(
                np.any(simultaneous_excludes)
            ),
            simultaneous_excludes_zero_everywhere=bool(
                np.all(simultaneous_excludes)
            ),
            bootstrap_effects=bootstrap_quantiles,
        ),
    )


def _particle_bootstrap_summaries(
    runs: RunBatch,
    outcome: str,
    matched_rows: np.ndarray,
    quantile_grid: FloatArray,
    replicates: int,
    generator: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    particle_means = np.empty((replicates, len(matched_rows)), dtype=float)
    particle_quantiles = np.empty(
        (replicates, len(matched_rows), len(quantile_grid)), dtype=float
    )
    samples = runs.particle_samples[outcome]
    for local_index, row_index in enumerate(matched_rows):
        values = samples[int(row_index)]
        bootstrap_indices = generator.integers(
            0, len(values), size=(replicates, len(values))
        )
        resampled = values[bootstrap_indices]
        particle_means[:, local_index] = resampled.mean(axis=1)
        particle_quantiles[:, local_index] = np.quantile(
            resampled,
            quantile_grid,
            axis=1,
            method="linear",
        ).T
    return particle_means, particle_quantiles


def _mean_effect_interval(
    point_estimate: float,
    bootstrap_effects: FloatArray,
    interval_level: float,
) -> MeanEffectInterval:
    tail_probability = (1.0 - interval_level) / 2.0
    lower_bound, upper_bound = np.quantile(
        bootstrap_effects,
        [tail_probability, 1.0 - tail_probability],
        method="linear",
    )
    return MeanEffectInterval(
        point_estimate=point_estimate,
        bootstrap_mean=float(bootstrap_effects.mean()),
        bootstrap_standard_error=float(bootstrap_effects.std(ddof=1)),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        same_sign_probability=float(
            np.mean(bootstrap_effects * point_estimate > 0)
        ),
        interval_excludes_zero=bool(lower_bound > 0 or upper_bound < 0),
        bootstrap_effects=bootstrap_effects.copy(),
    )


def _matched_arm_rows(
    runs: RunBatch, estimand: DOEEstimand
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
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
        treatment_values = runs.treatment_values[rows, treatment_index]
        reference_rows = rows[
            treatment_values == estimand.reference_value
        ]
        intervention_rows = rows[
            treatment_values == estimand.intervention_value
        ]
        if len(reference_rows) and len(intervention_rows):
            strata.append((reference_rows, intervention_rows))
    if len(strata) < 2:
        raise ValueError("At least two matched strata are required")
    return tuple(strata)
