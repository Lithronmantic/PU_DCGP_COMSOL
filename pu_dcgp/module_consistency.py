
from dataclasses import dataclass
import re

import numpy as np

from .config import PUDCGPConfig
from .contracts import FloatArray, RunBatch
from .estimands import DOEEstimand
from .matched_effects import estimate_matched_distribution_effects
from .mean_baselines import run_mean_targets


@dataclass(frozen=True, slots=True)
class ModuleOutcomeEffect:

    mean_difference: float
    quantile_difference: FloatArray
    wasserstein_norm: float


@dataclass(frozen=True, slots=True)
class ModuleEffect:

    module_code: str
    matched_strata: int
    reference_runs: int
    intervention_runs: int
    outcome_effects: dict[str, ModuleOutcomeEffect]


@dataclass(frozen=True, slots=True)
class ModuleOutcomeConsistency:

    pooled_mean_effect: float
    module_balanced_mean_effect: float
    module_mean_effects: dict[str, float]
    direction_consistent: bool
    quantile_direction_consistent: bool
    module_balanced_sign_retained: bool
    absolute_magnitude_ratio: float


@dataclass(frozen=True, slots=True)
class ModuleConsistencyResult:

    estimand: DOEEstimand
    modules: tuple[ModuleEffect, ...]
    all_modules_have_multiple_strata: bool
    outcome_consistency: dict[str, ModuleOutcomeConsistency]


def estimate_module_consistency(
    runs: RunBatch,
    config: PUDCGPConfig,
    estimand: DOEEstimand,
) -> ModuleConsistencyResult:

    pooled = estimate_matched_distribution_effects(runs, config, estimand)
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
    module_codes = np.asarray([_module_code(value) for value in runs.doe_modules])
    module_effects = []
    for code in sorted(set(module_codes)):
        effect = _estimate_module_effect(
            runs,
            estimand,
            code,
            np.flatnonzero(module_codes == code),
            outcome_names,
            mean_targets,
            quantile_curves,
            integration_weights,
        )
        if effect is not None:
            module_effects.append(effect)
    modules = tuple(module_effects)

    outcome_consistency = {}
    for outcome in outcome_names:
        module_means = {
            module.module_code: module.outcome_effects[outcome].mean_difference
            for module in modules
        }
        values = np.asarray(tuple(module_means.values()), dtype=float)
        pooled_mean = pooled.aggregate_effects[outcome].mean_difference
        pooled_quantiles = pooled.aggregate_effects[outcome].quantile_difference
        module_balanced_mean = float(values.mean())
        magnitudes = np.abs(values)
        outcome_consistency[outcome] = ModuleOutcomeConsistency(
            pooled_mean_effect=pooled_mean,
            module_balanced_mean_effect=module_balanced_mean,
            module_mean_effects=module_means,
            direction_consistent=bool(np.all(values * pooled_mean > 0)),
            quantile_direction_consistent=bool(
                all(
                    np.all(
                        module.outcome_effects[outcome].quantile_difference
                        * pooled_quantiles
                        > 0
                    )
                    for module in modules
                )
            ),
            module_balanced_sign_retained=bool(
                module_balanced_mean * pooled_mean > 0
            ),
            absolute_magnitude_ratio=float(magnitudes.max() / magnitudes.min()),
        )

    return ModuleConsistencyResult(
        estimand=estimand,
        modules=modules,
        all_modules_have_multiple_strata=all(
            module.matched_strata >= 2 for module in modules
        ),
        outcome_consistency=outcome_consistency,
    )


def _estimate_module_effect(
    runs: RunBatch,
    estimand: DOEEstimand,
    module_code: str,
    module_rows: np.ndarray,
    outcome_names: tuple[str, ...],
    mean_targets: FloatArray,
    quantile_curves: dict[str, FloatArray],
    integration_weights: FloatArray,
) -> ModuleEffect | None:
    treatment_index = runs.treatment_names.index(estimand.treatment_name)
    other_indices = [
        index
        for index in range(len(runs.treatment_names))
        if index != treatment_index
    ]
    grouped_rows: dict[tuple[float, ...], list[int]] = {}
    for row_index in module_rows:
        treatment_row = runs.treatment_values[row_index]
        grouped_rows.setdefault(tuple(treatment_row[other_indices]), []).append(
            int(row_index)
        )

    mean_differences = {outcome: [] for outcome in outcome_names}
    quantile_differences = {outcome: [] for outcome in outcome_names}
    matched_strata = 0
    reference_runs = 0
    intervention_runs = 0
    for _, row_indices in sorted(grouped_rows.items()):
        rows = np.asarray(row_indices, dtype=int)
        values = runs.treatment_values[rows, treatment_index]
        reference = rows[values == estimand.reference_value]
        intervention = rows[values == estimand.intervention_value]
        if len(reference) == 0 or len(intervention) == 0:
            continue
        matched_strata += 1
        reference_runs += len(reference)
        intervention_runs += len(intervention)
        for outcome_index, outcome in enumerate(outcome_names):
            mean_differences[outcome].append(
                mean_targets[intervention, outcome_index].mean()
                - mean_targets[reference, outcome_index].mean()
            )
            quantile_differences[outcome].append(
                quantile_curves[outcome][intervention].mean(axis=0)
                - quantile_curves[outcome][reference].mean(axis=0)
            )
    if matched_strata == 0:
        return None

    outcome_effects = {}
    for outcome in outcome_names:
        aggregate_quantiles = np.vstack(quantile_differences[outcome]).mean(axis=0)
        outcome_effects[outcome] = ModuleOutcomeEffect(
            mean_difference=float(np.mean(mean_differences[outcome])),
            quantile_difference=aggregate_quantiles,
            wasserstein_norm=float(
                np.sqrt(
                    np.average(
                        np.square(aggregate_quantiles),
                        weights=integration_weights,
                    )
                )
            ),
        )
    return ModuleEffect(
        module_code=module_code,
        matched_strata=matched_strata,
        reference_runs=reference_runs,
        intervention_runs=intervention_runs,
        outcome_effects=outcome_effects,
    )


def _module_code(value: str) -> str:
    return re.match(r"DOE-\d+", value).group(0)


def _trapezoid_weights(grid: FloatArray) -> FloatArray:
    differences = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights / weights.sum()
