"""Diagnostics for the empirical support available before causal estimation."""

from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import RunBatch
from .data_source import subset_run_batch
from .gaussian_process import GaussianProcessMeanModel
from .gp_evaluation import DistributionMetrics
from .mean_baselines import (
    OutcomeMetrics,
    grouped_setting_folds,
    outcome_metrics,
    run_mean_targets,
)


@dataclass(frozen=True, slots=True)
class RepeatedSettingAudit:
    """Leave-one-run-out reproducibility among settings with replicates."""

    repeated_runs: int
    repeated_settings: int
    global_mean_metrics: dict[str, OutcomeMetrics]
    same_setting_mean_metrics: dict[str, OutcomeMetrics]
    global_distribution_metrics: dict[str, DistributionMetrics]
    same_setting_distribution_metrics: dict[str, DistributionMetrics]


@dataclass(frozen=True, slots=True)
class ModulePredictionAudit:
    """Grouped out-of-setting mean prediction within one DOE module."""

    doe_module: str
    runs: int
    unique_settings: int
    repeated_runs: int
    n_folds: int | None
    fold_mean_metrics: dict[str, OutcomeMetrics] | None
    mean_gp_metrics: dict[str, OutcomeMetrics] | None


@dataclass(frozen=True, slots=True)
class MatchedFactorSupport:
    """Exact matching support for one factor with the other factors fixed."""

    treatment_name: str
    observed_levels: tuple[float, ...]
    matched_strata: int
    supported_runs: int
    max_levels_per_stratum: int
    level_set_counts: tuple[tuple[tuple[float, ...], int], ...]


def audit_repeated_settings(
    runs: RunBatch,
    config: PUDCGPConfig,
) -> RepeatedSettingAudit:
    """Compare sibling-run predictions with a leave-one-run-out global mean."""

    setting_rows: dict[tuple[float, ...], list[int]] = {}
    for index, treatment_row in enumerate(runs.treatment_values):
        setting_rows.setdefault(tuple(treatment_row), []).append(index)
    repeated_groups = [indices for indices in setting_rows.values() if len(indices) >= 2]
    repeated_indices = np.asarray(
        [index for indices in repeated_groups for index in indices],
        dtype=int,
    )
    outcome_names, mean_targets = run_mean_targets(runs)
    global_mean_predictions = np.empty(
        (len(repeated_indices), len(outcome_names)),
        dtype=float,
    )
    same_setting_mean_predictions = np.empty_like(global_mean_predictions)

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
    global_distribution_predictions = {
        outcome: np.empty((len(repeated_indices), len(quantile_grid)), dtype=float)
        for outcome in outcome_names
    }
    same_setting_distribution_predictions = {
        outcome: np.empty_like(global_distribution_predictions[outcome])
        for outcome in outcome_names
    }
    all_indices = np.arange(len(runs.run_ids))

    for prediction_row, test_index in enumerate(repeated_indices):
        setting = tuple(runs.treatment_values[test_index])
        sibling_indices = np.asarray(
            [index for index in setting_rows[setting] if index != test_index],
            dtype=int,
        )
        global_indices = all_indices[all_indices != test_index]
        global_mean_predictions[prediction_row] = mean_targets[global_indices].mean(
            axis=0
        )
        same_setting_mean_predictions[prediction_row] = mean_targets[
            sibling_indices
        ].mean(axis=0)
        for outcome in outcome_names:
            global_distribution_predictions[outcome][prediction_row] = (
                quantile_curves[outcome][global_indices].mean(axis=0)
            )
            same_setting_distribution_predictions[outcome][prediction_row] = (
                quantile_curves[outcome][sibling_indices].mean(axis=0)
            )

    observed_means = mean_targets[repeated_indices]
    weights = _trapezoid_weights(quantile_grid)
    return RepeatedSettingAudit(
        repeated_runs=len(repeated_indices),
        repeated_settings=len(repeated_groups),
        global_mean_metrics=outcome_metrics(
            outcome_names,
            observed_means,
            global_mean_predictions,
        ),
        same_setting_mean_metrics=outcome_metrics(
            outcome_names,
            observed_means,
            same_setting_mean_predictions,
        ),
        global_distribution_metrics={
            outcome: _distribution_metrics(
                quantile_curves[outcome][repeated_indices],
                global_distribution_predictions[outcome],
                weights,
            )
            for outcome in outcome_names
        },
        same_setting_distribution_metrics={
            outcome: _distribution_metrics(
                quantile_curves[outcome][repeated_indices],
                same_setting_distribution_predictions[outcome],
                weights,
            )
            for outcome in outcome_names
        },
    )


def audit_module_predictions(
    runs: RunBatch,
    config: PUDCGPConfig,
    max_folds: int = 5,
) -> tuple[ModulePredictionAudit, ...]:
    """Evaluate held-out settings separately inside each DOE module."""

    audits = []
    for doe_module in dict.fromkeys(runs.doe_modules):
        module_indices = np.flatnonzero(
            np.asarray(runs.doe_modules, dtype=object) == doe_module
        )
        module_runs = subset_run_batch(runs, module_indices)
        _, counts = np.unique(
            module_runs.treatment_values,
            axis=0,
            return_counts=True,
        )
        unique_settings = len(counts)
        repeated_runs = int(counts[counts >= 2].sum())
        if unique_settings < 2:
            audits.append(
                ModulePredictionAudit(
                    doe_module=doe_module,
                    runs=len(module_indices),
                    unique_settings=unique_settings,
                    repeated_runs=repeated_runs,
                    n_folds=None,
                    fold_mean_metrics=None,
                    mean_gp_metrics=None,
                )
            )
            continue

        n_folds = min(max_folds, unique_settings)
        folds = grouped_setting_folds(module_runs.treatment_values, n_folds)
        outcome_names, targets = run_mean_targets(module_runs)
        fold_mean_predictions = np.empty_like(targets)
        gp_predictions = np.empty_like(targets)
        all_indices = np.arange(len(module_indices))
        for test_indices in folds:
            train_indices = np.setdiff1d(all_indices, test_indices)
            fold_mean_predictions[test_indices] = targets[train_indices].mean(axis=0)
            training_runs = subset_run_batch(module_runs, train_indices)
            model = GaussianProcessMeanModel(config)
            model.fit(training_runs)
            predicted_means, _ = model.predict(
                module_runs.treatment_values[test_indices],
                module_runs.context_values[test_indices],
            )
            gp_predictions[test_indices] = predicted_means

        audits.append(
            ModulePredictionAudit(
                doe_module=doe_module,
                runs=len(module_indices),
                unique_settings=unique_settings,
                repeated_runs=repeated_runs,
                n_folds=n_folds,
                fold_mean_metrics=outcome_metrics(
                    outcome_names,
                    targets,
                    fold_mean_predictions,
                ),
                mean_gp_metrics=outcome_metrics(
                    outcome_names,
                    targets,
                    gp_predictions,
                ),
            )
        )
    return tuple(audits)


def audit_matched_factor_support(
    runs: RunBatch,
) -> tuple[MatchedFactorSupport, ...]:
    """Count exact factor contrasts while holding the other factors fixed."""

    results = []
    treatments = np.asarray(runs.treatment_values, dtype=float)
    for factor_index, treatment_name in enumerate(runs.treatment_names):
        other_indices = [
            index
            for index in range(treatments.shape[1])
            if index != factor_index
        ]
        strata: dict[tuple[float, ...], list[int]] = {}
        for row_index, row in enumerate(treatments):
            key = tuple(row[other_indices])
            strata.setdefault(key, []).append(row_index)

        matched = []
        level_set_counts: dict[tuple[float, ...], int] = {}
        for row_indices in strata.values():
            level_set = tuple(
                sorted(set(treatments[row_indices, factor_index]))
            )
            if len(level_set) < 2:
                continue
            matched.append((row_indices, level_set))
            level_set_counts[level_set] = level_set_counts.get(level_set, 0) + 1

        ordered_level_sets = tuple(
            sorted(
                level_set_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        results.append(
            MatchedFactorSupport(
                treatment_name=treatment_name,
                observed_levels=tuple(sorted(set(treatments[:, factor_index]))),
                matched_strata=len(matched),
                supported_runs=sum(len(row_indices) for row_indices, _ in matched),
                max_levels_per_stratum=max(
                    (len(level_set) for _, level_set in matched),
                    default=0,
                ),
                level_set_counts=ordered_level_sets,
            )
        )
    return tuple(results)


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
