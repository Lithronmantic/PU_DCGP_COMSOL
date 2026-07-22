
from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import RunBatch
from .gp_evaluation import DistributionMetrics
from .mean_baselines import OutcomeMetrics, outcome_metrics, run_mean_targets


@dataclass(frozen=True, slots=True)
class SharedAnchorSupport:

    treatment_setting: tuple[float, ...]
    module_run_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class AnchorOffsetFold:

    treatment_setting: tuple[float, ...]
    baseline_rmse: dict[str, float]
    offset_rmse: dict[str, float]


@dataclass(frozen=True, slots=True)
class CrossModuleOffsetAudit:

    reference_module: str
    target_module: str
    anchor_settings: tuple[tuple[float, ...], ...]
    observed_mean_offsets: dict[str, tuple[float, ...]]
    folds: tuple[AnchorOffsetFold, ...]
    baseline_mean_metrics: dict[str, OutcomeMetrics]
    offset_mean_metrics: dict[str, OutcomeMetrics]
    baseline_distribution_metrics: dict[str, DistributionMetrics]
    offset_distribution_metrics: dict[str, DistributionMetrics]


def find_shared_module_anchors(runs: RunBatch) -> tuple[SharedAnchorSupport, ...]:

    settings: dict[tuple[float, ...], dict[str, int]] = {}
    for index, treatment_row in enumerate(runs.treatment_values):
        setting = tuple(treatment_row)
        module = _module_code(runs.doe_modules[index])
        module_counts = settings.setdefault(setting, {})
        module_counts[module] = module_counts.get(module, 0) + 1

    return tuple(
        SharedAnchorSupport(
            treatment_setting=setting,
            module_run_counts=tuple(sorted(module_counts.items())),
        )
        for setting, module_counts in sorted(settings.items())
        if len(module_counts) >= 2
    )


def audit_cross_module_offset(
    runs: RunBatch,
    config: PUDCGPConfig,
    reference_module: str,
    target_module: str,
) -> CrossModuleOffsetAudit:

    modules = np.asarray([_module_code(module) for module in runs.doe_modules])
    anchor_settings = tuple(
        anchor.treatment_setting
        for anchor in find_shared_module_anchors(runs)
        if {reference_module, target_module}.issubset(
            dict(anchor.module_run_counts)
        )
    )
    if len(anchor_settings) < 2:
        raise ValueError("At least two shared anchors are required for validation")

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
    anchor_indices = {
        setting: {
            module: np.flatnonzero(
                (modules == module)
                & np.all(runs.treatment_values == setting, axis=1)
            )
            for module in (reference_module, target_module)
        }
        for setting in anchor_settings
    }
    mean_offsets = {
        setting: mean_targets[indices[target_module]].mean(axis=0)
        - mean_targets[indices[reference_module]].mean(axis=0)
        for setting, indices in anchor_indices.items()
    }
    quantile_offsets = {
        setting: {
            outcome: quantile_curves[outcome][indices[target_module]].mean(axis=0)
            - quantile_curves[outcome][indices[reference_module]].mean(axis=0)
            for outcome in outcome_names
        }
        for setting, indices in anchor_indices.items()
    }

    observed_rows = []
    baseline_mean_rows = []
    offset_mean_rows = []
    observed_distribution_rows = {outcome: [] for outcome in outcome_names}
    baseline_distribution_rows = {outcome: [] for outcome in outcome_names}
    offset_distribution_rows = {outcome: [] for outcome in outcome_names}
    folds = []

    for test_setting in anchor_settings:
        training_settings = [
            setting for setting in anchor_settings if setting != test_setting
        ]
        learned_mean_offset = np.vstack(
            [mean_offsets[setting] for setting in training_settings]
        ).mean(axis=0)
        indices = anchor_indices[test_setting]
        reference_indices = indices[reference_module]
        target_indices = indices[target_module]
        reference_mean = mean_targets[reference_indices].mean(axis=0)
        baseline_means = np.tile(reference_mean, (len(target_indices), 1))
        offset_means = baseline_means + learned_mean_offset
        observed_means = mean_targets[target_indices]

        folds.append(
            AnchorOffsetFold(
                treatment_setting=test_setting,
                baseline_rmse={
                    outcome: float(
                        np.sqrt(
                            np.square(
                                observed_means[:, outcome_index]
                                - baseline_means[:, outcome_index]
                            ).mean()
                        )
                    )
                    for outcome_index, outcome in enumerate(outcome_names)
                },
                offset_rmse={
                    outcome: float(
                        np.sqrt(
                            np.square(
                                observed_means[:, outcome_index]
                                - offset_means[:, outcome_index]
                            ).mean()
                        )
                    )
                    for outcome_index, outcome in enumerate(outcome_names)
                },
            )
        )
        observed_rows.append(observed_means)
        baseline_mean_rows.append(baseline_means)
        offset_mean_rows.append(offset_means)

        for outcome in outcome_names:
            reference_quantiles = quantile_curves[outcome][reference_indices].mean(
                axis=0
            )
            learned_quantile_offset = np.vstack(
                [quantile_offsets[setting][outcome] for setting in training_settings]
            ).mean(axis=0)
            baseline_quantiles = np.tile(
                reference_quantiles,
                (len(target_indices), 1),
            )
            adjusted_quantiles = np.sort(
                baseline_quantiles + learned_quantile_offset,
                axis=1,
            )
            observed_distribution_rows[outcome].append(
                quantile_curves[outcome][target_indices]
            )
            baseline_distribution_rows[outcome].append(baseline_quantiles)
            offset_distribution_rows[outcome].append(adjusted_quantiles)

    observed = np.vstack(observed_rows)
    baseline_predictions = np.vstack(baseline_mean_rows)
    offset_predictions = np.vstack(offset_mean_rows)
    weights = _trapezoid_weights(quantile_grid)
    return CrossModuleOffsetAudit(
        reference_module=reference_module,
        target_module=target_module,
        anchor_settings=anchor_settings,
        observed_mean_offsets={
            outcome: tuple(
                float(mean_offsets[setting][outcome_index])
                for setting in anchor_settings
            )
            for outcome_index, outcome in enumerate(outcome_names)
        },
        folds=tuple(folds),
        baseline_mean_metrics=outcome_metrics(
            outcome_names,
            observed,
            baseline_predictions,
        ),
        offset_mean_metrics=outcome_metrics(
            outcome_names,
            observed,
            offset_predictions,
        ),
        baseline_distribution_metrics={
            outcome: _distribution_metrics(
                np.vstack(observed_distribution_rows[outcome]),
                np.vstack(baseline_distribution_rows[outcome]),
                weights,
            )
            for outcome in outcome_names
        },
        offset_distribution_metrics={
            outcome: _distribution_metrics(
                np.vstack(observed_distribution_rows[outcome]),
                np.vstack(offset_distribution_rows[outcome]),
                weights,
            )
            for outcome in outcome_names
        },
    )


def _module_code(module: str) -> str:
    return module.partition("｜")[0]


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
