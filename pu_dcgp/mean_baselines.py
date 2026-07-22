
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .contracts import FloatArray, RunBatch


IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class OutcomeMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True, slots=True)
class MeanBaselineCVResult:
    model_name: str
    outcome_names: tuple[str, ...]
    predictions: FloatArray
    fold_ids: IntArray
    metrics: dict[str, OutcomeMetrics]
    selected_alphas: FloatArray | None = None


class PolynomialRidgeMeanModel:

    def __init__(self, degree: int) -> None:
        self.degree = degree
        self.treatment_mean: FloatArray | None = None
        self.treatment_scale: FloatArray | None = None
        self.coefficients: FloatArray | None = None

    def fit(self, treatments: FloatArray, targets: FloatArray, alphas: FloatArray) -> None:
        self.treatment_mean = treatments.mean(axis=0)
        self.treatment_scale = treatments.std(axis=0)
        self.treatment_scale[self.treatment_scale == 0] = 1.0
        design = self._design_matrix(treatments)
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.empty((design.shape[1], targets.shape[1]))

        for outcome_index, alpha in enumerate(alphas):
            coefficients[:, outcome_index] = np.linalg.solve(
                design.T @ design + alpha * penalty,
                design.T @ targets[:, outcome_index],
            )
        self.coefficients = coefficients

    def predict(self, treatments: FloatArray) -> FloatArray:
        if self.coefficients is None:
            raise RuntimeError("The mean baseline has not been fitted")
        return self._design_matrix(treatments) @ self.coefficients

    def _design_matrix(self, treatments: FloatArray) -> FloatArray:
        if self.treatment_mean is None or self.treatment_scale is None:
            raise RuntimeError("The mean baseline has not been fitted")
        standardized = (treatments - self.treatment_mean) / self.treatment_scale
        columns = [np.ones(len(treatments))]
        columns.extend(standardized[:, index] for index in range(standardized.shape[1]))

        if self.degree == 2:
            columns.extend(
                np.square(standardized[:, index])
                for index in range(standardized.shape[1])
            )
            columns.extend(
                standardized[:, left] * standardized[:, right]
                for left in range(standardized.shape[1])
                for right in range(left + 1, standardized.shape[1])
            )
        return np.column_stack(columns)


def run_mean_targets(runs: RunBatch) -> tuple[tuple[str, ...], FloatArray]:
    outcome_names = tuple(runs.particle_samples)
    targets = np.column_stack(
        [
            [float(values.mean()) for values in runs.particle_samples[outcome]]
            for outcome in outcome_names
        ]
    )
    return outcome_names, targets


def grouped_setting_folds(treatments: FloatArray, n_folds: int) -> tuple[IntArray, ...]:
    unique_settings, group_ids, group_counts = np.unique(
        treatments,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    group_order = sorted(
        range(len(unique_settings)),
        key=lambda index: (-group_counts[index], tuple(unique_settings[index])),
    )
    fold_loads = np.zeros(n_folds, dtype=int)
    group_folds = np.empty(len(unique_settings), dtype=int)

    for group_index in group_order:
        fold_index = int(np.argmin(fold_loads))
        group_folds[group_index] = fold_index
        fold_loads[fold_index] += group_counts[group_index]

    row_folds = group_folds[group_ids]
    return tuple(np.flatnonzero(row_folds == fold) for fold in range(n_folds))


def cross_validate_mean_baselines(
    runs: RunBatch,
    n_folds: int = 5,
    inner_folds: int = 4,
    alpha_grid: tuple[float, ...] = (1e-4, 1e-2, 1.0, 100.0, 10_000.0),
) -> dict[str, MeanBaselineCVResult]:
    treatments = np.asarray(runs.treatment_values, dtype=float)
    outcome_names, targets = run_mean_targets(runs)
    folds = grouped_setting_folds(treatments, n_folds)
    fold_ids = np.empty(len(treatments), dtype=int)
    model_specs = {
        "linear_ridge": (1, treatments),
        "quadratic_ridge": (2, treatments),
        "quadratic_ridge_with_order": (
            2,
            np.column_stack([treatments, runs.context_values[:, 0]]),
        ),
    }
    predictions = {"global_mean": np.empty_like(targets)}
    predictions.update({name: np.empty_like(targets) for name in model_specs})
    selected_alphas = {
        name: np.empty((n_folds, targets.shape[1])) for name in model_specs
    }

    all_indices = np.arange(len(treatments))
    for fold_index, test_indices in enumerate(folds):
        train_indices = np.setdiff1d(all_indices, test_indices)
        fold_ids[test_indices] = fold_index
        predictions["global_mean"][test_indices] = targets[train_indices].mean(axis=0)

        for model_name, (degree, predictors) in model_specs.items():
            alphas = _select_alphas(
                predictors[train_indices],
                treatments[train_indices],
                targets[train_indices],
                degree,
                inner_folds,
                alpha_grid,
            )
            selected_alphas[model_name][fold_index] = alphas
            model = PolynomialRidgeMeanModel(degree)
            model.fit(predictors[train_indices], targets[train_indices], alphas)
            predictions[model_name][test_indices] = model.predict(
                predictors[test_indices]
            )

    return {
        model_name: MeanBaselineCVResult(
            model_name=model_name,
            outcome_names=outcome_names,
            predictions=model_predictions,
            fold_ids=fold_ids.copy(),
            metrics=outcome_metrics(outcome_names, targets, model_predictions),
            selected_alphas=selected_alphas.get(model_name),
        )
        for model_name, model_predictions in predictions.items()
    }


def _select_alphas(
    predictors: FloatArray,
    grouping_treatments: FloatArray,
    targets: FloatArray,
    degree: int,
    n_folds: int,
    alpha_grid: tuple[float, ...],
) -> FloatArray:
    folds = grouped_setting_folds(grouping_treatments, n_folds)
    squared_errors = np.zeros((len(alpha_grid), targets.shape[1]))
    all_indices = np.arange(len(predictors))

    for validation_indices in folds:
        training_indices = np.setdiff1d(all_indices, validation_indices)
        for alpha_index, alpha in enumerate(alpha_grid):
            model = PolynomialRidgeMeanModel(degree)
            model.fit(
                predictors[training_indices],
                targets[training_indices],
                np.full(targets.shape[1], alpha),
            )
            errors = targets[validation_indices] - model.predict(
                predictors[validation_indices]
            )
            squared_errors[alpha_index] += np.square(errors).sum(axis=0)

    best_indices = squared_errors.argmin(axis=0)
    return np.asarray(alpha_grid, dtype=float)[best_indices]


def outcome_metrics(
    outcome_names: tuple[str, ...],
    targets: FloatArray,
    predictions: FloatArray,
) -> dict[str, OutcomeMetrics]:
    metrics: dict[str, OutcomeMetrics] = {}
    for outcome_index, outcome in enumerate(outcome_names):
        observed = targets[:, outcome_index]
        predicted = predictions[:, outcome_index]
        errors = observed - predicted
        metrics[outcome] = OutcomeMetrics(
            mae=float(np.abs(errors).mean()),
            rmse=float(np.sqrt(np.square(errors).mean())),
            r2=float(
                1
                - np.square(errors).sum()
                / np.square(observed - observed.mean()).sum()
            ),
        )
    return metrics
