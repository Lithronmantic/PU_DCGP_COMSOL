
from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import (
    DistributionPrediction,
    DistributionRepresentation,
    FloatArray,
    JointDistributionPrediction,
    JointScorePrediction,
    RunBatch,
    ScorePrediction,
)
from .interfaces import DistributionEncoder


@dataclass(frozen=True, slots=True)
class OutcomeFPCAState:

    mean_quantile: FloatArray
    quadrature_weights: FloatArray
    scaled_components: FloatArray
    explained_variance_ratio: FloatArray

    @property
    def n_components(self) -> int:
        return self.scaled_components.shape[0]

    @property
    def cumulative_explained_variance(self) -> float:
        return float(self.explained_variance_ratio[: self.n_components].sum())


class BootstrapWassersteinFPCAEncoder(DistributionEncoder):

    def __init__(self, config: PUDCGPConfig) -> None:
        self.config = config
        self.quantile_grid = np.asarray(config.quantile_grid, dtype=float)
        self.quadrature_weights = self._trapezoid_weights(self.quantile_grid)
        self.states: dict[str, OutcomeFPCAState] = {}

    def fit_transform(self, runs: RunBatch) -> DistributionRepresentation:
        curves = self.empirical_quantiles(runs)
        self.states = {
            outcome: self._fit_outcome(outcome_curves)
            for outcome, outcome_curves in curves.items()
        }
        return self._transform(runs, curves)

    def transform(self, runs: RunBatch) -> DistributionRepresentation:
        if not self.states:
            raise RuntimeError("The distribution encoder has not been fitted")
        return self._transform(runs, self.empirical_quantiles(runs))

    def inverse_transform(self, prediction: ScorePrediction) -> DistributionPrediction:
        distribution_means: dict[str, FloatArray] = {}
        distribution_variances: dict[str, FloatArray] = {}

        for outcome, state in self.states.items():
            loadings = state.scaled_components / np.sqrt(
                state.quadrature_weights[None, :]
            )
            raw_means = state.mean_quantile + np.asarray(
                prediction.means[outcome]
            ) @ loadings
            raw_variances = np.asarray(
                prediction.variances[outcome]
            ) @ np.square(loadings)
            rearrangement = np.argsort(raw_means, axis=1)
            distribution_means[outcome] = np.take_along_axis(
                raw_means,
                rearrangement,
                axis=1,
            )
            distribution_variances[outcome] = np.take_along_axis(
                raw_variances,
                rearrangement,
                axis=1,
            )

        return DistributionPrediction(
            quantile_grid=self.quantile_grid.copy(),
            means=distribution_means,
            variances=distribution_variances,
        )

    def inverse_transform_joint(
        self,
        prediction: JointScorePrediction,
    ) -> JointDistributionPrediction:

        distribution_means: dict[str, FloatArray] = {}
        distribution_covariances: dict[str, FloatArray] = {}
        for outcome, state in self.states.items():
            loadings = state.scaled_components / np.sqrt(
                state.quadrature_weights[None, :]
            )
            raw_means = state.mean_quantile + np.asarray(
                prediction.means[outcome]
            ) @ loadings
            raw_covariance = np.einsum(
                "cij,cq,cr->iqjr",
                np.asarray(prediction.covariances[outcome]),
                loadings,
                loadings,
            )
            rearrangement = np.argsort(raw_means, axis=1)
            distribution_means[outcome] = np.take_along_axis(
                raw_means,
                rearrangement,
                axis=1,
            )
            point_count, quantile_count = raw_means.shape
            permutation = (
                np.arange(point_count)[:, None] * quantile_count
                + rearrangement
            ).ravel()
            flat_covariance = raw_covariance.reshape(
                point_count * quantile_count,
                point_count * quantile_count,
            )
            distribution_covariances[outcome] = flat_covariance[
                permutation[:, None],
                permutation[None, :],
            ].reshape(
                point_count,
                quantile_count,
                point_count,
                quantile_count,
            )

        return JointDistributionPrediction(
            quantile_grid=self.quantile_grid.copy(),
            means=distribution_means,
            covariances=distribution_covariances,
        )

    def empirical_quantiles(self, runs: RunBatch) -> dict[str, FloatArray]:
        return {
            outcome: np.vstack(
                [
                    np.quantile(values, self.quantile_grid, method="linear")
                    for values in samples
                ]
            )
            for outcome, samples in runs.particle_samples.items()
        }

    def _fit_outcome(self, curves: FloatArray) -> OutcomeFPCAState:
        mean_quantile = curves.mean(axis=0)
        scaled_centered = (curves - mean_quantile) * np.sqrt(
            self.quadrature_weights[None, :]
        )
        _, singular_values, right_vectors = np.linalg.svd(
            scaled_centered,
            full_matrices=False,
        )
        explained = np.square(singular_values)
        explained /= explained.sum()
        threshold_components = int(
            np.searchsorted(
                np.cumsum(explained),
                self.config.fpca_variance_target,
            )
            + 1
        )
        n_components = min(
            max(self.config.fpca_min_components, threshold_components),
            right_vectors.shape[0],
        )
        return OutcomeFPCAState(
            mean_quantile=mean_quantile,
            quadrature_weights=self.quadrature_weights.copy(),
            scaled_components=right_vectors[:n_components],
            explained_variance_ratio=explained,
        )

    def _transform(
        self,
        runs: RunBatch,
        curves: dict[str, FloatArray],
    ) -> DistributionRepresentation:
        rng = np.random.default_rng(self.config.random_seed)
        scores: dict[str, FloatArray] = {}
        score_variances: dict[str, FloatArray] = {}

        for outcome, state in self.states.items():
            scores[outcome] = self._project(curves[outcome], state)
            score_variances[outcome] = self._bootstrap_score_variances(
                runs.particle_samples[outcome],
                state,
                rng,
            )

        return DistributionRepresentation(
            run_ids=runs.run_ids,
            outcome_names=tuple(self.states),
            quantile_grid=self.quantile_grid.copy(),
            scores=scores,
            score_variances=score_variances,
        )

    @staticmethod
    def _project(curves: FloatArray, state: OutcomeFPCAState) -> FloatArray:
        scaled = (curves - state.mean_quantile) * np.sqrt(
            state.quadrature_weights[None, :]
        )
        return scaled @ state.scaled_components.T

    def _bootstrap_score_variances(
        self,
        samples: tuple[FloatArray, ...],
        state: OutcomeFPCAState,
        rng: np.random.Generator,
    ) -> FloatArray:
        variances = np.empty((len(samples), state.n_components), dtype=float)
        for run_index, values in enumerate(samples):
            indices = rng.integers(
                0,
                len(values),
                size=(self.config.bootstrap_replicates, len(values)),
            )
            bootstrap_samples = values[indices]
            bootstrap_curves = np.quantile(
                bootstrap_samples,
                self.quantile_grid,
                axis=1,
                method="linear",
            ).T
            bootstrap_scores = self._project(bootstrap_curves, state)
            variances[run_index] = bootstrap_scores.var(axis=0, ddof=1)
        return variances

    @staticmethod
    def _trapezoid_weights(grid: FloatArray) -> FloatArray:
        differences = np.diff(grid)
        weights = np.empty_like(grid)
        weights[0] = differences[0] / 2
        weights[-1] = differences[-1] / 2
        weights[1:-1] = (differences[:-1] + differences[1:]) / 2
        return weights / weights.sum()
