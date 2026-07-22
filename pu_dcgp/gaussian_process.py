"""NumPy exact Gaussian-process models for the PU-DCGP response layer."""

from dataclasses import dataclass

import numpy as np

from .config import PUDCGPConfig
from .contracts import (
    FloatArray,
    JointScorePrediction,
    PreparedData,
    RunBatch,
    ScorePrediction,
)
from .interfaces import DistributionResponseModel
from .mean_baselines import run_mean_targets


@dataclass(frozen=True, slots=True)
class GPHyperparameters:
    process_lengthscale: float
    context_lengthscale: float
    signal_variance: float
    noise_variance: float


class ExactGaussianProcessRegressor:
    """Single-output exact GP with optional known observation variance."""

    def __init__(
        self,
        config: PUDCGPConfig,
        n_process_features: int | None = None,
    ) -> None:
        self.config = config
        self.n_process_features = n_process_features
        self.predictor_mean: FloatArray | None = None
        self.predictor_scale: FloatArray | None = None
        self.target_mean: float | None = None
        self.target_scale: float | None = None
        self.training_predictors: FloatArray | None = None
        self.cholesky: FloatArray | None = None
        self.alpha: FloatArray | None = None
        self.hyperparameters: GPHyperparameters | None = None

    def fit(
        self,
        predictors: FloatArray,
        targets: FloatArray,
        observation_variance: FloatArray | None = None,
    ) -> None:
        predictors = np.asarray(predictors, dtype=float)
        targets = np.asarray(targets, dtype=float)
        self.predictor_mean = predictors.mean(axis=0)
        self.predictor_scale = predictors.std(axis=0)
        self.predictor_scale[self.predictor_scale == 0] = 1.0
        self.target_mean = float(targets.mean())
        self.target_scale = float(targets.std())
        if self.target_scale == 0:
            self.target_scale = 1.0

        standardized_predictors = (
            predictors - self.predictor_mean
        ) / self.predictor_scale
        standardized_targets = (targets - self.target_mean) / self.target_scale
        if observation_variance is None:
            known_variance = np.zeros(len(targets))
        else:
            known_variance = np.asarray(observation_variance, dtype=float) / (
                self.target_scale**2
            )

        process_distances, context_distances = self._distance_components(
            standardized_predictors,
            standardized_predictors,
        )
        best_nll = np.inf
        best_hyperparameters: GPHyperparameters | None = None

        context_candidates = (
            self.config.gp_context_lengthscale_candidates
            if self.n_process_features is not None
            and self.n_process_features < standardized_predictors.shape[1]
            else (1.0,)
        )
        for process_lengthscale in self.config.gp_lengthscale_candidates:
            for context_lengthscale in context_candidates:
                scaled_distances = (
                    process_distances / process_lengthscale**2
                    + context_distances / context_lengthscale**2
                )
                correlation = self._matern32_correlation(scaled_distances)
                for signal_variance in self.config.gp_signal_variance_candidates:
                    signal_kernel = signal_variance * correlation
                    for noise_variance in self.config.gp_noise_variance_candidates:
                        covariance = signal_kernel.copy()
                        covariance.flat[:: len(covariance) + 1] += (
                            known_variance + noise_variance + 1e-9
                        )
                        cholesky = np.linalg.cholesky(covariance)
                        alpha = np.linalg.solve(
                            cholesky.T,
                            np.linalg.solve(cholesky, standardized_targets),
                        )
                        nll = (
                            0.5 * standardized_targets @ alpha
                            + np.log(np.diag(cholesky)).sum()
                            + 0.5 * len(targets) * np.log(2 * np.pi)
                        )
                        if nll < best_nll:
                            best_nll = float(nll)
                            best_hyperparameters = GPHyperparameters(
                                process_lengthscale=process_lengthscale,
                                context_lengthscale=context_lengthscale,
                                signal_variance=signal_variance,
                                noise_variance=noise_variance,
                            )

        self.training_predictors = standardized_predictors
        self.hyperparameters = best_hyperparameters
        correlation = self._matern32_correlation(
            process_distances / best_hyperparameters.process_lengthscale**2
            + context_distances / best_hyperparameters.context_lengthscale**2
        )
        covariance = best_hyperparameters.signal_variance * correlation
        covariance.flat[:: len(covariance) + 1] += (
            known_variance + best_hyperparameters.noise_variance + 1e-9
        )
        self.cholesky = np.linalg.cholesky(covariance)
        self.alpha = np.linalg.solve(
            self.cholesky.T,
            np.linalg.solve(self.cholesky, standardized_targets),
        )

    def predict(self, predictors: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return latent predictive means and marginal variances."""

        means, covariance = self.predict_joint(predictors)
        return means, np.diag(covariance).copy()

    def predict_joint(
        self,
        predictors: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return latent predictive means and covariance across test points."""

        if self.hyperparameters is None:
            raise RuntimeError("The Gaussian process has not been fitted")
        standardized_predictors = (
            np.asarray(predictors, dtype=float) - self.predictor_mean
        ) / self.predictor_scale
        process_distances, context_distances = self._distance_components(
            self.training_predictors,
            standardized_predictors,
        )
        cross_covariance = (
            self.hyperparameters.signal_variance
            * self._matern32_correlation(
                process_distances / self.hyperparameters.process_lengthscale**2
                + context_distances / self.hyperparameters.context_lengthscale**2
            )
        )
        standardized_mean = cross_covariance.T @ self.alpha
        solved = np.linalg.solve(self.cholesky, cross_covariance)
        test_process_distances, test_context_distances = self._distance_components(
            standardized_predictors,
            standardized_predictors,
        )
        test_covariance = (
            self.hyperparameters.signal_variance
            * self._matern32_correlation(
                test_process_distances
                / self.hyperparameters.process_lengthscale**2
                + test_context_distances
                / self.hyperparameters.context_lengthscale**2
            )
        )
        standardized_covariance = test_covariance - solved.T @ solved
        standardized_covariance = (
            standardized_covariance + standardized_covariance.T
        ) / 2
        diagonal = np.diag_indices_from(standardized_covariance)
        standardized_covariance[diagonal] = np.maximum(
            standardized_covariance[diagonal],
            0.0,
        )
        return (
            self.target_mean + self.target_scale * standardized_mean,
            self.target_scale**2 * standardized_covariance,
        )

    def _distance_components(
        self,
        left: FloatArray,
        right: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        split = self.n_process_features or left.shape[1]
        process_differences = (
            left[:, None, :split] - right[None, :, :split]
        )
        process_distances = np.square(process_differences).sum(axis=2)
        if split == left.shape[1]:
            context_distances = np.zeros_like(process_distances)
        else:
            context_differences = (
                left[:, None, split:] - right[None, :, split:]
            )
            context_distances = np.square(context_differences).sum(axis=2)
        return process_distances, context_distances

    @staticmethod
    def _matern32_correlation(
        scaled_squared_distances: FloatArray,
    ) -> FloatArray:
        scaled_distance = np.sqrt(3 * scaled_squared_distances)
        return (1 + scaled_distance) * np.exp(-scaled_distance)


class GaussianProcessDistributionModel(DistributionResponseModel):
    """Independent score GPs with optional particle-bootstrap uncertainty."""

    def __init__(
        self,
        config: PUDCGPConfig,
        use_particle_uncertainty: bool,
    ) -> None:
        self.config = config
        self.use_particle_uncertainty = use_particle_uncertainty
        self.models: dict[str, tuple[ExactGaussianProcessRegressor, ...]] = {}

    def fit(self, data: PreparedData) -> None:
        predictors = response_predictors(
            data.runs.treatment_values,
            data.runs.context_values,
        )
        models: dict[str, tuple[ExactGaussianProcessRegressor, ...]] = {}
        for outcome, scores in data.distributions.scores.items():
            outcome_models = []
            for component in range(scores.shape[1]):
                model = ExactGaussianProcessRegressor(
                    self.config,
                    n_process_features=4,
                )
                observation_variance = None
                if self.use_particle_uncertainty:
                    observation_variance = data.distributions.score_variances[
                        outcome
                    ][:, component]
                model.fit(
                    predictors,
                    scores[:, component],
                    observation_variance,
                )
                outcome_models.append(model)
            models[outcome] = tuple(outcome_models)
        self.models = models

    def predict(self, treatments: FloatArray, contexts: FloatArray) -> ScorePrediction:
        predictors = response_predictors(treatments, contexts)
        means: dict[str, FloatArray] = {}
        variances: dict[str, FloatArray] = {}
        for outcome, outcome_models in self.models.items():
            predictions = [model.predict(predictors) for model in outcome_models]
            means[outcome] = np.column_stack([prediction[0] for prediction in predictions])
            variances[outcome] = np.column_stack(
                [prediction[1] for prediction in predictions]
            )
        return ScorePrediction(means=means, variances=variances)

    def predict_joint(
        self,
        treatments: FloatArray,
        contexts: FloatArray,
    ) -> JointScorePrediction:
        """Return independent-component score covariances across test points."""

        predictors = response_predictors(treatments, contexts)
        means: dict[str, FloatArray] = {}
        covariances: dict[str, FloatArray] = {}
        for outcome, outcome_models in self.models.items():
            predictions = [
                model.predict_joint(predictors) for model in outcome_models
            ]
            means[outcome] = np.column_stack(
                [prediction[0] for prediction in predictions]
            )
            covariances[outcome] = np.stack(
                [prediction[1] for prediction in predictions]
            )
        return JointScorePrediction(
            means=means,
            covariances=covariances,
        )


class GaussianProcessMeanModel:
    """Independent exact GPs for the three run-level outcome means."""

    def __init__(self, config: PUDCGPConfig) -> None:
        self.config = config
        self.outcome_names: tuple[str, ...] = ()
        self.models: tuple[ExactGaussianProcessRegressor, ...] = ()

    def fit(self, runs: RunBatch) -> None:
        self.outcome_names, targets = run_mean_targets(runs)
        predictors = response_predictors(
            runs.treatment_values,
            runs.context_values,
        )
        fitted_models = []
        for outcome_index in range(targets.shape[1]):
            model = ExactGaussianProcessRegressor(
                self.config,
                n_process_features=4,
            )
            model.fit(predictors, targets[:, outcome_index])
            fitted_models.append(model)
        self.models = tuple(fitted_models)

    def predict(
        self,
        treatments: FloatArray,
        contexts: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        predictors = response_predictors(treatments, contexts)
        predictions = [model.predict(predictors) for model in self.models]
        return (
            np.column_stack([prediction[0] for prediction in predictions]),
            np.column_stack([prediction[1] for prediction in predictions]),
        )

    def predict_joint(
        self,
        treatments: FloatArray,
        contexts: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return outcome means and outcome-wise covariance across test points."""

        predictors = response_predictors(treatments, contexts)
        predictions = [model.predict_joint(predictors) for model in self.models]
        return (
            np.column_stack([prediction[0] for prediction in predictions]),
            np.stack([prediction[1] for prediction in predictions]),
        )


def response_predictors(
    treatments: FloatArray,
    contexts: FloatArray,
) -> FloatArray:
    """Combine process settings with execution order for the A-group response model."""

    return np.column_stack(
        [
            np.asarray(treatments, dtype=float),
            np.asarray(contexts, dtype=float)[:, 0],
        ]
    )
