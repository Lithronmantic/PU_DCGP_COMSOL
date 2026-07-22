"""Abstract module boundaries for PU-DCGP."""

from abc import ABC, abstractmethod

from .contracts import (
    ContrastSpec,
    DistributionEffect,
    DistributionPrediction,
    DistributionRepresentation,
    EvaluationResult,
    FloatArray,
    PreparedData,
    RunBatch,
    ScorePrediction,
)


class RunDataSource(ABC):
    """Load and align run-level process settings and particle observations."""

    @abstractmethod
    def load(self) -> RunBatch:
        """Return one analysis-ready run batch."""


class DistributionEncoder(ABC):
    """Convert particle samples into uncertainty-aware distribution scores."""

    @abstractmethod
    def fit_transform(self, runs: RunBatch) -> DistributionRepresentation:
        """Fit the representation and encode the observed runs."""

    @abstractmethod
    def transform(self, runs: RunBatch) -> DistributionRepresentation:
        """Encode new runs using the fitted representation."""

    @abstractmethod
    def inverse_transform(self, prediction: ScorePrediction) -> DistributionPrediction:
        """Reconstruct outcome distributions from predicted scores."""


class DistributionResponseModel(ABC):
    """Model process-to-distribution relationships in score space."""

    @abstractmethod
    def fit(self, data: PreparedData) -> None:
        """Fit the response model using score uncertainty."""

    @abstractmethod
    def predict(self, treatments: FloatArray, contexts: FloatArray) -> ScorePrediction:
        """Predict distribution scores at requested settings."""


class CausalContrastEstimator(ABC):
    """Estimate controlled distributional effects from a fitted model."""

    @abstractmethod
    def estimate(
        self,
        model: DistributionResponseModel,
        encoder: DistributionEncoder,
        contrast: ContrastSpec,
    ) -> DistributionEffect:
        """Return one distributional intervention contrast."""


class ModelEvaluator(ABC):
    """Evaluate predictive and effect stability under a declared protocol."""

    @abstractmethod
    def evaluate(
        self,
        model: DistributionResponseModel,
        encoder: DistributionEncoder,
        data: PreparedData,
    ) -> EvaluationResult:
        """Return named validation metrics and concise notes."""
