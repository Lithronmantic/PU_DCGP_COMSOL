
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

    @abstractmethod
    def load(self) -> RunBatch:


class DistributionEncoder(ABC):

    @abstractmethod
    def fit_transform(self, runs: RunBatch) -> DistributionRepresentation:

    @abstractmethod
    def transform(self, runs: RunBatch) -> DistributionRepresentation:

    @abstractmethod
    def inverse_transform(self, prediction: ScorePrediction) -> DistributionPrediction:


class DistributionResponseModel(ABC):

    @abstractmethod
    def fit(self, data: PreparedData) -> None:

    @abstractmethod
    def predict(self, treatments: FloatArray, contexts: FloatArray) -> ScorePrediction:


class CausalContrastEstimator(ABC):

    @abstractmethod
    def estimate(
        self,
        model: DistributionResponseModel,
        encoder: DistributionEncoder,
        contrast: ContrastSpec,
    ) -> DistributionEffect:


class ModelEvaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        model: DistributionResponseModel,
        encoder: DistributionEncoder,
        data: PreparedData,
    ) -> EvaluationResult:
