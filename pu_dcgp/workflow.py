"""Thin orchestration layer for PU-DCGP components."""

from dataclasses import dataclass

from .config import PUDCGPConfig
from .contracts import ContrastSpec, DistributionEffect, EvaluationResult, PreparedData
from .interfaces import (
    CausalContrastEstimator,
    DistributionEncoder,
    DistributionResponseModel,
    ModelEvaluator,
    RunDataSource,
)


@dataclass(slots=True)
class PUDCGPWorkflow:
    """Connect the declared components without implementing their statistics."""

    config: PUDCGPConfig
    data_source: RunDataSource
    encoder: DistributionEncoder
    model: DistributionResponseModel
    contrast_estimator: CausalContrastEstimator
    evaluator: ModelEvaluator

    def prepare(self) -> PreparedData:
        runs = self.data_source.load()
        distributions = self.encoder.fit_transform(runs)
        return PreparedData(runs=runs, distributions=distributions)

    def fit(self, data: PreparedData) -> None:
        self.model.fit(data)

    def estimate_effect(self, contrast: ContrastSpec) -> DistributionEffect:
        return self.contrast_estimator.estimate(self.model, self.encoder, contrast)

    def evaluate(self, data: PreparedData) -> EvaluationResult:
        return self.evaluator.evaluate(self.model, self.encoder, data)
