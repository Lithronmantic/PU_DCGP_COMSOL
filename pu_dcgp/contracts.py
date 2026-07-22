
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RunBatch:

    run_ids: tuple[str, ...]
    groups: tuple[str, ...]
    doe_modules: tuple[str, ...]
    treatment_names: tuple[str, ...]
    treatment_values: FloatArray
    controlled_process_names: tuple[str, ...]
    controlled_process_values: FloatArray
    context_names: tuple[str, ...]
    context_values: FloatArray
    particle_samples: Mapping[str, tuple[FloatArray, ...]]


@dataclass(frozen=True, slots=True)
class DistributionRepresentation:

    run_ids: tuple[str, ...]
    outcome_names: tuple[str, ...]
    quantile_grid: FloatArray
    scores: Mapping[str, FloatArray]
    score_variances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class PreparedData:

    runs: RunBatch
    distributions: DistributionRepresentation


@dataclass(frozen=True, slots=True)
class ScorePrediction:

    means: Mapping[str, FloatArray]
    variances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class JointScorePrediction:

    means: Mapping[str, FloatArray]
    covariances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class DistributionPrediction:

    quantile_grid: FloatArray
    means: Mapping[str, FloatArray]
    variances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class JointDistributionPrediction:

    quantile_grid: FloatArray
    means: Mapping[str, FloatArray]
    covariances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class ContrastSpec:

    treatment_name: str
    reference_value: float
    intervention_value: float
    fixed_treatments: Mapping[str, float]
    fixed_context: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DistributionEffect:

    treatment_name: str
    quantile_grid: FloatArray
    effects: Mapping[str, FloatArray]
    lower_bounds: Mapping[str, FloatArray]
    upper_bounds: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class EvaluationResult:

    metrics: Mapping[str, float]
    notes: tuple[str, ...] = ()
