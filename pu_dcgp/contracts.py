"""Data objects exchanged between PU-DCGP modules."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RunBatch:
    """Run-level process settings with particle observations for each outcome."""

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
    """Low-dimensional distribution scores and their measurement uncertainty."""

    run_ids: tuple[str, ...]
    outcome_names: tuple[str, ...]
    quantile_grid: FloatArray
    scores: Mapping[str, FloatArray]
    score_variances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Raw run data paired with its fitted distribution representation."""

    runs: RunBatch
    distributions: DistributionRepresentation


@dataclass(frozen=True, slots=True)
class ScorePrediction:
    """Predictive moments in distribution-score space."""

    means: Mapping[str, FloatArray]
    variances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class JointScorePrediction:
    """Score means and component-wise covariance across prediction points."""

    means: Mapping[str, FloatArray]
    covariances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class DistributionPrediction:
    """Predictive moments reconstructed on a common outcome quantile grid."""

    quantile_grid: FloatArray
    means: Mapping[str, FloatArray]
    variances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class JointDistributionPrediction:
    """Quantile means and covariance indexed as point, quantile, point, quantile."""

    quantile_grid: FloatArray
    means: Mapping[str, FloatArray]
    covariances: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    """Controlled comparison for one process variable."""

    treatment_name: str
    reference_value: float
    intervention_value: float
    fixed_treatments: Mapping[str, float]
    fixed_context: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DistributionEffect:
    """Estimated intervention effect across the outcome quantile grid."""

    treatment_name: str
    quantile_grid: FloatArray
    effects: Mapping[str, FloatArray]
    lower_bounds: Mapping[str, FloatArray]
    upper_bounds: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Named evaluation values returned by a validation protocol."""

    metrics: Mapping[str, float]
    notes: tuple[str, ...] = ()
