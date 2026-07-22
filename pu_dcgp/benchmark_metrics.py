"""Known-truth effect and simultaneous-band metrics for benchmark methods."""

from dataclasses import dataclass

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import SyntheticBenchmarkDataset
from .benchmark_methods import BenchmarkMethodResult


@dataclass(frozen=True, slots=True)
class BenchmarkEffectMetric:
    """Known-truth error and coverage for one estimated effect curve."""

    estimand_id: str
    outcome: str
    is_active: bool
    is_shape_effect: bool
    normalized_irmse: float
    simultaneous_covered: bool
    normalized_mean_band_width: float


@dataclass(frozen=True, slots=True)
class BenchmarkMethodMetrics:
    """Effect-level metrics and prespecified within-replicate summaries."""

    method_name: str
    scenario_id: str
    sample_size: int
    replicate_index: int
    effect_metrics: tuple[BenchmarkEffectMetric, ...]
    median_normalized_irmse: float
    maximum_normalized_irmse: float
    shape_median_normalized_irmse: float
    simultaneous_coverage_rate: float
    active_coverage_rate: float
    shape_coverage_rate: float
    normalized_mean_band_width: float
    runtime_seconds: float


def evaluate_benchmark_method(
    result: BenchmarkMethodResult,
    dataset: SyntheticBenchmarkDataset,
    contract: SyntheticBenchmarkContract,
) -> BenchmarkMethodMetrics:
    """Compare one fitted method with the dataset's aligned analytic truths."""

    truth_map = {
        (truth.estimand_id, truth.outcome): truth for truth in dataset.truths
    }
    outcome_scales = {
        mechanism.outcome: mechanism.baseline_scale
        for mechanism in contract.mechanisms
    }
    effect_metrics = []
    for effect in result.effects:
        truth = truth_map[(effect.estimand_id, effect.outcome)]
        scale = outcome_scales[effect.outcome]
        is_shape_effect = bool(np.ptp(truth.effect) > 1e-12)
        effect_metrics.append(
            BenchmarkEffectMetric(
                estimand_id=effect.estimand_id,
                outcome=effect.outcome,
                is_active=truth.is_active,
                is_shape_effect=is_shape_effect,
                normalized_irmse=float(
                    np.sqrt(np.mean(np.square(effect.point_effect - truth.effect)))
                    / scale
                ),
                simultaneous_covered=bool(
                    np.all(truth.effect >= effect.lower_bound)
                    and np.all(truth.effect <= effect.upper_bound)
                ),
                normalized_mean_band_width=float(
                    np.mean(effect.upper_bound - effect.lower_bound) / scale
                ),
            )
        )
    errors = np.asarray(
        [metric.normalized_irmse for metric in effect_metrics]
    )
    coverage = np.asarray(
        [metric.simultaneous_covered for metric in effect_metrics],
        dtype=float,
    )
    active = np.asarray([metric.is_active for metric in effect_metrics])
    shape = np.asarray([metric.is_shape_effect for metric in effect_metrics])
    widths = np.asarray(
        [metric.normalized_mean_band_width for metric in effect_metrics]
    )
    return BenchmarkMethodMetrics(
        method_name=result.method_name,
        scenario_id=result.scenario_id,
        sample_size=result.sample_size,
        replicate_index=result.replicate_index,
        effect_metrics=tuple(effect_metrics),
        median_normalized_irmse=float(np.median(errors)),
        maximum_normalized_irmse=float(np.max(errors)),
        shape_median_normalized_irmse=float(np.median(errors[shape])),
        simultaneous_coverage_rate=float(np.mean(coverage)),
        active_coverage_rate=float(np.mean(coverage[active])),
        shape_coverage_rate=float(np.mean(coverage[shape])),
        normalized_mean_band_width=float(np.mean(widths)),
        runtime_seconds=(
            result.preparation_seconds
            + result.fit_seconds
            + result.prediction_seconds
        ),
    )
