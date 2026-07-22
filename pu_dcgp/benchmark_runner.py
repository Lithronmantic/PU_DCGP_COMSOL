"""Replicated execution and aggregation for the frozen synthetic benchmark."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .benchmark_admission import evaluate_synthetic_admission_decisions
from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import (
    generate_identified_balanced_dataset,
    generate_identified_heterogeneous_dataset,
    generate_insufficient_overlap_dataset,
    generate_module_sign_reversal_dataset,
    generate_sequence_aligned_drift_dataset,
)
from .benchmark_methods import (
    apply_admission_decisions,
    fit_benchmark_point_effect_methods,
)
from .benchmark_metrics import evaluate_benchmark_method
from .benchmark_selection import evaluate_benchmark_selection
from .benchmark_prediction import evaluate_benchmark_predictions
from .config import PUDCGPConfig


@dataclass(frozen=True, slots=True)
class BenchmarkReplicateRecord:
    """All prespecified effect, coverage, and selection summaries for one fit."""

    scenario_id: str
    sample_size: int
    replicate_index: int
    method_name: str
    median_normalized_irmse: float
    shape_median_normalized_irmse: float
    simultaneous_coverage_rate: float
    active_coverage_rate: float
    shape_coverage_rate: float
    normalized_mean_band_width: float
    active_admission_rate: float
    null_false_admission_rate: float
    target_unsupported_admitted: bool | None
    runtime_seconds: float
    normalized_mean_prediction_rmse: float
    normalized_wasserstein_prediction_rmse: float | None
    prediction_validation_seconds: float


@dataclass(frozen=True, slots=True)
class BenchmarkAggregateRecord:
    """Replicate aggregation for one scenario, sample size, and method."""

    scenario_id: str
    sample_size: int
    method_name: str
    replicate_count: int
    median_normalized_irmse: float
    median_shape_normalized_irmse: float
    simultaneous_coverage_rate: float
    active_coverage_rate: float
    shape_coverage_rate: float
    normalized_mean_band_width: float
    active_admission_rate: float
    null_false_admission_rate: float
    target_unsupported_admission_rate: float | None
    median_runtime_seconds: float
    normalized_mean_prediction_rmse: float
    normalized_wasserstein_prediction_rmse: float | None
    median_prediction_validation_seconds: float


SCENARIO_GENERATORS = {
    "identified_balanced_particles": generate_identified_balanced_dataset,
    "identified_heterogeneous_particles": (
        generate_identified_heterogeneous_dataset
    ),
    "sequence_aligned_drift": generate_sequence_aligned_drift_dataset,
    "module_sign_reversal": generate_module_sign_reversal_dataset,
    "insufficient_overlap": generate_insufficient_overlap_dataset,
}


def run_benchmark_replicates(
    contract: SyntheticBenchmarkContract,
    config: PUDCGPConfig,
    sample_size: int,
    replicate_indices: tuple[int, ...],
    scenario_ids: tuple[str, ...],
) -> tuple[BenchmarkReplicateRecord, ...]:
    """Run selected frozen scenarios without interpreting hypotheses."""

    records = []
    for scenario_id in scenario_ids:
        generator = SCENARIO_GENERATORS[scenario_id]
        for replicate_index in replicate_indices:
            dataset = generator(contract, sample_size, replicate_index)
            validation_start = perf_counter()
            prediction_metrics = evaluate_benchmark_predictions(
                dataset,
                contract,
                config,
                n_folds=config.benchmark_cv_folds,
            )
            prediction_validation_seconds = perf_counter() - validation_start
            point_results = fit_benchmark_point_effect_methods(
                dataset,
                contract,
                config,
            )
            decisions = evaluate_synthetic_admission_decisions(
                dataset,
                contract,
                config,
                point_results[2],
            )
            gated_result = apply_admission_decisions(
                point_results[2],
                decisions,
            )
            for result in point_results + (gated_result,):
                method = evaluate_benchmark_method(result, dataset, contract)
                selection = evaluate_benchmark_selection(
                    result,
                    dataset,
                    contract,
                )
                prediction = prediction_metrics[result.method_name]
                records.append(
                    BenchmarkReplicateRecord(
                        scenario_id=scenario_id,
                        sample_size=sample_size,
                        replicate_index=replicate_index,
                        method_name=result.method_name,
                        median_normalized_irmse=(
                            method.median_normalized_irmse
                        ),
                        shape_median_normalized_irmse=(
                            method.shape_median_normalized_irmse
                        ),
                        simultaneous_coverage_rate=(
                            method.simultaneous_coverage_rate
                        ),
                        active_coverage_rate=method.active_coverage_rate,
                        shape_coverage_rate=method.shape_coverage_rate,
                        normalized_mean_band_width=(
                            method.normalized_mean_band_width
                        ),
                        active_admission_rate=(
                            selection.active_admission_rate
                        ),
                        null_false_admission_rate=(
                            selection.null_false_admission_rate
                        ),
                        target_unsupported_admitted=(
                            selection.target_unsupported_admitted
                        ),
                        runtime_seconds=method.runtime_seconds,
                        normalized_mean_prediction_rmse=(
                            prediction.normalized_mean_rmse
                        ),
                        normalized_wasserstein_prediction_rmse=(
                            prediction.normalized_wasserstein_rmse
                        ),
                        prediction_validation_seconds=(
                            prediction_validation_seconds
                        ),
                    )
                )
    return tuple(records)


def aggregate_benchmark_records(
    records: tuple[BenchmarkReplicateRecord, ...],
) -> tuple[BenchmarkAggregateRecord, ...]:
    """Aggregate replicate records without changing frozen metric definitions."""

    grouped: dict[tuple[str, int, str], list[BenchmarkReplicateRecord]] = {}
    for record in records:
        grouped.setdefault(
            (record.scenario_id, record.sample_size, record.method_name),
            [],
        ).append(record)
    aggregates = []
    for (scenario_id, sample_size, method_name), group in sorted(grouped.items()):
        target_values = [
            record.target_unsupported_admitted
            for record in group
            if record.target_unsupported_admitted is not None
        ]
        aggregates.append(
            BenchmarkAggregateRecord(
                scenario_id=scenario_id,
                sample_size=sample_size,
                method_name=method_name,
                replicate_count=len(group),
                median_normalized_irmse=float(
                    np.median(
                        [record.median_normalized_irmse for record in group]
                    )
                ),
                median_shape_normalized_irmse=float(
                    np.median(
                        [
                            record.shape_median_normalized_irmse
                            for record in group
                        ]
                    )
                ),
                simultaneous_coverage_rate=float(
                    np.mean(
                        [record.simultaneous_coverage_rate for record in group]
                    )
                ),
                active_coverage_rate=float(
                    np.mean([record.active_coverage_rate for record in group])
                ),
                shape_coverage_rate=float(
                    np.mean([record.shape_coverage_rate for record in group])
                ),
                normalized_mean_band_width=float(
                    np.mean(
                        [record.normalized_mean_band_width for record in group]
                    )
                ),
                active_admission_rate=float(
                    np.mean([record.active_admission_rate for record in group])
                ),
                null_false_admission_rate=float(
                    np.mean(
                        [record.null_false_admission_rate for record in group]
                    )
                ),
                target_unsupported_admission_rate=(
                    float(np.mean(target_values)) if target_values else None
                ),
                median_runtime_seconds=float(
                    np.median([record.runtime_seconds for record in group])
                ),
                normalized_mean_prediction_rmse=float(
                    np.mean(
                        [
                            record.normalized_mean_prediction_rmse
                            for record in group
                        ]
                    )
                ),
                normalized_wasserstein_prediction_rmse=(
                    float(
                        np.mean(
                            [
                                record.normalized_wasserstein_prediction_rmse
                                for record in group
                            ]
                        )
                    )
                    if group[0].normalized_wasserstein_prediction_rmse
                    is not None
                    else None
                ),
                median_prediction_validation_seconds=float(
                    np.median(
                        [
                            record.prediction_validation_seconds
                            for record in group
                        ]
                    )
                ),
            )
        )
    return tuple(aggregates)
