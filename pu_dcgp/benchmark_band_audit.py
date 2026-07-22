
from dataclasses import dataclass

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_generator import (
    generate_identified_balanced_dataset,
    generate_identified_heterogeneous_dataset,
    generate_insufficient_overlap_dataset,
    generate_module_sign_reversal_dataset,
    generate_sequence_aligned_drift_dataset,
)
from .benchmark_admission import evaluate_synthetic_admission_decisions
from .benchmark_methods import (
    apply_admission_decisions,
    fit_benchmark_point_effect_methods,
)
from .benchmark_metrics import evaluate_benchmark_method
from .benchmark_selection import evaluate_benchmark_selection
from .benchmark_settings import benchmark_pilot_config


@dataclass(frozen=True, slots=True)
class MethodBandPilotEntry:

    scenario_id: str
    method_name: str
    effect_count: int
    median_normalized_irmse: float
    shape_median_normalized_irmse: float
    simultaneous_coverage_rate: float
    active_coverage_rate: float
    shape_coverage_rate: float
    normalized_mean_band_width: float
    runtime_seconds: float
    active_admission_rate: float
    null_false_admission_rate: float
    target_unsupported_admitted: bool | None


@dataclass(frozen=True, slots=True)
class MethodBandPilotAudit:

    sample_size: int
    replicate_index: int
    scenario_count: int
    entries: tuple[MethodBandPilotEntry, ...]
    comparison_authorized: bool
    passed: bool


def audit_method_band_pilot(
    contract: SyntheticBenchmarkContract,
    sample_size: int = 48,
    replicate_index: int = 0,
) -> MethodBandPilotAudit:

    config = benchmark_pilot_config(contract)
    generators = (
        generate_identified_balanced_dataset,
        generate_identified_heterogeneous_dataset,
        generate_sequence_aligned_drift_dataset,
        generate_module_sign_reversal_dataset,
        generate_insufficient_overlap_dataset,
    )
    entries = []
    scenario_ids = []
    for generator in generators:
        dataset = generator(contract, sample_size, replicate_index)
        scenario_ids.append(dataset.scenario_id)
        results = fit_benchmark_point_effect_methods(dataset, contract, config)
        decisions = evaluate_synthetic_admission_decisions(
            dataset,
            contract,
            config,
            results[2],
        )
        gated_result = apply_admission_decisions(results[2], decisions)
        for result in results + (gated_result,):
            metrics = evaluate_benchmark_method(result, dataset, contract)
            selection = evaluate_benchmark_selection(result, dataset, contract)
            entries.append(
                MethodBandPilotEntry(
                    scenario_id=metrics.scenario_id,
                    method_name=metrics.method_name,
                    effect_count=len(metrics.effect_metrics),
                    median_normalized_irmse=metrics.median_normalized_irmse,
                    shape_median_normalized_irmse=(
                        metrics.shape_median_normalized_irmse
                    ),
                    simultaneous_coverage_rate=(
                        metrics.simultaneous_coverage_rate
                    ),
                    active_coverage_rate=metrics.active_coverage_rate,
                    shape_coverage_rate=metrics.shape_coverage_rate,
                    normalized_mean_band_width=(
                        metrics.normalized_mean_band_width
                    ),
                    runtime_seconds=metrics.runtime_seconds,
                    active_admission_rate=selection.active_admission_rate,
                    null_false_admission_rate=(
                        selection.null_false_admission_rate
                    ),
                    target_unsupported_admitted=(
                        selection.target_unsupported_admitted
                    ),
                )
            )
    finite_values = np.asarray(
        [
            value
            for entry in entries
            for value in (
                entry.median_normalized_irmse,
                entry.shape_median_normalized_irmse,
                entry.simultaneous_coverage_rate,
                entry.active_coverage_rate,
                entry.shape_coverage_rate,
                entry.normalized_mean_band_width,
                entry.runtime_seconds,
                entry.active_admission_rate,
                entry.null_false_admission_rate,
            )
        ]
    )
    passed = (
        len(set(scenario_ids)) == len(generators)
        and len(entries) == len(generators) * 4
        and all(entry.effect_count == 12 for entry in entries)
        and np.all(np.isfinite(finite_values))
        and all(
            0.0 <= rate <= 1.0
            for entry in entries
            for rate in (
                entry.simultaneous_coverage_rate,
                entry.active_coverage_rate,
                entry.shape_coverage_rate,
                entry.active_admission_rate,
                entry.null_false_admission_rate,
            )
        )
    )
    return MethodBandPilotAudit(
        sample_size=sample_size,
        replicate_index=replicate_index,
        scenario_count=len(generators),
        entries=tuple(entries),
        comparison_authorized=False,
        passed=bool(passed),
    )
