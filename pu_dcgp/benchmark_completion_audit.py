"""Requirement-level integrity audit for formal benchmark records."""

from dataclasses import dataclass

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_runner import BenchmarkReplicateRecord
from .benchmark_settings import FormalBenchmarkPlan


@dataclass(frozen=True, slots=True)
class FormalCheckpointAudit:
    """Structural and numerical evidence for formal benchmark completion."""

    record_count: int
    completed_dataset_count: int
    expected_dataset_count: int
    missing_dataset_count: int
    unexpected_dataset_count: int
    incomplete_dataset_count: int
    invalid_record_count: int
    pu_gated_mismatch_count: int
    integrity_passed: bool
    formal_complete: bool


def audit_formal_checkpoint_records(
    records: tuple[BenchmarkReplicateRecord, ...],
    contract: SyntheticBenchmarkContract,
    plan: FormalBenchmarkPlan,
) -> FormalCheckpointAudit:
    """Check axes, four-method completeness, numeric ranges, and invariants."""

    expected_keys = {
        (scenario_id, sample_size, replicate_index)
        for scenario_id in plan.scenario_ids
        for sample_size in plan.sample_sizes
        for replicate_index in plan.replicate_indices
    }
    methods_by_dataset: dict[tuple[str, int, int], set[str]] = {}
    records_by_key = {}
    invalid_record_count = 0
    for record in records:
        dataset_key = (
            record.scenario_id,
            record.sample_size,
            record.replicate_index,
        )
        methods_by_dataset.setdefault(dataset_key, set()).add(record.method_name)
        records_by_key[dataset_key + (record.method_name,)] = record
        if not _record_is_valid(record):
            invalid_record_count += 1
    actual_keys = set(methods_by_dataset)
    required_methods = set(contract.methods)
    complete_keys = {
        key
        for key, method_names in methods_by_dataset.items()
        if method_names == required_methods
    }
    incomplete_keys = {
        key
        for key, method_names in methods_by_dataset.items()
        if method_names != required_methods
    }
    pu_gated_mismatch_count = 0
    for key in complete_keys:
        pu = records_by_key[key + ("pu_dcgp",)]
        gated = records_by_key[key + ("support_gated_pu_dcgp",)]
        if not _pu_outputs_match(pu, gated):
            pu_gated_mismatch_count += 1
    unexpected_keys = actual_keys - expected_keys
    missing_keys = expected_keys - complete_keys
    integrity_passed = (
        not unexpected_keys
        and not incomplete_keys
        and invalid_record_count == 0
        and pu_gated_mismatch_count == 0
    )
    return FormalCheckpointAudit(
        record_count=len(records),
        completed_dataset_count=len(complete_keys),
        expected_dataset_count=len(expected_keys),
        missing_dataset_count=len(missing_keys),
        unexpected_dataset_count=len(unexpected_keys),
        incomplete_dataset_count=len(incomplete_keys),
        invalid_record_count=invalid_record_count,
        pu_gated_mismatch_count=pu_gated_mismatch_count,
        integrity_passed=integrity_passed,
        formal_complete=(integrity_passed and complete_keys == expected_keys),
    )


def _record_is_valid(record: BenchmarkReplicateRecord) -> bool:
    finite_values = (
        record.median_normalized_irmse,
        record.shape_median_normalized_irmse,
        record.normalized_mean_band_width,
        record.runtime_seconds,
        record.normalized_mean_prediction_rmse,
        record.prediction_validation_seconds,
    )
    probability_values = (
        record.simultaneous_coverage_rate,
        record.active_coverage_rate,
        record.shape_coverage_rate,
        record.active_admission_rate,
        record.null_false_admission_rate,
    )
    wasserstein_valid = (
        record.normalized_wasserstein_prediction_rmse is None
        if record.method_name == "mean_gp"
        else record.normalized_wasserstein_prediction_rmse is not None
        and np.isfinite(record.normalized_wasserstein_prediction_rmse)
        and record.normalized_wasserstein_prediction_rmse >= 0.0
    )
    target_valid = (
        record.target_unsupported_admitted is None
        or isinstance(record.target_unsupported_admitted, bool)
    )
    return bool(
        all(np.isfinite(value) and value >= 0.0 for value in finite_values)
        and all(0.0 <= value <= 1.0 for value in probability_values)
        and wasserstein_valid
        and target_valid
    )


def _pu_outputs_match(
    pu: BenchmarkReplicateRecord,
    gated: BenchmarkReplicateRecord,
) -> bool:
    fields = (
        "median_normalized_irmse",
        "shape_median_normalized_irmse",
        "simultaneous_coverage_rate",
        "active_coverage_rate",
        "shape_coverage_rate",
        "normalized_mean_band_width",
        "runtime_seconds",
        "normalized_mean_prediction_rmse",
        "normalized_wasserstein_prediction_rmse",
        "prediction_validation_seconds",
    )
    return all(getattr(pu, field) == getattr(gated, field) for field in fields)
