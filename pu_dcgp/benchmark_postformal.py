"""Typed contracts for post-formal benchmark diagnosis and paper tables."""

from dataclasses import dataclass

from .benchmark_hypotheses import BenchmarkHypothesisDecision
from .benchmark_runner import BenchmarkAggregateRecord


SUPPORTED_SCENARIOS = (
    "identified_balanced_particles",
    "identified_heterogeneous_particles",
)
FAILURE_SCENARIOS = (
    "sequence_aligned_drift",
    "module_sign_reversal",
    "insufficient_overlap",
)


@dataclass(frozen=True, slots=True)
class ShapeRecoveryCell:
    """H1 effect-shape recovery evidence for one supported cell."""

    scenario_id: str
    sample_size: int
    mean_gp_irmse: float
    no_pu_irmse: float
    pu_irmse: float
    no_pu_relative_reduction: float
    pu_relative_reduction: float


@dataclass(frozen=True, slots=True)
class CoverageCalibrationCell:
    """H2 simultaneous-coverage evidence at one sample size."""

    sample_size: int
    balanced_no_pu_coverage: float
    balanced_pu_coverage: float
    balanced_error_worsening: float
    heterogeneous_no_pu_coverage: float
    heterogeneous_pu_coverage: float
    heterogeneous_error_reduction: float


@dataclass(frozen=True, slots=True)
class UnsupportedAdmissionCell:
    """H3 pooled unsupported-target reporting at one sample size."""

    sample_size: int
    ungated_rate: float
    gated_rate: float
    relative_reduction: float


@dataclass(frozen=True, slots=True)
class RetainedPowerSummary:
    """H4 pooled supported-cell reporting behavior at 144 runs."""

    active_admission_power: float
    null_false_admission: float


@dataclass(frozen=True, slots=True)
class PredictionComparisonCell:
    """Held-out prediction comparison for one supported cell."""

    scenario_id: str
    sample_size: int
    mean_gp_normalized_rmse: float
    pu_normalized_rmse: float
    pu_mean_rmse_relative_reduction: float
    no_pu_wasserstein_rmse: float
    pu_wasserstein_rmse: float
    pu_wasserstein_relative_reduction: float


@dataclass(frozen=True, slots=True)
class PostFormalDiagnostics:
    """Complete evidence payload used by post-formal reports."""

    hypothesis_statuses: tuple[tuple[str, str], ...]
    shape_recovery: tuple[ShapeRecoveryCell, ...]
    coverage_calibration: tuple[CoverageCalibrationCell, ...]
    unsupported_admission: tuple[UnsupportedAdmissionCell, ...]
    retained_power: RetainedPowerSummary
    prediction: tuple[PredictionComparisonCell, ...]


def build_postformal_diagnostics(
    aggregates: tuple[BenchmarkAggregateRecord, ...],
    decisions: tuple[BenchmarkHypothesisDecision, ...],
    sample_sizes: tuple[int, ...],
    retained_power_sample_size: int = 144,
) -> PostFormalDiagnostics:
    """Assemble the formal decisions and their cell-level evidence."""

    return PostFormalDiagnostics(
        hypothesis_statuses=tuple(
            (decision.hypothesis_id, decision.status) for decision in decisions
        ),
        shape_recovery=diagnose_shape_recovery(aggregates, sample_sizes),
        coverage_calibration=diagnose_coverage_calibration(
            aggregates,
            sample_sizes,
        ),
        unsupported_admission=diagnose_unsupported_admission(
            aggregates,
            sample_sizes,
        ),
        retained_power=diagnose_retained_power(
            aggregates,
            retained_power_sample_size,
        ),
        prediction=diagnose_prediction(aggregates, sample_sizes),
    )


def diagnose_shape_recovery(
    aggregates: tuple[BenchmarkAggregateRecord, ...],
    sample_sizes: tuple[int, ...],
) -> tuple[ShapeRecoveryCell, ...]:
    """Expand the six H1 cells without collapsing them to the minimum."""

    records = _aggregate_index(aggregates)
    cells = []
    for scenario_id in SUPPORTED_SCENARIOS:
        for sample_size in sample_sizes:
            mean_error = records[
                (scenario_id, sample_size, "mean_gp")
            ].median_shape_normalized_irmse
            no_pu_error = records[
                (scenario_id, sample_size, "distribution_gp_no_pu")
            ].median_shape_normalized_irmse
            pu_error = records[
                (scenario_id, sample_size, "pu_dcgp")
            ].median_shape_normalized_irmse
            cells.append(
                ShapeRecoveryCell(
                    scenario_id=scenario_id,
                    sample_size=sample_size,
                    mean_gp_irmse=mean_error,
                    no_pu_irmse=no_pu_error,
                    pu_irmse=pu_error,
                    no_pu_relative_reduction=1.0 - no_pu_error / mean_error,
                    pu_relative_reduction=1.0 - pu_error / mean_error,
                )
            )
    return tuple(cells)


def diagnose_coverage_calibration(
    aggregates: tuple[BenchmarkAggregateRecord, ...],
    sample_sizes: tuple[int, ...],
    nominal_coverage: float = 0.95,
) -> tuple[CoverageCalibrationCell, ...]:
    """Expose the balanced and heterogeneous components of H2."""

    records = _aggregate_index(aggregates)
    balanced, heterogeneous = SUPPORTED_SCENARIOS
    cells = []
    for sample_size in sample_sizes:
        balanced_no_pu = records[
            (balanced, sample_size, "distribution_gp_no_pu")
        ].simultaneous_coverage_rate
        balanced_pu = records[
            (balanced, sample_size, "pu_dcgp")
        ].simultaneous_coverage_rate
        heterogeneous_no_pu = records[
            (heterogeneous, sample_size, "distribution_gp_no_pu")
        ].simultaneous_coverage_rate
        heterogeneous_pu = records[
            (heterogeneous, sample_size, "pu_dcgp")
        ].simultaneous_coverage_rate
        cells.append(
            CoverageCalibrationCell(
                sample_size=sample_size,
                balanced_no_pu_coverage=balanced_no_pu,
                balanced_pu_coverage=balanced_pu,
                balanced_error_worsening=(
                    abs(balanced_pu - nominal_coverage)
                    - abs(balanced_no_pu - nominal_coverage)
                ),
                heterogeneous_no_pu_coverage=heterogeneous_no_pu,
                heterogeneous_pu_coverage=heterogeneous_pu,
                heterogeneous_error_reduction=(
                    abs(heterogeneous_no_pu - nominal_coverage)
                    - abs(heterogeneous_pu - nominal_coverage)
                ),
            )
        )
    return tuple(cells)


def diagnose_unsupported_admission(
    aggregates: tuple[BenchmarkAggregateRecord, ...],
    sample_sizes: tuple[int, ...],
) -> tuple[UnsupportedAdmissionCell, ...]:
    """Expose the pooled ungated and gated components of H3."""

    records = _aggregate_index(aggregates)
    cells = []
    for sample_size in sample_sizes:
        ungated_rate = sum(
            records[
                (scenario_id, sample_size, "pu_dcgp")
            ].target_unsupported_admission_rate
            for scenario_id in FAILURE_SCENARIOS
        ) / len(FAILURE_SCENARIOS)
        gated_rate = sum(
            records[
                (scenario_id, sample_size, "support_gated_pu_dcgp")
            ].target_unsupported_admission_rate
            for scenario_id in FAILURE_SCENARIOS
        ) / len(FAILURE_SCENARIOS)
        cells.append(
            UnsupportedAdmissionCell(
                sample_size=sample_size,
                ungated_rate=ungated_rate,
                gated_rate=gated_rate,
                relative_reduction=(
                    (ungated_rate - gated_rate) / ungated_rate
                    if ungated_rate > 0.0
                    else 0.0
                ),
            )
        )
    return tuple(cells)


def diagnose_retained_power(
    aggregates: tuple[BenchmarkAggregateRecord, ...],
    sample_size: int = 144,
) -> RetainedPowerSummary:
    """Expose the two equally pooled reporting quantities used by H4."""

    records = _aggregate_index(aggregates)
    supported_records = tuple(
        records[(scenario_id, sample_size, "support_gated_pu_dcgp")]
        for scenario_id in SUPPORTED_SCENARIOS
    )
    return RetainedPowerSummary(
        active_admission_power=sum(
            record.active_admission_rate for record in supported_records
        )
        / len(supported_records),
        null_false_admission=sum(
            record.null_false_admission_rate for record in supported_records
        )
        / len(supported_records),
    )


def diagnose_prediction(
    aggregates: tuple[BenchmarkAggregateRecord, ...],
    sample_sizes: tuple[int, ...],
) -> tuple[PredictionComparisonCell, ...]:
    """Compare the two held-out prediction endpoints outside H1--H4."""

    records = _aggregate_index(aggregates)
    cells = []
    for scenario_id in SUPPORTED_SCENARIOS:
        for sample_size in sample_sizes:
            mean_rmse = records[
                (scenario_id, sample_size, "mean_gp")
            ].normalized_mean_prediction_rmse
            no_pu_wasserstein = records[
                (scenario_id, sample_size, "distribution_gp_no_pu")
            ].normalized_wasserstein_prediction_rmse
            pu_record = records[(scenario_id, sample_size, "pu_dcgp")]
            pu_mean_rmse = pu_record.normalized_mean_prediction_rmse
            pu_wasserstein = pu_record.normalized_wasserstein_prediction_rmse
            cells.append(
                PredictionComparisonCell(
                    scenario_id=scenario_id,
                    sample_size=sample_size,
                    mean_gp_normalized_rmse=mean_rmse,
                    pu_normalized_rmse=pu_mean_rmse,
                    pu_mean_rmse_relative_reduction=(
                        1.0 - pu_mean_rmse / mean_rmse
                    ),
                    no_pu_wasserstein_rmse=no_pu_wasserstein,
                    pu_wasserstein_rmse=pu_wasserstein,
                    pu_wasserstein_relative_reduction=(
                        1.0 - pu_wasserstein / no_pu_wasserstein
                    ),
                )
            )
    return tuple(cells)


def _aggregate_index(
    aggregates: tuple[BenchmarkAggregateRecord, ...],
) -> dict[tuple[str, int, str], BenchmarkAggregateRecord]:
    return {
        (record.scenario_id, record.sample_size, record.method_name): record
        for record in aggregates
    }
