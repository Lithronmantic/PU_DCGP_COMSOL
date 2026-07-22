"""Executable decisions for the four prespecified benchmark hypotheses."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_runner import BenchmarkAggregateRecord


@dataclass(frozen=True, slots=True)
class BenchmarkHypothesisDecision:
    """Formal pass, fail, or incomplete result for one frozen hypothesis."""

    hypothesis_id: str
    status: str
    evidence: Mapping[str, float]
    reason: str


def evaluate_benchmark_hypotheses(
    aggregates: tuple[BenchmarkAggregateRecord, ...],
    contract: SyntheticBenchmarkContract,
) -> tuple[BenchmarkHypothesisDecision, ...]:
    """Evaluate H1--H4 only when every required cell has 200 replicates."""

    records = {
        (record.scenario_id, record.sample_size, record.method_name): record
        for record in aggregates
    }
    supported = (
        "identified_balanced_particles",
        "identified_heterogeneous_particles",
    )
    failures = (
        "sequence_aligned_drift",
        "module_sign_reversal",
        "insufficient_overlap",
    )
    sample_sizes = contract.scenarios[0].sample_sizes

    h1_keys = tuple(
        (scenario, sample_size, method)
        for scenario in supported
        for sample_size in sample_sizes
        for method in ("mean_gp", "distribution_gp_no_pu", "pu_dcgp")
    )
    if not _is_formal_complete(records, h1_keys, contract.replicate_count):
        h1 = _incomplete("H1")
    else:
        pu_improvements = []
        no_pu_improvements = []
        for scenario in supported:
            for sample_size in sample_sizes:
                mean_error = records[
                    (scenario, sample_size, "mean_gp")
                ].median_shape_normalized_irmse
                no_pu_error = records[
                    (scenario, sample_size, "distribution_gp_no_pu")
                ].median_shape_normalized_irmse
                pu_error = records[
                    (scenario, sample_size, "pu_dcgp")
                ].median_shape_normalized_irmse
                no_pu_improvements.append(1.0 - no_pu_error / mean_error)
                pu_improvements.append(1.0 - pu_error / mean_error)
        minimum_pu = float(min(pu_improvements))
        h1 = BenchmarkHypothesisDecision(
            hypothesis_id="H1",
            status="pass" if minimum_pu >= 0.10 else "fail",
            evidence={
                "minimum_pu_relative_irmse_reduction": minimum_pu,
                "minimum_no_pu_relative_irmse_reduction": float(
                    min(no_pu_improvements)
                ),
            },
            reason="PU-DCGP must reduce shape-effect IRMSE by at least 0.10 in every supported scenario-size cell.",
        )

    h2_keys = tuple(
        (scenario, sample_size, method)
        for scenario in supported
        for sample_size in sample_sizes
        for method in ("distribution_gp_no_pu", "pu_dcgp")
    )
    if not _is_formal_complete(records, h2_keys, contract.replicate_count):
        h2 = _incomplete("H2")
    else:
        heterogeneous_improvements = []
        balanced_worsening = []
        for sample_size in sample_sizes:
            balanced_no_pu = records[
                (supported[0], sample_size, "distribution_gp_no_pu")
            ].simultaneous_coverage_rate
            balanced_pu = records[
                (supported[0], sample_size, "pu_dcgp")
            ].simultaneous_coverage_rate
            heterogeneous_no_pu = records[
                (supported[1], sample_size, "distribution_gp_no_pu")
            ].simultaneous_coverage_rate
            heterogeneous_pu = records[
                (supported[1], sample_size, "pu_dcgp")
            ].simultaneous_coverage_rate
            heterogeneous_improvements.append(
                abs(heterogeneous_no_pu - 0.95)
                - abs(heterogeneous_pu - 0.95)
            )
            balanced_worsening.append(
                abs(balanced_pu - 0.95) - abs(balanced_no_pu - 0.95)
            )
        minimum_improvement = float(min(heterogeneous_improvements))
        maximum_worsening = float(max(balanced_worsening))
        h2 = BenchmarkHypothesisDecision(
            hypothesis_id="H2",
            status=(
                "pass"
                if minimum_improvement >= 0.02 and maximum_worsening <= 0.02
                else "fail"
            ),
            evidence={
                "minimum_heterogeneous_coverage_error_reduction": (
                    minimum_improvement
                ),
                "maximum_balanced_coverage_error_worsening": maximum_worsening,
            },
            reason="Coverage improvement and balanced-count non-inferiority must hold at every sample size.",
        )

    h3_keys = tuple(
        (scenario, sample_size, method)
        for scenario in failures
        for sample_size in sample_sizes
        for method in ("pu_dcgp", "support_gated_pu_dcgp")
    )
    if not _is_formal_complete(records, h3_keys, contract.replicate_count):
        h3 = _incomplete("H3")
    else:
        gated_rates = []
        reductions = []
        for sample_size in sample_sizes:
            ungated = float(
                np.mean(
                    [
                        records[
                            (scenario, sample_size, "pu_dcgp")
                        ].target_unsupported_admission_rate
                        for scenario in failures
                    ]
                )
            )
            gated = float(
                np.mean(
                    [
                        records[
                            (scenario, sample_size, "support_gated_pu_dcgp")
                        ].target_unsupported_admission_rate
                        for scenario in failures
                    ]
                )
            )
            gated_rates.append(gated)
            reductions.append((ungated - gated) / ungated if ungated > 0 else 0.0)
        maximum_gated = float(max(gated_rates))
        minimum_reduction = float(min(reductions))
        h3 = BenchmarkHypothesisDecision(
            hypothesis_id="H3",
            status=(
                "pass"
                if maximum_gated <= 0.05 and minimum_reduction >= 0.50
                else "fail"
            ),
            evidence={
                "maximum_gated_unsupported_admission": maximum_gated,
                "minimum_relative_reduction_from_ungated": minimum_reduction,
            },
            reason="The three prespecified failure targets are pooled equally at each sample size.",
        )

    h4_keys = tuple(
        (scenario, 144, "support_gated_pu_dcgp") for scenario in supported
    )
    if not _is_formal_complete(records, h4_keys, contract.replicate_count):
        h4 = _incomplete("H4")
    else:
        power = float(
            np.mean([records[key].active_admission_rate for key in h4_keys])
        )
        null_false = float(
            np.mean([records[key].null_false_admission_rate for key in h4_keys])
        )
        h4 = BenchmarkHypothesisDecision(
            hypothesis_id="H4",
            status="pass" if power >= 0.80 and null_false <= 0.05 else "fail",
            evidence={
                "supported_active_admission_power": power,
                "supported_null_false_admission": null_false,
            },
            reason="The two identified 144-run scenarios are pooled equally.",
        )
    return h1, h2, h3, h4


def _is_formal_complete(
    records: Mapping[tuple[str, int, str], BenchmarkAggregateRecord],
    keys: tuple[tuple[str, int, str], ...],
    replicate_count: int,
) -> bool:
    return all(
        key in records and records[key].replicate_count == replicate_count
        for key in keys
    )


def _incomplete(hypothesis_id: str) -> BenchmarkHypothesisDecision:
    return BenchmarkHypothesisDecision(
        hypothesis_id=hypothesis_id,
        status="not_evaluable",
        evidence={},
        reason="Required scenario-size-method cells do not all contain 200 formal replicates.",
    )
