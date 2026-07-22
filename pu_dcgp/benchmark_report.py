"""Deterministic Markdown reporting for formal benchmark records."""

from pathlib import Path

from .benchmark_completion_audit import audit_formal_checkpoint_records
from .benchmark_contract import SyntheticBenchmarkContract
from .benchmark_hypotheses import evaluate_benchmark_hypotheses
from .benchmark_runner import (
    BenchmarkReplicateRecord,
    aggregate_benchmark_records,
)
from .benchmark_settings import benchmark_formal_config, formal_benchmark_plan


def render_formal_benchmark_report(
    records: tuple[BenchmarkReplicateRecord, ...],
    contract: SyntheticBenchmarkContract,
) -> str:
    """Render completion evidence and prespecified metrics without reinterpretation."""

    plan = formal_benchmark_plan(contract, benchmark_formal_config(contract))
    audit = audit_formal_checkpoint_records(records, contract, plan)
    aggregates = aggregate_benchmark_records(records)
    decisions = evaluate_benchmark_hypotheses(aggregates, contract)
    lines = [
        "# Formal PU-DCGP benchmark report",
        "",
        "## Completion audit",
        "",
        f"- Formal complete: `{str(audit.formal_complete).lower()}`",
        f"- Integrity passed: `{str(audit.integrity_passed).lower()}`",
        f"- Unique method records: {audit.record_count} / {plan.method_record_count}",
        f"- Complete datasets: {audit.completed_dataset_count} / {audit.expected_dataset_count}",
        f"- Missing / unexpected / incomplete datasets: {audit.missing_dataset_count} / {audit.unexpected_dataset_count} / {audit.incomplete_dataset_count}",
        f"- Invalid records / PU-gated mismatches: {audit.invalid_record_count} / {audit.pu_gated_mismatch_count}",
        "",
        "## Prespecified hypotheses",
        "",
        "| Hypothesis | Status | Evidence |",
        "|---|---|---|",
    ]
    for decision in decisions:
        evidence = "; ".join(
            f"{key}={value:.4f}" for key, value in decision.evidence.items()
        ) or "required formal cells incomplete"
        lines.append(
            f"| {decision.hypothesis_id} | {decision.status} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Prediction and effect metrics",
            "",
            "| Scenario | n | Method | Mean pred. RMSE | Wasserstein RMSE | Shape IRMSE | Simultaneous coverage | Active admission | Null false admission | Unsupported target admission |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in aggregates:
        wasserstein = _optional_number(
            record.normalized_wasserstein_prediction_rmse
        )
        target = _optional_number(record.target_unsupported_admission_rate)
        lines.append(
            "| "
            f"{record.scenario_id} | {record.sample_size} | {record.method_name} | "
            f"{record.normalized_mean_prediction_rmse:.4f} | {wasserstein} | "
            f"{record.median_shape_normalized_irmse:.4f} | "
            f"{record.simultaneous_coverage_rate:.4f} | "
            f"{record.active_admission_rate:.4f} | "
            f"{record.null_false_admission_rate:.4f} | {target} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "H1--H4 are formal only when the completion audit is true. "
                "Each component claim follows its own frozen decision and cannot "
                "be repaired by another passing hypothesis. Physics constraints "
                "are not part of this benchmark."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_formal_benchmark_report(
    path: Path,
    records: tuple[BenchmarkReplicateRecord, ...],
    contract: SyntheticBenchmarkContract,
) -> None:
    """Write the deterministic Markdown report."""

    path.write_text(
        render_formal_benchmark_report(records, contract),
        encoding="utf-8",
    )


def _optional_number(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"
