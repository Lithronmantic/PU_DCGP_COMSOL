"""Design-support audit for frozen A-group DOE estimands."""

from dataclasses import dataclass

import numpy as np

from .contracts import RunBatch
from .estimands import DOEEstimand


@dataclass(frozen=True, slots=True)
class MatchedContrastStratum:
    """One exact-matching stratum containing both contrast levels."""

    fixed_treatments: tuple[tuple[str, float], ...]
    reference_runs: int
    intervention_runs: int
    reference_mean_sequence: float
    intervention_mean_sequence: float
    sequence_gap: float
    shared_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContrastSupportAudit:
    """Structural and execution-sequence support for one DOE estimand."""

    estimand: DOEEstimand
    strata: tuple[MatchedContrastStratum, ...]
    reference_runs: int
    intervention_runs: int
    modules_within_comparison: tuple[str, ...]
    median_absolute_sequence_gap: float
    normalized_median_sequence_gap: float
    positive_sequence_gaps: int
    negative_sequence_gaps: int
    zero_sequence_gaps: int
    support_level: str
    support_reasons: tuple[str, ...]


def audit_contrast_support(
    runs: RunBatch,
    estimand: DOEEstimand,
) -> ContrastSupportAudit:
    """Audit exact matches without calculating particle-response effects."""

    treatment_index = runs.treatment_names.index(estimand.treatment_name)
    other_indices = [
        index
        for index in range(len(runs.treatment_names))
        if index != treatment_index
    ]
    grouped_rows: dict[tuple[float, ...], list[int]] = {}
    for row_index, treatment_row in enumerate(runs.treatment_values):
        grouped_rows.setdefault(tuple(treatment_row[other_indices]), []).append(
            row_index
        )

    sequence = np.asarray(runs.context_values[:, 0], dtype=float)
    strata = []
    for fixed_values, row_indices in sorted(grouped_rows.items()):
        rows = np.asarray(row_indices, dtype=int)
        treatment_values = runs.treatment_values[rows, treatment_index]
        reference_rows = rows[treatment_values == estimand.reference_value]
        intervention_rows = rows[treatment_values == estimand.intervention_value]
        if len(reference_rows) == 0 or len(intervention_rows) == 0:
            continue
        reference_modules = {
            _module_code(runs.doe_modules[index]) for index in reference_rows
        }
        intervention_modules = {
            _module_code(runs.doe_modules[index]) for index in intervention_rows
        }
        reference_mean = float(sequence[reference_rows].mean())
        intervention_mean = float(sequence[intervention_rows].mean())
        strata.append(
            MatchedContrastStratum(
                fixed_treatments=tuple(
                    (runs.treatment_names[index], float(value))
                    for index, value in zip(other_indices, fixed_values)
                ),
                reference_runs=len(reference_rows),
                intervention_runs=len(intervention_rows),
                reference_mean_sequence=reference_mean,
                intervention_mean_sequence=intervention_mean,
                sequence_gap=intervention_mean - reference_mean,
                shared_modules=tuple(
                    sorted(reference_modules & intervention_modules)
                ),
            )
        )

    gaps = np.asarray([stratum.sequence_gap for stratum in strata], dtype=float)
    reference_runs = sum(stratum.reference_runs for stratum in strata)
    intervention_runs = sum(stratum.intervention_runs for stratum in strata)
    modules = tuple(
        sorted(
            {
                module
                for stratum in strata
                for module in stratum.shared_modules
            }
        )
    )
    median_absolute_gap = float(np.median(np.abs(gaps))) if len(gaps) else np.inf
    sequence_span = float(sequence.max() - sequence.min())
    normalized_gap = median_absolute_gap / sequence_span
    positive_gaps = int((gaps > 0).sum())
    negative_gaps = int((gaps < 0).sum())
    zero_gaps = int((gaps == 0).sum())
    support_level, support_reasons = _classify_support(
        estimand,
        len(strata),
        reference_runs,
        intervention_runs,
        modules,
        normalized_gap,
        positive_gaps,
        negative_gaps,
    )
    return ContrastSupportAudit(
        estimand=estimand,
        strata=tuple(strata),
        reference_runs=reference_runs,
        intervention_runs=intervention_runs,
        modules_within_comparison=modules,
        median_absolute_sequence_gap=median_absolute_gap,
        normalized_median_sequence_gap=normalized_gap,
        positive_sequence_gaps=positive_gaps,
        negative_sequence_gaps=negative_gaps,
        zero_sequence_gaps=zero_gaps,
        support_level=support_level,
        support_reasons=support_reasons,
    )


def _classify_support(
    estimand: DOEEstimand,
    matched_strata: int,
    reference_runs: int,
    intervention_runs: int,
    modules: tuple[str, ...],
    normalized_gap: float,
    positive_gaps: int,
    negative_gaps: int,
) -> tuple[str, tuple[str, ...]]:
    reasons = []
    if matched_strata < 5:
        reasons.append("fewer than five exact-matching strata")
    if min(reference_runs, intervention_runs) < 8:
        reasons.append("fewer than eight matched runs in one contrast arm")
    if len(modules) < 2:
        reasons.append("within-module comparisons occur in fewer than two modules")
    if normalized_gap > 0.25:
        reasons.append("median sequence gap exceeds one quarter of the A sequence")
    if reasons:
        return "insufficient", tuple(reasons)

    conditional_reasons = []
    if normalized_gap > 0.10:
        conditional_reasons.append(
            "median sequence gap exceeds one tenth of the A sequence"
        )
    if positive_gaps == 0 or negative_gaps == 0:
        conditional_reasons.append("all nonzero sequence gaps have one direction")
    if estimand.claim_role == "exploratory":
        conditional_reasons.append("estimand was frozen as exploratory")
    if conditional_reasons:
        return "conditional", tuple(conditional_reasons)
    return "eligible", ("structural support gate passed",)


def _module_code(module: str) -> str:
    return module.partition("｜")[0]
