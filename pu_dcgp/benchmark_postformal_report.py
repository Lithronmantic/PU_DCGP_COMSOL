"""Paper-table rendering contracts for the post-formal diagnostics."""

from dataclasses import dataclass
from pathlib import Path

from .benchmark_postformal import PostFormalDiagnostics


SCENARIO_LABELS = {
    "identified_balanced_particles": "Balanced counts",
    "identified_heterogeneous_particles": "Heterogeneous counts",
}


@dataclass(frozen=True, slots=True)
class PaperTableSection:
    """One named Markdown table ready for manuscript assembly."""

    section_id: str
    title: str
    markdown: str


@dataclass(frozen=True, slots=True)
class PostFormalPaperTables:
    """Ordered paper-table sections generated from formal evidence."""

    sections: tuple[PaperTableSection, ...]


def build_postformal_paper_tables(
    diagnostics: PostFormalDiagnostics,
) -> PostFormalPaperTables:
    """Build all formal paper tables in their manuscript order."""

    return PostFormalPaperTables(
        sections=(
            render_shape_recovery_table(diagnostics),
            render_coverage_calibration_table(diagnostics),
            render_unsupported_admission_table(diagnostics),
            render_retained_power_table(diagnostics),
            render_prediction_table(diagnostics),
        )
    )


def render_postformal_paper_tables(
    tables: PostFormalPaperTables,
    hypothesis_statuses: tuple[tuple[str, str], ...],
) -> str:
    """Render one compact Markdown artifact for manuscript assembly."""

    statuses = ", ".join(
        f"{hypothesis_id} `{status}`"
        for hypothesis_id, status in hypothesis_statuses
    )
    sections = "\n\n".join(
        f"## {section.title}\n\n{section.markdown}"
        for section in tables.sections
    )
    return (
        "# Post-formal benchmark paper tables\n\n"
        f"Formal decisions: {statuses}.\n\n"
        f"{sections}\n\n"
        "## Interpretation boundary\n\n"
        "H1--H4 retain their prespecified separate decisions. Prediction "
        "endpoints are descriptive and do not alter those decisions. Physics "
        "constraints are outside the fitted benchmark methods.\n"
    )


def write_postformal_paper_tables(
    path: Path,
    diagnostics: PostFormalDiagnostics,
) -> None:
    """Write the complete post-formal paper-table artifact."""

    tables = build_postformal_paper_tables(diagnostics)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_postformal_paper_tables(
            tables,
            diagnostics.hypothesis_statuses,
        ),
        encoding="utf-8",
    )


def render_shape_recovery_table(
    diagnostics: PostFormalDiagnostics,
) -> PaperTableSection:
    """Render the six H1 effect-shape recovery cells."""

    rows = tuple(
        (
            SCENARIO_LABELS[cell.scenario_id],
            str(cell.sample_size),
            f"{cell.mean_gp_irmse:.4f}",
            f"{cell.no_pu_irmse:.4f}",
            f"{cell.pu_irmse:.4f}",
            f"{100.0 * cell.no_pu_relative_reduction:.1f}%",
            f"{100.0 * cell.pu_relative_reduction:.1f}%",
        )
        for cell in diagnostics.shape_recovery
    )
    return PaperTableSection(
        section_id="h1_shape_recovery",
        title="H1: distributional effect-shape recovery",
        markdown=_markdown_table(
            (
                "Scenario",
                "n",
                "Mean GP IRMSE",
                "No-PU IRMSE",
                "PU IRMSE",
                "No-PU reduction",
                "PU reduction",
            ),
            rows,
        ),
    )


def render_coverage_calibration_table(
    diagnostics: PostFormalDiagnostics,
) -> PaperTableSection:
    """Render the three H2 balanced/heterogeneous calibration cells."""

    rows = tuple(
        (
            str(cell.sample_size),
            f"{cell.balanced_no_pu_coverage:.4f}",
            f"{cell.balanced_pu_coverage:.4f}",
            f"{cell.balanced_error_worsening:+.4f}",
            f"{cell.heterogeneous_no_pu_coverage:.4f}",
            f"{cell.heterogeneous_pu_coverage:.4f}",
            f"{cell.heterogeneous_error_reduction:+.4f}",
        )
        for cell in diagnostics.coverage_calibration
    )
    return PaperTableSection(
        section_id="h2_coverage_calibration",
        title="H2: simultaneous coverage calibration",
        markdown=_markdown_table(
            (
                "n",
                "Balanced no-PU",
                "Balanced PU",
                "Balanced error change",
                "Heterogeneous no-PU",
                "Heterogeneous PU",
                "Heterogeneous error reduction",
            ),
            rows,
        ),
    )


def render_unsupported_admission_table(
    diagnostics: PostFormalDiagnostics,
) -> PaperTableSection:
    """Render the three H3 pooled unsupported-reporting cells."""

    rows = tuple(
        (
            str(cell.sample_size),
            f"{cell.ungated_rate:.3f}",
            f"{cell.gated_rate:.3f}",
            f"{100.0 * cell.relative_reduction:.1f}%",
        )
        for cell in diagnostics.unsupported_admission
    )
    return PaperTableSection(
        section_id="h3_unsupported_admission",
        title="H3: unsupported-target admission",
        markdown=_markdown_table(
            ("n", "Ungated rate", "Gated rate", "Relative reduction"),
            rows,
        ),
    )


def render_retained_power_table(
    diagnostics: PostFormalDiagnostics,
) -> PaperTableSection:
    """Render the pooled H4 power and null false-admission quantities."""

    summary = diagnostics.retained_power
    return PaperTableSection(
        section_id="h4_retained_power",
        title="H4: retained reporting power",
        markdown=_markdown_table(
            ("Active admission power", "Null false admission"),
            (
                (
                    f"{summary.active_admission_power:.4f}",
                    f"{summary.null_false_admission:.4f}",
                ),
            ),
        ),
    )


def render_prediction_table(
    diagnostics: PostFormalDiagnostics,
) -> PaperTableSection:
    """Render the six held-out prediction comparisons."""

    rows = tuple(
        (
            SCENARIO_LABELS[cell.scenario_id],
            str(cell.sample_size),
            f"{cell.mean_gp_normalized_rmse:.4f}",
            f"{cell.pu_normalized_rmse:.4f}",
            f"{100.0 * cell.pu_mean_rmse_relative_reduction:.2f}%",
            f"{cell.no_pu_wasserstein_rmse:.4f}",
            f"{cell.pu_wasserstein_rmse:.4f}",
            f"{100.0 * cell.pu_wasserstein_relative_reduction:.2f}%",
        )
        for cell in diagnostics.prediction
    )
    return PaperTableSection(
        section_id="prediction_endpoints",
        title="Held-out prediction endpoints",
        markdown=_markdown_table(
            (
                "Scenario",
                "n",
                "Mean GP mean RMSE",
                "PU mean RMSE",
                "Mean RMSE reduction",
                "No-PU Wasserstein RMSE",
                "PU Wasserstein RMSE",
                "Wasserstein reduction",
            ),
            rows,
        ),
    )


def _markdown_table(
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = tuple("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join((header, separator, *body))
