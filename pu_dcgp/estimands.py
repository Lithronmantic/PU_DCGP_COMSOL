"""Frozen A-group DOE estimands before effect calculation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DOEEstimand:
    """One controlled distributional contrast on the A-group design."""

    estimand_id: str
    treatment_name: str
    reference_value: float
    intervention_value: float
    claim_role: str
    effect_direction: str
    derived_reexpression: str | None = None


def a_group_doe_estimands() -> tuple[DOEEstimand, ...]:
    """Return the four frozen extreme-level DOE contrasts."""

    return (
        DOEEstimand(
            estimand_id="current_600_to_800",
            treatment_name="current_a",
            reference_value=600.0,
            intervention_value=800.0,
            claim_role="confirmatory",
            effect_direction="distribution_at_800_minus_distribution_at_600",
        ),
        DOEEstimand(
            estimand_id="argon_80_to_120",
            treatment_name="argon_flow_scfh",
            reference_value=80.0,
            intervention_value=120.0,
            claim_role="confirmatory",
            effect_direction="distribution_at_120_minus_distribution_at_80",
            derived_reexpression=(
                "H2/Ar setting ratio 0.03125 to 0.02083 if gas-flow units "
                "are confirmed compatible"
            ),
        ),
        DOEEstimand(
            estimand_id="powder_10_to_30",
            treatment_name="powder_feed_g_min",
            reference_value=10.0,
            intervention_value=30.0,
            claim_role="confirmatory",
            effect_direction="distribution_at_30_minus_distribution_at_10",
        ),
        DOEEstimand(
            estimand_id="distance_80_to_120",
            treatment_name="spray_distance_mm",
            reference_value=80.0,
            intervention_value=120.0,
            claim_role="exploratory",
            effect_direction="distribution_at_120_minus_distribution_at_80",
        ),
    )
