"""Non-overriding physical-direction annotations for admission decisions."""

from dataclasses import dataclass

from .admission_gate import EffectAdmissionDecision


PHYSICS_CONSISTENCY_STATUSES = (
    "direction_consistent",
    "direction_conflicting",
    "not_represented",
)


@dataclass(frozen=True, slots=True)
class PhysicsDirectionEvidence:
    """Direction-only evidence from a reduced physical model."""

    estimand_id: str
    outcome: str
    direction: int
    source_models: tuple[str, ...]
    directional_slopes: tuple[float, ...]
    fidelity: str
    calibrated_to_current_a: bool
    mechanism_scope: str


@dataclass(frozen=True, slots=True)
class PhysicsConsistencyAnnotation:
    """Read-only physical annotation attached to a statistical decision."""

    decision: EffectAdmissionDecision
    consistency_status: str
    statistical_direction: int
    physical_direction: int | None
    evidence: PhysicsDirectionEvidence | None
    admission_unchanged: bool = True


def annotate_physics_consistency(
    decision: EffectAdmissionDecision,
    evidence: tuple[PhysicsDirectionEvidence, ...],
) -> PhysicsConsistencyAnnotation:
    """Attach direction evidence without changing the admission decision."""

    statistical_direction = (
        1 if decision.point_mean_effect > 0.0
        else -1 if decision.point_mean_effect < 0.0
        else 0
    )
    represented = next(
        (
            item
            for item in evidence
            if item.estimand_id == decision.estimand.estimand_id
            and item.outcome == decision.outcome
        ),
        None,
    )
    if represented is None:
        return PhysicsConsistencyAnnotation(
            decision=decision,
            consistency_status="not_represented",
            statistical_direction=statistical_direction,
            physical_direction=None,
            evidence=None,
        )
    status = (
        "direction_consistent"
        if represented.direction == statistical_direction
        else "direction_conflicting"
    )
    return PhysicsConsistencyAnnotation(
        decision=decision,
        consistency_status=status,
        statistical_direction=statistical_direction,
        physical_direction=represented.direction,
        evidence=represented,
    )


def frozen_existing_physics_evidence() -> tuple[PhysicsDirectionEvidence, ...]:
    """Return physical evidence already represented by repository models."""

    return (
        PhysicsDirectionEvidence(
            estimand_id="argon_80_to_120",
            outcome="velocity_m_s",
            direction=1,
            source_models=("COMSOL-L4-laminar", "COMSOL-L5-k-omega"),
            directional_slopes=(
                0.3449331919867389,
                0.2238448099779388,
            ),
            fidelity="reduced_direction_probe",
            calibrated_to_current_a=False,
            mechanism_scope=(
                "argon flow to target-plane gas velocity; no calibrated "
                "particle or DPV observation model"
            ),
        ),
    )
