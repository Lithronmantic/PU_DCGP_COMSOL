
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .admission_gate import GATE_NAMES
from .benchmark_admission import SyntheticAdmissionObservation


@dataclass(frozen=True, slots=True)
class GatePowerDiagnostic:

    scenario_id: str
    sample_size: int
    replicate_count: int
    active_admission_rate: float
    null_false_admission_rate: float
    active_failed_gate_rates: Mapping[str, float]
    active_effect_admission_rates: Mapping[str, float]


def aggregate_gate_power_diagnostics(
    observations: tuple[SyntheticAdmissionObservation, ...],
) -> tuple[GatePowerDiagnostic, ...]:

    grouped: dict[
        tuple[str, int], list[SyntheticAdmissionObservation]
    ] = {}
    for observation in observations:
        grouped.setdefault(
            (observation.scenario_id, observation.sample_size),
            [],
        ).append(observation)
    admitted_statuses = {"admit", "conditional_admit", "exploratory_admit"}
    diagnostics = []
    for (scenario_id, sample_size), group in sorted(grouped.items()):
        active = [observation for observation in group if observation.is_active]
        null = [observation for observation in group if not observation.is_active]
        effect_keys = sorted(
            {
                (observation.treatment_name, observation.outcome)
                for observation in active
            }
        )
        diagnostics.append(
            GatePowerDiagnostic(
                scenario_id=scenario_id,
                sample_size=sample_size,
                replicate_count=len(
                    {observation.replicate_index for observation in group}
                ),
                active_admission_rate=float(
                    np.mean(
                        [
                            observation.status in admitted_statuses
                            for observation in active
                        ]
                    )
                ),
                null_false_admission_rate=float(
                    np.mean(
                        [
                            observation.status in admitted_statuses
                            for observation in null
                        ]
                    )
                ),
                active_failed_gate_rates={
                    gate: float(
                        np.mean(
                            [gate in observation.failed_gates for observation in active]
                        )
                    )
                    for gate in GATE_NAMES
                },
                active_effect_admission_rates={
                    f"{treatment}|{outcome}": float(
                        np.mean(
                            [
                                observation.status in admitted_statuses
                                for observation in active
                                if observation.treatment_name == treatment
                                and observation.outcome == outcome
                            ]
                        )
                    )
                    for treatment, outcome in effect_keys
                },
            )
        )
    return tuple(diagnostics)
