"""Frozen local COMSOL response design around the corrected effective exit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "effective_exit_local_jacobian"
MODEL_DIR = HERE / "comsol_models" / "h11_effective_exit_local_jacobian"
CONTRACT_PATH = OUTPUT_DIR / "h11_effective_exit_local_jacobian_contract.json"


@dataclass(frozen=True, slots=True)
class LocalJacobianContract:
    center_temperature_k: float = 11_160.0
    center_speed_m_s: float = 1_090.0
    temperature_step_k: float = 400.0
    speed_step_m_s: float = 100.0
    pilot_particles_per_size: int = 127
    final_particles_per_size: int = 1_023
    expected_diameter_nodes: int = 7
    minimum_pilot_primary_particles: int = 7
    minimum_final_primary_particles: int = 70
    particle_count_convergence_limit_fraction: float = 0.01
    maximum_jacobian_condition_number: float = 20.0
    minimum_absolute_determinant: float = 1.0e-5

    def validate(self) -> None:
        if (self.center_temperature_k, self.center_speed_m_s) != (11_160.0, 1_090.0):
            raise ValueError("The corrected effective-exit center is frozen")
        if (self.temperature_step_k, self.speed_step_m_s) != (400.0, 100.0):
            raise ValueError("The local perturbation sizes are frozen")
        if self.pilot_particles_per_size != 127 or self.final_particles_per_size != 1023:
            raise ValueError("The pilot/final particle ladder is frozen")
        if (
            self.minimum_final_primary_particles != 70
            or self.particle_count_convergence_limit_fraction != 0.01
        ):
            raise ValueError(
                "The inherited H11 particle-count gates cannot change"
            )
        for case in self.cases().values():
            if not 8_000 <= case["temperature_k"] <= 12_000:
                raise ValueError("A local temperature case is outside the tested envelope")
            if not 400 <= case["speed_m_s"] <= 1_200:
                raise ValueError("A local speed case is outside the tested envelope")

    def cases(self) -> dict[str, dict[str, float]]:
        return {
            "temperature_minus": {
                "temperature_k": self.center_temperature_k - self.temperature_step_k,
                "speed_m_s": self.center_speed_m_s,
            },
            "temperature_plus": {
                "temperature_k": self.center_temperature_k + self.temperature_step_k,
                "speed_m_s": self.center_speed_m_s,
            },
            "speed_minus": {
                "temperature_k": self.center_temperature_k,
                "speed_m_s": self.center_speed_m_s - self.speed_step_m_s,
            },
            "speed_plus": {
                "temperature_k": self.center_temperature_k,
                "speed_m_s": self.center_speed_m_s + self.speed_step_m_s,
            },
        }


def build_contract() -> dict:
    contract = LocalJacobianContract()
    contract.validate()
    return {
        "schema_version": "h11_effective_exit_local_jacobian_contract_v1",
        "status": "pass_frozen_local_response_design",
        "contract": asdict(contract),
        "cases": contract.cases(),
        "response": (
            "A-detected-diameter-weighted particle temperature and velocity medians "
            "inside the frozen low-speed DPV aperture at 100 mm"
        ),
        "pilot_gates": [
            "all four gas solves pass conservation and boundedness",
            "all four particle solves pass and retain seven diameter nodes",
            "centered direct derivatives have positive physical signs",
            "the two-by-two local Jacobian determinant and condition number pass",
        ],
        "escalation_rule": (
            "Run 1023 particles per size only after all pilot gates pass; "
            "no A or B outcome may select a perturbation case."
        ),
        "particle_count_rule": (
            "Reuse the previously frozen H11 gates: at least 70 primary-aperture "
            "particles at the final count and no more than one-percent change "
            "from 127 to 1023 for primary-aperture detected-weighted particle "
            "temperature and speed q10/q50/q90 in every local case."
        ),
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> None:
    payload = build_contract()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(CONTRACT_PATH)


if __name__ == "__main__":
    main()
