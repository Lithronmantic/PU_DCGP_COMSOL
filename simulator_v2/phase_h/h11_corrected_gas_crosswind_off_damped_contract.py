"""Freeze the same-mesh 0.35-to-0.36 turbulence-damping recovery test."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from simulator_v2.phase_h.h11_corrected_gas_crosswind_off import paths
from simulator_v2.phase_h.h11_corrected_gas_crosswind_off_contract import (
    CrosswindOffContract,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import _sha256


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "corrected_gas_crosswind_off_damped"
MODEL_DIR = HERE / "comsol_models" / "h11_corrected_gas_crosswind_off_damped"
CONTRACT_PATH = OUTPUT_DIR / "h11_corrected_gas_crosswind_off_damped_contract.json"
SOURCE_MODEL = paths(2)["model"].with_name(
    f"{paths(2)['model'].stem}_partial.mph"
)
SOURCE_FAILURE = paths(2)["audit"].with_name(
    f"{paths(2)['audit'].stem}_failure.json"
)


@dataclass(frozen=True, slots=True)
class DampedRecoveryContract:
    source_converged_load_fraction: float = 0.35
    target_load_fraction: float = 0.36
    flow_group_damping: float = 0.50
    turbulence_group_damping: float = 0.15
    relative_tolerance: float = 5e-4
    maximum_segregated_iterations: int = 2000
    mesh_level: int = 2
    expected_elements: int = 55_291

    def validate(self) -> None:
        if (
            self.source_converged_load_fraction,
            self.target_load_fraction,
        ) != (0.35, 0.36):
            raise ValueError("The local recovery interval is frozen")
        if (self.flow_group_damping, self.turbulence_group_damping) != (
            0.50,
            0.15,
        ):
            raise ValueError("The diagnostic damping values are frozen")
        if (
            self.relative_tolerance,
            self.maximum_segregated_iterations,
            self.mesh_level,
            self.expected_elements,
        ) != (5e-4, 2000, 2, 55_291):
            raise ValueError("The local numerical gate is frozen")


def build_contract() -> dict:
    base = CrosswindOffContract()
    base.validate()
    contract = DampedRecoveryContract()
    contract.validate()
    if not SOURCE_MODEL.is_file() or not SOURCE_FAILURE.is_file():
        raise FileNotFoundError("The audited crosswind-off failure checkpoint is absent")
    return {
        "schema_version": "h11_corrected_gas_crosswind_off_damped_contract_v1",
        "status": "pass_frozen_same_mesh_damping_recovery_contract",
        "contract": asdict(contract),
        "unchanged_physics_contract": asdict(base),
        "source_partial_model": str(SOURCE_MODEL.resolve()),
        "source_partial_model_sha256": _sha256(SOURCE_MODEL),
        "source_failure_audit": str(SOURCE_FAILURE.resolve()),
        "source_failure_audit_sha256": _sha256(SOURCE_FAILURE),
        "decision_rule": (
            "This is a local solver diagnostic only. Passing authorizes one new "
            "uniform full-ladder contract using the same damping on all meshes; "
            "the recovered state itself is never a paper result."
        ),
        "physics_changed": False,
        "mesh_changed": False,
        "boundary_conditions_changed": False,
        "paper_prediction_allowed": False,
    }


def main() -> None:
    payload = build_contract()
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(CONTRACT_PATH)


if __name__ == "__main__":
    main()
