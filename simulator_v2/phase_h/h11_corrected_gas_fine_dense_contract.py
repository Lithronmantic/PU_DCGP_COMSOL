"""Freeze the same-study dense continuation for the true fine gas mesh."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from simulator_v2.phase_h.h11_conservative_free_jet_nominal import (
    FreeJetSolveContract,
)
from simulator_v2.phase_h.h11_corrected_gas_fine_continuation_contract import (
    GAS_SKELETON_MODEL,
    MODEL_DIR,
    OUTPUT_DIR,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import _sha256


CONTRACT_PATH = OUTPUT_DIR / "h11_corrected_gas_fine_dense_contract.json"


def dense_load_fractions() -> tuple[float, ...]:
    prefix = tuple(
        value for value in FreeJetSolveContract().load_fractions if value <= 0.35
    )
    repair = (0.36, 0.37, 0.38, 0.39, 0.40)
    tail = tuple(round(0.40 + 0.025 * index, 12) for index in range(1, 25))
    return (*prefix, *repair, *tail)


@dataclass(frozen=True, slots=True)
class FineDenseContract:
    effective_exit_temperature_k: float = 11_160.0
    effective_exit_speed_m_s: float = 1_090.0
    automatic_mesh_level: int = 2
    expected_mesh_elements: int = 55_291
    diagnosed_last_successful_load: float = 0.35
    diagnosed_failed_attempted_load: float = 0.40
    maximum_step_after_0_35: float = 0.025
    continuation_relative_tolerance: float = 5.0e-4
    final_relative_tolerance: float = 1.0e-6

    @property
    def load_fractions(self) -> tuple[float, ...]:
        return dense_load_fractions()

    def validate(self) -> None:
        if (
            self.effective_exit_temperature_k,
            self.effective_exit_speed_m_s,
        ) != (11_160.0, 1_090.0):
            raise ValueError("The corrected workpoint is frozen")
        if (self.automatic_mesh_level, self.expected_mesh_elements) != (2, 55_291):
            raise ValueError("The true fine mesh is frozen")
        values = self.load_fractions
        if values[0] != 0.0 or values[-1] != 1.0:
            raise ValueError("Dense continuation must span zero to one")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("Dense continuation must strictly increase")
        post = tuple(value for value in values if value >= 0.35)
        if max(right - left for left, right in zip(post, post[1:])) > 0.025 + 1e-12:
            raise ValueError("A post-failure continuation step is too large")
        if (
            self.diagnosed_last_successful_load,
            self.diagnosed_failed_attempted_load,
        ) != (0.35, 0.40):
            raise ValueError("The numerical diagnosis is frozen")
        if (
            self.continuation_relative_tolerance,
            self.final_relative_tolerance,
        ) != (5.0e-4, 1.0e-6):
            raise ValueError("The solver tolerances cannot change")


def build_contract() -> dict:
    contract = FineDenseContract()
    contract.validate()
    return {
        "schema_version": "h11_corrected_gas_fine_dense_contract_v1",
        "status": "pass_frozen_same_study_dense_continuation",
        "contract": asdict(contract),
        "load_fractions": list(contract.load_fractions),
        "source_model": str(GAS_SKELETON_MODEL.resolve()),
        "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
        "repair_scope": (
            "Numerical continuation only. The fine mesh, governing equations, "
            "material tables, domain, boundary target, and final tolerances remain "
            "unchanged. All loads are solved in one COMSOL parametric study."
        ),
        "calibrated": False,
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
