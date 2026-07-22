"""Freeze the independent full-load continuation on the true fine gas mesh."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from simulator_v2.phase_h.h11_conservative_free_jet_skeleton import (
    MODEL_PATH as GAS_SKELETON_MODEL,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import _sha256


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "corrected_gas_fine_continuation"
MODEL_DIR = HERE / "comsol_models" / "h11_corrected_gas_fine_continuation"
CONTRACT_PATH = OUTPUT_DIR / "h11_corrected_gas_fine_continuation_contract.json"


@dataclass(frozen=True, slots=True)
class FineContinuationContract:
    effective_exit_temperature_k: float = 11_160.0
    effective_exit_speed_m_s: float = 1_090.0
    automatic_mesh_level: int = 2
    radial_domain_mm: float = 40.0
    axial_domain_mm: float = 140.0
    minimum_fine_elements: int = 29_116
    mass_imbalance_limit_fraction: float = 0.005
    energy_imbalance_limit_fraction: float = 0.02

    def validate(self) -> None:
        if (
            self.effective_exit_temperature_k,
            self.effective_exit_speed_m_s,
        ) != (11_160.0, 1_090.0):
            raise ValueError("The corrected effective-exit workpoint is frozen")
        if self.automatic_mesh_level != 2:
            raise ValueError("The independent continuation must use true level 2")
        if (self.radial_domain_mm, self.axial_domain_mm) != (40.0, 140.0):
            raise ValueError("The gas domain is frozen")
        if self.minimum_fine_elements != 29_116:
            raise ValueError("The fine mesh must exceed the 29,115-element bridge")
        if (
            self.mass_imbalance_limit_fraction,
            self.energy_imbalance_limit_fraction,
        ) != (0.005, 0.02):
            raise ValueError("The conservation gates are frozen")


def build_contract() -> dict:
    contract = FineContinuationContract()
    contract.validate()
    return {
        "schema_version": "h11_corrected_gas_fine_continuation_contract_v1",
        "status": "pass_frozen_fine_mesh_continuation_design",
        "contract": asdict(contract),
        "source_model": str(GAS_SKELETON_MODEL.resolve()),
        "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
        "initialization": (
            "Solve the unchanged level-2 equations from the frozen ambient-to-full-"
            "load continuation, then apply the unchanged 1e-6 full-load refinement."
        ),
        "forbidden_changes": [
            "no A or B output may select the mesh or continuation path",
            "no change to gas physics, material tables, boundary profiles, or domain",
            "the 29,115-element bridge is initialization evidence only",
        ],
        "acceptance": [
            "the actual mesh contains at least 29,116 elements",
            "all thirty continuation fractions and the strict refinement solve",
            "temperature and pressure remain inside the frozen bounds",
            "mass imbalance is at most 0.5 percent",
            "energy imbalance is at most 2 percent",
        ],
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
