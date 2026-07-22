"""Freeze the minimal H11 particle-parameter sensitivity matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "particle_parameter_sensitivity"
    / "h11_particle_sensitivity_contract.json"
)


@dataclass(frozen=True, slots=True)
class ParticleSensitivityCase:
    name: str
    effective_exit_speed_m_s: float
    emissivity: float
    role: str
    reuse_nominal_artifact: bool = False

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Sensitivity case name is required")
        if not 0 < self.effective_exit_speed_m_s <= 200:
            raise ValueError("Effective exit speed is outside the screening range")
        if not 0 <= self.emissivity <= 1:
            raise ValueError("Emissivity must be in [0, 1]")


def sensitivity_cases() -> tuple[ParticleSensitivityCase, ...]:
    cases = (
        ParticleSensitivityCase(
            "nominal_u25_eps06",
            25.0,
            0.6,
            "frozen_nominal_reference",
            True,
        ),
        ParticleSensitivityCase(
            "u_exit_10",
            10.0,
            0.6,
            "lower_speed_longer_hot_core_residence",
        ),
        ParticleSensitivityCase(
            "u_exit_75",
            75.0,
            0.6,
            "moderate_speed_screen",
        ),
        ParticleSensitivityCase(
            "u_exit_125",
            125.0,
            0.6,
            "high_speed_shorter_hot_core_residence",
        ),
        ParticleSensitivityCase(
            "emissivity_03",
            25.0,
            0.3,
            "lower_radiative_cooling_bound",
        ),
        ParticleSensitivityCase(
            "emissivity_09",
            25.0,
            0.9,
            "upper_radiative_cooling_bound",
        ),
    )
    for case in cases:
        case.validate()
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("Sensitivity case names must be unique")
    return cases


def build_sensitivity_contract() -> dict[str, Any]:
    cases = sensitivity_cases()
    return {
        "schema_version": "h11_particle_sensitivity_contract_v1",
        "status": "pass_frozen_minimal_sensitivity_matrix",
        "purpose": (
            "separate effective-exit momentum/residence effects from gray "
            "radiation uncertainty before any gas-field calibration"
        ),
        "cases": [asdict(case) for case in cases],
        "new_comsol_solves": sum(
            not case.reuse_nominal_artifact for case in cases
        ),
        "frozen_during_this_layer": [
            "accepted strict gas solution",
            "seven 16-90 um diameter nodes",
            "eight-shell radial enthalpy model",
            "Ranz-Marshall screening heat-transfer closure",
            "100 mm gun-relative observation plane",
            "5 ms time horizon",
            "detected-particle diameter weights",
        ],
        "forbidden_in_this_layer": [
            "joint parameter optimization",
            "changing the gas field",
            "selecting a favorable aperture as final",
            "dropping particles or DOE runs",
            "paper prediction claims",
        ],
        "acceptance_gates": [
            "all 105 particles and all seven diameter nodes cross 100 mm",
            "all particle fields are finite",
            "diameter support remains 16-90 um",
            "report all four aperture sensitivities for every case",
        ],
        "paper_use_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    contract = build_sensitivity_contract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, ensure_ascii=False)
    print(f"Wrote: {args.output}")
    print(f"New COMSOL solves: {contract['new_comsol_solves']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
