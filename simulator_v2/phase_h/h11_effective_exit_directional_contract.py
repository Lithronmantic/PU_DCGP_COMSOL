"""Freeze the first directional sensitivity layer for the effective exit.

This layer is diagnostic, not calibration.  It perturbs only the gas effective
exit temperature or speed around the accepted 10,000 K / 600 m/s field.  The
exit radius, radial profile, gas material, particle entry state, particle
properties, and DPV weighting remain fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "effective_exit_directional"
    / "h11_effective_exit_directional_contract.json"
)


@dataclass(frozen=True)
class EffectiveExitDirectionalCase:
    name: str
    temperature_k: float
    speed_m_s: float
    role: str
    reuse_nominal_artifact: bool = False

    def validate(self) -> None:
        if not self.name:
            raise ValueError("A case name is required")
        if not math.isfinite(self.temperature_k) or not 8_000 <= self.temperature_k <= 12_000:
            raise ValueError("Directional temperature must lie in [8000, 12000] K")
        if not math.isfinite(self.speed_m_s) or not 400 <= self.speed_m_s <= 800:
            raise ValueError("Directional speed must lie in [400, 800] m/s")


def directional_cases() -> tuple[EffectiveExitDirectionalCase, ...]:
    cases = (
        EffectiveExitDirectionalCase(
            "nominal_t10000_u600",
            10_000.0,
            600.0,
            "accepted nominal reference",
            True,
        ),
        EffectiveExitDirectionalCase(
            "t8000_u600",
            8_000.0,
            600.0,
            "temperature-minus directional derivative",
        ),
        EffectiveExitDirectionalCase(
            "t12000_u600",
            12_000.0,
            600.0,
            "temperature-plus directional derivative",
        ),
        EffectiveExitDirectionalCase(
            "t10000_u400",
            10_000.0,
            400.0,
            "speed-minus directional derivative",
        ),
        EffectiveExitDirectionalCase(
            "t10000_u800",
            10_000.0,
            800.0,
            "speed-plus directional derivative",
        ),
    )
    for case in cases:
        case.validate()
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("Directional case names must be unique")
    return cases


def contract_payload() -> dict[str, object]:
    cases = directional_cases()
    return {
        "schema_version": "h11_effective_exit_directional_contract_v1",
        "status": "pass_frozen_effective_exit_directional_matrix",
        "cases": [asdict(case) for case in cases],
        "frozen_parameters": {
            "effective_exit_radius_mm": 4.0,
            "radial_profile_power": 2.0,
            "gas_material": "COMSOL temperature-dependent pure Ar baseline",
            "particle_entry_speed_m_s": 25.0,
            "particle_emissivity": 0.6,
            "particle_diameter_nodes_um": "7 log-spaced nodes from 16 to 90",
            "observation_plane_mm": 100.0,
        },
        "interpretation": (
            "Estimate response signs, curvature, and temperature-speed tradeoff. "
            "Do not select a parameter value or DPV aperture from this layer."
        ),
        "forbidden_actions": [
            "jointly tune exit radius or profile",
            "change gas material between cases",
            "change particle entry speed or emissivity",
            "select an aperture by minimum empirical error",
            "claim held-out prediction",
        ],
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = contract_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
