
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET_PATH = (
    HERE / "h11_outputs" / "dpv_target_contract" / "h11_dpv_target_contract.json"
)
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "dpv_observation_operator"
    / "h11_dpv_sampling_volume_contract.json"
)
OFFICIAL_SOURCE = "https://spraysensors.tecnar.com/research/dpv/"


@dataclass(frozen=True)
class DpvSamplingVolumeContract:
    low_speed_volume_mm3: float = 0.15
    high_speed_volume_mm3: float = 0.43
    depth_of_field_mm: float = 5.0
    low_speed_min_m_s: float = 5.0
    low_speed_max_m_s: float = 400.0
    high_speed_min_m_s: float = 400.0
    high_speed_max_m_s: float = 1200.0

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("DPV sampling-volume values must be positive and finite")
        if self.low_speed_volume_mm3 != 0.15:
            raise ValueError("The official low-speed sampling volume is frozen")
        if self.high_speed_volume_mm3 != 0.43:
            raise ValueError("The official high-speed sampling volume is frozen")
        if self.depth_of_field_mm != 5.0:
            raise ValueError("The official depth of field is frozen")

    def equivalent_radius_mm(self, volume_mm3: float) -> float:

        return math.sqrt(
            volume_mm3 / self.depth_of_field_mm / math.pi
        )

    @property
    def low_speed_equivalent_radius_mm(self) -> float:
        return self.equivalent_radius_mm(self.low_speed_volume_mm3)

    @property
    def high_speed_equivalent_radius_mm(self) -> float:
        return self.equivalent_radius_mm(self.high_speed_volume_mm3)

    def instrument_informed_radii_mm(self) -> tuple[float, float]:
        self.validate()
        return (
            self.low_speed_equivalent_radius_mm,
            self.high_speed_equivalent_radius_mm,
        )


def contract_payload() -> dict[str, object]:
    contract = DpvSamplingVolumeContract()
    contract.validate()
    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    velocity = target["outcomes"]["velocity_m_s"]
    observed_max = float(velocity["pooled_max"])
    return {
        "schema_version": "h11_dpv_sampling_volume_contract_v1",
        "status": "pass_instrument_informed_sampling_volume_envelope",
        "official_specification": asdict(contract),
        "official_source": OFFICIAL_SOURCE,
        "a_group_observed_velocity_range_m_s": [
            float(velocity["pooled_min"]),
            observed_max,
        ],
        "all_observed_particles_within_official_low_speed_range": (
            observed_max <= contract.low_speed_max_m_s
        ),
        "axisymmetric_equal_area_aperture_radius_mm": {
            "low_speed_configuration": contract.low_speed_equivalent_radius_mm,
            "high_speed_configuration": contract.high_speed_equivalent_radius_mm,
        },
        "primary_and_sensitivity_rule": (
            "Use the low-speed equivalent-area radius as the prespecified primary "
            "because all retained A particles are below 400 m/s; also report the "
            "high-speed radius as a configuration sensitivity because the exact "
            "instrument configuration was not recorded."
        ),
        "geometry_interpretation": (
            "The official volume and 5 mm optical depth are converted to an "
            "equal-area circular aperture for the 2D axisymmetric simulation. "
            "This is an observation-operator approximation, not the true slit shape."
        ),
        "working_distance_warning": (
            "The manufacturer's optical working distance is not the gun-relative "
            "100 mm DPV plane and is never substituted for that coordinate."
        ),
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
