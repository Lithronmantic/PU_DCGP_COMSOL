
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "particle_time_step_convergence"
    / "h11_particle_time_step_convergence_contract.json"
)


@dataclass(frozen=True)
class ParticleTimeStepConvergenceContract:
    effective_exit_temperature_k: float = 11_160.0
    effective_exit_speed_m_s: float = 1_090.0
    particles_per_size: int = 1023
    output_step_us: float = 10.0
    maximum_step_ladder_us: tuple[float, ...] = (2.0, 1.0, 0.5)
    finest_change_limit_fraction: float = 0.005
    minimum_primary_aperture_particles: int = 70

    def validate(self) -> None:
        if (
            self.effective_exit_temperature_k != 11_160.0
            or self.effective_exit_speed_m_s != 1_090.0
        ):
            raise ValueError("The corrected effective-exit case is frozen")
        if self.particles_per_size != 1023:
            raise ValueError("Time-step convergence must use 1023 particles per size")
        if self.output_step_us != 10.0:
            raise ValueError("The result-output step is frozen")
        if self.maximum_step_ladder_us != (2.0, 1.0, 0.5):
            raise ValueError("The internal maximum-step ladder is frozen")
        if self.finest_change_limit_fraction != 0.005:
            raise ValueError("The time-step convergence gate is frozen")
        if self.minimum_primary_aperture_particles != 70:
            raise ValueError("The DPV sampling-count gate is frozen")


def contract_payload() -> dict[str, object]:
    contract = ParticleTimeStepConvergenceContract()
    contract.validate()
    return {
        "schema_version": "h11_particle_time_step_convergence_contract_v1",
        "status": "pass_frozen_particle_time_step_convergence_contract",
        "contract": asdict(contract),
        "decision_sequence": [
            "reuse the accepted corrected gas field without recalibration",
            "hold particle count, release distribution, and output step fixed",
            "solve maximum internal steps of 2, 1, and 0.5 microseconds",
            (
                "require the 1-to-0.5 microsecond primary-aperture temperature "
                "and speed q10/q50/q90 maximum relative change <=0.5%"
            ),
            "require at least 70 particles in the finest primary aperture",
        ],
        "claim_boundary": (
            "This is a numerical integration audit only. It cannot alter the "
            "effective-exit parameters or establish held-out predictive validity."
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
