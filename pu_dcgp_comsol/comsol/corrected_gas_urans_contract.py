
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = (
    HERE
    / "h11_outputs"
    / "corrected_gas_urans"
    / "h11_corrected_gas_urans_contract.json"
)


@dataclass(frozen=True)
class CorrectedGasUransContract:
    mesh_role: str = "geometric_bridge_39811_elements"
    preconditioner_relative_tolerance: float = 1e-4
    time_method: str = "bdf"
    minimum_bdf_order: int = 1
    maximum_bdf_order: int = 2
    time_relative_tolerance: float = 1e-3
    initial_time_step_s: float = 1e-8
    maximum_internal_time_step_s: float = 2e-6
    pilot_end_time_s: float = 2e-4
    pilot_output_step_s: float = 5e-6
    final_warmup_time_s: float = 1e-3
    final_end_time_s: float = 3e-3
    final_output_step_s: float = 1e-5
    maximum_segregated_iterations_per_step: int = 50
    temporal_mean_block_count: int = 4
    temporal_mean_block_limit_fraction: float = 0.01
    time_step_convergence_limit_fraction: float = 0.01
    mass_imbalance_limit_fraction: float = 0.005
    energy_imbalance_limit_fraction: float = 0.02
    workpiece_present: bool = False
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if self.mesh_role != "geometric_bridge_39811_elements":
            raise ValueError("The URANS pilot mesh is frozen")
        if self.preconditioner_relative_tolerance != 1e-4:
            raise ValueError("The URANS initial-condition tolerance is frozen")
        if (
            self.time_method,
            self.minimum_bdf_order,
            self.maximum_bdf_order,
        ) != ("bdf", 1, 2):
            raise ValueError("The BDF method and order range are frozen")
        if (
            self.time_relative_tolerance,
            self.initial_time_step_s,
            self.maximum_internal_time_step_s,
        ) != (1e-3, 1e-8, 2e-6):
            raise ValueError("The URANS integration accuracy is frozen")
        if (
            self.pilot_end_time_s,
            self.pilot_output_step_s,
            self.final_warmup_time_s,
            self.final_end_time_s,
            self.final_output_step_s,
        ) != (2e-4, 5e-6, 1e-3, 3e-3, 1e-5):
            raise ValueError("The URANS time windows are frozen")
        if self.pilot_output_step_s < self.maximum_internal_time_step_s:
            raise ValueError("Pilot output cannot be finer than internal steps")
        if not 0 < self.final_warmup_time_s < self.final_end_time_s:
            raise ValueError("The averaging window is invalid")
        if self.maximum_segregated_iterations_per_step != 50:
            raise ValueError("The per-step nonlinear ceiling is frozen")
        if (
            self.temporal_mean_block_count,
            self.temporal_mean_block_limit_fraction,
            self.time_step_convergence_limit_fraction,
        ) != (4, 0.01, 0.01):
            raise ValueError("The URANS convergence gates are frozen")
        if (
            self.mass_imbalance_limit_fraction,
            self.energy_imbalance_limit_fraction,
        ) != (0.005, 0.02):
            raise ValueError("The conservation gates are frozen")
        if (
            self.workpiece_present
            or self.calibrated
            or self.paper_prediction_allowed
        ):
            raise ValueError("URANS numerical verification cannot claim prediction")


def contract_payload() -> dict[str, object]:
    contract = CorrectedGasUransContract()
    contract.validate()
    return {
        "schema_version": "h11_corrected_gas_urans_contract_v1",
        "status": "pass_frozen_corrected_gas_urans_contract",
        "contract": asdict(contract),
        "decision_sequence": [
            "use the physical 39811-element 1e-4 state only as initialization",
            "run the fixed 0.2 ms pilot before any production time window",
            "verify bounded fields, conservation, and solvability of every step",
            "run maximum-time-step convergence before accepting time averages",
            "discard the first 1 ms and average 1-3 ms in four fixed blocks",
            "repeat on automatic mesh level 2 for time-averaged mesh convergence",
        ],
        "claim_boundary": (
            "The pilot establishes transient numerical feasibility only. "
            "Paper use requires time-step, averaging-window, mesh, domain, "
            "particle, and held-out A-group validation."
        ),
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    payload = contract_payload()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
