
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pu_dcgp_comsol.comsol.conservative_free_jet_skeleton import (
    MODEL_PATH as GAS_SKELETON_MODEL,
)
from pu_dcgp_comsol.comsol.corrected_gas_crosswind_off_contract import (
    CONTRACT_PATH as BASE_CONTRACT_PATH,
    CrosswindOffContract,
)
from pu_dcgp_comsol.comsol.corrected_gas_crosswind_off_damped import (
    AUDIT_PATH as RECOVERY_AUDIT_PATH,
)
from pu_dcgp_comsol.comsol.particle_radial_enthalpy_nominal import _sha256


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = (
    HERE / "h11_outputs" / "corrected_gas_crosswind_off_uniform_damped"
)
MODEL_DIR = (
    HERE / "comsol_models" / "h11_corrected_gas_crosswind_off_uniform_damped"
)
CONTRACT_PATH = (
    OUTPUT_DIR
    / "h11_corrected_gas_crosswind_off_uniform_damped_contract.json"
)


@dataclass(frozen=True, slots=True)
class UniformDampedContract:
    flow_group_damping: float = 0.50
    turbulence_group_damping: float = 0.15
    applies_to_studies: tuple[str, ...] = ("std1", "std_refine")
    applies_to_mesh_levels: tuple[int, ...] = (4, 3, 2)
    continuation_relative_tolerance: float = 5e-4
    refinement_relative_tolerance: float = 1e-6
    continuation_maximum_iterations: int = 2000
    refinement_maximum_iterations: int = 4000

    def validate(self) -> None:
        if (
            self.flow_group_damping,
            self.turbulence_group_damping,
        ) != (0.50, 0.15):
            raise ValueError("The recovered uniform damping is frozen")
        if self.applies_to_studies != ("std1", "std_refine"):
            raise ValueError("Both gas studies must use the same damping")
        if self.applies_to_mesh_levels != (4, 3, 2):
            raise ValueError("The full three-mesh ladder is required")
        if (
            self.continuation_relative_tolerance,
            self.refinement_relative_tolerance,
        ) != (5e-4, 1e-6):
            raise ValueError("The gas tolerances cannot change")
        if (
            self.continuation_maximum_iterations,
            self.refinement_maximum_iterations,
        ) != (2000, 4000):
            raise ValueError("The iteration ceilings cannot change")


def _validated_recovery() -> dict:
    if not RECOVERY_AUDIT_PATH.is_file():
        raise FileNotFoundError(RECOVERY_AUDIT_PATH)
    payload = json.loads(RECOVERY_AUDIT_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "pass_same_mesh_damping_recovery":
        raise RuntimeError("The fine-mesh damping recovery has not passed")
    if not payload.get("gates") or not all(payload["gates"].values()):
        raise RuntimeError("A fine-mesh recovery gate failed")
    if payload.get("paper_prediction_allowed") is not False:
        raise RuntimeError("The local recovery was incorrectly promoted")
    return payload


def build_contract() -> dict:
    base = CrosswindOffContract()
    base.validate()
    contract = UniformDampedContract()
    contract.validate()
    recovery = _validated_recovery()
    return {
        "schema_version": (
            "h11_corrected_gas_crosswind_off_uniform_damped_contract_v1"
        ),
        "status": "pass_frozen_uniform_full_ladder_damping_contract",
        "solver_contract": asdict(contract),
        "physical_numerical_contract": asdict(base),
        "load_fractions": list(base.load_fractions),
        "source_model": str(GAS_SKELETON_MODEL.resolve()),
        "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
        "base_contract": str(BASE_CONTRACT_PATH.resolve()),
        "base_contract_sha256": _sha256(BASE_CONTRACT_PATH),
        "recovery_audit": str(RECOVERY_AUDIT_PATH.resolve()),
        "recovery_audit_sha256": _sha256(RECOVERY_AUDIT_PATH),
        "recovery_model_sha256": recovery["model_sha256"],
        "decision_rule": (
            "Run every mesh from the common gas skeleton through the complete "
            "dense load ladder and 1e-6 refinement.  The 0.50/0.15 fixed "
            "segregated damping is identical on all meshes and both studies."
        ),
        "claim_boundary": (
            "This contract selects a solver only.  Paper use requires all "
            "three gas cases, particle mesh independence, physical sensitivity "
            "audits, and held-out experimental validation."
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
