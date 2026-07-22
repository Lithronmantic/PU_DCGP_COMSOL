
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pu_dcgp_comsol.comsol.corrected_gas_fine_continuation import MODEL_PATH
from pu_dcgp_comsol.comsol.particle_radial_enthalpy_nominal import _sha256


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "corrected_gas_fine_resume"
MODEL_DIR = HERE / "comsol_models" / "h11_corrected_gas_fine_resume"
SOURCE_PARTIAL_MODEL = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_partial.mph")
CONTRACT_PATH = OUTPUT_DIR / "h11_corrected_gas_fine_resume_contract.json"


def _resume_fractions() -> tuple[float, ...]:
    values = [0.36, 0.37, 0.38, 0.39, 0.40]
    values.extend(round(0.40 + 0.025 * index, 12) for index in range(1, 25))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class FineResumeContract:
    source_last_successful_load: float = 0.35
    failed_attempted_load: float = 0.40
    continuation_relative_tolerance: float = 5.0e-4
    final_relative_tolerance: float = 1.0e-6
    maximum_resume_step: float = 0.025
    automatic_mesh_level: int = 2
    expected_mesh_elements: int = 55_291

    @property
    def resume_fractions(self) -> tuple[float, ...]:
        return _resume_fractions()

    def validate(self) -> None:
        if (
            self.source_last_successful_load,
            self.failed_attempted_load,
        ) != (0.35, 0.40):
            raise ValueError("The observed fine-mesh failure boundary is frozen")
        values = self.resume_fractions
        if values[0] != 0.36 or values[-1] != 1.0:
            raise ValueError("The resume ladder must span 0.36 to 1.0")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("Resume fractions must strictly increase")
        if max(
            right - left
            for left, right in zip(
                (self.source_last_successful_load, *values[:-1]), values
            )
        ) > self.maximum_resume_step + 1.0e-12:
            raise ValueError("A resume step exceeds the frozen maximum")
        if (
            self.continuation_relative_tolerance,
            self.final_relative_tolerance,
        ) != (5.0e-4, 1.0e-6):
            raise ValueError("Fine-mesh tolerances cannot change")
        if (
            self.automatic_mesh_level,
            self.expected_mesh_elements,
        ) != (2, 55_291):
            raise ValueError("The true fine mesh is frozen")


def build_contract() -> dict:
    contract = FineResumeContract()
    contract.validate()
    if not SOURCE_PARTIAL_MODEL.is_file():
        raise FileNotFoundError(SOURCE_PARTIAL_MODEL)
    return {
        "schema_version": "h11_corrected_gas_fine_resume_contract_v1",
        "status": "pass_frozen_fine_mesh_resume_design",
        "contract": asdict(contract),
        "resume_fractions": list(contract.resume_fractions),
        "source_partial_model": str(SOURCE_PARTIAL_MODEL.resolve()),
        "source_partial_model_sha256": _sha256(SOURCE_PARTIAL_MODEL),
        "repair_scope": (
            "Initialization path only: restart from the last stored load-0.35 "
            "solution, reduce continuation increments, then solve the unchanged "
            "full-load equations to 1e-6."
        ),
        "forbidden_changes": [
            "no change to mesh, physics, material tables, domain, or boundary target",
            "no A or B outcome is used by the resume schedule",
            "a partial or tolerance-only solution cannot be accepted",
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
