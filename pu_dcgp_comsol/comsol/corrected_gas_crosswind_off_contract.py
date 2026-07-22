
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pu_dcgp_comsol.comsol.conservative_free_jet_skeleton import (
    MODEL_PATH as GAS_SKELETON_MODEL,
)
from pu_dcgp_comsol.comsol.corrected_gas_fine_dense_contract import (
    dense_load_fractions,
)
from pu_dcgp_comsol.comsol.particle_radial_enthalpy_nominal import _sha256


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "corrected_gas_crosswind_off"
MODEL_DIR = HERE / "comsol_models" / "h11_corrected_gas_crosswind_off"
CONTRACT_PATH = OUTPUT_DIR / "h11_corrected_gas_crosswind_off_contract.json"


@dataclass(frozen=True, slots=True)
class CrosswindOffContract:
    effective_exit_temperature_k: float = 11_160.0
    effective_exit_speed_m_s: float = 1_090.0
    automatic_mesh_levels: tuple[int, ...] = (4, 3, 2)
    expected_elements: tuple[int, ...] = (8_714, 15_772, 55_291)
    streamline_diffusion: int = 1
    rans_streamline_diffusion: int = 1
    heat_streamline_diffusion: int = 1
    crosswind_diffusion: int = 0
    rans_crosswind_diffusion: int = 0
    heat_crosswind_diffusion: int = 0
    mass_imbalance_limit_fraction: float = 0.005
    energy_imbalance_limit_fraction: float = 0.02
    radial_domain_mm: float = 40.0
    axial_domain_mm: float = 140.0
    observation_plane_mm: float = 100.0
    common_dpv_profile_radii_mm: tuple[float, ...] = (
        0.001,
        1.0,
        2.0,
        4.0,
        6.0,
        10.0,
        20.0,
        30.0,
    )
    gas_temperature_anomaly_l2_limit_fraction: float = 0.01
    gas_speed_l2_limit_fraction: float = 0.01
    gas_pressure_l2_over_ambient_limit_fraction: float = 1e-4
    particles_per_size: int = 1023
    particle_output_step_us: float = 10.0
    particle_maximum_step_us: float = 2.0
    minimum_primary_aperture_particles: int = 70
    particle_quantile_limit_fraction: float = 0.01

    @property
    def load_fractions(self) -> tuple[float, ...]:
        return dense_load_fractions()

    def expected_element_count(self, mesh_level: int) -> int:
        return dict(zip(self.automatic_mesh_levels, self.expected_elements))[mesh_level]

    def validate(self) -> None:
        if (
            self.effective_exit_temperature_k,
            self.effective_exit_speed_m_s,
        ) != (11_160.0, 1_090.0):
            raise ValueError("The corrected workpoint is frozen")
        if self.automatic_mesh_levels != (4, 3, 2):
            raise ValueError("The three-mesh ladder is frozen")
        if self.expected_elements != (8_714, 15_772, 55_291):
            raise ValueError("The generated mesh identities are frozen")
        if (
            self.streamline_diffusion,
            self.rans_streamline_diffusion,
            self.heat_streamline_diffusion,
            self.crosswind_diffusion,
            self.rans_crosswind_diffusion,
            self.heat_crosswind_diffusion,
        ) != (1, 1, 1, 0, 0, 0):
            raise ValueError("The uniform stabilization choice cannot change")
        if (
            self.mass_imbalance_limit_fraction,
            self.energy_imbalance_limit_fraction,
        ) != (0.005, 0.02):
            raise ValueError("The conservation gates are frozen")
        if (
            self.radial_domain_mm,
            self.axial_domain_mm,
            self.observation_plane_mm,
        ) != (40.0, 140.0, 100.0):
            raise ValueError("The gas domain and DPV plane are frozen")
        if max(self.common_dpv_profile_radii_mm) >= self.radial_domain_mm:
            raise ValueError("Every common-profile radius must lie inside the domain")
        if (
            self.gas_temperature_anomaly_l2_limit_fraction,
            self.gas_speed_l2_limit_fraction,
            self.gas_pressure_l2_over_ambient_limit_fraction,
            self.particle_quantile_limit_fraction,
        ) != (0.01, 0.01, 1e-4, 0.01):
            raise ValueError("The gas and particle mesh-independence gates are frozen")
        if (
            self.particles_per_size,
            self.particle_output_step_us,
            self.particle_maximum_step_us,
            self.minimum_primary_aperture_particles,
        ) != (1023, 10.0, 2.0, 70):
            raise ValueError("The particle observation operator is frozen")
        values = self.load_fractions
        if values[0] != 0.0 or values[-1] != 1.0:
            raise ValueError("The dense load ladder must span zero to one")


def build_contract() -> dict:
    contract = CrosswindOffContract()
    contract.validate()
    return {
        "schema_version": "h11_corrected_gas_crosswind_off_contract_v1",
        "status": "pass_frozen_uniform_crosswind_off_mesh_ladder",
        "contract": asdict(contract),
        "load_fractions": list(contract.load_fractions),
        "source_model": str(GAS_SKELETON_MODEL.resolve()),
        "source_model_sha256": _sha256(GAS_SKELETON_MODEL),
        "numerical_basis": {
            "source": "local COMSOL Multiphysics 6.3 Single-Phase Flow documentation",
            "statement": (
                "Streamline and crosswind diffusion are consistent stabilization "
                "methods; crosswind diffusion acts as shock capturing and can be "
                "deactivated when a more expensive solver is used."
            ),
            "local_document": (
                "E:/COMSOL63/Multiphysics/doc/help/wtpwebapps/ROOT/doc/"
                "com.comsol.help.ssf/ssf_ug_fluidflow_single.06.39.html"
            ),
        },
        "decision_rule": (
            "The same stabilization flags, gas equations, material tables, domain, "
            "boundary target, load ladder, and tolerances are used on every mesh. "
            "No single-mesh exception is allowed."
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
