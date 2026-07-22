"""H11 gas-property and effective-exit entry contract.

This layer records what may enter the COMSOL external-plume model before any
Ar--H2 mixture or effective-exit calibration is implemented.  It does not
solve COMSOL and it cannot grant prediction status.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_AUDIT = (
    HERE
    / "h11_outputs"
    / "gas_property_entry"
    / "h11_gas_property_entry_contract.json"
)


@dataclass(frozen=True)
class GasPropertyEntryContract:
    """Frozen scientific boundary for the next gas-physics layer."""

    argon_setting_min: float = 80.0
    argon_setting_max: float = 120.0
    argon_table_unit: str = "scfh"
    hydrogen_setting: float = 2.5
    hydrogen_operator_unit: str = "g/min"
    compatible_flow_units_confirmed: bool = False
    standard_reference_condition_confirmed: bool = False
    mixture_composition_identified: bool = False
    mixture_property_table_validated: bool = False
    particle_observation_operator_validated: bool = False
    grouped_held_out_calibration_required: bool = True
    dpv_plane_mm: float = 100.0
    calibrated: bool = False
    paper_prediction_allowed: bool = False

    def validate(self) -> None:
        if (self.argon_setting_min, self.argon_setting_max) != (80.0, 120.0):
            raise ValueError("Executed A argon settings must remain 80--120")
        if self.argon_table_unit != "scfh":
            raise ValueError("The executed table records argon in scfh")
        if self.hydrogen_setting != 2.5:
            raise ValueError("Hydrogen setting must remain 2.5")
        if self.hydrogen_operator_unit != "g/min":
            raise ValueError("Operator-reported hydrogen unit must be retained")
        if self.dpv_plane_mm != 100.0:
            raise ValueError("A-group DPV plane must remain 100 mm")
        if not self.grouped_held_out_calibration_required:
            raise ValueError("Calibration must retain grouped held-out testing")
        if self.calibrated or self.paper_prediction_allowed:
            raise ValueError("This entry contract cannot claim calibration")

    @property
    def setting_ratio_range(self) -> tuple[float, float]:
        return (
            self.hydrogen_setting / self.argon_setting_max,
            self.hydrogen_setting / self.argon_setting_min,
        )

    @property
    def mixture_release_allowed(self) -> bool:
        return all(
            (
                self.compatible_flow_units_confirmed,
                self.standard_reference_condition_confirmed,
                self.mixture_composition_identified,
                self.mixture_property_table_validated,
            )
        )

    @property
    def effective_exit_calibration_allowed(self) -> bool:
        return (
            self.particle_observation_operator_validated
            and self.grouped_held_out_calibration_required
        )


def build_payload(
    contract: GasPropertyEntryContract,
) -> dict[str, object]:
    contract.validate()
    ratio_min, ratio_max = contract.setting_ratio_range
    mixture_release = contract.mixture_release_allowed
    calibration_release = contract.effective_exit_calibration_allowed
    return {
        "schema_version": "h11_gas_property_entry_contract_v1",
        "status": "pass_release_locked_gas_property_entry_contract",
        "contract": asdict(contract),
        "identified_inputs": {
            "argon_setting": {
                "range": [
                    contract.argon_setting_min,
                    contract.argon_setting_max,
                ],
                "unit": contract.argon_table_unit,
            },
            "hydrogen_setting": {
                "value": contract.hydrogen_setting,
                "operator_reported_unit": contract.hydrogen_operator_unit,
            },
            "h2_to_ar_setting_ratio": {
                "range": [ratio_min, ratio_max],
                "admitted_use": "dimensionless setting covariate only",
                "not_admitted_as": [
                    "mass_fraction",
                    "mole_fraction",
                    "volume_fraction",
                ],
            },
        },
        "current_gas_model": {
            "mode": "pure_argon_numerical_reference_only",
            "mixture_release_allowed": mixture_release,
            "paper_prediction_allowed": False,
        },
        "mixture_release_requirements": {
            "compatible_Ar_H2_flow_units": (
                contract.compatible_flow_units_confirmed
            ),
            "declared_standard_reference_condition": (
                contract.standard_reference_condition_confirmed
            ),
            "mass_or_mole_composition_identified": (
                contract.mixture_composition_identified
            ),
            "one_atmosphere_temperature_table_validated": (
                contract.mixture_property_table_validated
            ),
            "required_functions": [
                "density_or_equilibrium_molar_mass",
                "specific_enthalpy_or_thermodynamically_consistent_Cp",
                "dynamic_viscosity",
                "thermal_conductivity",
                "volumetric_radiative_loss_or_documented_negligibility",
            ],
        },
        "effective_exit_calibration": {
            "allowed": calibration_release,
            "reason_if_locked": (
                None
                if calibration_release
                else (
                    "DPV observes particles, so gas exit temperature and "
                    "velocity cannot be calibrated before the particle and "
                    "detection operators are validated."
                )
            ),
            "joint_latent_parameters": [
                "effective_exit_temperature",
                "effective_exit_velocity",
                "effective_exit_radius",
                "radial_profile_exponent",
                "particle_injection_and_PSD_parameters",
                "DPV_detection_parameters",
            ],
            "validation": (
                "grouped held-out four-factor settings; all A runs retained"
            ),
        },
        "geometry_separation": {
            "dpv_branch": (
                "no workpiece; observation plane fixed 100 mm from gun"
            ),
            "impact_branch": (
                "workpiece present; spray distance changes target coordinate"
            ),
            "spray_distance_enters_dpv_geometry": False,
        },
        "sources": {
            "comsol_equilibrium_discharge_library": (
                "https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/"
                "comsol_ref_materials.25.77.html"
            ),
            "argon_hydrogen_transport_reference": (
                "https://doi.org/10.1023/A:1007099926249"
            ),
        },
        "next_gate": (
            "Resolve mixture composition or retain pure-Ar reference status; "
            "then build the particle population and DPV operator before joint "
            "effective-exit calibration."
        ),
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    payload = build_payload(GasPropertyEntryContract())
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote gas-property entry contract: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
