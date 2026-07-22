"""H11 input traceability for the dual-branch paper-usable COMSOL model.

The contract distinguishes confirmed apparatus inputs from values that are
numerically present but lack units, and from inputs that are absent.  Missing
quantities are never silently replaced by the constants used in H0--H8.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
RUN_MANIFEST = WORKSPACE / "experiments" / "pu_dcgp" / "data" / "run_manifest.json"
DEFAULT_OUTPUT = HERE / "h11_outputs" / "input_traceability" / "h11_input_traceability.json"


@dataclass(frozen=True)
class InputFact:
    name: str
    value: Any
    unit: str | None
    status: str
    source: str
    locator: str
    admitted_use: str


def _range(values: list[float]) -> list[float]:
    return [float(min(values)), float(max(values))]


def build_contract(manifest_path: Path = RUN_MANIFEST) -> dict[str, Any]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    a_runs = [row for row in manifest["runs"] if row["group"] == "A"]
    if len(a_runs) != 150:
        raise ValueError(f"Expected 150 A runs, found {len(a_runs)}")

    experiment = manifest["experiment"]
    facts = [
        InputFact(
            "current",
            _range([float(row["current_a"]) for row in a_runs]),
            "A",
            "confirmed_setpoint_range",
            str(manifest_path),
            "runs[group=A].current_a",
            "DOE treatment and electrical-current covariate; not arc power",
        ),
        InputFact(
            "argon_flow",
            _range([float(row["argon_flow_scfh"]) for row in a_runs]),
            "scfh",
            "confirmed_setpoint_range",
            str(manifest_path),
            "runs[group=A].argon_flow_scfh",
            "DOE treatment; mass-flow conversion requires declared standard conditions",
        ),
        InputFact(
            "spray_distance",
            _range([float(row["spray_distance_mm"]) for row in a_runs]),
            "mm",
            "confirmed_setpoint_range",
            str(manifest_path),
            "runs[group=A].spray_distance_mm",
            "independent robot/workpiece setting; never the DPV plane coordinate",
        ),
        InputFact(
            "powder_feed",
            _range([float(row["powder_feed_g_min"]) for row in a_runs]),
            "g/min",
            "confirmed_setpoint_range",
            str(manifest_path),
            "runs[group=A].powder_feed_g_min",
            "particle mass-loading treatment",
        ),
        InputFact(
            "dpv_axial_position",
            100.0,
            "mm",
            "operator_confirmed",
            "operator clarification; APS_YSZ_A方案.xlsx",
            "A方案!G4:G154 and current conversation",
            "fixed observation coordinate in the gun-attached frame",
        ),
        InputFact(
            "hydrogen_setting",
            float(experiment["hydrogen_setting"]),
            "operator reports same basis as argon; typed g/min conflicts with table scfh",
            "operator_reported_unit_conflict",
            "operator clarification and run manifest",
            "experiment.hydrogen_setting; current conversation",
            (
                "may form a dimensionless setting-ratio sensitivity with argon; "
                "must not be converted to mass/mole fraction"
            ),
        ),
        InputFact(
            "powder_carrier_gas_setting",
            float(experiment["powder_carrier_gas_setting"]),
            None,
            "confirmed_value_unit_missing",
            str(manifest_path),
            "experiment.powder_carrier_gas_setting",
            "excluded from the current model by operator instruction",
        ),
        InputFact(
            "current_response_tolerance",
            float(experiment["current_response_tolerance_a"]),
            "A",
            "operator_reported",
            str(manifest_path),
            "experiment.current_response_tolerance_a",
            "setpoint-response uncertainty",
        ),
        InputFact(
            "arc_voltage_or_power",
            None,
            "V or W",
            "missing",
            "repository search",
            "not found in executed A records",
            "required for a power/enthalpy inlet; current alone cannot determine energy input",
        ),
        InputFact(
            "nozzle_exit_geometry",
            None,
            "mm",
            "missing",
            "repository search",
            "torch/nozzle drawing not found",
            "required for exit area, velocity profile, and near-field resolution",
        ),
        InputFact(
            "gas_standard_reference",
            None,
            "temperature and pressure",
            "missing",
            "repository search",
            "SCFH reference condition not recorded",
            "required for traceable conversion of volumetric flow to mass flow",
        ),
        InputFact(
            "feedstock_material_and_size_bounds",
            {"material": "7YSZ", "diameter_min_um": 16.0, "diameter_max_um": 90.0},
            "um",
            "operator_confirmed_bounds_psd_shape_missing",
            "operator clarification",
            "current conversation",
            (
                "bounds a prespecified particle-size sensitivity ensemble; "
                "does not identify the PSD shape or d50"
            ),
        ),
        InputFact(
            "powder_injection_geometry_velocity",
            None,
            "mm and m/s",
            "missing",
            "repository search",
            "injector location, angle, and initial velocity not found",
            "required for trajectory and residence-time prediction",
        ),
        InputFact(
            "dpv_sampling_volume_and_thresholds",
            None,
            "mm3 and device settings",
            "missing",
            "DOE_YSZ_DPV_final.xlsx; metadata audit",
            "每炉记录项!A8:D20; metadata_missing_items.csv",
            "required for the detected-particle observation operator",
        ),
        InputFact(
            "dpv_measurement_workpiece_state",
            False,
            None,
            "operator_confirmed_and_photo_supported",
            "operator clarification; DPV setup photograph",
            "current conversation; simulator_v2/phase_h/evidence/dpv_no_workpiece_setup.jpg",
            "selects the no-workpiece measurement branch",
        ),
        InputFact(
            "impact_simulation_workpiece_state",
            True,
            None,
            "operator_confirmed_model_assumption",
            "operator clarification",
            "current conversation",
            "selects the workpiece-impact branch with target at z=d_spray",
        ),
        InputFact(
            "workpiece_temperature_during_process",
            [97.0, 119.0],
            "degC",
            "operator_reported_range",
            "operator clarification",
            "current conversation history",
            "isothermal target-wall sensitivity range in the impact branch",
        ),
    ]

    h2_setting = float(experiment["hydrogen_setting"])
    argon_range = _range([float(row["argon_flow_scfh"]) for row in a_runs])
    h2_to_argon_setting_ratio = [
        h2_setting / argon_range[1],
        h2_setting / argon_range[0],
    ]

    gates = {
        "coordinate_geometry": {
            "status": "pass",
            "reason": (
                "measurement and impact geometries are separated: z_DPV belongs "
                "only to the no-workpiece branch; d_spray locates the target only"
            ),
        },
        "external_jet_mass_momentum": {
            "status": "open_for_training_only_effective_boundary",
            "reason": (
                "absolute mass-flow conversion/nozzle state remain untraceable, "
                "so only a held-out-validated effective exit boundary is allowed"
            ),
        },
        "external_jet_energy": {
            "status": "open_for_training_only_effective_boundary",
            "reason": (
                "arc power is absent; exit thermal state may be calibrated inside "
                "training folds without claiming internal-arc reconstruction"
            ),
        },
        "particle_population": {
            "status": "open_for_bounded_size_sensitivity_release_locked",
            "reason": (
                "7YSZ and 16--90 um bounds are confirmed; PSD shape and injector "
                "geometry/velocity remain unresolved"
            ),
        },
        "dpv_observation_operator": {
            "status": "closed",
            "reason": "sampling volume and detection thresholds are absent",
        },
        "workpiece_geometry": {
            "status": "pass_dual_branch",
            "reason": (
                "DPV measurement has no workpiece; impact simulation has a target "
                "at d_spray with 97--119 degC temperature sensitivity"
            ),
        },
    }

    return {
        "schema_version": "h11_input_traceability_v1",
        "n_a_runs": len(a_runs),
        "facts": [asdict(fact) for fact in facts],
        "derived_quantities": {
            "h2_to_argon_setting_ratio": {
                "value_range": h2_to_argon_setting_ratio,
                "status": "admissible_as_setting_ratio_only",
                "reason": (
                    "operator reports the same setting basis, but typed g/min "
                    "conflicts with the table's scfh label; no mass/mole fraction is inferred"
                ),
            }
        },
        "model_entry_gates": gates,
        "paper_claim_boundary": (
            "The dual geometry, 7YSZ size bounds, target temperature range, and "
            "dimensionless H2/Ar setting-ratio sensitivity are admitted. Absolute "
            "plasma/particle prediction still requires training-only calibration "
            "and held-out validation; no internal-arc reconstruction is claimed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=RUN_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    contract = build_contract(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, ensure_ascii=False)
    print(f"Wrote: {args.output}")
    for name, gate in contract["model_entry_gates"].items():
        print(f"{name}: {gate['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
