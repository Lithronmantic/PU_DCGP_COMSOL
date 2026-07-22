
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PLUME_MODEL = (
    HERE
    / "comsol_models"
    / "h11_external_plume_skeleton"
    / "h11_external_plume_skeleton_latest.mph"
)
MODEL_DIR = HERE / "comsol_models" / "h11_particle_population_skeleton"
OUT_DIR = HERE / "h11_outputs" / "particle_population_skeleton"
MODEL_PATH = MODEL_DIR / "h11_particle_population_skeleton_latest.mph"
AUDIT_PATH = OUT_DIR / "h11_particle_population_skeleton_audit.json"
DPV_CONTRACT_PATH = (
    HERE / "h11_outputs" / "dpv_target_contract" / "h11_dpv_target_contract.json"
)


@dataclass(frozen=True)
class ParticleEntryContract:

    material_name: str = "7YSZ"
    feedstock_min_um: float = 16.0
    feedstock_max_um: float = 90.0
    feedstock_bounds_confirmed: bool = True
    psd_shape_traceable: bool = False
    dense_ysz_density_kg_m3: float = 5680.0
    provisional_cp_j_kg_k: float = 580.0
    provisional_k_w_m_k: float = 2.5
    provisional_emissivity: float = 0.4
    injection_geometry_traceable: bool = False
    injection_velocity_traceable: bool = False
    ysz_thermal_model_traceable: bool = False
    plume_solution_converged: bool = False
    dpv_operator_traceable: bool = False

    def validate(self) -> None:
        if self.material_name != "7YSZ":
            raise ValueError("The operator-confirmed feedstock is 7YSZ")
        if self.feedstock_min_um <= 0:
            raise ValueError("feedstock_min_um must be positive")
        if self.feedstock_max_um <= self.feedstock_min_um:
            raise ValueError("feedstock_max_um must exceed feedstock_min_um")
        if not self.feedstock_bounds_confirmed:
            raise ValueError("The 16--90 um feedstock bounds are operator-confirmed")
        if self.dense_ysz_density_kg_m3 <= 0:
            raise ValueError("particle density must be positive")
        if self.provisional_cp_j_kg_k <= 0 or self.provisional_k_w_m_k <= 0:
            raise ValueError("particle thermal properties must be positive")
        if not 0 < self.provisional_emissivity <= 1:
            raise ValueError("particle emissivity must be in (0, 1]")

    @property
    def solve_unlocked(self) -> bool:
        return all(
            (
                self.feedstock_bounds_confirmed,
                self.psd_shape_traceable,
                self.injection_geometry_traceable,
                self.injection_velocity_traceable,
                self.ysz_thermal_model_traceable,
                self.plume_solution_converged,
                self.dpv_operator_traceable,
            )
        )

    def unresolved_gates(self) -> list[str]:
        gates = {
            "feedstock_psd_shape_or_prespecified_sensitivity_ensemble": self.psd_shape_traceable,
            "injector_position_angle": self.injection_geometry_traceable,
            "particle_injection_velocity": self.injection_velocity_traceable,
            "ysz_enthalpy_and_phase_change": self.ysz_thermal_model_traceable,
            "converged_external_plume": self.plume_solution_converged,
            "dpv_sampling_and_detection_operator": self.dpv_operator_traceable,
        }
        return [name for name, passed in gates.items() if not passed]

    def size_sensitivity_anchors_um(self) -> dict[str, float]:

        return {
            "lower": self.feedstock_min_um,
            "geometric_center": math.sqrt(self.feedstock_min_um * self.feedstock_max_um),
            "upper": self.feedstock_max_um,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_param(params: Any, name: str, value: str, description: str) -> None:
    params.set(name, value, description)


def _selection_entities(comp: Any, tag: str) -> list[int]:
    return [int(value) for value in comp.selection(tag).entities()]


def _configure_particle_properties(
    feature: Any,
    *,
    diameter_expression: str,
    contract: ParticleEntryContract,
    label: str,
) -> None:
    feature.label(label)
    feature.set("ParticlePropertySpec", "SpecifyDensityAndDiameter")
    feature.set("ParticleType", "SolidParticles")
    feature.set("rhop_mat", "userdef")
    feature.set("rhop", "rho_ysz")
    feature.set("dp", diameter_expression)
    feature.set("Cp_mat", "userdef")
    feature.set("Cp", "cp_ysz_prior")
    feature.set("kp_mat", "userdef")
    feature.set("kp", "k_ysz_prior")


def build_model(client: Any, contract: ParticleEntryContract) -> tuple[Any, Any]:
    contract.validate()
    if not PLUME_MODEL.exists():
        raise FileNotFoundError(PLUME_MODEL)

    model = client.load(str(PLUME_MODEL))
    model.rename("h11_particle_population_skeleton")
    jm = model.java
    params = jm.param()
    comp = jm.component("comp1")

    _set_param(
        params,
        "d_feed_min",
        f"{contract.feedstock_min_um:.9g}[um]",
        "Operator-confirmed lower 7YSZ feedstock diameter bound",
    )
    _set_param(
        params,
        "d_feed_max",
        f"{contract.feedstock_max_um:.9g}[um]",
        "Operator-confirmed upper 7YSZ feedstock diameter bound",
    )
    _set_param(
        params,
        "d_feed_center",
        "sqrt(d_feed_min*d_feed_max)",
        "Neutral log-scale sensitivity center; not a measured d50",
    )
    _set_param(
        params,
        "rho_ysz",
        f"{contract.dense_ysz_density_kg_m3:.9g}[kg/m^3]",
        "Provisional dense YSZ density; source verification required",
    )
    _set_param(
        params,
        "cp_ysz_prior",
        f"{contract.provisional_cp_j_kg_k:.9g}[J/(kg*K)]",
        "Provisional constant YSZ heat capacity; enthalpy model required",
    )
    _set_param(
        params,
        "k_ysz_prior",
        f"{contract.provisional_k_w_m_k:.9g}[W/(m*K)]",
        "Provisional YSZ conductivity; Biot-number audit required",
    )
    _set_param(
        params,
        "eps_ysz_prior",
        f"{contract.provisional_emissivity:.9g}",
        "Provisional YSZ emissivity; sensitivity parameter",
    )

    try:
        comp.physics().remove("fpt")
    except Exception:
        pass
    fpt = comp.physics().create("fpt", "FluidParticleTracing", "geom1")
    fpt.label("Multi-size YSZ particle population - release locked")
    fpt.prop("ParticleReleaseSpecification").set(
        "ParticleReleaseSpecification", "SpecifyMassFlowRate"
    )
    fpt.prop("ParticleSizeDistribution").set(
        "ParticleSizeDistribution", "SpecifyParticleDiameter"
    )
    fpt.prop("ComputeParticleTemperature").set("ComputeParticleTemperature", "1")

    pp50 = fpt.feature("pp1")
    _configure_particle_properties(
        pp50,
        diameter_expression="d_feed_center",
        contract=contract,
        label="YSZ particle properties - release diameter distribution pending",
    )

    drag = fpt.create("df1", "DragForce", 2)
    drag.label("Finite-Re particle drag")
    drag.set("DragLaw", "SchillerNaumann")
    drag.set("u_src", "root.comp1.u")
    drag.set("rho_mat", "from_mat")
    drag.set("mu_mat", "from_mat")
    drag.set("TurbulentDispersionModel", "NoneTurbulenceModel")

    convective = fpt.create("chl1", "ConvectiveHeatLosses", 2)
    convective.label("Convective particle heating - Nu correlation pending")
    convective.set("minput_temperature_src", "userdef")
    convective.set("minput_temperature", "T")
    convective.set("HeatSourceDefinition", "SpecifyNusseltNumber")
    convective.set(
        "Nu",
        "2",
    )
    convective.set("k_mat", "from_mat")

    radiative = fpt.create("rhl1", "RadiativeHeatLosses", 2)
    radiative.label("Particle radiative heat loss")
    radiative.set("epsilonp", "eps_ysz_prior")
    radiative.set("minput_temperature_src", "userdef")
    radiative.set("minput_temperature", "T_amb")

    momentum_feedback = fpt.create("vfc1", "VolumeForceCalculation", 2)
    momentum_feedback.label("Accumulated particle momentum feedback - coupling pending")
    heat_feedback = fpt.create("dph1", "DissipatedParticleHeat", 2)
    heat_feedback.label("Accumulated particle heat feedback - coupling pending")

    axis_entities = _selection_entities(comp, "geom1_sel_axis")
    particle_symmetry = fpt.create("sym1", "Symmetry", 1)
    particle_symmetry.label("Particle-axis symmetry")
    particle_symmetry.selection().set(axis_entities)

    open_entities = sorted(
        set(
            _selection_entities(comp, "geom1_sel_ambient_in")
            + _selection_entities(comp, "geom1_sel_far_r")
            + _selection_entities(comp, "geom1_sel_far_z")
        )
    )
    particle_outlet = fpt.create("out1", "Outlet", 1)
    particle_outlet.label("Particle escape through open plume boundaries")
    particle_outlet.selection().set(open_entities)

    try:
        jm.study().remove("std_particle")
    except Exception:
        pass
    particle_study = jm.study().create("std_particle")
    particle_study.label("LOCKED particle-population transient study")
    transient = particle_study.create("time", "Transient")
    transient.set("tlist", "range(0,2e-6,2e-3)")
    try:
        transient.set("activate", ["spf", "off", "ht", "off", "fpt", "on"])
    except Exception:
        pass

    return model, jm


def _dpv_empirical_requirement() -> dict[str, Any]:
    with DPV_CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    associations = payload["within_run_rank_associations"]
    velocity_diameter = associations["velocity_m_s__particle_diameter_um"]
    temperature_diameter = associations["temperature_c__particle_diameter_um"]
    return {
        "joint_valid_particles": payload["joint_valid_particle_count"],
        "velocity_diameter_median_spearman": velocity_diameter["quantiles"]["q50"],
        "velocity_diameter_fraction_negative": 1.0 - velocity_diameter["fraction_positive"],
        "temperature_diameter_median_spearman": temperature_diameter["quantiles"]["q50"],
        "temperature_diameter_fraction_nonpositive": 1.0
        - temperature_diameter["fraction_positive"],
        "single_fixed_diameter_model_structurally_adequate": False,
    }


def audit_model(jm: Any, contract: ParticleEntryContract) -> dict[str, Any]:
    comp = jm.component("comp1")
    fpt = comp.physics("fpt")
    feature_types = {
        str(tag): str(fpt.feature(tag).getType()) for tag in fpt.feature().tags()
    }
    required = {
        "pp1": "ParticleProperties",
        "df1": "DragForce",
        "chl1": "ConvectiveHeatLosses",
        "rhl1": "RadiativeHeatLosses",
        "vfc1": "VolumeForceCalculation",
        "dph1": "DissipatedParticleHeat",
    }
    if any(feature_types.get(tag) != value for tag, value in required.items()):
        raise RuntimeError(f"Incomplete particle physics tree: {feature_types}")
    size_mode = str(
        fpt.prop("ParticleSizeDistribution").getString("ParticleSizeDistribution")
    )
    if size_mode != "SpecifyParticleDiameter":
        raise RuntimeError(f"Particle diameter distribution is not enabled: {size_mode}")

    release_types = {"Inlet", "Release", "ReleaseFromGrid", "Nozzle"}
    active_release_features = [
        tag for tag, feature_type in feature_types.items() if feature_type in release_types
    ]
    if active_release_features:
        raise RuntimeError(f"Particle releases must remain absent: {active_release_features}")

    anchors = contract.size_sensitivity_anchors_um()
    if not anchors["lower"] < anchors["geometric_center"] < anchors["upper"]:
        raise RuntimeError(f"Invalid particle sensitivity anchors: {anchors}")

    return {
        "schema_version": "h11_particle_population_skeleton_v1",
        "status": "pass_skeleton_solve_locked",
        "contract": asdict(contract),
        "solve_unlocked": contract.solve_unlocked,
        "unresolved_gates": contract.unresolved_gates(),
        "feedstock_material": contract.material_name,
        "particle_size_sensitivity_anchors_um": anchors,
        "particle_size_anchor_status": (
            "16--90 um bounds are operator-confirmed; the geometric center is "
            "a neutral sensitivity coordinate, not a measured d50"
        ),
        "physics_type": str(fpt.getType()),
        "particle_size_distribution_mode": size_mode,
        "feature_types": feature_types,
        "active_release_features": active_release_features,
        "dpv_empirical_requirement": _dpv_empirical_requirement(),
        "pending_before_solve": [
            "replace provisional particle properties by source-traceable YSZ enthalpy/phase-change data",
            "replace Nu=2 by a verified finite-Re/Pr heat-transfer correlation",
            "define injector geometry/velocity and prespecify or measure the bounded PSD shape",
            "connect particle momentum and heat feedback to the plume equations",
            "define the DPV sampling-volume and detection-probability operator",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    contract = ParticleEntryContract()
    contract.validate()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    started = time.time()
    print("H11 layer 3: multi-size particle-population COMSOL skeleton")
    client = mph.start(cores=args.cores, version=args.version)
    print(f"COMSOL {client.version}, cores={client.cores}, standalone={client.standalone}")
    try:
        model, jm = build_model(client, contract)
        audit = audit_model(jm, contract)
        model.save(str(args.model))
    finally:
        client.clear()

    audit.update(
        {
            "comsol_version": args.version,
            "cores": args.cores,
            "runtime_sec": time.time() - started,
            "model_path": str(args.model.resolve()),
            "model_sha256": _sha256(args.model),
            "source_plume_model": str(PLUME_MODEL),
            "source_plume_sha256": _sha256(PLUME_MODEL),
        }
    )
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    print(f"Saved model: {args.model}")
    print(f"Wrote audit: {args.audit}")
    print("Particle skeleton gate: PASS; release and solve remain LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
