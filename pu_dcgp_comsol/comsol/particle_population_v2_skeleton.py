
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from pu_dcgp_comsol.comsol.particle_physics_contract import (
    OUTPUT_PATH as PHYSICS_CONTRACT_PATH,
    ParticlePhysicsContract,
)


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_conservative_free_jet_fully_coupled"
    / "h11_conservative_free_jet_fully_coupled_300iter.mph"
)
MODEL_PATH = (
    HERE
    / "comsol_models"
    / "h11_particle_population_v2_skeleton"
    / "h11_particle_population_v2_skeleton.mph"
)
AUDIT_PATH = (
    HERE
    / "h11_outputs"
    / "particle_population_v2_skeleton"
    / "h11_particle_population_v2_skeleton.json"
)


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
    diameter_parameter: str,
) -> None:
    feature.label(f"7YSZ particle properties: {diameter_parameter}")
    feature.set("ParticlePropertySpec", "SpecifyDensityAndDiameter")
    feature.set("ParticleType", "SolidParticles")
    feature.set("rhop_mat", "userdef")
    feature.set("rhop", "rho_ysz")
    feature.set("dp", diameter_parameter)
    feature.set("Cp_mat", "userdef")
    feature.set("Cp", "cp_ysz_ref")
    feature.set("kp_mat", "userdef")
    feature.set("kp", "k_ysz_ref")


def build_model(
    client: Any,
    contract: ParticlePhysicsContract,
    source_model: Path = SOURCE_MODEL,
) -> tuple[Any, Any]:
    contract.validate()
    source_model = Path(source_model)
    if not source_model.exists():
        raise FileNotFoundError(source_model)

    model = client.load(str(source_model))
    model.rename("h11_particle_population_v2_skeleton")
    jm = model.java
    params = jm.param()
    comp = jm.component("comp1")

    _set_param(params, "rho_ysz", f"{contract.density_kg_m3:g}[kg/m^3]", "7YSZ density")
    _set_param(
        params,
        "cp_ysz_ref",
        f"{contract.solid_heat_capacity_j_kg_k:g}[J/(kg*K)]",
        "Source-traceable constant-property baseline",
    )
    _set_param(
        params,
        "k_ysz_ref",
        f"{contract.solid_thermal_conductivity_w_m_k:g}[W/(m*K)]",
        "Source-traceable constant-property baseline",
    )
    _set_param(params, "T_ysz_sol", f"{contract.solidus_k:g}[K]", "7YSZ solidus")
    _set_param(params, "T_ysz_liq", f"{contract.liquidus_k:g}[K]", "7YSZ liquidus")
    _set_param(
        params,
        "L_ysz",
        f"{contract.latent_heat_j_kg:g}[J/kg]",
        "7YSZ latent heat",
    )
    _set_param(
        params,
        "z_dpv",
        f"{contract.observation_plane_mm:g}[mm]",
        "Fixed gun-relative DPV observation plane",
    )
    for index, diameter_um in enumerate(contract.diameter_nodes_um(), start=1):
        _set_param(
            params,
            f"d_node_{index}",
            f"{diameter_um:.12g}[um]",
            "Fixed log-space numerical diameter support; weight not yet assigned",
        )

    try:
        comp.physics().remove("fpt")
    except Exception:
        pass
    fpt = comp.physics().create("fpt", "FluidParticleTracing", "geom1")
    fpt.label("7YSZ population v2 - build only, release absent")
    fpt.prop("ParticleReleaseSpecification").set(
        "ParticleReleaseSpecification", "SpecifyMassFlowRate"
    )
    fpt.prop("ParticleSizeDistribution").set(
        "ParticleSizeDistribution", "SpecifyParticleDiameter"
    )
    fpt.prop("ComputeParticleTemperature").set("ComputeParticleTemperature", "0")

    pp1 = fpt.feature("pp1")
    _configure_particle_properties(pp1, diameter_parameter="d_node_1")
    for index in range(2, contract.diameter_nodes + 1):
        pp = fpt.create(f"pp{index}", "ParticlePropertiesOther")
        _configure_particle_properties(pp, diameter_parameter=f"d_node_{index}")

    drag = fpt.create("df1", "DragForce", 2)
    drag.label("Wide-Re spherical baseline drag")
    drag.selection().all()
    drag.set("DragLaw", "StandardDragCorrelations")
    drag.set("u_src", "root.comp1.u")
    drag.set("rho_mat", "userdef")
    drag.set("rho", "hmnf.rho")
    drag.set("mu_mat", "userdef")
    drag.set("mu", "hmnf.mu")
    drag.set("TurbulentDispersionModel", "NoneTurbulenceModel")

    axis = fpt.create("sym1", "Symmetry", 1)
    axis.label("Particle-axis symmetry")
    axis.selection().set(_selection_entities(comp, "geom1_sel_axis"))
    open_entities = sorted(
        set(
            _selection_entities(comp, "geom1_sel_far_r")
            + _selection_entities(comp, "geom1_sel_far_z")
        )
    )
    outlet = fpt.create("out1", "Outlet", 1)
    outlet.label("Particle escape through open free-jet boundaries")
    outlet.selection().set(open_entities)

    try:
        jm.study().remove("std_particle_v2")
    except Exception:
        pass
    study = jm.study().create("std_particle_v2")
    study.label("LOCKED particle v2 transient - release and enthalpy shells absent")
    transient = study.create("time", "Transient")
    transient.set("tlist", "range(0,2e-6,2e-3)")
    try:
        transient.set("activate", ["hmnf", "off", "fpt", "on"])
    except Exception:
        pass
    return model, jm


def audit_model(
    jm: Any,
    contract: ParticlePhysicsContract,
) -> dict[str, Any]:
    comp = jm.component("comp1")
    fpt = comp.physics("fpt")
    feature_types = {
        str(tag): str(fpt.feature(tag).getType()) for tag in fpt.feature().tags()
    }
    particle_properties = [
        tag
        for tag, kind in feature_types.items()
        if kind in {"ParticleProperties", "ParticlePropertiesOther"}
    ]
    releases = [
        tag
        for tag, kind in feature_types.items()
        if kind in {"Inlet", "Release", "ReleaseFromGrid", "Nozzle"}
    ]
    drag_law = str(fpt.feature("df1").getString("DragLaw"))
    drag_density = str(fpt.feature("df1").getString("rho"))
    drag_viscosity = str(fpt.feature("df1").getString("mu"))
    drag_domains = [
        int(value) for value in fpt.feature("df1").selection().entities()
    ]
    temperature_enabled = str(
        fpt.prop("ComputeParticleTemperature").getString("ComputeParticleTemperature")
    )
    if len(particle_properties) != contract.diameter_nodes:
        raise RuntimeError(f"Expected seven particle-property nodes: {particle_properties}")
    if releases:
        raise RuntimeError(f"Release must remain absent in the skeleton: {releases}")
    if drag_law != "StandardDragCorrelations":
        raise RuntimeError(f"Unexpected drag law: {drag_law}")
    if drag_density != "hmnf.rho" or drag_viscosity != "hmnf.mu":
        raise RuntimeError(
            f"Drag must use the accepted compressible-gas fields: "
            f"rho={drag_density}, mu={drag_viscosity}"
        )
    if not drag_domains:
        raise RuntimeError("Drag force has no selected gas-flow domains")
    if temperature_enabled not in {"0", "off"}:
        raise RuntimeError("COMSOL's lumped particle temperature must remain disabled")

    return {
        "schema_version": "h11_particle_population_v2_skeleton_v1",
        "status": "pass_build_only_skeleton",
        "source_model": str(SOURCE_MODEL.resolve()),
        "source_model_sha256": _sha256(SOURCE_MODEL),
        "physics_contract": str(PHYSICS_CONTRACT_PATH.resolve()),
        "physics_contract_sha256": _sha256(PHYSICS_CONTRACT_PATH),
        "particle_physics_type": str(fpt.getType()),
        "feature_types": feature_types,
        "particle_property_nodes": particle_properties,
        "diameter_nodes_um": contract.diameter_nodes_um(),
        "drag_law": drag_law,
        "drag_gas_density": drag_density,
        "drag_gas_dynamic_viscosity": drag_viscosity,
        "drag_domain_entities": drag_domains,
        "comsol_lumped_particle_temperature_enabled": False,
        "release_features": releases,
        "fixed_observation_plane_mm": contract.observation_plane_mm,
        "workpiece_present": False,
        "next_layer": (
            "add radial enthalpy-shell auxiliary ODEs and a small effective-release "
            "screening case; do not expand the DOE"
        ),
        "calibrated": False,
        "paper_prediction_allowed": False,
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

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    contract = ParticlePhysicsContract()
    started = time.time()

    import mph

    print("H11: building particle-population v2 skeleton")
    client = mph.start(cores=args.cores, version=args.version)
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
        }
    )
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(f"Saved model: {args.model}")
    print(f"Wrote audit: {args.audit}")
    print("Build-only skeleton: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
