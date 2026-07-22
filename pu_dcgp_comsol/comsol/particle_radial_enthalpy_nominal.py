
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pu_dcgp_comsol.comsol.particle_physics_contract import (
    ParticlePhysicsContract,
)
from pu_dcgp_comsol.comsol.dpv_sampling_volume_contract import (
    DpvSamplingVolumeContract,
)
from pu_dcgp_comsol.comsol.radial_enthalpy_model import (
    RadialEnthalpyConfig,
    comsol_shell_rhs_expressions,
    comsol_temperature_expression,
)


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = (
    HERE
    / "comsol_models"
    / "h11_particle_population_v2_skeleton"
    / "h11_particle_population_v2_skeleton.mph"
)
MODEL_PATH = (
    HERE
    / "comsol_models"
    / "h11_particle_radial_enthalpy_nominal"
    / "h11_particle_radial_enthalpy_nominal.mph"
)
BUILD_AUDIT_PATH = (
    HERE
    / "h11_outputs"
    / "particle_radial_enthalpy_nominal"
    / "h11_particle_radial_enthalpy_nominal_build.json"
)
SOLVE_AUDIT_PATH = (
    HERE
    / "h11_outputs"
    / "particle_radial_enthalpy_nominal"
    / "h11_particle_radial_enthalpy_nominal_solve.json"
)
DPV_TARGET_CONTRACT_PATH = (
    HERE / "h11_outputs" / "dpv_target_contract" / "h11_dpv_target_contract.json"
)
DETECTED_DIAMETER_WEIGHT_PATH = (
    HERE
    / "h11_outputs"
    / "dpv_observation_operator"
    / "h11_dpv_detected_diameter_weights.json"
)
STUDY_TAG = "std_particle_nominal"
STEP_TAG = "time"
SOURCE_STUDY = "std_hmnf_fully_coupled_refine"
SOURCE_STEP = "stat"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_param(params: Any, name: str, value: str, description: str) -> None:
    params.set(name, value, description)


def _configure_parameters(
    params: Any,
    particle: ParticlePhysicsContract,
    radial: RadialEnthalpyConfig,
) -> None:
    _set_param(params, "N_shell", str(radial.shells), "Radial enthalpy shell count")
    _set_param(
        params,
        "T_ysz_ref",
        f"{radial.reference_temperature_k:g}[K]",
        "Particle enthalpy reference temperature",
    )
    _set_param(
        params,
        "H_ysz_sol",
        "cp_ysz_ref*(T_ysz_sol-T_ysz_ref)",
        "Specific enthalpy at the 7YSZ solidus",
    )
    _set_param(
        params,
        "cp_ysz_app",
        "cp_ysz_ref+L_ysz/(T_ysz_liq-T_ysz_sol)",
        "Apparent heat capacity across the source-traceable melting interval",
    )
    _set_param(
        params,
        "H_ysz_liq",
        "H_ysz_sol+cp_ysz_app*(T_ysz_liq-T_ysz_sol)",
        "Specific enthalpy at the 7YSZ liquidus",
    )
    _set_param(
        params,
        "sigma_sb",
        f"{radial.stefan_boltzmann_w_m2_k4:.12g}[W/(m^2*K^4)]",
        "Stefan-Boltzmann constant",
    )
    _set_param(
        params,
        "eps_ysz",
        "0.6",
        "Unmeasured gray emissivity; nominal sensitivity coordinate",
    )
    _set_param(
        params,
        "u_particle_exit_eff",
        "25[m/s]",
        "Uncalibrated effective particle axial speed at the model entry",
    )
    _set_param(
        params,
        "powder_feed_nominal",
        "20[g/min]",
        "Nominal A-group powder feed for the screening solve",
    )
    _set_param(
        params,
        "particles_per_size",
        "15",
        "Numerical particles released for each fixed diameter node",
    )
    _set_param(
        params,
        "particle_output_step",
        "2[us]",
        "Stored output interval; internal integrator maximum step remains 2 us",
    )
    _set_param(
        params,
        "t_particle_end",
        "5[ms]",
        "Nominal horizon exceeds the 4 ms ballistic time to the 100 mm plane",
    )
    if particle.radial_enthalpy_shells != radial.shells:
        raise ValueError("Particle and radial-shell contracts disagree")


def _particle_local_expressions(
    radial: RadialEnthalpyConfig,
) -> dict[str, str]:

    expressions: dict[str, str] = {
        "dp_eff_p": "max(fpt.dp,d_node_1)",
        "vrel_p": "sqrt((u-fpt.vr)^2+(w-fpt.vz)^2)",
    }
    expressions["dr_p"] = f"(({expressions['dp_eff_p']})/(2*N_shell))"
    for index in range(1, radial.shells + 1):
        expressions[f"Tsh{index}"] = comsol_temperature_expression(
            f"hsh{index}"
        )
        expressions[f"meltsh{index}"] = (
            f"min(1,max(0,(hsh{index}-H_ysz_sol)"
            f"/(H_ysz_liq-H_ysz_sol)))"
        )
    expressions["Re_p"] = (
        f"(hmnf.rho*({expressions['vrel_p']})"
        f"*({expressions['dp_eff_p']})/hmnf.mu)"
    )
    expressions["Pr_p"] = "(hmnf.Cp*hmnf.mu/mat_ar.def.k11)"
    expressions["a_g_p"] = "sqrt(hmnf.gamma*R_Ar*T)"
    expressions["Ma_rel_p"] = (
        f"(({expressions['vrel_p']})/({expressions['a_g_p']}))"
    )
    expressions["lambda_g_p"] = (
        "(hmnf.mu/max(p,1[Pa])*sqrt(pi*R_Ar*T/2))"
    )
    expressions["Kn_p"] = (
        f"(({expressions['lambda_g_p']})/({expressions['dp_eff_p']}))"
    )
    expressions["Nu_screen_p"] = (
        f"(2+0.6*sqrt(max(({expressions['Re_p']}),0))"
        f"*max(({expressions['Pr_p']}),1e-6)^(1/3))"
    )
    expressions["hconv_p"] = (
        f"(({expressions['Nu_screen_p']})*mat_ar.def.k11"
        f"/({expressions['dp_eff_p']}))"
    )
    expressions["Bi_screen_p"] = (
        f"(({expressions['hconv_p']})*({expressions['dp_eff_p']})"
        f"/(6*k_ysz_ref))"
    )
    expressions["Tsurf_p"] = expressions[f"Tsh{radial.shells}"]
    temperature_average = []
    melt_average = []
    for index in range(1, radial.shells + 1):
        weight = f"({index}^3-{index - 1}^3)/N_shell^3"
        temperature_average.append(
            f"({weight})*({expressions[f'Tsh{index}']})"
        )
        melt_average.append(
            f"({weight})*({expressions[f'meltsh{index}']})"
        )
    expressions["Tbulk_p"] = "(" + "+".join(temperature_average) + ")"
    expressions["meltfrac_p"] = "(" + "+".join(melt_average) + ")"
    return expressions


def _particle_local_shell_rhs(
    radial: RadialEnthalpyConfig,
) -> list[str]:
    expressions = _particle_local_expressions(radial)
    result = comsol_shell_rhs_expressions(radial)
    replacements = {
        "dr_p": expressions["dr_p"],
        "hconv_p": expressions["hconv_p"],
        **{
            f"Tsh{index}": expressions[f"Tsh{index}"]
            for index in range(1, radial.shells + 1)
        },
    }
    for old, new in replacements.items():
        result = [expression.replace(old, f"({new})") for expression in result]
    return result


def _configure_shell_odes(
    fpt: Any,
    radial: RadialEnthalpyConfig,
) -> None:
    rhs = _particle_local_shell_rhs(radial)
    for index, expression in enumerate(rhs, start=1):
        tag = f"auxh{index}"
        try:
            fpt.remove(tag)
        except Exception:
            pass
        feature = fpt.create(tag, "AuxiliaryField")
        feature.label(f"Radial specific enthalpy shell {index}")
        feature.set("fieldVariableName", f"hsh{index}")
        feature.set("R", expression)
        feature.set("Integrate", "WithRespectToTime")
        feature.set("DependentVariableQuantity", "specificenergy")
        feature.set("StudyStep", f"{STUDY_TAG}/{STEP_TAG}")


def _configure_releases(
    comp: Any,
    fpt: Any,
    particle: ParticlePhysicsContract,
    radial: RadialEnthalpyConfig,
) -> None:
    fpt.feature("df1").selection().all()
    for index in range(1, particle.diameter_nodes + 1):
        tag = f"inl_size_{index}"
        try:
            fpt.remove(tag)
        except Exception:
            pass
        inlet = fpt.create(tag, "Inlet", 1)
        inlet.label(f"Effective-exit release: diameter node {index}")
        inlet.selection().named("geom1_sel_nozzle_in")
        inlet.set("N", "particles_per_size")
        inlet.set("mdot", "powder_feed_nominal/7")
        inlet.set("InitialPosition", "UniformDistribution")
        inlet.set("VelocitySpecification", "SpecifyVelocity")
        inlet.set("InitialVelocity", "Expression")
        inlet.set("u_src", "userdef")
        inlet.set("u", ["0", "0", "u_particle_exit_eff"])
        inlet.set("ReleasedParticleProperties", f"pp{index}")
        inlet.set("DpDistributionFunction", "NoneDistribution")
        inlet.set("dp0", f"d_node_{index}")
        inlet.set("StudyStep", f"{STUDY_TAG}/{STEP_TAG}")
        for shell in range(1, radial.shells + 1):
            inlet.set(f"aux0_auxh{shell}", "0[J/kg]")

    try:
        fpt.remove("pcnt_dpv")
    except Exception:
        pass
    counter = fpt.create("pcnt_dpv", "ParticleCounter", 1)
    counter.label("Particles crossing the fixed gun-relative 100 mm DPV plane")
    counter.selection().named("geom1_sel_dpv")
    counter.set("StudyStep", f"{STUDY_TAG}/{STEP_TAG}")


def _configure_study(
    jm: Any,
    fpt: Any,
    source_study: str = SOURCE_STUDY,
) -> Any:


    if source_study not in {str(tag) for tag in jm.study().tags()}:
        raise ValueError(f"Unknown gas source study: {source_study}")
    fpt.prop("Formulation").set("Formulation", "Newtonian")
    try:
        jm.study().remove(STUDY_TAG)
    except Exception:
        pass
    study = jm.study().create(STUDY_TAG)
    study.label("Nominal one-way particle and radial-enthalpy screening")
    transient = study.create(STEP_TAG, "Transient")
    transient.set("tlist", "range(0,particle_output_step,t_particle_end)")
    transient.set("activate", ["hmnf", "off", "fpt", "on"])
    transient.set("usesol", "on")
    transient.set("notsolmethod", "sol")
    transient.set("notstudy", source_study)
    transient.set("notstudystep", SOURCE_STEP)
    transient.set("notsolnum", "auto")

    for tag in fpt.feature().tags():
        feature = fpt.feature(tag)
        if "StudyStep" in {str(value) for value in feature.properties()}:
            feature.set("StudyStep", f"{STUDY_TAG}/{STEP_TAG}")
    return study


def build_model(
    client: Any,
    source_model: Path = SOURCE_MODEL,
    source_study: str = SOURCE_STUDY,
) -> tuple[Any, Any]:

    source_model = Path(source_model)
    if not source_model.exists():
        raise FileNotFoundError(source_model)
    particle = ParticlePhysicsContract()
    particle.validate()
    radial = RadialEnthalpyConfig(shells=particle.radial_enthalpy_shells)
    radial.validate()

    model = client.load(str(source_model))
    model.rename("h11_particle_radial_enthalpy_nominal")
    jm = model.java
    comp = jm.component("comp1")
    fpt = comp.physics("fpt")
    _configure_parameters(jm.param(), particle, radial)
    _configure_study(jm, fpt, source_study)
    _configure_shell_odes(fpt, radial)
    _configure_releases(comp, fpt, particle, radial)
    return model, jm


def audit_build(
    jm: Any,
    source_study: str = SOURCE_STUDY,
) -> dict[str, Any]:
    fpt = jm.component("comp1").physics("fpt")
    formulation = str(fpt.prop("Formulation").getString("Formulation"))
    feature_types = {
        str(tag): str(fpt.feature(tag).getType()) for tag in fpt.feature().tags()
    }
    aux = [tag for tag, kind in feature_types.items() if kind == "AuxiliaryField"]
    releases = [tag for tag, kind in feature_types.items() if kind == "Inlet"]
    release_diameters = {
        tag: str(fpt.feature(tag).getString("dp0")) for tag in releases
    }
    expected_release_diameters = {
        f"inl_size_{index}": f"d_node_{index}" for index in range(1, 8)
    }
    drag_domains = [
        int(value) for value in fpt.feature("df1").selection().entities()
    ]
    counter_entities = [
        int(value) for value in fpt.feature("pcnt_dpv").selection().entities()
    ]
    step = jm.study(STUDY_TAG).feature(STEP_TAG)
    particles_per_release = int(
        round(float(str(jm.param().evaluate("particles_per_size"))))
    )
    source_mapping = {
        "usesol": str(step.getString("usesol")),
        "notsolmethod": str(step.getString("notsolmethod")),
        "notstudy": str(step.getString("notstudy")),
        "notstudystep": str(step.getString("notstudystep")),
    }
    if len(aux) != 8:
        raise RuntimeError(f"Expected eight enthalpy ODEs: {aux}")
    if formulation != "Newtonian":
        raise RuntimeError(f"Unexpected particle formulation: {formulation}")
    if len(releases) != 7:
        raise RuntimeError(f"Expected seven diameter releases: {releases}")
    if release_diameters != expected_release_diameters:
        raise RuntimeError(f"Incorrect release diameters: {release_diameters}")
    if not drag_domains:
        raise RuntimeError("Drag force has no selected gas-flow domains")
    if not counter_entities:
        raise RuntimeError("The fixed 100 mm DPV counter has no boundary selection")
    if source_mapping != {
        "usesol": "on",
        "notsolmethod": "sol",
        "notstudy": source_study,
        "notstudystep": SOURCE_STEP,
    }:
        raise RuntimeError(f"Incorrect accepted-gas solution mapping: {source_mapping}")
    return {
        "schema_version": "h11_particle_radial_enthalpy_nominal_build_v1",
        "status": "pass_build_ready_for_nominal_solve",
        "feature_types": feature_types,
        "radial_enthalpy_features": aux,
        "effective_exit_release_features": releases,
        "release_diameter_parameters": release_diameters,
        "drag_domain_entities": drag_domains,
        "particles_per_release": particles_per_release,
        "total_model_particles": 7 * particles_per_release,
        "particle_output_step_s": float(
            str(jm.param().evaluate("particle_output_step"))
        ),
        "dpv_counter_boundary_entities": counter_entities,
        "source_solution_mapping": source_mapping,
        "particle_equation_formulation": formulation,
        "temperature_state": "eight_shell_radial_specific_enthalpy",
        "phase_change": "apparent_heat_capacity_between_solidus_and_liquidus",
        "heat_transfer": "Ranz_Marshall_screening_only",
        "coupling": "one_way_regime_screening",
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def _create_particle_dataset(model: Any, jm: Any) -> Any:
    solution_tags = [
        str(tag)
        for tag in jm.sol().tags()
        if str(jm.sol(tag).study()) == STUDY_TAG
    ]
    if len(solution_tags) != 1:
        raise RuntimeError(
            f"Expected one particle solution for {STUDY_TAG}: {solution_tags}"
        )
    for node in list(model / "datasets"):
        if node.type() == "Particle":
            node.remove()
    dataset = (model / "datasets").create("Particle")
    dataset.property("solution", solution_tags[0])
    dataset.property("comp", "comp1")
    dataset.property("geom", "geom1")
    dataset.property("pgeomspec", "fromphysics")
    dataset.property("physicsinterface", "fpt")
    dataset.property("posdof", ["comp1.qr", "comp1.qphi", "comp1.qz"])
    return dataset


def _particle_array(model: Any, dataset: Any, expression: str) -> np.ndarray:
    value = np.asarray(model.evaluate(expression, dataset=dataset), dtype=float)
    return value


def _configure_implicit_particle_solver(
    jm: Any,
    maximum_step_s: float = 2.0e-6,
) -> dict[str, str]:

    if not 0.25e-6 <= maximum_step_s <= 2.0e-6:
        raise ValueError("Particle maximum time step is outside the audited range")
    jm.study(STUDY_TAG).createAutoSequences("all")
    solution_tags = [
        str(tag)
        for tag in jm.sol().tags()
        if str(jm.sol(tag).study()) == STUDY_TAG
    ]
    if len(solution_tags) != 1:
        raise RuntimeError(
            f"Expected one solver sequence for {STUDY_TAG}: {solution_tags}"
        )
    solution_tag = solution_tags[0]
    solution = jm.sol(solution_tag)
    variable_tags = [
        str(tag)
        for tag in solution.feature().tags()
        if str(solution.feature(tag).getType()) == "Variables"
    ]
    if len(variable_tags) != 1:
        raise RuntimeError(f"Expected one variables feature: {variable_tags}")
    variables = solution.feature(variable_tags[0])
    enthalpy_scale = RadialEnthalpyConfig().liquidus_enthalpy_j_kg
    enthalpy_fields = [
        str(tag)
        for tag in variables.feature().tags()
        if str(tag).startswith("comp1_hsh") and str(tag).endswith("fpt")
    ]
    if len(enthalpy_fields) != 8:
        raise RuntimeError(f"Expected eight enthalpy solver fields: {enthalpy_fields}")
    for tag in enthalpy_fields:
        field = variables.feature(tag)
        field.set("scalemethod", "manual")
        field.set("scaleval", f"{enthalpy_scale:.12g}")

    time_tags = [
        str(tag)
        for tag in solution.feature().tags()
        if str(solution.feature(tag).getType()) == "Time"
    ]
    if len(time_tags) != 1:
        raise RuntimeError(f"Expected one time integrator: {time_tags}")
    time_tag = time_tags[0]
    transient = solution.feature(time_tag)
    transient.set("odesolvertype", "implicit")
    transient.set("timemethod", "genalpha")
    transient.set("tstepsgenalpha", "free")
    transient.set("initialstepgenalphaactive", "on")
    transient.set("initialstepgenalpha", "5e-8")
    transient.set("maxstepconstraintgenalpha", "const")
    transient.set("maxstepgenalpha", f"{maximum_step_s:.12g}")
    return {
        "solution_tag": solution_tag,
        "time_feature_tag": time_tag,
        "ode_solver_type": str(transient.getString("odesolvertype")),
        "time_method": str(transient.getString("timemethod")),
        "initial_step_s": str(transient.getString("initialstepgenalpha")),
        "maximum_step_s": str(transient.getString("maxstepgenalpha")),
        "enthalpy_field_scale_j_kg": f"{enthalpy_scale:.12g}",
        "enthalpy_fields_scaled": ",".join(enthalpy_fields),
    }


def _first_plane_crossings(
    time_s: np.ndarray,
    axial_position_m: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    plane_z_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:

    time = np.asarray(time_s, dtype=float).ravel()
    z = np.asarray(axial_position_m, dtype=float)
    if z.ndim != 2 or z.shape[1] != time.size:
        raise ValueError("Particle axial positions must be particle-by-time")
    for name, value in fields.items():
        if np.asarray(value).shape != z.shape:
            raise ValueError(f"Crossing field {name} has shape {np.shape(value)}")

    hit_index = np.full(z.shape[0], -1, dtype=int)
    crossing = {
        "time_s": np.full(z.shape[0], np.nan, dtype=float),
        **{
            name: np.full(z.shape[0], np.nan, dtype=float)
            for name in fields
        },
    }
    for particle_index, trajectory in enumerate(z):
        candidates = np.flatnonzero(
            np.isfinite(trajectory) & (trajectory >= plane_z_m)
        )
        if not candidates.size:
            continue
        upper = int(candidates[0])
        lower = max(upper - 1, 0)
        z0 = float(trajectory[lower])
        z1 = float(trajectory[upper])
        if upper == lower or not np.isfinite(z0) or z1 == z0:
            fraction = 0.0
        else:
            fraction = float(np.clip((plane_z_m - z0) / (z1 - z0), 0.0, 1.0))
        hit_index[particle_index] = upper
        crossing["time_s"][particle_index] = (
            time[lower] + fraction * (time[upper] - time[lower])
        )
        for name, value in fields.items():
            array = np.asarray(value, dtype=float)
            value0 = float(array[particle_index, lower])
            value1 = float(array[particle_index, upper])
            crossing[name][particle_index] = value0 + fraction * (
                value1 - value0
            )
    return hit_index, crossing


def _finite_median(value: np.ndarray) -> float | None:
    finite = np.asarray(value, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    return float(np.median(finite))


def _weighted_median(
    value: np.ndarray,
    weight: np.ndarray,
) -> float | None:
    values = np.asarray(value, dtype=float).ravel()
    weights = np.asarray(weight, dtype=float).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[min(index, values.size - 1)])


def _weighted_quantile(
    value: np.ndarray,
    weight: np.ndarray,
    quantile: float,
) -> float | None:
    values = np.asarray(value, dtype=float).ravel()
    weights = np.asarray(weight, dtype=float).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any() or not 0 <= quantile <= 1:
        return None
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(
        np.searchsorted(cumulative, quantile * cumulative[-1], side="left")
    )
    return float(values[min(index, values.size - 1)])


def _weighted_quantile_triplet(
    value: np.ndarray,
    weight: np.ndarray,
    *,
    offset: float = 0.0,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for label, quantile in (("q10", 0.10), ("q50", 0.50), ("q90", 0.90)):
        estimate = _weighted_quantile(value, weight, quantile)
        result[label] = None if estimate is None else estimate + offset
    return result


def _trajectory_audit(
    time_s: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    particle = ParticlePhysicsContract()
    expected_diameters_um = np.asarray(particle.diameter_nodes_um(), dtype=float)
    particle_diameters_um = (
        np.nanmedian(arrays["diameter_m"], axis=1) * 1.0e6
    )
    observed_diameters_um = np.unique(
        np.round(particle_diameters_um[np.isfinite(particle_diameters_um)], 9)
    )
    diameter_support_matches = (
        observed_diameters_um.size == expected_diameters_um.size
        and np.allclose(
            observed_diameters_um,
            expected_diameters_um,
            rtol=0.0,
            atol=1.0e-6,
        )
    )

    speed = np.sqrt(
        arrays["velocity_r_m_s"] ** 2 + arrays["velocity_z_m_s"] ** 2
    )
    crossing_fields = {
        "radial_position_m": arrays["qr"],
        "speed_m_s": speed,
        "diameter_um": arrays["diameter_m"] * 1.0e6,
        "release_frequency_hz": arrays["release_frequency_hz"],
        "surface_temperature_k": arrays["Tsurf_p"],
        "bulk_temperature_k": arrays["Tbulk_p"],
        "melt_fraction": arrays["meltfrac_p"],
        "reynolds_number": arrays["Re_p"],
        "prandtl_number": arrays["Pr_p"],
        "relative_mach_number": arrays["Ma_rel_p"],
        "knudsen_number": arrays["Kn_p"],
        "screening_biot_number": arrays["Bi_screen_p"],
    }
    hit_index, crossing = _first_plane_crossings(
        time_s,
        arrays["qz"],
        crossing_fields,
        plane_z_m=particle.observation_plane_mm / 1000.0,
    )
    hit_mask = hit_index >= 0
    per_diameter = []
    for diameter_um in expected_diameters_um:
        group = np.isclose(
            particle_diameters_um,
            diameter_um,
            rtol=0.0,
            atol=1.0e-6,
        )
        group_hits = group & hit_mask
        per_diameter.append(
            {
                "diameter_um": float(diameter_um),
                "released_count": int(group.sum()),
                "crossing_count": int(group_hits.sum()),
                "crossing_fraction": (
                    float(group_hits.sum() / group.sum())
                    if group.sum()
                    else 0.0
                ),
                "median_crossing_time_us": (
                    None
                    if _finite_median(crossing["time_s"][group_hits]) is None
                    else 1.0e6
                    * float(_finite_median(crossing["time_s"][group_hits]))
                ),
                "median_speed_m_s": _finite_median(
                    crossing["speed_m_s"][group_hits]
                ),
                "median_surface_temperature_c": (
                    None
                    if _finite_median(
                        crossing["surface_temperature_k"][group_hits]
                    )
                    is None
                    else float(
                        _finite_median(
                            crossing["surface_temperature_k"][group_hits]
                        )
                        - 273.15
                    )
                ),
                "median_melt_fraction": _finite_median(
                    crossing["melt_fraction"][group_hits]
                ),
                "median_reynolds_number": _finite_median(
                    crossing["reynolds_number"][group_hits]
                ),
                "median_relative_mach_number": _finite_median(
                    crossing["relative_mach_number"][group_hits]
                ),
                "median_knudsen_number": _finite_median(
                    crossing["knudsen_number"][group_hits]
                ),
                "median_screening_biot_number": _finite_median(
                    crossing["screening_biot_number"][group_hits]
                ),
            }
        )

    if not DPV_TARGET_CONTRACT_PATH.exists():
        raise FileNotFoundError(DPV_TARGET_CONTRACT_PATH)
    with DPV_TARGET_CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        target = json.load(handle)
    target_temperature_c = float(
        target["outcomes"]["temperature_c"]["pooled_quantiles"]["q50"]
    )
    target_speed_m_s = float(
        target["outcomes"]["velocity_m_s"]["pooled_quantiles"]["q50"]
    )
    target_diameter_um = float(
        target["outcomes"]["particle_diameter_um"]["pooled_quantiles"]["q50"]
    )
    if not DETECTED_DIAMETER_WEIGHT_PATH.exists():
        raise FileNotFoundError(DETECTED_DIAMETER_WEIGHT_PATH)
    with DETECTED_DIAMETER_WEIGHT_PATH.open("r", encoding="utf-8") as handle:
        detected_weight_contract = json.load(handle)
    detected_nodes_um = np.asarray(
        detected_weight_contract["diameter_nodes_um"], dtype=float
    )
    detected_node_weights = np.asarray(
        detected_weight_contract["pooled_detected_weights"], dtype=float
    )
    if not np.allclose(
        detected_nodes_um,
        expected_diameters_um,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("Detected-particle and COMSOL diameter nodes disagree")
    empirical_particle_weight = np.zeros_like(particle_diameters_um)
    for diameter_um, node_weight in zip(
        expected_diameters_um,
        detected_node_weights,
    ):
        group = np.isclose(
            particle_diameters_um,
            diameter_um,
            rtol=0.0,
            atol=1.0e-6,
        )
        empirical_particle_weight[group] = node_weight / max(group.sum(), 1)
    predicted_temperature_c = (
        None
        if _finite_median(crossing["surface_temperature_k"][hit_mask]) is None
        else float(
            _finite_median(crossing["surface_temperature_k"][hit_mask])
            - 273.15
        )
    )
    predicted_speed_m_s = _finite_median(crossing["speed_m_s"][hit_mask])
    predicted_diameter_um = _finite_median(crossing["diameter_um"][hit_mask])
    comparison = {
        "role": "unweighted_nominal_diagnostic_not_a_calibrated_prediction",
        "dpv_prt_pooled_median": {
            "temperature_c": target_temperature_c,
            "speed_m_s": target_speed_m_s,
            "diameter_um": target_diameter_um,
        },
        "comsol_equal_node_crossing_median": {
            "temperature_c": predicted_temperature_c,
            "speed_m_s": predicted_speed_m_s,
            "diameter_um": predicted_diameter_um,
        },
        "signed_relative_gap": {
            "temperature": (
                None
                if predicted_temperature_c is None
                else (predicted_temperature_c - target_temperature_c)
                / target_temperature_c
            ),
            "speed": (
                None
                if predicted_speed_m_s is None
                else (predicted_speed_m_s - target_speed_m_s)
                / target_speed_m_s
            ),
            "diameter": (
                None
                if predicted_diameter_um is None
                else (predicted_diameter_um - target_diameter_um)
                / target_diameter_um
            ),
        },
    }
    frequency = crossing["release_frequency_hz"]
    weighted_temperature_k = _weighted_median(
        crossing["surface_temperature_k"][hit_mask],
        frequency[hit_mask],
    )
    weighted_speed_m_s = _weighted_median(
        crossing["speed_m_s"][hit_mask],
        frequency[hit_mask],
    )
    weighted_diameter_um = _weighted_median(
        crossing["diameter_um"][hit_mask],
        frequency[hit_mask],
    )
    frequency_weighted_comparison = {
        "role": (
            "equal_mass_per_node_release_frequency_weighting_sensitivity_only;"
            "_the_measured_feedstock_PSD_is_unknown"
        ),
        "comsol_crossing_weighted_median": {
            "temperature_c": (
                None
                if weighted_temperature_k is None
                else weighted_temperature_k - 273.15
            ),
            "speed_m_s": weighted_speed_m_s,
            "diameter_um": weighted_diameter_um,
        },
        "signed_relative_gap": {
            "temperature": (
                None
                if weighted_temperature_k is None
                else (
                    weighted_temperature_k - 273.15 - target_temperature_c
                )
                / target_temperature_c
            ),
            "speed": (
                None
                if weighted_speed_m_s is None
                else (weighted_speed_m_s - target_speed_m_s)
                / target_speed_m_s
            ),
            "diameter": (
                None
                if weighted_diameter_um is None
                else (weighted_diameter_um - target_diameter_um)
                / target_diameter_um
            ),
        },
    }
    empirical_temperature_k = _weighted_median(
        crossing["surface_temperature_k"][hit_mask],
        empirical_particle_weight[hit_mask],
    )
    empirical_speed_m_s = _weighted_median(
        crossing["speed_m_s"][hit_mask],
        empirical_particle_weight[hit_mask],
    )
    empirical_diameter_um = _weighted_median(
        crossing["diameter_um"][hit_mask],
        empirical_particle_weight[hit_mask],
    )
    empirical_detected_comparison = {
        "role": (
            "pooled_A_group_detected_diameter_weight_sensitivity; "
            "not_heldout_validation"
        ),
        "diameter_weight_contract": str(
            DETECTED_DIAMETER_WEIGHT_PATH.resolve()
        ),
        "comsol_crossing_weighted_median": {
            "temperature_c": (
                None
                if empirical_temperature_k is None
                else empirical_temperature_k - 273.15
            ),
            "speed_m_s": empirical_speed_m_s,
            "diameter_um": empirical_diameter_um,
        },
        "signed_relative_gap": {
            "temperature": (
                None
                if empirical_temperature_k is None
                else (
                    empirical_temperature_k - 273.15 - target_temperature_c
                )
                / target_temperature_c
            ),
            "speed": (
                None
                if empirical_speed_m_s is None
                else (empirical_speed_m_s - target_speed_m_s)
                / target_speed_m_s
            ),
            "diameter": (
                None
                if empirical_diameter_um is None
                else (empirical_diameter_um - target_diameter_um)
                / target_diameter_um
            ),
        },
    }
    aperture_sensitivity = []
    instrument_radii = (
        DpvSamplingVolumeContract().instrument_informed_radii_mm()
    )
    for aperture_radius_mm in [*instrument_radii, 0.5, 1.0, 2.0, 4.0]:
        selected = hit_mask & (
            crossing["radial_position_m"] <= aperture_radius_mm / 1000.0
        )
        selected_temperature_k = _finite_median(
            crossing["surface_temperature_k"][selected]
        )
        selected_speed_m_s = _finite_median(crossing["speed_m_s"][selected])
        selected_diameter_um = _finite_median(
            crossing["diameter_um"][selected]
        )
        selected_weighted_temperature_k = _weighted_median(
            crossing["surface_temperature_k"][selected],
            frequency[selected],
        )
        selected_weighted_speed_m_s = _weighted_median(
            crossing["speed_m_s"][selected],
            frequency[selected],
        )
        selected_weighted_diameter_um = _weighted_median(
            crossing["diameter_um"][selected],
            frequency[selected],
        )
        selected_empirical_temperature_k = _weighted_median(
            crossing["surface_temperature_k"][selected],
            empirical_particle_weight[selected],
        )
        selected_empirical_speed_m_s = _weighted_median(
            crossing["speed_m_s"][selected],
            empirical_particle_weight[selected],
        )
        selected_empirical_diameter_um = _weighted_median(
            crossing["diameter_um"][selected],
            empirical_particle_weight[selected],
        )
        empirical_quantiles = {
            "temperature_c": _weighted_quantile_triplet(
                crossing["surface_temperature_k"][selected],
                empirical_particle_weight[selected],
                offset=-273.15,
            ),
            "speed_m_s": _weighted_quantile_triplet(
                crossing["speed_m_s"][selected],
                empirical_particle_weight[selected],
            ),
            "diameter_um": _weighted_quantile_triplet(
                crossing["diameter_um"][selected],
                empirical_particle_weight[selected],
            ),
        }
        aperture_sensitivity.append(
            {
                "aperture_radius_mm": aperture_radius_mm,
                "selected_particle_count": int(selected.sum()),
                "selected_fraction_of_crossings": float(
                    selected.sum() / max(hit_mask.sum(), 1)
                ),
                "diameter_nodes_represented": int(
                    np.unique(
                        np.round(crossing["diameter_um"][selected], 9)
                    ).size
                ),
                "equal_model_particle_median": {
                    "temperature_c": (
                        None
                        if selected_temperature_k is None
                        else selected_temperature_k - 273.15
                    ),
                    "speed_m_s": selected_speed_m_s,
                    "diameter_um": selected_diameter_um,
                },
                "equal_mass_node_frequency_weighted_median": {
                    "temperature_c": (
                        None
                        if selected_weighted_temperature_k is None
                        else selected_weighted_temperature_k - 273.15
                    ),
                    "speed_m_s": selected_weighted_speed_m_s,
                    "diameter_um": selected_weighted_diameter_um,
                },
                "empirical_detected_diameter_weighted_median": {
                    "temperature_c": (
                        None
                        if selected_empirical_temperature_k is None
                        else selected_empirical_temperature_k - 273.15
                    ),
                    "speed_m_s": selected_empirical_speed_m_s,
                    "diameter_um": selected_empirical_diameter_um,
                },
                "empirical_detected_diameter_weighted_quantiles": (
                    empirical_quantiles
                ),
            }
        )
    diameter_crossing_complete = all(
        item["crossing_count"] > 0 for item in per_diameter
    )
    return {
        "observation_plane_mm": particle.observation_plane_mm,
        "trajectory_shape": list(arrays["qz"].shape),
        "released_particle_count": int(arrays["qz"].shape[0]),
        "crossing_count": int(hit_mask.sum()),
        "crossing_fraction": float(hit_mask.mean()),
        "expected_diameter_support_um": expected_diameters_um.tolist(),
        "observed_diameter_support_um": observed_diameters_um.tolist(),
        "diameter_support_matches_contract": bool(diameter_support_matches),
        "all_diameter_nodes_reach_observation_plane": bool(
            diameter_crossing_complete
        ),
        "per_diameter": per_diameter,
        "unweighted_target_comparison": comparison,
        "release_frequency_weighting_sensitivity": (
            frequency_weighted_comparison
        ),
        "empirical_detected_diameter_weighting_sensitivity": (
            empirical_detected_comparison
        ),
        "centerline_aperture_sensitivity": aperture_sensitivity,
    }


def solve_and_audit(
    model: Any,
    jm: Any,
    *,
    maximum_step_s: float = 2.0e-6,
) -> dict[str, Any]:
    solver = _configure_implicit_particle_solver(
        jm,
        maximum_step_s=maximum_step_s,
    )
    started = time.time()
    jm.sol(solver["solution_tag"]).runAll()
    runtime = time.time() - started
    dataset = _create_particle_dataset(model, jm)
    local = _particle_local_expressions(RadialEnthalpyConfig())
    expressions = {
        "qr": "qr",
        "qz": "qz",
        "velocity_r_m_s": "fpt.vr",
        "velocity_z_m_s": "fpt.vz",
        "diameter_m": "fpt.dp",
        "release_frequency_hz": "fpt.frel",
        **{
            name: local[name]
            for name in [
                "Tsurf_p",
                "Tbulk_p",
                "meltfrac_p",
                "Re_p",
                "Pr_p",
                "Ma_rel_p",
                "Kn_p",
                "Bi_screen_p",
            ]
        },
    }
    arrays: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for name, expression in expressions.items():
        try:
            arrays[name] = _particle_array(model, dataset, expression)
        except Exception as exc:
            errors[name] = str(exc)
    trajectory: dict[str, Any] = {}
    time_points = 0
    if not errors:
        try:
            time_s = np.asarray(
                model.evaluate("t", dataset=dataset), dtype=float
            ).ravel()
            time_points = int(time_s.size)
            trajectory = _trajectory_audit(time_s, arrays)
        except Exception as exc:
            errors["trajectory_audit"] = str(exc)
    if errors:
        status = "fail_particle_result_extraction"
    elif not trajectory["diameter_support_matches_contract"]:
        status = "fail_particle_diameter_support"
    elif not trajectory["all_diameter_nodes_reach_observation_plane"]:
        status = "fail_incomplete_diameter_crossing"
    else:
        status = "pass_nominal_comsol_trajectory_audit"
    return {
        "schema_version": "h11_particle_radial_enthalpy_nominal_solve_v2",
        "status": status,
        "runtime_sec": runtime,
        "solver": solver,
        "dataset": dataset.name(),
        "time_points": time_points,
        "array_shapes": {
            expression: list(value.shape) for expression, value in arrays.items()
        },
        "finite_ranges": {
            expression: {
                "minimum": float(np.nanmin(value)),
                "maximum": float(np.nanmax(value)),
            }
            for expression, value in arrays.items()
            if np.isfinite(value).any()
        },
        "extraction_errors": errors,
        "trajectory_audit": trajectory,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--source-model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--build-audit", type=Path, default=BUILD_AUDIT_PATH)
    parser.add_argument("--solve-audit", type=Path, default=SOLVE_AUDIT_PATH)
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.build_audit.parent.mkdir(parents=True, exist_ok=True)

    import mph

    total_started = time.time()
    print("H11: nominal radial-enthalpy particle model")
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model, jm = build_model(client, args.source_model)
        build_audit = audit_build(jm)
        model.save(str(args.model))
        build_audit.update(
            {
                "source_model": str(args.source_model.resolve()),
                "source_model_sha256": _sha256(args.source_model),
                "model_path": str(args.model.resolve()),
                "model_sha256": _sha256(args.model),
                "build_runtime_sec": time.time() - total_started,
            }
        )
        with args.build_audit.open("w", encoding="utf-8") as handle:
            json.dump(build_audit, handle, indent=2, ensure_ascii=False)
        print(f"Build audit: {args.build_audit}")
        if args.solve:
            solve_audit = solve_and_audit(model, jm)
            model.save(str(args.model))
            solve_audit.update(
                {
                    "model_path": str(args.model.resolve()),
                    "model_sha256": _sha256(args.model),
                }
            )
            with args.solve_audit.open("w", encoding="utf-8") as handle:
                json.dump(solve_audit, handle, indent=2, ensure_ascii=False)
            print(f"Solve audit: {args.solve_audit}")
    finally:
        client.clear()
    print(f"Saved model: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
