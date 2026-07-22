"""Extract frozen 90/110 mm particle observations from the saved COMSOL model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from simulator_v2.phase_h.h11_b_observation_plane_contract import (
    BObservationPlaneContract,
    CONTRACT_PATH,
    OUTPUT_DIR,
    SOURCE_AUDIT,
    SOURCE_MODEL,
)
from simulator_v2.phase_h.h11_particle_physics_contract import (
    ParticlePhysicsContract,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import (
    DETECTED_DIAMETER_WEIGHT_PATH,
    _create_particle_dataset,
    _first_plane_crossings,
    _particle_array,
    _particle_local_expressions,
    _sha256,
    _weighted_quantile_triplet,
)
from simulator_v2.phase_h.h11_radial_enthalpy_model import RadialEnthalpyConfig


AUDIT_PATH = OUTPUT_DIR / "h11_b_observation_plane_extraction.json"
PARTICLE_PATH = OUTPUT_DIR / "h11_b_observation_plane_particles.npz"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_arrays(model: Any, jm: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dataset = _create_particle_dataset(model, jm)
    local = _particle_local_expressions(RadialEnthalpyConfig())
    expressions = {
        "qr": "qr",
        "qz": "qz",
        "velocity_r_m_s": "fpt.vr",
        "velocity_z_m_s": "fpt.vz",
        "diameter_m": "fpt.dp",
        "release_frequency_hz": "fpt.frel",
        "Tsurf_p": local["Tsurf_p"],
        "Tbulk_p": local["Tbulk_p"],
        "meltfrac_p": local["meltfrac_p"],
        "Re_p": local["Re_p"],
        "Pr_p": local["Pr_p"],
        "Ma_rel_p": local["Ma_rel_p"],
        "Kn_p": local["Kn_p"],
        "Bi_screen_p": local["Bi_screen_p"],
    }
    arrays = {
        name: _particle_array(model, dataset, expression)
        for name, expression in expressions.items()
    }
    time_s = np.asarray(model.evaluate("t", dataset=dataset), dtype=float).ravel()
    return time_s, arrays


def _diameter_detection_weights(particle_diameter_um: np.ndarray) -> np.ndarray:
    contract = json.loads(DETECTED_DIAMETER_WEIGHT_PATH.read_text(encoding="utf-8"))
    nodes = np.asarray(contract["diameter_nodes_um"], dtype=float)
    node_weights = np.asarray(contract["pooled_detected_weights"], dtype=float)
    expected = np.asarray(ParticlePhysicsContract().diameter_nodes_um(), dtype=float)
    if not np.allclose(nodes, expected, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("Detected-diameter and COMSOL particle nodes disagree")
    weights = np.zeros_like(particle_diameter_um, dtype=float)
    for node, node_weight in zip(nodes, node_weights):
        group = np.isclose(particle_diameter_um, node, rtol=0.0, atol=1.0e-6)
        if not group.any():
            raise RuntimeError(f"Missing particle diameter node {node}")
        weights[group] = node_weight / int(group.sum())
    return weights


def _summarize_selected(
    crossing: dict[str, np.ndarray],
    selected: np.ndarray,
    weight: np.ndarray,
) -> dict[str, Any]:
    return {
        "selected_particle_count": int(selected.sum()),
        "diameter_nodes_represented": int(
            np.unique(np.round(crossing["diameter_um"][selected], 9)).size
        ),
        "quantiles": {
            "temperature_c": _weighted_quantile_triplet(
                crossing["surface_temperature_k"][selected],
                weight[selected],
                offset=-273.15,
            ),
            "velocity_m_s": _weighted_quantile_triplet(
                crossing["speed_m_s"][selected], weight[selected]
            ),
            "particle_diameter_um": _weighted_quantile_triplet(
                crossing["diameter_um"][selected], weight[selected]
            ),
        },
    }


def _extract_plane(
    time_s: np.ndarray,
    arrays: dict[str, np.ndarray],
    plane_mm: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    speed = np.sqrt(
        np.square(arrays["velocity_r_m_s"])
        + np.square(arrays["velocity_z_m_s"])
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
        plane_z_m=plane_mm / 1000.0,
    )
    hit = hit_index >= 0
    particle_diameter_um = np.nanmedian(arrays["diameter_m"], axis=1) * 1.0e6
    empirical_weight = _diameter_detection_weights(particle_diameter_um)
    frequency_weight = crossing["release_frequency_hz"]
    equal_weight = np.ones_like(frequency_weight)
    contract = BObservationPlaneContract()
    apertures = {}
    for label, radius_mm in (
        ("low_speed_primary", contract.primary_aperture_radius_mm),
        ("high_speed_sensitivity", contract.sensitivity_aperture_radius_mm),
    ):
        selected = hit & (
            crossing["radial_position_m"] <= radius_mm / 1000.0
        )
        apertures[label] = {
            "radius_mm": radius_mm,
            "equal_particle": _summarize_selected(
                crossing, selected, equal_weight
            ),
            "equal_mass_frequency": _summarize_selected(
                crossing, selected, frequency_weight
            ),
            "A_detected_diameter": _summarize_selected(
                crossing, selected, empirical_weight
            ),
        }
    expected_nodes = np.asarray(ParticlePhysicsContract().diameter_nodes_um())
    observed_nodes = np.unique(
        np.round(crossing["diameter_um"][hit], 9)
    )
    all_finite = bool(
        all(
            np.isfinite(crossing[name][hit]).all()
            for name in (
                "radial_position_m",
                "speed_m_s",
                "diameter_um",
                "surface_temperature_k",
            )
        )
    )
    audit = {
        "plane_mm": plane_mm,
        "crossing_particle_count": int(hit.sum()),
        "crossing_fraction": float(hit.mean()),
        "diameter_nodes_represented": int(observed_nodes.size),
        "diameter_support_matches_contract": bool(
            observed_nodes.size == expected_nodes.size
            and np.allclose(observed_nodes, expected_nodes, atol=1.0e-6, rtol=0.0)
        ),
        "all_primary_fields_finite": all_finite,
        "apertures": apertures,
    }
    particle_payload = {
        "particle_index": np.flatnonzero(hit).astype(np.int32),
        "radial_position_mm": crossing["radial_position_m"][hit] * 1000.0,
        "temperature_c": crossing["surface_temperature_k"][hit] - 273.15,
        "velocity_m_s": crossing["speed_m_s"][hit],
        "particle_diameter_um": crossing["diameter_um"][hit],
        "release_frequency_weight": frequency_weight[hit],
        "A_detected_diameter_weight": empirical_weight[hit],
    }
    return audit, particle_payload


def extract(client: Any) -> dict[str, Any]:
    frozen = BObservationPlaneContract()
    frozen.validate()
    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if source_audit["status"] != "pass_effective_exit_screen_particle_count":
        raise RuntimeError("The frozen source particle case did not pass")
    if int(source_audit["particles_per_size"]) != 1023:
        raise RuntimeError("The frozen source must use 1023 particles per size")
    if _sha256(SOURCE_MODEL) != json.loads(
        CONTRACT_PATH.read_text(encoding="utf-8")
    )["source_model_sha256"]:
        raise RuntimeError("The COMSOL source model changed after contract freeze")

    started = time.time()
    model = client.load(str(SOURCE_MODEL))
    try:
        time_s, arrays = _load_arrays(model, model.java)
        plane_audits = []
        particle_arrays: dict[str, np.ndarray] = {}
        for plane_mm in frozen.observation_planes_mm:
            audit, particles = _extract_plane(time_s, arrays, plane_mm)
            plane_audits.append(audit)
            prefix = f"plane_{int(plane_mm)}_"
            particle_arrays.update(
                {f"{prefix}{name}": value for name, value in particles.items()}
            )
        numerical_gates = {
            "all_source_particles_cross_both_planes": all(
                plane["crossing_fraction"] == 1.0 for plane in plane_audits
            ),
            "all_seven_diameter_nodes_at_both_planes": all(
                plane["diameter_support_matches_contract"] for plane in plane_audits
            ),
            "primary_particle_count_at_least_70_at_both_planes": all(
                plane["apertures"]["low_speed_primary"][
                    "A_detected_diameter"
                ]["selected_particle_count"]
                >= frozen.minimum_primary_particles_per_plane
                for plane in plane_audits
            ),
            "all_primary_fields_finite_at_both_planes": all(
                plane["all_primary_fields_finite"] for plane in plane_audits
            ),
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PARTICLE_PATH, **particle_arrays)
        return {
            "schema_version": "h11_b_observation_plane_extraction_v1",
            "status": (
                "pass_two_plane_comsol_extraction"
                if all(numerical_gates.values())
                else "fail_two_plane_comsol_extraction"
            ),
            "contract": str(CONTRACT_PATH.resolve()),
            "contract_sha256": _sha256(CONTRACT_PATH),
            "source_model": str(SOURCE_MODEL.resolve()),
            "source_model_sha256": _sha256(SOURCE_MODEL),
            "time_point_count": int(time_s.size),
            "trajectory_array_shape": list(arrays["qz"].shape),
            "planes": plane_audits,
            "numerical_gates": numerical_gates,
            "particle_archive": str(PARTICLE_PATH.resolve()),
            "particle_archive_sha256": _sha256(PARTICLE_PATH),
            "runtime_sec": time.time() - started,
            "calibrated_on_b": False,
            "paper_prediction_allowed": False,
        }
    finally:
        client.remove(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--output", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not CONTRACT_PATH.is_file():
        raise FileNotFoundError(CONTRACT_PATH)
    import mph

    client = mph.start(cores=args.cores, version=args.version)
    try:
        payload = extract(client)
    finally:
        client.clear()
    _write_json(args.output, payload)
    print(args.output)
    print(payload["status"])
    print(json.dumps(payload["numerical_gates"], ensure_ascii=False))
    return 0 if payload["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
