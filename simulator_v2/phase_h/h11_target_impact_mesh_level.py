"""H11 layer 7: independently solve one target-impact mesh level.

Each mesh starts from the same ambient seed and repeats the frozen continuation
and full-load refinement.  No field is projected from another mesh.  The
result is one input to the three-grid GCI and conservation audit; it remains
an uncalibrated numerical experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_target_impact_nominal_smoke import (
    NominalSmokeContract,
    SKELETON_MODEL,
    configure_continuation,
    configure_full_load_refinement,
    evaluate_final_solution,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "comsol_models" / "h11_target_impact_mesh_convergence"
OUT_DIR = HERE / "h11_outputs" / "target_impact_mesh_convergence"
MESH_LOAD_FRACTIONS = (
    0.0,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
)


def validate_mesh_level(level: int) -> None:
    if level not in {2, 3, 4, 5}:
        raise ValueError("COMSOL automatic mesh level must be one of 2, 3, 4, or 5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_constant_material_extrapolation(jm: Any) -> list[dict[str, str]]:
    group = jm.component("comp1").material("mat_ar").propertyGroup("def")
    records = []
    for tag in group.func().tags():
        function = group.func(tag)
        before = str(function.getString("extrap"))
        function.set("extrap", "const")
        records.append(
            {
                "tag": str(tag),
                "source_extrapolation": before,
                "applied_extrapolation": "const",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-level", type=int, required=True)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument("--model-in", type=Path, default=SKELETON_MODEL)
    args = parser.parse_args()

    validate_mesh_level(args.mesh_level)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.model_in.exists():
        raise FileNotFoundError(args.model_in)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"h11_target_impact_mesh_level_{args.mesh_level}.mph"
    audit_path = OUT_DIR / f"h11_target_impact_mesh_level_{args.mesh_level}.json"
    contract = NominalSmokeContract(
        load_fractions=MESH_LOAD_FRACTIONS,
        stationary_relative_tolerance=5e-4,
        maximum_segregated_iterations=1600,
    )
    contract.validate()

    import mph

    started = time.time()
    client = mph.start(cores=args.cores, version=args.version)
    try:
        model = client.load(str(args.model_in))
        model.rename(f"h11_target_impact_mesh_level_{args.mesh_level}")
        jm = model.java
        material_extrapolation = _set_constant_material_extrapolation(jm)
        jm.param().set(
            "Pr_t_continuation",
            "0.85",
            "Numerical continuation seed; final solve restores Kays-Crawford",
        )
        nonisothermal = jm.component("comp1").multiphysics("nitf1")
        nonisothermal.set("ThermalTurbType", "UserDefPrt")
        nonisothermal.set("Prt", "Pr_t_continuation")
        configure_continuation(jm, contract)
        mesh = jm.component("comp1").mesh("mesh1")
        mesh.autoMeshSize(args.mesh_level)
        mesh.run()
        mesh_statistics = {
            "automatic_level": args.mesh_level,
            "elements": int(mesh.getNumElem()),
            "vertices": int(mesh.getNumVertex()),
            "minimum_quality": float(mesh.getMinQuality()),
            "mean_quality": float(mesh.getMeanQuality()),
        }
        try:
            jm.study("std1").run()
        except Exception as exc:
            failure_model = MODEL_DIR / (
                f"h11_target_impact_mesh_level_{args.mesh_level}_partial.mph"
            )
            model.save(str(failure_model))
            last_converged_load = None
            n_converged = 0
            datasets = list(model / "datasets")
            if datasets:
                try:
                    indices, values = model.inner(datasets[0])
                    n_converged = int(len(indices))
                    if len(values):
                        last_converged_load = float(values[-1])
                except Exception:
                    pass
            failure = {
                "schema_version": "h11_target_impact_mesh_failure_v1",
                "status": "failed_continuation",
                "mesh": mesh_statistics,
                "n_converged_loads": n_converged,
                "last_converged_load": last_converged_load,
                "error": str(exc),
                "partial_model": str(failure_model.resolve()),
            }
            failure_path = OUT_DIR / (
                f"h11_target_impact_mesh_level_{args.mesh_level}_failure.json"
            )
            with failure_path.open("w", encoding="utf-8") as handle:
                json.dump(failure, handle, indent=2, ensure_ascii=False)
            print(f"Wrote failure audit: {failure_path}", flush=True)
            raise
        nonisothermal.set("ThermalTurbType", "KaysCrawford")
        configure_full_load_refinement(jm, contract)
        jm.study("std_refine").run()
        model.save(str(model_path))
        metrics = evaluate_final_solution(model)
    finally:
        client.clear()

    mass_pass = metrics["mass_flux_kg_s"]["imbalance_fraction"] < 0.005
    energy_pass = (
        metrics["energy_balance_w"]["imbalance_fraction_of_inlet"] < 0.02
    )
    bounded_pass = (
        metrics["temperature_k"]["minimum"]
        > contract.property_temperature_floor_k + 1e-6
    )
    audit = {
        "schema_version": "h11_target_impact_mesh_level_v1",
        "status": (
            "pass_mesh_level_numerical_gates"
            if mass_pass and energy_pass and bounded_pass
            else "pass_solve_fail_one_or_more_numerical_gates"
        ),
        "mesh": mesh_statistics,
        "metrics": metrics,
        "gates": {
            "mass_imbalance_below_0_5_percent": mass_pass,
            "energy_imbalance_below_2_percent": energy_pass,
            "temperature_floor_inactive": bounded_pass,
        },
        "material_extrapolation": material_extrapolation,
        "turbulent_prandtl_path": {
            "continuation": "UserDefPrt=0.85",
            "final_refinement": "KaysCrawford",
            "final_metrics_use": "KaysCrawford only",
        },
        "calibrated": False,
        "paper_prediction_allowed": False,
        "runtime_sec": time.time() - started,
        "comsol_version": args.version,
        "cores": args.cores,
        "source_model": str(args.model_in.resolve()),
        "source_sha256": _sha256(args.model_in),
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    print(f"Saved model: {model_path}")
    print(f"Wrote audit: {audit_path}")
    print(
        f"Mesh {args.mesh_level}: {audit['status']}; "
        f"mass={metrics['mass_flux_kg_s']['imbalance_fraction']:.3%}; "
        "energy="
        f"{metrics['energy_balance_w']['imbalance_fraction_of_inlet']:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
