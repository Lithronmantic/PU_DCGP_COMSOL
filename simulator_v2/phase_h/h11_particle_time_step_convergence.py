"""Run the frozen corrected-case particle time-step convergence ladder."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_dpv_sampling_volume_contract import (
    DpvSamplingVolumeContract,
)
from simulator_v2.phase_h.h11_effective_exit_joint_correction import SPEC
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import (
    _sha256,
    audit_build,
    build_model,
    solve_and_audit,
)
from simulator_v2.phase_h.h11_particle_time_step_convergence_contract import (
    OUTPUT_PATH as CONTRACT_PATH,
    ParticleTimeStepConvergenceContract,
)


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "particle_time_step_convergence"
MODEL_DIR = HERE / "comsol_models" / "h11_particle_time_step_convergence"
SUMMARY_PATH = OUTPUT_DIR / "h11_particle_time_step_convergence_summary.json"
BASELINE_PATH = SPEC.paths(1023)["particle_audit"]


def _step_tag(maximum_step_us: float) -> str:
    return f"{maximum_step_us:.1f}".replace(".", "p")


def case_paths(maximum_step_us: float) -> dict[str, Path]:
    tag = _step_tag(maximum_step_us)
    stem = f"corrected_t11160_u1090_n1023_dt{tag}us"
    return {
        "audit": OUTPUT_DIR / "cases" / f"{stem}.json",
        "model": MODEL_DIR / f"{stem}.mph",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _primary_aperture(solve_audit: dict[str, Any]) -> dict[str, Any]:
    radius = DpvSamplingVolumeContract().low_speed_equivalent_radius_mm
    return next(
        aperture
        for aperture in solve_audit["trajectory_audit"][
            "centerline_aperture_sensitivity"
        ]
        if math.isclose(
            aperture["aperture_radius_mm"],
            radius,
            rel_tol=0,
            abs_tol=1e-12,
        )
    )


def compact_case(
    maximum_step_us: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    solve = payload["particle_solve_audit"]
    aperture = _primary_aperture(solve)
    return {
        "maximum_step_us": maximum_step_us,
        "status": solve["status"],
        "solver_maximum_step_s": float(solve["solver"]["maximum_step_s"]),
        "released_particle_count": solve["trajectory_audit"][
            "released_particle_count"
        ],
        "crossing_count": solve["trajectory_audit"]["crossing_count"],
        "all_diameter_nodes_reach_observation_plane": solve[
            "trajectory_audit"
        ]["all_diameter_nodes_reach_observation_plane"],
        "primary_aperture": {
            "radius_mm": aperture["aperture_radius_mm"],
            "selected_particle_count": aperture["selected_particle_count"],
            "diameter_nodes_represented": aperture[
                "diameter_nodes_represented"
            ],
            "quantiles": aperture[
                "empirical_detected_diameter_weighted_quantiles"
            ],
        },
    }


def adjacent_change(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    changes: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for response in ("temperature_c", "speed_m_s"):
        changes[response] = {}
        for quantile in ("q10", "q50", "q90"):
            left_value = left["primary_aperture"]["quantiles"][response][quantile]
            right_value = right["primary_aperture"]["quantiles"][response][
                quantile
            ]
            change = abs(right_value - left_value) / max(
                abs(left_value),
                1e-12,
            )
            changes[response][quantile] = change
            maximum = max(maximum, change)
    return {
        "left_maximum_step_us": left["maximum_step_us"],
        "right_maximum_step_us": right["maximum_step_us"],
        "changes": changes,
        "maximum_relative_change": maximum,
    }


def load_case(maximum_step_us: float) -> dict[str, Any] | None:
    path = (
        BASELINE_PATH
        if maximum_step_us == 2.0
        else case_paths(maximum_step_us)["audit"]
    )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def solve_case(
    client: Any,
    maximum_step_us: float,
) -> dict[str, Any]:
    contract = ParticleTimeStepConvergenceContract()
    contract.validate()
    if maximum_step_us not in contract.maximum_step_ladder_us[1:]:
        raise ValueError("Only the 1 and 0.5 microsecond refinements are solved")
    paths = case_paths(maximum_step_us)
    paths["audit"].parent.mkdir(parents=True, exist_ok=True)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model, jm = build_model(
        client,
        SPEC.paths()["particle_skeleton"],
        source_study="std_refine",
    )
    try:
        jm.param().set("particles_per_size", str(contract.particles_per_size))
        jm.param().set(
            "particle_output_step",
            f"{contract.output_step_us:.12g}[us]",
        )
        build_audit = audit_build(jm, source_study="std_refine")
        started = time.time()
        solve_audit = solve_and_audit(
            model,
            jm,
            maximum_step_s=maximum_step_us * 1e-6,
        )
        model.save(str(paths["model"]))
        solve_audit.update(
            {
                "model_path": str(paths["model"].resolve()),
                "model_sha256": _sha256(paths["model"]),
            }
        )
        payload = {
            "schema_version": "h11_particle_time_step_case_v1",
            "status": (
                "pass_particle_time_step_case"
                if solve_audit["status"]
                == "pass_nominal_comsol_trajectory_audit"
                else "fail_particle_time_step_case"
            ),
            "maximum_step_us": maximum_step_us,
            "source_particle_skeleton": str(
                SPEC.paths()["particle_skeleton"].resolve()
            ),
            "source_particle_skeleton_sha256": _sha256(
                SPEC.paths()["particle_skeleton"]
            ),
            "particle_build_audit": build_audit,
            "particle_solve_audit": solve_audit,
            "wall_runtime_sec": time.time() - started,
            "calibrated": False,
            "paper_prediction_allowed": False,
        }
        _write_json(paths["audit"], payload)
        return payload
    finally:
        client.remove(model)


def build_summary() -> dict[str, Any] | None:
    contract = ParticleTimeStepConvergenceContract()
    contract.validate()
    payloads = [
        load_case(step) for step in contract.maximum_step_ladder_us
    ]
    if any(payload is None for payload in payloads):
        return None
    cases = [
        compact_case(step, payload)
        for step, payload in zip(
            contract.maximum_step_ladder_us,
            payloads,
        )
    ]
    comparisons = [
        adjacent_change(left, right)
        for left, right in zip(cases[:-1], cases[1:])
    ]
    finest = cases[-1]
    gates = {
        "all_cases_passed": all(
            case["status"] == "pass_nominal_comsol_trajectory_audit"
            for case in cases
        ),
        "solver_steps_match_contract": all(
            math.isclose(
                case["solver_maximum_step_s"],
                case["maximum_step_us"] * 1e-6,
                rel_tol=0,
                abs_tol=1e-15,
            )
            for case in cases
        ),
        "all_particles_cross_in_finest_case": (
            finest["released_particle_count"] == finest["crossing_count"]
            and finest["all_diameter_nodes_reach_observation_plane"]
        ),
        "finest_primary_count_at_least_70": (
            finest["primary_aperture"]["selected_particle_count"]
            >= contract.minimum_primary_aperture_particles
        ),
        "finest_adjacent_change_below_0_5_percent": (
            comparisons[-1]["maximum_relative_change"]
            <= contract.finest_change_limit_fraction
        ),
    }
    return {
        "schema_version": "h11_particle_time_step_convergence_summary_v1",
        "status": (
            "pass_particle_time_step_convergence"
            if all(gates.values())
            else "fail_particle_time_step_convergence"
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "baseline_case": str(BASELINE_PATH.resolve()),
        "cases": cases,
        "adjacent_changes": comparisons,
        "gates": gates,
        "calibrated": False,
        "paper_prediction_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument(
        "--maximum-step-us",
        type=float,
        action="append",
        choices=[1.0, 0.5],
    )
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not args.summarize_existing:
        import mph

        steps = args.maximum_step_us or [1.0, 0.5]
        client = mph.start(cores=args.cores, version=args.version)
        try:
            for index, step in enumerate(steps, start=1):
                print(
                    f"[{index}/{len(steps)}] maximum particle step={step} us",
                    flush=True,
                )
                payload = solve_case(client, step)
                compact = compact_case(step, payload)
                print(
                    f"  status={payload['status']}, "
                    f"primary_n={compact['primary_aperture']['selected_particle_count']}, "
                    f"wall={payload['wall_runtime_sec']:.1f}s",
                    flush=True,
                )
        finally:
            client.clear()
    summary = build_summary()
    if summary is None:
        print("Particle time-step ladder is incomplete; artifacts were retained.")
        return 0
    _write_json(args.output, summary)
    print(f"Summary: {args.output}")
    print(f"Time-step status: {summary['status']}")
    return 0 if summary["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
