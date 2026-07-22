"""Run the frozen one-factor H11 particle-parameter sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import (
    SOLVE_AUDIT_PATH as NOMINAL_SOLVE_AUDIT,
    SOURCE_MODEL,
    _sha256,
    audit_build,
    build_model,
    solve_and_audit,
)
from simulator_v2.phase_h.h11_particle_sensitivity_contract import (
    OUTPUT_PATH as CONTRACT_PATH,
    ParticleSensitivityCase,
    sensitivity_cases,
)


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "h11_outputs" / "particle_parameter_sensitivity"
SUMMARY_PATH = OUTPUT_DIR / "h11_particle_parameter_sensitivity_summary.json"


def _compact_case(
    case: ParticleSensitivityCase,
    solve_audit: dict[str, Any],
) -> dict[str, Any]:
    trajectory = solve_audit.get("trajectory_audit", {})
    unweighted = trajectory.get("unweighted_target_comparison", {})
    target = unweighted.get("dpv_prt_pooled_median", {})
    target_temperature = target.get("temperature_c")
    target_speed = target.get("speed_m_s")
    apertures = []
    for aperture in trajectory.get("centerline_aperture_sensitivity", []):
        prediction = aperture[
            "empirical_detected_diameter_weighted_median"
        ]
        temperature = prediction["temperature_c"]
        speed = prediction["speed_m_s"]
        if (
            temperature is None
            or speed is None
            or target_temperature is None
            or target_speed is None
        ):
            error = None
        else:
            error = math.sqrt(
                ((temperature - target_temperature) / target_temperature) ** 2
                + ((speed - target_speed) / target_speed) ** 2
            )
        apertures.append(
            {
                "radius_mm": aperture["aperture_radius_mm"],
                "selected_particle_count": aperture[
                    "selected_particle_count"
                ],
                "temperature_c": temperature,
                "speed_m_s": speed,
                "diameter_um": prediction["diameter_um"],
                "joint_relative_error": error,
            }
        )
    return {
        "case": case.name,
        "effective_exit_speed_m_s": case.effective_exit_speed_m_s,
        "emissivity": case.emissivity,
        "role": case.role,
        "status": solve_audit.get("status"),
        "runtime_sec": solve_audit.get("runtime_sec"),
        "crossing_count": trajectory.get("crossing_count"),
        "diameter_support_matches_contract": trajectory.get(
            "diameter_support_matches_contract"
        ),
        "all_diameter_nodes_reach_observation_plane": trajectory.get(
            "all_diameter_nodes_reach_observation_plane"
        ),
        "target": target,
        "apertures": apertures,
    }


def _case_path(case: ParticleSensitivityCase) -> Path:
    return OUTPUT_DIR / "cases" / f"{case.name}.json"


def _run_case(client: Any, case: ParticleSensitivityCase) -> dict[str, Any]:
    model = None
    try:
        model, jm = build_model(client)
        jm.param().set(
            "u_particle_exit_eff",
            f"{case.effective_exit_speed_m_s:.12g}[m/s]",
        )
        jm.param().set("eps_ysz", f"{case.emissivity:.12g}")
        build = audit_build(jm)
        solve = solve_and_audit(model, jm)
        return {
            "schema_version": "h11_particle_parameter_sensitivity_case_v1",
            "case": case.name,
            "parameters": {
                "u_particle_exit_eff_m_s": case.effective_exit_speed_m_s,
                "eps_ysz": case.emissivity,
            },
            "build_audit": build,
            "solve_audit": solve,
            "source_model": str(SOURCE_MODEL.resolve()),
            "source_model_sha256": _sha256(SOURCE_MODEL),
            "paper_use_allowed": False,
        }
    finally:
        if model is not None:
            client.remove(model)


def run_sensitivity(
    *,
    selected_names: set[str] | None = None,
    cores: int = 4,
    version: str = "6.3",
) -> dict[str, Any]:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(CONTRACT_PATH)
    if not NOMINAL_SOLVE_AUDIT.exists():
        raise FileNotFoundError(NOMINAL_SOLVE_AUDIT)
    cases = sensitivity_cases()
    known_names = {case.name for case in cases}
    if selected_names and not selected_names <= known_names:
        raise ValueError(f"Unknown sensitivity cases: {selected_names-known_names}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (_case_path(cases[0]).parent).mkdir(parents=True, exist_ok=True)
    nominal_audit = json.loads(NOMINAL_SOLVE_AUDIT.read_text(encoding="utf-8"))
    compact = [_compact_case(cases[0], nominal_audit)]

    import mph

    solve_cases = [
        case
        for case in cases
        if not case.reuse_nominal_artifact
        and (selected_names is None or case.name in selected_names)
    ]
    client = mph.start(cores=cores, version=version)
    try:
        for index, case in enumerate(solve_cases, start=1):
            print(
                f"[{index}/{len(solve_cases)}] {case.name}: "
                f"u={case.effective_exit_speed_m_s:g} m/s, "
                f"eps={case.emissivity:g}",
                flush=True,
            )
            started = time.time()
            payload = _run_case(client, case)
            path = _case_path(case)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            compact_case = _compact_case(case, payload["solve_audit"])
            compact.append(compact_case)
            print(
                f"  status={compact_case['status']}, "
                f"wall={time.time()-started:.1f}s, wrote={path.name}",
                flush=True,
            )
    finally:
        client.clear()

    candidates = [
        {
            "case": item["case"],
            **aperture,
        }
        for item in compact
        for aperture in item["apertures"]
        if aperture["joint_relative_error"] is not None
    ]
    best = (
        min(candidates, key=lambda item: item["joint_relative_error"])
        if candidates
        else None
    )
    return {
        "schema_version": "h11_particle_parameter_sensitivity_summary_v1",
        "status": (
            "pass_all_requested_sensitivity_cases"
            if all(
                item["status"] == "pass_nominal_comsol_trajectory_audit"
                for item in compact
            )
            else "fail_one_or_more_sensitivity_cases"
        ),
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "cases": compact,
        "best_screening_point_not_adopted": best,
        "interpretation_rule": (
            "Use signs and tradeoffs to diagnose the model. Do not adopt a "
            "case or aperture until calibration and heldout splits are frozen."
        ),
        "paper_use_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--version", default="6.3")
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in sensitivity_cases()],
    )
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    selected = set(args.case) if args.case else None
    summary = run_sensitivity(
        selected_names=selected,
        cores=args.cores,
        version=args.version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Summary: {args.output}")
    return 0 if summary["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
