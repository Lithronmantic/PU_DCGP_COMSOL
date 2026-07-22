"""External-consistency audit of frozen COMSOL 90/110 mm outputs against B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pu_dcgp import ManifestDataSource
from simulator_v2.phase_h.h11_b_observation_plane_contract import (
    BObservationPlaneContract,
    CONTRACT_PATH,
)
from simulator_v2.phase_h.h11_particle_radial_enthalpy_nominal import _sha256

from .b_group_auxiliary_analysis import load_b_run_means


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
B_SUMMARY_PATH = DATA_DIR / "b_group_auxiliary_summary.json"
SIMULATION_DIR = HERE.parents[1] / "simulator_v2" / "phase_h" / "h11_outputs" / "b_observation_planes"
SIMULATION_AUDIT_PATH = SIMULATION_DIR / "h11_b_observation_plane_extraction.json"
SIMULATION_PARTICLE_PATH = SIMULATION_DIR / "h11_b_observation_plane_particles.npz"
OUTPUT_PATH = DATA_DIR / "comsol_b_observation_validation.json"
REPORT_PATH = DATA_DIR / "COMSOL_B_OBSERVATION_VALIDATION.md"

OUTCOMES = (
    ("temperature_c", "temperature_c"),
    ("velocity_m_s", "velocity_m_s"),
    ("particle_diameter_um", "particle_diameter_um"),
)


def _simulation_weighted_mean(
    particles: Any,
    plane_mm: int,
    outcome: str,
    aperture_radius_mm: float,
) -> tuple[float, int]:
    prefix = f"plane_{plane_mm}_"
    selected = particles[prefix + "radial_position_mm"] <= aperture_radius_mm
    weights = particles[prefix + "A_detected_diameter_weight"][selected]
    values = particles[prefix + outcome][selected]
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise RuntimeError(f"No valid simulated {outcome} at {plane_mm} mm")
    return float(np.average(values[valid], weights=weights[valid])), int(valid.sum())


def build_validation() -> dict[str, Any]:
    frozen = BObservationPlaneContract()
    frozen.validate()
    for path in (
        CONTRACT_PATH,
        B_SUMMARY_PATH,
        SIMULATION_AUDIT_PATH,
        SIMULATION_PARTICLE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract_payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    simulation_audit = json.loads(
        SIMULATION_AUDIT_PATH.read_text(encoding="utf-8")
    )
    if simulation_audit["status"] != "pass_two_plane_comsol_extraction":
        raise RuntimeError("The two-plane COMSOL extraction did not pass")
    if simulation_audit["source_model_sha256"] != contract_payload["source_model_sha256"]:
        raise RuntimeError("The extracted and contracted COMSOL sources differ")
    if _sha256(SIMULATION_PARTICLE_PATH) != simulation_audit["particle_archive_sha256"]:
        raise RuntimeError("The extracted COMSOL particle archive changed")

    b_summary = json.loads(B_SUMMARY_PATH.read_text(encoding="utf-8"))
    _, run_means = load_b_run_means()
    a_runs = ManifestDataSource(groups=("A",)).load()
    a_current = a_runs.treatment_values[
        :, a_runs.treatment_names.index("current_a")
    ]
    a_argon = a_runs.treatment_values[
        :, a_runs.treatment_names.index("argon_flow_scfh")
    ]
    a_powder = a_runs.treatment_values[
        :, a_runs.treatment_names.index("powder_feed_g_min")
    ]
    a_distance = a_runs.treatment_values[
        :, a_runs.treatment_names.index("spray_distance_mm")
    ]
    a_same_process = (
        (a_current == 700.0)
        & (a_argon == 100.0)
        & (a_powder == 20.0)
    )
    a_same_process_distance_100 = a_same_process & (a_distance == 100.0)
    results = {}
    campaign_comparison = {}
    with np.load(SIMULATION_PARTICLE_PATH) as particles:
        for output_name, b_name in OUTCOMES:
            observed = {
                int(plane): float(
                    run_means.loc[
                        run_means["measurement_position_mm"] == plane,
                        b_name,
                    ].mean()
                )
                for plane in frozen.observation_planes_mm
            }
            simulated = {}
            selected_counts = {}
            for plane in frozen.observation_planes_mm:
                mean, count = _simulation_weighted_mean(
                    particles,
                    int(plane),
                    output_name,
                    frozen.primary_aperture_radius_mm,
                )
                simulated[int(plane)] = mean
                selected_counts[int(plane)] = count
            relative_errors = {
                plane: abs(simulated[plane] - observed[plane]) / abs(observed[plane])
                for plane in observed
            }
            simulated_difference = simulated[110] - simulated[90]
            observed_difference = float(
                b_summary["position_effects"][b_name][
                    "cell_weighted_average_far_minus_near"
                ]
            )
            difference_interval = [
                float(value)
                for value in b_summary["position_effects"][b_name][
                    "average_far_minus_near_bootstrap_95_interval"
                ]
            ]
            gates = {
                "absolute_mean_relative_error_at_90_below_10_percent": (
                    relative_errors[90] <= frozen.absolute_mean_relative_error_limit
                ),
                "absolute_mean_relative_error_at_110_below_10_percent": (
                    relative_errors[110] <= frozen.absolute_mean_relative_error_limit
                ),
                "simulated_difference_inside_B_run_bootstrap_interval": (
                    difference_interval[0]
                    <= simulated_difference
                    <= difference_interval[1]
                ),
            }
            results[b_name] = {
                "observed_run_balanced_mean": {
                    str(key): value for key, value in observed.items()
                },
                "simulated_A_detection_weighted_mean": {
                    str(key): value for key, value in simulated.items()
                },
                "simulated_primary_particle_count": {
                    str(key): value for key, value in selected_counts.items()
                },
                "absolute_relative_error": {
                    str(key): value for key, value in relative_errors.items()
                },
                "observed_average_far_minus_near": observed_difference,
                "observed_average_far_minus_near_95_interval": difference_interval,
                "simulated_far_minus_near": simulated_difference,
                "gates": gates,
                "status": (
                    "pass_B_external_consistency"
                    if all(gates.values())
                    else "fail_B_external_consistency"
                ),
            }
            a_outcome = np.asarray(
                [float(np.mean(sample)) for sample in a_runs.particle_samples[b_name]],
                dtype=float,
            )
            a_same_process_mean = float(np.mean(a_outcome[a_same_process]))
            a_distance_100_mean = float(
                np.mean(a_outcome[a_same_process_distance_100])
            )
            b_midplane_interpolation = 0.5 * (observed[90] + observed[110])
            campaign_comparison[b_name] = {
                "a_same_process_run_count": int(a_same_process.sum()),
                "a_same_process_all_spray_labels_mean": a_same_process_mean,
                "a_same_process_distance_100_run_count": int(
                    a_same_process_distance_100.sum()
                ),
                "a_same_process_distance_100_mean": a_distance_100_mean,
                "b_linear_midplane_interpolation": b_midplane_interpolation,
                "b_vs_a_same_process_relative_shift": (
                    (b_midplane_interpolation - a_same_process_mean)
                    / abs(a_same_process_mean)
                ),
                "b_vs_a_distance_100_relative_shift": (
                    (b_midplane_interpolation - a_distance_100_mean)
                    / abs(a_distance_100_mean)
                ),
            }
    admitted = [
        outcome
        for outcome, values in results.items()
        if values["status"] == "pass_B_external_consistency"
    ]
    diagnosis = {
        "temperature_c": (
            "The absolute scale is near the screen, but simulated axial cooling is much stronger than B."
        ),
        "velocity_m_s": (
            "The source model is too fast for B and predicts deceleration, whereas the B average is near-flat/slightly positive."
        ),
        "particle_diameter_um": (
            "The marginal scale is close, but fixed particle size and A-derived detection weights cannot reproduce the positive B detected-diameter shift."
        ),
        "cross_outcome": (
            "The A-pooled effective-exit correction is not an input-conditioned model for the fixed B setting. B block dependence is not a free-jet input and remains a discrepancy/measurement-state signal."
        ),
    }
    return {
        "schema_version": "pu_dcgp_v26_comsol_b_observation_validation_v1",
        "status": (
            "pass_all_B_external_consistency_channels"
            if len(admitted) == len(OUTCOMES)
            else "fail_current_comsol_for_B_external_consistency"
        ),
        "scope": "all 30 B runs and the frozen unretuned COMSOL 90/110 mm extraction",
        "contract": str(CONTRACT_PATH.resolve()),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "simulation_extraction": str(SIMULATION_AUDIT_PATH.resolve()),
        "simulation_extraction_sha256": _sha256(SIMULATION_AUDIT_PATH),
        "b_summary": str(B_SUMMARY_PATH.resolve()),
        "b_summary_sha256": _sha256(B_SUMMARY_PATH),
        "primary_aperture_radius_mm": frozen.primary_aperture_radius_mm,
        "primary_simulation_weighting": frozen.primary_weighting,
        "outcomes": results,
        "admitted_outcomes": admitted,
        "a_b_campaign_comparison": campaign_comparison,
        "a_b_observation_configuration_assessment": {
            "position_adjustment": (
                "B 90/110 mm run-balanced means are linearly interpolated to the "
                "bracketed 100 mm plane before comparison with A"
            ),
            "position_only_explanation": (
                "not_supported_by_the_observed_B_90_to_110_change_magnitude_but_not_formally_excluded"
            ),
            "remaining_candidates": [
                "nonlinear particle focusing or optical selection near 100 mm",
                "DPV sampling-volume or alignment differences",
                "measurement campaign or export-state differences",
            ],
            "decision": "B_absolute_scale_not_admitted_for_A_exit_calibration",
        },
        "diagnosis": diagnosis,
        "next_model_change": (
            "Fit the input-conditioned effective-exit map only inside grouped A training folds; "
            "retain B as a relative-plane falsification check and add a plane-dependent DPV detection model. "
            "Do not tune the A exit state to the shifted B absolute scale."
        ),
        "calibrated_on_b": False,
        "paper_prediction_allowed": False,
    }


def render_report(result: dict[str, Any]) -> str:
    rows = []
    labels = {
        "temperature_c": "粒子温度 (°C)",
        "velocity_m_s": "粒子速度 (m/s)",
        "particle_diameter_um": "检测粒径 (μm)",
    }
    for outcome, values in result["outcomes"].items():
        observed = values["observed_run_balanced_mean"]
        simulated = values["simulated_A_detection_weighted_mean"]
        error = values["absolute_relative_error"]
        interval = values["observed_average_far_minus_near_95_interval"]
        rows.append(
            f"| {labels[outcome]} | {observed['90']:.3f} / {observed['110']:.3f} | "
            f"{simulated['90']:.3f} / {simulated['110']:.3f} | "
            f"{100*error['90']:.1f}% / {100*error['110']:.1f}% | "
            f"{values['simulated_far_minus_near']:+.3f} | "
            f"[{interval[0]:+.3f}, {interval[1]:+.3f}] | {values['status']} |"
        )
    return "\n".join(
        [
            "# COMSOL—B组90/110 mm观测一致性审计",
            "",
            "本审计使用冻结后、未依据B组调参的COMSOL粒子模型。B组30次运行全部保留；仿真采用固定DPV主孔径和A组检测粒径权重。",
            "",
            "| 指标 | B组90 / 110均值 | COMSOL 90 / 110均值 | 相对误差90 / 110 | COMSOL 110−90 | B组95%区间 | 判定 |",
            "|---|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "## 结论",
            "",
            "当前COMSOL三个通道均未通过B组外部一致性门槛。温度的绝对尺度接近但轴向衰减过强；速度整体偏高且轴向方向不符；粒径尺度接近但不能再现检测粒径随观测面变化。",
            "",
            "同一700 A、100 scfh、20 g/min设置下，A组49次运行的速度均值约154.9 m/s，而B组90/110 mm线性中点约120.3 m/s。比较已经考虑了观测点差异：B组90和110 mm总体均值仅相差约0.82 m/s，因此其观测到的轴向变化量不足以解释34.6 m/s差距。但这不能排除100 mm附近的非线性粒子聚焦、光学采样/对中差异或批次状态。故将其记为未解释的A/B观测配置差异，而不是确定的批次漂移。",
            "",
            "这不是通过删点或修改阈值解决的问题。下一步应在A组训练折内拟合DOE输入决定的有效出口映射，同时增加随观测平面变化的DPV检测模型；B组继续保持为不参与调参的外部检查。",
            "",
        ]
    )


def main() -> None:
    result = build_validation()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(OUTPUT_PATH)
    print(result["status"])
    for outcome, values in result["outcomes"].items():
        print(outcome, values["status"], values["simulated_far_minus_near"])


if __name__ == "__main__":
    main()
