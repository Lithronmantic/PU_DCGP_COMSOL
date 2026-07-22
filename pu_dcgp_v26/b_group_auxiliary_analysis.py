
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols

from experiments.pu_dcgp import ManifestDataSource
from experiments.pu_dcgp.data_source import DEFAULT_MANIFEST_PATH


OUTPUT_DIR = Path(__file__).with_name("data")
SUMMARY_PATH = OUTPUT_DIR / "b_group_auxiliary_summary.json"
RUN_MEANS_PATH = OUTPUT_DIR / "b_group_run_means.csv"
CELL_MEANS_PATH = OUTPUT_DIR / "b_group_cell_means.csv"
POSITION_DIFFERENCES_PATH = OUTPUT_DIR / "b_group_position_differences.csv"
REPORT_PATH = OUTPUT_DIR / "B_GROUP_AUXILIARY_ANALYSIS.md"

OUTCOMES = (
    ("temperature_c", "temperature_c"),
    ("velocity_m_s", "velocity_m_s"),
    ("particle_diameter_um", "particle_diameter_um"),
)
DISTANCES = np.asarray((80.0, 90.0, 100.0, 110.0, 120.0))
POSITIONS = np.asarray((90.0, 110.0))
BOOTSTRAP_REPLICATES = 20_000
RANDOM_SEED = 260721


def load_b_run_means() -> tuple[object, pd.DataFrame]:

    runs = ManifestDataSource(groups=("B",)).load()
    distance_index = runs.treatment_names.index("spray_distance_mm")
    current_index = runs.treatment_names.index("current_a")
    argon_index = runs.treatment_names.index("argon_flow_scfh")
    powder_index = runs.treatment_names.index("powder_feed_g_min")
    order_index = runs.context_names.index("execution_order")
    position_index = runs.context_names.index("measurement_position_mm")
    frame = pd.DataFrame(
        {
            "run_id": runs.run_ids,
            "execution_order": runs.context_values[:, order_index].astype(int),
            "nominal_spray_distance_mm": runs.treatment_values[:, distance_index],
            "measurement_position_mm": runs.context_values[:, position_index],
            "current_a": runs.treatment_values[:, current_index],
            "argon_flow_scfh": runs.treatment_values[:, argon_index],
            "powder_feed_g_min": runs.treatment_values[:, powder_index],
        }
    )
    for source_name, output_name in OUTCOMES:
        frame[output_name] = [
            float(np.mean(sample)) for sample in runs.particle_samples[source_name]
        ]
        frame[f"{output_name}_particle_count"] = [
            int(len(sample)) for sample in runs.particle_samples[source_name]
        ]
    _assert_frozen_b_design(frame)
    return runs, frame.sort_values("execution_order").reset_index(drop=True)


def _assert_frozen_b_design(frame: pd.DataFrame) -> None:

    if len(frame) != 30 or frame["run_id"].nunique() != 30:
        raise AssertionError("B must contain 30 unique runs")
    if tuple(sorted(frame["execution_order"].tolist())) != tuple(range(151, 181)):
        raise AssertionError("B execution orders must be 151 through 180")
    expected_distances = (80.0, 90.0, 100.0, 110.0, 120.0)
    if tuple(sorted(frame["nominal_spray_distance_mm"].unique())) != expected_distances:
        raise AssertionError("Unexpected B nominal spray-distance blocks")
    if tuple(sorted(frame["measurement_position_mm"].unique())) != (90.0, 110.0):
        raise AssertionError("Unexpected B DPV measurement planes")
    cell_counts = frame.groupby(
        ["nominal_spray_distance_mm", "measurement_position_mm"]
    ).size()
    if len(cell_counts) != 10 or not (cell_counts == 3).all():
        raise AssertionError("B must contain three runs in every 5-by-2 cell")
    fixed = {
        "current_a": 700.0,
        "argon_flow_scfh": 100.0,
        "powder_feed_g_min": 20.0,
    }
    for column, value in fixed.items():
        if not np.allclose(frame[column], value):
            raise AssertionError(f"B {column} is not fixed at {value}")


def _cell_summary(run_means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (distance, position), cell in run_means.groupby(
        ["nominal_spray_distance_mm", "measurement_position_mm"], sort=True
    ):
        row = {
            "nominal_spray_distance_mm": float(distance),
            "measurement_position_mm": float(position),
            "run_count": int(len(cell)),
            "particle_count": int(cell["velocity_m_s_particle_count"].sum()),
        }
        for _, outcome in OUTCOMES:
            row[f"{outcome}_mean"] = float(cell[outcome].mean())
            row[f"{outcome}_run_sd"] = float(cell[outcome].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = float(spearmanr(x, y).statistic)
    null = np.asarray(
        [spearmanr(x, ordering).statistic for ordering in permutations(y)],
        dtype=float,
    )
    p_value = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
    return rho, p_value


def _position_effect_analysis(
    run_means: pd.DataFrame,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    output_rows = []
    outcomes = {}
    x = (DISTANCES - 100.0) / 10.0
    denominator = float(np.sum(np.square(x)))
    for outcome_index, (_, outcome) in enumerate(OUTCOMES):
        observed = []
        bootstrapped = []
        for distance in DISTANCES:
            cell_bootstrap = {}
            cell_means = {}
            for position in POSITIONS:
                values = run_means.loc[
                    (run_means["nominal_spray_distance_mm"] == distance)
                    & (run_means["measurement_position_mm"] == position),
                    outcome,
                ].to_numpy(dtype=float)
                if len(values) != 3:
                    raise AssertionError("Every B cell must contain three run means")
                cell_means[position] = float(np.mean(values))
                draws = rng.integers(
                    0,
                    len(values),
                    size=(bootstrap_replicates, len(values)),
                )
                cell_bootstrap[position] = np.mean(values[draws], axis=1)
            difference = cell_means[110.0] - cell_means[90.0]
            bootstrap_difference = (
                cell_bootstrap[110.0] - cell_bootstrap[90.0]
            )
            observed.append(difference)
            bootstrapped.append(bootstrap_difference)
            lower, upper = np.quantile(bootstrap_difference, (0.025, 0.975))
            output_rows.append(
                {
                    "outcome": outcome,
                    "nominal_spray_distance_mm": float(distance),
                    "near_90_mean": cell_means[90.0],
                    "far_110_mean": cell_means[110.0],
                    "far_minus_near": float(difference),
                    "bootstrap_95_lower": float(lower),
                    "bootstrap_95_upper": float(upper),
                }
            )
        differences = np.asarray(observed, dtype=float)
        bootstrap_matrix = np.column_stack(bootstrapped)
        rho, exact_p = _exact_spearman(DISTANCES, differences)
        slopes = bootstrap_matrix @ x / denominator
        point_slope = float(differences @ x / denominator)
        mean_difference_draws = np.mean(bootstrap_matrix, axis=1)
        leave_one = []
        for held_index, held_distance in enumerate(DISTANCES):
            keep = np.arange(len(DISTANCES)) != held_index
            loo_x = x[keep] - np.mean(x[keep])
            loo_y = differences[keep]
            loo_slope = float(
                np.sum(loo_x * (loo_y - np.mean(loo_y)))
                / np.sum(np.square(loo_x))
            )
            loo_rho, loo_p = _exact_spearman(DISTANCES[keep], loo_y)
            leave_one.append(
                {
                    "held_out_nominal_distance_mm": float(held_distance),
                    "slope_per_10_mm": loo_slope,
                    "spearman_rho": loo_rho,
                    "exact_p": loo_p,
                }
            )
        outcomes[outcome] = {
            "far_minus_near_by_nominal_distance": [float(v) for v in differences],
            "cell_weighted_average_far_minus_near": float(np.mean(differences)),
            "average_far_minus_near_bootstrap_95_interval": [
                float(v) for v in np.quantile(mean_difference_draws, (0.025, 0.975))
            ],
            "difference_trend_slope_per_10_mm": point_slope,
            "slope_bootstrap_95_interval": [
                float(v) for v in np.quantile(slopes, (0.025, 0.975))
            ],
            "bootstrap_probability_slope_positive": float(np.mean(slopes > 0.0)),
            "distance_difference_spearman_rho": rho,
            "exact_permutation_p": exact_p,
            "leave_one_nominal_distance_out": leave_one,
            "leave_one_slope_sign_consistent": bool(
                all(np.sign(row["slope_per_10_mm"]) == np.sign(point_slope) for row in leave_one)
            ),
        }
    return outcomes, pd.DataFrame(output_rows)


def _sequence_adjusted_models(run_means: pd.DataFrame) -> dict:
    frame = run_means.copy()
    frame["block_label"] = frame["nominal_spray_distance_mm"].astype(str)
    frame["position_label"] = frame["measurement_position_mm"].astype(str)
    frame["block_c"] = (frame["nominal_spray_distance_mm"] - 100.0) / 10.0
    frame["position_c"] = (frame["measurement_position_mm"] - 100.0) / 10.0
    frame["execution_order_z"] = (
        frame["execution_order"] - frame["execution_order"].mean()
    ) / frame["execution_order"].std(ddof=0)
    audits = {}
    for _, outcome in OUTCOMES:
        unadjusted = ols(
            f"{outcome} ~ C(block_label) * C(position_label)",
            data=frame,
        ).fit()
        unadjusted_anova = sm.stats.anova_lm(unadjusted, typ=2)
        unadjusted_hc3 = sm.stats.anova_lm(unadjusted, typ=2, robust="hc3")
        categorical = ols(
            f"{outcome} ~ C(block_label) * C(position_label) + execution_order_z",
            data=frame,
        ).fit()
        classical_anova = sm.stats.anova_lm(categorical, typ=2)
        anova = sm.stats.anova_lm(categorical, typ=2, robust="hc3")
        continuous = ols(
            f"{outcome} ~ block_c * position_c + execution_order_z",
            data=frame,
        ).fit(cov_type="HC3")
        interaction_name = "block_c:position_c"
        interval = continuous.conf_int().loc[interaction_name]
        audits[outcome] = {
            "unadjusted_block_by_position_p": {
                "classical": float(
                    unadjusted_anova.loc[
                        "C(block_label):C(position_label)", "PR(>F)"
                    ]
                ),
                "hc3": float(
                    unadjusted_hc3.loc[
                        "C(block_label):C(position_label)", "PR(>F)"
                    ]
                ),
            },
            "sequence_adjusted_classical_anova_p": {
                "nominal_distance_block": float(
                    classical_anova.loc["C(block_label)", "PR(>F)"]
                ),
                "measurement_position": float(
                    classical_anova.loc["C(position_label)", "PR(>F)"]
                ),
                "block_by_position": float(
                    classical_anova.loc[
                        "C(block_label):C(position_label)", "PR(>F)"
                    ]
                ),
                "execution_order": float(
                    classical_anova.loc["execution_order_z", "PR(>F)"]
                ),
            },
            "categorical_hc3_anova_p": {
                "nominal_distance_block": float(
                    anova.loc["C(block_label)", "PR(>F)"]
                ),
                "measurement_position": float(
                    anova.loc["C(position_label)", "PR(>F)"]
                ),
                "block_by_position": float(
                    anova.loc[
                        "C(block_label):C(position_label)", "PR(>F)"
                    ]
                ),
                "execution_order": float(anova.loc["execution_order_z", "PR(>F)"]),
            },
            "categorical_design_rank": int(np.linalg.matrix_rank(categorical.model.exog)),
            "categorical_parameter_count": int(categorical.model.exog.shape[1]),
            "categorical_design_condition_number": float(
                np.linalg.cond(categorical.model.exog)
            ),
            "continuous_interaction_coefficient": float(
                continuous.params[interaction_name]
            ),
            "continuous_interaction_hc3_95_interval": [
                float(interval.iloc[0]),
                float(interval.iloc[1]),
            ],
            "continuous_interaction_hc3_p": float(
                continuous.pvalues[interaction_name]
            ),
            "continuous_execution_order_hc3_p": float(
                continuous.pvalues["execution_order_z"]
            ),
        }
    return audits


def _read_raw_prt_velocity_mean(path: Path) -> float:
    values = []
    with path.open("r", encoding="ascii", errors="replace") as stream:
        next(stream)
        for line in stream:
            columns = line.split()
            if len(columns) == 9:
                value = float(columns[4])
                if np.isfinite(value) and value > 0:
                    values.append(value)
    if not values:
        raise ValueError(f"No valid raw PRT velocity in {path}")
    return float(np.mean(values))


def _source_position_summary(
    frame: pd.DataFrame, value_column: str, cell_statistic: str
) -> dict:
    differences = []
    for distance in DISTANCES:
        values = {}
        for position in POSITIONS:
            cell = frame.loc[
                (frame["nominal_spray_distance_mm"] == distance)
                & (frame["measurement_position_mm"] == position),
                value_column,
            ]
            values[position] = float(getattr(cell, cell_statistic)())
        differences.append(values[110.0] - values[90.0])
    rho, exact_p = _exact_spearman(DISTANCES, np.asarray(differences))
    return {
        "aggregation": f"cell_{cell_statistic}_of_run_means",
        "far_minus_near_velocity_m_s": [float(v) for v in differences],
        "distance_difference_spearman_rho": rho,
        "exact_permutation_p": exact_p,
    }


def _raw_prt_sensitivity(run_means: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    data_root = (DEFAULT_MANIFEST_PATH.parent / manifest["data_root"]).resolve()
    raw_paths = {
        run["run_id"]: data_root / run["process_export"]
        for run in manifest["runs"]
        if run["group"] == "B"
    }
    source_frame = run_means[
        [
            "run_id",
            "execution_order",
            "nominal_spray_distance_mm",
            "measurement_position_mm",
            "velocity_m_s",
        ]
    ].copy()
    source_frame["raw_prt_velocity_m_s"] = [
        _read_raw_prt_velocity_mean(raw_paths[run_id])
        for run_id in source_frame["run_id"]
    ]
    source_frame["absolute_source_difference_m_s"] = np.abs(
        source_frame["velocity_m_s"] - source_frame["raw_prt_velocity_m_s"]
    )
    largest = source_frame.loc[source_frame["absolute_source_difference_m_s"].idxmax()]
    summaries = {
        "processed_strips_cell_mean": _source_position_summary(
            source_frame, "velocity_m_s", "mean"
        ),
        "raw_prt_cell_mean": _source_position_summary(
            source_frame, "raw_prt_velocity_m_s", "mean"
        ),
        "raw_prt_cell_median": _source_position_summary(
            source_frame, "raw_prt_velocity_m_s", "median"
        ),
    }
    processed = summaries["processed_strips_cell_mean"]
    raw_mean = summaries["raw_prt_cell_mean"]
    raw_median = summaries["raw_prt_cell_median"]
    if (
        processed["distance_difference_spearman_rho"] >= 0.8
        and raw_median["distance_difference_spearman_rho"] >= 0.8
        and processed["exact_permutation_p"] <= 0.05
        and raw_median["exact_permutation_p"] <= 0.05
        and (
            raw_mean["distance_difference_spearman_rho"] < 0.8
            or raw_mean["exact_permutation_p"] > 0.05
        )
    ):
        conclusion = "robust_rank_trend_with_raw_mean_export_sensitivity"
    elif all(
        item["distance_difference_spearman_rho"] >= 0.8
        and item["exact_permutation_p"] <= 0.05
        for item in summaries.values()
    ):
        conclusion = "rank_trend_agrees_across_export_summaries"
    else:
        conclusion = "rank_trend_not_stable_across_export_summaries"
    rho = float(
        spearmanr(
            source_frame["velocity_m_s"], source_frame["raw_prt_velocity_m_s"]
        ).statistic
    )
    return (
        {
            "processed_vs_raw_run_spearman_rho": rho,
            "largest_source_discrepancy_run_id": str(largest["run_id"]),
            "largest_source_discrepancy_m_s": float(
                largest["absolute_source_difference_m_s"]
            ),
            "source_summaries": summaries,
            "conclusion": conclusion,
        },
        source_frame,
    )


def build_b_group_auxiliary_analysis() -> tuple[dict, dict[str, pd.DataFrame]]:

    _, run_means = load_b_run_means()
    cells = _cell_summary(run_means)
    position_effects, differences = _position_effect_analysis(run_means)
    sequence_models = _sequence_adjusted_models(run_means)
    source_sensitivity, source_frame = _raw_prt_sensitivity(run_means)
    position_order_rho = float(
        spearmanr(
            run_means["measurement_position_mm"], run_means["execution_order"]
        ).statistic
    )
    velocity = position_effects["velocity_m_s"]
    result = {
        "schema": "pu_dcgp_v26_b_group_auxiliary_analysis_v1",
        "scope": "B group only; all 30 runs; no workpiece",
        "physical_coordinates": {
            "dpv_measurement_positions_mm_from_gun": [90.0, 110.0],
            "nominal_spray_distance_block_labels_mm": [
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
            ],
            "workpiece_present": False,
        },
        "design": {
            "run_count": int(len(run_means)),
            "factorial_cell_count": int(len(cells)),
            "runs_per_cell": sorted(int(v) for v in cells["run_count"]),
            "position_execution_order_spearman_rho": position_order_rho,
            "measurement_position_alternated": False,
        },
        "bootstrap": {
            "unit": "run resampled within each distance-by-position cell",
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": RANDOM_SEED,
        },
        "position_effects": position_effects,
        "sequence_adjusted_models": sequence_models,
        "raw_prt_velocity_sensitivity": source_sensitivity,
        "primary_velocity_finding": {
            "far_minus_near_by_nominal_distance_m_s": velocity[
                "far_minus_near_by_nominal_distance"
            ],
            "monotonic_rank_trend_rho": velocity[
                "distance_difference_spearman_rho"
            ],
            "exact_permutation_p": velocity["exact_permutation_p"],
            "leave_one_block_sign_consistent": velocity[
                "leave_one_slope_sign_consistent"
            ],
            "sequence_adjusted_block_by_position_p": sequence_models[
                "velocity_m_s"
            ]["categorical_hc3_anova_p"]["block_by_position"],
            "source_sensitivity_conclusion": source_sensitivity["conclusion"],
        },
        "claim_boundary": (
            "B supports an observation-plane-by-executed-block diagnostic, not a "
            "workpiece spray-distance causal effect. Sequential acquisition prevents "
            "separating position from time/repositioning without additional data."
        ),
        "status": "experimental_analysis_complete_comsol_validation_pending",
    }
    return result, {
        "run_means": run_means,
        "cell_means": cells,
        "position_differences": differences,
        "velocity_source_comparison": source_frame,
    }


def _fmt_interval(values: list[float]) -> str:
    return f"[{values[0]:+.3f}, {values[1]:+.3f}]"


def render_report(result: dict) -> str:
    lines = [
        "# B组自由射流辅助验证分析",
        "",
        "## 数据与坐标",
        "",
        "B组共30次DPV采集，工件不在场。DPV测量面位于喷枪下游90和110 mm；80–120 mm仅作为实际执行的名义喷距区块标签，不作为B组中的工件边界。电流、氩气与送粉量分别固定为700 A、100 scfh和20 g/min。30次运行全部保留。",
        "",
        "## 位置差结果",
        "",
        "下表中的位置差均为同一名义区块内的 `110 mm − 90 mm` 运行均值差。区间通过在每个单元内重采样3次运行获得。",
        "",
        "| 指标 | 五个区块的位置差 | 平均位置差及95%区间 | 趋势斜率/10 mm及95%区间 | Spearman rho | 精确p值 | 留一区块符号一致 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for outcome, label in (
        ("temperature_c", "粒子温度 (°C)"),
        ("velocity_m_s", "粒子速度 (m/s)"),
        ("particle_diameter_um", "粒径 (μm)"),
    ):
        row = result["position_effects"][outcome]
        differences = ", ".join(f"{v:+.3f}" for v in row["far_minus_near_by_nominal_distance"])
        mean_text = (
            f"{row['cell_weighted_average_far_minus_near']:+.3f} "
            f"{_fmt_interval(row['average_far_minus_near_bootstrap_95_interval'])}"
        )
        slope_text = (
            f"{row['difference_trend_slope_per_10_mm']:+.3f} "
            f"{_fmt_interval(row['slope_bootstrap_95_interval'])}"
        )
        lines.append(
            f"| {label} | {differences} | {mean_text} | {slope_text} | "
            f"{row['distance_difference_spearman_rho']:+.3f} | "
            f"{row['exact_permutation_p']:.4f} | "
            f"{'是' if row['leave_one_slope_sign_consistent'] else '否'} |"
        )
    velocity_model = result["sequence_adjusted_models"]["velocity_m_s"]
    source = result["raw_prt_velocity_sensitivity"]
    lines.extend(
        [
            "",
            "## 稳健性与限制",
            "",
            f"速度的区块×位置项在加入执行顺序后，HC3稳健ANOVA p={velocity_model['categorical_hc3_anova_p']['block_by_position']:.4g}。该模型设计矩阵秩为{velocity_model['categorical_design_rank']}/{velocity_model['categorical_parameter_count']}，条件数为{velocity_model['categorical_design_condition_number']:.2f}。",
            "",
            f"作为口径敏感性，未使用HC3时同一顺序调整交互项的经典ANOVA p={velocity_model['sequence_adjusted_classical_anova_p']['block_by_position']:.4g}；正文采用更保守的HC3结果，不以较小的经典p值替代。",
            "",
            f"STRIPS与原始PRT的逐运行速度排序相关为{source['processed_vs_raw_run_spearman_rho']:.3f}；最大差异出现在{source['largest_source_discrepancy_run_id']}（{source['largest_source_discrepancy_m_s']:.3f} m/s）。原始PRT按单元均值汇总时趋势可能受该差异影响，按单元中位数汇总用于稳健性检查，但任何运行都没有被删除。",
            "",
            "B组最清楚的信号是：110−90 mm的速度差随名义区块单调上升。不过90和110 mm是先后两个完整采集块，并未交替，因此该信号不能单独区分真实轴向演化、时间漂移、机器人复位或测量状态变化。它可以作为COMSOL—DPV观测算子的待解释实验约束，不能直接写成工件喷距的因果效应。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result, frames = build_b_group_auxiliary_analysis()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames["run_means"].to_csv(RUN_MEANS_PATH, index=False, encoding="utf-8-sig")
    frames["cell_means"].to_csv(CELL_MEANS_PATH, index=False, encoding="utf-8-sig")
    frames["position_differences"].to_csv(
        POSITION_DIFFERENCES_PATH, index=False, encoding="utf-8-sig"
    )
    frames["velocity_source_comparison"].to_csv(
        OUTPUT_DIR / "b_group_velocity_source_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    SUMMARY_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
