"""Render the corrected A-group supported DOE-to-DPV evidence network."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DATA_PATH = Path(__file__).with_name("data") / "a_group_causal_network_evidence.json"
FIGURE_DIR = Path(__file__).with_name("figures")
PNG_PATH = FIGURE_DIR / "a_group_supported_causal_network.png"
SVG_PATH = FIGURE_DIR / "a_group_supported_causal_network.svg"

SOURCE_ORDER = [
    "current_a",
    "argon_flow_scfh",
    "powder_feed_g_min",
    "spray_distance_mm",
]
SOURCE_META = {
    "current_a": ("电流 600→800 A", "#4C78A8"),
    "argon_flow_scfh": ("氩气 80→120 scfh", "#59A14F"),
    "powder_feed_g_min": ("送粉 10→30 g/min", "#F28E2B"),
    "spray_distance_mm": ("喷距 80→120 mm", "#8B6FB0"),
}
TARGET_ORDER = ["temperature_c", "velocity_m_s", "particle_diameter_um"]
TARGET_META = {
    "temperature_c": "粒子温度分布",
    "velocity_m_s": "粒子速度分布",
    "particle_diameter_um": "检测粒径分布",
}
SHORT_TARGET = {
    "temperature_c": "温度",
    "velocity_m_s": "速度",
    "particle_diameter_um": "粒径",
}


def _font() -> font_manager.FontProperties:
    return font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")


def _format_effect(edge: dict) -> str:
    value = edge["mean_effect"]
    unit = {"deg C": "°C", "m/s": "m/s", "um": "μm"}[edge["unit"]]
    return f"{SHORT_TARGET[edge['target']]} {value:+.1f} {unit}"


def render() -> None:
    evidence = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    supported = [edge for edge in evidence["edges"] if edge["supported"]]
    by_source = {
        source: [edge for edge in supported if edge["source"] == source]
        for source in SOURCE_ORDER
    }
    font = _font()
    figure, axis = plt.subplots(figsize=(14, 8), dpi=180)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    source_y = dict(zip(SOURCE_ORDER, [0.80, 0.61, 0.42, 0.23]))
    target_y = dict(zip(TARGET_ORDER, [0.76, 0.50, 0.24]))

    for source in SOURCE_ORDER:
        label, color = SOURCE_META[source]
        y = source_y[source]
        box = FancyBboxPatch(
            (0.035, y - 0.07),
            0.29,
            0.14,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.4,
            edgecolor=color,
            facecolor=color + "18",
            zorder=3,
        )
        axis.add_patch(box)
        axis.text(0.055, y + 0.032, label, fontproperties=font, fontsize=13, color="#222222", va="center")
        effects = "   ".join(_format_effect(edge) for edge in by_source[source])
        if source == "spray_distance_mm":
            effects += "   速度/粒径：未支持"
        axis.text(0.055, y - 0.030, effects, fontproperties=font, fontsize=10.5, color="#4A4A4A", va="center")

    for target in TARGET_ORDER:
        y = target_y[target]
        box = FancyBboxPatch(
            (0.79, y - 0.055),
            0.17,
            0.11,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#777777",
            facecolor="#F4F5F7",
            zorder=3,
        )
        axis.add_patch(box)
        axis.text(0.875, y, TARGET_META[target], ha="center", va="center", fontproperties=font, fontsize=12, color="#222222")

    rad_by_source = {
        "current_a": [-0.10, -0.07, -0.04],
        "argon_flow_scfh": [-0.03, 0.00, 0.03],
        "powder_feed_g_min": [0.04, 0.07, 0.10],
        "spray_distance_mm": [0.14],
    }
    for source in SOURCE_ORDER:
        _, color = SOURCE_META[source]
        edges = by_source[source]
        for index, edge in enumerate(edges):
            arrow = FancyArrowPatch(
                (0.325, source_y[source]),
                (0.782, target_y[edge["target"]]),
                connectionstyle=f"arc3,rad={rad_by_source[source][index]}",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=2.0,
                linestyle="--" if edge["claim_role"] == "exploratory" else "-",
                color=color,
                alpha=0.78,
                zorder=1,
            )
            axis.add_patch(arrow)

    axis.text(
        0.035,
        0.955,
        "A组纯数据因果效应网络",
        fontproperties=font,
        fontsize=18,
        color="#222222",
        va="top",
    )
    axis.text(
        0.035,
        0.915,
        "正确映射：NNN.csv = UCE-RNNN = 表格第NNN行；n=150，66个DOE设置",
        fontproperties=font,
        fontsize=11,
        color="#666666",
        va="top",
    )
    axis.text(
        0.035,
        0.065,
        "实线：条件支持的确认性总效应   虚线：探索性总效应   数值：干预水平－参考水平的匹配均值效应",
        fontproperties=font,
        fontsize=10.5,
        color="#555555",
        va="center",
    )
    axis.text(
        0.035,
        0.025,
        "氢气与载气在A组固定，不能独立估计；H2/Ar与氩气完全别名，不作为第二处理变量。",
        fontproperties=font,
        fontsize=10,
        color="#777777",
        va="center",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    render()
