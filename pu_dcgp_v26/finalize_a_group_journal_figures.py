
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image


DATA_DIR = Path(__file__).with_name("data")
FIGURE_DIR = Path(__file__).with_name("figures") / "journal_a_group"
MANIFEST_PATH = DATA_DIR / "a_group_journal_figure_manifest.json"
INDEX_PATH = FIGURE_DIR / "A_GROUP_JOURNAL_FIGURE_INDEX.md"
FORMATS = ("pdf", "svg", "png", "tiff")

MAIN_CAPTIONS = {
    1: (
        "A组DOE到DPV分布的支持效应网络。实线表示通过全部预设检验的确认性条件总效应，虚线表示探索性喷距效应；节点内数值为干预水平减参考水平的匹配均值效应。",
        "Supported DOE-to-DPV distributional effect network for group A. Solid arrows denote confirmatory conditional total effects passing all prespecified checks; the dashed arrow denotes the exploratory spray-distance effect. Values are matched intervention-minus-reference mean effects.",
    ),
    2: (
        "四个DOE对比对粒子温度均值的匹配效应及95%分层bootstrap区间。",
        "Matched mean effects of the four DOE contrasts on particle temperature with 95% hierarchical-bootstrap intervals.",
    ),
    3: (
        "四个DOE对比对粒子速度均值的匹配效应及95%分层bootstrap区间。",
        "Matched mean effects of the four DOE contrasts on particle velocity with 95% hierarchical-bootstrap intervals.",
    ),
    4: (
        "四个DOE对比对DPV检测粒径均值的匹配效应及95%分层bootstrap区间。",
        "Matched mean effects of the four DOE contrasts on DPV-detected particle size with 95% hierarchical-bootstrap intervals.",
    ),
    5: (
        "粒子温度效应在主匹配分析、线性执行顺序调整及142次运行粒子计数敏感性分析之间的稳健性。浅蓝误差线为主分析95%区间。",
        "Robustness of particle-temperature effects across the primary matched analysis, linear execution-order adjustment, and the 142-run particle-count sensitivity. Light-blue error bars show primary 95% intervals.",
    ),
    6: (
        "粒子速度效应在主匹配分析、线性执行顺序调整及142次运行粒子计数敏感性分析之间的稳健性。",
        "Robustness of particle-velocity effects across the primary matched analysis, linear execution-order adjustment, and the 142-run particle-count sensitivity.",
    ),
    7: (
        "DPV检测粒径效应在主匹配分析、线性执行顺序调整及142次运行粒子计数敏感性分析之间的稳健性。",
        "Robustness of DPV-detected particle-size effects across the primary matched analysis, linear execution-order adjustment, and the 142-run particle-count sensitivity.",
    ),
}

ESTIMAND_CN = {
    "current_600_to_800": "电流600→800 A",
    "argon_80_to_120": "氩气80→120 scfh",
    "powder_10_to_30": "送粉10→30 g/min",
    "distance_80_to_120": "喷距80→120 mm",
}
ESTIMAND_EN = {
    "current_600_to_800": "arc current from 600 to 800 A",
    "argon_80_to_120": "argon flow from 80 to 120 scfh",
    "powder_10_to_30": "powder feed from 10 to 30 g min−1",
    "distance_80_to_120": "spray distance from 80 to 120 mm",
}
OUTCOME_CN = {
    "temperature_c": "粒子温度",
    "velocity_m_s": "粒子速度",
    "particle_diameter_um": "DPV检测粒径",
}
OUTCOME_EN = {
    "temperature_c": "particle temperature",
    "velocity_m_s": "particle velocity",
    "particle_diameter_um": "DPV-detected particle size",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_captions() -> dict[int, tuple[str, str]]:
    data = json.loads(
        (DATA_DIR / "a_group_quantile_figure_data.json").read_text(encoding="utf-8")
    )
    captions = {}
    for record in data["records"]:
        status_cn = (
            "该边为探索性支持。"
            if record["status"] == "exploratory_admit"
            else "该边未通过全部预设检验。"
            if record["status"] == "abstain"
            else "该边通过全部预设检验。"
        )
        status_en = (
            "The edge is supported as exploratory."
            if record["status"] == "exploratory_admit"
            else "The edge did not pass all prespecified checks."
            if record["status"] == "abstain"
            else "The edge passed all prespecified checks."
        )
        captions[record["figure_number"]] = (
            f"{ESTIMAND_CN[record['estimand_id']]}对{OUTCOME_CN[record['outcome']]}分布的匹配分位效应；阴影为2000次三层bootstrap得到的95%同时带。{status_cn}",
            f"Matched quantile effect of {ESTIMAND_EN[record['estimand_id']]} on the {OUTCOME_EN[record['outcome']]} distribution; shading denotes the 95% simultaneous band from 2,000 three-level bootstrap replicates. {status_en}",
        )
    return captions


def finalize() -> dict:
    png_paths = sorted(FIGURE_DIR.glob("fig*.png"))
    stems_by_number = {}
    for path in png_paths:
        match = re.match(r"fig(\d{2})_", path.stem)
        if match:
            stems_by_number[int(match.group(1))] = path.stem
    if set(stems_by_number) != set(range(1, 20)):
        raise AssertionError("Expected exactly figures 01 through 19")

    captions = {**MAIN_CAPTIONS, **_quantile_captions()}
    records = []
    for number in range(1, 20):
        stem = stems_by_number[number]
        files = {}
        for suffix in FORMATS:
            path = FIGURE_DIR / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size < 1000:
                raise AssertionError(f"Missing or empty journal figure: {path}")
            item = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            if suffix in {"png", "tiff"}:
                with Image.open(path) as image:
                    dpi = image.info.get("dpi", (None, None))
                    item.update(
                        {
                            "width_px": image.width,
                            "height_px": image.height,
                            "dpi_x": float(dpi[0]) if dpi[0] else None,
                            "dpi_y": float(dpi[1]) if dpi[1] else None,
                        }
                    )
                    if item["dpi_x"] is None or item["dpi_x"] < 590:
                        raise AssertionError(f"Raster resolution below 600-dpi target: {path}")
            files[suffix] = item
        records.append(
            {
                "figure_number": number,
                "stem": stem,
                "caption_cn": captions[number][0],
                "caption_en": captions[number][1],
                "files": files,
            }
        )
    return {
        "schema": "pu_dcgp_v26_a_group_journal_figure_manifest_v1",
        "figure_count": len(records),
        "formats_per_figure": list(FORMATS),
        "standalone_single_panel": True,
        "raster_target_dpi": 600,
        "records": records,
    }


def render_index(manifest: dict) -> str:
    lines = [
        "# A-group journal figure index",
        "",
        "All figures are standalone single-panel files. Each is available as PDF, SVG, 600-dpi PNG, and LZW-compressed 600-dpi TIFF.",
        "",
    ]
    for record in manifest["records"]:
        png_name = Path(record["files"]["png"]["path"]).name
        lines.extend(
            [
                f"## Figure {record['figure_number']:02d}",
                "",
                f"PNG: [{png_name}]({png_name})",
                "",
                f"中文图注：{record['caption_cn']}",
                "",
                f"English caption: {record['caption_en']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    manifest = finalize()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    INDEX_PATH.write_text(render_index(manifest), encoding="utf-8")
    print(json.dumps({
        "figure_count": manifest["figure_count"],
        "file_count": manifest["figure_count"] * len(FORMATS),
        "manifest": str(MANIFEST_PATH.resolve()),
        "index": str(INDEX_PATH.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
