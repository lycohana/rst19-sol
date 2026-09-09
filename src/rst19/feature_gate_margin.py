"""按特征类别统计数值质量门的有符号余量。

本模块只读取已经生成的 ``source_catalog.csv``，不重新检测 FITS，也不把
多个余量压成一个“真星分数”。对每个源分别计算通量 SNR、PSF 支持、FWHM、
椭圆率、sharpness 和足迹相对当前质量门的余量：正值表示该字段位于门内，
负值表示该字段越过门。输出用于解释类别机制和 hard negative，不能替代
星表/WCS、注入真值、PSF 留出或人工标注。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping


DEFAULT_FEATURE_GATE_THRESHOLDS: dict[str, float] = {
    "min_flux_snr": 5.0,
    "min_psf_support_pixels": 3.0,
    "min_fwhm": 0.8,
    "max_fwhm": 12.0,
    "max_ellipticity": 0.65,
    "min_sharpness": 0.005,
    "max_sharpness": 0.85,
    "min_footprint_pixels": 2.0,
}


@dataclass(frozen=True, slots=True)
class FeatureGateMarginRow:
    """一个类别和一个质量门字段的余量分布。"""

    feature_class: str
    feature_class_label: str
    class_count: int
    quality_count: int
    metric: str
    value_count: int
    negative_margin_count: int
    negative_margin_fraction: float
    median_margin: float
    p10_margin: float
    p90_margin: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureGateMarginResult:
    """质量门余量审计结果。"""

    catalog_path: str
    thresholds: dict[str, float]
    rows: tuple[FeatureGateMarginRow, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "thresholds": dict(self.thresholds),
            "rows": [row.as_dict() for row in self.rows],
            "interpretation_boundary": [
                "a positive margin means the selected numeric field is inside the configured gate",
                "a negative margin means that field is outside the configured gate",
                "margins have different units and must not be summed or ranked across metrics",
                "quality_passed and feature_class remain detector-level labels, not physical truth",
            ],
        }


def _finite_float(value: object) -> float | None:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _quality(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def compute_feature_gate_margins(
    row: Mapping[str, str],
    thresholds: Mapping[str, float],
) -> dict[str, float]:
    """计算一行候选的数值质量门余量。"""

    values = {key: _finite_float(row.get(key, "")) for key in (
        "flux_snr",
        "psf_support_pixels",
        "fwhm",
        "ellipticity",
        "sharpness",
        "footprint_pixels",
    )}
    margins: dict[str, float] = {}
    if values["flux_snr"] is not None:
        margins["flux_snr"] = values["flux_snr"] - thresholds["min_flux_snr"]
    if values["psf_support_pixels"] is not None:
        margins["psf_support_pixels"] = values["psf_support_pixels"] - thresholds["min_psf_support_pixels"]
    if values["fwhm"] is not None:
        margins["fwhm"] = min(
            values["fwhm"] - thresholds["min_fwhm"],
            thresholds["max_fwhm"] - values["fwhm"],
        )
    if values["ellipticity"] is not None:
        margins["ellipticity"] = thresholds["max_ellipticity"] - values["ellipticity"]
    if values["sharpness"] is not None:
        margins["sharpness"] = min(
            values["sharpness"] - thresholds["min_sharpness"],
            thresholds["max_sharpness"] - values["sharpness"],
        )
    if values["footprint_pixels"] is not None:
        margins["footprint_pixels"] = values["footprint_pixels"] - thresholds["min_footprint_pixels"]
    return margins


def _read_catalog(catalog_path: str | Path) -> tuple[Path, list[dict[str, str]]]:
    path = Path(catalog_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "flux_snr",
            "psf_support_pixels",
            "fwhm",
            "ellipticity",
            "sharpness",
            "footprint_pixels",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        records = list(reader)
    if not records:
        raise ValueError("source catalog is empty")
    return path, records


def run_feature_gate_margin(
    catalog_path: str | Path,
    *,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    min_fwhm: float = 0.8,
    max_fwhm: float = 12.0,
    max_ellipticity: float = 0.65,
    min_sharpness: float = 0.005,
    max_sharpness: float = 0.85,
    min_footprint_pixels: int = 2,
) -> FeatureGateMarginResult:
    """统计每一类候选在各数值质量门上的有符号余量。"""

    thresholds = {
        "min_flux_snr": float(min_flux_snr),
        "min_psf_support_pixels": float(min_psf_support_pixels),
        "min_fwhm": float(min_fwhm),
        "max_fwhm": float(max_fwhm),
        "max_ellipticity": float(max_ellipticity),
        "min_sharpness": float(min_sharpness),
        "max_sharpness": float(max_sharpness),
        "min_footprint_pixels": float(min_footprint_pixels),
    }
    if thresholds["min_flux_snr"] <= 0:
        raise ValueError("min_flux_snr must be positive")
    if thresholds["min_psf_support_pixels"] < 1:
        raise ValueError("min_psf_support_pixels must be positive")
    if thresholds["min_fwhm"] <= 0 or thresholds["max_fwhm"] < thresholds["min_fwhm"]:
        raise ValueError("fwhm thresholds are invalid")
    if not 0 <= thresholds["max_ellipticity"] <= 1:
        raise ValueError("max_ellipticity must be between 0 and 1")
    if thresholds["min_sharpness"] < 0 or thresholds["max_sharpness"] <= thresholds["min_sharpness"]:
        raise ValueError("sharpness thresholds are invalid")
    if thresholds["min_footprint_pixels"] < 1:
        raise ValueError("min_footprint_pixels must be positive")

    path, records = _read_catalog(catalog_path)
    seen_ids: set[str] = set()
    class_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        detection_id = str(record.get("detection_id", "")).strip()
        if not detection_id:
            raise ValueError("source catalog contains an empty detection_id")
        if detection_id in seen_ids:
            raise ValueError(f"source catalog contains duplicate detection_id: {detection_id}")
        seen_ids.add(detection_id)
        feature_class = str(record.get("feature_class", "")).strip()
        if feature_class:
            class_records[feature_class].append(record)

    output_rows: list[FeatureGateMarginRow] = []
    for feature_class in sorted(class_records):
        members = class_records[feature_class]
        margin_values: dict[str, list[float]] = defaultdict(list)
        for member in members:
            for metric, margin in compute_feature_gate_margins(member, thresholds).items():
                margin_values[metric].append(margin)
        quality_count = sum(_quality(member.get("quality_passed", "")) for member in members)
        label = members[0].get("feature_class_label", feature_class)
        for metric in sorted(margin_values):
            values = sorted(margin_values[metric])
            negative_count = sum(value < 0 for value in values)
            output_rows.append(
                FeatureGateMarginRow(
                    feature_class=feature_class,
                    feature_class_label=label,
                    class_count=len(members),
                    quality_count=quality_count,
                    metric=metric,
                    value_count=len(values),
                    negative_margin_count=negative_count,
                    negative_margin_fraction=negative_count / len(values),
                    median_margin=float(_quantile(values, 0.50)),
                    p10_margin=float(_quantile(values, 0.10)),
                    p90_margin=float(_quantile(values, 0.90)),
                )
            )
    output_rows.sort(key=lambda row: (row.feature_class, row.metric))
    return FeatureGateMarginResult(
        catalog_path=str(path),
        thresholds=thresholds,
        rows=tuple(output_rows),
    )


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile for an empty sequence")
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def write_feature_gate_margin_artifacts(
    result: FeatureGateMarginResult,
    output_dir: str | Path,
) -> Path:
    """写出质量门余量 CSV 和 JSON。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "feature_gate_margins.csv"
    json_path = output / "feature_gate_margins.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(FeatureGateMarginRow)])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "DEFAULT_FEATURE_GATE_THRESHOLDS",
    "FeatureGateMarginResult",
    "FeatureGateMarginRow",
    "compute_feature_gate_margins",
    "run_feature_gate_margin",
    "write_feature_gate_margin_artifacts",
]
