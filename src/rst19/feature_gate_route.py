"""按候选逐行解释当前质量门的观测规则路径。

本模块只读取已经生成的 ``source_catalog.csv``。它把数值字段的越界和
非数值/结构旗标分开统计，用于回答“某一类候选是被哪一类规则拦截”。
这不是反事实消融实验：旗标与质量层可能共享检测器实现，统计结果不能
直接解释为因果贡献、噪点概率或物理恒星真值。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .feature_gate_margin import (
    DEFAULT_FEATURE_GATE_THRESHOLDS,
    compute_feature_gate_margins,
)


# 这些旗标已经有对应的数值门。若对应数值确实越界，就不再把它们重复
# 计入“独立结构旗标”；若字段缺失或没有越界，则保留旗标以免隐藏冲突。
_NUMERIC_FLAG_TO_METRIC = {
    "LOW_FLUX_SNR": "flux_snr",
    "INSUFFICIENT_PSF_SUPPORT": "psf_support_pixels",
    "SMALL_FOOTPRINT": "footprint_pixels",
}


@dataclass(frozen=True, slots=True)
class FeatureGateRouteSourceRow:
    """一个候选的逐行质量门路径。"""

    detection_id: str
    feature_class: str
    feature_class_label: str
    quality_passed: bool
    route: str
    numeric_failure_reasons: str
    structural_flags: str
    flags: str
    flux_snr: str
    psf_support_pixels: str
    fwhm: str
    ellipticity: str
    sharpness: str
    footprint_pixels: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureGateRouteSummaryRow:
    """一个特征类别的质量门路径汇总。"""

    feature_class: str
    feature_class_label: str
    candidate_count: int
    quality_count: int
    rejected_count: int
    numeric_failure_count: int
    structural_flag_count: int
    rejected_numeric_only_count: int
    rejected_structural_only_count: int
    rejected_numeric_and_structural_count: int
    rejected_unexplained_count: int
    quality_passed_with_numeric_failure_count: int
    quality_passed_with_structural_flag_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureGateRouteResult:
    """逐候选路径和逐类别汇总。"""

    catalog_path: str
    thresholds: dict[str, float]
    source_rows: tuple[FeatureGateRouteSourceRow, ...]
    summary_rows: tuple[FeatureGateRouteSummaryRow, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "thresholds": dict(self.thresholds),
            "numeric_alias_flags": dict(_NUMERIC_FLAG_TO_METRIC),
            "source_rows": [row.as_dict() for row in self.source_rows],
            "summary_rows": [row.as_dict() for row in self.summary_rows],
            "interpretation_boundary": [
                "route is an observed detector-rule decomposition, not a counterfactual ablation",
                "flags and quality_passed can share implementation logic",
                "numeric and structural routes are not noise probabilities or physical star classes",
                "quality_passed remains a detector-level label until truth, WCS, catalog and injection checks close",
            ],
        }


def _quality(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _flags(value: object) -> tuple[str, ...]:
    return tuple(sorted({part.strip() for part in str(value).replace(";", "|").split("|") if part.strip()}))


def _read_catalog(catalog_path: str | Path) -> tuple[Path, list[dict[str, str]]]:
    path = Path(catalog_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "flags",
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


def _route(numeric_reasons: tuple[str, ...], structural_flags: tuple[str, ...], quality_passed: bool) -> str:
    if quality_passed:
        return "quality_passed"
    if numeric_reasons and structural_flags:
        return "rejected_numeric_and_structural"
    if numeric_reasons:
        return "rejected_numeric_only"
    if structural_flags:
        return "rejected_structural_only"
    return "rejected_unexplained"


def _numeric_reasons(record: dict[str, str]) -> tuple[str, ...]:
    margins = compute_feature_gate_margins(record, DEFAULT_FEATURE_GATE_THRESHOLDS)
    return tuple(sorted(metric for metric, margin in margins.items() if margin < 0))


def _structural_flags(record_flags: tuple[str, ...], numeric_reasons: tuple[str, ...]) -> tuple[str, ...]:
    numeric_reason_set = set(numeric_reasons)
    return tuple(
        flag
        for flag in record_flags
        if _NUMERIC_FLAG_TO_METRIC.get(flag) not in numeric_reason_set
    )


def run_feature_gate_route(catalog_path: str | Path) -> FeatureGateRouteResult:
    """按当前默认数值门逐候选计算观测规则路径。"""

    path, records = _read_catalog(catalog_path)
    seen_ids: set[str] = set()
    source_rows: list[FeatureGateRouteSourceRow] = []
    class_rows: dict[str, list[FeatureGateRouteSourceRow]] = defaultdict(list)

    for record in records:
        detection_id = str(record.get("detection_id", "")).strip()
        if not detection_id:
            raise ValueError("source catalog contains an empty detection_id")
        if detection_id in seen_ids:
            raise ValueError(f"source catalog contains duplicate detection_id: {detection_id}")
        seen_ids.add(detection_id)

        feature_class = str(record.get("feature_class", "")).strip()
        if not feature_class:
            raise ValueError(f"source catalog row {detection_id} has an empty feature_class")
        label = str(record.get("feature_class_label", "")).strip() or feature_class
        quality_passed = _quality(record.get("quality_passed", ""))
        all_flags = _flags(record.get("flags", ""))
        numeric_reasons = _numeric_reasons(record)
        structural_flags = _structural_flags(all_flags, numeric_reasons)
        row = FeatureGateRouteSourceRow(
            detection_id=detection_id,
            feature_class=feature_class,
            feature_class_label=label,
            quality_passed=quality_passed,
            route=_route(numeric_reasons, structural_flags, quality_passed),
            numeric_failure_reasons="|".join(numeric_reasons),
            structural_flags="|".join(structural_flags),
            flags="|".join(all_flags),
            flux_snr=str(record.get("flux_snr", "")),
            psf_support_pixels=str(record.get("psf_support_pixels", "")),
            fwhm=str(record.get("fwhm", "")),
            ellipticity=str(record.get("ellipticity", "")),
            sharpness=str(record.get("sharpness", "")),
            footprint_pixels=str(record.get("footprint_pixels", "")),
        )
        source_rows.append(row)
        class_rows[feature_class].append(row)

    summary_rows: list[FeatureGateRouteSummaryRow] = []
    for feature_class in sorted(class_rows):
        members = class_rows[feature_class]
        route_counts: defaultdict[str, int] = defaultdict(int)
        numeric_count = 0
        structural_count = 0
        for member in members:
            route_counts[member.route] += 1
            numeric_count += int(bool(member.numeric_failure_reasons))
            structural_count += int(bool(member.structural_flags))
        summary_rows.append(
            FeatureGateRouteSummaryRow(
                feature_class=feature_class,
                feature_class_label=members[0].feature_class_label,
                candidate_count=len(members),
                quality_count=sum(int(member.quality_passed) for member in members),
                rejected_count=sum(int(not member.quality_passed) for member in members),
                numeric_failure_count=numeric_count,
                structural_flag_count=structural_count,
                rejected_numeric_only_count=route_counts["rejected_numeric_only"],
                rejected_structural_only_count=route_counts["rejected_structural_only"],
                rejected_numeric_and_structural_count=route_counts["rejected_numeric_and_structural"],
                rejected_unexplained_count=route_counts["rejected_unexplained"],
                quality_passed_with_numeric_failure_count=sum(
                    int(member.route == "quality_passed" and bool(member.numeric_failure_reasons))
                    for member in members
                ),
                quality_passed_with_structural_flag_count=sum(
                    int(member.route == "quality_passed" and bool(member.structural_flags))
                    for member in members
                ),
            )
        )

    return FeatureGateRouteResult(
        catalog_path=str(path),
        thresholds=dict(DEFAULT_FEATURE_GATE_THRESHOLDS),
        source_rows=tuple(source_rows),
        summary_rows=tuple(summary_rows),
    )


def write_feature_gate_route_artifacts(
    result: FeatureGateRouteResult,
    output_dir: str | Path,
) -> Path:
    """写出逐候选路径、类别汇总和 JSON。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "feature_gate_route_sources.csv"
    summary_path = output / "feature_gate_route_summary.csv"
    json_path = output / "feature_gate_routes.json"
    with source_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[field.name for field in fields(FeatureGateRouteSourceRow)],
        )
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.source_rows)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[field.name for field in fields(FeatureGateRouteSummaryRow)],
        )
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.summary_rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "FeatureGateRouteResult",
    "FeatureGateRouteSourceRow",
    "FeatureGateRouteSummaryRow",
    "run_feature_gate_route",
    "write_feature_gate_route_artifacts",
]
