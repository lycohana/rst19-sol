"""孔径净通量与局部峰值比的类别条件审计。

该指标用于发现“候选自身峰值不高，但孔径通量异常吸收邻近亮斑/结构”的
情况。它受孔径半径、PSF、背景估计和类别机制影响，因此只输出描述性分布，
不把比值转换成真星概率、伪影概率或完备率。
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class AperturePeakRatioSummary:
    """一个互斥特征类别的 ``flux / peak`` 分布摘要。"""

    feature_class: str
    sample_count: int
    ratio_median: float
    ratio_q90: float
    ratio_q95: float
    ratio_q99: float
    ratio_gt10_count: int
    ratio_gt10_fraction: float
    ratio_gt20_count: int
    ratio_gt20_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AperturePeakRatioTarget:
    """指定候选在其自身类别分布中的比值位置。"""

    detection_id: str
    feature_class: str
    peak: float
    flux: float
    flux_to_peak: float
    class_sample_count: int
    class_percentile: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AperturePeakRatioMethodSummary:
    """一个特征类别与一个提议器来源组的 ``flux / peak`` 摘要。"""

    feature_class: str
    method_group: str
    sample_count: int
    ratio_median: float
    ratio_q90: float
    ratio_q95: float
    ratio_gt10_count: int
    ratio_gt10_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AperturePeakRatioResult:
    """孔径通量/峰值审计的完整结果。"""

    catalog_path: str
    targets: tuple[str, ...]
    summaries: tuple[AperturePeakRatioSummary, ...]
    method_summaries: tuple[AperturePeakRatioMethodSummary, ...]
    target_rows: tuple[AperturePeakRatioTarget, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "targets": list(self.targets),
            "summaries": [row.as_dict() for row in self.summaries],
            "method_summaries": [row.as_dict() for row in self.method_summaries],
            "target_rows": [row.as_dict() for row in self.target_rows],
            "ratio_definition": "finite aperture flux divided by positive peak value",
            "interpretation_boundary": (
                "descriptive aperture/PSF contamination diagnostic; depends on aperture, "
                "background, PSF and feature class; not star probability, artifact probability, "
                "completeness, precision, or FDR"
            ),
        }


@dataclass(frozen=True, slots=True)
class _RatioRecord:
    detection_id: str
    feature_class: str
    peak: float
    flux: float
    ratio: float
    method_group: str


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile from an empty sequence")
    return float(np.percentile(np.asarray(values, dtype=np.float64), probability * 100.0))


def _proposal_method_group(raw_methods: str) -> str:
    """把当前混合运行的来源集合归一为描述性分组。"""

    methods = frozenset(value.strip() for value in str(raw_methods).split("|") if value.strip())
    if methods == frozenset({"gaussian", "dog_narrow", "dog_broad"}):
        return "all_three"
    if methods == frozenset({"gaussian"}):
        return "gaussian_only"
    if methods and "gaussian" not in methods:
        return "dog_only"
    if methods:
        return "partial"
    return "no_method"


def _finite_ratio_records(rows: Iterable[dict[str, str]]) -> list[_RatioRecord]:
    records: list[_RatioRecord] = []
    for row in rows:
        feature_class = str(row.get("feature_class", "")).strip()
        detection_id = str(row.get("detection_id", "")).strip()
        if not feature_class or not detection_id:
            continue
        try:
            peak = float(row.get("peak", ""))
            flux = float(row.get("flux", ""))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(peak) and math.isfinite(flux)) or peak <= 0.0:
            continue
        ratio = flux / peak
        if math.isfinite(ratio):
            records.append(
                _RatioRecord(
                    detection_id,
                    feature_class,
                    peak,
                    flux,
                    ratio,
                    _proposal_method_group(row.get("proposal_methods", "")),
                )
            )
    return records


def run_aperture_peak_ratio_audit(
    catalog_path: str | Path,
    *,
    target_ids: Sequence[str] = ("82931", "82934"),
) -> AperturePeakRatioResult:
    """按特征类别汇总 ``flux / peak``，并定位指定候选的类别分位。"""

    path = Path(catalog_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        required = {"detection_id", "feature_class", "peak", "flux"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        records = _finite_ratio_records(reader)

    if not records:
        raise ValueError("source catalog has no rows with finite flux and positive peak")

    normalized_targets = tuple(dict.fromkeys(str(value).strip() for value in target_ids if str(value).strip()))
    by_class: dict[str, list[_RatioRecord]] = {}
    for record in records:
        by_class.setdefault(record.feature_class, []).append(record)

    summaries: list[AperturePeakRatioSummary] = []
    sorted_by_class: dict[str, list[float]] = {}
    for feature_class in sorted(by_class):
        values = sorted(record.ratio for record in by_class[feature_class])
        sorted_by_class[feature_class] = values
        count = len(values)
        gt10 = sum(value > 10.0 for value in values)
        gt20 = sum(value > 20.0 for value in values)
        summaries.append(
            AperturePeakRatioSummary(
                feature_class=feature_class,
                sample_count=count,
                ratio_median=_quantile(values, 0.50),
                ratio_q90=_quantile(values, 0.90),
                ratio_q95=_quantile(values, 0.95),
                ratio_q99=_quantile(values, 0.99),
                ratio_gt10_count=gt10,
                ratio_gt10_fraction=gt10 / count,
                ratio_gt20_count=gt20,
                ratio_gt20_fraction=gt20 / count,
            )
        )

    by_method: dict[tuple[str, str], list[float]] = {}
    for record in records:
        by_method.setdefault((record.feature_class, record.method_group), []).append(record.ratio)
    method_summaries: list[AperturePeakRatioMethodSummary] = []
    for feature_class, method_group in sorted(by_method):
        values = sorted(by_method[(feature_class, method_group)])
        count = len(values)
        gt10 = sum(value > 10.0 for value in values)
        method_summaries.append(
            AperturePeakRatioMethodSummary(
                feature_class=feature_class,
                method_group=method_group,
                sample_count=count,
                ratio_median=_quantile(values, 0.50),
                ratio_q90=_quantile(values, 0.90),
                ratio_q95=_quantile(values, 0.95),
                ratio_gt10_count=gt10,
                ratio_gt10_fraction=gt10 / count,
            )
        )

    target_rows: list[AperturePeakRatioTarget] = []
    wanted = set(normalized_targets)
    for record in records:
        if record.detection_id not in wanted:
            continue
        values = sorted_by_class[record.feature_class]
        rank = bisect.bisect_right(values, record.ratio)
        target_rows.append(
            AperturePeakRatioTarget(
                detection_id=record.detection_id,
                feature_class=record.feature_class,
                peak=record.peak,
                flux=record.flux,
                flux_to_peak=record.ratio,
                class_sample_count=len(values),
                class_percentile=rank / len(values),
            )
        )
    target_rows.sort(key=lambda row: (row.detection_id, row.feature_class))

    return AperturePeakRatioResult(
        catalog_path=str(path),
        targets=normalized_targets,
        summaries=tuple(summaries),
        method_summaries=tuple(method_summaries),
        target_rows=tuple(target_rows),
    )


def write_aperture_peak_ratio_artifacts(
    result: AperturePeakRatioResult,
    output_dir: str | Path,
) -> Path:
    """写出类别摘要、目标定位表和 JSON 说明。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "aperture_peak_ratio_summary.csv"
    method_summary_path = output / "aperture_peak_ratio_method_summary.csv"
    target_path = output / "aperture_peak_ratio_targets.csv"
    json_path = output / "aperture_peak_ratio.json"

    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(AperturePeakRatioSummary)])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.summaries)
    with method_summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[field.name for field in fields(AperturePeakRatioMethodSummary)],
        )
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.method_summaries)
    with target_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(AperturePeakRatioTarget)])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.target_rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output
