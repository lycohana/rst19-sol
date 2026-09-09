"""源级特征相关性与类别中位数审计。

这个模块只回答一个方法学问题：当前源表中的峰值、三种 SNR、PSF
支持和形状字段，有多少是在重复表达同一个响应，哪些类别仍需要独立的
形态、值域或时序证据。它不训练分类器、不把相关性当作因果关系，也不
生成恒星概率、precision、FDR 或物理真值。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


DEFAULT_CORRELATION_FEATURES: tuple[str, ...] = (
    "peak",
    "flux_snr",
    "filter_snr",
    "fwhm",
    "ellipticity",
    "sharpness",
    "footprint_pixels",
    "psf_support_pixels",
    "centroid_shift_px",
)


@dataclass(frozen=True, slots=True)
class FeatureCorrelationRow:
    """两个源级字段的 Spearman 相关性。"""

    feature_a: str
    feature_b: str
    sample_count: int
    spearman_rho: float | None
    absolute_spearman_rho: float | None
    strong_threshold: float
    strong_absolute_correlation: bool | None

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "sample_count": self.sample_count,
            "spearman_rho": self.spearman_rho,
            "absolute_spearman_rho": self.absolute_spearman_rho,
            "strong_threshold": self.strong_threshold,
            "strong_absolute_correlation": self.strong_absolute_correlation,
        }


@dataclass(frozen=True, slots=True)
class FeatureClassMedianRow:
    """一个首要类别的源级特征中位数。"""

    feature_class: str
    feature_class_label: str
    sample_count: int
    feature_names: tuple[str, ...]
    medians: tuple[float | None, ...]

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "sample_count": self.sample_count,
        }
        result.update(
            {
                feature: value
                for feature, value in zip(self.feature_names, self.medians, strict=True)
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class FeatureClassCorrelationRow:
    """一个首要类别内部的字段相关性。"""

    feature_class: str
    feature_class_label: str
    feature_a: str
    feature_b: str
    sample_count: int
    spearman_rho: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "sample_count": self.sample_count,
            "spearman_rho": self.spearman_rho,
        }


@dataclass(frozen=True, slots=True)
class FeatureCorrelationAuditResult:
    """特征相关性和类别中位数的可复核结果。"""

    catalog_path: str
    features: tuple[str, ...]
    strong_threshold: float
    correlation_rows: tuple[FeatureCorrelationRow, ...]
    class_median_rows: tuple[FeatureClassMedianRow, ...]
    class_correlation_rows: tuple[FeatureClassCorrelationRow, ...]

    def as_dict(self) -> dict[str, object]:
        strong_rows = [
            row.as_dict()
            for row in self.correlation_rows
            if row.strong_absolute_correlation is True
        ]
        return {
            "catalog_path": self.catalog_path,
            "features": list(self.features),
            "strong_threshold": self.strong_threshold,
            "correlations": [row.as_dict() for row in self.correlation_rows],
            "strong_correlations": strong_rows,
            "class_medians": [row.as_dict() for row in self.class_median_rows],
            "class_correlations": [row.as_dict() for row in self.class_correlation_rows],
            "interpretation_boundary": (
                "descriptive source-field dependence and class medians; not causality, "
                "classifier accuracy, precision, FDR, or star probability"
            ),
        }


def _finite_value(row: Mapping[str, str], feature: str) -> float | None:
    raw = row.get(feature, "")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) < 2 or len(values_a) != len(values_b):
        return None
    ranks_a = rankdata(np.asarray(values_a, dtype=np.float64), method="average")
    ranks_b = rankdata(np.asarray(values_b, dtype=np.float64), method="average")
    centered_a = ranks_a - float(np.mean(ranks_a))
    centered_b = ranks_b - float(np.mean(ranks_b))
    denominator = float(np.sqrt(np.sum(centered_a**2) * np.sum(centered_b**2)))
    if denominator <= 0.0:
        return None
    return float(np.sum(centered_a * centered_b) / denominator)


def _normalise_features(features: Sequence[str]) -> tuple[str, ...]:
    normalised = tuple(dict.fromkeys(str(feature).strip() for feature in features if str(feature).strip()))
    if len(normalised) < 2:
        raise ValueError("features must contain at least two non-empty names")
    return normalised


def run_feature_correlation_audit(
    catalog_path: str | Path,
    *,
    features: Sequence[str] = DEFAULT_CORRELATION_FEATURES,
    strong_threshold: float = 0.45,
) -> FeatureCorrelationAuditResult:
    """读取源表并计算字段相关性与按类别中位数。"""

    if not 0.0 < float(strong_threshold) <= 1.0:
        raise ValueError("strong_threshold must be in (0, 1]")
    feature_names = _normalise_features(features)
    path = Path(catalog_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        required = {"feature_class", *feature_names}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        records = list(reader)

    correlation_rows: list[FeatureCorrelationRow] = []
    for index, feature_a in enumerate(feature_names):
        for feature_b in feature_names[index + 1 :]:
            values_a: list[float] = []
            values_b: list[float] = []
            for record in records:
                value_a = _finite_value(record, feature_a)
                value_b = _finite_value(record, feature_b)
                if value_a is None or value_b is None:
                    continue
                values_a.append(value_a)
                values_b.append(value_b)
            rho = _spearman(values_a, values_b)
            absolute = abs(rho) if rho is not None else None
            correlation_rows.append(
                FeatureCorrelationRow(
                    feature_a=feature_a,
                    feature_b=feature_b,
                    sample_count=len(values_a),
                    spearman_rho=rho,
                    absolute_spearman_rho=absolute,
                    strong_threshold=float(strong_threshold),
                    strong_absolute_correlation=(
                        absolute >= float(strong_threshold) if absolute is not None else None
                    ),
                )
            )

    by_class: dict[str, list[Mapping[str, str]]] = {}
    labels: dict[str, str] = {}
    for record in records:
        feature_class = str(record.get("feature_class", "")).strip()
        if not feature_class:
            continue
        by_class.setdefault(feature_class, []).append(record)
        labels.setdefault(feature_class, str(record.get("feature_class_label", "")).strip())

    class_median_rows: list[FeatureClassMedianRow] = []
    class_correlation_rows: list[FeatureClassCorrelationRow] = []
    for feature_class, class_records in by_class.items():
        medians: list[float | None] = []
        for feature in feature_names:
            values = [
                value
                for record in class_records
                if (value := _finite_value(record, feature)) is not None
            ]
            medians.append(float(np.median(values)) if values else None)
        class_median_rows.append(
            FeatureClassMedianRow(
                feature_class=feature_class,
                feature_class_label=labels.get(feature_class, ""),
                sample_count=len(class_records),
                feature_names=feature_names,
                medians=tuple(medians),
            )
        )
        for index, feature_a in enumerate(feature_names):
            for feature_b in feature_names[index + 1 :]:
                values_a: list[float] = []
                values_b: list[float] = []
                for record in class_records:
                    value_a = _finite_value(record, feature_a)
                    value_b = _finite_value(record, feature_b)
                    if value_a is None or value_b is None:
                        continue
                    values_a.append(value_a)
                    values_b.append(value_b)
                class_correlation_rows.append(
                    FeatureClassCorrelationRow(
                        feature_class=feature_class,
                        feature_class_label=labels.get(feature_class, ""),
                        feature_a=feature_a,
                        feature_b=feature_b,
                        sample_count=len(values_a),
                        spearman_rho=_spearman(values_a, values_b),
                    )
                )

    return FeatureCorrelationAuditResult(
        catalog_path=str(path),
        features=feature_names,
        strong_threshold=float(strong_threshold),
        correlation_rows=tuple(correlation_rows),
        class_median_rows=tuple(class_median_rows),
        class_correlation_rows=tuple(class_correlation_rows),
    )


def write_feature_correlation_artifacts(
    result: FeatureCorrelationAuditResult,
    output_dir: str | Path,
) -> Path:
    """写出相关性表、类别中位数表和 JSON。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    correlation_path = output / "feature_correlations.csv"
    class_median_path = output / "feature_class_medians.csv"
    class_correlation_path = output / "feature_class_correlations.csv"
    json_path = output / "feature_correlation_audit.json"

    with correlation_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FeatureCorrelationRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.correlation_rows)

    class_fieldnames = ["feature_class", "feature_class_label", "sample_count", *result.features]
    with class_median_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=class_fieldnames)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.class_median_rows)

    with class_correlation_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(FeatureClassCorrelationRow.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.class_correlation_rows)

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "DEFAULT_CORRELATION_FEATURES",
    "FeatureClassMedianRow",
    "FeatureClassCorrelationRow",
    "FeatureCorrelationAuditResult",
    "FeatureCorrelationRow",
    "run_feature_correlation_audit",
    "write_feature_correlation_artifacts",
]
