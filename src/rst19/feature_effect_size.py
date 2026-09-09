"""特征类别的非参数效应量审计。

该模块回答的是一个有限的工程问题：在当前 ``source_catalog.csv`` 的
类别标签下，哪些源级字段能把 ``compact_quality`` 与各类落选候选分开。
它不把 ``compact_quality`` 当成物理真值，也不训练分类器；输出只是对
当前检测规则的描述性签名，避免把单一 SNR 或单一形态指标写成恒星概率。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import rankdata


DEFAULT_EFFECT_FEATURES: tuple[str, ...] = (
    "flux_snr",
    "filter_snr",
    "peak",
    "fwhm",
    "ellipticity",
    "sharpness",
    "footprint_pixels",
    "psf_support_pixels",
    "centroid_shift_px",
    "repeated_code_count",
    "range_anomaly_pixel_count",
)


@dataclass(frozen=True, slots=True)
class FeatureEffectSizeRow:
    """一个类别和一个源级特征的描述性效应量。"""

    reference_class: str
    comparison_class: str
    feature: str
    reference_count: int
    comparison_count: int
    reference_median: float
    comparison_median: float
    reference_iqr: float
    comparison_iqr: float
    auc_reference_greater: float
    cliffs_delta: float
    robust_median_gap: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureEffectSizeResult:
    """特征效应量审计结果。"""

    catalog_path: str
    reference_class: str
    comparison_classes: tuple[str, ...]
    features: tuple[str, ...]
    rows: tuple[FeatureEffectSizeRow, ...]
    top_features_by_class: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "reference_class": self.reference_class,
            "comparison_classes": list(self.comparison_classes),
            "features": list(self.features),
            "rows": [row.as_dict() for row in self.rows],
            "top_features_by_class": {
                key: list(value) for key, value in self.top_features_by_class.items()
            },
            "interpretation_boundary": (
                "descriptive detector-rule effect sizes; not physical truth, classifier accuracy, "
                "precision, FDR, or star probability"
            ),
        }


def _finite_values(rows: Iterable[dict[str, str]], feature: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(feature, "")
        if raw is None or not str(raw).strip():
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _iqr(values: Sequence[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 75) - np.percentile(values, 25))


def _auc_reference_greater(reference: Sequence[float], comparison: Sequence[float]) -> float:
    """返回 P(reference > comparison) + 0.5 P(tie)。"""

    reference_values = np.asarray(reference, dtype=np.float64)
    comparison_values = np.asarray(comparison, dtype=np.float64)
    combined = np.concatenate((reference_values, comparison_values))
    ranks = rankdata(combined, method="average")
    reference_count = len(reference_values)
    comparison_count = len(comparison_values)
    u = float(np.sum(ranks[:reference_count]) - reference_count * (reference_count + 1) / 2.0)
    return u / float(reference_count * comparison_count)


def run_feature_effect_size_audit(
    catalog_path: str | Path,
    *,
    reference_class: str = "compact_quality",
    comparison_classes: Sequence[str] | None = None,
    features: Sequence[str] = DEFAULT_EFFECT_FEATURES,
    top_n: int = 5,
) -> FeatureEffectSizeResult:
    """比较参考类别与各落选类别的源级特征分布。

    ``auc_reference_greater`` 的方向固定为“参考类别数值更大”。因此
    FWHM、椭圆率等更适合较小值的特征可能得到小于 0.5 的 AUC；这不是
    错误，而是保留特征方向，避免在没有物理标定时偷偷翻转指标。
    """

    path = Path(catalog_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        required = {"feature_class", *features}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        records = list(reader)

    by_class: dict[str, list[dict[str, str]]] = {}
    for row in records:
        class_name = str(row.get("feature_class", "")).strip()
        if class_name:
            by_class.setdefault(class_name, []).append(row)
    reference_rows = by_class.get(reference_class, [])
    if not reference_rows:
        raise ValueError(f"source catalog has no feature_class={reference_class!r}")

    if comparison_classes is None:
        comparisons = tuple(sorted(name for name in by_class if name not in {reference_class, "other_rejected"}))
    else:
        comparisons = tuple(dict.fromkeys(str(name).strip() for name in comparison_classes if str(name).strip()))
    unknown = [name for name in comparisons if name not in by_class]
    if unknown:
        raise ValueError(f"source catalog has no comparison class: {unknown[0]!r}")
    feature_names = tuple(dict.fromkeys(str(feature).strip() for feature in features if str(feature).strip()))
    if not feature_names:
        raise ValueError("features must not be empty")
    if top_n < 1:
        raise ValueError("top_n must be positive")

    output_rows: list[FeatureEffectSizeRow] = []
    top_features: dict[str, tuple[str, ...]] = {}
    for comparison_class in comparisons:
        comparison_rows = by_class[comparison_class]
        class_rows: list[FeatureEffectSizeRow] = []
        for feature in feature_names:
            reference_values = _finite_values(reference_rows, feature)
            comparison_values = _finite_values(comparison_rows, feature)
            if not reference_values or not comparison_values:
                continue
            auc = _auc_reference_greater(reference_values, comparison_values)
            reference_median = float(np.median(reference_values))
            comparison_median = float(np.median(comparison_values))
            reference_iqr = _iqr(reference_values)
            comparison_iqr = _iqr(comparison_values)
            scale = max((reference_iqr + comparison_iqr) / 2.0, 1e-12)
            row = FeatureEffectSizeRow(
                reference_class=reference_class,
                comparison_class=comparison_class,
                feature=feature,
                reference_count=len(reference_values),
                comparison_count=len(comparison_values),
                reference_median=reference_median,
                comparison_median=comparison_median,
                reference_iqr=reference_iqr,
                comparison_iqr=comparison_iqr,
                auc_reference_greater=auc,
                cliffs_delta=2.0 * auc - 1.0,
                robust_median_gap=(reference_median - comparison_median) / scale,
            )
            class_rows.append(row)
        class_rows.sort(key=lambda row: (-abs(row.cliffs_delta), row.feature))
        top_features[comparison_class] = tuple(row.feature for row in class_rows[:top_n])
        output_rows.extend(class_rows)

    return FeatureEffectSizeResult(
        catalog_path=str(path),
        reference_class=reference_class,
        comparison_classes=comparisons,
        features=feature_names,
        rows=tuple(output_rows),
        top_features_by_class=top_features,
    )


def write_feature_effect_size_artifacts(
    result: FeatureEffectSizeResult,
    output_dir: str | Path,
) -> Path:
    """写出 CSV/JSON 研究产物。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "feature_effect_sizes.csv"
    json_path = output / "feature_effect_sizes.json"
    fieldnames = list(FeatureEffectSizeRow.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row.as_dict())
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output
