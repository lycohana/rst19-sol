"""把指定候选放回其首要特征类别的经验分布中。

该工具只读取 ``rst19-sources`` 已生成的源级 CSV，不重新检测 FITS，也不把
经验分位数解释为真星概率。它用于回答一个很具体的审计问题：某个候选的
高 SNR、PSF 支持或形状指标，在它所属的算法类别中究竟是普通、偏高还是
极端值。这样可以识别“看起来像点源但仍被值域/去混叠规则拒绝”的 hard
negative，而不是只引用全图绝对数。
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONTEXT_METRIC_FIELDS: tuple[str, ...] = (
    "flux_snr",
    "filter_snr",
    "fwhm",
    "ellipticity",
    "sharpness",
    "psf_support_pixels",
    "footprint_pixels",
    "centroid_shift_px",
    "peak",
)


@dataclass(frozen=True, slots=True)
class FeatureClassContextMetric:
    """一个目标在所属类别中的单指标经验位置。"""

    detection_id: int
    feature_class: str
    metric: str
    value: float
    class_count: int
    class_p10: float
    class_median: float
    class_p90: float
    empirical_percentile: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureClassContextTarget:
    """目标的类别身份和质量层概览。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    quality_passed: bool
    flags: str
    class_count: int
    class_quality_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureClassContextResult:
    """类别内经验分位审计结果。"""

    catalog_path: str
    targets: tuple[FeatureClassContextTarget, ...]
    metrics: tuple[FeatureClassContextMetric, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "targets": [target.as_dict() for target in self.targets],
            "metrics": [metric.as_dict() for metric in self.metrics],
            "interpretation_guardrails": [
                "经验分位数只表示目标在当前算法类别中的相对位置，不是真星概率。",
                "feature_class 是按旗标优先级生成的审计标签，不是物理类别。",
                "高 SNR 或高 PSF 支持不能抵消值域、共享孔径和去混叠证据。",
            ],
        }


def _finite_float(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw in ("", "nan", "NaN", "None"):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _percentile(values: Sequence[float], value: float) -> float:
    return float(sum(item <= value for item in values) / len(values)) if values else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("无法对空指标集合计算经验分位数")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def run_feature_class_context(
    catalog_path: str | Path,
    target_ids: Iterable[int],
    *,
    metric_fields: Sequence[str] = CONTEXT_METRIC_FIELDS,
) -> FeatureClassContextResult:
    """将指定 detection ID 放回同类源的经验分布中。"""

    path = Path(catalog_path)
    requested_ids = tuple(dict.fromkeys(int(value) for value in target_ids))
    if not requested_ids:
        raise ValueError("至少需要一个 target detection ID")
    unknown_metrics = sorted(set(metric_fields) - set(CONTEXT_METRIC_FIELDS))
    if unknown_metrics:
        raise ValueError(f"不支持的类别上下文指标：{', '.join(unknown_metrics)}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise OSError(f"无法读取源级 CSV：{path}") from exc
    if not rows:
        raise ValueError(f"源级 CSV 为空：{path}")
    required_columns = {"detection_id", "feature_class", "feature_class_label", "quality_passed", "flags"}
    missing = sorted(required_columns - set(rows[0]))
    if missing:
        raise ValueError(f"源级 CSV 缺少列：{', '.join(missing)}")

    by_class: dict[str, list[dict[str, str]]] = {}
    by_id: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            detection_id = int(row["detection_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"源级 CSV 存在非法 detection_id：{row.get('detection_id')!r}") from exc
        if detection_id in by_id:
            raise ValueError(f"源级 CSV 存在重复 detection_id：{detection_id}")
        by_id[detection_id] = row
        by_class.setdefault(row.get("feature_class", ""), []).append(row)

    missing_targets = sorted(set(requested_ids) - set(by_id))
    if missing_targets:
        raise ValueError(f"源级 CSV 找不到 detection_id：{missing_targets}")

    targets: list[FeatureClassContextTarget] = []
    metrics: list[FeatureClassContextMetric] = []
    for detection_id in requested_ids:
        row = by_id[detection_id]
        feature_class = row.get("feature_class", "")
        class_rows = by_class.get(feature_class, [])
        class_quality_count = sum(row.get("quality_passed", "").lower() == "true" for row in class_rows)
        targets.append(
            FeatureClassContextTarget(
                detection_id=detection_id,
                feature_class=feature_class,
                feature_class_label=row.get("feature_class_label", feature_class),
                quality_passed=row.get("quality_passed", "").lower() == "true",
                flags=row.get("flags", ""),
                class_count=len(class_rows),
                class_quality_count=class_quality_count,
            )
        )
        for metric in metric_fields:
            value = _finite_float(row, metric)
            values = sorted(
                candidate
                for candidate_row in class_rows
                if (candidate := _finite_float(candidate_row, metric)) is not None
            )
            if value is None or not values:
                continue
            metrics.append(
                FeatureClassContextMetric(
                    detection_id=detection_id,
                    feature_class=feature_class,
                    metric=metric,
                    value=value,
                    class_count=len(values),
                    class_p10=_quantile(values, 0.10),
                    class_median=_quantile(values, 0.50),
                    class_p90=_quantile(values, 0.90),
                    empirical_percentile=_percentile(values, value),
                )
            )
    return FeatureClassContextResult(
        catalog_path=str(path),
        targets=tuple(targets),
        metrics=tuple(metrics),
    )


def _write_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].as_dict().keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_feature_class_context_artifacts(
    result: FeatureClassContextResult,
    out_dir: str | Path,
) -> Path:
    """写出目标摘要、指标分位表和 JSON。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_class_context.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output / "feature_class_context_targets.csv", result.targets)
    _write_csv(output / "feature_class_context_metrics.csv", result.metrics)
    return output


__all__ = [
    "CONTEXT_METRIC_FIELDS",
    "FeatureClassContextMetric",
    "FeatureClassContextTarget",
    "FeatureClassContextResult",
    "run_feature_class_context",
    "write_feature_class_context_artifacts",
]
