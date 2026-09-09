"""各特征类别的原始逐帧响应一致性审计。

该模块只读取已有的逐帧原始孔径摘要 CSV，并按 ``detection_id`` 连接
``source_catalog.csv``。它把每个抽样源的通量 SNR 序列压缩为稳健波动、正值
比例、符号翻转和重复值域状态，再按首要特征类别汇总。

这些响应是在固定 detector 坐标/测量口径下得到的 detector-level 诊断，不能
替代跨帧配准、PSF、WCS 或星表唯一匹配；“15 帧都有响应”也不等于 15 帧都
观测到了独立恒星。类别规则与部分输入字段存在重叠，结果只用于机制分流。

默认要求 ``feature_class`` 来自同一个 ``source_catalog.csv``。如果逐帧表本身
带有 artifact-local 的 ``feature_class``（例如 pair 专用审计），必须显式使用
``feature_class_source="frame"``；这是为了防止不同实验重复使用同一个
``detection_id`` 时发生静默错连。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


TEMPORAL_REQUIRED_FIELDS: tuple[str, ...] = (
    "flux_snr_raw",
    "negative_extreme_count",
    "repeated_code_count",
    "zero_count",
)

FEATURE_CLASS_SOURCE_CHOICES: tuple[str, ...] = ("catalog", "frame")

_TEMPORAL_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "repeated_code_count": ("repeated_code_count", "repeated_3990_3993_count"),
}


@dataclass(frozen=True, slots=True)
class TemporalConsistencySource:
    """一个抽样源的原始逐帧响应摘要。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    expected_frame_count: int
    frame_count: int
    finite_flux_frame_count: int
    coverage_fraction: float
    flux_snr_median: float
    flux_snr_mad_scaled: float
    flux_snr_relative_mad: float
    flux_snr_positive_fraction: float
    flux_snr_ge5_fraction: float
    flux_snr_sign_flip_count: int
    zero_frame_fraction: float
    negative_extreme_frame_fraction: float
    repeated_code_frame_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalConsistencyClassSummary:
    """一个首要特征类别的逐帧响应汇总。"""

    feature_class: str
    feature_class_label: str
    source_count: int
    frame_count_median: float
    flux_snr_median_median: float
    flux_snr_relative_mad_p10: float
    flux_snr_relative_mad_median: float
    flux_snr_relative_mad_p90: float
    flux_snr_positive_fraction_median: float
    flux_snr_ge5_fraction_median: float
    sign_flip_count_median: float
    source_any_sign_flip_fraction: float
    source_all_positive_fraction: float
    source_any_zero_fraction: float
    source_any_negative_extreme_fraction: float
    source_any_repeated_code_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalConsistencyAuditResult:
    """逐帧一致性审计的机器可读结果。"""

    catalog_path: str
    frame_metrics_path: str
    feature_class_source: str
    expected_frame_count: int
    selected_detection_ids: tuple[int, ...]
    feature_classes: tuple[str, ...]
    source_rows: tuple[TemporalConsistencySource, ...]
    class_summaries: tuple[TemporalConsistencyClassSummary, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "frame_metrics_path": self.frame_metrics_path,
            "feature_class_source": self.feature_class_source,
            "expected_frame_count": self.expected_frame_count,
            "selected_detection_ids": list(self.selected_detection_ids),
            "feature_classes": list(self.feature_classes),
            "source_rows": [row.as_dict() for row in self.source_rows],
            "class_summaries": [row.as_dict() for row in self.class_summaries],
            "interpretation_guardrails": [
                "逐帧响应来自固定 detector 坐标/孔径口径，不是星表身份匹配。",
                "稳健波动、正值比例和帧间重复是描述量，不是恒星概率、precision 或 FDR。",
                "feature_class 是检测器旗标优先级标签；部分 SNR/值域字段与其存在定义重叠。",
                "缺少 WCS、空间变化 PSF 和官方真值时，不能把响应持久升级为物理恒星确认。",
                "detection_id 只在同一实验产物内有意义；跨产物连接必须先通过类别血缘校验。",
            ],
        }


def _required_id(row: Mapping[str, str], field: str) -> int:
    raw = str(row.get(field, "")).strip()
    if not raw:
        raise ValueError(f"CSV row has empty {field}")
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ValueError(f"CSV row has invalid {field}={raw!r}") from exc
    if value < 0:
        raise ValueError(f"CSV row has negative {field}={value}")
    return value


def _required_float(row: Mapping[str, str], field: str) -> float:
    raw = str(row.get(field, "")).strip()
    if not raw:
        raise ValueError(f"CSV row has empty {field}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"CSV row has invalid {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"CSV row has non-finite {field}={raw!r}")
    return value


def _field_alias(row: Mapping[str, str], field: str) -> str:
    for candidate in _TEMPORAL_FIELD_ALIASES.get(field, (field,)):
        if candidate in row:
            return candidate
    aliases = "/".join(_TEMPORAL_FIELD_ALIASES.get(field, (field,)))
    raise ValueError(f"CSV row has no field alias for {field}: expected {aliases}")


def _required_float_alias(row: Mapping[str, str], field: str) -> float:
    return _required_float(row, _field_alias(row, field))


def _read_catalog(path_value: str | Path) -> tuple[Path, dict[int, tuple[str, str]]]:
    path = Path(path_value)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            required = {"detection_id", "feature_class", "feature_class_label"}
            missing = sorted(required - columns)
            if missing:
                raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
            records: dict[int, tuple[str, str]] = {}
            for row in reader:
                detection_id = _required_id(row, "detection_id")
                if detection_id in records:
                    raise ValueError(f"source catalog contains duplicate detection_id={detection_id}")
                feature_class = str(row.get("feature_class", "")).strip()
                if not feature_class:
                    raise ValueError(f"source catalog has empty feature_class for detection_id={detection_id}")
                label = str(row.get("feature_class_label", feature_class)).strip() or feature_class
                records[detection_id] = (feature_class, label)
    except OSError as exc:
        raise OSError(f"无法读取源级 CSV：{path}") from exc
    if not records:
        raise ValueError(f"source catalog is empty: {path}")
    return path, records


def _read_frame_metrics(
    path_value: str | Path,
    catalog: Mapping[int, tuple[str, str]],
) -> tuple[Path, dict[int, list[dict[str, str]]], bool]:
    path = Path(path_value)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            required = {"detection_id", "frame_index", "flux_snr_raw", "negative_extreme_count", "zero_count"}
            missing = sorted(required - columns)
            if not any(candidate in columns for candidate in _TEMPORAL_FIELD_ALIASES["repeated_code_count"]):
                missing.append("repeated_code_count/repeated_3990_3993_count")
            if missing:
                raise ValueError(f"frame metrics are missing required columns: {', '.join(missing)}")
            has_feature_class = "feature_class" in columns
            grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
            seen_frames: set[tuple[int, int]] = set()
            for row in reader:
                detection_id = _required_id(row, "detection_id")
                if detection_id not in catalog:
                    raise ValueError(
                        f"frame metrics detection_id={detection_id} is absent from source catalog"
                    )
                frame_index = _required_id(row, "frame_index")
                key = (detection_id, frame_index)
                if key in seen_frames:
                    raise ValueError(
                        f"frame metrics contains duplicate detection_id={detection_id}, frame_index={frame_index}"
                    )
                seen_frames.add(key)
                for field in TEMPORAL_REQUIRED_FIELDS:
                    _required_float_alias(row, field)
                grouped[detection_id].append(row)
    except OSError as exc:
        raise OSError(f"无法读取逐帧摘要 CSV：{path}") from exc
    if not grouped:
        raise ValueError(f"frame metrics are empty: {path}")
    for rows in grouped.values():
        rows.sort(key=lambda item: _required_id(item, "frame_index"))
    return path, grouped, has_feature_class


def _source_summary(
    detection_id: int,
    rows: Sequence[Mapping[str, str]],
    feature_class: str,
    feature_class_label: str,
    expected_frame_count: int,
) -> TemporalConsistencySource:
    flux_values = np.asarray([_required_float(row, "flux_snr_raw") for row in rows], dtype=np.float64)
    median = float(np.median(flux_values))
    mad_scaled = float(1.4826 * np.median(np.abs(flux_values - median)))
    relative_mad = mad_scaled / max(abs(median), 1.0)
    positive_fraction = float(np.count_nonzero(flux_values > 0) / len(flux_values))
    ge5_fraction = float(np.count_nonzero(flux_values >= 5) / len(flux_values))
    sign_flip_count = int(
        sum((left > 0) != (right > 0) for left, right in zip(flux_values[:-1], flux_values[1:]))
    )
    zero_fraction = float(
        sum(_required_float(row, "zero_count") > 0 for row in rows) / len(rows)
    )
    negative_fraction = float(
        sum(_required_float(row, "negative_extreme_count") > 0 for row in rows) / len(rows)
    )
    repeated_fraction = float(
        sum(_required_float_alias(row, "repeated_code_count") > 0 for row in rows) / len(rows)
    )
    return TemporalConsistencySource(
        detection_id=detection_id,
        feature_class=feature_class,
        feature_class_label=feature_class_label,
        expected_frame_count=expected_frame_count,
        frame_count=len(rows),
        finite_flux_frame_count=len(flux_values),
        coverage_fraction=float(len(rows) / expected_frame_count),
        flux_snr_median=median,
        flux_snr_mad_scaled=mad_scaled,
        flux_snr_relative_mad=float(relative_mad),
        flux_snr_positive_fraction=positive_fraction,
        flux_snr_ge5_fraction=ge5_fraction,
        flux_snr_sign_flip_count=sign_flip_count,
        zero_frame_fraction=zero_fraction,
        negative_extreme_frame_fraction=negative_fraction,
        repeated_code_frame_fraction=repeated_fraction,
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty class")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty class")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _class_summary(
    feature_class: str,
    rows: Sequence[TemporalConsistencySource],
) -> TemporalConsistencyClassSummary:
    if not rows:
        raise ValueError(f"feature class has no temporal source rows: {feature_class!r}")
    label = rows[0].feature_class_label
    if any(row.feature_class_label != label for row in rows):
        raise ValueError(f"feature class has inconsistent labels: {feature_class!r}")
    return TemporalConsistencyClassSummary(
        feature_class=feature_class,
        feature_class_label=label,
        source_count=len(rows),
        frame_count_median=_median([row.frame_count for row in rows]),
        flux_snr_median_median=_median([row.flux_snr_median for row in rows]),
        flux_snr_relative_mad_p10=_quantile([row.flux_snr_relative_mad for row in rows], 0.10),
        flux_snr_relative_mad_median=_median([row.flux_snr_relative_mad for row in rows]),
        flux_snr_relative_mad_p90=_quantile([row.flux_snr_relative_mad for row in rows], 0.90),
        flux_snr_positive_fraction_median=_median([row.flux_snr_positive_fraction for row in rows]),
        flux_snr_ge5_fraction_median=_median([row.flux_snr_ge5_fraction for row in rows]),
        sign_flip_count_median=_median([row.flux_snr_sign_flip_count for row in rows]),
        source_any_sign_flip_fraction=float(
            sum(row.flux_snr_sign_flip_count > 0 for row in rows) / len(rows)
        ),
        source_all_positive_fraction=float(
            sum(row.flux_snr_positive_fraction == 1.0 for row in rows) / len(rows)
        ),
        source_any_zero_fraction=float(sum(row.zero_frame_fraction > 0 for row in rows) / len(rows)),
        source_any_negative_extreme_fraction=float(
            sum(row.negative_extreme_frame_fraction > 0 for row in rows) / len(rows)
        ),
        source_any_repeated_code_fraction=float(
            sum(row.repeated_code_frame_fraction > 0 for row in rows) / len(rows)
        ),
    )


def _resolve_feature_class(
    detection_id: int,
    rows: Sequence[Mapping[str, str]],
    catalog: Mapping[int, tuple[str, str]],
    *,
    feature_class_source: str,
    frame_has_feature_class: bool,
) -> str:
    catalog_class = catalog[detection_id][0]
    frame_classes = {
        str(row.get("feature_class", "")).strip()
        for row in rows
        if str(row.get("feature_class", "")).strip()
    }
    if feature_class_source == "catalog":
        if frame_classes and frame_classes != {catalog_class}:
            frame_class = sorted(frame_classes)[0]
            raise ValueError(
                "feature_class mismatch for detection_id="
                f"{detection_id}: catalog={catalog_class!r}, frame={frame_class!r}; "
                "use feature_class_source='frame' only for an explicitly artifact-local table"
            )
        return catalog_class
    if feature_class_source != "frame":
        raise ValueError(
            f"feature_class_source must be one of {', '.join(FEATURE_CLASS_SOURCE_CHOICES)}"
        )
    if not frame_has_feature_class or not frame_classes:
        raise ValueError(
            "feature_class_source='frame' requires a non-empty feature_class column"
        )
    if len(frame_classes) != 1 or any(
        str(row.get("feature_class", "")).strip() == "" for row in rows
    ):
        raise ValueError(
            f"frame metrics has inconsistent feature_class for detection_id={detection_id}"
        )
    return next(iter(frame_classes))


def run_feature_temporal_consistency_audit(
    catalog_path: str | Path,
    frame_metrics_path: str | Path,
    *,
    expected_frame_count: int = 15,
    feature_classes: Iterable[str] | None = None,
    feature_class_source: str = "catalog",
    detection_ids: Iterable[int] | None = None,
) -> TemporalConsistencyAuditResult:
    """按类别汇总已有固定口径的原始逐帧响应。"""

    if expected_frame_count < 1:
        raise ValueError("expected_frame_count 必须为正整数")
    if feature_class_source not in FEATURE_CLASS_SOURCE_CHOICES:
        raise ValueError(
            f"feature_class_source must be one of {', '.join(FEATURE_CLASS_SOURCE_CHOICES)}"
        )
    catalog_file, catalog = _read_catalog(catalog_path)
    metrics_file, grouped, frame_has_feature_class = _read_frame_metrics(frame_metrics_path, catalog)
    if detection_ids is None:
        selected_ids = tuple(sorted(grouped))
    else:
        selected_ids = tuple(dict.fromkeys(int(value) for value in detection_ids))
        invalid_ids = [value for value in selected_ids if value not in grouped]
        if invalid_ids:
            raise ValueError(f"frame metrics has no detection_id={invalid_ids[0]}")
        if any(value < 0 for value in selected_ids):
            raise ValueError("detection_ids 必须为非负整数")
        if not selected_ids:
            raise ValueError("detection_ids 不能为空")
    selected_grouped = {detection_id: grouped[detection_id] for detection_id in selected_ids}
    resolved_classes = {
        detection_id: _resolve_feature_class(
            detection_id,
            selected_grouped[detection_id],
            catalog,
            feature_class_source=feature_class_source,
            frame_has_feature_class=frame_has_feature_class,
        )
        for detection_id in selected_grouped
    }
    sampled_classes = {
        resolved_classes[detection_id]
        for detection_id in selected_grouped
        if resolved_classes[detection_id] != "other_rejected"
    }
    if feature_classes is None:
        requested_classes = tuple(sorted(sampled_classes))
    else:
        requested_classes = tuple(dict.fromkeys(str(name).strip() for name in feature_classes if str(name).strip()))
    if not requested_classes:
        raise ValueError("feature_classes 不能为空，且逐帧表中没有可用类别")
    catalog_classes = {feature_class for feature_class, _label in catalog.values()}
    available_classes = catalog_classes | set(resolved_classes.values())
    unknown = [name for name in requested_classes if name not in available_classes]
    if unknown:
        raise ValueError(f"source catalog has no feature_class={unknown[0]!r}")
    missing = [
        name
        for name in requested_classes
        if not any(resolved_classes[detection_id] == name for detection_id in selected_grouped)
    ]
    if missing:
        raise ValueError(f"frame metrics has no sampled rows for feature_class={missing[0]!r}")

    class_labels: dict[str, str] = {}
    for feature_class, label in catalog.values():
        class_labels.setdefault(feature_class, label)
    source_rows_list: list[TemporalConsistencySource] = []
    for detection_id in selected_ids:
        feature_class = resolved_classes[detection_id]
        if feature_class not in requested_classes:
            continue
        source_rows_list.append(
            _source_summary(
                detection_id,
                selected_grouped[detection_id],
                feature_class,
                class_labels.get(feature_class, feature_class),
                expected_frame_count,
            )
        )
    source_rows = tuple(source_rows_list)
    by_class: dict[str, list[TemporalConsistencySource]] = defaultdict(list)
    for row in source_rows:
        by_class[row.feature_class].append(row)
    summaries = tuple(_class_summary(name, by_class[name]) for name in requested_classes)
    return TemporalConsistencyAuditResult(
        catalog_path=str(catalog_file),
        frame_metrics_path=str(metrics_file),
        feature_class_source=feature_class_source,
        expected_frame_count=expected_frame_count,
        selected_detection_ids=selected_ids,
        feature_classes=requested_classes,
        source_rows=source_rows,
        class_summaries=summaries,
    )


def write_feature_temporal_consistency_artifacts(
    result: TemporalConsistencyAuditResult,
    out_dir: str | Path,
) -> Path:
    """写出逐源、类别摘要和 JSON 产物。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_temporal_consistency_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_fields = list(TemporalConsistencySource.__dataclass_fields__)
    with (output / "feature_temporal_consistency_sources.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields)
        writer.writeheader()
        for row in result.source_rows:
            writer.writerow(row.as_dict())
    summary_fields = list(TemporalConsistencyClassSummary.__dataclass_fields__)
    with (output / "feature_temporal_consistency_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for row in result.class_summaries:
            writer.writerow(row.as_dict())
    return output


__all__ = [
    "FEATURE_CLASS_SOURCE_CHOICES",
    "TEMPORAL_REQUIRED_FIELDS",
    "TemporalConsistencyAuditResult",
    "TemporalConsistencyClassSummary",
    "TemporalConsistencySource",
    "run_feature_temporal_consistency_audit",
    "write_feature_temporal_consistency_artifacts",
]
