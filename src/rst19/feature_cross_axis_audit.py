"""首要特征类别的跨证据轴一致性审计。

该模块把已有的源目录、原始局部抽样摘要、raw 局部峰审计和逐帧响应摘要
按 ``detection_id`` 严格连接。它不重新读取 FITS，也不把多个证据轴加权成
“真星分数”；输出的是每个源的证据并置、类别级中位数和离散模式计数。

``feature_class`` 仍是检测器的优先级旗标，不是物理类别。原始支持、值域
计数和部分时序字段也可能参与过原规则，所以本模块用于发现规则重叠、证据
冲突和复核优先级，不用于估计 precision、FDR、伪影率或恒星概率。
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureCrossAxisSource:
    """一个抽样源在多个 detector-level 证据轴上的并置结果。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    quality_passed: bool
    raw_valid: bool
    raw_local_maximum_r1: bool
    raw_local_max_gap_adu_r1: float
    core_fraction_median: float
    local_noise_median: float
    support_frames_ge3: float
    flux_snr_frames_ge5: float
    negative_extreme_frames: float
    repeated_code_frames: float
    expected_frame_count: int
    temporal_frame_count: int
    temporal_coverage_fraction: float
    temporal_flux_snr_relative_mad: float
    temporal_positive_fraction: float
    temporal_flux_snr_ge5_fraction: float
    temporal_sign_flip_count: int
    raw_peak_status: str
    value_domain_status: str
    support_status: str
    evidence_pattern: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureCrossAxisClassSummary:
    """一个首要类别的跨证据轴描述性汇总。"""

    feature_class: str
    feature_class_label: str
    source_count: int
    quality_passed_fraction: float
    raw_valid_fraction: float
    raw_local_maximum_fraction: float
    raw_non_local_maximum_fraction: float
    core_fraction_median: float
    local_noise_median: float
    support_frames_ge3_median: float
    flux_snr_frames_ge5_median: float
    value_domain_anomaly_fraction: float
    temporal_coverage_median: float
    temporal_flux_snr_relative_mad_median: float
    temporal_positive_fraction_median: float
    temporal_flux_snr_ge5_fraction_median: float
    temporal_any_sign_flip_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureCrossAxisPatternSummary:
    """一个类别中某种证据组合模式的计数。"""

    feature_class: str
    feature_class_label: str
    evidence_pattern: str
    source_count: int
    source_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureCrossAxisAuditResult:
    """跨证据轴审计的机器可读结果。"""

    catalog_path: str
    raw_summary_path: str
    peak_consistency_path: str
    temporal_sources_path: str
    selected_detection_ids: tuple[int, ...]
    feature_classes: tuple[str, ...]
    source_rows: tuple[FeatureCrossAxisSource, ...]
    class_summaries: tuple[FeatureCrossAxisClassSummary, ...]
    pattern_summaries: tuple[FeatureCrossAxisPatternSummary, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "raw_summary_path": self.raw_summary_path,
            "peak_consistency_path": self.peak_consistency_path,
            "temporal_sources_path": self.temporal_sources_path,
            "selected_detection_ids": list(self.selected_detection_ids),
            "feature_classes": list(self.feature_classes),
            "source_rows": [row.as_dict() for row in self.source_rows],
            "class_summaries": [row.as_dict() for row in self.class_summaries],
            "pattern_summaries": [row.as_dict() for row in self.pattern_summaries],
            "interpretation_guardrails": [
                "feature_class 是检测器优先级旗标，不是物理恒星类别。",
                "raw 局部峰、二维支持、值域计数和逐帧响应不是彼此独立的真值标签。",
                "evidence_pattern 只描述证据并置，不是分类器、恒星概率或伪影率。",
                "跨表连接要求抽样 detection_id 集合一致，避免把不同实验产物静默拼接。",
            ],
        }


def _read_csv(path_value: str | Path, name: str) -> tuple[Path, list[dict[str, str]]]:
    path = Path(path_value)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"{name} has no header: {path}")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise OSError(f"无法读取{name}：{path}") from exc
    if not rows:
        raise ValueError(f"{name} is empty: {path}")
    return path, rows


def _required_columns(rows: Sequence[Mapping[str, str]], fields: Iterable[str], name: str) -> None:
    columns = set(rows[0])
    missing = sorted(set(fields) - columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _required_id(row: Mapping[str, str], field: str, name: str) -> int:
    raw = str(row.get(field, "")).strip()
    if not raw:
        raise ValueError(f"{name} has empty {field}")
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ValueError(f"{name} has invalid {field}={raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} has negative {field}={value}")
    return value


def _required_float(row: Mapping[str, str], field: str, name: str) -> float:
    raw = str(row.get(field, "")).strip()
    if not raw:
        raise ValueError(f"{name} has empty {field}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} has invalid {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} has non-finite {field}={raw!r}")
    return value


def _required_bool(row: Mapping[str, str], field: str, name: str) -> bool:
    raw = str(row.get(field, "")).strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} has invalid boolean {field}={raw!r}")


def _index_unique(rows: Sequence[Mapping[str, str]], field: str, name: str) -> dict[int, Mapping[str, str]]:
    indexed: dict[int, Mapping[str, str]] = {}
    for row in rows:
        detection_id = _required_id(row, field, name)
        if detection_id in indexed:
            raise ValueError(f"{name} contains duplicate detection_id={detection_id}")
        indexed[detection_id] = row
    return indexed


def _check_same_ids(
    reference_ids: set[int],
    candidate_ids: set[int],
    name: str,
) -> None:
    missing = sorted(reference_ids - candidate_ids)
    extra = sorted(candidate_ids - reference_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing[:3]}")
        if extra:
            details.append(f"extra={extra[:3]}")
        raise ValueError(f"{name} detection_id set does not match raw sample ({'; '.join(details)})")


def _pattern(raw_peak: bool, value_anomaly: bool, support_seen: bool) -> tuple[str, str, str, str]:
    raw_status = "raw_local_peak" if raw_peak else "raw_non_local_peak"
    domain_status = "value_domain_anomaly" if value_anomaly else "value_domain_clean"
    support_status = "support_seen" if support_seen else "no_support_seen"
    return raw_status, domain_status, support_status, "+".join(
        (raw_status, domain_status, support_status)
    )


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _make_class_summary(
    feature_class: str,
    label: str,
    rows: Sequence[FeatureCrossAxisSource],
) -> FeatureCrossAxisClassSummary:
    if not rows:
        raise ValueError(f"feature class has no rows: {feature_class!r}")
    return FeatureCrossAxisClassSummary(
        feature_class=feature_class,
        feature_class_label=label,
        source_count=len(rows),
        quality_passed_fraction=float(sum(row.quality_passed for row in rows) / len(rows)),
        raw_valid_fraction=float(sum(row.raw_valid for row in rows) / len(rows)),
        raw_local_maximum_fraction=float(sum(row.raw_local_maximum_r1 for row in rows) / len(rows)),
        raw_non_local_maximum_fraction=float(
            sum(not row.raw_local_maximum_r1 for row in rows) / len(rows)
        ),
        core_fraction_median=_median([row.core_fraction_median for row in rows]),
        local_noise_median=_median([row.local_noise_median for row in rows]),
        support_frames_ge3_median=_median([row.support_frames_ge3 for row in rows]),
        flux_snr_frames_ge5_median=_median([row.flux_snr_frames_ge5 for row in rows]),
        value_domain_anomaly_fraction=float(
            sum(row.value_domain_status == "value_domain_anomaly" for row in rows) / len(rows)
        ),
        temporal_coverage_median=_median([row.temporal_coverage_fraction for row in rows]),
        temporal_flux_snr_relative_mad_median=_median(
            [row.temporal_flux_snr_relative_mad for row in rows]
        ),
        temporal_positive_fraction_median=_median([row.temporal_positive_fraction for row in rows]),
        temporal_flux_snr_ge5_fraction_median=_median(
            [row.temporal_flux_snr_ge5_fraction for row in rows]
        ),
        temporal_any_sign_flip_fraction=float(
            sum(row.temporal_sign_flip_count > 0 for row in rows) / len(rows)
        ),
    )


def run_feature_cross_axis_audit(
    catalog_path: str | Path,
    raw_summary_path: str | Path,
    peak_consistency_path: str | Path,
    temporal_sources_path: str | Path,
    *,
    feature_classes: Iterable[str] | None = None,
    detection_ids: Iterable[int] | None = None,
) -> FeatureCrossAxisAuditResult:
    """严格对齐四类既有表，输出跨证据轴的描述性审计。"""

    catalog_file, catalog_rows = _read_csv(catalog_path, "source catalog")
    raw_file, raw_rows = _read_csv(raw_summary_path, "raw source summary")
    peak_file, peak_rows = _read_csv(peak_consistency_path, "peak consistency table")
    temporal_file, temporal_rows = _read_csv(temporal_sources_path, "temporal source table")

    _required_columns(
        catalog_rows,
        ("detection_id", "feature_class", "feature_class_label", "quality_passed"),
        "source catalog",
    )
    _required_columns(
        raw_rows,
        (
            "detection_id",
            "core_fraction_median",
            "noise_median",
            "frames_support_ge3",
            "frames_flux_snr_ge5",
            "frames_negative_extreme_count_gt0",
            "frames_repeated_code_count_gt0",
        ),
        "raw source summary",
    )
    _required_columns(
        peak_rows,
        (
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "raw_valid",
            "raw_local_maximum_r1",
            "raw_local_max_gap_adu_r1",
        ),
        "peak consistency table",
    )
    _required_columns(
        temporal_rows,
        (
            "detection_id",
            "feature_class",
            "expected_frame_count",
            "frame_count",
            "coverage_fraction",
            "flux_snr_relative_mad",
            "flux_snr_positive_fraction",
            "flux_snr_ge5_fraction",
            "flux_snr_sign_flip_count",
        ),
        "temporal source table",
    )

    catalog = _index_unique(catalog_rows, "detection_id", "source catalog")
    raw = _index_unique(raw_rows, "detection_id", "raw source summary")
    peak = _index_unique(peak_rows, "detection_id", "peak consistency table")
    temporal = _index_unique(temporal_rows, "detection_id", "temporal source table")
    sample_ids = set(raw)
    _check_same_ids(sample_ids, set(temporal), "temporal source table")
    if not sample_ids <= set(catalog):
        missing = sorted(sample_ids - set(catalog))
        raise ValueError(f"source catalog is missing raw sample detection_id={missing[0]}")
    if not sample_ids <= set(peak):
        missing = sorted(sample_ids - set(peak))
        raise ValueError(f"peak consistency table is missing raw sample detection_id={missing[0]}")

    if detection_ids is None:
        selected_ids = tuple(sorted(sample_ids))
    else:
        selected_ids = tuple(dict.fromkeys(int(value) for value in detection_ids))
        if not selected_ids:
            raise ValueError("detection_ids 不能为空")
        if any(value < 0 for value in selected_ids):
            raise ValueError("detection_ids 必须为非负整数")
        missing = [value for value in selected_ids if value not in sample_ids]
        if missing:
            raise ValueError(f"raw source summary has no detection_id={missing[0]}")

    selected_classes = {
        str(catalog[detection_id].get("feature_class", "")).strip()
        for detection_id in selected_ids
    }
    if feature_classes is None:
        requested_classes = tuple(sorted(selected_classes))
    else:
        requested_classes = tuple(
            dict.fromkeys(str(value).strip() for value in feature_classes if str(value).strip())
        )
    if not requested_classes:
        raise ValueError("feature_classes 不能为空")
    missing_classes = [name for name in requested_classes if name not in selected_classes]
    if missing_classes:
        raise ValueError(f"raw sample has no feature_class={missing_classes[0]!r}")

    source_result: list[FeatureCrossAxisSource] = []
    for detection_id in selected_ids:
        catalog_row = catalog[detection_id]
        feature_class = str(catalog_row.get("feature_class", "")).strip()
        if feature_class not in requested_classes:
            continue
        raw_row = raw[detection_id]
        peak_row = peak[detection_id]
        temporal_row = temporal[detection_id]
        temporal_class = str(temporal_row.get("feature_class", "")).strip()
        if temporal_class and temporal_class != feature_class:
            raise ValueError(
                f"feature_class mismatch for detection_id={detection_id}: "
                f"catalog={feature_class!r}, temporal={temporal_class!r}"
            )
        peak_class = str(peak_row.get("feature_class", "")).strip()
        if peak_class != feature_class:
            raise ValueError(
                f"feature_class mismatch for detection_id={detection_id}: "
                f"catalog={feature_class!r}, peak={peak_class!r}"
            )
        catalog_label = str(catalog_row.get("feature_class_label", "")).strip() or feature_class
        peak_label = str(peak_row.get("feature_class_label", "")).strip() or peak_class
        if peak_label != catalog_label:
            raise ValueError(
                f"feature_class_label mismatch for detection_id={detection_id}: "
                f"catalog={catalog_label!r}, peak={peak_label!r}"
            )
        catalog_quality = _required_bool(catalog_row, "quality_passed", "source catalog")
        peak_quality = _required_bool(peak_row, "quality_passed", "peak consistency table")
        if peak_quality != catalog_quality:
            raise ValueError(
                f"quality_passed mismatch for detection_id={detection_id}: "
                f"catalog={catalog_quality}, peak={peak_quality}"
            )
        expected_frames = int(_required_float(temporal_row, "expected_frame_count", "temporal source table"))
        frame_count = int(_required_float(temporal_row, "frame_count", "temporal source table"))
        if expected_frames < 1 or frame_count < 0 or frame_count > expected_frames:
            raise ValueError(f"temporal frame counts are invalid for detection_id={detection_id}")
        negative_frames = _required_float(
            raw_row, "frames_negative_extreme_count_gt0", "raw source summary"
        )
        repeated_frames = _required_float(
            raw_row, "frames_repeated_code_count_gt0", "raw source summary"
        )
        support_frames = _required_float(raw_row, "frames_support_ge3", "raw source summary")
        raw_peak = _required_bool(peak_row, "raw_local_maximum_r1", "peak consistency table")
        value_anomaly = negative_frames > 0 or repeated_frames > 0
        raw_status, domain_status, support_status, pattern = _pattern(
            raw_peak,
            value_anomaly,
            support_frames > 0,
        )
        source_result.append(
            FeatureCrossAxisSource(
                detection_id=detection_id,
                feature_class=feature_class,
                feature_class_label=catalog_label,
                quality_passed=catalog_quality,
                raw_valid=_required_bool(peak_row, "raw_valid", "peak consistency table"),
                raw_local_maximum_r1=raw_peak,
                raw_local_max_gap_adu_r1=_required_float(
                    peak_row, "raw_local_max_gap_adu_r1", "peak consistency table"
                ),
                core_fraction_median=_required_float(
                    raw_row, "core_fraction_median", "raw source summary"
                ),
                local_noise_median=_required_float(raw_row, "noise_median", "raw source summary"),
                support_frames_ge3=support_frames,
                flux_snr_frames_ge5=_required_float(
                    raw_row, "frames_flux_snr_ge5", "raw source summary"
                ),
                negative_extreme_frames=negative_frames,
                repeated_code_frames=repeated_frames,
                expected_frame_count=expected_frames,
                temporal_frame_count=frame_count,
                temporal_coverage_fraction=_required_float(
                    temporal_row, "coverage_fraction", "temporal source table"
                ),
                temporal_flux_snr_relative_mad=_required_float(
                    temporal_row, "flux_snr_relative_mad", "temporal source table"
                ),
                temporal_positive_fraction=_required_float(
                    temporal_row, "flux_snr_positive_fraction", "temporal source table"
                ),
                temporal_flux_snr_ge5_fraction=_required_float(
                    temporal_row, "flux_snr_ge5_fraction", "temporal source table"
                ),
                temporal_sign_flip_count=int(
                    _required_float(
                        temporal_row, "flux_snr_sign_flip_count", "temporal source table"
                    )
                ),
                raw_peak_status=raw_status,
                value_domain_status=domain_status,
                support_status=support_status,
                evidence_pattern=pattern,
            )
        )

    if not source_result:
        raise ValueError("selected sources have no requested feature class")
    source_rows = tuple(source_result)
    by_class: dict[str, list[FeatureCrossAxisSource]] = {}
    class_labels: dict[str, str] = {}
    for row in source_rows:
        by_class.setdefault(row.feature_class, []).append(row)
        class_labels.setdefault(row.feature_class, row.feature_class_label)
    class_summaries = tuple(
        _make_class_summary(name, class_labels[name], by_class[name])
        for name in requested_classes
        if name in by_class
    )
    pattern_rows: list[FeatureCrossAxisPatternSummary] = []
    for feature_class in requested_classes:
        rows = by_class.get(feature_class, [])
        counts = Counter(row.evidence_pattern for row in rows)
        for pattern, count in sorted(counts.items()):
            pattern_rows.append(
                FeatureCrossAxisPatternSummary(
                    feature_class=feature_class,
                    feature_class_label=class_labels[feature_class],
                    evidence_pattern=pattern,
                    source_count=count,
                    source_fraction=float(count / len(rows)),
                )
            )
    return FeatureCrossAxisAuditResult(
        catalog_path=str(catalog_file),
        raw_summary_path=str(raw_file),
        peak_consistency_path=str(peak_file),
        temporal_sources_path=str(temporal_file),
        selected_detection_ids=tuple(row.detection_id for row in source_rows),
        feature_classes=requested_classes,
        source_rows=source_rows,
        class_summaries=class_summaries,
        pattern_summaries=tuple(pattern_rows),
    )


def _write_rows(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8-sig")
        return
    first = rows[0]
    if not hasattr(first, "as_dict"):
        raise TypeError("audit row has no as_dict method")
    fieldnames = list(first.as_dict().keys())  # type: ignore[union-attr]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())  # type: ignore[union-attr]


def write_feature_cross_axis_artifacts(
    result: FeatureCrossAxisAuditResult,
    out_dir: str | Path,
) -> Path:
    """写出逐源、类别、模式摘要和 JSON。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_cross_axis_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_rows(output / "feature_cross_axis_sources.csv", result.source_rows)
    _write_rows(output / "feature_cross_axis_summary.csv", result.class_summaries)
    _write_rows(output / "feature_cross_axis_patterns.csv", result.pattern_summaries)
    return output


__all__ = [
    "FeatureCrossAxisAuditResult",
    "FeatureCrossAxisClassSummary",
    "FeatureCrossAxisPatternSummary",
    "FeatureCrossAxisSource",
    "run_feature_cross_axis_audit",
    "write_feature_cross_axis_artifacts",
]
