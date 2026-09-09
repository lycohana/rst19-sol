"""各特征类别的原始局部证据审计。

该模块把已有的 ``source_catalog.csv`` 与原始 FITS 抽样产出的
``stratified_raw_source_summary.csv`` 按 ``detection_id`` 连接，汇总核心能量、
局部噪声、零值/负值/重复码和 15 帧支持等字段。它不重新读取 FITS，也不把
``feature_class`` 当作物理类别；用途是检查“类别旗标之外，原始局部证据是否
呈现同方向的机制差异”。

抽样表中的部分字段（例如 flux SNR、PSF 支持、范围异常）参与了质量或类别
规则，因此结果会显式保留 ``definition_overlap``，不能当作独立分类器性能、
precision、FDR、真星概率或物理真值。
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


RAW_EVIDENCE_METRICS: tuple[str, ...] = (
    "core_fraction_median",
    "noise_median",
    "zero_count_max",
    "frames_zero_count_gt0",
    "frames_negative_extreme_count_gt0",
    "frames_repeated_code_count_gt0",
    "frames_peak_snr_ge5",
    "frames_flux_snr_ge5",
    "frames_support_ge3",
)

_METRIC_AXIS: dict[str, tuple[str, str]] = {
    "core_fraction_median": (
        "raw_profile",
        "局部核心能量占比；不直接等同于 PSF 拟合或恒星身份。",
    ),
    "noise_median": (
        "local_noise",
        "局部背景环噪声；用于判断相同峰值在不同背景中的可比性。",
    ),
    "zero_count_max": (
        "validity",
        "单源孔径内零值计数最大值；是原始有效性线索，不是坏点真值。",
    ),
    "frames_zero_count_gt0": (
        "validity_temporal",
        "15 帧中出现零值的帧数；描述值域/孔径状态的时间重复。",
    ),
    "frames_negative_extreme_count_gt0": (
        "signed_value",
        "15 帧中出现极端负码的帧数；需结合编码说明解释。",
    ),
    "frames_repeated_code_count_gt0": (
        "value_pattern",
        "15 帧中出现重复码的帧数；是工程模式线索，不是坏点证明。",
    ),
    "frames_peak_snr_ge5": (
        "temporal_peak",
        "15 帧峰值 SNR 达到当前工程线的帧数；不是恒星概率。",
    ),
    "frames_flux_snr_ge5": (
        "temporal_flux",
        "15 帧通量 SNR 达到当前工程线的帧数；受孔径和背景估计影响。",
    ),
    "frames_support_ge3": (
        "temporal_support",
        "15 帧满足 3×3 支持下限的帧数；是形态支持线索，不是独立真值。",
    ),
}

_METRIC_OVERLAP: dict[str, str] = {
    "core_fraction_median": "间接相关：与形态/支持有关，但未直接作为首要类别旗标。",
    "noise_median": "低重叠：局部噪声未直接定义首要类别。",
    "zero_count_max": "低重叠：零值是值域/有效孔径线索，未直接定义首要类别。",
    "frames_zero_count_gt0": "低重叠：用于补充值域时间行为，未直接定义首要类别。",
    "frames_negative_extreme_count_gt0": "高重叠：与 NEGATIVE_OVERFLOW/范围审计共享值域信息。",
    "frames_repeated_code_count_gt0": "部分重叠：CODE_PATTERN 使用重复码与范围异常的组合。",
    "frames_peak_snr_ge5": "部分重叠：峰值 SNR 与候选显著性相关，但不是唯一类别条件。",
    "frames_flux_snr_ge5": "高重叠：LOW_FLUX_SNR 直接使用通量 SNR。",
    "frames_support_ge3": "高重叠：INSUFFICIENT_PSF_SUPPORT 与支持像素直接相关。",
}


@dataclass(frozen=True, slots=True)
class RawEvidenceSummary:
    """一个首要特征类别在某一原始证据轴上的抽样分布。"""

    feature_class: str
    feature_class_label: str
    sample_count: int
    control_sample_count: int
    diagnostic_sample_count: int
    metric: str
    evidence_axis: str
    metric_note: str
    definition_overlap: str
    finite_value_count: int
    p10: float
    median: float
    p90: float
    minimum: float
    maximum: float
    nonzero_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RawEvidenceAuditResult:
    """原始局部证据审计的机器可读结果。"""

    catalog_path: str
    raw_summary_path: str
    feature_classes: tuple[str, ...]
    metrics: tuple[str, ...]
    sample_counts: dict[str, int]
    origin_counts: dict[str, dict[str, int]]
    rows: tuple[RawEvidenceSummary, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "raw_summary_path": self.raw_summary_path,
            "feature_classes": list(self.feature_classes),
            "metrics": list(self.metrics),
            "sample_counts": dict(self.sample_counts),
            "origin_counts": {key: dict(value) for key, value in self.origin_counts.items()},
            "rows": [row.as_dict() for row in self.rows],
            "interpretation_guardrails": [
                "raw summary 是有限抽样，不是全图 precision、完备率或真星率。",
                "feature_class 是检测器旗标优先级产生的互斥审计标签，不是物理类别。",
                "definition_overlap 显式标记了与类别/质量规则共享的字段，不能当作独立验证。",
                "core_fraction 和局部噪声可补充类别解释，但也不能单独区分恒星、噪声和伪影。",
            ],
        }


def _finite_float(row: Mapping[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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


def _read_raw_summary(
    path_value: str | Path,
    catalog: Mapping[int, tuple[str, str]],
    metrics: Sequence[str],
) -> tuple[Path, list[tuple[int, str, dict[str, str]]]]:
    path = Path(path_value)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            required = {"detection_id", "origin", *metrics}
            missing = sorted(required - columns)
            if missing:
                raise ValueError(f"raw source summary is missing required columns: {', '.join(missing)}")
            records: list[tuple[int, str, dict[str, str]]] = []
            seen: set[int] = set()
            for row in reader:
                detection_id = _required_id(row, "detection_id")
                if detection_id in seen:
                    raise ValueError(f"raw source summary contains duplicate detection_id={detection_id}")
                seen.add(detection_id)
                if detection_id not in catalog:
                    raise ValueError(f"raw source summary detection_id={detection_id} is absent from source catalog")
                origin = str(row.get("origin", "")).strip()
                if not origin:
                    raise ValueError(f"raw source summary has empty origin for detection_id={detection_id}")
                records.append((detection_id, origin, row))
    except OSError as exc:
        raise OSError(f"无法读取原始抽样摘要：{path}") from exc
    if not records:
        raise ValueError(f"raw source summary is empty: {path}")
    return path, records


def _summary(values: Sequence[float]) -> tuple[int, float, float, float, float, float, float]:
    if not values:
        raise ValueError("cannot summarize an empty raw evidence metric")
    array = np.asarray(values, dtype=np.float64)
    return (
        len(values),
        float(np.quantile(array, 0.10)),
        float(np.median(array)),
        float(np.quantile(array, 0.90)),
        float(np.min(array)),
        float(np.max(array)),
        float(np.count_nonzero(array > 0) / len(array)),
    )


def run_feature_raw_evidence_audit(
    catalog_path: str | Path,
    raw_summary_path: str | Path,
    *,
    feature_classes: Iterable[str] | None = None,
    metrics: Sequence[str] = RAW_EVIDENCE_METRICS,
) -> RawEvidenceAuditResult:
    """按首要类别汇总已有原始 FITS 抽样摘要的局部证据。"""

    requested_metrics = tuple(dict.fromkeys(str(metric).strip() for metric in metrics if str(metric).strip()))
    if not requested_metrics:
        raise ValueError("metrics 不能为空")
    unknown_metrics = [metric for metric in requested_metrics if metric not in RAW_EVIDENCE_METRICS]
    if unknown_metrics:
        raise ValueError(f"unknown raw evidence metric: {unknown_metrics[0]!r}")

    catalog_file, catalog = _read_catalog(catalog_path)
    raw_file, raw_records = _read_raw_summary(raw_summary_path, catalog, requested_metrics)
    sampled_classes = {
        catalog[detection_id][0]
        for detection_id, _origin, _row in raw_records
        if catalog[detection_id][0] != "other_rejected"
    }
    if feature_classes is None:
        requested_classes = tuple(sorted(sampled_classes))
    else:
        requested_classes = tuple(dict.fromkeys(str(name).strip() for name in feature_classes if str(name).strip()))
    if not requested_classes:
        raise ValueError("feature_classes 不能为空，且抽样表中没有可用类别")
    catalog_classes = {feature_class for feature_class, _label in catalog.values()}
    unknown_classes = [name for name in requested_classes if name not in catalog_classes]
    if unknown_classes:
        raise ValueError(f"source catalog has no feature_class={unknown_classes[0]!r}")
    missing_sample_classes = [name for name in requested_classes if name not in sampled_classes]
    if missing_sample_classes:
        raise ValueError(f"raw source summary has no sampled rows for feature_class={missing_sample_classes[0]!r}")

    values_by_class_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_counts: Counter[str] = Counter()
    origin_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for detection_id, origin, row in raw_records:
        feature_class, _label = catalog[detection_id]
        if feature_class not in requested_classes:
            continue
        sample_counts[feature_class] += 1
        origin_counts[feature_class][origin] += 1
        for metric in requested_metrics:
            value = _finite_float(row, metric)
            if value is not None:
                values_by_class_metric[(feature_class, metric)].append(value)

    rows: list[RawEvidenceSummary] = []
    for feature_class in requested_classes:
        labels = {
            catalog[detection_id][1]
            for detection_id, _origin, _row in raw_records
            if catalog[detection_id][0] == feature_class
        }
        if not labels:
            raise ValueError(f"raw source summary has no label for feature_class={feature_class!r}")
        label = sorted(labels)[0]
        for metric in requested_metrics:
            values = values_by_class_metric.get((feature_class, metric), [])
            if not values:
                continue
            count, p10, median, p90, minimum, maximum, nonzero_fraction = _summary(values)
            axis, note = _METRIC_AXIS[metric]
            rows.append(
                RawEvidenceSummary(
                    feature_class=feature_class,
                    feature_class_label=label,
                    sample_count=sample_counts[feature_class],
                    control_sample_count=origin_counts[feature_class].get("control", 0),
                    diagnostic_sample_count=origin_counts[feature_class].get("diagnostic", 0),
                    metric=metric,
                    evidence_axis=axis,
                    metric_note=note,
                    definition_overlap=_METRIC_OVERLAP[metric],
                    finite_value_count=count,
                    p10=p10,
                    median=median,
                    p90=p90,
                    minimum=minimum,
                    maximum=maximum,
                    nonzero_fraction=nonzero_fraction,
                )
            )

    if not rows:
        raise ValueError("raw source summary has no finite requested metrics")
    return RawEvidenceAuditResult(
        catalog_path=str(catalog_file),
        raw_summary_path=str(raw_file),
        feature_classes=requested_classes,
        metrics=requested_metrics,
        sample_counts={name: sample_counts[name] for name in requested_classes},
        origin_counts={
            name: dict(origin_counts[name])
            for name in requested_classes
        },
        rows=tuple(rows),
    )


def write_feature_raw_evidence_artifacts(
    result: RawEvidenceAuditResult,
    out_dir: str | Path,
) -> Path:
    """写出原始局部证据 CSV/JSON 产物。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_raw_evidence_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(RawEvidenceSummary.__dataclass_fields__)
    with (output / "feature_raw_evidence_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row.as_dict())
    return output


__all__ = [
    "RAW_EVIDENCE_METRICS",
    "RawEvidenceAuditResult",
    "RawEvidenceSummary",
    "run_feature_raw_evidence_audit",
    "write_feature_raw_evidence_artifacts",
]
