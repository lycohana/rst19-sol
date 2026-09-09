"""近邻候选的跨帧强制测光共变审计。

这个模块只读取 ``forced_stability_frame_metrics.csv``，把一对候选在同一
组 15 帧中的强制测光 SNR 放在一起比较。它不重新检测、不修改质量层、GUI
或缓存；相关性和控制分位数只是共享响应/孔径混叠的诊断，不是双星概率。
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PAIR_FLUX_METRICS = ("flux_snr", "local_peak_flux_snr")


@dataclass(frozen=True, slots=True)
class ForcedStabilityFluxRow:
    """强制测光表中一个候选的一帧数值。"""

    detection_id: int
    frame_index: int
    flux_snr: float | None
    local_peak_flux_snr: float | None


@dataclass(frozen=True, slots=True)
class PairFluxCovarianceMetric:
    """一类强制测光量的 pair 共变和控制分布。"""

    metric_name: str
    valid_frame_count: int
    primary_pearson: float | None
    primary_spearman: float | None
    frame_median_normalized_pearson: float | None
    frame_median_normalized_spearman: float | None
    below_pair_total_median_frame_count: int
    above_pair_total_median_frame_count: int
    below_pair_total_median_pearson: float | None
    above_pair_total_median_pearson: float | None
    control_source_count: int
    control_pair_count: int
    control_pearson_p05: float | None
    control_pearson_median: float | None
    control_pearson_p95: float | None
    control_normalized_pearson_p05: float | None
    control_normalized_pearson_median: float | None
    control_normalized_pearson_p95: float | None
    raw_upper_tail_fraction: float | None
    normalized_upper_tail_fraction: float | None
    primary_fraction_median: float | None
    primary_fraction_min: float | None
    primary_fraction_max: float | None
    primary_fraction_range: float | None
    primary_fraction_mad_scaled: float | None
    below_pair_total_median_primary_fraction_median: float | None
    above_pair_total_median_primary_fraction_median: float | None
    primary_fraction_shift_high_minus_low: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "valid_frame_count": self.valid_frame_count,
            "primary_pearson": self.primary_pearson,
            "primary_spearman": self.primary_spearman,
            "frame_median_normalized_pearson": self.frame_median_normalized_pearson,
            "frame_median_normalized_spearman": self.frame_median_normalized_spearman,
            "below_pair_total_median_frame_count": self.below_pair_total_median_frame_count,
            "above_pair_total_median_frame_count": self.above_pair_total_median_frame_count,
            "below_pair_total_median_pearson": self.below_pair_total_median_pearson,
            "above_pair_total_median_pearson": self.above_pair_total_median_pearson,
            "control_source_count": self.control_source_count,
            "control_pair_count": self.control_pair_count,
            "control_pearson_p05": self.control_pearson_p05,
            "control_pearson_median": self.control_pearson_median,
            "control_pearson_p95": self.control_pearson_p95,
            "control_normalized_pearson_p05": self.control_normalized_pearson_p05,
            "control_normalized_pearson_median": self.control_normalized_pearson_median,
            "control_normalized_pearson_p95": self.control_normalized_pearson_p95,
            "raw_upper_tail_fraction": self.raw_upper_tail_fraction,
            "normalized_upper_tail_fraction": self.normalized_upper_tail_fraction,
            "primary_fraction_median": self.primary_fraction_median,
            "primary_fraction_min": self.primary_fraction_min,
            "primary_fraction_max": self.primary_fraction_max,
            "primary_fraction_range": self.primary_fraction_range,
            "primary_fraction_mad_scaled": self.primary_fraction_mad_scaled,
            "below_pair_total_median_primary_fraction_median": (
                self.below_pair_total_median_primary_fraction_median
            ),
            "above_pair_total_median_primary_fraction_median": (
                self.above_pair_total_median_primary_fraction_median
            ),
            "primary_fraction_shift_high_minus_low": self.primary_fraction_shift_high_minus_low,
        }


@dataclass(frozen=True, slots=True)
class PairFluxCovarianceAuditResult:
    """一对候选的跨帧强制测光共变审计结果。"""

    frame_count: int
    primary_detection_id: int
    secondary_detection_id: int
    control_source_count: int
    metrics: tuple[PairFluxCovarianceMetric, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "control_source_count": self.control_source_count,
            "metrics": [metric.as_dict() for metric in self.metrics],
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        }


def _optional_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    if value is None or not str(value).strip():
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"forced stability field {name!r} is not numeric: {value!r}") from exc
    return converted if math.isfinite(converted) else None


def load_forced_stability_flux_csv(path: str | Path) -> tuple[ForcedStabilityFluxRow, ...]:
    """读取强制测光表中的固定位置和局部位置通量 SNR。"""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"detection_id", "frame_index", *PAIR_FLUX_METRICS}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"forced stability CSV is missing required columns: {', '.join(missing)}")
        rows: list[ForcedStabilityFluxRow] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                detection_id = int(float(row["detection_id"]))
                frame_index = int(float(row["frame_index"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid forced stability row {line_number}: id/frame is invalid") from exc
            rows.append(
                ForcedStabilityFluxRow(
                    detection_id=detection_id,
                    frame_index=frame_index,
                    flux_snr=_optional_float(row, "flux_snr"),
                    local_peak_flux_snr=_optional_float(row, "local_peak_flux_snr"),
                )
            )
    return tuple(rows)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x -= float(np.mean(x))
    y -= float(np.mean(y))
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator <= np.finfo(np.float64).eps:
        return None
    value = float(np.sum(x * y) / denominator)
    return value if math.isfinite(value) else None


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[int(order[end + 1])] == values[int(order[start])]:
            end += 1
        rank = 0.5 * (start + end) + 1.0
        ranks[order[start : end + 1]] = rank
        start = end + 1
    return [float(value) for value in ranks]


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    return _pearson(_average_ranks(left), _average_ranks(right))


def _finite_pair(
    left: Sequence[float | None], right: Sequence[float | None]
) -> tuple[list[float], list[float]]:
    selected = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    return [item[0] for item in selected], [item[1] for item in selected]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _upper_tail(target: float | None, controls: Sequence[float]) -> float | None:
    if target is None or not controls:
        return None
    count = sum(value >= target for value in controls)
    return float((1 + count) / (1 + len(controls)))


def _scaled_mad(values: Sequence[float]) -> float | None:
    """返回 1.4826×MAD；只作小样本的稳健离散度描述。"""

    if not values:
        return None
    median = float(np.median(np.asarray(values, dtype=np.float64)))
    deviations = np.abs(np.asarray(values, dtype=np.float64) - median)
    return float(1.4826 * np.median(deviations))


def _metric_values(row: ForcedStabilityFluxRow, metric_name: str) -> float | None:
    if metric_name not in PAIR_FLUX_METRICS:
        raise ValueError(f"unsupported pair flux metric: {metric_name!r}")
    return getattr(row, metric_name)


def run_pair_flux_covariance_audit(
    rows: Iterable[ForcedStabilityFluxRow],
    primary_detection_id: int,
    secondary_detection_id: int,
    *,
    metrics: Sequence[str] = PAIR_FLUX_METRICS,
) -> PairFluxCovarianceAuditResult:
    """比较 pair 的跨帧通量 SNR，并与同表其他源的两两相关控制分布比较。

    ``frame_median_normalized_*`` 先以每帧其它完整源的中位数除掉一个粗略的
    frame scale，再计算相关性；这是控制共同帧尺度的敏感性分析，不是光度
    标定。控制 pair 使用同一张表中除目标外的所有完整源组合，因此输出的
    upper-tail 只是探索性参考，不能当作正式 p 值。
    """

    primary_id = int(primary_detection_id)
    secondary_id = int(secondary_detection_id)
    if primary_id == secondary_id:
        raise ValueError("primary and secondary detection IDs must differ")
    metric_names = tuple(dict.fromkeys(str(name) for name in metrics))
    if not metric_names:
        raise ValueError("metrics cannot be empty")
    for name in metric_names:
        if name not in PAIR_FLUX_METRICS:
            raise ValueError(f"unsupported pair flux metric: {name!r}")

    by_id: dict[int, dict[int, ForcedStabilityFluxRow]] = {}
    for row in rows:
        item = by_id.setdefault(int(row.detection_id), {})
        frame_index = int(row.frame_index)
        if frame_index in item:
            raise ValueError(f"duplicate forced stability row for id={row.detection_id}, frame={frame_index}")
        item[frame_index] = row
    if primary_id not in by_id or secondary_id not in by_id:
        raise ValueError("both target detection IDs must be present in forced stability rows")
    frame_indices = sorted(set(by_id[primary_id]) & set(by_id[secondary_id]))
    if len(frame_indices) < 3:
        raise ValueError("target pair needs at least three common frames")

    control_ids = sorted(
        source_id
        for source_id, frame_rows in by_id.items()
        if source_id not in {primary_id, secondary_id}
        and all(frame_index in frame_rows for frame_index in frame_indices)
    )
    metrics_out: list[PairFluxCovarianceMetric] = []
    for metric_name in metric_names:
        primary_values = [_metric_values(by_id[primary_id][frame], metric_name) for frame in frame_indices]
        secondary_values = [_metric_values(by_id[secondary_id][frame], metric_name) for frame in frame_indices]
        primary_series, secondary_series = _finite_pair(primary_values, secondary_values)
        if len(primary_series) < 3:
            metrics_out.append(
                PairFluxCovarianceMetric(
                    metric_name=metric_name,
                    valid_frame_count=len(primary_series),
                    primary_pearson=None,
                    primary_spearman=None,
                    frame_median_normalized_pearson=None,
                    frame_median_normalized_spearman=None,
                    below_pair_total_median_frame_count=0,
                    above_pair_total_median_frame_count=0,
                    below_pair_total_median_pearson=None,
                    above_pair_total_median_pearson=None,
                    control_source_count=0,
                    control_pair_count=0,
                    control_pearson_p05=None,
                    control_pearson_median=None,
                    control_pearson_p95=None,
                    control_normalized_pearson_p05=None,
                    control_normalized_pearson_median=None,
                    control_normalized_pearson_p95=None,
                    raw_upper_tail_fraction=None,
                    normalized_upper_tail_fraction=None,
                    primary_fraction_median=None,
                    primary_fraction_min=None,
                    primary_fraction_max=None,
                    primary_fraction_range=None,
                    primary_fraction_mad_scaled=None,
                    below_pair_total_median_primary_fraction_median=None,
                    above_pair_total_median_primary_fraction_median=None,
                    primary_fraction_shift_high_minus_low=None,
                )
            )
            continue

        frame_medians: dict[int, float] = {}
        normalized_control_ids: list[int] = []
        for source_id in control_ids:
            values = [
                _metric_values(by_id[source_id][frame], metric_name)
                for frame in frame_indices
            ]
            if all(value is not None and math.isfinite(float(value)) for value in values):
                normalized_control_ids.append(source_id)
        for frame in frame_indices:
            values = [
                float(_metric_values(by_id[source_id][frame], metric_name))
                for source_id in normalized_control_ids
                if _metric_values(by_id[source_id][frame], metric_name) is not None
                and math.isfinite(float(_metric_values(by_id[source_id][frame], metric_name)))
            ]
            if values:
                frame_medians[frame] = float(np.median(np.asarray(values, dtype=np.float64)))

        normalized_primary: list[float] = []
        normalized_secondary: list[float] = []
        raw_primary: list[float] = []
        raw_secondary: list[float] = []
        pair_totals: list[float] = []
        split_fractions: list[float] = []
        pair_fractions: list[float | None] = []
        for frame, primary_value, secondary_value in zip(frame_indices, primary_values, secondary_values):
            if primary_value is None or secondary_value is None:
                continue
            primary_float = float(primary_value)
            secondary_float = float(secondary_value)
            if not math.isfinite(primary_float) or not math.isfinite(secondary_float):
                continue
            raw_primary.append(primary_float)
            raw_secondary.append(secondary_float)
            total = primary_float + secondary_float
            pair_totals.append(total)
            if abs(total) > np.finfo(np.float64).eps:
                fraction = primary_float / total
                split_fractions.append(fraction)
                pair_fractions.append(fraction)
            else:
                pair_fractions.append(None)
            scale = frame_medians.get(frame)
            if scale is not None and abs(scale) > np.finfo(np.float64).eps:
                normalized_primary.append(primary_float / scale)
                normalized_secondary.append(secondary_float / scale)

        raw_controls: list[float] = []
        normalized_controls: list[float] = []
        for left_id, right_id in itertools.combinations(normalized_control_ids, 2):
            left = [
                _metric_values(by_id[left_id][frame], metric_name)
                for frame in frame_indices
            ]
            right = [
                _metric_values(by_id[right_id][frame], metric_name)
                for frame in frame_indices
            ]
            left_series, right_series = _finite_pair(left, right)
            raw_corr = _pearson(left_series, right_series)
            if raw_corr is not None:
                raw_controls.append(raw_corr)
            if len(frame_medians) == len(frame_indices):
                left_normalized = [
                    float(value) / frame_medians[frame]
                    for value, frame in zip(left_series, frame_indices)
                ]
                right_normalized = [
                    float(value) / frame_medians[frame]
                    for value, frame in zip(right_series, frame_indices)
                ]
                normalized_corr = _pearson(left_normalized, right_normalized)
                if normalized_corr is not None:
                    normalized_controls.append(normalized_corr)

        primary_pearson = _pearson(raw_primary, raw_secondary)
        normalized_pearson = _pearson(normalized_primary, normalized_secondary)
        total_median = float(np.median(np.asarray(pair_totals, dtype=np.float64)))
        below_indices = [index for index, total in enumerate(pair_totals) if total < total_median]
        above_indices = [index for index, total in enumerate(pair_totals) if total >= total_median]
        below_pearson = _pearson(
            [raw_primary[index] for index in below_indices],
            [raw_secondary[index] for index in below_indices],
        )
        above_pearson = _pearson(
            [raw_primary[index] for index in above_indices],
            [raw_secondary[index] for index in above_indices],
        )
        below_fractions = [
            float(pair_fractions[index])
            for index in below_indices
            if pair_fractions[index] is not None
        ]
        above_fractions = [
            float(pair_fractions[index])
            for index in above_indices
            if pair_fractions[index] is not None
        ]
        primary_fraction_median = _percentile(split_fractions, 50.0)
        primary_fraction_min = _percentile(split_fractions, 0.0)
        primary_fraction_max = _percentile(split_fractions, 100.0)
        primary_fraction_range = (
            primary_fraction_max - primary_fraction_min
            if primary_fraction_min is not None and primary_fraction_max is not None
            else None
        )
        below_fraction_median = _percentile(below_fractions, 50.0)
        above_fraction_median = _percentile(above_fractions, 50.0)
        primary_fraction_shift = (
            above_fraction_median - below_fraction_median
            if below_fraction_median is not None and above_fraction_median is not None
            else None
        )
        metrics_out.append(
            PairFluxCovarianceMetric(
                metric_name=metric_name,
                valid_frame_count=len(raw_primary),
                primary_pearson=primary_pearson,
                primary_spearman=_spearman(raw_primary, raw_secondary),
                frame_median_normalized_pearson=normalized_pearson,
                frame_median_normalized_spearman=_spearman(normalized_primary, normalized_secondary),
                below_pair_total_median_frame_count=len(below_indices),
                above_pair_total_median_frame_count=len(above_indices),
                below_pair_total_median_pearson=below_pearson,
                above_pair_total_median_pearson=above_pearson,
                control_source_count=len(normalized_control_ids),
                control_pair_count=len(raw_controls),
                control_pearson_p05=_percentile(raw_controls, 5.0),
                control_pearson_median=_percentile(raw_controls, 50.0),
                control_pearson_p95=_percentile(raw_controls, 95.0),
                control_normalized_pearson_p05=_percentile(normalized_controls, 5.0),
                control_normalized_pearson_median=_percentile(normalized_controls, 50.0),
                control_normalized_pearson_p95=_percentile(normalized_controls, 95.0),
                raw_upper_tail_fraction=_upper_tail(primary_pearson, raw_controls),
                normalized_upper_tail_fraction=_upper_tail(normalized_pearson, normalized_controls),
                primary_fraction_median=primary_fraction_median,
                primary_fraction_min=primary_fraction_min,
                primary_fraction_max=primary_fraction_max,
                primary_fraction_range=primary_fraction_range,
                primary_fraction_mad_scaled=_scaled_mad(split_fractions),
                below_pair_total_median_primary_fraction_median=below_fraction_median,
                above_pair_total_median_primary_fraction_median=above_fraction_median,
                primary_fraction_shift_high_minus_low=primary_fraction_shift,
            )
        )

    descriptions = []
    for metric in metrics_out:
        descriptions.append(
            f"{metric.metric_name}: Pearson={metric.primary_pearson}, "
            f"按帧中位数归一后={metric.frame_median_normalized_pearson}, "
            f"控制上尾={metric.normalized_upper_tail_fraction}"
        )
    conclusion = (
        f"pair 强制测光共变审计完成：{len(frame_indices)} 帧，控制源 {len(control_ids)} 个；"
        + "；".join(descriptions)
        + "。共变只能说明局部响应可能共享结构、孔径或值域状态，不能单独区分混合星、异常码和真实近邻，"
        "也不是双星概率；"
        "正式结论仍需原始值域、PSF、位置协方差、WCS/星表唯一性或已知真值注入共同支持。"
    )
    return PairFluxCovarianceAuditResult(
        frame_count=len(frame_indices),
        primary_detection_id=primary_id,
        secondary_detection_id=secondary_id,
        control_source_count=len(control_ids),
        metrics=tuple(metrics_out),
        parameters={
            "metric_names": list(metric_names),
            "frame_indices": frame_indices,
            "normalization": "divide by the same-frame median across other complete sources",
            "control_pair_scope": "all complete source pairs excluding the two targets",
            "upper_tail_is_not_p_value": True,
            "interpretation_boundary": "diagnostic of shared response, not binary-star probability",
            "primary_fraction_definition": "primary metric / (primary metric + secondary metric) per common frame",
            "primary_fraction_mad_scale": 1.4826,
            "pair_total_split": "below < median and above >= median of the target pair total",
        },
        conclusion=conclusion,
    )


def write_pair_flux_covariance_artifacts(
    result: PairFluxCovarianceAuditResult,
    out_dir: str | Path,
) -> Path:
    """写 pair 共变审计的 CSV 和 JSON。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(result.metrics[0].as_dict()) if result.metrics else ["metric_name"]
    with (output / "pair_flux_covariance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric.as_dict() for metric in result.metrics)
    payload = result.as_dict()
    payload["note"] = (
        "这里的量是同一强制测光流程产生的 flux_snr/local_peak_flux_snr；"
        "控制分布用于发现异常共变，不是正式显著性检验，也不是双星概率。"
    )
    (output / "pair_flux_covariance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output
