"""15 帧分层强制测光稳定性审计。

这个研究工具消费已经生成的首帧 ``source_catalog.csv`` 和
``sequence_feature_persistence.json``，把首帧峰位置沿累计平移投回每一帧
未降噪 FITS，输出逐源逐帧的孔径通量、PSF 支持和形状诊断。它故意不重跑
候选提案、不修改 ``quality_passed``、不把结果写入 rst19 缓存，也不把
“质量样式”计数解释成物理恒星身份。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .fits import auxiliary_mask, read_fits
from .sequence import _stack_forced_frame_measure_batch


FEATURE_CLASS_ORDER: tuple[tuple[str, str], ...] = (
    ("range_anomaly", "数据有效性/范围异常邻域"),
    ("linear_artifact", "线状/拖影候选"),
    ("masked_or_edge", "边缘/掩膜候选"),
    ("crowded_blend", "拥挤/未分辨近邻"),
    ("spike_or_support", "尖峰/PSF 支持不足"),
    ("weak_or_background", "弱通量/背景不确定"),
    ("shape_outlier", "形状异常"),
    ("compact_quality", "质量通过的紧凑候选"),
    ("other_rejected", "其他拒绝候选"),
)


@dataclass(frozen=True, slots=True)
class ForcedAnchor:
    """从首帧源表选出的一个强制测光锚点。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    x: float
    y: float
    flux_snr: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "x": self.x,
            "y": self.y,
            "flux_snr": self.flux_snr,
        }


@dataclass(frozen=True, slots=True)
class ForcedFrameMetric:
    """一个锚点在一个原始帧上的强制测光结果。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    frame_index: int
    predicted_x: float
    predicted_y: float
    flux_snr: float | None
    support_3x3: int | None
    center_valid: bool
    fwhm: float | None
    ellipticity: float | None
    quality_like: bool
    local_peak_x: float | None = None
    local_peak_y: float | None = None
    local_peak_flux_snr: float | None = None
    local_peak_support_3x3: int | None = None
    local_peak_center_valid: bool | None = None
    local_peak_fwhm: float | None = None
    local_peak_ellipticity: float | None = None
    local_peak_quality_like: bool | None = None
    matched_response_snr: float | None = None
    local_peak_response_snr: float | None = None
    relocalization_dx_px: float | None = None
    relocalization_dy_px: float | None = None
    relocalization_distance_px: float | None = None
    relocalized: bool | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "frame_index": self.frame_index,
            "predicted_x": self.predicted_x,
            "predicted_y": self.predicted_y,
            "flux_snr": self.flux_snr,
            "support_3x3": self.support_3x3,
            "center_valid": self.center_valid,
            "fwhm": self.fwhm,
            "ellipticity": self.ellipticity,
            "quality_like": self.quality_like,
            "local_peak_x": self.local_peak_x,
            "local_peak_y": self.local_peak_y,
            "local_peak_flux_snr": self.local_peak_flux_snr,
            "local_peak_support_3x3": self.local_peak_support_3x3,
            "local_peak_center_valid": self.local_peak_center_valid,
            "local_peak_fwhm": self.local_peak_fwhm,
            "local_peak_ellipticity": self.local_peak_ellipticity,
            "local_peak_quality_like": self.local_peak_quality_like,
            "matched_response_snr": self.matched_response_snr,
            "local_peak_response_snr": self.local_peak_response_snr,
            "relocalization_dx_px": self.relocalization_dx_px,
            "relocalization_dy_px": self.relocalization_dy_px,
            "relocalization_distance_px": self.relocalization_distance_px,
            "relocalized": self.relocalized,
        }


@dataclass(frozen=True, slots=True)
class ForcedFrameSpec:
    """序列审计 JSON 中的一帧及其背景/配准信息。"""

    frame_index: int
    path: Path
    background_adu: float
    noise_adu: float
    shift_x: float
    shift_y: float


@dataclass(frozen=True, slots=True)
class ForcedStabilityResult:
    """强制测光审计的全部可序列化结果。"""

    source_catalog: Path
    persistence_json: Path
    anchor_coordinate: str
    included_detection_ids: tuple[int, ...]
    frame_count: int
    anchors: tuple[ForcedAnchor, ...]
    frame_metrics: tuple[ForcedFrameMetric, ...]
    summary_rows: tuple[dict[str, object], ...]
    aperture_radius: int
    min_psf_support_pixels: int
    min_flux_snr: float
    min_fwhm: float
    max_fwhm: float
    max_ellipticity: float
    local_peak_search_radius: int = 0
    local_peak_psf_fwhm: float = 3.0


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _required_float(value: object, field: str, context: str) -> float:
    number = _finite_float(value)
    if number is None:
        raise ValueError(f"{context} has invalid {field}: {value!r}")
    return number


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _normalize_anchor_coordinate(value: str) -> str:
    coordinate = str(value).strip().lower()
    if coordinate not in {"peak", "centroid"}:
        raise ValueError("anchor_coordinate must be 'peak' or 'centroid'")
    return coordinate


def _feature_order(observed: Sequence[str]) -> tuple[str, ...]:
    known = [feature_class for feature_class, _label in FEATURE_CLASS_ORDER]
    extra = sorted({str(value) for value in observed if str(value) not in known})
    return tuple(known) + tuple(extra)


def load_forced_anchors(
    source_catalog: str | Path,
    *,
    max_per_class: int = 32,
    anchor_coordinate: str = "peak",
    include_detection_ids: Sequence[int] = (),
) -> tuple[ForcedAnchor, ...]:
    """读取源表，并按类别做与既有研究表一致的确定性分位点抽样。

    ``peak`` 与当前序列叠加层的提案坐标一致；``centroid`` 用源表的
    测量质心作一个敏感性对照。两种口径都保留在 API 中，避免把一个
    未记录的坐标选择隐藏在论文数字里。``include_detection_ids`` 用于
    把截图或其他指定反例追加到分层样本中，不会挤掉原有分位点样本。
    """

    if max_per_class < 1:
        raise ValueError("max_per_class must be positive")
    coordinate = _normalize_anchor_coordinate(anchor_coordinate)
    included_ids = {int(value) for value in include_detection_ids}
    path = Path(source_catalog)
    if not path.is_file():
        raise ValueError(f"source catalog does not exist: {path}")

    labels = dict(FEATURE_CLASS_ORDER)
    grouped: dict[str, list[ForcedAnchor]] = {}
    observed_ids: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"source catalog has no header: {path}")
        required = {"detection_id", "feature_class", "x", "y"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"source catalog is missing fields: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            feature_class = str(row.get("feature_class", "")).strip()
            if not feature_class:
                raise ValueError(f"source catalog row {row_number} has empty feature_class")
            if coordinate == "peak":
                x = _finite_float(row.get("peak_x"))
                y = _finite_float(row.get("peak_y"))
                if x is None:
                    x = _required_float(row.get("x"), "peak_x/x", f"row {row_number}")
                if y is None:
                    y = _required_float(row.get("y"), "peak_y/y", f"row {row_number}")
            else:
                x = _finite_float(row.get("x"))
                y = _finite_float(row.get("y"))
                if x is None:
                    x = _required_float(row.get("peak_x"), "x/peak_x", f"row {row_number}")
                if y is None:
                    y = _required_float(row.get("peak_y"), "y/peak_y", f"row {row_number}")
            detection_text = str(row.get("detection_id", "")).strip()
            try:
                detection_id = int(detection_text)
            except ValueError as exc:
                raise ValueError(f"row {row_number} has invalid detection_id: {detection_text!r}") from exc
            observed_ids.add(detection_id)
            grouped.setdefault(feature_class, []).append(
                ForcedAnchor(
                    detection_id=detection_id,
                    feature_class=feature_class,
                    feature_class_label=str(row.get("feature_class_label") or labels.get(feature_class, feature_class)),
                    x=x,
                    y=y,
                    flux_snr=_finite_float(row.get("flux_snr")),
                )
            )

    selected: list[ForcedAnchor] = []
    for feature_class in _feature_order(tuple(grouped)):
        members = sorted(
            grouped.get(feature_class, ()),
            key=lambda anchor: (
                float("-inf") if anchor.flux_snr is None else anchor.flux_snr,
                anchor.detection_id,
            ),
        )
        if len(members) > max_per_class:
            indices = np.rint(np.linspace(0, len(members) - 1, max_per_class)).astype(int)
            selected_indices = {int(index) for index in indices}
            members = [
                member
                for index, member in enumerate(members)
                if index in selected_indices or member.detection_id in included_ids
            ]
        selected.extend(members)
    missing_included = sorted(included_ids.difference(observed_ids))
    if missing_included:
        missing_text = ", ".join(str(value) for value in missing_included)
        raise ValueError(f"source catalog does not contain included detection_id(s): {missing_text}")
    return tuple(selected)


def _load_frame_specs(
    persistence_json: str | Path,
    *,
    base_dir: str | Path = ".",
) -> tuple[Path, tuple[ForcedFrameSpec, ...]]:
    path = Path(persistence_json)
    if not path.is_file():
        raise ValueError(f"persistence JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid persistence JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("persistence JSON root must be an object")
    frame_count = int(payload.get("frame_count", 0))
    if frame_count < 1:
        raise ValueError("persistence JSON has invalid frame_count")
    raw_shifts = payload.get("cumulative_shifts")
    if not isinstance(raw_shifts, list) or len(raw_shifts) != frame_count:
        raise ValueError("cumulative_shifts length must match frame_count")
    shifts: list[tuple[float, float]] = []
    for index, value in enumerate(raw_shifts, start=1):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"cumulative_shifts[{index}] must contain x/y")
        shifts.append(
            (
                _required_float(value[0], "shift_x", f"cumulative_shifts[{index}]"),
                _required_float(value[1], "shift_y", f"cumulative_shifts[{index}]"),
            )
        )

    raw_rows = payload.get("frame_rows")
    if not isinstance(raw_rows, list):
        raise ValueError("persistence JSON is missing frame_rows")
    by_index: dict[int, ForcedFrameSpec] = {}
    root = Path(base_dir)
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        try:
            frame_index = int(row["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frame_rows contains an invalid frame_index") from exc
        if frame_index < 1 or frame_index > frame_count:
            raise ValueError(f"frame_rows contains out-of-range frame_index={frame_index}")
        frame_path = _resolve_path(str(row.get("path", "")), root)
        spec = ForcedFrameSpec(
            frame_index=frame_index,
            path=frame_path,
            background_adu=_required_float(row.get("background_adu"), "background_adu", f"frame {frame_index}"),
            noise_adu=max(
                _required_float(row.get("noise_adu"), "noise_adu", f"frame {frame_index}"),
                np.finfo(float).eps,
            ),
            shift_x=shifts[frame_index - 1][0],
            shift_y=shifts[frame_index - 1][1],
        )
        previous = by_index.get(frame_index)
        if previous is not None:
            if (
                previous.path != spec.path
                or not np.isclose(previous.background_adu, spec.background_adu)
                or not np.isclose(previous.noise_adu, spec.noise_adu)
            ):
                raise ValueError(f"frame_rows has inconsistent duplicate frame {frame_index}")
        else:
            by_index[frame_index] = spec
    missing = [str(index) for index in range(1, frame_count + 1) if index not in by_index]
    if missing:
        raise ValueError(f"frame_rows is missing frame(s): {', '.join(missing)}")
    return path, tuple(by_index[index] for index in range(1, frame_count + 1))


def _median(values: Sequence[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.median(finite)) if finite.size else None


def _robust_cv(values: Sequence[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return None
    center = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - center)))
    denominator = abs(center)
    return scale / denominator if denominator > np.finfo(float).eps else None


def _local_search_offsets(radius: int) -> np.ndarray:
    """返回局部重定位的整数偏移，中心点排在第一位。"""

    radius = int(radius)
    if radius not in (0, 1):
        raise ValueError("local_peak_search_radius must be 0 or 1 px")
    offsets: list[tuple[int, int]] = [(0, 0)]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (dx, dy) != (0, 0):
                offsets.append((dx, dy))
    return np.asarray(offsets, dtype=np.intp)


def _select_local_peak_indices(response_snr: np.ndarray) -> np.ndarray:
    """在每行的窄窗响应中选最高位置统计量，中心点用于平局和全无效回退。

    这只是“预测位置是否因配准/质心量化而偏离”的诊断选择，不是再次发现
    星点。由于它在多个位置中取最大值，结果有明确的 look-elsewhere 偏差；
    调用方必须同时保留中心固定测量和重定位后的测量，不能只报后者。
    """

    values = np.asarray(response_snr, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("response_snr local-search array must have shape (source_count, offset_count)")
    finite = np.isfinite(values)
    best = np.argmax(np.where(finite, values, -np.inf), axis=1).astype(np.intp)
    has_finite = finite.any(axis=1)
    return np.where(has_finite, best, 0).astype(np.intp, copy=False)


def _local_matched_filter_response(
    image: np.ndarray,
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    global_background: float,
    global_noise: float,
    psf_fwhm: float,
) -> np.ndarray:
    """计算窄窗候选的多尺度高斯匹配响应。

    重定位不能直接用九个孔径通量的最大值：孔径重叠、局部背景重估和
    邻近亮源会让最大值系统性偏向窗口边缘。这里先在整帧上做三种 PSF
    尺度的归一化高斯响应，再在候选坐标处双线性取样；它只负责选“最像
    点源中心”的局部位置，通量/FWHM/支持仍来自同一套强制孔径测光。
    ``xs/ys`` 超出有限图像区域的响应返回 ``-inf``。
    """

    values = np.asarray(image, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) == 0
    valid &= np.isfinite(values)
    residual = np.where(valid, values - float(global_background), 0.0)
    noise = max(float(global_noise), np.finfo(np.float64).eps)
    scales = tuple(
        max(float(psf_fwhm) * factor / 2.35482, 0.5)
        for factor in (0.65, 1.0, 1.35)
    )
    response_values: list[np.ndarray] = []
    for sigma in scales:
        radius = max(3, int(np.ceil(4.0 * sigma)))
        axis = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel_1d = np.exp(-0.5 * (axis / sigma) ** 2)
        kernel_1d /= max(float(kernel_1d.sum()), np.finfo(np.float64).eps)
        kernel_l2 = max(float(np.sqrt(np.sum(np.outer(kernel_1d, kernel_1d) ** 2))), np.finfo(float).eps)
        filtered = ndimage.gaussian_filter(residual, sigma=sigma, mode="constant", cval=0.0)
        coverage = ndimage.gaussian_filter(valid.astype(np.float64), sigma=sigma, mode="constant", cval=0.0)
        # 对有限掩膜做归一化；白噪声标准差随有效核面积按 sqrt(coverage)
        # 缩放，从而避免边缘/掩膜处因为少采样而人为抬高响应。
        denominator = noise * kernel_l2 * np.sqrt(np.maximum(coverage, np.finfo(np.float64).eps))
        response = np.divide(
            filtered,
            denominator,
            out=np.full(values.shape, -np.inf, dtype=np.float64),
            where=coverage >= 0.5,
        )
        response_values.append(response)
    stacked = np.stack(response_values, axis=0)
    response_map = np.max(stacked, axis=0)
    coordinates = np.vstack((np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)))
    sampled = ndimage.map_coordinates(
        response_map,
        coordinates,
        order=1,
        mode="constant",
        cval=-np.inf,
        prefilter=False,
    )
    return np.asarray(sampled, dtype=np.float64)


def _quality_like_measurement(
    flux_snr: float | None,
    support_3x3: int | None,
    center_valid: bool,
    fwhm: float | None,
    ellipticity: float | None,
    *,
    min_psf_support_pixels: int,
    min_flux_snr: float,
    min_fwhm: float,
    max_fwhm: float,
    max_ellipticity: float,
) -> bool:
    """应用仅供审计的质量样式门控；不等价于 ``quality_passed``。"""

    return bool(
        center_valid
        and flux_snr is not None
        and flux_snr >= float(min_flux_snr)
        and support_3x3 is not None
        and support_3x3 >= int(min_psf_support_pixels)
        and fwhm is not None
        and float(min_fwhm) <= fwhm <= float(max_fwhm)
        and ellipticity is not None
        and ellipticity <= float(max_ellipticity)
    )


def summarize_forced_stability(
    anchors: Sequence[ForcedAnchor],
    frame_metrics: Sequence[ForcedFrameMetric],
    *,
    frame_count: int,
) -> tuple[dict[str, object], ...]:
    """按类别汇总逐源强制测光结果，不把它转成物理身份概率。"""

    by_anchor: dict[int, list[ForcedFrameMetric]] = {}
    anchors_by_class: dict[str, list[ForcedAnchor]] = {}
    for anchor in anchors:
        anchors_by_class.setdefault(anchor.feature_class, []).append(anchor)
    for metric in frame_metrics:
        by_anchor.setdefault(metric.detection_id, []).append(metric)

    labels = dict(FEATURE_CLASS_ORDER)
    rows: list[dict[str, object]] = []
    for feature_class in _feature_order(tuple(anchors_by_class)):
        class_anchors = anchors_by_class.get(feature_class, [])
        if not class_anchors:
            continue
        per_source_flux: list[float] = []
        per_source_cv: list[float] = []
        per_source_support: list[float] = []
        per_source_fwhm: list[float] = []
        per_source_ellipticity: list[float] = []
        per_source_valid_fraction: list[float] = []
        per_source_quality_presence: list[int] = []
        per_source_local_flux: list[float] = []
        per_source_local_cv: list[float] = []
        per_source_local_support: list[float] = []
        per_source_local_fwhm: list[float] = []
        per_source_local_ellipticity: list[float] = []
        per_source_local_quality_presence: list[int] = []
        per_source_response: list[float] = []
        per_source_local_response: list[float] = []
        per_source_local_gain: list[float] = []
        per_source_modal_relocalization: list[float] = []
        per_source_relocalization_distance: list[float] = []
        per_source_relocalized_fraction: list[float] = []
        for anchor in class_anchors:
            metrics = sorted(by_anchor.get(anchor.detection_id, ()), key=lambda metric: metric.frame_index)
            if len(metrics) != frame_count:
                raise ValueError(
                    f"anchor {anchor.detection_id} has {len(metrics)} frame metrics; expected {frame_count}"
                )
            flux = [metric.flux_snr for metric in metrics if metric.flux_snr is not None]
            support = [float(metric.support_3x3) for metric in metrics if metric.support_3x3 is not None]
            fwhm = [metric.fwhm for metric in metrics if metric.fwhm is not None]
            ellipticity = [metric.ellipticity for metric in metrics if metric.ellipticity is not None]
            per_source_flux_value = _median(flux)
            per_source_cv_value = _robust_cv(flux)
            per_source_support_value = _median(support)
            per_source_fwhm_value = _median(fwhm)
            per_source_ellipticity_value = _median(ellipticity)
            if per_source_flux_value is not None:
                per_source_flux.append(per_source_flux_value)
            if per_source_cv_value is not None:
                per_source_cv.append(per_source_cv_value)
            if per_source_support_value is not None:
                per_source_support.append(per_source_support_value)
            if per_source_fwhm_value is not None:
                per_source_fwhm.append(per_source_fwhm_value)
            if per_source_ellipticity_value is not None:
                per_source_ellipticity.append(per_source_ellipticity_value)
            per_source_valid_fraction.append(sum(metric.center_valid for metric in metrics) / frame_count)
            per_source_quality_presence.append(sum(metric.quality_like for metric in metrics))

            local_flux = [
                metric.local_peak_flux_snr
                if metric.local_peak_flux_snr is not None
                else metric.flux_snr
                for metric in metrics
            ]
            local_support = [
                float(
                    metric.local_peak_support_3x3
                    if metric.local_peak_support_3x3 is not None
                    else metric.support_3x3
                )
                for metric in metrics
                if (
                    metric.local_peak_support_3x3 is not None
                    or metric.support_3x3 is not None
                )
            ]
            local_fwhm = [
                metric.local_peak_fwhm if metric.local_peak_fwhm is not None else metric.fwhm
                for metric in metrics
                if metric.local_peak_fwhm is not None or metric.fwhm is not None
            ]
            local_ellipticity = [
                (
                    metric.local_peak_ellipticity
                    if metric.local_peak_ellipticity is not None
                    else metric.ellipticity
                )
                for metric in metrics
                if metric.local_peak_ellipticity is not None or metric.ellipticity is not None
            ]
            local_flux_value = _median([value for value in local_flux if value is not None])
            local_cv_value = _robust_cv([value for value in local_flux if value is not None])
            local_support_value = _median(local_support)
            local_fwhm_value = _median(local_fwhm)
            local_ellipticity_value = _median(local_ellipticity)
            if local_flux_value is not None:
                per_source_local_flux.append(local_flux_value)
            if local_cv_value is not None:
                per_source_local_cv.append(local_cv_value)
            if local_support_value is not None:
                per_source_local_support.append(local_support_value)
            if local_fwhm_value is not None:
                per_source_local_fwhm.append(local_fwhm_value)
            if local_ellipticity_value is not None:
                per_source_local_ellipticity.append(local_ellipticity_value)
            response_value = _median(
                [metric.matched_response_snr for metric in metrics if metric.matched_response_snr is not None]
            )
            local_response_value = _median(
                [
                    metric.local_peak_response_snr
                    if metric.local_peak_response_snr is not None
                    else metric.matched_response_snr
                    for metric in metrics
                    if metric.local_peak_response_snr is not None or metric.matched_response_snr is not None
                ]
            )
            if response_value is not None:
                per_source_response.append(response_value)
            if local_response_value is not None:
                per_source_local_response.append(local_response_value)
            if per_source_flux_value is not None and per_source_flux_value > np.finfo(float).eps:
                if local_flux_value is not None:
                    per_source_local_gain.append(local_flux_value / per_source_flux_value - 1.0)
            relocation_offsets = [
                (
                    round(
                        metric.relocalization_dx_px
                        if metric.relocalization_dx_px is not None
                        else 0.0,
                        6,
                    ),
                    round(
                        metric.relocalization_dy_px
                        if metric.relocalization_dy_px is not None
                        else 0.0,
                        6,
                    ),
                )
                for metric in metrics
            ]
            offset_counts = dict.fromkeys(relocation_offsets, 0)
            for offset in relocation_offsets:
                offset_counts[offset] += 1
            per_source_modal_relocalization.append(max(offset_counts.values()) / frame_count)
            per_source_local_quality_presence.append(
                sum(
                    metric.local_peak_quality_like
                    if metric.local_peak_quality_like is not None
                    else metric.quality_like
                    for metric in metrics
                )
            )
            per_source_relocalization_distance.append(
                float(
                    np.median(
                        [
                            metric.relocalization_distance_px
                            if metric.relocalization_distance_px is not None
                            else 0.0
                            for metric in metrics
                        ]
                    )
                )
            )
            per_source_relocalized_fraction.append(
                sum(
                    metric.relocalized
                    if metric.relocalized is not None
                    else False
                    for metric in metrics
                )
                / frame_count
            )

        quality_ge_required = sum(value >= 12 for value in per_source_quality_presence)
        local_quality_ge_required = sum(value >= 12 for value in per_source_local_quality_presence)
        rows.append(
            {
                "feature_class": feature_class,
                "feature_class_label": labels.get(feature_class, class_anchors[0].feature_class_label),
                "sample_count": len(class_anchors),
                "frame_count": frame_count,
                "median_frame_valid_fraction": _median(per_source_valid_fraction),
                "median_quality_like_presence": _median([float(value) for value in per_source_quality_presence]),
                "quality_like_presence_ge_12_count": quality_ge_required,
                "median_flux_snr_across_frames": _median(per_source_flux),
                "robust_flux_snr_cv_median": _median(per_source_cv),
                "median_support_3x3": _median(per_source_support),
                "median_fwhm": _median(per_source_fwhm),
                "median_ellipticity": _median(per_source_ellipticity),
                "median_local_peak_quality_like_presence": _median(
                    [float(value) for value in per_source_local_quality_presence]
                ),
                "local_peak_quality_like_presence_ge_12_count": local_quality_ge_required,
                "median_local_peak_flux_snr_across_frames": _median(per_source_local_flux),
                "robust_local_peak_flux_snr_cv_median": _median(per_source_local_cv),
                "median_local_peak_support_3x3": _median(per_source_local_support),
                "median_local_peak_fwhm": _median(per_source_local_fwhm),
                "median_local_peak_ellipticity": _median(per_source_local_ellipticity),
                "median_matched_response_snr": _median(per_source_response),
                "median_local_peak_response_snr": _median(per_source_local_response),
                "median_local_peak_flux_gain_fraction": _median(per_source_local_gain),
                "median_modal_relocalization_fraction": _median(per_source_modal_relocalization),
                "median_relocalization_distance_px": _median(per_source_relocalization_distance),
                "median_relocalized_frame_fraction": _median(per_source_relocalized_fraction),
                "sample_detection_ids": "|".join(str(anchor.detection_id) for anchor in class_anchors),
            }
        )
    return tuple(rows)


def summarize_forced_pair_relationships(
    anchors: Sequence[ForcedAnchor],
    frame_metrics: Sequence[ForcedFrameMetric],
    detection_ids: Sequence[int],
    *,
    frame_count: int,
) -> tuple[dict[str, object], ...]:
    """汇总显式指定候选对的相对几何变化。

    这个层只回答“两个候选的局部位置是否相对收缩”，不回答它们是不是两颗
    独立恒星。预测位置已经包含序列累计平移，因此相对间距中的共同平移会被
    消掉；若两个候选在多个帧分别朝向对方移动，说明共享亮斑、局部结构或
    去混叠模型不足值得优先复核。所有位置仍来自一个有限的 ±1 px 诊断窗，
    不允许把它解释成新的候选搜索或轨迹。
    """

    explicit_ids = tuple(sorted({int(value) for value in detection_ids}))
    if len(explicit_ids) < 2:
        return ()
    anchor_by_id = {anchor.detection_id: anchor for anchor in anchors}
    metric_by_id_frame: dict[tuple[int, int], ForcedFrameMetric] = {
        (metric.detection_id, metric.frame_index): metric for metric in frame_metrics
    }
    rows: list[dict[str, object]] = []
    for detection_id_a, detection_id_b in combinations(explicit_ids, 2):
        anchor_a = anchor_by_id.get(detection_id_a)
        anchor_b = anchor_by_id.get(detection_id_b)
        if anchor_a is None or anchor_b is None:
            raise ValueError(
                f"pair anchor missing for detection_id(s) {detection_id_a}, {detection_id_b}"
            )
        anchor_dx = float(anchor_b.x - anchor_a.x)
        anchor_dy = float(anchor_b.y - anchor_a.y)
        anchor_separation = float(np.hypot(anchor_dx, anchor_dy))
        if anchor_separation <= np.finfo(float).eps:
            raise ValueError(f"pair anchors {detection_id_a}, {detection_id_b} have identical coordinates")
        unit_ab = np.asarray((anchor_dx, anchor_dy), dtype=np.float64) / anchor_separation
        for frame_index in range(1, frame_count + 1):
            metric_a = metric_by_id_frame.get((detection_id_a, frame_index))
            metric_b = metric_by_id_frame.get((detection_id_b, frame_index))
            if metric_a is None or metric_b is None:
                raise ValueError(
                    f"pair {detection_id_a}/{detection_id_b} is missing frame {frame_index}"
                )
            local_a = np.asarray(
                (
                    metric_a.local_peak_x if metric_a.local_peak_x is not None else metric_a.predicted_x,
                    metric_a.local_peak_y if metric_a.local_peak_y is not None else metric_a.predicted_y,
                ),
                dtype=np.float64,
            )
            local_b = np.asarray(
                (
                    metric_b.local_peak_x if metric_b.local_peak_x is not None else metric_b.predicted_x,
                    metric_b.local_peak_y if metric_b.local_peak_y is not None else metric_b.predicted_y,
                ),
                dtype=np.float64,
            )
            if not np.isfinite(local_a).all() or not np.isfinite(local_b).all():
                rows.append(
                    {
                        "detection_id_a": detection_id_a,
                        "detection_id_b": detection_id_b,
                        "frame_index": frame_index,
                        "anchor_separation_px": anchor_separation,
                        "local_separation_px": None,
                        "separation_change_px": None,
                        "a_shift_x_px": None,
                        "a_shift_y_px": None,
                        "b_shift_x_px": None,
                        "b_shift_y_px": None,
                        "a_toward_b_projection_px": None,
                        "b_toward_a_projection_px": None,
                        "both_shift_toward_each_other": False,
                        "local_offset_same": False,
                        "anchor_order_preserved": None,
                    }
                )
                continue
            local_vector = local_b - local_a
            local_separation = float(np.hypot(local_vector[0], local_vector[1]))
            shift_a = local_a - np.asarray((metric_a.predicted_x, metric_a.predicted_y), dtype=np.float64)
            shift_b = local_b - np.asarray((metric_b.predicted_x, metric_b.predicted_y), dtype=np.float64)
            projection_a = float(np.dot(shift_a, unit_ab))
            projection_b = float(np.dot(shift_b, -unit_ab))
            local_offset_same = bool(
                np.isclose(shift_a[0], shift_b[0], atol=1e-6)
                and np.isclose(shift_a[1], shift_b[1], atol=1e-6)
            )
            rows.append(
                {
                    "detection_id_a": detection_id_a,
                    "detection_id_b": detection_id_b,
                    "frame_index": frame_index,
                    "anchor_separation_px": anchor_separation,
                    "local_separation_px": local_separation,
                    "separation_change_px": local_separation - anchor_separation,
                    "a_shift_x_px": float(shift_a[0]),
                    "a_shift_y_px": float(shift_a[1]),
                    "b_shift_x_px": float(shift_b[0]),
                    "b_shift_y_px": float(shift_b[1]),
                    "a_toward_b_projection_px": projection_a,
                    "b_toward_a_projection_px": projection_b,
                    "both_shift_toward_each_other": bool(projection_a > 0.0 and projection_b > 0.0),
                    "local_offset_same": local_offset_same,
                    "anchor_order_preserved": bool(
                        np.dot(local_vector, unit_ab) > 0.0
                    ),
                }
            )

    summaries: list[dict[str, object]] = []
    for pair in combinations(explicit_ids, 2):
        pair_rows = [
            row
            for row in rows
            if row["detection_id_a"] == pair[0] and row["detection_id_b"] == pair[1]
        ]
        valid_rows = [row for row in pair_rows if row["local_separation_px"] is not None]
        separations = [float(row["local_separation_px"]) for row in valid_rows]
        changes = [float(row["separation_change_px"]) for row in valid_rows]
        summaries.append(
            {
                "detection_id_a": pair[0],
                "detection_id_b": pair[1],
                "feature_class_a": anchor_by_id[pair[0]].feature_class,
                "feature_class_b": anchor_by_id[pair[1]].feature_class,
                "frame_count": frame_count,
                "valid_frame_count": len(valid_rows),
                "anchor_separation_px": float(pair_rows[0]["anchor_separation_px"]),
                "median_local_separation_px": _median(separations),
                "median_separation_change_px": _median(changes),
                "local_separation_contract_fraction": (
                    sum(value < 0.0 for value in changes) / len(changes) if changes else None
                ),
                "both_shift_toward_each_other_fraction": (
                    sum(bool(row["both_shift_toward_each_other"]) for row in valid_rows) / len(valid_rows)
                    if valid_rows
                    else None
                ),
                "local_offset_same_fraction": (
                    sum(bool(row["local_offset_same"]) for row in valid_rows) / len(valid_rows)
                    if valid_rows
                    else None
                ),
                "anchor_order_reversal_count": sum(
                    row["anchor_order_preserved"] is False for row in valid_rows
                ),
            }
        )
    return tuple(summaries)


def _forced_pair_frame_rows(
    anchors: Sequence[ForcedAnchor],
    frame_metrics: Sequence[ForcedFrameMetric],
    detection_ids: Sequence[int],
    *,
    frame_count: int,
) -> tuple[dict[str, object], ...]:
    """返回与 pair 汇总相同定义的逐帧几何行，供机器表输出。"""

    explicit_ids = tuple(sorted({int(value) for value in detection_ids}))
    if len(explicit_ids) < 2:
        return ()
    anchor_by_id = {anchor.detection_id: anchor for anchor in anchors}
    metric_by_id_frame: dict[tuple[int, int], ForcedFrameMetric] = {
        (metric.detection_id, metric.frame_index): metric for metric in frame_metrics
    }
    rows: list[dict[str, object]] = []
    for detection_id_a, detection_id_b in combinations(explicit_ids, 2):
        anchor_a = anchor_by_id[detection_id_a]
        anchor_b = anchor_by_id[detection_id_b]
        anchor_dx = float(anchor_b.x - anchor_a.x)
        anchor_dy = float(anchor_b.y - anchor_a.y)
        anchor_separation = float(np.hypot(anchor_dx, anchor_dy))
        unit_ab = np.asarray((anchor_dx, anchor_dy), dtype=np.float64) / anchor_separation
        for frame_index in range(1, frame_count + 1):
            metric_a = metric_by_id_frame[(detection_id_a, frame_index)]
            metric_b = metric_by_id_frame[(detection_id_b, frame_index)]
            local_a = np.asarray(
                (
                    metric_a.local_peak_x if metric_a.local_peak_x is not None else metric_a.predicted_x,
                    metric_a.local_peak_y if metric_a.local_peak_y is not None else metric_a.predicted_y,
                ),
                dtype=np.float64,
            )
            local_b = np.asarray(
                (
                    metric_b.local_peak_x if metric_b.local_peak_x is not None else metric_b.predicted_x,
                    metric_b.local_peak_y if metric_b.local_peak_y is not None else metric_b.predicted_y,
                ),
                dtype=np.float64,
            )
            finite = bool(np.isfinite(local_a).all() and np.isfinite(local_b).all())
            if finite:
                local_vector = local_b - local_a
                local_separation = float(np.hypot(local_vector[0], local_vector[1]))
                shift_a = local_a - np.asarray((metric_a.predicted_x, metric_a.predicted_y), dtype=np.float64)
                shift_b = local_b - np.asarray((metric_b.predicted_x, metric_b.predicted_y), dtype=np.float64)
                projection_a = float(np.dot(shift_a, unit_ab))
                projection_b = float(np.dot(shift_b, -unit_ab))
                same_offset = bool(
                    np.allclose(shift_a, shift_b, atol=1e-6, rtol=0.0)
                )
                order_preserved: bool | None = bool(np.dot(local_vector, unit_ab) > 0.0)
                both_inward = bool(projection_a > 0.0 and projection_b > 0.0)
            else:
                local_separation = None
                shift_a = np.asarray((np.nan, np.nan), dtype=np.float64)
                shift_b = np.asarray((np.nan, np.nan), dtype=np.float64)
                projection_a = None
                projection_b = None
                same_offset = False
                order_preserved = None
                both_inward = False
            rows.append(
                {
                    "detection_id_a": detection_id_a,
                    "detection_id_b": detection_id_b,
                    "frame_index": frame_index,
                    "anchor_separation_px": anchor_separation,
                    "local_separation_px": local_separation,
                    "separation_change_px": (
                        local_separation - anchor_separation if local_separation is not None else None
                    ),
                    "a_shift_x_px": float(shift_a[0]) if finite else None,
                    "a_shift_y_px": float(shift_a[1]) if finite else None,
                    "b_shift_x_px": float(shift_b[0]) if finite else None,
                    "b_shift_y_px": float(shift_b[1]) if finite else None,
                    "a_toward_b_projection_px": projection_a,
                    "b_toward_a_projection_px": projection_b,
                    "both_shift_toward_each_other": both_inward,
                    "local_offset_same": same_offset,
                    "anchor_order_preserved": order_preserved,
                }
            )
    return tuple(rows)


def run_forced_stability(
    source_catalog: str | Path,
    persistence_json: str | Path,
    *,
    base_dir: str | Path = ".",
    max_per_class: int = 32,
    anchor_coordinate: str = "peak",
    include_detection_ids: Sequence[int] = (),
    aperture_radius: int = 4,
    min_psf_support_pixels: int = 3,
    min_flux_snr: float = 5.0,
    min_fwhm: float = 0.8,
    max_fwhm: float = 12.0,
    max_ellipticity: float = 0.65,
    local_peak_search_radius: int = 1,
    local_peak_psf_fwhm: float = 3.0,
    progress: Callable[[int, int], None] | None = None,
) -> ForcedStabilityResult:
    """运行不修改检测结果的分层逐帧强制测光审计。"""

    if aperture_radius < 1 or min_psf_support_pixels < 1 or max_per_class < 1:
        raise ValueError("aperture_radius, min_psf_support_pixels and max_per_class must be positive")
    if min_flux_snr <= 0 or min_fwhm <= 0 or max_fwhm < min_fwhm or not 0.0 < max_ellipticity:
        raise ValueError("invalid forced-stability quality-style thresholds")
    if local_peak_psf_fwhm <= 0:
        raise ValueError("local_peak_psf_fwhm must be positive")
    local_offsets = _local_search_offsets(local_peak_search_radius)

    coordinate = _normalize_anchor_coordinate(anchor_coordinate)
    anchors = load_forced_anchors(
        source_catalog,
        max_per_class=max_per_class,
        anchor_coordinate=coordinate,
        include_detection_ids=include_detection_ids,
    )
    persistence_path, frames = _load_frame_specs(persistence_json, base_dir=base_dir)
    if not anchors:
        raise ValueError("source catalog has no anchors")

    frame_metrics: list[ForcedFrameMetric] = []
    anchor_x = np.asarray([anchor.x for anchor in anchors], dtype=np.float64)
    anchor_y = np.asarray([anchor.y for anchor in anchors], dtype=np.float64)
    for completed, spec in enumerate(frames, start=1):
        frame = read_fits(spec.path)
        values = np.asarray(frame.data)
        mask = auxiliary_mask(values.shape)
        xs = anchor_x + float(spec.shift_x)
        ys = anchor_y + float(spec.shift_y)
        window_x = (xs[:, None] + local_offsets[None, :, 0]).reshape(-1)
        window_y = (ys[:, None] + local_offsets[None, :, 1]).reshape(-1)
        flux_snr, support, center_valid, fwhm, ellipticity = _stack_forced_frame_measure_batch(
            values,
            mask,
            window_x,
            window_y,
            aperture_radius=aperture_radius,
            min_psf_support_pixels=min_psf_support_pixels,
            global_background=spec.background_adu,
            global_noise=spec.noise_adu,
        )
        source_count = len(anchors)
        offset_count = int(local_offsets.shape[0])
        flux_snr = flux_snr.reshape(source_count, offset_count)
        support = support.reshape(source_count, offset_count)
        center_valid = center_valid.reshape(source_count, offset_count)
        fwhm = fwhm.reshape(source_count, offset_count)
        ellipticity = ellipticity.reshape(source_count, offset_count)
        matched_response = _local_matched_filter_response(
            values,
            mask,
            window_x,
            window_y,
            global_background=spec.background_adu,
            global_noise=spec.noise_adu,
            psf_fwhm=local_peak_psf_fwhm,
        ).reshape(source_count, offset_count)
        local_indices = _select_local_peak_indices(matched_response)
        for anchor_index, anchor in enumerate(anchors):
            fixed_flux_snr = flux_snr[anchor_index, 0]
            fixed_support = support[anchor_index, 0]
            fixed_center_valid = bool(center_valid[anchor_index, 0])
            fixed_fwhm = fwhm[anchor_index, 0]
            fixed_ellipticity = ellipticity[anchor_index, 0]
            local_index = int(local_indices[anchor_index])
            local_offset_x = float(local_offsets[local_index, 0])
            local_offset_y = float(local_offsets[local_index, 1])
            local_flux_snr = flux_snr[anchor_index, local_index]
            local_support = support[anchor_index, local_index]
            local_center_valid = bool(center_valid[anchor_index, local_index])
            local_fwhm = fwhm[anchor_index, local_index]
            local_ellipticity = ellipticity[anchor_index, local_index]
            fixed_response = matched_response[anchor_index, 0]
            local_response = matched_response[anchor_index, local_index]
            flux_value = float(fixed_flux_snr) if np.isfinite(fixed_flux_snr) else None
            support_value = int(fixed_support) if fixed_center_valid else None
            fwhm_value = float(fixed_fwhm) if np.isfinite(fixed_fwhm) else None
            ellipticity_value = (
                float(fixed_ellipticity) if np.isfinite(fixed_ellipticity) else None
            )
            local_flux_value = float(local_flux_snr) if np.isfinite(local_flux_snr) else None
            local_support_value = int(local_support) if local_center_valid else None
            local_fwhm_value = float(local_fwhm) if np.isfinite(local_fwhm) else None
            local_ellipticity_value = (
                float(local_ellipticity) if np.isfinite(local_ellipticity) else None
            )
            fixed_response_value = float(fixed_response) if np.isfinite(fixed_response) else None
            local_response_value = float(local_response) if np.isfinite(local_response) else None
            quality_like = _quality_like_measurement(
                flux_value,
                support_value,
                fixed_center_valid,
                fwhm_value,
                ellipticity_value,
                min_psf_support_pixels=min_psf_support_pixels,
                min_flux_snr=min_flux_snr,
                min_fwhm=min_fwhm,
                max_fwhm=max_fwhm,
                max_ellipticity=max_ellipticity,
            )
            local_quality_like = _quality_like_measurement(
                local_flux_value,
                local_support_value,
                local_center_valid,
                local_fwhm_value,
                local_ellipticity_value,
                min_psf_support_pixels=min_psf_support_pixels,
                min_flux_snr=min_flux_snr,
                min_fwhm=min_fwhm,
                max_fwhm=max_fwhm,
                max_ellipticity=max_ellipticity,
            )
            frame_metrics.append(
                ForcedFrameMetric(
                    detection_id=anchor.detection_id,
                    feature_class=anchor.feature_class,
                    feature_class_label=anchor.feature_class_label,
                    frame_index=spec.frame_index,
                    predicted_x=float(xs[anchor_index]),
                    predicted_y=float(ys[anchor_index]),
                    flux_snr=flux_value,
                    support_3x3=support_value,
                    center_valid=fixed_center_valid,
                    fwhm=fwhm_value,
                    ellipticity=ellipticity_value,
                    quality_like=quality_like,
                    local_peak_x=float(xs[anchor_index] + local_offset_x),
                    local_peak_y=float(ys[anchor_index] + local_offset_y),
                    local_peak_flux_snr=local_flux_value,
                    local_peak_support_3x3=local_support_value,
                    local_peak_center_valid=local_center_valid,
                    local_peak_fwhm=local_fwhm_value,
                    local_peak_ellipticity=local_ellipticity_value,
                    local_peak_quality_like=local_quality_like,
                    matched_response_snr=fixed_response_value,
                    local_peak_response_snr=local_response_value,
                    relocalization_dx_px=local_offset_x,
                    relocalization_dy_px=local_offset_y,
                    relocalization_distance_px=float(np.hypot(local_offset_x, local_offset_y)),
                    relocalized=bool(local_index != 0),
                )
            )
        if progress is not None:
            progress(completed, len(frames))

    summary_rows = summarize_forced_stability(anchors, frame_metrics, frame_count=len(frames))
    return ForcedStabilityResult(
        source_catalog=Path(source_catalog),
        persistence_json=persistence_path,
        anchor_coordinate=coordinate,
        included_detection_ids=tuple(sorted({int(value) for value in include_detection_ids})),
        frame_count=len(frames),
        anchors=tuple(anchors),
        frame_metrics=tuple(frame_metrics),
        summary_rows=summary_rows,
        aperture_radius=aperture_radius,
        min_psf_support_pixels=min_psf_support_pixels,
        min_flux_snr=float(min_flux_snr),
        min_fwhm=float(min_fwhm),
        max_fwhm=float(max_fwhm),
        max_ellipticity=float(max_ellipticity),
        local_peak_search_radius=int(local_peak_search_radius),
        local_peak_psf_fwhm=float(local_peak_psf_fwhm),
    )


def write_forced_stability_artifacts(result: ForcedStabilityResult, out_dir: str | Path) -> Path:
    """写逐帧 CSV、类别汇总 CSV 和带参数边界的 JSON。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_fields = (
        "detection_id",
        "feature_class",
        "feature_class_label",
        "frame_index",
        "predicted_x",
        "predicted_y",
        "flux_snr",
        "support_3x3",
        "center_valid",
        "fwhm",
        "ellipticity",
        "quality_like",
        "local_peak_x",
        "local_peak_y",
        "local_peak_flux_snr",
        "local_peak_support_3x3",
        "local_peak_center_valid",
        "local_peak_fwhm",
        "local_peak_ellipticity",
        "local_peak_quality_like",
        "matched_response_snr",
        "local_peak_response_snr",
        "relocalization_dx_px",
        "relocalization_dy_px",
        "relocalization_distance_px",
        "relocalized",
    )
    with (output / "forced_stability_frame_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(metric.as_dict() for metric in result.frame_metrics)

    summary_fields = (
        "feature_class",
        "feature_class_label",
        "sample_count",
        "frame_count",
        "median_frame_valid_fraction",
        "median_quality_like_presence",
        "quality_like_presence_ge_12_count",
        "median_flux_snr_across_frames",
        "robust_flux_snr_cv_median",
        "median_support_3x3",
        "median_fwhm",
        "median_ellipticity",
        "median_local_peak_quality_like_presence",
        "local_peak_quality_like_presence_ge_12_count",
        "median_local_peak_flux_snr_across_frames",
        "robust_local_peak_flux_snr_cv_median",
        "median_local_peak_support_3x3",
        "median_local_peak_fwhm",
        "median_local_peak_ellipticity",
        "median_matched_response_snr",
        "median_local_peak_response_snr",
        "median_local_peak_flux_gain_fraction",
        "median_modal_relocalization_fraction",
        "median_relocalization_distance_px",
        "median_relocalized_frame_fraction",
        "sample_detection_ids",
    )
    with (output / "forced_stability_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(result.summary_rows)

    pair_frame_rows = _forced_pair_frame_rows(
        result.anchors,
        result.frame_metrics,
        result.included_detection_ids,
        frame_count=result.frame_count,
    )
    pair_summary_rows = summarize_forced_pair_relationships(
        result.anchors,
        result.frame_metrics,
        result.included_detection_ids,
        frame_count=result.frame_count,
    )
    if pair_frame_rows:
        pair_frame_fields = (
            "detection_id_a",
            "detection_id_b",
            "frame_index",
            "anchor_separation_px",
            "local_separation_px",
            "separation_change_px",
            "a_shift_x_px",
            "a_shift_y_px",
            "b_shift_x_px",
            "b_shift_y_px",
            "a_toward_b_projection_px",
            "b_toward_a_projection_px",
            "both_shift_toward_each_other",
            "local_offset_same",
            "anchor_order_preserved",
        )
        with (output / "forced_stability_pair_frame_metrics.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=pair_frame_fields)
            writer.writeheader()
            writer.writerows(pair_frame_rows)
        pair_summary_fields = (
            "detection_id_a",
            "detection_id_b",
            "feature_class_a",
            "feature_class_b",
            "frame_count",
            "valid_frame_count",
            "anchor_separation_px",
            "median_local_separation_px",
            "median_separation_change_px",
            "local_separation_contract_fraction",
            "both_shift_toward_each_other_fraction",
            "local_offset_same_fraction",
            "anchor_order_reversal_count",
        )
        with (output / "forced_stability_pair_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=pair_summary_fields)
            writer.writeheader()
            writer.writerows(pair_summary_rows)

    payload = {
        "source_catalog": str(result.source_catalog),
        "persistence_json": str(result.persistence_json),
        "frame_count": result.frame_count,
        "anchor_count": len(result.anchors),
        "frame_metric_count": len(result.frame_metrics),
        "parameters": {
            "aperture_radius": result.aperture_radius,
            "min_psf_support_pixels": result.min_psf_support_pixels,
            "min_flux_snr": result.min_flux_snr,
            "min_fwhm": result.min_fwhm,
            "max_fwhm": result.max_fwhm,
            "max_ellipticity": result.max_ellipticity,
            "anchor_coordinate": result.anchor_coordinate,
            "local_peak_search_radius": result.local_peak_search_radius,
            "local_peak_psf_fwhm": result.local_peak_psf_fwhm,
            "local_peak_selection": (
                "argmax of three-scale normalized Gaussian matched response in the bounded "
                "integer-offset window; center offset is first for deterministic ties"
            ),
            "anchor_coordinate_definition": {
                "peak": "source_catalog peak_x/peak_y with measured x/y fallback",
                "centroid": "source_catalog measured x/y with peak_x/peak_y fallback",
            }[result.anchor_coordinate],
            "included_detection_ids": list(result.included_detection_ids),
            "raw_measurement": "unscaled FITS; per-frame background_adu/noise_adu from persistence JSON",
        },
        "anchors": [anchor.as_dict() for anchor in result.anchors],
        "summary_rows": list(result.summary_rows),
        "pair_relationship_summary": list(pair_summary_rows),
        "note": (
            "Fixed fields (flux_snr/support_3x3/fwhm/ellipticity/quality_like) are forced "
            "photometry at predicted registered positions. local_peak_* is a bounded "
            "diagnostic relocation within at most ±1 px, not a reclassification or a "
            "WCS/catalog identity match. Taking the maximum over multiple positions has "
            "a look-elsewhere bias over positions and scales, so fixed and local values "
            "must be reported together. "
            "quality_like is not quality_passed; value-domain flags and deblending evidence "
            "must be applied separately."
            " When two detection IDs are explicitly included, pair relationship files also "
            "report registered local separation and inward-shift fractions. They are bounded "
            "diagnostics for shared structure or model mismatch, not evidence of a binary "
            "or two independent catalog sources."
        ),
    }
    (output / "forced_stability.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 15 帧原始 FITS 做分层强制测光稳定性审计")
    parser.add_argument("source_catalog", type=Path, help="rst19-sources 生成的首帧 source_catalog.csv")
    parser.add_argument(
        "persistence_json",
        type=Path,
        help="rst19-feature-sequence 生成的 sequence_feature_persistence.json",
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="序列 JSON 中相对 FITS 路径的基准目录")
    parser.add_argument("--max-per-class", type=int, default=32, help="每个首要类别的分位点抽样上限")
    parser.add_argument(
        "--anchor-coordinate",
        choices=("peak", "centroid"),
        default="peak",
        help="强制测光锚点：peak 与序列提案一致；centroid 用测量质心作敏感性对照",
    )
    parser.add_argument(
        "--include-detection-id",
        action="append",
        type=int,
        default=[],
        help="追加指定 detection_id 到分层样本，可重复指定；不占用每类分位抽样名额",
    )
    parser.add_argument("--aperture-radius", type=int, default=4, help="强制孔径半径 pixel")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="质量样式的最少 3×3 支持像素")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量样式的通量 SNR 下限")
    parser.add_argument("--min-fwhm", type=float, default=0.8, help="质量样式 FWHM 下限")
    parser.add_argument("--max-fwhm", type=float, default=12.0, help="质量样式 FWHM 上限")
    parser.add_argument("--max-ellipticity", type=float, default=0.65, help="质量样式椭圆率上限")
    parser.add_argument(
        "--local-peak-search-radius",
        type=int,
        choices=(0, 1),
        default=1,
        help="每帧在预测坐标周围做局部强制响应重定位的半径；默认 ±1 px，0 关闭",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印逐帧进度")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/forced-stability-audit"), help="CSV/JSON 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        progress = None if args.quiet else lambda completed, total: print(
            f"forced stability: frame {completed}/{total}"
        )
        result = run_forced_stability(
            args.source_catalog,
            args.persistence_json,
            base_dir=args.base_dir,
            max_per_class=args.max_per_class,
            anchor_coordinate=args.anchor_coordinate,
            include_detection_ids=args.include_detection_id,
            aperture_radius=args.aperture_radius,
            min_psf_support_pixels=args.min_psf_support_pixels,
            min_flux_snr=args.min_flux_snr,
            min_fwhm=args.min_fwhm,
            max_fwhm=args.max_fwhm,
            max_ellipticity=args.max_ellipticity,
            local_peak_search_radius=args.local_peak_search_radius,
            progress=progress,
        )
        output = write_forced_stability_artifacts(result, args.out_dir)
        print(f"rst19-forced-stability: 审计已写入 {output}")
        print(f"anchors={len(result.anchors)} frames={result.frame_count} metrics={len(result.frame_metrics)}")
        for row in result.summary_rows:
            print(
                f"{row['feature_class']}: n={row['sample_count']} "
                f"fixed_flux_snr={row['median_flux_snr_across_frames']} "
                f"local_flux_snr={row['median_local_peak_flux_snr_across_frames']} "
                f"quality_like_ge12={row['quality_like_presence_ge_12_count']} "
                f"local_quality_like_ge12={row['local_peak_quality_like_presence_ge_12_count']} "
                f"reloc_dist={row['median_relocalization_distance_px']}"
            )
        for row in summarize_forced_pair_relationships(
            result.anchors,
            result.frame_metrics,
            result.included_detection_ids,
            frame_count=result.frame_count,
        ):
            print(
                f"pair {row['detection_id_a']}/{row['detection_id_b']}: "
                f"anchor_sep={row['anchor_separation_px']} "
                f"local_sep={row['median_local_separation_px']} "
                f"contract={row['local_separation_contract_fraction']} "
                f"inward={row['both_shift_toward_each_other_fraction']}"
            )
        return 0
    except (OSError, ValueError) as exc:
        print(f"rst19-forced-stability: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
