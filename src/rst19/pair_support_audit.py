"""近邻双候选的独立像素支持与留出掩膜审计。

这个研究工具专门回答一个容易被源表数量掩盖的问题：同一团局部结构在
两个相距很近的坐标上被提案时，两个坐标是否各自有独立、有效的像素支持？

审计同时输出两类证据：

* 在目标局部裁剪上重新执行当前 Gaussian+DoG 宽筛和质量层，并用全局
  最大基数/最小残差的一对一分配，防止一个重叠峰同时算给两个目标；
* 对每个目标把孔径拆成总孔径、共同孔径、非共同孔径和近邻二分区，分别
  计算有效像素、净通量、正残差通量和噪声尺度。随后屏蔽重复码、极端负
  异常和共同孔径，观察候选是否仍有独立支持。

它不修改默认检测器、质量规则或 GUI 缓存。尤其要注意：屏蔽共同孔径会
同时拿掉真实双星的核心像素，因此“屏蔽后消失”只能说明独立支持没有被
当前控制证明，不能单独当作物理上的假星结论；最终还需要星表/WCS、跨帧
配准、经验 PSF 和注入回收等正向证据。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .detection import Detection, _default_negative_overflow_limit, detect_sources
from .fits import FitsFrame, read_fits
from .matching import _global_assignment


PAIR_SUPPORT_MASK_MODES: tuple[str, ...] = (
    "raw",
    "range_masked",
    "repeated_code_masked",
    "local_anomaly_masked",
    "local_anomaly_repeated_code_masked",
    "shared_aperture_masked",
    "shared_local_anomaly_repeated_code_masked",
)

PAIR_SUPPORT_SUMMARY_FIELDS: tuple[str, ...] = (
    "mask_mode",
    "frame_count",
    "assigned_two_count",
    "assigned_one_count",
    "assigned_zero_count",
    "quality_two_count",
    "assigned_two_fraction",
    "quality_two_fraction",
    "masked_pixel_median",
    "primary_shared_positive_fraction_median",
    "secondary_shared_positive_fraction_median",
    "primary_exclusive_snr_median",
    "secondary_exclusive_snr_median",
    "primary_partition_snr_median",
    "secondary_partition_snr_median",
)

_MASK_COMPONENTS = {
    "raw": frozenset(),
    "range_masked": frozenset({"range"}),
    "repeated_code_masked": frozenset({"repeated_code"}),
    "local_anomaly_masked": frozenset({"local_anomaly"}),
    "local_anomaly_repeated_code_masked": frozenset({"local_anomaly", "repeated_code"}),
    "shared_aperture_masked": frozenset({"shared_aperture"}),
    "shared_local_anomaly_repeated_code_masked": frozenset(
        {"shared_aperture", "local_anomaly", "repeated_code"}
    ),
}


@dataclass(frozen=True, slots=True)
class _SupportMetrics:
    """一个目标在一个掩膜下的像素分区测量。"""

    background_adu: float | None
    noise_adu: float | None
    total_valid_pixels: int
    shared_valid_pixels: int
    exclusive_valid_pixels: int
    partition_valid_pixels: int
    total_net_flux_adu: float | None
    shared_net_flux_adu: float | None
    exclusive_net_flux_adu: float | None
    partition_net_flux_adu: float | None
    total_positive_excess_adu: float | None
    shared_positive_excess_adu: float | None
    exclusive_positive_excess_adu: float | None
    partition_positive_excess_adu: float | None
    shared_positive_fraction: float | None
    exclusive_snr: float | None
    partition_snr: float | None


@dataclass(frozen=True, slots=True)
class PairSupportAuditRow:
    """一帧、一个留出掩膜下的 pair 支持证据。"""

    frame_index: int
    path: str
    frame_shift_x_px: float
    frame_shift_y_px: float
    mask_mode: str
    crop_x0: int
    crop_y0: int
    crop_width: int
    crop_height: int
    masked_pixel_count: int
    local_negative_limit_adu: float | None
    candidate_count: int
    quality_count: int
    nearby_candidate_count: int
    nearby_quality_count: int
    assigned_candidate_count: int
    primary_re_detected_id: int | None
    secondary_re_detected_id: int | None
    primary_match_distance_px: float | None
    secondary_match_distance_px: float | None
    primary_local_quality_passed: bool | None
    secondary_local_quality_passed: bool | None
    primary_local_flags: tuple[str, ...]
    secondary_local_flags: tuple[str, ...]
    primary_catalog_quality_passed: bool | None
    secondary_catalog_quality_passed: bool | None
    primary_catalog_flags: tuple[str, ...]
    secondary_catalog_flags: tuple[str, ...]
    primary_background_adu: float | None
    primary_noise_adu: float | None
    primary_total_valid_pixels: int
    primary_shared_valid_pixels: int
    primary_exclusive_valid_pixels: int
    primary_partition_valid_pixels: int
    primary_total_net_flux_adu: float | None
    primary_shared_net_flux_adu: float | None
    primary_exclusive_net_flux_adu: float | None
    primary_partition_net_flux_adu: float | None
    primary_total_positive_excess_adu: float | None
    primary_shared_positive_excess_adu: float | None
    primary_exclusive_positive_excess_adu: float | None
    primary_partition_positive_excess_adu: float | None
    primary_shared_positive_fraction: float | None
    primary_exclusive_snr: float | None
    primary_partition_snr: float | None
    secondary_background_adu: float | None
    secondary_noise_adu: float | None
    secondary_total_valid_pixels: int
    secondary_shared_valid_pixels: int
    secondary_exclusive_valid_pixels: int
    secondary_partition_valid_pixels: int
    secondary_total_net_flux_adu: float | None
    secondary_shared_net_flux_adu: float | None
    secondary_exclusive_net_flux_adu: float | None
    secondary_partition_net_flux_adu: float | None
    secondary_total_positive_excess_adu: float | None
    secondary_shared_positive_excess_adu: float | None
    secondary_exclusive_positive_excess_adu: float | None
    secondary_partition_positive_excess_adu: float | None
    secondary_shared_positive_fraction: float | None
    secondary_exclusive_snr: float | None
    secondary_partition_snr: float | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairSupportAuditResult:
    """近邻双候选的逐帧、逐掩膜审计结果。"""

    frame_count: int
    primary_detection_id: int | None
    secondary_detection_id: int | None
    primary_target_xy: tuple[float, float]
    secondary_target_xy: tuple[float, float]
    frame_shifts: tuple[tuple[float, float], ...]
    mask_modes: tuple[str, ...]
    repeated_code_values: tuple[int, ...]
    crop_padding_px: float
    aperture_radius: int
    match_radius_px: float
    rows: tuple[PairSupportAuditRow, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_target_xy": list(self.primary_target_xy),
            "secondary_target_xy": list(self.secondary_target_xy),
            "frame_shifts": [list(shift) for shift in self.frame_shifts],
            "mask_modes": list(self.mask_modes),
            "repeated_code_values": list(self.repeated_code_values),
            "crop_padding_px": self.crop_padding_px,
            "aperture_radius": self.aperture_radius,
            "match_radius_px": self.match_radius_px,
            "rows": [row.as_dict() for row in self.rows],
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
        }


def _xy(point: Sequence[float], label: str) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{label} must contain x and y")
    result = (float(point[0]), float(point[1]))
    if not all(np.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite coordinates")
    return result


def _frame_values(frame: FitsFrame | np.ndarray | str | Path) -> tuple[np.ndarray, str]:
    if isinstance(frame, FitsFrame):
        values = np.asarray(frame.data)
        label = str(frame.path)
    elif isinstance(frame, (str, Path)):
        loaded = read_fits(frame)
        values = np.asarray(loaded.data)
        label = str(loaded.path)
    else:
        values = np.asarray(frame)
        label = "<array>"
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    return values, label


def _crop_bounds(
    shape: tuple[int, int],
    positions: Sequence[tuple[float, float]],
    padding_px: float,
) -> tuple[int, int, int, int]:
    if padding_px <= 0 or not np.isfinite(float(padding_px)):
        raise ValueError("crop_padding_px must be positive and finite")
    height, width = shape
    x0 = max(0, int(np.floor(min(point[0] for point in positions) - padding_px)))
    x1 = min(width, int(np.ceil(max(point[0] for point in positions) + padding_px)) + 1)
    y0 = max(0, int(np.floor(min(point[1] for point in positions) - padding_px)))
    y1 = min(height, int(np.ceil(max(point[1] for point in positions) + padding_px)) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("pair crop is empty")
    return x0, y0, x1, y1


def _robust_location_scale(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(values) & ~np.asarray(mask, dtype=bool)
    samples = np.asarray(values[valid], dtype=np.float64)
    if samples.size == 0:
        raise ValueError("pair crop has no valid pixels")
    center = float(np.median(samples))
    mad = float(1.4826 * np.median(np.abs(samples - center)))
    if not np.isfinite(mad) or mad <= np.finfo(np.float64).eps:
        mad = float(np.std(samples))
    return center, max(mad, np.finfo(np.float64).eps)


def _mask_components(
    crop: np.ndarray,
    *,
    crop_origin: tuple[int, int],
    positions: Sequence[tuple[float, float]],
    aperture_radius: int,
    repeated_code_values: Sequence[int],
    negative_overflow_limit: float | None,
) -> tuple[dict[str, np.ndarray], float | None]:
    """构造留出掩膜，并返回局部负异常阈值。"""

    if len(positions) != 2:
        raise ValueError("positions must contain primary and secondary coordinates")
    base = ~np.isfinite(crop)
    local_background, local_noise = _robust_location_scale(crop, base)
    # 与检测器的 CODE_PATTERN 局部审计保持相同的数量级：普通负噪声不
    # 会进入这里，远离局部背景 50σ 且至少低于 -1000 ADU 的值才留出。
    local_negative_limit = min(-1000.0, local_background - 50.0 * local_noise)
    x0, y0 = crop_origin
    yy, xx = np.indices(crop.shape, dtype=np.float64)
    global_x = xx + x0
    global_y = yy + y0
    primary_distance = np.hypot(global_x - positions[0][0], global_y - positions[0][1])
    secondary_distance = np.hypot(global_x - positions[1][0], global_y - positions[1][1])
    shared_aperture = (primary_distance <= aperture_radius) & (secondary_distance <= aperture_radius)
    repeated = (
        np.isin(crop, np.asarray(tuple(repeated_code_values), dtype=crop.dtype))
        if repeated_code_values
        else np.zeros(crop.shape, dtype=bool)
    )
    local_anomaly = np.isfinite(crop) & (crop <= local_negative_limit)
    range_mask = np.zeros(crop.shape, dtype=bool)
    if negative_overflow_limit is not None:
        range_mask = np.isfinite(crop) & (
            (crop <= float(negative_overflow_limit)) | (crop >= -float(negative_overflow_limit))
        )
    components = {
        "range": range_mask,
        "repeated_code": repeated,
        "local_anomaly": local_anomaly,
        "shared_aperture": shared_aperture,
    }
    masks: dict[str, np.ndarray] = {}
    for mode, names in _MASK_COMPONENTS.items():
        mode_mask = base.copy()
        for name in names:
            mode_mask |= components[name]
        masks[mode] = np.asarray(mode_mask, dtype=bool)
    return masks, float(local_negative_limit)


def _support_region_metrics(
    values: np.ndarray,
    mask: np.ndarray,
    region: np.ndarray,
    *,
    background: float,
    noise: float,
    annulus_count: int,
) -> tuple[int, float | None, float | None, float | None]:
    valid = region & np.isfinite(values) & ~mask
    count = int(np.count_nonzero(valid))
    if count == 0:
        return 0, None, None, None
    residual = np.asarray(values[valid], dtype=np.float64) - float(background)
    net = float(np.sum(residual))
    positive = float(np.sum(np.maximum(residual, 0.0)))
    error = float(noise) * np.sqrt(count + (count * count / max(1, int(annulus_count))))
    snr = net / max(error, np.finfo(np.float64).eps)
    return count, net, positive, float(snr) if np.isfinite(snr) else None


def _support_metrics(
    values: np.ndarray,
    mask: np.ndarray,
    target: tuple[float, float],
    other: tuple[float, float],
    *,
    crop_origin: tuple[int, int],
    aperture_radius: int,
) -> _SupportMetrics:
    """把一个孔径拆成共同区、几何非共同区和近邻二分区。"""

    x0, y0 = crop_origin
    yy, xx = np.indices(values.shape, dtype=np.float64)
    global_x = xx + x0
    global_y = yy + y0
    target_distance = np.hypot(global_x - target[0], global_y - target[1])
    other_distance = np.hypot(global_x - other[0], global_y - other[1])
    aperture = target_distance <= aperture_radius
    other_aperture = other_distance <= aperture_radius
    shared = aperture & other_aperture
    exclusive = aperture & ~other_aperture
    # 二分区把 union aperture 的每个像素只分给一个目标，避免“共同区
    # 同时计给两颗”造成伪独立通量。等距像素归主目标，规则固定可复现。
    partition = (aperture | other_aperture) & (target_distance <= other_distance)
    base_mask = np.asarray(mask, dtype=bool) | ~np.isfinite(values)
    annulus = (
        (target_distance >= float(aperture_radius + 2))
        & (target_distance <= float(aperture_radius + 6))
        & ~base_mask
    )
    annulus_values = np.asarray(values[annulus], dtype=np.float64)
    if annulus_values.size:
        background = float(np.median(annulus_values))
        mad = float(1.4826 * np.median(np.abs(annulus_values - background)))
        if not np.isfinite(mad) or mad <= np.finfo(np.float64).eps:
            mad = float(np.std(annulus_values))
        noise = max(mad, np.finfo(np.float64).eps)
    else:
        fallback = np.asarray(values[~base_mask], dtype=np.float64)
        if fallback.size == 0:
            return _SupportMetrics(None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None)
        background = float(np.median(fallback))
        noise = max(float(np.std(fallback)), np.finfo(np.float64).eps)
    annulus_count = int(annulus_values.size)
    total = _support_region_metrics(values, base_mask, aperture, background=background, noise=noise, annulus_count=annulus_count)
    shared_metrics = _support_region_metrics(values, base_mask, shared, background=background, noise=noise, annulus_count=annulus_count)
    exclusive_metrics = _support_region_metrics(values, base_mask, exclusive, background=background, noise=noise, annulus_count=annulus_count)
    partition_metrics = _support_region_metrics(values, base_mask, partition, background=background, noise=noise, annulus_count=annulus_count)
    total_count, total_net, total_positive, _total_snr = total
    shared_count, shared_net, shared_positive, _shared_snr = shared_metrics
    exclusive_count, exclusive_net, exclusive_positive, exclusive_snr = exclusive_metrics
    partition_count, partition_net, partition_positive, partition_snr = partition_metrics
    shared_fraction = (
        float(shared_positive / total_positive)
        if shared_positive is not None and total_positive is not None and total_positive > 0
        else None
    )
    return _SupportMetrics(
        background_adu=background,
        noise_adu=noise,
        total_valid_pixels=total_count,
        shared_valid_pixels=shared_count,
        exclusive_valid_pixels=exclusive_count,
        partition_valid_pixels=partition_count,
        total_net_flux_adu=total_net,
        shared_net_flux_adu=shared_net,
        exclusive_net_flux_adu=exclusive_net,
        partition_net_flux_adu=partition_net,
        total_positive_excess_adu=total_positive,
        shared_positive_excess_adu=shared_positive,
        exclusive_positive_excess_adu=exclusive_positive,
        partition_positive_excess_adu=partition_positive,
        shared_positive_fraction=shared_fraction,
        exclusive_snr=exclusive_snr,
        partition_snr=partition_snr,
    )


def _global_detection_sources(
    sources: Sequence[Detection],
    *,
    crop_origin: tuple[int, int],
) -> tuple[Detection, ...]:
    x0, y0 = crop_origin
    shifted: list[Detection] = []
    for source in sources:
        shifted.append(
            replace(
                source,
                x=float(source.x + x0),
                y=float(source.y + y0),
                peak_x=None if source.peak_x is None else float(source.peak_x + x0),
                peak_y=None if source.peak_y is None else float(source.peak_y + y0),
            )
        )
    return tuple(shifted)


def _assign_target_sources(
    sources: Sequence[Detection],
    targets: Sequence[tuple[float, float]],
    *,
    match_radius_px: float,
) -> tuple[tuple[int, int, float], ...]:
    """按最大基数/最小残差把局部候选分给两个目标。"""

    if len(targets) != 2:
        raise ValueError("targets must contain primary and secondary coordinates")
    if match_radius_px <= 0 or not np.isfinite(float(match_radius_px)):
        raise ValueError("match_radius_px must be positive and finite")
    candidates: list[tuple[float, int, int]] = []
    for detection_index, source in enumerate(sources):
        for target_index, target in enumerate(targets):
            distance = float(np.hypot(source.x - target[0], source.y - target[1]))
            if distance <= match_radius_px:
                candidates.append((distance, detection_index, target_index))
    assigned = _global_assignment(
        candidates,
        detection_count=len(sources),
        catalog_count=len(targets),
        radius_px=float(match_radius_px),
    )
    return tuple((int(target_index), int(detection_index), float(distance)) for distance, detection_index, target_index in assigned)


def _target_assignment(
    sources: Sequence[Detection],
    targets: Sequence[tuple[float, float]],
    *,
    match_radius_px: float,
) -> tuple[tuple[int, int, float] | None, tuple[int, int, float] | None]:
    assigned = _assign_target_sources(sources, targets, match_radius_px=match_radius_px)
    by_target = {
        target_index: (target_index, detection_index, distance)
        for target_index, detection_index, distance in assigned
    }
    return by_target.get(0), by_target.get(1)


def _flags_from_assignment(
    sources: Sequence[Detection],
    assignment: tuple[int, int, float] | None,
) -> tuple[str, ...]:
    if assignment is None:
        return ()
    return tuple(sources[assignment[1]].flags)


def _source_from_assignment(
    sources: Sequence[Detection],
    assignment: tuple[int, int, float] | None,
) -> Detection | None:
    return None if assignment is None else sources[assignment[1]]


def _support_kwargs(prefix: str, metrics: _SupportMetrics) -> dict[str, object]:
    return {
        f"{prefix}_background_adu": metrics.background_adu,
        f"{prefix}_noise_adu": metrics.noise_adu,
        f"{prefix}_total_valid_pixels": metrics.total_valid_pixels,
        f"{prefix}_shared_valid_pixels": metrics.shared_valid_pixels,
        f"{prefix}_exclusive_valid_pixels": metrics.exclusive_valid_pixels,
        f"{prefix}_partition_valid_pixels": metrics.partition_valid_pixels,
        f"{prefix}_total_net_flux_adu": metrics.total_net_flux_adu,
        f"{prefix}_shared_net_flux_adu": metrics.shared_net_flux_adu,
        f"{prefix}_exclusive_net_flux_adu": metrics.exclusive_net_flux_adu,
        f"{prefix}_partition_net_flux_adu": metrics.partition_net_flux_adu,
        f"{prefix}_total_positive_excess_adu": metrics.total_positive_excess_adu,
        f"{prefix}_shared_positive_excess_adu": metrics.shared_positive_excess_adu,
        f"{prefix}_exclusive_positive_excess_adu": metrics.exclusive_positive_excess_adu,
        f"{prefix}_partition_positive_excess_adu": metrics.partition_positive_excess_adu,
        f"{prefix}_shared_positive_fraction": metrics.shared_positive_fraction,
        f"{prefix}_exclusive_snr": metrics.exclusive_snr,
        f"{prefix}_partition_snr": metrics.partition_snr,
    }


def _resolve_frame_shifts(
    frame_count: int,
    frame_shifts: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, float], ...]:
    if frame_shifts is None:
        return tuple((0.0, 0.0) for _ in range(frame_count))
    if len(frame_shifts) != frame_count:
        raise ValueError("frame_shifts length must match frame paths")
    resolved = tuple((float(item[0]), float(item[1])) for item in frame_shifts)
    if not all(np.isfinite(value) for item in resolved for value in item):
        raise ValueError("frame_shifts must contain finite values")
    return resolved


def _resolve_mask_modes(mask_modes: Iterable[str]) -> tuple[str, ...]:
    resolved = tuple(dict.fromkeys(str(value).strip() for value in mask_modes if str(value).strip()))
    if not resolved:
        raise ValueError("mask_modes must contain at least one mode")
    unknown = sorted(set(resolved) - set(PAIR_SUPPORT_MASK_MODES))
    if unknown:
        raise ValueError(f"unknown pair support mask mode(s): {', '.join(unknown)}")
    return resolved


def _resolve_repeated_codes(
    template_sources: Sequence[Detection] | None,
    repeated_code_values: Iterable[int] | None,
) -> tuple[int, ...]:
    if repeated_code_values is not None:
        return tuple(sorted({int(value) for value in repeated_code_values}))
    if template_sources is None:
        return ()
    return tuple(sorted({int(code) for source in template_sources for code in source.repeated_code_values}))


def _row_note(
    *,
    mask_mode: str,
    primary_assignment: tuple[int, int, float] | None,
    secondary_assignment: tuple[int, int, float] | None,
    primary_metrics: _SupportMetrics,
    secondary_metrics: _SupportMetrics,
) -> str:
    if primary_assignment is None or secondary_assignment is None:
        assignment_note = "一对一局部重检未同时分配两个目标"
    else:
        assignment_note = "一对一局部重检同时分配了两个候选"
    fractions = [
        value
        for value in (primary_metrics.shared_positive_fraction, secondary_metrics.shared_positive_fraction)
        if value is not None
    ]
    fraction_note = (
        f"共同孔径正残差占比中位约 {float(np.median(fractions)):.1%}"
        if fractions
        else "共同孔径正残差占比不可计算"
    )
    if mask_mode == "raw":
        return f"{assignment_note}；{fraction_note}；raw 只作候选层基线"
    return f"{assignment_note}；{fraction_note}；掩膜敏感性是诊断证据，不是物理真值"


def run_pair_support_audit(
    frame_paths: Sequence[str | Path],
    primary_target_xy: Sequence[float],
    secondary_target_xy: Sequence[float],
    *,
    template_sources: Sequence[Detection] | None = None,
    primary_detection_id: int | None = None,
    secondary_detection_id: int | None = None,
    frame_shifts: Sequence[Sequence[float]] | None = None,
    mask_modes: Iterable[str] = PAIR_SUPPORT_MASK_MODES,
    repeated_code_values: Iterable[int] | None = None,
    crop_padding_px: float = 128.0,
    aperture_radius: int = 4,
    match_radius_px: float = 2.0,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    psf_fwhm: float = 2.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
    reject_linear_artifacts: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> PairSupportAuditResult:
    """在局部裁剪中执行双候选的留出支持审计。"""

    paths = tuple(Path(path) for path in frame_paths)
    if not paths:
        raise ValueError("frame_paths must contain at least one frame")
    primary = _xy(primary_target_xy, "primary_target_xy")
    secondary = _xy(secondary_target_xy, "secondary_target_xy")
    if primary == secondary:
        raise ValueError("primary and secondary target coordinates must differ")
    if aperture_radius < 1:
        raise ValueError("aperture_radius must be positive")
    if not np.isfinite(float(psf_fwhm)) or psf_fwhm <= 0:
        raise ValueError("psf_fwhm must be positive and finite")
    resolved_modes = _resolve_mask_modes(mask_modes)
    shifts = _resolve_frame_shifts(len(paths), frame_shifts)
    codes = _resolve_repeated_codes(template_sources, repeated_code_values)
    catalog_by_id = {source.detection_id: source for source in (template_sources or ())}
    rows: list[PairSupportAuditRow] = []

    for frame_number, (path, shift) in enumerate(zip(paths, shifts, strict=True), start=1):
        values, path_label = _frame_values(path)
        frame_positions = (
            (primary[0] + shift[0], primary[1] + shift[1]),
            (secondary[0] + shift[0], secondary[1] + shift[1]),
        )
        crop_x0, crop_y0, crop_x1, crop_y1 = _crop_bounds(values.shape, frame_positions, crop_padding_px)
        crop = np.asarray(values[crop_y0:crop_y1, crop_x0:crop_x1])
        negative_limit = _default_negative_overflow_limit(values)
        masks, local_negative_limit = _mask_components(
            crop,
            crop_origin=(crop_x0, crop_y0),
            positions=frame_positions,
            aperture_radius=aperture_radius,
            repeated_code_values=codes,
            negative_overflow_limit=negative_limit,
        )
        for mode in resolved_modes:
            mode_mask = masks[mode]
            detection = detect_sources(
                crop,
                mask=mode_mask,
                threshold_sigma=threshold_sigma,
                min_distance=min_distance,
                aperture_radius=aperture_radius,
                max_sources=None,
                negative_overflow_limit=negative_limit,
                psf_fwhm=psf_fwhm,
                background_box_size=background_box_size,
                min_flux_snr=min_flux_snr,
                min_psf_support_pixels=min_psf_support_pixels,
                mask_zero_pixels=False,
                allow_partial_zero_mask=False,
                reject_linear_artifacts=reject_linear_artifacts,
                proposal_mode=proposal_mode,
                enable_local_deblend=False,
            )
            global_sources = _global_detection_sources(detection.sources, crop_origin=(crop_x0, crop_y0))
            assignments = _target_assignment(global_sources, frame_positions, match_radius_px=match_radius_px)
            primary_assignment, secondary_assignment = assignments
            nearby_radius = max(float(match_radius_px), 5.0)
            nearby = [
                source
                for source in global_sources
                if min(
                    float(np.hypot(source.x - frame_positions[0][0], source.y - frame_positions[0][1])),
                    float(np.hypot(source.x - frame_positions[1][0], source.y - frame_positions[1][1])),
                )
                <= nearby_radius
            ]
            primary_metrics = _support_metrics(
                crop,
                mode_mask,
                frame_positions[0],
                frame_positions[1],
                crop_origin=(crop_x0, crop_y0),
                aperture_radius=aperture_radius,
            )
            secondary_metrics = _support_metrics(
                crop,
                mode_mask,
                frame_positions[1],
                frame_positions[0],
                crop_origin=(crop_x0, crop_y0),
                aperture_radius=aperture_radius,
            )
            primary_source = _source_from_assignment(global_sources, primary_assignment)
            secondary_source = _source_from_assignment(global_sources, secondary_assignment)
            primary_catalog = catalog_by_id.get(primary_detection_id) if primary_detection_id is not None else None
            secondary_catalog = catalog_by_id.get(secondary_detection_id) if secondary_detection_id is not None else None
            row_kwargs: dict[str, object] = {
                "frame_index": frame_number,
                "path": path_label,
                "frame_shift_x_px": shift[0],
                "frame_shift_y_px": shift[1],
                "mask_mode": mode,
                "crop_x0": crop_x0,
                "crop_y0": crop_y0,
                "crop_width": crop_x1 - crop_x0,
                "crop_height": crop_y1 - crop_y0,
                "masked_pixel_count": int(np.count_nonzero(mode_mask)),
                "local_negative_limit_adu": local_negative_limit,
                "candidate_count": int(detection.candidate_count),
                "quality_count": int(detection.star_count),
                "nearby_candidate_count": len(nearby),
                "nearby_quality_count": int(sum(source.quality_passed for source in nearby)),
                "assigned_candidate_count": int(sum(item is not None for item in assignments)),
                "primary_re_detected_id": None if primary_assignment is None else int(primary_source.detection_id),
                "secondary_re_detected_id": None if secondary_assignment is None else int(secondary_source.detection_id),
                "primary_match_distance_px": None if primary_assignment is None else primary_assignment[2],
                "secondary_match_distance_px": None if secondary_assignment is None else secondary_assignment[2],
                "primary_local_quality_passed": None if primary_source is None else bool(primary_source.quality_passed),
                "secondary_local_quality_passed": None if secondary_source is None else bool(secondary_source.quality_passed),
                "primary_local_flags": _flags_from_assignment(global_sources, primary_assignment),
                "secondary_local_flags": _flags_from_assignment(global_sources, secondary_assignment),
                "primary_catalog_quality_passed": None if primary_catalog is None else bool(primary_catalog.quality_passed),
                "secondary_catalog_quality_passed": None if secondary_catalog is None else bool(secondary_catalog.quality_passed),
                "primary_catalog_flags": () if primary_catalog is None else tuple(primary_catalog.flags),
                "secondary_catalog_flags": () if secondary_catalog is None else tuple(secondary_catalog.flags),
                "note": _row_note(
                    mask_mode=mode,
                    primary_assignment=primary_assignment,
                    secondary_assignment=secondary_assignment,
                    primary_metrics=primary_metrics,
                    secondary_metrics=secondary_metrics,
                ),
            }
            row_kwargs.update(_support_kwargs("primary", primary_metrics))
            row_kwargs.update(_support_kwargs("secondary", secondary_metrics))
            rows.append(PairSupportAuditRow(**row_kwargs))  # type: ignore[arg-type]
        if progress is not None:
            progress(frame_number, len(paths))

    raw_rows = [row for row in rows if row.mask_mode == "raw"]
    code_rows = [row for row in rows if row.mask_mode == "repeated_code_masked"]
    local_rows = [row for row in rows if row.mask_mode == "local_anomaly_masked"]
    raw_two = sum(row.assigned_candidate_count == 2 for row in raw_rows)
    code_two = sum(row.assigned_candidate_count == 2 for row in code_rows)
    local_two = sum(row.assigned_candidate_count == 2 for row in local_rows)
    raw_shared = [
        value
        for row in raw_rows
        for value in (row.primary_shared_positive_fraction, row.secondary_shared_positive_fraction)
        if value is not None
    ]
    shared_summary = (
        f"raw 共同孔径正残差占比中位 {float(np.median(raw_shared)):.1%}"
        if raw_shared
        else "raw 共同孔径正残差占比不可计算"
    )
    conclusion = (
        f"局部双候选留出审计完成：{len(paths)} 帧；raw 模式有两个一对一候选的帧数 "
        f"{raw_two}/{len(raw_rows)}，重复码屏蔽后 {code_two}/{len(code_rows)}，"
        f"局部负异常屏蔽后 {local_two}/{len(local_rows)}；{shared_summary}。"
        "这说明 raw 双计数对共同孔径和数据有效性控制的敏感性，不能直接解释为两颗恒星；"
        "共同孔径屏蔽会损伤真实双源核心，因此仍需用有效独占支持、PSF/星表和跨帧证据共同裁决。"
    )
    parameters = {
        "detector": "current detect_sources on local crop",
        "threshold_sigma": float(threshold_sigma),
        "min_distance": int(min_distance),
        "aperture_radius": int(aperture_radius),
        "psf_fwhm": float(psf_fwhm),
        "background_box_size": int(background_box_size),
        "min_flux_snr": float(min_flux_snr),
        "min_psf_support_pixels": int(min_psf_support_pixels),
        "proposal_mode": str(proposal_mode),
        "reject_linear_artifacts": bool(reject_linear_artifacts),
        "coordinate_note": "目标坐标为首帧 detector 质心，按 frame_shifts 平移；局部候选坐标已加回全图原点",
        "assignment_note": "每个候选最多分给一个目标；先最大化匹配数，再最小化距离",
        "support_note": "shared/exclusive 是几何分区；partition 在 union aperture 上按最近目标单计数",
        "raw_quality_boundary": "局部 raw 重检没有全幅重复码频次先验；catalog flags 单独保留",
        "interpretation_boundary": "掩膜后的候选数、通量分区和 SNR 是诊断量，不是恒星概率或物理真值",
    }
    return PairSupportAuditResult(
        frame_count=len(paths),
        primary_detection_id=primary_detection_id,
        secondary_detection_id=secondary_detection_id,
        primary_target_xy=primary,
        secondary_target_xy=secondary,
        frame_shifts=shifts,
        mask_modes=resolved_modes,
        repeated_code_values=codes,
        crop_padding_px=float(crop_padding_px),
        aperture_radius=int(aperture_radius),
        match_radius_px=float(match_radius_px),
        rows=tuple(rows),
        parameters=parameters,
        conclusion=conclusion,
    )


def _csv_value(value: object) -> object:
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    if isinstance(value, bool):
        return "True" if value else "False"
    return value


def _optional_median(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not numeric else float(np.median(numeric))


def summarize_pair_support(
    result: PairSupportAuditResult,
) -> tuple[dict[str, object], ...]:
    """按留出掩膜汇总逐帧 pair 支持，保留候选层/质量层的分母。

    ``assigned_two_count`` 只表示两个不同候选被一对一分配；
    ``quality_two_count`` 还要求这两个局部检测都通过质量层。共享孔径、
    独占区和 partition 的连续量是诊断统计，不能直接解释成真星概率。
    """

    summaries: list[dict[str, object]] = []
    for mode in result.mask_modes:
        rows = [row for row in result.rows if row.mask_mode == mode]
        frame_count = len(rows)
        assigned_two = sum(row.assigned_candidate_count == 2 for row in rows)
        assigned_one = sum(row.assigned_candidate_count == 1 for row in rows)
        assigned_zero = sum(row.assigned_candidate_count == 0 for row in rows)
        quality_two = sum(
            row.primary_local_quality_passed is True and row.secondary_local_quality_passed is True
            for row in rows
        )
        fraction = lambda count: float(count / frame_count) if frame_count else 0.0
        summaries.append(
            {
                "mask_mode": mode,
                "frame_count": frame_count,
                "assigned_two_count": assigned_two,
                "assigned_one_count": assigned_one,
                "assigned_zero_count": assigned_zero,
                "quality_two_count": quality_two,
                "assigned_two_fraction": fraction(assigned_two),
                "quality_two_fraction": fraction(quality_two),
                "masked_pixel_median": _optional_median(row.masked_pixel_count for row in rows),
                "primary_shared_positive_fraction_median": _optional_median(
                    row.primary_shared_positive_fraction for row in rows
                ),
                "secondary_shared_positive_fraction_median": _optional_median(
                    row.secondary_shared_positive_fraction for row in rows
                ),
                "primary_exclusive_snr_median": _optional_median(row.primary_exclusive_snr for row in rows),
                "secondary_exclusive_snr_median": _optional_median(row.secondary_exclusive_snr for row in rows),
                "primary_partition_snr_median": _optional_median(row.primary_partition_snr for row in rows),
                "secondary_partition_snr_median": _optional_median(row.secondary_partition_snr for row in rows),
            }
        )
    return tuple(summaries)


def write_pair_support_artifacts(result: PairSupportAuditResult, out_dir: str | Path) -> Path:
    """写入逐掩膜 CSV 和可复现 JSON。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(PairSupportAuditRow)]
    with (output / "pair_support_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.as_dict().items()}
            for row in result.rows
        )
    summary_rows = summarize_pair_support(result)
    with (output / "pair_support_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PAIR_SUPPORT_SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = result.as_dict()
    payload["summary"] = list(summary_rows)
    payload["outputs"] = ["pair_support_audit.csv", "pair_support_summary.csv", "pair_support_audit.json"]
    payload["note"] = (
        "该审计把局部重检、共同/非共同孔径和留出掩膜作为相互独立的控制；"
        "它不修改默认候选表，也不把屏蔽后消失直接解释为噪点。"
    )
    (output / "pair_support_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


__all__ = [
    "PAIR_SUPPORT_MASK_MODES",
    "PAIR_SUPPORT_SUMMARY_FIELDS",
    "PairSupportAuditResult",
    "PairSupportAuditRow",
    "run_pair_support_audit",
    "summarize_pair_support",
    "write_pair_support_artifacts",
]
