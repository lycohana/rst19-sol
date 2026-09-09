"""符号反相的源级阴性对照。

本模块把真实图像相对于稳健背景取反，再用完全相同的检测器参数运行一遍。
它回答的是一个受限问题：如果背景/噪声中的负向波动在符号上近似对称，检测器
会把多少这样的结构提出为候选？结果是 ``signed-tail leakage`` 诊断，不是
FDR、p 值、precision，也不能代替热像素、宇宙线和亮线等正向伪影的独立审计。

该命令只写研究汇总表，不写源目录、不修改 GUI 默认值，也不写项目缓存。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import ndimage

from .detection import Detection, DetectionResult, detect_sources, sigma_clipped_stats
from .experiments import summarize_source_features
from .fits import auxiliary_mask, read_fits
from .models import FitsFrame


SIGNED_NULL_SNR_THRESHOLDS: tuple[float, ...] = (4.0, 5.0, 7.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True, slots=True)
class SignedNullThresholdRow:
    """正向/反相结果在一个 SNR 下限上的计数对照。"""

    threshold: float
    positive_filter_count: int
    negative_filter_count: int
    positive_flux_count: int
    negative_flux_count: int
    filter_leakage_ratio: float | None
    flux_leakage_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "positive_filter_count": self.positive_filter_count,
            "negative_filter_count": self.negative_filter_count,
            "positive_flux_count": self.positive_flux_count,
            "negative_flux_count": self.negative_flux_count,
            "filter_leakage_ratio": self.filter_leakage_ratio,
            "flux_leakage_ratio": self.flux_leakage_ratio,
        }


@dataclass(frozen=True, slots=True)
class SignedNullFeatureRow:
    """一个控制方向和一个首要形态类别的源级汇总。"""

    control: str
    feature_class: str
    feature_class_label: str
    candidate_count: int
    quality_count: int
    quality_fraction: float | None
    median_flux_snr: float | None
    max_flux_snr: float | None
    common_flags: str

    def as_dict(self) -> dict[str, object]:
        return {
            "control": self.control,
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "candidate_count": self.candidate_count,
            "quality_count": self.quality_count,
            "quality_fraction": self.quality_fraction,
            "median_flux_snr": self.median_flux_snr,
            "max_flux_snr": self.max_flux_snr,
            "common_flags": self.common_flags,
        }


@dataclass(frozen=True, slots=True)
class SignedNullSourceRow:
    """反相侧通过质量层源的坐标、形态和原始值域局部证据。"""

    detection_id: int
    x: float
    y: float
    peak_x: float
    peak_y: float
    peak: float
    flux_snr: float | None
    filter_snr: float | None
    fwhm: float | None
    psf_support_pixels: int | None
    raw_peak_adu: float
    raw_negative_pixel_count: int
    raw_negative_anomaly_pixel_count: int
    raw_extreme_negative_pixel_count: int
    raw_minus_one_count: int
    raw_min_adu: float | None
    raw_max_adu: float | None
    range_transfer_overlap: bool
    raw_evidence_layer: str
    flags: str

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "peak_x": self.peak_x,
            "peak_y": self.peak_y,
            "peak": self.peak,
            "flux_snr": self.flux_snr,
            "filter_snr": self.filter_snr,
            "fwhm": self.fwhm,
            "psf_support_pixels": self.psf_support_pixels,
            "raw_peak_adu": self.raw_peak_adu,
            "raw_negative_pixel_count": self.raw_negative_pixel_count,
            "raw_negative_anomaly_pixel_count": self.raw_negative_anomaly_pixel_count,
            "raw_extreme_negative_pixel_count": self.raw_extreme_negative_pixel_count,
            "raw_minus_one_count": self.raw_minus_one_count,
            "raw_min_adu": self.raw_min_adu,
            "raw_max_adu": self.raw_max_adu,
            "range_transfer_overlap": self.range_transfer_overlap,
            "raw_evidence_layer": self.raw_evidence_layer,
            "flags": self.flags,
        }


@dataclass(frozen=True, slots=True)
class SignedNullAuditResult:
    """符号反相阴性对照的完整、可复现汇总。"""

    frame_path: str
    image_shape: tuple[int, int]
    background_adu: float
    positive_noise_adu: float
    negative_noise_adu: float
    positive_candidate_count: int
    positive_quality_count: int
    negative_candidate_count: int
    negative_quality_count: int
    negative_quality_count_after_range_exclusion: int
    candidate_leakage_ratio: float | None
    quality_leakage_ratio: float | None
    quality_leakage_ratio_after_range_exclusion: float | None
    negative_range_transfer_candidate_count: int
    negative_range_transfer_quality_count: int
    thresholds: tuple[SignedNullThresholdRow, ...]
    feature_rows: tuple[SignedNullFeatureRow, ...]
    parameters: dict[str, object]
    conclusion: str
    negative_quality_sources: tuple[SignedNullSourceRow, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_path": self.frame_path,
            "image_shape": list(self.image_shape),
            "background_adu": self.background_adu,
            "positive_noise_adu": self.positive_noise_adu,
            "negative_noise_adu": self.negative_noise_adu,
            "positive_candidate_count": self.positive_candidate_count,
            "positive_quality_count": self.positive_quality_count,
            "negative_candidate_count": self.negative_candidate_count,
            "negative_quality_count": self.negative_quality_count,
            "negative_quality_count_after_range_exclusion": self.negative_quality_count_after_range_exclusion,
            "candidate_leakage_ratio": self.candidate_leakage_ratio,
            "quality_leakage_ratio": self.quality_leakage_ratio,
            "quality_leakage_ratio_after_range_exclusion": self.quality_leakage_ratio_after_range_exclusion,
            "negative_range_transfer_candidate_count": self.negative_range_transfer_candidate_count,
            "negative_range_transfer_quality_count": self.negative_range_transfer_quality_count,
            "thresholds": [row.as_dict() for row in self.thresholds],
            "feature_rows": [row.as_dict() for row in self.feature_rows],
            "negative_quality_sources": [row.as_dict() for row in self.negative_quality_sources],
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        }


def _finite_snr_values(sources: Iterable[Detection], field: str) -> np.ndarray:
    """取出某个源字段中的有限 SNR 值，忽略缺失和非有限值。"""

    if field not in {"filter_snr", "flux_snr"}:
        raise ValueError(f"unsupported SNR field: {field!r}")
    values = [
        float(value)
        for source in sources
        if (value := getattr(source, field, None)) is not None and np.isfinite(value)
    ]
    return np.asarray(values, dtype=np.float64)


def _count_at_least(values: np.ndarray, threshold: float) -> int:
    """统计 SNR 大于等于阈值的源数。"""

    return int(np.count_nonzero(np.asarray(values, dtype=np.float64) >= float(threshold)))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """分母为零时返回 ``None``，不把无定义比值写成 0。"""

    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _signed_integer_range_transfer_mask(
    values: np.ndarray,
    aperture_radius: int,
) -> tuple[np.ndarray, int | None, int]:
    """找出反相会把原始有符号整型负侧异常变成亮正源的区域。

    ``detect_sources`` 对真实有符号整型使用负侧 90% 满量程作为数据有效性
    审计线。符号反相会把这些像素映射到正方向；这里仅生成一个研究标签，
    不把它混入负向检测的主掩膜，因而原始反相计数仍然可复现。
    """

    array = np.asarray(values)
    empty = np.zeros(array.shape, dtype=bool)
    dtype = array.dtype
    if not np.issubdtype(dtype, np.signedinteger):
        return empty, None, 0
    info = np.iinfo(dtype)
    if info.bits < 16:
        return empty, None, 0
    limit = -0.9 * float(info.max)
    invalid = np.isfinite(array) & (array <= limit)
    if not np.any(invalid):
        return empty, limit, 0
    axis = np.arange(-aperture_radius, aperture_radius + 1, dtype=np.int32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    structure = (xx * xx + yy * yy) <= int(aperture_radius * aperture_radius)
    expanded = ndimage.binary_dilation(invalid, structure=structure)
    return np.asarray(expanded, dtype=bool), limit, int(np.count_nonzero(invalid))


def _source_hits_mask(source: Detection, mask: np.ndarray) -> bool:
    """按峰值锚点判断候选是否落在一个预先扩张的数据异常区内。"""

    x_value = source.peak_x if source.peak_x is not None else source.x
    y_value = source.peak_y if source.peak_y is not None else source.y
    x = int(round(float(x_value)))
    y = int(round(float(y_value)))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _raw_evidence_layer(
    *,
    range_transfer_overlap: bool,
    raw_negative_pixel_count: int,
    raw_negative_anomaly_pixel_count: int,
    raw_minus_one_count: int,
) -> str:
    """为反相质量源生成互斥的原始值域审计层标签。

    标签只描述当前孔径中观察到的值域证据，不表示硬件故障、噪声概率或
    物理源身份。完整计数仍保留在源表中，避免互斥标签隐藏重叠证据。
    """

    if range_transfer_overlap:
        return "extreme_range_transfer"
    if raw_negative_anomaly_pixel_count > 0:
        return "raw_negative_anomaly"
    if raw_minus_one_count > 0:
        return "fixed_sentinel_candidate"
    if raw_negative_pixel_count > 0:
        return "raw_negative_other"
    return "no_negative_evidence"


def _raw_aperture_summary(
    values: np.ndarray,
    mask: np.ndarray,
    source: Detection,
    *,
    aperture_radius: int,
    negative_anomaly_limit: float,
    range_transfer_limit: float | None,
) -> dict[str, object]:
    """汇总反相质量源对应原始孔径的负值证据，不复制完整源目录。"""

    x_value = source.peak_x if source.peak_x is not None else source.x
    y_value = source.peak_y if source.peak_y is not None else source.y
    x_peak = int(round(float(x_value)))
    y_peak = int(round(float(y_value)))
    numeric = np.asarray(values, dtype=np.float64)
    height, width = numeric.shape
    y0 = max(0, y_peak - aperture_radius)
    y1 = min(height, y_peak + aperture_radius + 1)
    x0 = max(0, x_peak - aperture_radius)
    x1 = min(width, x_peak + aperture_radius + 1)
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)
    geometric = np.hypot(xx - (x_peak - x0), yy - (y_peak - y0)) <= aperture_radius
    local_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    local_values = numeric[y0:y1, x0:x1]
    valid = geometric & ~local_mask & np.isfinite(local_values)
    aperture_values = local_values[valid]
    if aperture_values.size == 0:
        raw_peak = float("nan")
        raw_min = raw_max = None
        negative_count = anomaly_count = extreme_count = minus_one_count = 0
    else:
        raw_peak = float(numeric[y_peak, x_peak]) if 0 <= y_peak < height and 0 <= x_peak < width else float("nan")
        raw_min = float(np.min(aperture_values))
        raw_max = float(np.max(aperture_values))
        negative_count = int(np.count_nonzero(aperture_values < 0.0))
        anomaly_count = int(np.count_nonzero(aperture_values <= float(negative_anomaly_limit)))
        extreme_count = (
            int(np.count_nonzero(aperture_values <= float(range_transfer_limit)))
            if range_transfer_limit is not None
            else 0
        )
        minus_one_count = int(np.count_nonzero(aperture_values == -1.0))
    return {
        "peak_x": float(x_peak),
        "peak_y": float(y_peak),
        "raw_peak_adu": raw_peak,
        "raw_negative_pixel_count": negative_count,
        "raw_negative_anomaly_pixel_count": anomaly_count,
        "raw_extreme_negative_pixel_count": extreme_count,
        "raw_minus_one_count": minus_one_count,
        "raw_min_adu": raw_min,
        "raw_max_adu": raw_max,
    }


def _negative_quality_source_rows(
    sources: Sequence[Detection],
    values: np.ndarray,
    mask: np.ndarray,
    range_transfer_mask: np.ndarray,
    *,
    aperture_radius: int,
    background: float,
    noise: float,
    range_transfer_limit: float | None,
) -> tuple[SignedNullSourceRow, ...]:
    """只保留反相质量源的局部证据，便于审计跨帧固定坐标重复。"""

    negative_anomaly_limit = min(-1000.0, float(background) - 50.0 * max(float(noise), 1e-12))
    rows: list[SignedNullSourceRow] = []
    for source in sources:
        if not source.quality_passed:
            continue
        raw = _raw_aperture_summary(
            values,
            mask,
            source,
            aperture_radius=aperture_radius,
            negative_anomaly_limit=negative_anomaly_limit,
            range_transfer_limit=range_transfer_limit,
        )
        peak_x = float(raw["peak_x"])
        peak_y = float(raw["peak_y"])
        raw_negative_pixel_count = int(raw["raw_negative_pixel_count"])
        raw_negative_anomaly_pixel_count = int(raw["raw_negative_anomaly_pixel_count"])
        raw_minus_one_count = int(raw["raw_minus_one_count"])
        range_transfer_overlap = _source_hits_mask(source, range_transfer_mask)
        rows.append(
            SignedNullSourceRow(
                detection_id=int(source.detection_id),
                x=float(source.x),
                y=float(source.y),
                peak_x=peak_x,
                peak_y=peak_y,
                peak=float(source.peak),
                flux_snr=None if source.flux_snr is None else float(source.flux_snr),
                filter_snr=None if source.filter_snr is None else float(source.filter_snr),
                fwhm=None if source.fwhm is None else float(source.fwhm),
                psf_support_pixels=(
                    None if source.psf_support_pixels is None else int(source.psf_support_pixels)
                ),
                raw_peak_adu=float(raw["raw_peak_adu"]),
                raw_negative_pixel_count=raw_negative_pixel_count,
                raw_negative_anomaly_pixel_count=raw_negative_anomaly_pixel_count,
                raw_extreme_negative_pixel_count=int(raw["raw_extreme_negative_pixel_count"]),
                raw_minus_one_count=raw_minus_one_count,
                raw_min_adu=None if raw["raw_min_adu"] is None else float(raw["raw_min_adu"]),
                raw_max_adu=None if raw["raw_max_adu"] is None else float(raw["raw_max_adu"]),
                range_transfer_overlap=range_transfer_overlap,
                raw_evidence_layer=_raw_evidence_layer(
                    range_transfer_overlap=range_transfer_overlap,
                    raw_negative_pixel_count=raw_negative_pixel_count,
                    raw_negative_anomaly_pixel_count=raw_negative_anomaly_pixel_count,
                    raw_minus_one_count=raw_minus_one_count,
                ),
                flags="|".join(source.flags),
            )
        )
    return tuple(rows)


def _feature_rows(control: str, sources: Sequence[Detection]) -> tuple[SignedNullFeatureRow, ...]:
    """把现有互斥首要类别汇总转换为符号反相专用表。"""

    if control not in {"positive", "sign_flipped"}:
        raise ValueError(f"unsupported control: {control!r}")
    return tuple(
        SignedNullFeatureRow(
            control=control,
            feature_class=str(row["feature_class"]),
            feature_class_label=str(row["feature_class_label"]),
            candidate_count=int(row["candidate_count"]),
            quality_count=int(row["quality_count"]),
            quality_fraction=(None if row["quality_fraction"] is None else float(row["quality_fraction"])),
            median_flux_snr=(None if row["median_flux_snr"] is None else float(row["median_flux_snr"])),
            max_flux_snr=(None if row["max_flux_snr"] is None else float(row["max_flux_snr"])),
            common_flags=str(row["common_flags"]),
        )
        for row in summarize_source_features(sources)
    )


def _jsonable(value: object) -> object:
    """将检测器参数中的 NumPy 标量递归转换为 JSON 可写值。"""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _resolve_frame(
    frame: FitsFrame | np.ndarray | str | Path,
) -> tuple[str, np.ndarray, np.ndarray]:
    """统一路径、已读 FITS 和数组输入，并生成辅助区掩膜。"""

    if isinstance(frame, FitsFrame):
        frame_path = str(frame.path)
        values = np.asarray(frame.data)
    elif isinstance(frame, (str, Path)):
        loaded = read_fits(frame)
        frame_path = str(loaded.path)
        values = np.asarray(loaded.data)
    else:
        frame_path = "<array>"
        values = np.asarray(frame)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("image must not be empty")
    try:
        np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("image must contain numeric values") from exc
    return frame_path, values, auxiliary_mask((int(values.shape[0]), int(values.shape[1])))


def _validate_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in thresholds)
    if not values:
        raise ValueError("snr_thresholds must not be empty")
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("snr_thresholds must contain finite positive values")
    if tuple(sorted(set(values))) != values:
        raise ValueError("snr_thresholds must be strictly increasing")
    return values


def run_signed_null_audit(
    frame: FitsFrame | np.ndarray | str | Path,
    *,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 2.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
    reject_linear_artifacts: bool = True,
    snr_thresholds: Sequence[float] = SIGNED_NULL_SNR_THRESHOLDS,
    progress: Callable[[float, str], None] | None = None,
) -> SignedNullAuditResult:
    """在同一帧上运行正向和符号反相检测，输出源级泄漏诊断。"""

    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")
    if min_distance < 1 or aperture_radius < 1:
        raise ValueError("min_distance and aperture_radius must be positive")
    if psf_fwhm <= 0 or background_box_size < 16 or min_flux_snr <= 0:
        raise ValueError("psf_fwhm and min_flux_snr must be positive; background_box_size must be at least 16")
    if not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_psf_support_pixels must be between 1 and 9")
    thresholds = _validate_thresholds(snr_thresholds)
    frame_path, values, mask = _resolve_frame(frame)
    numeric = np.asarray(values, dtype=np.float64)
    range_transfer_mask, range_transfer_limit, range_transfer_pixel_count = _signed_integer_range_transfer_mask(
        values,
        int(aperture_radius),
    )
    background, _global_noise = sigma_clipped_stats(
        numeric,
        mask=mask,
        sample_limit=1_000_000,
    )

    def report(value: float, label: str) -> None:
        if progress is not None:
            progress(max(0.0, min(100.0, float(value))), label)

    def mapped_progress(start: float, end: float, prefix: str) -> Callable[[float, str], None]:
        def callback(value: float, label: str) -> None:
            mapped = start + (end - start) * max(0.0, min(100.0, float(value))) / 100.0
            report(mapped, f"{prefix}{label}")

        return callback

    detector_kwargs: dict[str, object] = {
        "mask": mask,
        "threshold_sigma": float(threshold_sigma),
        "min_distance": int(min_distance),
        "aperture_radius": int(aperture_radius),
        "psf_fwhm": float(psf_fwhm),
        "background_box_size": int(background_box_size),
        "min_flux_snr": float(min_flux_snr),
        "min_psf_support_pixels": int(min_psf_support_pixels),
        "proposal_mode": str(proposal_mode),
        "reject_linear_artifacts": bool(reject_linear_artifacts),
    }
    report(1.0, "估计符号反相对照背景")
    positive = detect_sources(
        values,
        **detector_kwargs,
        progress=mapped_progress(5.0, 48.0, "正向 · "),
    )
    report(50.0, "构造背景对称的符号反相图")
    mirrored = 2.0 * float(background) - numeric
    negative = detect_sources(
        mirrored,
        **detector_kwargs,
        progress=mapped_progress(52.0, 96.0, "符号反相 · "),
    )
    report(98.0, "汇总 SNR 与特征类别")

    positive_filter = _finite_snr_values(positive.sources, "filter_snr")
    negative_filter = _finite_snr_values(negative.sources, "filter_snr")
    positive_flux = _finite_snr_values(positive.sources, "flux_snr")
    negative_flux = _finite_snr_values(negative.sources, "flux_snr")
    threshold_rows = tuple(
        SignedNullThresholdRow(
            threshold=threshold,
            positive_filter_count=_count_at_least(positive_filter, threshold),
            negative_filter_count=_count_at_least(negative_filter, threshold),
            positive_flux_count=_count_at_least(positive_flux, threshold),
            negative_flux_count=_count_at_least(negative_flux, threshold),
            filter_leakage_ratio=_safe_ratio(
                _count_at_least(negative_filter, threshold),
                _count_at_least(positive_filter, threshold),
            ),
            flux_leakage_ratio=_safe_ratio(
                _count_at_least(negative_flux, threshold),
                _count_at_least(positive_flux, threshold),
            ),
        )
        for threshold in thresholds
    )
    candidate_leakage_ratio = _safe_ratio(negative.candidate_count, positive.candidate_count)
    quality_leakage_ratio = _safe_ratio(negative.star_count, positive.star_count)
    range_transfer_sources = tuple(
        source for source in negative.sources if _source_hits_mask(source, range_transfer_mask)
    )
    range_transfer_quality_count = sum(bool(source.quality_passed) for source in range_transfer_sources)
    negative_quality_count_after_range_exclusion = max(0, negative.star_count - range_transfer_quality_count)
    quality_leakage_ratio_after_range_exclusion = _safe_ratio(
        negative_quality_count_after_range_exclusion,
        positive.star_count,
    )
    negative_quality_sources = _negative_quality_source_rows(
        negative.sources,
        values,
        mask,
        range_transfer_mask,
        aperture_radius=int(aperture_radius),
        background=float(background),
        noise=float(positive.noise),
        range_transfer_limit=range_transfer_limit,
    )
    parameters = {
        "thresholds": list(thresholds),
        "detector": _jsonable(detector_kwargs),
        "positive_detector_parameters": _jsonable(positive.parameters),
        "negative_detector_parameters": _jsonable(negative.parameters),
        "background_estimator": "sigma_clipped_stats(mask=auxiliary_mask, sample_limit=1000000)",
        "mirror_definition": "I_mirror = 2 * B_global - I",
        "mask_policy": "FITS first-row auxiliary bytes are masked; detector additionally masks non-finite values",
        "source_count_definition": (
            "candidate_count is post-NMS proposal count; quality_count is source-level quality pass"
        ),
        "diagnostic_name": "signed-tail leakage",
        "not_formal_fdr": True,
        "negative_range_transfer": {
            "enabled": range_transfer_limit is not None,
            "original_negative_overflow_limit_adu": range_transfer_limit,
            "original_extreme_negative_pixel_count": range_transfer_pixel_count,
            "expanded_mask_radius_px": int(aperture_radius),
            "expanded_mask_pixel_count": int(np.count_nonzero(range_transfer_mask)),
            "candidate_overlap_count": len(range_transfer_sources),
            "quality_overlap_count": int(range_transfer_quality_count),
            "effective_quality_count_after_exclusion": int(negative_quality_count_after_range_exclusion),
            "label": "original signed-integer negative-range response transferred to positive by mirroring",
        },
        "limitations": [
            "符号反相只约束近似对称的背景/噪声尾部，不是 FDR、p 值、precision 或真实源概率。",
            "正向热像素、宇宙线、亮线、饱和和非对称值域结构不会被该对照完整模拟。",
            "反相图使用 float64，因此原始有符号整型的负侧满量程编码审计不等价于真实帧。",
        ],
    }
    conclusion = (
        "符号反相只约束可近似镜像的背景/噪声尾部；正负响应比是泄漏诊断，"
        "不是 FDR、p 值或真实源概率。正向热像素、宇宙线、亮线和非对称值域结构"
        "仍需通过类别、PSF、跨帧和星表/注入证据独立确认。"
    )
    report(100.0, "符号反相阴性对照完成")
    return SignedNullAuditResult(
        frame_path=frame_path,
        image_shape=(int(values.shape[0]), int(values.shape[1])),
        background_adu=float(background),
        positive_noise_adu=float(positive.noise),
        negative_noise_adu=float(negative.noise),
        positive_candidate_count=int(positive.candidate_count),
        positive_quality_count=int(positive.star_count),
        negative_candidate_count=int(negative.candidate_count),
        negative_quality_count=int(negative.star_count),
        negative_quality_count_after_range_exclusion=int(negative_quality_count_after_range_exclusion),
        candidate_leakage_ratio=candidate_leakage_ratio,
        quality_leakage_ratio=quality_leakage_ratio,
        quality_leakage_ratio_after_range_exclusion=quality_leakage_ratio_after_range_exclusion,
        negative_range_transfer_candidate_count=len(range_transfer_sources),
        negative_range_transfer_quality_count=int(range_transfer_quality_count),
        thresholds=threshold_rows,
        feature_rows=_feature_rows("positive", positive.sources) + _feature_rows("sign_flipped", negative.sources),
        parameters=parameters,
        conclusion=conclusion,
        negative_quality_sources=negative_quality_sources,
    )


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = tuple(rows[0].keys()) if rows else ()
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_signed_null_artifacts(result: SignedNullAuditResult, out_dir: str | Path) -> Path:
    """写入符号反相汇总 JSON、SNR 分档表和特征类别表。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "signed_null_summary.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_rows(output / "signed_null_thresholds.csv", [row.as_dict() for row in result.thresholds])
    _write_rows(output / "signed_null_feature_counts.csv", [row.as_dict() for row in result.feature_rows])
    return output
