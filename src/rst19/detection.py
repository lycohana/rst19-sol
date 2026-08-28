"""可解释的点源检测、局部噪声估计和候选质量判定。

检测结果明确分成两层：``candidate_count`` 是匹配滤波后经过非极大值
抑制的候选峰数量，``quality_count`` 是通过局部测光、信噪比、点源形状、
边缘、掩膜和饱和检查的数量。两者都保留，避免把“算法产生的候选数”
误写成未经验证的物理恒星真值。

本模块不依赖 Photutils，方便比赛现场离线复现。默认检测核是圆对称
Gaussian，适合作为当前数据的可解释基线；若数据证明存在拖影、畸变或
明显非高斯 PSF，应替换成实测 PSF，而不是继续调一个数量目标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class Detection:
    """单个检测源的可解释属性。

    坐标以图像左上角为原点，x 向右、y 向下。``snr`` 是峰值相对局部
    背景的 SNR，``flux_snr`` 是孔径净通量的 SNR；两者不能混用。旧版
    结果只有峰值 SNR 时，``flux_snr`` 可以为 ``None``。
    """

    detection_id: int
    x: float
    y: float
    peak: float
    flux: float
    background: float
    noise: float
    snr: float
    fwhm: float | None
    flags: tuple[str, ...]
    flux_error: float | None = None
    flux_snr: float | None = None
    filter_snr: float | None = None
    fwhm_x: float | None = None
    fwhm_y: float | None = None
    ellipticity: float | None = None
    sharpness: float | None = None
    footprint_pixels: int | None = None
    quality_passed: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "peak": self.peak,
            "flux": self.flux,
            "background": self.background,
            "noise": self.noise,
            "snr": self.snr,
            "flux_error": self.flux_error,
            "flux_snr": self.flux_snr,
            "filter_snr": self.filter_snr,
            "fwhm": self.fwhm,
            "fwhm_x": self.fwhm_x,
            "fwhm_y": self.fwhm_y,
            "ellipticity": self.ellipticity,
            "sharpness": self.sharpness,
            "footprint_pixels": self.footprint_pixels,
            "quality_passed": self.quality_passed,
            "flags": list(self.flags),
        }


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """一帧图像的检测结果和用于复现的参数。"""

    image_shape: tuple[int, int]
    background: float
    noise: float
    threshold: float
    candidate_count: int
    sources: tuple[Detection, ...]
    parameters: dict[str, float | int]
    quality_count: int | None = None

    @property
    def star_count(self) -> int:
        """通过当前质量规则的星点数，不等同于原始候选峰数。"""

        return self.quality_count if self.quality_count is not None else sum(source.quality_passed for source in self.sources)

    @property
    def quality_sources(self) -> tuple[Detection, ...]:
        """返回通过质量判定的源；原始候选仍保存在 ``sources`` 中。"""

        return tuple(source for source in self.sources if source.quality_passed)

    @property
    def returned_count(self) -> int:
        """实际返回并完成属性计算的候选源数。"""

        return len(self.sources)

    @property
    def rejected_count(self) -> int:
        """已返回但未通过质量判定的候选数。"""

        return self.returned_count - self.star_count

    @property
    def truncated(self) -> bool:
        """是否因为显式 ``max_sources`` 而少返回了候选源。"""

        return self.candidate_count > self.returned_count

    def as_dict(self) -> dict[str, object]:
        return {
            "image_shape": list(self.image_shape),
            "background": self.background,
            "noise": self.noise,
            "threshold": self.threshold,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "quality_count": self.star_count,
            "rejected_count": self.rejected_count,
            "truncated": self.truncated,
            "parameters": self.parameters,
            "sources": [source.as_dict() for source in self.sources],
        }


def _finite_values(image: np.ndarray, mask: np.ndarray | None, sample_limit: int) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= ~mask
    flat = values[valid]
    if flat.size > sample_limit:
        stride = max(1, flat.size // sample_limit)
        flat = flat[::stride]
    return flat


def sigma_clipped_stats(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float = 3.0,
    iterations: int = 4,
    sample_limit: int = 1_000_000,
) -> tuple[float, float]:
    """用中位数和 MAD 估计背景位置与噪声尺度。"""

    values = _finite_values(image, mask, sample_limit)
    if values.size == 0:
        raise ValueError("image has no valid pixels for background estimation")
    selected = values
    for _ in range(max(1, iterations)):
        center = float(np.median(selected))
        mad = float(np.median(np.abs(selected - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(selected))
        if not np.isfinite(scale) or scale <= 0:
            scale = np.finfo(np.float64).eps
        keep = np.abs(selected - center) <= sigma * scale
        if keep.all():
            break
        selected = selected[keep]
        if selected.size == 0:
            break
    center = float(np.median(selected))
    mad = float(np.median(np.abs(selected - center)))
    scale = max(1.4826 * mad, float(np.std(selected)), np.finfo(np.float64).eps)
    return center, scale


def _resize_grid(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把块统计网格线性插值到图像大小。"""

    height, width = shape
    if grid.shape == (1, 1):
        return np.full(shape, float(grid[0, 0]), dtype=np.float64)
    zoom = (
        (height - 1) / max(1, grid.shape[0] - 1),
        (width - 1) / max(1, grid.shape[1] - 1),
    )
    resized = ndimage.zoom(grid, zoom=zoom, order=1, mode="nearest", prefilter=False)
    if resized.shape != shape:
        result = np.empty(shape, dtype=np.float64)
        result[...] = resized[-1, -1]
        result[: min(height, resized.shape[0]), : min(width, resized.shape[1])] = resized[:height, :width]
        return result
    return resized


def local_background_rms(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    box_size: int = 128,
    fallback: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """用网格化 sigma-clipping 生成二维背景和 RMS 图。

    块统计避免把一个亮星的像素直接当作背景；线性插值只用于把块统计
    变成逐像素权重。它不是对复杂散射光场的最终模型，但比单一全图
    标量更能解释局部 SNR。
    """

    if box_size < 16:
        raise ValueError("background box_size must be at least 16 pixels")
    values = np.asarray(image, dtype=np.float64)
    height, width = values.shape
    rows = range(0, height, box_size)
    cols = range(0, width, box_size)
    global_stats = fallback or sigma_clipped_stats(values, mask=mask)
    backgrounds: list[list[float]] = []
    noises: list[list[float]] = []
    for y0 in rows:
        background_row: list[float] = []
        noise_row: list[float] = []
        for x0 in cols:
            block = values[y0 : y0 + box_size, x0 : x0 + box_size]
            block_mask = mask[y0 : y0 + box_size, x0 : x0 + box_size]
            try:
                block_stats = sigma_clipped_stats(block, mask=block_mask, sample_limit=100_000)
            except ValueError:
                block_stats = global_stats
            if not np.isfinite(block_stats[0]) or not np.isfinite(block_stats[1]) or block_stats[1] <= 0:
                block_stats = global_stats
            background_row.append(float(block_stats[0]))
            noise_row.append(float(block_stats[1]))
        backgrounds.append(background_row)
        noises.append(noise_row)
    background_map = _resize_grid(np.asarray(backgrounds), values.shape)
    noise_map = _resize_grid(np.asarray(noises), values.shape)
    noise_map = np.maximum(noise_map, np.finfo(np.float64).eps)
    return background_map, noise_map


def _suppress_close_candidates(candidates: Iterable[tuple[int, int, float]], min_distance: int) -> list[tuple[int, int, float]]:
    """按峰值从高到低做网格化非极大值抑制，避免 O(n²) 全量比较。"""

    if min_distance < 1:
        return list(candidates)
    cell_size = float(min_distance)
    accepted: list[tuple[int, int, float]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    distance_sq = float(min_distance * min_distance)
    for x, y, peak in candidates:
        cell = (int(x // cell_size), int(y // cell_size))
        too_close = False
        for gx in range(cell[0] - 1, cell[0] + 2):
            for gy in range(cell[1] - 1, cell[1] + 2):
                for accepted_index in grid.get((gx, gy), ()):
                    ax, ay, _ = accepted[accepted_index]
                    if (x - ax) ** 2 + (y - ay) ** 2 < distance_sq:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if not too_close:
            grid.setdefault(cell, []).append(len(accepted))
            accepted.append((x, y, peak))
    return accepted


def _linear_artifact_mask(
    residual: np.ndarray,
    noise_map: np.ndarray,
    valid: np.ndarray,
    *,
    detection_sigma: float,
    psf_fwhm: float,
) -> np.ndarray:
    """识别跨越多个 PSF 宽度的线状正残差，并返回其膨胀掩膜。

    这是点源检测前的结构审计，不是把所有长源都武断地当成伪迹：只有
    同时满足最小面积、主轴长度和 PCA 长宽比的连通域才会被标记。掩膜
    只用于给候选加 ``LINE_ARTIFACT`` 标志，候选仍保留在审计输出中。
    """

    support_threshold = max(4.0, float(detection_sigma))
    support = valid & np.isfinite(residual) & np.isfinite(noise_map) & (residual / noise_map >= support_threshold)
    labels, component_count = ndimage.label(support, structure=np.ones((3, 3), dtype=bool))
    line_mask = np.zeros(valid.shape, dtype=bool)
    if component_count == 0:
        return line_mask

    min_pixels = max(24, int(round(2.5 * psf_fwhm**2)))
    min_length = max(16.0, 5.0 * psf_fwhm)
    slices = ndimage.find_objects(labels)
    sizes = ndimage.sum(support, labels, index=np.arange(1, component_count + 1))
    for component_id, size in enumerate(sizes, start=1):
        if float(size) < min_pixels:
            continue
        component_slice = slices[component_id - 1]
        if component_slice is None:
            continue
        ys, xs = component_slice
        component = labels[component_slice] == component_id
        local_y, local_x = np.nonzero(component)
        if local_x.size < 3:
            continue
        points = np.column_stack((local_x + xs.start, local_y + ys.start)).astype(np.float64)
        covariance = np.cov(points, rowvar=False, bias=True)
        eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
        major_variance = float(eigenvalues[-1])
        minor_variance = max(0.0, float(eigenvalues[0]))
        major_length = 4.0 * np.sqrt(max(major_variance, 0.0))
        axis_ratio = np.sqrt(major_variance / max(minor_variance, np.finfo(np.float64).eps))
        if major_length >= min_length and axis_ratio >= 4.0:
            line_mask[component_slice] |= component

    dilation = max(1, int(round(psf_fwhm / 2.0)))
    return ndimage.binary_dilation(line_mask, iterations=dilation)


def _source_from_peak(
    image: np.ndarray,
    mask: np.ndarray,
    line_artifact_mask: np.ndarray | None,
    x_peak: int,
    y_peak: int,
    background_map: np.ndarray,
    noise_map: np.ndarray,
    aperture_radius: int,
    saturation_level: float | None,
    filter_snr: float,
    min_flux_snr: float,
    min_fwhm: float,
    max_fwhm: float,
    max_ellipticity: float,
    min_sharpness: float,
    max_sharpness: float,
    min_footprint_pixels: int,
    gain_e_per_adu: float | None,
    read_noise_adu: float,
) -> Detection:
    """在候选峰周围做局部背景、孔径测光和点源形状计算。"""

    height, width = image.shape
    outer_radius = aperture_radius + 4
    y0 = max(0, y_peak - outer_radius)
    y1 = min(height, y_peak + outer_radius + 1)
    x0 = max(0, x_peak - outer_radius)
    x1 = min(width, x_peak + outer_radius + 1)
    patch = image[y0:y1, x0:x1]
    patch_mask = mask[y0:y1, x0:x1] | ~np.isfinite(patch)
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    distance = np.sqrt((xx + x0 - x_peak) ** 2 + (yy + y0 - y_peak) ** 2)
    geometric_aperture = distance <= aperture_radius
    aperture = geometric_aperture & ~patch_mask
    annulus = (distance >= aperture_radius + 2) & (distance <= outer_radius) & ~patch_mask
    annulus_values = patch[annulus]
    fallback_background = float(background_map[y_peak, x_peak])
    fallback_noise = float(noise_map[y_peak, x_peak])
    annulus_is_usable = annulus_values.size >= 16
    if annulus_is_usable:
        local_background, local_noise = sigma_clipped_stats(annulus_values, sample_limit=100_000)
    else:
        local_background, local_noise = fallback_background, fallback_noise
    local_noise = max(float(local_noise), np.finfo(np.float64).eps)

    residual = patch - local_background
    valid_aperture = residual[aperture]
    net_flux = float(valid_aperture.sum()) if valid_aperture.size else 0.0
    aperture_pixels = int(aperture.sum())
    background_pixels = max(1, int(annulus.sum()))
    source_variance = max(net_flux, 0.0) / gain_e_per_adu if gain_e_per_adu is not None else 0.0
    source_variance += aperture_pixels * (read_noise_adu**2)
    background_variance = aperture_pixels * local_noise**2
    background_variance += (aperture_pixels**2 / background_pixels) * local_noise**2
    flux_error = float(np.sqrt(max(source_variance + background_variance, np.finfo(np.float64).eps)))
    flux_snr = net_flux / flux_error

    # 用正残差估计形状，避免负噪声把质心拉向边缘。
    positive = np.where(aperture, np.maximum(residual, 0.0), 0.0)
    positive_sum = float(positive.sum())
    if positive_sum > 0:
        x = float((positive * (xx + x0)).sum() / positive_sum)
        y = float((positive * (yy + y0)).sum() / positive_sum)
        variance_x = float((positive * ((xx + x0) - x) ** 2).sum() / positive_sum)
        variance_y = float((positive * ((yy + y0) - y) ** 2).sum() / positive_sum)
        fwhm_x = 2.35482 * float(np.sqrt(max(0.0, variance_x)))
        fwhm_y = 2.35482 * float(np.sqrt(max(0.0, variance_y)))
        fwhm = 2.35482 * float(np.sqrt(max(0.0, (variance_x + variance_y) / 2.0)))
        ellipticity = abs(fwhm_x - fwhm_y) / max(fwhm_x, fwhm_y, np.finfo(np.float64).eps)
        sharpness = float(max(residual[aperture].max(), 0.0) / max(positive_sum, np.finfo(np.float64).eps))
        if not np.isfinite(fwhm) or fwhm <= 0:
            fwhm = None
    else:
        x, y = float(x_peak), float(y_peak)
        fwhm = fwhm_x = fwhm_y = ellipticity = sharpness = None

    peak = float(image[y_peak, x_peak])
    peak_snr = (peak - local_background) / local_noise
    footprint_pixels = int(((residual >= local_noise) & aperture).sum())
    flags: list[str] = []
    if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
        flags.append("EDGE")
    if patch_mask[geometric_aperture].any():
        flags.append("MASKED")
    if saturation_level is not None and np.any(
        geometric_aperture & np.isfinite(patch) & (patch >= saturation_level)
    ):
        flags.append("SATURATED")
    if line_artifact_mask is not None and line_artifact_mask[y0:y1, x0:x1][geometric_aperture].any():
        flags.append("LINE_ARTIFACT")
    if not annulus_is_usable:
        flags.append("BACKGROUND_UNCERTAIN")
    if net_flux <= 0:
        flags.append("NON_POSITIVE_FLUX")
    if not np.isfinite(flux_snr) or flux_snr < min_flux_snr:
        flags.append("LOW_FLUX_SNR")
    if fwhm is None or fwhm_x is None or fwhm_y is None:
        flags.append("NO_SHAPE")
    else:
        if fwhm < min_fwhm:
            flags.append("NARROW")
        if fwhm > max_fwhm:
            flags.append("BROAD")
        if ellipticity is not None and ellipticity > max_ellipticity:
            flags.append("ELONGATED")
    if sharpness is None or sharpness < min_sharpness:
        flags.append("DIFFUSE")
    elif sharpness > max_sharpness:
        flags.append("SPIKE")
    if footprint_pixels < min_footprint_pixels:
        flags.append("SMALL_FOOTPRINT")

    reject_flags = {
        "EDGE",
        "MASKED",
        "LINE_ARTIFACT",
        "BACKGROUND_UNCERTAIN",
        "SATURATED",
        "NON_POSITIVE_FLUX",
        "LOW_FLUX_SNR",
        "NO_SHAPE",
        "NARROW",
        "BROAD",
        "ELONGATED",
        "DIFFUSE",
        "SPIKE",
        "SMALL_FOOTPRINT",
    }
    return Detection(
        detection_id=-1,
        x=x,
        y=y,
        peak=peak,
        flux=net_flux,
        background=float(local_background),
        noise=float(local_noise),
        snr=float(peak_snr),
        flux_error=flux_error,
        flux_snr=float(flux_snr),
        filter_snr=float(filter_snr),
        fwhm=fwhm,
        fwhm_x=fwhm_x,
        fwhm_y=fwhm_y,
        ellipticity=ellipticity,
        sharpness=sharpness,
        footprint_pixels=footprint_pixels,
        flags=tuple(flags),
        quality_passed=not reject_flags.intersection(flags),
    )


def _working_mask(
    values: np.ndarray,
    mask: np.ndarray | None,
    *,
    background: float,
    mask_zero_pixels: bool | None,
    saturation_level: float | None,
) -> tuple[np.ndarray, float | None, bool]:
    """生成通用无效像素掩膜，并返回自动推断的饱和上限。"""

    numeric = np.asarray(values, dtype=np.float64)
    effective_mask = np.zeros(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if effective_mask.shape != values.shape:
        raise ValueError(f"mask shape {effective_mask.shape} does not match image shape {values.shape}")
    effective_mask |= ~np.isfinite(numeric)
    is_integer = np.issubdtype(np.asarray(values).dtype, np.integer)
    if mask_zero_pixels is None:
        mask_zero_pixels = bool(is_integer and np.isfinite(background) and abs(background) > 1.0)
    if mask_zero_pixels:
        effective_mask |= numeric == 0.0

    inferred_saturation = saturation_level
    if inferred_saturation is None and is_integer:
        dtype_info = np.iinfo(np.asarray(values).dtype)
        inferred_saturation = float(dtype_info.max - 32)
        effective_mask |= numeric >= inferred_saturation
        effective_mask |= numeric <= float(dtype_info.min + 32)
    elif saturation_level is not None:
        effective_mask |= numeric >= saturation_level
    return effective_mask, inferred_saturation, bool(mask_zero_pixels)


def detect_sources(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold_sigma: float = 4.0,
    min_distance: int = 3,
    aperture_radius: int = 4,
    max_sources: int | None = None,
    saturation_level: float | None = None,
    psf_fwhm: float = 3.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_fwhm: float = 0.8,
    max_fwhm: float = 12.0,
    max_ellipticity: float = 0.65,
    min_sharpness: float = 0.005,
    max_sharpness: float = 0.85,
    min_footprint_pixels: int = 2,
    gain_e_per_adu: float | None = None,
    read_noise_adu: float = 0.0,
    mask_zero_pixels: bool | None = None,
    reject_linear_artifacts: bool = True,
) -> DetectionResult:
    """检测点源候选并进行可解释的局部质量判定。

    检测阶段对背景扣除图像做 Gaussian PSF 匹配滤波，再以局部峰和
    ``threshold_sigma`` 取得高召回候选。测量阶段在原始图像上用局部环
    估计背景，计算净通量、误差和 ``flux_snr``，最后按点源形状和数据
    有效性打标签。默认不限制源数量；只有调用方显式传入
    ``max_sources`` 才会截断返回列表。

    ``gain_e_per_adu`` 和 ``read_noise_adu`` 未知时不虚构仪器噪声参数：
    flux SNR 至少包含孔径内背景噪声和局部背景估计误差；若提供增益，
    再加入源光子的 Poisson 方差。
    """

    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")
    if min_distance < 1 or aperture_radius < 1:
        raise ValueError("min_distance and aperture_radius must be positive")
    if psf_fwhm <= 0:
        raise ValueError("psf_fwhm must be positive")
    if min_flux_snr <= 0 or background_box_size < 16:
        raise ValueError("min_flux_snr must be positive and background_box_size must be at least 16")
    if min_fwhm <= 0 or max_fwhm < min_fwhm:
        raise ValueError("fwhm limits are invalid")
    if not 0 <= max_ellipticity <= 1:
        raise ValueError("max_ellipticity must be between 0 and 1")
    if min_sharpness < 0 or max_sharpness <= min_sharpness:
        raise ValueError("sharpness limits are invalid")
    if min_footprint_pixels < 1:
        raise ValueError("min_footprint_pixels must be positive")
    if gain_e_per_adu is not None and gain_e_per_adu <= 0:
        raise ValueError("gain_e_per_adu must be positive when provided")
    if read_noise_adu < 0:
        raise ValueError("read_noise_adu cannot be negative")

    numeric = np.asarray(values, dtype=np.float64)
    finite_mask = ~np.isfinite(numeric)
    base_mask = finite_mask if mask is None else (np.asarray(mask, dtype=bool) | finite_mask)
    if base_mask.shape != values.shape:
        raise ValueError(f"mask shape {base_mask.shape} does not match image shape {values.shape}")
    global_background, global_noise = sigma_clipped_stats(numeric, mask=base_mask)
    effective_mask, inferred_saturation, used_zero_mask = _working_mask(
        values,
        mask,
        background=global_background,
        mask_zero_pixels=mask_zero_pixels,
        saturation_level=saturation_level,
    )
    background, noise = sigma_clipped_stats(numeric, mask=effective_mask)
    background_map, noise_map = local_background_rms(
        numeric,
        effective_mask,
        box_size=background_box_size,
        fallback=(background, noise),
    )

    psf_sigma = psf_fwhm / 2.35482
    valid = ~effective_mask
    residual = np.where(valid, numeric - background_map, 0.0)
    line_artifact_mask = (
        _linear_artifact_mask(
            residual,
            noise_map,
            valid,
            detection_sigma=threshold_sigma,
            psf_fwhm=psf_fwhm,
        )
        if reject_linear_artifacts
        else None
    )
    # 对掩膜区域做归一化卷积，避免边界/坏点的 0 值把附近源的响应压低。
    filtered_sum = ndimage.gaussian_filter(residual * valid, sigma=psf_sigma, mode="nearest")
    filtered_weight = ndimage.gaussian_filter(valid.astype(np.float64), sigma=psf_sigma, mode="nearest")
    filtered = np.divide(
        filtered_sum,
        filtered_weight,
        out=np.zeros_like(filtered_sum),
        where=filtered_weight > 0.5,
    )
    filter_background, filter_noise = sigma_clipped_stats(filtered, mask=effective_mask)
    filter_noise = max(filter_noise, np.finfo(np.float64).eps)
    local_noise_scale = np.maximum(noise_map / max(float(np.median(noise_map[valid])), np.finfo(np.float64).eps), 0.25)
    filter_noise_map = filter_noise * local_noise_scale
    filter_snr_image = (filtered - filter_background) / filter_noise_map

    peak_radius = max(min_distance, int(np.ceil(2.0 * psf_sigma)))
    neighborhood = 2 * peak_radius + 1
    candidate_mask = np.isfinite(filter_snr_image) & valid & (filter_snr_image >= threshold_sigma)
    local_max = filter_snr_image == ndimage.maximum_filter(filter_snr_image, size=neighborhood, mode="nearest")
    yy, xx = np.nonzero(candidate_mask & local_max)
    candidates = sorted(
        ((int(x), int(y), float(filter_snr_image[y, x])) for x, y in zip(xx, yy, strict=True)),
        key=lambda item: item[2],
        reverse=True,
    )
    selected = _suppress_close_candidates(candidates, min_distance)
    candidate_count = len(selected)
    if max_sources is not None:
        if max_sources < 1:
            raise ValueError("max_sources must be positive when provided")
        selected = selected[:max_sources]

    sources: list[Detection] = []
    for x_peak, y_peak, candidate_filter_snr in selected:
        sources.append(
            _source_from_peak(
                numeric,
                effective_mask,
                line_artifact_mask,
                x_peak,
                y_peak,
                background_map,
                noise_map,
                aperture_radius,
                inferred_saturation,
                candidate_filter_snr,
                min_flux_snr,
                min_fwhm,
                max_fwhm,
                max_ellipticity,
                min_sharpness,
                max_sharpness,
                min_footprint_pixels,
                gain_e_per_adu,
                read_noise_adu,
            )
        )
    sources.sort(key=lambda source: (source.y, source.x))
    numbered = [
        Detection(
            detection_id=index,
            x=source.x,
            y=source.y,
            peak=source.peak,
            flux=source.flux,
            background=source.background,
            noise=source.noise,
            snr=source.snr,
            fwhm=source.fwhm,
            flags=source.flags,
            flux_error=source.flux_error,
            flux_snr=source.flux_snr,
            filter_snr=source.filter_snr,
            fwhm_x=source.fwhm_x,
            fwhm_y=source.fwhm_y,
            ellipticity=source.ellipticity,
            sharpness=source.sharpness,
            footprint_pixels=source.footprint_pixels,
            quality_passed=source.quality_passed,
        )
        for index, source in enumerate(sources)
    ]
    quality_count = sum(source.quality_passed for source in numbered)
    return DetectionResult(
        image_shape=(int(values.shape[0]), int(values.shape[1])),
        background=float(background),
        noise=float(noise),
        threshold=float(background + threshold_sigma * noise),
        candidate_count=candidate_count,
        sources=tuple(numbered),
        quality_count=quality_count,
        parameters={
            "threshold_sigma": float(threshold_sigma),
            "min_distance": int(min_distance),
            "aperture_radius": int(aperture_radius),
            "max_sources": -1 if max_sources is None else int(max_sources),
            "psf_fwhm": float(psf_fwhm),
            "background_box_size": int(background_box_size),
            "min_flux_snr": float(min_flux_snr),
            "min_fwhm": float(min_fwhm),
            "max_fwhm": float(max_fwhm),
            "max_ellipticity": float(max_ellipticity),
            "min_sharpness": float(min_sharpness),
            "max_sharpness": float(max_sharpness),
            "min_footprint_pixels": int(min_footprint_pixels),
            "gain_e_per_adu": -1.0 if gain_e_per_adu is None else float(gain_e_per_adu),
            "read_noise_adu": float(read_noise_adu),
            "mask_zero_pixels": int(used_zero_mask),
            "reject_linear_artifacts": int(reject_linear_artifacts),
            "line_artifact_pixels": 0 if line_artifact_mask is None else int(line_artifact_mask.sum()),
            "saturation_level": -1.0 if inferred_saturation is None else float(inferred_saturation),
            "global_background": float(global_background),
            "global_noise": float(global_noise),
            "filter_noise": float(filter_noise),
        },
    )
