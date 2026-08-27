"""鲁棒背景估计和星点候选检测。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class Detection:
    """单个检测源的可解释属性。坐标以图像左上角为原点，x 向右、y 向下。"""

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
            "fwhm": self.fwhm,
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

    @property
    def star_count(self) -> int:
        return len(self.sources)

    def as_dict(self) -> dict[str, object]:
        return {
            "image_shape": list(self.image_shape),
            "background": self.background,
            "noise": self.noise,
            "threshold": self.threshold,
            "candidate_count": self.candidate_count,
            "returned_count": self.star_count,
            "truncated": self.candidate_count > self.star_count,
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


def _source_from_peak(
    image: np.ndarray,
    mask: np.ndarray,
    x_peak: int,
    y_peak: int,
    background: float,
    noise: float,
    aperture_radius: int,
    saturation_level: float | None,
) -> Detection:
    height, width = image.shape
    y0 = max(0, y_peak - aperture_radius)
    y1 = min(height, y_peak + aperture_radius + 1)
    x0 = max(0, x_peak - aperture_radius)
    x1 = min(width, x_peak + aperture_radius + 1)
    patch = image[y0:y1, x0:x1]
    patch_mask = mask[y0:y1, x0:x1] | ~np.isfinite(patch)
    signal = np.where(patch_mask, 0.0, np.maximum(patch - background, 0.0))
    flux = float(signal.sum())
    yy, xx = np.indices(signal.shape, dtype=np.float64)
    if flux > 0:
        x = float((signal * (xx + x0)).sum() / flux)
        y = float((signal * (yy + y0)).sum() / flux)
        var_x = float((signal * ((xx + x0) - x) ** 2).sum() / flux)
        var_y = float((signal * ((yy + y0) - y) ** 2).sum() / flux)
        fwhm = 2.35482 * float(np.sqrt(max(0.0, (var_x + var_y) / 2.0)))
        if fwhm <= 0 or not np.isfinite(fwhm):
            fwhm = None
    else:
        x, y, fwhm = float(x_peak), float(y_peak), None

    peak = float(image[y_peak, x_peak])
    snr = (peak - background) / noise if noise > 0 else float("inf")
    flags: list[str] = []
    if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
        flags.append("EDGE")
    if patch_mask.any():
        flags.append("MASKED")
    if saturation_level is not None and float(np.nanmax(patch)) >= saturation_level:
        flags.append("SATURATED")
    return Detection(
        detection_id=-1,
        x=x,
        y=y,
        peak=peak,
        flux=flux,
        background=background,
        noise=noise,
        snr=float(snr),
        fwhm=fwhm,
        flags=tuple(flags),
    )


def detect_sources(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold_sigma: float = 5.0,
    min_distance: int = 3,
    aperture_radius: int = 4,
    max_sources: int | None = None,
    saturation_level: float | None = None,
) -> DetectionResult:
    """检测超过鲁棒背景阈值的局部峰值。

    这是可解释的第一版基线，不声称完成 PSF 拟合或星点真值分类。所有
    参数会写入结果，便于后续做阈值扫描和注入实验。
    """

    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")
    if min_distance < 1 or aperture_radius < 1:
        raise ValueError("min_distance and aperture_radius must be positive")
    effective_mask = np.zeros(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if effective_mask.shape != values.shape:
        raise ValueError(f"mask shape {effective_mask.shape} does not match image shape {values.shape}")
    background, noise = sigma_clipped_stats(values, mask=effective_mask)
    numeric = np.asarray(values, dtype=np.float64)
    threshold = background + threshold_sigma * noise
    candidate_mask = np.isfinite(numeric) & ~effective_mask & (numeric >= threshold)
    neighborhood = 2 * min_distance + 1
    local_max = numeric == ndimage.maximum_filter(numeric, size=neighborhood, mode="nearest")
    yy, xx = np.nonzero(candidate_mask & local_max)
    candidates = sorted(((int(x), int(y), float(numeric[y, x])) for x, y in zip(xx, yy, strict=True)), key=lambda item: item[2], reverse=True)
    selected = _suppress_close_candidates(candidates, min_distance)
    candidate_count = len(selected)
    if max_sources is not None:
        if max_sources < 1:
            raise ValueError("max_sources must be positive when provided")
        selected = selected[:max_sources]

    sources: list[Detection] = []
    for x_peak, y_peak, _ in selected:
        source = _source_from_peak(
            numeric,
            effective_mask,
            x_peak,
            y_peak,
            background,
            noise,
            aperture_radius,
            saturation_level,
        )
        sources.append(source)
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
        )
        for index, source in enumerate(sources)
    ]
    return DetectionResult(
        image_shape=(int(values.shape[0]), int(values.shape[1])),
        background=background,
        noise=noise,
        threshold=threshold,
        candidate_count=candidate_count,
        sources=tuple(numbered),
        parameters={
            "threshold_sigma": float(threshold_sigma),
            "min_distance": int(min_distance),
            "aperture_radius": int(aperture_radius),
            "max_sources": -1 if max_sources is None else int(max_sources),
        },
    )
