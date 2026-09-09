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

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


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
    psf_support_pixels: int | None = None
    quality_passed: bool = True
    # ``x/y`` are the measured (sub-pixel) centroid used for photometry and
    # tracking.  Keep the integer matched-filter peak separately so the UI and
    # audit tools can show where the candidate was actually detected.  Older
    # serialized results may not have these fields; callers must fall back to
    # ``x/y`` when they are ``None``.
    peak_x: float | None = None
    peak_y: float | None = None
    centroid_shift_px: float | None = None
    # 宽筛选层的来源证据。``filter_snr`` 始终保留 Gaussian 匹配滤波
    # 的语义；DoG 等互补检测器的最高响应单独放在 ``proposal_snr``，
    # 避免把不同响应统计混成一个 SNR。
    proposal_methods: tuple[str, ...] = ()
    proposal_scales: tuple[float, ...] = ()
    proposal_snr: float | None = None
    nearest_gaussian_px: float | None = None
    deblend_delta_bic: float | None = None
    deblend_component_snr: float | None = None
    # 数据有效性审计：整型 FITS 中若某些高位码在全幅异常重复，且同一
    # 孔径还出现远超背景噪声的负值，这两个字段记录局部证据。它们不
    # 修改原始 ADU，也不把“疑似编码模式”直接命名为饱和真值。
    repeated_code_count: int | None = None
    range_anomaly_pixel_count: int | None = None
    repeated_code_values: tuple[int, ...] = ()

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
            "psf_support_pixels": self.psf_support_pixels,
            "quality_passed": self.quality_passed,
            "peak_x": self.peak_x,
            "peak_y": self.peak_y,
            "centroid_shift_px": self.centroid_shift_px,
            "proposal_methods": list(self.proposal_methods),
            "proposal_scales": list(self.proposal_scales),
            "proposal_snr": self.proposal_snr,
            "nearest_gaussian_px": self.nearest_gaussian_px,
            "deblend_delta_bic": self.deblend_delta_bic,
            "deblend_component_snr": self.deblend_component_snr,
            "repeated_code_count": self.repeated_code_count,
            "range_anomaly_pixel_count": self.range_anomaly_pixel_count,
            "repeated_code_values": list(self.repeated_code_values),
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
    parameters: dict[str, object]
    quality_count: int | None = None
    # Sequence fast-path only: all merged matched-filter peaks before an
    # optional source-level working-set cut.  This is intentionally transient
    # and is not serialized by ``as_dict``; the sequence consumes it to build a
    # temporal consensus without measuring every candidate in every frame.
    candidate_peaks: np.ndarray | tuple[tuple[int, int, float], ...] = ()
    # Sequence fast-path only: sparse (x, y) coordinates of the line-artifact
    # mask produced during detection. This is transient and intentionally not
    # serialized; sequence consensus uses it to reject candidates whose local
    # aperture intersects a known extended linear component.
    line_artifact_coordinates: np.ndarray | tuple[tuple[int, int], ...] = ()

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
    """返回稳健统计样本，避免序列模式无意义的整图升精度复制。"""

    values = np.asarray(image)
    if not np.issubdtype(values.dtype, np.floating):
        values = np.asarray(values, dtype=np.float64)
    flat_values = values.reshape(-1)
    flat_mask = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
    if flat_mask is not None and flat_mask.size != flat_values.size:
        raise ValueError("mask shape does not match image shape")
    valid = np.isfinite(flat_values)
    if flat_mask is not None:
        valid &= ~flat_mask
    flat = flat_values[valid]
    if flat.size > sample_limit:
        # 保留原有“先去除掩膜/非有限值，再按有效值抽样”的统计口径；
        # 这里只避免把 float32 输入先复制成 float64。
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
            # 当前样本已经没有需要剔除的异常值。循环末尾原本会再次
            # 计算同一个 center/MAD；小孔径环会调用数万次，这个等价
            # 短路能显著减少排序/partition 次数，同时保留原来的最终
            # scale=max(1.4826*MAD, std) 定义。
            return center, max(scale, float(np.std(selected)), np.finfo(np.float64).eps)
        selected = selected[keep]
        if selected.size == 0:
            break
    center = float(np.median(selected))
    mad = float(np.median(np.abs(selected - center)))
    scale = max(1.4826 * mad, float(np.std(selected)), np.finfo(np.float64).eps)
    return center, scale


def _resize_grid(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把块统计网格线性插值到图像大小。

    保留旧版 ``ndimage.zoom`` 再裁剪/补齐的坐标口径，但直接生成需要的
    部分，避免 4096² 图像先产生更大的中间数组。
    """

    height, width = shape
    if grid.shape == (1, 1):
        dtype = np.float32 if np.asarray(grid).dtype == np.float32 else np.float64
        return np.full(shape, float(grid[0, 0]), dtype=dtype)
    values = np.asarray(grid, dtype=np.float32 if np.asarray(grid).dtype == np.float32 else np.float64)
    grid_height, grid_width = values.shape
    zoom_y = (height - 1) / max(1, grid_height - 1) if height > 1 else 1.0
    zoom_x = (width - 1) / max(1, grid_width - 1) if width > 1 else 1.0
    resized_height = max(1, int(round(grid_height * zoom_y)))
    resized_width = max(1, int(round(grid_width * zoom_x)))
    output_height = min(height, resized_height)
    output_width = min(width, resized_width)
    if resized_height == 1:
        target_y = np.zeros(output_height, dtype=np.float64)
    else:
        target_y = np.arange(output_height, dtype=np.float64) * (grid_height - 1) / (resized_height - 1)
    source_y = np.arange(grid_height, dtype=np.float64)
    rows = np.empty((output_height, grid_width), dtype=values.dtype)
    for column in range(grid_width):
        rows[:, column] = np.interp(target_y, source_y, values[:, column])
    if resized_width == 1:
        resized = rows[:, :1].copy()
    else:
        source_x = np.arange(grid_width, dtype=np.float64)
        target_x = np.arange(output_width, dtype=np.float64) * (grid_width - 1) / (resized_width - 1)
        resized = np.empty((output_height, output_width), dtype=values.dtype)
        for row_index in range(output_height):
            resized[row_index] = np.interp(target_x, source_x, rows[row_index])
    if resized.shape == shape:
        return resized
    result = np.empty(shape, dtype=values.dtype)
    result[...] = resized[-1, -1]
    result[:output_height, :output_width] = resized
    return result


def local_background_rms(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    box_size: int = 128,
    fallback: tuple[float, float] | None = None,
    iterations: int = 4,
    sample_limit: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """用网格化 sigma-clipping 生成二维背景和 RMS 图。

    块统计避免把一个亮星的像素直接当作背景；线性插值只用于把块统计
    变成逐像素权重。``iterations`` 允许序列快速路径使用较少的稳健
    裁剪轮数；``sample_limit`` 是每个网格块的确定性步进抽样上限，适合
    背景缓慢变化且需要低延迟的序列路径。单图精测默认使用完整网格像素；
    这两者都不是对复杂散射光场的最终模型，但比单一全图标量更能解释
    局部 SNR。
    """

    if box_size < 16:
        raise ValueError("background box_size must be at least 16 pixels")
    if iterations < 1:
        raise ValueError("background iterations must be positive")
    if sample_limit < 1:
        raise ValueError("background sample_limit must be positive")
    input_values = np.asarray(image)
    calculation_dtype = np.float32 if input_values.dtype == np.float32 else np.float64
    values = np.asarray(input_values, dtype=calculation_dtype)
    height, width = values.shape
    global_stats = fallback or sigma_clipped_stats(values, mask=mask, iterations=iterations)

    backgrounds: list[list[float]] = []
    noises: list[list[float]] = []
    rows = range(0, height, box_size)
    cols = range(0, width, box_size)
    for y0 in rows:
        background_row: list[float] = []
        noise_row: list[float] = []
        for x0 in cols:
            block = values[y0 : y0 + box_size, x0 : x0 + box_size]
            block_mask = mask[y0 : y0 + box_size, x0 : x0 + box_size]
            try:
                block_stats = sigma_clipped_stats(
                    block,
                    mask=block_mask,
                    iterations=iterations,
                    sample_limit=sample_limit,
                )
            except ValueError:
                block_stats = global_stats
            if not np.isfinite(block_stats[0]) or not np.isfinite(block_stats[1]) or block_stats[1] <= 0:
                block_stats = global_stats
            background_row.append(float(block_stats[0]))
            noise_row.append(float(block_stats[1]))
        backgrounds.append(background_row)
        noises.append(noise_row)
    background_map = _resize_grid(np.asarray(backgrounds), values.shape).astype(calculation_dtype, copy=False)
    noise_map = _resize_grid(np.asarray(noises), values.shape).astype(calculation_dtype, copy=False)
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


@dataclass(frozen=True, slots=True)
class _Proposal:
    """宽筛选层的一个算法提案；尚未经过源级质量判定。"""

    x: int
    y: int
    score: float
    method: str
    scale_fwhm: float


@dataclass(slots=True)
class _MergedProposal:
    """同一帧内按空间合并后的候选及其算法来源。"""

    x: int
    y: int
    score: float
    methods: list[str]
    scales: list[float]
    nearest_gaussian_px: float | None = None


def _response_proposals(
    response_snr: np.ndarray,
    valid: np.ndarray,
    *,
    threshold_sigma: float,
    min_distance: int,
    scale_fwhm: float,
    method: str,
    candidate_gate: np.ndarray | None = None,
) -> list[_Proposal]:
    """从一个标准化响应图中提取局部峰，并先在算法内部做 NMS。"""

    response_sigma = max(float(scale_fwhm) / 2.35482, 0.5)
    # 极大值窗口只负责回答“这里是不是一个局部响应峰”，不能把最终
    # 的候选间距 ``min_distance`` 再提前用一遍。旧实现把两者绑定：默认
    # FWHM=3、min_distance=4 时使用 9×9 窗口，4--7 px 的真实近邻峰会在
    # NMS 之前就消失，后面的去混叠没有机会看到它。这里把窗口限制在
    # 响应核约 2σ 的范围，最终候选间距仍由
    # ``_suppress_close_candidates`` 单独控制。
    peak_radius = max(1, int(np.ceil(2.0 * response_sigma)))
    neighborhood = 2 * peak_radius + 1
    candidate_mask = np.isfinite(response_snr) & valid & (response_snr >= threshold_sigma)
    if candidate_gate is not None:
        if candidate_gate.shape != response_snr.shape:
            raise ValueError("candidate_gate shape must match response_snr")
        candidate_mask &= candidate_gate
    local_max = response_snr == ndimage.maximum_filter(response_snr, size=neighborhood, mode="nearest")
    yy, xx = np.nonzero(candidate_mask & local_max)
    ranked = sorted(
        ((int(x), int(y), float(response_snr[y, x])) for x, y in zip(xx, yy, strict=True)),
        key=lambda item: item[2],
        reverse=True,
    )
    return [
        _Proposal(x=x, y=y, score=score, method=method, scale_fwhm=float(scale_fwhm))
        for x, y, score in _suppress_close_candidates(ranked, min_distance)
    ]


def _merge_proposals(
    primary: Iterable[_Proposal],
    extras: Iterable[_Proposal],
    *,
    min_distance: int,
) -> list[_MergedProposal]:
    """合并多算法候选，同时保留来源；Gaussian 主峰优先作为测量锚点。"""

    primary_rows = list(primary)
    accepted: list[_MergedProposal] = []
    grid: dict[tuple[int, int], list[int]] = {}
    cell_size = float(max(1, min_distance))
    distance_sq = float(min_distance * min_distance)

    def add(proposal: _Proposal) -> None:
        cell = (int(proposal.x // cell_size), int(proposal.y // cell_size))
        nearest_index: int | None = None
        nearest_distance = float("inf")
        for gx in range(cell[0] - 1, cell[0] + 2):
            for gy in range(cell[1] - 1, cell[1] + 2):
                for candidate_index in grid.get((gx, gy), ()):
                    candidate = accepted[candidate_index]
                    current_distance = float((proposal.x - candidate.x) ** 2 + (proposal.y - candidate.y) ** 2)
                    if current_distance < distance_sq and current_distance < nearest_distance:
                        nearest_index = candidate_index
                        nearest_distance = current_distance
        if nearest_index is None:
            grid.setdefault(cell, []).append(len(accepted))
            accepted.append(
                _MergedProposal(
                    x=proposal.x,
                    y=proposal.y,
                    score=proposal.score,
                    methods=[proposal.method],
                    scales=[proposal.scale_fwhm],
                )
            )
            return
        candidate = accepted[nearest_index]
        candidate.score = max(candidate.score, proposal.score)
        if proposal.method not in candidate.methods:
            candidate.methods.append(proposal.method)
        if not any(abs(proposal.scale_fwhm - scale) < 1e-6 for scale in candidate.scales):
            candidate.scales.append(proposal.scale_fwhm)

    # 先种下 Gaussian 候选，确保已有基线的整数峰仍是测光锚点；额外
    # 方法只补充来源或建立 Gaussian 未命中的新候选。
    for proposal in primary_rows:
        add(proposal)
    for proposal in sorted(extras, key=lambda item: item.score, reverse=True):
        add(proposal)
    for candidate in accepted:
        candidate.scales.sort()
    if primary_rows:
        gaussian_tree = cKDTree(np.asarray([(proposal.x, proposal.y) for proposal in primary_rows], dtype=np.float64))
        non_gaussian = [candidate for candidate in accepted if "gaussian" not in candidate.methods]
        if non_gaussian:
            distances, _indices = gaussian_tree.query(
                np.asarray([(candidate.x, candidate.y) for candidate in non_gaussian], dtype=np.float64),
                k=1,
            )
            for candidate, distance in zip(non_gaussian, np.atleast_1d(distances), strict=True):
                candidate.nearest_gaussian_px = float(distance)
    return sorted(accepted, key=lambda item: item.score, reverse=True)


def _local_deblend_proposals(
    image: np.ndarray,
    valid: np.ndarray,
    line_artifact_mask: np.ndarray | None,
    primary: Iterable[_Proposal],
    *,
    background_map: np.ndarray,
    noise_map: np.ndarray,
    psf_fwhm: float,
    min_primary_snr: float,
    min_residual_sigma: float,
    search_radius_factor: float,
) -> list[_Proposal]:
    """从强主峰的单 PSF 残差中提出被 NMS 吞掉的近邻峰。

    宽筛的 Gaussian/DoG 峰检测只能在原始响应图已有局部极大值时工作；
    两个相距约一个 PSF 的源可能只形成一个极大值，单纯再降低阈值也
    不会凭空产生第二个峰。这里对强 Gaussian 主峰做一次小窗口的
    ``背景平面 + 单 PSF`` 拟合，在拟合残差中寻找第二个正峰。该函数
    只生成审计候选，不直接认定为恒星；后续仍必须经过同一套原图测光、
    线状掩膜和 ``ΔBIC + 次分量 SNR`` 检验。

    为控制运行时间，只围绕匹配滤波 SNR 足够高的主峰搜索，每个主峰
    最多提出一个最强次峰。搜索半径与 PSF 成比例，避免把远处独立源
    或整条亮线错误地当成近邻双星。
    """

    if min_primary_snr <= 0 or min_residual_sigma <= 0 or search_radius_factor <= 0:
        raise ValueError("local deblend thresholds must be positive")
    if image.shape != valid.shape or background_map.shape != image.shape or noise_map.shape != image.shape:
        raise ValueError("local deblend arrays must have the same shape")
    if line_artifact_mask is not None and line_artifact_mask.shape != image.shape:
        raise ValueError("line_artifact_mask shape does not match image")

    height, width = image.shape
    sigma = max(float(psf_fwhm) / 2.35482, 0.5)
    search_radius = max(3.0, float(search_radius_factor) * float(psf_fwhm))
    search_pixels = max(4, int(np.ceil(search_radius)))
    # 小于约 0.8 FWHM 的“次峰”通常只是单个 PSF 核心的采样起伏；
    # 把它们交给双源模型会让一个孤立亮星被拆成两个源。真正可分辨的
    # 近邻双源仍会在 1 FWHM 左右及更远留下残差峰，后续 BIC 再做第二道门。
    min_separation = max(1.5, 0.8 * float(psf_fwhm))
    proposals: list[_Proposal] = []

    # 绝大多数主峰位于图像内部且其局部窗口没有坏像素。对这些窗口，
    # ``背景平面 + 固定中心 PSF`` 的设计矩阵完全相同；原来逐个调用
    # ``np.linalg.lstsq`` 会为每个主峰重复一次小矩阵 SVD，真实 4096²
    # 图像上会把去混叠拖到分钟级。这里按批次提取滑窗，复用一次固定
    # 设计矩阵的投影。含掩膜/非有限像素的窗口和边界窗口仍走下面的
    # 标量回退，宁可少一部分深筛候选，也不改变坏像素附近的安全语义。
    primary_rows = [proposal for proposal in primary if float(proposal.score) >= float(min_primary_snr)]
    if not primary_rows:
        return proposals
    batch_size = 4096
    patch_size = 2 * search_pixels + 1
    interior_rows: list[_Proposal] = []
    fallback_rows: list[_Proposal] = []
    for proposal in primary_rows:
        if (
            search_pixels <= proposal.x < width - search_pixels
            and search_pixels <= proposal.y < height - search_pixels
        ):
            interior_rows.append(proposal)
        else:
            fallback_rows.append(proposal)

    if interior_rows:
        values = np.asarray(image, dtype=np.float64)
        background_values = np.asarray(background_map, dtype=np.float64)
        noise_values = np.maximum(np.asarray(noise_map, dtype=np.float64), np.finfo(np.float64).eps)
        image_windows = np.lib.stride_tricks.sliding_window_view(values, (patch_size, patch_size))
        valid_windows = np.lib.stride_tricks.sliding_window_view(np.asarray(valid, dtype=bool), (patch_size, patch_size))
        background_windows = np.lib.stride_tricks.sliding_window_view(background_values, (patch_size, patch_size))
        noise_windows = np.lib.stride_tricks.sliding_window_view(noise_values, (patch_size, patch_size))
        line_windows = (
            np.lib.stride_tricks.sliding_window_view(np.asarray(line_artifact_mask, dtype=bool), (patch_size, patch_size))
            if line_artifact_mask is not None
            else None
        )
        grid_y, grid_x = np.mgrid[-search_pixels : search_pixels + 1, -search_pixels : search_pixels + 1]
        grid_x = np.asarray(grid_x, dtype=np.float64)
        grid_y = np.asarray(grid_y, dtype=np.float64)
        scale = float(patch_size)
        primary_psf = np.exp(-0.5 * ((grid_x / sigma) ** 2 + (grid_y / sigma) ** 2))
        design = np.column_stack(
            (
                np.ones((patch_size, patch_size), dtype=np.float64).reshape(-1),
                (grid_x / scale).reshape(-1),
                (grid_y / scale).reshape(-1),
                primary_psf.reshape(-1),
            )
        )
        projection = np.linalg.pinv(design, rcond=1e-10)
        distance_grid = np.hypot(grid_x, grid_y)
        for batch_start in range(0, len(interior_rows), batch_size):
            batch_rows = interior_rows[batch_start : batch_start + batch_size]
            xs = np.asarray([proposal.x - search_pixels for proposal in batch_rows], dtype=np.intp)
            ys = np.asarray([proposal.y - search_pixels for proposal in batch_rows], dtype=np.intp)
            patch_valid = valid_windows[ys, xs] & np.isfinite(image_windows[ys, xs])
            if line_windows is not None:
                patch_valid &= ~line_windows[ys, xs]
            usable = np.all(patch_valid, axis=(1, 2))
            if not np.all(usable):
                fallback_rows.extend(row for row, keep in zip(batch_rows, usable, strict=True) if not keep)
            if not np.any(usable):
                continue
            usable_xs = xs[usable]
            usable_ys = ys[usable]
            patches = np.asarray(image_windows[usable_ys, usable_xs], dtype=np.float64)
            beta = patches.reshape(patches.shape[0], -1) @ projection.T
            positive_primary = np.isfinite(beta[:, -1]) & (beta[:, -1] > 0.0)
            if not np.any(positive_primary):
                continue
            model = beta @ design.T
            residual_batch = (patches.reshape(patches.shape[0], -1) - model).reshape(
                patches.shape[0], patch_size, patch_size
            )
            local_noise = np.maximum(
                np.asarray(noise_windows[usable_ys, usable_xs], dtype=np.float64),
                np.finfo(np.float64).eps,
            )
            raw_excess = np.asarray(patches, dtype=np.float64) - np.asarray(
                background_windows[usable_ys, usable_xs], dtype=np.float64
            )
            local_max = residual_batch == ndimage.maximum_filter(
                residual_batch,
                size=(1, 3, 3),
                mode="nearest",
            )
            candidate_mask = (
                local_max
                & (distance_grid[None, :, :] >= min_separation)
                & (distance_grid[None, :, :] <= search_radius)
                & (residual_batch >= float(min_residual_sigma) * local_noise)
                & (raw_excess >= 2.0 * local_noise)
            )
            candidate_scores = np.where(
                candidate_mask,
                residual_batch / local_noise,
                -np.inf,
            )
            best_flat = np.argmax(candidate_scores.reshape(candidate_scores.shape[0], -1), axis=1)
            best_scores = np.take_along_axis(
                candidate_scores.reshape(candidate_scores.shape[0], -1),
                best_flat[:, None],
                axis=1,
            )[:, 0]
            for row, is_positive, flat_index, score in zip(
                (batch_rows[index] for index in np.flatnonzero(usable)),
                positive_primary,
                best_flat,
                best_scores,
                strict=True,
            ):
                if not is_positive or not np.isfinite(score):
                    continue
                local_y, local_x = divmod(int(flat_index), patch_size)
                proposals.append(
                    _Proposal(
                        x=int(row.x - search_pixels + local_x),
                        y=int(row.y - search_pixels + local_y),
                        score=float(score),
                        method="deblend_local",
                        scale_fwhm=float(psf_fwhm),
                    )
                )

    for proposal in fallback_rows:
        px = float(proposal.x)
        py = float(proposal.y)
        x0 = max(0, int(proposal.x) - search_pixels)
        x1 = min(width, int(proposal.x) + search_pixels + 1)
        y0 = max(0, int(proposal.y) - search_pixels)
        y1 = min(height, int(proposal.y) + search_pixels + 1)
        patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
        patch_valid = np.asarray(valid[y0:y1, x0:x1], dtype=bool) & np.isfinite(patch)
        if line_artifact_mask is not None:
            patch_valid &= ~np.asarray(line_artifact_mask[y0:y1, x0:x1], dtype=bool)
        if int(np.count_nonzero(patch_valid)) < 24:
            continue

        yy, xx = np.indices(patch.shape, dtype=np.float64)
        gx = xx + float(x0)
        gy = yy + float(y0)
        scale = max(float(x1 - x0), float(y1 - y0), 1.0)
        primary_psf = np.exp(-0.5 * (((gx - px) / sigma) ** 2 + ((gy - py) / sigma) ** 2))
        terms = (
            np.ones(patch.shape, dtype=np.float64),
            (gx - px) / scale,
            (gy - py) / scale,
            primary_psf,
        )
        design = np.column_stack([term[patch_valid] for term in terms])
        try:
            beta, *_ = np.linalg.lstsq(design, patch[patch_valid], rcond=None)
        except np.linalg.LinAlgError:
            continue
        primary_amplitude = float(beta[-1])
        if not np.isfinite(primary_amplitude) or primary_amplitude <= 0:
            continue

        model = sum(float(coefficient) * term for coefficient, term in zip(beta, terms, strict=True))
        residual = patch - model
        local_noise = np.maximum(
            np.asarray(noise_map[y0:y1, x0:x1], dtype=np.float64),
            np.finfo(np.float64).eps,
        )
        raw_excess = patch - np.asarray(background_map[y0:y1, x0:x1], dtype=np.float64)
        distance = np.hypot(gx - px, gy - py)
        local_max = residual == ndimage.maximum_filter(residual, size=3, mode="nearest")
        candidate_mask = (
            patch_valid
            & local_max
            & (distance >= min_separation)
            & (distance <= search_radius)
            & (residual >= float(min_residual_sigma) * local_noise)
            # 残差峰还必须在原图中有正向像素证据。这个门槛不要求它
            # 已经是原图局部极大值，否则真正的近邻弱源会被强主峰压住。
            & (raw_excess >= 2.0 * local_noise)
        )
        if not np.any(candidate_mask):
            continue
        candidate_y, candidate_x = np.nonzero(candidate_mask)
        candidate_scores = residual[candidate_y, candidate_x] / local_noise[candidate_y, candidate_x]
        best_index = int(np.argmax(candidate_scores))
        x_candidate = int(candidate_x[best_index] + x0)
        y_candidate = int(candidate_y[best_index] + y0)
        score = float(candidate_scores[best_index])
        if not np.isfinite(score):
            continue
        proposals.append(
            _Proposal(
                x=x_candidate,
                y=y_candidate,
                score=score,
                method="deblend_local",
                scale_fwhm=float(psf_fwhm),
            )
        )
    return proposals


def _append_deblend_proposals(
    merged: Iterable[_MergedProposal],
    deblend: Iterable[_Proposal],
    *,
    gaussian_primary: Iterable[_Proposal],
    min_distance: int,
) -> list[_MergedProposal]:
    """把有意允许靠近主峰的去混叠候选追加到常规 NMS 结果。

    常规 ``_merge_proposals`` 会把小于 ``min_distance`` 的点合并；这对
    普通噪声抑制是正确的，却会把本函数专门找出的近邻次峰再次吞掉。
    因此去混叠候选单独追加，只用较小半径去除多个主峰重复提出的同一点，
    并保留它到最近 Gaussian 主峰的距离供后续审计。
    """

    accepted = list(merged)
    primary_rows = list(gaussian_primary)
    primary_tree = (
        cKDTree(np.asarray([(proposal.x, proposal.y) for proposal in primary_rows], dtype=np.float64))
        if primary_rows
        else None
    )
    dedupe_radius = max(1, int(np.floor(max(1.5, 0.5 * float(min_distance)))))
    ranked = sorted(
        ((proposal.x, proposal.y, proposal.score, proposal) for proposal in deblend),
        key=lambda item: float(item[2]),
        reverse=True,
    )
    unique: list[_Proposal] = []
    for _x, _y, _score, proposal in ranked:
        if any(
            (proposal.x - other.x) ** 2 + (proposal.y - other.y) ** 2 < float(dedupe_radius * dedupe_radius)
            for other in unique
        ):
            continue
        # 同一个候选若已由常规通道提出，不能再复制一个 ID；只有相距
        # 约一个像素以内才视为同一测量锚点，保留更近的真实近邻源。
        if any((proposal.x - row.x) ** 2 + (proposal.y - row.y) ** 2 < 2.25 for row in accepted):
            continue
        unique.append(proposal)
    for proposal in unique:
        nearest = None
        if primary_tree is not None:
            nearest_distance, _index = primary_tree.query((proposal.x, proposal.y), k=1)
            nearest = float(nearest_distance)
        accepted.append(
            _MergedProposal(
                x=proposal.x,
                y=proposal.y,
                score=float(proposal.score),
                methods=[proposal.method],
                scales=[proposal.scale_fwhm],
                nearest_gaussian_px=nearest,
            )
        )
    return sorted(accepted, key=lambda item: item.score, reverse=True)


def _masked_gaussian_response(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: float,
    calculation_dtype: type[np.float32] | type[np.float64],
    fast_sequence: bool,
) -> tuple[np.ndarray, str]:
    """对掩膜归一化后的残差做 Gaussian 滤波，并返回实现模式。"""

    if fast_sequence and float(np.mean(~valid)) <= 0.02:
        filtered = ndimage.gaussian_filter(residual, sigma=sigma, mode="nearest")
        filtered[~valid] = 0.0
        return filtered, "sparse_mask_single_pass"
    filtered_sum = ndimage.gaussian_filter(residual * valid, sigma=sigma, mode="nearest")
    filtered_weight = ndimage.gaussian_filter(valid.astype(calculation_dtype), sigma=sigma, mode="nearest")
    filtered = np.divide(
        filtered_sum,
        filtered_weight,
        out=np.zeros_like(filtered_sum),
        where=filtered_weight > 0.5,
    )
    return filtered, "normalized_two_pass"


def _masked_starlet_smooth(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    step: int,
    calculation_dtype: type[np.float32] | type[np.float64],
    fast_sequence: bool,
) -> tuple[np.ndarray, str]:
    """用带孔 B3-spline 核做一层 à trous/starlet 平滑。

    基础核为 ``[1, 4, 6, 4, 1] / 16``；第 ``j`` 层在相邻系数间插入
    ``2**(j-1)-1`` 个零。这里直接接收像素步长，使调用方可以构造
    ``w1=c0-c1``、``w2=c1-c2``。掩膜处理与 Gaussian 通道一致，避免
    无效 0 值在小波系数中制造亮边。
    """

    if step < 1:
        raise ValueError("starlet step must be positive")
    kernel = np.zeros(4 * int(step) + 1, dtype=calculation_dtype)
    kernel[:: int(step)] = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=calculation_dtype) / 16.0

    def smooth(array: np.ndarray) -> np.ndarray:
        horizontal = ndimage.convolve1d(array, kernel, axis=1, mode="nearest")
        return ndimage.convolve1d(horizontal, kernel, axis=0, mode="nearest")

    if fast_sequence and float(np.mean(~valid)) <= 0.02:
        result = smooth(np.where(valid, values, 0.0))
        result[~valid] = 0.0
        return result, "sparse_mask_single_pass"
    weighted = smooth(np.where(valid, values, 0.0))
    weights = smooth(valid.astype(calculation_dtype))
    result = np.divide(weighted, weights, out=np.zeros_like(weighted), where=weights > 0.5)
    result[~valid] = 0.0
    return result, "normalized_two_pass"


def _standardize_response(
    response: np.ndarray,
    *,
    effective_mask: np.ndarray,
    noise_map: np.ndarray,
    valid: np.ndarray,
    sample_limit: int,
) -> tuple[np.ndarray, float, float]:
    """将滤波响应转换为带局部噪声缩放的经验 SNR 图。"""

    response_background, response_noise = sigma_clipped_stats(
        response,
        mask=effective_mask,
        sample_limit=sample_limit,
    )
    response_noise = max(response_noise, np.finfo(np.float64).eps)
    median_noise = max(float(np.median(noise_map[valid])), np.finfo(np.float64).eps)
    local_noise_scale = np.maximum(noise_map / median_noise, 0.25)
    response_noise_map = response_noise * local_noise_scale
    return (response - response_background) / response_noise_map, float(response_background), float(response_noise)


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


@lru_cache(maxsize=512)
def _patch_geometry(
    shape: tuple[int, int],
    peak_x: int,
    peak_y: int,
    aperture_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """缓存相同孔径窗口的几何数组，减少数千次小数组分配。"""

    yy, xx = np.indices(shape, dtype=np.float64)
    distance = np.hypot(xx - peak_x, yy - peak_y)
    geometric_aperture = distance <= aperture_radius
    outer_radius = aperture_radius + 4
    annulus = (distance >= aperture_radius + 2) & (distance <= outer_radius)
    return yy, xx, distance, geometric_aperture, annulus


_QUALITY_REJECT_FLAGS = frozenset(
    {
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
        "INSUFFICIENT_PSF_SUPPORT",
        "UNRESOLVED_BLEND",
        "NEGATIVE_OVERFLOW",
        "CODE_PATTERN",
    }
)


def _default_negative_overflow_limit(values: np.ndarray) -> float | None:
    """返回有符号整型图像的极端负码审计线。

    当前比赛 FITS 是有符号 ``BITPIX=16``，但局部可见 ``-3.0e4`` 量级的
    负码只占极少数，并且与接近正满量程的像素成对出现。这与普通背景
    噪声不同，更像 ADC/存储溢出或饱和邻域的异常编码。这里不把所有负值
    都判坏，只对有符号整型的负侧 90% 满量程以外的值做“可疑数据范围”
    审计；浮点图像和无符号图像不套用这个假设。

    该线不是相机真实满阱标定，也不修改像素值；它只让受异常码污染的
    候选不能进入质量星点层，并把候选留在审计表中。
    """

    dtype = np.asarray(values).dtype
    if not np.issubdtype(dtype, np.signedinteger):
        return None
    info = np.iinfo(dtype)
    if info.bits < 16:
        return None
    return -0.9 * float(info.max)


def _repeated_high_code_values(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    background: float,
    noise: float,
    minimum_count: int = 128,
) -> tuple[int, ...]:
    """找出全幅异常重复的高位整型数据码。

    这不是把某个数值硬编码成“饱和线”。当前样本中 `3990--3993` 在
    4096² 图像里各自重复数百至上千次，而同一亮度尾部的普通码频次远低
    于它们；这种全幅频次异常可以作为局部数据有效性审计的先验。只有
    在候选孔径内又同时出现远离背景的负值时，调用方才会把它升级为
    ``CODE_PATTERN``，因此单个合法高亮像素不会因数值恰好相同而被拒绝。

    ``minimum_count`` 是工程审计的最小重复次数，不是相机标定参数；对
    非有符号整型和浮点输入不启用，以免把正常浮点测光值误当作编码问题。
    """

    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.signedinteger):
        return ()
    info = np.iinfo(raw.dtype)
    if info.bits < 16:
        return ()
    if mask.shape != raw.shape:
        raise ValueError("mask shape does not match image for repeated-code audit")
    if minimum_count < 1 or not np.isfinite(float(background)) or not np.isfinite(float(noise)):
        return ()

    numeric = np.asarray(raw, dtype=np.float64)
    valid = ~np.asarray(mask, dtype=bool) & np.isfinite(numeric)
    # 500 ADU 是当前数据背景（约 21 ADU）以上的保守亮部起点；同时用
    # 20σ 防止在不同背景标度下把中等亮星的自然量化重复当作异常码。
    high_floor = max(500.0, float(background) + 20.0 * max(float(noise), 1e-12))
    high_values = raw[valid & (numeric >= high_floor)]
    if high_values.size == 0:
        return ()
    unique, counts = np.unique(high_values, return_counts=True)
    return tuple(
        int(value)
        for value, count in zip(unique, counts, strict=True)
        if int(count) >= int(minimum_count)
    )


def _pair_psf_evidence(
    image: np.ndarray,
    mask: np.ndarray,
    primary_xy: tuple[float, float],
    secondary_xy: tuple[float, float],
    *,
    psf_fwhm: float,
) -> tuple[float | None, float | None]:
    """比较局部单 PSF 与双 PSF 模型，返回 ``(ΔBIC, 次源幅度 SNR)``。

    两个模型共享常数和一阶背景平面，位置固定在宽筛峰上；双源模型只多
    一个非负 PSF 幅度。这样 ``ΔBIC = BIC_single - BIC_pair`` 可直接解释
    为第二个点源是否足以抵偿额外参数，而不会把邻近主星的翼部通量自动
    当成新星。该检验只决定是否解除 ``UNRESOLVED_BLEND``，不绕过原有
    孔径通量、形状、掩膜和伪迹质量规则。
    """

    sigma = max(float(psf_fwhm) / 2.35482, 0.5)
    margin = max(4, int(np.ceil(3.5 * sigma)))
    px, py = primary_xy
    sx, sy = secondary_xy
    height, width = image.shape
    x0 = max(0, int(np.floor(min(px, sx))) - margin)
    x1 = min(width, int(np.ceil(max(px, sx))) + margin + 1)
    y0 = max(0, int(np.floor(min(py, sy))) - margin)
    y1 = min(height, int(np.ceil(max(py, sy))) + margin + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None, None
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    valid = ~np.asarray(mask[y0:y1, x0:x1], dtype=bool) & np.isfinite(patch)
    if np.count_nonzero(valid) < 24:
        return None, None
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    gx = xx + float(x0)
    gy = yy + float(y0)
    primary = np.exp(-0.5 * (((gx - px) / sigma) ** 2 + ((gy - py) / sigma) ** 2))
    secondary = np.exp(-0.5 * (((gx - sx) / sigma) ** 2 + ((gy - sy) / sigma) ** 2))
    center_x = 0.5 * (x0 + x1 - 1)
    center_y = 0.5 * (y0 + y1 - 1)
    scale = max(float(x1 - x0), float(y1 - y0), 1.0)
    background_terms = (
        np.ones(patch.shape, dtype=np.float64),
        (gx - center_x) / scale,
        (gy - center_y) / scale,
    )
    y_values = patch[valid]
    design_single = np.column_stack([term[valid] for term in (*background_terms, primary)])
    design_pair = np.column_stack([term[valid] for term in (*background_terms, primary, secondary)])
    # 这里每个候选只拟合 4/5 个线性系数。原先对每一个近邻候选都调用
    # ``lstsq``，随后再调用一次 ``pinv`` 求协方差；在真实 4096² 图上
    # 数千个去混叠候选会把小矩阵 SVD 的固定开销放大到分钟级。对已经
    # 构造好的设计矩阵改用正规方程，并先检查条件数；病态的近重合源
    # 回退到 SVD，保留原实现的数值安全边界。
    normal_single = design_single.T @ design_single
    normal_pair = design_pair.T @ design_pair
    try:
        beta_single = np.linalg.solve(normal_single, design_single.T @ y_values)
        beta_pair = np.linalg.solve(normal_pair, design_pair.T @ y_values)
        if not np.all(np.isfinite(beta_single)) or not np.all(np.isfinite(beta_pair)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        try:
            beta_single, *_ = np.linalg.lstsq(design_single, y_values, rcond=None)
            beta_pair, *_ = np.linalg.lstsq(design_pair, y_values, rcond=None)
        except np.linalg.LinAlgError:
            return None, None
    if beta_pair[-2] <= 0.0 or beta_pair[-1] <= 0.0:
        return None, None
    residual_single = y_values - design_single @ beta_single
    residual_pair = y_values - design_pair @ beta_pair
    epsilon = np.finfo(np.float64).eps
    rss_single = max(float(residual_single @ residual_single), epsilon)
    rss_pair = max(float(residual_pair @ residual_pair), epsilon)
    sample_count = int(y_values.size)
    bic_single = sample_count * np.log(rss_single / sample_count) + design_single.shape[1] * np.log(sample_count)
    bic_pair = sample_count * np.log(rss_pair / sample_count) + design_pair.shape[1] * np.log(sample_count)
    degrees_of_freedom = max(1, sample_count - design_pair.shape[1])
    variance = rss_pair / degrees_of_freedom
    try:
        inverse_normal_pair = np.linalg.solve(normal_pair, np.eye(normal_pair.shape[0], dtype=np.float64))
        amplitude_error = float(np.sqrt(max(variance * inverse_normal_pair[-1, -1], epsilon)))
    except np.linalg.LinAlgError:
        try:
            covariance = variance * np.linalg.pinv(normal_pair)
            amplitude_error = float(np.sqrt(max(covariance[-1, -1], epsilon)))
        except np.linalg.LinAlgError:
            return float(bic_single - bic_pair), None
    return float(bic_single - bic_pair), float(beta_pair[-1] / amplitude_error)


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
    min_psf_support_pixels: int,
    gain_e_per_adu: float | None,
    read_noise_adu: float,
    refine_local_background: bool,
    psf_fwhm: float,
    proposal_methods: tuple[str, ...],
    proposal_scales: tuple[float, ...],
    proposal_snr: float,
    nearest_gaussian_px: float | None,
    dog_blend_radius_px: float,
    allow_partial_zero_mask: bool = False,
    negative_overflow_limit: float | None = None,
    repeated_high_code_values: tuple[int, ...] = (),
) -> Detection:
    """在候选峰周围做孔径测光和点源形状计算。

    单图模式默认用环形 sigma-clipping 精修背景；15 帧模式可以复用已经
    生成的局部背景/RMS 图，避免对每个候选重复执行统计迭代。
    """

    height, width = image.shape
    outer_radius = aperture_radius + 4
    y0 = max(0, y_peak - outer_radius)
    y1 = min(height, y_peak + outer_radius + 1)
    x0 = max(0, x_peak - outer_radius)
    x1 = min(width, x_peak + outer_radius + 1)
    patch = image[y0:y1, x0:x1]
    patch_mask = mask[y0:y1, x0:x1] | ~np.isfinite(patch)
    yy, xx, distance, geometric_aperture, annulus_geometry = _patch_geometry(
        patch.shape,
        x_peak - x0,
        y_peak - y0,
        aperture_radius,
    )
    aperture = geometric_aperture & ~patch_mask
    annulus = annulus_geometry & ~patch_mask
    annulus_values = patch[annulus]
    fallback_background = float(background_map[y_peak, x_peak])
    fallback_noise = float(noise_map[y_peak, x_peak])
    annulus_is_usable = annulus_values.size >= 16
    if annulus_is_usable and refine_local_background:
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
        # 测光孔径和定位孔径不是同一个问题。把整个孔径的正残差直接
        # 当作质心权重，会把外圈随机正噪声也算进去；在 5--6 的弱源
        # 上，这个偏移足以让标记落到亮点旁边，并污染跨帧关联。定位时
        # 只使用候选峰附近的 PSF 加权、扣除半个噪声尺度后的正信号；
        # 原来的 ``positive`` 仍用于通量形状指标，避免改变测光口径。
        psf_sigma = max(float(psf_fwhm) / 2.35482, 0.5)
        centroid_radius = min(
            float(aperture_radius),
            max(2.0, 2.5 * psf_sigma),
        )
        centroid_mask = aperture & (distance <= centroid_radius)
        centroid_signal = np.maximum(residual - 0.5 * local_noise, 0.0)
        centroid_weights = np.where(centroid_mask, centroid_signal, 0.0)
        centroid_weights *= np.exp(-0.5 * (distance / psf_sigma) ** 2)
        centroid_sum = float(centroid_weights.sum())
        if centroid_sum <= np.finfo(np.float64).eps:
            centroid_weights = positive
            centroid_sum = positive_sum
        x = float((centroid_weights * (xx + x0)).sum() / centroid_sum)
        y = float((centroid_weights * (yy + y0)).sum() / centroid_sum)
        # 形状统计继续围绕“全孔径正残差质心”计算；定位质心和形状
        # 质心分开，避免仅为修正弱源标注位置就改变 FWHM/椭圆率质量
        # 规则的历史口径。
        shape_x = float((positive * (xx + x0)).sum() / positive_sum)
        shape_y = float((positive * (yy + y0)).sum() / positive_sum)
        variance_x = float((positive * ((xx + x0) - shape_x) ** 2).sum() / positive_sum)
        variance_y = float((positive * ((yy + y0) - shape_y) ** 2).sum() / positive_sum)
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
    # 1σ footprint 对“一个很高的单像素 + 一圈随机正噪声”过于宽松：
    # 这种候选的孔径通量、FWHM 和 sharpness 都可能被随机像素伪装成点源。
    # 用候选峰周围 3×3 的原始残差做独立的 PSF 支持审计，阈值同时受
    # 局部噪声和峰值幅度约束。它不改变测光通量，只决定该候选能否进入
    # quality 层；过弱源仍可保留在 candidate 审计层。
    peak_excess = max(float(peak - local_background), 0.0)
    core_y = y_peak - y0
    core_x = x_peak - x0
    core_y0, core_y1 = max(0, core_y - 1), min(residual.shape[0], core_y + 2)
    core_x0, core_x1 = max(0, core_x - 1), min(residual.shape[1], core_x + 2)
    core_slice = residual[core_y0:core_y1, core_x0:core_x1]
    core_valid = ~patch_mask[core_y0:core_y1, core_x0:core_x1]
    psf_support_threshold = max(2.0 * local_noise, 0.1 * peak_excess)
    psf_support_pixels = int(
        np.count_nonzero(
            core_valid
            & np.isfinite(core_slice)
            & (core_slice >= psf_support_threshold)
        )
    )
    repeated_code_values = tuple(int(value) for value in repeated_high_code_values)
    repeated_code_count = int(
        np.count_nonzero(
            geometric_aperture
            & ~patch_mask
            & np.isin(patch, repeated_code_values)
        )
    ) if repeated_code_values else 0
    # 这里的负值线只用于和“全幅异常重复高位码”联合判定。单独出现的
    # 负噪声、精确 -1 值或普通暗像素不触发 CODE_PATTERN；它不是硬件
    # 满阱/BLANK 的替代标定。
    local_range_pattern_limit = min(-1000.0, float(local_background) - 50.0 * local_noise)
    range_anomaly_pixel_count = int(
        np.count_nonzero(
            geometric_aperture
            & ~patch_mask
            & np.isfinite(patch)
            & (patch <= local_range_pattern_limit)
        )
    )
    # CODE_PATTERN 只拒绝“候选峰本身坐落在全幅异常重复高位码上、且孔径
    # 还含远离背景的负值”的检测；它不应因为孔径内恰好含有高位码就拒绝
    # 正常亮星。真实首帧中，亮星的饱和/溢出出血列会把 3991/3992 等重复
    # 码和 -20628 这类负值同时带进孔径，但峰值仍是一个正常的高亮度值；
    # 若把“孔径内含有重复码”当作充分条件，就会把这些亮星误标为数据编码
    # 伪迹。只有峰值本身落在重复码上（此时它通常还紧邻一个更强的真源），
    # 才判定为 CODE_PATTERN。
    peak_is_repeated_code = bool(repeated_code_values) and int(round(peak)) in repeated_code_values
    code_pattern = bool(peak_is_repeated_code and range_anomaly_pixel_count > 0)
    flags: list[str] = []
    if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
        flags.append("EDGE")
    masked_aperture = patch_mask & geometric_aperture
    if masked_aperture.any():
        # 当前 16 位 FITS 中的精确 0 更像稀疏坏像素/未写入值，而不是
        # 整颗星都不可信。只在“候选中心有效 + 被掩膜值全部是有限精确 0
        # + 圆孔径仍保留至少 80% 且不少于 9 个像素”时放宽；任意外部
        # 掩膜、NaN、负值/饱和值或中心被遮住，仍保持 MASKED 严格拒绝。
        masked_values = patch[masked_aperture]
        geometric_pixels = int(np.count_nonzero(geometric_aperture))
        minimum_aperture_valid = max(9, int(np.ceil(0.8 * geometric_pixels)))
        center_valid = not bool(patch_mask[core_y, core_x])
        only_zero_values = bool(
            masked_values.size
            and np.all(np.isfinite(masked_values))
            and np.all(masked_values == 0.0)
        )
        partial_zero_mask = bool(
            allow_partial_zero_mask
            and center_valid
            and only_zero_values
            and aperture_pixels >= minimum_aperture_valid
        )
        flags.append("PARTIAL_MASKED" if partial_zero_mask else "MASKED")
    if saturation_level is not None and np.any(
        geometric_aperture & np.isfinite(patch) & (patch >= saturation_level)
    ):
        flags.append("SATURATED")
    if negative_overflow_limit is not None and np.any(
        geometric_aperture & np.isfinite(patch) & (patch <= negative_overflow_limit)
    ):
        # 只审计极端负码，不把背景扣除后常见的少量负噪声误判成坏点。
        # 当前 FITS 的这类像素在亮峰邻域会与正满量程附近像素同时出现，
        # 若继续做孔径求和，可能把一个饱和/溢出星像拆成第二个“高 SNR”源。
        flags.append("NEGATIVE_OVERFLOW")
    if code_pattern:
        # 该标志保留候选，但拒绝把“重复数据码 + 局部异常负值”的孔径
        # 当作干净点源。它专门覆盖当前截图中 3991/3992 与负值共现、
        # 但尚未达到旧版 -0.9*int16 上限的情况。
        flags.append("CODE_PATTERN")
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
    if psf_support_pixels < min_psf_support_pixels:
        flags.append("INSUFFICIENT_PSF_SUPPORT")
    if (
        "gaussian" not in proposal_methods
        and nearest_gaussian_px is not None
        and nearest_gaussian_px < dog_blend_radius_px
    ):
        # 非 Gaussian 补充峰若紧邻一个 Gaussian 主峰，孔径通量和形状会被邻源
        # 污染。没有联合 PSF 去混叠前保留为宽筛候选，但不能自动进入
        # 最终质量计数。
        flags.append("UNRESOLVED_BLEND")

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
        psf_support_pixels=psf_support_pixels,
        flags=tuple(flags),
        quality_passed=not _QUALITY_REJECT_FLAGS.intersection(flags),
        peak_x=float(x_peak),
        peak_y=float(y_peak),
        centroid_shift_px=float(np.hypot(x - float(x_peak), y - float(y_peak))),
        proposal_methods=proposal_methods,
        proposal_scales=proposal_scales,
        proposal_snr=float(proposal_snr),
        nearest_gaussian_px=nearest_gaussian_px,
        repeated_code_count=repeated_code_count,
        range_anomaly_pixel_count=range_anomaly_pixel_count,
        repeated_code_values=repeated_code_values,
    )


def _source_peak_position(source: Detection) -> tuple[float, float]:
    """返回检测源的整数匹配滤波峰坐标，兼容旧缓存字段缺失。"""

    return (
        float(source.peak_x if source.peak_x is not None else source.x),
        float(source.peak_y if source.peak_y is not None else source.y),
    )


def _source_measurement_position(source: Detection) -> tuple[float, float]:
    """返回源级测量质心，用于近邻重叠审计。"""

    return float(source.x), float(source.y)


_PAIR_AUDIT_FLAGS = frozenset({"CODE_PATTERN", "NEGATIVE_OVERFLOW", "MASKED", "SATURATED"})


def _guard_unresolved_gaussian_pairs(
    sources: Sequence[Detection],
    image: np.ndarray,
    mask: np.ndarray,
    *,
    psf_fwhm: float,
    delta_bic_min: float,
    component_snr_min: float,
) -> tuple[list[Detection], int, int, float]:
    """对极近 Gaussian 候选做一次轻量双 PSF 门控。

    ``min_distance`` 只约束匹配滤波峰的整数坐标，不能证明两个测量源
    是两个独立天体；质心会把两个相邻峰重新拉近，亮斑的非高斯翼部也
    可能形成两个峰。因此按源级测量质心寻找约 ``1.5 FWHM`` 内的近邻，
    并纳入少量带值域审计旗标的强 Gaussian 邻峰。这样可以覆盖“主峰先
    因 ``CODE_PATTERN`` 或溢出被拒、副峰却仍通过普通质量规则”的情况，
    同时不把所有低 SNR 拒绝候选成批送入昂贵的 PSF 拟合。双模型没有同时
    达到 ΔBIC 和次分量 SNR 证据线时，仅将较弱分量标成
    ``UNRESOLVED_BLEND``，主峰和两个候选都保留在审计层。

    这是“候选层全保留、质量层不重复计数”的折中：它不声称相距更远
    的源不可分辨，也不把一个成功的局部模型直接升级为星表身份。
    返回值为 ``(sources, tested_pairs, guarded_sources, radius_px)``。
    """

    if psf_fwhm <= 0 or delta_bic_min <= 0 or component_snr_min <= 0:
        raise ValueError("PSF pair guard parameters must be positive")
    if not sources:
        return [], 0, 0, max(2.5, 1.5 * float(psf_fwhm))
    radius_px = max(2.5, 1.5 * float(psf_fwhm))
    eligible_indices = [
        index
        for index, source in enumerate(sources)
        if "gaussian" in source.proposal_methods
        and (
            source.quality_passed
            or bool(_PAIR_AUDIT_FLAGS.intersection(source.flags))
        )
    ]
    if len(eligible_indices) < 2:
        return list(sources), 0, 0, radius_px
    coordinates = np.asarray(
        [
            _source_measurement_position(source)
            for source in (sources[index] for index in eligible_indices)
        ],
        dtype=np.float64,
    )
    tree = cKDTree(coordinates)
    pair_indices = sorted(tree.query_pairs(radius_px))
    if not pair_indices:
        return list(sources), 0, 0, radius_px

    replacements: dict[int, tuple[float | None, float | None]] = {}
    tested_pairs = 0
    for local_i, local_j in pair_indices:
        source_i = sources[eligible_indices[local_i]]
        source_j = sources[eligible_indices[local_j]]
        primary_index = eligible_indices[local_i]
        secondary_index = eligible_indices[local_j]
        # 保留峰值更强的源作为当前 blend 的代表；若峰值相同，再用
        # 匹配滤波 SNR 和孔径通量稳定打破平局。
        strength_i = (
            float(source_i.peak),
            float(source_i.filter_snr if source_i.filter_snr is not None else source_i.snr),
            float(source_i.flux),
        )
        strength_j = (
            float(source_j.peak),
            float(source_j.filter_snr if source_j.filter_snr is not None else source_j.snr),
            float(source_j.flux),
        )
        if strength_j > strength_i:
            primary_index, secondary_index = secondary_index, primary_index
            source_i, source_j = source_j, source_i
        delta_bic, component_snr = _pair_psf_evidence(
            np.asarray(image),
            np.asarray(mask, dtype=bool),
            _source_peak_position(source_i),
            _source_peak_position(source_j),
            psf_fwhm=float(psf_fwhm),
        )
        tested_pairs += 1
        pair_supported = bool(
            delta_bic is not None
            and component_snr is not None
            and delta_bic >= float(delta_bic_min)
            and component_snr >= float(component_snr_min)
        )
        if not pair_supported:
            # 只压掉较弱的重复质量计数；主峰代表仍可作为一个混合源
            # 保留，候选视图仍会显示另一局部响应和完整落选原因。
            replacements[secondary_index] = (delta_bic, component_snr)

    if not replacements:
        return list(sources), tested_pairs, 0, radius_px
    guarded: list[Detection] = []
    for index, source in enumerate(sources):
        evidence = replacements.get(index)
        if evidence is None:
            guarded.append(source)
            continue
        delta_bic, component_snr = evidence
        flags = tuple(source.flags)
        if "UNRESOLVED_BLEND" not in flags:
            flags = (*flags, "UNRESOLVED_BLEND")
        guarded.append(
            replace(
                source,
                flags=flags,
                quality_passed=False,
                deblend_delta_bic=delta_bic,
                deblend_component_snr=component_snr,
            )
        )
    return guarded, tested_pairs, len(replacements), radius_px


def _working_mask(
    values: np.ndarray,
    mask: np.ndarray | None,
    *,
    background: float,
    mask_zero_pixels: bool | None,
    saturation_level: float | None,
    numeric: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None, bool]:
    """生成通用无效像素掩膜，并返回自动推断的饱和上限。"""

    raw_values = np.asarray(values)
    numeric_values = np.asarray(values, dtype=np.float64) if numeric is None else np.asarray(numeric)
    effective_mask = np.zeros(raw_values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if effective_mask.shape != raw_values.shape:
        raise ValueError(f"mask shape {effective_mask.shape} does not match image shape {raw_values.shape}")
    effective_mask |= ~np.isfinite(numeric_values)
    is_integer = np.issubdtype(raw_values.dtype, np.integer)
    if mask_zero_pixels is None:
        mask_zero_pixels = bool(is_integer and np.isfinite(background) and abs(background) > 1.0)
    if mask_zero_pixels:
        effective_mask |= numeric_values == 0.0

    inferred_saturation = saturation_level
    if inferred_saturation is None and is_integer:
        dtype_info = np.iinfo(raw_values.dtype)
        inferred_saturation = float(dtype_info.max - 32)
        effective_mask |= numeric_values >= inferred_saturation
        effective_mask |= numeric_values <= float(dtype_info.min + 32)
    elif saturation_level is not None:
        effective_mask |= numeric_values >= saturation_level
    return effective_mask, inferred_saturation, bool(mask_zero_pixels)


def build_background_model(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    background_box_size: int = 128,
    background_sample_limit: int = 100_000,
    mask_zero_pixels: bool | None = None,
    saturation_level: float | None = None,
    refine_local_background: bool = True,
    use_float32: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """为同一组序列帧建立可复用的局部背景/RMS 模型。

    15 张开运一号图像来自同一段连续观测，背景场和读出噪声在帧间通常
    变化远小于星点位置。序列快速路径可以先用首帧生成二维背景模型，后续
    帧仍各自计算全局背景、坏点和饱和掩膜，但跳过重复的 16×16 网格
    sigma-clipping 与插值。模型只在 ``analyze_sequence`` 的临时生命周期
    内使用，不写入缓存，也不会改变单图精测的默认口径。

    返回值是 ``(background_map, noise_map)``；调用方必须检查图像尺寸，
    ``detect_sources`` 会在尺寸不一致时安全回退到逐帧模型。
    """

    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if background_box_size < 16 or background_sample_limit < 1:
        raise ValueError("background_box_size must be at least 16 and sample_limit must be positive")
    calculation_dtype = np.float32 if (
        use_float32 and (np.issubdtype(values.dtype, np.integer) or values.dtype == np.float32)
    ) else np.float64
    numeric = np.asarray(values, dtype=calculation_dtype)
    finite_mask = ~np.isfinite(numeric)
    base_mask = finite_mask if mask is None else (np.asarray(mask, dtype=bool) | finite_mask)
    if base_mask.shape != values.shape:
        raise ValueError(f"mask shape {base_mask.shape} does not match image shape {values.shape}")
    global_background, _global_noise = sigma_clipped_stats(
        numeric,
        mask=base_mask,
        sample_limit=1_000_000,
    )
    effective_mask, _saturation, _used_zero = _working_mask(
        values,
        mask,
        background=global_background,
        mask_zero_pixels=mask_zero_pixels,
        saturation_level=saturation_level,
        numeric=numeric,
    )
    background, noise = sigma_clipped_stats(
        numeric,
        mask=effective_mask,
        sample_limit=1_000_000,
    )
    return local_background_rms(
        numeric,
        effective_mask,
        box_size=background_box_size,
        fallback=(background, noise),
        iterations=2 if not refine_local_background else 4,
        sample_limit=background_sample_limit,
    )


def detect_sources(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold_sigma: float = 4.0,
    min_distance: int = 3,
    aperture_radius: int = 4,
    max_sources: int | None = None,
    saturation_level: float | None = None,
    negative_overflow_limit: float | None = None,
    psf_fwhm: float = 3.0,
    background_box_size: int = 128,
    background_sample_limit: int = 100_000,
    min_flux_snr: float = 5.0,
    min_fwhm: float = 0.8,
    max_fwhm: float = 12.0,
    max_ellipticity: float = 0.65,
    min_sharpness: float = 0.005,
    max_sharpness: float = 0.85,
    min_footprint_pixels: int = 2,
    min_psf_support_pixels: int = 3,
    gain_e_per_adu: float | None = None,
    read_noise_adu: float = 0.0,
    mask_zero_pixels: bool | None = None,
    allow_partial_zero_mask: bool | None = None,
    reject_linear_artifacts: bool = True,
    proposal_mode: str = "gaussian",
    dog_threshold_sigma: float | None = None,
    dog_min_peak_sigma: float = 2.0,
    dog_blend_radius_factor: float = 2.5,
    dog_scale_factors: tuple[float, float] = (0.75, 1.60),
    starlet_threshold_sigma: float | None = None,
    starlet_min_peak_sigma: float = 2.5,
    deblend_delta_bic_min: float = 10.0,
    deblend_component_snr_min: float = 5.0,
    deblend_primary_snr_min: float = 12.0,
    deblend_min_residual_sigma: float = 4.0,
    deblend_search_radius_factor: float = 2.0,
    enable_local_deblend: bool = False,
    refine_local_background: bool = True,
    use_float32: bool = False,
    fast_sequence: bool = False,
    background_model: tuple[np.ndarray, np.ndarray] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> DetectionResult:
    """检测点源候选并进行可解释的局部质量判定。

    检测阶段至少对背景扣除图像做 Gaussian PSF 匹配滤波；当
    ``proposal_mode='hybrid'`` 时，再用两个尺度的 DoG 响应补充候选；
    ``proposal_mode='ensemble'`` 还增加两层 à trous/starlet 小波提案。
    同一位置合并后保留算法来源。测量阶段在原始图像上用局部环
    估计背景，计算净通量、误差和 ``flux_snr``，最后按点源形状和数据
    有效性打标签。默认不限制源数量；只有调用方显式传入
    ``max_sources`` 才会截断返回列表。

    ``gain_e_per_adu`` 和 ``read_noise_adu`` 未知时不虚构仪器噪声参数：
    flux SNR 至少包含孔径内背景噪声和局部背景估计误差；若提供增益，
    再加入源光子的 Poisson 方差。

    ``refine_local_background=False`` 只复用已经生成的局部背景/RMS 图，
    并在序列路径使用两轮网格稳健裁剪，适合 15 帧配准的快速工作集；
    单图精确星等路径默认保持为四轮裁剪和环形局部精修。
    ``use_float32=True`` 只在整数/float32 输入上启用精确的低带宽中间阵列，
    适合 15 帧序列；单图默认保持 float64。
    ``fast_sequence=True`` 是序列专用的低延迟路径：在掩膜稀疏时省略
    归一化卷积的第二次 Gaussian pass；全图线性伪迹连通域审计仍保留，
    点源形状、通量 SNR、边缘和饱和规则也照常执行。单图科学测光不应启用它。
    ``background_model`` 可传入同尺寸的 ``(background_map, noise_map)``，
    供连续序列复用首帧背景；尺寸不一致时自动回退到逐帧网格模型。
    ``deblend_primary_snr_min``、``deblend_min_residual_sigma`` 和
    ``deblend_search_radius_factor`` 控制局部双源宽筛。它只在非快速
    单帧路径、且 ``enable_local_deblend=True`` 时启用；快速 15 帧路径
    保留 Gaussian/DoG/Starlet 候选和现有质量规则，把去混叠留给单帧
    精测，避免把序列吞吐拖慢。默认关闭是为了让常规抽测不会意外进入
    分钟级的局部联合拟合。

    有符号整型输入还会对测光孔径内接近负侧满量程的极端负码增加
    ``NEGATIVE_OVERFLOW`` 标志。它只拒绝受污染的质量源，不删除候选，
    也不是相机真实饱和阈值；真实满阱线仍需设备标定。若输入图像是
    为注入/仿真而转换过的浮点副本，可以显式传入原始整型图像推导的
    ``negative_overflow_limit``，避免在转换后丢失这项数据有效性审计。
    """

    def report(value: float, label: str) -> None:
        if progress is not None:
            progress(max(0.0, min(100.0, float(value))), label)

    report(2.0, "准备检测")
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive")
    if min_distance < 1 or aperture_radius < 1:
        raise ValueError("min_distance and aperture_radius must be positive")
    if psf_fwhm <= 0:
        raise ValueError("psf_fwhm must be positive")
    if min_flux_snr <= 0 or background_box_size < 16 or background_sample_limit < 1:
        raise ValueError("min_flux_snr and background_sample_limit must be positive; background_box_size must be at least 16")
    if min_fwhm <= 0 or max_fwhm < min_fwhm:
        raise ValueError("fwhm limits are invalid")
    if not 0 <= max_ellipticity <= 1:
        raise ValueError("max_ellipticity must be between 0 and 1")
    if min_sharpness < 0 or max_sharpness <= min_sharpness:
        raise ValueError("sharpness limits are invalid")
    if min_footprint_pixels < 1:
        raise ValueError("min_footprint_pixels must be positive")
    if not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_psf_support_pixels must be between 1 and 9")
    if gain_e_per_adu is not None and gain_e_per_adu <= 0:
        raise ValueError("gain_e_per_adu must be positive when provided")
    if read_noise_adu < 0:
        raise ValueError("read_noise_adu cannot be negative")
    if negative_overflow_limit is not None and not np.isfinite(float(negative_overflow_limit)):
        raise ValueError("negative_overflow_limit must be finite when provided")
    proposal_mode = str(proposal_mode).strip().lower()
    if proposal_mode not in {"gaussian", "hybrid", "ensemble"}:
        raise ValueError("proposal_mode must be 'gaussian', 'hybrid', or 'ensemble'")
    if dog_threshold_sigma is not None and dog_threshold_sigma <= 0:
        raise ValueError("dog_threshold_sigma must be positive when provided")
    if dog_min_peak_sigma <= 0:
        raise ValueError("dog_min_peak_sigma must be positive")
    if dog_blend_radius_factor <= 0:
        raise ValueError("dog_blend_radius_factor must be positive")
    if len(dog_scale_factors) != 2 or not (0 < dog_scale_factors[0] < 1.0 < dog_scale_factors[1]):
        raise ValueError("dog_scale_factors must contain one scale below 1 and one above 1")
    if starlet_threshold_sigma is not None and starlet_threshold_sigma <= 0:
        raise ValueError("starlet_threshold_sigma must be positive when provided")
    if starlet_min_peak_sigma <= 0:
        raise ValueError("starlet_min_peak_sigma must be positive")
    if deblend_delta_bic_min <= 0 or deblend_component_snr_min <= 0:
        raise ValueError("deblend BIC and component SNR thresholds must be positive")
    if deblend_primary_snr_min <= 0 or deblend_min_residual_sigma <= 0 or deblend_search_radius_factor <= 0:
        raise ValueError("local deblend thresholds must be positive")

    # 当前 FITS 是 16 位整数，float32 可以精确表示每个输入 ADU，并把
    # 4096² 像素级中间阵列的带宽/内存压力减半。单图默认仍是 float64；
    # 序列入口才显式打开这个实现级优化，避免改变单图科学测光口径。
    use_float32 = bool(use_float32 and (np.issubdtype(values.dtype, np.integer) or values.dtype == np.float32))
    calculation_dtype = np.float32 if use_float32 else np.float64
    # 全图统计决定整体背景、零值掩膜和滤波噪声，继续使用原有的大样本
    # 口径；序列优化只抽样每个局部网格块，避免把全局阈值的变化误当成
    # 性能优化带来的科学结果。
    statistics_sample_limit = 1_000_000
    numeric = np.asarray(values, dtype=calculation_dtype)
    finite_mask = ~np.isfinite(numeric)
    base_mask = finite_mask if mask is None else (np.asarray(mask, dtype=bool) | finite_mask)
    if base_mask.shape != values.shape:
        raise ValueError(f"mask shape {base_mask.shape} does not match image shape {values.shape}")
    global_background, global_noise = sigma_clipped_stats(
        numeric,
        mask=base_mask,
        sample_limit=statistics_sample_limit,
    )
    effective_mask, inferred_saturation, used_zero_mask = _working_mask(
        values,
        mask,
        background=global_background,
        mask_zero_pixels=mask_zero_pixels,
        saturation_level=saturation_level,
        numeric=numeric,
    )
    negative_overflow_limit = (
        _default_negative_overflow_limit(values)
        if negative_overflow_limit is None
        else float(negative_overflow_limit)
    )
    if allow_partial_zero_mask is None:
        # 只对本检测器识别出的精确零值掩膜启用局部容错；显式传入的
        # 非零外部 mask 仍会在源级测量中严格标记为 MASKED。
        allow_partial_zero_mask = bool(used_zero_mask)
    background, noise = sigma_clipped_stats(
        numeric,
        mask=effective_mask,
        sample_limit=statistics_sample_limit,
    )
    repeated_high_code_values = _repeated_high_code_values(
        values,
        effective_mask,
        background=float(background),
        noise=float(noise),
    )
    background_model_mode = "per_frame_local"
    if background_model is not None:
        candidate_background, candidate_noise = background_model
        candidate_background = np.asarray(candidate_background)
        candidate_noise = np.asarray(candidate_noise)
        if candidate_background.shape == values.shape and candidate_noise.shape == values.shape:
            # 模型由序列首帧生成，后续检测只读不写；转换只在 dtype 不一致
            # 时发生，避免每帧再次复制 2 个 128 MiB 的数组。
            background_map = np.asarray(candidate_background, dtype=calculation_dtype)
            noise_map = np.asarray(candidate_noise, dtype=calculation_dtype)
            background_model_mode = "shared_sequence_pilot"
        else:
            # 不同尺寸的帧不能共用空间模型。安全回退到逐帧模型，而不是
            # 为了速度广播/裁剪一个错误的背景场。
            background_map, noise_map = local_background_rms(
                numeric,
                effective_mask,
                box_size=background_box_size,
                fallback=(background, noise),
                iterations=2 if not refine_local_background else 4,
                sample_limit=background_sample_limit,
            )
            background_model_mode = "per_frame_local_shape_fallback"
    else:
        background_map, noise_map = local_background_rms(
            numeric,
            effective_mask,
            box_size=background_box_size,
            fallback=(background, noise),
            iterations=2 if not refine_local_background else 4,
            sample_limit=background_sample_limit,
        )
    report(22.0, "计算局部背景与噪声")

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
    # Gaussian 仍是稳定基线；hybrid 只增加候选提案，不改变后续原图
    # 测光和质量规则。
    filtered, masked_filter_mode = _masked_gaussian_response(
        residual,
        valid,
        sigma=psf_sigma,
        calculation_dtype=calculation_dtype,
        fast_sequence=fast_sequence,
    )
    filter_snr_image, filter_background, filter_noise = _standardize_response(
        filtered,
        effective_mask=effective_mask,
        noise_map=noise_map,
        valid=valid,
        sample_limit=statistics_sample_limit,
    )
    gaussian_proposals = _response_proposals(
        filter_snr_image,
        valid,
        threshold_sigma=threshold_sigma,
        min_distance=min_distance,
        scale_fwhm=psf_fwhm,
        method="gaussian",
    )
    report(43.0, f"Gaussian 提案 {len(gaussian_proposals):,}")

    extra_proposals: list[_Proposal] = []
    resolved_dog_threshold = float(max(6.0, threshold_sigma + 2.0) if dog_threshold_sigma is None else dog_threshold_sigma)
    resolved_starlet_threshold = float(
        max(7.0, threshold_sigma + 3.0) if starlet_threshold_sigma is None else starlet_threshold_sigma
    )
    dog_blend_radius_px = float(dog_blend_radius_factor * psf_fwhm)
    dog_narrow_count = 0
    dog_broad_count = 0
    starlet_scale1_count = 0
    starlet_scale2_count = 0
    raw_local_max: np.ndarray | None = None
    if proposal_mode in {"hybrid", "ensemble"}:
        narrow_factor, broad_factor = (float(dog_scale_factors[0]), float(dog_scale_factors[1]))
        # DoG 对负点源周围的正环也会有强响应。宽筛候选必须同时落在
        # 原始背景扣除图的局部正峰，并达到最低局部峰值显著性；否则反相
        # 图会产生大量没有正向像素证据的环状候选。
        raw_local_max = residual == ndimage.maximum_filter(residual, size=3, mode="nearest")
        dog_candidate_gate = valid & raw_local_max & (residual >= float(dog_min_peak_sigma) * noise_map)
        narrow, _narrow_mode = _masked_gaussian_response(
            residual,
            valid,
            sigma=psf_sigma * narrow_factor,
            calculation_dtype=calculation_dtype,
            fast_sequence=fast_sequence,
        )
        dog_narrow_snr, _dog_narrow_background, _dog_narrow_noise = _standardize_response(
            narrow - filtered,
            effective_mask=effective_mask,
            noise_map=noise_map,
            valid=valid,
            sample_limit=statistics_sample_limit,
        )
        dog_narrow = _response_proposals(
            dog_narrow_snr,
            valid,
            threshold_sigma=resolved_dog_threshold,
            min_distance=min_distance,
            scale_fwhm=psf_fwhm * float(np.sqrt(narrow_factor)),
            method="dog_narrow",
            candidate_gate=dog_candidate_gate,
        )
        dog_narrow_count = len(dog_narrow)
        extra_proposals.extend(dog_narrow)
        del narrow, dog_narrow_snr, dog_narrow
        report(48.0, f"DoG 窄尺度提案 {dog_narrow_count:,}")

        broad, _broad_mode = _masked_gaussian_response(
            residual,
            valid,
            sigma=psf_sigma * broad_factor,
            calculation_dtype=calculation_dtype,
            fast_sequence=fast_sequence,
        )
        dog_broad_snr, _dog_broad_background, _dog_broad_noise = _standardize_response(
            filtered - broad,
            effective_mask=effective_mask,
            noise_map=noise_map,
            valid=valid,
            sample_limit=statistics_sample_limit,
        )
        dog_broad = _response_proposals(
            dog_broad_snr,
            valid,
            threshold_sigma=resolved_dog_threshold,
            min_distance=min_distance,
            scale_fwhm=psf_fwhm * float(np.sqrt(broad_factor)),
            method="dog_broad",
            candidate_gate=dog_candidate_gate,
        )
        dog_broad_count = len(dog_broad)
        extra_proposals.extend(dog_broad)
        del broad, dog_broad_snr, dog_broad
        report(53.0, f"DoG 宽尺度提案 {dog_broad_count:,}")

    if proposal_mode == "ensemble":
        if raw_local_max is None:  # 防御式保护；ensemble 必然先执行 DoG 分支。
            raw_local_max = residual == ndimage.maximum_filter(residual, size=3, mode="nearest")
        starlet_gate = valid & raw_local_max & (residual >= float(starlet_min_peak_sigma) * noise_map)
        starlet_c1, _starlet_mode1 = _masked_starlet_smooth(
            residual,
            valid,
            step=1,
            calculation_dtype=calculation_dtype,
            fast_sequence=fast_sequence,
        )
        starlet_w1_snr, _starlet_w1_background, _starlet_w1_noise = _standardize_response(
            residual - starlet_c1,
            effective_mask=effective_mask,
            noise_map=noise_map,
            valid=valid,
            sample_limit=statistics_sample_limit,
        )
        starlet_scale1 = _response_proposals(
            starlet_w1_snr,
            valid,
            threshold_sigma=resolved_starlet_threshold,
            min_distance=min_distance,
            scale_fwhm=1.7,
            method="starlet_s1",
            candidate_gate=starlet_gate,
        )
        starlet_scale1_count = len(starlet_scale1)
        extra_proposals.extend(starlet_scale1)
        del starlet_w1_snr, starlet_scale1
        report(56.0, f"Starlet S1 提案 {starlet_scale1_count:,}")

        starlet_c2, _starlet_mode2 = _masked_starlet_smooth(
            starlet_c1,
            valid,
            step=2,
            calculation_dtype=calculation_dtype,
            fast_sequence=fast_sequence,
        )
        starlet_w2_snr, _starlet_w2_background, _starlet_w2_noise = _standardize_response(
            starlet_c1 - starlet_c2,
            effective_mask=effective_mask,
            noise_map=noise_map,
            valid=valid,
            sample_limit=statistics_sample_limit,
        )
        starlet_scale2 = _response_proposals(
            starlet_w2_snr,
            valid,
            threshold_sigma=resolved_starlet_threshold,
            min_distance=min_distance,
            scale_fwhm=3.5,
            method="starlet_s2",
            candidate_gate=starlet_gate,
        )
        starlet_scale2_count = len(starlet_scale2)
        extra_proposals.extend(starlet_scale2)
        del starlet_c1, starlet_c2, starlet_w2_snr, starlet_scale2
        report(59.0, f"Starlet S2 提案 {starlet_scale2_count:,}")

    selected = _merge_proposals(gaussian_proposals, extra_proposals, min_distance=min_distance)
    deblend_proposals: list[_Proposal] = []
    if enable_local_deblend and not fast_sequence:
        # 这是“宽筛”的补充通道，不改变 Gaussian/DoG 的常规候选排序。
        # 候选先经过单主峰残差搜索，后面还会经过原图测光与双 PSF/BIC，
        # 所以不能把这里的数量直接解释为新增恒星数量。
        deblend_proposals = _local_deblend_proposals(
            numeric,
            valid,
            line_artifact_mask,
            gaussian_proposals,
            background_map=background_map,
            noise_map=noise_map,
            psf_fwhm=psf_fwhm,
            min_primary_snr=deblend_primary_snr_min,
            min_residual_sigma=deblend_min_residual_sigma,
            search_radius_factor=deblend_search_radius_factor,
        )
        selected = _append_deblend_proposals(
            selected,
            deblend_proposals,
            gaussian_primary=gaussian_proposals,
            min_distance=min_distance,
        )
        report(60.0, f"局部去混叠提案 {len(deblend_proposals):,}")
    candidate_count = len(selected)
    candidate_peaks: np.ndarray | tuple[tuple[int, int, float], ...]
    candidate_peaks = (
        np.asarray(
            [(int(item.x), int(item.y), float(item.score)) for item in selected],
            dtype=np.float32,
        )
        if fast_sequence
        else ()
    )
    if max_sources is not None:
        if max_sources < 1:
            raise ValueError("max_sources must be positive when provided")
        selected = selected[:max_sources]
    report(62.0, f"候选峰 {candidate_count:,} · 开始源级测量")

    sources: list[Detection] = []
    measurement_total = len(selected)
    report_step = max(1, measurement_total // 20)
    for index, candidate in enumerate(selected, start=1):
        x_peak = int(candidate.x)
        y_peak = int(candidate.y)
        candidate_filter_snr = float(filter_snr_image[y_peak, x_peak])
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
                min_psf_support_pixels,
                gain_e_per_adu,
                read_noise_adu,
                refine_local_background,
                psf_fwhm,
                tuple(candidate.methods),
                tuple(candidate.scales),
                float(candidate.score),
                candidate.nearest_gaussian_px,
                dog_blend_radius_px,
                allow_partial_zero_mask=bool(allow_partial_zero_mask),
                negative_overflow_limit=negative_overflow_limit,
                repeated_high_code_values=repeated_high_code_values,
            )
        )
        if index == measurement_total or index % report_step == 0:
            report(62.0 + 32.0 * index / max(1, measurement_total), f"源级测量 {index:,}/{measurement_total:,}")

    # 对非 Gaussian 近邻候选做局部联合 PSF 证据检验。Gaussian 候选仍是
    # 主分量；只有双源模型相对单源模型取得强 BIC 优势，且新增分量自身
    # 幅度显著，才解除 UNRESOLVED_BLEND。其他质量旗标原样保留。
    gaussian_sources = [source for source in sources if "gaussian" in source.proposal_methods]
    if gaussian_sources:
        gaussian_tree = cKDTree(
            np.asarray(
                [
                    (
                        source.peak_x if source.peak_x is not None else source.x,
                        source.peak_y if source.peak_y is not None else source.y,
                    )
                    for source in gaussian_sources
                ],
                dtype=np.float64,
            )
        )
        resolved_sources: list[Detection] = []
        for source in sources:
            if "UNRESOLVED_BLEND" not in source.flags:
                resolved_sources.append(source)
                continue
            secondary_xy = (
                float(source.peak_x if source.peak_x is not None else source.x),
                float(source.peak_y if source.peak_y is not None else source.y),
            )
            _distance, neighbor_index = gaussian_tree.query(np.asarray(secondary_xy), k=1)
            neighbor = gaussian_sources[int(neighbor_index)]
            primary_xy = (
                float(neighbor.peak_x if neighbor.peak_x is not None else neighbor.x),
                float(neighbor.peak_y if neighbor.peak_y is not None else neighbor.y),
            )
            delta_bic, component_snr = _pair_psf_evidence(
                numeric,
                effective_mask,
                primary_xy,
                secondary_xy,
                psf_fwhm=psf_fwhm,
            )
            pair_supported = (
                delta_bic is not None
                and component_snr is not None
                and delta_bic >= float(deblend_delta_bic_min)
                and component_snr >= float(deblend_component_snr_min)
            )
            flags = (
                tuple(flag for flag in source.flags if flag != "UNRESOLVED_BLEND")
                if pair_supported
                else source.flags
            )
            resolved_sources.append(
                replace(
                    source,
                    flags=flags,
                    quality_passed=not _QUALITY_REJECT_FLAGS.intersection(flags),
                    deblend_delta_bic=delta_bic,
                    deblend_component_snr=component_snr,
                )
            )
        sources = resolved_sources
    sources, close_pair_tested_count, close_pair_guarded_count, close_pair_radius_px = _guard_unresolved_gaussian_pairs(
        sources,
        numeric,
        effective_mask,
        psf_fwhm=float(psf_fwhm),
        delta_bic_min=float(deblend_delta_bic_min),
        component_snr_min=float(deblend_component_snr_min),
    )
    sources.sort(key=lambda source: (source.y, source.x))
    report(98.0, "整理质量标记")
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
            psf_support_pixels=source.psf_support_pixels,
            quality_passed=source.quality_passed,
            peak_x=source.peak_x,
            peak_y=source.peak_y,
            centroid_shift_px=source.centroid_shift_px,
            proposal_methods=source.proposal_methods,
            proposal_scales=source.proposal_scales,
            proposal_snr=source.proposal_snr,
            nearest_gaussian_px=source.nearest_gaussian_px,
            deblend_delta_bic=source.deblend_delta_bic,
            deblend_component_snr=source.deblend_component_snr,
            repeated_code_count=source.repeated_code_count,
            range_anomaly_pixel_count=source.range_anomaly_pixel_count,
            repeated_code_values=source.repeated_code_values,
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
        candidate_peaks=candidate_peaks,
        line_artifact_coordinates=(
            np.column_stack(np.nonzero(line_artifact_mask)[::-1]).astype(np.int32, copy=False)
            if line_artifact_mask is not None and np.any(line_artifact_mask)
            else np.empty((0, 2), dtype=np.int32)
        ),
        parameters={
            "threshold_sigma": float(threshold_sigma),
            "min_distance": int(min_distance),
            "aperture_radius": int(aperture_radius),
            "max_sources": -1 if max_sources is None else int(max_sources),
            "psf_fwhm": float(psf_fwhm),
            "background_box_size": int(background_box_size),
            "background_sample_limit": int(background_sample_limit),
            "statistics_sample_limit": int(statistics_sample_limit),
            "min_flux_snr": float(min_flux_snr),
            "min_fwhm": float(min_fwhm),
            "max_fwhm": float(max_fwhm),
            "max_ellipticity": float(max_ellipticity),
            "min_sharpness": float(min_sharpness),
            "max_sharpness": float(max_sharpness),
            "min_footprint_pixels": int(min_footprint_pixels),
            "min_psf_support_pixels": int(min_psf_support_pixels),
            "gain_e_per_adu": -1.0 if gain_e_per_adu is None else float(gain_e_per_adu),
            "read_noise_adu": float(read_noise_adu),
            "mask_zero_pixels": int(used_zero_mask),
            "allow_partial_zero_mask": int(bool(allow_partial_zero_mask)),
            "partial_mask_policy": (
                "exact_zero_center_valid_aperture_ge_80pct"
                if allow_partial_zero_mask
                else "disabled"
            ),
            "reject_linear_artifacts": int(reject_linear_artifacts),
            "proposal_mode": proposal_mode,
            "dog_threshold_sigma": float(resolved_dog_threshold),
            "dog_min_peak_sigma": float(dog_min_peak_sigma),
            "dog_blend_radius_factor": float(dog_blend_radius_factor),
            "dog_blend_radius_px": float(dog_blend_radius_px),
            "dog_narrow_factor": float(dog_scale_factors[0]),
            "dog_broad_factor": float(dog_scale_factors[1]),
            "starlet_threshold_sigma": float(resolved_starlet_threshold),
            "starlet_min_peak_sigma": float(starlet_min_peak_sigma),
            "deblend_delta_bic_min": float(deblend_delta_bic_min),
            "deblend_component_snr_min": float(deblend_component_snr_min),
            "deblend_primary_snr_min": float(deblend_primary_snr_min),
            "deblend_min_residual_sigma": float(deblend_min_residual_sigma),
            "deblend_search_radius_factor": float(deblend_search_radius_factor),
            "enable_local_deblend": int(bool(enable_local_deblend)),
            "repeated_high_code_values": list(repeated_high_code_values),
            "repeated_high_code_min_count": 128,
            "code_pattern_negative_limit_adu": -1000.0,
            "close_pair_guard_radius_px": float(close_pair_radius_px),
            "close_pair_guard_tested_count": int(close_pair_tested_count),
            "close_pair_guarded_source_count": int(close_pair_guarded_count),
            "proposal_count_deblend_local": int(len(deblend_proposals)),
            "proposal_count_gaussian": len(gaussian_proposals),
            "proposal_count_dog_narrow": int(dog_narrow_count),
            "proposal_count_dog_broad": int(dog_broad_count),
            "proposal_count_starlet_s1": int(starlet_scale1_count),
            "proposal_count_starlet_s2": int(starlet_scale2_count),
            "proposal_count_merged": int(candidate_count),
            "refine_local_background": int(refine_local_background),
            "background_clip_iterations": 4 if refine_local_background else 2,
            "fast_sequence": int(fast_sequence),
            "masked_filter_mode": masked_filter_mode,
            "background_stats_mode": (
                "sampled_grid_4096"
                if use_float32 and not refine_local_background
                else "full_grid"
            ),
            "background_model_mode": background_model_mode,
            "use_float32": int(use_float32),
            "line_artifact_pixels": 0 if line_artifact_mask is None else int(line_artifact_mask.sum()),
            "saturation_level": -1.0 if inferred_saturation is None else float(inferred_saturation),
            "negative_overflow_limit": (
                None if negative_overflow_limit is None else float(negative_overflow_limit)
            ),
            "global_background": float(global_background),
            "global_noise": float(global_noise),
            "filter_noise": float(filter_noise),
            "centroid_method": "psf_weighted_noise_floor",
        },
    )
