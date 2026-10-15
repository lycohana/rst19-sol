"""真实 FITS 背景上的空间分区留出注入审计。

该模块回答的是检测器在不同 detector 区域的条件性灵敏度是否一致，
而不是把不同区域的检测数量直接解释成恒星密度。注入位置先远离当前
候选、特殊值和局部亮结构，再在每个空间单元内用已知信号注入；候选层
和质量层回收率分别记录。结果只用于空间 PSF/噪声的校准诊断，不修改
默认检测器、GUI 质量层或缓存。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .detection import Detection, _default_negative_overflow_limit, detect_sources
from .experiments import (
    EmpiricalPSF,
    _RealInjectionSite,
    _count_outside_injection_regions,
    _greedy_real_injection_sites,
    _inject_source_signal,
    _local_patch_stats,
    _matched_count,
    _real_injection_special_mask,
    estimate_empirical_psf,
)
from .fits import FitsFrame, auxiliary_mask, read_fits


@dataclass(frozen=True, slots=True)
class SpatialInjectionAuditRow:
    """一个空间单元和控制信号档位的留出注入汇总。"""

    source_path: str
    grid_size: int
    cell_x: int
    cell_y: int
    cell_id: str
    cell_x0: int
    cell_y0: int
    cell_x1: int
    cell_y1: int
    proposal_mode: str
    psf_model: str
    psf_source_count: int
    psf_median_fwhm_px: float | None
    signal_normalization: str
    control_signal_adu: float
    mean_injected_peak_excess_adu: float
    trial_count: int
    injected_count: int
    candidate_recovered_count: int
    quality_recovered_count: int
    candidate_recall: float
    quality_recall: float
    mean_candidate_count: float
    mean_quality_count: float
    mean_filter_noise: float
    mean_background_candidate_count: float
    mean_background_quality_count: float
    paired_control_candidate_count: int
    paired_control_quality_count: int
    paired_control_filter_noise: float
    candidate_delta_vs_paired_control: float
    quality_delta_vs_paired_control: float
    filter_noise_delta_vs_paired_control: float
    background_candidate_delta_vs_paired_control: float
    background_quality_delta_vs_paired_control: float
    local_background_adu: float
    local_noise_adu: float
    special_pixel_fraction: float
    nearest_baseline_source_px: float | None
    site_condition_json: str
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "grid_size": self.grid_size,
            "cell_x": self.cell_x,
            "cell_y": self.cell_y,
            "cell_id": self.cell_id,
            "cell_x0": self.cell_x0,
            "cell_y0": self.cell_y0,
            "cell_x1": self.cell_x1,
            "cell_y1": self.cell_y1,
            "proposal_mode": self.proposal_mode,
            "psf_model": self.psf_model,
            "psf_source_count": self.psf_source_count,
            "psf_median_fwhm_px": self.psf_median_fwhm_px,
            "signal_normalization": self.signal_normalization,
            "control_signal_adu": self.control_signal_adu,
            "mean_injected_peak_excess_adu": self.mean_injected_peak_excess_adu,
            "trial_count": self.trial_count,
            "injected_count": self.injected_count,
            "candidate_recovered_count": self.candidate_recovered_count,
            "quality_recovered_count": self.quality_recovered_count,
            "candidate_recall": self.candidate_recall,
            "quality_recall": self.quality_recall,
            "mean_candidate_count": self.mean_candidate_count,
            "mean_quality_count": self.mean_quality_count,
            "mean_filter_noise": self.mean_filter_noise,
            "mean_background_candidate_count": self.mean_background_candidate_count,
            "mean_background_quality_count": self.mean_background_quality_count,
            "paired_control_candidate_count": self.paired_control_candidate_count,
            "paired_control_quality_count": self.paired_control_quality_count,
            "paired_control_filter_noise": self.paired_control_filter_noise,
            "candidate_delta_vs_paired_control": self.candidate_delta_vs_paired_control,
            "quality_delta_vs_paired_control": self.quality_delta_vs_paired_control,
            "filter_noise_delta_vs_paired_control": self.filter_noise_delta_vs_paired_control,
            "background_candidate_delta_vs_paired_control": self.background_candidate_delta_vs_paired_control,
            "background_quality_delta_vs_paired_control": self.background_quality_delta_vs_paired_control,
            "local_background_adu": self.local_background_adu,
            "local_noise_adu": self.local_noise_adu,
            "special_pixel_fraction": self.special_pixel_fraction,
            "nearest_baseline_source_px": self.nearest_baseline_source_px,
            "site_condition_json": self.site_condition_json,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class SpatialInjectionAuditResult:
    """空间分区留出注入审计的完整结果。"""

    source_path: str
    image_shape: tuple[int, int]
    grid_size: int
    peak_levels: tuple[float, ...]
    trials_per_level: int
    sources_per_cell: int
    baseline_candidate_count: int
    baseline_quality_count: int
    paired_control_candidate_count: int
    paired_control_quality_count: int
    paired_control_filter_noise: float
    psf_model: str
    psf_source_count: int
    psf_median_fwhm_px: float | None
    parameters: dict[str, object]
    rows: tuple[SpatialInjectionAuditRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "image_shape": list(self.image_shape),
            "grid_size": self.grid_size,
            "peak_levels": list(self.peak_levels),
            "trials_per_level": self.trials_per_level,
            "sources_per_cell": self.sources_per_cell,
            "baseline_candidate_count": self.baseline_candidate_count,
            "baseline_quality_count": self.baseline_quality_count,
            "paired_control_candidate_count": self.paired_control_candidate_count,
            "paired_control_quality_count": self.paired_control_quality_count,
            "paired_control_filter_noise": self.paired_control_filter_noise,
            "psf_model": self.psf_model,
            "psf_source_count": self.psf_source_count,
            "psf_median_fwhm_px": self.psf_median_fwhm_px,
            "parameters": self.parameters,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
        }


def _cell_bounds(shape: tuple[int, int], grid_size: int, cell_x: int, cell_y: int) -> tuple[int, int, int, int]:
    height, width = (int(shape[0]), int(shape[1]))
    x0 = int(math.floor(width * cell_x / grid_size))
    x1 = int(math.floor(width * (cell_x + 1) / grid_size))
    y0 = int(math.floor(height * cell_y / grid_size))
    y1 = int(math.floor(height * (cell_y + 1) / grid_size))
    return x0, y0, x1, y1


def _select_spatial_blank_sites(
    image: np.ndarray,
    existing_sources: Sequence[Detection],
    rng: np.random.Generator,
    count: int,
    *,
    cell_bounds: tuple[int, int, int, int],
    baseline_noise_adu: float,
    psf_fwhm: float,
    aperture_radius: int,
) -> tuple[_RealInjectionSite, ...]:
    """在一个 detector 单元内选择远离候选的相对空白位置。"""

    if count < 1:
        raise ValueError("count must be positive")
    values = np.asarray(image)
    x0, y0, x1, y1 = cell_bounds
    neighborhood_radius = max(8, int(math.ceil(2.0 * float(psf_fwhm))))
    source_exclusion_radius = max(12.0, 2.0 * float(psf_fwhm))
    minimum_distance = max(10.0, 2.5 * float(psf_fwhm))
    interior_margin = max(
        int(aperture_radius) + 4,
        int(math.ceil(4.0 * float(psf_fwhm) / 2.354820045)) + 2,
    )
    if x1 - x0 <= 2 * interior_margin or y1 - y0 <= 2 * interior_margin:
        raise ValueError("spatial cell is too small for the requested injection margins")

    special_mask = _real_injection_special_mask(values)
    baseline_points = np.asarray([(source.x, source.y) for source in existing_sources], dtype=np.float64)
    baseline_tree = cKDTree(baseline_points) if baseline_points.size else None
    pool: list[_RealInjectionSite] = []
    target_pool = max(64, count * 80)
    attempts_limit = max(20_000, target_pool * 160)
    for _attempt in range(attempts_limit):
        if len(pool) >= target_pool:
            break
        x = float(rng.uniform(x0 + interior_margin, x1 - interior_margin))
        y = float(rng.uniform(y0 + interior_margin, y1 - interior_margin))
        if baseline_tree is not None:
            distance = baseline_tree.query((x, y), distance_upper_bound=source_exclusion_radius)[0]
            if np.isfinite(distance):
                continue
        site = _spatial_site(
            values,
            x,
            y,
            baseline_tree=baseline_tree,
            special_mask=special_mask,
            neighborhood_radius=neighborhood_radius,
        )
        if site is None or site.special_pixel_fraction > 0.0:
            continue
        allowed_excess = max(3.0 * site.local_noise_adu, 3.0 * float(baseline_noise_adu))
        if site.upper_excess_adu > allowed_excess:
            continue
        pool.append(site)

    if len(pool) < count:
        raise RuntimeError(
            f"空间单元 ({x0},{y0}) 合规位置不足：仅找到 {len(pool)} 个，需要 {count} 个"
        )
    # 随机排列候选池，避免按随机采样顺序把某个角落系统性排在前面。
    order = rng.permutation(len(pool))
    ordered = [pool[int(index)] for index in order]
    selected = _greedy_real_injection_sites(ordered, count, minimum_distance_px=minimum_distance)
    if len(selected) != count:
        raise RuntimeError(
            f"空间单元 ({x0},{y0}) 无法得到 {count} 个互相分离的注入位置，仅找到 {len(selected)} 个"
        )
    return tuple(selected)


def _spatial_site(
    image: np.ndarray,
    x: float,
    y: float,
    *,
    baseline_tree: cKDTree | None,
    special_mask: np.ndarray,
    neighborhood_radius: int,
) -> _RealInjectionSite | None:
    stats = _local_patch_stats(image, x, y, radius=5)
    if stats is None:
        return None
    local_background, local_noise, upper_excess = stats
    center_x = int(round(x))
    center_y = int(round(y))
    y0 = max(0, center_y - neighborhood_radius)
    y1 = min(image.shape[0], center_y + neighborhood_radius + 1)
    x0 = max(0, center_x - neighborhood_radius)
    x1 = min(image.shape[1], center_x + neighborhood_radius + 1)
    local_special = special_mask[y0:y1, x0:x1]
    special_fraction = float(np.mean(local_special)) if local_special.size else 0.0
    if baseline_tree is None:
        neighbor_count = 0
        nearest_distance: float | None = None
    else:
        neighbor_count = len(baseline_tree.query_ball_point((x, y), r=float(neighborhood_radius)))
        nearest = baseline_tree.query((x, y), distance_upper_bound=float(neighborhood_radius))[0]
        nearest_distance = float(nearest) if np.isfinite(nearest) else None
    return _RealInjectionSite(
        x=float(x),
        y=float(y),
        local_background_adu=float(local_background),
        local_noise_adu=float(local_noise),
        upper_excess_adu=float(upper_excess),
        special_pixel_fraction=special_fraction,
        baseline_neighbor_count=int(neighbor_count),
        nearest_baseline_source_px=nearest_distance,
    )


def run_spatial_injection_audit(
    path: str | Path | FitsFrame,
    *,
    grid_size: int = 2,
    peak_levels: Sequence[float] = (56.0, 128.0),
    trials_per_level: int = 1,
    sources_per_cell: int = 1,
    psf_fwhm: float = 2.0,
    injected_psf_fwhm: float | None = None,
    proposal_mode: str = "hybrid",
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    max_sources: int | None = None,
    match_radius_px: float | None = None,
    seed: int = 19019,
    reject_linear_artifacts: bool = True,
    psf_model: str = "gaussian",
    signal_normalization: str = "integrated_excess",
    empirical_psf_radius: int = 7,
    empirical_psf_sources: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> SpatialInjectionAuditResult:
    """在每个 detector 空间单元执行真实背景留出注入-回收。

    ``peak_levels`` 在 ``peak_excess`` 模式表示峰值超额，在
    ``integrated_excess`` 模式表示离散注入小窗的总积分超额。候选和质量
    回收率均以已知注入位置为分母；它们不等于真实图像的 precision、误检
    率或物理恒星完备率。
    """

    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if trials_per_level < 1 or sources_per_cell < 1:
        raise ValueError("trials_per_level and sources_per_cell must be positive")
    if psf_fwhm <= 0 or threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1:
        raise ValueError("spatial injection detector parameters must be positive")
    if background_box_size < 16 or min_flux_snr <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("background_box_size/min_flux_snr/min_psf_support_pixels are invalid")
    if match_radius_px is not None and match_radius_px <= 0:
        raise ValueError("match_radius_px must be positive when provided")
    if psf_model not in {"gaussian", "empirical"}:
        raise ValueError("psf_model must be gaussian or empirical")
    if signal_normalization not in {"peak_excess", "integrated_excess"}:
        raise ValueError("signal_normalization must be peak_excess or integrated_excess")
    if empirical_psf_radius < 3 or empirical_psf_sources < 1:
        raise ValueError("empirical PSF radius must be at least 3 and source count must be positive")
    if not peak_levels or any(float(level) <= 0 for level in peak_levels):
        raise ValueError("peak_levels must contain positive values")

    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
    height, width = (int(image.shape[0]), int(image.shape[1]))
    if height < 2 * (aperture_radius + 8) or width < 2 * (aperture_radius + 8):
        raise ValueError("image is too small for spatial injection margins")

    detector_options = {
        "mask": auxiliary_mask(image.shape),
        "threshold_sigma": threshold_sigma,
        "min_distance": min_distance,
        "aperture_radius": aperture_radius,
        "max_sources": max_sources,
        "psf_fwhm": psf_fwhm,
        "background_box_size": background_box_size,
        "min_flux_snr": min_flux_snr,
        "min_psf_support_pixels": min_psf_support_pixels,
        "reject_linear_artifacts": reject_linear_artifacts,
        "proposal_mode": proposal_mode,
    }
    baseline = detect_sources(image, **detector_options)
    empirical_psf: EmpiricalPSF | None = None
    if psf_model == "empirical":
        empirical_psf = estimate_empirical_psf(
            image,
            baseline.sources,
            support_radius=empirical_psf_radius,
            max_sources=empirical_psf_sources,
        )
        if empirical_psf is None:
            raise RuntimeError("真实质量源不足，无法建立实测 PSF；请改用 gaussian 或增加模板源")

    injected_detector_options = dict(detector_options)
    saturation_level = float(baseline.parameters.get("saturation_level", -1.0))
    injected_detector_options["saturation_level"] = saturation_level if saturation_level > 0 else None
    negative_overflow_limit = baseline.parameters.get("negative_overflow_limit")
    injected_detector_options["negative_overflow_limit"] = (
        None if negative_overflow_limit is None else float(negative_overflow_limit)
    )
    injected_detector_options["mask_zero_pixels"] = bool(int(baseline.parameters.get("mask_zero_pixels", 0)))

    # 注入图像会被转换为 float32；若只拿原始 int16 基线比较总候选数，
    # 任何 dtype/值域口径差异都会被误读为空间效应。这里固定一个与注入
    # 图像完全同 dtype、同参数、无注入的配对控制。它不是 GUI 的官方基线，
    # 只用于估计“这次注入相对于同条件控制改变了多少候选”。
    paired_control = detect_sources(
        np.asarray(image, dtype=np.float32).copy(),
        **injected_detector_options,
    )

    resolved_peaks = tuple(float(level) for level in peak_levels)
    rng = np.random.default_rng(seed)
    cells = tuple(
        (cell_x, cell_y, _cell_bounds(image.shape, grid_size, cell_x, cell_y))
        for cell_y in range(grid_size)
        for cell_x in range(grid_size)
    )
    layouts: dict[tuple[int, int], tuple[tuple[_RealInjectionSite, ...], ...]] = {}
    baseline_noise = float(baseline.noise)
    for cell_x, cell_y, bounds in cells:
        layouts[(cell_x, cell_y)] = tuple(
            _select_spatial_blank_sites(
                image,
                baseline.sources,
                rng,
                sources_per_cell,
                cell_bounds=bounds,
                baseline_noise_adu=baseline_noise,
                psf_fwhm=psf_fwhm,
                aperture_radius=aperture_radius,
            )
            for _trial in range(trials_per_level)
        )

    recovery_radius = max(3.0, float(match_radius_px if match_radius_px is not None else psf_fwhm))
    protection_radius = max(8.0, 2.0 * float(psf_fwhm))
    total_work = len(cells) * len(resolved_peaks) * trials_per_level
    completed_work = 0
    if progress is not None:
        progress(0, total_work)
    rows: list[SpatialInjectionAuditRow] = []
    for cell_x, cell_y, bounds in cells:
        layouts_for_cell = layouts[(cell_x, cell_y)]
        site_condition_json = json.dumps(
            [
                {
                    "trial_index": trial_index,
                    "site_index": site_index,
                    "x": site.x,
                    "y": site.y,
                    "local_background_adu": site.local_background_adu,
                    "local_noise_adu": site.local_noise_adu,
                    "upper_excess_adu": site.upper_excess_adu,
                    "special_pixel_fraction": site.special_pixel_fraction,
                    "baseline_neighbor_count": site.baseline_neighbor_count,
                    "nearest_baseline_source_px": site.nearest_baseline_source_px,
                }
                for trial_index, sites in enumerate(layouts_for_cell, start=1)
                for site_index, site in enumerate(sites, start=1)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        for control_signal in resolved_peaks:
            injected_total = 0
            candidate_recovered = 0
            quality_recovered = 0
            candidate_counts: list[int] = []
            quality_counts: list[int] = []
            filter_noises: list[float] = []
            background_candidate_counts: list[int] = []
            background_quality_counts: list[int] = []
            actual_peaks: list[float] = []
            local_backgrounds: list[float] = []
            local_noises: list[float] = []
            special_fractions: list[float] = []
            nearest_distances: list[float] = []
            for sites in layouts_for_cell:
                test_image = np.asarray(image, dtype=np.float32).copy()
                positions = [(site.x, site.y) for site in sites]
                for site in sites:
                    actual_peaks.append(
                        _inject_source_signal(
                            test_image,
                            site.x,
                            site.y,
                            control_signal,
                            signal_normalization=signal_normalization,
                            psf_model=psf_model,
                            gaussian_fwhm=float(injected_psf_fwhm if injected_psf_fwhm is not None else psf_fwhm),
                            empirical_psf=empirical_psf,
                        )
                    )
                detection = detect_sources(test_image, **injected_detector_options)
                candidate_recovered += _matched_count(positions, detection.sources, recovery_radius)
                quality_recovered += _matched_count(positions, detection.quality_sources, recovery_radius)
                injected_total += len(positions)
                candidate_counts.append(int(detection.candidate_count))
                quality_counts.append(int(detection.star_count))
                filter_noise = detection.parameters.get("filter_noise")
                if filter_noise is None or not np.isfinite(float(filter_noise)):
                    raise RuntimeError("检测结果缺少有限的 filter_noise，无法进行空间注入配对控制")
                filter_noises.append(float(filter_noise))
                background_candidate_counts.append(
                    _count_outside_injection_regions(detection.sources, positions, protection_radius)
                )
                background_quality_counts.append(
                    _count_outside_injection_regions(detection.quality_sources, positions, protection_radius)
                )
                local_backgrounds.extend(site.local_background_adu for site in sites)
                local_noises.extend(site.local_noise_adu for site in sites)
                special_fractions.extend(site.special_pixel_fraction for site in sites)
                nearest_distances.extend(
                    site.nearest_baseline_source_px
                    for site in sites
                    if site.nearest_baseline_source_px is not None
                )
                completed_work += 1
                if progress is not None:
                    progress(completed_work, total_work)
            rows.append(
                SpatialInjectionAuditRow(
                    source_path=str(frame.path),
                    grid_size=grid_size,
                    cell_x=cell_x,
                    cell_y=cell_y,
                    cell_id=f"r{cell_y}c{cell_x}",
                    cell_x0=bounds[0],
                    cell_y0=bounds[1],
                    cell_x1=bounds[2],
                    cell_y1=bounds[3],
                    proposal_mode=str(proposal_mode),
                    psf_model=psf_model,
                    psf_source_count=0 if empirical_psf is None else empirical_psf.source_count,
                    psf_median_fwhm_px=None if empirical_psf is None else empirical_psf.median_fwhm_px,
                    signal_normalization=signal_normalization,
                    control_signal_adu=control_signal,
                    mean_injected_peak_excess_adu=float(np.mean(actual_peaks)),
                    trial_count=trials_per_level,
                    injected_count=injected_total,
                    candidate_recovered_count=candidate_recovered,
                    quality_recovered_count=quality_recovered,
                    candidate_recall=candidate_recovered / injected_total,
                    quality_recall=quality_recovered / injected_total,
                    mean_candidate_count=float(np.mean(candidate_counts)),
                    mean_quality_count=float(np.mean(quality_counts)),
                    mean_filter_noise=float(np.mean(filter_noises)),
                    mean_background_candidate_count=float(np.mean(background_candidate_counts)),
                    mean_background_quality_count=float(np.mean(background_quality_counts)),
                    paired_control_candidate_count=int(paired_control.candidate_count),
                    paired_control_quality_count=int(paired_control.star_count),
                    paired_control_filter_noise=float(paired_control.parameters["filter_noise"]),
                    candidate_delta_vs_paired_control=(
                        float(np.mean(candidate_counts)) - float(paired_control.candidate_count)
                    ),
                    quality_delta_vs_paired_control=(
                        float(np.mean(quality_counts)) - float(paired_control.star_count)
                    ),
                    filter_noise_delta_vs_paired_control=(
                        float(np.mean(filter_noises)) - float(paired_control.parameters["filter_noise"])
                    ),
                    background_candidate_delta_vs_paired_control=(
                        float(np.mean(background_candidate_counts)) - float(paired_control.candidate_count)
                    ),
                    background_quality_delta_vs_paired_control=(
                        float(np.mean(background_quality_counts)) - float(paired_control.star_count)
                    ),
                    local_background_adu=float(np.mean(local_backgrounds)),
                    local_noise_adu=float(np.mean(local_noises)),
                    special_pixel_fraction=float(np.mean(special_fractions)),
                    nearest_baseline_source_px=(
                        float(np.median(nearest_distances)) if nearest_distances else None
                    ),
                    site_condition_json=site_condition_json,
                    note=(
                        "每个空间单元在注入前远离当前候选和特殊值；候选/质量回收是已知注入真值召回，"
                        "不是当前真实 FITS 的 precision、误检率或物理恒星完备率。"
                    ),
                )
            )

    result = SpatialInjectionAuditResult(
        source_path=str(frame.path),
        image_shape=(height, width),
        grid_size=grid_size,
        peak_levels=resolved_peaks,
        trials_per_level=trials_per_level,
        sources_per_cell=sources_per_cell,
        baseline_candidate_count=int(baseline.candidate_count),
        baseline_quality_count=int(baseline.star_count),
        paired_control_candidate_count=int(paired_control.candidate_count),
        paired_control_quality_count=int(paired_control.star_count),
        paired_control_filter_noise=float(paired_control.parameters["filter_noise"]),
        psf_model=psf_model,
        psf_source_count=0 if empirical_psf is None else empirical_psf.source_count,
        psf_median_fwhm_px=None if empirical_psf is None else empirical_psf.median_fwhm_px,
        parameters={
            "grid_size": grid_size,
            "peak_levels": list(resolved_peaks),
            "trials_per_level": trials_per_level,
            "sources_per_cell": sources_per_cell,
            "psf_fwhm": float(psf_fwhm),
            "injected_psf_fwhm": float(injected_psf_fwhm if injected_psf_fwhm is not None else psf_fwhm),
            "proposal_mode": proposal_mode,
            "threshold_sigma": float(threshold_sigma),
            "min_distance": int(min_distance),
            "aperture_radius": int(aperture_radius),
            "background_box_size": int(background_box_size),
            "min_flux_snr": float(min_flux_snr),
            "min_psf_support_pixels": int(min_psf_support_pixels),
            "max_sources": max_sources,
            "match_radius_px": float(recovery_radius),
            "seed": int(seed),
            "reject_linear_artifacts": bool(reject_linear_artifacts),
            "signal_normalization": signal_normalization,
            "empirical_psf_radius": int(empirical_psf_radius),
            "empirical_psf_sources": int(empirical_psf_sources),
            "baseline_noise_adu": baseline_noise,
            "paired_control_dtype": "float32",
            "paired_control_candidate_count": int(paired_control.candidate_count),
            "paired_control_quality_count": int(paired_control.star_count),
            "paired_control_filter_noise": float(paired_control.parameters["filter_noise"]),
            "negative_overflow_limit": _default_negative_overflow_limit(image),
        },
        rows=tuple(rows),
        conclusion=(
            f"空间分区留出注入完成：{grid_size}×{grid_size} 单元、{len(resolved_peaks)} 个信号档位、"
            f"每档 {trials_per_level} 次；候选/质量回收只表示已知注入源召回。"
            "候选总数变化以同 dtype 配对控制为参照；各单元差异用于定位噪声、PSF 或边界校准方向，"
            "不直接解释为恒星密度或物理完备率。"
        ),
    )
    return result


def write_spatial_injection_artifacts(result: SpatialInjectionAuditResult, out_dir: str | Path) -> Path:
    """写空间分区注入 CSV/JSON 研究产物。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(result.rows[0].as_dict()) if result.rows else ["cell_id", "control_signal_adu"]
    with (output / "spatial_injection_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    (output / "spatial_injection_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output
