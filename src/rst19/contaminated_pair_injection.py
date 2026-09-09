"""真实污染结构中的已知双源注入审计。

这个研究工具把两个已知位置、已知强度的 Gaussian 双源叠加到真实图像中
已有的局部背景、紧凑源或线状结构上，然后把三种口径分开记录：

* baseline：注入前该位置已经有的候选/质量源；
* injected：注入后检测器是否命中两个真值位置；
* new：注入后相对 baseline 新增的命中。

因此它不能把原图中本来就存在的亮点冒充成注入回收，也不把“候选命中”
直接写成物理恒星真值。``pair_*_resolved`` 要求两个真值由两个不同的
候选分别匹配，``merged_*`` 记录一个候选是否同时落入两个匹配半径。

``analysis_scope='local_roi'`` 是为大图多锚点研究提供的加速口径：baseline
仍在全幅检测一次，注入后只在每个锚点的局部窗口运行同一检测器，结果中的
``injected_candidate_count``/``injected_quality_count`` 明确是 ROI 局部数量。
它不应与全幅候选总数混合，也不改变默认检测、GUI 或缓存。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .detection import Detection, detect_sources
from .experiments import _inject_source_signal, _match_target_flags, _merged_pair_fraction
from .fits import FitsFrame, auxiliary_mask, read_fits


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class ContaminatedPairAnchor:
    """真实污染位置的中心和双源方向。

    ``direction_x/y`` 只表示方向，不要求已经归一化；运行时会统一归一化。
    ``name`` 应该带有位置语义，例如 ``compact_quality_22381``，但不暗示
    该锚点本身就是物理真值。
    """

    name: str
    x: float
    y: float
    direction_x: float
    direction_y: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContaminatedPairInjectionRow:
    """一个污染位置、间距和强度比条件的回收结果。"""

    source_path: str
    anchor_name: str
    anchor_x: float
    anchor_y: float
    direction_x: float
    direction_y: float
    separation_px: float
    secondary_to_primary_ratio: float
    total_peak_excess_adu: float
    primary_signal_adu: float
    secondary_signal_adu: float
    primary_x: float
    primary_y: float
    secondary_x: float
    secondary_y: float
    baseline_candidate_primary: bool
    baseline_candidate_secondary: bool
    injected_candidate_primary: bool
    injected_candidate_secondary: bool
    new_candidate_primary: bool
    new_candidate_secondary: bool
    baseline_quality_primary: bool
    baseline_quality_secondary: bool
    injected_quality_primary: bool
    injected_quality_secondary: bool
    new_quality_primary: bool
    new_quality_secondary: bool
    candidate_pair_resolved: bool
    quality_pair_resolved: bool
    merged_candidate: bool
    merged_quality: bool
    baseline_candidate_count: int
    injected_candidate_count: int
    baseline_quality_count: int
    injected_quality_count: int
    new_candidate_hit_count: int
    new_quality_hit_count: int
    match_radius_px: float
    detector_psf_fwhm_px: float
    injected_psf_fwhm_px: float
    local_background_adu: float
    local_noise_adu: float
    note: str
    analysis_scope: str = "full_frame"
    roi_bounds: tuple[int, int, int, int] | None = None
    injected_primary_match_id: int | None = None
    injected_secondary_match_id: int | None = None
    injected_primary_match_distance_px: float | None = None
    injected_secondary_match_distance_px: float | None = None
    injected_primary_quality_reason: str = ""
    injected_secondary_quality_reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContaminatedPairInjectionResult:
    """污染结构双源注入审计的完整结果。"""

    source_path: str
    anchors: tuple[ContaminatedPairAnchor, ...]
    separations_px: tuple[float, ...]
    ratios: tuple[float, ...]
    baseline_candidate_count: int
    baseline_quality_count: int
    parameters: dict[str, object]
    rows: tuple[ContaminatedPairInjectionRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "anchors": [anchor.as_dict() for anchor in self.anchors],
            "separations_px": list(self.separations_px),
            "ratios": list(self.ratios),
            "baseline_candidate_count": self.baseline_candidate_count,
            "baseline_quality_count": self.baseline_quality_count,
            "parameters": dict(self.parameters),
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
            "interpretation_boundary": (
                "contaminated-background injection pilot; baseline overlap is not injection truth; "
                "candidate/quality recovery is not precision or physical star truth"
            ),
        }


def _finite_positive(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _normalised_anchor(anchor: ContaminatedPairAnchor) -> ContaminatedPairAnchor:
    values = (anchor.x, anchor.y, anchor.direction_x, anchor.direction_y)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"anchor {anchor.name!r} contains a non-finite value")
    if not str(anchor.name).strip():
        raise ValueError("anchor name must not be empty")
    norm = math.hypot(float(anchor.direction_x), float(anchor.direction_y))
    if norm <= 0:
        raise ValueError(f"anchor {anchor.name!r} direction must be non-zero")
    return ContaminatedPairAnchor(
        name=str(anchor.name).strip(),
        x=float(anchor.x),
        y=float(anchor.y),
        direction_x=float(anchor.direction_x) / norm,
        direction_y=float(anchor.direction_y) / norm,
    )


def _pair_positions(
    anchor: ContaminatedPairAnchor,
    separation_px: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    half = 0.5 * float(separation_px)
    dx = half * float(anchor.direction_x)
    dy = half * float(anchor.direction_y)
    return ((anchor.x + dx, anchor.y + dy), (anchor.x - dx, anchor.y - dy))


def _roi_bounds(
    image_shape: tuple[int, int],
    anchor: ContaminatedPairAnchor,
    half_size_px: int,
) -> tuple[int, int, int, int]:
    """返回包含锚点的局部窗口，顺序为 ``x0, y0, x1, y1``。"""

    height, width = image_shape
    center_x = int(round(anchor.x))
    center_y = int(round(anchor.y))
    x0 = max(0, center_x - int(half_size_px))
    y0 = max(0, center_y - int(half_size_px))
    x1 = min(width, center_x + int(half_size_px) + 1)
    y1 = min(height, center_y + int(half_size_px) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"anchor {anchor.name!r} has an empty local ROI")
    return x0, y0, x1, y1


def _offset_detection(source: Detection, *, x_offset: int, y_offset: int) -> Detection:
    """把局部 ROI 检测坐标恢复为全幅坐标。"""

    return replace(
        source,
        x=float(source.x) + float(x_offset),
        y=float(source.y) + float(y_offset),
        peak_x=None if source.peak_x is None else float(source.peak_x) + float(x_offset),
        peak_y=None if source.peak_y is None else float(source.peak_y) + float(y_offset),
    )


def _match_target_sources(
    injected: Sequence[tuple[float, float]],
    detections: Sequence[Detection],
    radius_px: float,
) -> tuple[Detection | None, ...]:
    """按 ``_match_target_flags`` 的贪心口径返回实际匹配源。

    研究表除了记录端点是否命中，还需要知道命中的候选为何没有通过质量层。
    这里严格复用原有的一对一、按目标顺序取最近源的匹配规则，不改变命中
    统计，只把被选中的 ``Detection`` 带回供审计。
    """

    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    remaining = list(detections)
    matched: list[Detection | None] = []
    for x, y in injected:
        best_index: int | None = None
        best_distance = float(radius_px)
        for index, detection in enumerate(remaining):
            distance = math.hypot(detection.x - x, detection.y - y)
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            matched.append(None)
        else:
            matched.append(remaining.pop(best_index))
    return tuple(matched)


def _match_distance(
    target: tuple[float, float],
    source: Detection | None,
) -> float | None:
    if source is None:
        return None
    return float(math.hypot(source.x - target[0], source.y - target[1]))


def _quality_reason(source: Detection | None) -> str:
    """将命中源压缩成可读、可聚合的质量层原因。"""

    if source is None:
        return "NO_CANDIDATE"
    if source.quality_passed:
        return "QUALITY_PASS"
    return "|".join(source.flags) if source.flags else "QUALITY_REJECTED_UNKNOWN"


def _local_stats(image: np.ndarray, anchor: ContaminatedPairAnchor, radius: int = 5) -> tuple[float, float]:
    """返回锚点周围的稳健局部背景和噪声，供结果解释而非注入归一化。"""

    x = int(round(anchor.x))
    y = int(round(anchor.y))
    y0 = max(0, y - radius)
    y1 = min(image.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(image.shape[1], x + radius + 1)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return float("nan"), float("nan")
    background = float(np.median(finite))
    mad = float(np.median(np.abs(finite - background)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(finite))
    if not math.isfinite(scale) or scale <= 0:
        scale = np.finfo(np.float64).eps
    # 负溢出/重复码可能远超普通背景 RMS；先做一个宽松的 MAD 截断，
    # 避免把“局部含异常值”错误记录成数千 ADU 的噪声尺度。
    clipped = finite[np.abs(finite - background) <= 5.0 * scale]
    if clipped.size >= 3:
        background = float(np.median(clipped))
        clipped_mad = float(np.median(np.abs(clipped - background)))
        clipped_scale = 1.4826 * clipped_mad
        noise = max(clipped_scale, float(np.std(clipped)), np.finfo(np.float64).eps)
    else:
        noise = scale
    return background, float(noise)


def _validate_sequence(values: Sequence[float], name: str, *, upper: float | None = None) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError(f"{name} cannot be empty")
    if any(not math.isfinite(value) or value <= 0 for value in resolved):
        raise ValueError(f"{name} must contain finite positive values")
    if upper is not None and any(value > upper for value in resolved):
        raise ValueError(f"{name} must be <= {upper}")
    return resolved


def run_contaminated_pair_injection_audit(
    path: str | Path | FitsFrame,
    anchors: Sequence[ContaminatedPairAnchor],
    *,
    separations_px: Sequence[float] = (2.738, 4.123),
    secondary_to_primary_ratios: Sequence[float] = (0.143, 1.0),
    total_peak_excess_adu: float = 4096.0,
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
    match_radius_px: float = 2.0,
    reject_linear_artifacts: bool = True,
    signal_normalization: str = "peak_excess",
    analysis_scope: str = "full_frame",
    roi_half_size_px: int = 192,
    progress: ProgressCallback | None = None,
) -> ContaminatedPairInjectionResult:
    """在指定的真实局部结构上注入双源并分离 baseline/new 命中。

    当前研究口径使用 Gaussian 注入和 ``peak_excess`` 总峰值固定；真实
    图像背景、原有候选和 detector 质量规则保持不变。``full_frame`` 会对
    每个条件运行全图检测；``local_roi`` 只把注入后的检测裁到每个锚点窗口，
    但会把检测坐标恢复到全幅后再匹配。该函数不修改输入 FITS，不写缓存，
    所有条件在内存副本上完成。
    """

    if not anchors:
        raise ValueError("anchors cannot be empty")
    resolved_anchors = tuple(_normalised_anchor(anchor) for anchor in anchors)
    names = [anchor.name for anchor in resolved_anchors]
    if len(set(names)) != len(names):
        raise ValueError("anchor names must be unique")
    resolved_separations = _validate_sequence(separations_px, "separations_px")
    resolved_ratios = _validate_sequence(secondary_to_primary_ratios, "secondary_to_primary_ratios", upper=1.0)
    total_peak = _finite_positive(total_peak_excess_adu, "total_peak_excess_adu")
    detector_fwhm = _finite_positive(psf_fwhm, "psf_fwhm")
    injected_fwhm = (
        detector_fwhm
        if injected_psf_fwhm is None
        else _finite_positive(injected_psf_fwhm, "injected_psf_fwhm")
    )
    if signal_normalization != "peak_excess":
        raise ValueError("contaminated pair audit currently supports signal_normalization='peak_excess' only")
    analysis_scope = str(analysis_scope).strip().lower()
    if analysis_scope not in {"full_frame", "local_roi"}:
        raise ValueError("analysis_scope must be 'full_frame' or 'local_roi'")
    if int(roi_half_size_px) < 1:
        raise ValueError("roi_half_size_px must be positive")
    roi_half_size = int(roi_half_size_px)
    if analysis_scope == "local_roi":
        minimum_roi_half_size = int(
            math.ceil(max(resolved_separations) / 2.0) + aperture_radius + 2
        )
        if roi_half_size < minimum_roi_half_size:
            raise ValueError(
                "roi_half_size_px is too small for the requested separation/aperture; "
                f"need at least {minimum_roi_half_size}"
            )
    if threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1 or background_box_size < 16:
        raise ValueError("detector geometry and threshold parameters are invalid")
    if min_flux_snr <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_flux_snr/min_psf_support_pixels are invalid")
    match_radius = _finite_positive(match_radius_px, "match_radius_px")
    if max_sources is not None and max_sources < 1:
        raise ValueError("max_sources must be positive when provided")

    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
    detector_options = {
        "mask": auxiliary_mask(image.shape),
        "threshold_sigma": float(threshold_sigma),
        "min_distance": int(min_distance),
        "aperture_radius": int(aperture_radius),
        "max_sources": max_sources,
        "psf_fwhm": detector_fwhm,
        "background_box_size": int(background_box_size),
        "min_flux_snr": float(min_flux_snr),
        "min_psf_support_pixels": int(min_psf_support_pixels),
        "reject_linear_artifacts": bool(reject_linear_artifacts),
        "proposal_mode": str(proposal_mode),
    }
    baseline = detect_sources(image, **detector_options)
    injected_detector_options = dict(detector_options)
    saturation_level = float(baseline.parameters.get("saturation_level", -1.0))
    injected_detector_options["saturation_level"] = saturation_level if saturation_level > 0 else None
    negative_overflow_limit = baseline.parameters.get("negative_overflow_limit")
    injected_detector_options["negative_overflow_limit"] = (
        None if negative_overflow_limit is None else float(negative_overflow_limit)
    )
    injected_detector_options["mask_zero_pixels"] = bool(int(baseline.parameters.get("mask_zero_pixels", 0)))

    total_work = len(resolved_anchors) * len(resolved_separations) * len(resolved_ratios)
    completed_work = 0
    if progress is not None:
        progress(0, total_work)
    rows: list[ContaminatedPairInjectionRow] = []
    for anchor in resolved_anchors:
        local_background, local_noise = _local_stats(image, anchor)
        roi = None
        local_injected_options = injected_detector_options
        if analysis_scope == "local_roi":
            roi = _roi_bounds(image.shape, anchor, roi_half_size)
            x0, y0, x1, y1 = roi
            local_injected_options = dict(injected_detector_options)
            local_injected_options["mask"] = np.asarray(detector_options["mask"])[y0:y1, x0:x1]
        for separation in resolved_separations:
            primary_position, secondary_position = _pair_positions(anchor, separation)
            pair = (primary_position, secondary_position)
            for ratio in resolved_ratios:
                primary_signal = total_peak / (1.0 + ratio)
                secondary_signal = total_peak - primary_signal
                baseline_candidate_flags = _match_target_flags(pair, baseline.sources, match_radius)
                baseline_quality_flags = _match_target_flags(pair, baseline.quality_sources, match_radius)
                if roi is None:
                    test_image = np.asarray(image, dtype=np.float32).copy()
                    injection_x_offset = 0
                    injection_y_offset = 0
                else:
                    x0, y0, x1, y1 = roi
                    test_image = np.asarray(image[y0:y1, x0:x1], dtype=np.float32).copy()
                    injection_x_offset = x0
                    injection_y_offset = y0
                _inject_source_signal(
                    test_image,
                    primary_position[0] - injection_x_offset,
                    primary_position[1] - injection_y_offset,
                    primary_signal,
                    signal_normalization=signal_normalization,
                    psf_model="gaussian",
                    gaussian_fwhm=injected_fwhm,
                    empirical_psf=None,
                )
                _inject_source_signal(
                    test_image,
                    secondary_position[0] - injection_x_offset,
                    secondary_position[1] - injection_y_offset,
                    secondary_signal,
                    signal_normalization=signal_normalization,
                    psf_model="gaussian",
                    gaussian_fwhm=injected_fwhm,
                    empirical_psf=None,
                )
                injected_local = detect_sources(test_image, **local_injected_options)
                injected_sources = tuple(
                    _offset_detection(
                        source,
                        x_offset=injection_x_offset,
                        y_offset=injection_y_offset,
                    )
                    for source in injected_local.sources
                )
                injected_quality_sources = tuple(
                    source for source in injected_sources if source.quality_passed
                )
                injected_candidate_flags = _match_target_flags(pair, injected_sources, match_radius)
                injected_quality_flags = _match_target_flags(pair, injected_quality_sources, match_radius)
                injected_match_sources = _match_target_sources(pair, injected_sources, match_radius)
                if tuple(source is not None for source in injected_match_sources) != injected_candidate_flags:
                    raise RuntimeError("injected candidate diagnostic matching diverged from hit matching")
                new_candidate_flags = tuple(
                    injected_flag and not baseline_flag
                    for injected_flag, baseline_flag in zip(injected_candidate_flags, baseline_candidate_flags)
                )
                new_quality_flags = tuple(
                    injected_flag and not baseline_flag
                    for injected_flag, baseline_flag in zip(injected_quality_flags, baseline_quality_flags)
                )
                rows.append(
                    ContaminatedPairInjectionRow(
                        source_path=str(frame.path),
                        anchor_name=anchor.name,
                        anchor_x=anchor.x,
                        anchor_y=anchor.y,
                        direction_x=anchor.direction_x,
                        direction_y=anchor.direction_y,
                        separation_px=separation,
                        secondary_to_primary_ratio=ratio,
                        total_peak_excess_adu=total_peak,
                        primary_signal_adu=primary_signal,
                        secondary_signal_adu=secondary_signal,
                        primary_x=primary_position[0],
                        primary_y=primary_position[1],
                        secondary_x=secondary_position[0],
                        secondary_y=secondary_position[1],
                        baseline_candidate_primary=baseline_candidate_flags[0],
                        baseline_candidate_secondary=baseline_candidate_flags[1],
                        injected_candidate_primary=injected_candidate_flags[0],
                        injected_candidate_secondary=injected_candidate_flags[1],
                        new_candidate_primary=new_candidate_flags[0],
                        new_candidate_secondary=new_candidate_flags[1],
                        baseline_quality_primary=baseline_quality_flags[0],
                        baseline_quality_secondary=baseline_quality_flags[1],
                        injected_quality_primary=injected_quality_flags[0],
                        injected_quality_secondary=injected_quality_flags[1],
                        new_quality_primary=new_quality_flags[0],
                        new_quality_secondary=new_quality_flags[1],
                        candidate_pair_resolved=all(injected_candidate_flags),
                        quality_pair_resolved=all(injected_quality_flags),
                        merged_candidate=_merged_pair_fraction((pair,), injected_sources, match_radius) > 0,
                        merged_quality=_merged_pair_fraction((pair,), injected_quality_sources, match_radius) > 0,
                        baseline_candidate_count=baseline.candidate_count,
                        injected_candidate_count=injected_local.candidate_count,
                        baseline_quality_count=baseline.star_count,
                        injected_quality_count=injected_local.star_count,
                        new_candidate_hit_count=sum(new_candidate_flags),
                        new_quality_hit_count=sum(new_quality_flags),
                        match_radius_px=match_radius,
                        detector_psf_fwhm_px=detector_fwhm,
                        injected_psf_fwhm_px=injected_fwhm,
                        local_background_adu=local_background,
                        local_noise_adu=local_noise,
                        note=(
                            "baseline flags describe pre-existing detections at the injected truth positions; "
                            "new flags are injected-minus-baseline diagnostics, not precision or physical truth; "
                            f"analysis_scope={analysis_scope}; injected counts are "
                            f"{'ROI-local' if analysis_scope == 'local_roi' else 'full-frame'}"
                        ),
                        analysis_scope=analysis_scope,
                        roi_bounds=roi,
                        injected_primary_match_id=(
                            None
                            if injected_match_sources[0] is None
                            else int(injected_match_sources[0].detection_id)
                        ),
                        injected_secondary_match_id=(
                            None
                            if injected_match_sources[1] is None
                            else int(injected_match_sources[1].detection_id)
                        ),
                        injected_primary_match_distance_px=_match_distance(
                            primary_position, injected_match_sources[0]
                        ),
                        injected_secondary_match_distance_px=_match_distance(
                            secondary_position, injected_match_sources[1]
                        ),
                        injected_primary_quality_reason=_quality_reason(injected_match_sources[0]),
                        injected_secondary_quality_reason=_quality_reason(injected_match_sources[1]),
                    )
                )
                completed_work += 1
                if progress is not None:
                    progress(completed_work, total_work)

    resolved_rows = tuple(rows)
    resolved_parameters = {
        "proposal_mode": str(proposal_mode),
        "threshold_sigma": float(threshold_sigma),
        "min_distance": int(min_distance),
        "aperture_radius": int(aperture_radius),
        "background_box_size": int(background_box_size),
        "min_flux_snr": float(min_flux_snr),
        "min_psf_support_pixels": int(min_psf_support_pixels),
        "max_sources": max_sources,
        "reject_linear_artifacts": bool(reject_linear_artifacts),
        "signal_normalization": signal_normalization,
        "analysis_scope": analysis_scope,
        "roi_half_size_px": roi_half_size if analysis_scope == "local_roi" else None,
        "roi_bounds_order": "x0,y0,x1,y1" if analysis_scope == "local_roi" else None,
        "baseline_scope": "full_frame",
        "injected_count_scope": "local_roi" if analysis_scope == "local_roi" else "full_frame",
        "detector_psf_fwhm_px": detector_fwhm,
        "injected_psf_fwhm_px": injected_fwhm,
        "match_radius_px": match_radius,
    }
    resolved_count = sum(row.quality_pair_resolved for row in resolved_rows)
    candidate_count = sum(row.candidate_pair_resolved for row in resolved_rows)
    merged_count = sum(row.merged_candidate for row in resolved_rows)
    conclusion = (
        f"污染结构双源注入 pilot：{len(resolved_rows)} 个条件中，候选双源分别命中 "
        f"{candidate_count}/{len(resolved_rows)}，质量层分别通过 {resolved_count}/{len(resolved_rows)}，"
        f"候选合并 {merged_count}/{len(resolved_rows)}；baseline 命中必须与新增命中分开解释。"
    )
    return ContaminatedPairInjectionResult(
        source_path=str(frame.path),
        anchors=resolved_anchors,
        separations_px=resolved_separations,
        ratios=resolved_ratios,
        baseline_candidate_count=baseline.candidate_count,
        baseline_quality_count=baseline.star_count,
        parameters=resolved_parameters,
        rows=resolved_rows,
        conclusion=conclusion,
    )


def write_contaminated_pair_injection_artifacts(
    result: ContaminatedPairInjectionResult,
    output_dir: str | Path,
) -> Path:
    """写出可复核的 CSV/JSON 产物。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "contaminated_pair_injection.csv"
    json_path = output / "contaminated_pair_injection.json"
    fieldnames = list(ContaminatedPairInjectionRow.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row.as_dict())
    json_path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output
