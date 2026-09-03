"""可复现的星点检测实验：注入-回收和真实图像参数扫描。"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.optimize import least_squares, nnls
from scipy.spatial import cKDTree

from .detection import (
    Detection,
    DetectionResult,
    _default_negative_overflow_limit,
    _pair_psf_evidence,
    detect_sources,
)
from .fits import FitsFrame, auxiliary_mask, read_fits
from .pipeline import analyze_frame
from .sequence import _estimate_translation, detect_long_trails


@dataclass(frozen=True, slots=True)
class InjectionRecoveryRow:
    """一个峰值强度档位的注入-回收统计。"""

    proposal_mode: str
    detector_psf_fwhm_px: float
    injected_psf_fwhm_px: float
    peak_excess_adu: float
    injected_count: int
    candidate_recovered_count: int
    quality_recovered_count: int
    candidate_recall: float
    quality_recall: float
    mean_candidate_count: float
    mean_quality_count: float

    def as_dict(self) -> dict[str, object]:
        return {
            "proposal_mode": self.proposal_mode,
            "detector_psf_fwhm_px": self.detector_psf_fwhm_px,
            "injected_psf_fwhm_px": self.injected_psf_fwhm_px,
            "peak_excess_adu": self.peak_excess_adu,
            "injected_count": self.injected_count,
            "candidate_recovered_count": self.candidate_recovered_count,
            "quality_recovered_count": self.quality_recovered_count,
            "candidate_recall": self.candidate_recall,
            "quality_recall": self.quality_recall,
            "mean_candidate_count": self.mean_candidate_count,
            "mean_quality_count": self.mean_quality_count,
        }


@dataclass(frozen=True, slots=True)
class FeatureAuditRow:
    """一个特征场景的正样本回收或阴性伪影泄漏审计。"""

    feature_class: str
    scenario: str
    control_type: str
    truth_count: int
    candidate_true_hits: int
    quality_true_hits: int
    candidate_recall: float | None
    quality_recall: float | None
    nearby_candidate_count: int
    nearby_quality_count: int
    candidate_count: int
    quality_count: int
    nearby_flags: str
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_class": self.feature_class,
            "scenario": self.scenario,
            "control_type": self.control_type,
            "truth_count": self.truth_count,
            "candidate_true_hits": self.candidate_true_hits,
            "quality_true_hits": self.quality_true_hits,
            "candidate_recall": self.candidate_recall,
            "quality_recall": self.quality_recall,
            "nearby_candidate_count": self.nearby_candidate_count,
            "nearby_quality_count": self.nearby_quality_count,
            "candidate_count": self.candidate_count,
            "quality_count": self.quality_count,
            "nearby_flags": self.nearby_flags,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class SequenceFeatureFrameRow:
    """15 帧特征类别的逐帧计数。"""

    frame_index: int
    path: str
    candidate_count: int
    returned_count: int
    quality_count: int
    background_adu: float
    noise_adu: float
    feature_class: str
    feature_class_label: str
    feature_candidate_count: int
    feature_quality_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "quality_count": self.quality_count,
            "background_adu": self.background_adu,
            "noise_adu": self.noise_adu,
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "feature_candidate_count": self.feature_candidate_count,
            "feature_quality_count": self.feature_quality_count,
        }


@dataclass(frozen=True, slots=True)
class SequenceFeatureTemporalProfile:
    """15 帧特征类别计数、占比和质量通过率的时间剖面。"""

    feature_class: str
    feature_class_label: str
    frame_count: int
    active_frame_count: int
    quality_active_frame_count: int
    candidate_count_mean: float
    candidate_count_min: int
    candidate_count_max: int
    candidate_count_cv: float | None
    candidate_fraction_mean: float
    candidate_fraction_cv: float | None
    quality_count_mean: float
    quality_count_min: int
    quality_count_max: int
    quality_count_cv: float | None
    quality_fraction_mean: float
    quality_fraction_weighted: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "frame_count": self.frame_count,
            "active_frame_count": self.active_frame_count,
            "quality_active_frame_count": self.quality_active_frame_count,
            "candidate_count_mean": self.candidate_count_mean,
            "candidate_count_min": self.candidate_count_min,
            "candidate_count_max": self.candidate_count_max,
            "candidate_count_cv": self.candidate_count_cv,
            "candidate_fraction_mean": self.candidate_fraction_mean,
            "candidate_fraction_cv": self.candidate_fraction_cv,
            "quality_count_mean": self.quality_count_mean,
            "quality_count_min": self.quality_count_min,
            "quality_count_max": self.quality_count_max,
            "quality_count_cv": self.quality_count_cv,
            "quality_fraction_mean": self.quality_fraction_mean,
            "quality_fraction_weighted": self.quality_fraction_weighted,
        }


@dataclass(frozen=True, slots=True)
class SequenceFeaturePersistenceRow:
    """首帧特征类别在后续帧的候选/质量邻域持久性。"""

    feature_class: str
    feature_class_label: str
    anchor_candidate_count: int
    anchor_quality_count: int
    frame_count: int
    required_presence: int
    association_radius_px: float
    candidate_median_presence: float | None
    candidate_mean_presence: float | None
    candidate_presence_ge_required_count: int
    candidate_presence_all_frames_count: int
    quality_median_presence: float | None
    quality_mean_presence: float | None
    quality_presence_ge_required_count: int
    quality_presence_all_frames_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "anchor_candidate_count": self.anchor_candidate_count,
            "anchor_quality_count": self.anchor_quality_count,
            "frame_count": self.frame_count,
            "required_presence": self.required_presence,
            "association_radius_px": self.association_radius_px,
            "candidate_median_presence": self.candidate_median_presence,
            "candidate_mean_presence": self.candidate_mean_presence,
            "candidate_presence_ge_required_count": self.candidate_presence_ge_required_count,
            "candidate_presence_all_frames_count": self.candidate_presence_all_frames_count,
            "quality_median_presence": self.quality_median_presence,
            "quality_mean_presence": self.quality_mean_presence,
            "quality_presence_ge_required_count": self.quality_presence_ge_required_count,
            "quality_presence_all_frames_count": self.quality_presence_all_frames_count,
        }


@dataclass(frozen=True, slots=True)
class SequenceFeatureClassTransitionRow:
    """首帧特征类别到后续帧实际响应类别的逐层转移统计。"""

    layer: str
    anchor_feature_class: str
    anchor_feature_class_label: str
    response_feature_class: str
    response_feature_class_label: str
    anchor_count: int
    frame_count: int
    possible_match_count: int
    matched_count: int
    match_rate: float | None
    mean_matches_per_anchor: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "anchor_feature_class": self.anchor_feature_class,
            "anchor_feature_class_label": self.anchor_feature_class_label,
            "response_feature_class": self.response_feature_class,
            "response_feature_class_label": self.response_feature_class_label,
            "anchor_count": self.anchor_count,
            "frame_count": self.frame_count,
            "possible_match_count": self.possible_match_count,
            "matched_count": self.matched_count,
            "match_rate": self.match_rate,
            "mean_matches_per_anchor": self.mean_matches_per_anchor,
        }


@dataclass(frozen=True, slots=True)
class SequenceFeatureAuditResult:
    """真实 FITS 的跨帧特征类别审计结果。"""

    frame_count: int
    association_radius_px: float
    required_presence: int
    association_method: str
    candidate_source_mode: str
    quality_source_mode: str
    cumulative_shifts: tuple[tuple[float, float], ...]
    frame_rows: tuple[SequenceFeatureFrameRow, ...]
    persistence_rows: tuple[SequenceFeaturePersistenceRow, ...]
    class_transition_rows: tuple[SequenceFeatureClassTransitionRow, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "association_radius_px": self.association_radius_px,
            "required_presence": self.required_presence,
            "association_method": self.association_method,
            "candidate_source_mode": self.candidate_source_mode,
            "quality_source_mode": self.quality_source_mode,
            "cumulative_shifts": [list(shift) for shift in self.cumulative_shifts],
            "frame_rows": [row.as_dict() for row in self.frame_rows],
            "persistence_rows": [row.as_dict() for row in self.persistence_rows],
            "class_transition_rows": [row.as_dict() for row in self.class_transition_rows],
        }


@dataclass(frozen=True, slots=True)
class FeatureCrossAuditRow:
    """一个特征类别在测光、候选持久性和质量持久性之间的交叉审计。"""

    feature_class: str
    feature_class_label: str
    anchor_candidate_count: int
    anchor_quality_count: int
    median_flux_snr: float | None
    max_flux_snr: float | None
    required_presence: int
    frame_count: int
    candidate_presence_ge_required_count: int
    candidate_presence_fraction: float | None
    quality_presence_ge_required_count: int
    quality_presence_fraction: float | None
    persistence_gap: float | None
    high_snr_rejected: bool
    interpretation: str
    candidate_any_response_fraction: float | None = None
    candidate_same_class_response_fraction: float | None = None
    quality_any_response_fraction: float | None = None
    quality_same_class_response_fraction: float | None = None
    dominant_cross_class: str | None = None
    dominant_cross_class_label: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "anchor_candidate_count": self.anchor_candidate_count,
            "anchor_quality_count": self.anchor_quality_count,
            "median_flux_snr": self.median_flux_snr,
            "max_flux_snr": self.max_flux_snr,
            "required_presence": self.required_presence,
            "frame_count": self.frame_count,
            "candidate_presence_ge_required_count": self.candidate_presence_ge_required_count,
            "candidate_presence_fraction": self.candidate_presence_fraction,
            "quality_presence_ge_required_count": self.quality_presence_ge_required_count,
            "quality_presence_fraction": self.quality_presence_fraction,
            "persistence_gap": self.persistence_gap,
            "high_snr_rejected": self.high_snr_rejected,
            "interpretation": self.interpretation,
            "candidate_any_response_fraction": self.candidate_any_response_fraction,
            "candidate_same_class_response_fraction": self.candidate_same_class_response_fraction,
            "quality_any_response_fraction": self.quality_any_response_fraction,
            "quality_same_class_response_fraction": self.quality_same_class_response_fraction,
            "dominant_cross_class": self.dominant_cross_class,
            "dominant_cross_class_label": self.dominant_cross_class_label,
        }


@dataclass(frozen=True, slots=True)
class FeatureCrossAuditResult:
    """由单帧特征汇总和 15 帧持久性汇总生成的交叉反例报告。"""

    source_summary_path: str
    persistence_path: str
    frame_count: int
    required_presence: int
    high_snr_threshold: float
    rows: tuple[FeatureCrossAuditRow, ...]
    conclusion: str
    class_transition_path: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "source_summary_path": self.source_summary_path,
            "persistence_path": self.persistence_path,
            "frame_count": self.frame_count,
            "required_presence": self.required_presence,
            "high_snr_threshold": self.high_snr_threshold,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
            "class_transition_path": self.class_transition_path,
        }


@dataclass(frozen=True, slots=True)
class SourcePairAuditRow:
    """一个固定或注册 detector 目标近邻双源的逐帧原始证据。"""

    frame_index: int
    path: str
    frame_shift_x_px: float
    frame_shift_y_px: float
    target_peak_distance_px: float
    detected_peak_distance_px: float | None
    detected_centroid_distance_px: float | None
    nearest_primary_distance_px: float | None
    nearest_secondary_distance_px: float | None
    nearest_source_peak_distance_px: float | None
    nearest_source_same_for_targets: bool | None
    nearest_secondary_detection_id: int | None
    nearest_secondary_peak_x: float | None
    nearest_secondary_peak_y: float | None
    nearest_secondary_peak_adu: float | None
    nearest_secondary_quality_passed: bool | None
    nearest_secondary_flags: str
    primary_detection_id: int | None
    secondary_detection_id: int | None
    primary_peak_adu: float | None
    secondary_peak_adu: float | None
    primary_flux_snr: float | None
    secondary_flux_snr: float | None
    primary_quality_passed: bool | None
    secondary_quality_passed: bool | None
    primary_feature_class: str | None
    secondary_feature_class: str | None
    primary_flags: str
    secondary_flags: str
    secondary_aperture_min_adu: float | None
    secondary_aperture_max_adu: float | None
    secondary_aperture_median_adu: float | None
    secondary_aperture_unique_values: int
    secondary_mode_adu: float | None
    secondary_mode_count: int
    secondary_mode_fraction: float | None
    secondary_fixed_minus_one_count: int
    secondary_negative_overflow_count: int
    secondary_positive_extreme_count: int
    pair_window_negative_overflow_count: int
    pair_window_positive_extreme_count: int
    pair_delta_bic_raw: float | None
    pair_component_snr_raw: float | None
    pair_delta_bic_range_masked: float | None
    pair_component_snr_range_masked: float | None
    pair_delta_bic_sentinel_masked: float | None
    pair_component_snr_sentinel_masked: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "frame_shift_x_px": self.frame_shift_x_px,
            "frame_shift_y_px": self.frame_shift_y_px,
            "target_peak_distance_px": self.target_peak_distance_px,
            "detected_peak_distance_px": self.detected_peak_distance_px,
            "detected_centroid_distance_px": self.detected_centroid_distance_px,
            "nearest_primary_distance_px": self.nearest_primary_distance_px,
            "nearest_secondary_distance_px": self.nearest_secondary_distance_px,
            "nearest_source_peak_distance_px": self.nearest_source_peak_distance_px,
            "nearest_source_same_for_targets": self.nearest_source_same_for_targets,
            "nearest_secondary_detection_id": self.nearest_secondary_detection_id,
            "nearest_secondary_peak_x": self.nearest_secondary_peak_x,
            "nearest_secondary_peak_y": self.nearest_secondary_peak_y,
            "nearest_secondary_peak_adu": self.nearest_secondary_peak_adu,
            "nearest_secondary_quality_passed": self.nearest_secondary_quality_passed,
            "nearest_secondary_flags": self.nearest_secondary_flags,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_peak_adu": self.primary_peak_adu,
            "secondary_peak_adu": self.secondary_peak_adu,
            "primary_flux_snr": self.primary_flux_snr,
            "secondary_flux_snr": self.secondary_flux_snr,
            "primary_quality_passed": self.primary_quality_passed,
            "secondary_quality_passed": self.secondary_quality_passed,
            "primary_feature_class": self.primary_feature_class,
            "secondary_feature_class": self.secondary_feature_class,
            "primary_flags": self.primary_flags,
            "secondary_flags": self.secondary_flags,
            "secondary_aperture_min_adu": self.secondary_aperture_min_adu,
            "secondary_aperture_max_adu": self.secondary_aperture_max_adu,
            "secondary_aperture_median_adu": self.secondary_aperture_median_adu,
            "secondary_aperture_unique_values": self.secondary_aperture_unique_values,
            "secondary_mode_adu": self.secondary_mode_adu,
            "secondary_mode_count": self.secondary_mode_count,
            "secondary_mode_fraction": self.secondary_mode_fraction,
            "secondary_fixed_minus_one_count": self.secondary_fixed_minus_one_count,
            "secondary_negative_overflow_count": self.secondary_negative_overflow_count,
            "secondary_positive_extreme_count": self.secondary_positive_extreme_count,
            "pair_window_negative_overflow_count": self.pair_window_negative_overflow_count,
            "pair_window_positive_extreme_count": self.pair_window_positive_extreme_count,
            "pair_delta_bic_raw": self.pair_delta_bic_raw,
            "pair_component_snr_raw": self.pair_component_snr_raw,
            "pair_delta_bic_range_masked": self.pair_delta_bic_range_masked,
            "pair_component_snr_range_masked": self.pair_component_snr_range_masked,
            "pair_delta_bic_sentinel_masked": self.pair_delta_bic_sentinel_masked,
            "pair_component_snr_sentinel_masked": self.pair_component_snr_sentinel_masked,
        }


@dataclass(frozen=True, slots=True)
class SourcePairAuditResult:
    """固定或按平移注册 detector 目标的跨帧证据汇总。"""

    frame_count: int
    primary_target_xy: tuple[float, float]
    secondary_target_xy: tuple[float, float]
    target_peak_distance_px: float
    match_radius_px: float
    nearest_search_radius_px: float
    frame_shifts: tuple[tuple[float, float], ...]
    aperture_radius_px: int
    association_coordinate_system: str
    parameters: dict[str, object]
    rows: tuple[SourcePairAuditRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "primary_target_xy": list(self.primary_target_xy),
            "secondary_target_xy": list(self.secondary_target_xy),
            "target_peak_distance_px": self.target_peak_distance_px,
            "match_radius_px": self.match_radius_px,
            "nearest_search_radius_px": self.nearest_search_radius_px,
            "frame_shifts": [list(shift) for shift in self.frame_shifts],
            "aperture_radius_px": self.aperture_radius_px,
            "association_coordinate_system": self.association_coordinate_system,
            "parameters": self.parameters,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True, slots=True)
class LocalMultiPSFAuditRow:
    """固定候选位置的 K 源局部 PSF 模型比较结果。

    该表是研究审计，不是默认检测器的去混叠结果。位置来自宽筛候选或
    人工指定的 detector 坐标；每个模型共享同一局部背景平面和同一像素
    窗口，只改变固定位置的 PSF 分量数。这样可以把“多了一个局部峰”
    与“数据确实需要第二/第三个 PSF”分开记录。
    """

    frame_index: int
    path: str
    mask_mode: str
    component_count: int
    target_positions: tuple[tuple[float, float], ...]
    sample_count: int
    masked_pixel_count: int
    rss: float | None
    bic: float | None
    delta_bic_from_single: float | None
    delta_bic_from_previous: float | None
    amplitudes: tuple[float, ...]
    component_snr: tuple[float, ...]
    active_component_count: int
    max_template_correlation: float | None
    design_condition_number: float | None
    confirmation_line: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "mask_mode": self.mask_mode,
            "component_count": self.component_count,
            "target_positions": [list(point) for point in self.target_positions],
            "sample_count": self.sample_count,
            "masked_pixel_count": self.masked_pixel_count,
            "rss": self.rss,
            "bic": self.bic,
            "delta_bic_from_single": self.delta_bic_from_single,
            "delta_bic_from_previous": self.delta_bic_from_previous,
            "amplitudes": list(self.amplitudes),
            "component_snr": list(self.component_snr),
            "active_component_count": self.active_component_count,
            "max_template_correlation": self.max_template_correlation,
            "design_condition_number": self.design_condition_number,
            "confirmation_line": self.confirmation_line,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LocalMultiPSFAuditResult:
    """一组局部 K 源 PSF 对照的可复核汇总。"""

    frame_count: int
    target_positions: tuple[tuple[float, float], ...]
    frame_shifts: tuple[tuple[float, float], ...]
    mask_modes: tuple[str, ...]
    psf_fwhm_px: float
    parameters: dict[str, object]
    rows: tuple[LocalMultiPSFAuditRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "target_positions": [list(point) for point in self.target_positions],
            "frame_shifts": [list(shift) for shift in self.frame_shifts],
            "mask_modes": list(self.mask_modes),
            "psf_fwhm_px": self.psf_fwhm_px,
            "parameters": self.parameters,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True, slots=True)
class LocalFreeMultiPSFAuditRow:
    """允许候选中心小范围移动的 K 源局部 PSF 审计结果。"""

    frame_index: int
    path: str
    mask_mode: str
    component_count: int
    initial_positions: tuple[tuple[float, float], ...]
    fitted_positions: tuple[tuple[float, float], ...]
    position_offsets: tuple[tuple[float, float], ...]
    positions_at_bound: tuple[bool, ...]
    sample_count: int
    masked_pixel_count: int
    rss: float | None
    bic: float | None
    delta_bic_from_single: float | None
    delta_bic_from_previous: float | None
    amplitudes: tuple[float, ...]
    component_snr: tuple[float, ...]
    active_component_count: int
    max_template_correlation: float | None
    design_condition_number: float | None
    optimizer_success: bool
    optimizer_status: int | None
    optimizer_nfev: int | None
    confirmation_line: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "mask_mode": self.mask_mode,
            "component_count": self.component_count,
            "initial_positions": [list(point) for point in self.initial_positions],
            "fitted_positions": [list(point) for point in self.fitted_positions],
            "position_offsets": [list(point) for point in self.position_offsets],
            "positions_at_bound": list(self.positions_at_bound),
            "sample_count": self.sample_count,
            "masked_pixel_count": self.masked_pixel_count,
            "rss": self.rss,
            "bic": self.bic,
            "delta_bic_from_single": self.delta_bic_from_single,
            "delta_bic_from_previous": self.delta_bic_from_previous,
            "amplitudes": list(self.amplitudes),
            "component_snr": list(self.component_snr),
            "active_component_count": self.active_component_count,
            "max_template_correlation": self.max_template_correlation,
            "design_condition_number": self.design_condition_number,
            "optimizer_success": self.optimizer_success,
            "optimizer_status": self.optimizer_status,
            "optimizer_nfev": self.optimizer_nfev,
            "confirmation_line": self.confirmation_line,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LocalFreeMultiPSFAuditResult:
    """一组允许小范围中心优化的 K 源 PSF 对照汇总。"""

    frame_count: int
    target_positions: tuple[tuple[float, float], ...]
    frame_shifts: tuple[tuple[float, float], ...]
    mask_modes: tuple[str, ...]
    psf_fwhm_px: float
    position_radius_px: float
    parameters: dict[str, object]
    rows: tuple[LocalFreeMultiPSFAuditRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "target_positions": [list(point) for point in self.target_positions],
            "frame_shifts": [list(shift) for shift in self.frame_shifts],
            "mask_modes": list(self.mask_modes),
            "psf_fwhm_px": self.psf_fwhm_px,
            "position_radius_px": self.position_radius_px,
            "parameters": self.parameters,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True, slots=True)
class LocalFreeMultiPSFRadiusSweepRow:
    """不同位置搜索半径下的自由位置 PSF 敏感性摘要行。"""

    position_radius_px: float
    frame_index: int
    path: str
    mask_mode: str
    component_count: int
    delta_bic_from_single: float | None
    component_snr: tuple[float, ...]
    fitted_positions: tuple[tuple[float, float], ...]
    positions_at_bound: tuple[bool, ...]
    optimizer_success: bool
    confirmation_line: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "position_radius_px": self.position_radius_px,
            "frame_index": self.frame_index,
            "path": self.path,
            "mask_mode": self.mask_mode,
            "component_count": self.component_count,
            "delta_bic_from_single": self.delta_bic_from_single,
            "component_snr": list(self.component_snr),
            "fitted_positions": [list(point) for point in self.fitted_positions],
            "positions_at_bound": list(self.positions_at_bound),
            "optimizer_success": self.optimizer_success,
            "confirmation_line": self.confirmation_line,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LocalFreeMultiPSFRadiusSweepResult:
    """多组位置搜索半径的自由位置 PSF 对照汇总。"""

    frame_count: int
    target_positions: tuple[tuple[float, float], ...]
    position_radii_px: tuple[float, ...]
    mask_modes: tuple[str, ...]
    psf_fwhm_px: float
    parameters: dict[str, object]
    rows: tuple[LocalFreeMultiPSFRadiusSweepRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "target_positions": [list(point) for point in self.target_positions],
            "position_radii_px": list(self.position_radii_px),
            "mask_modes": list(self.mask_modes),
            "psf_fwhm_px": self.psf_fwhm_px,
            "parameters": self.parameters,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True, slots=True)
class TemporalCodeAuditResult:
    """15 帧 detector 坐标中的固定值/低变化值审计结果。"""

    frame_count: int
    image_shape: tuple[int, int]
    paths: tuple[str, ...]
    exact_stable_pixel_count: int
    exact_stable_value_counts: tuple[tuple[int, int], ...]
    low_variation_span_adu: int
    low_variation_pixel_count: int
    code_range_adu: tuple[int, int]
    code_pixel_count: int
    code_component_count: int
    code_components_ge_2: int
    code_largest_component: int
    code_focus_xy: tuple[int, int] | None
    code_focus_component_size: int | None
    code_focus_component_coordinates: tuple[tuple[int, int], ...]
    sentinel_value: int
    sentinel_exact_stable_pixel_count: int
    sentinel_component_count: int
    sentinel_largest_component: int
    sentinel_focus_xy: tuple[int, int] | None
    sentinel_focus_component_size: int | None
    sentinel_focus_component_coordinates: tuple[tuple[int, int], ...]
    components: tuple[dict[str, object], ...]
    interpretation_boundary: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "image_shape": list(self.image_shape),
            "paths": list(self.paths),
            "exact_stable_pixel_count": self.exact_stable_pixel_count,
            "exact_stable_value_counts": {
                str(value): count for value, count in self.exact_stable_value_counts
            },
            "low_variation_span_adu": self.low_variation_span_adu,
            "low_variation_pixel_count": self.low_variation_pixel_count,
            "focus_code_range_adu": list(self.code_range_adu),
            "focus_code_pixel_count": self.code_pixel_count,
            "focus_code_component_count": self.code_component_count,
            "focus_code_components_ge_2": self.code_components_ge_2,
            "focus_code_largest_component": self.code_largest_component,
            "focus_code_xy": None if self.code_focus_xy is None else list(self.code_focus_xy),
            "focus_code_component_size": self.code_focus_component_size,
            "focus_code_component_coordinates": [
                list(point) for point in self.code_focus_component_coordinates
            ],
            "sentinel": {
                "value": self.sentinel_value,
                "exact_stable_pixel_count": self.sentinel_exact_stable_pixel_count,
                "component_count": self.sentinel_component_count,
                "largest_component": self.sentinel_largest_component,
                "focus_xy": None if self.sentinel_focus_xy is None else list(self.sentinel_focus_xy),
                "focus_component_size": self.sentinel_focus_component_size,
                "focus_component_coordinates": [
                    list(point) for point in self.sentinel_focus_component_coordinates
                ],
            },
            "components": list(self.components),
            "interpretation_boundary": self.interpretation_boundary,
        }


@dataclass(frozen=True, slots=True)
class ProposalModeComparisonRow:
    """同一真实背景、同一注入坐标下 Gaussian 与 hybrid 的公平对照。"""

    source_path: str
    detector_psf_fwhm_px: float
    injected_psf_fwhm_px: float
    pair_separation_px: float | None
    peak_excess_adu: float
    injected_count: int
    gaussian_candidate_recall: float
    hybrid_candidate_recall: float
    ensemble_candidate_recall: float
    gaussian_quality_recall: float
    hybrid_quality_recall: float
    ensemble_quality_recall: float
    gaussian_mean_candidate_count: float
    hybrid_mean_candidate_count: float
    ensemble_mean_candidate_count: float
    gaussian_mean_quality_count: float
    hybrid_mean_quality_count: float
    ensemble_mean_quality_count: float
    trial_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "detector_psf_fwhm_px": self.detector_psf_fwhm_px,
            "injected_psf_fwhm_px": self.injected_psf_fwhm_px,
            "pair_separation_px": self.pair_separation_px,
            "peak_excess_adu": self.peak_excess_adu,
            "injected_count": self.injected_count,
            "gaussian_candidate_recall": self.gaussian_candidate_recall,
            "hybrid_candidate_recall": self.hybrid_candidate_recall,
            "ensemble_candidate_recall": self.ensemble_candidate_recall,
            "gaussian_quality_recall": self.gaussian_quality_recall,
            "hybrid_quality_recall": self.hybrid_quality_recall,
            "ensemble_quality_recall": self.ensemble_quality_recall,
            "gaussian_mean_candidate_count": self.gaussian_mean_candidate_count,
            "hybrid_mean_candidate_count": self.hybrid_mean_candidate_count,
            "ensemble_mean_candidate_count": self.ensemble_mean_candidate_count,
            "gaussian_mean_quality_count": self.gaussian_mean_quality_count,
            "hybrid_mean_quality_count": self.hybrid_mean_quality_count,
            "ensemble_mean_quality_count": self.ensemble_mean_quality_count,
            "trial_count": self.trial_count,
        }


@dataclass(frozen=True, slots=True)
class PairFluxRatioAuditRow:
    """固定总峰值下，不同双源强弱比的分离审计结果。

    ``primary`` 与 ``secondary`` 是注入真值的标签，不是检测器输出的
    亮度排序。``*_resolution_recall`` 要求同一对双源被两个不同候选分别
    命中；它比单独的 source recall 更严格，专门回答“看到了两个局部峰”
    是否真的等于“解析出了两颗源”。
    """

    source_path: str
    proposal_mode: str
    psf_model: str
    signal_normalization: str
    psf_source_count: int
    psf_median_fwhm_px: float | None
    psf_kernel_sum: float | None
    detector_psf_fwhm_px: float
    injected_psf_fwhm_px: float
    pair_separation_px: float
    total_peak_excess_adu: float | None
    total_control_signal_adu: float
    primary_control_signal_adu: float
    secondary_control_signal_adu: float
    secondary_to_primary_ratio: float
    primary_peak_excess_adu: float | None
    secondary_peak_excess_adu: float | None
    mean_injected_primary_peak_excess_adu: float
    mean_injected_secondary_peak_excess_adu: float
    pair_count: int
    injected_count: int
    primary_candidate_recall: float
    secondary_candidate_recall: float
    primary_quality_recall: float
    secondary_quality_recall: float
    source_candidate_recall: float
    source_quality_recall: float
    pair_candidate_resolution_recall: float
    pair_quality_resolution_recall: float
    merged_candidate_fraction: float
    merged_quality_fraction: float
    baseline_candidate_count: int
    baseline_quality_count: int
    mean_candidate_count: float
    mean_quality_count: float
    local_background_adu: float
    local_noise_adu: float
    trial_count: int
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "proposal_mode": self.proposal_mode,
            "psf_model": self.psf_model,
            "signal_normalization": self.signal_normalization,
            "psf_source_count": self.psf_source_count,
            "psf_median_fwhm_px": self.psf_median_fwhm_px,
            "psf_kernel_sum": self.psf_kernel_sum,
            "detector_psf_fwhm_px": self.detector_psf_fwhm_px,
            "injected_psf_fwhm_px": self.injected_psf_fwhm_px,
            "pair_separation_px": self.pair_separation_px,
            "total_peak_excess_adu": self.total_peak_excess_adu,
            "total_control_signal_adu": self.total_control_signal_adu,
            "primary_control_signal_adu": self.primary_control_signal_adu,
            "secondary_control_signal_adu": self.secondary_control_signal_adu,
            "secondary_to_primary_ratio": self.secondary_to_primary_ratio,
            "primary_peak_excess_adu": self.primary_peak_excess_adu,
            "secondary_peak_excess_adu": self.secondary_peak_excess_adu,
            "mean_injected_primary_peak_excess_adu": self.mean_injected_primary_peak_excess_adu,
            "mean_injected_secondary_peak_excess_adu": self.mean_injected_secondary_peak_excess_adu,
            "pair_count": self.pair_count,
            "injected_count": self.injected_count,
            "primary_candidate_recall": self.primary_candidate_recall,
            "secondary_candidate_recall": self.secondary_candidate_recall,
            "primary_quality_recall": self.primary_quality_recall,
            "secondary_quality_recall": self.secondary_quality_recall,
            "source_candidate_recall": self.source_candidate_recall,
            "source_quality_recall": self.source_quality_recall,
            "pair_candidate_resolution_recall": self.pair_candidate_resolution_recall,
            "pair_quality_resolution_recall": self.pair_quality_resolution_recall,
            "merged_candidate_fraction": self.merged_candidate_fraction,
            "merged_quality_fraction": self.merged_quality_fraction,
            "baseline_candidate_count": self.baseline_candidate_count,
            "baseline_quality_count": self.baseline_quality_count,
            "mean_candidate_count": self.mean_candidate_count,
            "mean_quality_count": self.mean_quality_count,
            "local_background_adu": self.local_background_adu,
            "local_noise_adu": self.local_noise_adu,
            "trial_count": self.trial_count,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class CrowdedBlendAuditRow:
    """固定真实背景下，多源拥挤组的合并/解析审计结果。

    ``group_*_resolution_recall`` 要求同一注入组内的每个真值分别由不同
    候选命中；``merged_*_fraction`` 则记录一个候选同时落入至少两个真值
    匹配半径的组比例。``extra_*`` 只描述注入组邻域内的候选数超过真值数，
    不把它直接解释成误检率。
    """

    source_path: str
    proposal_mode: str
    psf_model: str
    signal_normalization: str
    psf_source_count: int
    psf_median_fwhm_px: float | None
    psf_kernel_sum: float | None
    detector_psf_fwhm_px: float
    injected_psf_fwhm_px: float
    group_size: int
    nearest_separation_px: float
    total_control_signal_adu: float
    per_source_control_signal_adu: float
    group_count: int
    injected_count: int
    source_candidate_recall: float
    source_quality_recall: float
    group_candidate_resolution_recall: float
    group_quality_resolution_recall: float
    merged_candidate_fraction: float
    merged_quality_fraction: float
    extra_candidate_fraction: float
    extra_quality_fraction: float
    mean_candidate_count_in_group: float
    mean_quality_count_in_group: float
    baseline_candidate_count: int
    baseline_quality_count: int
    mean_candidate_count: float
    mean_quality_count: float
    local_background_adu: float
    local_noise_adu: float
    trial_count: int
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "proposal_mode": self.proposal_mode,
            "psf_model": self.psf_model,
            "signal_normalization": self.signal_normalization,
            "psf_source_count": self.psf_source_count,
            "psf_median_fwhm_px": self.psf_median_fwhm_px,
            "psf_kernel_sum": self.psf_kernel_sum,
            "detector_psf_fwhm_px": self.detector_psf_fwhm_px,
            "injected_psf_fwhm_px": self.injected_psf_fwhm_px,
            "group_size": self.group_size,
            "nearest_separation_px": self.nearest_separation_px,
            "total_control_signal_adu": self.total_control_signal_adu,
            "per_source_control_signal_adu": self.per_source_control_signal_adu,
            "group_count": self.group_count,
            "injected_count": self.injected_count,
            "source_candidate_recall": self.source_candidate_recall,
            "source_quality_recall": self.source_quality_recall,
            "group_candidate_resolution_recall": self.group_candidate_resolution_recall,
            "group_quality_resolution_recall": self.group_quality_resolution_recall,
            "merged_candidate_fraction": self.merged_candidate_fraction,
            "merged_quality_fraction": self.merged_quality_fraction,
            "extra_candidate_fraction": self.extra_candidate_fraction,
            "extra_quality_fraction": self.extra_quality_fraction,
            "mean_candidate_count_in_group": self.mean_candidate_count_in_group,
            "mean_quality_count_in_group": self.mean_quality_count_in_group,
            "baseline_candidate_count": self.baseline_candidate_count,
            "baseline_quality_count": self.baseline_quality_count,
            "mean_candidate_count": self.mean_candidate_count,
            "mean_quality_count": self.mean_quality_count,
            "local_background_adu": self.local_background_adu,
            "local_noise_adu": self.local_noise_adu,
            "trial_count": self.trial_count,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class DetectionSweepRow:
    """真实图像上一个候选阈值的检测统计。"""

    threshold_sigma: float
    candidate_count: int
    returned_count: int
    quality_count: int
    rejected_count: int
    line_artifact_count: int
    background_adu: float
    noise_adu: float

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold_sigma": self.threshold_sigma,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "quality_count": self.quality_count,
            "rejected_count": self.rejected_count,
            "line_artifact_count": self.line_artifact_count,
            "background_adu": self.background_adu,
            "noise_adu": self.noise_adu,
        }


@dataclass(frozen=True, slots=True)
class DetectionPSFSweepRow:
    """真实图像上一个匹配滤波 PSF 尺度的检测统计。"""

    psf_fwhm_px: float
    candidate_count: int
    returned_count: int
    quality_count: int
    rejected_count: int
    line_artifact_count: int
    background_adu: float
    noise_adu: float

    def as_dict(self) -> dict[str, object]:
        return {
            "psf_fwhm_px": self.psf_fwhm_px,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "quality_count": self.quality_count,
            "rejected_count": self.rejected_count,
            "line_artifact_count": self.line_artifact_count,
            "background_adu": self.background_adu,
            "noise_adu": self.noise_adu,
        }


@dataclass(frozen=True, slots=True)
class SingleFrameTrailAuditRow:
    """一张图的长线候选审计结果；不把单帧形状直接命名为运动真值。"""

    frame_number: int
    path: str
    timestamp: str | None
    candidate_count: int
    quality_count: int
    trail_count: int
    max_length_px: float | None
    max_width_px: float | None
    max_residual_snr: float | None
    edge_trail_count: int
    total_area_pixels: int
    longest_angle_deg: float | None
    longest_bbox: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_number": self.frame_number,
            "path": self.path,
            "timestamp": self.timestamp,
            "candidate_count": self.candidate_count,
            "quality_count": self.quality_count,
            "trail_count": self.trail_count,
            "max_length_px": self.max_length_px,
            "max_width_px": self.max_width_px,
            "max_residual_snr": self.max_residual_snr,
            "edge_trail_count": self.edge_trail_count,
            "total_area_pixels": self.total_area_pixels,
            "longest_angle_deg": self.longest_angle_deg,
            "longest_bbox": self.longest_bbox,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalPSF:
    """由真实质量源切片中值叠加得到的归一化 PSF。"""

    kernel: np.ndarray
    support_radius: int
    source_count: int
    median_fwhm_px: float | None
    median_ellipticity: float | None

    @property
    def model(self) -> str:
        return "empirical"

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "support_radius": self.support_radius,
            "source_count": self.source_count,
            "median_fwhm_px": self.median_fwhm_px,
            "median_ellipticity": self.median_ellipticity,
            "kernel_shape": list(self.kernel.shape),
            "kernel_sum": float(np.sum(self.kernel)),
        }


@dataclass(frozen=True, slots=True)
class RealBackgroundInjectionRow:
    """真实 FITS 背景上的一个注入强度档位统计。"""

    source_path: str
    proposal_mode: str
    psf_model: str
    psf_source_count: int
    psf_median_fwhm_px: float | None
    psf_kernel_sum: float | None
    peak_excess_adu: float
    injected_count: int
    candidate_recovered_count: int
    quality_recovered_count: int
    candidate_recall: float
    quality_recall: float
    baseline_candidate_count: int
    baseline_quality_count: int
    mean_candidate_count: float
    mean_quality_count: float
    mean_background_candidate_count: float
    mean_background_quality_count: float
    net_candidate_delta: float
    net_quality_delta: float
    local_background_adu: float
    local_noise_adu: float
    trial_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "proposal_mode": self.proposal_mode,
            "psf_model": self.psf_model,
            "psf_source_count": self.psf_source_count,
            "psf_median_fwhm_px": self.psf_median_fwhm_px,
            "psf_kernel_sum": self.psf_kernel_sum,
            "peak_excess_adu": self.peak_excess_adu,
            "injected_count": self.injected_count,
            "candidate_recovered_count": self.candidate_recovered_count,
            "quality_recovered_count": self.quality_recovered_count,
            "candidate_recall": self.candidate_recall,
            "quality_recall": self.quality_recall,
            "baseline_candidate_count": self.baseline_candidate_count,
            "baseline_quality_count": self.baseline_quality_count,
            "mean_candidate_count": self.mean_candidate_count,
            "mean_quality_count": self.mean_quality_count,
            "mean_background_candidate_count": self.mean_background_candidate_count,
            "mean_background_quality_count": self.mean_background_quality_count,
            "net_candidate_delta": self.net_candidate_delta,
            "net_quality_delta": self.net_quality_delta,
            "local_background_adu": self.local_background_adu,
            "local_noise_adu": self.local_noise_adu,
            "trial_count": self.trial_count,
        }


@dataclass(frozen=True, slots=True)
class StratifiedRealBackgroundInjectionRow:
    """真实 FITS 背景分层注入的一档统计。

    ``stratum`` 描述注入前的图像条件，而不是注入源的物理类别。候选/质量
    回收率是已知注入真值的测量；``ambiguous_injection_count`` 表示该位置
    在基线候选附近，因而更适合解释为拥挤/混合条件下的可检出性，而不是
    普通孤立源完备率。
    """

    source_path: str
    stratum: str
    stratum_label: str
    proposal_mode: str
    psf_model: str
    psf_source_count: int
    psf_median_fwhm_px: float | None
    psf_kernel_sum: float | None
    peak_excess_adu: float
    injected_count: int
    candidate_recovered_count: int
    quality_recovered_count: int
    candidate_recall: float
    quality_recall: float
    ambiguous_injection_count: int
    unambiguous_injected_count: int
    unambiguous_candidate_recall: float | None
    unambiguous_quality_recall: float | None
    baseline_candidate_count: int
    baseline_quality_count: int
    mean_candidate_count: float
    mean_quality_count: float
    mean_background_candidate_count: float
    mean_background_quality_count: float
    net_candidate_delta: float
    net_quality_delta: float
    local_background_adu: float
    local_noise_adu: float
    special_pixel_fraction: float
    nearest_baseline_source_px: float | None
    site_condition_json: str
    trial_count: int
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "stratum": self.stratum,
            "stratum_label": self.stratum_label,
            "proposal_mode": self.proposal_mode,
            "psf_model": self.psf_model,
            "psf_source_count": self.psf_source_count,
            "psf_median_fwhm_px": self.psf_median_fwhm_px,
            "psf_kernel_sum": self.psf_kernel_sum,
            "peak_excess_adu": self.peak_excess_adu,
            "injected_count": self.injected_count,
            "candidate_recovered_count": self.candidate_recovered_count,
            "quality_recovered_count": self.quality_recovered_count,
            "candidate_recall": self.candidate_recall,
            "quality_recall": self.quality_recall,
            "ambiguous_injection_count": self.ambiguous_injection_count,
            "unambiguous_injected_count": self.unambiguous_injected_count,
            "unambiguous_candidate_recall": self.unambiguous_candidate_recall,
            "unambiguous_quality_recall": self.unambiguous_quality_recall,
            "baseline_candidate_count": self.baseline_candidate_count,
            "baseline_quality_count": self.baseline_quality_count,
            "mean_candidate_count": self.mean_candidate_count,
            "mean_quality_count": self.mean_quality_count,
            "mean_background_candidate_count": self.mean_background_candidate_count,
            "mean_background_quality_count": self.mean_background_quality_count,
            "net_candidate_delta": self.net_candidate_delta,
            "net_quality_delta": self.net_quality_delta,
            "local_background_adu": self.local_background_adu,
            "local_noise_adu": self.local_noise_adu,
            "special_pixel_fraction": self.special_pixel_fraction,
            "nearest_baseline_source_px": self.nearest_baseline_source_px,
            "site_condition_json": self.site_condition_json,
            "trial_count": self.trial_count,
            "note": self.note,
        }


# ``Detection.flags`` are deliberately overlapping: one source can be weak,
# masked and narrow at the same time.  For a human audit we also need one
# deterministic *primary* class, otherwise a table of counts is impossible to
# reconcile.  The class is an algorithmic explanation, not a physical label.
_SOURCE_FEATURE_CLASSES: tuple[tuple[str, str], ...] = (
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

_SOURCE_FEATURE_FLAG_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("range_anomaly", frozenset({"SATURATED", "NEGATIVE_OVERFLOW", "CODE_PATTERN"})),
    ("linear_artifact", frozenset({"LINE_ARTIFACT"})),
    ("masked_or_edge", frozenset({"MASKED", "PARTIAL_MASKED", "EDGE"})),
    ("crowded_blend", frozenset({"UNRESOLVED_BLEND"})),
    (
        "spike_or_support",
        frozenset({"SPIKE", "NARROW", "SMALL_FOOTPRINT", "INSUFFICIENT_PSF_SUPPORT"}),
    ),
    ("weak_or_background", frozenset({"LOW_FLUX_SNR", "NON_POSITIVE_FLUX", "BACKGROUND_UNCERTAIN"})),
    ("shape_outlier", frozenset({"NO_SHAPE", "BROAD", "ELONGATED", "DIFFUSE"})),
)


def classify_source_feature(source: Detection) -> str:
    """返回一个可解释的互斥源级审计类别。

    ``Detection.flags`` 仍然是完整的、允许重叠的原因集合；本函数只按
    固定优先级挑一个首要类别，方便分层计数和抽样。优先级先处理数据
    范围/线状/有效性问题，再处理拥挤、尖峰、弱源和形状问题。它不能把
    ``compact_quality`` 解释成已证实的物理恒星，也不能替代星表或注入
    真值。
    """

    flags = frozenset(str(flag) for flag in source.flags)
    for feature_class, flag_group in _SOURCE_FEATURE_FLAG_GROUPS:
        if flags.intersection(flag_group):
            return feature_class
    if source.quality_passed:
        return "compact_quality"
    return "other_rejected"


def summarize_source_features(sources: Sequence[Detection]) -> tuple[dict[str, object], ...]:
    """按首要类别汇总候选/质量层，并保留可重叠 flags 的补充语义。

    ``candidate_count`` 是该类别在返回源表中的数量，``quality_count`` 是
    其中通过质量层的数量。类别本身互斥，但 ``flags`` 仍可能重叠；因此
    该表适合回答“首要特征构成”，不能用来把所有拒绝原因相加核对。
    """

    grouped: dict[str, list[Detection]] = {feature_class: [] for feature_class, _label in _SOURCE_FEATURE_CLASSES}
    for source in sources:
        grouped.setdefault(classify_source_feature(source), []).append(source)

    rows: list[dict[str, object]] = []
    for feature_class, label in _SOURCE_FEATURE_CLASSES:
        members = grouped[feature_class]
        snr_values = np.asarray(
            [
                float(source.flux_snr if source.flux_snr is not None else source.snr)
                for source in members
                if np.isfinite(source.flux_snr if source.flux_snr is not None else source.snr)
            ],
            dtype=np.float64,
        )
        if snr_values.size:
            median_snr: float | None = float(np.median(snr_values))
            maximum_snr: float | None = float(np.max(snr_values))
        else:
            median_snr = None
            maximum_snr = None
        quality_count = sum(bool(source.quality_passed) for source in members)
        flag_counts = Counter(flag for source in members for flag in source.flags)
        rows.append(
            {
                "feature_class": feature_class,
                "feature_class_label": label,
                "candidate_count": len(members),
                "quality_count": quality_count,
                "rejected_count": len(members) - quality_count,
                "quality_fraction": quality_count / len(members) if members else None,
                "median_flux_snr": median_snr,
                "max_flux_snr": maximum_snr,
                "common_flags": "|".join(
                    flag for flag, _count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
                ),
            }
        )
    return tuple(rows)


def _finite_source_attribute(sources: Sequence[Detection], attribute: str) -> np.ndarray:
    """Return finite numeric values for one source attribute.

    Research summaries must tolerate old serialized detections and diagnostic
    fields that are legitimately ``None``.  Keeping this conversion in one
    place also prevents a missing morphology measurement from being silently
    interpreted as zero.
    """

    values: list[float] = []
    for source in sources:
        value = getattr(source, attribute, None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return np.asarray(values, dtype=np.float64)


def _percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    """Return a percentile or ``None`` when an attribute is unavailable."""

    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def summarize_feature_morphology(
    sources: Sequence[Detection],
    *,
    high_flux_snr_threshold: float = 10.0,
) -> tuple[dict[str, object], ...]:
    """按互斥首要类别汇总形态、显著性、空间和拒绝原因。

    这是对 ``summarize_source_features`` 的第二层审计。前者回答“每类
    有多少候选/质量源”，本函数回答“这些候选的形态是否像点源、是否被
    某种边界条件主导、是否在高 SNR 下仍被拒绝”。

    ``high_flux_snr_threshold`` 只是诊断切片，不是物理真值标签；高 SNR
    仍可能来自线状结构、饱和/编码异常、未分辨混叠或错误背景估计。空间
    中位数同样只描述该类别在图像中的位置，不能替代星表/WCS 身份。
    """

    if not np.isfinite(high_flux_snr_threshold) or high_flux_snr_threshold <= 0:
        raise ValueError("high_flux_snr_threshold must be finite and positive")

    grouped: dict[str, list[Detection]] = {feature_class: [] for feature_class, _label in _SOURCE_FEATURE_CLASSES}
    for source in sources:
        grouped.setdefault(classify_source_feature(source), []).append(source)

    flag_groups = {
        "range_anomaly": frozenset({"SATURATED", "NEGATIVE_OVERFLOW", "CODE_PATTERN"}),
        "linear_artifact": frozenset({"LINE_ARTIFACT"}),
        "masked_or_edge": frozenset({"MASKED", "PARTIAL_MASKED", "EDGE"}),
        "crowded_blend": frozenset({"UNRESOLVED_BLEND"}),
        "spike_or_support": frozenset({"SPIKE", "NARROW", "SMALL_FOOTPRINT", "INSUFFICIENT_PSF_SUPPORT"}),
        "weak_or_background": frozenset({"LOW_FLUX_SNR", "NON_POSITIVE_FLUX", "BACKGROUND_UNCERTAIN"}),
        "shape_outlier": frozenset({"NO_SHAPE", "BROAD", "ELONGATED", "DIFFUSE"}),
    }
    morphology_attributes = (
        ("peak_snr", "snr"),
        ("flux_snr", "flux_snr"),
        ("filter_snr", "filter_snr"),
        ("fwhm_px", "fwhm"),
        ("ellipticity", "ellipticity"),
        ("sharpness", "sharpness"),
        ("psf_support_pixels", "psf_support_pixels"),
        ("footprint_pixels", "footprint_pixels"),
        ("centroid_shift_px", "centroid_shift_px"),
        ("x_px", "x"),
        ("y_px", "y"),
    )

    rows: list[dict[str, object]] = []
    for feature_class, label in _SOURCE_FEATURE_CLASSES:
        members = grouped[feature_class]
        quality_count = sum(bool(source.quality_passed) for source in members)
        rejected_count = len(members) - quality_count
        signal_values = _finite_source_attribute(members, "flux_snr")
        peak_values = _finite_source_attribute(members, "snr")
        high_snr_count = int(np.count_nonzero(signal_values >= high_flux_snr_threshold))
        high_snr_rejected_count = sum(
            1
            for source in members
            if not source.quality_passed
            and source.flux_snr is not None
            and np.isfinite(source.flux_snr)
            and float(source.flux_snr) >= high_flux_snr_threshold
        )
        flag_counts = Counter(flag for source in members for flag in source.flags)
        row: dict[str, object] = {
            "feature_class": feature_class,
            "feature_class_label": label,
            "candidate_count": len(members),
            "quality_count": quality_count,
            "rejected_count": rejected_count,
            "quality_fraction": quality_count / len(members) if members else None,
            "high_flux_snr_threshold": float(high_flux_snr_threshold),
            "high_flux_snr_count": high_snr_count,
            "high_flux_snr_rejected_count": high_snr_rejected_count,
            "high_flux_snr_rejection_fraction": high_snr_rejected_count / high_snr_count if high_snr_count else None,
            "median_peak_snr": _percentile_or_none(peak_values, 50.0),
            "median_flux_snr": _percentile_or_none(signal_values, 50.0),
            "common_flags": "|".join(
                flag for flag, _count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
            ),
        }
        for output_name, attribute in morphology_attributes:
            values = _finite_source_attribute(members, attribute)
            row[f"{output_name}_p10"] = _percentile_or_none(values, 10.0)
            row[f"{output_name}_median"] = _percentile_or_none(values, 50.0)
            row[f"{output_name}_p90"] = _percentile_or_none(values, 90.0)
        flags_by_source = [frozenset(str(flag) for flag in source.flags) for source in members]
        for group_name, group_flags in flag_groups.items():
            row[f"{group_name}_fraction"] = (
                sum(bool(flags.intersection(group_flags)) for flags in flags_by_source) / len(members)
                if members
                else None
            )
        rows.append(row)
    return tuple(rows)


def summarize_feature_spatial_distribution(
    sources: Sequence[Detection],
    image_shape: tuple[int, int],
    *,
    grid_size: int = 4,
    high_flux_snr_threshold: float = 10.0,
) -> tuple[dict[str, object], ...]:
    """按首要特征类别和粗空间网格汇总候选分布。

    空间集中性是区分线状结构、边缘问题和随机背景的重要诊断，但它不能
    单独证明某个候选是伪影：真实恒星也可能在一个局部视场内成团，固定
    detector 结构还可能跨帧重复。因此这里保留每个类别的候选/质量数、
    高 SNR 拒绝数和 SNR 分位数，只用于研究表和人工抽检，不写回质量层。
    ``grid_row`` 从图像上方开始，``grid_column`` 从左侧开始。
    """

    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if not np.isfinite(high_flux_snr_threshold) or high_flux_snr_threshold <= 0:
        raise ValueError("high_flux_snr_threshold must be finite and positive")
    if len(image_shape) != 2 or int(image_shape[0]) < 1 or int(image_shape[1]) < 1:
        raise ValueError("image_shape must contain two positive dimensions")
    height, width = int(image_shape[0]), int(image_shape[1])

    grouped: dict[str, list[Detection]] = {feature_class: [] for feature_class, _label in _SOURCE_FEATURE_CLASSES}
    for source in sources:
        grouped.setdefault(classify_source_feature(source), []).append(source)

    cells: dict[tuple[str, int, int], list[Detection]] = {}
    for feature_class, _label in _SOURCE_FEATURE_CLASSES:
        for grid_row in range(grid_size):
            for grid_column in range(grid_size):
                cells[(feature_class, grid_row, grid_column)] = []
    for source in sources:
        feature_class = classify_source_feature(source)
        raw_x = source.peak_x if source.peak_x is not None else source.x
        raw_y = source.peak_y if source.peak_y is not None else source.y
        try:
            x = float(raw_x)
            y = float(raw_y)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        grid_column = min(grid_size - 1, max(0, int(x * grid_size / width)))
        grid_row = min(grid_size - 1, max(0, int(y * grid_size / height)))
        cells[(feature_class, grid_row, grid_column)].append(source)

    rows: list[dict[str, object]] = []
    for feature_class, feature_label in _SOURCE_FEATURE_CLASSES:
        for grid_row in range(grid_size):
            for grid_column in range(grid_size):
                members = cells[(feature_class, grid_row, grid_column)]
                signal_values = _finite_source_attribute(members, "flux_snr")
                high_snr_count = int(np.count_nonzero(signal_values >= high_flux_snr_threshold))
                high_snr_rejected_count = sum(
                    1
                    for source in members
                    if not source.quality_passed
                    and source.flux_snr is not None
                    and np.isfinite(source.flux_snr)
                    and float(source.flux_snr) >= high_flux_snr_threshold
                )
                x_min = width * grid_column / grid_size
                x_max = width * (grid_column + 1) / grid_size
                y_min = height * grid_row / grid_size
                y_max = height * (grid_row + 1) / grid_size
                rows.append(
                    {
                        "feature_class": feature_class,
                        "feature_class_label": feature_label,
                        "grid_size": int(grid_size),
                        "grid_row": int(grid_row),
                        "grid_column": int(grid_column),
                        "x_min_px": float(x_min),
                        "x_max_px": float(x_max),
                        "y_min_px": float(y_min),
                        "y_max_px": float(y_max),
                        "candidate_count": len(members),
                        "quality_count": sum(bool(source.quality_passed) for source in members),
                        "rejected_count": sum(not bool(source.quality_passed) for source in members),
                        "quality_fraction": (
                            sum(bool(source.quality_passed) for source in members) / len(members)
                            if members
                            else None
                        ),
                        "high_flux_snr_threshold": float(high_flux_snr_threshold),
                        "high_flux_snr_count": high_snr_count,
                        "high_flux_snr_rejected_count": high_snr_rejected_count,
                        "median_flux_snr": _percentile_or_none(signal_values, 50.0),
                        "flux_snr_p90": _percentile_or_none(signal_values, 90.0),
                    }
                )
    return tuple(rows)


def _gaussian_kernel_patch(
    shape: tuple[int, int],
    x: float,
    y: float,
    fwhm: float,
) -> tuple[int, int, int, int, np.ndarray]:
    """返回与 ``_gaussian_source`` 相同的离散 Gaussian 小窗。"""

    sigma = max(float(fwhm) / 2.354820045, np.finfo(np.float64).eps)
    radius = max(4, int(math.ceil(4.0 * sigma)))
    x_min = max(0, int(math.floor(x)) - radius)
    x_max = min(shape[1], int(math.floor(x)) + radius + 1)
    y_min = max(0, int(math.floor(y)) - radius)
    y_max = min(shape[0], int(math.floor(y)) + radius + 1)
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    kernel = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    return y_min, y_max, x_min, x_max, kernel


def _gaussian_source(image: np.ndarray, x: float, y: float, peak_excess_adu: float, fwhm: float) -> None:
    """以峰值超额 ADU 注入一个离散 Gaussian 源。"""

    y_min, y_max, x_min, x_max, kernel = _gaussian_kernel_patch(image.shape, x, y, fwhm)
    image[y_min:y_max, x_min:x_max] += peak_excess_adu * kernel


def estimate_empirical_psf(
    image: np.ndarray,
    sources: Sequence[Detection],
    *,
    support_radius: int = 7,
    max_sources: int = 64,
    isolation_sources: Sequence[Detection] | None = None,
) -> EmpiricalPSF | None:
    """从亮、孤立、未饱和质量源提取并中值叠加实测 PSF。

    每个切片先按源的局部背景扣除，再做亚像素中心对齐和峰值归一化。
    ``sources`` 是用于抽取模板的候选池；函数内部再要求
    ``quality_passed``。默认也用这张表检查邻近污染；构造局部模板时，
    ``isolation_sources`` 可以传入整幅图的候选表，让“局部取样池”和
    “全局隔离参照”分离，避免局部窗口边界隐藏邻峰。若调用方只传质量
    子集作为两者，隔离审计会变得过宽。该函数只建立注入模型，不改变
    输入图像；若合格源不足则返回 None，由调用方明确回退到 Gaussian，
    而不是静默伪造实测 PSF。
    """

    if support_radius < 3 or max_sources < 1:
        raise ValueError("support_radius must be at least 3 and max_sources must be positive")
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    radius = int(support_radius)
    margin = radius + 1
    aux_mask = auxiliary_mask(values.shape)
    eligible = [
        source
        for source in sources
        if margin <= source.x < values.shape[1] - margin
        and margin <= source.y < values.shape[0] - margin
        and source.quality_passed
        and not any(
            flag in source.flags
            for flag in (
                "EDGE",
                "MASKED",
                "PARTIAL_MASKED",
                "SATURATED",
                "LINE_ARTIFACT",
                "NEGATIVE_OVERFLOW",
            )
        )
        and source.ellipticity is not None
        and float(source.ellipticity) <= 0.45
        and source.fwhm is not None
        and 1.0 <= float(source.fwhm) <= 8.0
    ]
    eligible.sort(
        key=lambda source: float(source.flux_snr if source.flux_snr is not None else source.snr),
        reverse=True,
    )
    isolation_pool = sources if isolation_sources is None else isolation_sources
    all_points = np.asarray([(source.x, source.y) for source in isolation_pool], dtype=np.float64)
    all_tree = cKDTree(all_points) if all_points.size else None
    stamps: list[np.ndarray] = []
    fwhm_values: list[float] = []
    ellipticity_values: list[float] = []
    for source in eligible:
        if len(stamps) >= max_sources:
            break
        if all_tree is not None and len(isolation_pool) > 1:
            # ``sources`` may be only a local template pool while
            # ``isolation_pool`` is the full-frame candidate table.  The
            # nearest-neighbour query must therefore be sized from the
            # isolation pool, otherwise a small local pool can hide a nearby
            # rejected/crowded candidate.
            distances, _indices = all_tree.query((source.x, source.y), k=2)
            distances = np.atleast_1d(distances)
            nearest_other = float(distances[-1]) if np.isclose(distances[0], 0.0) else float(distances[0])
            if nearest_other < max(3.0 * radius, 18.0):
                continue
        center_x = int(round(source.x))
        center_y = int(round(source.y))
        y0, y1 = center_y - radius, center_y + radius + 1
        x0, x1 = center_x - radius, center_x + radius + 1
        stamp = np.asarray(values[y0:y1, x0:x1], dtype=np.float64)
        if stamp.shape != (2 * radius + 1, 2 * radius + 1) or not np.all(np.isfinite(stamp)):
            continue
        if aux_mask[y0:y1, x0:x1].any():
            continue
        local_background = float(source.background) if np.isfinite(source.background) else float(np.median(stamp))
        stamp = np.maximum(stamp - local_background, 0.0)
        shift_y = float(center_y - source.y)
        shift_x = float(center_x - source.x)
        aligned = ndimage.shift(
            stamp,
            shift=(shift_y, shift_x),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        peak = float(np.max(aligned))
        if not np.isfinite(peak) or peak <= 0:
            continue
        stamps.append((aligned / peak).astype(np.float32))
        fwhm_values.append(float(source.fwhm))
        if source.ellipticity is not None and np.isfinite(source.ellipticity):
            ellipticity_values.append(float(source.ellipticity))
    if len(stamps) < 3:
        return None
    kernel = np.median(np.stack(stamps, axis=0), axis=0).astype(np.float32)
    kernel = np.maximum(kernel, 0.0)
    kernel_peak = float(np.max(kernel))
    if not np.isfinite(kernel_peak) or kernel_peak <= 0:
        return None
    kernel /= kernel_peak
    return EmpiricalPSF(
        kernel=kernel,
        support_radius=radius,
        source_count=len(stamps),
        median_fwhm_px=float(np.median(fwhm_values)) if fwhm_values else None,
        median_ellipticity=float(np.median(ellipticity_values)) if ellipticity_values else None,
    )


def _psf_similarity_metrics(
    values: np.ndarray,
    source: Detection,
    psf: EmpiricalPSF,
    aux_mask: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """计算单个原始裁剪相对经验 PSF 的诊断量。"""

    radius = int(psf.support_radius)
    kernel = np.asarray(psf.kernel, dtype=np.float64)
    expected_shape = (2 * radius + 1, 2 * radius + 1)
    if kernel.shape != expected_shape:
        raise ValueError(f"empirical PSF shape {kernel.shape} does not match support radius {radius}")
    center_x = int(round(float(source.x)))
    center_y = int(round(float(source.y)))
    y0, y1 = center_y - radius, center_y + radius + 1
    x0, x1 = center_x - radius, center_x + radius + 1
    stamp = np.asarray(values[y0:y1, x0:x1], dtype=np.float64)
    if stamp.shape != kernel.shape:
        return None
    valid = np.isfinite(stamp) & ~aux_mask[y0:y1, x0:x1]
    if int(np.count_nonzero(valid)) < int(np.ceil(0.8 * stamp.size)):
        return None
    raw_background = getattr(source, "background", None)
    try:
        background = float(raw_background)
    except (TypeError, ValueError):
        background = float(np.nanmedian(stamp))
    if not np.isfinite(background):
        background = float(np.nanmedian(stamp))
    observed = np.where(valid, np.maximum(stamp - background, 0.0), 0.0)
    shift_y = float(center_y - source.y)
    shift_x = float(center_x - source.x)
    observed = ndimage.shift(
        observed,
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    template = np.where(valid, kernel, 0.0)
    data_norm = float(np.sqrt(np.sum(observed**2)))
    template_norm = float(np.sqrt(np.sum(template**2)))
    denominator = data_norm * template_norm
    epsilon = np.finfo(np.float64).eps
    if denominator <= epsilon:
        return None
    dot = float(np.sum(observed * template))
    amplitude = max(dot / max(float(np.sum(template**2)), epsilon), 0.0)
    residual = float(np.sqrt(np.sum((observed - amplitude * template) ** 2)) / max(data_norm, epsilon))
    positive_sum = float(np.sum(observed))
    if positive_sum <= epsilon:
        return None
    central = np.zeros_like(observed, dtype=bool)
    central[radius - 1 : radius + 2, radius - 1 : radius + 2] = True
    central_fraction = float(np.sum(observed[central]) / positive_sum)
    return (dot / denominator, residual, central_fraction, amplitude)


def summarize_feature_psf_similarity(
    frame: FitsFrame | np.ndarray | str | Path,
    sources: Sequence[Detection],
    *,
    psf: EmpiricalPSF | None = None,
    per_class: int = 8,
    support_radius: int = 7,
    correlation_threshold: float = 0.8,
) -> tuple[dict[str, object], ...]:
    """对每个首要类别的原始裁剪计算经验 PSF 相似度。

    该函数只做确定性抽样和诊断，不改变 ``quality_passed``。观测裁剪先
    扣除该源的局部背景并取正残差，再与归一化经验核计算余弦相似度；同时
    输出相对残差和中心 3×3 能量占比。极端码、邻近源和 PSF 空间变化都会
    影响这些量，因此 ``correlation_threshold`` 不是恒星概率，也不直接
    进入默认质量规则。
    """

    if per_class < 1 or support_radius < 3:
        raise ValueError("per_class must be positive and support_radius must be at least 3")
    if not 0.0 < float(correlation_threshold) <= 1.0:
        raise ValueError("correlation_threshold must be between 0 and 1")
    if isinstance(frame, FitsFrame):
        values = np.asarray(frame.data)
    elif isinstance(frame, (str, Path)):
        values = np.asarray(read_fits(frame).data)
    else:
        values = np.asarray(frame)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if psf is None:
        psf = estimate_empirical_psf(values, sources, support_radius=support_radius, max_sources=64)

    kernel: np.ndarray | None = None
    radius = int(support_radius)
    psf_source_count = 0
    psf_model = "unavailable"
    if psf is not None:
        radius = int(psf.support_radius)
        kernel = np.asarray(psf.kernel, dtype=np.float64)
        expected_shape = (2 * radius + 1, 2 * radius + 1)
        if kernel.shape != expected_shape:
            raise ValueError(f"empirical PSF shape {kernel.shape} does not match support radius {radius}")
        if not np.all(np.isfinite(kernel)) or float(np.max(kernel)) <= 0:
            raise ValueError("empirical PSF kernel must be finite and positive")
        psf_source_count = int(psf.source_count)
        psf_model = psf.model

    def signal_snr(source: Detection) -> float:
        value = source.flux_snr if source.flux_snr is not None else source.snr
        return float(value) if value is not None and np.isfinite(value) else float("-inf")

    grouped: dict[str, list[Detection]] = {feature_class: [] for feature_class, _label in _SOURCE_FEATURE_CLASSES}
    for source in sources:
        grouped.setdefault(classify_source_feature(source), []).append(source)
    selected_by_class: dict[str, list[Detection]] = {}
    for feature_class, _label in _SOURCE_FEATURE_CLASSES:
        members = sorted(grouped[feature_class], key=lambda source: (signal_snr(source), source.detection_id))
        if len(members) > per_class:
            indices = np.rint(np.linspace(0, len(members) - 1, per_class)).astype(int)
            members = [members[int(index)] for index in indices]
        selected_by_class[feature_class] = members

    aux_mask = auxiliary_mask(values.shape)
    rows: list[dict[str, object]] = []
    for feature_class, feature_label in _SOURCE_FEATURE_CLASSES:
        selected = selected_by_class[feature_class]
        correlations: list[float] = []
        residuals: list[float] = []
        central_fractions: list[float] = []
        amplitudes: list[float] = []
        sample_ids: list[str] = []
        for source in selected:
            sample_ids.append(str(source.detection_id))
            if kernel is None:
                continue
            metrics = _psf_similarity_metrics(values, source, psf, aux_mask)
            if metrics is None:
                continue
            correlation, residual, central_fraction, amplitude = metrics
            correlations.append(correlation)
            residuals.append(residual)
            central_fractions.append(central_fraction)
            amplitudes.append(amplitude)

        def median(values_: Sequence[float]) -> float | None:
            return float(np.median(values_)) if values_ else None

        def percentile(values_: Sequence[float], percentile_: float) -> float | None:
            return float(np.percentile(values_, percentile_)) if values_ else None

        count_ge_threshold = sum(value >= float(correlation_threshold) for value in correlations)
        rows.append(
            {
                "feature_class": feature_class,
                "feature_class_label": feature_label,
                "psf_model": psf_model,
                "psf_source_count": psf_source_count,
                "correlation_threshold": float(correlation_threshold),
                "sample_count": len(selected),
                "valid_count": len(correlations),
                "valid_fraction": len(correlations) / len(selected) if selected else None,
                "correlation_ge_threshold_count": count_ge_threshold,
                "correlation_ge_threshold_fraction": count_ge_threshold / len(correlations) if correlations else None,
                "psf_correlation_p10": percentile(correlations, 10.0),
                "psf_correlation_median": median(correlations),
                "psf_correlation_p90": percentile(correlations, 90.0),
                "relative_residual_p10": percentile(residuals, 10.0),
                "relative_residual_median": median(residuals),
                "relative_residual_p90": percentile(residuals, 90.0),
                "central_energy_fraction_p10": percentile(central_fractions, 10.0),
                "central_energy_fraction_median": median(central_fractions),
                "central_energy_fraction_p90": percentile(central_fractions, 90.0),
                "fitted_amplitude_adu_median": median(amplitudes),
                "sample_detection_ids": "|".join(sample_ids),
            }
        )
    return tuple(rows)


def summarize_spatial_psf_similarity(
    frame: FitsFrame | np.ndarray | str | Path,
    sources: Sequence[Detection],
    *,
    global_psf: EmpiricalPSF | None = None,
    per_class: int = 8,
    support_radius: int = 7,
    grid_size: int = 4,
    correlation_threshold: float = 0.8,
) -> tuple[dict[str, object], ...]:
    """比较全局经验 PSF 与按 detector 网格构造的局部经验 PSF。

    每个类别只抽取确定性的少量样本；局部模板从样本之外的候选池构造，
    避免把被比较的裁剪直接泄漏进模板。局部模板不足时不伪造结果，保留
    ``local_fallback_count``，并把这一步限定为 PSF 空间变化诊断。输出的
    改善比例不能解释为召回率，也不会改变 ``quality_passed``。
    """

    if per_class < 1 or support_radius < 3 or grid_size < 1:
        raise ValueError("per_class and grid_size must be positive; support_radius must be at least 3")
    if not 0.0 < float(correlation_threshold) <= 1.0:
        raise ValueError("correlation_threshold must be between 0 and 1")
    if isinstance(frame, FitsFrame):
        values = np.asarray(frame.data)
    elif isinstance(frame, (str, Path)):
        values = np.asarray(read_fits(frame).data)
    else:
        values = np.asarray(frame)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    height, width = (int(values.shape[0]), int(values.shape[1]))
    if global_psf is None:
        global_psf = estimate_empirical_psf(values, sources, support_radius=support_radius, max_sources=64)

    def signal_snr(source: Detection) -> float:
        value = source.flux_snr if source.flux_snr is not None else source.snr
        return float(value) if value is not None and np.isfinite(value) else float("-inf")

    def source_xy(source: Detection) -> tuple[float, float] | None:
        try:
            x, y = float(source.x), float(source.y)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return x, y

    grouped: dict[str, list[Detection]] = {feature_class: [] for feature_class, _label in _SOURCE_FEATURE_CLASSES}
    for source in sources:
        grouped.setdefault(classify_source_feature(source), []).append(source)
    selected_by_class: dict[str, list[Detection]] = {}
    for feature_class, _label in _SOURCE_FEATURE_CLASSES:
        members = sorted(grouped[feature_class], key=lambda source: (signal_snr(source), source.detection_id))
        if len(members) > per_class:
            indices = np.rint(np.linspace(0, len(members) - 1, per_class)).astype(int)
            members = [members[int(index)] for index in indices]
        selected_by_class[feature_class] = members
    selected_ids = {int(source.detection_id) for members in selected_by_class.values() for source in members}
    aux_mask = auxiliary_mask(values.shape)
    padding = max(18.0, float(2 * support_radius + 4))
    local_psf_cache: dict[tuple[int, int], EmpiricalPSF | None] = {}

    def cell_for(source: Detection) -> tuple[int, int] | None:
        coordinates = source_xy(source)
        if coordinates is None:
            return None
        x, y = coordinates
        column = min(grid_size - 1, max(0, int(x * grid_size / width)))
        row = min(grid_size - 1, max(0, int(y * grid_size / height)))
        return row, column

    def local_psf_for(cell: tuple[int, int]) -> EmpiricalPSF | None:
        if cell in local_psf_cache:
            return local_psf_cache[cell]
        row, column = cell
        x_min = width * column / grid_size - padding
        x_max = width * (column + 1) / grid_size + padding
        y_min = height * row / grid_size - padding
        y_max = height * (row + 1) / grid_size + padding
        pool = [
            source
            for source in sources
            if int(source.detection_id) not in selected_ids
            and (coordinates := source_xy(source)) is not None
            and x_min <= coordinates[0] < x_max
            and y_min <= coordinates[1] < y_max
        ]
        local_psf_cache[cell] = estimate_empirical_psf(
            values,
            pool,
            support_radius=support_radius,
            max_sources=64,
            isolation_sources=sources,
        )
        return local_psf_cache[cell]

    def median(values_: Sequence[float]) -> float | None:
        return float(np.median(values_)) if values_ else None

    def percentile(values_: Sequence[float], percentile_: float) -> float | None:
        return float(np.percentile(values_, percentile_)) if values_ else None

    rows: list[dict[str, object]] = []
    for feature_class, feature_label in _SOURCE_FEATURE_CLASSES:
        selected = selected_by_class[feature_class]
        global_correlations: list[float] = []
        local_correlations: list[float] = []
        global_residuals: list[float] = []
        local_residuals: list[float] = []
        paired_correlation_deltas: list[float] = []
        paired_residual_deltas: list[float] = []
        local_source_counts: list[float] = []
        local_fallback_count = 0
        used_cells: set[tuple[int, int]] = set()
        sample_ids: list[str] = []
        for source in selected:
            sample_ids.append(str(source.detection_id))
            global_metrics = (
                _psf_similarity_metrics(values, source, global_psf, aux_mask) if global_psf is not None else None
            )
            cell = cell_for(source)
            local_psf = local_psf_for(cell) if cell is not None else None
            if cell is not None:
                used_cells.add(cell)
            if local_psf is None:
                local_fallback_count += 1
            local_metrics = (
                _psf_similarity_metrics(values, source, local_psf, aux_mask) if local_psf is not None else None
            )
            if global_metrics is not None:
                global_correlations.append(float(global_metrics[0]))
                global_residuals.append(float(global_metrics[1]))
            if local_metrics is not None:
                local_source_counts.append(float(local_psf.source_count))
                local_correlations.append(float(local_metrics[0]))
                local_residuals.append(float(local_metrics[1]))
            if global_metrics is not None and local_metrics is not None:
                paired_correlation_deltas.append(float(local_metrics[0] - global_metrics[0]))
                paired_residual_deltas.append(float(local_metrics[1] - global_metrics[1]))

        local_psfs = [local_psf_cache[cell] for cell in used_cells]
        available_local_psfs = [psf for psf in local_psfs if psf is not None]
        local_corr_improved = sum(delta > 0.0 for delta in paired_correlation_deltas)
        local_residual_reduced = sum(delta < 0.0 for delta in paired_residual_deltas)
        rows.append(
            {
                "feature_class": feature_class,
                "feature_class_label": feature_label,
                "grid_size": int(grid_size),
                "global_psf_source_count": int(global_psf.source_count) if global_psf is not None else 0,
                "local_template_cell_count": len(used_cells),
                "local_template_available_cell_count": len(available_local_psfs),
                "local_template_source_count_median": median([float(psf.source_count) for psf in available_local_psfs]),
                "sample_count": len(selected),
                "global_valid_count": len(global_correlations),
                "local_valid_count": len(local_correlations),
                "local_fallback_count": local_fallback_count,
                "correlation_threshold": float(correlation_threshold),
                "global_correlation_median": median(global_correlations),
                "local_correlation_median": median(local_correlations),
                "paired_correlation_count": len(paired_correlation_deltas),
                "local_correlation_improved_count": local_corr_improved,
                "local_correlation_improved_fraction": (
                    local_corr_improved / len(paired_correlation_deltas) if paired_correlation_deltas else None
                ),
                "global_residual_median": median(global_residuals),
                "local_residual_median": median(local_residuals),
                "paired_residual_count": len(paired_residual_deltas),
                "local_residual_reduced_count": local_residual_reduced,
                "local_residual_reduced_fraction": (
                    local_residual_reduced / len(paired_residual_deltas) if paired_residual_deltas else None
                ),
                "median_correlation_delta_local_minus_global": median(paired_correlation_deltas),
                "median_residual_delta_local_minus_global": median(paired_residual_deltas),
                "sample_detection_ids": "|".join(sample_ids),
            }
        )
    return tuple(rows)


def _empirical_kernel_patch(
    shape: tuple[int, int],
    x: float,
    y: float,
    psf: EmpiricalPSF,
) -> tuple[int, int, int, int, np.ndarray]:
    """返回与 ``_empirical_source`` 相同的亚像素实测 PSF 小窗。"""

    radius = psf.support_radius
    center_x = int(round(x))
    center_y = int(round(y))
    fractional_shift = (float(y - center_y), float(x - center_x))
    kernel = ndimage.shift(
        np.asarray(psf.kernel, dtype=np.float32),
        shift=fractional_shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    y0 = max(0, center_y - radius)
    y1 = min(shape[0], center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(shape[1], center_x + radius + 1)
    kernel_y0 = y0 - (center_y - radius)
    kernel_x0 = x0 - (center_x - radius)
    clipped_kernel = kernel[
        kernel_y0 : kernel_y0 + (y1 - y0),
        kernel_x0 : kernel_x0 + (x1 - x0),
    ]
    return y0, y1, x0, x1, clipped_kernel


def _empirical_source(image: np.ndarray, x: float, y: float, peak_excess_adu: float, psf: EmpiricalPSF) -> None:
    """把归一化实测 PSF 以亚像素中心叠加到图像副本。"""

    y0, y1, x0, x1, kernel = _empirical_kernel_patch(image.shape, x, y, psf)
    image[y0:y1, x0:x1] += float(peak_excess_adu) * kernel


def _inject_source_signal(
    image: np.ndarray,
    x: float,
    y: float,
    signal_adu: float,
    *,
    signal_normalization: str,
    psf_model: str,
    gaussian_fwhm: float,
    empirical_psf: EmpiricalPSF | None,
) -> float:
    """注入源并返回实际使用的峰值超额。

    ``peak_excess`` 延续原实验口径；``integrated_excess`` 则先生成单位
    峰值的离散核，再按该小窗的核和归一化，使每个源的离散积分超额相同。
    后者会随亚像素位置产生轻微峰值变化，但避免把 Gaussian 与经验 PSF
    的核和差异误写成 PSF 性能差异。
    """

    if signal_adu <= 0 or not np.isfinite(signal_adu):
        raise ValueError("signal_adu must be finite and positive")
    if signal_normalization == "peak_excess":
        if psf_model == "empirical":
            if empirical_psf is None:
                raise ValueError("empirical_psf is required for empirical injection")
            _empirical_source(image, x, y, signal_adu, empirical_psf)
        elif psf_model == "gaussian":
            _gaussian_source(image, x, y, signal_adu, gaussian_fwhm)
        else:
            raise ValueError("psf_model must be gaussian or empirical")
        return float(signal_adu)
    if signal_normalization != "integrated_excess":
        raise ValueError("signal_normalization must be peak_excess or integrated_excess")
    if psf_model == "empirical":
        if empirical_psf is None:
            raise ValueError("empirical_psf is required for empirical injection")
        y0, y1, x0, x1, kernel = _empirical_kernel_patch(image.shape, x, y, empirical_psf)
    elif psf_model == "gaussian":
        y0, y1, x0, x1, kernel = _gaussian_kernel_patch(image.shape, x, y, gaussian_fwhm)
    else:
        raise ValueError("psf_model must be gaussian or empirical")
    kernel_sum = float(np.sum(kernel, dtype=np.float64))
    if not np.isfinite(kernel_sum) or kernel_sum <= 0:
        raise ValueError("injection PSF kernel must have a finite positive sum")
    peak_excess = float(signal_adu) / kernel_sum
    image[y0:y1, x0:x1] += peak_excess * kernel
    return peak_excess


def _random_positions(rng: np.random.Generator, count: int, shape: tuple[int, int], *, margin: int = 14) -> list[tuple[float, float]]:
    height, width = shape
    positions: list[tuple[float, float]] = []
    minimum_distance = 12.0
    for _attempt in range(max(1000, count * 100)):
        if len(positions) >= count:
            break
        x = float(rng.uniform(margin, width - margin))
        y = float(rng.uniform(margin, height - margin))
        if all(math.hypot(x - previous_x, y - previous_y) >= minimum_distance for previous_x, previous_y in positions):
            positions.append((x, y))
    if len(positions) != count:
        raise RuntimeError("无法为注入实验生成足够的非拥挤位置")
    return positions


def _matched_count(injected: Sequence[tuple[float, float]], detections: Sequence[Detection], radius_px: float) -> int:
    return int(sum(_match_target_flags(injected, detections, radius_px)))


def _match_target_flags(
    injected: Sequence[tuple[float, float]],
    detections: Sequence[Detection],
    radius_px: float,
) -> tuple[bool, ...]:
    """对注入真值做一对一匹配，并返回每个真值是否被命中。

    逐个目标贪心匹配保持了既有实验的口径，同时把“一个候选同时落在
    两个真值半径内”明确记成只能命中一个目标。这一点对双源实验很关键：
    单个中间峰可以提高 source recall 的一半，但不能被计为 pair resolved。
    """

    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    remaining = list(detections)
    matched: list[bool] = []
    for x, y in injected:
        best_index: int | None = None
        best_distance = float(radius_px)
        for index, detection in enumerate(remaining):
            distance = math.hypot(detection.x - x, detection.y - y)
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None:
            remaining.pop(best_index)
            matched.append(True)
        else:
            matched.append(False)
    return tuple(matched)


def _merged_pair_fraction(
    pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    detections: Sequence[Detection],
    radius_px: float,
) -> float:
    """返回至少一个候选同时落入同一对两个匹配半径的比例。"""

    if not pairs:
        return 0.0
    merged = 0
    for (primary_x, primary_y), (secondary_x, secondary_y) in pairs:
        if any(
            math.hypot(detection.x - primary_x, detection.y - primary_y) <= radius_px
            and math.hypot(detection.x - secondary_x, detection.y - secondary_y) <= radius_px
            for detection in detections
        ):
            merged += 1
    return merged / len(pairs)


def _merged_group_flag(
    group: Sequence[tuple[float, float]],
    detections: Sequence[Detection],
    radius_px: float,
) -> bool:
    """判断一个候选是否同时落入拥挤组内至少两个真值的匹配半径。"""

    if len(group) < 2:
        return False
    return any(
        sum(math.hypot(detection.x - x, detection.y - y) <= radius_px for x, y in group) >= 2
        for detection in detections
    )


def _group_region_candidate_count(
    group: Sequence[tuple[float, float]],
    detections: Sequence[Detection],
    radius_px: float,
) -> int:
    """统计注入组包络邻域内的候选数，供合并/多分裂诊断使用。"""

    if not group:
        return 0
    center_x = float(np.mean([point[0] for point in group]))
    center_y = float(np.mean([point[1] for point in group]))
    region_radius = max(
        float(radius_px),
        max(math.hypot(x - center_x, y - center_y) for x, y in group) + float(radius_px),
    )
    return int(
        sum(math.hypot(detection.x - center_x, detection.y - center_y) <= region_radius for detection in detections)
    )


def _local_patch_stats(image: np.ndarray, x: float, y: float, *, radius: int = 5) -> tuple[float, float, float] | None:
    """返回注入位置附近的中位背景、MAD 噪声和高分位超额。"""

    center_x = int(round(x))
    center_y = int(round(y))
    y0 = max(0, center_y - radius)
    y1 = min(image.shape[0], center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(image.shape[1], center_x + radius + 1)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    finite = patch[np.isfinite(patch)]
    if finite.size < 16:
        return None
    median = float(np.median(finite))
    mad_noise = max(1.4826 * float(np.median(np.abs(finite - median))), np.finfo(np.float64).eps)
    upper_excess = float(np.quantile(finite, 0.98) - median)
    return median, mad_noise, upper_excess


def _select_real_background_positions(
    image: np.ndarray,
    existing_sources: Sequence[Detection],
    rng: np.random.Generator,
    count: int,
    *,
    noise_adu: float,
    margin: int = 14,
    source_exclusion_radius_px: float = 12.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """在真实图像中选取相对空白、彼此不拥挤的注入位置。

    位置选择只用于控制实验条件，不声称这些窗口在物理上绝对没有恒星。
    它们必须远离当前检测候选，并通过局部高分位超额检查；筛选失败时
    明确报错，不静默退化成把已有亮源当作注入真值。
    """

    if count < 1:
        raise ValueError("count must be positive")
    height, width = image.shape
    if height <= 2 * margin or width <= 2 * margin:
        raise ValueError("image is too small for real-background injection margins")
    existing_points = np.asarray([(source.x, source.y) for source in existing_sources], dtype=np.float64)
    existing_tree = cKDTree(existing_points) if existing_points.size else None
    positions: list[tuple[float, float]] = []
    local_stats: list[tuple[float, float]] = []
    attempts_limit = max(5000, count * 1500)
    for _ in range(attempts_limit):
        if len(positions) >= count:
            break
        x = float(rng.uniform(margin, width - margin))
        y = float(rng.uniform(margin, height - margin))
        if positions and min(math.hypot(x - px, y - py) for px, py in positions) < source_exclusion_radius_px:
            continue
        if existing_tree is not None:
            distance, _index = existing_tree.query((x, y), distance_upper_bound=source_exclusion_radius_px)
            if np.isfinite(distance) and float(distance) < source_exclusion_radius_px:
                continue
        patch_stats = _local_patch_stats(image, x, y)
        if patch_stats is None:
            continue
        local_background, local_noise, upper_excess = patch_stats
        allowed_excess = max(3.0 * local_noise, 3.0 * float(noise_adu), np.finfo(np.float64).eps)
        if upper_excess > allowed_excess:
            continue
        if not np.isfinite(local_background) or not np.isfinite(local_noise):
            continue
        positions.append((x, y))
        local_stats.append((local_background, local_noise))
    if len(positions) != count:
        raise RuntimeError(
            f"无法在真实背景中找到 {count} 个合规注入位置，仅找到 {len(positions)} 个；"
            "请减少每次注入数量或先检查图像拥挤/条纹区域"
        )
    return positions, local_stats


def _count_outside_injection_regions(
    detections: Sequence[Detection],
    injected: Sequence[tuple[float, float]],
    radius_px: float,
) -> int:
    """统计不在注入保护区内的检测；不将其解释为真实误检。"""

    if not detections:
        return 0
    if not injected:
        return len(detections)
    tree = cKDTree(np.asarray(injected, dtype=np.float64))
    points = np.asarray([(source.x, source.y) for source in detections], dtype=np.float64)
    distances, _indices = tree.query(points, distance_upper_bound=radius_px)
    return int(np.count_nonzero(~np.isfinite(distances)))


def _nearby_detections(
    detections: Sequence[Detection],
    positions: Sequence[tuple[float, float]],
    radius_px: float,
) -> tuple[Detection, ...]:
    """返回特征位置邻域内的唯一检测源。"""

    if not detections or not positions:
        return ()
    tree = cKDTree(np.asarray(positions, dtype=np.float64))
    points = np.asarray([(source.x, source.y) for source in detections], dtype=np.float64)
    distances, _indices = tree.query(points, distance_upper_bound=float(radius_px))
    return tuple(source for source, distance in zip(detections, distances, strict=True) if np.isfinite(distance))


@dataclass(frozen=True, slots=True)
class _RealInjectionSite:
    """真实背景上的一个带条件描述的注入位置。"""

    x: float
    y: float
    local_background_adu: float
    local_noise_adu: float
    upper_excess_adu: float
    special_pixel_fraction: float
    baseline_neighbor_count: int
    nearest_baseline_source_px: float | None


_REAL_INJECTION_STRATUM_LABELS: dict[str, str] = {
    "blank": "候选稀疏/低纹理背景",
    "high_background": "候选稀疏/高局部噪声背景",
    "edge": "图像边缘截断",
    "crowded": "亮源邻域/拥挤混合",
    "special_code": "特殊值域邻域",
    "line": "线状伪影邻域",
}


FeatureScenario = tuple[
    str,
    str,
    str,
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
    float,
    Callable[[np.ndarray], object],
    np.ndarray | None,
    float | None,
    str,
]


def _real_injection_special_mask(image: np.ndarray) -> np.ndarray:
    """返回只用于分层选择的特殊值域掩膜。

    这里不把 ``-1`` 解释成坏点，也不把极端正负码解释成已经标定的
    满阱/溢出值；它们只是从原始数组中提取的可复核条件。真正的检测
    质量判定仍由 ``detect_sources`` 的原图测量负责。
    """

    values = np.asarray(image)
    numeric = np.asarray(values, dtype=np.float64)
    special = np.isfinite(numeric) & (numeric == -1.0)
    negative_limit = _default_negative_overflow_limit(values)
    if negative_limit is not None:
        special |= np.isfinite(numeric) & (numeric <= float(negative_limit))
    if np.issubdtype(values.dtype, np.integer):
        dtype_info = np.iinfo(values.dtype)
        special |= np.isfinite(numeric) & (numeric >= float(dtype_info.max - 32))
    return special


def _real_injection_site(
    image: np.ndarray,
    x: float,
    y: float,
    *,
    baseline_tree: cKDTree | None,
    special_mask: np.ndarray,
    neighborhood_radius: int,
) -> _RealInjectionSite | None:
    """计算注入位置的原始背景、特殊值域和基线邻域描述。"""

    stats = _local_patch_stats(image, x, y, radius=5)
    if stats is None:
        return None
    local_background, local_noise, upper_excess = stats
    height, width = image.shape
    center_x = int(round(x))
    center_y = int(round(y))
    y0 = max(0, center_y - neighborhood_radius)
    y1 = min(height, center_y + neighborhood_radius + 1)
    x0 = max(0, center_x - neighborhood_radius)
    x1 = min(width, center_x + neighborhood_radius + 1)
    local_special = special_mask[y0:y1, x0:x1]
    special_fraction = float(np.mean(local_special)) if local_special.size else 0.0
    if baseline_tree is None:
        neighbor_count = 0
        nearest_distance: float | None = None
    else:
        neighbor_indices = baseline_tree.query_ball_point((x, y), r=float(neighborhood_radius))
        neighbor_count = len(neighbor_indices)
        nearest = baseline_tree.query((x, y), distance_upper_bound=float(neighborhood_radius))
        nearest_distance = float(nearest[0]) if np.isfinite(nearest[0]) else None
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


def _greedy_real_injection_sites(
    sites: Sequence[_RealInjectionSite],
    count: int,
    *,
    minimum_distance_px: float,
) -> list[_RealInjectionSite]:
    """按给定顺序选取彼此分离的位置，失败时不偷偷降低距离。"""

    selected: list[_RealInjectionSite] = []
    for site in sites:
        if all(
            math.hypot(site.x - other.x, site.y - other.y) >= minimum_distance_px
            for other in selected
        ):
            selected.append(site)
            if len(selected) >= count:
                return selected
    return selected


def _select_stratified_real_injection_sites(
    image: np.ndarray,
    existing_sources: Sequence[Detection],
    rng: np.random.Generator,
    count: int,
    *,
    stratum: str,
    noise_adu: float,
    psf_fwhm: float,
    aperture_radius: int,
) -> tuple[_RealInjectionSite, ...]:
    """从真实 FITS 中选取一个明确的注入条件层。

    ``blank``/``high_background`` 是远离基线候选的控制层；``edge``、
    ``crowded``、``special_code`` 和 ``line`` 则故意保留相应条件，用来
    测量质量规则在困难区域的代价。任何一层样本不足都显式报错，避免
    把另一类背景悄悄冒充成该层。
    """

    if stratum not in _REAL_INJECTION_STRATUM_LABELS:
        allowed = ", ".join(sorted(_REAL_INJECTION_STRATUM_LABELS))
        raise ValueError(f"unknown real-background injection stratum {stratum!r}; choose from {allowed}")
    if count < 1:
        raise ValueError("count must be positive")
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    height, width = values.shape
    neighborhood_radius = max(8, int(math.ceil(2.0 * float(psf_fwhm))))
    recovery_radius = max(3.0, float(psf_fwhm))
    minimum_distance = max(10.0, 2.5 * float(psf_fwhm))
    interior_margin = max(
        neighborhood_radius + 2,
        int(math.ceil(4.0 * float(psf_fwhm) / 2.354820045)) + 2,
    )
    special_mask = _real_injection_special_mask(values)
    baseline_points = np.asarray([(source.x, source.y) for source in existing_sources], dtype=np.float64)
    baseline_tree = cKDTree(baseline_points) if baseline_points.size else None

    def is_inside(x: float, y: float, margin: int = 0) -> bool:
        return margin <= x < width - margin and margin <= y < height - margin

    def is_far_from_baseline(site: _RealInjectionSite, radius: float) -> bool:
        return site.nearest_baseline_source_px is None or site.nearest_baseline_source_px >= radius

    if stratum in {"blank", "high_background"}:
        pool: list[_RealInjectionSite] = []
        target_pool = max(256, count * 80)
        attempts_limit = max(20_000, target_pool * 160)
        for _attempt in range(attempts_limit):
            if len(pool) >= target_pool:
                break
            x = float(rng.uniform(interior_margin, width - interior_margin))
            y = float(rng.uniform(interior_margin, height - interior_margin))
            if baseline_tree is not None:
                distance = baseline_tree.query((x, y), distance_upper_bound=float(neighborhood_radius))[0]
                if np.isfinite(distance):
                    continue
            site = _real_injection_site(
                values,
                x,
                y,
                baseline_tree=baseline_tree,
                special_mask=special_mask,
                neighborhood_radius=neighborhood_radius,
            )
            if site is None or site.special_pixel_fraction > 0.0:
                continue
            # 只排除明显局部亮结构；保留背景噪声分布，才能把安静/高噪声
            # 层分开，而不是先把困难背景裁掉。
            allowed_excess = max(4.0 * site.local_noise_adu, 4.0 * float(noise_adu))
            if site.upper_excess_adu > allowed_excess:
                continue
            pool.append(site)
        if len(pool) < max(count, 8):
            raise RuntimeError(
                f"真实背景 {stratum} 层候选不足：仅 {len(pool)} 个，至少需要 {max(count, 8)} 个；"
                "请减少注入数量或更换背景分层参数"
            )
        noise_values = np.asarray([site.local_noise_adu for site in pool], dtype=np.float64)
        if stratum == "blank":
            noise_cut = float(np.quantile(noise_values, 0.50))
            ordered = sorted(
                (site for site in pool if site.local_noise_adu <= noise_cut),
                key=lambda site: (site.local_noise_adu, site.upper_excess_adu, site.y, site.x),
            )
        else:
            noise_cut = float(np.quantile(noise_values, 0.75))
            ordered = sorted(
                (site for site in pool if site.local_noise_adu >= noise_cut),
                key=lambda site: (-site.local_noise_adu, -site.upper_excess_adu, site.y, site.x),
            )
        selected = _greedy_real_injection_sites(ordered, count, minimum_distance_px=minimum_distance)
    elif stratum == "edge":
        edge_band = max(aperture_radius + 4, int(math.ceil(2.0 * float(psf_fwhm))))
        candidates: list[_RealInjectionSite] = []
        attempts_limit = max(20_000, count * 2_000)
        for _attempt in range(attempts_limit):
            side = int(rng.integers(0, 2))
            x = (
                float(rng.uniform(1.5, max(1.6, edge_band - 0.25)))
                if side == 0
                else float(rng.uniform(width - edge_band + 0.25, width - 1.5))
            )
            y = float(rng.uniform(max(12, edge_band + 2), height - edge_band - 2))
            if not is_inside(x, y, margin=1):
                continue
            site = _real_injection_site(
                values,
                x,
                y,
                baseline_tree=baseline_tree,
                special_mask=special_mask,
                neighborhood_radius=neighborhood_radius,
            )
            if site is None or not is_far_from_baseline(site, neighborhood_radius):
                continue
            candidates.append(site)
        selected = _greedy_real_injection_sites(candidates, count, minimum_distance_px=minimum_distance)
    elif stratum == "crowded":
        eligible = [
            source
            for source in existing_sources
            if source.quality_passed
            and is_inside(source.x, source.y, margin=interior_margin)
            and not any(flag in source.flags for flag in ("EDGE", "MASKED", "SATURATED", "LINE_ARTIFACT"))
        ]
        eligible.sort(
            key=lambda source: float(source.flux_snr if source.flux_snr is not None else source.snr),
            reverse=True,
        )
        eligible = eligible[: max(256, count * 32)]
        candidates = []
        min_neighbor_distance = max(recovery_radius + 1.0, 1.4 * float(psf_fwhm))
        max_neighbor_distance = max(min_neighbor_distance + 1.0, 3.5 * float(psf_fwhm))
        attempts_limit = max(20_000, count * 3_000)
        for _attempt in range(attempts_limit):
            if not eligible:
                break
            anchor = eligible[int(rng.integers(0, len(eligible)))]
            distance = float(rng.uniform(min_neighbor_distance, max_neighbor_distance))
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            x = float(anchor.x + distance * math.cos(angle))
            y = float(anchor.y + distance * math.sin(angle))
            if not is_inside(x, y, margin=interior_margin):
                continue
            site = _real_injection_site(
                values,
                x,
                y,
                baseline_tree=baseline_tree,
                special_mask=special_mask,
                neighborhood_radius=neighborhood_radius,
            )
            if site is None or site.baseline_neighbor_count < 1:
                continue
            if site.nearest_baseline_source_px is not None and site.nearest_baseline_source_px <= recovery_radius:
                continue
            candidates.append(site)
        selected = _greedy_real_injection_sites(candidates, count, minimum_distance_px=minimum_distance)
    else:
        if stratum == "special_code":
            candidate_mask = ndimage.binary_dilation(
                special_mask,
                iterations=max(1, int(math.ceil(float(psf_fwhm)))),
            )
            candidate_coordinates = np.column_stack(np.nonzero(candidate_mask))
            if candidate_coordinates.size:
                order = rng.permutation(len(candidate_coordinates))
                candidates = []
                for index in order:
                    y, x = (int(value) for value in candidate_coordinates[index])
                    if not is_inside(float(x), float(y), margin=interior_margin):
                        continue
                    site = _real_injection_site(
                        values,
                        float(x),
                        float(y),
                        baseline_tree=baseline_tree,
                        special_mask=special_mask,
                        neighborhood_radius=neighborhood_radius,
                    )
                    if site is None or site.special_pixel_fraction <= 0.0:
                        continue
                    candidates.append(site)
                selected = _greedy_real_injection_sites(candidates, count, minimum_distance_px=minimum_distance)
            else:
                selected = []
        else:  # line
            line_sources = [source for source in existing_sources if "LINE_ARTIFACT" in source.flags]
            candidates = []
            attempts_limit = max(20_000, count * 3_000)
            for _attempt in range(attempts_limit):
                if not line_sources:
                    break
                anchor = line_sources[int(rng.integers(0, len(line_sources)))]
                distance = float(
                    rng.uniform(
                        max(recovery_radius + 1.0, 1.4 * float(psf_fwhm)),
                        max(3.0, 3.0 * float(psf_fwhm)),
                    )
                )
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                x = float(anchor.x + distance * math.cos(angle))
                y = float(anchor.y + distance * math.sin(angle))
                if not is_inside(x, y, margin=interior_margin):
                    continue
                site = _real_injection_site(
                    values,
                    x,
                    y,
                    baseline_tree=baseline_tree,
                    special_mask=special_mask,
                    neighborhood_radius=neighborhood_radius,
                )
                if (
                    site is None
                    or site.baseline_neighbor_count < 1
                    or (
                        site.nearest_baseline_source_px is not None
                        and site.nearest_baseline_source_px <= recovery_radius
                    )
                ):
                    continue
                candidates.append(site)
            selected = _greedy_real_injection_sites(candidates, count, minimum_distance_px=minimum_distance)

    if len(selected) != count:
        raise RuntimeError(
            f"真实背景 {stratum} 层无法得到 {count} 个互相分离的位置，仅找到 {len(selected)} 个；"
            "该层未执行，避免静默改变实验条件"
        )
    return tuple(selected)


def run_feature_audit(
    *,
    image_shape: tuple[int, int] = (128, 128),
    background_adu: float = 21.0,
    noise_sigma_adu: float = 5.0,
    psf_fwhm: float = 2.0,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    background_box_size: int = 32,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
    seed: int = 19019,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[FeatureAuditRow, ...]:
    """对孤立源、弱源、近邻和典型伪影做可复现的分层控制实验。

    正样本场景有已知的 Gaussian 源坐标，报告候选/质量层回收率；尖峰、
    长线等阴性场景没有注入恒星，报告特征邻域中有多少候选或质量源泄漏。
    这些图像只用于检查算法机制，不代表真实 FITS 的伪影比例，也不把
    ``quality_recall`` 解释成真实恒星完备率。
    """

    height, width = image_shape
    if height < 48 or width < 48:
        raise ValueError("feature audit image_shape must be at least 48x48")
    if background_adu <= 0 or noise_sigma_adu <= 0 or psf_fwhm <= 0:
        raise ValueError("feature audit background/noise/psf parameters must be positive")
    if threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1:
        raise ValueError("feature audit detection parameters must be positive")
    if background_box_size < 16 or min_flux_snr <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("feature audit background/quality parameters are invalid")

    center_x, center_y = width / 2.0, height / 2.0
    line_positions = tuple((64.0 if width > 80 else center_x, float(y)) for y in range(18, height - 18))

    def add_single_pixel_spike(image: np.ndarray) -> None:
        y = int(round(center_y))
        x = int(round(center_x))
        image[y, x] += 400.0

    def add_long_line(image: np.ndarray) -> None:
        for x, y in line_positions:
            pixel_y = int(round(y))
            pixel_x = int(round(x))
            image[pixel_y, pixel_x] += 80.0
            image[pixel_y, pixel_x + 1] += 50.0

    scenario_specs: tuple[FeatureScenario, ...] = (
        (
            "compact_quality",
            "isolated_fwhm2",
            "positive",
            ((center_x, center_y),),
            ((center_x, center_y),),
            5.0,
            lambda image: _gaussian_source(image, center_x, center_y, 128.0, 2.0),
            None,
            None,
            "已知孤立 Gaussian PSF；检查标准紧凑源回收",
        ),
        (
            "weak_or_background",
            "weak_fwhm2_peak28",
            "positive",
            ((center_x, center_y),),
            ((center_x, center_y),),
            5.0,
            lambda image: _gaussian_source(image, center_x, center_y, 28.0, 2.0),
            None,
            None,
            "已知弱 Gaussian PSF；检查候选层与质量层的差距",
        ),
        (
            "shape_outlier",
            "broad_fwhm6",
            "positive",
            ((center_x, center_y),),
            ((center_x, center_y),),
            5.0,
            lambda image: _gaussian_source(image, center_x, center_y, 128.0, 6.0),
            None,
            None,
            "已知宽 PSF；不预设它一定是假源，观察模型失配代价",
        ),
        (
            "spike_or_support",
            "narrow_fwhm0.8",
            "positive",
            ((center_x, center_y),),
            ((center_x, center_y),),
            5.0,
            lambda image: _gaussian_source(image, center_x, center_y, 160.0, 0.8),
            None,
            None,
            "已知欠采样窄源；检查尖峰/PSF 支持审计是否生效",
        ),
        (
            "crowded_blend",
            "pair_separation3",
            "positive",
            ((center_x - 1.5, center_y), (center_x + 1.5, center_y)),
            ((center_x - 1.5, center_y), (center_x + 1.5, center_y)),
            5.0,
            lambda image: (
                _gaussian_source(image, center_x - 1.5, center_y, 128.0, 2.0),
                _gaussian_source(image, center_x + 1.5, center_y, 128.0, 2.0),
            ),
            None,
            None,
            "已知 3 px 双源；低于/接近分辨尺度，允许合并结果",
        ),
        (
            "crowded_blend",
            "pair_separation6",
            "positive",
            ((center_x - 3.0, center_y), (center_x + 3.0, center_y)),
            ((center_x - 3.0, center_y), (center_x + 3.0, center_y)),
            5.0,
            lambda image: (
                _gaussian_source(image, center_x - 3.0, center_y, 128.0, 2.0),
                _gaussian_source(image, center_x + 3.0, center_y, 128.0, 2.0),
            ),
            None,
            None,
            "已知 6 px 双源；与当前双源细筛证据线对照",
        ),
        (
            "spike_or_support",
            "single_pixel_spike",
            "negative",
            (),
            ((center_x, center_y),),
            5.0,
            add_single_pixel_spike,
            None,
            None,
            "阴性样本：只有单像素尖峰，没有注入恒星",
        ),
        (
            "masked_or_edge",
            "masked_source",
            "positive",
            ((center_x, center_y),),
            ((center_x, center_y),),
            5.0,
            lambda image: _gaussian_source(image, center_x, center_y, 128.0, 2.0),
            np.zeros(image_shape, dtype=bool),
            None,
            "已知 PSF 的孔径外圈叠加显式掩膜；检查有效性拒绝",
        ),
        (
            "masked_or_edge",
            "edge_source",
            "positive",
            ((3.0, center_y),),
            ((3.0, center_y),),
            5.0,
            lambda image: _gaussian_source(image, 3.0, center_y, 128.0, 2.0),
            None,
            None,
            "已知靠近边界的 PSF；检查不完整孔径不被当作普通星",
        ),
        (
            "range_anomaly",
            "saturated_source",
            "positive",
            ((center_x, center_y),),
            ((center_x, center_y),),
            5.0,
            lambda image: _gaussian_source(image, center_x, center_y, 800.0, 2.0),
            None,
            250.0,
            "已知高幅度 PSF，检测器显式设置饱和线；检查范围审计",
        ),
        (
            "linear_artifact",
            "long_line_negative",
            "negative",
            (),
            line_positions,
            4.0,
            add_long_line,
            None,
            None,
            "阴性样本：长线残差，不注入恒星；检查线状源泄漏",
        ),
    )

    rows: list[FeatureAuditRow] = []
    detector_total = len(scenario_specs)
    for scenario_index, (
        feature_class,
        scenario,
        control_type,
        truth_positions,
        feature_positions,
        nearby_radius,
        builder,
        external_mask,
        saturation_level,
        note,
    ) in enumerate(scenario_specs):
        if progress is not None:
            progress(scenario_index + 1, detector_total)
        rng = np.random.default_rng(int(seed) + scenario_index * 1009)
        image = rng.normal(background_adu, noise_sigma_adu, image_shape).astype(np.float32)
        builder(image)
        scenario_mask = auxiliary_mask(image_shape)
        if external_mask is not None:
            external_mask = np.asarray(external_mask, dtype=bool).copy()
            external_mask[int(round(center_y)), int(round(center_x + 3.0))] = True
            scenario_mask |= external_mask
        detection = detect_sources(
            image,
            mask=scenario_mask,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            saturation_level=saturation_level,
            reject_linear_artifacts=True,
            proposal_mode=proposal_mode,
        )
        nearby_sources = _nearby_detections(detection.sources, feature_positions, nearby_radius)
        nearby_flags = Counter(flag for source in nearby_sources for flag in source.flags)
        flag_text = "|".join(
            f"{flag}:{count}" for flag, count in sorted(nearby_flags.items(), key=lambda item: (-item[1], item[0]))
        )
        candidate_hits = _matched_count(truth_positions, detection.sources, max(3.0, float(psf_fwhm)))
        quality_hits = _matched_count(truth_positions, detection.quality_sources, max(3.0, float(psf_fwhm)))
        truth_count = len(truth_positions)
        rows.append(
            FeatureAuditRow(
                feature_class=feature_class,
                scenario=scenario,
                control_type=control_type,
                truth_count=truth_count,
                candidate_true_hits=candidate_hits,
                quality_true_hits=quality_hits,
                candidate_recall=candidate_hits / truth_count if truth_count else None,
                quality_recall=quality_hits / truth_count if truth_count else None,
                nearby_candidate_count=len(nearby_sources),
                nearby_quality_count=sum(source.quality_passed for source in nearby_sources),
                candidate_count=detection.candidate_count,
                quality_count=detection.star_count,
                nearby_flags=flag_text,
                note=note,
            )
        )
    return tuple(rows)


def run_sequence_feature_persistence(
    paths: Iterable[str | Path],
    *,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 2.0,
    background_box_size: int = 128,
    background_sample_limit: int = 100_000,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
    max_sources: int | None = None,
    registration_radius_px: float = 8.0,
    association_radius_px: float = 1.0,
    required_presence: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SequenceFeatureAuditResult:
    """审计首帧各类候选在真实序列中的跨帧响应持久性。

    每一帧都先按同一套单帧规则检测，使用通过质量层的源估计全局平移，
    再把后续帧的候选峰和质量源变换到首帧坐标。输出的 ``presence`` 是
    固定小邻域内的响应次数，不是已经完成一对一身份确认的轨迹；在密集
    星场中，邻近源仍可能造成偶然命中。因此该函数把候选持久性和质量源
    持久性分开，并把 ``max_sources`` 是否截断写入结果语义。

    首帧的 ``feature_class`` 是按当前旗标优先级生成的算法审计标签：它
    描述首帧为什么被保留/拒绝，不是物理真值标签。默认 ``max_sources=None``
    保留每帧完整返回源，适合论文实验；若为减少运行时间传入上限，结果
    只能解释为该返回工作集内的持久性，不能与全量质量率直接比较。
    """

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    if threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1 or psf_fwhm <= 0:
        raise ValueError("sequence feature detector parameters must be positive")
    if background_box_size < 16 or background_sample_limit < 1 or min_flux_snr <= 0:
        raise ValueError("sequence feature background/quality parameters are invalid")
    if not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_psf_support_pixels must be within 1..9")
    if max_sources is not None and max_sources < 1:
        raise ValueError("max_sources must be positive or None")
    if registration_radius_px <= 0 or association_radius_px <= 0:
        raise ValueError("registration_radius_px and association_radius_px must be positive")

    frame_count = len(frame_paths)
    resolved_required_presence = (
        min(frame_count, max(1, int(np.ceil(frame_count * 0.8))))
        if required_presence is None
        else int(required_presence)
    )
    if not 1 <= resolved_required_presence <= frame_count:
        raise ValueError("required_presence must be within the frame count")

    def source_points(sources: Sequence[Detection], *, peak: bool) -> np.ndarray:
        if peak:
            values = [
                (
                    float(source.peak_x if source.peak_x is not None else source.x),
                    float(source.peak_y if source.peak_y is not None else source.y),
                )
                for source in sources
            ]
        else:
            values = [(float(source.x), float(source.y)) for source in sources]
        return np.asarray(values, dtype=np.float32).reshape((-1, 2))

    candidate_frames: list[np.ndarray] = []
    quality_frames: list[np.ndarray] = []
    candidate_class_frames: list[np.ndarray] = []
    quality_class_frames: list[np.ndarray] = []
    frame_rows: list[SequenceFeatureFrameRow] = []
    cumulative_shifts: list[tuple[float, float]] = [(0.0, 0.0)]
    previous_quality: tuple[Detection, ...] | None = None
    anchor_peak_points: np.ndarray | None = None
    anchor_centroid_points: np.ndarray | None = None
    anchor_classes: np.ndarray | None = None
    anchor_quality: np.ndarray | None = None

    for frame_index, path in enumerate(frame_paths):
        analysis = analyze_frame(
            path,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            background_sample_limit=background_sample_limit,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            max_sources=max_sources,
            reject_linear_artifacts=True,
            proposal_mode=proposal_mode,
            refine_local_background=True,
        )
        sources = analysis.detection.sources
        quality_sources = analysis.detection.quality_sources
        if previous_quality is not None:
            delta_x, delta_y = _estimate_translation(
                previous_quality,
                quality_sources,
                radius_px=registration_radius_px,
            )
            previous_shift = cumulative_shifts[-1]
            cumulative_shifts.append(
                (previous_shift[0] + float(delta_x), previous_shift[1] + float(delta_y))
            )
        shift_x, shift_y = cumulative_shifts[-1]
        candidate_points = source_points(sources, peak=True)
        centroid_points = source_points(sources, peak=False)
        quality_points = source_points(quality_sources, peak=False)
        candidate_points -= np.asarray((shift_x, shift_y), dtype=np.float32)
        centroid_points -= np.asarray((shift_x, shift_y), dtype=np.float32)
        quality_points -= np.asarray((shift_x, shift_y), dtype=np.float32)
        candidate_classes = np.asarray(
            [classify_source_feature(source) for source in sources],
            dtype=object,
        )
        quality_classes = np.asarray(
            [classify_source_feature(source) for source in quality_sources],
            dtype=object,
        )
        candidate_frames.append(candidate_points)
        quality_frames.append(quality_points)
        candidate_class_frames.append(candidate_classes)
        quality_class_frames.append(quality_classes)

        feature_summary = summarize_source_features(sources)
        for summary in feature_summary:
            frame_rows.append(
                SequenceFeatureFrameRow(
                    frame_index=frame_index + 1,
                    path=str(path),
                    candidate_count=analysis.detection.candidate_count,
                    returned_count=analysis.detection.returned_count,
                    quality_count=analysis.detection.star_count,
                    background_adu=float(analysis.detection.background),
                    noise_adu=float(analysis.detection.noise),
                    feature_class=str(summary["feature_class"]),
                    feature_class_label=str(summary["feature_class_label"]),
                    feature_candidate_count=int(summary["candidate_count"]),
                    feature_quality_count=int(summary["quality_count"]),
                )
            )

        if frame_index == 0:
            anchor_peak_points = candidate_points.copy()
            anchor_centroid_points = centroid_points.copy()
            anchor_classes = np.asarray(
                candidate_classes,
                dtype=object,
            )
            anchor_quality = np.asarray(
                [bool(source.quality_passed) for source in sources],
                dtype=bool,
            )
        previous_quality = quality_sources
        if progress is not None:
            progress(frame_index + 1, frame_count)
        del analysis, sources, quality_sources

    if (
        anchor_peak_points is None
        or anchor_centroid_points is None
        or anchor_classes is None
        or anchor_quality is None
    ):
        raise RuntimeError("sequence feature audit did not produce an anchor frame")

    def reciprocal_match_pairs(
        anchor_points: np.ndarray,
        frames: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        """返回每帧互相最近一对一匹配的 ``(anchor_index, frame_index)``。"""

        empty = np.empty((0, 2), dtype=np.intp)
        if anchor_points.size == 0:
            return tuple(empty.copy() for _points in frames)
        anchor_tree = cKDTree(anchor_points)
        matches: list[np.ndarray] = []
        for points in frames:
            if points.size == 0:
                matches.append(empty.copy())
                continue
            current_tree = cKDTree(points)
            anchor_distances, current_indices = current_tree.query(
                anchor_points,
                distance_upper_bound=association_radius_px,
            )
            current_distances, anchor_indices = anchor_tree.query(
                points,
                distance_upper_bound=association_radius_px,
            )
            valid = np.isfinite(anchor_distances) & (current_indices < len(points))
            anchor_positions = np.flatnonzero(valid)
            if not anchor_positions.size:
                matches.append(empty.copy())
                continue
            current_positions = np.asarray(current_indices[valid], dtype=np.intp)
            reciprocal = (
                np.isfinite(current_distances[current_positions])
                & (anchor_indices[current_positions] < len(anchor_points))
                & (anchor_indices[current_positions] == anchor_positions)
            )
            matches.append(
                np.column_stack(
                    (
                        anchor_positions[reciprocal],
                        current_positions[reciprocal],
                    )
                ).astype(np.intp, copy=False)
            )
        return tuple(matches)

    def reciprocal_presence_counts(anchor_points: np.ndarray, frames: Sequence[np.ndarray]) -> np.ndarray:
        """按互相最近邻统计每帧的唯一邻域命中。"""

        counts = np.zeros(anchor_points.shape[0], dtype=np.int16)
        for pairs in reciprocal_match_pairs(anchor_points, frames):
            if pairs.size:
                counts[pairs[:, 0]] += 1
        return counts

    candidate_presence = reciprocal_presence_counts(anchor_peak_points, candidate_frames)
    quality_presence = reciprocal_presence_counts(anchor_centroid_points, quality_frames)
    label_by_class = dict(_SOURCE_FEATURE_CLASSES)
    persistence_rows: list[SequenceFeaturePersistenceRow] = []
    for feature_class, feature_label in _SOURCE_FEATURE_CLASSES:
        class_mask = anchor_classes == feature_class
        class_candidate_presence = candidate_presence[class_mask]
        class_quality_presence = quality_presence[class_mask]
        persistence_rows.append(
            SequenceFeaturePersistenceRow(
                feature_class=feature_class,
                feature_class_label=label_by_class[feature_class],
                anchor_candidate_count=int(np.count_nonzero(class_mask)),
                anchor_quality_count=int(np.count_nonzero(anchor_quality[class_mask])),
                frame_count=frame_count,
                required_presence=resolved_required_presence,
                association_radius_px=float(association_radius_px),
                candidate_median_presence=(
                    float(np.median(class_candidate_presence))
                    if class_candidate_presence.size
                    else None
                ),
                candidate_mean_presence=(
                    float(np.mean(class_candidate_presence))
                    if class_candidate_presence.size
                    else None
                ),
                candidate_presence_ge_required_count=int(
                    np.count_nonzero(class_candidate_presence >= resolved_required_presence)
                ),
                candidate_presence_all_frames_count=int(
                    np.count_nonzero(class_candidate_presence >= frame_count)
                ),
                quality_median_presence=(
                    float(np.median(class_quality_presence))
                    if class_quality_presence.size
                    else None
                ),
                quality_mean_presence=(
                    float(np.mean(class_quality_presence))
                    if class_quality_presence.size
                    else None
                ),
                quality_presence_ge_required_count=int(
                    np.count_nonzero(class_quality_presence >= resolved_required_presence)
                ),
                quality_presence_all_frames_count=int(
                    np.count_nonzero(class_quality_presence >= frame_count)
                ),
            )
        )

    candidate_pairs = reciprocal_match_pairs(anchor_peak_points, candidate_frames)
    quality_anchor_points = anchor_centroid_points[anchor_quality]
    quality_anchor_classes = anchor_classes[anchor_quality]
    quality_pairs = reciprocal_match_pairs(quality_anchor_points, quality_frames)
    transition_counts: Counter[tuple[str, str, str]] = Counter()
    for pairs, response_classes in zip(candidate_pairs, candidate_class_frames, strict=True):
        for anchor_index, response_index in pairs:
            transition_counts[
                (
                    "candidate",
                    str(anchor_classes[int(anchor_index)]),
                    str(response_classes[int(response_index)]),
                )
            ] += 1
    for pairs, response_classes in zip(quality_pairs, quality_class_frames, strict=True):
        for anchor_index, response_index in pairs:
            transition_counts[
                (
                    "quality",
                    str(quality_anchor_classes[int(anchor_index)]),
                    str(response_classes[int(response_index)]),
                )
            ] += 1

    class_transition_rows: list[SequenceFeatureClassTransitionRow] = []
    for layer, anchor_class_values in (
        ("candidate", anchor_classes),
        ("quality", quality_anchor_classes),
    ):
        for anchor_feature_class, anchor_feature_class_label in _SOURCE_FEATURE_CLASSES:
            anchor_count = int(np.count_nonzero(anchor_class_values == anchor_feature_class))
            possible_match_count = anchor_count * frame_count
            for response_feature_class, response_feature_class_label in _SOURCE_FEATURE_CLASSES:
                matched_count = int(
                    transition_counts[(layer, anchor_feature_class, response_feature_class)]
                )
                class_transition_rows.append(
                    SequenceFeatureClassTransitionRow(
                        layer=layer,
                        anchor_feature_class=anchor_feature_class,
                        anchor_feature_class_label=anchor_feature_class_label,
                        response_feature_class=response_feature_class,
                        response_feature_class_label=response_feature_class_label,
                        anchor_count=anchor_count,
                        frame_count=frame_count,
                        possible_match_count=possible_match_count,
                        matched_count=matched_count,
                        match_rate=(
                            matched_count / possible_match_count
                            if possible_match_count > 0
                            else None
                        ),
                        mean_matches_per_anchor=(
                            matched_count / anchor_count if anchor_count > 0 else None
                        ),
                    )
                )

    candidate_mode = (
        "all returned source measurements"
        if max_sources is None
        else f"returned source measurements capped at max_sources={max_sources}"
    )
    quality_mode = (
        "all returned quality sources"
        if max_sources is None
        else f"quality sources within max_sources={max_sources} working set"
    )
    return SequenceFeatureAuditResult(
        frame_count=frame_count,
        association_radius_px=float(association_radius_px),
        required_presence=resolved_required_presence,
        association_method="reciprocal_nearest_one_to_one",
        candidate_source_mode=candidate_mode,
        quality_source_mode=quality_mode,
        cumulative_shifts=tuple(cumulative_shifts),
        frame_rows=tuple(frame_rows),
        persistence_rows=tuple(persistence_rows),
        class_transition_rows=tuple(class_transition_rows),
    )


def summarize_sequence_feature_temporal_profiles(
    frame_rows: Sequence[SequenceFeatureFrameRow],
) -> tuple[SequenceFeatureTemporalProfile, ...]:
    """汇总每类在全序列中的计数稳定性和质量层渗透率。

    该表与首帧锚点持久性不同：它不回答某个首帧候选是否被逐帧
    邻域命中，而回答某个算法类别在每帧占多少候选、数量是否稳定、
    以及每帧有多少比例进入质量层。由于 ``compact_quality`` 本身是
    按 ``quality_passed=True`` 定义的，质量通过率有一部分是分类定义
    的账本属性，不是独立验证；整个输出仍是 detector-level 统计，不能
    替代源身份、星表匹配或官方真值。
    """

    if not frame_rows:
        return ()
    frame_ids = tuple(sorted({int(row.frame_index) for row in frame_rows}))
    if any(frame_id < 1 for frame_id in frame_ids):
        raise ValueError("frame_index must be positive")
    by_key: dict[tuple[int, str], SequenceFeatureFrameRow] = {}
    for row in frame_rows:
        key = (int(row.frame_index), str(row.feature_class))
        if key in by_key:
            raise ValueError(f"duplicate frame/category row: {key}")
        by_key[key] = row
    total_candidates: dict[int, int] = {}
    for frame_id in frame_ids:
        candidates = {
            int(row.candidate_count)
            for (row_frame, _feature_class), row in by_key.items()
            if row_frame == frame_id
        }
        if len(candidates) != 1:
            raise ValueError(f"frame {frame_id} has inconsistent candidate_count")
        total_candidates[frame_id] = candidates.pop()

    def coefficient_of_variation(values: Sequence[float]) -> float | None:
        mean = float(np.mean(values)) if values else 0.0
        if mean <= 0.0:
            return None
        return float(np.std(np.asarray(values, dtype=np.float64), ddof=0) / mean)

    labels = dict(_SOURCE_FEATURE_CLASSES)
    profiles: list[SequenceFeatureTemporalProfile] = []
    for feature_class in labels:
        members = [
            by_key[(frame_id, feature_class)]
            for frame_id in frame_ids
            if (frame_id, feature_class) in by_key
        ]
        if len(members) != len(frame_ids):
            raise ValueError(f"missing frame/category rows for {feature_class}")
        candidate_counts = [int(row.feature_candidate_count) for row in members]
        quality_counts = [int(row.feature_quality_count) for row in members]
        candidate_fractions = [
            count / total_candidates[int(row.frame_index)]
            if total_candidates[int(row.frame_index)] > 0
            else 0.0
            for row, count in zip(members, candidate_counts, strict=True)
        ]
        quality_fractions = [
            quality / candidate if candidate > 0 else 0.0
            for quality, candidate in zip(quality_counts, candidate_counts, strict=True)
        ]
        total_candidate_count = sum(candidate_counts)
        total_quality_count = sum(quality_counts)
        profiles.append(
            SequenceFeatureTemporalProfile(
                feature_class=feature_class,
                feature_class_label=labels[feature_class],
                frame_count=len(frame_ids),
                active_frame_count=sum(count > 0 for count in candidate_counts),
                quality_active_frame_count=sum(count > 0 for count in quality_counts),
                candidate_count_mean=float(np.mean(candidate_counts)),
                candidate_count_min=min(candidate_counts),
                candidate_count_max=max(candidate_counts),
                candidate_count_cv=coefficient_of_variation(candidate_counts),
                candidate_fraction_mean=float(np.mean(candidate_fractions)),
                candidate_fraction_cv=coefficient_of_variation(candidate_fractions),
                quality_count_mean=float(np.mean(quality_counts)),
                quality_count_min=min(quality_counts),
                quality_count_max=max(quality_counts),
                quality_count_cv=coefficient_of_variation(quality_counts),
                quality_fraction_mean=float(np.mean(quality_fractions)),
                quality_fraction_weighted=(
                    total_quality_count / total_candidate_count if total_candidate_count > 0 else None
                ),
            )
        )
    return tuple(profiles)


def run_feature_cross_audit(
    source_summary_path: str | Path,
    persistence_path: str | Path,
    *,
    high_snr_threshold: float = 10.0,
    class_transition_path: str | Path | None = None,
) -> FeatureCrossAuditResult:
    """把单帧特征汇总与 15 帧持久性汇总连接成可复现的反例审计。

    ``source_summary_path`` 通常来自 ``rst19-sources`` 的
    ``source_feature_summary.csv``，``persistence_path`` 来自
    ``rst19-feature-sequence`` 的 ``sequence_feature_persistence.csv``。
    如果提供 ``class_transition_path``，还会连接
    ``sequence_feature_class_transition.csv``，把“任意类别响应”和“同类响应”
    追加到同一张交叉表。该函数只做 CSV 的确定性连接，不重新检测 FITS，
    也不把“质量邻域命中”解释成星表身份。这样可以把论文里的“高 SNR
    但被拒绝”“候选重复但质量不重复”和“邻域响应发生类别切换”从手工
    整理变成可复现产物。
    """

    source_path = Path(source_summary_path)
    persistence_file = Path(persistence_path)
    if high_snr_threshold <= 0:
        raise ValueError("high_snr_threshold must be positive")

    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"CSV has no rows: {path}")
        return rows

    source_rows = read_rows(source_path)
    persistence_rows = read_rows(persistence_file)

    def keyed(rows: Sequence[Mapping[str, str]], path: Path) -> dict[str, Mapping[str, str]]:
        result: dict[str, Mapping[str, str]] = {}
        for row in rows:
            feature_class = str(row.get("feature_class", "")).strip()
            if not feature_class:
                raise ValueError(f"CSV row has empty feature_class: {path}")
            if feature_class in result:
                raise ValueError(f"CSV has duplicate feature_class={feature_class}: {path}")
            result[feature_class] = row
        return result

    source_by_class = keyed(source_rows, source_path)
    persistence_by_class = keyed(persistence_rows, persistence_file)
    if set(source_by_class) != set(persistence_by_class):
        missing_in_persistence = sorted(set(source_by_class) - set(persistence_by_class))
        missing_in_source = sorted(set(persistence_by_class) - set(source_by_class))
        raise ValueError(
            "feature classes do not match: "
            f"missing_in_persistence={missing_in_persistence}, "
            f"missing_in_source={missing_in_source}"
        )

    def optional_float(value: object, field: str, feature_class: str) -> float | None:
        text = "" if value is None else str(value).strip()
        if not text:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {field} for {feature_class}: {value!r}") from exc
        if not np.isfinite(number):
            return None
        return number

    def required_int(value: object, field: str, feature_class: str) -> int:
        number = optional_float(value, field, feature_class)
        if number is None or not float(number).is_integer():
            raise ValueError(f"invalid integer {field} for {feature_class}: {value!r}")
        return int(number)

    first_persistence = next(iter(persistence_by_class.values()))
    frame_count = required_int(first_persistence.get("frame_count"), "frame_count", "<header>")
    required_presence = required_int(
        first_persistence.get("required_presence"),
        "required_presence",
        "<header>",
    )
    if frame_count < 1 or not 1 <= required_presence <= frame_count:
        raise ValueError("persistence frame_count/required_presence is invalid")

    source_anchor_counts: dict[str, tuple[int, int]] = {}
    for source_row in source_rows:
        feature_class = str(source_row["feature_class"]).strip()
        if feature_class in source_anchor_counts:
            raise ValueError(f"duplicate feature_class in source summary: {feature_class}")
        source_anchor_counts[feature_class] = (
            required_int(source_row.get("candidate_count"), "candidate_count", feature_class),
            required_int(source_row.get("quality_count"), "quality_count", feature_class),
        )

    transition_file = Path(class_transition_path) if class_transition_path is not None else None
    transition_metrics: dict[tuple[str, str, str], tuple[int, int, str]] = {}
    if transition_file is not None:
        transition_rows = read_rows(transition_file)
        expected_classes = set(source_by_class)
        for transition_row in transition_rows:
            layer = str(transition_row.get("layer", "")).strip()
            anchor_class = str(transition_row.get("anchor_feature_class", "")).strip()
            response_class = str(transition_row.get("response_feature_class", "")).strip()
            if layer not in {"candidate", "quality"}:
                raise ValueError(f"invalid transition layer: {layer!r}")
            if anchor_class not in expected_classes or response_class not in expected_classes:
                raise ValueError(
                    "transition feature class is not present in source summary: "
                    f"anchor={anchor_class!r}, response={response_class!r}"
                )
            key = (layer, anchor_class, response_class)
            if key in transition_metrics:
                raise ValueError(f"duplicate class transition row: {key}")
            frame_value = required_int(transition_row.get("frame_count"), "frame_count", str(key))
            if frame_value != frame_count:
                raise ValueError(
                    f"transition frame_count={frame_value} does not match persistence frame_count={frame_count}"
                )
            expected_anchor_count = source_anchor_counts[anchor_class][0 if layer == "candidate" else 1]
            anchor_count = required_int(transition_row.get("anchor_count"), "anchor_count", str(key))
            possible_match_count = required_int(
                transition_row.get("possible_match_count"), "possible_match_count", str(key)
            )
            matched_count = required_int(transition_row.get("matched_count"), "matched_count", str(key))
            if anchor_count != expected_anchor_count:
                raise ValueError(
                    f"transition anchor_count={anchor_count} does not match source summary "
                    f"{layer} anchor_count={expected_anchor_count} for {anchor_class}"
                )
            expected_possible = anchor_count * frame_count
            if possible_match_count != expected_possible:
                raise ValueError(
                    f"transition possible_match_count={possible_match_count} does not equal "
                    f"anchor_count*frame_count={expected_possible} for {key}"
                )
            if matched_count < 0 or matched_count > possible_match_count:
                raise ValueError(f"transition matched_count is outside its denominator for {key}")
            response_label = str(
                transition_row.get("response_feature_class_label", response_class)
            ).strip() or response_class
            transition_metrics[key] = (matched_count, possible_match_count, response_label)

        expected_keys = {
            (layer, anchor_class, response_class)
            for layer in ("candidate", "quality")
            for anchor_class in expected_classes
            for response_class in expected_classes
        }
        actual_keys = set(transition_metrics)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                "class transition matrix is incomplete or has unexpected rows: "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        for layer in ("candidate", "quality"):
            for anchor_class in expected_classes:
                possible = source_anchor_counts[anchor_class][0 if layer == "candidate" else 1] * frame_count
                matched = sum(
                    transition_metrics[(layer, anchor_class, response_class)][0]
                    for response_class in expected_classes
                )
                if matched > possible:
                    raise ValueError(
                        f"class transition rows exceed one-to-one denominator for "
                        f"layer={layer}, anchor={anchor_class}"
                    )

    def transition_fraction(layer: str, anchor_class: str, response_class: str) -> float | None:
        metric = transition_metrics.get((layer, anchor_class, response_class))
        if metric is None or metric[1] <= 0:
            return None
        return float(metric[0] / metric[1])

    rows: list[FeatureCrossAuditRow] = []
    for source_row in source_rows:
        feature_class = str(source_row["feature_class"]).strip()
        persistence_row = persistence_by_class[feature_class]
        label = str(source_row.get("feature_class_label", "")).strip()
        if not label:
            label = str(persistence_row.get("feature_class_label", feature_class)).strip()
        anchor_candidates = required_int(
            source_row.get("candidate_count"), "candidate_count", feature_class
        )
        anchor_quality = required_int(source_row.get("quality_count"), "quality_count", feature_class)
        candidate_presence = required_int(
            persistence_row.get("candidate_presence_ge_required_count"),
            "candidate_presence_ge_required_count",
            feature_class,
        )
        quality_presence = required_int(
            persistence_row.get("quality_presence_ge_required_count"),
            "quality_presence_ge_required_count",
            feature_class,
        )
        if min(anchor_candidates, anchor_quality, candidate_presence, quality_presence) < 0:
            raise ValueError(f"negative count in feature class: {feature_class}")
        if anchor_quality > anchor_candidates:
            raise ValueError(f"quality_count exceeds candidate_count: {feature_class}")
        if candidate_presence > anchor_candidates or quality_presence > anchor_candidates:
            raise ValueError(f"persistence count exceeds anchor count: {feature_class}")
        candidate_fraction = (
            float(candidate_presence / anchor_candidates) if anchor_candidates else None
        )
        quality_fraction = float(quality_presence / anchor_candidates) if anchor_candidates else None
        persistence_gap = (
            None
            if candidate_fraction is None or quality_fraction is None
            else float(candidate_fraction - quality_fraction)
        )
        median_flux_snr = optional_float(source_row.get("median_flux_snr"), "median_flux_snr", feature_class)
        max_flux_snr = optional_float(source_row.get("max_flux_snr"), "max_flux_snr", feature_class)
        high_snr_rejected = bool(
            max_flux_snr is not None
            and max_flux_snr >= float(high_snr_threshold)
            and anchor_quality == 0
        )

        candidate_any_response_fraction: float | None = None
        candidate_same_class_response_fraction: float | None = None
        quality_any_response_fraction: float | None = None
        quality_same_class_response_fraction: float | None = None
        dominant_cross_class: str | None = None
        dominant_cross_class_label: str | None = None
        if transition_file is not None:
            candidate_transitions = [
                transition_metrics[("candidate", feature_class, response_class)]
                for response_class in source_by_class
            ]
            candidate_possible = candidate_transitions[0][1]
            candidate_any_response_fraction = (
                float(sum(metric[0] for metric in candidate_transitions) / candidate_possible)
                if candidate_possible > 0
                else None
            )
            candidate_same_class_response_fraction = transition_fraction(
                "candidate", feature_class, feature_class
            )

            quality_transitions = [
                transition_metrics[("quality", feature_class, response_class)]
                for response_class in source_by_class
            ]
            quality_possible = quality_transitions[0][1]
            quality_any_response_fraction = (
                float(sum(metric[0] for metric in quality_transitions) / quality_possible)
                if quality_possible > 0
                else None
            )
            quality_same_class_response_fraction = transition_fraction(
                "quality", feature_class, feature_class
            )

            off_diagonal = [
                (
                    transition_metrics[("candidate", feature_class, response_class)][0],
                    response_class,
                    transition_metrics[("candidate", feature_class, response_class)][2],
                )
                for response_class in source_by_class
                if response_class != feature_class
            ]
            if off_diagonal:
                matched_count, response_class, response_label = max(
                    off_diagonal,
                    key=lambda item: (item[0], item[1]),
                )
                if matched_count > 0:
                    dominant_cross_class = response_class
                    dominant_cross_class_label = response_label

        if feature_class == "compact_quality" and anchor_quality == anchor_candidates:
            interpretation = "当前形态与质量规则下稳定；仍需星表或注入确认物理身份"
        elif feature_class == "crowded_blend":
            interpretation = "近邻/去混叠证据不足；低质量持久性不能等同于伪影"
        elif high_snr_rejected and candidate_fraction is not None and candidate_fraction >= 0.7 and quality_fraction is not None and quality_fraction <= 0.1:
            interpretation = "高显著且候选层可重复，但质量层拒绝；不能计为恒星"
        elif high_snr_rejected:
            interpretation = "高显著但质量层拒绝；优先复核形态、范围或线状结构"
        elif candidate_fraction is not None and candidate_fraction >= 0.7 and quality_fraction is not None and quality_fraction <= 0.1:
            interpretation = "候选层重复而质量层不重复；可能是弱源或固定结构"
        else:
            interpretation = "需结合原始像素、PSF、跨帧和星表证据"

        rows.append(
            FeatureCrossAuditRow(
                feature_class=feature_class,
                feature_class_label=label,
                anchor_candidate_count=anchor_candidates,
                anchor_quality_count=anchor_quality,
                median_flux_snr=median_flux_snr,
                max_flux_snr=max_flux_snr,
                required_presence=required_presence,
                frame_count=frame_count,
                candidate_presence_ge_required_count=candidate_presence,
                candidate_presence_fraction=candidate_fraction,
                quality_presence_ge_required_count=quality_presence,
                quality_presence_fraction=quality_fraction,
                persistence_gap=persistence_gap,
                high_snr_rejected=high_snr_rejected,
                interpretation=interpretation,
                candidate_any_response_fraction=candidate_any_response_fraction,
                candidate_same_class_response_fraction=candidate_same_class_response_fraction,
                quality_any_response_fraction=quality_any_response_fraction,
                quality_same_class_response_fraction=quality_same_class_response_fraction,
                dominant_cross_class=dominant_cross_class,
                dominant_cross_class_label=dominant_cross_class_label,
            )
        )

    high_snr_classes = [row.feature_class for row in rows if row.high_snr_rejected]
    large_gap_classes = [
        row.feature_class
        for row in rows
        if row.persistence_gap is not None and row.persistence_gap >= 0.5
    ]
    conclusion = (
        f"交叉审计连接 {len(rows)} 个特征类别；高 SNR 但质量层拒绝的类别为 "
        f"{','.join(high_snr_classes) if high_snr_classes else '无'}，"
        f"候选-质量持久性差距至少 0.5 的类别为 "
        f"{','.join(large_gap_classes) if large_gap_classes else '无'}。"
        "SNR、候选持久性和质量持久性分别描述显著性、稳定性和点源约束，"
        "不能替代星表/WCS 身份确认。"
    )
    if transition_file is not None:
        conclusion += (
            " 已连接类别条件转移；任意类别响应与同类响应被分别报告，"
            "仍不代表物理类别变化或逐星身份。"
        )
    return FeatureCrossAuditResult(
        source_summary_path=str(source_path),
        persistence_path=str(persistence_file),
        frame_count=frame_count,
        required_presence=required_presence,
        high_snr_threshold=float(high_snr_threshold),
        rows=tuple(rows),
        conclusion=conclusion,
        class_transition_path=None if transition_file is None else str(transition_file),
    )


def _nearest_source_to_peak(
    sources: Sequence[Detection],
    target_xy: tuple[float, float],
    match_radius_px: float,
) -> Detection | None:
    """按整数匹配峰在固定探测器坐标附近找最近源，不用检测 ID 跨帧追踪。"""

    if not sources:
        return None
    tx, ty = target_xy
    ranked: list[tuple[float, Detection]] = []
    for source in sources:
        source_x = source.peak_x if source.peak_x is not None else source.x
        source_y = source.peak_y if source.peak_y is not None else source.y
        distance = float(np.hypot(float(source_x) - tx, float(source_y) - ty))
        if distance <= float(match_radius_px):
            ranked.append((distance, source))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1].detection_id))
    return ranked[0][1]


def _nearest_source_with_distance(
    sources: Sequence[Detection],
    target_xy: tuple[float, float],
    search_radius_px: float,
) -> tuple[float, Detection] | None:
    """返回目标附近最近候选及峰坐标距离，用于识别合并/漂移响应。"""

    if not sources:
        return None
    tx, ty = target_xy
    ranked: list[tuple[float, Detection]] = []
    for source in sources:
        source_x = source.peak_x if source.peak_x is not None else source.x
        source_y = source.peak_y if source.peak_y is not None else source.y
        distance = float(np.hypot(float(source_x) - tx, float(source_y) - ty))
        if distance <= float(search_radius_px):
            ranked.append((distance, source))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1].detection_id))
    return ranked[0]


def _raw_region_statistics(
    image: np.ndarray,
    center_xy: tuple[float, float],
    radius_px: int,
    *,
    negative_overflow_limit: float | None,
) -> dict[str, object]:
    """提取一个原始像素圆孔径的范围、重复码和固定值统计。"""

    values = np.asarray(image)
    height, width = values.shape
    center_x, center_y = (int(round(float(center_xy[0]))), int(round(float(center_xy[1]))))
    x0 = max(0, center_x - int(radius_px))
    x1 = min(width, center_x + int(radius_px) + 1)
    y0 = max(0, center_y - int(radius_px))
    y1 = min(height, center_y + int(radius_px) + 1)
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)
    circular = np.hypot(xx + x0 - float(center_xy[0]), yy + y0 - float(center_xy[1])) <= float(radius_px)
    selected = np.asarray(values[y0:y1, x0:x1])[circular]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {
            "min": None,
            "max": None,
            "median": None,
            "unique": 0,
            "mode": None,
            "mode_count": 0,
            "mode_fraction": None,
            "fixed_minus_one": 0,
            "negative_overflow": 0,
            "positive_extreme": 0,
        }
    unique, counts = np.unique(selected, return_counts=True)
    mode_index = int(np.argmax(counts))
    mode_count = int(counts[mode_index])
    positive_extreme = (
        int(np.count_nonzero(selected >= -float(negative_overflow_limit)))
        if negative_overflow_limit is not None
        else 0
    )
    negative_extreme = (
        int(np.count_nonzero(selected <= float(negative_overflow_limit)))
        if negative_overflow_limit is not None
        else 0
    )
    return {
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
        "median": float(np.median(selected)),
        "unique": int(unique.size),
        "mode": float(unique[mode_index]),
        "mode_count": mode_count,
        "mode_fraction": float(mode_count / selected.size),
        "fixed_minus_one": int(np.count_nonzero(selected == -1)),
        "negative_overflow": negative_extreme,
        "positive_extreme": positive_extreme,
    }


def _pair_peak_evidence(
    image: np.ndarray,
    mask: np.ndarray,
    primary: Detection | None,
    secondary: Detection | None,
    primary_target_xy: tuple[float, float],
    secondary_target_xy: tuple[float, float],
    *,
    psf_fwhm: float,
) -> tuple[float | None, float | None]:
    """用检测峰（缺失时用目标坐标）计算局部单/双 PSF 证据。"""

    def peak_xy(source: Detection | None, fallback: tuple[float, float]) -> tuple[float, float]:
        if source is None:
            return fallback
        return (
            float(source.peak_x if source.peak_x is not None else source.x),
            float(source.peak_y if source.peak_y is not None else source.y),
        )

    return _pair_psf_evidence(
        image,
        mask,
        peak_xy(primary, primary_target_xy),
        peak_xy(secondary, secondary_target_xy),
        psf_fwhm=psf_fwhm,
    )


def _fixed_position_multi_psf_fit(
    image: np.ndarray,
    mask: np.ndarray,
    model_positions: Sequence[tuple[float, float]],
    *,
    patch_positions: Sequence[tuple[float, float]],
    psf_fwhm: float,
) -> dict[str, object] | None:
    """在相同局部窗口内拟合固定位置的非负 K 分量 PSF。

    位置和 PSF 宽度是外部给定的，只有背景平面和分量幅度参与拟合。
    背景项不受非负约束；先将背景从数据和 PSF 基函数中投影掉，再对
    PSF 幅度做 NNLS，避免把大 ADU 的背景截距错误地限制为正值。该函数
    只用于模型敏感性审计，不能替代自由位置、空间变 PSF 或完整多源
    最大似然拟合。
    """

    positions = tuple((float(point[0]), float(point[1])) for point in model_positions)
    geometry_positions = tuple((float(point[0]), float(point[1])) for point in patch_positions)
    if not positions or not geometry_positions or any(
        not np.isfinite(value) for point in (*positions, *geometry_positions) for value in point
    ):
        return None
    if psf_fwhm <= 0:
        return None
    sigma = max(float(psf_fwhm) / 2.35482, 0.5)
    margin = max(4, int(np.ceil(3.5 * sigma)))
    height, width = np.asarray(image).shape
    x0 = max(0, int(np.floor(min(point[0] for point in geometry_positions))) - margin)
    x1 = min(width, int(np.ceil(max(point[0] for point in geometry_positions))) + margin + 1)
    y0 = max(0, int(np.floor(min(point[1] for point in geometry_positions))) - margin)
    y1 = min(height, int(np.ceil(max(point[1] for point in geometry_positions))) + margin + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None

    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    patch_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    valid = ~patch_mask & np.isfinite(patch)
    sample_count = int(np.count_nonzero(valid))
    component_count = len(positions)
    background_count = 3
    parameter_count = background_count + component_count
    if sample_count < max(24, parameter_count + 8):
        return None

    yy, xx = np.indices(patch.shape, dtype=np.float64)
    gx = xx + float(x0)
    gy = yy + float(y0)
    center_x = 0.5 * (x0 + x1 - 1)
    center_y = 0.5 * (y0 + y1 - 1)
    scale = max(float(x1 - x0), float(y1 - y0), 1.0)
    background_terms = np.column_stack(
        [
            np.ones(patch.shape, dtype=np.float64)[valid],
            ((gx - center_x) / scale)[valid],
            ((gy - center_y) / scale)[valid],
        ]
    )
    source_terms = np.column_stack(
        [
            np.exp(-0.5 * (((gx - px) / sigma) ** 2 + ((gy - py) / sigma) ** 2))[valid]
            for px, py in positions
        ]
    )
    y_values = patch[valid]
    try:
        background_from_y, *_ = np.linalg.lstsq(background_terms, y_values, rcond=None)
        background_from_source, *_ = np.linalg.lstsq(background_terms, source_terms, rcond=None)
    except np.linalg.LinAlgError:
        return None
    source_residual = source_terms - background_terms @ background_from_source
    data_residual = y_values - background_terms @ background_from_y
    try:
        amplitudes, _nnls_residual = nnls(source_residual, data_residual)
    except (RuntimeError, ValueError):
        return None
    try:
        background_coefficients, *_ = np.linalg.lstsq(
            background_terms,
            y_values - source_terms @ amplitudes,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None
    model = background_terms @ background_coefficients + source_terms @ amplitudes
    residual = y_values - model
    epsilon = np.finfo(np.float64).eps
    rss = max(float(residual @ residual), epsilon)
    bic = float(
        sample_count * np.log(rss / sample_count)
        + parameter_count * np.log(sample_count)
    )

    degrees_of_freedom = max(1, sample_count - parameter_count)
    variance = rss / degrees_of_freedom
    active = amplitudes > max(1e-12, np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(amplitudes)))))
    component_snr = np.zeros(component_count, dtype=np.float64)
    if np.any(active):
        active_design = source_residual[:, active]
        try:
            covariance = variance * np.linalg.pinv(active_design.T @ active_design)
            errors = np.sqrt(np.maximum(np.diag(covariance), epsilon))
            component_snr[active] = amplitudes[active] / errors
        except (np.linalg.LinAlgError, ValueError):
            pass

    max_template_correlation: float | None = None
    if component_count >= 2:
        gram = source_residual.T @ source_residual
        norms = np.sqrt(np.maximum(np.diag(gram), epsilon))
        correlation = np.abs(gram / np.outer(norms, norms))
        upper = correlation[np.triu_indices(component_count, k=1)]
        if upper.size:
            max_template_correlation = float(np.max(upper))
    try:
        design_condition_number = float(np.linalg.cond(source_residual))
    except (np.linalg.LinAlgError, ValueError):
        design_condition_number = None
    if design_condition_number is not None and not np.isfinite(design_condition_number):
        design_condition_number = None

    return {
        "sample_count": sample_count,
        "masked_pixel_count": int(patch.size - sample_count),
        "rss": float(rss),
        "bic": bic,
        "amplitudes": tuple(float(value) for value in amplitudes),
        "component_snr": tuple(float(value) for value in component_snr),
        "active_component_count": int(np.count_nonzero(active)),
        "max_template_correlation": max_template_correlation,
        "design_condition_number": design_condition_number,
    }


def _free_position_multi_psf_fit(
    image: np.ndarray,
    mask: np.ndarray,
    model_positions: Sequence[tuple[float, float]],
    *,
    patch_positions: Sequence[tuple[float, float]],
    psf_fwhm: float,
    position_radius_px: float,
) -> dict[str, object] | None:
    """在候选中心附近小范围优化位置的非负 K 分量 PSF。"""

    positions = tuple((float(point[0]), float(point[1])) for point in model_positions)
    geometry_positions = tuple((float(point[0]), float(point[1])) for point in patch_positions)
    if not positions or not geometry_positions or any(
        not np.isfinite(value) for point in (*positions, *geometry_positions) for value in point
    ):
        return None
    if psf_fwhm <= 0 or not np.isfinite(float(psf_fwhm)):
        return None
    if position_radius_px <= 0 or not np.isfinite(float(position_radius_px)):
        return None

    sigma = max(float(psf_fwhm) / 2.35482, 0.5)
    margin = max(4, int(np.ceil(3.5 * sigma)))
    height, width = np.asarray(image).shape
    x0 = max(0, int(np.floor(min(point[0] for point in geometry_positions))) - margin)
    x1 = min(width, int(np.ceil(max(point[0] for point in geometry_positions))) + margin + 1)
    y0 = max(0, int(np.floor(min(point[1] for point in geometry_positions))) - margin)
    y1 = min(height, int(np.ceil(max(point[1] for point in geometry_positions))) + margin + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None

    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    patch_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    valid = ~patch_mask & np.isfinite(patch)
    sample_count = int(np.count_nonzero(valid))
    component_count = len(positions)
    parameter_count = 3 + 3 * component_count
    if sample_count < max(24, parameter_count + 8):
        return None

    yy, xx = np.indices(patch.shape, dtype=np.float64)
    gx = xx + float(x0)
    gy = yy + float(y0)
    center_x = 0.5 * (x0 + x1 - 1)
    center_y = 0.5 * (y0 + y1 - 1)
    scale = max(float(x1 - x0), float(y1 - y0), 1.0)
    background_terms = np.column_stack(
        [
            np.ones(patch.shape, dtype=np.float64)[valid],
            ((gx - center_x) / scale)[valid],
            ((gy - center_y) / scale)[valid],
        ]
    )
    y_values = patch[valid]

    def source_terms_at(current_positions: Sequence[tuple[float, float]]) -> np.ndarray:
        return np.column_stack(
            [
                np.exp(
                    -0.5
                    * (((gx - px) / sigma) ** 2 + ((gy - py) / sigma) ** 2)
                )[valid]
                for px, py in current_positions
            ]
        )

    initial_fixed = _fixed_position_multi_psf_fit(
        image,
        mask,
        positions,
        patch_positions=geometry_positions,
        psf_fwhm=psf_fwhm,
    )
    if initial_fixed is None:
        return None
    initial_amplitudes = np.maximum(
        np.asarray(initial_fixed["amplitudes"], dtype=np.float64),
        0.0,
    )
    initial_source_terms = source_terms_at(positions)
    try:
        initial_background, *_ = np.linalg.lstsq(
            background_terms,
            y_values - initial_source_terms @ initial_amplitudes,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None

    parameter_start = np.empty(parameter_count, dtype=np.float64)
    parameter_start[:3] = initial_background
    lower = np.full(parameter_count, -np.inf, dtype=np.float64)
    upper = np.full(parameter_count, np.inf, dtype=np.float64)
    for index, ((px, py), amplitude) in enumerate(zip(positions, initial_amplitudes, strict=True)):
        base = 3 + 3 * index
        parameter_start[base] = float(amplitude)
        parameter_start[base + 1] = px
        parameter_start[base + 2] = py
        lower[base] = 0.0
        lower[base + 1] = px - float(position_radius_px)
        lower[base + 2] = py - float(position_radius_px)
        upper[base + 1] = px + float(position_radius_px)
        upper[base + 2] = py + float(position_radius_px)

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        fitted_background = background_terms @ parameters[:3]
        model = fitted_background.copy()
        for index in range(component_count):
            base = 3 + 3 * index
            amplitude = parameters[base]
            fitted_x = parameters[base + 1]
            fitted_y = parameters[base + 2]
            model += amplitude * np.exp(
                -0.5
                * (((gx[valid] - fitted_x) / sigma) ** 2 + ((gy[valid] - fitted_y) / sigma) ** 2)
            )
        return model - y_values

    try:
        optimized = least_squares(
            residual_function,
            parameter_start,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            loss="linear",
            max_nfev=300,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    parameters = np.asarray(optimized.x, dtype=np.float64)
    residual = np.asarray(optimized.fun, dtype=np.float64)
    if parameters.shape != (parameter_count,) or not np.all(np.isfinite(parameters)):
        return None
    if residual.size != sample_count or not np.all(np.isfinite(residual)):
        return None

    epsilon = np.finfo(np.float64).eps
    rss = max(float(residual @ residual), epsilon)
    bic = float(sample_count * np.log(rss / sample_count) + parameter_count * np.log(sample_count))
    amplitudes = np.asarray(
        [parameters[3 + 3 * index] for index in range(component_count)],
        dtype=np.float64,
    )
    fitted_positions = tuple(
        (
            float(parameters[3 + 3 * index + 1]),
            float(parameters[3 + 3 * index + 2]),
        )
        for index in range(component_count)
    )
    position_offsets = tuple(
        (fitted[0] - initial[0], fitted[1] - initial[1])
        for initial, fitted in zip(positions, fitted_positions, strict=True)
    )
    position_tolerance = max(1e-5, float(position_radius_px) * 1e-5)
    positions_at_bound = tuple(
        abs(offset[0]) >= float(position_radius_px) - position_tolerance
        or abs(offset[1]) >= float(position_radius_px) - position_tolerance
        for offset in position_offsets
    )

    active = amplitudes > max(
        1e-12,
        np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(amplitudes)))),
    )
    component_snr = np.zeros(component_count, dtype=np.float64)
    degrees_of_freedom = max(1, sample_count - parameter_count)
    variance = rss / degrees_of_freedom
    try:
        jacobian = np.asarray(optimized.jac, dtype=np.float64)
        covariance = variance * np.linalg.pinv(jacobian.T @ jacobian)
        errors = np.sqrt(np.maximum(np.diag(covariance), epsilon))
        for index in range(component_count):
            amplitude_index = 3 + 3 * index
            if active[index] and amplitude_index < errors.size:
                component_snr[index] = amplitudes[index] / errors[amplitude_index]
    except (np.linalg.LinAlgError, ValueError, TypeError):
        pass

    source_terms = source_terms_at(fitted_positions)
    try:
        background_from_source, *_ = np.linalg.lstsq(
            background_terms,
            source_terms,
            rcond=None,
        )
        source_residual = source_terms - background_terms @ background_from_source
    except np.linalg.LinAlgError:
        source_residual = source_terms
    max_template_correlation: float | None = None
    if component_count >= 2:
        gram = source_residual.T @ source_residual
        norms = np.sqrt(np.maximum(np.diag(gram), epsilon))
        correlation = np.abs(gram / np.outer(norms, norms))
        upper_triangle = correlation[np.triu_indices(component_count, k=1)]
        if upper_triangle.size:
            max_template_correlation = float(np.max(upper_triangle))
    try:
        design_condition_number = float(np.linalg.cond(source_residual))
    except (np.linalg.LinAlgError, ValueError):
        design_condition_number = None
    if design_condition_number is not None and not np.isfinite(design_condition_number):
        design_condition_number = None

    return {
        "sample_count": sample_count,
        "masked_pixel_count": int(patch.size - sample_count),
        "rss": rss,
        "bic": bic,
        "initial_positions": positions,
        "fitted_positions": fitted_positions,
        "position_offsets": position_offsets,
        "positions_at_bound": positions_at_bound,
        "amplitudes": tuple(float(value) for value in amplitudes),
        "component_snr": tuple(float(value) for value in component_snr),
        "active_component_count": int(np.count_nonzero(active)),
        "max_template_correlation": max_template_correlation,
        "design_condition_number": design_condition_number,
        "optimizer_success": bool(optimized.success),
        "optimizer_status": int(optimized.status),
        "optimizer_nfev": int(optimized.nfev),
    }


def _multi_psf_mask_modes(
    image: np.ndarray,
    *,
    mask_modes: Sequence[str],
) -> dict[str, np.ndarray]:
    """构造多源局部模型使用的平行掩膜，不改写原图。"""

    values = np.asarray(image)
    base = auxiliary_mask(values.shape)
    negative_limit = _default_negative_overflow_limit(values)
    result: dict[str, np.ndarray] = {}
    for raw_mode in mask_modes:
        mode = str(raw_mode)
        if mode == "raw":
            mode_mask = base.copy()
        elif mode == "range_masked":
            mode_mask = base.copy()
            if negative_limit is not None:
                mode_mask |= values <= float(negative_limit)
                mode_mask |= values >= -float(negative_limit)
        elif mode == "sentinel_masked":
            mode_mask = base | (values == -1)
        elif mode == "range_sentinel_masked":
            mode_mask = base | (values == -1)
            if negative_limit is not None:
                mode_mask |= values <= float(negative_limit)
                mode_mask |= values >= -float(negative_limit)
        else:
            raise ValueError(
                "mask_modes must contain only raw, range_masked, sentinel_masked, "
                "or range_sentinel_masked"
            )
        result[mode] = np.asarray(mode_mask, dtype=bool)
    return result


def run_local_multipsf_audit(
    paths: Iterable[str | Path],
    target_positions: Sequence[tuple[float, float]],
    *,
    psf_fwhm: float = 2.0,
    mask_modes: Sequence[str] = ("raw", "range_masked", "sentinel_masked"),
    frame_shifts: Sequence[tuple[float, float]] | None = None,
    confirmation_delta_bic: float = 10.0,
    confirmation_component_snr: float = 5.0,
    progress: Callable[[int, int], None] | None = None,
) -> LocalMultiPSFAuditResult:
    """比较同一局部的 K=1..N 固定位置 PSF 模型。

    该实验用于回答“两个或多个框是否必须解释为多个点源”。所有 K
    模型在同一帧、同一窗口、同一掩膜和同一背景平面上比较；因此它能
    发现额外分量是否只是主源翼部的替代解释，但不能证明存在真实星表
    身份。``frame_shifts`` 仅用于把 detector 目标移动到另一帧，仍不是
    WCS 配准。
    """

    frame_paths = tuple(paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    positions = tuple((float(point[0]), float(point[1])) for point in target_positions)
    if len(positions) < 2:
        raise ValueError("target_positions must contain at least two positions")
    if any(not np.isfinite(value) for point in positions for value in point):
        raise ValueError("target_positions must contain finite coordinates")
    if any(
        np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]) <= 1e-6
        for index, point_a in enumerate(positions)
        for point_b in positions[index + 1 :]
    ):
        raise ValueError("target_positions must be distinct")
    if psf_fwhm <= 0 or not np.isfinite(float(psf_fwhm)):
        raise ValueError("psf_fwhm must be finite and positive")
    resolved_modes = tuple(str(mode) for mode in mask_modes)
    if not resolved_modes:
        raise ValueError("mask_modes cannot be empty")
    if len(set(resolved_modes)) != len(resolved_modes):
        raise ValueError("mask_modes must not contain duplicates")
    allowed_modes = {"raw", "range_masked", "sentinel_masked", "range_sentinel_masked"}
    if any(mode not in allowed_modes for mode in resolved_modes):
        raise ValueError(
            "mask_modes must contain only raw, range_masked, sentinel_masked, "
            "or range_sentinel_masked"
        )

    total = len(frame_paths)
    if frame_shifts is None:
        resolved_shifts = tuple((0.0, 0.0) for _path in frame_paths)
    else:
        if len(frame_shifts) != total:
            raise ValueError("frame_shifts length must match paths")
        resolved_shifts = tuple((float(shift[0]), float(shift[1])) for shift in frame_shifts)
        if any(not np.isfinite(value) for shift in resolved_shifts for value in shift):
            raise ValueError("frame_shifts must contain finite values")

    if confirmation_delta_bic <= 0 or confirmation_component_snr <= 0:
        raise ValueError("confirmation thresholds must be positive")
    rows: list[LocalMultiPSFAuditRow] = []
    for frame_index, path in enumerate(frame_paths):
        frame = read_fits(path)
        image = np.asarray(frame.data)
        if image.ndim != 2:
            raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
        shift_x, shift_y = resolved_shifts[frame_index]
        frame_positions = tuple((x + shift_x, y + shift_y) for x, y in positions)
        masks = _multi_psf_mask_modes(image, mask_modes=resolved_modes)
        for mask_mode, mode_mask in masks.items():
            fits: list[dict[str, object] | None] = []
            for component_count in range(1, len(frame_positions) + 1):
                fits.append(
                    _fixed_position_multi_psf_fit(
                        image,
                        mode_mask,
                        frame_positions[:component_count],
                        patch_positions=frame_positions,
                        psf_fwhm=float(psf_fwhm),
                    )
                )
            single_bic = None if fits[0] is None else float(fits[0]["bic"])
            for component_index, fit in enumerate(fits, start=1):
                previous_fit = fits[component_index - 2] if component_index >= 2 else None
                bic = None if fit is None else float(fit["bic"])
                delta_single = (
                    None if single_bic is None or bic is None else float(single_bic - bic)
                )
                previous_bic = None if previous_fit is None else float(previous_fit["bic"])
                delta_previous = (
                    None if previous_bic is None or bic is None else float(previous_bic - bic)
                )
                snr_values = () if fit is None else tuple(float(value) for value in fit["component_snr"])
                added_snr = snr_values[1:] if component_index >= 2 else ()
                confirmation = bool(
                    component_index >= 2
                    and delta_single is not None
                    and delta_single >= float(confirmation_delta_bic)
                    and added_snr
                    and min(added_snr) >= float(confirmation_component_snr)
                )
                if fit is None:
                    note = "局部有效像素不足或线性模型数值不稳定"
                    target_values: tuple[tuple[float, float], ...] = frame_positions[:component_index]
                    sample_count = 0
                    masked_pixel_count = 0
                    rss = bic_value = None
                    delta_single_value = delta_previous_value = None
                    amplitudes: tuple[float, ...] = ()
                    max_template_correlation = condition_number = None
                    active_component_count = 0
                else:
                    note = (
                        "达到当前工程双/多源证据线；仍需实测 PSF、注入和星表核验"
                        if confirmation
                        else "局部模型比较；不等于物理恒星身份"
                    )
                    target_values = frame_positions[:component_index]
                    sample_count = int(fit["sample_count"])
                    masked_pixel_count = int(fit["masked_pixel_count"])
                    rss = float(fit["rss"])
                    bic_value = bic
                    delta_single_value = delta_single
                    delta_previous_value = delta_previous
                    amplitudes = tuple(float(value) for value in fit["amplitudes"])
                    max_template_correlation = (
                        None
                        if fit["max_template_correlation"] is None
                        else float(fit["max_template_correlation"])
                    )
                    condition_number = (
                        None
                        if fit["design_condition_number"] is None
                        else float(fit["design_condition_number"])
                    )
                    active_component_count = int(fit["active_component_count"])
                rows.append(
                    LocalMultiPSFAuditRow(
                        frame_index=frame_index + 1,
                        path=str(frame.path),
                        mask_mode=mask_mode,
                        component_count=component_index,
                        target_positions=target_values,
                        sample_count=sample_count,
                        masked_pixel_count=masked_pixel_count,
                        rss=rss,
                        bic=bic_value,
                        delta_bic_from_single=delta_single_value,
                        delta_bic_from_previous=delta_previous_value,
                        amplitudes=amplitudes,
                        component_snr=snr_values,
                        active_component_count=active_component_count,
                        max_template_correlation=max_template_correlation,
                        design_condition_number=condition_number,
                        confirmation_line=confirmation,
                        note=note,
                    )
                )
        if progress is not None:
            progress(frame_index + 1, total)

    raw_rows = [row for row in rows if row.mask_mode == "raw"]
    confirmed_by_k = {
        component_count: sum(
            row.confirmation_line and row.component_count == component_count
            for row in raw_rows
        )
        for component_count in range(2, len(positions) + 1)
    }
    best_raw_by_frame: list[str] = []
    for frame_index in range(1, total + 1):
        frame_rows = [
            row
            for row in raw_rows
            if row.frame_index == frame_index and row.bic is not None
        ]
        if not frame_rows:
            best_raw_by_frame.append(f"F{frame_index:02d}: 无有效局部拟合")
            continue
        best = min(frame_rows, key=lambda row: float(row.bic))
        delta = best.delta_bic_from_single
        best_raw_by_frame.append(
            f"F{frame_index:02d}: BIC 最优 K={best.component_count}"
            + (f" (ΔBIC={delta:.2f})" if delta is not None else "")
        )
    confirmation_summary = ", ".join(
        f"K={component_count}: {count}/{total} 帧"
        for component_count, count in confirmed_by_k.items()
    )
    conclusion = (
        f"固定位置 K 源 PSF 审计完成：{total} 帧、{len(positions)} 个候选位置，"
        f"当前证据线下 raw 掩膜确认统计为 {confirmation_summary}。"
        "K 源模型只说明局部像素对某个受限模型的相对支持；它不能替代自由位置、"
        "空间变 PSF、完整噪声似然、人工注入或 WCS/星表身份。"
        + (" 首帧/各帧 raw 最优模型：" + "; ".join(best_raw_by_frame) + "。" if best_raw_by_frame else "")
    )
    parameters = {
        "psf_fwhm_px": float(psf_fwhm),
        "mask_modes": list(resolved_modes),
        "component_counts": list(range(1, len(positions) + 1)),
        "confirmation_delta_bic": float(confirmation_delta_bic),
        "confirmation_component_snr": float(confirmation_component_snr),
        "frame_shifts": [list(shift) for shift in resolved_shifts],
        "model": "shared linear background + fixed-position Gaussian PSF + nonnegative amplitudes",
        "interpretation_boundary": "local model evidence, not WCS/catalog identity",
    }
    return LocalMultiPSFAuditResult(
        frame_count=total,
        target_positions=positions,
        frame_shifts=resolved_shifts,
        mask_modes=resolved_modes,
        psf_fwhm_px=float(psf_fwhm),
        parameters=parameters,
        rows=tuple(rows),
        conclusion=conclusion,
    )


def run_local_free_multipsf_audit(
    paths: Iterable[str | Path],
    target_positions: Sequence[tuple[float, float]],
    *,
    psf_fwhm: float = 2.0,
    position_radius_px: float = 1.25,
    mask_modes: Sequence[str] = ("raw", "range_masked", "sentinel_masked"),
    frame_shifts: Sequence[tuple[float, float]] | None = None,
    confirmation_delta_bic: float = 10.0,
    confirmation_component_snr: float = 5.0,
    progress: Callable[[int, int], None] | None = None,
) -> LocalFreeMultiPSFAuditResult:
    """比较候选中心小范围自由移动时的 K=1..N 局部 PSF 模型。

    该实验专门检查固定 detector 坐标是否人为制造了副峰。位置只允许在
    初始候选周围的有限半径内移动，避免把整个局部窗口中的噪声峰任意吸收
    成源；模型仍固定 Gaussian FWHM 和线性背景，因而只是研究性对照。
    """

    frame_paths = tuple(paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    positions = tuple((float(point[0]), float(point[1])) for point in target_positions)
    if len(positions) < 2:
        raise ValueError("target_positions must contain at least two positions")
    if any(not np.isfinite(value) for point in positions for value in point):
        raise ValueError("target_positions must contain finite coordinates")
    if any(
        np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]) <= 1e-6
        for index, point_a in enumerate(positions)
        for point_b in positions[index + 1 :]
    ):
        raise ValueError("target_positions must be distinct")
    if psf_fwhm <= 0 or not np.isfinite(float(psf_fwhm)):
        raise ValueError("psf_fwhm must be finite and positive")
    if position_radius_px <= 0 or not np.isfinite(float(position_radius_px)):
        raise ValueError("position_radius_px must be finite and positive")
    resolved_modes = tuple(str(mode) for mode in mask_modes)
    if not resolved_modes:
        raise ValueError("mask_modes cannot be empty")
    if len(set(resolved_modes)) != len(resolved_modes):
        raise ValueError("mask_modes must not contain duplicates")
    allowed_modes = {"raw", "range_masked", "sentinel_masked", "range_sentinel_masked"}
    if any(mode not in allowed_modes for mode in resolved_modes):
        raise ValueError(
            "mask_modes must contain only raw, range_masked, sentinel_masked, "
            "or range_sentinel_masked"
        )
    if confirmation_delta_bic <= 0 or confirmation_component_snr <= 0:
        raise ValueError("confirmation thresholds must be positive")

    total = len(frame_paths)
    if frame_shifts is None:
        resolved_shifts = tuple((0.0, 0.0) for _path in frame_paths)
    else:
        if len(frame_shifts) != total:
            raise ValueError("frame_shifts length must match paths")
        resolved_shifts = tuple((float(shift[0]), float(shift[1])) for shift in frame_shifts)
        if any(not np.isfinite(value) for shift in resolved_shifts for value in shift):
            raise ValueError("frame_shifts must contain finite values")

    rows: list[LocalFreeMultiPSFAuditRow] = []
    for frame_index, path in enumerate(frame_paths):
        frame = read_fits(path)
        image = np.asarray(frame.data)
        if image.ndim != 2:
            raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
        shift_x, shift_y = resolved_shifts[frame_index]
        frame_positions = tuple((x + shift_x, y + shift_y) for x, y in positions)
        masks = _multi_psf_mask_modes(image, mask_modes=resolved_modes)
        for mask_mode, mode_mask in masks.items():
            fits: list[dict[str, object] | None] = []
            for component_count in range(1, len(frame_positions) + 1):
                fits.append(
                    _free_position_multi_psf_fit(
                        image,
                        mode_mask,
                        frame_positions[:component_count],
                        patch_positions=frame_positions,
                        psf_fwhm=float(psf_fwhm),
                        position_radius_px=float(position_radius_px),
                    )
                )
            single_bic = None if fits[0] is None else float(fits[0]["bic"])
            for component_index, fit in enumerate(fits, start=1):
                previous_fit = fits[component_index - 2] if component_index >= 2 else None
                bic = None if fit is None else float(fit["bic"])
                delta_single = None if single_bic is None or bic is None else float(single_bic - bic)
                previous_bic = None if previous_fit is None else float(previous_fit["bic"])
                delta_previous = None if previous_bic is None or bic is None else float(previous_bic - bic)
                if fit is None:
                    initial_positions = frame_positions[:component_index]
                    fitted_positions = initial_positions
                    position_offsets = tuple((0.0, 0.0) for _point in initial_positions)
                    positions_at_bound = tuple(False for _point in initial_positions)
                    sample_count = 0
                    masked_pixel_count = 0
                    rss = bic_value = None
                    delta_single_value = delta_previous_value = None
                    amplitudes: tuple[float, ...] = ()
                    snr_values: tuple[float, ...] = ()
                    active_component_count = 0
                    max_template_correlation = condition_number = None
                    optimizer_success = False
                    optimizer_status = optimizer_nfev = None
                    confirmation = False
                    note = "局部有效像素不足或自由位置拟合数值不稳定"
                else:
                    initial_positions = tuple(
                        (float(point[0]), float(point[1])) for point in fit["initial_positions"]
                    )
                    fitted_positions = tuple(
                        (float(point[0]), float(point[1])) for point in fit["fitted_positions"]
                    )
                    position_offsets = tuple(
                        (float(point[0]), float(point[1])) for point in fit["position_offsets"]
                    )
                    positions_at_bound = tuple(bool(value) for value in fit["positions_at_bound"])
                    snr_values = tuple(float(value) for value in fit["component_snr"])
                    added_snr = snr_values[1:] if component_index >= 2 else ()
                    added_positions_interior = not any(positions_at_bound[1:])
                    confirmation = bool(
                        component_index >= 2
                        and delta_single is not None
                        and delta_single >= float(confirmation_delta_bic)
                        and added_snr
                        and min(added_snr) >= float(confirmation_component_snr)
                        and added_positions_interior
                        and bool(fit["optimizer_success"])
                    )
                    if confirmation:
                        note = "达到当前工程双/多源证据线；仍需实测 PSF、注入和星表核验"
                    elif not added_positions_interior and component_index >= 2:
                        note = "新增分量触碰位置搜索边界；局部模型比较，不等于物理恒星身份"
                    elif not bool(fit["optimizer_success"]):
                        note = "优化器未报告收敛；局部模型比较，不等于物理恒星身份"
                    else:
                        note = "局部模型比较；不等于物理恒星身份"
                    sample_count = int(fit["sample_count"])
                    masked_pixel_count = int(fit["masked_pixel_count"])
                    rss = float(fit["rss"])
                    bic_value = bic
                    delta_single_value = delta_single
                    delta_previous_value = delta_previous
                    amplitudes = tuple(float(value) for value in fit["amplitudes"])
                    active_component_count = int(fit["active_component_count"])
                    max_template_correlation = (
                        None
                        if fit["max_template_correlation"] is None
                        else float(fit["max_template_correlation"])
                    )
                    condition_number = (
                        None
                        if fit["design_condition_number"] is None
                        else float(fit["design_condition_number"])
                    )
                    optimizer_success = bool(fit["optimizer_success"])
                    optimizer_status = int(fit["optimizer_status"])
                    optimizer_nfev = int(fit["optimizer_nfev"])
                rows.append(
                    LocalFreeMultiPSFAuditRow(
                        frame_index=frame_index + 1,
                        path=str(frame.path),
                        mask_mode=mask_mode,
                        component_count=component_index,
                        initial_positions=initial_positions,
                        fitted_positions=fitted_positions,
                        position_offsets=position_offsets,
                        positions_at_bound=positions_at_bound,
                        sample_count=sample_count,
                        masked_pixel_count=masked_pixel_count,
                        rss=rss,
                        bic=bic_value,
                        delta_bic_from_single=delta_single_value,
                        delta_bic_from_previous=delta_previous_value,
                        amplitudes=amplitudes,
                        component_snr=snr_values,
                        active_component_count=active_component_count,
                        max_template_correlation=max_template_correlation,
                        design_condition_number=condition_number,
                        optimizer_success=optimizer_success,
                        optimizer_status=optimizer_status,
                        optimizer_nfev=optimizer_nfev,
                        confirmation_line=confirmation,
                        note=note,
                    )
                )
        if progress is not None:
            progress(frame_index + 1, total)

    raw_rows = [row for row in rows if row.mask_mode == "raw"]
    confirmed_by_k = {
        component_count: sum(
            row.confirmation_line and row.component_count == component_count
            for row in raw_rows
        )
        for component_count in range(2, len(positions) + 1)
    }
    best_raw_by_frame: list[str] = []
    for frame_index in range(1, total + 1):
        frame_rows = [
            row for row in raw_rows if row.frame_index == frame_index and row.bic is not None
        ]
        if not frame_rows:
            best_raw_by_frame.append(f"F{frame_index:02d}: 无有效局部拟合")
            continue
        best = min(frame_rows, key=lambda row: float(row.bic))
        delta = best.delta_bic_from_single
        best_raw_by_frame.append(
            f"F{frame_index:02d}: BIC 最优 K={best.component_count}"
            + (f" (ΔBIC={delta:.2f})" if delta is not None else "")
        )
    confirmation_summary = ", ".join(
        f"K={component_count}: {count}/{total} 帧"
        for component_count, count in confirmed_by_k.items()
    )
    conclusion = (
        f"自由位置 K 源 PSF 审计完成：{total} 帧、{len(positions)} 个候选位置，"
        f"raw 证据线下确认统计为 {confirmation_summary}。位置仅允许在初始候选周围 "
        f"±{float(position_radius_px):g} px 移动；该模型仍不能替代空间变 PSF、"
        "完整噪声似然、人工注入或 WCS/星表身份。"
        + (" 各帧 raw 最优模型：" + "; ".join(best_raw_by_frame) + "。" if best_raw_by_frame else "")
    )
    parameters = {
        "psf_fwhm_px": float(psf_fwhm),
        "position_radius_px": float(position_radius_px),
        "mask_modes": list(resolved_modes),
        "component_counts": list(range(1, len(positions) + 1)),
        "confirmation_delta_bic": float(confirmation_delta_bic),
        "confirmation_component_snr": float(confirmation_component_snr),
        "require_added_positions_interior": True,
        "frame_shifts": [list(shift) for shift in resolved_shifts],
        "model": "shared linear background + bounded free-position Gaussian PSF + nonnegative amplitudes",
        "interpretation_boundary": "local model evidence, not WCS/catalog identity",
    }
    return LocalFreeMultiPSFAuditResult(
        frame_count=total,
        target_positions=positions,
        frame_shifts=resolved_shifts,
        mask_modes=resolved_modes,
        psf_fwhm_px=float(psf_fwhm),
        position_radius_px=float(position_radius_px),
        parameters=parameters,
        rows=tuple(rows),
        conclusion=conclusion,
    )


def run_local_free_multipsf_radius_sweep(
    paths: Iterable[str | Path],
    target_positions: Sequence[tuple[float, float]],
    *,
    position_radii_px: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.25),
    psf_fwhm: float = 2.0,
    mask_modes: Sequence[str] = ("raw",),
    frame_shifts: Sequence[tuple[float, float]] | None = None,
    confirmation_delta_bic: float = 10.0,
    confirmation_component_snr: float = 5.0,
    progress: Callable[[int, int], None] | None = None,
) -> LocalFreeMultiPSFRadiusSweepResult:
    """扫描位置搜索半径，检查自由位置双源证据是否稳定。"""

    frame_paths = tuple(paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    radii = tuple(float(radius) for radius in position_radii_px)
    if not radii or any(not np.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError("position_radii_px must contain finite positive values")
    if len(set(radii)) != len(radii):
        raise ValueError("position_radii_px must not contain duplicates")
    total_work = len(frame_paths) * len(radii)
    rows: list[LocalFreeMultiPSFRadiusSweepRow] = []
    for radius_index, radius in enumerate(radii):
        def report_radius(index: int, total: int, *, offset: int = radius_index * len(frame_paths)) -> None:
            if progress is not None:
                progress(offset + index, total_work)

        audit = run_local_free_multipsf_audit(
            frame_paths,
            target_positions,
            psf_fwhm=psf_fwhm,
            position_radius_px=radius,
            mask_modes=mask_modes,
            frame_shifts=frame_shifts,
            confirmation_delta_bic=confirmation_delta_bic,
            confirmation_component_snr=confirmation_component_snr,
            progress=report_radius,
        )
        for row in audit.rows:
            rows.append(
                LocalFreeMultiPSFRadiusSweepRow(
                    position_radius_px=radius,
                    frame_index=row.frame_index,
                    path=row.path,
                    mask_mode=row.mask_mode,
                    component_count=row.component_count,
                    delta_bic_from_single=row.delta_bic_from_single,
                    component_snr=row.component_snr,
                    fitted_positions=row.fitted_positions,
                    positions_at_bound=row.positions_at_bound,
                    optimizer_success=row.optimizer_success,
                    confirmation_line=row.confirmation_line,
                    note=row.note,
                )
            )

    summaries: list[str] = []
    for radius in radii:
        raw_pair_rows = [
            row
            for row in rows
            if row.position_radius_px == radius
            and row.mask_mode == "raw"
            and row.component_count == 2
        ]
        confirmed = sum(row.confirmation_line for row in raw_pair_rows)
        at_bound = sum(any(row.positions_at_bound[1:]) for row in raw_pair_rows)
        not_converged = sum(not row.optimizer_success for row in raw_pair_rows)
        summaries.append(
            f"r={radius:g}: raw K=2 确认 {confirmed}/{len(frame_paths)}，"
            f"新增位置触边 {at_bound}/{len(frame_paths)}，未收敛 {not_converged}/{len(frame_paths)}"
        )
    conclusion = (
        f"自由位置半径敏感性审计完成：{len(radii)} 个半径、{len(frame_paths)} 帧。"
        + "；".join(summaries)
        + "。半径改变会改变可吸收的局部结构，因此稳定性是必要的审计条件，"
        "不等于物理恒星身份。"
    )
    parameters = {
        "position_radii_px": list(radii),
        "psf_fwhm_px": float(psf_fwhm),
        "mask_modes": [str(mode) for mode in mask_modes],
        "confirmation_delta_bic": float(confirmation_delta_bic),
        "confirmation_component_snr": float(confirmation_component_snr),
        "frame_shifts": (
            None
            if frame_shifts is None
            else [[float(shift[0]), float(shift[1])] for shift in frame_shifts]
        ),
        "interpretation_boundary": "radius stability is a necessary local check, not catalog identity",
    }
    return LocalFreeMultiPSFRadiusSweepResult(
        frame_count=len(frame_paths),
        target_positions=tuple((float(point[0]), float(point[1])) for point in target_positions),
        position_radii_px=radii,
        mask_modes=tuple(str(mode) for mode in mask_modes),
        psf_fwhm_px=float(psf_fwhm),
        parameters=parameters,
        rows=tuple(rows),
        conclusion=conclusion,
    )


def run_source_pair_audit(
    paths: Iterable[str | Path],
    primary_target_xy: tuple[float, float],
    secondary_target_xy: tuple[float, float],
    *,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 2.0,
    background_box_size: int = 128,
    background_sample_limit: int = 100_000,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
    max_sources: int | None = None,
    match_radius_px: float = 2.0,
    nearest_search_radius_px: float = 8.0,
    frame_shifts: Sequence[tuple[float, float]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SourcePairAuditResult:
    """对两个近邻响应做逐帧原始证据审计。

    该实验专门回答“两个圈是不是两颗星”，并有意不把两个坐标当成
    已经确认的物理身份：单帧检测只用于找到对应候选，原始圆孔径统计
    检查重复量化码、固定 ``-1`` 和极端正负码，单/双 PSF 只提供局部
    模型比较。默认使用固定 detector 坐标来找热像素/数据编码结构；传入
    ``frame_shifts`` 后，两个目标会按每帧的 detector 平移移动，用于检查
    注册坐标下的响应。两种模式都不是 WCS/星表身份匹配；真实星表验证仍
    需要授权的 WCS 和目录。
    """

    frame_paths = tuple(paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    total = len(frame_paths)
    if frame_shifts is None:
        resolved_frame_shifts = tuple((0.0, 0.0) for _ in frame_paths)
        using_registered_targets = False
    else:
        if len(frame_shifts) != total:
            raise ValueError("frame_shifts length must match paths")
        resolved_frame_shifts = tuple(
            (float(shift[0]), float(shift[1])) for shift in frame_shifts
        )
        if not all(np.isfinite(value) for shift in resolved_frame_shifts for value in shift):
            raise ValueError("frame_shifts must contain finite values")
        using_registered_targets = True
    if not all(np.isfinite(float(value)) for point in (primary_target_xy, secondary_target_xy) for value in point):
        raise ValueError("target coordinates must be finite")
    if primary_target_xy == secondary_target_xy:
        raise ValueError("primary and secondary target coordinates must differ")
    if threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1 or psf_fwhm <= 0:
        raise ValueError("pair audit detector parameters must be positive")
    if background_box_size < 16 or background_sample_limit < 1 or min_flux_snr <= 0:
        raise ValueError("pair audit background/quality parameters are invalid")
    if not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_psf_support_pixels must be within 1..9")
    if max_sources is not None and max_sources < 1:
        raise ValueError("max_sources must be positive or None")
    if match_radius_px <= 0:
        raise ValueError("match_radius_px must be positive")
    if nearest_search_radius_px < match_radius_px:
        raise ValueError("nearest_search_radius_px must be at least match_radius_px")

    target_distance = float(np.hypot(
        float(secondary_target_xy[0]) - float(primary_target_xy[0]),
        float(secondary_target_xy[1]) - float(primary_target_xy[1]),
    ))
    base_midpoint = (
        0.5 * (float(primary_target_xy[0]) + float(secondary_target_xy[0])),
        0.5 * (float(primary_target_xy[1]) + float(secondary_target_xy[1])),
    )
    pair_window_radius = max(int(aperture_radius) + 2, int(np.ceil(target_distance / 2.0)) + int(aperture_radius))
    rows: list[SourcePairAuditRow] = []
    for frame_index, path in enumerate(frame_paths):
        shift_x, shift_y = resolved_frame_shifts[frame_index]
        frame_primary_target = (
            float(primary_target_xy[0]) + shift_x,
            float(primary_target_xy[1]) + shift_y,
        )
        frame_secondary_target = (
            float(secondary_target_xy[0]) + shift_x,
            float(secondary_target_xy[1]) + shift_y,
        )
        frame_midpoint = (base_midpoint[0] + shift_x, base_midpoint[1] + shift_y)
        frame = read_fits(path)
        analysis = analyze_frame(
            frame,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            background_sample_limit=background_sample_limit,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            max_sources=max_sources,
            reject_linear_artifacts=True,
            proposal_mode=proposal_mode,
            refine_local_background=True,
        )
        primary = _nearest_source_to_peak(analysis.detection.sources, frame_primary_target, match_radius_px)
        secondary = _nearest_source_to_peak(analysis.detection.sources, frame_secondary_target, match_radius_px)
        nearest_primary_row = _nearest_source_with_distance(
            analysis.detection.sources,
            frame_primary_target,
            nearest_search_radius_px,
        )
        nearest_secondary_row = _nearest_source_with_distance(
            analysis.detection.sources,
            frame_secondary_target,
            nearest_search_radius_px,
        )
        nearest_primary = None if nearest_primary_row is None else nearest_primary_row[1]
        nearest_secondary = None if nearest_secondary_row is None else nearest_secondary_row[1]
        image = np.asarray(frame.data)
        auxiliary = auxiliary_mask(image.shape)
        negative_limit = _default_negative_overflow_limit(image)
        secondary_stats = _raw_region_statistics(
            image,
            frame_secondary_target,
            aperture_radius,
            negative_overflow_limit=negative_limit,
        )
        pair_stats = _raw_region_statistics(
            image,
            frame_midpoint,
            pair_window_radius,
            negative_overflow_limit=negative_limit,
        )
        range_mask = auxiliary.copy()
        sentinel_mask = auxiliary | (image == -1)
        if negative_limit is not None:
            range_mask |= image <= float(negative_limit)
            range_mask |= image >= -float(negative_limit)
        raw_bic, raw_component = _pair_peak_evidence(
            image,
            auxiliary,
            primary,
            secondary,
            frame_primary_target,
            frame_secondary_target,
            psf_fwhm=psf_fwhm,
        )
        range_bic, range_component = _pair_peak_evidence(
            image,
            range_mask,
            primary,
            secondary,
            frame_primary_target,
            frame_secondary_target,
            psf_fwhm=psf_fwhm,
        )
        sentinel_bic, sentinel_component = _pair_peak_evidence(
            image,
            sentinel_mask,
            primary,
            secondary,
            frame_primary_target,
            frame_secondary_target,
            psf_fwhm=psf_fwhm,
        )

        primary_peak = None if primary is None else float(primary.peak)
        secondary_peak = None if secondary is None else float(secondary.peak)
        primary_flux_snr = None if primary is None or primary.flux_snr is None else float(primary.flux_snr)
        secondary_flux_snr = None if secondary is None or secondary.flux_snr is None else float(secondary.flux_snr)
        primary_peak_xy = None if primary is None else (
            float(primary.peak_x if primary.peak_x is not None else primary.x),
            float(primary.peak_y if primary.peak_y is not None else primary.y),
        )
        secondary_peak_xy = None if secondary is None else (
            float(secondary.peak_x if secondary.peak_x is not None else secondary.x),
            float(secondary.peak_y if secondary.peak_y is not None else secondary.y),
        )
        detected_peak_distance = (
            None
            if primary_peak_xy is None or secondary_peak_xy is None
            else float(np.hypot(
                secondary_peak_xy[0] - primary_peak_xy[0],
                secondary_peak_xy[1] - primary_peak_xy[1],
            ))
        )
        detected_centroid_distance = (
            None
            if primary is None or secondary is None
            else float(np.hypot(float(secondary.x) - float(primary.x), float(secondary.y) - float(primary.y)))
        )
        nearest_source_peak_distance = (
            None
            if nearest_primary is None or nearest_secondary is None
            else float(np.hypot(
                float(nearest_secondary.peak_x if nearest_secondary.peak_x is not None else nearest_secondary.x)
                - float(nearest_primary.peak_x if nearest_primary.peak_x is not None else nearest_primary.x),
                float(nearest_secondary.peak_y if nearest_secondary.peak_y is not None else nearest_secondary.y)
                - float(nearest_primary.peak_y if nearest_primary.peak_y is not None else nearest_primary.y),
            ))
        )

        def finite_float(value: object) -> float | None:
            try:
                converted = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return converted if np.isfinite(converted) else None

        def flag_text(source: Detection | None) -> str:
            return "|".join(source.flags) if source is not None else "NOT_FOUND"

        rows.append(
            SourcePairAuditRow(
                frame_index=frame_index + 1,
                path=str(frame.path),
                frame_shift_x_px=shift_x,
                frame_shift_y_px=shift_y,
                target_peak_distance_px=target_distance,
                detected_peak_distance_px=detected_peak_distance,
                detected_centroid_distance_px=detected_centroid_distance,
                nearest_primary_distance_px=(
                    None if nearest_primary_row is None else float(nearest_primary_row[0])
                ),
                nearest_secondary_distance_px=(
                    None if nearest_secondary_row is None else float(nearest_secondary_row[0])
                ),
                nearest_source_peak_distance_px=nearest_source_peak_distance,
                nearest_source_same_for_targets=(
                    None
                    if nearest_primary is None or nearest_secondary is None
                    else bool(nearest_primary.detection_id == nearest_secondary.detection_id)
                ),
                nearest_secondary_detection_id=(
                    None if nearest_secondary is None else int(nearest_secondary.detection_id)
                ),
                nearest_secondary_peak_x=(
                    None
                    if nearest_secondary is None
                    else float(nearest_secondary.peak_x if nearest_secondary.peak_x is not None else nearest_secondary.x)
                ),
                nearest_secondary_peak_y=(
                    None
                    if nearest_secondary is None
                    else float(nearest_secondary.peak_y if nearest_secondary.peak_y is not None else nearest_secondary.y)
                ),
                nearest_secondary_peak_adu=(None if nearest_secondary is None else float(nearest_secondary.peak)),
                nearest_secondary_quality_passed=(
                    None if nearest_secondary is None else bool(nearest_secondary.quality_passed)
                ),
                nearest_secondary_flags=flag_text(nearest_secondary),
                primary_detection_id=None if primary is None else int(primary.detection_id),
                secondary_detection_id=None if secondary is None else int(secondary.detection_id),
                primary_peak_adu=primary_peak,
                secondary_peak_adu=secondary_peak,
                primary_flux_snr=finite_float(None if primary is None else primary_flux_snr),
                secondary_flux_snr=finite_float(None if secondary is None else secondary_flux_snr),
                primary_quality_passed=None if primary is None else bool(primary.quality_passed),
                secondary_quality_passed=None if secondary is None else bool(secondary.quality_passed),
                primary_feature_class=None if primary is None else classify_source_feature(primary),
                secondary_feature_class=None if secondary is None else classify_source_feature(secondary),
                primary_flags=flag_text(primary),
                secondary_flags=flag_text(secondary),
                secondary_aperture_min_adu=finite_float(secondary_stats["min"]),
                secondary_aperture_max_adu=finite_float(secondary_stats["max"]),
                secondary_aperture_median_adu=finite_float(secondary_stats["median"]),
                secondary_aperture_unique_values=int(secondary_stats["unique"]),
                secondary_mode_adu=finite_float(secondary_stats["mode"]),
                secondary_mode_count=int(secondary_stats["mode_count"]),
                secondary_mode_fraction=finite_float(secondary_stats["mode_fraction"]),
                secondary_fixed_minus_one_count=int(secondary_stats["fixed_minus_one"]),
                secondary_negative_overflow_count=int(secondary_stats["negative_overflow"]),
                secondary_positive_extreme_count=int(secondary_stats["positive_extreme"]),
                pair_window_negative_overflow_count=int(pair_stats["negative_overflow"]),
                pair_window_positive_extreme_count=int(pair_stats["positive_extreme"]),
                pair_delta_bic_raw=finite_float(raw_bic),
                pair_component_snr_raw=finite_float(raw_component),
                pair_delta_bic_range_masked=finite_float(range_bic),
                pair_component_snr_range_masked=finite_float(range_component),
                pair_delta_bic_sentinel_masked=finite_float(sentinel_bic),
                pair_component_snr_sentinel_masked=finite_float(sentinel_component),
            )
        )
        if progress is not None:
            progress(frame_index + 1, total)
        del analysis, frame

    found_secondary = sum(row.secondary_detection_id is not None for row in rows)
    quality_secondary = sum(row.secondary_quality_passed is True for row in rows)
    range_frames = sum(
        row.secondary_negative_overflow_count > 0 or row.secondary_positive_extreme_count > 0
        for row in rows
    )
    raw_pair_supported = sum(
        row.pair_delta_bic_raw is not None
        and row.pair_component_snr_raw is not None
        and row.pair_delta_bic_raw >= 10.0
        and row.pair_component_snr_raw >= 5.0
        for row in rows
    )
    nearest_found = sum(row.nearest_secondary_detection_id is not None for row in rows)
    nearest_same = sum(row.nearest_source_same_for_targets is True for row in rows)
    coordinate_label = "按累计平移调整的 detector 目标" if using_registered_targets else "固定 detector 坐标"
    coordinate_note = (
        "仅基于平移注册的 detector 目标诊断"
        if using_registered_targets
        else "固定 detector 坐标的合并/漂移诊断"
    )
    conclusion = (
        f"{coordinate_label}审计：副目标在 {found_secondary}/{total} 帧被候选表找到，"
        f"其中 {quality_secondary}/{total} 帧通过质量层；{range_frames}/{total} 帧的副孔径含"
        "极端正/负范围码。达到当前双 PSF 工程证据线的帧数为 "
        f"{raw_pair_supported}/{total}。放宽到目标周围 {nearest_search_radius_px:g} px 后，"
        f"附近候选在 {nearest_found}/{total} 帧出现，但两个目标查询落到同一检测源的情况有 "
        f"{nearest_same}/{total} 帧；这只是{coordinate_note}，不是天空坐标匹配。"
        "这些结果支持把两个框解释为‘两个局部提案，副响应需经过异常编码和 PSF 复核’，"
        "不能直接报告为两颗已确认恒星。"
    )
    return SourcePairAuditResult(
        frame_count=total,
        primary_target_xy=(float(primary_target_xy[0]), float(primary_target_xy[1])),
        secondary_target_xy=(float(secondary_target_xy[0]), float(secondary_target_xy[1])),
        target_peak_distance_px=target_distance,
        match_radius_px=float(match_radius_px),
        nearest_search_radius_px=float(nearest_search_radius_px),
        frame_shifts=resolved_frame_shifts,
        aperture_radius_px=int(aperture_radius),
        association_coordinate_system=(
            "registered detector coordinates from supplied translations; not WCS/sky matching"
            if using_registered_targets
            else "fixed detector coordinates; not WCS/sky matching"
        ),
        parameters={
            "threshold_sigma": float(threshold_sigma),
            "min_distance": int(min_distance),
            "aperture_radius": int(aperture_radius),
            "psf_fwhm": float(psf_fwhm),
            "background_box_size": int(background_box_size),
            "background_sample_limit": int(background_sample_limit),
            "min_flux_snr": float(min_flux_snr),
            "min_psf_support_pixels": int(min_psf_support_pixels),
            "proposal_mode": str(proposal_mode),
            "max_sources": max_sources,
            "match_radius_px": float(match_radius_px),
            "nearest_search_radius_px": float(nearest_search_radius_px),
            "frame_shifts": [list(shift) for shift in resolved_frame_shifts],
            "pair_psf_confirmation_line": "delta_bic >= 10 and component_snr >= 5; engineering audit only",
            "range_code_threshold": "signed integer negative <= -0.9 * dtype max; positive counterpart is diagnostic only",
        },
        rows=tuple(rows),
        conclusion=conclusion,
    )


def run_temporal_code_audit(
    paths: Iterable[str | Path],
    *,
    low_variation_span_adu: int = 2,
    code_min_adu: int = 3990,
    code_max_adu: int = 3993,
    sentinel_value: int = -1,
    code_focus_xy: tuple[int, int] | None = (1274, 3466),
    sentinel_focus_xy: tuple[int, int] | None = (1271, 3465),
    progress: Callable[[int, int], None] | None = None,
) -> TemporalCodeAuditResult:
    """审计 15 帧 detector 坐标中的固定值和低变化值。"""

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    if low_variation_span_adu < 0:
        raise ValueError("low_variation_span_adu cannot be negative")
    if code_min_adu > code_max_adu:
        raise ValueError("code_min_adu cannot exceed code_max_adu")

    def validate_focus(point: tuple[int, int] | None, label: str) -> tuple[int, int] | None:
        if point is None:
            return None
        if len(point) != 2:
            raise ValueError(f"{label} must contain x and y")
        try:
            x, y = int(point[0]), int(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain integer coordinates") from exc
        return (x, y)

    resolved_code_focus = validate_focus(code_focus_xy, "code_focus_xy")
    resolved_sentinel_focus = validate_focus(sentinel_focus_xy, "sentinel_focus_xy")
    total = len(frame_paths)
    first: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    for frame_index, path in enumerate(frame_paths):
        frame = read_fits(path)
        values = np.asarray(frame.data)
        if values.ndim != 2:
            raise ValueError(f"temporal code audit expects a 2-D FITS image: {path}")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"temporal code audit expects integer FITS pixels: {path}")
        numeric = np.asarray(values, dtype=np.int32)
        if first is None:
            first = numeric.copy()
            minimum = numeric.copy()
            maximum = numeric.copy()
        else:
            if numeric.shape != first.shape:
                raise ValueError("all FITS images must have the same shape")
            minimum = np.minimum(minimum, numeric)
            maximum = np.maximum(maximum, numeric)
        if progress is not None:
            progress(frame_index + 1, total)

    if first is None or minimum is None or maximum is None:
        raise RuntimeError("temporal code audit did not read any image")
    span = maximum - minimum

    def component_summary(
        mask: np.ndarray,
        kind: str,
        focus: tuple[int, int] | None,
    ) -> tuple[int, int, int, int | None, tuple[tuple[int, int], ...], tuple[dict[str, object], ...]]:
        labels, component_count = ndimage.label(
            mask,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        sizes = np.bincount(labels.ravel())[1:]
        focus_label = None
        focus_coordinates: tuple[tuple[int, int], ...] = ()
        focus_size: int | None = None
        if focus is not None:
            x, y = focus
            if 0 <= x < first.shape[1] and 0 <= y < first.shape[0]:
                label = int(labels[y, x])
                if label > 0:
                    focus_label = label
                    focus_size = int(sizes[label - 1])
                    yy, xx = np.where(labels == label)
                    focus_coordinates = tuple((int(px), int(py)) for py, px in zip(yy, xx))
        components: list[dict[str, object]] = []
        for label, size in enumerate(sizes, 1):
            yy, xx = np.where(labels == label)
            components.append(
                {
                    "kind": kind,
                    "component_id": int(label),
                    "size": int(size),
                    "x0": int(xx.min()),
                    "y0": int(yy.min()),
                    "x1": int(xx.max()),
                    "y1": int(yy.max()),
                    "is_focus_component": bool(label == focus_label),
                }
            )
        largest = int(sizes.max()) if sizes.size else 0
        at_least_two = int(np.count_nonzero(sizes >= 2))
        return (
            int(component_count),
            at_least_two,
            largest,
            focus_size,
            focus_coordinates,
            tuple(components),
        )

    exact_stable = span == 0
    stable_values, stable_counts = np.unique(first[exact_stable], return_counts=True)
    exact_value_counts = tuple(
        (int(value), int(count))
        for value, count in sorted(
            zip(stable_values, stable_counts, strict=True),
            key=lambda item: (-int(item[1]), int(item[0])),
        )
    )
    code_mask = (
        (first >= int(code_min_adu))
        & (first <= int(code_max_adu))
        & (span <= int(low_variation_span_adu))
    )
    sentinel_mask = (first == int(sentinel_value)) & exact_stable
    (
        code_component_count,
        code_components_ge_2,
        code_largest_component,
        code_focus_component_size,
        code_focus_coordinates,
        code_components,
    ) = component_summary(code_mask, "low_variation_code", resolved_code_focus)
    (
        sentinel_component_count,
        _sentinel_components_ge_2,
        sentinel_largest_component,
        sentinel_focus_component_size,
        sentinel_focus_coordinates,
        sentinel_components,
    ) = component_summary(sentinel_mask, "exact_stable_sentinel", resolved_sentinel_focus)
    del _sentinel_components_ge_2
    return TemporalCodeAuditResult(
        frame_count=total,
        image_shape=(int(first.shape[0]), int(first.shape[1])),
        paths=tuple(str(path) for path in frame_paths),
        exact_stable_pixel_count=int(np.count_nonzero(exact_stable)),
        exact_stable_value_counts=exact_value_counts,
        low_variation_span_adu=int(low_variation_span_adu),
        low_variation_pixel_count=int(np.count_nonzero(span <= int(low_variation_span_adu))),
        code_range_adu=(int(code_min_adu), int(code_max_adu)),
        code_pixel_count=int(np.count_nonzero(code_mask)),
        code_component_count=code_component_count,
        code_components_ge_2=code_components_ge_2,
        code_largest_component=code_largest_component,
        code_focus_xy=resolved_code_focus,
        code_focus_component_size=code_focus_component_size,
        code_focus_component_coordinates=code_focus_coordinates,
        sentinel_value=int(sentinel_value),
        sentinel_exact_stable_pixel_count=int(np.count_nonzero(sentinel_mask)),
        sentinel_component_count=sentinel_component_count,
        sentinel_largest_component=sentinel_largest_component,
        sentinel_focus_xy=resolved_sentinel_focus,
        sentinel_focus_component_size=sentinel_focus_component_size,
        sentinel_focus_component_coordinates=sentinel_focus_coordinates,
        components=tuple((*code_components, *sentinel_components)),
        interpretation_boundary=(
            "低变化/相邻特殊值是固定结构候选，不是已由 FITS 格式说明确认的坏点或物理编码；"
            "需主办方格式、暗场/坏点资料和注入验证。"
        ),
    )


def run_injection_recovery(
    *,
    peak_levels: Sequence[float] = (8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 128.0),
    trials_per_level: int = 4,
    sources_per_trial: int = 24,
    image_shape: tuple[int, int] = (256, 256),
    background_adu: float = 21.0,
    noise_sigma_adu: float = 5.0,
    psf_fwhm: float = 3.0,
    injected_psf_fwhm: float | None = None,
    proposal_mode: str = "gaussian",
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    seed: int = 19019,
) -> tuple[InjectionRecoveryRow, ...]:
    """在可控 Gaussian 噪声图上测试候选层与质量层的召回率。"""

    if trials_per_level < 1 or sources_per_trial < 1:
        raise ValueError("trials_per_level and sources_per_trial must be positive")
    resolved_injected_fwhm = float(psf_fwhm if injected_psf_fwhm is None else injected_psf_fwhm)
    if psf_fwhm <= 0 or resolved_injected_fwhm <= 0 or noise_sigma_adu <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("psf_fwhm/noise_sigma_adu/min_psf_support_pixels are invalid")
    rng = np.random.default_rng(seed)
    rows: list[InjectionRecoveryRow] = []
    for level in peak_levels:
        peak = float(level)
        if peak <= 0:
            raise ValueError("peak_levels must be positive")
        candidate_recovered = 0
        quality_recovered = 0
        injected_total = 0
        candidate_counts: list[int] = []
        quality_counts: list[int] = []
        for _trial in range(trials_per_level):
            image = rng.normal(background_adu, noise_sigma_adu, image_shape).astype(np.float32)
            injected = _random_positions(rng, sources_per_trial, image_shape)
            for x, y in injected:
                _gaussian_source(image, x, y, peak, resolved_injected_fwhm)
            detection = detect_sources(
                image,
                mask=auxiliary_mask(image.shape),
                threshold_sigma=threshold_sigma,
                min_distance=min_distance,
                aperture_radius=4,
                psf_fwhm=psf_fwhm,
                background_box_size=64,
                min_flux_snr=min_flux_snr,
                min_psf_support_pixels=min_psf_support_pixels,
                reject_linear_artifacts=True,
                proposal_mode=proposal_mode,
            )
            candidate_recovered += _matched_count(injected, detection.sources, max(3.0, psf_fwhm))
            quality_recovered += _matched_count(injected, detection.quality_sources, max(3.0, psf_fwhm))
            injected_total += len(injected)
            candidate_counts.append(detection.returned_count)
            quality_counts.append(detection.star_count)
        rows.append(
            InjectionRecoveryRow(
                proposal_mode=str(proposal_mode),
                detector_psf_fwhm_px=float(psf_fwhm),
                injected_psf_fwhm_px=float(resolved_injected_fwhm),
                peak_excess_adu=peak,
                injected_count=injected_total,
                candidate_recovered_count=candidate_recovered,
                quality_recovered_count=quality_recovered,
                candidate_recall=candidate_recovered / injected_total,
                quality_recall=quality_recovered / injected_total,
                mean_candidate_count=float(np.mean(candidate_counts)),
                mean_quality_count=float(np.mean(quality_counts)),
            )
        )
    return tuple(rows)


def run_real_background_injection(
    path: str | Path | FitsFrame,
    *,
    peak_levels: Sequence[float] = (12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 128.0),
    trials_per_level: int = 2,
    sources_per_trial: int = 16,
    psf_fwhm: float = 3.0,
    proposal_mode: str = "gaussian",
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
    empirical_psf_radius: int = 7,
    empirical_psf_sources: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[RealBackgroundInjectionRow, ...]:
    """在真实 FITS 背景上注入点源并测量候选/质量层召回率。

    原始 FitsFrame 不会被修改；每个强度档位只在内存副本上叠加
    Gaussian PSF。实验先对未注入原图运行一次基线检测，然后在远离已有
    候选且局部高分位超额较低的位置注入源。由于真实背景本身没有逐星
    真值，输出 background_* 和 net_*_delta，而不把背景检测数
    冒充 precision/false-positive。
    """

    if trials_per_level < 1 or sources_per_trial < 1:
        raise ValueError("trials_per_level and sources_per_trial must be positive")
    if psf_fwhm <= 0 or threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1:
        raise ValueError("real-background injection parameters must be positive")
    if background_box_size < 16 or min_flux_snr <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("background_box_size/min_flux_snr/min_psf_support_pixels are invalid")
    if match_radius_px is not None and match_radius_px <= 0:
        raise ValueError("match_radius_px must be positive when provided")
    if psf_model not in {"gaussian", "empirical"}:
        raise ValueError("psf_model must be gaussian or empirical")
    if empirical_psf_radius < 3 or empirical_psf_sources < 1:
        raise ValueError("empirical PSF radius must be at least 3 and source count must be positive")
    if not peak_levels:
        raise ValueError("peak_levels cannot be empty")

    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    # 保留原始存储 dtype 做基线检测；否则 int16 图像转成 float32 后，
    # 检测器无法沿用其 dtype 推导的边界/饱和掩膜。
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
    image_mask = auxiliary_mask(image.shape)
    detector_options = {
        "mask": image_mask,
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
    empirical_psf = (
        estimate_empirical_psf(
            image,
            # 传入完整候选表，而不是只传质量子集。函数内部会先要求
            # source.quality_passed，再用所有候选检查近邻；如果这里只传
            # quality_sources，被拒绝但仍会污染 PSF 的邻峰会从隔离审计中
            # 消失，经验核可能再次吸收拥挤/伪影结构。
            baseline.sources,
            support_radius=empirical_psf_radius,
            max_sources=empirical_psf_sources,
        )
        if psf_model == "empirical"
        else None
    )
    if psf_model == "empirical" and empirical_psf is None:
        raise RuntimeError("真实质量源不足，无法建立实测 PSF；请改用 gaussian 或放宽 PSF 样本条件")
    injected_detector_options = dict(detector_options)
    saturation_level = float(baseline.parameters.get("saturation_level", -1.0))
    injected_detector_options["saturation_level"] = saturation_level if saturation_level > 0 else None
    negative_overflow_limit = baseline.parameters.get("negative_overflow_limit")
    injected_detector_options["negative_overflow_limit"] = (
        None if negative_overflow_limit is None else float(negative_overflow_limit)
    )
    injected_detector_options["mask_zero_pixels"] = bool(int(baseline.parameters.get("mask_zero_pixels", 0)))
    rng = np.random.default_rng(seed)
    recovery_radius = max(3.0, float(match_radius_px if match_radius_px is not None else psf_fwhm))
    protection_radius = max(8.0, 2.0 * float(psf_fwhm))
    layouts = [
        _select_real_background_positions(
            image,
            baseline.sources,
            rng,
            sources_per_trial,
            noise_adu=float(baseline.noise),
            source_exclusion_radius_px=max(12.0, 2.0 * float(psf_fwhm)),
        )
        for _trial in range(trials_per_level)
    ]
    rows: list[RealBackgroundInjectionRow] = []
    total_levels = len(peak_levels)
    for level_index, level in enumerate(peak_levels, start=1):
        if progress is not None:
            progress(level_index, total_levels)
        peak = float(level)
        if peak <= 0:
            raise ValueError("peak_levels must be positive")
        injected_total = 0
        candidate_recovered = 0
        quality_recovered = 0
        candidate_counts: list[int] = []
        quality_counts: list[int] = []
        background_candidate_counts: list[int] = []
        background_quality_counts: list[int] = []
        local_backgrounds: list[float] = []
        local_noises: list[float] = []
        for injected, local_stats in layouts:
            test_image = np.asarray(image, dtype=np.float32).copy()
            for x, y in injected:
                if empirical_psf is None:
                    _gaussian_source(test_image, x, y, peak, psf_fwhm)
                else:
                    _empirical_source(test_image, x, y, peak, empirical_psf)
            detection = detect_sources(test_image, **injected_detector_options)
            candidate_recovered += _matched_count(injected, detection.sources, recovery_radius)
            quality_recovered += _matched_count(injected, detection.quality_sources, recovery_radius)
            injected_total += len(injected)
            candidate_counts.append(detection.candidate_count)
            quality_counts.append(detection.star_count)
            background_candidate_counts.append(
                _count_outside_injection_regions(detection.sources, injected, protection_radius)
            )
            background_quality_counts.append(
                _count_outside_injection_regions(detection.quality_sources, injected, protection_radius)
            )
            local_backgrounds.extend(background for background, _noise in local_stats)
            local_noises.extend(noise for _background, noise in local_stats)
        mean_candidate = float(np.mean(candidate_counts))
        mean_quality = float(np.mean(quality_counts))
        rows.append(
            RealBackgroundInjectionRow(
                source_path=str(frame.path),
                proposal_mode=str(proposal_mode),
                psf_model=psf_model,
                psf_source_count=0 if empirical_psf is None else empirical_psf.source_count,
                psf_median_fwhm_px=None if empirical_psf is None else empirical_psf.median_fwhm_px,
                psf_kernel_sum=None if empirical_psf is None else float(np.sum(empirical_psf.kernel)),
                peak_excess_adu=peak,
                injected_count=injected_total,
                candidate_recovered_count=candidate_recovered,
                quality_recovered_count=quality_recovered,
                candidate_recall=candidate_recovered / injected_total,
                quality_recall=quality_recovered / injected_total,
                baseline_candidate_count=baseline.candidate_count,
                baseline_quality_count=baseline.star_count,
                mean_candidate_count=mean_candidate,
                mean_quality_count=mean_quality,
                mean_background_candidate_count=float(np.mean(background_candidate_counts)),
                mean_background_quality_count=float(np.mean(background_quality_counts)),
                net_candidate_delta=mean_candidate - float(baseline.candidate_count),
                net_quality_delta=mean_quality - float(baseline.star_count),
                local_background_adu=float(np.mean(local_backgrounds)),
                local_noise_adu=float(np.mean(local_noises)),
                trial_count=trials_per_level,
            )
        )
    return tuple(rows)


def run_stratified_real_background_injection(
    path: str | Path | FitsFrame,
    *,
    strata: Sequence[str] = ("blank", "high_background", "edge", "crowded", "special_code"),
    peak_levels: Sequence[float] = (24.0, 56.0, 128.0),
    trials_per_level: int = 1,
    sources_per_trial: int = 4,
    psf_fwhm: float = 3.0,
    proposal_mode: str = "gaussian",
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
    empirical_psf_radius: int = 7,
    empirical_psf_sources: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[StratifiedRealBackgroundInjectionRow, ...]:
    """在真实 FITS 背景的不同图像条件下做分层注入-回收。

    该实验把“源亮度”和“背景条件”拆开。``blank`` 与
    ``high_background`` 远离基线候选，分别代表安静和局部噪声较高的
    背景；``edge``、``crowded``、``special_code`` 和可选的 ``line``
    故意保留困难条件。每个位置都保存注入前的局部统计和基线邻域信息。

    输出的 candidate/quality recall 是注入真值的回收率；它们不等于真实
    图像的 precision、误检率或物理恒星完备率。尤其是 ``crowded``、
    ``special_code`` 和 ``line`` 层，``ambiguous_injection_count`` 大于
    零时只能解释为困难条件下的可检出性审计。原始 FITS 和传入的
    ``FitsFrame`` 都不会被修改。
    """

    resolved_strata = tuple(str(value).strip().lower() for value in strata)
    if not resolved_strata:
        raise ValueError("strata cannot be empty")
    if len(set(resolved_strata)) != len(resolved_strata):
        raise ValueError("strata must not contain duplicates")
    unknown_strata = [value for value in resolved_strata if value not in _REAL_INJECTION_STRATUM_LABELS]
    if unknown_strata:
        allowed = ", ".join(sorted(_REAL_INJECTION_STRATUM_LABELS))
        raise ValueError(f"unknown real-background injection strata {unknown_strata!r}; choose from {allowed}")
    if trials_per_level < 1 or sources_per_trial < 1:
        raise ValueError("trials_per_level and sources_per_trial must be positive")
    if psf_fwhm <= 0 or threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1:
        raise ValueError("stratified real-background injection parameters must be positive")
    if background_box_size < 16 or min_flux_snr <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("background_box_size/min_flux_snr/min_psf_support_pixels are invalid")
    if match_radius_px is not None and match_radius_px <= 0:
        raise ValueError("match_radius_px must be positive when provided")
    if psf_model not in {"gaussian", "empirical"}:
        raise ValueError("psf_model must be gaussian or empirical")
    if empirical_psf_radius < 3 or empirical_psf_sources < 1:
        raise ValueError("empirical PSF radius must be at least 3 and source count must be positive")
    if not peak_levels:
        raise ValueError("peak_levels cannot be empty")
    resolved_peaks = tuple(float(level) for level in peak_levels)
    if any(level <= 0 for level in resolved_peaks):
        raise ValueError("peak_levels must be positive")

    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
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
    empirical_psf = (
        estimate_empirical_psf(
            image,
            baseline.sources,
            support_radius=empirical_psf_radius,
            max_sources=empirical_psf_sources,
        )
        if psf_model == "empirical"
        else None
    )
    if psf_model == "empirical" and empirical_psf is None:
        raise RuntimeError("真实质量源不足，无法建立实测 PSF；请改用 gaussian 或增加可用模板")
    injected_detector_options = dict(detector_options)
    saturation_level = float(baseline.parameters.get("saturation_level", -1.0))
    injected_detector_options["saturation_level"] = saturation_level if saturation_level > 0 else None
    negative_overflow_limit = baseline.parameters.get("negative_overflow_limit")
    injected_detector_options["negative_overflow_limit"] = (
        None if negative_overflow_limit is None else float(negative_overflow_limit)
    )
    injected_detector_options["mask_zero_pixels"] = bool(int(baseline.parameters.get("mask_zero_pixels", 0)))

    rng = np.random.default_rng(seed)
    site_layouts: dict[str, list[tuple[_RealInjectionSite, ...]]] = {}
    for stratum in resolved_strata:
        site_layouts[stratum] = [
            _select_stratified_real_injection_sites(
                image,
                baseline.sources,
                rng,
                sources_per_trial,
                stratum=stratum,
                noise_adu=float(baseline.noise),
                psf_fwhm=psf_fwhm,
                aperture_radius=aperture_radius,
            )
            for _trial in range(trials_per_level)
        ]
    site_condition_json = {
        stratum: json.dumps(
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
                for trial_index, sites in enumerate(site_layouts[stratum], start=1)
                for site_index, site in enumerate(sites, start=1)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        for stratum in resolved_strata
    }

    recovery_radius = max(3.0, float(match_radius_px if match_radius_px is not None else psf_fwhm))
    protection_radius = max(8.0, 2.0 * float(psf_fwhm))
    total_work = len(resolved_strata) * len(resolved_peaks) * trials_per_level
    completed_work = 0
    if progress is not None:
        progress(0, total_work)
    rows: list[StratifiedRealBackgroundInjectionRow] = []
    for stratum in resolved_strata:
        for peak in resolved_peaks:
            injected_total = 0
            candidate_recovered = 0
            quality_recovered = 0
            ambiguous_total = 0
            unambiguous_total = 0
            candidate_counts: list[int] = []
            quality_counts: list[int] = []
            background_candidate_counts: list[int] = []
            background_quality_counts: list[int] = []
            local_backgrounds: list[float] = []
            local_noises: list[float] = []
            special_fractions: list[float] = []
            nearest_distances: list[float] = []
            for sites in site_layouts[stratum]:
                test_image = np.asarray(image, dtype=np.float32).copy()
                positions = [(site.x, site.y) for site in sites]
                unambiguous_sites = tuple(
                    site
                    for site in sites
                    if site.nearest_baseline_source_px is None
                    or site.nearest_baseline_source_px > recovery_radius
                )
                unambiguous_positions = [(site.x, site.y) for site in unambiguous_sites]
                for site in sites:
                    if empirical_psf is None:
                        _gaussian_source(test_image, site.x, site.y, peak, psf_fwhm)
                    else:
                        _empirical_source(test_image, site.x, site.y, peak, empirical_psf)
                detection = detect_sources(test_image, **injected_detector_options)
                # 注入位置若已在基线源的直接匹配半径内，检测结果可能只是
                # 原有源；这类位置不计入严格新增源回收，避免特殊值域/拥挤
                # 层把既有邻源误算成注入真值。
                candidate_recovered += _matched_count(unambiguous_positions, detection.sources, recovery_radius)
                quality_recovered += _matched_count(
                    unambiguous_positions,
                    detection.quality_sources,
                    recovery_radius,
                )
                injected_total += len(sites)
                ambiguous_total += sum(site.baseline_neighbor_count > 0 for site in sites)
                unambiguous_total += len(unambiguous_sites)
                candidate_counts.append(detection.candidate_count)
                quality_counts.append(detection.star_count)
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
            mean_candidate = float(np.mean(candidate_counts))
            mean_quality = float(np.mean(quality_counts))
            rows.append(
                StratifiedRealBackgroundInjectionRow(
                    source_path=str(frame.path),
                    stratum=stratum,
                    stratum_label=_REAL_INJECTION_STRATUM_LABELS[stratum],
                    proposal_mode=str(proposal_mode),
                    psf_model=psf_model,
                    psf_source_count=0 if empirical_psf is None else empirical_psf.source_count,
                    psf_median_fwhm_px=None if empirical_psf is None else empirical_psf.median_fwhm_px,
                    psf_kernel_sum=None if empirical_psf is None else float(np.sum(empirical_psf.kernel)),
                    peak_excess_adu=peak,
                    injected_count=injected_total,
                    candidate_recovered_count=candidate_recovered,
                    quality_recovered_count=quality_recovered,
                    candidate_recall=candidate_recovered / injected_total,
                    quality_recall=quality_recovered / injected_total,
                    ambiguous_injection_count=ambiguous_total,
                    unambiguous_injected_count=unambiguous_total,
                    unambiguous_candidate_recall=(
                        candidate_recovered / unambiguous_total if unambiguous_total else None
                    ),
                    unambiguous_quality_recall=(
                        quality_recovered / unambiguous_total if unambiguous_total else None
                    ),
                    baseline_candidate_count=baseline.candidate_count,
                    baseline_quality_count=baseline.star_count,
                    mean_candidate_count=mean_candidate,
                    mean_quality_count=mean_quality,
                    mean_background_candidate_count=float(np.mean(background_candidate_counts)),
                    mean_background_quality_count=float(np.mean(background_quality_counts)),
                    net_candidate_delta=mean_candidate - float(baseline.candidate_count),
                    net_quality_delta=mean_quality - float(baseline.star_count),
                    local_background_adu=float(np.mean(local_backgrounds)),
                    local_noise_adu=float(np.mean(local_noises)),
                    special_pixel_fraction=float(np.mean(special_fractions)),
                    nearest_baseline_source_px=(
                        float(np.median(nearest_distances)) if nearest_distances else None
                    ),
                    site_condition_json=site_condition_json[stratum],
                    trial_count=trials_per_level,
                    note=(
                        "该层按注入前图像条件选点；候选/质量回收是已知注入真值的召回率，"
                        "不是当前真实 FITS 的 precision 或逐星完备率。"
                    ),
                )
            )
    return tuple(rows)


def run_proposal_mode_comparison(
    path: str | Path | FitsFrame,
    *,
    peak_levels: Sequence[float] = (12.0, 24.0, 56.0),
    trials_per_level: int = 2,
    sources_per_trial: int = 12,
    psf_fwhm: float = 3.0,
    injected_psf_fwhm: float | None = None,
    pair_separation_px: float | None = None,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    seed: int = 19019,
) -> tuple[ProposalModeComparisonRow, ...]:
    """在完全相同的真实背景和注入坐标上比较三种提案模式。

    注入位置由 ensemble 候选并集确定，因此不会让 Gaussian/DoG 模式因为
    候选较少而抽到一组更容易的位置。两个模式消费同一张内存图像副本，
    候选与质量召回差异才可归因于提案/细筛逻辑。
    """

    if trials_per_level < 1 or sources_per_trial < 1:
        raise ValueError("trials_per_level and sources_per_trial must be positive")
    resolved_injected_fwhm = float(psf_fwhm if injected_psf_fwhm is None else injected_psf_fwhm)
    if (
        psf_fwhm <= 0
        or resolved_injected_fwhm <= 0
        or threshold_sigma <= 0
        or min_distance < 1
        or aperture_radius < 1
    ):
        raise ValueError("proposal comparison parameters must be positive")
    if pair_separation_px is not None:
        if pair_separation_px <= 0:
            raise ValueError("pair_separation_px must be positive when provided")
        if sources_per_trial % 2:
            raise ValueError("sources_per_trial must be even for paired injections")
    if not peak_levels:
        raise ValueError("peak_levels cannot be empty")
    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
    mask = auxiliary_mask(image.shape)
    common = {
        "mask": mask,
        "threshold_sigma": threshold_sigma,
        "min_distance": min_distance,
        "aperture_radius": aperture_radius,
        "psf_fwhm": psf_fwhm,
        "background_box_size": background_box_size,
        "min_flux_snr": min_flux_snr,
        "min_psf_support_pixels": min_psf_support_pixels,
        "reject_linear_artifacts": True,
    }
    ensemble_baseline = detect_sources(image, proposal_mode="ensemble", **common)
    rng = np.random.default_rng(seed)
    layouts: list[list[tuple[float, float]]] = []
    for _trial in range(trials_per_level):
        anchor_count = sources_per_trial if pair_separation_px is None else sources_per_trial // 2
        anchors = _select_real_background_positions(
            image,
            ensemble_baseline.sources,
            rng,
            anchor_count,
            noise_adu=float(ensemble_baseline.noise),
            margin=max(14, int(np.ceil((pair_separation_px or 0.0) / 2.0)) + 10),
            source_exclusion_radius_px=max(12.0, 2.0 * float(psf_fwhm) + float(pair_separation_px or 0.0)),
        )[0]
        if pair_separation_px is None:
            layouts.append(anchors)
            continue
        paired: list[tuple[float, float]] = []
        half_separation = 0.5 * float(pair_separation_px)
        for anchor_x, anchor_y in anchors:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            offset_x = half_separation * float(np.cos(angle))
            offset_y = half_separation * float(np.sin(angle))
            paired.extend(
                [
                    (anchor_x - offset_x, anchor_y - offset_y),
                    (anchor_x + offset_x, anchor_y + offset_y),
                ]
            )
        layouts.append(paired)
    recovery_radius = max(3.0, float(psf_fwhm))
    rows: list[ProposalModeComparisonRow] = []
    for level in peak_levels:
        peak = float(level)
        if peak <= 0:
            raise ValueError("peak_levels must be positive")
        recovered = {
            "gaussian": {"candidate": 0, "quality": 0, "candidate_counts": [], "quality_counts": []},
            "hybrid": {"candidate": 0, "quality": 0, "candidate_counts": [], "quality_counts": []},
            "ensemble": {"candidate": 0, "quality": 0, "candidate_counts": [], "quality_counts": []},
        }
        injected_total = 0
        for injected in layouts:
            test_image = np.asarray(image, dtype=np.float32).copy()
            for x, y in injected:
                _gaussian_source(test_image, x, y, peak, resolved_injected_fwhm)
            injected_total += len(injected)
            for mode in ("gaussian", "hybrid", "ensemble"):
                detection = detect_sources(test_image, proposal_mode=mode, **common)
                recovered[mode]["candidate"] += _matched_count(injected, detection.sources, recovery_radius)
                recovered[mode]["quality"] += _matched_count(injected, detection.quality_sources, recovery_radius)
                recovered[mode]["candidate_counts"].append(detection.candidate_count)
                recovered[mode]["quality_counts"].append(detection.star_count)
        rows.append(
            ProposalModeComparisonRow(
                source_path=str(frame.path),
                detector_psf_fwhm_px=float(psf_fwhm),
                injected_psf_fwhm_px=resolved_injected_fwhm,
                pair_separation_px=None if pair_separation_px is None else float(pair_separation_px),
                peak_excess_adu=peak,
                injected_count=injected_total,
                gaussian_candidate_recall=float(recovered["gaussian"]["candidate"]) / injected_total,
                hybrid_candidate_recall=float(recovered["hybrid"]["candidate"]) / injected_total,
                ensemble_candidate_recall=float(recovered["ensemble"]["candidate"]) / injected_total,
                gaussian_quality_recall=float(recovered["gaussian"]["quality"]) / injected_total,
                hybrid_quality_recall=float(recovered["hybrid"]["quality"]) / injected_total,
                ensemble_quality_recall=float(recovered["ensemble"]["quality"]) / injected_total,
                gaussian_mean_candidate_count=float(np.mean(recovered["gaussian"]["candidate_counts"])),
                hybrid_mean_candidate_count=float(np.mean(recovered["hybrid"]["candidate_counts"])),
                ensemble_mean_candidate_count=float(np.mean(recovered["ensemble"]["candidate_counts"])),
                gaussian_mean_quality_count=float(np.mean(recovered["gaussian"]["quality_counts"])),
                hybrid_mean_quality_count=float(np.mean(recovered["hybrid"]["quality_counts"])),
                ensemble_mean_quality_count=float(np.mean(recovered["ensemble"]["quality_counts"])),
                trial_count=trials_per_level,
            )
        )
    return tuple(rows)


def run_pair_flux_ratio_audit(
    path: str | Path | FitsFrame,
    *,
    total_peak_levels: Sequence[float] = (128.0, 256.0, 512.0),
    secondary_to_primary_ratios: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
    trials_per_condition: int = 1,
    pairs_per_trial: int = 4,
    pair_separation_px: float = 5.4,
    psf_fwhm: float = 3.0,
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
    signal_normalization: str = "peak_excess",
    empirical_psf_radius: int = 7,
    empirical_psf_sources: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[PairFluxRatioAuditRow, ...]:
    """在同一真实背景上审计双源强弱比对“解析为两颗”的影响。

    ``signal_normalization=peak_excess`` 时保持总峰值超额
    ``primary + secondary`` 不变；``integrated_excess`` 时保持离散注入
    小窗积分超额不变。两种模式都只改变 ``secondary / primary``。注入
    位置、双源方向和背景窗口在所有比值及强度档位之间复用，因而可以
    把“总信号变了”和“弱源变弱了”分开。后者还避免不同 PSF 的离散核
    和差异混入比较。
    ``source_*_recall`` 是单个真值的回收率；``pair_*_resolution_recall``
    要求一对真值由两个不同检测候选分别命中，是判断双峰是否被真正解析
    的主要指标。实验不把未注入真实 FITS 中的候选数解释成 precision。
    """

    if trials_per_condition < 1 or pairs_per_trial < 1:
        raise ValueError("trials_per_condition and pairs_per_trial must be positive")
    if pair_separation_px <= 0 or psf_fwhm <= 0 or threshold_sigma <= 0:
        raise ValueError("pair flux-ratio audit geometry and thresholds must be positive")
    if injected_psf_fwhm is not None and injected_psf_fwhm <= 0:
        raise ValueError("injected_psf_fwhm must be positive when provided")
    if min_distance < 1 or aperture_radius < 1 or background_box_size < 16:
        raise ValueError("pair flux-ratio audit detector parameters are invalid")
    if min_flux_snr <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_flux_snr/min_psf_support_pixels are invalid")
    if match_radius_px is not None and match_radius_px <= 0:
        raise ValueError("match_radius_px must be positive when provided")
    if psf_model not in {"gaussian", "empirical"}:
        raise ValueError("psf_model must be gaussian or empirical")
    if signal_normalization not in {"peak_excess", "integrated_excess"}:
        raise ValueError("signal_normalization must be peak_excess or integrated_excess")
    if empirical_psf_radius < 3 or empirical_psf_sources < 1:
        raise ValueError("empirical PSF radius must be at least 3 and source count must be positive")
    if not total_peak_levels:
        raise ValueError("total_peak_levels cannot be empty")
    if not secondary_to_primary_ratios:
        raise ValueError("secondary_to_primary_ratios cannot be empty")
    resolved_peaks = tuple(float(value) for value in total_peak_levels)
    resolved_ratios = tuple(float(value) for value in secondary_to_primary_ratios)
    if any(value <= 0 for value in resolved_peaks):
        raise ValueError("total_peak_levels must be positive")
    # 让 primary 保持“较亮或等亮”标签；大于 1 的情况可以通过交换标签
    # 表达，但会破坏读者对 secondary 的直觉，因此直接要求调用方排序。
    if any(value <= 0 or value > 1 for value in resolved_ratios):
        raise ValueError("secondary_to_primary_ratios must be in (0, 1]")

    resolved_injected_fwhm = float(psf_fwhm if injected_psf_fwhm is None else injected_psf_fwhm)
    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
    image_mask = auxiliary_mask(image.shape)
    detector_options = {
        "mask": image_mask,
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
    empirical_psf = (
        estimate_empirical_psf(
            image,
            baseline.sources,
            support_radius=empirical_psf_radius,
            max_sources=empirical_psf_sources,
        )
        if psf_model == "empirical"
        else None
    )
    if psf_model == "empirical" and empirical_psf is None:
        raise RuntimeError("真实质量源不足，无法建立实测 PSF；请改用 gaussian 或增加可用模板")

    injected_detector_options = dict(detector_options)
    saturation_level = float(baseline.parameters.get("saturation_level", -1.0))
    injected_detector_options["saturation_level"] = saturation_level if saturation_level > 0 else None
    negative_overflow_limit = baseline.parameters.get("negative_overflow_limit")
    injected_detector_options["negative_overflow_limit"] = (
        None if negative_overflow_limit is None else float(negative_overflow_limit)
    )
    injected_detector_options["mask_zero_pixels"] = bool(int(baseline.parameters.get("mask_zero_pixels", 0)))

    # 只生成一次布局：不同强度和比值看到同一批真实背景窗口。
    rng = np.random.default_rng(seed)
    layouts: list[tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[tuple[float, float]]]] = []
    for _trial in range(trials_per_condition):
        anchors, local_stats = _select_real_background_positions(
            image,
            baseline.sources,
            rng,
            pairs_per_trial,
            noise_adu=float(baseline.noise),
            margin=max(14, int(np.ceil(pair_separation_px / 2.0)) + 10),
            # 将锚点再向外推半个分离距离，避免双源端点重新落回基线源附近。
            source_exclusion_radius_px=max(16.0, 2.0 * float(psf_fwhm) + float(pair_separation_px)),
        )
        pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        half_separation = 0.5 * float(pair_separation_px)
        for anchor_x, anchor_y in anchors:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            offset_x = half_separation * float(np.cos(angle))
            offset_y = half_separation * float(np.sin(angle))
            pairs.append(
                (
                    (anchor_x - offset_x, anchor_y - offset_y),
                    (anchor_x + offset_x, anchor_y + offset_y),
                )
            )
        layouts.append((pairs, local_stats))

    recovery_radius = max(3.0, float(match_radius_px if match_radius_px is not None else psf_fwhm))
    total_work = len(resolved_peaks) * len(resolved_ratios) * trials_per_condition
    completed_work = 0
    if progress is not None:
        progress(0, total_work)
    rows: list[PairFluxRatioAuditRow] = []
    for total_peak in resolved_peaks:
        for ratio in resolved_ratios:
            primary_control_signal = total_peak / (1.0 + ratio)
            secondary_control_signal = total_peak - primary_control_signal
            primary_peak = (
                primary_control_signal if signal_normalization == "peak_excess" else None
            )
            secondary_peak = (
                secondary_control_signal if signal_normalization == "peak_excess" else None
            )
            primary_candidate_hits = 0
            secondary_candidate_hits = 0
            primary_quality_hits = 0
            secondary_quality_hits = 0
            candidate_pair_resolved = 0
            quality_pair_resolved = 0
            merged_candidate_pairs = 0.0
            merged_quality_pairs = 0.0
            pair_total = 0
            candidate_counts: list[int] = []
            quality_counts: list[int] = []
            injected_primary_peaks: list[float] = []
            injected_secondary_peaks: list[float] = []
            local_backgrounds: list[float] = []
            local_noises: list[float] = []
            for pairs, local_stats in layouts:
                test_image = np.asarray(image, dtype=np.float32).copy()
                for (primary_x, primary_y), (secondary_x, secondary_y) in pairs:
                    injected_primary_peaks.append(
                        _inject_source_signal(
                            test_image,
                            primary_x,
                            primary_y,
                            primary_control_signal,
                            signal_normalization=signal_normalization,
                            psf_model=psf_model,
                            gaussian_fwhm=resolved_injected_fwhm,
                            empirical_psf=empirical_psf,
                        )
                    )
                    injected_secondary_peaks.append(
                        _inject_source_signal(
                            test_image,
                            secondary_x,
                            secondary_y,
                            secondary_control_signal,
                            signal_normalization=signal_normalization,
                            psf_model=psf_model,
                            gaussian_fwhm=resolved_injected_fwhm,
                            empirical_psf=empirical_psf,
                        )
                    )
                detection = detect_sources(test_image, **injected_detector_options)
                flat_targets = [point for pair in pairs for point in pair]
                candidate_flags = _match_target_flags(flat_targets, detection.sources, recovery_radius)
                quality_flags = _match_target_flags(flat_targets, detection.quality_sources, recovery_radius)
                primary_candidate_hits += sum(candidate_flags[0::2])
                secondary_candidate_hits += sum(candidate_flags[1::2])
                primary_quality_hits += sum(quality_flags[0::2])
                secondary_quality_hits += sum(quality_flags[1::2])
                candidate_pair_resolved += sum(
                    candidate_flags[index] and candidate_flags[index + 1]
                    for index in range(0, len(candidate_flags), 2)
                )
                quality_pair_resolved += sum(
                    quality_flags[index] and quality_flags[index + 1]
                    for index in range(0, len(quality_flags), 2)
                )
                merged_candidate_pairs += _merged_pair_fraction(pairs, detection.sources, recovery_radius) * len(pairs)
                merged_quality_pairs += _merged_pair_fraction(pairs, detection.quality_sources, recovery_radius) * len(pairs)
                pair_total += len(pairs)
                candidate_counts.append(detection.candidate_count)
                quality_counts.append(detection.star_count)
                local_backgrounds.extend(background for background, _noise in local_stats)
                local_noises.extend(noise for _background, noise in local_stats)
                completed_work += 1
                if progress is not None:
                    progress(completed_work, total_work)
            injected_total = 2 * pair_total
            rows.append(
                PairFluxRatioAuditRow(
                    source_path=str(frame.path),
                    proposal_mode=str(proposal_mode),
                    psf_model=psf_model,
                    signal_normalization=signal_normalization,
                    psf_source_count=0 if empirical_psf is None else empirical_psf.source_count,
                    psf_median_fwhm_px=None if empirical_psf is None else empirical_psf.median_fwhm_px,
                    psf_kernel_sum=None if empirical_psf is None else float(np.sum(empirical_psf.kernel)),
                    detector_psf_fwhm_px=float(psf_fwhm),
                    injected_psf_fwhm_px=resolved_injected_fwhm,
                    pair_separation_px=float(pair_separation_px),
                    total_peak_excess_adu=(
                        total_peak if signal_normalization == "peak_excess" else None
                    ),
                    total_control_signal_adu=total_peak,
                    primary_control_signal_adu=primary_control_signal,
                    secondary_control_signal_adu=secondary_control_signal,
                    secondary_to_primary_ratio=ratio,
                    primary_peak_excess_adu=primary_peak,
                    secondary_peak_excess_adu=secondary_peak,
                    mean_injected_primary_peak_excess_adu=float(np.mean(injected_primary_peaks)),
                    mean_injected_secondary_peak_excess_adu=float(np.mean(injected_secondary_peaks)),
                    pair_count=pair_total,
                    injected_count=injected_total,
                    primary_candidate_recall=primary_candidate_hits / pair_total,
                    secondary_candidate_recall=secondary_candidate_hits / pair_total,
                    primary_quality_recall=primary_quality_hits / pair_total,
                    secondary_quality_recall=secondary_quality_hits / pair_total,
                    source_candidate_recall=(primary_candidate_hits + secondary_candidate_hits) / injected_total,
                    source_quality_recall=(primary_quality_hits + secondary_quality_hits) / injected_total,
                    pair_candidate_resolution_recall=candidate_pair_resolved / pair_total,
                    pair_quality_resolution_recall=quality_pair_resolved / pair_total,
                    merged_candidate_fraction=merged_candidate_pairs / pair_total,
                    merged_quality_fraction=merged_quality_pairs / pair_total,
                    baseline_candidate_count=baseline.candidate_count,
                    baseline_quality_count=baseline.star_count,
                    mean_candidate_count=float(np.mean(candidate_counts)),
                    mean_quality_count=float(np.mean(quality_counts)),
                    local_background_adu=float(np.mean(local_backgrounds)),
                    local_noise_adu=float(np.mean(local_noises)),
                    trial_count=trials_per_condition,
                    note=(
                        f"固定同一总{'峰值超额' if signal_normalization == 'peak_excess' else '离散积分超额'}与真实背景，"
                        "仅改变 secondary/primary；"
                        "pair resolution 要求一对真值各由不同候选命中，不能把单个中间峰算作两颗。"
                    ),
                )
            )
    return tuple(rows)


def run_crowded_blend_audit(
    path: str | Path | FitsFrame,
    *,
    group_sizes: Sequence[int] = (2, 3),
    separations_px: Sequence[float] = (3.0, 5.4, 6.0),
    total_control_levels: Sequence[float] = (256.0, 512.0),
    trials_per_condition: int = 1,
    groups_per_trial: int = 4,
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
) -> tuple[CrowdedBlendAuditRow, ...]:
    """在真实 FITS 背景上审计 2/3 源拥挤组的合并与独立解析。

    每个组沿随机方向等间距排列，``nearest_separation_px`` 是相邻真值
    的距离；同一组内的总控制信号固定并平均分给各源。组级解析率要求组内
    每个真值被不同候选一对一命中，``merged_*`` 表示至少一个候选同时靠近
    两个真值，``extra_*`` 只表示包络邻域内候选数超过真值数。该实验用于
    解释拥挤场的机制，不把真实 FITS 中的候选数转换为 precision。
    """

    if trials_per_condition < 1 or groups_per_trial < 1:
        raise ValueError("trials_per_condition and groups_per_trial must be positive")
    if psf_fwhm <= 0 or threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1:
        raise ValueError("crowded blend geometry and detector parameters must be positive")
    if injected_psf_fwhm is not None and injected_psf_fwhm <= 0:
        raise ValueError("injected_psf_fwhm must be positive when provided")
    if background_box_size < 16 or min_flux_snr <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("crowded blend detector parameters are invalid")
    if match_radius_px is not None and match_radius_px <= 0:
        raise ValueError("match_radius_px must be positive when provided")
    if psf_model not in {"gaussian", "empirical"}:
        raise ValueError("psf_model must be gaussian or empirical")
    if signal_normalization not in {"peak_excess", "integrated_excess"}:
        raise ValueError("signal_normalization must be peak_excess or integrated_excess")
    if empirical_psf_radius < 3 or empirical_psf_sources < 1:
        raise ValueError("empirical PSF radius must be at least 3 and source count must be positive")
    resolved_group_sizes = tuple(int(size) for size in group_sizes)
    resolved_separations = tuple(float(value) for value in separations_px)
    resolved_levels = tuple(float(value) for value in total_control_levels)
    if not resolved_group_sizes or any(size < 2 for size in resolved_group_sizes):
        raise ValueError("group_sizes must contain integers at least 2")
    if not resolved_separations or any(value <= 0 for value in resolved_separations):
        raise ValueError("separations_px must contain positive values")
    if not resolved_levels or any(value <= 0 for value in resolved_levels):
        raise ValueError("total_control_levels must contain positive values")

    resolved_injected_fwhm = float(psf_fwhm if injected_psf_fwhm is None else injected_psf_fwhm)
    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
    image_mask = auxiliary_mask(image.shape)
    detector_options = {
        "mask": image_mask,
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
    empirical_psf = (
        estimate_empirical_psf(
            image,
            baseline.sources,
            support_radius=empirical_psf_radius,
            max_sources=empirical_psf_sources,
        )
        if psf_model == "empirical"
        else None
    )
    if psf_model == "empirical" and empirical_psf is None:
        raise RuntimeError("真实质量源不足，无法建立实测 PSF；请改用 gaussian 或增加可用模板")

    injected_detector_options = dict(detector_options)
    saturation_level = float(baseline.parameters.get("saturation_level", -1.0))
    injected_detector_options["saturation_level"] = saturation_level if saturation_level > 0 else None
    negative_overflow_limit = baseline.parameters.get("negative_overflow_limit")
    injected_detector_options["negative_overflow_limit"] = (
        None if negative_overflow_limit is None else float(negative_overflow_limit)
    )
    injected_detector_options["mask_zero_pixels"] = bool(int(baseline.parameters.get("mask_zero_pixels", 0)))

    rng = np.random.default_rng(seed)
    layouts: dict[
        tuple[int, float],
        list[tuple[list[list[tuple[float, float]]], list[tuple[float, float]]]],
    ] = {}
    for group_size in resolved_group_sizes:
        for separation in resolved_separations:
            extent = (group_size - 1) * separation
            layout_trials: list[tuple[list[list[tuple[float, float]]], list[tuple[float, float]]]] = []
            for _trial in range(trials_per_condition):
                anchors, local_stats = _select_real_background_positions(
                    image,
                    baseline.sources,
                    rng,
                    groups_per_trial,
                    noise_adu=float(baseline.noise),
                    margin=max(14, int(np.ceil(extent / 2.0)) + 10),
                    source_exclusion_radius_px=max(16.0, 2.0 * float(psf_fwhm) + extent + 4.0),
                )
                groups: list[list[tuple[float, float]]] = []
                for anchor_x, anchor_y in anchors:
                    angle = float(rng.uniform(0.0, 2.0 * np.pi))
                    unit_x = float(np.cos(angle))
                    unit_y = float(np.sin(angle))
                    center = 0.5 * float(group_size - 1)
                    groups.append(
                        [
                            (
                                anchor_x + (index - center) * separation * unit_x,
                                anchor_y + (index - center) * separation * unit_y,
                            )
                            for index in range(group_size)
                        ]
                    )
                layout_trials.append((groups, local_stats))
            layouts[(group_size, separation)] = layout_trials

    recovery_radius = max(3.0, float(match_radius_px if match_radius_px is not None else psf_fwhm))
    total_work = len(resolved_group_sizes) * len(resolved_separations) * len(resolved_levels) * trials_per_condition
    completed_work = 0
    if progress is not None:
        progress(0, total_work)
    rows: list[CrowdedBlendAuditRow] = []
    for group_size in resolved_group_sizes:
        for separation in resolved_separations:
            layout_trials = layouts[(group_size, separation)]
            for total_signal in resolved_levels:
                per_source_signal = total_signal / float(group_size)
                source_candidate_hits = 0
                source_quality_hits = 0
                group_candidate_resolved = 0
                group_quality_resolved = 0
                merged_candidate_groups = 0
                merged_quality_groups = 0
                extra_candidate_groups = 0
                extra_quality_groups = 0
                candidate_count_in_groups: list[int] = []
                quality_count_in_groups: list[int] = []
                candidate_counts: list[int] = []
                quality_counts: list[int] = []
                local_backgrounds: list[float] = []
                local_noises: list[float] = []
                group_total = 0
                for groups, local_stats in layout_trials:
                    test_image = np.asarray(image, dtype=np.float32).copy()
                    for group in groups:
                        for x, y in group:
                            _inject_source_signal(
                                test_image,
                                x,
                                y,
                                per_source_signal,
                                signal_normalization=signal_normalization,
                                psf_model=psf_model,
                                gaussian_fwhm=resolved_injected_fwhm,
                                empirical_psf=empirical_psf,
                            )
                    detection = detect_sources(test_image, **injected_detector_options)
                    for group in groups:
                        candidate_flags = _match_target_flags(group, detection.sources, recovery_radius)
                        quality_flags = _match_target_flags(group, detection.quality_sources, recovery_radius)
                        source_candidate_hits += sum(candidate_flags)
                        source_quality_hits += sum(quality_flags)
                        group_candidate_resolved += int(all(candidate_flags))
                        group_quality_resolved += int(all(quality_flags))
                        merged_candidate_groups += int(
                            _merged_group_flag(group, detection.sources, recovery_radius)
                        )
                        merged_quality_groups += int(
                            _merged_group_flag(group, detection.quality_sources, recovery_radius)
                        )
                        candidate_region_count = _group_region_candidate_count(
                            group, detection.sources, recovery_radius
                        )
                        quality_region_count = _group_region_candidate_count(
                            group, detection.quality_sources, recovery_radius
                        )
                        candidate_count_in_groups.append(candidate_region_count)
                        quality_count_in_groups.append(quality_region_count)
                        extra_candidate_groups += int(candidate_region_count > group_size)
                        extra_quality_groups += int(quality_region_count > group_size)
                        group_total += 1
                    candidate_counts.append(detection.candidate_count)
                    quality_counts.append(detection.star_count)
                    local_backgrounds.extend(background for background, _noise in local_stats)
                    local_noises.extend(noise for _background, noise in local_stats)
                    completed_work += 1
                    if progress is not None:
                        progress(completed_work, total_work)
                injected_total = group_size * group_total
                rows.append(
                    CrowdedBlendAuditRow(
                        source_path=str(frame.path),
                        proposal_mode=str(proposal_mode),
                        psf_model=psf_model,
                        signal_normalization=signal_normalization,
                        psf_source_count=0 if empirical_psf is None else empirical_psf.source_count,
                        psf_median_fwhm_px=None if empirical_psf is None else empirical_psf.median_fwhm_px,
                        psf_kernel_sum=None if empirical_psf is None else float(np.sum(empirical_psf.kernel)),
                        detector_psf_fwhm_px=float(psf_fwhm),
                        injected_psf_fwhm_px=resolved_injected_fwhm,
                        group_size=group_size,
                        nearest_separation_px=separation,
                        total_control_signal_adu=total_signal,
                        per_source_control_signal_adu=per_source_signal,
                        group_count=group_total,
                        injected_count=injected_total,
                        source_candidate_recall=source_candidate_hits / injected_total,
                        source_quality_recall=source_quality_hits / injected_total,
                        group_candidate_resolution_recall=group_candidate_resolved / group_total,
                        group_quality_resolution_recall=group_quality_resolved / group_total,
                        merged_candidate_fraction=merged_candidate_groups / group_total,
                        merged_quality_fraction=merged_quality_groups / group_total,
                        extra_candidate_fraction=extra_candidate_groups / group_total,
                        extra_quality_fraction=extra_quality_groups / group_total,
                        mean_candidate_count_in_group=float(np.mean(candidate_count_in_groups)),
                        mean_quality_count_in_group=float(np.mean(quality_count_in_groups)),
                        baseline_candidate_count=baseline.candidate_count,
                        baseline_quality_count=baseline.star_count,
                        mean_candidate_count=float(np.mean(candidate_counts)),
                        mean_quality_count=float(np.mean(quality_counts)),
                        local_background_adu=float(np.mean(local_backgrounds)),
                        local_noise_adu=float(np.mean(local_noises)),
                        trial_count=trials_per_condition,
                        note=(
                            f"固定同一真实背景和总{('峰值超额' if signal_normalization == 'peak_excess' else '离散积分超额')}，"
                            "组内各源等分信号；group resolution 要求每个真值由不同候选一对一命中；"
                            "merged/extra 仅是注入组邻域诊断，不是 precision。"
                        ),
                    )
                )
    return tuple(rows)


def write_injection_artifacts(rows: Sequence[InjectionRecoveryRow], out_dir: str | Path) -> Path:
    """写注入实验 CSV 和轻量 PNG 曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "injection_recovery.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].as_dict()) if rows else ["peak_excess_adu"])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    from .innovation import _line_chart

    _line_chart(
        output / "injection_recovery.png",
        "INJECTION RECOVERY · CANDIDATE VS QUALITY RECALL",
        [row.peak_excess_adu for row in rows],
        (
            ("candidate recall", [row.candidate_recall for row in rows], "#d79432"),
            ("quality recall", [row.quality_recall for row in rows], "#4f9b83"),
        ),
        y_label="recall",
        x_label="injected peak excess / ADU",
    )
    return output


def write_feature_audit_artifacts(rows: Sequence[FeatureAuditRow], out_dir: str | Path) -> Path:
    """写特征分层控制实验的 CSV、JSON 和正样本回收率曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].as_dict()) if rows else ("feature_class", "scenario")
    with (output / "feature_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    payload = {
        "row_count": len(rows),
        "positive_scenarios": sum(row.control_type == "positive" for row in rows),
        "negative_scenarios": sum(row.control_type == "negative" for row in rows),
        "note": "正样本回收率和阴性特征邻域泄漏是合成控制量，不代表真实 FITS 的物理真值或伪影比例。",
        "rows": [row.as_dict() for row in rows],
    }
    (output / "feature_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    positive = [row for row in rows if row.control_type == "positive" and row.truth_count > 0]
    from .innovation import _line_chart

    _line_chart(
        output / "feature_audit_recall.png",
        "FEATURE AUDIT · KNOWN-SOURCE RECALL",
        list(range(1, len(positive) + 1)),
        (
            ("candidate recall", [row.candidate_recall or 0.0 for row in positive], "#d79432"),
            ("quality recall", [row.quality_recall or 0.0 for row in positive], "#4f9b83"),
        ),
        y_label="recall",
        x_label="positive scenario order (see feature_audit.csv)",
    )
    return output


def write_sequence_feature_persistence_artifacts(
    result: SequenceFeatureAuditResult,
    out_dir: str | Path,
) -> Path:
    """写真实 15 帧特征持久性审计的 CSV、JSON 和曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    persistence_fields = tuple(result.persistence_rows[0].as_dict()) if result.persistence_rows else ("feature_class",)
    with (output / "sequence_feature_persistence.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=persistence_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.persistence_rows)
    frame_fields = tuple(result.frame_rows[0].as_dict()) if result.frame_rows else ("frame_index", "feature_class")
    with (output / "sequence_feature_frame_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.frame_rows)
    temporal_profiles = summarize_sequence_feature_temporal_profiles(result.frame_rows)
    temporal_fields = tuple(temporal_profiles[0].as_dict()) if temporal_profiles else ("feature_class",)
    with (output / "sequence_feature_temporal_profile.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=temporal_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in temporal_profiles)
    transition_fields = (
        tuple(result.class_transition_rows[0].as_dict())
        if result.class_transition_rows
        else ("layer", "anchor_feature_class", "response_feature_class")
    )
    with (output / "sequence_feature_class_transition.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=transition_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.class_transition_rows)
    payload = result.as_dict()
    payload["note"] = (
        "presence 是首帧坐标在固定小邻域内的互相最近一对一响应次数，不是星表身份；"
        "密集候选场中的邻近命中不能替代星表、注入和人工真值。"
    )
    payload["temporal_profiles"] = [row.as_dict() for row in temporal_profiles]
    payload["class_transition_note"] = (
        "class_transition_rows 只统计注册坐标中互相最近的一对一响应，并以首帧类别与响应帧类别分开计数；"
        "它仍不是星表身份匹配，也不代表物理类别转移。"
    )
    payload["temporal_profile_note"] = (
        "candidate_fraction_cv 描述类别构成的时间稳定性；quality_fraction_weighted 对 compact_quality "
        "包含按 quality_passed 分类造成的定义性泄漏，不是真阳性率。"
    )
    (output / "sequence_feature_persistence.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    from .innovation import _line_chart

    labels = [row.feature_class for row in result.persistence_rows]
    candidate_fraction = [
        row.candidate_presence_ge_required_count / row.anchor_candidate_count
        if row.anchor_candidate_count
        else 0.0
        for row in result.persistence_rows
    ]
    quality_fraction = [
        row.quality_presence_ge_required_count / row.anchor_candidate_count
        if row.anchor_candidate_count
        else 0.0
        for row in result.persistence_rows
    ]
    _line_chart(
        output / "sequence_feature_persistence.png",
        "SEQUENCE FEATURE PERSISTENCE · ANCHOR FRAME",
        list(range(1, len(labels) + 1)),
        (
            ("candidate neighborhood", candidate_fraction, "#d79432"),
            ("quality neighborhood", quality_fraction, "#4f9b83"),
        ),
        y_label=f"fraction present >= {result.required_presence}/{result.frame_count}",
        x_label="feature class order (see sequence_feature_persistence.csv)",
    )
    return output


def write_feature_cross_audit_artifacts(
    result: FeatureCrossAuditResult,
    out_dir: str | Path,
) -> Path:
    """写特征类别交叉反例审计的 CSV、JSON 和比例图。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(result.rows[0].as_dict()) if result.rows else ("feature_class",)
    with (output / "feature_cross_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    payload = result.as_dict()
    payload["note"] = (
        "该报告只连接单帧特征汇总和 15 帧响应持久性；质量邻域持久性不是特征类别的真阳性，"
        "高 SNR 反例也不能替代星表/WCS 身份核验。"
    )
    if result.class_transition_path is not None:
        payload["class_transition_note"] = (
            "类别转移字段来自注册坐标中的互相最近一对一响应；任意类别响应和同类响应是"
            "检测器稳定性描述，不是物理类别变化或逐星身份。"
        )
    (output / "feature_cross_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    from .innovation import _line_chart

    labels = [row.feature_class for row in result.rows]
    candidate_fraction = [
        row.candidate_presence_fraction if row.candidate_presence_fraction is not None else 0.0
        for row in result.rows
    ]
    quality_fraction = [
        row.quality_presence_fraction if row.quality_presence_fraction is not None else 0.0
        for row in result.rows
    ]
    _line_chart(
        output / "feature_cross_audit.png",
        "FEATURE CROSS AUDIT · CANDIDATE VS QUALITY PERSISTENCE",
        list(range(1, len(labels) + 1)),
        (
            ("candidate >= required", candidate_fraction, "#d79432"),
            ("quality >= required", quality_fraction, "#4f9b83"),
        ),
        y_label=f"fraction present >= {result.required_presence}/{result.frame_count}",
        x_label="feature class order (see feature_cross_audit.csv)",
    )
    return output


def write_local_multipsf_audit_artifacts(
    result: LocalMultiPSFAuditResult,
    out_dir: str | Path,
) -> Path:
    """写固定位置 K 源局部 PSF 对照的 CSV、JSON 和 BIC 图。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(result.rows[0].as_dict()) if result.rows else ("frame_index", "mask_mode")
    array_fields = {"target_positions", "amplitudes", "component_snr"}
    with (output / "local_multipsf_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result.rows:
            record = row.as_dict()
            for field in array_fields:
                record[field] = json.dumps(record[field], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(record)
    payload = result.as_dict()
    payload["note"] = (
        "位置由 detector 候选/人工审计坐标固定给出；K=1..N 共享局部背景和像素窗口，"
        "非负幅度用 NNLS 拟合。结果是局部模型敏感性证据，不是物理恒星身份。"
    )
    (output / "local_multipsf_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    from .innovation import _line_chart

    component_counts = list(range(1, len(result.target_positions) + 1))
    series: list[tuple[str, list[float], str]] = []
    colors = {
        "raw": "#d79432",
        "range_masked": "#4f9b83",
        "sentinel_masked": "#4d91b8",
        "range_sentinel_masked": "#c7359e",
    }
    for mask_mode in result.mask_modes:
        values: list[float] = []
        for component_count in component_counts:
            candidates = [
                row.delta_bic_from_single
                for row in result.rows
                if row.mask_mode == mask_mode
                and row.component_count == component_count
                and row.delta_bic_from_single is not None
            ]
            values.append(float(np.median(candidates)) if candidates else float("nan"))
        series.append((f"{mask_mode} · median ΔBIC", values, colors.get(mask_mode, "#626b7d")))
    series.append(
        (
            "confirmation ΔBIC=10",
            [10.0 for _component_count in component_counts],
            "#20293d",
        )
    )
    _line_chart(
        output / "local_multipsf_bic.png",
        "LOCAL MULTI-PSF AUDIT · MODEL ORDER",
        [float(value) for value in component_counts],
        series,
        y_label="ΔBIC = BIC(K=1) - BIC(K)",
        x_label="fixed PSF component count K",
    )
    return output


def write_local_free_multipsf_audit_artifacts(
    result: LocalFreeMultiPSFAuditResult,
    out_dir: str | Path,
) -> Path:
    """写有限自由位置 K 源局部 PSF 对照的 CSV、JSON 和 BIC 图。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(result.rows[0].as_dict()) if result.rows else ("frame_index", "mask_mode")
    array_fields = {
        "initial_positions",
        "fitted_positions",
        "position_offsets",
        "positions_at_bound",
        "amplitudes",
        "component_snr",
    }
    with (output / "local_free_multipsf_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result.rows:
            record = row.as_dict()
            for field in array_fields:
                record[field] = json.dumps(record[field], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(record)
    payload = result.as_dict()
    payload["note"] = (
        "中心只允许在初始 detector 候选周围的有限半径内优化；K=1..N 共享局部背景和像素窗口，"
        "源幅度用非负约束。结果是局部模型敏感性证据，不是物理恒星身份。"
    )
    (output / "local_free_multipsf_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    from .innovation import _line_chart

    component_counts = list(range(1, len(result.target_positions) + 1))
    series: list[tuple[str, list[float], str]] = []
    colors = {
        "raw": "#d79432",
        "range_masked": "#4f9b83",
        "sentinel_masked": "#4d91b8",
        "range_sentinel_masked": "#c7359e",
    }
    for mask_mode in result.mask_modes:
        values: list[float] = []
        for component_count in component_counts:
            candidates = [
                row.delta_bic_from_single
                for row in result.rows
                if row.mask_mode == mask_mode
                and row.component_count == component_count
                and row.delta_bic_from_single is not None
            ]
            values.append(float(np.median(candidates)) if candidates else float("nan"))
        series.append((f"{mask_mode} · median ΔBIC", values, colors.get(mask_mode, "#626b7d")))
    series.append(
        (
            "confirmation ΔBIC=10",
            [10.0 for _component_count in component_counts],
            "#20293d",
        )
    )
    _line_chart(
        output / "local_free_multipsf_bic.png",
        "LOCAL FREE-POSITION MULTI-PSF · MODEL ORDER",
        [float(value) for value in component_counts],
        series,
        y_label="ΔBIC = BIC(K=1) - BIC(K)",
        x_label="bounded free PSF component count K",
    )
    return output


def write_local_free_multipsf_radius_sweep_artifacts(
    result: LocalFreeMultiPSFRadiusSweepResult,
    out_dir: str | Path,
) -> Path:
    """写自由位置搜索半径敏感性摘要的 CSV、JSON 和 BIC 图。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(result.rows[0].as_dict()) if result.rows else ("position_radius_px", "frame_index")
    array_fields = {"component_snr", "fitted_positions", "positions_at_bound"}
    with (output / "local_free_multipsf_radius_sweep.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result.rows:
            record = row.as_dict()
            for field in array_fields:
                record[field] = json.dumps(record[field], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(record)
    payload = result.as_dict()
    payload["note"] = (
        "位置搜索半径敏感性只用于检查局部模型是否稳定；位置触边、优化未收敛和"
        "跨半径不稳定都不能作为物理恒星身份。"
    )
    (output / "local_free_multipsf_radius_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    from .innovation import _line_chart

    series: list[tuple[str, list[float], str]] = []
    colors = {
        "raw": "#d79432",
        "range_masked": "#4f9b83",
        "sentinel_masked": "#4d91b8",
        "range_sentinel_masked": "#c7359e",
    }
    max_colors = {
        "raw": "#a95d16",
        "range_masked": "#28715e",
        "sentinel_masked": "#2b6385",
        "range_sentinel_masked": "#8c1f6d",
    }
    for mask_mode in result.mask_modes:
        median_values: list[float] = []
        maximum_values: list[float] = []
        for radius in result.position_radii_px:
            candidates = [
                row.delta_bic_from_single
                for row in result.rows
                if row.position_radius_px == radius
                and row.mask_mode == mask_mode
                and row.component_count == 2
                and row.delta_bic_from_single is not None
            ]
            median_values.append(float(np.median(candidates)) if candidates else float("nan"))
            maximum_values.append(float(np.max(candidates)) if candidates else float("nan"))
        series.append(
            (
                f"{mask_mode} · median K=2 ΔBIC",
                median_values,
                colors.get(mask_mode, "#626b7d"),
            )
        )
        series.append(
            (
                f"{mask_mode} · max K=2 ΔBIC",
                maximum_values,
                max_colors.get(mask_mode, "#3f4655"),
            )
        )
    series.append(
        (
            "confirmation ΔBIC=10",
            [10.0 for _radius in result.position_radii_px],
            "#20293d",
        )
    )
    _line_chart(
        output / "local_free_multipsf_radius_sweep.png",
        "LOCAL FREE-POSITION MULTI-PSF · RADIUS SENSITIVITY",
        [float(radius) for radius in result.position_radii_px],
        series,
        y_label="K=2 ΔBIC (median / max)",
        x_label="position search radius / px",
    )
    return output


def write_source_pair_audit_artifacts(
    result: SourcePairAuditResult,
    out_dir: str | Path,
) -> Path:
    """写近邻双源逐帧原始证据、PSF 对照和审计曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(result.rows[0].as_dict()) if result.rows else ("frame_index", "path")
    with (output / "source_pair_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    payload = result.as_dict()
    payload["note"] = (
        "ID 只在单次检测运行内有效；secondary 的原始像素统计使用固定 detector 坐标，"
        "pair PSF 数值是局部模型比较，不是星表身份或物理真值。"
    )
    (output / "source_pair_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    from .innovation import _line_chart

    frame_numbers = [float(row.frame_index) for row in result.rows]
    _line_chart(
        output / "source_pair_raw_codes.png",
        "SOURCE PAIR AUDIT · RAW RANGE / REPEATED CODE EVIDENCE",
        frame_numbers,
        (
            (
                "secondary negative overflow",
                [float(row.secondary_negative_overflow_count) for row in result.rows],
                "#c7359e",
            ),
            (
                "secondary positive extreme",
                [float(row.secondary_positive_extreme_count) for row in result.rows],
                "#d79432",
            ),
            (
                "secondary exact -1",
                [float(row.secondary_fixed_minus_one_count) for row in result.rows],
                "#4f9b83",
            ),
        ),
        y_label="pixel count in secondary aperture",
        x_label="frame",
    )
    _line_chart(
        output / "source_pair_psf_evidence.png",
        "SOURCE PAIR AUDIT · SINGLE / DOUBLE PSF EVIDENCE",
        frame_numbers,
        (
            (
                "raw delta BIC",
                [
                    float(row.pair_delta_bic_raw)
                    if row.pair_delta_bic_raw is not None
                    else float("nan")
                    for row in result.rows
                ],
                "#d79432",
            ),
            (
                "range-masked delta BIC",
                [
                    float(row.pair_delta_bic_range_masked)
                    if row.pair_delta_bic_range_masked is not None
                    else float("nan")
                    for row in result.rows
                ],
                "#4f9b83",
            ),
            (
                "confirmation line delta BIC=10",
                [10.0 for _row in result.rows],
                "#c7359e",
            ),
        ),
        y_label="delta BIC = BIC(single) - BIC(double)",
        x_label="frame",
    )
    return output


def write_temporal_code_audit_artifacts(
    result: TemporalCodeAuditResult,
    out_dir: str | Path,
) -> Path:
    """写固定码审计的 JSON 和连通簇 CSV。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "temporal_code_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    fields = ("kind", "component_id", "size", "x0", "y0", "x1", "y1", "is_focus_component")
    with (output / "temporal_code_components.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: component.get(field) for field in fields}
            for component in result.components
        )
    return output


def write_real_background_injection_artifacts(
    rows: Sequence[RealBackgroundInjectionRow],
    out_dir: str | Path,
) -> Path:
    """写真实背景注入实验 CSV 和召回率曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "real_background_injection.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].as_dict()) if rows else ["peak_excess_adu"])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    from .innovation import _line_chart

    _line_chart(
        output / "real_background_injection.png",
        "REAL FITS BACKGROUND · INJECTION RECOVERY",
        [row.peak_excess_adu for row in rows],
        (
            ("candidate recall", [row.candidate_recall for row in rows], "#d79432"),
            ("quality recall", [row.quality_recall for row in rows], "#4f9b83"),
        ),
        y_label="recall",
        x_label="injected peak excess / ADU",
    )
    return output


def write_stratified_real_background_injection_artifacts(
    rows: Sequence[StratifiedRealBackgroundInjectionRow],
    out_dir: str | Path,
) -> Path:
    """写真实背景分层注入的 CSV、JSON 和条件分组曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].as_dict()) if rows else ["stratum", "peak_excess_adu"]
    with (output / "stratified_real_background_injection.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    payload = {
        "row_count": len(rows),
        "strata": list(dict.fromkeys(row.stratum for row in rows)),
        "peak_levels": list(dict.fromkeys(row.peak_excess_adu for row in rows)),
        "note": (
            "分层注入只测量已知注入源在真实背景条件下的候选/质量回收；"
            "没有逐星真值时，背景检测数和净数量变化不能解释成 precision、误检率或物理恒星完备率。"
        ),
        "rows": [row.as_dict() for row in rows],
    }
    (output / "stratified_real_background_injection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    from .innovation import _line_chart

    if rows:
        _line_chart(
            output / "stratified_real_background_injection.png",
            "REAL FITS BACKGROUND · STRATIFIED INJECTION RECOVERY",
            list(range(1, len(rows) + 1)),
            (
                ("candidate recall", [row.candidate_recall for row in rows], "#d79432"),
                ("quality recall", [row.quality_recall for row in rows], "#4f9b83"),
            ),
            y_label="recall",
            x_label="stratum / injected peak (see CSV)",
        )
    return output


def write_proposal_mode_comparison_artifacts(
    rows: Sequence[ProposalModeComparisonRow],
    out_dir: str | Path,
) -> Path:
    """写同坐标 Gaussian/hybrid 对照 CSV 与召回曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "proposal_mode_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].as_dict()) if rows else ["peak_excess_adu"])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    from .innovation import _line_chart

    _line_chart(
        output / "proposal_mode_comparison.png",
        "SAME-POSITION INJECTION · GAUSSIAN VS DOG VS STARLET",
        [row.peak_excess_adu for row in rows],
        (
            ("gaussian candidate", [row.gaussian_candidate_recall for row in rows], "#72b9d4"),
            ("hybrid candidate", [row.hybrid_candidate_recall for row in rows], "#d79432"),
            ("ensemble candidate", [row.ensemble_candidate_recall for row in rows], "#a56cc1"),
            ("gaussian quality", [row.gaussian_quality_recall for row in rows], "#8bdcff"),
            ("hybrid quality", [row.hybrid_quality_recall for row in rows], "#4f9b83"),
            ("ensemble quality", [row.ensemble_quality_recall for row in rows], "#d65f91"),
        ),
        y_label="recall",
        x_label="injected peak excess / ADU",
    )
    return output


def write_pair_flux_ratio_audit_artifacts(
    rows: Sequence[PairFluxRatioAuditRow],
    out_dir: str | Path,
) -> Path:
    """写双源强弱比审计的 CSV、JSON 和解析率曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].as_dict()) if rows else ["total_control_signal_adu", "secondary_to_primary_ratio"]
    with (output / "pair_flux_ratio_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    payload = {
        "row_count": len(rows),
        "total_control_signal_levels": list(dict.fromkeys(row.total_control_signal_adu for row in rows)),
        "signal_normalizations": list(dict.fromkeys(row.signal_normalization for row in rows)),
        "secondary_to_primary_ratios": list(
            dict.fromkeys(row.secondary_to_primary_ratio for row in rows)
        ),
        "note": (
            "source recall 只表示单个注入真值被候选命中；pair resolution recall 要求同一对双源"
            "由两个不同候选分别命中；merged fraction 表示一个候选同时落入两个匹配半径。"
            "peak_excess 是峰值超额控制，integrated_excess 是离散注入小窗积分超额控制。"
        ),
        "rows": [row.as_dict() for row in rows],
    }
    (output / "pair_flux_ratio_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    from .innovation import _line_chart

    if rows:
        _line_chart(
            output / "pair_flux_ratio_audit.png",
            "FIXED TOTAL SIGNAL · UNEQUAL PAIR RESOLUTION",
            list(range(1, len(rows) + 1)),
            (
                (
                    "primary candidate",
                    [row.primary_candidate_recall for row in rows],
                    "#72b9d4",
                ),
                (
                    "secondary candidate",
                    [row.secondary_candidate_recall for row in rows],
                    "#d79432",
                ),
                (
                    "pair candidate resolved",
                    [row.pair_candidate_resolution_recall for row in rows],
                    "#a56cc1",
                ),
                (
                    "pair quality resolved",
                    [row.pair_quality_resolution_recall for row in rows],
                    "#4f9b83",
                ),
            ),
            y_label="recall",
            x_label="condition order (see CSV for total peak / ratio)",
        )
    return output


def write_crowded_blend_audit_artifacts(
    rows: Sequence[CrowdedBlendAuditRow],
    out_dir: str | Path,
) -> Path:
    """写拥挤多源组的合并/解析审计 CSV、JSON 和曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].as_dict()) if rows else ["group_size", "nearest_separation_px"]
    with (output / "crowded_blend_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    payload = {
        "row_count": len(rows),
        "group_sizes": list(dict.fromkeys(row.group_size for row in rows)),
        "separations_px": list(dict.fromkeys(row.nearest_separation_px for row in rows)),
        "total_control_signal_levels": list(dict.fromkeys(row.total_control_signal_adu for row in rows)),
        "signal_normalizations": list(dict.fromkeys(row.signal_normalization for row in rows)),
        "note": (
            "group resolution 要求组内每个真值被不同候选一对一命中；merged 表示一个候选"
            "同时靠近至少两个真值；extra 只描述组邻域候选数超过真值数，不是误检率。"
        ),
        "rows": [row.as_dict() for row in rows],
    }
    (output / "crowded_blend_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    from .innovation import _line_chart

    if rows:
        _line_chart(
            output / "crowded_blend_audit.png",
            "CROWDED BLEND · GROUP RESOLUTION / MERGE",
            list(range(1, len(rows) + 1)),
            (
                (
                    "source candidate recall",
                    [row.source_candidate_recall for row in rows],
                    "#72b9d4",
                ),
                (
                    "group candidate resolved",
                    [row.group_candidate_resolution_recall for row in rows],
                    "#d79432",
                ),
                (
                    "group quality resolved",
                    [row.group_quality_resolution_recall for row in rows],
                    "#4f9b83",
                ),
                (
                    "merged candidate",
                    [row.merged_candidate_fraction for row in rows],
                    "#a56cc1",
                ),
            ),
            y_label="fraction",
            x_label="condition order (see CSV for size / separation / signal)",
        )
    return output


def run_single_frame_trail_audit(
    paths: Iterable[str | Path],
    *,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 3.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    max_sources: int | None = None,
    residual_sigma: float = 15.0,
    min_residual_adu: float = 100.0,
    min_feature_area: int = 40,
    min_axis_ratio: float = 4.0,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[SingleFrameTrailAuditRow, ...]:
    """逐张 FITS 运行单帧长线检测，输出可复核的候选统计。

    该审计刻意不跨帧关联，也不输出 ``moving`` 结论：它回答的是“这张
    图里是否存在满足几何和显著性门槛的长线候选”。跨帧持续性和速度
    仍由 ``sequence.py`` 的独立路径判断。
    """

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    if threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1:
        raise ValueError("single-frame trail detection parameters must be positive")
    if psf_fwhm <= 0 or background_box_size < 16 or min_flux_snr <= 0:
        raise ValueError("psf_fwhm/background_box_size/min_flux_snr are invalid")
    if residual_sigma <= 0 or min_residual_adu <= 0 or min_feature_area < 3 or min_axis_ratio <= 0:
        raise ValueError("trail geometry parameters must be positive")

    rows: list[SingleFrameTrailAuditRow] = []
    total = len(frame_paths)
    for index, path in enumerate(frame_paths, start=1):
        if progress is not None:
            progress(index, total)
        frame = read_fits(path)
        analysis = analyze_frame(
            frame,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            max_sources=max_sources,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            min_flux_snr=min_flux_snr,
            reject_linear_artifacts=True,
        )
        trails = detect_long_trails(
            analysis,
            psf_fwhm=psf_fwhm,
            residual_sigma=residual_sigma,
            min_residual_adu=min_residual_adu,
            min_feature_area=min_feature_area,
            min_axis_ratio=min_axis_ratio,
        )
        points = [point for trail in trails for point in trail.points]
        longest = max(points, key=lambda point: point.length_px, default=None)
        rows.append(
            SingleFrameTrailAuditRow(
                frame_number=index,
                path=str(path),
                timestamp=str(frame.header.get("DATE-OBS")) if frame.header.get("DATE-OBS") is not None else None,
                candidate_count=analysis.detection.candidate_count,
                quality_count=analysis.detection.star_count,
                trail_count=len(trails),
                max_length_px=float(longest.length_px) if longest is not None else None,
                max_width_px=float(longest.width_px) if longest is not None else None,
                max_residual_snr=float(max(point.residual_snr for point in points)) if points else None,
                edge_trail_count=sum(point.touches_edge for point in points),
                total_area_pixels=sum(point.area_pixels for point in points),
                longest_angle_deg=float(longest.angle_deg) if longest is not None else None,
                longest_bbox=",".join(str(value) for value in longest.bbox) if longest is not None else None,
            )
        )
    return tuple(rows)


def write_single_frame_trail_artifacts(
    rows: Sequence[SingleFrameTrailAuditRow],
    out_dir: str | Path,
) -> Path:
    """写单帧长线审计 CSV 及候选数量/长度曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "single_frame_trails.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].as_dict()) if rows else ["frame_number"])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    from .innovation import _line_chart

    frame_numbers = [row.frame_number for row in rows]
    _line_chart(
        output / "single_frame_trail_counts.png",
        "SINGLE-FRAME TRAIL AUDIT · CANDIDATE COUNT",
        frame_numbers,
        (("long-trail candidates", [row.trail_count for row in rows], "#d79432"),),
        y_label="candidate count",
        x_label="frame number",
    )
    _line_chart(
        output / "single_frame_trail_lengths.png",
        "SINGLE-FRAME TRAIL AUDIT · MAX LENGTH",
        frame_numbers,
        (("max length", [row.max_length_px or 0.0 for row in rows], "#4f9b83"),),
        y_label="pixel",
        x_label="frame number",
    )
    return output


def _source_quality_flag_chart(
    path: Path,
    detection: DetectionResult,
    flag_counts: Counter[str],
) -> None:
    """绘制质量通过量和可重叠拒绝标志；纵轴使用 log1p 保留小类。"""

    rows = [("QUALITY_PASS", detection.star_count), *sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))]
    width, height = 1200, 680
    image = Image.new("RGB", (width, height), "#f4f0e7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 90, 75, width - 45, height - 110
    draw.text((left, 22), "SOURCE QUALITY · OVERLAPPING FLAGS (LOG1P)", fill="#20293d", font=font)
    if not rows:
        image.save(path)
        return
    values = [math.log1p(max(0, int(count))) for _label, count in rows]
    maximum = max(values) or 1.0
    bar_slot = (right - left) / len(rows)
    for index, ((label, raw_count), value) in enumerate(zip(rows, values, strict=True)):
        x0 = left + index * bar_slot + 0.12 * bar_slot
        x1 = left + (index + 1) * bar_slot - 0.12 * bar_slot
        y = bottom - value / maximum * (bottom - top)
        color = "#4f9b83" if label == "QUALITY_PASS" else "#d79432"
        draw.rectangle((x0, y, x1, bottom), fill=color)
        draw.text((x0, max(top, y - 16)), f"{raw_count:,}", fill="#273147", font=font)
        short_label = label if len(label) <= 15 else label[:14] + "…"
        draw.text((x0, bottom + 10), short_label, fill="#626b7d", font=font)
    draw.line((left, bottom, right, bottom), fill="#626b7d", width=1)
    draw.text((left, height - 30), "Flags may overlap; bar heights are log1p(count), labels preserve raw counts.", fill="#626b7d", font=font)
    image.save(path)


def _source_feature_class_chart(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """绘制互斥首要特征类别；柱高使用 log1p，柱顶保留原始计数。"""

    width, height = 1280, 720
    image = Image.new("RGB", (width, height), "#f4f0e7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 82, 80, width - 42, height - 120
    draw.text((left, 24), "SOURCE FEATURE CLASSES · MUTUALLY EXCLUSIVE PRIMARY AUDIT", fill="#20293d", font=font)
    values = [max(0, int(row.get("candidate_count", 0) or 0)) for row in rows]
    if not values:
        image.save(path)
        return
    scaled = [math.log1p(value) for value in values]
    maximum = max(scaled) or 1.0
    slot = (right - left) / len(rows)
    for index, (row, value, transformed) in enumerate(zip(rows, values, scaled, strict=True)):
        x0 = left + index * slot + 0.10 * slot
        x1 = left + (index + 1) * slot - 0.10 * slot
        y = bottom - transformed / maximum * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill="#4f9b83" if row.get("quality_count", 0) else "#d79432")
        draw.text((x0, max(top, y - 17)), f"{value:,}", fill="#273147", font=font)
        label = str(row.get("feature_class", ""))
        short_label = label if len(label) <= 15 else label[:14] + "…"
        draw.text((x0, bottom + 10), short_label, fill="#626b7d", font=font)
    draw.line((left, bottom, right, bottom), fill="#626b7d", width=1)
    draw.text(
        (left, height - 34),
        "Primary classes are mutually exclusive; original flags can overlap. Bar heights are log1p(count).",
        fill="#626b7d",
        font=font,
    )
    image.save(path)


def _source_spatial_grid_artifacts(
    detection: DetectionResult,
    csv_path: Path,
    chart_path: Path,
    *,
    grid_size: int = 16,
) -> dict[str, float]:
    """输出返回源密度与质量通过率的空间网格，检查视场位置偏差。"""

    height_px, width_px = detection.image_shape
    returned = np.zeros((grid_size, grid_size), dtype=np.int64)
    quality = np.zeros((grid_size, grid_size), dtype=np.int64)
    for source in detection.sources:
        col = min(grid_size - 1, max(0, int(float(source.x) / max(1, width_px) * grid_size)))
        row = min(grid_size - 1, max(0, int(float(source.y) / max(1, height_px) * grid_size)))
        returned[row, col] += 1
        if source.quality_passed:
            quality[row, col] += 1
    fraction = np.divide(quality, returned, out=np.full_like(quality, np.nan, dtype=np.float64), where=returned > 0)
    rows: list[dict[str, object]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            rows.append(
                {
                    "grid_row": row,
                    "grid_column": col,
                    "x_min_px": col * width_px / grid_size,
                    "x_max_px": (col + 1) * width_px / grid_size,
                    "y_min_px": row * height_px / grid_size,
                    "y_max_px": (row + 1) * height_px / grid_size,
                    "returned_count": int(returned[row, col]),
                    "quality_count": int(quality[row, col]),
                    "rejected_count": int(returned[row, col] - quality[row, col]),
                    "quality_fraction": float(fraction[row, col]) if np.isfinite(fraction[row, col]) else None,
                }
            )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    width, height = 1200, 680
    image = Image.new("RGB", (width, height), "#f4f0e7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((75, 22), "SOURCE SPATIAL AUDIT · RETURNED DENSITY / QUALITY FRACTION", fill="#20293d", font=font)
    panels = ((returned.astype(np.float64), "returned density · log1p(count)", "density"), (fraction, "quality fraction", "fraction"))
    for panel_index, (values, title, mode) in enumerate(panels):
        panel_left = 75 + panel_index * 560
        panel_top = 82
        panel_size = 480
        draw.text((panel_left, 55), title, fill="#273147", font=font)
        finite = values[np.isfinite(values)]
        if mode == "density":
            scaled = np.log1p(values)
            maximum = float(np.max(scaled)) if scaled.size else 1.0
            scaled = scaled / (maximum or 1.0)
        else:
            scaled = np.clip(values, 0.0, 1.0)
        cell = panel_size / grid_size
        for row in range(grid_size):
            for col in range(grid_size):
                value = float(scaled[row, col]) if np.isfinite(scaled[row, col]) else 0.0
                if mode == "density":
                    low, high = np.array((244, 240, 231)), np.array((215, 148, 50))
                else:
                    low, high = np.array((222, 216, 202)), np.array((79, 155, 131))
                rgb = tuple(int(channel) for channel in np.rint(low + value * (high - low)))
                x0 = panel_left + col * cell
                y0 = panel_top + row * cell
                draw.rectangle((x0, y0, x0 + cell + 1, y0 + cell + 1), fill=rgb, outline="#f4f0e7")
        draw.rectangle((panel_left, panel_top, panel_left + panel_size, panel_top + panel_size), outline="#626b7d", width=1)
        draw.text((panel_left, panel_top + panel_size + 12), "X →", fill="#626b7d", font=font)
        draw.text((panel_left - 34, panel_top - 14), "Y ↓", fill="#626b7d", font=font)
        if finite.size:
            draw.text((panel_left + 90, panel_top + panel_size + 12), f"min {float(np.min(finite)):.3f} · median {float(np.median(finite)):.3f} · max {float(np.max(finite)):.3f}", fill="#626b7d", font=font)
    note = "Each cell preserves raw counts in source_spatial_grid.csv. Spatial maps describe returned sources; truncated exports are SNR-selected."
    draw.text((75, height - 32), note, fill="#626b7d", font=font)
    image.save(chart_path)

    nonempty_fraction = fraction[np.isfinite(fraction)]
    density_mean = float(np.mean(quality))
    return {
        "spatial_grid_size": float(grid_size),
        "quality_density_cv": float(np.std(quality) / density_mean) if density_mean > 0 else math.nan,
        "quality_fraction_median": float(np.median(nonempty_fraction)) if nonempty_fraction.size else math.nan,
        "quality_fraction_min": float(np.min(nonempty_fraction)) if nonempty_fraction.size else math.nan,
        "quality_fraction_max": float(np.max(nonempty_fraction)) if nonempty_fraction.size else math.nan,
    }


def write_detection_source_artifacts(
    detection: DetectionResult,
    out_dir: str | Path,
    *,
    frame: FitsFrame | None = None,
    spatial_psf_per_class: int = 8,
    spatial_psf_grid_size: int = 4,
) -> Path:
    """导出源级研究表、质量拒绝统计和 SNR 分布图。

    该函数只消费已经完成的 ``DetectionResult``，不会重新检测，也不会把
    ``quality_count`` 改写成物理恒星真值。拒绝标志允许重叠，因此汇总表中
    的旗标计数不能简单相加为候选总数。空间 PSF 诊断的抽样量和网格数可
    单独放大，但默认保持轻量的 ``8/4×4`` 口径，不影响检测结果。
    """
    if spatial_psf_per_class < 1 or spatial_psf_grid_size < 1:
        raise ValueError("spatial_psf_per_class and spatial_psf_grid_size must be positive")

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_fields = (
        "detection_id",
        "x",
        "y",
        "peak_x",
        "peak_y",
        "centroid_shift_px",
        "peak",
        "flux",
        "background",
        "noise",
        "snr",
        "flux_error",
        "flux_snr",
        "filter_snr",
        "proposal_snr",
        "proposal_methods",
        "proposal_scales",
        "nearest_gaussian_px",
        "deblend_delta_bic",
        "deblend_component_snr",
        "repeated_code_count",
        "range_anomaly_pixel_count",
        "repeated_code_values",
        "fwhm",
        "fwhm_x",
        "fwhm_y",
        "ellipticity",
        "sharpness",
        "footprint_pixels",
        "psf_support_pixels",
        "quality_passed",
        "quality_class",
        "feature_class",
        "feature_class_label",
        "flags",
    )
    feature_rows = summarize_source_features(detection.sources)
    morphology_rows = summarize_feature_morphology(detection.sources)
    feature_spatial_rows = summarize_feature_spatial_distribution(
        detection.sources,
        detection.image_shape,
    )
    feature_by_class = {str(row["feature_class"]): row for row in feature_rows}
    with (output / "source_catalog.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields, extrasaction="ignore")
        writer.writeheader()
        for source in detection.sources:
            row = source.as_dict()
            row["flags"] = "|".join(source.flags)
            row["proposal_methods"] = "|".join(source.proposal_methods)
            row["proposal_scales"] = "|".join(f"{scale:.3f}" for scale in source.proposal_scales)
            row["repeated_code_values"] = "|".join(str(value) for value in source.repeated_code_values)
            row["quality_class"] = "quality" if source.quality_passed else "rejected"
            feature_class = classify_source_feature(source)
            row["feature_class"] = feature_class
            row["feature_class_label"] = feature_by_class[feature_class]["feature_class_label"]
            writer.writerow(row)

    flag_counts = Counter(flag for source in detection.sources for flag in source.flags)
    spatial_summary = _source_spatial_grid_artifacts(
        detection,
        output / "source_spatial_grid.csv",
        output / "source_spatial_density.png",
    )
    _source_quality_flag_chart(output / "source_quality_flags.png", detection, flag_counts)
    with (output / "source_feature_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        feature_fields = (
            "feature_class",
            "feature_class_label",
            "candidate_count",
            "quality_count",
            "rejected_count",
            "quality_fraction",
            "median_flux_snr",
            "max_flux_snr",
            "common_flags",
        )
        writer = csv.DictWriter(stream, fieldnames=feature_fields)
        writer.writeheader()
        writer.writerows(feature_rows)
    _source_feature_class_chart(output / "source_feature_classes.png", feature_rows)
    morphology_fields = (
        "feature_class",
        "feature_class_label",
        "candidate_count",
        "quality_count",
        "rejected_count",
        "quality_fraction",
        "high_flux_snr_threshold",
        "high_flux_snr_count",
        "high_flux_snr_rejected_count",
        "high_flux_snr_rejection_fraction",
        "median_peak_snr",
        "median_flux_snr",
        "common_flags",
        "peak_snr_p10",
        "peak_snr_median",
        "peak_snr_p90",
        "flux_snr_p10",
        "flux_snr_median",
        "flux_snr_p90",
        "filter_snr_p10",
        "filter_snr_median",
        "filter_snr_p90",
        "fwhm_px_p10",
        "fwhm_px_median",
        "fwhm_px_p90",
        "ellipticity_p10",
        "ellipticity_median",
        "ellipticity_p90",
        "sharpness_p10",
        "sharpness_median",
        "sharpness_p90",
        "psf_support_pixels_p10",
        "psf_support_pixels_median",
        "psf_support_pixels_p90",
        "footprint_pixels_p10",
        "footprint_pixels_median",
        "footprint_pixels_p90",
        "centroid_shift_px_p10",
        "centroid_shift_px_median",
        "centroid_shift_px_p90",
        "x_px_p10",
        "x_px_median",
        "x_px_p90",
        "y_px_p10",
        "y_px_median",
        "y_px_p90",
        "range_anomaly_fraction",
        "linear_artifact_fraction",
        "masked_or_edge_fraction",
        "crowded_blend_fraction",
        "spike_or_support_fraction",
        "weak_or_background_fraction",
        "shape_outlier_fraction",
    )
    with (output / "source_feature_morphology.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=morphology_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(morphology_rows)
    feature_spatial_fields = (
        "feature_class",
        "feature_class_label",
        "grid_size",
        "grid_row",
        "grid_column",
        "x_min_px",
        "x_max_px",
        "y_min_px",
        "y_max_px",
        "candidate_count",
        "quality_count",
        "rejected_count",
        "quality_fraction",
        "high_flux_snr_threshold",
        "high_flux_snr_count",
        "high_flux_snr_rejected_count",
        "median_flux_snr",
        "flux_snr_p90",
    )
    with (output / "source_feature_spatial.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=feature_spatial_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(feature_spatial_rows)
    parameter = detection.parameters
    quality_support_counts = Counter(
        int(source.psf_support_pixels) if source.psf_support_pixels is not None else -1
        for source in detection.quality_sources
    )
    summary_rows = [
        {"metric": "candidate_count", "value": detection.candidate_count, "definition": "NMS 后的候选峰数量"},
        {"metric": "returned_count", "value": detection.returned_count, "definition": "已计算源级属性的返回数量"},
        {"metric": "quality_count", "value": detection.star_count, "definition": "通过通量、形状、掩膜和有效性规则的数量"},
        {"metric": "rejected_count", "value": detection.rejected_count, "definition": "返回但未通过质量规则的数量"},
        {
            "metric": "quality_count_support_ge_4",
            "value": sum(count for support, count in quality_support_counts.items() if support >= 4),
            "definition": "诊断口径：质量源中 3×3 PSF 支持像素至少 4 个；不改变 quality_count",
        },
        {
            "metric": "quality_count_support_ge_5",
            "value": sum(count for support, count in quality_support_counts.items() if support >= 5),
            "definition": "诊断口径：质量源中 3×3 PSF 支持像素至少 5 个；不因接近外部数量而自动采用",
        },
        {"metric": "background_adu", "value": detection.background, "definition": "全图稳健背景基线"},
        {"metric": "noise_adu", "value": detection.noise, "definition": "全图稳健噪声尺度"},
        {"metric": "threshold_sigma", "value": parameter.get("threshold_sigma"), "definition": "匹配滤波候选阈值"},
        {"metric": "proposal_mode", "value": parameter.get("proposal_mode"), "definition": "宽筛选提案模式"},
        {"metric": "proposal_count_gaussian", "value": parameter.get("proposal_count_gaussian"), "definition": "Gaussian 算法内 NMS 后提案数"},
        {"metric": "proposal_count_dog_narrow", "value": parameter.get("proposal_count_dog_narrow"), "definition": "DoG 窄尺度算法内 NMS 后提案数"},
        {"metric": "proposal_count_dog_broad", "value": parameter.get("proposal_count_dog_broad"), "definition": "DoG 宽尺度算法内 NMS 后提案数"},
        {"metric": "proposal_count_starlet_s1", "value": parameter.get("proposal_count_starlet_s1"), "definition": "Starlet 第一尺度算法内 NMS 后提案数"},
        {"metric": "proposal_count_starlet_s2", "value": parameter.get("proposal_count_starlet_s2"), "definition": "Starlet 第二尺度算法内 NMS 后提案数"},
        {"metric": "min_distance_px", "value": parameter.get("min_distance"), "definition": "候选峰最小间距"},
        {"metric": "psf_fwhm_px", "value": parameter.get("psf_fwhm"), "definition": "Gaussian 匹配滤波 PSF FWHM"},
        {"metric": "min_flux_snr", "value": parameter.get("min_flux_snr"), "definition": "质量层通量 SNR 下限"},
        {
            "metric": "negative_overflow_limit",
            "value": parameter.get("negative_overflow_limit"),
            "definition": "有符号整型图像测光孔径内的极端负码审计线；不是相机满阱标定",
        },
        {
            "metric": "repeated_high_code_values",
            "value": "|".join(str(value) for value in parameter.get("repeated_high_code_values", ())),
            "definition": "全幅频次异常的高位整型码；只作数据有效性先验，不是饱和标定",
        },
        {
            "metric": "repeated_high_code_min_count",
            "value": parameter.get("repeated_high_code_min_count"),
            "definition": "全幅重复高位码的工程审计最小频次",
        },
        {
            "metric": "close_pair_guarded_source_count",
            "value": parameter.get("close_pair_guarded_source_count"),
            "definition": "近邻 Gaussian 双 PSF 证据不足而从质量层降级的源数；候选仍保留",
        },
        {
            "metric": "min_psf_support_pixels",
            "value": parameter.get("min_psf_support_pixels"),
            "definition": "候选峰中心 3×3 核心中必须超过支持阈值的最少像素数",
        },
        {"metric": "truncated", "value": int(detection.truncated), "definition": "是否因显式返回上限截断"},
        {"metric": "spatial_grid_size", "value": int(spatial_summary["spatial_grid_size"]), "definition": "空间诊断每轴网格数量"},
        {"metric": "quality_density_cv", "value": spatial_summary["quality_density_cv"], "definition": "16×16 网格质量源计数的变异系数；描述空间不均匀，不等同于灵敏度"},
        {"metric": "quality_fraction_median", "value": spatial_summary["quality_fraction_median"], "definition": "非空网格中质量通过率中位数"},
        {"metric": "quality_fraction_min", "value": spatial_summary["quality_fraction_min"], "definition": "非空网格中质量通过率最小值；需结合边缘/掩膜解释"},
        {"metric": "quality_fraction_max", "value": spatial_summary["quality_fraction_max"], "definition": "非空网格中质量通过率最大值"},
        {
            "metric": "feature_class_count",
            "value": len(feature_rows),
            "definition": "互斥首要特征类别数量；类别内仍保留原始重叠 flags",
        },
    ]
    summary_rows.extend(
        {
            "metric": f"feature:{row['feature_class']}:candidate_count",
            "value": row["candidate_count"],
            "definition": f"首要特征类别：{row['feature_class_label']}；互斥审计计数",
        }
        for row in feature_rows
    )
    summary_rows.extend(
        {
            "metric": f"feature:{row['feature_class']}:quality_count",
            "value": row["quality_count"],
            "definition": f"首要特征类别：{row['feature_class_label']}；其中通过质量层的数量",
        }
        for row in feature_rows
    )
    summary_rows.extend(
        {"metric": f"flag:{flag}", "value": count, "definition": "可重叠的源拒绝/审计标志计数"}
        for flag, count in sorted(flag_counts.items())
    )
    summary_rows.extend(
        {
            "metric": f"psf_support_pixels:{support}",
            "value": quality_support_counts.get(support, 0),
            "definition": "质量源的 3×3 PSF 支持像素分布；用于人工抽检，不是独立真值标签",
        }
        for support in range(10)
    )
    with (output / "source_quality_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value", "definition"))
        writer.writeheader()
        writer.writerows(summary_rows)

    if frame is not None:
        write_detection_source_cutouts(frame, detection, output)
        empirical_psf = estimate_empirical_psf(
            np.asarray(frame.data),
            detection.sources,
            support_radius=7,
            max_sources=64,
        )
        psf_similarity_rows = summarize_feature_psf_similarity(
            frame,
            detection.sources,
            psf=empirical_psf,
        )
        psf_similarity_fields = (
            "feature_class",
            "feature_class_label",
            "psf_model",
            "psf_source_count",
            "correlation_threshold",
            "sample_count",
            "valid_count",
            "valid_fraction",
            "correlation_ge_threshold_count",
            "correlation_ge_threshold_fraction",
            "psf_correlation_p10",
            "psf_correlation_median",
            "psf_correlation_p90",
            "relative_residual_p10",
            "relative_residual_median",
            "relative_residual_p90",
            "central_energy_fraction_p10",
            "central_energy_fraction_median",
            "central_energy_fraction_p90",
            "fitted_amplitude_adu_median",
            "sample_detection_ids",
        )
        with (output / "source_feature_psf_similarity.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=psf_similarity_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(psf_similarity_rows)
        spatial_psf_rows = summarize_spatial_psf_similarity(
            frame,
            detection.sources,
            global_psf=empirical_psf,
            per_class=spatial_psf_per_class,
            grid_size=spatial_psf_grid_size,
        )
        spatial_psf_fields = (
            "feature_class",
            "feature_class_label",
            "grid_size",
            "global_psf_source_count",
            "local_template_cell_count",
            "local_template_available_cell_count",
            "local_template_source_count_median",
            "sample_count",
            "global_valid_count",
            "local_valid_count",
            "local_fallback_count",
            "correlation_threshold",
            "global_correlation_median",
            "local_correlation_median",
            "paired_correlation_count",
            "local_correlation_improved_count",
            "local_correlation_improved_fraction",
            "global_residual_median",
            "local_residual_median",
            "paired_residual_count",
            "local_residual_reduced_count",
            "local_residual_reduced_fraction",
            "median_correlation_delta_local_minus_global",
            "median_residual_delta_local_minus_global",
            "sample_detection_ids",
        )
        with (output / "source_feature_psf_spatial.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=spatial_psf_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(spatial_psf_rows)

    from .innovation import _line_chart

    source_snr = np.asarray(
        [float(source.flux_snr) for source in detection.sources if source.flux_snr is not None and np.isfinite(source.flux_snr) and source.flux_snr > 0],
        dtype=np.float64,
    )
    filter_snr = np.asarray(
        [float(source.filter_snr) for source in detection.sources if source.filter_snr is not None and np.isfinite(source.filter_snr) and source.filter_snr > 0],
        dtype=np.float64,
    )
    rank_count = max(source_snr.size, filter_snr.size)
    if rank_count:
        rank = list(range(1, rank_count + 1))
        flux_rank = np.full(rank_count, np.nan, dtype=np.float64)
        filter_rank = np.full(rank_count, np.nan, dtype=np.float64)
        if source_snr.size:
            flux_rank[: source_snr.size] = np.log10(np.sort(source_snr)[::-1])
        if filter_snr.size:
            filter_rank[: filter_snr.size] = np.log10(np.sort(filter_snr)[::-1])
        _line_chart(
            output / "source_snr_rank.png",
            "SOURCE QUALITY AUDIT · SNR RANK",
            rank,
            (
                ("log10 flux SNR", flux_rank.tolist(), "#4f9b83"),
                ("log10 filter SNR", filter_rank.tolist(), "#d79432"),
            ),
            y_label="log10 SNR",
            x_label="source rank (descending)",
        )
    else:
        _line_chart(output / "source_snr_rank.png", "SOURCE QUALITY AUDIT · SNR RANK", [], (), y_label="log10 SNR", x_label="source rank")

    if source_snr.size:
        upper = max(10.0, float(np.percentile(source_snr, 99.5)))
        bins = np.linspace(0.0, upper, 33)
        counts, edges = np.histogram(source_snr, bins=bins)
        centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
        _line_chart(
            output / "source_flux_snr_distribution.png",
            "SOURCE QUALITY AUDIT · FLUX SNR DISTRIBUTION",
            centers,
            (("sources / bin", counts.astype(float).tolist(), "#4f9b83"),),
            y_label="sources / bin",
            x_label="flux SNR (linear bins; upper clipped at p99.5)",
        )
    else:
        _line_chart(output / "source_flux_snr_distribution.png", "SOURCE QUALITY AUDIT · FLUX SNR DISTRIBUTION", [], (), y_label="sources / bin", x_label="flux SNR")
    return output


def write_detection_source_cutouts(
    frame: FitsFrame | str | Path,
    detection: DetectionResult,
    out_dir: str | Path,
    *,
    per_group: int = 8,
    radius_px: int = 20,
    scale: int = 4,
) -> Path:
    """Write deterministic source-quality cutouts for human inspection.

    The contact sheet deliberately samples several SNR/flag strata and one
    deterministic sample group for every mutually exclusive feature class,
    instead of showing only the brightest sources.  It is an audit artifact,
    not a labelled truth set: a cutout can look star-like and still be an
    unrelated background fluctuation, especially near the single-frame
    threshold.
    """

    if per_group < 1 or radius_px < 3 or scale < 1:
        raise ValueError("per_group, radius_px and scale must be positive")
    source_frame = frame if isinstance(frame, FitsFrame) else read_fits(frame)
    values = np.asarray(source_frame.data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cutout_dir = output / "source_cutouts"
    cutout_dir.mkdir(parents=True, exist_ok=True)

    def signal_snr(source: Detection) -> float:
        value = source.flux_snr if source.flux_snr is not None else source.snr
        return float(value) if value is not None and np.isfinite(value) else float("-inf")

    groups: tuple[tuple[str, str, Callable[[Detection], bool]], ...] = (
        (
            "quality_near_threshold",
            "QUALITY 5≤flux SNR<6",
            lambda source: source.quality_passed and 5.0 <= signal_snr(source) < 6.0,
        ),
        (
            "quality_mid_snr",
            "QUALITY 6≤flux SNR<20",
            lambda source: source.quality_passed and 6.0 <= signal_snr(source) < 20.0,
        ),
        (
            "quality_bright",
            "QUALITY flux SNR≥20",
            lambda source: source.quality_passed and signal_snr(source) >= 20.0,
        ),
        (
            "quality_extra_only",
            "QUALITY proposed only by supplemental algorithms",
            lambda source: source.quality_passed
            and bool(source.proposal_methods)
            and "gaussian" not in source.proposal_methods,
        ),
        (
            "rejected_extra_only",
            "REJECTED proposed only by supplemental algorithms",
            lambda source: not source.quality_passed
            and bool(source.proposal_methods)
            and "gaussian" not in source.proposal_methods,
        ),
        (
            "quality_starlet_added",
            "QUALITY non-Gaussian candidate hit by Starlet",
            lambda source: source.quality_passed
            and "gaussian" not in source.proposal_methods
            and any(method.startswith("starlet_") for method in source.proposal_methods),
        ),
        (
            "rejected_starlet_added",
            "REJECTED non-Gaussian candidate hit by Starlet",
            lambda source: not source.quality_passed
            and "gaussian" not in source.proposal_methods
            and any(method.startswith("starlet_") for method in source.proposal_methods),
        ),
        (
            "quality_deblended",
            "QUALITY pair-PSF deblend supported",
            lambda source: source.quality_passed
            and source.deblend_delta_bic is not None
            and "UNRESOLVED_BLEND" not in source.flags,
        ),
        (
            "rejected_deblended",
            "REJECTED despite pair-PSF support",
            lambda source: not source.quality_passed
            and source.deblend_delta_bic is not None
            and "UNRESOLVED_BLEND" not in source.flags,
        ),
        (
            "rejected_low_flux_snr",
            "REJECTED LOW_FLUX_SNR",
            lambda source: "LOW_FLUX_SNR" in source.flags,
        ),
        (
            "rejected_negative_overflow",
            "REJECTED NEGATIVE_OVERFLOW",
            lambda source: "NEGATIVE_OVERFLOW" in source.flags,
        ),
        (
            "rejected_spike",
            "REJECTED SPIKE",
            lambda source: "SPIKE" in source.flags,
        ),
        (
            "rejected_masked",
            "REJECTED MASKED",
            lambda source: "MASKED" in source.flags,
        ),
        (
            "rejected_line_artifact",
            "REJECTED LINE_ARTIFACT",
            lambda source: "LINE_ARTIFACT" in source.flags,
        ),
        (
            "rejected_edge",
            "REJECTED EDGE",
            lambda source: "EDGE" in source.flags,
        ),
    )
    # The fixed diagnostic strata above are useful for regression checks, but
    # they do not show the mutually exclusive feature classes used by the
    # paper.  Add one deterministic sample group per class so that every
    # category can be checked against the original pixels, not only against
    # aggregate CSV statistics.  Capture the loop value explicitly: otherwise
    # all lambdas would inspect the final class.
    groups += tuple(
        (
            f"feature_{feature_class}",
            f"FEATURE {feature_label}",
            lambda source, expected_class=feature_class: classify_source_feature(source) == expected_class,
        )
        for feature_class, feature_label in _SOURCE_FEATURE_CLASSES
    )

    def selected_sources(predicate: Callable[[Detection], bool]) -> list[Detection]:
        rows = sorted(
            (source for source in detection.sources if predicate(source)),
            key=lambda source: (signal_snr(source), source.detection_id),
        )
        if len(rows) <= per_group:
            return rows
        indices = np.rint(np.linspace(0, len(rows) - 1, per_group)).astype(int)
        return [rows[int(index)] for index in indices]

    manifest_fields = (
        "group",
        "group_label",
        "sample_index",
        "detection_id",
        "centroid_x",
        "centroid_y",
        "peak_x",
        "peak_y",
        "centroid_shift_px",
        "flux_snr",
        "proposal_snr",
        "proposal_methods",
        "nearest_gaussian_px",
        "deblend_delta_bic",
        "deblend_component_snr",
        "fwhm",
        "ellipticity",
        "sharpness",
        "quality_passed",
        "flags",
        "file",
    )
    manifest_rows: list[dict[str, object]] = []
    contact_items: list[tuple[str, str, Image.Image, Detection]] = []
    crop_size = 2 * int(radius_px) + 1
    median_value = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0

    def peak_position(source: Detection) -> tuple[float, float]:
        if (
            source.peak_x is not None
            and source.peak_y is not None
            and np.isfinite(source.peak_x)
            and np.isfinite(source.peak_y)
        ):
            return float(source.peak_x), float(source.peak_y)
        return float(source.x), float(source.y)

    for group_name, group_label, predicate in groups:
        rows = selected_sources(predicate)
        for sample_index, source in enumerate(rows, start=1):
            marker_x, marker_y = peak_position(source)
            center_x = int(round(marker_x))
            center_y = int(round(marker_y))
            x_start = center_x - radius_px
            y_start = center_y - radius_px
            crop = np.full((crop_size, crop_size), median_value, dtype=np.float32)
            source_y0 = max(0, y_start)
            source_y1 = min(values.shape[0], y_start + crop_size)
            source_x0 = max(0, x_start)
            source_x1 = min(values.shape[1], x_start + crop_size)
            if source_y1 > source_y0 and source_x1 > source_x0:
                crop_y0 = source_y0 - y_start
                crop_y1 = crop_y0 + source_y1 - source_y0
                crop_x0 = source_x0 - x_start
                crop_x1 = crop_x0 + source_x1 - source_x0
                crop[crop_y0:crop_y1, crop_x0:crop_x1] = values[source_y0:source_y1, source_x0:source_x1]
            finite = crop[np.isfinite(crop)]
            low = float(np.percentile(finite, 5.0)) if finite.size else median_value
            high = float(np.percentile(finite, 99.7)) if finite.size else low + 1.0
            if high <= low:
                high = low + 1.0
            normalized = np.clip((crop - low) / (high - low), 0.0, 1.0)
            normalized[~np.isfinite(normalized)] = 0.0
            image = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L").convert("RGB")
            image = image.resize((crop_size * scale, crop_size * scale), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(image)
            draw_x = (marker_x - x_start) * scale
            draw_y = (marker_y - y_start) * scale
            color = "#4fd6aa" if source.quality_passed else "#f0b34b"
            marker_radius = max(4, scale * 2)
            draw.ellipse(
                (draw_x - marker_radius, draw_y - marker_radius, draw_x + marker_radius, draw_y + marker_radius),
                outline=color,
                width=max(1, scale // 2),
            )
            draw.line((draw_x - marker_radius * 1.4, draw_y, draw_x + marker_radius * 1.4, draw_y), fill=color, width=1)
            draw.line((draw_x, draw_y - marker_radius * 1.4, draw_x, draw_y + marker_radius * 1.4), fill=color, width=1)
            file_name = f"{group_name}_{sample_index:02d}_id{source.detection_id:05d}.png"
            image.save(cutout_dir / file_name)
            manifest_rows.append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "sample_index": sample_index,
                    "detection_id": source.detection_id,
                    "centroid_x": source.x,
                    "centroid_y": source.y,
                    "peak_x": marker_x,
                    "peak_y": marker_y,
                    "centroid_shift_px": source.centroid_shift_px,
                    "flux_snr": source.flux_snr,
                    "proposal_snr": source.proposal_snr,
                    "proposal_methods": "|".join(source.proposal_methods),
                    "nearest_gaussian_px": source.nearest_gaussian_px,
                    "deblend_delta_bic": source.deblend_delta_bic,
                    "deblend_component_snr": source.deblend_component_snr,
                    "fwhm": source.fwhm,
                    "ellipticity": source.ellipticity,
                    "sharpness": source.sharpness,
                    "quality_passed": source.quality_passed,
                    "flags": "|".join(source.flags),
                    "file": f"source_cutouts/{file_name}",
                }
            )
            contact_items.append((group_name, group_label, image, source))

    with (output / "source_cutout_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    columns = 4
    caption_height = 30
    tile_width = crop_size * scale
    tile_height = tile_width + caption_height
    rows_count = max(1, int(math.ceil(len(contact_items) / columns)))
    sheet = Image.new("RGB", (columns * tile_width, 32 + rows_count * tile_height), "#182033")
    sheet_draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    sheet_draw.text((6, 8), "SOURCE CUTOUT AUDIT · colors show algorithmic quality class, not truth", fill="#f6f1e7", font=font)
    for index, (group_name, group_label, image, source) in enumerate(contact_items):
        x0 = (index % columns) * tile_width
        y0 = 32 + (index // columns) * tile_height
        sheet.paste(image, (x0, y0))
        snr_text = signal_snr(source)
        caption = f"{group_name} · ID {source.detection_id} · SNR {snr_text:.1f}"
        sheet_draw.rectangle((x0, y0 + tile_width, x0 + tile_width, y0 + tile_height), fill="#182033")
        sheet_draw.text((x0 + 3, y0 + tile_width + 3), caption[:52], fill="#bce0d1", font=font)
    sheet.save(output / "source_cutout_contact_sheet.png")
    return output


def write_sequence_source_audit(
    sequence: Mapping[str, Any] | Any,
    out_dir: str | Path,
    *,
    per_group: int = 6,
    radius_px: int = 10,
    scale: int = 2,
) -> Path:
    """导出 15 帧点源轨迹的时序小图和抽样清单。

    该审计专门回答“一个点在多帧中是否仍保持点状”这个问题。它把严格
    静态轨迹、低置信持续轨迹和低持续性瞬态轨迹分开抽样；缺帧位置仍按
    配准坐标回投到原图，并用灰色边框明确标出缺帧。输出只用于人工复核，
    不把抽样小图或轨迹分类写成逐星真值。

    ``sequence`` 可以是 ``SequenceResult``，也可以是其 ``as_dict()`` 结果。
    每条轨迹只抽取确定性的分位样本，避免审计图被最亮或最容易展示的源
    主导。原始 FITS 只读，不会被修改。
    """

    if per_group < 1 or radius_px < 3 or scale < 1:
        raise ValueError("per_group, radius_px and scale must be positive")
    payload: Mapping[str, Any]
    if hasattr(sequence, "as_dict"):
        rendered = sequence.as_dict()
        if not isinstance(rendered, Mapping):
            raise ValueError("sequence.as_dict() must return a mapping")
        payload = rendered
    elif isinstance(sequence, Mapping):
        payload = sequence
    else:
        raise TypeError("sequence must be a SequenceResult or mapping")

    raw_frames = payload.get("frames", [])
    raw_tracks = payload.get("tracks", [])
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
        raise ValueError("sequence frames must be a sequence")
    if not isinstance(raw_tracks, Sequence) or isinstance(raw_tracks, (str, bytes)):
        raise ValueError("sequence tracks must be a sequence")
    if not raw_frames:
        raise ValueError("sequence frames cannot be empty")

    frame_rows = [row for row in raw_frames if isinstance(row, Mapping)]
    if len(frame_rows) != len(raw_frames):
        raise ValueError("sequence frame rows must be mappings")
    frame_paths: list[Path] = []
    for row in frame_rows:
        raw_path = row.get("path")
        if raw_path is None:
            raise ValueError("sequence frame is missing path")
        path = Path(str(raw_path))
        if not path.is_file():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"sequence frame does not exist: {raw_path}")
        frame_paths.append(path)

    frame_data = [np.asarray(read_fits(path).data, dtype=np.float32) for path in frame_paths]
    frame_count = len(frame_data)
    shifts_raw = payload.get("cumulative_shifts", [])
    shifts = np.zeros((frame_count, 2), dtype=np.float64)
    if isinstance(shifts_raw, Sequence) and not isinstance(shifts_raw, (str, bytes)):
        for index, value in enumerate(shifts_raw[:frame_count]):
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
                try:
                    shifts[index] = (float(value[0]), float(value[1]))
                except (TypeError, ValueError):
                    continue

    def track_points(track: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        points = track.get("points", [])
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            return []
        valid = [point for point in points if isinstance(point, Mapping)]
        valid.sort(key=lambda point: int(point.get("frame_index", 0)))
        return valid

    def track_snr(track: Mapping[str, Any]) -> float:
        values = []
        for point in track_points(track):
            value = point.get("flux_snr")
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values.append(number)
        return min(values) if values else float("-inf")

    def select_tracks(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        ordered = sorted(rows, key=lambda row: (track_snr(row), int(row.get("track_id", 0))))
        if len(ordered) <= per_group:
            return list(ordered)
        indices = np.rint(np.linspace(0, len(ordered) - 1, per_group)).astype(int)
        return [ordered[int(index)] for index in indices]

    persistent_threshold = int(payload.get("persistent_min_presence", max(3, int(np.ceil(frame_count * 0.5)))) or 0)
    groups: tuple[tuple[str, str, str, Callable[[Mapping[str, Any]], bool]], ...] = (
        (
            "static_strict",
            "严格静态 · 至少约 80% 帧",
            "#4fd6aa",
            lambda track: str(track.get("classification", "")) == "static",
        ),
        (
            "persistent",
            f"持续候选 · {persistent_threshold}/{frame_count} 帧门槛",
            "#72b9d4",
            lambda track: str(track.get("classification", "")) == "persistent",
        ),
        (
            "transient_one_frame",
            "瞬态 · 1 帧",
            "#d99a4e",
            lambda track: str(track.get("classification", "")) == "transient" and len(track_points(track)) == 1,
        ),
        (
            "transient_near_threshold",
            "瞬态 · 低于持续门槛",
            "#9c8b6a",
            lambda track: str(track.get("classification", "")) == "transient"
            and 1 < len(track_points(track)) < persistent_threshold,
        ),
    )

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    track_dir = output / "source_track_cutouts"
    track_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 11)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
    median_values = []
    for data in frame_data:
        finite = data[np.isfinite(data)]
        median_values.append(float(np.median(finite)) if finite.size else 0.0)

    def cutout(
        data: np.ndarray,
        x: float,
        y: float,
        *,
        label: str,
        colour: str | None,
        missing: bool,
        background_value: float,
    ) -> Image.Image:
        height, width = data.shape
        size = 2 * radius_px + 1
        center_x, center_y = int(round(x)), int(round(y))
        left, top = center_x - radius_px, center_y - radius_px
        crop = np.full((size, size), background_value, dtype=np.float32)
        source_left, source_top = max(0, left), max(0, top)
        source_right, source_bottom = min(width, left + size), min(height, top + size)
        if source_right > source_left and source_bottom > source_top:
            crop_top = source_top - top
            crop_left = source_left - left
            crop[crop_top : crop_top + source_bottom - source_top, crop_left : crop_left + source_right - source_left] = data[
                source_top:source_bottom,
                source_left:source_right,
            ]
        finite = crop[np.isfinite(crop)]
        low, high = np.percentile(finite, [2.0, 99.5]) if finite.size else (0.0, 1.0)
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        normalized = np.clip((crop - low) / (high - low), 0.0, 1.0)
        normalized[~np.isfinite(normalized)] = 0.0
        image = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L").convert("RGB")
        image = image.resize((size * scale, size * scale), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, min(image.width - 1, 50), 12), fill="#182033")
        draw.text((2, 2), label[:9], fill="#f6f1e7", font=font)
        if not missing and colour is not None:
            marker_x = (float(x) - left) * scale
            marker_y = (float(y) - top) * scale
            marker_radius = max(3, scale * 2)
            draw.ellipse(
                (
                    marker_x - marker_radius,
                    marker_y - marker_radius,
                    marker_x + marker_radius,
                    marker_y + marker_radius,
                ),
                outline=colour,
                width=max(1, scale),
            )
            draw.line((marker_x - marker_radius * 1.4, marker_y, marker_x + marker_radius * 1.4, marker_y), fill=colour, width=1)
            draw.line((marker_x, marker_y - marker_radius * 1.4, marker_x, marker_y + marker_radius * 1.4), fill=colour, width=1)
        if missing:
            draw.rectangle((0, 0, image.width - 1, image.height - 1), outline="#697386", width=1)
        return image

    manifest_fields = (
        "group",
        "group_label",
        "track_id",
        "classification",
        "presence",
        "frame_indices",
        "min_flux_snr",
        "median_flux_snr",
        "max_flux_snr",
        "displacement_px",
        "fit_rms_px",
        "file",
    )
    manifest_rows: list[dict[str, object]] = []
    group_sheets: list[Image.Image] = []
    for group_name, group_label, colour, predicate in groups:
        candidates = [
            track
            for track in raw_tracks
            if isinstance(track, Mapping) and predicate(track) and track_points(track)
        ]
        selected = select_tracks(candidates)
        if not selected:
            continue
        track_images: list[Image.Image] = []
        for track in selected:
            points = track_points(track)
            point_by_frame = {int(point.get("frame_index", -1)): point for point in points}
            aligned_x = float(np.median([float(point.get("aligned_x", point.get("x", 0.0))) for point in points]))
            aligned_y = float(np.median([float(point.get("aligned_y", point.get("y", 0.0))) for point in points]))
            tiles: list[Image.Image] = []
            frame_indices: list[int] = []
            snr_values: list[float] = []
            for frame_index, data in enumerate(frame_data):
                point = point_by_frame.get(frame_index)
                if point is None:
                    x = aligned_x + shifts[frame_index, 0]
                    y = aligned_y + shifts[frame_index, 1]
                    tile = cutout(
                        data,
                        x,
                        y,
                        label=f"F{frame_index + 1:02d} —",
                        colour=None,
                        missing=True,
                        background_value=median_values[frame_index],
                    )
                else:
                    try:
                        x = float(point.get("x", aligned_x + shifts[frame_index, 0]))
                        y = float(point.get("y", aligned_y + shifts[frame_index, 1]))
                    except (TypeError, ValueError):
                        x = aligned_x + shifts[frame_index, 0]
                        y = aligned_y + shifts[frame_index, 1]
                    value = point.get("flux_snr")
                    try:
                        snr = float(value)
                    except (TypeError, ValueError):
                        snr = float("nan")
                    label = f"F{frame_index + 1:02d} {snr:.0f}" if np.isfinite(snr) else f"F{frame_index + 1:02d} ?"
                    tile = cutout(
                        data,
                        x,
                        y,
                        label=label,
                        colour=colour,
                        missing=False,
                        background_value=median_values[frame_index],
                    )
                    if np.isfinite(snr):
                        snr_values.append(snr)
                    frame_indices.append(frame_index + 1)
                tiles.append(tile)
            tile_width, tile_height = tiles[0].size
            track_image = Image.new("RGB", (frame_count * tile_width, tile_height + 25), "#182033")
            draw = ImageDraw.Draw(track_image)
            for frame_index, tile in enumerate(tiles):
                track_image.paste(tile, (frame_index * tile_width, 25))
            presence = len(points)
            min_snr = min(snr_values) if snr_values else float("nan")
            median_snr = float(np.median(snr_values)) if snr_values else float("nan")
            max_snr = max(snr_values) if snr_values else float("nan")
            track_id = int(track.get("track_id", 0))
            caption = (
                f"{group_name} · ID {track_id:05d} · {presence}/{frame_count}F · "
                f"min/med/max SNR {min_snr:.1f}/{median_snr:.1f}/{max_snr:.1f} · "
                f"disp {float(track.get('displacement_px', 0.0)):.2f}px"
            )
            draw.text((3, 6), caption[:180], fill="#f6f1e7", font=font)
            file_name = f"{group_name}_track{track_id:05d}.png"
            track_image.save(track_dir / file_name)
            track_images.append(track_image)
            manifest_rows.append(
                {
                    "group": group_name,
                    "group_label": group_label,
                    "track_id": track_id,
                    "classification": str(track.get("classification", "")),
                    "presence": presence,
                    "frame_indices": "|".join(str(value) for value in frame_indices),
                    "min_flux_snr": min_snr,
                    "median_flux_snr": median_snr,
                    "max_flux_snr": max_snr,
                    "displacement_px": track.get("displacement_px"),
                    "fit_rms_px": track.get("fit_rms_px"),
                    "file": str(Path("source_track_cutouts") / file_name),
                }
            )
        group_height = 25 + sum(image.height for image in track_images)
        group_width = max(image.width for image in track_images)
        group_sheet = Image.new("RGB", (group_width, group_height), "#182033")
        draw = ImageDraw.Draw(group_sheet)
        draw.text((3, 5), f"{group_label} · sampled {len(selected)}/{len(candidates)} · color is algorithmic class, not truth", fill="#bce0d1", font=font)
        offset = 25
        for image in track_images:
            group_sheet.paste(image, (0, offset))
            offset += image.height
        group_sheets.append(group_sheet)

    with (output / "source_track_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "frame_count": frame_count,
        "strict_static_count": sum(str(track.get("classification", "")) == "static" for track in raw_tracks if isinstance(track, Mapping)),
        "persistent_count": sum(str(track.get("classification", "")) == "persistent" for track in raw_tracks if isinstance(track, Mapping)),
        "transient_count": sum(str(track.get("classification", "")) == "transient" for track in raw_tracks if isinstance(track, Mapping)),
        "persistent_min_presence": persistent_threshold,
        "sampled_track_count": len(manifest_rows),
        "note": "抽样时序图用于人工检查点状持续性，不是逐星真值；缺帧位置以灰框显示。",
    }
    (output / "source_track_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if group_sheets:
        width = max(image.width for image in group_sheets)
        height = sum(image.height for image in group_sheets)
        sheet = Image.new("RGB", (width, height), "#182033")
        offset = 0
        for image in group_sheets:
            sheet.paste(image, (0, offset))
            offset += image.height
    else:
        sheet = Image.new("RGB", (max(1, frame_count * (2 * radius_px + 1) * scale), 40), "#182033")
        ImageDraw.Draw(sheet).text((4, 12), "SOURCE TRACK AUDIT · no tracks selected", fill="#f6f1e7", font=font)
    sheet.save(output / "source_track_contact_sheet.png")
    return output


def run_detection_psf_sweep(
    path: str | Path | FitsFrame,
    *,
    psf_levels: Sequence[float] = (2.0, 2.5, 3.0, 3.5, 4.0),
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    max_sources: int | None = None,
    reject_linear_artifacts: bool = True,
) -> tuple[DetectionPSFSweepRow, ...]:
    """在同一真实图像上固定阈值并扫描匹配滤波 PSF FWHM。"""

    if not psf_levels:
        raise ValueError("psf_levels cannot be empty")
    if threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1:
        raise ValueError("detection PSF sweep parameters must be positive")
    if background_box_size < 16 or min_flux_snr <= 0 or not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("background_box_size/min_flux_snr/min_psf_support_pixels are invalid")
    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    rows: list[DetectionPSFSweepRow] = []
    for level in psf_levels:
        psf_fwhm = float(level)
        if psf_fwhm <= 0:
            raise ValueError("psf_levels must be positive")
        detection = detect_sources(
            frame.data,
            mask=auxiliary_mask(frame.data.shape),
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            max_sources=max_sources,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            reject_linear_artifacts=reject_linear_artifacts,
        )
        line_artifacts = sum("LINE_ARTIFACT" in source.flags for source in detection.sources)
        rows.append(
            DetectionPSFSweepRow(
                psf_fwhm_px=psf_fwhm,
                candidate_count=detection.candidate_count,
                returned_count=detection.returned_count,
                quality_count=detection.star_count,
                rejected_count=max(0, detection.returned_count - detection.star_count),
                line_artifact_count=line_artifacts,
                background_adu=float(detection.background),
                noise_adu=float(detection.noise),
            )
        )
    return tuple(rows)


def write_detection_psf_sweep_artifacts(
    rows: Sequence[DetectionPSFSweepRow],
    out_dir: str | Path,
) -> Path:
    """写 PSF FWHM 敏感性 CSV 和数量曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "detection_psf_sweep.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].as_dict()) if rows else ["psf_fwhm_px"])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    from .innovation import _line_chart

    _line_chart(
        output / "detection_psf_sweep.png",
        "REAL FRAME · PSF FWHM SENSITIVITY",
        [row.psf_fwhm_px for row in rows],
        (
            ("candidate peaks", [row.candidate_count for row in rows], "#d79432"),
            ("quality sources", [row.quality_count for row in rows], "#4f9b83"),
        ),
        y_label="sources",
        x_label="matched-filter PSF FWHM / px",
    )
    return output


def run_detection_threshold_sweep(
    path: str | Path | FitsFrame,
    *,
    threshold_levels: Sequence[float] = (4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0),
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 3.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    max_sources: int | None = None,
    reject_linear_artifacts: bool = True,
) -> tuple[DetectionSweepRow, ...]:
    """在同一真实图像上扫描候选阈值，保留可比较的数量口径。"""

    if not threshold_levels:
        raise ValueError("threshold_levels cannot be empty")
    if (
        min_distance < 1
        or aperture_radius < 1
        or psf_fwhm <= 0
        or background_box_size < 8
        or min_flux_snr <= 0
        or not 1 <= min_psf_support_pixels <= 9
    ):
        raise ValueError("detection sweep parameters must be positive")
    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    rows: list[DetectionSweepRow] = []
    for level in threshold_levels:
        threshold = float(level)
        if not 0 < threshold <= 100:
            raise ValueError("threshold_levels must be between 0 and 100 sigma")
        detection = detect_sources(
            frame.data,
            mask=auxiliary_mask(frame.data.shape),
            threshold_sigma=threshold,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            max_sources=max_sources,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            reject_linear_artifacts=reject_linear_artifacts,
        )
        line_artifacts = sum("LINE_ARTIFACT" in source.flags for source in detection.sources)
        rows.append(
            DetectionSweepRow(
                threshold_sigma=threshold,
                candidate_count=detection.candidate_count,
                returned_count=detection.returned_count,
                quality_count=detection.star_count,
                rejected_count=max(0, detection.returned_count - detection.star_count),
                line_artifact_count=line_artifacts,
                background_adu=float(detection.background),
                noise_adu=float(detection.noise),
            )
        )
    return tuple(rows)


def write_detection_sweep_artifacts(rows: Sequence[DetectionSweepRow], out_dir: str | Path) -> Path:
    """写真实图像阈值扫描的 CSV 和数量曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "detection_threshold_sweep.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].as_dict()) if rows else ["threshold_sigma"])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    from .innovation import _line_chart

    _line_chart(
        output / "detection_threshold_sweep.png",
        "REAL FRAME · THRESHOLD SENSITIVITY",
        [row.threshold_sigma for row in rows],
        (
            ("candidate peaks", [row.candidate_count for row in rows], "#d79432"),
            ("quality sources", [row.quality_count for row in rows], "#4f9b83"),
        ),
        y_label="sources",
        x_label="candidate threshold / sigma",
    )
    return output
