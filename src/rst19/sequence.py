"""15 帧星点稳定性和运动目标关联。

这里不把帧间最近邻直接叫作运动目标。先用高质量、高 SNR 源估计每一帧
相对于上一帧的全局平移，再在配准坐标中关联检测源；只有具有足够帧数、
并且相对于静态背景呈现可重复位移的轨迹，才标为 ``moving``。这是一种
适用于当前数据的可解释基线，不替代带旋转/畸变项的完整图像配准。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .detection import Detection, DetectionResult, _working_mask, build_background_model, detect_sources, sigma_clipped_stats
from .fits import auxiliary_mask, read_fits
from .models import FitsFrame
from .pipeline import FrameAnalysis, analyze_frame


# 15 帧关联的默认工作集上限。候选总数仍然完整统计，只有进入跨帧
# 配准/点轨迹关联的源级测量数量受限；单图分析不使用这个上限。
DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT = 6000
# 当前 4096×4096 数据在开发机实测 4 个 worker 略快于 3 个；仍提供
# CLI ``--workers 1/2/3`` 给内存紧张或需要降低峰值带宽争用的机器。
DEFAULT_SEQUENCE_WORKERS = 4
DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT = 4_096
# 时间参考图只用于“宽筛提案”。当前数据的亮源实测 FWHM 约在 1.9 px
# 附近，而序列默认检测器参数是 3 px；只用一个尺度会让窄 PSF 源在参考图
# 中被匹配滤波稀释。用相对基准 FWHM 的小型尺度组覆盖窄/基准/宽三类
# 响应，后续仍必须通过逐帧原图支持、形状和持续性细筛。
TEMPORAL_PSF_FWHM_FACTORS = (0.65, 1.0, 1.35)


def _temporal_psf_fwhm_bank(psf_fwhm: float) -> tuple[float, ...]:
    """返回时间宽筛使用的 PSF FWHM 尺度组。"""

    if psf_fwhm <= 0:
        raise ValueError("psf_fwhm must be positive")
    return tuple(
        sorted(
            {
                round(max(float(psf_fwhm) * factor, 0.8), 6)
                for factor in TEMPORAL_PSF_FWHM_FACTORS
            }
        )
    )


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """轨迹中的一次源观测；aligned 坐标已扣除全局帧间平移。"""

    frame_index: int
    detection_id: int
    x: float
    y: float
    aligned_x: float
    aligned_y: float
    flux_snr: float | None
    quality_passed: bool = True
    candidate_snr: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "aligned_x": self.aligned_x,
            "aligned_y": self.aligned_y,
            "flux_snr": self.flux_snr,
            "quality_passed": self.quality_passed,
            "candidate_snr": self.candidate_snr,
        }


@dataclass(frozen=True, slots=True)
class SourceTrack:
    """一条跨帧轨迹及其静态/持续/运动判定。"""

    track_id: int
    classification: str
    points: tuple[TrackPoint, ...]
    displacement_px: float
    speed_px_per_frame: float
    fit_rms_px: float | None
    evidence_level: str = "quality"

    @property
    def presence(self) -> int:
        return len(self.points)

    @property
    def quality_presence(self) -> int:
        """轨迹中通过单帧质量规则的观测次数。"""

        return sum(point.quality_passed for point in self.points)

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "classification": self.classification,
            "presence": self.presence,
            "displacement_px": self.displacement_px,
            "speed_px_per_frame": self.speed_px_per_frame,
            "fit_rms_px": self.fit_rms_px,
            "quality_presence": self.quality_presence,
            "evidence_level": self.evidence_level,
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class MotionFeaturePoint:
    """一帧中被时序/形状规则检出的线状运动候选。"""

    frame_index: int
    x: float
    y: float
    aligned_x: float
    aligned_y: float
    residual_snr: float
    area_pixels: int
    length_px: float
    width_px: float
    angle_deg: float
    bbox: tuple[int, int, int, int]
    touches_edge: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "x": self.x,
            "y": self.y,
            "aligned_x": self.aligned_x,
            "aligned_y": self.aligned_y,
            "residual_snr": self.residual_snr,
            "area_pixels": self.area_pixels,
            "length_px": self.length_px,
            "width_px": self.width_px,
            "angle_deg": self.angle_deg,
            "bbox": list(self.bbox),
            "touches_edge": self.touches_edge,
        }


@dataclass(frozen=True, slots=True)
class MotionFeatureTrack:
    """由线状候选跨帧关联得到的运动目标候选轨迹。"""

    track_id: int
    classification: str
    points: tuple[MotionFeaturePoint, ...]
    displacement_px: float
    speed_px_per_frame: float
    fit_rms_px: float | None

    @property
    def presence(self) -> int:
        return len(self.points)

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "classification": self.classification,
            "presence": self.presence,
            "displacement_px": self.displacement_px,
            "speed_px_per_frame": self.speed_px_per_frame,
            "fit_rms_px": self.fit_rms_px,
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class MotionFrameAudit:
    """逐帧线状筛选的中间计数，便于解释候选为何被保留或剔除。"""

    frame_index: int
    threshold_adu: float
    residual_noise_adu: float
    valid_pixel_count: int
    support_pixel_count: int
    component_count: int
    area_pass_count: int
    geometry_pass_count: int
    edge_rejected_count: int
    feature_count: int
    max_feature_residual_snr: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "threshold_adu": self.threshold_adu,
            "residual_noise_adu": self.residual_noise_adu,
            "valid_pixel_count": self.valid_pixel_count,
            "support_pixel_count": self.support_pixel_count,
            "component_count": self.component_count,
            "area_pass_count": self.area_pass_count,
            "geometry_pass_count": self.geometry_pass_count,
            "edge_rejected_count": self.edge_rejected_count,
            "feature_count": self.feature_count,
            "max_feature_residual_snr": self.max_feature_residual_snr,
        }


@dataclass(frozen=True, slots=True)
class FrameSequenceSummary:
    frame_index: int
    path: str
    candidate_count: int
    returned_count: int
    quality_count: int
    timestamp: str | None = None
    exposure_ms: float | None = None
    auxiliary: tuple[tuple[str, float], ...] = ()
    width_px: int | None = None
    height_px: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "quality_count": self.quality_count,
            "timestamp": self.timestamp,
            "exposure_ms": self.exposure_ms,
            "auxiliary": dict(self.auxiliary),
            "width_px": self.width_px,
            "height_px": self.height_px,
        }

    @property
    def auxiliary_dict(self) -> Mapping[str, float]:
        return dict(self.auxiliary)


@dataclass(frozen=True, slots=True)
class FixedSentinelAudit:
    """跨帧固定像素值审计；只提供诊断证据，不修改检测输入。"""

    sentinel_value: int
    frame_count: int
    image_shape: tuple[int, int] | None
    same_shape: bool
    total_occurrences: int
    per_frame_occurrences: tuple[int, ...]
    fixed_coordinate_count: int
    fixed_coordinates: tuple[tuple[int, int], ...]
    coordinates_truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "sentinel_value": self.sentinel_value,
            "frame_count": self.frame_count,
            "image_shape": list(self.image_shape) if self.image_shape is not None else None,
            "same_shape": self.same_shape,
            "total_occurrences": self.total_occurrences,
            "per_frame_occurrences": list(self.per_frame_occurrences),
            "fixed_coordinate_count": self.fixed_coordinate_count,
            "fixed_coordinates": [list(coordinate) for coordinate in self.fixed_coordinates],
            "coordinates_truncated": self.coordinates_truncated,
        }


def audit_fixed_sentinel(
    images: Sequence[np.ndarray],
    *,
    sentinel_value: int = -1,
    max_coordinates: int = 10_000,
) -> FixedSentinelAudit:
    """统计序列中恒定出现的精确像素值。

    比较前会排除比赛格式第一行前 104 个辅助字段位置。该函数刻意不把
    固定值直接加入质量掩膜：固定值可能是坏像素、填充值或编码约定，
    只有结合 FITS 头、原始范围和局部 PSF 证据后才能决定其物理含义。
    ``fixed_coordinates`` 使用 ``(x, y)`` 坐标，最多返回
    ``max_coordinates`` 个；``fixed_coordinate_count`` 始终是完整计数。
    """

    if max_coordinates < 0:
        raise ValueError("max_coordinates must be non-negative")
    frame_count = len(images)
    if frame_count == 0:
        return FixedSentinelAudit(
            sentinel_value=int(sentinel_value),
            frame_count=0,
            image_shape=None,
            same_shape=True,
            total_occurrences=0,
            per_frame_occurrences=(),
            fixed_coordinate_count=0,
            fixed_coordinates=(),
        )

    first_shape: tuple[int, int] | None = None
    same_shape = True
    fixed_mask: np.ndarray | None = None
    per_frame_occurrences: list[int] = []
    total_occurrences = 0
    for image in images:
        values = np.asarray(image)
        if values.ndim != 2:
            raise ValueError(f"expected a 2-D image, got shape {values.shape}")
        shape = (int(values.shape[0]), int(values.shape[1]))
        if first_shape is None:
            first_shape = shape
        elif shape != first_shape:
            same_shape = False

        mask = np.equal(values, sentinel_value)
        mask &= ~auxiliary_mask(shape)
        count = int(np.count_nonzero(mask))
        per_frame_occurrences.append(count)
        total_occurrences += count

        if same_shape:
            if fixed_mask is None:
                fixed_mask = mask.copy()
            else:
                fixed_mask &= mask
        else:
            # 不对不同尺寸的帧猜测坐标对应关系；仍保留逐帧/总量统计。
            fixed_mask = None

    if same_shape and fixed_mask is not None:
        coordinates_yx = np.argwhere(fixed_mask)
        fixed_coordinate_count = int(coordinates_yx.shape[0])
        kept_coordinates = coordinates_yx[:max_coordinates]
        fixed_coordinates = tuple(
            (int(x), int(y)) for y, x in kept_coordinates
        )
        coordinates_truncated = fixed_coordinate_count > len(fixed_coordinates)
    else:
        fixed_coordinate_count = 0
        fixed_coordinates = ()
        coordinates_truncated = False

    return FixedSentinelAudit(
        sentinel_value=int(sentinel_value),
        frame_count=frame_count,
        image_shape=first_shape,
        same_shape=same_shape,
        total_occurrences=total_occurrences,
        per_frame_occurrences=tuple(per_frame_occurrences),
        fixed_coordinate_count=fixed_coordinate_count,
        fixed_coordinates=fixed_coordinates,
        coordinates_truncated=coordinates_truncated,
    )


@dataclass(frozen=True, slots=True)
class FixedSentinelSourceImpact:
    """一个已测量源与固定异常坐标的孔径重叠证据。"""

    frame_index: int
    detection_id: int
    x: float
    y: float
    peak_x: float
    peak_y: float
    nearest_fixed_distance_px: float
    fixed_coordinate_count: int
    peak: float
    flux_snr: float | None
    quality_passed: bool
    flags: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "peak_x": self.peak_x,
            "peak_y": self.peak_y,
            "nearest_fixed_distance_px": self.nearest_fixed_distance_px,
            "fixed_coordinate_count": self.fixed_coordinate_count,
            "peak": self.peak,
            "flux_snr": self.flux_snr,
            "quality_passed": self.quality_passed,
            "flags": list(self.flags),
        }


@dataclass(frozen=True, slots=True)
class FixedSentinelImpactAudit:
    """固定异常坐标对候选峰和已测量源的影响汇总。"""

    frame_count: int
    affected_frame_count: int
    candidate_peak_data_available: bool
    candidate_peak_frame_count: int
    candidate_peak_count: int
    candidate_peak_affected_count: int
    affected_returned_source_count: int
    affected_quality_source_count: int
    records: tuple[FixedSentinelSourceImpact, ...]
    record_count: int
    records_truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "affected_frame_count": self.affected_frame_count,
            "candidate_peak_data_available": self.candidate_peak_data_available,
            "candidate_peak_frame_count": self.candidate_peak_frame_count,
            "candidate_peak_count": self.candidate_peak_count,
            "candidate_peak_affected_count": self.candidate_peak_affected_count,
            "affected_returned_source_count": self.affected_returned_source_count,
            "affected_quality_source_count": self.affected_quality_source_count,
            "record_count": self.record_count,
            "records_truncated": self.records_truncated,
            "records": [record.as_dict() for record in self.records],
        }


def audit_fixed_sentinel_impacts(
    frame_analyses: Sequence[FrameAnalysis],
    fixed_sentinel_audit: FixedSentinelAudit,
    *,
    max_records: int = 5_000,
) -> FixedSentinelImpactAudit:
    """把固定异常坐标关联到候选峰和已测量源。

    “受影响”定义为：固定坐标落入该帧检测参数中的圆孔径半径内。
    候选峰数组通常包含完整宽筛峰，而 ``Detection`` 记录只包含已经
    完成源级测量的返回候选；两种计数分开保存。这个函数不改变任何
    检测掩膜、像素或质量结果。
    """

    if max_records < 0:
        raise ValueError("max_records must be non-negative")
    frame_count = len(frame_analyses)
    fixed_coordinates = fixed_sentinel_audit.fixed_coordinates
    fixed_xy = (
        np.asarray(fixed_coordinates, dtype=np.float64)
        if fixed_coordinates
        else np.empty((0, 2), dtype=np.float64)
    )
    fixed_tree = cKDTree(fixed_xy) if fixed_xy.size else None
    candidate_peak_data_available = False
    candidate_peak_frame_count = 0
    candidate_peak_count = 0
    candidate_peak_affected_count = 0
    affected_frame_count = 0
    affected_returned_source_count = 0
    affected_quality_source_count = 0
    records: list[FixedSentinelSourceImpact] = []
    record_count = 0

    for frame_index, analysis in enumerate(frame_analyses):
        parameters = analysis.detection.parameters
        try:
            aperture_radius = float(parameters.get("aperture_radius", 4.0))
        except (TypeError, ValueError):
            aperture_radius = 4.0
        if not np.isfinite(aperture_radius) or aperture_radius <= 0:
            aperture_radius = 4.0

        frame_affected = False
        raw_candidates = np.asarray(analysis.detection.candidate_peaks)
        if raw_candidates.ndim == 2 and raw_candidates.shape[1] >= 2:
            candidate_peak_data_available = True
            candidate_peak_frame_count += 1
            candidate_peak_count += int(raw_candidates.shape[0])
            candidate_xy = np.asarray(raw_candidates[:, :2], dtype=np.float64)
            finite_candidates = np.isfinite(candidate_xy).all(axis=1)
            if fixed_tree is not None and finite_candidates.any():
                distances, indices = fixed_tree.query(
                    candidate_xy[finite_candidates],
                    distance_upper_bound=aperture_radius,
                )
                affected = np.isfinite(distances) & (indices < len(fixed_coordinates))
                candidate_peak_affected_count += int(np.count_nonzero(affected))
                frame_affected |= bool(np.any(affected))

        if fixed_tree is not None:
            for source in analysis.detection.sources:
                peak_x = (
                    float(source.peak_x)
                    if source.peak_x is not None and np.isfinite(source.peak_x)
                    else float(source.x)
                )
                peak_y = (
                    float(source.peak_y)
                    if source.peak_y is not None and np.isfinite(source.peak_y)
                    else float(source.y)
                )
                nearby = fixed_tree.query_ball_point((peak_x, peak_y), r=aperture_radius)
                if not nearby:
                    continue
                nearest_distance = float(fixed_tree.query((peak_x, peak_y))[0])
                record_count += 1
                affected_returned_source_count += 1
                affected_quality_source_count += int(bool(source.quality_passed))
                frame_affected = True
                if len(records) < max_records:
                    records.append(
                        FixedSentinelSourceImpact(
                            frame_index=frame_index,
                            detection_id=int(source.detection_id),
                            x=float(source.x),
                            y=float(source.y),
                            peak_x=peak_x,
                            peak_y=peak_y,
                            nearest_fixed_distance_px=nearest_distance,
                            fixed_coordinate_count=len(nearby),
                            peak=float(source.peak),
                            flux_snr=(
                                float(source.flux_snr)
                                if source.flux_snr is not None
                                else None
                            ),
                            quality_passed=bool(source.quality_passed),
                            flags=tuple(source.flags),
                        )
                    )
        affected_frame_count += int(frame_affected)

    return FixedSentinelImpactAudit(
        frame_count=frame_count,
        affected_frame_count=affected_frame_count,
        candidate_peak_data_available=candidate_peak_data_available,
        candidate_peak_frame_count=candidate_peak_frame_count,
        candidate_peak_count=candidate_peak_count,
        candidate_peak_affected_count=candidate_peak_affected_count,
        affected_returned_source_count=affected_returned_source_count,
        affected_quality_source_count=affected_quality_source_count,
        records=tuple(records),
        record_count=record_count,
        records_truncated=record_count > len(records),
    )


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """15 帧检测和关联的汇总结果。

    ``fixed_sentinel_audit`` 与 ``fixed_sentinel_impact_audit`` 是数据质量
    诊断字段，不会改变候选、质量源或轨迹的物理分类。
    """

    frames: tuple[FrameSequenceSummary, ...]
    cumulative_shifts: tuple[tuple[float, float], ...]
    tracks: tuple[SourceTrack, ...]
    link_radius_px: float
    min_presence: int
    motion_min_displacement_px: float
    max_motion_fit_rms_px: float
    motion_features: tuple[MotionFeatureTrack, ...] = ()
    motion_residual_threshold_adu: float = 100.0
    motion_reference_mode: str = "per_frame_background"
    motion_frame_audits: tuple[MotionFrameAudit, ...] = ()
    source_working_limit: int | None = None
    calculation_dtype: str = "float64"
    background_sample_limit: int | None = None
    fast_sequence: bool = False
    background_model_mode: str = "per_frame_local"
    persistent_min_presence: int = 0
    temporal_reference_candidate_count: int = 0
    temporal_coadd_candidate_count: int = 0
    temporal_proposal_mode: str = "median"
    temporal_candidate_min_snr: float = 7.5
    temporal_reference_min_snr: float = 15.0
    temporal_multiscale: bool = False
    temporal_min_psf_correlation: float = 0.8
    candidate_consensus_audit: tuple[tuple[str, int], ...] = ()
    stack_faint_candidate_count: int = 0
    stack_reference_mode: str = "median"
    stack_threshold_sigma: float = 4.0
    stack_min_flux_snr: float = 5.0
    stack_frame_min_flux_snr: float = 3.0
    stack_min_presence: int = 0
    fixed_sentinel_audit: FixedSentinelAudit | None = None
    fixed_sentinel_impact_audit: FixedSentinelImpactAudit | None = None

    @property
    def stable_source_count(self) -> int:
        return sum(track.classification == "static" for track in self.tracks)

    @property
    def moving_track_count(self) -> int:
        return sum(track.classification == "moving" for track in self.tracks)

    @property
    def transient_track_count(self) -> int:
        return sum(track.classification == "transient" for track in self.tracks)

    @property
    def persistent_source_count(self) -> int:
        """达到较低持续性门槛、但未达到严格静态门槛的点源轨迹数。

        叠加参考图恢复的 ``stack_faint`` 轨迹单独计入
        :meth:`stack_faint_count`，不混入本计数。
        """

        return sum(
            track.classification == "persistent" and track.evidence_level != "stack_faint"
            for track in self.tracks
        )

    @property
    def candidate_consensus_count(self) -> int:
        """由补充候选通道跨帧共识得到的低置信稳定候选总数。"""

        return sum(
            track.evidence_level in {"candidate_consensus", "temporal_reference"}
            for track in self.tracks
        )

    @property
    def temporal_reference_count(self) -> int:
        """由注册时间中值补提案形成的低置信候选数。"""

        return sum(track.evidence_level == "temporal_reference" for track in self.tracks)

    @property
    def stack_faint_count(self) -> int:
        """由叠加参考图恢复、经逐帧强制测光确认的低置信暗星轨迹数。

        这是单帧质量门下的低置信补充层，不是官方逐星真值。
        """

        return sum(track.evidence_level == "stack_faint" for track in self.tracks)

    @property
    def stable_field_candidate_count(self) -> int:
        """严格静态源与持续源候选的合计；不是物理恒星真值。"""

        return self.stable_source_count + self.persistent_source_count

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": len(self.frames),
            "stable_source_count": self.stable_source_count,
            "persistent_source_count": self.persistent_source_count,
            "stable_field_candidate_count": self.stable_field_candidate_count,
            "candidate_consensus_count": self.candidate_consensus_count,
            "temporal_reference_count": self.temporal_reference_count,
            "stack_faint_count": self.stack_faint_count,
            "moving_track_count": self.moving_track_count,
            "transient_track_count": self.transient_track_count,
            "link_radius_px": self.link_radius_px,
            "min_presence": self.min_presence,
            "motion_min_displacement_px": self.motion_min_displacement_px,
            "max_motion_fit_rms_px": self.max_motion_fit_rms_px,
            "motion_residual_threshold_adu": self.motion_residual_threshold_adu,
            "motion_reference_mode": self.motion_reference_mode,
            "motion_frame_audits": [audit.as_dict() for audit in self.motion_frame_audits],
            "source_working_limit": self.source_working_limit,
            "calculation_dtype": self.calculation_dtype,
            "background_sample_limit": self.background_sample_limit,
            "fast_sequence": self.fast_sequence,
            "background_model_mode": self.background_model_mode,
            "persistent_min_presence": self.persistent_min_presence,
            "temporal_reference_candidate_count": self.temporal_reference_candidate_count,
            "temporal_coadd_candidate_count": self.temporal_coadd_candidate_count,
            "temporal_proposal_mode": self.temporal_proposal_mode,
            "temporal_candidate_min_snr": self.temporal_candidate_min_snr,
            "temporal_reference_min_snr": self.temporal_reference_min_snr,
            "temporal_multiscale": self.temporal_multiscale,
            "temporal_min_psf_correlation": self.temporal_min_psf_correlation,
            "candidate_consensus_audit": dict(self.candidate_consensus_audit),
            "stack_faint_candidate_count": self.stack_faint_candidate_count,
            "stack_reference_mode": self.stack_reference_mode,
            "stack_threshold_sigma": self.stack_threshold_sigma,
            "stack_min_flux_snr": self.stack_min_flux_snr,
            "stack_frame_min_flux_snr": self.stack_frame_min_flux_snr,
            "stack_min_presence": self.stack_min_presence,
            "fixed_sentinel_audit": (
                self.fixed_sentinel_audit.as_dict()
                if self.fixed_sentinel_audit is not None
                else None
            ),
            "fixed_sentinel_impact_audit": (
                self.fixed_sentinel_impact_audit.as_dict()
                if self.fixed_sentinel_impact_audit is not None
                else None
            ),
            "frames": [frame.as_dict() for frame in self.frames],
            "cumulative_shifts": [list(shift) for shift in self.cumulative_shifts],
            "tracks": [track.as_dict() for track in self.tracks],
            "motion_features": [track.as_dict() for track in self.motion_features],
        }


def _signal_snr(source: Detection) -> float:
    value = source.flux_snr if source.flux_snr is not None else source.snr
    return float(value) if np.isfinite(value) else 0.0


def _estimate_translation(
    previous: Sequence[Detection],
    current: Sequence[Detection],
    *,
    radius_px: float,
    reference_limit: int = 1500,
) -> tuple[float, float]:
    """用高 SNR 源的稳健中位数估计 current - previous 的平移。"""

    if not previous or not current:
        return 0.0, 0.0
    previous_top = sorted(previous, key=_signal_snr, reverse=True)[:reference_limit]
    current_top = sorted(current, key=_signal_snr, reverse=True)[:reference_limit]
    previous_points = np.array([(source.x, source.y) for source in previous_top], dtype=np.float64)
    current_points = np.array([(source.x, source.y) for source in current_top], dtype=np.float64)
    current_tree = cKDTree(current_points)
    previous_tree = cKDTree(previous_points)
    previous_distances, previous_indices = current_tree.query(previous_points, distance_upper_bound=radius_px)
    _current_distances, current_indices = previous_tree.query(current_points, distance_upper_bound=radius_px)
    valid = np.isfinite(previous_distances) & (previous_indices < len(current_top))
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return 0.0, 0.0
    reciprocal = current_indices[previous_indices[valid_indices]] == valid_indices
    valid_indices = valid_indices[reciprocal]
    if valid_indices.size == 0:
        return 0.0, 0.0
    deltas = current_points[previous_indices[valid_indices]] - previous_points[valid_indices]
    # 只保留互相最近的一对一匹配，避免一个当前源被多个前帧源重复使用。
    center = np.median(deltas, axis=0)
    residual = np.linalg.norm(deltas - center, axis=1)
    mad = np.median(np.abs(residual - np.median(residual)))
    cutoff = max(1.0, 3.0 * 1.4826 * mad)
    inliers = residual <= cutoff
    if np.count_nonzero(inliers) < 3:
        return float(center[0]), float(center[1])
    return tuple(float(value) for value in np.median(deltas[inliers], axis=0))


def _fit_track(points: Sequence[TrackPoint]) -> tuple[float, float, float | None]:
    if len(points) < 2:
        return 0.0, 0.0, None
    time = np.array([point.frame_index for point in points], dtype=np.float64)
    x = np.array([point.aligned_x for point in points], dtype=np.float64)
    y = np.array([point.aligned_y for point in points], dtype=np.float64)
    x_slope, x_intercept = np.polyfit(time, x, 1)
    y_slope, y_intercept = np.polyfit(time, y, 1)
    fitted_x = x_slope * time + x_intercept
    fitted_y = y_slope * time + y_intercept
    rms = float(np.sqrt(np.mean((x - fitted_x) ** 2 + (y - fitted_y) ** 2)))
    # 使用拟合速度在整个观测跨度上的位移，而不是首末两个质心的距离。
    # 后者会把首帧/末帧一次性的质心抖动误判为持续运动。
    duration = float(time[-1] - time[0])
    displacement = float(np.hypot(x_slope * duration, y_slope * duration))
    speed = float(np.hypot(x_slope, y_slope))
    return displacement, speed, rms


def _fit_motion_feature_track(points: Sequence[MotionFeaturePoint]) -> tuple[float, float, float | None]:
    """在线状候选的配准坐标中拟合常速度轨迹。"""

    if len(points) < 2:
        return 0.0, 0.0, None
    time = np.array([point.frame_index for point in points], dtype=np.float64)
    x = np.array([point.aligned_x for point in points], dtype=np.float64)
    y = np.array([point.aligned_y for point in points], dtype=np.float64)
    x_slope, x_intercept = np.polyfit(time, x, 1)
    y_slope, y_intercept = np.polyfit(time, y, 1)
    fitted_x = x_slope * time + x_intercept
    fitted_y = y_slope * time + y_intercept
    rms = float(np.sqrt(np.mean((x - fitted_x) ** 2 + (y - fitted_y) ** 2)))
    # 与点源轨迹保持同一语义：用拟合速度乘以观测跨度，避免首末帧
    # 的单次质心/连通域形状抖动改变运动判定。
    duration = float(time[-1] - time[0])
    displacement = float(np.hypot(x_slope * duration, y_slope * duration))
    speed = float(np.hypot(x_slope, y_slope))
    return displacement, speed, rms


def _registered_median_reference(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    tile_rows: int = 256,
    progress: Callable[[float, str], None] | None = None,
) -> np.ndarray | None:
    """以低内存方式生成注册后的时间中值参考图。

    每帧先按 ``aligned = raw - cumulative_shift`` 的约定重采样到第 1 帧
    坐标，再对时间轴取中值。使用分块而不是一次性堆叠完整的 15 帧，避免
    4096² 图像在 GUI 后台额外占用约 1 GB 浮点内存。
    """

    if len(frame_analyses) < 2:
        return None
    if len(frame_analyses) != len(cumulative_shifts):
        raise ValueError("cumulative_shifts length must match frame_analyses")
    shape = tuple(int(value) for value in frame_analyses[0].frame.data.shape)
    if len(shape) != 2 or any(tuple(int(value) for value in analysis.frame.data.shape) != shape for analysis in frame_analyses):
        return None
    height, width = shape
    tile_height = max(16, int(tile_rows))
    tile_count = max(1, int(np.ceil(height / tile_height)))
    reference = np.full(shape, np.nan, dtype=np.float32)
    source_x = np.arange(width, dtype=np.float32)[None, :]
    # 当前 15 帧的累计配准量小于 1 px。此时直接按原坐标分块取中值，
    # 把亚像素差异交给后续高阈值/细长形态筛选，能显著降低 GUI 的峰值
    # 内存；只有较大平移时才使用全图重采样。
    direct_reference = max(float(np.max(np.abs(cumulative_shifts))), 0.0) <= 0.75
    for tile_index, y_start in enumerate(range(0, height, tile_height), start=1):
        y_stop = min(height, y_start + tile_height)
        output_y = np.arange(y_start, y_stop, dtype=np.float32)[:, None]
        stack = np.full((len(frame_analyses), y_stop - y_start, width), np.nan, dtype=np.float32)
        for frame_index, analysis in enumerate(frame_analyses):
            parameters = analysis.detection.parameters
            if direct_reference:
                # 只把当前行块转成 float32，避免把整帧再复制一份。
                working = np.asarray(analysis.frame.data[y_start:y_stop], dtype=np.float32).copy()
                if y_start == 0:
                    working[0, : min(working.shape[1], 104)] = np.nan
            else:
                # 较大平移需要在统一坐标中插值；复制只活在本次 frame 循环。
                working = np.asarray(analysis.frame.data, dtype=np.float32).copy()
                working[0, : min(working.shape[1], 104)] = np.nan
            if int(parameters.get("mask_zero_pixels", 0)):
                working[working == 0.0] = np.nan
            saturation_level = float(parameters.get("saturation_level", -1.0))
            if saturation_level > 0:
                working[working >= saturation_level] = np.nan
            shift_x, shift_y = cumulative_shifts[frame_index]
            if direct_reference:
                stack[frame_index] = working
            else:
                source_y = output_y + np.float32(shift_y)
                source_x_frame = source_x + np.float32(shift_x)
                coordinates = (
                    np.broadcast_to(source_y, (y_stop - y_start, width)),
                    np.broadcast_to(source_x_frame, (y_stop - y_start, width)),
                )
                stack[frame_index] = ndimage.map_coordinates(
                    working,
                    coordinates,
                    order=1,
                    mode="constant",
                    cval=np.nan,
                    prefilter=False,
                ).astype(np.float32, copy=False)
        tile_reference = np.full((y_stop - y_start, width), np.nan, dtype=np.float32)
        valid_positions = np.any(np.isfinite(stack), axis=0)
        if np.any(valid_positions):
            with np.errstate(invalid="ignore"):
                tile_reference[valid_positions] = np.nanmedian(stack[:, valid_positions], axis=0).astype(np.float32, copy=False)
        reference[y_start:y_stop] = tile_reference
        if progress is not None:
            progress(
                5.0 + 15.0 * tile_index / tile_count,
                f"构建时间中值参考 {tile_index}/{tile_count}",
            )
    return reference


def _registered_coadd_reference(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    tile_rows: int = 256,
    clip_sigma: float = 3.0,
    progress: Callable[[float, str], None] | None = None,
) -> np.ndarray | None:
    """生成用于宽筛提案的注册稳健均值图。

    与时间中值参考不同，稳健均值对稳定弱源的信噪比更友好；对每个像素
    先围绕时间中位数做一次 MAD 裁剪，再求均值，尽量压制只出现在少数
    帧中的亮线、宇宙线或瞬态结构。这个结果只用于提出坐标，不能绕过
    ``_candidate_consensus_tracks`` 的逐帧原图支持、形状和持续性细筛。

    分块读取和累加保持与中值参考相同的内存上限。当前 15 帧累计平移小
    于 1 px 时直接在原坐标上累加；平移较大时才使用一次线性重采样。
    ``clip_sigma`` 只控制多帧提案的瞬态抑制，不改变单帧检测阈值。
    """

    if len(frame_analyses) < 2:
        return None
    if len(frame_analyses) != len(cumulative_shifts):
        raise ValueError("cumulative_shifts length must match frame_analyses")
    if clip_sigma <= 0:
        raise ValueError("clip_sigma must be positive")
    shape = tuple(int(value) for value in frame_analyses[0].frame.data.shape)
    if len(shape) != 2 or any(tuple(int(value) for value in analysis.frame.data.shape) != shape for analysis in frame_analyses):
        return None
    height, width = shape
    tile_height = max(16, int(tile_rows))
    tile_count = max(1, int(np.ceil(height / tile_height)))
    coadd = np.full(shape, np.nan, dtype=np.float32)
    source_x = np.arange(width, dtype=np.float32)[None, :]
    direct_reference = max(float(np.max(np.abs(cumulative_shifts))), 0.0) <= 0.75
    for tile_index, y_start in enumerate(range(0, height, tile_height), start=1):
        y_stop = min(height, y_start + tile_height)
        output_y = np.arange(y_start, y_stop, dtype=np.float32)[:, None]
        stack = np.full(
            (len(frame_analyses), y_stop - y_start, width),
            np.nan,
            dtype=np.float32,
        )
        for frame_index, analysis in enumerate(frame_analyses):
            parameters = analysis.detection.parameters
            if direct_reference:
                working = np.asarray(
                    analysis.frame.data[y_start:y_stop],
                    dtype=np.float32,
                ).copy()
                if y_start == 0:
                    working[0, : min(working.shape[1], 104)] = np.nan
            else:
                working = np.asarray(analysis.frame.data, dtype=np.float32).copy()
                working[0, : min(working.shape[1], 104)] = np.nan
            if int(parameters.get("mask_zero_pixels", 0)):
                working[working == 0.0] = np.nan
            saturation_level = float(parameters.get("saturation_level", -1.0))
            if saturation_level > 0:
                working[working >= saturation_level] = np.nan
            shift_x, shift_y = cumulative_shifts[frame_index]
            if direct_reference:
                stack[frame_index] = working
            else:
                source_y = output_y + np.float32(shift_y)
                source_x_frame = source_x + np.float32(shift_x)
                coordinates = (
                    np.broadcast_to(source_y, (y_stop - y_start, width)),
                    np.broadcast_to(source_x_frame, (y_stop - y_start, width)),
                )
                stack[frame_index] = ndimage.map_coordinates(
                    working,
                    coordinates,
                    order=1,
                    mode="constant",
                    cval=np.nan,
                    prefilter=False,
                ).astype(np.float32, copy=False)

        valid_positions = np.any(np.isfinite(stack), axis=0)
        median = np.full((y_stop - y_start, width), np.nan, dtype=np.float32)
        if np.any(valid_positions):
            with np.errstate(invalid="ignore"):
                median[valid_positions] = np.nanmedian(
                    stack[:, valid_positions],
                    axis=0,
                ).astype(np.float32, copy=False)
        deviation = np.abs(stack - median[None, :, :])
        mad = np.full((y_stop - y_start, width), np.nan, dtype=np.float32)
        if np.any(valid_positions):
            with np.errstate(invalid="ignore"):
                mad[valid_positions] = np.nanmedian(
                    deviation[:, valid_positions],
                    axis=0,
                ).astype(np.float32, copy=False)
        scale = np.maximum(
            1.4826 * mad,
            np.finfo(np.float32).eps,
        )
        keep = np.isfinite(stack) & (
            np.abs(stack - median[None, :, :]) <= float(clip_sigma) * scale[None, :, :]
        )
        finite_count = np.count_nonzero(keep, axis=0)
        summed = np.sum(np.where(keep, stack, 0.0), axis=0, dtype=np.float32)
        tile_coadd = np.full((y_stop - y_start, width), np.nan, dtype=np.float32)
        valid_positions = finite_count > 0
        tile_coadd[valid_positions] = (
            summed[valid_positions] / finite_count[valid_positions]
        ).astype(np.float32, copy=False)
        coadd[y_start:y_stop] = tile_coadd
        if progress is not None:
            progress(
                5.0 + 15.0 * tile_index / tile_count,
                f"构建稳健时间叠加 {tile_index}/{tile_count}",
            )
    return coadd


def _temporal_reference_candidate_frames(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    reference: np.ndarray | None,
    *,
    psf_fwhm: float,
    min_candidate_snr: float,
    temporal_multiscale: bool = False,
) -> tuple[tuple[np.ndarray, ...], int]:
    """从注册时间中值中提出需要回到逐帧核验的补充候选。

    单帧 Gaussian 峰提案有一个结构性盲点：如果两个紧邻源的响应重叠，
    或某一帧的峰恰好被坏像素/线状结构污染，源可能从首帧候选集合中消失。
    15 帧注册后的时间中值可以作为“宽筛补充”，但不能直接作为星点真值：
    固定图样和静态伪影也会被中值保留。因此这里只生成临时的逐帧强制测量
    坐标，并要求后续候选共识再次通过逐帧 PSF 响应、3×3 支持、孔径有效性
    和位置拟合；结果仍进入低置信 ``candidate_consensus`` 层。

    返回值是每帧的 ``(x, y, frame_filter_snr, channel, reference_snr)`` 数组
    以及时间中值宽筛提案数。``channel=1`` 和最后一列的参考 SNR 让后续
    共识可以使用较低的逐帧强制响应门槛，同时仍要求时间中值种子自身达到
    正式候选的显著性；不能把这个较低门槛误用于普通单帧候选。
    ``temporal_multiscale=True`` 时，参考图和逐帧强制响应使用相对
    ``psf_fwhm`` 的窄/基准/宽三尺度组，取各尺度的最大标准化响应；这样
    可以做窄 PSF 漏检抽测，但也会增加纹理候选，默认关闭。数组不写入
    序列缓存，避免把一个依赖当前参考构造的中间层误当成正式源表。
    """

    frame_count = len(frame_analyses)
    empty = tuple(np.empty((0, 5), dtype=np.float32) for _ in range(frame_count))
    if reference is None or frame_count < 2 or len(cumulative_shifts) != frame_count:
        return empty, 0
    if psf_fwhm <= 0 or min_candidate_snr <= 0:
        raise ValueError("temporal reference candidate thresholds must be positive")

    values = np.asarray(reference, dtype=np.float32)
    if values.ndim != 2:
        return empty, 0
    valid = np.isfinite(values)
    if not np.any(valid):
        return empty, 0
    # 注册中值已经把无效孔径变成 NaN；第一行辅助区再显式排除一次，防止
    # 在参考帧平移较大时把辅助字节重采样成一个亮的局部峰。
    valid = valid.copy()
    valid[0, : min(values.shape[1], 104)] = False
    background, noise = sigma_clipped_stats(
        values,
        mask=~valid,
        sample_limit=1_000_000,
    )
    noise = max(float(noise), np.finfo(np.float32).eps)
    residual = np.where(valid, values - float(background), 0.0).astype(np.float32, copy=False)
    temporal_fwhms = (
        _temporal_psf_fwhm_bank(float(psf_fwhm))
        if temporal_multiscale
        else (float(psf_fwhm),)
    )
    temporal_sigmas = tuple(max(fwhm / 2.35482, 0.5) for fwhm in temporal_fwhms)
    response_snr_bank: list[np.ndarray] = []
    response_local_max_bank: list[np.ndarray] = []
    for sigma in temporal_sigmas:
        response = ndimage.gaussian_filter(residual, sigma=sigma, mode="nearest")
        response_values = response[valid]
        response_background, response_noise = sigma_clipped_stats(
            response_values,
            sample_limit=1_000_000,
        )
        response_noise = max(float(response_noise), np.finfo(np.float32).eps)
        response_snr = (response - float(response_background)) / response_noise
        response_snr_bank.append(response_snr.astype(np.float32, copy=False))
        peak_radius = max(1, int(np.ceil(2.0 * sigma)))
        response_local_max_bank.append(
            response_snr == ndimage.maximum_filter(
                response_snr,
                size=2 * peak_radius + 1,
                mode="nearest",
            )
        )

    # 时间中值只负责补提案，所以阈值可以低于逐帧候选门槛；真正进入
    # 共识的逐帧行仍按 min_candidate_snr 过滤。这里的 0.5 倍是宽筛层，
    # 不是把 15σ 质量口径偷偷改成 7.5σ。
    seed_threshold = max(5.0, 0.5 * float(min_candidate_snr))
    raw_local_max = values == ndimage.maximum_filter(values, size=3, mode="nearest")
    response_snr_map = np.maximum.reduce(response_snr_bank)
    response_peak_mask = np.zeros(values.shape, dtype=bool)
    for response_snr, response_local_max in zip(
        response_snr_bank,
        response_local_max_bank,
        strict=True,
    ):
        response_peak_mask |= response_local_max & (response_snr >= seed_threshold)
    candidate_mask = (
        valid
        & response_peak_mask
        & raw_local_max
        & ((values - float(background)) >= 2.0 * noise)
    )
    candidate_y, candidate_x = np.nonzero(candidate_mask)
    if candidate_x.size == 0:
        return empty, 0

    first_parameters = frame_analyses[0].detection.parameters
    required_support = max(3, int(first_parameters.get("min_psf_support_pixels", 3)))
    # 只在参考图上做一次宽松形状审计，主要挡住时间中值中的长条和极窄
    # 单像素尖峰；最终的单帧质量规则仍然不被这个宽筛替代。
    shape_radius = max(4, int(np.ceil(3.5 * max(temporal_sigmas))))
    reference_rows: list[tuple[float, float, float]] = []
    for x_raw, y_raw in zip(candidate_x, candidate_y, strict=True):
        x = int(x_raw)
        y = int(y_raw)
        if (
            x < shape_radius
            or y < shape_radius
            or x >= values.shape[1] - shape_radius
            or y >= values.shape[0] - shape_radius
        ):
            continue
        patch = values[
            y - shape_radius : y + shape_radius + 1,
            x - shape_radius : x + shape_radius + 1,
        ].astype(np.float64, copy=False) - float(background)
        peak_excess = max(float(values[y, x] - float(background)), 0.0)
        core_slice = patch[shape_radius - 1 : shape_radius + 2, shape_radius - 1 : shape_radius + 2]
        support_threshold = max(2.0 * noise, 0.1 * peak_excess)
        if int(np.count_nonzero(core_slice >= support_threshold)) < required_support:
            continue
        weights = np.maximum(patch - 0.5 * noise, 0.0)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= np.finfo(np.float64).eps:
            continue
        yy, xx = np.indices(patch.shape, dtype=np.float64)
        center_x = float((weights * xx).sum() / weight_sum)
        center_y = float((weights * yy).sum() / weight_sum)
        variance_x = float((weights * (xx - center_x) ** 2).sum() / weight_sum)
        variance_y = float((weights * (yy - center_y) ** 2).sum() / weight_sum)
        covariance = float((weights * (xx - center_x) * (yy - center_y)).sum() / weight_sum)
        eigenvalues = np.linalg.eigvalsh(
            np.asarray(((variance_x, covariance), (covariance, variance_y)), dtype=np.float64)
        )
        major = max(float(eigenvalues[-1]), 0.0)
        minor = max(float(eigenvalues[0]), 0.0)
        fwhm = 2.35482 * float(np.sqrt(max((major + minor) / 2.0, 0.0)))
        ellipticity = 1.0 - np.sqrt(minor) / max(np.sqrt(major), np.finfo(np.float64).eps)
        if not np.isfinite(fwhm) or not 0.5 <= fwhm <= 12.0:
            continue
        if not np.isfinite(ellipticity) or ellipticity > 0.75:
            continue
        reference_rows.append((float(x), float(y), float(response_snr_map[y, x])))
    if not reference_rows:
        return empty, 0

    reference_coordinates = np.asarray(
        [(row[0], row[1]) for row in reference_rows],
        dtype=np.float32,
    )
    # 用多尺度 Gaussian 权重在每帧原图的预测位置附近做廉价强制响应。
    # 参考图经过注册和四舍五入后，真实 PSF 中心可能落在预测整数点的
    # 相邻像素；若只在一个像素上测量，会把亚像素配准误差误判成形状不符。
    # 因此只在 ±1 px 的 3×3 小窗口内寻找最高匹配响应，再把同一个中心
    # 用于 3×3 支持和后续 PSF 相关。搜索半径保持很小，避免把空间邻域中
    # 随机噪声的最大值当成星点坐标。
    local_search_radius = 1
    kernel_radius = max(3, int(np.ceil(2.75 * max(temporal_sigmas))))
    patch_radius = kernel_radius + local_search_radius
    offset_y, offset_x = np.mgrid[-patch_radius : patch_radius + 1, -patch_radius : patch_radius + 1]
    offset_y = offset_y.reshape(-1)
    offset_x = offset_x.reshape(-1)
    patch_side = 2 * patch_radius + 1
    kernel_offset_y, kernel_offset_x = np.mgrid[
        -kernel_radius : kernel_radius + 1,
        -kernel_radius : kernel_radius + 1,
    ]
    kernel_offset_y = kernel_offset_y.reshape(-1)
    kernel_offset_x = kernel_offset_x.reshape(-1)
    search_offset_y, search_offset_x = np.mgrid[
        -local_search_radius : local_search_radius + 1,
        -local_search_radius : local_search_radius + 1,
    ]
    search_offset_y = search_offset_y.reshape(-1)
    search_offset_x = search_offset_x.reshape(-1)
    search_patch_indices = np.asarray(
        [
            (kernel_offset_y + dy + patch_radius) * patch_side
            + (kernel_offset_x + dx + patch_radius)
            for dy, dx in zip(search_offset_y, search_offset_x, strict=True)
        ],
        dtype=np.intp,
    )
    core_offset_y, core_offset_x = np.mgrid[-1:2, -1:2]
    core_offset_y = core_offset_y.reshape(-1)
    core_offset_x = core_offset_x.reshape(-1)
    search_core_indices = np.asarray(
        [
            (core_offset_y + dy + patch_radius) * patch_side
            + (core_offset_x + dx + patch_radius)
            for dy, dx in zip(search_offset_y, search_offset_x, strict=True)
        ],
        dtype=np.intp,
    )
    search_center_indices = np.asarray(
        [
            (dy + patch_radius) * patch_side + (dx + patch_radius)
            for dy, dx in zip(search_offset_y, search_offset_x, strict=True)
        ],
        dtype=np.intp,
    )
    kernels = tuple(
        np.exp(
            -0.5
            * ((kernel_offset_x**2 + kernel_offset_y**2) / sigma**2)
        ).astype(np.float32)
        for sigma in temporal_sigmas
    )
    output: list[np.ndarray] = []
    frame_gate = max(4.0, 0.5 * float(min_candidate_snr))
    for frame_index, (analysis, shift) in enumerate(zip(frame_analyses, cumulative_shifts, strict=True)):
        image = np.asarray(analysis.frame.data, dtype=np.float32)
        frame_background = float(analysis.detection.background)
        frame_noise = max(float(analysis.detection.noise), np.finfo(np.float32).eps)
        x_integer = np.rint(reference_coordinates[:, 0] + float(shift[0])).astype(np.intp)
        y_integer = np.rint(reference_coordinates[:, 1] + float(shift[1])).astype(np.intp)
        inside = (
            (x_integer >= patch_radius)
            & (x_integer < image.shape[1] - patch_radius)
            & (y_integer >= patch_radius)
            & (y_integer < image.shape[0] - patch_radius)
        )
        frame_rows = np.empty((0, 5), dtype=np.float32)
        valid_indices = np.flatnonzero(inside)
        if valid_indices.size:
            patches = image[
                y_integer[valid_indices, None] + offset_y[None, :],
                x_integer[valid_indices, None] + offset_x[None, :],
            ]
            pixel_valid = np.isfinite(patches)
            parameters = analysis.detection.parameters
            if int(parameters.get("mask_zero_pixels", 0)):
                pixel_valid &= patches != 0.0
            saturation_level = float(parameters.get("saturation_level", -1.0))
            if saturation_level > 0:
                pixel_valid &= patches < saturation_level
            absolute_y = y_integer[valid_indices, None] + offset_y[None, :]
            absolute_x = x_integer[valid_indices, None] + offset_x[None, :]
            pixel_valid &= ~((absolute_y == 0) & (absolute_x < 104))
            residual_patch = patches - frame_background
            scores = np.full(
                (valid_indices.size, search_offset_x.size),
                -np.inf,
                dtype=np.float32,
            )
            for kernel in kernels:
                kernel_sq = kernel**2
                for search_index, patch_indices in enumerate(search_patch_indices):
                    sampled = patches[:, patch_indices]
                    sampled_valid = pixel_valid[:, patch_indices]
                    denominator = frame_noise * np.sqrt(
                        np.maximum(
                            np.sum(kernel_sq[None, :] * sampled_valid, axis=1),
                            np.finfo(np.float32).eps,
                        )
                    )
                    scale_scores = np.sum(
                        np.where(sampled_valid, sampled - frame_background, 0.0)
                        * kernel[None, :],
                        axis=1,
                    ) / denominator
                    scores[:, search_index] = np.maximum(
                        scores[:, search_index], scale_scores
                    )
            best_search = np.argmax(scores, axis=1)
            best_scores = scores[np.arange(valid_indices.size), best_search]
            selected_core_indices = search_core_indices[best_search]
            row_indices = np.arange(valid_indices.size, dtype=np.intp)[:, None]
            core = residual_patch[row_indices, selected_core_indices]
            core_valid = pixel_valid[row_indices, selected_core_indices]
            selected_centers = search_center_indices[best_search]
            center_valid = pixel_valid[np.arange(valid_indices.size), selected_centers]
            core_peak = np.maximum(residual_patch[np.arange(valid_indices.size), selected_centers], 0.0)
            support_threshold = np.maximum(2.0 * frame_noise, 0.1 * core_peak)
            support = np.count_nonzero(
                (core >= support_threshold[:, None]) & core_valid,
                axis=1,
            )
            keep = (
                np.isfinite(best_scores)
                & (best_scores >= frame_gate)
                & center_valid
                & (support >= required_support)
            )
            if np.any(keep):
                kept_indices = valid_indices[keep]
                kept_search = best_search[keep]
                frame_rows = np.column_stack(
                    (
                        x_integer[kept_indices] + search_offset_x[kept_search],
                        y_integer[kept_indices] + search_offset_y[kept_search],
                        best_scores[keep],
                        np.ones(kept_indices.size, dtype=np.float32),
                        response_snr_map[y_integer[kept_indices], x_integer[kept_indices]],
                    )
                ).astype(np.float32, copy=False)
        output.append(frame_rows)
    return tuple(output), len(reference_rows)


def detect_motion_features(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    psf_fwhm: float,
    residual_sigma: float = 15.0,
    min_residual_adu: float = 100.0,
    min_feature_area: int = 40,
    min_axis_ratio: float = 4.0,
    temporal_reference: np.ndarray | None = None,
    progress: Callable[[float, str], None] | None = None,
    audit_sink: list[MotionFrameAudit] | None = None,
) -> tuple[tuple[MotionFeatureTrack, ...], float]:
    """从每帧原图中提取长线，再用配准坐标做跨帧关联。

    当前数据中的运动目标不是一个每帧只移动几像素的点，而是高亮、细长的
    拖影/线状结构。旧的点源质量筛选会正确地拒绝它，但也因此把它完全从
    运动层抹掉。这里不把它重新塞回星点列表，而是单独建立“线状候选”层：
    先用原图背景和 RMS 找显著正残差，再用连通域 PCA 排除普通圆点和孤立
    热像素，最后在已经估计的全局平移坐标中关联。只有跨帧形成稳定轨迹的
    线状候选才标成 ``moving``；单帧线保留为 ``candidate`` 供人工复核。
    """

    if not frame_analyses:
        return (), float(min_residual_adu)
    if len(cumulative_shifts) != len(frame_analyses):
        raise ValueError("cumulative_shifts length must match frame_analyses")
    if psf_fwhm <= 0:
        raise ValueError("psf_fwhm must be positive")

    if progress is not None:
        progress(0.0, "准备线状残差筛选")
    if temporal_reference is None:
        temporal_reference = _registered_median_reference(
            frame_analyses,
            cumulative_shifts,
            progress=progress,
        )
    elif tuple(np.asarray(temporal_reference).shape) != tuple(int(value) for value in frame_analyses[0].frame.data.shape):
        # 预计算参考图尺寸不一致时安全回退，不让一个旧结果改变当前
        # 线状候选的检测坐标。
        temporal_reference = _registered_median_reference(
            frame_analyses,
            cumulative_shifts,
            progress=progress,
        )
    if progress is not None:
        progress(20.0, "完成时间中值参考")
    difference_noise_factor = np.sqrt(2.0) if temporal_reference is not None else 1.0
    min_pixels = max(int(min_feature_area), int(np.ceil(4.0 * psf_fwhm**2)))
    min_length = max(18.0, 6.0 * psf_fwhm)
    feature_frames: list[list[MotionFeaturePoint]] = []
    thresholds: list[float] = []
    frame_audits: list[MotionFrameAudit] = []

    for frame_index, (analysis, shift) in enumerate(zip(frame_analyses, cumulative_shifts, strict=True)):
        values = np.asarray(analysis.frame.data, dtype=np.float32)
        valid = np.isfinite(values)
        valid &= ~auxiliary_mask(values.shape)
        parameters = analysis.detection.parameters
        if int(parameters.get("mask_zero_pixels", 0)):
            valid &= values != 0.0
        saturation_level = float(parameters.get("saturation_level", -1.0))
        if saturation_level > 0:
            valid &= values < saturation_level
        background = float(analysis.detection.background)
        noise = max(float(analysis.detection.noise), np.finfo(np.float32).eps)
        residual_noise = noise * difference_noise_factor
        threshold = max(float(min_residual_adu), float(residual_sigma) * residual_noise)
        thresholds.append(threshold)
        if temporal_reference is None:
            residual = np.where(valid, values - background, 0.0)
        else:
            shift_x, shift_y = cumulative_shifts[frame_index]
            raw_reference = ndimage.shift(
                temporal_reference,
                shift=(float(shift_y), float(shift_x)),
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            residual = np.where(valid & np.isfinite(raw_reference), values - raw_reference, 0.0)
            del raw_reference
        support = valid & (residual >= threshold)
        valid_pixel_count = int(np.count_nonzero(valid))
        support_pixel_count = int(np.count_nonzero(support))
        labels, component_count = ndimage.label(support, structure=np.ones((3, 3), dtype=bool))
        slices = ndimage.find_objects(labels)
        sizes = ndimage.sum(support, labels, index=np.arange(1, component_count + 1))
        frame_features: list[MotionFeaturePoint] = []
        area_pass_count = 0
        geometry_pass_count = 0
        edge_rejected_count = 0
        for component_id, size in enumerate(sizes, start=1):
            if float(size) < min_pixels:
                continue
            area_pass_count += 1
            component_slice = slices[component_id - 1]
            if component_slice is None:
                continue
            ys, xs = component_slice
            component = labels[component_slice] == component_id
            local_y, local_x = np.nonzero(component)
            if local_x.size < 3:
                continue
            raw_x = local_x + xs.start
            raw_y = local_y + ys.start
            covariance = np.cov(np.column_stack((raw_x, raw_y)), rowvar=False, bias=True)
            eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
            major_variance = max(0.0, float(eigenvalues[-1]))
            minor_variance = max(0.0, float(eigenvalues[0]))
            length_px = 4.0 * np.sqrt(major_variance)
            width_px = 4.0 * np.sqrt(minor_variance)
            axis_ratio = np.sqrt(major_variance / max(minor_variance, np.finfo(np.float64).eps))
            if length_px < min_length or axis_ratio < min_axis_ratio:
                continue
            geometry_pass_count += 1
            x_min, x_max = int(raw_x.min()), int(raw_x.max())
            y_min, y_max = int(raw_y.min()), int(raw_y.max())
            touches_edge = x_min == 0 or y_min == 0 or x_max == values.shape[1] - 1 or y_max == values.shape[0] - 1
            # 小型边缘横线通常是数组边界/截断异常；保留足够长的真实拖影，
            # 但不把几十个像素的边缘噪声当成目标。
            if touches_edge and length_px < max(min_length, 8.0 * psf_fwhm):
                edge_rejected_count += 1
                continue
            component_values = residual[component_slice][component]
            weights = np.maximum(component_values.astype(np.float64), 0.0)
            weight_sum = float(weights.sum())
            center_x = float(np.average(raw_x, weights=weights)) if weight_sum > 0 else float(raw_x.mean())
            center_y = float(np.average(raw_y, weights=weights)) if weight_sum > 0 else float(raw_y.mean())
            major_vector = eigenvectors[:, -1]
            angle_deg = float(np.degrees(np.arctan2(major_vector[1], major_vector[0])))
            residual_peak = float(component_values.max())
            frame_features.append(
                MotionFeaturePoint(
                    frame_index=frame_index,
                    x=center_x,
                    y=center_y,
                    aligned_x=center_x - float(shift[0]),
                    aligned_y=center_y - float(shift[1]),
                    residual_snr=residual_peak / residual_noise,
                    area_pixels=int(size),
                    length_px=float(length_px),
                    width_px=float(width_px),
                    angle_deg=angle_deg,
                    bbox=(x_min, y_min, x_max, y_max),
                    touches_edge=touches_edge,
                )
            )
        feature_frames.append(frame_features)
        frame_audits.append(
            MotionFrameAudit(
                frame_index=frame_index,
                threshold_adu=float(threshold),
                residual_noise_adu=float(residual_noise),
                valid_pixel_count=valid_pixel_count,
                support_pixel_count=support_pixel_count,
                component_count=int(component_count),
                area_pass_count=area_pass_count,
                geometry_pass_count=geometry_pass_count,
                edge_rejected_count=edge_rejected_count,
                feature_count=len(frame_features),
                max_feature_residual_snr=(
                    max((float(point.residual_snr) for point in frame_features), default=None)
                ),
            )
        )
        if progress is not None:
            progress(
                20.0 + 60.0 * (frame_index + 1) / len(frame_analyses),
                f"线状残差筛选 F{frame_index + 1:02d}/{len(frame_analyses)}",
            )

    # 当前真实数据中线状目标的帧间位移约为几十像素；半径过小会漏掉，
    # 过大又会把不同边缘伪迹串成一条轨迹，因此单独使用 96 px 门限。
    link_radius_px = max(48.0, 32.0 * psf_fwhm)
    if progress is not None:
        progress(82.0, "关联线状候选")
    tracks: list[list[MotionFeaturePoint]] = []
    for frame_index, current in enumerate(feature_frames):
        if not current:
            continue
        possible: list[tuple[float, int, int]] = []
        for track_index, history in enumerate(tracks):
            last = history[-1]
            if last.frame_index != frame_index - 1:
                continue
            if len(history) >= 2:
                previous = history[-2]
                predicted_x = last.aligned_x + (last.aligned_x - previous.aligned_x)
                predicted_y = last.aligned_y + (last.aligned_y - previous.aligned_y)
            else:
                predicted_x, predicted_y = last.aligned_x, last.aligned_y
            for source_index, point in enumerate(current):
                distance = float(np.hypot(point.aligned_x - predicted_x, point.aligned_y - predicted_y))
                if distance <= link_radius_px:
                    possible.append((distance, track_index, source_index))
        possible.sort(key=lambda item: (item[0], item[1], item[2]))
        assigned_tracks: set[int] = set()
        assigned_sources: set[int] = set()
        for _distance, track_index, source_index in possible:
            if track_index in assigned_tracks or source_index in assigned_sources:
                continue
            tracks[track_index].append(current[source_index])
            assigned_tracks.add(track_index)
            assigned_sources.add(source_index)
        for source_index, point in enumerate(current):
            if source_index not in assigned_sources:
                tracks.append([point])

    rendered: list[MotionFeatureTrack] = []
    feature_min_presence = 3
    feature_min_displacement = max(20.0, 6.0 * psf_fwhm)
    feature_max_rms = max(4.0, 1.5 * psf_fwhm)
    if progress is not None:
        progress(94.0, "拟合运动轨迹")
    for track_id, points in enumerate(tracks):
        displacement, speed, fit_rms = _fit_motion_feature_track(points)
        classification = (
            "moving"
            if len(points) >= feature_min_presence
            and displacement >= feature_min_displacement
            and fit_rms is not None
            and fit_rms <= feature_max_rms
            else "candidate"
        )
        rendered.append(
            MotionFeatureTrack(
                track_id=track_id,
                classification=classification,
                points=tuple(points),
                displacement_px=displacement,
                speed_px_per_frame=speed,
                fit_rms_px=fit_rms,
            )
        )
    if progress is not None:
        progress(100.0, "线状目标筛选完成")
    if audit_sink is not None:
        audit_sink.extend(frame_audits)
    return tuple(rendered), float(np.median(thresholds)) if thresholds else float(min_residual_adu)


def detect_long_trails(
    frame_analysis: FrameAnalysis,
    *,
    psf_fwhm: float,
    residual_sigma: float = 15.0,
    min_residual_adu: float = 100.0,
    min_feature_area: int = 40,
    min_axis_ratio: float = 4.0,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[MotionFeatureTrack, ...]:
    """对单帧分析结果提取长线候选。

    ``detect_motion_features`` 原本只从 15 帧序列入口调用，导致单张图
    即使存在明显长拖影，也没有可以显示在界面上的运动线索。这里复用同一
    套残差、连通域和 PCA 几何筛选，只关闭跨帧关联；返回值明确标记为
    ``candidate``，不能把单帧形状证据冒充为已经确认的运动目标。
    """

    tracks, _threshold = detect_motion_features(
        (frame_analysis,),
        ((0.0, 0.0),),
        psf_fwhm=psf_fwhm,
        residual_sigma=residual_sigma,
        min_residual_adu=min_residual_adu,
        min_feature_area=min_feature_area,
        min_axis_ratio=min_axis_ratio,
        progress=progress,
    )
    return tracks


def detect_single_frame_long_trails(
    frame: FitsFrame,
    *,
    psf_fwhm: float,
    residual_sigma: float = 15.0,
    min_residual_adu: float = 100.0,
    min_feature_area: int = 40,
    min_axis_ratio: float = 4.0,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[MotionFeatureTrack, ...]:
    """在不等待全量星点测光的情况下提取单帧长线候选。

    单帧长线只需要全图稳健背景、有效像素和连通域几何，不需要先完成
    数万候选的孔径测光。这里构造一个只含背景基线的轻量 ``FrameAnalysis``
    再复用 ``detect_long_trails``；结果仍严格是 ``candidate``，不会把单帧
    形状证据升级成跨帧 ``moving``。
    """

    if psf_fwhm <= 0 or residual_sigma <= 0 or min_residual_adu <= 0:
        raise ValueError("single-frame trail parameters must be positive")
    if min_feature_area < 3 or min_axis_ratio <= 0:
        raise ValueError("single-frame trail geometry parameters are invalid")
    values = np.asarray(frame.data)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if progress is not None:
        progress(1.0, "估计单帧背景与噪声")
    # 这里是“先显示”的候选预览，不承担单图星等最终测量；对当前 16
    # 位 FITS，float32 可以精确表示每个 ADU，避免为这次轻量预览复制
    # 一份 128 MiB 的 float64 全图。
    calculation_dtype = np.float32 if (
        np.issubdtype(values.dtype, np.integer) or values.dtype == np.float32
    ) else np.float64
    numeric = np.asarray(values, dtype=calculation_dtype)
    auxiliary = auxiliary_mask(values.shape)
    base_mask = ~np.isfinite(numeric) | auxiliary
    global_background, _global_noise = sigma_clipped_stats(numeric, mask=base_mask)
    effective_mask, saturation_level, used_zero_mask = _working_mask(
        values,
        auxiliary,
        background=global_background,
        mask_zero_pixels=None,
        saturation_level=None,
        numeric=numeric,
    )
    background, noise = sigma_clipped_stats(numeric, mask=effective_mask)
    detection = DetectionResult(
        image_shape=(int(values.shape[0]), int(values.shape[1])),
        background=float(background),
        noise=float(noise),
        threshold=float(background + 4.0 * noise),
        candidate_count=0,
        sources=(),
        parameters={
            "mask_zero_pixels": int(used_zero_mask),
            "saturation_level": -1.0 if saturation_level is None else float(saturation_level),
        },
        quality_count=0,
    )
    analysis = FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=None)
    if progress is not None:
        progress(2.0, "开始单帧长线几何筛选")

        def trail_progress(value: float, label: str) -> None:
            progress(2.0 + 0.98 * max(0.0, min(100.0, float(value))), label)

    else:
        trail_progress = None
    return detect_long_trails(
        analysis,
        psf_fwhm=psf_fwhm,
        residual_sigma=residual_sigma,
        min_residual_adu=min_residual_adu,
        min_feature_area=min_feature_area,
        min_axis_ratio=min_axis_ratio,
        progress=trail_progress,
    )


def track_detections(
    frame_sources: Sequence[Sequence[Detection]],
    *,
    frame_names: Sequence[str] | None = None,
    link_radius_px: float = 4.0,
    min_presence: int | None = None,
    motion_min_displacement_px: float = 2.0,
    registration_radius_px: float = 8.0,
    max_motion_fit_rms_px: float = 0.75,
    persistent_min_presence: int | None = None,
) -> SequenceResult:
    """对已经检测的质量源做平移配准和轨迹关联。

    ``frame_sources`` 应传入 ``DetectionResult.quality_sources``，这样被
    局部 SNR、形状或坏像素规则拒绝的候选不会污染运动轨迹。为了便于
    单元测试和缓存复用，本函数不强制重新读取 FITS。
    """

    if not frame_sources:
        raise ValueError("frame_sources cannot be empty")
    if link_radius_px <= 0 or registration_radius_px <= 0:
        raise ValueError("link and registration radii must be positive")
    frame_count = len(frame_sources)
    if max_motion_fit_rms_px <= 0:
        raise ValueError("max_motion_fit_rms_px must be positive")
    required_presence = min_presence if min_presence is not None else max(3, int(np.ceil(frame_count * 0.8)))
    if required_presence < 1 or required_presence > frame_count:
        raise ValueError("min_presence must be within the frame count")
    persistent_required = (
        persistent_min_presence
        if persistent_min_presence is not None
        else min(required_presence, max(3, int(np.ceil(frame_count * 0.5))))
    )
    if persistent_required < 1 or persistent_required > required_presence:
        raise ValueError("persistent_min_presence must be within 1..min_presence")
    names = tuple(frame_names or (str(index) for index in range(frame_count)))
    if len(names) != frame_count:
        raise ValueError("frame_names length must match frame_sources")

    cumulative: list[tuple[float, float]] = [(0.0, 0.0)]
    for index in range(1, frame_count):
        delta = _estimate_translation(
            frame_sources[index - 1],
            frame_sources[index],
            radius_px=registration_radius_px,
        )
        previous = cumulative[-1]
        cumulative.append((previous[0] + delta[0], previous[1] + delta[1]))

    aligned_frames: list[list[tuple[Detection, float, float]]] = []
    for index, sources in enumerate(frame_sources):
        shift_x, shift_y = cumulative[index]
        aligned_frames.append([(source, source.x - shift_x, source.y - shift_y) for source in sources])

    # 每条轨迹保存最后一个点和其历史；未匹配源会在当前帧新建轨迹。
    tracks: list[list[TrackPoint]] = []
    if aligned_frames:
        for source, x, y in aligned_frames[0]:
            tracks.append(
                [
                    TrackPoint(
                        0,
                        source.detection_id,
                        source.x,
                        source.y,
                        x,
                        y,
                        source.flux_snr,
                        quality_passed=bool(source.quality_passed),
                    )
                ]
            )
    for frame_index in range(1, frame_count):
        current = aligned_frames[frame_index]
        if not current:
            continue
        points = np.array([(x, y) for _, x, y in current], dtype=np.float64)
        tree = cKDTree(points)
        possible: list[tuple[float, int, int]] = []
        for track_index, history in enumerate(tracks):
            last = history[-1]
            # 严格的连续观测：漏掉一帧就结束旧轨迹，避免把两个不相邻
            # 的偶然亮点重新串起来；运动候选的独立检测层负责处理短暂
            # 出现或拖线目标。
            if last.frame_index != frame_index - 1:
                continue
            distance, source_index = tree.query((last.aligned_x, last.aligned_y), distance_upper_bound=link_radius_px)
            if np.isfinite(distance) and source_index < len(current):
                possible.append((float(distance), track_index, int(source_index)))
        possible.sort(key=lambda item: (item[0], item[1], item[2]))
        assigned_tracks: set[int] = set()
        assigned_sources: set[int] = set()
        for _distance, track_index, source_index in possible:
            if track_index in assigned_tracks or source_index in assigned_sources:
                continue
            source, x, y = current[source_index]
            tracks[track_index].append(
                TrackPoint(
                    frame_index,
                    source.detection_id,
                    source.x,
                    source.y,
                    x,
                    y,
                    source.flux_snr,
                    quality_passed=bool(source.quality_passed),
                )
            )
            assigned_tracks.add(track_index)
            assigned_sources.add(source_index)
        for source_index, (source, x, y) in enumerate(current):
            if source_index not in assigned_sources:
                tracks.append(
                    [
                        TrackPoint(
                            frame_index,
                            source.detection_id,
                            source.x,
                            source.y,
                            x,
                            y,
                            source.flux_snr,
                            quality_passed=bool(source.quality_passed),
                        )
                    ]
                )

    rendered_tracks: list[SourceTrack] = []
    for track_id, points in enumerate(tracks):
        displacement, speed, fit_rms = _fit_track(points)
        if len(points) >= required_presence and displacement >= motion_min_displacement_px and fit_rms is not None and fit_rms <= max_motion_fit_rms_px:
            classification = "moving"
        elif len(points) >= required_presence:
            classification = "static"
        elif (
            len(points) >= persistent_required
            and displacement < motion_min_displacement_px
            and fit_rms is not None
            and fit_rms <= max_motion_fit_rms_px
        ):
            classification = "persistent"
        else:
            classification = "transient"
        rendered_tracks.append(
            SourceTrack(
                track_id=track_id,
                classification=classification,
                points=tuple(points),
                displacement_px=displacement,
                speed_px_per_frame=speed,
                fit_rms_px=fit_rms,
            )
        )
    frames = tuple(
        FrameSequenceSummary(
            frame_index=index,
            path=names[index],
            candidate_count=len(frame_sources[index]),
            returned_count=len(frame_sources[index]),
            quality_count=len(frame_sources[index]),
        )
        for index in range(frame_count)
    )
    return SequenceResult(
        frames=frames,
        cumulative_shifts=tuple(cumulative),
        tracks=tuple(rendered_tracks),
        link_radius_px=link_radius_px,
        min_presence=required_presence,
        motion_min_displacement_px=motion_min_displacement_px,
        max_motion_fit_rms_px=max_motion_fit_rms_px,
        persistent_min_presence=persistent_required,
    )


def _candidate_consensus_tracks(
    candidate_peak_frames: Sequence[object],
    cumulative_shifts: Sequence[tuple[float, float]],
    quality_tracks: Sequence[SourceTrack],
    *,
    frame_analyses: Sequence[FrameAnalysis] | None = None,
    link_radius_px: float,
    min_presence: int,
    persistent_min_presence: int,
    motion_min_displacement_px: float,
    max_motion_fit_rms_px: float,
    min_candidate_snr: float,
    temporal_candidate_min_snr: float | None = None,
    temporal_reference_min_snr: float | None = None,
    temporal_min_psf_correlation: float = 0.8,
    line_artifact_frames: Sequence[object] | None = None,
    audit_sink: MutableMapping[str, int] | None = None,
) -> tuple[SourceTrack, ...]:
    """从全量滤波峰中提取跨帧稳定、但尚未通过单帧质量的候选。

    15 帧快速路径会为了速度只测量每帧前 ``N`` 个源；如果稳定星较暗，
    它们可能在单帧质量层之外，却仍然在完整匹配滤波候选层中反复出现。
    这里不把这些峰直接当作恒星，而是用首帧候选作为锚点，在配准坐标中
    做互相最近的一对一匹配，并要求至少达到持续性门槛、位置拟合稳定、
    且普通候选峰 SNR 不低于 ``min_candidate_snr``。如果提供了原始帧分析，
    还要求候选在多数出现帧中满足 3×3 核心 PSF 支持，并排除边缘、无效
    像素、线状伪迹和单像素尖峰。时间中值/稳健叠加补提案使用独立的较低强制响应
    门槛；其参考 SNR 默认达到 ``min_candidate_snr``，也可用
    ``temporal_reference_min_snr`` 单独降低做召回抽测。已经由质量源轨迹解释的位置会被去重；返回轨迹
    统一标为 ``persistent``，供 UI 以低置信颜色显示，不能与严格单帧
    质量源混写。

    ``candidate_peak_frames`` 是检测器的瞬时内存结果，不是缓存格式的一
    部分。支持 ``numpy`` 数组和测试用的三元组序列，缺少该证据时安全地
    返回空元组。``frame_analyses`` 只用于廉价的时序核心支持审计，不会
    重新执行逐源环形测光。
    """

    if not candidate_peak_frames or len(candidate_peak_frames) != len(cumulative_shifts):
        return ()
    try:
        if all(np.asarray(raw).size == 0 for raw in candidate_peak_frames):
            return ()
    except (TypeError, ValueError):
        return ()
    if link_radius_px <= 0 or min_presence < 1 or persistent_min_presence < 1:
        raise ValueError("candidate consensus thresholds are invalid")
    if persistent_min_presence > min_presence or min_candidate_snr <= 0:
        raise ValueError("candidate consensus presence/SNR thresholds are invalid")
    temporal_reference_gate = (
        float(min_candidate_snr)
        if temporal_reference_min_snr is None
        else float(temporal_reference_min_snr)
    )
    if temporal_reference_gate <= 0 or temporal_reference_gate > float(min_candidate_snr):
        raise ValueError("temporal reference SNR threshold is invalid")
    temporal_min_snr = (
        max(5.0, 0.5 * float(min_candidate_snr))
        if temporal_candidate_min_snr is None
        else float(temporal_candidate_min_snr)
    )
    if temporal_min_snr <= 0 or temporal_min_snr > float(min_candidate_snr):
        raise ValueError("temporal candidate SNR threshold is invalid")
    if not 0.0 < float(temporal_min_psf_correlation) <= 1.0:
        raise ValueError("temporal PSF correlation threshold is invalid")

    normalized: list[np.ndarray] = []
    input_row_count = 0
    gated_row_count = 0
    for raw in candidate_peak_frames:
        try:
            array = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            return ()
        if array.size == 0:
            normalized.append(np.empty((0, 5), dtype=np.float64))
            continue
        if array.ndim != 2 or array.shape[1] < 3:
            return ()
        input_row_count += int(array.shape[0])
        finite = np.isfinite(array[:, :3]).all(axis=1)
        temporal_channel = np.zeros(array.shape[0], dtype=bool)
        reference_snr = np.full(array.shape[0], np.nan, dtype=np.float64)
        if array.shape[1] >= 4:
            temporal_channel = np.isfinite(array[:, 3]) & (array[:, 3] >= 0.5)
        if array.shape[1] >= 5:
            reference_snr = np.asarray(array[:, 4], dtype=np.float64)
        score_gate = np.where(temporal_channel, temporal_min_snr, float(min_candidate_snr))
        finite &= array[:, 2] >= score_gate
        if np.any(temporal_channel):
            finite &= (~temporal_channel) | (
                np.isfinite(reference_snr) & (reference_snr >= temporal_reference_gate)
            )
        filtered = np.column_stack(
            (
                np.asarray(array[:, :3], dtype=np.float64),
                temporal_channel.astype(np.float64),
                reference_snr,
            )
        )[finite]
        gated_row_count += int(filtered.shape[0])
        normalized.append(filtered)
    if audit_sink is not None:
        audit_sink["input_rows"] = int(input_row_count)
        audit_sink["rows_after_snr_gate"] = int(gated_row_count)
    if not normalized or not any(array.size for array in normalized):
        return ()
    if line_artifact_frames is not None and len(line_artifact_frames) != len(normalized):
        raise ValueError("line_artifact_frames length must match candidate_peak_frames")

    # 不再把第 1 帧硬编码成唯一锚点。真实序列中某一帧可能被坏像素、
    # 亮线、重叠源或局部低响应遮住；如果只从第 1 帧起轨迹，这类源即使
    # 在后续 14 帧稳定出现，也没有机会进入共识。先在配准坐标中建立各帧
    # 候选的空间并集，再对并集逐帧匹配。新增锚点只来自距离现有锚点超过
    # link_radius_px 的位置，避免同一源被重复播种；最终仍要求 presence、
    # 形状、线状、静态拟合和质量轨迹去重等细筛条件。
    anchor_aligned_parts: list[np.ndarray] = []
    anchor_metadata_parts: list[np.ndarray] = []
    for frame_index, current in enumerate(normalized):
        if current.size == 0:
            continue
        shift_x, shift_y = cumulative_shifts[frame_index]
        current_aligned = current[:, :2] - np.asarray((shift_x, shift_y), dtype=np.float64)
        if anchor_aligned_parts:
            existing_aligned = np.vstack(anchor_aligned_parts)
            anchor_tree = cKDTree(existing_aligned)
            distances, _existing_indices = anchor_tree.query(
                current_aligned,
                k=1,
                distance_upper_bound=float(link_radius_px),
            )
            new_mask = ~np.isfinite(distances)
        else:
            new_mask = np.ones(current.shape[0], dtype=bool)
        if np.any(new_mask):
            anchor_aligned_parts.append(current_aligned[new_mask])
            anchor_metadata_parts.append(current[new_mask, 2:5])
    if not anchor_aligned_parts:
        return ()
    anchor_aligned = np.vstack(anchor_aligned_parts).astype(np.float64, copy=False)
    # anchors 的 x/y 仅保留配准坐标；源级元数据列仍是候选分数、通道和
    # 时间中值参考分数。后续输出点始终从具体帧的 current 行读取原始坐标。
    anchors = np.column_stack(
        (anchor_aligned, np.vstack(anchor_metadata_parts).astype(np.float64, copy=False))
    )
    anchor_tree = cKDTree(anchor_aligned)
    match_indices = np.full((anchors.shape[0], len(normalized)), -1, dtype=np.int32)
    match_scores = np.full((anchors.shape[0], len(normalized)), np.nan, dtype=np.float64)

    for frame_index in range(len(normalized)):
        current = normalized[frame_index]
        if current.size == 0:
            continue
        shift_x, shift_y = cumulative_shifts[frame_index]
        current_aligned = current[:, :2] - np.asarray((shift_x, shift_y), dtype=np.float64)
        current_tree = cKDTree(current_aligned)
        distances, current_indices = current_tree.query(
            anchor_aligned,
            k=1,
            distance_upper_bound=float(link_radius_px),
        )
        reverse_distances, reverse_indices = anchor_tree.query(
            current_aligned,
            k=1,
            distance_upper_bound=float(link_radius_px),
        )
        del reverse_distances
        valid = np.isfinite(distances) & (current_indices < len(current))
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            continue
        selected_current = current_indices[valid_indices].astype(np.intp, copy=False)
        reciprocal = (selected_current < len(reverse_indices)) & (
            reverse_indices[selected_current] == valid_indices
        )
        valid_indices = valid_indices[reciprocal]
        if valid_indices.size == 0:
            continue
        selected_current = current_indices[valid_indices].astype(np.intp, copy=False)
        # reciprocal 最近邻已经排除了大部分多对一情况；这里再按距离做一
        # 次显式的一对一选择，避免两个并集锚点在密集源区共享同一个峰。
        order = np.argsort(distances[valid_indices], kind="stable")
        used_current: set[int] = set()
        selected_anchors: list[int] = []
        selected_sources: list[int] = []
        for ordered_index in order:
            anchor_index = int(valid_indices[ordered_index])
            source_index = int(selected_current[ordered_index])
            if source_index in used_current:
                continue
            used_current.add(source_index)
            selected_anchors.append(anchor_index)
            selected_sources.append(source_index)
        if selected_anchors:
            selected_anchor_array = np.asarray(selected_anchors, dtype=np.intp)
            selected_source_array = np.asarray(selected_sources, dtype=np.intp)
            match_indices[selected_anchor_array, frame_index] = selected_source_array
            match_scores[selected_anchor_array, frame_index] = current[selected_source_array, 2]

    presence = np.count_nonzero(match_indices >= 0, axis=1)
    candidate_anchor_indices = np.flatnonzero(presence >= int(persistent_min_presence))
    if audit_sink is not None:
        audit_sink["anchor_total"] = int(anchors.shape[0])
        audit_sink["anchor_presence_ge_persistent"] = int(candidate_anchor_indices.size)
    if candidate_anchor_indices.size == 0:
        return ()

    # 对候选补充一个跨帧的 3×3 核心支持审计。候选峰分数只描述匹配滤波
    # 响应，固定读出纹理也可能在多帧重复；要求核心周围有多个同量级正
    # 像素，才能把“重复出现”与“重复单像素尖峰”分开。这里按帧向量化，
    # 不调用逐候选的完整测光函数，以免稳定候选补检重新变成分钟级。
    temporal_support: np.ndarray | None = None
    temporal_valid: np.ndarray | None = None
    temporal_line_overlap: np.ndarray | None = None
    temporal_shape_support: np.ndarray | None = None
    temporal_psf_fit_support: np.ndarray | None = None
    if frame_analyses is not None:
        if len(frame_analyses) != len(normalized):
            raise ValueError("frame_analyses length must match candidate_peak_frames")
        temporal_support = np.zeros(anchors.shape[0], dtype=np.int16)
        temporal_valid = np.zeros(anchors.shape[0], dtype=np.int16)
        temporal_line_overlap = np.zeros(anchors.shape[0], dtype=np.int16)
        temporal_shape_support = np.zeros(anchors.shape[0], dtype=np.int16)
        temporal_psf_fit_support = np.zeros(anchors.shape[0], dtype=np.int16)
        offsets_y, offsets_x = np.mgrid[-1:2, -1:2]
        offsets_y = offsets_y.reshape(-1)
        offsets_x = offsets_x.reshape(-1)
        required_support = max(
            3,
            int(frame_analyses[0].detection.parameters.get("min_psf_support_pixels", 3)),
        )
        aperture_radius = max(1, int(frame_analyses[0].detection.parameters.get("aperture_radius", 4)))
        aperture_offsets_y, aperture_offsets_x = np.mgrid[
            -aperture_radius : aperture_radius + 1,
            -aperture_radius : aperture_radius + 1,
        ]
        aperture_offsets_y = aperture_offsets_y.reshape(-1)
        aperture_offsets_x = aperture_offsets_x.reshape(-1)
        aperture_geometry = (
            (aperture_offsets_x.astype(np.float64) ** 2)
            + (aperture_offsets_y.astype(np.float64) ** 2)
            <= float(aperture_radius**2)
        )
        annulus_inner_radius = max(1.5, 0.75 * float(aperture_radius))
        annulus_geometry = aperture_geometry & (
            (aperture_offsets_x.astype(np.float64) ** 2)
            + (aperture_offsets_y.astype(np.float64) ** 2)
            >= annulus_inner_radius**2
        )
        minimum_annulus_pixels = max(8, int(np.ceil(0.25 * int(np.count_nonzero(aperture_geometry)))))
        try:
            temporal_psf_fwhm = float(frame_analyses[0].detection.parameters.get("psf_fwhm", 3.0))
        except (AttributeError, TypeError, ValueError):
            temporal_psf_fwhm = 3.0
        psf_fit_sigmas = tuple(
            max(fwhm / 2.35482, 0.5)
            for fwhm in _temporal_psf_fwhm_bank(temporal_psf_fwhm)
        )
        psf_fit_kernels = tuple(
            np.exp(
                -0.5
                * (
                    (aperture_offsets_x.astype(np.float64) ** 2)
                    + (aperture_offsets_y.astype(np.float64) ** 2)
                )
                / sigma**2
            )
            for sigma in psf_fit_sigmas
        )
        for frame_index, (current, analysis) in enumerate(zip(normalized, frame_analyses, strict=True)):
            active = candidate_anchor_indices[match_indices[candidate_anchor_indices, frame_index] >= 0]
            if active.size == 0:
                continue
            current_indices = match_indices[active, frame_index].astype(np.intp, copy=False)
            points = current[current_indices]
            integer_x = np.rint(points[:, 0]).astype(np.intp)
            integer_y = np.rint(points[:, 1]).astype(np.intp)
            height, width = analysis.frame.data.shape
            inside = (
                (integer_x >= 1)
                & (integer_x < width - 1)
                & (integer_y >= 1)
                & (integer_y < height - 1)
            )
            edge_margin = aperture_radius + 4
            inside &= (
                (integer_x >= edge_margin)
                & (integer_x < width - edge_margin)
                & (integer_y >= edge_margin)
                & (integer_y < height - edge_margin)
            )
            if not np.any(inside):
                continue
            active = active[inside]
            integer_x = integer_x[inside]
            integer_y = integer_y[inside]
            pixel_values = np.asarray(analysis.frame.data)[
                integer_y[:, None] + offsets_y[None, :],
                integer_x[:, None] + offsets_x[None, :],
            ].astype(np.float64, copy=False)
            aperture_values = np.asarray(analysis.frame.data)[
                integer_y[:, None] + aperture_offsets_y[None, :],
                integer_x[:, None] + aperture_offsets_x[None, :],
            ]
            global_background = float(analysis.detection.background)
            global_noise = max(float(analysis.detection.noise), np.finfo(np.float64).eps)
            # 零值/NaN 是像素级坏点，不能因为孔径中恰好有一个坏像素就
            # 把整帧观测判成无效。中心必须有效，圆形孔径仍需保留足够的
            # 有效像素；后续支持和形状统计只消费对应的有效像素。
            pixel_valid = np.isfinite(pixel_values)
            aperture_valid = np.isfinite(aperture_values)
            if int(analysis.detection.parameters.get("mask_zero_pixels", 0)):
                pixel_valid &= pixel_values != 0.0
                aperture_valid &= aperture_values != 0.0
            saturation = float(analysis.detection.parameters.get("saturation_level", -1.0))
            if saturation > 0:
                pixel_valid &= pixel_values < saturation
                aperture_valid &= aperture_values < saturation
            aperture_y = integer_y[:, None] + aperture_offsets_y[None, :]
            aperture_x = integer_x[:, None] + aperture_offsets_x[None, :]
            aperture_valid &= ~((aperture_y == 0) & (aperture_x < 104))
            core_y = integer_y[:, None] + offsets_y[None, :]
            core_x = integer_x[:, None] + offsets_x[None, :]
            pixel_valid &= ~((core_y == 0) & (core_x < 104))
            aperture_valid_count = np.count_nonzero(
                aperture_valid & aperture_geometry[None, :],
                axis=1,
            )
            minimum_aperture_valid = max(
                9,
                int(np.ceil(0.8 * int(np.count_nonzero(aperture_geometry)))),
            )
            finite = (
                pixel_valid[:, 4]
                & (aperture_valid_count >= minimum_aperture_valid)
            )
            # 候选共识阶段不能只减每帧一个全局背景。当前 FITS 存在明显的
            # 空间纹理/亮度梯度；若直接用全局背景，宽纹理会在 3×3 支持和
            # 二阶矩中伪装成“重复的星点”。用孔径外环估计每个候选的局部
            # 中位数和 MAD，并保留全局噪声作为无效/样本不足时的回退。
            annulus_valid = aperture_valid & annulus_geometry[None, :]
            annulus_count = np.count_nonzero(annulus_valid, axis=1)
            annulus_values = np.where(annulus_valid, aperture_values, np.nan).astype(np.float64, copy=False)
            with np.errstate(invalid="ignore"):
                local_background = np.nanmedian(annulus_values, axis=1)
                local_deviation = np.abs(annulus_values - local_background[:, None])
                local_noise = 1.4826 * np.nanmedian(local_deviation, axis=1)
            usable_annulus = (
                (annulus_count >= minimum_annulus_pixels)
                & np.isfinite(local_background)
                & np.isfinite(local_noise)
                & (local_noise > np.finfo(np.float64).eps)
            )
            local_background = np.where(usable_annulus, local_background, global_background)
            local_noise = np.where(usable_annulus, local_noise, global_noise)
            # MAD 在非常平坦的局部块中可能低估真实读出噪声；不让它低于
            # 全局噪声的 50%，避免把个别像素波动夸大成高显著性点源。
            local_noise = np.maximum(local_noise, 0.5 * global_noise)
            if line_artifact_frames is not None:
                raw_line_coordinates = np.asarray(line_artifact_frames[frame_index])
                if raw_line_coordinates.size:
                    if raw_line_coordinates.ndim != 2 or raw_line_coordinates.shape[1] < 2:
                        raise ValueError("line artifact coordinates must be an Nx2 array")
                    line_tree = cKDTree(
                        np.asarray(raw_line_coordinates[:, :2], dtype=np.float64)
                    )
                    line_distances, _line_indices = line_tree.query(
                        np.column_stack((integer_x, integer_y)),
                        k=1,
                        distance_upper_bound=float(aperture_radius),
                    )
                    line_overlap = np.isfinite(line_distances)
                    temporal_line_overlap[active[line_overlap]] += 1
                    finite &= ~line_overlap
            temporal_valid[active[finite]] += 1
            if not np.any(finite):
                continue
            valid_pixels = pixel_values[finite]
            valid_core = pixel_valid[finite]
            local_background_valid = local_background[finite]
            local_noise_valid = local_noise[finite]
            peak_excess = np.maximum(valid_pixels[:, 4] - local_background_valid, 0.0)
            support_threshold = np.maximum(2.0 * local_noise_valid, 0.1 * peak_excess)
            support_pixels = np.count_nonzero(
                (valid_pixels - local_background_valid[:, None] >= support_threshold[:, None]) & valid_core,
                axis=1,
            )
            temporal_support[active[finite]] += support_pixels >= required_support

            # 3×3 支持能排除单像素尖峰，但不能排除“宽而连续”的读出纹理。
            # 用与单帧源级测量相同的正残差二阶矩和 peak/positive_sum 指标
            # 做一个廉价形状细筛；这里只计数证据，不重算孔径通量。
            shape_values = np.where(
                aperture_valid[finite],
                aperture_values[finite].astype(np.float64, copy=False) - local_background_valid[:, None],
                0.0,
            )
            shape_geometry = aperture_geometry[None, :] & aperture_valid[finite]
            positive = np.where(
                shape_geometry,
                np.maximum(shape_values, 0.0),
                0.0,
            )
            positive_sum = positive.sum(axis=1)
            shape_peak = np.max(
                np.where(shape_geometry, shape_values, -np.inf),
                axis=1,
            )
            safe_sum = np.maximum(positive_sum, np.finfo(np.float64).eps)
            center_x = (positive * aperture_offsets_x[None, :]).sum(axis=1) / safe_sum
            center_y = (positive * aperture_offsets_y[None, :]).sum(axis=1) / safe_sum
            variance_x = (
                positive
                * (aperture_offsets_x[None, :] - center_x[:, None]) ** 2
            ).sum(axis=1) / safe_sum
            variance_y = (
                positive
                * (aperture_offsets_y[None, :] - center_y[:, None]) ** 2
            ).sum(axis=1) / safe_sum
            fwhm = 2.35482 * np.sqrt(np.maximum((variance_x + variance_y) / 2.0, 0.0))
            ellipticity = np.abs(np.sqrt(variance_x) - np.sqrt(variance_y)) / np.maximum(
                np.maximum(np.sqrt(variance_x), np.sqrt(variance_y)),
                np.finfo(np.float64).eps,
            )
            sharpness = np.maximum(shape_peak, 0.0) / safe_sum
            shape_pass = (
                np.isfinite(fwhm)
                & (fwhm >= 0.8)
                & (fwhm <= 4.5)
                & np.isfinite(ellipticity)
                & (ellipticity <= 0.65)
                & np.isfinite(sharpness)
                & (sharpness >= 0.08)
            )
            temporal_shape_support[active[finite]] += shape_pass
            # 仅靠 FWHM/椭圆率仍可能让低幅度宽纹理通过。计算局部孔径
            # 与窄/基准/宽 Gaussian 模板的归一化相关系数，取最相符尺度；
            # 这不是完整 PSF 拟合，只作为时序候选的点源形状证据。
            fit_values = np.where(shape_geometry, shape_values, 0.0)
            data_norm = np.sqrt(np.sum(fit_values**2, axis=1))
            best_correlation = np.full(fit_values.shape[0], -np.inf, dtype=np.float64)
            for kernel in psf_fit_kernels:
                template = np.where(shape_geometry, kernel[None, :], 0.0)
                template_norm = np.sqrt(np.sum(template**2, axis=1))
                denominator = data_norm * template_norm
                correlation = np.divide(
                    np.sum(fit_values * template, axis=1),
                    denominator,
                    out=np.full_like(denominator, -np.inf, dtype=np.float64),
                    where=denominator > np.finfo(np.float64).eps,
                )
                best_correlation = np.maximum(best_correlation, correlation)
            temporal_psf_fit_support[active[finite]] += (
                best_correlation >= float(temporal_min_psf_correlation)
            )

    # 只有已经达到持续门槛的质量轨迹才足以“解释掉”补检位置。此前
    # 任何只出现 1--3 帧的瞬态质量轨迹都会让整条 15 帧候选补检被去重，
    # 这会遮蔽真实但偶尔被局部峰/坏像素打断的星。短质量轨迹保留为
    # 独立瞬态证据，不得压制更强的跨帧共识。
    quality_by_frame: list[list[tuple[float, float]]] = [[] for _ in normalized]
    for track in quality_tracks:
        if track.presence < int(persistent_min_presence):
            continue
        for point in track.points:
            if 0 <= point.frame_index < len(quality_by_frame):
                quality_by_frame[point.frame_index].append((point.aligned_x, point.aligned_y))
    quality_trees = [cKDTree(points) if points else None for points in quality_by_frame]
    merge_radius = max(1.25, min(2.0, float(link_radius_px) * 0.5))

    rendered: list[SourceTrack] = []
    for anchor_index in candidate_anchor_indices:
        if audit_sink is not None:
            audit_sink["anchors_evaluated"] = audit_sink.get("anchors_evaluated", 0) + 1
        frame_indices = np.flatnonzero(match_indices[anchor_index] >= 0)
        points: list[TrackPoint] = []
        overlaps_quality = False
        for frame_index in frame_indices:
            current = normalized[frame_index]
            current_index = int(match_indices[anchor_index, frame_index])
            raw_x, raw_y, score = current[current_index, :3]
            shift_x, shift_y = cumulative_shifts[frame_index]
            point = TrackPoint(
                frame_index=int(frame_index),
                detection_id=-1 - int(anchor_index),
                x=float(raw_x),
                y=float(raw_y),
                aligned_x=float(raw_x - shift_x),
                aligned_y=float(raw_y - shift_y),
                flux_snr=None,
                quality_passed=False,
                candidate_snr=float(score),
            )
            points.append(point)
            tree = quality_trees[frame_index]
            if tree is not None:
                distance, _index = tree.query((point.aligned_x, point.aligned_y), k=1)
                if np.isfinite(distance) and float(distance) <= merge_radius:
                    overlaps_quality = True
        if overlaps_quality:
            if audit_sink is not None:
                audit_sink["reject_quality_overlap"] = audit_sink.get("reject_quality_overlap", 0) + 1
            continue
        if len(points) < int(persistent_min_presence):
            if audit_sink is not None:
                audit_sink["reject_presence"] = audit_sink.get("reject_presence", 0) + 1
            continue
        median_candidate_snr = float(np.nanmedian(match_scores[anchor_index, frame_indices]))
        anchor_is_temporal = bool(anchors[anchor_index, 3] >= 0.5)
        score_threshold = temporal_min_snr if anchor_is_temporal else float(min_candidate_snr)
        if not np.isfinite(median_candidate_snr) or median_candidate_snr < score_threshold:
            if audit_sink is not None:
                audit_sink["reject_median_candidate_snr"] = audit_sink.get("reject_median_candidate_snr", 0) + 1
            continue
        if anchor_is_temporal and (
            not np.isfinite(anchors[anchor_index, 4])
            or anchors[anchor_index, 4] < temporal_reference_gate
        ):
            if audit_sink is not None:
                audit_sink["reject_temporal_reference_snr"] = audit_sink.get("reject_temporal_reference_snr", 0) + 1
            continue
        if temporal_support is not None and temporal_valid is not None:
            if audit_sink is not None:
                for name, value in (
                    ("presence", len(points)),
                    ("temporal_valid", int(temporal_valid[anchor_index])),
                    ("temporal_support", int(temporal_support[anchor_index])),
                    (
                        "temporal_shape",
                        int(temporal_shape_support[anchor_index])
                        if temporal_shape_support is not None
                        else -1,
                    ),
                    (
                        "temporal_psf_correlation",
                        int(temporal_psf_fit_support[anchor_index])
                        if temporal_psf_fit_support is not None
                        else -1,
                    ),
                    (
                        "line_overlap",
                        int(temporal_line_overlap[anchor_index])
                        if temporal_line_overlap is not None
                        else 0,
                    ),
                ):
                    key = f"{name}_{value}"
                    audit_sink[key] = audit_sink.get(key, 0) + 1
            # 单个坏像素可能只影响一帧的孔径；要求大多数而非所有
            # 观测孔径可用，避免把真实但偶尔被坏点覆盖的暗星全部漏掉。
            minimum_valid = max(1, int(np.ceil(0.8 * len(points))))
            if temporal_valid[anchor_index] < minimum_valid:
                if audit_sink is not None:
                    audit_sink["reject_temporal_valid"] = audit_sink.get("reject_temporal_valid", 0) + 1
                continue
            if temporal_support[anchor_index] < max(1, int(np.ceil(0.6 * temporal_valid[anchor_index]))):
                if audit_sink is not None:
                    audit_sink["reject_temporal_support"] = audit_sink.get("reject_temporal_support", 0) + 1
                    support_key = f"support_reject_valid_{int(temporal_valid[anchor_index])}_support_{int(temporal_support[anchor_index])}"
                    audit_sink[support_key] = audit_sink.get(support_key, 0) + 1
                continue
            if temporal_line_overlap is not None:
                # 亮轨迹只擦过一两帧时，不能把其余 13--14 帧的稳定
                # 点源证据一并丢掉；但短序列保持保守，且整条轨迹被
                # 亮线覆盖时仍必须拒绝。允许的污染比例最多为 20%，
                # 最少需要 5 帧才启用“偶发擦过”例外。
                line_overlap_count = int(temporal_line_overlap[anchor_index])
                allowed_line_overlap = (
                    max(1, int(np.floor(0.2 * len(points))))
                    if len(points) >= 5
                    else 0
                )
                if line_overlap_count > allowed_line_overlap:
                    if audit_sink is not None:
                        audit_sink["reject_line_overlap"] = audit_sink.get("reject_line_overlap", 0) + 1
                    continue
            if temporal_shape_support is not None and temporal_valid[anchor_index] > 0:
                if temporal_shape_support[anchor_index] < max(
                    1,
                    int(np.ceil(0.6 * temporal_valid[anchor_index])),
                ):
                    if audit_sink is not None:
                        audit_sink["reject_temporal_shape"] = audit_sink.get("reject_temporal_shape", 0) + 1
                    continue
            if anchor_is_temporal and temporal_psf_fit_support is not None and temporal_valid[anchor_index] > 0:
                if temporal_psf_fit_support[anchor_index] < max(
                    1,
                    int(np.ceil(0.6 * temporal_valid[anchor_index])),
                ):
                    if audit_sink is not None:
                        audit_sink["reject_temporal_psf_correlation"] = audit_sink.get(
                            "reject_temporal_psf_correlation",
                            0,
                        ) + 1
                    continue
        displacement, speed, fit_rms = _fit_track(points)
        if (
            fit_rms is None
            or fit_rms > float(max_motion_fit_rms_px)
            or displacement >= float(motion_min_displacement_px)
        ):
            # 候选峰也可能来自随帧移动的伪迹；不把它们塞进稳定场。
            if audit_sink is not None:
                audit_sink["reject_motion_fit"] = audit_sink.get("reject_motion_fit", 0) + 1
            continue
        rendered.append(
            SourceTrack(
                track_id=-1,
                classification="persistent",
                points=tuple(points),
                displacement_px=float(displacement),
                speed_px_per_frame=float(speed),
                fit_rms_px=float(fit_rms),
                evidence_level=(
                    "temporal_reference"
                    if anchor_is_temporal
                    else "candidate_consensus"
                ),
            )
        )
        if audit_sink is not None:
            audit_sink["accepted"] = audit_sink.get("accepted", 0) + 1
            key = "accepted_temporal_reference" if anchor_is_temporal else "accepted_candidate_consensus"
            audit_sink[key] = audit_sink.get(key, 0) + 1
    return tuple(rendered)


def _stack_forced_frame_measure(
    image: np.ndarray,
    mask: np.ndarray,
    x: float,
    y: float,
    *,
    aperture_radius: int,
    min_psf_support_pixels: int,
    psf_fwhm: float,
    global_background: float,
    global_noise: float,
) -> tuple[float | None, int, bool, float | None, float | None]:
    """在单帧原图 (x, y) 处做紧凑强制测光。

    叠加参考图把噪声压到约 ``1/√N``，但固定热像素和静态纹理也会一起
    保留；因此任何叠加候选都必须回到每帧原始图像做局部环测光、3×3 支持
    和点源形状审计，才能算作“该帧确实出现”。这里只做最小必要测量，
    复用与 ``_source_from_peak`` 相同的局部环背景、孔径通量、flux_snr、
    ``3×3`` 支持与二阶矩形状口径，避免为叠加候选再走一遍全帧背景网格。
    返回 ``(flux_snr, support, center_valid, fwhm, ellipticity)``。
    """

    height, width = image.shape
    outer = aperture_radius + 4
    y0 = max(0, int(np.floor(y)) - outer)
    y1 = min(height, int(np.floor(y)) + outer + 1)
    x0 = max(0, int(np.floor(x)) - outer)
    x1 = min(width, int(np.floor(x)) + outer + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None, 0, False, None, None
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    patch_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool) | ~np.isfinite(patch)
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    center_x = x - x0
    center_y = y - y0
    distance = np.hypot(xx - center_x, yy - center_y)
    aperture = (distance <= aperture_radius) & ~patch_mask
    annulus = (distance > aperture_radius + 1) & (distance <= outer) & ~patch_mask
    annulus_values = patch[annulus]
    if annulus_values.size < 8:
        local_background = float(global_background)
        local_noise = max(float(global_noise), np.finfo(np.float64).eps)
    else:
        local_background, local_noise = sigma_clipped_stats(
            annulus_values,
            sample_limit=100_000,
        )
        local_noise = max(float(local_noise), np.finfo(np.float64).eps)
    # 局部环可能落在亮星翼部里，噪声被系统性抬高；但相对全局噪声过高的
    # 环不可靠，直接退回全局噪声，避免把翼部纹理误判成高显著性暗星。
    if local_noise > 3.0 * max(float(global_noise), np.finfo(np.float64).eps):
        local_noise = max(float(global_noise), np.finfo(np.float64).eps)
    residual = patch - local_background
    valid_aperture = residual[aperture]
    net_flux = float(valid_aperture.sum()) if valid_aperture.size else 0.0
    aperture_pixels = int(aperture.sum())
    if aperture_pixels < 3:
        return None, 0, False, None, None
    background_pixels = max(1, int(annulus.sum()))
    background_variance = aperture_pixels * local_noise**2
    background_variance += (aperture_pixels**2 / background_pixels) * local_noise**2
    flux_error = float(np.sqrt(max(background_variance, np.finfo(np.float64).eps)))
    flux_snr = net_flux / flux_error

    positive = np.where(aperture, np.maximum(residual, 0.0), 0.0)
    positive_sum = float(positive.sum())
    if positive_sum > 0:
        shape_x = float((positive * (xx + x0)).sum() / positive_sum)
        shape_y = float((positive * (yy + y0)).sum() / positive_sum)
        variance_x = float((positive * ((xx + x0) - shape_x) ** 2).sum() / positive_sum)
        variance_y = float((positive * ((yy + y0) - shape_y) ** 2).sum() / positive_sum)
        fwhm_x = 2.35482 * float(np.sqrt(max(0.0, variance_x)))
        fwhm_y = 2.35482 * float(np.sqrt(max(0.0, variance_y)))
        fwhm = 2.35482 * float(np.sqrt(max(0.0, (variance_x + variance_y) / 2.0)))
        ellipticity = abs(fwhm_x - fwhm_y) / max(fwhm_x, fwhm_y, np.finfo(np.float64).eps)
        if not np.isfinite(fwhm) or fwhm <= 0:
            fwhm = None
            ellipticity = None
    else:
        fwhm = None
        ellipticity = None

    core_x = int(np.rint(center_x))
    core_y = int(np.rint(center_y))
    center_valid = bool(not patch_mask[core_y, core_x]) if 0 <= core_y < patch.shape[0] and 0 <= core_x < patch.shape[1] else False
    if not center_valid:
        return flux_snr, 0, False, fwhm, ellipticity
    peak_excess = max(float(patch[core_y, core_x] - local_background), 0.0)
    core_y0, core_y1 = max(0, core_y - 1), min(patch.shape[0], core_y + 2)
    core_x0, core_x1 = max(0, core_x - 1), min(patch.shape[1], core_x + 2)
    core_slice = residual[core_y0:core_y1, core_x0:core_x1]
    core_valid = ~patch_mask[core_y0:core_y1, core_x0:core_x1]
    support_threshold = max(2.0 * local_noise, 0.1 * peak_excess)
    support = int(
        np.count_nonzero(
            core_valid
            & np.isfinite(core_slice)
            & (core_slice >= support_threshold)
        )
    )
    return flux_snr, support, True, fwhm, ellipticity


def _stack_faint_tracks(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    reference: np.ndarray | None,
    *,
    threshold_sigma: float,
    min_distance: int,
    aperture_radius: int,
    psf_fwhm: float,
    min_flux_snr: float,
    min_psf_support_pixels: int,
    proposal_mode: str,
    stack_frame_min_flux_snr: float,
    min_presence: int,
    link_radius_px: float,
    existing_tracks: Sequence[SourceTrack],
    audit_sink: MutableMapping[str, int] | None = None,
) -> tuple[SourceTrack, ...]:
    """从注册中值/稳健叠加参考图恢复单帧漏掉的暗星。

    叠加把噪声降到约 ``1/√N``，能把单帧 ``flux_snr<5`` 的暗星抬到参考图
    ``flux_snr≥5``；但叠加也保留固定热像素和静态纹理，所以这里不把参考图
    检测直接当结果。流程：参考图局部测光 → 与已有质量/共识轨迹按
    ``link_radius_px`` 去重 → 回到 15 帧原图做强制测光/3×3 支持/形状审计
    → 出现帧数达到 ``min_presence`` 且逐帧 ``flux_snr`` 达到放宽门槛。
    结果标为低置信 ``evidence_level="stack_faint"`` 轨迹，分类为
    ``persistent``，不与单帧质量源或严格静态源混写。

    参考图坐标已对齐到第 1 帧；第 i 帧的原始坐标是
    ``(x + shift_x, y + shift_y)``。
    """

    if reference is None or len(frame_analyses) < 2:
        return ()
    if len(frame_analyses) != len(cumulative_shifts):
        raise ValueError("cumulative_shifts length must match frame_analyses")
    # 参考图本身是 float32（注册中值/稳健均值）。叠加上只剩“再筛一遍”，
    # 不追求单帧式环形精修；使用 float32 + 网格背景即可显著降低 4096²
    # 中间阵列的内存带宽，避免 GUI 后台在 15 帧之上再加一次 float64 全图
    # 检测而耗尽内存。参考图候选仍需逐帧原图强制测光，最终质量不依赖这
    # 一层测光的绝对精度。
    values = np.asarray(reference, dtype=np.float32)
    if values.ndim != 2:
        return ()
    reference_mask = ~np.isfinite(values)
    # 第一行辅助区在参考图中已经是 NaN；这里不再重复屏蔽，但防御性地
    # 保持与单帧一致的口径：仅当参考图没有 NaN 时才依赖 detect_sources
    # 自身的有限掩膜。
    try:
        stack_detection = detect_sources(
            values,
            mask=reference_mask,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            max_sources=None,
            psf_fwhm=psf_fwhm,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            proposal_mode=proposal_mode,
            refine_local_background=False,
            use_float32=True,
            mask_zero_pixels=False,
            allow_partial_zero_mask=False,
        )
    except ValueError:
        return ()
    stack_candidates = tuple(stack_detection.quality_sources)
    if audit_sink is not None:
        audit_sink["stack_reference_quality_sources"] = len(stack_candidates)
    if not stack_candidates:
        return ()

    # 去重：任何已经被质量轨迹或候选共识轨迹解释过的位置不再重复输出。
    explained: list[tuple[float, float]] = []
    for track in existing_tracks:
        for point in track.points:
            explained.append((point.aligned_x, point.aligned_y))
    if explained:
        tree = cKDTree(np.asarray(explained, dtype=np.float64))
    else:
        tree = None

    # 逐帧原始图、有效掩膜和全局背景/噪声只准备一次；若在候选循环内
    # 重复构造 4096² 的辅助掩膜或 ``image == 0`` 布尔阵列，数千候选会
    # 放大成数百万次整图分配，把这一层拖到不可接受的耗时。
    frame_images = tuple(np.asarray(analysis.frame.data) for analysis in frame_analyses)
    frame_masks = tuple(
        (
            auxiliary_mask(image.shape)
            | (image == 0)
            if int(analysis.detection.parameters.get("mask_zero_pixels", 0))
            else auxiliary_mask(image.shape)
        )
        for image, analysis in zip(frame_images, frame_analyses, strict=True)
    )
    frame_backgrounds = tuple(float(analysis.detection.background) for analysis in frame_analyses)
    frame_noises = tuple(float(analysis.detection.noise) for analysis in frame_analyses)

    rendered: list[SourceTrack] = []
    for candidate in stack_candidates:
        ref_x = float(candidate.peak_x if candidate.peak_x is not None else candidate.x)
        ref_y = float(candidate.peak_y if candidate.peak_y is not None else candidate.y)
        if tree is not None:
            distance, _index = tree.query((ref_x, ref_y), k=1)
            if np.isfinite(distance) and float(distance) <= float(link_radius_px):
                if audit_sink is not None:
                    audit_sink["reject_stack_explained"] = audit_sink.get("reject_stack_explained", 0) + 1
                continue
        points: list[TrackPoint] = []
        frame_flux_snrs: list[float] = []
        for frame_index, (image, mask, background, noise, shift) in enumerate(
            zip(
                frame_images,
                frame_masks,
                frame_backgrounds,
                frame_noises,
                cumulative_shifts,
                strict=True,
            )
        ):
            raw_x = ref_x + float(shift[0])
            raw_y = ref_y + float(shift[1])
            flux_snr, support, center_valid, fwhm, ellipticity = _stack_forced_frame_measure(
                image,
                mask,
                raw_x,
                raw_y,
                aperture_radius=aperture_radius,
                min_psf_support_pixels=min_psf_support_pixels,
                psf_fwhm=psf_fwhm,
                global_background=background,
                global_noise=noise,
            )
            if not center_valid:
                continue
            if support < min_psf_support_pixels:
                continue
            if flux_snr is None or not np.isfinite(flux_snr) or flux_snr < float(stack_frame_min_flux_snr):
                continue
            if fwhm is None or not 0.8 <= fwhm <= 12.0:
                continue
            if ellipticity is None or ellipticity > 0.65:
                continue
            points.append(
                TrackPoint(
                    frame_index=int(frame_index),
                    detection_id=-1,
                    x=float(raw_x),
                    y=float(raw_y),
                    aligned_x=ref_x,
                    aligned_y=ref_y,
                    flux_snr=float(flux_snr),
                    quality_passed=False,
                    candidate_snr=float(candidate.filter_snr)
                    if candidate.filter_snr is not None
                    else float(flux_snr),
                )
            )
            frame_flux_snrs.append(float(flux_snr))
        if len(points) < int(min_presence):
            if audit_sink is not None:
                audit_sink["reject_stack_presence"] = audit_sink.get("reject_stack_presence", 0) + 1
            continue
        # 少数帧被亮星翼部抬噪导致的低 flux_snr 可以吸收；要求多数出现帧
        # 有正的稳定通量，避免只凭 3×3 支持就把固定纹理算成星。
        if frame_flux_snrs:
            median_frame_flux_snr = float(np.median(frame_flux_snrs))
        else:
            median_frame_flux_snr = 0.0
        if median_frame_flux_snr < float(stack_frame_min_flux_snr):
            if audit_sink is not None:
                audit_sink["reject_stack_flux_snr"] = audit_sink.get("reject_stack_flux_snr", 0) + 1
            continue
        displacement, speed, fit_rms = _fit_track(points)
        rendered.append(
            SourceTrack(
                track_id=-1,
                classification="persistent",
                points=tuple(points),
                displacement_px=float(displacement),
                speed_px_per_frame=float(speed),
                fit_rms_px=float(fit_rms),
                evidence_level="stack_faint",
            )
        )
        if audit_sink is not None:
            audit_sink["accepted_stack_faint"] = audit_sink.get("accepted_stack_faint", 0) + 1
    return tuple(rendered)


def analyze_sequence(
    paths: Iterable[str | Path],
    *,
    link_radius_px: float = 4.0,
    min_presence: int | None = None,
    motion_min_displacement_px: float = 2.0,
    registration_radius_px: float = 8.0,
    max_motion_fit_rms_px: float = 0.75,
    persistent_min_presence: int | None = None,
    sequence_max_sources: int | None = DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT,
    sequence_workers: int = DEFAULT_SEQUENCE_WORKERS,
    candidate_consensus_min_snr: float = 15.0,
    temporal_candidate_min_snr: float | None = None,
    temporal_reference_min_snr: float | None = None,
    temporal_multiscale: bool = False,
    temporal_min_psf_correlation: float = 0.8,
    temporal_proposal_mode: str = "median",
    stack_faint_recovery: bool = True,
    stack_reference_mode: str = "median",
    stack_threshold_sigma: float = 4.0,
    stack_min_flux_snr: float = 5.0,
    stack_frame_min_flux_snr: float = 3.0,
    stack_min_presence: int | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    detail_progress: Callable[[int, int, float, str], None] | None = None,
    **detector_kwargs: object,
) -> SequenceResult:
    """读取并检测一组 FITS，再关联质量源。

    序列默认只对每帧按匹配滤波 SNR 排名前 ``sequence_max_sources`` 个源
    做源级测量和点轨迹关联，但仍保留匹配滤波得到的完整 ``candidate_count``。
    传入 ``None`` 可恢复全量序列工作集；单图 ``analyze_frame`` 不受此默认值影响。
    快速路径还保留每帧完整的匹配滤波峰坐标。序列关联完成后，会对没有
    进入单帧质量层、但在配准坐标中持续出现的峰做一次廉价的一对一共识
    检验，作为 ``persistent`` 低置信稳定候选；它不改变单帧质量计数。
    ``temporal_reference_min_snr`` 只作用于时间中值/稳健叠加提出的补充坐标，
    不能降低普通逐帧候选的 SNR 门槛；所有补提案仍需逐帧原图支持、形状、
    掩膜和持续性审计。
    ``temporal_multiscale=True`` 只扩大时间参考图的宽筛 PSF 尺度组，默认
    关闭；它适合研究窄 PSF 漏检，不能仅凭候选数增加就当作正式星表口径。
    序列还会输出固定像素值审计及其孔径影响关联；这些字段只用于定位
    坏像素/填充值对候选测量的影响，不会自动屏蔽固定值。
    整数 FITS 默认启用序列快速路径：每个 256 px 背景块最多抽样
    ``DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT`` 个像素，并在稀疏掩膜时
    使用单遍匹配滤波；快速口径会写入返回结果，单图默认不启用。
    """

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    total = len(frame_paths)
    if sequence_max_sources is not None and sequence_max_sources < 1:
        raise ValueError("sequence_max_sources must be positive or None")
    if sequence_workers < 1:
        raise ValueError("sequence_workers must be positive")
    if candidate_consensus_min_snr <= 0:
        raise ValueError("candidate_consensus_min_snr must be positive")
    temporal_proposal_mode = str(temporal_proposal_mode).strip().lower()
    if temporal_proposal_mode not in {"median", "coadd", "both"}:
        raise ValueError("temporal_proposal_mode must be 'median', 'coadd', or 'both'")
    resolved_temporal_reference_min_snr = (
        float(candidate_consensus_min_snr)
        if temporal_reference_min_snr is None
        else float(temporal_reference_min_snr)
    )
    resolved_temporal_candidate_min_snr = (
        max(5.0, 0.5 * float(candidate_consensus_min_snr))
        if temporal_candidate_min_snr is None
        else float(temporal_candidate_min_snr)
    )
    if (
        resolved_temporal_candidate_min_snr <= 0
        or resolved_temporal_candidate_min_snr > float(candidate_consensus_min_snr)
    ):
        raise ValueError("temporal_candidate_min_snr must be positive and no greater than candidate_consensus_min_snr")
    if (
        resolved_temporal_reference_min_snr <= 0
        or resolved_temporal_reference_min_snr > float(candidate_consensus_min_snr)
    ):
        raise ValueError("temporal_reference_min_snr must be positive and no greater than candidate_consensus_min_snr")
    if not 0.0 < float(temporal_min_psf_correlation) <= 1.0:
        raise ValueError("temporal_min_psf_correlation must be between 0 and 1")
    stack_reference_mode = str(stack_reference_mode).strip().lower()
    if stack_reference_mode not in {"median", "coadd"}:
        raise ValueError("stack_reference_mode must be 'median' or 'coadd'")
    if stack_threshold_sigma <= 0 or stack_min_flux_snr <= 0 or stack_frame_min_flux_snr <= 0:
        raise ValueError("stack_faint thresholds must be positive")
    if stack_min_presence is not None and stack_min_presence < 1:
        raise ValueError("stack_min_presence must be positive when provided")
    if detector_kwargs.get("max_sources") is None and sequence_max_sources is not None:
        detector_kwargs["max_sources"] = int(sequence_max_sources)
    # 序列只需要稳定的空间噪声场；更大的网格减少背景统计开销，仍会在
    # 每个像素位置插值得到 background_map/noise_map，不改变输入图像。
    detector_kwargs.setdefault("background_box_size", 256)
    # 15 帧主要需要稳定的配准参考，不需要为数万候选逐个重复做环形
    # sigma-clipping；允许调用方显式传 True 进行高精度单帧式测量。
    detector_kwargs.setdefault("refine_local_background", False)
    # 当前数据为 16 位 FITS；float32 对每个输入 ADU 是精确的，只减少
    # 4096² 像素级中间阵列的内存带宽。单图入口仍保持 float64 默认口径。
    detector_kwargs.setdefault("use_float32", True)
    detector_kwargs.setdefault(
        "background_sample_limit",
        DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT if detector_kwargs.get("use_float32", True) else 100_000,
    )
    detector_kwargs.setdefault("fast_sequence", bool(detector_kwargs.get("use_float32", True)))
    if progress is not None:
        progress("prepare", 0, total)

    # 连续观测帧共享同一片天空背景。快速序列路径先用首帧生成二维
    # background/RMS 模型，后续帧仍各自计算全局统计和有效像素掩膜，
    # 但不再重复 256 个局部网格的 sigma-clipping。测试中的虚拟路径
    # 没有真实文件，因此自然回退到逐帧模型。
    shared_background_model: tuple[np.ndarray, np.ndarray] | None = None
    if (
        total >= 2
        and bool(detector_kwargs.get("fast_sequence", False))
        and bool(detector_kwargs.get("use_float32", True))
        and frame_paths[0].is_file()
    ):
        pilot = read_fits(frame_paths[0])
        shared_background_model = build_background_model(
            pilot.data,
            mask=auxiliary_mask(pilot.data.shape),
            background_box_size=int(detector_kwargs["background_box_size"]),
            background_sample_limit=int(detector_kwargs["background_sample_limit"]),
            mask_zero_pixels=(
                bool(detector_kwargs["mask_zero_pixels"])
                if detector_kwargs.get("mask_zero_pixels") is not None
                else None
            ),
            refine_local_background=bool(detector_kwargs.get("refine_local_background", False)),
            use_float32=True,
        )
        del pilot
        if progress is not None:
            progress("background", total, total)

    def analyze_one(item: tuple[int, Path]) -> tuple[int, FrameAnalysis]:
        frame_index, path = item

        def frame_progress(value: float, label: str) -> None:
            if detail_progress is not None:
                detail_progress(frame_index, total, value, label)

        frame_kwargs = dict(detector_kwargs)
        if shared_background_model is not None:
            frame_kwargs["background_model"] = shared_background_model
        analysis = analyze_frame(path, progress=frame_progress, **frame_kwargs)
        return frame_index, analysis

    analyses: list[FrameAnalysis | None] = [None] * total
    work_items = tuple(enumerate(frame_paths, start=1))
    worker_count = min(int(sequence_workers), total)
    if worker_count == 1:
        completed_items = map(analyze_one, work_items)
        for completed, (frame_index, analysis) in enumerate(completed_items, start=1):
            analyses[frame_index - 1] = analysis
            if progress is not None:
                progress("frame", completed, total)
    else:
        # NumPy/SciPy 的卷积、分位数和极大值滤波会释放 GIL；按帧并行能
        # NumPy/SciPy 的卷积、分位数和极大值滤波会释放 GIL；默认 4 个
        # worker 是当前 4096² 数据的实测吞吐点。结果按 frame_index 写回，
        # 输出顺序稳定；内存紧张时调用方可显式降到 1/2/3。
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="rst19-frame") as executor:
            completed_items = executor.map(analyze_one, work_items)
            for completed, (frame_index, analysis) in enumerate(completed_items, start=1):
                analyses[frame_index - 1] = analysis
                if progress is not None:
                    progress("frame", completed, total)
    resolved_analyses = [analysis for analysis in analyses if analysis is not None]
    if len(resolved_analyses) != total:
        raise RuntimeError("sequence frame analysis did not produce all results")
    # 对疑似填充值/坏像素做序列级固定值审计。它只保留统计与坐标证据，
    # 不改变单帧检测的原图、掩膜或质量计数，避免把“固定异常码”误当成
    # 已经确认的物理坏像素。
    fixed_sentinel_audit = audit_fixed_sentinel(
        tuple(analysis.frame.data for analysis in resolved_analyses),
        sentinel_value=-1,
    )
    fixed_sentinel_impact_audit = audit_fixed_sentinel_impacts(
        resolved_analyses,
        fixed_sentinel_audit,
    )
    if progress is not None:
        progress("sentinel-audit", total, total)
    calculation_dtype = (
        "float32"
        if all(int(analysis.detection.parameters.get("use_float32", 0)) for analysis in resolved_analyses)
        else "float64"
    )
    if progress is not None:
        progress("registration", total, total)
    result = track_detections(
        [analysis.detection.quality_sources for analysis in resolved_analyses],
        frame_names=[str(path) for path in frame_paths],
        link_radius_px=link_radius_px,
        min_presence=min_presence,
        motion_min_displacement_px=motion_min_displacement_px,
        registration_radius_px=registration_radius_px,
        max_motion_fit_rms_px=max_motion_fit_rms_px,
        persistent_min_presence=persistent_min_presence,
    )
    # 线状层本来就需要注册时间中值参考；提前构建一次并复用到候选补检，
    # 避免为了补回单帧 Gaussian 漏掉的稳定源而再复制/计算一遍 4096² 参考图。
    temporal_reference = _registered_median_reference(
        resolved_analyses,
        result.cumulative_shifts,
    )
    temporal_candidate_frames, temporal_candidate_count = _temporal_reference_candidate_frames(
        resolved_analyses,
        result.cumulative_shifts,
        temporal_reference,
        psf_fwhm=float(detector_kwargs.get("psf_fwhm", 3.0)),
        min_candidate_snr=resolved_temporal_reference_min_snr,
        temporal_multiscale=bool(temporal_multiscale),
    )
    temporal_coadd_candidate_count = 0
    if temporal_proposal_mode in {"coadd", "both"}:
        if progress is not None:
            progress("temporal-coadd", 0, total)
        temporal_coadd_reference = _registered_coadd_reference(
            resolved_analyses,
            result.cumulative_shifts,
        )
        coadd_candidate_frames, temporal_coadd_candidate_count = _temporal_reference_candidate_frames(
            resolved_analyses,
            result.cumulative_shifts,
            temporal_coadd_reference,
            psf_fwhm=float(detector_kwargs.get("psf_fwhm", 3.0)),
            min_candidate_snr=resolved_temporal_reference_min_snr,
            temporal_multiscale=bool(temporal_multiscale),
        )
        del temporal_coadd_reference
        if temporal_proposal_mode == "coadd":
            temporal_candidate_frames = coadd_candidate_frames
        else:
            temporal_candidate_frames = tuple(
                np.vstack((median_rows, coadd_rows)).astype(np.float32, copy=False)
                if median_rows.size and coadd_rows.size
                else coadd_rows
                if coadd_rows.size
                else median_rows
                for median_rows, coadd_rows in zip(
                    temporal_candidate_frames,
                    coadd_candidate_frames,
                    strict=True,
                )
            )
        if progress is not None:
            progress("temporal-coadd", total, total)
    candidate_peak_frames: list[np.ndarray] = []
    for analysis, temporal_rows in zip(resolved_analyses, temporal_candidate_frames, strict=True):
        raw = np.asarray(analysis.detection.candidate_peaks, dtype=np.float32)
        if raw.size == 0:
            base_rows = np.empty((0, 3), dtype=np.float32)
        elif raw.ndim == 2 and raw.shape[1] >= 3:
            base_rows = raw[:, :3]
        else:
            # 旧/非快速结果没有临时候选峰证据；不要把异常数组猜成坐标。
            base_rows = np.empty((0, 3), dtype=np.float32)
        if base_rows.size:
            base_rows = np.column_stack(
                (
                    base_rows[:, :3],
                    np.zeros(base_rows.shape[0], dtype=np.float32),
                    np.full(base_rows.shape[0], np.nan, dtype=np.float32),
                )
            ).astype(np.float32, copy=False)
        if temporal_rows.size == 0:
            candidate_peak_frames.append(base_rows)
        elif base_rows.size == 0:
            candidate_peak_frames.append(temporal_rows)
        else:
            candidate_peak_frames.append(np.vstack((base_rows, temporal_rows)).astype(np.float32, copy=False))
    candidate_audit: dict[str, int] = {}
    candidate_consensus = _candidate_consensus_tracks(
        candidate_peak_frames,
        result.cumulative_shifts,
        result.tracks,
        frame_analyses=resolved_analyses,
        link_radius_px=link_radius_px,
        min_presence=result.min_presence,
        persistent_min_presence=int(
            getattr(
                result,
                "persistent_min_presence",
                persistent_min_presence
                if persistent_min_presence is not None
                else max(3, int(np.ceil(total * 0.5))),
            )
        ),
        motion_min_displacement_px=motion_min_displacement_px,
        max_motion_fit_rms_px=max_motion_fit_rms_px,
        min_candidate_snr=float(candidate_consensus_min_snr),
        temporal_candidate_min_snr=resolved_temporal_candidate_min_snr,
        temporal_reference_min_snr=resolved_temporal_reference_min_snr,
        temporal_min_psf_correlation=float(temporal_min_psf_correlation),
        line_artifact_frames=[
            analysis.detection.line_artifact_coordinates
            for analysis in resolved_analyses
        ],
        audit_sink=candidate_audit,
    )
    quality_tracks = tuple(result.tracks)
    candidate_tracks = tuple(
        SourceTrack(
            track_id=len(quality_tracks) + index,
            classification=track.classification,
            points=track.points,
            displacement_px=track.displacement_px,
            speed_px_per_frame=track.speed_px_per_frame,
            fit_rms_px=track.fit_rms_px,
            evidence_level=track.evidence_level,
        )
        for index, track in enumerate(candidate_consensus)
    )
    all_tracks = quality_tracks + candidate_tracks
    resolved_stack_min_presence = (
        int(getattr(result, "persistent_min_presence", max(3, int(np.ceil(total * 0.5)))))
        if stack_min_presence is None
        else int(stack_min_presence)
    )
    stack_faint_candidate_count = 0
    stack_faint_tracks: tuple[SourceTrack, ...] = ()
    if stack_faint_recovery and total >= 2:
        if progress is not None:
            progress("stack-faint", total, total)
        stack_reference = temporal_reference
        if stack_reference_mode == "coadd":
            stack_reference = _registered_coadd_reference(
                resolved_analyses,
                result.cumulative_shifts,
            )
        stack_audit: dict[str, int] = {}
        stack_faint_tracks = _stack_faint_tracks(
            resolved_analyses,
            result.cumulative_shifts,
            stack_reference,
            threshold_sigma=float(detector_kwargs.get("threshold_sigma", 4.0)),
            min_distance=int(detector_kwargs.get("min_distance", 3)),
            aperture_radius=int(detector_kwargs.get("aperture_radius", 4)),
            psf_fwhm=float(detector_kwargs.get("psf_fwhm", 3.0)),
            min_flux_snr=float(stack_min_flux_snr),
            min_psf_support_pixels=int(detector_kwargs.get("min_psf_support_pixels", 3)),
            proposal_mode=str(detector_kwargs.get("proposal_mode", "gaussian")),
            stack_frame_min_flux_snr=float(stack_frame_min_flux_snr),
            min_presence=resolved_stack_min_presence,
            link_radius_px=link_radius_px,
            existing_tracks=all_tracks,
            audit_sink=stack_audit,
        )
        stack_faint_candidate_count = int(stack_audit.get("stack_reference_quality_sources", 0))
        if stack_reference is not temporal_reference:
            del stack_reference
        if stack_faint_tracks:
            stack_faint_with_ids = tuple(
                SourceTrack(
                    track_id=len(all_tracks) + index,
                    classification=track.classification,
                    points=track.points,
                    displacement_px=track.displacement_px,
                    speed_px_per_frame=track.speed_px_per_frame,
                    fit_rms_px=track.fit_rms_px,
                    evidence_level=track.evidence_level,
                )
                for index, track in enumerate(stack_faint_tracks)
            )
            all_tracks = all_tracks + stack_faint_with_ids
    if progress is not None:
        progress("consensus", total, total)
    summaries = tuple(
        FrameSequenceSummary(
            frame_index=index,
            path=str(path),
            candidate_count=analysis.detection.candidate_count,
            returned_count=analysis.detection.returned_count,
            quality_count=analysis.detection.star_count,
            timestamp=str(analysis.frame.header.get("DATE-OBS")) if analysis.frame.header.get("DATE-OBS") is not None else None,
            exposure_ms=float(analysis.frame.header["EXPOSURE"]) if isinstance(analysis.frame.header.get("EXPOSURE"), (int, float)) else None,
            auxiliary=tuple(analysis.frame.auxiliary.as_dict().items()) if analysis.frame.auxiliary is not None else (),
            width_px=analysis.frame.width,
            height_px=analysis.frame.height,
        )
        for index, (path, analysis) in enumerate(zip(frame_paths, resolved_analyses, strict=True))
    )
    # 点轨迹关联已经消费了完整的 quality_sources。线状检测只需要原图、
    # 背景/噪声和屏蔽参数，不需要再次持有每帧数万条 Detection 对象；构造
    # 轻量视图后释放完整单帧分析，避免 GUI 后台在时间中值阶段继续占用峰值内存。
    motion_analyses = tuple(
        FrameAnalysis(
            frame=analysis.frame,
            detection=DetectionResult(
                image_shape=analysis.detection.image_shape,
                background=analysis.detection.background,
                noise=analysis.detection.noise,
                threshold=analysis.detection.threshold,
                candidate_count=analysis.detection.candidate_count,
                sources=(),
                parameters=dict(analysis.detection.parameters),
                quality_count=0,
            ),
            matching=None,
            faintest=None,
        )
        for analysis in resolved_analyses
    )
    model_modes = {
        str(analysis.detection.parameters.get("background_model_mode", "per_frame_local"))
        for analysis in resolved_analyses
    }
    del analyses, resolved_analyses
    motion_audits: list[MotionFrameAudit] = []
    def motion_detail_progress(value: float, label: str) -> None:
        if progress is not None:
            progress("motion-detail", int(round(float(value))), 100)

    motion_features, motion_threshold = detect_motion_features(
        motion_analyses,
        result.cumulative_shifts,
        psf_fwhm=float(detector_kwargs.get("psf_fwhm", 3.0)),
        temporal_reference=temporal_reference,
        progress=motion_detail_progress,
        audit_sink=motion_audits,
    )
    del motion_analyses
    del temporal_reference
    if progress is not None:
        progress("motion", total, total)
        progress("complete", total, total)
    if model_modes == {"shared_sequence_pilot"}:
        background_model_mode = "shared_sequence_pilot"
    elif "shared_sequence_pilot" in model_modes:
        background_model_mode = "shared_sequence_pilot_with_fallback"
    else:
        background_model_mode = "per_frame_local"
    return SequenceResult(
        frames=summaries,
        cumulative_shifts=result.cumulative_shifts,
        tracks=all_tracks,
        link_radius_px=result.link_radius_px,
        min_presence=result.min_presence,
        motion_min_displacement_px=result.motion_min_displacement_px,
        max_motion_fit_rms_px=result.max_motion_fit_rms_px,
        persistent_min_presence=int(
            getattr(
                result,
                "persistent_min_presence",
                persistent_min_presence
                if persistent_min_presence is not None
                else max(3, int(np.ceil(total * 0.5))),
            )
        ),
        motion_features=motion_features,
        motion_residual_threshold_adu=motion_threshold,
        motion_reference_mode=(
            "per_frame_background"
            if len(frame_paths) < 2
            else "temporal_median_small_shift"
            if max(float(np.max(np.abs(result.cumulative_shifts))), 0.0) <= 0.75
            else "registered_median"
        ),
        motion_frame_audits=tuple(motion_audits),
        source_working_limit=(
            int(detector_kwargs["max_sources"])
            if detector_kwargs.get("max_sources") is not None
            else None
        ),
        calculation_dtype=calculation_dtype,
        background_sample_limit=(
            int(detector_kwargs["background_sample_limit"])
            if detector_kwargs.get("background_sample_limit") is not None
            else None
        ),
        fast_sequence=bool(detector_kwargs.get("fast_sequence", False)),
        background_model_mode=background_model_mode,
        temporal_reference_candidate_count=int(temporal_candidate_count),
        temporal_coadd_candidate_count=int(temporal_coadd_candidate_count),
        temporal_proposal_mode=temporal_proposal_mode,
        temporal_candidate_min_snr=float(resolved_temporal_candidate_min_snr),
        temporal_reference_min_snr=float(resolved_temporal_reference_min_snr),
        temporal_multiscale=bool(temporal_multiscale),
        temporal_min_psf_correlation=float(temporal_min_psf_correlation),
        candidate_consensus_audit=tuple(sorted(candidate_audit.items())),
        stack_faint_candidate_count=stack_faint_candidate_count,
        stack_reference_mode=stack_reference_mode,
        stack_threshold_sigma=float(stack_threshold_sigma),
        stack_min_flux_snr=float(stack_min_flux_snr),
        stack_frame_min_flux_snr=float(stack_frame_min_flux_snr),
        stack_min_presence=resolved_stack_min_presence,
        fixed_sentinel_audit=fixed_sentinel_audit,
        fixed_sentinel_impact_audit=fixed_sentinel_impact_audit,
    )
