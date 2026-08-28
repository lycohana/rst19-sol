"""15 帧星点稳定性和运动目标关联。

这里不把帧间最近邻直接叫作运动目标。先用高质量、高 SNR 源估计每一帧
相对于上一帧的全局平移，再在配准坐标中关联检测源；只有具有足够帧数、
并且相对于静态背景呈现可重复位移的轨迹，才标为 ``moving``。这是一种
适用于当前数据的可解释基线，不替代带旋转/畸变项的完整图像配准。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .detection import Detection
from .pipeline import FrameAnalysis, analyze_frame


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

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "aligned_x": self.aligned_x,
            "aligned_y": self.aligned_y,
            "flux_snr": self.flux_snr,
        }


@dataclass(frozen=True, slots=True)
class SourceTrack:
    """一条跨帧轨迹及其静态/运动判定。"""

    track_id: int
    classification: str
    points: tuple[TrackPoint, ...]
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
class FrameSequenceSummary:
    frame_index: int
    path: str
    candidate_count: int
    returned_count: int
    quality_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "quality_count": self.quality_count,
        }


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """15 帧检测和关联的汇总结果。"""

    frames: tuple[FrameSequenceSummary, ...]
    cumulative_shifts: tuple[tuple[float, float], ...]
    tracks: tuple[SourceTrack, ...]
    link_radius_px: float
    min_presence: int
    motion_min_displacement_px: float
    max_motion_fit_rms_px: float

    @property
    def stable_source_count(self) -> int:
        return sum(track.classification == "static" for track in self.tracks)

    @property
    def moving_track_count(self) -> int:
        return sum(track.classification == "moving" for track in self.tracks)

    @property
    def transient_track_count(self) -> int:
        return sum(track.classification == "transient" for track in self.tracks)

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": len(self.frames),
            "stable_source_count": self.stable_source_count,
            "moving_track_count": self.moving_track_count,
            "transient_track_count": self.transient_track_count,
            "link_radius_px": self.link_radius_px,
            "min_presence": self.min_presence,
            "motion_min_displacement_px": self.motion_min_displacement_px,
            "max_motion_fit_rms_px": self.max_motion_fit_rms_px,
            "frames": [frame.as_dict() for frame in self.frames],
            "cumulative_shifts": [list(shift) for shift in self.cumulative_shifts],
            "tracks": [track.as_dict() for track in self.tracks],
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
    displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    speed = float(np.hypot(x_slope, y_slope))
    return displacement, speed, rms


def track_detections(
    frame_sources: Sequence[Sequence[Detection]],
    *,
    frame_names: Sequence[str] | None = None,
    link_radius_px: float = 4.0,
    min_presence: int | None = None,
    motion_min_displacement_px: float = 2.0,
    registration_radius_px: float = 8.0,
    max_motion_fit_rms_px: float = 0.75,
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
            tracks.append([TrackPoint(0, source.detection_id, source.x, source.y, x, y, source.flux_snr)])
    for frame_index in range(1, frame_count):
        current = aligned_frames[frame_index]
        if not current:
            continue
        points = np.array([(x, y) for _, x, y in current], dtype=np.float64)
        tree = cKDTree(points)
        possible: list[tuple[float, int, int]] = []
        for track_index, history in enumerate(tracks):
            last = history[-1]
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
                TrackPoint(frame_index, source.detection_id, source.x, source.y, x, y, source.flux_snr)
            )
            assigned_tracks.add(track_index)
            assigned_sources.add(source_index)
        for source_index, (source, x, y) in enumerate(current):
            if source_index not in assigned_sources:
                tracks.append([TrackPoint(frame_index, source.detection_id, source.x, source.y, x, y, source.flux_snr)])

    rendered_tracks: list[SourceTrack] = []
    for track_id, points in enumerate(tracks):
        displacement, speed, fit_rms = _fit_track(points)
        if len(points) < required_presence:
            classification = "transient"
        elif displacement >= motion_min_displacement_px and fit_rms is not None and fit_rms <= max_motion_fit_rms_px:
            classification = "moving"
        else:
            classification = "static"
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
    )


def analyze_sequence(
    paths: Iterable[str | Path],
    *,
    link_radius_px: float = 4.0,
    min_presence: int | None = None,
    motion_min_displacement_px: float = 2.0,
    registration_radius_px: float = 8.0,
    max_motion_fit_rms_px: float = 0.75,
    **detector_kwargs: object,
) -> SequenceResult:
    """读取并检测一组 FITS，再关联质量源；参数与 ``analyze_frame`` 对齐。"""

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    analyses: list[FrameAnalysis] = [analyze_frame(path, **detector_kwargs) for path in frame_paths]
    result = track_detections(
        [analysis.detection.quality_sources for analysis in analyses],
        frame_names=[str(path) for path in frame_paths],
        link_radius_px=link_radius_px,
        min_presence=min_presence,
        motion_min_displacement_px=motion_min_displacement_px,
        registration_radius_px=registration_radius_px,
        max_motion_fit_rms_px=max_motion_fit_rms_px,
    )
    summaries = tuple(
        FrameSequenceSummary(
            frame_index=index,
            path=str(path),
            candidate_count=analysis.detection.candidate_count,
            returned_count=analysis.detection.returned_count,
            quality_count=analysis.detection.star_count,
        )
        for index, (path, analysis) in enumerate(zip(frame_paths, analyses, strict=True))
    )
    return SequenceResult(
        frames=summaries,
        cumulative_shifts=result.cumulative_shifts,
        tracks=result.tracks,
        link_radius_px=result.link_radius_px,
        min_presence=result.min_presence,
        motion_min_displacement_px=result.motion_min_displacement_px,
        max_motion_fit_rms_px=result.max_motion_fit_rms_px,
    )
