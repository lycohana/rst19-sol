"""把 15 帧科学结果转换成面向用户的谨慎简报。

这里不重新检测、不产生概率，也不把 ``moving`` 自动升级成真实天体。
它只是把 ``SequenceResult``（以及可选的 ``MosaicResult``）中的证据，
按“先说发现，再说边界”的顺序组织成 UI 和论文都能复用的文本。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


@dataclass(frozen=True, slots=True)
class BriefTarget:
    """一条可以在简报中直接解释的目标/候选。"""

    track_id: int
    kind: str
    classification: str
    first_frame: int
    last_frame: int
    presence: int
    duration_s: float | None
    speed_px_per_s: float | None
    direction_deg: float | None
    fit_rms_px: float | None
    evidence_text: str

    @property
    def frame_text(self) -> str:
        return (
            f"F{self.first_frame:02d}"
            if self.first_frame == self.last_frame
            else f"F{self.first_frame:02d}–F{self.last_frame:02d}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "kind": self.kind,
            "classification": self.classification,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "presence": self.presence,
            "duration_s": self.duration_s,
            "speed_px_per_s": self.speed_px_per_s,
            "direction_deg": self.direction_deg,
            "fit_rms_px": self.fit_rms_px,
            "evidence_text": self.evidence_text,
        }


@dataclass(frozen=True, slots=True)
class SequenceBrief:
    """15 帧简报的结构化结果。"""

    headline: str
    findings: tuple[str, ...]
    boundaries: tuple[str, ...]
    targets: tuple[BriefTarget, ...]
    mosaic_note: str
    evidence_refs: tuple[str, ...]
    frame_count: int
    duration_s: float | None
    interval_median_s: float | None
    registration_max_shift_px: float

    def text(self) -> str:
        lines = [self.headline]
        lines.extend(f"• {line}" for line in self.findings)
        lines.append("结论边界：")
        lines.extend(f"• {line}" for line in self.boundaries)
        lines.append(f"• {self.mosaic_note}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "headline": self.headline,
            "findings": list(self.findings),
            "boundaries": list(self.boundaries),
            "targets": [target.as_dict() for target in self.targets],
            "mosaic_note": self.mosaic_note,
            "evidence_refs": list(self.evidence_refs),
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "interval_median_s": self.interval_median_s,
            "registration_max_shift_px": self.registration_max_shift_px,
        }


def _fit_motion(
    points: Sequence[Any],
    frames: Sequence[Any],
    fallback_speed_per_frame: float,
) -> tuple[float | None, float | None, float | None]:
    """返回图像平面速度、方向和观测跨度；不报告真实物理速度。"""

    if len(points) < 2:
        return None, None, 0.0
    timestamps = [_timestamp(getattr(frame, "timestamp", None)) for frame in frames]
    first_index = int(getattr(points[0], "frame_index", 0))
    first_time = timestamps[first_index] if 0 <= first_index < len(timestamps) else None
    elapsed: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        index = int(getattr(point, "frame_index", 0))
        current = timestamps[index] if 0 <= index < len(timestamps) else None
        if first_time is None or current is None:
            continue
        elapsed.append((current - first_time).total_seconds())
        xs.append(float(getattr(point, "aligned_x")))
        ys.append(float(getattr(point, "aligned_y")))
    if len(elapsed) >= 2 and elapsed[-1] > elapsed[0]:
        mean_t = sum(elapsed) / len(elapsed)
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((value - mean_t) ** 2 for value in elapsed)
        if denominator > 0:
            vx = sum((time - mean_t) * (x - mean_x) for time, x in zip(elapsed, xs)) / denominator
            vy = sum((time - mean_t) * (y - mean_y) for time, y in zip(elapsed, ys)) / denominator
            return math.hypot(vx, vy), math.degrees(math.atan2(vy, vx)), elapsed[-1] - elapsed[0]
    first = points[0]
    last = points[-1]
    dx = float(getattr(last, "aligned_x")) - float(getattr(first, "aligned_x"))
    dy = float(getattr(last, "aligned_y")) - float(getattr(first, "aligned_y"))
    duration = float(max(1, int(getattr(last, "frame_index", 0)) - int(getattr(first, "frame_index", 0))))
    return (
        # With no usable DATE-OBS, keep the native frame-based speed instead
        # of pretending that a frame index is a number of seconds.
        float(fallback_speed_per_frame),
        math.degrees(math.atan2(dy, dx)),
        duration,
    )


def _target_from_track(track: Any, result: Any, *, kind: str) -> BriefTarget | None:
    points = tuple(getattr(track, "points", ()))
    if not points:
        return None
    first_frame = int(getattr(points[0], "frame_index", 0)) + 1
    last_frame = int(getattr(points[-1], "frame_index", 0)) + 1
    speed, direction, duration = _fit_motion(
        points,
        getattr(result, "frames", ()),
        float(getattr(track, "speed_px_per_frame", 0.0)),
    )
    classification = str(getattr(track, "classification", "candidate"))
    evidence_level = str(getattr(track, "evidence_level", "quality"))
    if classification == "moving":
        evidence_text = "连续帧位置/形状通过当前 moving 门槛，仍待星表或人工真值确认"
    elif len(points) == 1:
        evidence_text = "只在单帧出现，不能单独估计运行速度，也没有与其它轨迹强行合并"
    else:
        evidence_text = "多帧出现但未达到严格 moving 门槛，保留为待复核候选"
    return BriefTarget(
        track_id=int(getattr(track, "track_id", 0)),
        kind=kind,
        classification=classification,
        first_frame=first_frame,
        last_frame=last_frame,
        presence=len(points),
        duration_s=duration,
        speed_px_per_s=speed,
        direction_deg=direction,
        fit_rms_px=(float(getattr(track, "fit_rms_px")) if getattr(track, "fit_rms_px", None) is not None else None),
        evidence_text=evidence_text,
    )


def build_sequence_brief(result: Any, mosaic: Any | None = None) -> SequenceBrief:
    """从当前序列结果生成一份短、谨慎、可复核的中文简报。"""

    frames = tuple(getattr(result, "frames", ()))
    datetimes = [_timestamp(getattr(frame, "timestamp", None)) for frame in frames]
    intervals = [
        (right - left).total_seconds()
        for left, right in zip(datetimes, datetimes[1:])
        if left is not None and right is not None and (right - left).total_seconds() > 0
    ]
    duration = (
        (datetimes[-1] - datetimes[0]).total_seconds()
        if len(datetimes) >= 2 and datetimes[0] is not None and datetimes[-1] is not None
        else None
    )
    interval_median = _median(intervals)
    max_shift = max(
        (math.hypot(float(shift[0]), float(shift[1])) for shift in getattr(result, "cumulative_shifts", ())),
        default=0.0,
    )

    targets: list[BriefTarget] = []
    for track in getattr(result, "motion_features", ()):
        target = _target_from_track(track, result, kind="line")
        if target is not None:
            targets.append(target)
    for track in getattr(result, "tracks", ()):
        if str(getattr(track, "classification", "")) != "moving":
            continue
        target = _target_from_track(track, result, kind="point")
        if target is not None:
            targets.append(target)
    targets.sort(key=lambda target: (target.classification != "moving", -target.presence, target.track_id))

    moving_targets = [target for target in targets if target.classification == "moving"]
    pending_targets = [target for target in targets if target.classification != "moving"]
    if moving_targets:
        headline = f"先说结论：15 帧中形成 {len(moving_targets)} 条连续运动候选。"
    elif pending_targets:
        headline = "先说结论：当前门槛下没有形成严格 moving，只有待复核候选。"
    else:
        headline = "先说结论：当前门槛下没有形成运动候选。"

    findings: list[str] = []
    timing = f"{len(frames)} 帧连续观测"
    if duration is not None:
        timing += f"，总时长约 {duration:.3f} s"
    if interval_median is not None:
        timing += f"，相邻帧约每 {interval_median:.3f} s 一张"
    findings.append(timing + "。")
    findings.append(
        f"固定星场配准后的最大共同位移约 {max_shift:.3f} px，说明本次序列的共同星场变化很小，适合在配准坐标里找异常运动。"
    )
    for target in targets[:3]:
        if target.classification == "moving" and target.speed_px_per_s is not None and target.direction_deg is not None:
            fit = f"，拟合 RMS {target.fit_rms_px:.2f} px" if target.fit_rms_px is not None else ""
            findings.append(
                f"{target.kind == 'line' and '线状' or '点状'}目标 {target.track_id:04d} 出现在 {target.frame_text}（{target.presence} 帧），"
                f"图像平面速度约 {target.speed_px_per_s:.2f} px/s，方向约 {target.direction_deg:.1f}°{fit}。"
            )
        elif target.first_frame == target.last_frame:
            findings.append(
                f"{target.kind == 'line' and '线状' or '点状'}候选 {target.track_id:04d} 只在 {target.frame_text} 出现；没有把它和其它时间段的轨迹硬合并。"
            )
        else:
            findings.append(
                f"候选 {target.track_id:04d} 出现在 {target.frame_text}，但当前证据不足以升级为严格 moving。"
            )
    if not targets:
        findings.append("没有生成可展示的点状或线状运动轨迹；这表示当前算法层未找到，不等于物理上绝对没有目标。")
    point_moving = [target for target in moving_targets if target.kind == "point"]
    if len(point_moving) >= 2:
        findings.append(
            "多条点状候选的方向/速度相近，先按同向运动候选报告；还要排除共同配准误差或错误关联，不能直接按数量认定为多个真实天体。"
        )

    boundaries = [
        "moving 是通过当前图像几何、跨帧关联和拟合门槛的算法候选，不等于已经完成星表身份或人工真值确认。",
        "速度和方向目前是图像平面 px/s 与图像坐标角度；没有像元角尺度和完整 WCS 时，不能写成真实角速度或轨道速度。",
        f"严格静态 {int(getattr(result, 'stable_source_count', 0)):,}、持续候选 {int(getattr(result, 'persistent_source_count', 0)):,} 等是证据层计数，不是物理恒星总数。",
    ]
    if pending_targets:
        boundaries.append("单帧长线只能说明本帧有形状异常，必须结合其它帧、差分图和原始像素复核。")

    if mosaic is None:
        mosaic_note = "注册合成图尚未生成；生成后只用于查看联合覆盖、重叠和像素稳定性。"
    else:
        shape = getattr(mosaic, "output_shape", (0, 0))
        mosaic_note = (
            f"合成图为 {int(shape[1])}×{int(shape[0])} px，覆盖 {int(getattr(mosaic, 'covered_pixel_count', 0)):,} 个像素，"
            f"重叠 {int(getattr(mosaic, 'overlap_pixel_count', 0)):,} 个像素；它说明多帧如何融合，不会自动增加恒星数或确认运动目标。"
        )
    return SequenceBrief(
        headline=headline,
        findings=tuple(findings),
        boundaries=tuple(boundaries),
        targets=tuple(targets),
        mosaic_note=mosaic_note,
        evidence_refs=("innovation_report.json", "motion_evidence.csv", "motion_frame_audit.csv", "registered_mosaic_coverage.npy"),
        frame_count=len(frames),
        duration_s=duration,
        interval_median_s=interval_median,
        registration_max_shift_px=max_shift,
    )
