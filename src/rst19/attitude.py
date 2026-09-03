"""比赛 FITS 辅助姿态字段的可复核审计。

格式说明把相机物理指向定义为卫星本体坐标系的 ``-Y`` 轴，并把四元数
写成矢量在前、标量在后的 ``(q1, q2, q3, q4)``。本模块只验证辅助字段
之间的几何自洽性，不把它扩展成完整的像素到天球 WCS。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .fits import read_fits
from .models import AuxiliaryData


def quaternion_to_active_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """将 ``(x, y, z, w)`` 四元数转换为标准 Hamilton 旋转矩阵。

    输入的四元数会归一化，但会拒绝长度错误、非有限或零范数输入。格式
    说明把 q 描述为 J2000 到本体的坐标变换；在本数据中，按此矩阵将
    本体 ``-Y`` 轴变换到参考系后，恰好与辅助 ``ra/dec`` 一致。因此下游
    使用时应把这里的结果称为“格式约定下的光轴一致性审计”，而不是
    未经相机坐标定义验证的物理 active/passive 结论。
    """

    values = np.asarray(tuple(quaternion), dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = values / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def vector_to_radec_deg(vector: Sequence[float]) -> tuple[float, float]:
    """将有限三维方向向量转换为 ``(RA, Dec)``，单位为度。"""

    values = np.asarray(tuple(vector), dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("direction vector must contain three finite values")
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("direction vector norm must be positive")
    x, y, z = values / norm
    return math.degrees(math.atan2(float(y), float(x))) % 360.0, math.degrees(math.asin(float(z)))


def auxiliary_boresight_radec(
    auxiliary: AuxiliaryData,
    *,
    camera_body_axis: Sequence[float] = (0.0, -1.0, 0.0),
) -> tuple[float, float]:
    """由格式约定下的姿态和相机本体轴推导光轴 ``(RA, Dec)``。

    默认使用格式说明中的本体 ``-Y`` 轴。返回值只代表光轴中心先验；它
    不包含像元尺度、图像 x/y 轴方向、旋转、畸变或曝光期间姿态变化。
    """

    axis = np.asarray(tuple(camera_body_axis), dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("camera_body_axis must contain three finite values")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= np.finfo(np.float64).eps:
        raise ValueError("camera_body_axis norm must be positive")
    values = auxiliary.as_dict()
    quaternion = [values[name] for name in ("q1", "q2", "q3", "q4")]
    return vector_to_radec_deg(quaternion_to_active_matrix(quaternion) @ (axis / axis_norm))


def _wrap_ra_delta_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


@dataclass(frozen=True, slots=True)
class AuxiliaryBoresightAudit:
    """单帧辅助姿态、光轴一致性和头部交叉校验结果。"""

    frame_index: int
    path: str
    timestamp: str | None
    auxiliary_ra_deg: float | None
    auxiliary_dec_deg: float | None
    derived_ra_deg: float | None
    derived_dec_deg: float | None
    delta_ra_arcsec: float | None
    delta_dec_arcsec: float | None
    angular_residual_arcsec: float | None
    quaternion_norm: float | None
    p_az_deg: float | None
    p_el_deg: float | None
    roll_deg: float | None
    pitch_deg: float | None
    yaw_deg: float | None
    header_azimuth_deg: float | None
    header_elevation_deg: float | None
    header_pointing_match: bool | None
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_number": self.frame_index + 1,
            "path": self.path,
            "timestamp": self.timestamp,
            "auxiliary_ra_deg": self.auxiliary_ra_deg,
            "auxiliary_dec_deg": self.auxiliary_dec_deg,
            "derived_ra_deg": self.derived_ra_deg,
            "derived_dec_deg": self.derived_dec_deg,
            "delta_ra_arcsec": self.delta_ra_arcsec,
            "delta_dec_arcsec": self.delta_dec_arcsec,
            "angular_residual_arcsec": self.angular_residual_arcsec,
            "quaternion_norm": self.quaternion_norm,
            "p_az_deg": self.p_az_deg,
            "p_el_deg": self.p_el_deg,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "header_azimuth_deg": self.header_azimuth_deg,
            "header_elevation_deg": self.header_elevation_deg,
            "header_pointing_match": self.header_pointing_match,
            "status": self.status,
        }


def audit_auxiliary_boresight(
    paths: Iterable[str | Path],
    *,
    camera_body_axis: Sequence[float] = (0.0, -1.0, 0.0),
) -> tuple[AuxiliaryBoresightAudit, ...]:
    """逐文件审计辅助 ``ra/dec``、四元数和 FITS 头指向字段。"""

    rows: list[AuxiliaryBoresightAudit] = []
    for frame_index, raw_path in enumerate(paths):
        path = Path(raw_path)
        frame = read_fits(path)
        auxiliary = frame.auxiliary
        header_azimuth = _optional_float(frame.header.get("AZIMUTH"))
        header_elevation = _optional_float(frame.header.get("ELEVATIO"))
        if auxiliary is None:
            rows.append(
                AuxiliaryBoresightAudit(
                    frame_index=frame_index,
                    path=str(path.resolve()),
                    timestamp=str(frame.header.get("DATE-OBS")) if frame.header.get("DATE-OBS") is not None else None,
                    auxiliary_ra_deg=None,
                    auxiliary_dec_deg=None,
                    derived_ra_deg=None,
                    derived_dec_deg=None,
                    delta_ra_arcsec=None,
                    delta_dec_arcsec=None,
                    angular_residual_arcsec=None,
                    quaternion_norm=None,
                    p_az_deg=None,
                    p_el_deg=None,
                    roll_deg=None,
                    pitch_deg=None,
                    yaw_deg=None,
                    header_azimuth_deg=header_azimuth,
                    header_elevation_deg=header_elevation,
                    header_pointing_match=None,
                    status="missing_auxiliary",
                )
            )
            continue

        values = auxiliary.as_dict()
        quaternion = np.asarray([values[name] for name in ("q1", "q2", "q3", "q4")], dtype=np.float64)
        quaternion_norm = float(np.linalg.norm(quaternion))
        derived_ra, derived_dec = auxiliary_boresight_radec(auxiliary, camera_body_axis=camera_body_axis)
        delta_ra_arcsec = _wrap_ra_delta_deg(derived_ra - auxiliary.ra_deg) * 3600.0
        delta_dec_arcsec = (derived_dec - auxiliary.dec_deg) * 3600.0
        angular_residual = math.hypot(
            delta_ra_arcsec * math.cos(math.radians(auxiliary.dec_deg)),
            delta_dec_arcsec,
        )
        p_az = _optional_float(values.get("p_az"))
        p_el = _optional_float(values.get("p_el"))
        header_pointing_match = (
            p_az is not None
            and p_el is not None
            and header_azimuth is not None
            and header_elevation is not None
            and math.isclose(p_az, header_azimuth, abs_tol=1e-9)
            and math.isclose(p_el, header_elevation, abs_tol=1e-9)
        )
        rows.append(
            AuxiliaryBoresightAudit(
                frame_index=frame_index,
                path=str(path.resolve()),
                timestamp=str(frame.header.get("DATE-OBS")) if frame.header.get("DATE-OBS") is not None else None,
                auxiliary_ra_deg=float(auxiliary.ra_deg),
                auxiliary_dec_deg=float(auxiliary.dec_deg),
                derived_ra_deg=float(derived_ra),
                derived_dec_deg=float(derived_dec),
                delta_ra_arcsec=float(delta_ra_arcsec),
                delta_dec_arcsec=float(delta_dec_arcsec),
                angular_residual_arcsec=float(angular_residual),
                quaternion_norm=quaternion_norm,
                p_az_deg=p_az,
                p_el_deg=p_el,
                roll_deg=_optional_float(values.get("roll")),
                pitch_deg=_optional_float(values.get("pitch")),
                yaw_deg=_optional_float(values.get("yaw")),
                header_azimuth_deg=header_azimuth,
                header_elevation_deg=header_elevation,
                header_pointing_match=bool(header_pointing_match),
                status="ok",
            )
        )
    return tuple(rows)


def _range(rows: Sequence[AuxiliaryBoresightAudit], field: str) -> dict[str, float | None]:
    values = [float(getattr(row, field)) for row in rows if getattr(row, field) is not None]
    if not values:
        return {"min": None, "max": None, "range": None}
    return {"min": min(values), "max": max(values), "range": max(values) - min(values)}


def summarize_auxiliary_boresight(
    rows: Sequence[AuxiliaryBoresightAudit],
    *,
    camera_body_axis: Sequence[float] = (0.0, -1.0, 0.0),
) -> dict[str, object]:
    """汇总光轴一致性、姿态范围和头部交叉校验。"""

    residuals = [row.angular_residual_arcsec for row in rows if row.angular_residual_arcsec is not None]
    norms = [row.quaternion_norm for row in rows if row.quaternion_norm is not None]
    pointing_rows = [row for row in rows if row.header_pointing_match is not None]
    ra_range = _range(rows, "auxiliary_ra_deg")
    dec_range = _range(rows, "auxiliary_dec_deg")
    return {
        "frame_count": len(rows),
        "ok_frame_count": sum(row.status == "ok" for row in rows),
        "missing_auxiliary_count": sum(row.status == "missing_auxiliary" for row in rows),
        "camera_body_axis": [float(value) for value in camera_body_axis],
        "quaternion_norm": {
            "min": min(norms) if norms else None,
            "max": max(norms) if norms else None,
            "max_abs_deviation_from_one": max((abs(value - 1.0) for value in norms), default=None),
        },
        "boresight_residual_arcsec": {
            "max": max(residuals) if residuals else None,
            "rms": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
        },
        "auxiliary_ra_deg": ra_range,
        "auxiliary_dec_deg": dec_range,
        "auxiliary_ra_range_arcsec": float(ra_range["range"] * 3600.0) if ra_range["range"] is not None else None,
        "auxiliary_dec_range_arcsec": float(dec_range["range"] * 3600.0) if dec_range["range"] is not None else None,
        "attitude_ranges": {
            field: _range(rows, field) for field in ("roll_deg", "pitch_deg", "yaw_deg")
        },
        "header_pointing_match_count": sum(bool(row.header_pointing_match) for row in pointing_rows),
        "header_pointing_checked_count": len(pointing_rows),
        "note": (
            "四元数-光轴一致性只验证辅助字段之间的约定；它不提供像元角尺度、图像轴方向、"
            "畸变或曝光期间姿态变化，因此不能单独构成完整 WCS。"
        ),
    }


def write_auxiliary_boresight_artifacts(
    rows: Sequence[AuxiliaryBoresightAudit],
    output_dir: str | Path,
    *,
    camera_body_axis: Sequence[float] = (0.0, -1.0, 0.0),
) -> Path:
    """写出辅助姿态审计 CSV/JSON，并返回输出目录。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    serialized = [row.as_dict() for row in rows]
    fieldnames = list(serialized[0].keys()) if serialized else ["frame_index", "status"]
    with (output / "auxiliary_boresight_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialized)
    payload = {
        "schema_version": 1,
        "summary": summarize_auxiliary_boresight(rows, camera_body_axis=camera_body_axis),
        "frames": serialized,
    }
    (output / "auxiliary_boresight_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
