"""把 15 帧序列整理为可审计的创新分析证据。

这里输出的是观测量和像素平面派生量。没有 WCS 或像元尺度时，不把图像
平面速度解释成角速度、天体真实速度或轨道参数；辅助遥测则同时保留原始
字段、格式说明中的单位线索和位置差分自洽检查。报告既可以供 Tkinter
界面读取，也可以导出 CSV/JSON 作为答辩材料的表格基础。
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .fits import auxiliary_mask, exposure_milliseconds, read_fits


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        timestamp = datetime.fromisoformat(text)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        return timestamp
    except ValueError:
        return None


def _timestamp_text(value: object) -> str | None:
    timestamp = _parse_timestamp(value)
    return timestamp.isoformat() if timestamp is not None else str(value) if value is not None else None


def _resolve_frame_path(raw_path: str | Path, sequence_json: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.is_file():
        return path
    candidates = (
        Path.cwd() / path,
        sequence_json.parent / path,
        sequence_json.parent.parent / path,
        sequence_json.parent.parent.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # 保留一个有意义的错误路径，而不是在报告中静默丢失一帧。
    return (Path.cwd() / path).resolve()


def _robust_image_stats(data: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(values) & ~auxiliary_mask(values.shape)
    finite = values[valid]
    if finite.size == 0:
        return {
            "valid_pixels": 0,
            "background_median_adu": None,
            "background_mad_adu": None,
            "background_rms_adu": None,
            "p01_adu": None,
            "p50_adu": None,
            "p99_adu": None,
            "p998_adu": None,
            "raw_max_adu": None,
            "zero_fraction": None,
        }
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return {
        "valid_pixels": int(finite.size),
        "background_median_adu": median,
        "background_mad_adu": mad,
        "background_rms_adu": float(1.4826 * mad),
        "p01_adu": float(np.percentile(finite, 1.0)),
        "p50_adu": float(np.percentile(finite, 50.0)),
        "p99_adu": float(np.percentile(finite, 99.0)),
        "p998_adu": float(np.percentile(finite, 99.8)),
        "raw_max_adu": float(np.max(finite)),
        "zero_fraction": float(np.count_nonzero(finite == 0.0) / finite.size),
    }


def _vector_metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    """从辅助字段计算向量模长和方向，保留 ``*_raw`` 兼容字段。"""

    metrics: dict[str, float | None] = {}
    for prefix in ("j2000", "wgs84"):
        position = [_as_float(row.get(f"{prefix}_{axis}")) for axis in ("x", "y", "z")]
        velocity = [_as_float(row.get(f"{prefix}_{axis}v")) for axis in ("x", "y", "z")]
        if all(value is not None for value in position):
            vector = np.asarray(position, dtype=np.float64)
            metrics[f"{prefix}_position_norm_raw"] = float(np.linalg.norm(vector))
            metrics[f"{prefix}_position_azimuth_deg"] = float(np.degrees(np.arctan2(vector[1], vector[0])))
            metrics[f"{prefix}_position_elevation_deg"] = float(np.degrees(np.arctan2(vector[2], np.hypot(vector[0], vector[1]))))
        else:
            metrics[f"{prefix}_position_norm_raw"] = None
            metrics[f"{prefix}_position_azimuth_deg"] = None
            metrics[f"{prefix}_position_elevation_deg"] = None
        if all(value is not None for value in velocity):
            vector = np.asarray(velocity, dtype=np.float64)
            metrics[f"{prefix}_velocity_norm_raw"] = float(np.linalg.norm(vector))
            metrics[f"{prefix}_velocity_azimuth_deg"] = float(np.degrees(np.arctan2(vector[1], vector[0])))
            metrics[f"{prefix}_velocity_elevation_deg"] = float(np.degrees(np.arctan2(vector[2], np.hypot(vector[0], vector[1]))))
        else:
            metrics[f"{prefix}_velocity_norm_raw"] = None
            metrics[f"{prefix}_velocity_azimuth_deg"] = None
            metrics[f"{prefix}_velocity_elevation_deg"] = None
    return metrics


def _relative_rates(rows: Sequence[Mapping[str, Any]], fields: Iterable[str]) -> dict[str, dict[str, float | None]]:
    timestamps = [_parse_timestamp(row.get("timestamp")) for row in rows]
    result: dict[str, dict[str, float | None]] = {}
    for field in fields:
        values = [_as_float(row.get(field)) for row in rows]
        rates: list[float] = []
        for index in range(1, len(rows)):
            if timestamps[index - 1] is None or timestamps[index] is None:
                continue
            dt = (timestamps[index] - timestamps[index - 1]).total_seconds()
            if dt <= 0 or values[index - 1] is None or values[index] is None:
                continue
            rates.append((values[index] - values[index - 1]) / dt)
        result[field] = {
            "min_per_s": float(min(rates)) if rates else None,
            "max_per_s": float(max(rates)) if rates else None,
            "median_per_s": float(np.median(rates)) if rates else None,
        }
    return result


def _telemetry_consistency(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    """检查位置差分速度与辅助速度向量是否数值自洽。

    位置字段在格式说明中标为米，因此由位置和 ``DATE-OBS`` 差分得到的
    速度量纲必然是 m/s。速度字段的表格原文却写成 ``m``；本函数只做
    “按 m/s 假设比较”的可复核计算，不替主办方修正文档，并把端点的一侧
    差分和中间帧的中心差分分开汇总。
    """

    position_keys = tuple(f"{prefix}_{axis}" for axis in ("x", "y", "z"))
    velocity_keys = tuple(f"{prefix}_{axis}v" for axis in ("x", "y", "z"))
    timestamps = [_parse_timestamp(row.get("timestamp")) for row in rows]
    positions = [
        np.asarray([_as_float(row.get(key)) for key in position_keys], dtype=np.float64)
        for row in rows
    ]
    velocities = [
        np.asarray([_as_float(row.get(key)) for key in velocity_keys], dtype=np.float64)
        for row in rows
    ]
    rate_norms: list[float] = []
    provided_norms: list[float] = []
    residual_norms: list[float] = []
    relative_errors: list[float] = []
    interior_relative_errors: list[float] = []

    row_rate_key = f"{prefix}_position_rate_norm_m_per_s"
    row_residual_key = f"{prefix}_velocity_consistency_residual_norm_assuming_m_per_s"
    row_error_key = f"{prefix}_velocity_consistency_relative_error_assuming_m_per_s"
    for row in rows:
        # Mapping 对象通常是 dict；无法写入时仍允许调用方只取汇总结果。
        for key in (row_rate_key, row_residual_key, row_error_key):
            try:
                row[key] = None  # type: ignore[index]
            except TypeError:
                pass

    for index in range(len(rows)):
        if len(rows) < 2:
            continue
        if index == 0:
            left, right = 0, 1
        elif index == len(rows) - 1:
            left, right = len(rows) - 2, len(rows) - 1
        else:
            left, right = index - 1, index + 1
        if timestamps[left] is None or timestamps[right] is None:
            continue
        delta_s = (timestamps[right] - timestamps[left]).total_seconds()
        if delta_s <= 0:
            continue
        if not np.all(np.isfinite(positions[index])) or not np.all(np.isfinite(velocities[index])):
            continue
        if not np.all(np.isfinite(positions[left])) or not np.all(np.isfinite(positions[right])):
            continue
        position_rate = (positions[right] - positions[left]) / delta_s
        provided_velocity = velocities[index]
        rate_norm = float(np.linalg.norm(position_rate))
        provided_norm = float(np.linalg.norm(provided_velocity))
        residual_norm = float(np.linalg.norm(position_rate - provided_velocity))
        relative_error = residual_norm / provided_norm if provided_norm > 1e-12 else None
        rate_norms.append(rate_norm)
        provided_norms.append(provided_norm)
        residual_norms.append(residual_norm)
        if relative_error is not None:
            relative_errors.append(relative_error)
            if 0 < index < len(rows) - 1:
                interior_relative_errors.append(relative_error)
        try:
            row_rate = rows[index]
            row_rate[row_rate_key] = rate_norm  # type: ignore[index]
            row_rate[row_residual_key] = residual_norm  # type: ignore[index]
            row_rate[row_error_key] = relative_error  # type: ignore[index]
        except TypeError:
            pass

    def _summary(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
        return (
            float(min(values)) if values else None,
            float(np.median(values)) if values else None,
            float(max(values)) if values else None,
        )

    rate_min, rate_median, rate_max = _summary(rate_norms)
    velocity_min, velocity_median, velocity_max = _summary(provided_norms)
    residual_min, residual_median, residual_max = _summary(residual_norms)
    _, relative_median, relative_max = _summary(relative_errors)
    _, interior_median, interior_max = _summary(interior_relative_errors)
    return {
        "sample_count": len(rows),
        "valid_sample_count": len(rate_norms),
        "interior_sample_count": len(interior_relative_errors),
        "difference_method": "首帧/末帧使用一侧差分，中间帧使用中心差分；position(m)/time(s) 得到 m/s",
        "position_rate_norm_m_per_s_min": rate_min,
        "position_rate_norm_m_per_s_median": rate_median,
        "position_rate_norm_m_per_s_max": rate_max,
        "provided_velocity_norm_doc_unit_min": velocity_min,
        "provided_velocity_norm_doc_unit_median": velocity_median,
        "provided_velocity_norm_doc_unit_max": velocity_max,
        "residual_norm_assuming_m_per_s_min": residual_min,
        "residual_norm_assuming_m_per_s_median": residual_median,
        "residual_norm_assuming_m_per_s_max": residual_max,
        "relative_error_assuming_m_per_s_median": relative_median,
        "relative_error_assuming_m_per_s_max": relative_max,
        "interior_relative_error_assuming_m_per_s_median": interior_median,
        "interior_relative_error_assuming_m_per_s_max": interior_max,
        "velocity_unit_hypothesis": "m/s",
        "velocity_unit_status": "由位置(m)/DATE-OBS(s) 差分高度支持；格式说明速度字段原文写 m，待主办方确认",
    }


def _telemetry_prediction(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
    horizon_s: float | None,
) -> dict[str, Any]:
    """用末帧位置和速度做一个可复核的短期常速度外推。

    这是遥测状态量的基线预测，不是轨道传播。速度字段的单位仍保留
    ``m/s?`` 状态，因此预测位置也不包装成经过轨道模型确认的物理量。
    """

    result: dict[str, Any] = {
        "prefix": prefix,
        "method": "last_state_constant_velocity",
        "horizon_s": horizon_s,
        "frame_index": None,
        "status": "unavailable",
        "last_position_m": None,
        "last_velocity_m_per_s_assumed": None,
        "predicted_position_m_assuming_m_per_s": None,
        "predicted_position_norm_m_assuming_m_per_s": None,
        "velocity_azimuth_deg": None,
        "velocity_elevation_deg": None,
        "unit_status": "速度单位待主办方确认",
    }
    if not rows or horizon_s is None or not math.isfinite(float(horizon_s)) or horizon_s < 0:
        return result
    last = rows[-1]
    position = np.asarray([_as_float(last.get(f"{prefix}_{axis}")) for axis in ("x", "y", "z")], dtype=np.float64)
    velocity = np.asarray([_as_float(last.get(f"{prefix}_{axis}v")) for axis in ("x", "y", "z")], dtype=np.float64)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        return result
    predicted = position + velocity * float(horizon_s)
    result.update(
        {
            "frame_index": int(last.get("frame_index", len(rows) - 1)),
            "status": "available",
            "last_position_m": [float(value) for value in position],
            "last_velocity_m_per_s_assumed": [float(value) for value in velocity],
            "predicted_position_m_assuming_m_per_s": [float(value) for value in predicted],
            "predicted_position_norm_m_assuming_m_per_s": float(np.linalg.norm(predicted)),
            "velocity_azimuth_deg": float(np.degrees(np.arctan2(velocity[1], velocity[0]))),
            "velocity_elevation_deg": float(np.degrees(np.arctan2(velocity[2], np.hypot(velocity[0], velocity[1])))),
        }
    )
    return result


def _telemetry_prediction_rows(predictions: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把 J2000/WGS84 预测摘要展平为适合答辩表格的两行。"""

    rows: list[dict[str, Any]] = []
    for prefix in ("j2000", "wgs84"):
        prediction = predictions.get(prefix, {})
        last_position = prediction.get("last_position_m") or [None, None, None]
        velocity = prediction.get("last_velocity_m_per_s_assumed") or [None, None, None]
        predicted = prediction.get("predicted_position_m_assuming_m_per_s") or [None, None, None]
        rows.append(
            {
                "reference_frame": prefix.upper(),
                "status": prediction.get("status"),
                "frame_index": prediction.get("frame_index"),
                "horizon_s": prediction.get("horizon_s"),
                "last_x_m": last_position[0],
                "last_y_m": last_position[1],
                "last_z_m": last_position[2],
                "velocity_x_m_per_s_assumed": velocity[0],
                "velocity_y_m_per_s_assumed": velocity[1],
                "velocity_z_m_per_s_assumed": velocity[2],
                "predicted_x_m_assuming_m_per_s": predicted[0],
                "predicted_y_m_assuming_m_per_s": predicted[1],
                "predicted_z_m_assuming_m_per_s": predicted[2],
                "predicted_position_norm_m_assuming_m_per_s": prediction.get("predicted_position_norm_m_assuming_m_per_s"),
                "velocity_azimuth_deg": prediction.get("velocity_azimuth_deg"),
                "velocity_elevation_deg": prediction.get("velocity_elevation_deg"),
                "unit_status": prediction.get("unit_status"),
            }
        )
    return rows


def _frame_rows(payload: Mapping[str, Any], sequence_json: Path) -> list[dict[str, Any]]:
    summaries = payload.get("frames")
    shifts = payload.get("cumulative_shifts", [])
    if not isinstance(summaries, list):
        raise ValueError("sequence JSON 缺少 frames 列表")
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries):
        if not isinstance(summary, Mapping):
            raise ValueError(f"frames[{index}] 不是对象")
        raw_path = str(summary.get("path", ""))
        frame_path = _resolve_frame_path(raw_path, sequence_json)
        frame = read_fits(frame_path)
        timestamp_raw = summary.get("timestamp") or frame.header.get("DATE-OBS")
        summary_auxiliary = summary.get("auxiliary")
        auxiliary = dict(summary_auxiliary) if isinstance(summary_auxiliary, Mapping) else (frame.auxiliary.as_dict() if frame.auxiliary is not None else {})
        shift = shifts[index] if index < len(shifts) and isinstance(shifts[index], Sequence) else (0.0, 0.0)
        shift_x = _as_float(shift[0]) if len(shift) > 0 else 0.0
        shift_y = _as_float(shift[1]) if len(shift) > 1 else 0.0
        stats = _robust_image_stats(frame.data)
        exposure_ms = _as_float(summary.get("exposure_ms"))
        if exposure_ms is None:
            exposure_ms = exposure_milliseconds(frame.header)
        row: dict[str, Any] = {
            "frame_index": int(summary.get("frame_index", index)),
            "path": str(frame_path),
            "timestamp": _timestamp_text(timestamp_raw),
            "exposure_ms": exposure_ms,
            "width_px": int(frame.data.shape[1]),
            "height_px": int(frame.data.shape[0]),
            "candidate_count": int(summary.get("candidate_count", 0)),
            "returned_count": int(summary.get("returned_count", 0)),
            "quality_count": int(summary.get("quality_count", 0)),
            "cumulative_shift_x_px": shift_x,
            "cumulative_shift_y_px": shift_y,
            "cumulative_shift_norm_px": float(math.hypot(shift_x or 0.0, shift_y or 0.0)),
            **stats,
            **auxiliary,
        }
        row.update(_vector_metrics(row))
        rows.append(row)
    for prefix in ("j2000", "wgs84"):
        first_position = np.asarray(
            [_as_float(rows[0].get(f"{prefix}_{axis}")) for axis in ("x", "y", "z")],
            dtype=np.float64,
        ) if rows else np.empty(0, dtype=np.float64)
        first_valid = first_position.size == 3 and np.all(np.isfinite(first_position))
        for row in rows:
            position = np.asarray(
                [_as_float(row.get(f"{prefix}_{axis}")) for axis in ("x", "y", "z")],
                dtype=np.float64,
            )
            row[f"{prefix}_position_delta_norm_raw"] = float(np.linalg.norm(position - first_position)) if first_valid and np.all(np.isfinite(position)) else None
    timestamps = [_parse_timestamp(row.get("timestamp")) for row in rows]
    first_timestamp = next((item for item in timestamps if item is not None), None)
    for index, row in enumerate(rows):
        timestamp = timestamps[index]
        row["elapsed_s"] = (timestamp - first_timestamp).total_seconds() if timestamp is not None and first_timestamp is not None else None
        if index == 0 or timestamp is None or timestamps[index - 1] is None:
            row["interval_s"] = None
        else:
            row["interval_s"] = (timestamp - timestamps[index - 1]).total_seconds()
    return rows


def _t95_critical(degrees_of_freedom: int) -> float:
    """返回双侧 95% 区间的 Student-t 临界值。

    轨迹通常只有 3–15 个时间点，不能直接用 1.96 代替小样本临界值。
    表中保留到 30 自由度；更大样本使用正态近似。
    """

    table = (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    )
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    return table[degrees_of_freedom - 1] if degrees_of_freedom <= len(table) else 1.96


def _fit_constant_velocity(
    elapsed_s: Sequence[float],
    x_px: Sequence[float],
    y_px: Sequence[float],
    forecast_horizon_s: float | None,
) -> dict[str, Any]:
    """拟合二维常速度轨迹，并给出小样本 OLS 不确定度。

    X/Y 分量使用同一时间设计矩阵分别拟合。速度和方向标准误由一阶
    delta method 传播；预测区间是拟合轨迹均值的逐轴 95% 置信区间，
    不是未来单次观测的 prediction interval，也不是轨道动力学包络。
    """

    elapsed = np.asarray(elapsed_s, dtype=np.float64)
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    if elapsed.ndim != 1 or x.shape != elapsed.shape or y.shape != elapsed.shape:
        raise ValueError("elapsed_s, x_px and y_px must be one-dimensional arrays of equal length")
    if elapsed.size < 2 or not np.all(np.isfinite(elapsed)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("constant-velocity fit requires at least two finite observations")
    if float(np.ptp(elapsed)) <= 0:
        raise ValueError("constant-velocity fit requires distinct observation times")

    design = np.column_stack((np.ones(elapsed.size, dtype=np.float64), elapsed))
    x_coeff, *_ = np.linalg.lstsq(design, x, rcond=None)
    y_coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted_x = design @ x_coeff
    fitted_y = design @ y_coeff
    x_residual = x - fitted_x
    y_residual = y - fitted_y
    residual_norm = np.hypot(x_residual, y_residual)
    fit_rms = float(np.sqrt(np.mean(residual_norm**2)))
    vx = float(x_coeff[1])
    vy = float(y_coeff[1])
    speed = float(math.hypot(vx, vy))
    direction = float(math.degrees(math.atan2(vy, vx)))
    degrees_of_freedom = int(elapsed.size - 2)

    vx_se: float | None = None
    vy_se: float | None = None
    speed_se: float | None = None
    speed_ci95_low: float | None = None
    speed_ci95_high: float | None = None
    direction_ci95_half_width: float | None = None
    predicted_x_ci95_half_width: float | None = None
    predicted_y_ci95_half_width: float | None = None
    t95: float | None = None
    covariance_x: np.ndarray | None = None
    covariance_y: np.ndarray | None = None
    if degrees_of_freedom > 0:
        inverse_normal = np.linalg.inv(design.T @ design)
        covariance_x = float(np.sum(x_residual**2) / degrees_of_freedom) * inverse_normal
        covariance_y = float(np.sum(y_residual**2) / degrees_of_freedom) * inverse_normal
        vx_se = float(math.sqrt(max(0.0, covariance_x[1, 1])))
        vy_se = float(math.sqrt(max(0.0, covariance_y[1, 1])))
        t95 = _t95_critical(degrees_of_freedom)
        if speed > 0:
            speed_se = float(math.sqrt((vx / speed) ** 2 * vx_se**2 + (vy / speed) ** 2 * vy_se**2))
            direction_se_rad = math.sqrt(vy**2 * vx_se**2 + vx**2 * vy_se**2) / (speed**2)
            direction_ci95_half_width = float(math.degrees(t95 * direction_se_rad))
            speed_ci95_low = float(max(0.0, speed - t95 * speed_se))
            speed_ci95_high = float(speed + t95 * speed_se)

    predicted_x: float | None = None
    predicted_y: float | None = None
    forecast_time: float | None = None
    if forecast_horizon_s is not None and math.isfinite(float(forecast_horizon_s)) and float(forecast_horizon_s) >= 0:
        forecast_time = float(elapsed[-1] + float(forecast_horizon_s))
        forecast_design = np.asarray([1.0, forecast_time], dtype=np.float64)
        predicted_x = float(forecast_design @ x_coeff)
        predicted_y = float(forecast_design @ y_coeff)
        if covariance_x is not None and covariance_y is not None and t95 is not None:
            predicted_x_ci95_half_width = float(t95 * math.sqrt(max(0.0, forecast_design @ covariance_x @ forecast_design)))
            predicted_y_ci95_half_width = float(t95 * math.sqrt(max(0.0, forecast_design @ covariance_y @ forecast_design)))

    return {
        "velocity_x_px_per_s": vx,
        "velocity_y_px_per_s": vy,
        "velocity_x_se_px_per_s": vx_se,
        "velocity_y_se_px_per_s": vy_se,
        "speed_px_per_s": speed,
        "speed_se_px_per_s": speed_se,
        "speed_ci95_low_px_per_s": speed_ci95_low,
        "speed_ci95_high_px_per_s": speed_ci95_high,
        "direction_deg_image": direction,
        "direction_ci95_half_width_deg": direction_ci95_half_width,
        "fit_rms_px": fit_rms,
        "degrees_of_freedom": degrees_of_freedom,
        "t95_critical": t95,
        "fitted_x_px": fitted_x,
        "fitted_y_px": fitted_y,
        "fit_residual_px": residual_norm,
        "prediction_horizon_s": float(forecast_horizon_s) if predicted_x is not None else None,
        "forecast_elapsed_s": forecast_time,
        "predicted_x_px": predicted_x,
        "predicted_y_px": predicted_y,
        "predicted_x_ci95_half_width_px": predicted_x_ci95_half_width,
        "predicted_y_ci95_half_width_px": predicted_y_ci95_half_width,
        "uncertainty_note": "逐轴 OLS 轨迹均值 95% 置信区间；未包含模型失配、配准系统误差或未来单次观测噪声。",
    }


def _motion_rows(payload: Mapping[str, Any], frame_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    features = payload.get("motion_features", [])
    if not isinstance(features, list):
        return []
    timestamps = [_parse_timestamp(row.get("timestamp")) for row in frame_rows]
    interval_values = [
        float(row["interval_s"])
        for row in frame_rows
        if row.get("interval_s") is not None and float(row["interval_s"]) > 0
    ]
    prediction_horizon_s = float(np.median(interval_values) * 5.0) if interval_values else None
    result: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        points = feature.get("points", [])
        if not isinstance(points, list) or not points:
            continue
        valid_points = [point for point in points if isinstance(point, Mapping)]
        if not valid_points:
            continue
        valid_points.sort(key=lambda point: int(point.get("frame_index", 0)))
        first = valid_points[0]
        last = valid_points[-1]
        first_index = int(first.get("frame_index", 0))
        last_index = int(last.get("frame_index", first_index))
        first_time = timestamps[first_index] if 0 <= first_index < len(timestamps) else None
        last_time = timestamps[last_index] if 0 <= last_index < len(timestamps) else None
        duration_s = (last_time - first_time).total_seconds() if first_time is not None and last_time is not None else None
        x = np.array([float(point.get("aligned_x", point.get("x", 0.0))) for point in valid_points], dtype=np.float64)
        y = np.array([float(point.get("aligned_y", point.get("y", 0.0))) for point in valid_points], dtype=np.float64)
        if len(valid_points) >= 2 and duration_s is not None and duration_s > 0:
            elapsed = np.array(
                [
                    (timestamps[int(point.get("frame_index", 0))] - first_time).total_seconds()
                    if 0 <= int(point.get("frame_index", 0)) < len(timestamps) and timestamps[int(point.get("frame_index", 0))] is not None and first_time is not None
                    else float(index)
                    for index, point in enumerate(valid_points)
                ],
                dtype=np.float64,
            )
            kinematics = _fit_constant_velocity(elapsed, x, y, prediction_horizon_s)
            fit_rms = float(kinematics["fit_rms_px"])
            point_fit_residuals = np.asarray(kinematics["fit_residual_px"], dtype=np.float64)
            speed_px_per_s = float(kinematics["speed_px_per_s"])
            displacement_px = float(speed_px_per_s * duration_s)
            predicted_x_px = _as_float(kinematics["predicted_x_px"])
            predicted_y_px = _as_float(kinematics["predicted_y_px"])
        else:
            kinematics = {}
            fit_rms = _as_float(feature.get("fit_rms_px"))
            point_fit_residuals = np.full(len(valid_points), np.nan, dtype=np.float64)
            speed_px_per_s = None
            displacement_px = _as_float(feature.get("displacement_px"))
            predicted_x_px = None
            predicted_y_px = None
        dx = float(x[-1] - x[0])
        dy = float(y[-1] - y[0])
        lengths = [float(point.get("length_px", 0.0)) for point in valid_points]
        widths = [float(point.get("width_px", 0.0)) for point in valid_points]
        snrs = [float(point.get("residual_snr", 0.0)) for point in valid_points]
        point_records: list[dict[str, Any]] = []
        for point, point_residual in zip(valid_points, point_fit_residuals, strict=True):
            record = dict(point)
            record["fit_residual_px"] = float(point_residual) if np.isfinite(point_residual) else None
            point_records.append(record)
        result.append(
            {
                "track_id": int(feature.get("track_id", len(result))),
                "classification": str(feature.get("classification", "candidate")),
                "presence": len(valid_points),
                "first_frame": first_index,
                "last_frame": last_index,
                "duration_s": duration_s,
                "displacement_px": displacement_px,
                "speed_px_per_frame": _as_float(feature.get("speed_px_per_frame")),
                "speed_px_per_s": speed_px_per_s,
                "velocity_x_px_per_s": _as_float(kinematics.get("velocity_x_px_per_s")),
                "velocity_y_px_per_s": _as_float(kinematics.get("velocity_y_px_per_s")),
                "velocity_x_se_px_per_s": _as_float(kinematics.get("velocity_x_se_px_per_s")),
                "velocity_y_se_px_per_s": _as_float(kinematics.get("velocity_y_se_px_per_s")),
                "speed_se_px_per_s": _as_float(kinematics.get("speed_se_px_per_s")),
                "speed_ci95_low_px_per_s": _as_float(kinematics.get("speed_ci95_low_px_per_s")),
                "speed_ci95_high_px_per_s": _as_float(kinematics.get("speed_ci95_high_px_per_s")),
                "direction_deg_image": _as_float(kinematics.get("direction_deg_image")) if kinematics else (_as_float(first.get("angle_deg")) if len(valid_points) == 1 else float(math.degrees(math.atan2(dy, dx)))),
                "direction_ci95_half_width_deg": _as_float(kinematics.get("direction_ci95_half_width_deg")),
                "fit_rms_px": fit_rms,
                "kinematic_model": "constant_velocity_ols_date_obs" if kinematics else None,
                "uncertainty_method": "student_t_95_delta_method_independent_xy" if kinematics and kinematics.get("t95_critical") is not None else None,
                "fit_degrees_of_freedom": int(kinematics["degrees_of_freedom"]) if kinematics else None,
                "fit_t95_critical": _as_float(kinematics.get("t95_critical")),
                "median_length_px": float(np.median(lengths)),
                "max_length_px": float(max(lengths)),
                "median_width_px": float(np.median(widths)),
                "max_residual_snr": float(max(snrs)),
                "start_x_px": float(x[0]),
                "start_y_px": float(y[0]),
                "end_x_px": float(x[-1]),
                "end_y_px": float(y[-1]),
                "fitted_start_x_px": float(kinematics["fitted_x_px"][0]) if kinematics else None,
                "fitted_start_y_px": float(kinematics["fitted_y_px"][0]) if kinematics else None,
                "fitted_end_x_px": float(kinematics["fitted_x_px"][-1]) if kinematics else None,
                "fitted_end_y_px": float(kinematics["fitted_y_px"][-1]) if kinematics else None,
                "prediction_horizon_s": prediction_horizon_s if predicted_x_px is not None else None,
                "predicted_x_px": predicted_x_px,
                "predicted_y_px": predicted_y_px,
                "predicted_x_ci95_half_width_px": _as_float(kinematics.get("predicted_x_ci95_half_width_px")),
                "predicted_y_ci95_half_width_px": _as_float(kinematics.get("predicted_y_ci95_half_width_px")),
                "uncertainty_note": kinematics.get("uncertainty_note") if kinematics else "单帧或两点轨迹不能估计小样本拟合不确定度。",
                "frame_indices": ",".join(str(int(point.get("frame_index", 0)) + 1) for point in valid_points),
                "points": point_records,
            }
        )
    return result


def _motion_audit_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """规范化逐帧线状筛选计数，并补充面向用户的 1-based 帧号。"""

    raw_audits = payload.get("motion_frame_audits", [])
    if not isinstance(raw_audits, Sequence) or isinstance(raw_audits, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_audits:
        if not isinstance(raw, Mapping):
            continue
        try:
            frame_index = int(raw.get("frame_index", len(rows)))
        except (TypeError, ValueError):
            frame_index = len(rows)
        row = dict(raw)
        row["frame_index"] = frame_index
        row["frame_number"] = frame_index + 1
        rows.append(row)
    return rows


_PHOTOMETRIC_LEVELS = (
    "INSTRUMENTAL_ONLY",
    "RELATIVE_CALIBRATED",
    "APPARENT_CALIBRATED",
    "ABSOLUTE_ELIGIBLE",
    "GEOMETRY_UNAVAILABLE",
)
_RELATIVE_VALID_STATUSES = {"VALID", "VALID_NO_HOLDOUT"}
_PHOTOMETRIC_REPORT_ALIASES = (
    "photometric_quality_report",
    "photometric_quality",
    "photometric_report",
    "photometry_report",
)


def _evidence_mapping(value: object) -> dict[str, Any] | None:
    """把映射、结果对象或 ``as_dict`` 对象转为只读报告视图。

    ``innovation`` 的输入既可能来自 JSON，也可能来自 GUI 传入的结果对象。
    这里不修改输入对象，也不要求调用方先导入任意一个具体的光度结果类，
    因而可以兼容旧序列结果和新的 dataclass payload。
    """

    if isinstance(value, Mapping):
        return dict(value)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        converted = as_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return dict(attributes)
    return None


def _evidence_key(value: object) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value).strip().lower()).strip("_")


def _evidence_lookup(mapping: Mapping[str, Any] | None, aliases: Iterable[str]) -> tuple[bool, Any]:
    if mapping is None:
        return False, None
    values = {_evidence_key(key): value for key, value in mapping.items()}
    for alias in aliases:
        key = _evidence_key(alias)
        if key in values:
            return True, values[key]
    return False, None


def _evidence_value(mapping: Mapping[str, Any] | None, aliases: Iterable[str], default: Any = None) -> Any:
    found, value = _evidence_lookup(mapping, aliases)
    return value if found else default


def _evidence_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "y", "1", "pass", "passed", "ok"}:
            return True
        if token in {"false", "no", "n", "0", "fail", "failed", "invalid"}:
            return False
    return None


def _evidence_status(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.upper().replace("-", "_").replace(" ", "_") if text else None


def _evidence_count(value: object) -> int | None:
    number = _as_float(value)
    if number is None or number < 0:
        return None
    return int(number)


def _evidence_strings(value: object, *, limit: int = 64) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return [str(item) for item in list(value)[:limit] if item is not None]


def _finite_evidence_mapping(value: object) -> dict[str, float]:
    mapping = _evidence_mapping(value)
    if mapping is None:
        return {}
    result: dict[str, float] = {}
    for key, raw in mapping.items():
        number = _as_float(raw)
        if number is not None:
            result[str(key)] = number
    return result


def _count_evidence_mapping(value: object) -> dict[str, int]:
    mapping = _evidence_mapping(value)
    if mapping is None:
        return {}
    result: dict[str, int] = {}
    for key, raw in mapping.items():
        count = _evidence_count(raw)
        if count is not None:
            result[str(key)] = count
    return {key: result[key] for key in sorted(result)}


def _evidence_rows(value: object) -> list[dict[str, Any]]:
    """提取光度报告行，但不把原始检测源表误当成测光表。"""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [mapping for item in value if (mapping := _evidence_mapping(item)) is not None]
    mapping = _evidence_mapping(value)
    if mapping is None:
        return []
    for alias in (
        "rows",
        "source_rows",
        "source_photometry",
        "photometry_rows",
        "photometry_results",
        "results",
        "items",
    ):
        found, nested = _evidence_lookup(mapping, (alias,))
        if not found:
            continue
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            return [item for raw in nested if (item := _evidence_mapping(raw)) is not None]
    return []


def _normalise_evidence_level(value: object) -> str | None:
    status = _evidence_status(value)
    if status in _PHOTOMETRIC_LEVELS:
        return status
    return None


def _absolute_value_from_row(row: Mapping[str, Any]) -> float | None:
    found, raw = _evidence_lookup(row, ("M", "absolute_magnitude", "absolute_magnitude_value", "absolute_mag", "M_V", "m_abs"))
    if found:
        nested = _evidence_mapping(raw)
        if nested is not None:
            return _as_float(_evidence_value(nested, ("value", "M", "absolute_magnitude", "absolute_magnitude_value")))
        return _as_float(raw)
    return None


def _row_evidence_level(row: Mapping[str, Any]) -> str | None:
    explicit = _normalise_evidence_level(_evidence_value(row, ("observability_level", "observability", "level")))
    if explicit is not None:
        return explicit
    status = _evidence_status(_evidence_value(row, ("status", "state", "calibration_status")))
    if status in {"CATALOG_INCONSISTENT", "PHOTOMETRIC_OUTLIER", "PHOTOMETRICALLY_INCONSISTENT"}:
        # A retained numerical m_cal is diagnostic evidence only when its
        # catalog residual failed the source-level gate; do not count it as a
        # usable apparent magnitude in the innovation summary.
        return "INSTRUMENTAL_ONLY"
    absolute_raw_found, absolute_raw = _evidence_lookup(
        row, ("absolute_magnitude", "absolute", "absolute_result")
    )
    absolute_mapping = _evidence_mapping(absolute_raw) if absolute_raw_found else None
    absolute_status = _evidence_status(
        _evidence_value(row, ("absolute_status", "M_status", "absolute_magnitude_status"))
    )
    if absolute_status is None and absolute_mapping is not None:
        absolute_status = _evidence_status(_evidence_value(absolute_mapping, ("status", "state", "absolute_status")))
    # SourcePhotometry keeps the row status as CALIBRATED while the stricter
    # absolute status lives in the nested estimate.  Inspect the nested status
    # before falling through to APPARENT_CALIBRATED, otherwise valid GSP-Phot
    # model-distance estimates would disappear from the innovation counts.
    if _absolute_value_from_row(row) is not None and absolute_status in {
        "VALID",
        "VALID_MODEL_DISTANCE",
        "VALID_MODEL_DISTANCE_NO_INTERVAL",
        "ABSOLUTE_VALID",
        "ABSOLUTE_ELIGIBLE",
        "CALIBRATED_ABSOLUTE",
    }:
        return "ABSOLUTE_ELIGIBLE"
    if status in {"RELATIVE_CALIBRATED", "RELATIVE", "RELATIVE_CALIBRATION"}:
        return "RELATIVE_CALIBRATED"
    if status in {"CALIBRATED", "APPARENT_CALIBRATED", "APPARENT", "VALID_NO_HOLDOUT"}:
        return "APPARENT_CALIBRATED"
    if _absolute_value_from_row(row) is not None and status in {
        "VALID",
        "VALID_MODEL_DISTANCE",
        "VALID_MODEL_DISTANCE_NO_INTERVAL",
        "ABSOLUTE_VALID",
        "ABSOLUTE_ELIGIBLE",
        "CALIBRATED_ABSOLUTE",
    }:
        return "ABSOLUTE_ELIGIBLE"
    if _as_float(_evidence_value(row, ("m_inst", "instrumental_magnitude", "instrumental_mag"))) is not None:
        return "INSTRUMENTAL_ONLY"
    return None


def _compact_photometric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """保留一小段逐源证据，避免把大型原始检测表复制进创新报告。"""

    absolute = _absolute_value_from_row(row)
    result: dict[str, Any] = {
        "row_key": _evidence_value(row, ("row_key", "key")),
        "frame_id": _evidence_value(row, ("frame_id", "frame")),
        "source_id": _evidence_value(row, ("source_id", "detection_id", "id")),
        "observability_level": _row_evidence_level(row),
        "status": _evidence_value(row, ("status", "state")),
        "m_inst": _as_float(_evidence_value(row, ("m_inst", "instrumental_magnitude", "instrumental_mag"))),
        "m_cal": _as_float(_evidence_value(row, ("m_cal", "calibrated_magnitude", "apparent_magnitude", "m_std"))),
        "absolute_magnitude": absolute,
        "photometric_residual_mag": _as_float(
            _evidence_value(row, ("photometric_residual_mag", "photometric_residual", "catalog_residual_mag"))
        ),
        "photometric_consistent": _evidence_bool(_evidence_value(row, ("photometric_consistent",))),
        "photometric_outlier_reason": _evidence_value(
            row, ("photometric_outlier_reason", "outlier_reason")
        ),
        "calibration_sample_role": _evidence_value(row, ("calibration_sample_role", "sample_role")),
        "flags": _evidence_strings(_evidence_value(row, ("flags",))),
        "rejection_reasons": _evidence_strings(_evidence_value(row, ("rejection_reasons", "reasons"))),
        "missing_inputs": _evidence_strings(_evidence_value(row, ("missing_inputs",))),
    }
    return result


def _photometric_quality_summary(value: object, *, source_name: str) -> dict[str, Any]:
    mapping = _evidence_mapping(value)
    rows = _evidence_rows(value)
    if mapping is None and not rows:
        return {
            "present": False,
            "source": source_name,
            "row_count": 0,
            "sample_rows": [],
            "sample_truncated": False,
        }

    declared_counts: dict[str, int] = {}
    for alias in ("observability_counts", "level_counts", "observability_levels"):
        candidate = _evidence_value(mapping, (alias,))
        declared_counts = {
            _evidence_status(key) or str(key): count
            for key, count in _count_evidence_mapping(candidate).items()
            if (_evidence_status(key) or str(key)) in _PHOTOMETRIC_LEVELS
        }
        if declared_counts:
            break
    if not declared_counts:
        counts_mapping = _evidence_mapping(_evidence_value(mapping, ("counts",)))
        if counts_mapping is not None:
            declared_counts = {
                _evidence_status(key) or str(key): count
                for key, count in _count_evidence_mapping(_evidence_value(counts_mapping, ("observability",))).items()
                if (_evidence_status(key) or str(key)) in _PHOTOMETRIC_LEVELS
            }

    row_counts = {level: 0 for level in _PHOTOMETRIC_LEVELS}
    instrumental_count = 0
    apparent_count = 0
    absolute_value_count = 0
    absolute_eligible_count = 0
    derived_reasons: dict[str, int] = {}
    derived_missing: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    photometric_consistent_count = 0
    photometric_inconsistent_count = 0
    photometric_unknown_count = 0
    for row in rows:
        level = _row_evidence_level(row)
        if level is not None:
            row_counts[level] += 1
        if _as_float(_evidence_value(row, ("m_inst", "instrumental_magnitude", "instrumental_mag"))) is not None:
            instrumental_count += 1
        if _as_float(_evidence_value(row, ("m_cal", "calibrated_magnitude", "apparent_magnitude", "m_std"))) is not None:
            apparent_count += 1
        absolute = _absolute_value_from_row(row)
        if absolute is not None:
            absolute_value_count += 1
        if level == "ABSOLUTE_ELIGIBLE":
            absolute_eligible_count += 1
        status = _evidence_status(_evidence_value(row, ("status", "state")))
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        consistent = _evidence_bool(_evidence_value(row, ("photometric_consistent",)))
        if consistent is True:
            photometric_consistent_count += 1
        elif consistent is False:
            photometric_inconsistent_count += 1
        elif _as_float(
            _evidence_value(row, ("photometric_residual_mag", "photometric_residual", "catalog_residual_mag"))
        ) is not None:
            photometric_unknown_count += 1
        for reason in _evidence_strings(_evidence_value(row, ("rejection_reasons", "reasons"))):
            derived_reasons[reason] = derived_reasons.get(reason, 0) + 1
        for missing in _evidence_strings(_evidence_value(row, ("missing_inputs",))):
            derived_missing[missing] = derived_missing.get(missing, 0) + 1

    level_counts = declared_counts or {key: value for key, value in row_counts.items() if value}
    false_valid = _evidence_bool(_evidence_value(mapping, ("false_valid",)))
    gate = _evidence_mapping(_evidence_value(mapping, ("false_valid_gate",)))
    gate_passed = _evidence_bool(_evidence_value(gate, ("passed",)))
    if false_valid is None and gate_passed is not None:
        false_valid = not gate_passed
    if gate_passed is None and false_valid is not None:
        gate_passed = not false_valid

    source_count = _evidence_count(_evidence_value(mapping, ("source_count",)))
    if source_count is None:
        source_count = len(rows)
    frame_count = _evidence_count(_evidence_value(mapping, ("frame_count", "frames")))
    missing_inputs = _count_evidence_mapping(_evidence_value(mapping, ("missing_inputs", "missing_input_counts"))) or {
        key: derived_missing[key] for key in sorted(derived_missing)
    }
    rejection_reasons = _count_evidence_mapping(_evidence_value(mapping, ("rejection_reasons", "reason_counts"))) or {
        key: derived_reasons[key] for key in sorted(derived_reasons)
    }
    sample_rows = [_compact_photometric_row(row) for row in rows[:20]]
    provenance = _evidence_mapping(_evidence_value(mapping, ("provenance",)))
    compact_provenance: dict[str, Any] = {}
    if provenance is not None:
        for key, raw in sorted(provenance.items(), key=lambda item: str(item[0])):
            if isinstance(raw, (str, int, bool)):
                compact_provenance[str(key)] = raw
            else:
                number = _as_float(raw)
                if number is not None:
                    compact_provenance[str(key)] = number
                elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                    compact_provenance[str(key)] = _evidence_strings(raw, limit=16)

    return {
        "present": True,
        "source": source_name,
        "status": _evidence_status(_evidence_value(mapping, ("status", "state"))),
        "frame_count": frame_count,
        "source_count": source_count,
        "row_count": len(rows),
        "level_counts": {key: level_counts[key] for key in sorted(level_counts)},
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "photometric_consistent_count": photometric_consistent_count,
        "photometric_inconsistent_count": photometric_inconsistent_count,
        "photometric_consistency_unknown_count": photometric_unknown_count,
        "instrumental_magnitude_count": instrumental_count,
        "apparent_magnitude_count": apparent_count,
        "absolute_magnitude_value_count": absolute_value_count,
        "absolute_eligible_count": absolute_eligible_count or level_counts.get("ABSOLUTE_ELIGIBLE", 0),
        "false_valid": false_valid,
        "false_valid_gate_passed": gate_passed,
        "flags": _evidence_strings(_evidence_value(mapping, ("flags",))),
        "errors": _evidence_strings(_evidence_value(mapping, ("errors",))),
        "rejection_reasons": rejection_reasons,
        "missing_inputs": missing_inputs,
        "provenance": compact_provenance,
        "sample_rows": sample_rows,
        "sample_truncated": len(rows) > len(sample_rows),
    }


def _relative_photometry_summary(value: object) -> dict[str, Any]:
    mapping = _evidence_mapping(value)
    if mapping is None:
        return {
            "present": False,
            "status": None,
            "usable": False,
            "flags": [],
            "relative_magnitudes": {},
            "frame_zero_points": {},
        }
    relative_values = _finite_evidence_mapping(_evidence_value(mapping, ("relative_magnitudes", "source_relative_magnitudes")))
    frame_offsets = _finite_evidence_mapping(_evidence_value(mapping, ("frame_zero_points", "zero_points")))
    status = _evidence_status(_evidence_value(mapping, ("status", "state")))
    reference_count = _evidence_count(_evidence_value(mapping, ("reference_sample_count", "reference_count")))
    frame_count = len(frame_offsets)
    source_count = len(relative_values)
    usable = status in _RELATIVE_VALID_STATUSES and reference_count is not None and reference_count > 0 and frame_count > 0 and source_count > 0
    values = list(relative_values.values())
    return {
        "present": True,
        "status": status,
        "usable": usable,
        "flags": _evidence_strings(_evidence_value(mapping, ("flags",))),
        "reference_sample_count": reference_count,
        "reference_sample_count_by_frame": _count_evidence_mapping(_evidence_value(mapping, ("reference_sample_count_by_frame",))),
        "reference_sample_count_by_source": _count_evidence_mapping(_evidence_value(mapping, ("reference_sample_count_by_source",))),
        "frame_count": frame_count,
        "source_count": source_count,
        "frame_zero_point_count": len(frame_offsets),
        "relative_magnitude_count": len(relative_values),
        "frame_zero_points": frame_offsets,
        "relative_magnitudes": relative_values,
        "relative_magnitude_range": {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        },
        "training_residual_rms_mag": _as_float(_evidence_value(mapping, ("training_residual_rms", "train_rms"))),
        "training_residual_mad_mag": _as_float(_evidence_value(mapping, ("training_residual_mad", "train_mad"))),
        "validation_residual_rms_mag": _as_float(_evidence_value(mapping, ("validation_residual_rms", "validation_rms"))),
        "validation_residual_mad_mag": _as_float(_evidence_value(mapping, ("validation_residual_mad", "validation_mad"))),
        "validation_sample_count": _evidence_count(_evidence_value(mapping, ("validation_sample_count",))),
        "validation_source_count": len(_evidence_strings(_evidence_value(mapping, ("validation_source_ids",)))),
        "excluded_observation_count": _evidence_count(_evidence_value(mapping, ("excluded_observation_count",))),
        "spatial_order": _evidence_count(_evidence_value(mapping, ("spatial_order",))),
        "absolute_magnitude_claim": "NOT_DERIVED_FROM_RELATIVE_SCALE",
        "note": "这些数值是相对光度尺度和帧零点；创新报告不把它们改名为表观或绝对星等。",
    }


def _photometric_calibration_summary(value: object) -> dict[str, Any]:
    mapping = _evidence_mapping(value)
    if mapping is None:
        return {"present": False}
    coefficients = [number for number in (_as_float(item) for item in _evidence_strings(_evidence_value(mapping, ("coefficients",)))) if number is not None]
    # ``coefficients`` 通常是数值列表；上面的字符串路径只为兼容极旧的
    # JSON 包装，下面再直接读取序列以避免丢失正常输入。
    raw_coefficients = _evidence_value(mapping, ("coefficients",))
    if isinstance(raw_coefficients, Sequence) and not isinstance(raw_coefficients, (str, bytes, bytearray)):
        coefficients = [number for number in (_as_float(item) for item in raw_coefficients) if number is not None]
    return {
        "present": True,
        "status": _evidence_status(_evidence_value(mapping, ("status", "state"))),
        "photometric_system": _evidence_value(mapping, ("photometric_system", "system")),
        "photometric_band": _evidence_value(mapping, ("photometric_band", "band")),
        "color_name": _evidence_value(mapping, ("color_name", "color")),
        "color_order": _evidence_count(_evidence_value(mapping, ("color_order",))),
        "coefficients": coefficients,
        "calibrator_count": _evidence_count(_evidence_value(mapping, ("calibrator_count",))),
        "inlier_count": _evidence_count(_evidence_value(mapping, ("inlier_count",))),
        "validation_count": _evidence_count(_evidence_value(mapping, ("validation_count",))),
        "fit_rms_mag": _as_float(_evidence_value(mapping, ("fit_rms_mag",))),
        "validation_rms_mag": _as_float(_evidence_value(mapping, ("validation_rms_mag",))),
        "flags": _evidence_strings(_evidence_value(mapping, ("flags",))),
        "note": "这是输入中已有的同设备/同条件标定声明；创新报告不重新拟合零点。",
    }


def _photometric_evidence_section(payload: Mapping[str, Any] | object) -> dict[str, Any]:
    """把星等相关输入接入创新报告，同时显式保留物理证据边界。

    该函数只做报告消费和审计，不执行新的星表匹配、不从相对尺度推导
    绝对星等，也不把 ``m_inst`` 或输入中未经质量门确认的数字升级为
    ``m_cal``/``M``。旧的序列 JSON 没有这些字段时仍返回稳定的
    ``NOT_PRESENT`` section。
    """

    mapping = _evidence_mapping(payload) or {}
    relative_raw = _evidence_value(mapping, ("relative_photometry",))
    relative = _relative_photometry_summary(relative_raw)

    quality_raw: object = None
    quality_source: str | None = None
    for alias in _PHOTOMETRIC_REPORT_ALIASES:
        found, value = _evidence_lookup(mapping, (alias,))
        if found and value is not None:
            quality_raw = value
            quality_source = alias
            break
    photometry_container = _evidence_mapping(_evidence_value(mapping, ("photometry",)))
    if quality_source is None and photometry_container is not None:
        for alias in ("quality_report", "quality", "report"):
            found, value = _evidence_lookup(photometry_container, (alias,))
            if found and value is not None:
                quality_raw = value
                quality_source = f"photometry.{alias}"
                break
    if quality_source is None:
        photometry_found, photometry_value = _evidence_lookup(mapping, ("photometry",))
        if photometry_found and photometry_value is not None:
            quality_raw = photometry_value
            quality_source = "photometry"
    if quality_source is None:
        source_rows_found, source_rows = _evidence_lookup(mapping, ("source_photometry",))
        if source_rows_found and source_rows is not None:
            quality_raw = {"rows": source_rows}
            quality_source = "source_photometry"
    quality = _photometric_quality_summary(quality_raw, source_name=quality_source or "photometric_quality_report")

    calibration_raw = _evidence_value(mapping, ("photometric_calibration", "calibration"))
    calibration = _photometric_calibration_summary(calibration_raw)
    sources: list[str] = []
    if relative["present"]:
        sources.append("relative_photometry")
    if quality["present"]:
        sources.append(str(quality["source"]))
    if calibration["present"]:
        sources.append("photometric_calibration")

    quality_rejected = quality.get("false_valid") is True
    quality_level_count = sum(int(value) for value in quality.get("level_counts", {}).values()) if quality.get("present") else 0
    apparent_evidence_count = int(quality.get("apparent_magnitude_count", 0)) + int(quality.get("absolute_magnitude_value_count", 0))
    apparent_evidence_count += int(quality.get("level_counts", {}).get("APPARENT_CALIBRATED", 0))
    apparent_evidence_count += int(quality.get("level_counts", {}).get("ABSOLUTE_ELIGIBLE", 0))
    available = bool(relative.get("usable") or apparent_evidence_count > 0 or calibration.get("status") in {"VALID", "VALID_NO_HOLDOUT"})
    if not sources:
        status = "NOT_PRESENT"
    elif quality_rejected:
        status = "PRESENT_BUT_REJECTED"
    elif available:
        status = "AVAILABLE"
    elif quality.get("present") and (quality.get("row_count", 0) > 0 or quality_level_count > 0) and quality.get("instrumental_magnitude_count", 0) > 0:
        status = "INSTRUMENTAL_ONLY"
    else:
        status = "PRESENT_BUT_INCOMPLETE"

    apparent_count = int(quality.get("apparent_magnitude_count", 0))
    if apparent_count == 0:
        apparent_count = int(quality.get("level_counts", {}).get("APPARENT_CALIBRATED", 0)) + int(quality.get("level_counts", {}).get("ABSOLUTE_ELIGIBLE", 0))
    absolute_value_count = int(quality.get("absolute_magnitude_value_count", 0))
    absolute_eligible_count = int(quality.get("absolute_eligible_count", 0))
    return {
        "status": status,
        "evidence_sources": sources,
        "relative_photometry": relative,
        "photometric_quality": quality,
        "photometric_calibration": calibration,
        "summary": {
            "relative_scale_usable": bool(relative.get("usable")),
            "relative_source_count": int(relative.get("relative_magnitude_count", 0)),
            "relative_frame_count": int(relative.get("frame_zero_point_count", 0)),
            "reported_apparent_magnitude_count": apparent_count,
            "reported_absolute_magnitude_value_count": absolute_value_count,
            "reported_absolute_eligible_count": absolute_eligible_count,
            "absolute_magnitude_status": "INPUT_EVIDENCE_ONLY" if absolute_value_count or absolute_eligible_count else "NOT_AVAILABLE",
        },
        "boundary": {
            "instrumental_magnitude": {
                "status": "INPUT_MEASUREMENT_ONLY",
                "note": "m_inst/仪器星等可作为观测量消费，但不含同设备绝对零点。",
            },
            "relative_magnitude": {
                "status": "CONSUMED_WITH_RELATIVE_LABEL" if relative.get("present") else "NOT_PRESENT",
                "note": "relative_photometry 只提供跨帧相对尺度、帧零点和残差证据。",
            },
            "apparent_magnitude": {
                "status": "REPORTED_BY_INPUT_ONLY" if apparent_count else "NOT_INFERRED",
                "count": apparent_count,
                "note": "只有输入质量报告明确提供并通过其自身状态门时才保留为表观星等证据；本模块不重新标定。",
            },
            "absolute_magnitude": {
                "status": "INPUT_EVIDENCE_ONLY" if absolute_value_count or absolute_eligible_count else "NOT_INFERRED",
                "value_count": absolute_value_count,
                "eligible_count": absolute_eligible_count,
                "note": "创新模块不会从 m_inst、相对光度或亮度排序推导绝对星等；需要目录身份、距离/视差、消光和质量门。",
            },
        },
        "notes": [
            "星等 section 是创新报告对已有光度证据的审计摘要，不是新的星表匹配或重新测光结果。",
            "运动目标没有被自动当作 Gaia 恒星；相对光度证据也不会改变运动/静态分类。",
        ],
    }


def _build_innovation_report_payload(payload: Mapping[str, Any], json_path: Path) -> dict[str, Any]:
    """从已经载入的 ``SequenceResult.as_dict`` 对象构建完整序列证据报告。"""

    if not isinstance(payload, Mapping):
        converted = _evidence_mapping(payload)
        if converted is None:
            raise ValueError("sequence JSON 根节点必须是对象")
        payload = converted
    frame_rows = _frame_rows(payload, json_path)
    motion_rows = _motion_rows(payload, frame_rows)
    motion_audit_rows = _motion_audit_rows(payload)
    intervals = [float(row["interval_s"]) for row in frame_rows if row.get("interval_s") is not None and float(row["interval_s"]) > 0]
    timestamps = [_parse_timestamp(row.get("timestamp")) for row in frame_rows]
    elapsed = [row.get("elapsed_s") for row in frame_rows if row.get("elapsed_s") is not None]
    shift_norms = [float(row["cumulative_shift_norm_px"]) for row in frame_rows]
    attitude_fields = ("roll", "pitch", "yaw")
    attitude_ranges = {
        field: {
            "start": _as_float(frame_rows[0].get(field)) if frame_rows else None,
            "end": _as_float(frame_rows[-1].get(field)) if frame_rows else None,
            "min": float(min(values)) if (values := [_as_float(row.get(field)) for row in frame_rows if _as_float(row.get(field)) is not None]) else None,
            "max": float(max(values)) if values else None,
        }
        for field in attitude_fields
    }
    telemetry: dict[str, Any] = {}
    telemetry_consistency: dict[str, Any] = {}
    telemetry_prediction_horizon_s = float(np.median(intervals) * 5.0) if intervals else None
    telemetry_predictions: dict[str, dict[str, Any]] = {}
    for prefix in ("j2000", "wgs84"):
        velocity_norms = [
            float(row[f"{prefix}_velocity_norm_raw"])
            for row in frame_rows
            if row.get(f"{prefix}_velocity_norm_raw") is not None
        ]
        azimuths = [
            float(row[f"{prefix}_velocity_azimuth_deg"])
            for row in frame_rows
            if row.get(f"{prefix}_velocity_azimuth_deg") is not None
        ]
        telemetry[prefix] = {
            "velocity_norm_raw_min": min(velocity_norms) if velocity_norms else None,
            "velocity_norm_raw_max": max(velocity_norms) if velocity_norms else None,
            "velocity_azimuth_deg_min": min(azimuths) if azimuths else None,
            "velocity_azimuth_deg_max": max(azimuths) if azimuths else None,
            "start_position_norm_raw": frame_rows[0].get(f"{prefix}_position_norm_raw") if frame_rows else None,
            "end_position_norm_raw": frame_rows[-1].get(f"{prefix}_position_norm_raw") if frame_rows else None,
            "end_position_delta_norm_raw": frame_rows[-1].get(f"{prefix}_position_delta_norm_raw") if frame_rows else None,
        }
        telemetry_consistency[prefix] = _telemetry_consistency(frame_rows, prefix)
        telemetry_predictions[prefix] = _telemetry_prediction(frame_rows, prefix, telemetry_prediction_horizon_s)
    reference_mode = str(payload.get("motion_reference_mode", "per_frame_background"))
    if reference_mode == "registered_median":
        reference_note = "插值注册后的时间中值参考差分；差分噪声按 sqrt(2) 倍单帧噪声估计。"
    elif reference_mode == "temporal_median_small_shift":
        reference_note = "累计平移小于 0.75 px 时的原坐标时间中值近似差分；差分噪声按 sqrt(2) 倍单帧噪声估计。"
    else:
        reference_note = "单帧背景差分；没有跨帧参考，因此线状结果只能作为形状候选。"
    report = {
        "schema_version": 2,
        "source_sequence_json": str(json_path),
        "relation": {
            "frame_count": len(frame_rows),
            "same_shape": len({(row["height_px"], row["width_px"]) for row in frame_rows}) == 1 if frame_rows else False,
            "shapes": sorted({f"{row['width_px']}x{row['height_px']}" for row in frame_rows}),
            "exposure_ms": sorted({row["exposure_ms"] for row in frame_rows if row.get("exposure_ms") is not None}),
            "start_timestamp": frame_rows[0].get("timestamp") if frame_rows else None,
            "end_timestamp": frame_rows[-1].get("timestamp") if frame_rows else None,
            "duration_s": float(elapsed[-1]) if elapsed else None,
            "interval_median_s": float(np.median(intervals)) if intervals else None,
            "interval_std_s": float(np.std(intervals)) if intervals else None,
            "intervals_s": intervals,
        },
        "registration": {
            "final_cumulative_shift_x_px": frame_rows[-1].get("cumulative_shift_x_px") if frame_rows else None,
            "final_cumulative_shift_y_px": frame_rows[-1].get("cumulative_shift_y_px") if frame_rows else None,
            "max_cumulative_shift_norm_px": max(shift_norms) if shift_norms else None,
            "note": "当前为平移配准基线；无标准 WCS 时不换算为角秒或姿态角。",
        },
        "attitude_auxiliary": {
            "ranges": attitude_ranges,
            "rates": _relative_rates(frame_rows, attitude_fields),
            "quaternion_norm_range": {
                "min": min((_as_float(row.get("q1")) or 0.0) ** 2 + (_as_float(row.get("q2")) or 0.0) ** 2 + (_as_float(row.get("q3")) or 0.0) ** 2 + (_as_float(row.get("q4")) or 0.0) ** 2 for row in frame_rows),
                "max": max((_as_float(row.get("q1")) or 0.0) ** 2 + (_as_float(row.get("q2")) or 0.0) ** 2 + (_as_float(row.get("q3")) or 0.0) ** 2 + (_as_float(row.get("q4")) or 0.0) ** 2 for row in frame_rows),
            } if frame_rows else {"min": None, "max": None},
            "note": "辅助字段已按项目格式解码；单位、参考系和字段语义仍以主办方定义为准。",
        },
        "telemetry_vectors": {
            "vectors": telemetry,
            "units": {
                "position": "m（格式说明）",
                "provided_velocity": "m/s?（格式说明速度行原文写 m；由位置/时间差分高度支持，待主办方确认）",
                "angular_velocity": "deg/s（格式说明）",
            },
            "consistency": telemetry_consistency,
            "note": "保留 *_raw 字段以兼容旧报告；位置差分自洽检查仅验证辅助速度字段的数值一致性，不等同于图像平面运动目标速度，也不构成轨道解算。",
        },
        "telemetry_predictions": {
            "horizon_s": telemetry_prediction_horizon_s,
            "method": "末帧位置 + 末帧速度 × 5 个中位帧间隔",
            "predictions": telemetry_predictions,
            "note": "常速度外推基线；速度单位和参考系仍以主办方定义为准，不是轨道传播或未来真实位置保证。",
        },
        "quality": {
            "candidate_count_min": min((row["candidate_count"] for row in frame_rows), default=None),
            "candidate_count_max": max((row["candidate_count"] for row in frame_rows), default=None),
            "returned_count_min": min((row["returned_count"] for row in frame_rows), default=None),
            "returned_count_max": max((row["returned_count"] for row in frame_rows), default=None),
            "quality_count_min": min((row["quality_count"] for row in frame_rows), default=None),
            "quality_count_max": max((row["quality_count"] for row in frame_rows), default=None),
            "source_working_limit": (
                int(payload["source_working_limit"])
                if payload.get("source_working_limit") is not None
                else None
            ),
            "calculation_dtype": str(payload.get("calculation_dtype", "float64")),
            "background_model_mode": str(payload.get("background_model_mode", "per_frame_local")),
            "background_rms_median_adu": float(np.median([row["background_rms_adu"] for row in frame_rows if row["background_rms_adu"] is not None])) if any(row["background_rms_adu"] is not None for row in frame_rows) else None,
            "note": (
                "candidate_count 是全量候选审计；returned_count、quality_count 和点轨迹只在每帧最高 SNR 工作集内统计。"
                if payload.get("source_working_limit") is not None
                else "当前序列使用全量源级工作集。统计排除了首行 208 字节辅助区；未设置饱和阈值，不报告饱和率。"
            ),
        },
        "photometric_evidence": _photometric_evidence_section(payload),
        "motion": {
            "feature_count": len(motion_rows),
            "moving_count": sum(row["classification"] == "moving" for row in motion_rows),
            "candidate_count": sum(row["classification"] == "candidate" for row in motion_rows),
            "reference_mode": reference_mode,
            "tracks": motion_rows,
            "frame_audits": motion_audit_rows,
            "note": f"{reference_note} speed_px_per_s 是图像平面速度；三点及以上轨迹附带小样本 OLS 速度、方向和轨迹均值外推的 95% 区间。区间不包含配准系统误差、模型失配或未来单次观测噪声；没有像元尺度/WCS 时不解释为真实天体速度。",
        },
        "frames": frame_rows,
    }
    return report


def build_innovation_report(sequence_json: str | Path) -> dict[str, Any]:
    """从 ``SequenceResult.as_dict`` JSON 构建完整序列证据报告。"""

    json_path = Path(sequence_json).resolve()
    with json_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return _build_innovation_report_payload(payload, json_path)


def build_innovation_report_from_payload(
    payload: Mapping[str, Any],
    source_hint: str | Path | None = None,
) -> dict[str, Any]:
    """从内存中的序列结果构建报告，供 Tkinter 按需导出图证。"""

    hint = Path(source_hint) if source_hint is not None else Path.cwd() / "rst19-gui-sequence.json"
    return _build_innovation_report_payload(payload, hint.resolve())


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, exclude_keys: Iterable[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = set(exclude_keys)
    clean_rows = [{key: value for key, value in row.items() if key not in excluded} for row in rows]
    if not clean_rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in clean_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_rows)


def _motion_point_rows(tracks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把 JSON 中的嵌套轨迹点展开为适合表格审计的长表。"""

    rows: list[dict[str, Any]] = []
    for track in tracks:
        track_id = int(track.get("track_id", len(rows)))
        classification = str(track.get("classification", "candidate"))
        points = track.get("points", [])
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            continue
        for point in points:
            if not isinstance(point, Mapping):
                continue
            bbox = point.get("bbox", [])
            rows.append(
                {
                    "track_id": track_id,
                    "classification": classification,
                    "frame_index": int(point.get("frame_index", 0)),
                    "frame_number": int(point.get("frame_index", 0)) + 1,
                    "x_px": _as_float(point.get("x")),
                    "y_px": _as_float(point.get("y")),
                    "aligned_x_px": _as_float(point.get("aligned_x")),
                    "aligned_y_px": _as_float(point.get("aligned_y")),
                    "length_px": _as_float(point.get("length_px")),
                    "width_px": _as_float(point.get("width_px")),
                    "angle_deg": _as_float(point.get("angle_deg")),
                    "residual_snr": _as_float(point.get("residual_snr")),
                    "fit_residual_px": _as_float(point.get("fit_residual_px")),
                    "bbox": ",".join(str(value) for value in bbox) if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) else None,
                    "touches_edge": bool(point.get("touches_edge", False)),
                }
            )
    return rows


def _normalise_cutout(data: np.ndarray) -> Image.Image:
    values = np.asarray(data, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Image.new("RGB", (max(1, values.shape[1]), max(1, values.shape[0])), "#182033")
    low, high = np.percentile(finite, [1.0, 99.7])
    if high <= low:
        high = low + 1.0
    normalised = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalised[~np.isfinite(normalised)] = 0.0
    return Image.fromarray(np.rint(normalised * 255.0).astype(np.uint8), mode="L").convert("RGB")


def _write_motion_cutouts(report: Mapping[str, Any], output: Path) -> None:
    """为每个线状候选输出原图裁剪和接触表，不把裁剪图当作真值。"""

    frame_rows = report.get("frames", [])
    tracks = report.get("motion", {}).get("tracks", [])
    if not isinstance(frame_rows, Sequence) or isinstance(frame_rows, (str, bytes)) or not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
        return
    cutout_dir = output / "motion_cutouts"
    cutout_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    contact_tiles: list[tuple[Image.Image, dict[str, Any]]] = []
    frame_cache: dict[int, np.ndarray] = {}
    for track in tracks:
        if not isinstance(track, Mapping):
            continue
        track_id = int(track.get("track_id", len(summary_rows)))
        classification = str(track.get("classification", "candidate"))
        points = track.get("points", [])
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            continue
        for point in points:
            if not isinstance(point, Mapping):
                continue
            frame_index = int(point.get("frame_index", 0))
            if not 0 <= frame_index < len(frame_rows):
                continue
            raw_path = frame_rows[frame_index].get("path") if isinstance(frame_rows[frame_index], Mapping) else None
            if raw_path is None:
                continue
            if frame_index not in frame_cache:
                frame_cache[frame_index] = read_fits(Path(str(raw_path))).data
            data = frame_cache[frame_index]
            bbox = point.get("bbox", [])
            if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) >= 4:
                x_min, y_min, x_max, y_max = (int(float(value)) for value in bbox[:4])
            else:
                center_x = int(round(float(point.get("x", 0.0))))
                center_y = int(round(float(point.get("y", 0.0))))
                half = max(16, int(round(float(point.get("length_px", 20.0)) / 2.0)))
                x_min, y_min, x_max, y_max = center_x - half, center_y - half, center_x + half, center_y + half
            margin = max(12, int(round(float(point.get("width_px", 3.0)) * 3.0)))
            left = max(0, x_min - margin)
            top = max(0, y_min - margin)
            right = min(data.shape[1], x_max + margin + 1)
            bottom = min(data.shape[0], y_max + margin + 1)
            if right <= left or bottom <= top:
                continue
            crop = _normalise_cutout(data[top:bottom, left:right])
            draw = ImageDraw.Draw(crop)
            center_x = float(point.get("x", 0.0)) - left
            center_y = float(point.get("y", 0.0)) - top
            angle = math.radians(float(point.get("angle_deg", 0.0)))
            half_length = 0.5 * float(point.get("length_px", 0.0))
            dx = math.cos(angle) * half_length
            dy = math.sin(angle) * half_length
            colour = "#ff3bd4" if classification == "moving" else "#ef7d45"
            draw.line((center_x - dx, center_y - dy, center_x + dx, center_y + dy), fill="#182033", width=5)
            draw.line((center_x - dx, center_y - dy, center_x + dx, center_y + dy), fill=colour, width=2)
            draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), outline=colour, width=2)
            residual = point.get("fit_residual_px")
            residual_text = f"fit {float(residual):.2f}px" if residual is not None else "fit —"
            draw.rectangle((3, 3, min(crop.width - 3, 190), 18), fill="#182033")
            draw.text((6, 5), f"TRAIL {track_id:04d} · F{frame_index + 1:02d} · {residual_text}", fill="#f4f0e7", font=ImageFont.load_default())
            filename = f"track_{track_id:04d}_frame_{frame_index + 1:02d}.png"
            path = cutout_dir / filename
            crop.save(path)
            summary = {
                "track_id": track_id,
                "classification": classification,
                "frame_index": frame_index,
                "timestamp": frame_rows[frame_index].get("timestamp") if isinstance(frame_rows[frame_index], Mapping) else None,
                "x_px": float(point.get("x", 0.0)),
                "y_px": float(point.get("y", 0.0)),
                "length_px": _as_float(point.get("length_px")),
                "width_px": _as_float(point.get("width_px")),
                "angle_deg": _as_float(point.get("angle_deg")),
                "residual_snr": _as_float(point.get("residual_snr")),
                "fit_residual_px": _as_float(point.get("fit_residual_px")),
                "touches_edge": bool(point.get("touches_edge", False)),
                "crop_left": left,
                "crop_top": top,
                "crop_right": right,
                "crop_bottom": bottom,
                "path": str(Path("motion_cutouts") / filename),
            }
            summary_rows.append(summary)
            tile = crop.copy()
            tile.thumbnail((300, 205), Image.Resampling.LANCZOS)
            contact_tiles.append((tile, summary))
    _write_csv(output / "motion_cutouts.csv", summary_rows)
    if not contact_tiles:
        return
    columns = 3
    tile_width, tile_height = 320, 240
    rows_count = math.ceil(len(contact_tiles) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "#182033")
    sheet_draw = ImageDraw.Draw(sheet)
    for index, (tile, summary) in enumerate(contact_tiles):
        column = index % columns
        row = index // columns
        x = column * tile_width + (tile_width - tile.width) // 2
        y = row * tile_height + 24 + (205 - tile.height) // 2
        sheet.paste(tile, (x, y))
        label = f"TRAIL {int(summary['track_id']):04d} · F{int(summary['frame_index']) + 1:02d} · {summary['classification']}"
        sheet_draw.text((column * tile_width + 8, row * tile_height + 7), label, fill="#f4f0e7", font=ImageFont.load_default())
    sheet.save(output / "motion_cutout_contact_sheet.png")


def _chart_canvas(width: int = 1200, height: int = 680) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "#f4f0e7")
    return image, ImageDraw.Draw(image)


def _line_chart(
    path: Path,
    title: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    y_label: str,
    x_label: str,
    include_zero: bool = True,
) -> None:
    image, draw = _chart_canvas()
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    left, top, right, bottom = 105, 74, image.width - 50, image.height - 82
    draw.text((left, 22), title, fill="#20293d", font=title_font)
    draw.line((left, top, left, bottom), fill="#626b7d", width=2)
    draw.line((left, bottom, right, bottom), fill="#626b7d", width=2)
    finite_y = [float(value) for _name, values, _color in series for value in values if math.isfinite(float(value))]
    if not x_values or not finite_y:
        image.save(path)
        return
    x_min, x_max = min(x_values), max(x_values)
    raw_y_min, raw_y_max = min(finite_y), max(finite_y)
    y_min = min(0.0, raw_y_min) if include_zero else raw_y_min
    y_max = raw_y_max
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    elif not include_zero:
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding
    def format_tick(value: float) -> str:
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    for tick_index in range(6):
        fraction = tick_index / 5.0
        tick_value = y_max - fraction * (y_max - y_min)
        sy = top + fraction * (bottom - top)
        draw.line((left + 1, sy, right, sy), fill="#ded8ca", width=1)
        tick_text = format_tick(tick_value)
        tick_box = draw.textbbox((0, 0), tick_text, font=font)
        draw.text((left - 10 - (tick_box[2] - tick_box[0]), sy - 7), tick_text, fill="#626b7d", font=font)
    x_ticks = list(dict.fromkeys(float(value) for value in x_values))
    if len(x_ticks) > 20:
        step = max(1, math.ceil(len(x_ticks) / 10))
        x_ticks = x_ticks[::step]
        if x_ticks[-1] != float(x_values[-1]):
            x_ticks.append(float(x_values[-1]))
    for x_value in x_ticks:
        sx = left + (x_value - x_min) / (x_max - x_min) * (right - left)
        draw.line((sx, bottom, sx, bottom + 5), fill="#626b7d", width=1)
        tick_text = f"{x_value:g}"
        tick_box = draw.textbbox((0, 0), tick_text, font=font)
        draw.text((sx - (tick_box[2] - tick_box[0]) / 2, bottom + 8), tick_text, fill="#626b7d", font=font)
    draw.text((right - 38, bottom + 36), x_label, fill="#626b7d", font=font)
    draw.text((8, top - 22), y_label, fill="#626b7d", font=font)
    for name, values, color in series:
        points = []
        for x_value, y_value in zip(x_values, values, strict=False):
            if not math.isfinite(float(y_value)):
                continue
            sx = left + (float(x_value) - x_min) / (x_max - x_min) * (right - left)
            sy = bottom - (float(y_value) - y_min) / (y_max - y_min) * (bottom - top)
            points.append((sx, sy))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        legend_x = right - 180
        legend_y = top + 18 * series.index((name, values, color))
        draw.line((legend_x, legend_y + 5, legend_x + 18, legend_y + 5), fill=color, width=3)
        draw.text((legend_x + 25, legend_y), name, fill="#273147", font=font)
    image.save(path)


def _motion_audit_chart(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """绘制线状候选从连通域到最终特征的逐帧筛选漏斗。"""

    x_values = [float(row.get("frame_number", index + 1)) for index, row in enumerate(rows)]

    def values(key: str) -> list[float]:
        # 连通域数量通常是数千，而最终线状特征是 0–1 条；使用 log1p
        # 让同一张图同时看见“原始结构很多”和“最终输出很少”。CSV 仍保留
        # 原始计数，图中的纵轴明确写出变换，避免把变换后的数值当作计数。
        return [math.log1p(float(row.get(key, 0) or 0)) for row in rows]

    _line_chart(
        path,
        "MOTION AUDIT · RESIDUAL COMPONENT FILTERING (LOG1P)",
        x_values,
        (
            ("components", values("component_count"), "#d79432"),
            ("area pass", values("area_pass_count"), "#e5a94d"),
            ("geometry pass", values("geometry_pass_count"), "#72b9d4"),
            ("emitted lines", values("feature_count"), "#c7359e"),
        ),
        y_label="log1p(count)",
        x_label="frame",
        include_zero=False,
    )


def _trajectory_chart(path: Path, tracks: Sequence[Mapping[str, Any]]) -> None:
    image, draw = _chart_canvas()
    font = ImageFont.load_default()
    left, top, right, bottom = 95, 72, image.width - 165, image.height - 78
    draw.text((left, 22), "CROSS-FRAME MOTION · CONSTANT-VELOCITY OLS", fill="#20293d", font=font)
    moving = [track for track in tracks if track.get("classification") == "moving"]
    candidates = moving or [track for track in tracks if track.get("classification") == "candidate"]
    coords = [
        float(track[key])
        for track in candidates
        for key in ("start_x_px", "end_x_px", "predicted_x_px", "start_y_px", "end_y_px", "predicted_y_px")
        if track.get(key) is not None
    ]
    if not candidates or not coords:
        image.save(path)
        return
    xs = [float(track[key]) for track in candidates for key in ("start_x_px", "end_x_px", "predicted_x_px") if track.get(key) is not None]
    ys = [float(track[key]) for track in candidates for key in ("start_y_px", "end_y_px", "predicted_y_px") if track.get(key) is not None]
    for track in candidates:
        predicted_x = _as_float(track.get("predicted_x_px"))
        predicted_y = _as_float(track.get("predicted_y_px"))
        x_half = _as_float(track.get("predicted_x_ci95_half_width_px"))
        y_half = _as_float(track.get("predicted_y_ci95_half_width_px"))
        if predicted_x is not None and x_half is not None:
            xs.extend((predicted_x - x_half, predicted_x + x_half))
        if predicted_y is not None and y_half is not None:
            ys.extend((predicted_y - y_half, predicted_y + y_half))
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    x_padding = max(1.0, 0.06 * (x_max - x_min))
    y_padding = max(1.0, 0.08 * (y_max - y_min))
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    draw.line((left, top, left, bottom), fill="#626b7d", width=1)
    draw.line((left, bottom, right, bottom), fill="#626b7d", width=1)
    for tick in range(6):
        fraction = tick / 5.0
        sx = left + fraction * (right - left)
        sy = top + fraction * (bottom - top)
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        draw.line((sx, top, sx, bottom), fill="#ded8ca", width=1)
        draw.line((left, sy, right, sy), fill="#ded8ca", width=1)
        draw.text((sx - 18, bottom + 10), f"{x_value:.0f}", fill="#626b7d", font=font)
        draw.text((left - 48, sy - 5), f"{y_value:.0f}", fill="#626b7d", font=font)
    draw.text(((left + right) / 2 - 38, bottom + 35), "aligned X / px", fill="#273147", font=font)
    draw.text((12, top - 20), "aligned Y / px (down)", fill="#273147", font=font)

    def project(x_value: float, y_value: float) -> tuple[float, float]:
        return (
            left + (x_value - x_min) / (x_max - x_min) * (right - left),
            top + (y_value - y_min) / (y_max - y_min) * (bottom - top),
        )

    for track in candidates:
        x0, x1 = float(track["start_x_px"]), float(track["end_x_px"])
        y0, y1 = float(track["start_y_px"]), float(track["end_y_px"])
        sx0, sy0 = project(x0, y0)
        sx1, sy1 = project(x1, y1)
        color = "#d79432" if track.get("classification") == "candidate" else "#c7359e"
        fitted_start_x = _as_float(track.get("fitted_start_x_px"))
        fitted_start_y = _as_float(track.get("fitted_start_y_px"))
        fitted_end_x = _as_float(track.get("fitted_end_x_px"))
        fitted_end_y = _as_float(track.get("fitted_end_y_px"))
        if None not in (fitted_start_x, fitted_start_y, fitted_end_x, fitted_end_y):
            fit_start = project(float(fitted_start_x), float(fitted_start_y))
            fit_end = project(float(fitted_end_x), float(fitted_end_y))
            draw.line((*fit_start, *fit_end), fill="#20293d", width=2)
        else:
            draw.line((sx0, sy0, sx1, sy1), fill="#20293d", width=2)
        points = track.get("points", [])
        if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
            for point_index, point in enumerate(points):
                if not isinstance(point, Mapping):
                    continue
                point_x = _as_float(point.get("aligned_x", point.get("x")))
                point_y = _as_float(point.get("aligned_y", point.get("y")))
                if point_x is None or point_y is None:
                    continue
                sx, sy = project(point_x, point_y)
                draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), fill=color, outline="#20293d", width=1)
                draw.text((sx + 7, sy + 4), f"F{int(point.get('frame_index', point_index)) + 1:02d}", fill="#626b7d", font=font)
        draw.text((right + 12, top + 18 * candidates.index(track)), f"ID {track.get('track_id')} · {track.get('classification')}", fill=color, font=font)
        predicted_x = track.get("predicted_x_px")
        predicted_y = track.get("predicted_y_px")
        if predicted_x is not None and predicted_y is not None:
            px, py = project(float(predicted_x), float(predicted_y))
            fit_end_x = fitted_end_x if fitted_end_x is not None else x1
            fit_end_y = fitted_end_y if fitted_end_y is not None else y1
            forecast_start = project(float(fit_end_x), float(fit_end_y))
            draw.line((*forecast_start, px, py), fill="#72b9d4", width=2)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline="#72b9d4", width=2)
            x_half = _as_float(track.get("predicted_x_ci95_half_width_px"))
            y_half = _as_float(track.get("predicted_y_ci95_half_width_px"))
            if x_half is not None and y_half is not None:
                rx = x_half / (x_max - x_min) * (right - left)
                ry = y_half / (y_max - y_min) * (bottom - top)
                draw.ellipse((px - rx, py - ry, px + rx, py + ry), outline="#4f9b83", width=2)
                label_x = min(px + 8, right - 150)
                draw.text((label_x, py - 18), "forecast / OLS mean 95% CI", fill="#626b7d", font=font)
            else:
                label_x = min(px + 8, right - 150)
                draw.text((label_x, py - 18), "forecast / uncertainty unavailable", fill="#626b7d", font=font)
    draw.text((left, image.height - 25), "Points: observed centroids · dark line: OLS fit · blue: extrapolation · green ellipse: axis-wise mean 95% CI", fill="#626b7d", font=font)
    image.save(path)


def _telemetry_position_chart(
    path: Path,
    frame_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> None:
    """绘制两种参考系相对首帧的 XY 位置轨迹和末帧外推。"""

    image, draw = _chart_canvas()
    font = ImageFont.load_default()
    draw.text((90, 22), "AUXILIARY TELEMETRY · POSITION DELTA / XY PROJECTION", fill="#20293d", font=font)
    panel_specs = (("j2000", "J2000", "#c7359e"), ("wgs84", "WGS84", "#72b9d4"))
    panel_width = (image.width - 90 - 50 - 30) / 2.0
    for panel_index, (prefix, label, color) in enumerate(panel_specs):
        left = 90 + panel_index * (panel_width + 30)
        right = left + panel_width
        top, bottom = 70, image.height - 70
        points: list[tuple[float, float]] = []
        for row in frame_rows:
            values = [_as_float(row.get(f"{prefix}_{axis}")) for axis in ("x", "y")]
            if all(value is not None for value in values):
                points.append((float(values[0]), float(values[1])))
        prediction = predictions.get(prefix, {})
        predicted_values = prediction.get("predicted_position_m_assuming_m_per_s")
        if not points:
            draw.rectangle((left, top, right, bottom), outline="#ded8ca", fill="#f4f0e7")
            draw.text((left + 12, top + 12), f"{label} · no valid position", fill="#626b7d", font=font)
            continue
        origin = np.asarray(points[0], dtype=np.float64)
        delta = np.asarray(points, dtype=np.float64) - origin
        plotted = [tuple(float(value) for value in row) for row in delta]
        if isinstance(predicted_values, Sequence) and not isinstance(predicted_values, (str, bytes)) and len(predicted_values) >= 2:
            last = np.asarray(points[-1], dtype=np.float64)
            predicted_delta = np.asarray([float(predicted_values[0]), float(predicted_values[1])], dtype=np.float64) - origin
            plotted.append((float(predicted_delta[0]), float(predicted_delta[1])))
        else:
            predicted_delta = None
        x_values = [point[0] for point in plotted]
        y_values = [point[1] for point in plotted]
        x_min, x_max = min(0.0, min(x_values)), max(0.0, max(x_values))
        y_min, y_max = min(0.0, min(y_values)), max(0.0, max(y_values))
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        x_margin = 0.08 * x_span
        y_margin = 0.08 * y_span
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin
        x_span = x_max - x_min
        y_span = y_max - y_min

        def project(point: tuple[float, float]) -> tuple[float, float]:
            return (
                left + (point[0] - x_min) / x_span * (right - left),
                bottom - (point[1] - y_min) / y_span * (bottom - top),
            )

        draw.rectangle((left, top, right, bottom), outline="#ded8ca", fill="#f4f0e7")
        draw.text((left + 10, top + 9), f"{label} · ΔX/ΔY from first frame", fill="#273147", font=font)
        observed_points = [project(point) for point in plotted[: len(points)]]
        if len(observed_points) >= 2:
            draw.line(observed_points, fill=color, width=3)
        for index, point in enumerate(observed_points):
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
            if index in {0, len(observed_points) - 1}:
                draw.text((point[0] + 7, point[1] - 7), f"F{index + 1:02d}" if index == 0 else "F15", fill="#626b7d", font=font)
        if predicted_delta is not None:
            last_point = project(plotted[len(points) - 1])
            predicted_point = project((float(predicted_delta[0]), float(predicted_delta[1])))
            draw.line((last_point, predicted_point), fill="#d79432", width=2)
            draw.ellipse((predicted_point[0] - 5, predicted_point[1] - 5, predicted_point[0] + 5, predicted_point[1] + 5), outline="#d79432", width=2)
            draw.text((predicted_point[0] + 7, predicted_point[1] - 7), "forecast", fill="#626b7d", font=font)
        draw.text((left + 8, bottom - 18), "ΔX / m?", fill="#626b7d", font=font)
        draw.text((left + 8, top + 27), "ΔY / m?", fill="#626b7d", font=font)
    image.save(path)


def _registered_stack(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    max_width: int = 1200,
    normalize: bool = True,
) -> tuple[list[np.ndarray], int, int]:
    """读取并按累计平移对齐帧，返回图像栈和原始尺寸。"""

    if not frame_rows:
        return [], 0, 0
    first_frame = read_fits(frame_rows[0]["path"])
    raw_height, raw_width = first_frame.data.shape
    width = min(int(max_width), raw_width)
    height = max(1, round(raw_height * width / raw_width))
    low_values = [float(row["p01_adu"]) for row in frame_rows if row.get("p01_adu") is not None]
    high_values = [float(row["p998_adu"]) for row in frame_rows if row.get("p998_adu") is not None]
    low = float(np.median(low_values)) if low_values else 0.0
    high = float(np.median(high_values)) if high_values else low + 1.0
    if high <= low:
        high = low + 1.0
    stack: list[np.ndarray] = []
    scale_x = width / raw_width
    scale_y = height / raw_height
    for row in frame_rows:
        frame = first_frame if row["path"] == frame_rows[0]["path"] else read_fits(row["path"])
        values = np.asarray(frame.data, dtype=np.float32)
        valid = np.isfinite(values) & ~auxiliary_mask(values.shape)
        source = np.clip((values - low) / (high - low), 0.0, 1.0) if normalize else values
        source = np.where(valid, source, 0.0).astype(np.float32, copy=False)
        thumbnail = np.asarray(
            Image.fromarray(source, mode="F").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).copy()
        valid_thumbnail = np.asarray(
            Image.fromarray(valid.astype(np.float32), mode="F").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        thumbnail[valid_thumbnail < 0.5] = np.nan
        shift_x = float(row.get("cumulative_shift_x_px") or 0.0) * scale_x
        shift_y = float(row.get("cumulative_shift_y_px") or 0.0) * scale_y
        aligned = ndimage.shift(thumbnail, shift=(-shift_y, -shift_x), order=1, mode="constant", cval=np.nan, prefilter=False)
        stack.append(aligned)
    return stack, raw_width, raw_height


def _difference_rows(
    frame_rows: Sequence[Mapping[str, Any]],
    registered_stack: Sequence[np.ndarray],
    reference: np.ndarray,
) -> list[dict[str, Any]]:
    """计算每帧相对注册中值参考图的残差统计。"""

    rows: list[dict[str, Any]] = []
    for row, aligned in zip(frame_rows, registered_stack, strict=True):
        residual = np.asarray(aligned, dtype=np.float64) - reference
        finite = np.isfinite(residual)
        values = residual[finite]
        noise = max(_as_float(row.get("background_rms_adu")) or 1.0, np.finfo(np.float64).eps)
        if values.size == 0:
            rows.append(
                {
                    "frame_index": int(row.get("frame_index", len(rows))),
                    "timestamp": row.get("timestamp"),
                    "residual_median_adu": None,
                    "residual_rms_adu": None,
                    "residual_p99_abs_adu": None,
                    "positive_peak_sigma": None,
                    "negative_peak_sigma": None,
                    "pixels_abs_gt_5sigma": 0,
                    "valid_fraction": 0.0,
                }
            )
            continue
        residual_median = float(np.median(values))
        residual_mad = float(np.median(np.abs(values - residual_median)))
        abs_values = np.abs(values)
        rows.append(
            {
                "frame_index": int(row.get("frame_index", len(rows))),
                "timestamp": row.get("timestamp"),
                "residual_median_adu": residual_median,
                "residual_rms_adu": float(1.4826 * residual_mad),
                "residual_p99_abs_adu": float(np.percentile(abs_values, 99.0)),
                "positive_peak_sigma": float(np.max(values) / noise),
                "negative_peak_sigma": float(np.min(values) / noise),
                "pixels_abs_gt_5sigma": int(np.count_nonzero(abs_values >= 5.0 * noise)),
                "valid_fraction": float(values.size / residual.size),
            }
        )
    return rows


def _difference_image(registered_stack: Sequence[np.ndarray], reference: np.ndarray) -> Image.Image:
    """把所有帧的最大绝对注册残差渲染为正/负残差图。"""

    if not registered_stack:
        return Image.new("RGB", (1, 1), "#182033")
    stack = np.stack(registered_stack, axis=0).astype(np.float64, copy=False)
    residual_stack = stack - reference
    positive = np.where(np.isfinite(residual_stack) & (residual_stack > 0.0), residual_stack, 0.0)
    negative = np.where(np.isfinite(residual_stack) & (residual_stack < 0.0), residual_stack, 0.0)
    positive_max = np.max(positive, axis=0)
    negative_min = np.min(negative, axis=0)
    signed = np.where(positive_max >= np.abs(negative_min), positive_max, negative_min)
    finite = np.abs(signed[np.isfinite(signed)])
    clip = float(np.percentile(finite, 99.8)) if finite.size else 1.0
    clip = max(clip, np.finfo(np.float64).eps)
    strength = np.clip(np.abs(signed) / clip, 0.0, 1.0)
    neutral = np.asarray((20.0, 28.0, 44.0), dtype=np.float64)
    positive_color = np.asarray((239.0, 111.0, 70.0), dtype=np.float64)
    negative_color = np.asarray((73.0, 145.0, 221.0), dtype=np.float64)
    colors = np.where((signed >= 0.0)[..., None], positive_color, negative_color)
    rgb = neutral + (colors - neutral) * strength[..., None]
    rgb[~np.isfinite(rgb)] = neutral[0]
    return Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _registered_difference(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    max_width: int = 1200,
) -> tuple[list[dict[str, Any]], Image.Image]:
    """生成注册中值差分统计及一张正/负残差图。"""

    registered_stack, _raw_width, _raw_height = _registered_stack(frame_rows, max_width=max_width, normalize=False)
    if not registered_stack:
        return [], Image.new("RGB", (1, 1), "#182033")
    reference = np.nanmedian(np.stack(registered_stack, axis=0), axis=0)
    return _difference_rows(frame_rows, registered_stack, reference), _difference_image(registered_stack, reference)


def _registered_mosaic(path: Path, frame_rows: Sequence[Mapping[str, Any]], *, max_width: int = 1200) -> None:
    """生成基于平移配准基线的中值拼图；仅用于可视化和一致性检查。"""

    registered_stack, _raw_width, _raw_height = _registered_stack(frame_rows, max_width=max_width)
    if not registered_stack:
        return
    composite = np.nanmedian(np.stack(registered_stack, axis=0), axis=0)
    composite = np.clip(np.nan_to_num(composite, nan=0.0) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(composite, mode="L").convert("RGB").save(path)


def write_innovation_artifacts(report: Mapping[str, Any], out_dir: str | Path) -> Path:
    """写出 JSON、逐帧/逐轨迹/差分 CSV 和图像证据，返回输出目录。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "frame_evidence.csv", report.get("frames", []))
    telemetry_prediction_payload = report.get("telemetry_predictions", {})
    predictions: Mapping[str, Mapping[str, Any]] = {}
    if isinstance(telemetry_prediction_payload, Mapping):
        raw_predictions = telemetry_prediction_payload.get("predictions", {})
        if isinstance(raw_predictions, Mapping):
            predictions = raw_predictions
            _write_csv(output / "telemetry_prediction.csv", _telemetry_prediction_rows(predictions))
    motion_tracks = report.get("motion", {}).get("tracks", [])
    _write_csv(output / "motion_evidence.csv", motion_tracks, exclude_keys=("points",))
    _write_csv(output / "motion_points_evidence.csv", _motion_point_rows(motion_tracks))
    motion_audits = report.get("motion", {}).get("frame_audits", [])
    _write_csv(output / "motion_frame_audit.csv", motion_audits)
    _motion_audit_chart(output / "motion_frame_audit.png", motion_audits)
    _write_motion_cutouts(report, output)
    frames = report.get("frames", [])
    difference_rows, difference_image = _registered_difference(frames)
    _write_csv(output / "difference_evidence.csv", difference_rows)
    difference_image.save(output / "registered_difference.png")
    report_output = dict(report)
    residual_rms = [row["residual_rms_adu"] for row in difference_rows if row.get("residual_rms_adu") is not None]
    report_output["difference"] = {
        "frame_count": len(difference_rows),
        "residual_rms_median_adu": float(np.median(residual_rms)) if residual_rms else None,
        "note": "这是平移配准后相对中值参考的残差基线；未做 PSF matching 或 proper image subtraction。",
    }
    (output / "innovation_report.json").write_text(json.dumps(report_output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    x_values = [float(row.get("frame_index", index)) + 1 for index, row in enumerate(frames)]
    quality_summary = report.get("quality", {})
    source_working_limit = quality_summary.get("source_working_limit") if isinstance(quality_summary, Mapping) else None
    quality_title = (
        f"FRAME QUALITY · ALL CANDIDATES VS WORKSET (LOG1P, LIMIT {int(source_working_limit):,})"
        if source_working_limit is not None
        else "FRAME QUALITY · CANDIDATES VS QUALITY SOURCES (LOG1P)"
    )
    quality_label = "quality / workset" if source_working_limit is not None else "quality"
    _line_chart(
        output / "quality_counts.png",
        quality_title,
        x_values,
        (
            ("candidate", [math.log1p(float(row.get("candidate_count", 0))) for row in frames], "#d79432"),
            (quality_label, [math.log1p(float(row.get("quality_count", 0))) for row in frames], "#4f9b83"),
        ),
        y_label="log1p(count)",
        x_label="frame",
        include_zero=False,
    )
    _line_chart(
        output / "registration_shift.png",
        "REGISTRATION BASELINE · CUMULATIVE SHIFT",
        x_values,
        (
            ("shift norm px", [float(row.get("cumulative_shift_norm_px", 0.0)) for row in frames], "#72b9d4"),
        ),
        y_label="pixel",
        x_label="frame",
    )
    telemetry_rows = report.get("frames", [])
    _line_chart(
        output / "telemetry_velocity.png",
        "AUXILIARY TELEMETRY · VECTOR NORM (m/s?)",
        x_values,
        (
            ("J2000 |v|", [float(row.get("j2000_velocity_norm_raw")) if row.get("j2000_velocity_norm_raw") is not None else math.nan for row in telemetry_rows], "#c7359e"),
            ("WGS84 |v|", [float(row.get("wgs84_velocity_norm_raw")) if row.get("wgs84_velocity_norm_raw") is not None else math.nan for row in telemetry_rows], "#72b9d4"),
        ),
        y_label="m/s?",
        x_label="frame",
    )
    _telemetry_position_chart(output / "telemetry_position.png", telemetry_rows, predictions)
    _trajectory_chart(output / "motion_trajectory.png", report.get("motion", {}).get("tracks", []))
    _registered_mosaic(output / "registered_mosaic.png", telemetry_rows)
    return output
