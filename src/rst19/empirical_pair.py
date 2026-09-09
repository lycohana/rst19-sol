"""经验 PSF 对近邻候选的形状复核。

这个模块是研究审计工具，不参与默认候选计数、质量判定或 GUI 缓存。它把
首帧源表中的干净隔离源叠加成经验 PSF，然后在指定的两个 detector 位置上
逐帧计算形状相似度和相对残差。局部模板不足时明确返回 ``None``，不把
“没有模板”静默解释成“不是星”。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .detection import Detection
from .experiments import EmpiricalPSF, _psf_similarity_metrics, estimate_empirical_psf
from .fits import FitsFrame, auxiliary_mask, read_fits


@dataclass(frozen=True, slots=True)
class EmpiricalPSFPairAuditRow:
    """一帧中两个目标相对经验 PSF 的形状诊断。"""

    frame_index: int
    path: str
    frame_shift_x_px: float
    frame_shift_y_px: float
    primary_x: float
    primary_y: float
    secondary_x: float
    secondary_y: float
    primary_background_adu: float | None
    secondary_background_adu: float | None
    global_psf_source_count: int
    local_psf_source_count: int
    local_template_available: bool
    primary_global_correlation: float | None
    primary_global_relative_residual: float | None
    primary_global_central_energy_fraction: float | None
    primary_global_fitted_amplitude_adu: float | None
    secondary_global_correlation: float | None
    secondary_global_relative_residual: float | None
    secondary_global_central_energy_fraction: float | None
    secondary_global_fitted_amplitude_adu: float | None
    primary_local_correlation: float | None
    primary_local_relative_residual: float | None
    primary_local_central_energy_fraction: float | None
    primary_local_fitted_amplitude_adu: float | None
    secondary_local_correlation: float | None
    secondary_local_relative_residual: float | None
    secondary_local_central_energy_fraction: float | None
    secondary_local_fitted_amplitude_adu: float | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "frame_shift_x_px": self.frame_shift_x_px,
            "frame_shift_y_px": self.frame_shift_y_px,
            "primary_x": self.primary_x,
            "primary_y": self.primary_y,
            "secondary_x": self.secondary_x,
            "secondary_y": self.secondary_y,
            "primary_background_adu": self.primary_background_adu,
            "secondary_background_adu": self.secondary_background_adu,
            "global_psf_source_count": self.global_psf_source_count,
            "local_psf_source_count": self.local_psf_source_count,
            "local_template_available": self.local_template_available,
            "primary_global_correlation": self.primary_global_correlation,
            "primary_global_relative_residual": self.primary_global_relative_residual,
            "primary_global_central_energy_fraction": self.primary_global_central_energy_fraction,
            "primary_global_fitted_amplitude_adu": self.primary_global_fitted_amplitude_adu,
            "secondary_global_correlation": self.secondary_global_correlation,
            "secondary_global_relative_residual": self.secondary_global_relative_residual,
            "secondary_global_central_energy_fraction": self.secondary_global_central_energy_fraction,
            "secondary_global_fitted_amplitude_adu": self.secondary_global_fitted_amplitude_adu,
            "primary_local_correlation": self.primary_local_correlation,
            "primary_local_relative_residual": self.primary_local_relative_residual,
            "primary_local_central_energy_fraction": self.primary_local_central_energy_fraction,
            "primary_local_fitted_amplitude_adu": self.primary_local_fitted_amplitude_adu,
            "secondary_local_correlation": self.secondary_local_correlation,
            "secondary_local_relative_residual": self.secondary_local_relative_residual,
            "secondary_local_central_energy_fraction": self.secondary_local_central_energy_fraction,
            "secondary_local_fitted_amplitude_adu": self.secondary_local_fitted_amplitude_adu,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalPSFPairAuditResult:
    """两个近邻候选的经验 PSF 逐帧审计结果。"""

    frame_count: int
    primary_detection_id: int | None
    secondary_detection_id: int | None
    primary_target_xy: tuple[float, float]
    secondary_target_xy: tuple[float, float]
    frame_shifts: tuple[tuple[float, float], ...]
    template_frame_path: str
    template_catalog_path: str | None
    support_radius: int
    grid_size: int
    local_padding_px: float
    global_psf: EmpiricalPSF | None
    local_psf: EmpiricalPSF | None
    parameters: dict[str, object]
    rows: tuple[EmpiricalPSFPairAuditRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_target_xy": list(self.primary_target_xy),
            "secondary_target_xy": list(self.secondary_target_xy),
            "frame_shifts": [list(shift) for shift in self.frame_shifts],
            "template_frame_path": self.template_frame_path,
            "template_catalog_path": self.template_catalog_path,
            "support_radius": self.support_radius,
            "grid_size": self.grid_size,
            "local_padding_px": self.local_padding_px,
            "global_psf": None if self.global_psf is None else self.global_psf.as_dict(),
            "local_psf": None if self.local_psf is None else self.local_psf.as_dict(),
            "parameters": self.parameters,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
        }


def _optional_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    if value is None or not str(value).strip():
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"source catalog field {name!r} is not numeric: {value!r}") from exc
    return converted if np.isfinite(converted) else None


def _required_float(row: dict[str, str], name: str) -> float:
    value = _optional_float(row, name)
    if value is None:
        raise ValueError(f"source catalog field {name!r} is required")
    return value


def _optional_int(row: dict[str, str], name: str) -> int | None:
    value = row.get(name, "")
    if value is None or not str(value).strip():
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"source catalog field {name!r} is not an integer: {value!r}") from exc


def _pipe_values(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split("|") if item.strip())


def _pipe_floats(value: str | None) -> tuple[float, ...]:
    values: list[float] = []
    for item in _pipe_values(value):
        try:
            converted = float(item)
        except ValueError as exc:
            raise ValueError(f"source catalog list contains a non-numeric value: {item!r}") from exc
        if not np.isfinite(converted):
            raise ValueError(f"source catalog list contains a non-finite value: {item!r}")
        values.append(converted)
    return tuple(values)


def _pipe_ints(value: str | None) -> tuple[int, ...]:
    values: list[int] = []
    for item in _pipe_values(value):
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"source catalog list contains a non-integer value: {item!r}") from exc
    return tuple(values)


def _parse_bool(value: str | None, *, default: bool = True) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"true", "1", "yes", "y", "是"}:
        return True
    if text in {"false", "0", "no", "n", "否"}:
        return False
    raise ValueError(f"source catalog boolean value is invalid: {value!r}")


def load_detection_catalog_csv(path: str | Path) -> tuple[Detection, ...]:
    """读取 ``rst19-sources`` 产生的全量 ``source_catalog.csv``。

    这是研究 CLI 的输入适配器，不把 CSV 当作跨帧身份表；检测 ID 仍只在
    生成该表的单次检测运行内有意义。
    """

    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"detection_id", "x", "y", "peak", "flux", "background", "noise", "snr"}
        missing = sorted(required_columns - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        sources: list[Detection] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                detection_id = _optional_int(row, "detection_id")
                if detection_id is None:
                    raise ValueError("detection_id is required")
                sources.append(
                    Detection(
                        detection_id=detection_id,
                        x=_required_float(row, "x"),
                        y=_required_float(row, "y"),
                        peak=_required_float(row, "peak"),
                        flux=_required_float(row, "flux"),
                        background=_required_float(row, "background"),
                        noise=_required_float(row, "noise"),
                        snr=_required_float(row, "snr"),
                        fwhm=_optional_float(row, "fwhm"),
                        flags=_pipe_values(row.get("flags")),
                        flux_error=_optional_float(row, "flux_error"),
                        flux_snr=_optional_float(row, "flux_snr"),
                        filter_snr=_optional_float(row, "filter_snr"),
                        fwhm_x=_optional_float(row, "fwhm_x"),
                        fwhm_y=_optional_float(row, "fwhm_y"),
                        ellipticity=_optional_float(row, "ellipticity"),
                        sharpness=_optional_float(row, "sharpness"),
                        footprint_pixels=_optional_int(row, "footprint_pixels"),
                        psf_support_pixels=_optional_int(row, "psf_support_pixels"),
                        quality_passed=_parse_bool(row.get("quality_passed")),
                        peak_x=_optional_float(row, "peak_x"),
                        peak_y=_optional_float(row, "peak_y"),
                        centroid_shift_px=_optional_float(row, "centroid_shift_px"),
                        proposal_methods=_pipe_values(row.get("proposal_methods")),
                        proposal_scales=_pipe_floats(row.get("proposal_scales")),
                        proposal_snr=_optional_float(row, "proposal_snr"),
                        nearest_gaussian_px=_optional_float(row, "nearest_gaussian_px"),
                        deblend_delta_bic=_optional_float(row, "deblend_delta_bic"),
                        deblend_component_snr=_optional_float(row, "deblend_component_snr"),
                        repeated_code_count=_optional_int(row, "repeated_code_count"),
                        range_anomaly_pixel_count=_optional_int(row, "range_anomaly_pixel_count"),
                        repeated_code_values=_pipe_ints(row.get("repeated_code_values")),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"invalid source catalog row {line_number}: {exc}") from exc
    return tuple(sources)


def _xy(point: Sequence[float], label: str) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{label} must contain x and y")
    result = (float(point[0]), float(point[1]))
    if not all(np.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite coordinates")
    return result


def _frame_values(frame: FitsFrame | np.ndarray | str | Path) -> tuple[np.ndarray, str]:
    if isinstance(frame, FitsFrame):
        values = np.asarray(frame.data)
        label = str(frame.path)
    elif isinstance(frame, (str, Path)):
        loaded = read_fits(frame)
        values = np.asarray(loaded.data)
        label = str(loaded.path)
    else:
        values = np.asarray(frame)
        label = "<array>"
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    return values, label


def _probe_background(
    values: np.ndarray,
    x: float,
    y: float,
    support_radius: int,
    aux_mask: np.ndarray,
) -> float | None:
    """用 PSF 小窗外圈估计强制探针的局部背景。"""

    radius = max(int(support_radius) + 4, 8)
    center_x, center_y = int(round(x)), int(round(y))
    y0 = max(0, center_y - radius)
    y1 = min(values.shape[0], center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(values.shape[1], center_x + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)
    distance = np.hypot(xx + x0 - float(x), yy + y0 - float(y))
    valid = np.isfinite(values[y0:y1, x0:x1]) & ~aux_mask[y0:y1, x0:x1]
    valid &= distance >= float(support_radius)
    samples = np.asarray(values[y0:y1, x0:x1], dtype=np.float64)[valid]
    if samples.size == 0:
        return None
    return float(np.median(samples))


def _probe_source(detection_id: int, x: float, y: float, background: float | None) -> Detection:
    return Detection(
        detection_id=int(detection_id),
        x=float(x),
        y=float(y),
        peak=0.0,
        flux=0.0,
        background=0.0 if background is None else float(background),
        noise=1.0,
        snr=0.0,
        fwhm=None,
        flags=(),
    )


def _metric(metrics: tuple[float, float, float, float] | None, index: int) -> float | None:
    if metrics is None:
        return None
    value = float(metrics[index])
    return value if np.isfinite(value) else None


def _grid_cell(x: float, y: float, shape: tuple[int, int], grid_size: int) -> tuple[int, int]:
    height, width = shape
    column = min(grid_size - 1, max(0, int(float(x) * grid_size / width)))
    row = min(grid_size - 1, max(0, int(float(y) * grid_size / height)))
    return row, column


def run_empirical_psf_pair_audit(
    paths: Iterable[str | Path],
    primary_target_xy: tuple[float, float],
    secondary_target_xy: tuple[float, float],
    *,
    template_sources: Sequence[Detection],
    primary_detection_id: int | None = None,
    secondary_detection_id: int | None = None,
    template_frame: FitsFrame | np.ndarray | str | Path | None = None,
    template_catalog_path: str | Path | None = None,
    frame_shifts: Sequence[tuple[float, float]] | None = None,
    support_radius: int = 7,
    max_template_sources: int = 64,
    grid_size: int = 4,
    local_padding_px: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EmpiricalPSFPairAuditResult:
    """在真实 FITS 序列上比较一对候选与全局/局部经验 PSF 的形状。

    ``template_sources`` 只用于构造首帧模板；``frame_shifts`` 是 detector
    平移，不是天空坐标运动。输出的相关、残差和中心能量比例是形状诊断，
    不代表概率、星表身份或物理真值。
    """

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    primary_xy = _xy(primary_target_xy, "primary_target_xy")
    secondary_xy = _xy(secondary_target_xy, "secondary_target_xy")
    if np.hypot(secondary_xy[0] - primary_xy[0], secondary_xy[1] - primary_xy[1]) <= 1e-6:
        raise ValueError("primary and secondary target coordinates must differ")
    if support_radius < 3 or max_template_sources < 1 or grid_size < 1:
        raise ValueError("support_radius must be at least 3; source count and grid size must be positive")
    if local_padding_px is None:
        local_padding = max(18.0, float(2 * support_radius + 4))
    else:
        local_padding = float(local_padding_px)
        if not np.isfinite(local_padding) or local_padding < 0:
            raise ValueError("local_padding_px must be finite and non-negative")

    if frame_shifts is None:
        shifts = tuple((0.0, 0.0) for _path in frame_paths)
    else:
        if len(frame_shifts) != len(frame_paths):
            raise ValueError("frame_shifts length must match paths")
        shifts = tuple((float(item[0]), float(item[1])) for item in frame_shifts)
        if any(not np.isfinite(value) for shift in shifts for value in shift):
            raise ValueError("frame_shifts must contain finite values")

    if template_frame is None:
        template_values, template_label = _frame_values(frame_paths[0])
    else:
        template_values, template_label = _frame_values(template_frame)
    if not template_sources:
        raise ValueError("template_sources cannot be empty")
    global_psf = estimate_empirical_psf(
        template_values,
        template_sources,
        support_radius=int(support_radius),
        max_sources=int(max_template_sources),
        isolation_sources=template_sources,
    )

    pair_ids = {int(value) for value in (primary_detection_id, secondary_detection_id) if value is not None}
    pair_midpoint = (
        0.5 * (primary_xy[0] + secondary_xy[0]),
        0.5 * (primary_xy[1] + secondary_xy[1]),
    )
    pair_cell = _grid_cell(pair_midpoint[0], pair_midpoint[1], template_values.shape, int(grid_size))
    x_min = template_values.shape[1] * pair_cell[1] / grid_size - local_padding
    x_max = template_values.shape[1] * (pair_cell[1] + 1) / grid_size + local_padding
    y_min = template_values.shape[0] * pair_cell[0] / grid_size - local_padding
    y_max = template_values.shape[0] * (pair_cell[0] + 1) / grid_size + local_padding
    local_pool = [
        source
        for source in template_sources
        if int(source.detection_id) not in pair_ids
        and np.isfinite(float(source.x))
        and np.isfinite(float(source.y))
        and x_min <= float(source.x) < x_max
        and y_min <= float(source.y) < y_max
    ]
    local_psf = estimate_empirical_psf(
        template_values,
        local_pool,
        support_radius=int(support_radius),
        max_sources=int(max_template_sources),
        isolation_sources=template_sources,
    )

    rows: list[EmpiricalPSFPairAuditRow] = []
    for frame_index, path in enumerate(frame_paths):
        values, path_label = _frame_values(path)
        shift_x, shift_y = shifts[frame_index]
        primary_x, primary_y = primary_xy[0] + shift_x, primary_xy[1] + shift_y
        secondary_x, secondary_y = secondary_xy[0] + shift_x, secondary_xy[1] + shift_y
        aux_mask = auxiliary_mask(values.shape)
        primary_background = _probe_background(values, primary_x, primary_y, int(support_radius), aux_mask)
        secondary_background = _probe_background(values, secondary_x, secondary_y, int(support_radius), aux_mask)
        primary_probe = _probe_source(
            -1 if primary_detection_id is None else int(primary_detection_id),
            primary_x,
            primary_y,
            primary_background,
        )
        secondary_probe = _probe_source(
            -2 if secondary_detection_id is None else int(secondary_detection_id),
            secondary_x,
            secondary_y,
            secondary_background,
        )
        primary_global = (
            None if global_psf is None else _psf_similarity_metrics(values, primary_probe, global_psf, aux_mask)
        )
        secondary_global = (
            None if global_psf is None else _psf_similarity_metrics(values, secondary_probe, global_psf, aux_mask)
        )
        primary_local = (
            None if local_psf is None else _psf_similarity_metrics(values, primary_probe, local_psf, aux_mask)
        )
        secondary_local = (
            None if local_psf is None else _psf_similarity_metrics(values, secondary_probe, local_psf, aux_mask)
        )
        if global_psf is None:
            note = "全局经验 PSF 模板不足；形状诊断不可用"
        elif local_psf is None:
            note = "局部隔离模板不足；仅报告全局 PSF 形状诊断，不等于星表身份"
        else:
            note = "全局/局部经验 PSF 仅为形状诊断，不等于星表身份"
        rows.append(
            EmpiricalPSFPairAuditRow(
                frame_index=frame_index + 1,
                path=path_label,
                frame_shift_x_px=shift_x,
                frame_shift_y_px=shift_y,
                primary_x=primary_x,
                primary_y=primary_y,
                secondary_x=secondary_x,
                secondary_y=secondary_y,
                primary_background_adu=primary_background,
                secondary_background_adu=secondary_background,
                global_psf_source_count=0 if global_psf is None else int(global_psf.source_count),
                local_psf_source_count=0 if local_psf is None else int(local_psf.source_count),
                local_template_available=local_psf is not None,
                primary_global_correlation=_metric(primary_global, 0),
                primary_global_relative_residual=_metric(primary_global, 1),
                primary_global_central_energy_fraction=_metric(primary_global, 2),
                primary_global_fitted_amplitude_adu=_metric(primary_global, 3),
                secondary_global_correlation=_metric(secondary_global, 0),
                secondary_global_relative_residual=_metric(secondary_global, 1),
                secondary_global_central_energy_fraction=_metric(secondary_global, 2),
                secondary_global_fitted_amplitude_adu=_metric(secondary_global, 3),
                primary_local_correlation=_metric(primary_local, 0),
                primary_local_relative_residual=_metric(primary_local, 1),
                primary_local_central_energy_fraction=_metric(primary_local, 2),
                primary_local_fitted_amplitude_adu=_metric(primary_local, 3),
                secondary_local_correlation=_metric(secondary_local, 0),
                secondary_local_relative_residual=_metric(secondary_local, 1),
                secondary_local_central_energy_fraction=_metric(secondary_local, 2),
                secondary_local_fitted_amplitude_adu=_metric(secondary_local, 3),
                note=note,
            )
        )
        if progress is not None:
            progress(frame_index + 1, len(frame_paths))

    def median(values: Sequence[float | None]) -> float | None:
        finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
        return float(np.median(finite)) if finite else None

    primary_global_median = median([row.primary_global_correlation for row in rows])
    secondary_global_median = median([row.secondary_global_correlation for row in rows])
    primary_global_residual = median([row.primary_global_relative_residual for row in rows])
    secondary_global_residual = median([row.secondary_global_relative_residual for row in rows])
    valid_global = sum(
        row.primary_global_correlation is not None and row.secondary_global_correlation is not None
        for row in rows
    )
    local_available = sum(row.local_template_available for row in rows)
    conclusion = (
        f"经验 PSF 对照完成：{len(rows)} 帧；全局模板 "
        f"{0 if global_psf is None else global_psf.source_count} 个，局部模板 "
        f"{0 if local_psf is None else local_psf.source_count} 个。"
        f"两个目标均得到全局形状指标的帧数为 {valid_global}/{len(rows)}，"
        f"局部模板可用帧数为 {local_available}/{len(rows)}。"
        f"全局相关中位数 primary={primary_global_median}, secondary={secondary_global_median}；"
        f"相对残差中位数 primary={primary_global_residual}, secondary={secondary_global_residual}。"
        "这些指标受混叠、背景估计、空间 PSF 变化和模板选择影响，只能支持形状复核，"
        "不能单独确认或否定星表身份。"
    )
    return EmpiricalPSFPairAuditResult(
        frame_count=len(rows),
        primary_detection_id=None if primary_detection_id is None else int(primary_detection_id),
        secondary_detection_id=None if secondary_detection_id is None else int(secondary_detection_id),
        primary_target_xy=primary_xy,
        secondary_target_xy=secondary_xy,
        frame_shifts=shifts,
        template_frame_path=template_label,
        template_catalog_path=None if template_catalog_path is None else str(Path(template_catalog_path)),
        support_radius=int(support_radius),
        grid_size=int(grid_size),
        local_padding_px=float(local_padding),
        global_psf=global_psf,
        local_psf=local_psf,
        parameters={
            "support_radius": int(support_radius),
            "max_template_sources": int(max_template_sources),
            "grid_size": int(grid_size),
            "pair_cell": list(pair_cell),
            "local_padding_px": float(local_padding),
            "background_estimator": "median of valid pixels outside the PSF support radius",
            "frame_shifts": [list(shift) for shift in shifts],
            "interpretation_boundary": "empirical PSF shape diagnostic, not catalog identity or probability",
        },
        rows=tuple(rows),
        conclusion=conclusion,
    )


def write_empirical_psf_pair_audit_artifacts(
    result: EmpiricalPSFPairAuditResult,
    out_dir: str | Path,
) -> Path:
    """写经验 PSF 近邻对照 CSV、JSON 和相关曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(result.rows[0].as_dict()) if result.rows else ["frame_index", "path"]
    with (output / "empirical_pair_psf_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    payload = result.as_dict()
    payload["note"] = (
        "经验 PSF 由模板帧的质量通过、未饱和、未掩膜且隔离源构造；"
        "局部模板不足时保持 None。所有 correlation/residual/central_fraction 均为形状诊断，"
        "不是星表身份、恒星概率或独立去混叠通量。"
    )
    (output / "empirical_pair_psf_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    from .innovation import _line_chart

    frame_numbers = [float(row.frame_index) for row in result.rows]
    def series_values(name: str) -> list[float]:
        return [
            float(getattr(row, name)) if getattr(row, name) is not None else float("nan")
            for row in result.rows
        ]

    _line_chart(
        output / "empirical_pair_psf_similarity.png",
        "EMPIRICAL PSF · NEIGHBOR PAIR SHAPE DIAGNOSTIC",
        frame_numbers,
        (
            ("primary global correlation", series_values("primary_global_correlation"), "#d79432"),
            ("secondary global correlation", series_values("secondary_global_correlation"), "#c7359e"),
            ("primary local correlation", series_values("primary_local_correlation"), "#4f9b83"),
            ("secondary local correlation", series_values("secondary_local_correlation"), "#72b9d4"),
        ),
        y_label="cosine correlation",
        x_label="frame",
    )
    return output
