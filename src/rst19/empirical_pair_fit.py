"""经验 PSF 的单源/双源锚点敏感性审计。

这个模块专门检验近邻候选的双源证据是否依赖坐标锚点。所有模型使用
同一帧、同一局部窗口、同一背景平面和同一掩膜；固定位置 K=1/K=2
用于可比的 BIC 诊断，局部单源位置网格用于控制“单源中心没有放在最佳
位置”的偏差。它是研究工具，不参与默认检测、质量筛选或 GUI 缓存。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import ndimage
from scipy.optimize import nnls

from .detection import Detection
from .detection import _default_negative_overflow_limit
from .experiments import EmpiricalPSF, estimate_empirical_psf
from .fits import FitsFrame, auxiliary_mask, read_fits


_MASK_MODES = {
    "raw",
    "range_masked",
    "sentinel_masked",
    "range_sentinel_masked",
    "repeated_code_masked",
    "range_repeated_code_masked",
    "sentinel_repeated_code_masked",
    "range_sentinel_repeated_code_masked",
}


def _finite_or_none(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _xy(point: Sequence[float], label: str) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{label} must contain x and y")
    result = (float(point[0]), float(point[1]))
    if not all(np.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite coordinates")
    return result


@dataclass(frozen=True, slots=True)
class EmpiricalPairFitRow:
    """一帧、一个掩膜下的经验 PSF K=1/K=2 比较。"""

    frame_index: int
    path: str
    coordinate_mode: str
    mask_mode: str
    patch_x0: int
    patch_y0: int
    patch_width: int
    patch_height: int
    sample_count: int
    masked_pixel_count: int
    fixed_single_rss: float | None
    fixed_double_rss: float | None
    fixed_single_bic: float | None
    fixed_double_bic: float | None
    fixed_delta_bic: float | None
    best_single_grid_rss: float | None
    best_single_grid_bic: float | None
    best_single_grid_x: float | None
    best_single_grid_y: float | None
    best_single_grid_evaluated_count: int
    best_single_grid_delta_bic: float | None
    double_amplitudes: tuple[float, ...]
    double_component_snr: tuple[float, ...]
    double_second_component_snr: float | None
    double_amplitude_correlation: float | None
    double_design_condition_number: float | None
    double_active_component_count: int
    fixed_confirmation_line: bool
    best_single_grid_control_line: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "coordinate_mode": self.coordinate_mode,
            "mask_mode": self.mask_mode,
            "patch_x0": self.patch_x0,
            "patch_y0": self.patch_y0,
            "patch_width": self.patch_width,
            "patch_height": self.patch_height,
            "sample_count": self.sample_count,
            "masked_pixel_count": self.masked_pixel_count,
            "fixed_single_rss": self.fixed_single_rss,
            "fixed_double_rss": self.fixed_double_rss,
            "fixed_single_bic": self.fixed_single_bic,
            "fixed_double_bic": self.fixed_double_bic,
            "fixed_delta_bic": self.fixed_delta_bic,
            "best_single_grid_rss": self.best_single_grid_rss,
            "best_single_grid_bic": self.best_single_grid_bic,
            "best_single_grid_x": self.best_single_grid_x,
            "best_single_grid_y": self.best_single_grid_y,
            "best_single_grid_evaluated_count": self.best_single_grid_evaluated_count,
            "best_single_grid_delta_bic": self.best_single_grid_delta_bic,
            "double_amplitudes": list(self.double_amplitudes),
            "double_component_snr": list(self.double_component_snr),
            "double_second_component_snr": self.double_second_component_snr,
            "double_amplitude_correlation": self.double_amplitude_correlation,
            "double_design_condition_number": self.double_design_condition_number,
            "double_active_component_count": self.double_active_component_count,
            "fixed_confirmation_line": self.fixed_confirmation_line,
            "best_single_grid_control_line": self.best_single_grid_control_line,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalPairFitResult:
    """经验 PSF 锚点敏感性审计的序列汇总。"""

    frame_count: int
    coordinate_mode: str
    primary_detection_id: int | None
    secondary_detection_id: int | None
    primary_target_xy: tuple[float, float]
    secondary_target_xy: tuple[float, float]
    frame_shifts: tuple[tuple[float, float], ...]
    template_frame_path: str
    template_catalog_path: str | None
    mask_modes: tuple[str, ...]
    support_radius: int
    patch_padding_px: float
    best_single_search_radius_px: float
    best_single_grid_step_px: float
    confirmation_delta_bic: float
    confirmation_component_snr: float
    empirical_psf: EmpiricalPSF | None
    parameters: dict[str, object]
    rows: tuple[EmpiricalPairFitRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "coordinate_mode": self.coordinate_mode,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_target_xy": list(self.primary_target_xy),
            "secondary_target_xy": list(self.secondary_target_xy),
            "frame_shifts": [list(shift) for shift in self.frame_shifts],
            "template_frame_path": self.template_frame_path,
            "template_catalog_path": self.template_catalog_path,
            "mask_modes": list(self.mask_modes),
            "support_radius": self.support_radius,
            "patch_padding_px": self.patch_padding_px,
            "best_single_search_radius_px": self.best_single_search_radius_px,
            "best_single_grid_step_px": self.best_single_grid_step_px,
            "confirmation_delta_bic": self.confirmation_delta_bic,
            "confirmation_component_snr": self.confirmation_component_snr,
            "empirical_psf": None if self.empirical_psf is None else self.empirical_psf.as_dict(),
            "parameters": self.parameters,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
        }


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


def _patch_bounds(
    shape: tuple[int, int],
    positions: Sequence[tuple[float, float]],
    padding_px: float,
) -> tuple[int, int, int, int]:
    height, width = shape
    x0 = max(0, int(np.floor(min(point[0] for point in positions) - padding_px)))
    x1 = min(width, int(np.ceil(max(point[0] for point in positions) + padding_px)) + 1)
    y0 = max(0, int(np.floor(min(point[1] for point in positions) - padding_px)))
    y1 = min(height, int(np.ceil(max(point[1] for point in positions) + padding_px)) + 1)
    return x0, y0, x1, y1


def _resolve_repeated_code_values(
    template_sources: Sequence[Detection],
    repeated_code_values: Sequence[int] | None,
) -> tuple[int, ...]:
    if repeated_code_values is None:
        values = {
            int(code)
            for source in template_sources
            for code in source.repeated_code_values
        }
    else:
        values = {int(code) for code in repeated_code_values}
    return tuple(sorted(values))


def _pair_mask_modes(
    image: np.ndarray,
    *,
    mask_modes: Sequence[str],
    repeated_code_values: Sequence[int],
) -> dict[str, np.ndarray]:
    """构造 pair 审计的值域掩膜，显式区分工程重复码与极值范围线。"""

    values = np.asarray(image)
    base = auxiliary_mask(values.shape)
    negative_limit = _default_negative_overflow_limit(values)
    codes = np.asarray(tuple(repeated_code_values))
    result: dict[str, np.ndarray] = {}
    for raw_mode in mask_modes:
        mode = str(raw_mode)
        mode_mask = base.copy()
        if mode in {
            "range_masked",
            "range_sentinel_masked",
            "range_repeated_code_masked",
            "range_sentinel_repeated_code_masked",
        }:
            if negative_limit is not None:
                mode_mask |= values <= float(negative_limit)
                mode_mask |= values >= -float(negative_limit)
        if mode in {
            "sentinel_masked",
            "range_sentinel_masked",
            "sentinel_repeated_code_masked",
            "range_sentinel_repeated_code_masked",
        }:
            mode_mask |= values == -1
        if mode in {
            "repeated_code_masked",
            "range_repeated_code_masked",
            "sentinel_repeated_code_masked",
            "range_sentinel_repeated_code_masked",
        }:
            if codes.size:
                mode_mask |= np.isin(values, codes)
        result[mode] = np.asarray(mode_mask, dtype=bool)
    return result


def _shifted_empirical_basis(
    patch_shape: tuple[int, int],
    patch_origin: tuple[int, int],
    position: tuple[float, float],
    psf: EmpiricalPSF,
) -> np.ndarray:
    """把以整数中心为基准的经验核移动到一个亚像素位置。"""

    radius = int(psf.support_radius)
    kernel = np.asarray(psf.kernel, dtype=np.float64)
    expected_shape = (2 * radius + 1, 2 * radius + 1)
    if kernel.shape != expected_shape:
        raise ValueError(f"empirical PSF shape {kernel.shape} does not match support radius {radius}")
    x, y = position
    center_x = int(round(x))
    center_y = int(round(y))
    shifted = ndimage.shift(
        kernel,
        shift=(float(center_y) - y, float(center_x) - x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    basis = np.zeros(patch_shape, dtype=np.float64)
    x0, y0 = patch_origin
    for kernel_y in range(expected_shape[0]):
        global_y = center_y - radius + kernel_y
        local_y = global_y - y0
        if not 0 <= local_y < patch_shape[0]:
            continue
        for kernel_x in range(expected_shape[1]):
            global_x = center_x - radius + kernel_x
            local_x = global_x - x0
            if 0 <= local_x < patch_shape[1]:
                basis[local_y, local_x] = shifted[kernel_y, kernel_x]
    return basis


def _correlation_from_covariance(covariance: np.ndarray) -> float | None:
    if covariance.shape != (2, 2):
        return None
    denominator = float(np.sqrt(max(covariance[0, 0], 0.0) * max(covariance[1, 1], 0.0)))
    if denominator <= np.finfo(np.float64).eps:
        return None
    value = float(covariance[0, 1] / denominator)
    return value if np.isfinite(value) else None


def _fit_fixed_empirical_psf(
    image: np.ndarray,
    mask: np.ndarray,
    positions: Sequence[tuple[float, float]],
    *,
    patch_origin: tuple[int, int],
    patch_bounds: tuple[int, int, int, int],
    psf: EmpiricalPSF,
) -> dict[str, object] | None:
    """同一窗口中拟合固定位置的非负经验 PSF 分量。"""

    x0, y0, x1, y1 = patch_bounds
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    patch_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    valid = np.isfinite(patch) & ~patch_mask
    sample_count = int(np.count_nonzero(valid))
    component_count = len(positions)
    parameter_count = 3 + component_count
    if sample_count < max(24, parameter_count + 8):
        return None

    yy, xx = np.indices(patch.shape, dtype=np.float64)
    center_x = 0.5 * (x0 + x1 - 1)
    center_y = 0.5 * (y0 + y1 - 1)
    scale = max(float(x1 - x0), float(y1 - y0), 1.0)
    background_terms = np.column_stack(
        [
            np.ones(patch.shape, dtype=np.float64)[valid],
            ((xx + x0 - center_x) / scale)[valid],
            ((yy + y0 - center_y) / scale)[valid],
        ]
    )
    source_terms = np.column_stack(
        [
            _shifted_empirical_basis(patch.shape, patch_origin, position, psf)[valid]
            for position in positions
        ]
    )
    y_values = patch[valid]
    try:
        background_from_y, *_ = np.linalg.lstsq(background_terms, y_values, rcond=None)
        background_from_source, *_ = np.linalg.lstsq(
            background_terms,
            source_terms,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None
    source_residual = source_terms - background_terms @ background_from_source
    data_residual = y_values - background_terms @ background_from_y
    try:
        amplitudes, _nnls_residual = nnls(source_residual, data_residual)
        background_coefficients, *_ = np.linalg.lstsq(
            background_terms,
            y_values - source_terms @ amplitudes,
            rcond=None,
        )
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return None
    model = background_terms @ background_coefficients + source_terms @ amplitudes
    residual = y_values - model
    epsilon = np.finfo(np.float64).eps
    rss = max(float(residual @ residual), epsilon)
    bic = float(sample_count * np.log(rss / sample_count) + parameter_count * np.log(sample_count))
    active = amplitudes > max(
        1e-12,
        np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(amplitudes)))),
    )
    degrees_of_freedom = max(1, sample_count - parameter_count)
    variance = rss / degrees_of_freedom
    component_snr = np.zeros(component_count, dtype=np.float64)
    covariance: np.ndarray | None = None
    try:
        active_design = source_residual[:, active]
        covariance = variance * np.linalg.pinv(active_design.T @ active_design)
        errors = np.sqrt(np.maximum(np.diag(covariance), epsilon))
        component_snr[active] = amplitudes[active] / errors
    except (np.linalg.LinAlgError, ValueError):
        covariance = None
    try:
        condition_number = float(np.linalg.cond(source_residual))
    except (np.linalg.LinAlgError, ValueError):
        condition_number = None
    if condition_number is not None and not np.isfinite(condition_number):
        condition_number = None
    amplitude_correlation = None
    if covariance is not None and component_count == 2 and np.all(active):
        amplitude_correlation = _correlation_from_covariance(covariance)
    return {
        "sample_count": sample_count,
        "masked_pixel_count": int(patch.size - sample_count),
        "rss": float(rss),
        "bic": bic,
        "amplitudes": tuple(float(value) for value in amplitudes),
        "component_snr": tuple(float(value) for value in component_snr),
        "active_component_count": int(np.count_nonzero(active)),
        "amplitude_correlation": amplitude_correlation,
        "design_condition_number": condition_number,
    }


def _single_position_grid(
    center: tuple[float, float],
    radius_px: float,
    step_px: float,
) -> tuple[tuple[float, float], ...]:
    offsets = np.arange(-radius_px, radius_px + 0.5 * step_px, step_px, dtype=np.float64)
    return tuple(
        (float(center[0] + dx), float(center[1] + dy))
        for dy in offsets
        for dx in offsets
    )


def _csv_value(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def run_empirical_pair_fit_audit(
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
    coordinate_mode: str = "centroid",
    mask_modes: Sequence[str] = ("raw", "range_masked", "sentinel_masked"),
    repeated_code_values: Sequence[int] | None = None,
    support_radius: int = 7,
    max_template_sources: int = 64,
    patch_padding_px: float | None = None,
    best_single_search_radius_px: float = 2.0,
    best_single_grid_step_px: float = 0.5,
    confirmation_delta_bic: float = 10.0,
    confirmation_component_snr: float = 5.0,
    empirical_psf: EmpiricalPSF | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EmpiricalPairFitResult:
    """比较固定锚点双源与最佳单源控制，保持所有像素条件一致。"""

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    primary_xy = _xy(primary_target_xy, "primary_target_xy")
    secondary_xy = _xy(secondary_target_xy, "secondary_target_xy")
    if np.hypot(secondary_xy[0] - primary_xy[0], secondary_xy[1] - primary_xy[1]) <= 1e-6:
        raise ValueError("primary and secondary target coordinates must differ")
    if coordinate_mode not in {"centroid", "peak"}:
        raise ValueError("coordinate_mode must be centroid or peak")
    resolved_modes = tuple(str(mode) for mode in mask_modes)
    if not resolved_modes or len(set(resolved_modes)) != len(resolved_modes):
        raise ValueError("mask_modes must be a non-empty sequence without duplicates")
    if any(mode not in _MASK_MODES for mode in resolved_modes):
        raise ValueError("mask_modes contains an unsupported mode")
    resolved_codes = _resolve_repeated_code_values(template_sources, repeated_code_values)
    if any("repeated_code" in mode for mode in resolved_modes) and not resolved_codes:
        raise ValueError("repeated-code mask mode requires repeated_code_values")
    if support_radius < 3 or max_template_sources < 1:
        raise ValueError("support_radius must be at least 3 and source count must be positive")
    search_radius = float(best_single_search_radius_px)
    grid_step = float(best_single_grid_step_px)
    if not np.isfinite(search_radius) or search_radius <= 0:
        raise ValueError("best_single_search_radius_px must be finite and positive")
    if not np.isfinite(grid_step) or grid_step <= 0:
        raise ValueError("best_single_grid_step_px must be finite and positive")
    if confirmation_delta_bic <= 0 or confirmation_component_snr <= 0:
        raise ValueError("confirmation thresholds must be positive")
    if template_frame is None:
        template_values, template_label = _frame_values(frame_paths[0])
    else:
        template_values, template_label = _frame_values(template_frame)
    if not template_sources:
        raise ValueError("template_sources cannot be empty")
    if empirical_psf is None:
        empirical_psf = estimate_empirical_psf(
            template_values,
            template_sources,
            support_radius=int(support_radius),
            max_sources=int(max_template_sources),
            isolation_sources=template_sources,
        )
    if empirical_psf is None:
        raise ValueError("not enough eligible isolated sources to build an empirical PSF")
    if int(empirical_psf.support_radius) != int(support_radius):
        raise ValueError("empirical_psf support radius does not match support_radius")
    radius = float(empirical_psf.support_radius)
    if patch_padding_px is None:
        patch_padding = max(radius + 3.0, 10.0)
    else:
        patch_padding = float(patch_padding_px)
    if not np.isfinite(patch_padding) or patch_padding < radius + 1.0:
        raise ValueError("patch_padding_px must be at least support_radius + 1")
    if frame_shifts is None:
        shifts = tuple((0.0, 0.0) for _path in frame_paths)
    else:
        if len(frame_shifts) != len(frame_paths):
            raise ValueError("frame_shifts length must match paths")
        shifts = tuple((float(item[0]), float(item[1])) for item in frame_shifts)
        if any(not np.isfinite(value) for shift in shifts for value in shift):
            raise ValueError("frame_shifts must contain finite values")

    pair_positions = (primary_xy, secondary_xy)
    midpoint = (
        0.5 * (primary_xy[0] + secondary_xy[0]),
        0.5 * (primary_xy[1] + secondary_xy[1]),
    )
    rows: list[EmpiricalPairFitRow] = []
    for frame_number, path in enumerate(frame_paths, start=1):
        values, path_label = _frame_values(path)
        shift_x, shift_y = shifts[frame_number - 1]
        frame_positions = tuple((x + shift_x, y + shift_y) for x, y in pair_positions)
        frame_midpoint = (midpoint[0] + shift_x, midpoint[1] + shift_y)
        patch_bounds = _patch_bounds(values.shape, frame_positions, patch_padding)
        x0, y0, x1, y1 = patch_bounds
        patch_shape = (y1 - y0, x1 - x0)
        if patch_shape[0] < 2 * int(radius) + 3 or patch_shape[1] < 2 * int(radius) + 3:
            raise ValueError(f"pair patch is too small in frame {frame_number}")
        masks = _pair_mask_modes(
            values,
            mask_modes=resolved_modes,
            repeated_code_values=resolved_codes,
        )
        grid_positions = _single_position_grid(frame_midpoint, search_radius, grid_step)
        for mask_mode, mode_mask in masks.items():
            single_fit = _fit_fixed_empirical_psf(
                values,
                mode_mask,
                (frame_positions[0],),
                patch_origin=(x0, y0),
                patch_bounds=patch_bounds,
                psf=empirical_psf,
            )
            double_fit = _fit_fixed_empirical_psf(
                values,
                mode_mask,
                frame_positions,
                patch_origin=(x0, y0),
                patch_bounds=patch_bounds,
                psf=empirical_psf,
            )
            best_grid_fit: dict[str, object] | None = None
            best_grid_position: tuple[float, float] | None = None
            evaluated_count = 0
            for grid_position in grid_positions:
                candidate = _fit_fixed_empirical_psf(
                    values,
                    mode_mask,
                    (grid_position,),
                    patch_origin=(x0, y0),
                    patch_bounds=patch_bounds,
                    psf=empirical_psf,
                )
                if candidate is None:
                    continue
                evaluated_count += 1
                if best_grid_fit is None or float(candidate["rss"]) < float(best_grid_fit["rss"]):
                    best_grid_fit = candidate
                    best_grid_position = grid_position
            fixed_single_bic = None if single_fit is None else float(single_fit["bic"])
            fixed_double_bic = None if double_fit is None else float(double_fit["bic"])
            fixed_delta = (
                None
                if fixed_single_bic is None or fixed_double_bic is None
                else fixed_single_bic - fixed_double_bic
            )
            grid_bic = None if best_grid_fit is None else float(best_grid_fit["bic"])
            grid_delta = (
                None
                if grid_bic is None or fixed_double_bic is None
                else grid_bic - fixed_double_bic
            )
            amplitudes = () if double_fit is None else tuple(double_fit["amplitudes"])
            component_snr = () if double_fit is None else tuple(double_fit["component_snr"])
            second_snr = component_snr[1] if len(component_snr) >= 2 else None
            fixed_line = bool(
                fixed_delta is not None
                and fixed_delta >= float(confirmation_delta_bic)
                and second_snr is not None
                and second_snr >= float(confirmation_component_snr)
            )
            grid_line = bool(
                grid_delta is not None
                and grid_delta >= float(confirmation_delta_bic)
                and second_snr is not None
                and second_snr >= float(confirmation_component_snr)
            )
            if double_fit is None:
                note = "双源经验 PSF 拟合无有效结果"
            elif fixed_line and grid_line:
                note = "固定锚点和最佳单源位置控制均显示双源改善；仍非星表身份"
            elif fixed_line:
                note = "固定锚点双源改善，但最佳单源位置控制未通过；不作双星结论"
            else:
                note = "固定位置双源未形成当前证据线；不等于排除混叠或弱源"
            rows.append(
                EmpiricalPairFitRow(
                    frame_index=frame_number,
                    path=path_label,
                    coordinate_mode=coordinate_mode,
                    mask_mode=mask_mode,
                    patch_x0=x0,
                    patch_y0=y0,
                    patch_width=x1 - x0,
                    patch_height=y1 - y0,
                    sample_count=int(double_fit["sample_count"]) if double_fit is not None else 0,
                    masked_pixel_count=(
                        int(double_fit["masked_pixel_count"]) if double_fit is not None else 0
                    ),
                    fixed_single_rss=None if single_fit is None else float(single_fit["rss"]),
                    fixed_double_rss=None if double_fit is None else float(double_fit["rss"]),
                    fixed_single_bic=fixed_single_bic,
                    fixed_double_bic=fixed_double_bic,
                    fixed_delta_bic=fixed_delta,
                    best_single_grid_rss=None if best_grid_fit is None else float(best_grid_fit["rss"]),
                    best_single_grid_bic=grid_bic,
                    best_single_grid_x=None if best_grid_position is None else best_grid_position[0],
                    best_single_grid_y=None if best_grid_position is None else best_grid_position[1],
                    best_single_grid_evaluated_count=evaluated_count,
                    best_single_grid_delta_bic=grid_delta,
                    double_amplitudes=amplitudes,
                    double_component_snr=component_snr,
                    double_second_component_snr=second_snr,
                    double_amplitude_correlation=(
                        None
                        if double_fit is None
                        else _finite_or_none(double_fit["amplitude_correlation"])
                    ),
                    double_design_condition_number=(
                        None
                        if double_fit is None
                        else _finite_or_none(double_fit["design_condition_number"])
                    ),
                    double_active_component_count=(
                        0 if double_fit is None else int(double_fit["active_component_count"])
                    ),
                    fixed_confirmation_line=fixed_line,
                    best_single_grid_control_line=grid_line,
                    note=note,
                )
            )
        if progress is not None:
            progress(frame_number, len(frame_paths))

    raw_rows = [row for row in rows if row.mask_mode == "raw"]
    fixed_count = sum(row.fixed_confirmation_line for row in raw_rows)
    grid_count = sum(row.best_single_grid_control_line for row in raw_rows)
    conclusion = (
        f"经验 PSF 锚点敏感性审计完成：{len(frame_paths)} 帧，坐标锚点={coordinate_mode}；"
        f"raw 固定位置 K=2 证据线 {fixed_count}/{len(raw_rows)} 帧，"
        f"加入局部最佳单源位置控制后 {grid_count}/{len(raw_rows)} 帧。"
        "后一个数不是形式化的自由位置 BIC，而是同窗网格搜索的保守诊断；"
        "若双源优势在该控制下消失，优先解释为单源位置锚定或非高斯残差的模型自由度，"
        "不能据此确认两颗真实恒星。"
    )
    parameters = {
        "model": "shared planar background + fixed empirical PSF + nonnegative amplitudes",
        "coordinate_mode": coordinate_mode,
        "mask_modes": list(resolved_modes),
        "repeated_code_values": list(resolved_codes),
        "support_radius": int(support_radius),
        "max_template_sources": int(max_template_sources),
        "patch_padding_px": float(patch_padding),
        "best_single_search_radius_px": search_radius,
        "best_single_grid_step_px": grid_step,
        "best_single_grid_center": list(midpoint),
        "confirmation_delta_bic": float(confirmation_delta_bic),
        "confirmation_component_snr": float(confirmation_component_snr),
        "frame_shifts": [list(shift) for shift in shifts],
        "interpretation_boundary": (
            "固定位置 BIC 是局部模型证据；最佳单源网格搜索是位置锚定控制；"
            "二者均不是星表身份、恒星概率或物理真值"
        ),
    }
    return EmpiricalPairFitResult(
        frame_count=len(frame_paths),
        coordinate_mode=coordinate_mode,
        primary_detection_id=primary_detection_id,
        secondary_detection_id=secondary_detection_id,
        primary_target_xy=primary_xy,
        secondary_target_xy=secondary_xy,
        frame_shifts=shifts,
        template_frame_path=template_label,
        template_catalog_path=None if template_catalog_path is None else str(template_catalog_path),
        mask_modes=resolved_modes,
        support_radius=int(support_radius),
        patch_padding_px=float(patch_padding),
        best_single_search_radius_px=search_radius,
        best_single_grid_step_px=grid_step,
        confirmation_delta_bic=float(confirmation_delta_bic),
        confirmation_component_snr=float(confirmation_component_snr),
        empirical_psf=empirical_psf,
        parameters=parameters,
        rows=tuple(rows),
        conclusion=conclusion,
    )


def write_empirical_pair_fit_artifacts(result: EmpiricalPairFitResult, out_dir: str | Path) -> Path:
    """写经验 PSF 锚点审计的 CSV、JSON 和简图。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(result.rows[0].as_dict()) if result.rows else ["frame_index", "path"]
    with (output / "empirical_pair_fit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.as_dict().items()}
            for row in result.rows
        )
    payload = result.as_dict()
    payload["note"] = (
        "固定位置 K=1/K=2 使用相同局部窗口和掩膜；最佳单源网格为位置锚定控制，"
        "不能直接当作自由位置模型的正式 BIC。经验 PSF 只表示模板帧的形状样本，"
        "所有结果均不是星表身份或物理真值。"
    )
    (output / "empirical_pair_fit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    from .innovation import _line_chart

    frame_numbers = [float(row.frame_index) for row in result.rows if row.mask_mode == "raw"]
    raw_rows = [row for row in result.rows if row.mask_mode == "raw"]
    if raw_rows:
        def series(name: str) -> list[float]:
            return [
                float(getattr(row, name)) if getattr(row, name) is not None else float("nan")
                for row in raw_rows
            ]

        _line_chart(
            output / "empirical_pair_fit_bic.png",
            "EMPIRICAL PSF · FIXED PAIR VS BEST SINGLE POSITION",
            frame_numbers,
            (
                ("fixed K1 - K2", series("fixed_delta_bic"), "#d79432"),
                ("best single grid - K2", series("best_single_grid_delta_bic"), "#4f9b83"),
            ),
            y_label="delta BIC (positive favors K2)",
            x_label="frame",
        )
    return output


__all__ = [
    "EmpiricalPairFitResult",
    "EmpiricalPairFitRow",
    "run_empirical_pair_fit_audit",
    "write_empirical_pair_fit_artifacts",
]
