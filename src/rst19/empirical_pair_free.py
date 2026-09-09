"""自由位置经验 PSF 的近邻双源研究审计。

本模块把 ``empirical_pair_fit`` 的位置网格控制推进为一个显式的连续
参数模型：在相同局部窗口、相同三项背景、相同原始像素掩膜和相同经验
PSF 下，分别拟合 K=1 与 K=2，并让幅度、位置和背景共同优化。

它只用于研究，不参与默认候选数、``quality_passed``、GUI 或缓存。由于
经验 PSF 的空间变化、噪声似然和星表真值仍未完全标定，输出是局部模型
证据，不是双星概率或物理身份。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares, nnls

from .detection import Detection
from .empirical_pair_fit import (
    _MASK_MODES,
    _frame_values,
    _pair_mask_modes,
    _patch_bounds,
    _resolve_repeated_code_values,
    _shifted_empirical_basis,
    _xy,
)
from .experiments import EmpiricalPSF, estimate_empirical_psf


@dataclass(frozen=True, slots=True)
class EmpiricalFreePairRow:
    """一帧、一个掩膜下的自由位置 K=1/K=2 比较。"""

    frame_index: int
    path: str
    mask_mode: str
    patch_x0: int
    patch_y0: int
    patch_width: int
    patch_height: int
    sample_count: int
    masked_pixel_count: int
    single_rss: float | None
    double_rss: float | None
    single_bic: float | None
    double_bic: float | None
    delta_bic_single_minus_double: float | None
    single_initial_position: tuple[float, float] | None
    single_fitted_position: tuple[float, float] | None
    single_position_offset: tuple[float, float] | None
    single_position_at_bound: bool
    single_component_snr: float | None
    single_optimizer_success: bool
    single_optimizer_nfev: int | None
    double_initial_positions: tuple[tuple[float, float], ...]
    double_fitted_positions: tuple[tuple[float, float], ...]
    double_position_offsets: tuple[tuple[float, float], ...]
    double_positions_at_bound: tuple[bool, ...]
    double_amplitudes: tuple[float, ...]
    double_component_snr: tuple[float, ...]
    double_second_component_snr: float | None
    double_active_component_count: int
    double_optimizer_success: bool
    double_optimizer_nfev: int | None
    fitted_separation_px: float | None
    confirmation_line: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EmpiricalFreePairResult:
    """自由位置经验 PSF 审计的序列汇总。"""

    frame_count: int
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
    single_position_radius_px: float
    double_position_radius_px: float
    confirmation_delta_bic: float
    confirmation_component_snr: float
    minimum_fitted_separation_px: float
    empirical_psf: EmpiricalPSF | None
    parameters: dict[str, object]
    rows: tuple[EmpiricalFreePairRow, ...]
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
            "mask_modes": list(self.mask_modes),
            "support_radius": self.support_radius,
            "patch_padding_px": self.patch_padding_px,
            "single_position_radius_px": self.single_position_radius_px,
            "double_position_radius_px": self.double_position_radius_px,
            "confirmation_delta_bic": self.confirmation_delta_bic,
            "confirmation_component_snr": self.confirmation_component_snr,
            "minimum_fitted_separation_px": self.minimum_fitted_separation_px,
            "empirical_psf": None if self.empirical_psf is None else self.empirical_psf.as_dict(),
            "parameters": self.parameters,
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
        }


def _finite_xy(point: Sequence[float]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _position_is_inside(
    position: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    return max(abs(position[0] - center[0]), abs(position[1] - center[1])) <= radius + 1e-9


def _fit_free_empirical_psf(
    image: np.ndarray,
    mask: np.ndarray,
    anchor_positions: Sequence[tuple[float, float]],
    *,
    bounds_centers: Sequence[tuple[float, float]],
    start_positions: Sequence[tuple[float, float]],
    patch_origin: tuple[int, int],
    patch_bounds: tuple[int, int, int, int],
    psf: EmpiricalPSF,
    position_radius_px: float,
) -> dict[str, object] | None:
    """拟合一个带位置边界的非负经验 PSF 模型。"""

    anchors = tuple(_finite_xy(point) for point in anchor_positions)
    centers = tuple(_finite_xy(point) for point in bounds_centers)
    starts = tuple(_finite_xy(point) for point in start_positions)
    component_count = len(anchors)
    if component_count < 1 or len(centers) != component_count or len(starts) != component_count:
        return None
    if any(
        not _position_is_inside(start, center, position_radius_px)
        for start, center in zip(starts, centers, strict=True)
    ):
        return None

    x0, y0, x1, y1 = patch_bounds
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    patch_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    valid = np.isfinite(patch) & ~patch_mask
    sample_count = int(np.count_nonzero(valid))
    parameter_count = 3 + 3 * component_count
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
    y_values = patch[valid]
    patch_shape = patch.shape

    def source_terms_at(positions: Sequence[tuple[float, float]]) -> np.ndarray:
        return np.column_stack(
            [
                _shifted_empirical_basis(patch_shape, patch_origin, position, psf)[valid]
                for position in positions
            ]
        )

    initial_source_terms = source_terms_at(starts)
    try:
        background_from_y, *_ = np.linalg.lstsq(background_terms, y_values, rcond=None)
        background_from_source, *_ = np.linalg.lstsq(
            background_terms,
            initial_source_terms,
            rcond=None,
        )
        source_residual = initial_source_terms - background_terms @ background_from_source
        data_residual = y_values - background_terms @ background_from_y
        amplitudes, _ = nnls(source_residual, data_residual)
        initial_background, *_ = np.linalg.lstsq(
            background_terms,
            y_values - initial_source_terms @ amplitudes,
            rcond=None,
        )
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return None

    parameter_start = np.empty(parameter_count, dtype=np.float64)
    parameter_start[:3] = initial_background
    lower = np.full(parameter_count, -np.inf, dtype=np.float64)
    upper = np.full(parameter_count, np.inf, dtype=np.float64)
    for index, (start, center, amplitude) in enumerate(
        zip(starts, centers, amplitudes, strict=True)
    ):
        base = 3 + 3 * index
        parameter_start[base] = float(amplitude)
        parameter_start[base + 1] = start[0]
        parameter_start[base + 2] = start[1]
        lower[base] = 0.0
        lower[base + 1] = center[0] - position_radius_px
        lower[base + 2] = center[1] - position_radius_px
        upper[base + 1] = center[0] + position_radius_px
        upper[base + 2] = center[1] + position_radius_px

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        model = background_terms @ parameters[:3]
        for index in range(component_count):
            base = 3 + 3 * index
            position = (float(parameters[base + 1]), float(parameters[base + 2]))
            basis = _shifted_empirical_basis(patch_shape, patch_origin, position, psf)[valid]
            model = model + float(parameters[base]) * basis
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
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None

    parameters = np.asarray(optimized.x, dtype=np.float64)
    residual = np.asarray(optimized.fun, dtype=np.float64)
    if parameters.shape != (parameter_count,) or residual.size != sample_count:
        return None
    if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(residual)):
        return None

    epsilon = np.finfo(np.float64).eps
    rss = max(float(residual @ residual), epsilon)
    bic = float(sample_count * np.log(rss / sample_count) + parameter_count * np.log(sample_count))
    fitted_positions = tuple(
        (
            float(parameters[3 + 3 * index + 1]),
            float(parameters[3 + 3 * index + 2]),
        )
        for index in range(component_count)
    )
    position_offsets = tuple(
        (fitted[0] - center[0], fitted[1] - center[1])
        for fitted, center in zip(fitted_positions, centers, strict=True)
    )
    position_tolerance = max(1e-5, position_radius_px * 1e-5)
    positions_at_bound = tuple(
        abs(offset[0]) >= position_radius_px - position_tolerance
        or abs(offset[1]) >= position_radius_px - position_tolerance
        for offset in position_offsets
    )
    amplitudes = np.asarray(
        [parameters[3 + 3 * index] for index in range(component_count)],
        dtype=np.float64,
    )
    active = amplitudes > max(1e-12, epsilon * max(1.0, float(np.max(np.abs(amplitudes)))))
    degrees_of_freedom = max(1, sample_count - parameter_count)
    variance = rss / degrees_of_freedom
    component_snr = np.zeros(component_count, dtype=np.float64)
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

    return {
        "sample_count": sample_count,
        "masked_pixel_count": int(patch.size - sample_count),
        "rss": rss,
        "bic": bic,
        "initial_positions": starts,
        "fitted_positions": fitted_positions,
        "position_offsets": position_offsets,
        "positions_at_bound": positions_at_bound,
        "amplitudes": tuple(float(value) for value in amplitudes),
        "component_snr": tuple(float(value) for value in component_snr),
        "active_component_count": int(np.count_nonzero(active)),
        "optimizer_success": bool(optimized.success),
        "optimizer_status": int(optimized.status),
        "optimizer_nfev": int(optimized.nfev),
    }


def _best_free_fit(
    image: np.ndarray,
    mask: np.ndarray,
    anchor_positions: Sequence[tuple[float, float]],
    *,
    bounds_centers: Sequence[tuple[float, float]],
    starts: Sequence[Sequence[tuple[float, float]]],
    patch_origin: tuple[int, int],
    patch_bounds: tuple[int, int, int, int],
    psf: EmpiricalPSF,
    position_radius_px: float,
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for start_positions in starts:
        candidate = _fit_free_empirical_psf(
            image,
            mask,
            anchor_positions,
            bounds_centers=bounds_centers,
            start_positions=start_positions,
            patch_origin=patch_origin,
            patch_bounds=patch_bounds,
            psf=psf,
            position_radius_px=position_radius_px,
        )
        if candidate is None:
            continue
        if best is None or float(candidate["rss"]) < float(best["rss"]):
            best = candidate
    return best


def _single_start_positions(
    midpoint: tuple[float, float],
    primary: tuple[float, float],
    secondary: tuple[float, float],
    radius: float,
) -> tuple[tuple[float, float], ...]:
    offsets = (
        (0.0, 0.0),
        (radius * 0.5, 0.0),
        (-radius * 0.5, 0.0),
        (0.0, radius * 0.5),
        (0.0, -radius * 0.5),
        (radius * 0.5, radius * 0.5),
        (radius * 0.5, -radius * 0.5),
        (-radius * 0.5, radius * 0.5),
        (-radius * 0.5, -radius * 0.5),
    )
    positions = [
        (midpoint[0] + dx, midpoint[1] + dy)
        for dx, dy in offsets
        if _position_is_inside((midpoint[0] + dx, midpoint[1] + dy), midpoint, radius)
    ]
    for position in (primary, secondary):
        if _position_is_inside(position, midpoint, radius):
            positions.append(position)
    unique: list[tuple[float, float]] = []
    for position in positions:
        if not any(np.hypot(position[0] - old[0], position[1] - old[1]) < 1e-8 for old in unique):
            unique.append(position)
    return tuple(unique)


def _fit_summary(
    fit: dict[str, object] | None,
    *,
    component_count: int,
) -> dict[str, object]:
    if fit is None:
        return {
            "rss": None,
            "bic": None,
            "initial_positions": (),
            "fitted_positions": (),
            "position_offsets": (),
            "positions_at_bound": (),
            "amplitudes": (),
            "component_snr": (),
            "active_component_count": 0,
            "optimizer_success": False,
            "optimizer_nfev": None,
            "sample_count": 0,
            "masked_pixel_count": 0,
        }
    return fit


def run_empirical_free_pair_audit(
    paths: Iterable[str | Path],
    primary_target_xy: tuple[float, float],
    secondary_target_xy: tuple[float, float],
    *,
    template_sources: Sequence[Detection],
    primary_detection_id: int | None = None,
    secondary_detection_id: int | None = None,
    template_frame: object | None = None,
    template_catalog_path: str | Path | None = None,
    frame_shifts: Sequence[tuple[float, float]] | None = None,
    mask_modes: Sequence[str] = ("raw", "range_masked", "sentinel_masked"),
    repeated_code_values: Sequence[int] | None = None,
    support_radius: int = 7,
    max_template_sources: int = 64,
    patch_padding_px: float | None = None,
    single_position_radius_px: float = 2.0,
    double_position_radius_px: float = 1.25,
    confirmation_delta_bic: float = 10.0,
    confirmation_component_snr: float = 5.0,
    minimum_fitted_separation_px: float = 1.0,
    empirical_psf: EmpiricalPSF | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EmpiricalFreePairResult:
    """在边界内联合优化 K=1/K=2 的经验 PSF 位置和幅度。"""

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    primary_xy = _xy(primary_target_xy, "primary_target_xy")
    secondary_xy = _xy(secondary_target_xy, "secondary_target_xy")
    if np.hypot(secondary_xy[0] - primary_xy[0], secondary_xy[1] - primary_xy[1]) <= 1e-6:
        raise ValueError("primary and secondary target coordinates must differ")
    resolved_modes = tuple(str(mode) for mode in mask_modes)
    if not resolved_modes or len(set(resolved_modes)) != len(resolved_modes):
        raise ValueError("mask_modes must be a non-empty sequence without duplicates")
    if any(mode not in _MASK_MODES for mode in resolved_modes):
        raise ValueError("mask_modes contains an unsupported mode")
    resolved_codes = _resolve_repeated_code_values(template_sources, repeated_code_values)
    if any("repeated_code" in mode for mode in resolved_modes) and not resolved_codes:
        raise ValueError("repeated-code mask mode requires repeated_code_values")
    for name, value in (
        ("single_position_radius_px", single_position_radius_px),
        ("double_position_radius_px", double_position_radius_px),
        ("confirmation_delta_bic", confirmation_delta_bic),
        ("confirmation_component_snr", confirmation_component_snr),
        ("minimum_fitted_separation_px", minimum_fitted_separation_px),
    ):
        if value <= 0 or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite and positive")
    if support_radius < 3 or max_template_sources < 1:
        raise ValueError("support_radius must be at least 3 and source count must be positive")
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
    psf_radius = float(empirical_psf.support_radius)
    patch_padding = max(psf_radius + 3.0, 10.0) if patch_padding_px is None else float(patch_padding_px)
    if not np.isfinite(patch_padding) or patch_padding < psf_radius + 1.0:
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
    rows: list[EmpiricalFreePairRow] = []
    for frame_number, path in enumerate(frame_paths, start=1):
        values, path_label = _frame_values(path)
        shift_x, shift_y = shifts[frame_number - 1]
        frame_positions = tuple((x + shift_x, y + shift_y) for x, y in pair_positions)
        frame_midpoint = (midpoint[0] + shift_x, midpoint[1] + shift_y)
        patch_bounds = _patch_bounds(values.shape, frame_positions, patch_padding)
        x0, y0, x1, y1 = patch_bounds
        if y1 - y0 < 2 * int(psf_radius) + 3 or x1 - x0 < 2 * int(psf_radius) + 3:
            raise ValueError(f"pair patch is too small in frame {frame_number}")
        masks = _pair_mask_modes(
            values,
            mask_modes=resolved_modes,
            repeated_code_values=resolved_codes,
        )
        single_starts = _single_start_positions(
            frame_midpoint,
            frame_positions[0],
            frame_positions[1],
            float(single_position_radius_px),
        )
        for mask_mode, mode_mask in masks.items():
            single_fit = _best_free_fit(
                values,
                mode_mask,
                (frame_midpoint,),
                bounds_centers=(frame_midpoint,),
                starts=tuple((position,) for position in single_starts),
                patch_origin=(x0, y0),
                patch_bounds=patch_bounds,
                psf=empirical_psf,
                position_radius_px=float(single_position_radius_px),
            )
            single_best_position = (
                tuple(float(value) for value in single_fit["fitted_positions"][0])
                if single_fit is not None and single_fit["fitted_positions"]
                else frame_midpoint
            )
            double_start_positions: list[tuple[tuple[float, float], ...]] = [
                frame_positions,
                (frame_midpoint, frame_midpoint),
                (frame_positions[0], frame_midpoint),
                (frame_midpoint, frame_positions[1]),
                (single_best_position, frame_midpoint),
                (single_best_position, single_best_position),
            ]
            double_fit = _best_free_fit(
                values,
                mode_mask,
                frame_positions,
                bounds_centers=frame_positions,
                starts=tuple(double_start_positions),
                patch_origin=(x0, y0),
                patch_bounds=patch_bounds,
                psf=empirical_psf,
                position_radius_px=float(double_position_radius_px),
            )
            single = _fit_summary(single_fit, component_count=1)
            double = _fit_summary(double_fit, component_count=2)
            single_bic = None if single["bic"] is None else float(single["bic"])
            double_bic = None if double["bic"] is None else float(double["bic"])
            delta = None if single_bic is None or double_bic is None else single_bic - double_bic
            double_positions = tuple(
                (float(point[0]), float(point[1])) for point in double["fitted_positions"]
            )
            double_snr = tuple(float(value) for value in double["component_snr"])
            second_snr = double_snr[1] if len(double_snr) >= 2 else None
            separation = None
            if len(double_positions) >= 2:
                separation = float(
                    np.hypot(
                        double_positions[0][0] - double_positions[1][0],
                        double_positions[0][1] - double_positions[1][1],
                    )
                )
            double_at_bound = tuple(bool(value) for value in double["positions_at_bound"])
            confirmation = bool(
                delta is not None
                and delta >= float(confirmation_delta_bic)
                and second_snr is not None
                and second_snr >= float(confirmation_component_snr)
                and int(double["active_component_count"]) >= 2
                and bool(double["optimizer_success"])
                and not any(double_at_bound)
                and separation is not None
                and separation >= float(minimum_fitted_separation_px)
            )
            if double_fit is None or single_fit is None:
                note = "自由位置拟合无完整 K=1/K=2 结果"
            elif confirmation:
                note = "自由位置模型达到当前工程证据线；仍需空间 PSF、注入和星表核验"
            elif any(double_at_bound):
                note = "双源位置触碰搜索边界；局部模型比较，不等于物理恒星身份"
            elif not bool(double["optimizer_success"]):
                note = "双源优化器未报告收敛；局部模型比较，不等于物理恒星身份"
            else:
                note = "自由位置模型未达到当前双源证据线；不等于排除真实近邻"
            single_positions = tuple(
                (float(point[0]), float(point[1])) for point in single["fitted_positions"]
            )
            single_offsets = tuple(
                (float(point[0]), float(point[1])) for point in single["position_offsets"]
            )
            double_offsets = tuple(
                (float(point[0]), float(point[1])) for point in double["position_offsets"]
            )
            single_snr_values = tuple(float(value) for value in single["component_snr"])
            rows.append(
                EmpiricalFreePairRow(
                    frame_index=frame_number,
                    path=path_label,
                    mask_mode=mask_mode,
                    patch_x0=x0,
                    patch_y0=y0,
                    patch_width=x1 - x0,
                    patch_height=y1 - y0,
                    sample_count=int(double["sample_count"] or single["sample_count"]),
                    masked_pixel_count=int(double["masked_pixel_count"] or single["masked_pixel_count"]),
                    single_rss=None if single["rss"] is None else float(single["rss"]),
                    double_rss=None if double["rss"] is None else float(double["rss"]),
                    single_bic=single_bic,
                    double_bic=double_bic,
                    delta_bic_single_minus_double=delta,
                    single_initial_position=(
                        tuple(float(value) for value in single["initial_positions"][0])
                        if single["initial_positions"]
                        else None
                    ),
                    single_fitted_position=single_positions[0] if single_positions else None,
                    single_position_offset=single_offsets[0] if single_offsets else None,
                    single_position_at_bound=(
                        bool(single["positions_at_bound"][0]) if single["positions_at_bound"] else False
                    ),
                    single_component_snr=single_snr_values[0] if single_snr_values else None,
                    single_optimizer_success=bool(single["optimizer_success"]),
                    single_optimizer_nfev=(
                        None if single["optimizer_nfev"] is None else int(single["optimizer_nfev"])
                    ),
                    double_initial_positions=tuple(
                        (float(point[0]), float(point[1])) for point in double["initial_positions"]
                    ),
                    double_fitted_positions=double_positions,
                    double_position_offsets=double_offsets,
                    double_positions_at_bound=double_at_bound,
                    double_amplitudes=tuple(float(value) for value in double["amplitudes"]),
                    double_component_snr=double_snr,
                    double_second_component_snr=second_snr,
                    double_active_component_count=int(double["active_component_count"]),
                    double_optimizer_success=bool(double["optimizer_success"]),
                    double_optimizer_nfev=(
                        None if double["optimizer_nfev"] is None else int(double["optimizer_nfev"])
                    ),
                    fitted_separation_px=separation,
                    confirmation_line=confirmation,
                    note=note,
                )
            )
        if progress is not None:
            progress(frame_number, len(frame_paths))

    raw_rows = [row for row in rows if row.mask_mode == "raw"]
    confirmed = sum(row.confirmation_line for row in raw_rows)
    bounded = sum(any(row.double_positions_at_bound) for row in raw_rows)
    conclusion = (
        f"自由位置经验 PSF 审计完成：{len(frame_paths)} 帧；raw K=2 达到当前证据线 "
        f"{confirmed}/{len(raw_rows)} 帧，双源位置触碰边界 {bounded}/{len(raw_rows)} 帧。"
        "该模型仍是有边界的局部研究模型；未通过不能排除真实近邻，达到证据线也不能替代空间 PSF、"
        "注入真值或星表/WCS 身份。"
    )
    parameters = {
        "model": "shared planar background + free-position empirical PSF + nonnegative amplitudes",
        "mask_modes": list(resolved_modes),
        "repeated_code_values": list(resolved_codes),
        "support_radius": int(support_radius),
        "max_template_sources": int(max_template_sources),
        "patch_padding_px": float(patch_padding),
        "single_position_radius_px": float(single_position_radius_px),
        "double_position_radius_px": float(double_position_radius_px),
        "confirmation_delta_bic": float(confirmation_delta_bic),
        "confirmation_component_snr": float(confirmation_component_snr),
        "minimum_fitted_separation_px": float(minimum_fitted_separation_px),
        "frame_shifts": [list(shift) for shift in shifts],
        "interpretation_boundary": (
            "有边界自由位置 BIC 是局部模型证据；不是星表身份、恒星概率或物理真值"
        ),
    }
    return EmpiricalFreePairResult(
        frame_count=len(frame_paths),
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
        single_position_radius_px=float(single_position_radius_px),
        double_position_radius_px=float(double_position_radius_px),
        confirmation_delta_bic=float(confirmation_delta_bic),
        confirmation_component_snr=float(confirmation_component_snr),
        minimum_fitted_separation_px=float(minimum_fitted_separation_px),
        empirical_psf=empirical_psf,
        parameters=parameters,
        rows=tuple(rows),
        conclusion=conclusion,
    )


def _csv_value(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_empirical_free_pair_artifacts(
    result: EmpiricalFreePairResult,
    out_dir: str | Path,
) -> Path:
    """写自由位置经验 PSF 的 CSV、JSON 和 BIC 曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(result.rows[0].as_dict()) if result.rows else ["frame_index", "path"]
    with (output / "empirical_pair_free.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.as_dict().items()}
            for row in result.rows
        )
    payload = result.as_dict()
    payload["note"] = (
        "K=1/K=2 在同一窗口和掩膜下联合优化位置、幅度与平面背景；位置有边界，"
        "因此 BIC 只表示局部研究模型证据，不是星表身份、恒星概率或物理真值。"
    )
    (output / "empirical_pair_free.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    raw_rows = [row for row in result.rows if row.mask_mode == "raw"]
    if raw_rows:
        from .innovation import _line_chart

        frame_numbers = [float(row.frame_index) for row in raw_rows]
        delta_values = [
            float(row.delta_bic_single_minus_double)
            if row.delta_bic_single_minus_double is not None
            else float("nan")
            for row in raw_rows
        ]
        _line_chart(
            output / "empirical_pair_free_bic.png",
            "FREE EMPIRICAL PSF · K=1 VS K=2",
            frame_numbers,
            (("free K1 - free K2", delta_values, "#4f9b83"),),
            y_label="delta BIC (positive favors K2)",
            x_label="frame",
        )
    return output


__all__ = [
    "EmpiricalFreePairResult",
    "EmpiricalFreePairRow",
    "run_empirical_free_pair_audit",
    "write_empirical_free_pair_artifacts",
]
