"""经验 PSF 下的孔径测光敏感性审计。

该模块把“同一真实背景、同一注入位置、同一注入信号，只改变测光孔径”
固定下来，用来区分三个容易混淆的量：注入源是否被宽筛提出、孔径通量是否
达到质量门，以及全图滤波噪声重估造成的非局部候选变化。它是研究命令，
不修改 GUI 默认参数、质量规则或检测缓存。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .detection import Detection, _default_negative_overflow_limit, detect_sources
from .experiments import (
    EmpiricalPSF,
    _inject_source_signal,
    estimate_empirical_psf,
)
from .fits import FitsFrame, auxiliary_mask, read_fits
from .spatial_injection import _cell_bounds, _select_spatial_blank_sites


@dataclass(frozen=True, slots=True)
class ApertureSensitivityRow:
    """一个孔径、一个空间位置和一个已知注入源的复核记录。"""

    source_path: str
    grid_size: int
    cell_id: str
    x: float
    y: float
    local_background_adu: float | None
    local_noise_adu: float | None
    aperture_radius: int
    reference_aperture_radius: int
    control_signal_adu: float
    injected_peak_excess_adu: float
    paired_control_candidate_count: int
    paired_control_quality_count: int
    paired_control_filter_noise: float
    candidate_count: int
    quality_count: int
    candidate_delta_vs_control: int
    quality_delta_vs_control: int
    filter_noise: float
    filter_noise_delta_vs_control: float
    candidate_recovered: bool
    quality_recovered: bool
    nearest_candidate_distance_px: float | None
    detection_id: int | None
    peak_snr: float | None
    flux_snr: float | None
    filter_snr: float | None
    fwhm: float | None
    psf_support_pixels: int | None
    quality_passed: bool | None
    flags: tuple[str, ...]
    note: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        return payload


@dataclass(frozen=True, slots=True)
class ApertureSensitivityAuditResult:
    """经验 PSF 孔径敏感性审计的完整结果。"""

    source_path: str
    image_shape: tuple[int, int]
    grid_size: int
    aperture_radii: tuple[int, ...]
    reference_aperture_radius: int
    control_signal_adu: float
    baseline_candidate_count: int
    baseline_quality_count: int
    psf_model: str
    psf_source_count: int
    psf_support_radius: int
    psf_median_fwhm_px: float | None
    psf_kernel_sum: float
    psf_encircled_energy_by_radius: dict[str, float]
    parameters: dict[str, object]
    positions: tuple[dict[str, object], ...]
    rows: tuple[ApertureSensitivityRow, ...]
    summary_by_aperture: tuple[dict[str, object], ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "image_shape": list(self.image_shape),
            "grid_size": self.grid_size,
            "aperture_radii": list(self.aperture_radii),
            "reference_aperture_radius": self.reference_aperture_radius,
            "control_signal_adu": self.control_signal_adu,
            "baseline_candidate_count": self.baseline_candidate_count,
            "baseline_quality_count": self.baseline_quality_count,
            "psf_model": self.psf_model,
            "psf_source_count": self.psf_source_count,
            "psf_support_radius": self.psf_support_radius,
            "psf_median_fwhm_px": self.psf_median_fwhm_px,
            "psf_kernel_sum": self.psf_kernel_sum,
            "psf_encircled_energy_by_radius": dict(self.psf_encircled_energy_by_radius),
            "parameters": dict(self.parameters),
            "positions": [dict(position) for position in self.positions],
            "rows": [row.as_dict() for row in self.rows],
            "summary_by_aperture": [dict(summary) for summary in self.summary_by_aperture],
            "conclusion": self.conclusion,
        }


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _closest_source(
    sources: Sequence[Detection],
    x: float,
    y: float,
    recovery_radius_px: float,
) -> tuple[Detection | None, float | None]:
    if not sources:
        return None, None
    distances = np.asarray(
        [np.hypot(float(source.x) - x, float(source.y) - y) for source in sources],
        dtype=np.float64,
    )
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance > recovery_radius_px:
        return None, distance
    return sources[index], distance


def _psf_encircled_energy(psf: EmpiricalPSF) -> dict[str, float]:
    kernel = np.asarray(psf.kernel, dtype=np.float64)
    radius = int(psf.support_radius)
    yy, xx = np.indices(kernel.shape, dtype=np.float64)
    distance = np.hypot(xx - radius, yy - radius)
    total = float(kernel.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("empirical PSF kernel must have a finite positive sum")
    return {
        str(aperture_radius): float(kernel[distance <= aperture_radius].sum() / total)
        for aperture_radius in range(1, radius + 1)
    }


def _normalise_positions(
    positions: Sequence[tuple[str, float, float, float | None, float | None]],
) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for cell_id, x, y, local_background, local_noise in positions:
        if not np.isfinite(float(x)) or not np.isfinite(float(y)):
            raise ValueError("aperture sensitivity positions must be finite")
        result.append(
            {
                "cell_id": str(cell_id),
                "x": float(x),
                "y": float(y),
                "local_background_adu": _optional_float(local_background),
                "local_noise_adu": _optional_float(local_noise),
            }
        )
    if not result:
        raise ValueError("at least one aperture sensitivity position is required")
    return tuple(result)


def run_aperture_sensitivity_audit(
    path: str | Path | FitsFrame,
    *,
    aperture_radii: Sequence[int] = (3, 4, 5, 6),
    reference_aperture_radius: int = 4,
    control_signal_adu: float = 512.0,
    grid_size: int = 2,
    psf_fwhm: float = 2.0,
    proposal_mode: str = "hybrid",
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    max_sources: int | None = None,
    empirical_psf_radius: int = 7,
    empirical_psf_sources: int = 64,
    seed: int = 19019,
    positions: Sequence[tuple[str, float, float, float | None, float | None]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ApertureSensitivityAuditResult:
    """固定注入位置并扫描孔径半径，记录候选/质量层和源级测光变化。

    默认位置选择与 ``rst19-spatial-injection`` 相同：首帧候选、特殊值和
    局部高分位亮结构之外的 ``2×2`` 相对空白位置。若传入 ``positions``，
    位置必须为 ``(cell_id, x, y, local_background, local_noise)``，用于复现
    已有空间注入布局。每个位置只注入一次，不把四个位置同时叠加。
    """

    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if reference_aperture_radius < 1:
        raise ValueError("reference_aperture_radius must be positive")
    radii = tuple(sorted({int(radius) for radius in aperture_radii}))
    if not radii or any(radius < 1 for radius in radii):
        raise ValueError("aperture_radii must contain positive integers")
    if not np.isfinite(float(control_signal_adu)) or control_signal_adu <= 0:
        raise ValueError("control_signal_adu must be finite and positive")
    if psf_fwhm <= 0 or threshold_sigma <= 0 or min_distance < 1:
        raise ValueError("detector parameters must be positive")
    if background_box_size < 16 or min_flux_snr <= 0:
        raise ValueError("background_box_size/min_flux_snr are invalid")
    if not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_psf_support_pixels must be between 1 and 9")
    if empirical_psf_radius < 3 or empirical_psf_sources < 1:
        raise ValueError("empirical PSF radius must be at least 3 and source count must be positive")

    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")

    detector_options = {
        "mask": auxiliary_mask(image.shape),
        "threshold_sigma": threshold_sigma,
        "min_distance": min_distance,
        "aperture_radius": reference_aperture_radius,
        "max_sources": max_sources,
        "psf_fwhm": psf_fwhm,
        "background_box_size": background_box_size,
        "min_flux_snr": min_flux_snr,
        "min_psf_support_pixels": min_psf_support_pixels,
        "reject_linear_artifacts": True,
        "proposal_mode": proposal_mode,
    }
    baseline = detect_sources(image, **detector_options)
    empirical_psf = estimate_empirical_psf(
        image,
        baseline.sources,
        support_radius=empirical_psf_radius,
        max_sources=empirical_psf_sources,
    )
    if empirical_psf is None:
        raise RuntimeError("真实质量源不足，无法建立实测 PSF")

    if positions is None:
        rng = np.random.default_rng(seed)
        generated: list[tuple[str, float, float, float | None, float | None]] = []
        for cell_y in range(grid_size):
            for cell_x in range(grid_size):
                bounds = _cell_bounds(image.shape, grid_size, cell_x, cell_y)
                site = _select_spatial_blank_sites(
                    image,
                    baseline.sources,
                    rng,
                    1,
                    cell_bounds=bounds,
                    baseline_noise_adu=float(baseline.noise),
                    psf_fwhm=psf_fwhm,
                    aperture_radius=reference_aperture_radius,
                )[0]
                generated.append(
                    (
                        f"r{cell_y}c{cell_x}",
                        float(site.x),
                        float(site.y),
                        float(site.local_background_adu),
                        float(site.local_noise_adu),
                    )
                )
        position_payload = _normalise_positions(generated)
    else:
        position_payload = _normalise_positions(positions)

    injected_options_base = dict(detector_options)
    saturation_level = _optional_float(baseline.parameters.get("saturation_level"))
    injected_options_base["saturation_level"] = saturation_level if saturation_level and saturation_level > 0 else None
    negative_limit = _optional_float(baseline.parameters.get("negative_overflow_limit"))
    injected_options_base["negative_overflow_limit"] = (
        negative_limit if negative_limit is not None else _default_negative_overflow_limit(image)
    )
    injected_options_base["mask_zero_pixels"] = bool(int(baseline.parameters.get("mask_zero_pixels", 0)))

    total = len(radii) * (1 + len(position_payload))
    completed = 0
    if progress is not None:
        progress(0, total)
    rows: list[ApertureSensitivityRow] = []
    summary: list[dict[str, object]] = []
    recovery_radius = max(3.0, float(psf_fwhm))

    for aperture_radius in radii:
        options = dict(injected_options_base)
        options["aperture_radius"] = aperture_radius
        control = detect_sources(np.asarray(image, dtype=np.float32).copy(), **options)
        control_filter_noise = _optional_float(control.parameters.get("filter_noise"))
        if control_filter_noise is None:
            raise RuntimeError("检测结果缺少有限的 filter_noise")
        completed += 1
        if progress is not None:
            progress(completed, total)

        aperture_rows: list[ApertureSensitivityRow] = []
        for position in position_payload:
            test_image = np.asarray(image, dtype=np.float32).copy()
            x = float(position["x"])
            y = float(position["y"])
            injected_peak = _inject_source_signal(
                test_image,
                x,
                y,
                float(control_signal_adu),
                signal_normalization="integrated_excess",
                psf_model="empirical",
                gaussian_fwhm=psf_fwhm,
                empirical_psf=empirical_psf,
            )
            result = detect_sources(test_image, **options)
            candidate, distance = _closest_source(result.sources, x, y, recovery_radius)
            quality, _quality_distance = _closest_source(result.quality_sources, x, y, recovery_radius)
            filter_noise = _optional_float(result.parameters.get("filter_noise"))
            if filter_noise is None:
                raise RuntimeError("注入检测结果缺少有限的 filter_noise")
            flags = tuple(str(flag) for flag in candidate.flags) if candidate is not None else ()
            row = ApertureSensitivityRow(
                source_path=str(frame.path),
                grid_size=grid_size,
                cell_id=str(position["cell_id"]),
                x=x,
                y=y,
                local_background_adu=_optional_float(position.get("local_background_adu")),
                local_noise_adu=_optional_float(position.get("local_noise_adu")),
                aperture_radius=aperture_radius,
                reference_aperture_radius=reference_aperture_radius,
                control_signal_adu=float(control_signal_adu),
                injected_peak_excess_adu=float(injected_peak),
                paired_control_candidate_count=int(control.candidate_count),
                paired_control_quality_count=int(control.star_count),
                paired_control_filter_noise=float(control_filter_noise),
                candidate_count=int(result.candidate_count),
                quality_count=int(result.star_count),
                candidate_delta_vs_control=int(result.candidate_count - control.candidate_count),
                quality_delta_vs_control=int(result.star_count - control.star_count),
                filter_noise=float(filter_noise),
                filter_noise_delta_vs_control=float(filter_noise - control_filter_noise),
                candidate_recovered=candidate is not None,
                quality_recovered=quality is not None,
                nearest_candidate_distance_px=distance,
                detection_id=None if candidate is None else int(candidate.detection_id),
                peak_snr=None if candidate is None else _optional_float(candidate.snr),
                flux_snr=None if candidate is None else _optional_float(candidate.flux_snr),
                filter_snr=None if candidate is None else _optional_float(candidate.filter_snr),
                fwhm=None if candidate is None else _optional_float(candidate.fwhm),
                psf_support_pixels=(
                    None if candidate is None or candidate.psf_support_pixels is None else int(candidate.psf_support_pixels)
                ),
                quality_passed=None if candidate is None else bool(candidate.quality_passed),
                flags=flags,
                note=(
                    "同一经验 PSF、同一离散积分注入和同一空间位置；只改变测光孔径。"
                    "candidate/quality 是注入真值回收，不是当前 FITS 的恒星完备率。"
                ),
            )
            rows.append(row)
            aperture_rows.append(row)
            completed += 1
            if progress is not None:
                progress(completed, total)

        flux_values = [row.flux_snr for row in aperture_rows if row.flux_snr is not None]
        summary.append(
            {
                "aperture_radius": aperture_radius,
                "paired_control_candidate_count": int(control.candidate_count),
                "paired_control_quality_count": int(control.star_count),
                "paired_control_filter_noise": float(control_filter_noise),
                "candidate_recovered_count": int(sum(row.candidate_recovered for row in aperture_rows)),
                "quality_recovered_count": int(sum(row.quality_recovered for row in aperture_rows)),
                "injected_count": len(aperture_rows),
                "candidate_recall": float(np.mean([row.candidate_recovered for row in aperture_rows])),
                "quality_recall": float(np.mean([row.quality_recovered for row in aperture_rows])),
                "median_candidate_flux_snr": (
                    float(np.median(flux_values)) if flux_values else None
                ),
                "median_candidate_fwhm_px": (
                    float(np.median([row.fwhm for row in aperture_rows if row.fwhm is not None]))
                    if any(row.fwhm is not None for row in aperture_rows)
                    else None
                ),
                "quality_flags_by_position": {
                    row.cell_id: list(row.flags) for row in aperture_rows
                },
            }
        )

    summary_json = json.dumps(summary, ensure_ascii=False, allow_nan=False)
    quality_recall_by_radius = {
        int(item["aperture_radius"]): float(item["quality_recall"])
        for item in summary
    }
    best_radius = max(quality_recall_by_radius, key=quality_recall_by_radius.get)
    result = ApertureSensitivityAuditResult(
        source_path=str(frame.path),
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        grid_size=grid_size,
        aperture_radii=radii,
        reference_aperture_radius=reference_aperture_radius,
        control_signal_adu=float(control_signal_adu),
        baseline_candidate_count=int(baseline.candidate_count),
        baseline_quality_count=int(baseline.star_count),
        psf_model="empirical",
        psf_source_count=int(empirical_psf.source_count),
        psf_support_radius=int(empirical_psf.support_radius),
        psf_median_fwhm_px=empirical_psf.median_fwhm_px,
        psf_kernel_sum=float(np.asarray(empirical_psf.kernel, dtype=np.float64).sum()),
        psf_encircled_energy_by_radius=_psf_encircled_energy(empirical_psf),
        parameters={
            "reference_detector_aperture_radius": reference_aperture_radius,
            "aperture_radii": list(radii),
            "control_signal_adu": float(control_signal_adu),
            "signal_normalization": "integrated_excess",
            "psf_model": "empirical",
            "psf_fwhm_for_detector": float(psf_fwhm),
            "proposal_mode": proposal_mode,
            "threshold_sigma": float(threshold_sigma),
            "min_distance": int(min_distance),
            "background_box_size": int(background_box_size),
            "min_flux_snr": float(min_flux_snr),
            "min_psf_support_pixels": int(min_psf_support_pixels),
            "empirical_psf_radius": int(empirical_psf_radius),
            "empirical_psf_sources": int(empirical_psf_sources),
            "seed": int(seed),
            "position_count": len(position_payload),
            "position_source": "generated_with_spatial_blank_selector" if positions is None else "provided",
            "paired_control_dtype": "float32",
            "baseline_dtype": str(image.dtype),
            "baseline_negative_overflow_limit": _default_negative_overflow_limit(image),
            "summary_json": summary_json,
        },
        positions=position_payload,
        rows=tuple(rows),
        summary_by_aperture=tuple(summary),
        conclusion=(
            f"经验 PSF 孔径敏感性完成：{len(position_payload)} 个固定位置、{len(radii)} 个孔径；"
            f"质量回收率最高的本次 pilot 孔径为 r={best_radius}，"
            "该选择只描述当前信号、背景和模板条件，不是普适最优孔径。"
            "扩大孔径会同时改变通量截取、背景方差、形状测量和质量源计数，"
            "因此不能把质量回收变化直接解释为真实恒星数变化。"
        ),
    )
    return result


def write_aperture_sensitivity_artifacts(
    result: ApertureSensitivityAuditResult,
    out_dir: str | Path,
) -> Path:
    """写孔径敏感性 CSV/JSON 研究产物。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(result.rows[0].as_dict()) if result.rows else ["aperture_radius"]
    with (output / "aperture_sensitivity_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    (output / "aperture_sensitivity_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output
