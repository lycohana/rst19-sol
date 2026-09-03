"""对多帧星表匹配结果做逐帧局部 WCS 验证。

该模块刻意把“有一帧拟合成功”和“15 帧都稳定”分开：每一帧都独立
拟合完整二维仿射模型，并保留匹配点数、离群剔除、留一残差和尺度变化。
它不是盲解算，也不会把没有提供星表的本地数据强行标成已验证。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .catalog import CatalogSource
from .matching import CatalogMatch
from .pipeline import analyze_frame
from .wcs import AffineWCSCalibration, TangentPlaneWCS, fit_affine_wcs_from_matches


@dataclass(frozen=True, slots=True)
class WCSFrameValidation:
    """单帧局部 WCS 验证结果。"""

    frame_index: int
    frame_path: str
    status: str
    matched_count: int
    inlier_count: int
    inlier_ratio: float
    rms_residual_px: float | None
    max_residual_px: float | None
    leave_one_out_rms_residual_px: float | None
    leave_one_out_max_residual_px: float | None
    plate_scale_arcsec_per_pixel: float | None
    rotation_deg: float | None
    parity: int | None
    anisotropy_ratio: float | None
    condition_number: float | None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_number": self.frame_index + 1,
            "frame_path": self.frame_path,
            "status": self.status,
            "matched_count": self.matched_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "rms_residual_px": self.rms_residual_px,
            "max_residual_px": self.max_residual_px,
            "leave_one_out_rms_residual_px": self.leave_one_out_rms_residual_px,
            "leave_one_out_max_residual_px": self.leave_one_out_max_residual_px,
            "plate_scale_arcsec_per_pixel": self.plate_scale_arcsec_per_pixel,
            "rotation_deg": self.rotation_deg,
            "parity": self.parity,
            "anisotropy_ratio": self.anisotropy_ratio,
            "condition_number": self.condition_number,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class WCSValidationReport:
    """一组帧的逐帧 WCS 验证汇总。"""

    frame_rows: tuple[WCSFrameValidation, ...]
    catalog_source_count: int
    reference_wcs: TangentPlaneWCS
    min_matches: int
    method: str = "每帧独立局部仿射拟合 + MAD 离群剔除 + 匹配点留一验证"

    @property
    def frame_count(self) -> int:
        return len(self.frame_rows)

    @property
    def validated_count(self) -> int:
        return sum(row.status == "validated" for row in self.frame_rows)

    @property
    def validation_ratio(self) -> float:
        return self.validated_count / self.frame_count if self.frame_count else 0.0

    def as_dict(self) -> dict[str, object]:
        validated = [row for row in self.frame_rows if row.status == "validated"]

        def finite_values(name: str) -> list[float]:
            values: list[float] = []
            for row in validated:
                value = getattr(row, name)
                if value is not None and math.isfinite(float(value)):
                    values.append(float(value))
            return values

        def range_dict(name: str) -> dict[str, float | None]:
            values = finite_values(name)
            return {
                "min": min(values) if values else None,
                "median": _median(values),
                "max": max(values) if values else None,
            }

        return {
            "schema_version": 1,
            "method": self.method,
            "catalog_source_count": self.catalog_source_count,
            "reference_wcs": self.reference_wcs.as_dict(),
            "min_matches": self.min_matches,
            "frame_count": self.frame_count,
            "validated_count": self.validated_count,
            "validation_ratio": self.validation_ratio,
            "summary": {
                "rms_residual_px": range_dict("rms_residual_px"),
                "leave_one_out_rms_residual_px": range_dict("leave_one_out_rms_residual_px"),
                "plate_scale_arcsec_per_pixel": range_dict("plate_scale_arcsec_per_pixel"),
                "anisotropy_ratio": range_dict("anisotropy_ratio"),
            },
            "frames": [row.as_dict() for row in self.frame_rows],
            "note": (
                "validated 只表示该帧有足够的唯一匹配并通过本地仿射拟合；"
                "它不是全天空盲解算，也不等于官方逐星真值。应继续检查跨帧尺度、旋转、"
                "parity、空间残差和人工抽检。"
            ),
        }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _row_from_calibration(
    frame_index: int,
    frame_path: str,
    matches: Sequence[CatalogMatch],
    calibration: AffineWCSCalibration,
) -> WCSFrameValidation:
    return WCSFrameValidation(
        frame_index=frame_index,
        frame_path=frame_path,
        status="validated",
        matched_count=calibration.matched_count,
        inlier_count=calibration.inlier_count,
        inlier_ratio=calibration.inlier_ratio,
        rms_residual_px=calibration.rms_residual_px,
        max_residual_px=calibration.max_residual_px,
        leave_one_out_rms_residual_px=calibration.leave_one_out_rms_residual_px,
        leave_one_out_max_residual_px=calibration.leave_one_out_max_residual_px,
        plate_scale_arcsec_per_pixel=calibration.plate_scale_arcsec_per_pixel,
        rotation_deg=calibration.rotation_deg,
        parity=calibration.parity,
        anisotropy_ratio=calibration.anisotropy_ratio,
        condition_number=calibration.condition_number,
    )


def validate_frame_matches(
    frame_matches: Sequence[Sequence[CatalogMatch]],
    catalog: Sequence[CatalogSource],
    reference_wcs: TangentPlaneWCS,
    *,
    frame_paths: Sequence[str | Path] | None = None,
    epoch: float | None = None,
    min_matches: int = 6,
    clip_sigma: float = 3.5,
    min_clip_residual_px: float = 1.0,
    max_iterations: int = 6,
) -> WCSValidationReport:
    """从多帧已有匹配结果生成逐帧 WCS 验证报告。

    ``frame_matches`` 应来自同一套先验 WCS、匹配半径和星表版本。函数
    不会重新搜索匹配，也不会在不足 6 点时用低阶模型凑出尺度。
    """

    if min_matches < 3:
        raise ValueError("min_matches must be at least 3")
    if frame_paths is not None and len(frame_paths) != len(frame_matches):
        raise ValueError("frame_paths length must match frame_matches")
    rows: list[WCSFrameValidation] = []
    for frame_index, raw_matches in enumerate(frame_matches):
        matches = tuple(raw_matches)
        frame_path = str(frame_paths[frame_index]) if frame_paths is not None else f"frame-{frame_index + 1:02d}"
        try:
            calibration = fit_affine_wcs_from_matches(
                matches,
                catalog,
                reference_wcs,
                epoch=epoch,
                min_matches=min_matches,
                clip_sigma=clip_sigma,
                min_clip_residual_px=min_clip_residual_px,
                max_iterations=max_iterations,
            )
        except (TypeError, ValueError) as exc:
            status = "insufficient" if len(matches) < min_matches else "failed"
            rows.append(
                WCSFrameValidation(
                    frame_index=frame_index,
                    frame_path=frame_path,
                    status=status,
                    matched_count=len(matches),
                    inlier_count=0,
                    inlier_ratio=0.0,
                    rms_residual_px=None,
                    max_residual_px=None,
                    leave_one_out_rms_residual_px=None,
                    leave_one_out_max_residual_px=None,
                    plate_scale_arcsec_per_pixel=None,
                    rotation_deg=None,
                    parity=None,
                    anisotropy_ratio=None,
                    condition_number=None,
                    reason=str(exc),
                )
            )
        else:
            rows.append(_row_from_calibration(frame_index, frame_path, matches, calibration))
    return WCSValidationReport(
        frame_rows=tuple(rows),
        catalog_source_count=len(catalog),
        reference_wcs=reference_wcs,
        min_matches=min_matches,
    )


def run_sequence_wcs_validation(
    frame_paths: Sequence[str | Path],
    catalog: Sequence[CatalogSource],
    reference_wcs: TangentPlaneWCS,
    *,
    detector_kwargs: Mapping[str, Any] | None = None,
    match_radius_px: float = 3.0,
    epoch: float | None = None,
    min_matches: int = 6,
    progress: Callable[[int, int], None] | None = None,
) -> WCSValidationReport:
    """对整组 FITS 做检测、先验匹配，再逐帧拟合局部 WCS。"""

    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")
    if match_radius_px <= 0:
        raise ValueError("match_radius_px must be positive")
    parameters = dict(detector_kwargs or {})
    frame_matches: list[tuple[CatalogMatch, ...]] = []
    paths = tuple(Path(path).resolve() for path in frame_paths)
    for index, path in enumerate(paths, start=1):
        analysis = analyze_frame(
            path,
            catalog=catalog,
            wcs=reference_wcs,
            match_radius_px=match_radius_px,
            epoch=epoch,
            **parameters,
        )
        frame_matches.append(tuple(analysis.matching.matches) if analysis.matching is not None else ())
        if progress is not None:
            progress(index, len(paths))
    return validate_frame_matches(
        frame_matches,
        catalog,
        reference_wcs,
        frame_paths=paths,
        epoch=epoch,
        min_matches=min_matches,
    )


def write_wcs_validation_artifacts(report: WCSValidationReport, out_dir: str | Path) -> Path:
    """导出逐帧 WCS 验证 CSV、JSON 和残差/尺度曲线。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [row.as_dict() for row in report.frame_rows]
    fieldnames = (
        "frame_index",
        "frame_number",
        "frame_path",
        "status",
        "matched_count",
        "inlier_count",
        "inlier_ratio",
        "rms_residual_px",
        "max_residual_px",
        "leave_one_out_rms_residual_px",
        "leave_one_out_max_residual_px",
        "plate_scale_arcsec_per_pixel",
        "rotation_deg",
        "parity",
        "anisotropy_ratio",
        "condition_number",
        "reason",
    )
    with (output / "wcs_frame_validation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "wcs_validation_report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    from .innovation import _line_chart

    frame_numbers = [row.frame_index + 1 for row in report.frame_rows]
    _line_chart(
        output / "wcs_validation_residual.png",
        "WCS VALIDATION · PER-FRAME RESIDUAL",
        frame_numbers,
        (
            ("fit RMS / px", [row.rms_residual_px or math.nan for row in report.frame_rows], "#4f9b83"),
            ("leave-one-out RMS / px", [row.leave_one_out_rms_residual_px or math.nan for row in report.frame_rows], "#d79432"),
        ),
        y_label="pixel",
        x_label="frame",
    )
    _line_chart(
        output / "wcs_validation_scale.png",
        "WCS VALIDATION · PLATE SCALE",
        frame_numbers,
        (("arcsec / pixel", [row.plate_scale_arcsec_per_pixel or math.nan for row in report.frame_rows], "#72b9d4"),),
        y_label="arcsec / pixel",
        x_label="frame",
    )
    return output

