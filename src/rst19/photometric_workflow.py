"""把公共 Gaia 参考星、自动板解和单帧测光串成一个显式工作流。

这个模块不主动联网。调用方先准备本地 CSV（例如由
``public_catalog.download_public_gaia_catalog_for_frame`` 生成），再调用
``run_auto_photometric_workflow``。这样“允许联网”和“如何解释测光结果”
仍然是两个可审计的步骤，也不会因为 GUI 启动而产生隐式网络请求。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .catalog import CatalogSource, load_catalog_csv
from .fits import read_fits
from .models import FitsFrame
from .pipeline import FrameAnalysis, analyze_frame, recalibrate_frame_analysis
from .plate_solver import PlateSolveResult, solve_plate
from .wcs import AffineWCSCalibration, TangentPlaneWCS, fit_affine_wcs_from_matches


DEFAULT_CAMERA_PIXEL_SCALE_ARCSEC = 8.5
DEFAULT_PLATE_SCALE_TOLERANCE = 0.04
DEFAULT_PLATE_MIN_MATCHES = 8
DEFAULT_PLATE_MIN_COVERAGE = 0.02
DEFAULT_PLATE_MAX_RMS_PX = 2.0
DEFAULT_PLATE_MAX_LOO_RMS_PX = 3.0


ProgressCallback = Callable[[float, str], object]


@dataclass(frozen=True, slots=True)
class AutoPhotometricResult:
    """自动板解与 Gaia 经验测光的完整结果。

    ``analysis`` 在板解失败时仍保留单帧检测结果，但此时
    ``affine_wcs`` 为 ``None``，调用方只能展示 ``m_inst``。只有
    ``calibrated`` 为真时，才允许把 ``analysis`` 中的标定星等作为主结果。
    """

    analysis: FrameAnalysis
    catalog_sources: tuple[CatalogSource, ...]
    catalog_path: Path | None
    reference_wcs: TangentPlaneWCS
    plate_solution: PlateSolveResult
    affine_wcs: AffineWCSCalibration | None
    status: str
    reason: str
    catalog_provenance_status: str = "IN_MEMORY_UNVERIFIED"

    @property
    def calibrated(self) -> bool:
        calibration = self.analysis.photometric_calibration
        return (
            self.affine_wcs is not None
            and self.plate_solution.valid
            and calibration is not None
            and calibration.status in {"VALID", "VALID_NO_HOLDOUT"}
            and self.catalog_provenance_status in {"COMPLETE", "IN_MEMORY_UNVERIFIED"}
        )

    @property
    def photometric_status(self) -> str | None:
        calibration = self.analysis.photometric_calibration
        return None if calibration is None else calibration.status

    @property
    def faintest_global_claim(self) -> bool:
        """Whether the result supports a whole-frame calibrated faintest claim.

        A valid calibration can cover only the matched subset of detections.
        Keep that distinction explicit so a successful local fit is not
        presented as proof that no unmatched source in the frame is fainter.
        """

        faintest = self.analysis.faintest
        return bool(
            self.calibrated
            and faintest is not None
            and faintest.selection_scope == "CALIBRATED_MATCHES"
        )

    def as_dict(self, *, include_all_sources: bool = False) -> dict[str, object]:
        """返回可审计 JSON；默认不重复展开全量检测源。

        自动测光的主要证据是检测统计、通过验收的 WCS、匹配星和逐源
        光度结果。``FrameAnalysis.as_dict()`` 为了缓存/研究审计会保留
        每一个检测源；如果这里无条件再次嵌入它，单帧全量候选会把一个
        本应轻量的 JSON 放大到百 MB 级。需要完整源表时仍可显式传入
        ``include_all_sources=True``，默认导出则保留计数和已匹配测光行。
        """

        analysis_payload = self.analysis.as_dict()
        detection_payload = analysis_payload.get("detection")
        matching_payload = analysis_payload.get("matching")
        source_rows = analysis_payload.get("source_photometry")
        if not include_all_sources:
            if isinstance(detection_payload, dict):
                detection_payload["sources"] = []
                detection_payload["evidence_scope"] = "summary_only"
            if isinstance(matching_payload, dict):
                unmatched_detection_ids = matching_payload.pop("unmatched_detection_ids", [])
                unmatched_catalog_ids = matching_payload.pop("unmatched_catalog_ids", [])
                matching_payload["unmatched_detection_count"] = len(unmatched_detection_ids)
                matching_payload["unmatched_catalog_count"] = len(unmatched_catalog_ids)
                matching_payload["evidence_scope"] = "matches_and_counts"
            if isinstance(source_rows, list):
                analysis_payload["source_photometry"] = [
                    row for row in source_rows if isinstance(row, dict) and row.get("source_id") is not None
                ]
            analysis_payload["evidence_scope"] = "compact_auto_photometry"
            analysis_payload["source_photometry_total_count"] = len(source_rows) if isinstance(source_rows, list) else 0
            analysis_payload["source_photometry_included_count"] = (
                len(analysis_payload["source_photometry"])
                if isinstance(analysis_payload.get("source_photometry"), list)
                else 0
            )
        else:
            analysis_payload["evidence_scope"] = "full_source_evidence"

        return {
            "schema_version": 1,
            "status": self.status,
            "calibrated": self.calibrated,
            "reason": self.reason,
            "frame_path": str(self.analysis.frame.path),
            "catalog_path": str(self.catalog_path) if self.catalog_path is not None else None,
            "catalog_provenance_status": self.catalog_provenance_status,
            "catalog_source_count": len(self.catalog_sources),
            "reference_wcs": self.reference_wcs.as_dict(),
            "plate_solution": self.plate_solution.as_dict(),
            "affine_wcs": self.affine_wcs.as_dict() if self.affine_wcs is not None else None,
            "photometric_status": self.photometric_status,
            "faintest_global_claim": self.faintest_global_claim,
            "faintest_selection_scope": (
                self.analysis.faintest.selection_scope
                if self.analysis.faintest is not None
                else None
            ),
            "analysis": analysis_payload,
            "interpretation": (
                "m_cal is an empirically calibrated apparent magnitude in the declared catalog system; "
                "it is not an automatic claim of camera-specific radiometric calibration."
            ),
        }


def _report(progress: ProgressCallback | None, value: float, label: str) -> None:
    if progress is not None:
        progress(max(0.0, min(100.0, float(value))), str(label))


def _resolve_catalog(catalog: str | Path | Sequence[CatalogSource]) -> tuple[tuple[CatalogSource, ...], Path | None]:
    if isinstance(catalog, (str, Path)):
        path = Path(catalog).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"星表文件不存在：{path}")
        sources = tuple(load_catalog_csv(path))
        catalog_path: Path | None = path
    else:
        sources = tuple(catalog)
        catalog_path = None
    if not sources:
        raise ValueError("星表为空，无法进行自动板解")
    return sources, catalog_path


def _catalog_provenance_status(path: Path | None) -> str:
    """Return the evidence status of a CSV's sidecar audit.

    A hand-supplied in-memory catalog remains usable for diagnostic tests.
    A file-backed catalog without an explicit sidecar is deliberately marked
    provisional; this prevents an old or truncated CSV from silently becoming
    a final calibrated result.
    """

    if path is None:
        return "IN_MEMORY_UNVERIFIED"
    audit_path = path.with_suffix(path.suffix + ".meta.json")
    if not audit_path.is_file():
        return "PROVENANCE_UNKNOWN"
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "PROVENANCE_INVALID"
    if not isinstance(payload, Mapping):
        return "PROVENANCE_INVALID"
    if payload.get("complete") is False:
        return "INCOMPLETE"
    if payload.get("complete") is not True:
        return "PROVENANCE_UNKNOWN"
    if payload.get("coverage_warning"):
        return "COMPLETE_WITH_WARNING"
    return "COMPLETE"


def _affine_wcs_rejection_reason(
    affine_wcs: AffineWCSCalibration,
    *,
    min_matches: int,
    max_rms_residual_px: float,
    max_leave_one_out_rms_px: float,
) -> str | None:
    """Return why a refined WCS cannot be used for photometric calibration.

    ``solve_plate`` validates its own affine candidate before the refinement,
    but the second fit can change the inlier set and residuals.  Requiring a
    finite leave-one-out result here prevents a locally good-looking fit from
    silently becoming the coordinate basis for ``m_cal``.
    """

    failures: list[str] = []
    if affine_wcs.inlier_count < min_matches:
        failures.append(f"仿射内点数 {affine_wcs.inlier_count} < {min_matches}")
    if not math.isfinite(float(affine_wcs.rms_residual_px)):
        failures.append("仿射 RMS 无效")
    elif affine_wcs.rms_residual_px > max_rms_residual_px:
        failures.append(
            f"仿射 RMS {affine_wcs.rms_residual_px:.3f}px > {max_rms_residual_px:.3f}px"
        )
    if not math.isfinite(float(affine_wcs.condition_number)) or affine_wcs.condition_number > 1.0e10:
        failures.append("仿射条件数无效或过大")
    if not math.isfinite(float(affine_wcs.anisotropy_ratio)) or affine_wcs.anisotropy_ratio <= 0.0:
        failures.append("仿射各向异性指标无效")
    loo_rms = affine_wcs.leave_one_out_rms_residual_px
    if affine_wcs.validation_count < 1 or loo_rms is None:
        failures.append("仿射留一验证无有效样本")
    elif not math.isfinite(float(loo_rms)):
        failures.append("仿射留一残差无效")
    elif loo_rms > max_leave_one_out_rms_px:
        failures.append(
            f"仿射留一 RMS {loo_rms:.3f}px > {max_leave_one_out_rms_px:.3f}px"
        )
    return "；".join(failures) if failures else None


def run_auto_photometric_workflow(
    frame: str | Path | FitsFrame,
    catalog: str | Path | Sequence[CatalogSource],
    *,
    pixel_scale_arcsec: float = DEFAULT_CAMERA_PIXEL_SCALE_ARCSEC,
    match_radius_px: float = 3.0,
    epoch: float | None = None,
    scale_tolerance: float = DEFAULT_PLATE_SCALE_TOLERANCE,
    min_matches: int = DEFAULT_PLATE_MIN_MATCHES,
    min_coverage_area: float = DEFAULT_PLATE_MIN_COVERAGE,
    max_rms_residual_px: float = DEFAULT_PLATE_MAX_RMS_PX,
    max_leave_one_out_rms_px: float = DEFAULT_PLATE_MAX_LOO_RMS_PX,
    photometric_min_calibrators: int = 6,
    detector_kwargs: Mapping[str, Any] | None = None,
    initial_analysis: FrameAnalysis | None = None,
    progress: ProgressCallback | None = None,
) -> AutoPhotometricResult:
    """用相机尺度先验自动板解，并在通过后拟合 Gaia G 经验星等。

    该函数的“自动”指自动尝试旋转、奇偶性、平移和局部仿射细化，
    不是无先验全天空盲解。视轴中心取自 FITS 辅助数据，像元尺度取
    相机先验；任何验收失败都只返回未标定的单帧检测结果。
    """

    loaded = frame if isinstance(frame, FitsFrame) else read_fits(frame)
    if loaded.auxiliary is None:
        raise ValueError("FITS 没有可用的辅助 RA/DEC，无法自动规划 WCS")
    sources, catalog_path = _resolve_catalog(catalog)
    catalog_provenance_status = _catalog_provenance_status(catalog_path)
    try:
        scale = float(pixel_scale_arcsec)
        match_radius = float(match_radius_px)
    except (TypeError, ValueError) as exc:
        raise ValueError("像元尺度和匹配半径必须是数字") from exc
    if scale <= 0.0 or match_radius <= 0.0:
        raise ValueError("像元尺度和匹配半径必须为正数")
    if initial_analysis is not None and initial_analysis.frame.path.resolve() != loaded.path.resolve():
        raise ValueError("initial_analysis 与当前 FITS 不是同一帧")

    reference_wcs = TangentPlaneWCS(
        center_ra_deg=loaded.auxiliary.ra_deg,
        center_dec_deg=loaded.auxiliary.dec_deg,
        pixel_scale_arcsec=scale,
        crpix_x=loaded.width / 2.0,
        crpix_y=loaded.height / 2.0,
        rotation_deg=0.0,
        parity=1,
    )
    _report(progress, 2.0, "准备自动 Gaia 测光")
    if initial_analysis is None:
        options = dict(detector_kwargs or {})
        forbidden = {"catalog", "wcs", "fit_photometry", "progress"}.intersection(options)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"detector_kwargs 不允许覆盖工作流参数：{names}")

        def detection_progress(value: float, label: str) -> None:
            _report(progress, 5.0 + float(value) * 0.62, label)

        analysis = analyze_frame(loaded, progress=detection_progress, **options)
    else:
        analysis = initial_analysis
        _report(progress, 67.0, "复用当前帧检测结果")

    _report(progress, 72.0, "正在用 Gaia 星对搜索 WCS")
    plate_solution = solve_plate(
        analysis.detection.quality_sources,
        sources,
        reference_wcs,
        epoch=epoch,
        image_shape=loaded.data.shape,
        scale_tolerance=scale_tolerance,
        match_radius_px=match_radius,
        min_matches=min_matches,
        min_coverage_area=min_coverage_area,
        max_rms_residual_px=max_rms_residual_px,
        max_leave_one_out_rms_px=max_leave_one_out_rms_px,
    )
    if not plate_solution.valid or plate_solution.best is None:
        _report(progress, 100.0, f"自动板解未通过 · {plate_solution.reason}")
        return AutoPhotometricResult(
            analysis=analysis,
            catalog_sources=sources,
            catalog_path=catalog_path,
            reference_wcs=reference_wcs,
            plate_solution=plate_solution,
            affine_wcs=None,
            status=f"WCS_{plate_solution.status}",
            reason=f"自动板解未通过：{plate_solution.reason}；仅保留 m_inst。",
            catalog_provenance_status=catalog_provenance_status,
        )

    _report(progress, 82.0, f"板解通过 · 匹配 {plate_solution.best.matched_count} · 正在细化仿射 WCS")
    try:
        affine_wcs = fit_affine_wcs_from_matches(
            plate_solution.best.matches,
            sources,
            reference_wcs,
            epoch=epoch,
            min_matches=min(6, min_matches),
        )
    except (OSError, ValueError) as exc:
        _report(progress, 100.0, f"WCS 细化失败 · {exc}")
        return AutoPhotometricResult(
            analysis=analysis,
            catalog_sources=sources,
            catalog_path=catalog_path,
            reference_wcs=reference_wcs,
            plate_solution=plate_solution,
            affine_wcs=None,
            status="WCS_REFINEMENT_FAILED",
            reason=f"星对板解通过但仿射细化失败：{exc}；仅保留 m_inst。",
            catalog_provenance_status=catalog_provenance_status,
        )

    affine_gate_reason = _affine_wcs_rejection_reason(
        affine_wcs,
        min_matches=min(6, min_matches),
        max_rms_residual_px=max_rms_residual_px,
        max_leave_one_out_rms_px=max_leave_one_out_rms_px,
    )
    if affine_gate_reason is not None:
        _report(progress, 100.0, f"仿射 WCS 验收失败 · {affine_gate_reason}")
        return AutoPhotometricResult(
            analysis=analysis,
            catalog_sources=sources,
            catalog_path=catalog_path,
            reference_wcs=reference_wcs,
            plate_solution=plate_solution,
            affine_wcs=None,
            status="WCS_REFINEMENT_REJECTED",
            reason=f"仿射 WCS 验收失败：{affine_gate_reason}；仅保留 m_inst。",
            catalog_provenance_status=catalog_provenance_status,
        )

    first_source = sources[0]
    system = first_source.photometric_system or "Gaia Vega"
    band = first_source.photometric_band or "G"
    color_name = first_source.color_name or "BP-RP"
    _report(progress, 90.0, "WCS 已细化 · 正在拟合多参考星光度零点")
    try:
        refined_analysis = recalibrate_frame_analysis(
            analysis,
            sources,
            affine_wcs,
            match_radius_px=match_radius,
            epoch=epoch,
            photometric_system=system,
            photometric_band=band,
            photometric_color_name=color_name,
            photometric_color_order=1,
            photometric_min_calibrators=photometric_min_calibrators,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _report(progress, 100.0, f"光度层失败 · {exc}")
        return AutoPhotometricResult(
            analysis=analysis,
            catalog_sources=sources,
            catalog_path=catalog_path,
            reference_wcs=reference_wcs,
            plate_solution=plate_solution,
            affine_wcs=affine_wcs,
            status="WCS_PHOTOMETRY_ERROR",
            reason=f"WCS 已通过但光度层发生异常：{exc}；仅保留 m_inst。",
            catalog_provenance_status=catalog_provenance_status,
        )
    calibration = refined_analysis.photometric_calibration
    if calibration is not None and calibration.status in {"VALID", "VALID_NO_HOLDOUT"}:
        selection_scope = (
            refined_analysis.faintest.selection_scope
            if refined_analysis.faintest is not None
            else "NO_FAINTEST_RESULT"
        )
        scope_status = (
            "CALIBRATED"
            if selection_scope in {"CALIBRATED_MATCHES", "NO_FAINTEST_RESULT"}
            else "CALIBRATED_PARTIAL"
        )
        status = (
            scope_status
            if catalog_provenance_status in {"COMPLETE", "IN_MEMORY_UNVERIFIED"}
            else "CALIBRATED_PROVISIONAL_REFERENCE"
        )
        reason = (
            f"WCS 与 Gaia 经验光度标定通过：参考星 {calibration.calibrator_count}，"
            f"光度状态 {calibration.status}；最暗判定集合为 {selection_scope}。"
            f"主结果为 {system}/{band} 表观星等。"
            + (
                f" 星表完整性状态为 {catalog_provenance_status}，暂不作为正式校准结果。"
                if status == "CALIBRATED_PROVISIONAL_REFERENCE"
                else ""
            )
        )
    else:
        status = f"WCS_VALID_PHOTOMETRY_{calibration.status if calibration is not None else 'MISSING'}"
        reason = (
            f"WCS 通过，但光度标定未通过（{calibration.status if calibration is not None else '无结果'}）；"
            "仅保留 m_inst，不能把未通过的数值作为主星等。"
        )
    _report(progress, 100.0, reason)
    return AutoPhotometricResult(
        analysis=refined_analysis,
        catalog_sources=sources,
        catalog_path=catalog_path,
        reference_wcs=reference_wcs,
        plate_solution=plate_solution,
        affine_wcs=affine_wcs,
        status=status,
        reason=reason,
        catalog_provenance_status=catalog_provenance_status,
    )


__all__ = [
    "AutoPhotometricResult",
    "DEFAULT_CAMERA_PIXEL_SCALE_ARCSEC",
    "DEFAULT_PLATE_MAX_LOO_RMS_PX",
    "DEFAULT_PLATE_MAX_RMS_PX",
    "DEFAULT_PLATE_MIN_COVERAGE",
    "DEFAULT_PLATE_MIN_MATCHES",
    "DEFAULT_PLATE_SCALE_TOLERANCE",
    "run_auto_photometric_workflow",
]
