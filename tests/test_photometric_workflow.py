from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from rst19 import photometric_workflow as workflow
from rst19.catalog import CatalogSource
from rst19.detection import Detection, DetectionResult
from rst19.matching import CatalogMatch
from rst19.models import AuxiliaryData, FitsFrame
from rst19.photometry import PhotometricCalibration
from rst19.pipeline import FrameAnalysis
from rst19.plate_solver import PlateSolveCandidate, PlateSolveResult, PlateTransform
from rst19.wcs import AffineWCSCalibration


def _frame(tmp_path: Path) -> FitsFrame:
    auxiliary = [0.0] * 26
    auxiliary[2] = 10.0
    auxiliary[3] = 20.0
    return FitsFrame(
        path=tmp_path / "frame.fits",
        header={"EXPOSURE": 1000},
        data=np.zeros((128, 128), dtype=np.int16),
        auxiliary=AuxiliaryData.from_values(auxiliary),
        data_offset=0,
    )


def _analysis(frame: FitsFrame) -> FrameAnalysis:
    source = Detection(
        detection_id=1,
        x=64.0,
        y=64.0,
        peak=100.0,
        flux=500.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=2.0,
        flags=(),
        flux_error=25.0,
        flux_snr=20.0,
        filter_snr=20.0,
        quality_passed=True,
    )
    detection = DetectionResult(
        image_shape=(128, 128),
        background=10.0,
        noise=2.0,
        threshold=8.0,
        candidate_count=1,
        sources=(source,),
        parameters={},
        quality_count=1,
    )
    return FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=None)


def _catalog() -> tuple[CatalogSource, ...]:
    return (
        CatalogSource(
            source_id="gaia-1",
            ra_deg=10.0,
            dec_deg=20.0,
            magnitude=10.0,
            color=0.6,
            color_name="BP-RP",
            catalog_name="Gaia DR3",
            photometric_system="Gaia Vega",
            photometric_band="G",
        ),
    )


def test_auto_workflow_keeps_detection_only_when_plate_solution_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _frame(tmp_path)
    analysis = _analysis(frame)
    monkeypatch.setattr(workflow, "analyze_frame", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(
        workflow,
        "solve_plate",
        lambda *_args, **_kwargs: PlateSolveResult(
            status="NO_SOLUTION",
            reason="synthetic negative control",
            best=None,
            alternatives=(),
            image_points_considered=1,
            catalog_points_considered=1,
            pair_hypotheses=0,
            unique_hypotheses=0,
            acceptance={},
        ),
    )
    progress: list[tuple[float, str]] = []

    result = workflow.run_auto_photometric_workflow(
        frame,
        _catalog(),
        progress=lambda value, label: progress.append((value, label)),
    )

    assert result.status == "WCS_NO_SOLUTION"
    assert not result.calibrated
    assert result.affine_wcs is None
    assert result.analysis is analysis
    assert progress[-1][0] == 100.0
    assert "仅保留 m_inst" in result.reason


def test_auto_workflow_only_calls_calibrated_result_after_wcs_and_photometry_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _frame(tmp_path)
    analysis = _analysis(frame)
    catalog = _catalog()
    match = CatalogMatch(
        detection_id=1,
        source_id="gaia-1",
        detection_x=64.0,
        detection_y=64.0,
        predicted_x=64.0,
        predicted_y=64.0,
        residual_px=0.1,
        catalog_magnitude=10.0,
    )
    candidate = PlateSolveCandidate(
        transform=PlateTransform(
            matrix_px_per_arcsec=((0.1, 0.0), (0.0, 0.1)),
            offset_px=(64.0, 64.0),
            plate_scale_arcsec_per_pixel=10.0,
            rotation_deg=0.0,
            parity=1,
            anisotropy_ratio=1.0,
        ),
        matches=(match,),
        rms_residual_px=0.1,
        max_residual_px=0.1,
        coverage_x=0.5,
        coverage_y=0.5,
        coverage_area=0.25,
        leave_one_out_rms_residual_px=0.1,
        leave_one_out_max_residual_px=0.1,
        seed_image_pair=(1, 1),
        seed_catalog_pair=("gaia-1", "gaia-1"),
    )
    plate_solution = PlateSolveResult(
        status="VALID",
        reason="synthetic positive control",
        best=candidate,
        alternatives=(),
        image_points_considered=1,
        catalog_points_considered=1,
        pair_hypotheses=1,
        unique_hypotheses=1,
        acceptance={},
    )
    affine = AffineWCSCalibration(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        matrix_px_per_arcsec=((0.1, 0.0), (0.0, 0.1)),
        offset_px=(64.0, 64.0),
        plate_scale_arcsec_per_pixel=10.0,
        rotation_deg=0.0,
        parity=1,
        anisotropy_ratio=1.0,
        matched_count=1,
        inlier_count=1,
        rms_residual_px=0.1,
        max_residual_px=0.1,
        all_rms_residual_px=0.1,
        all_max_residual_px=0.1,
        condition_number=1.0,
        inlier_source_ids=("gaia-1",),
    )
    photometric = PhotometricCalibration(
        photometric_system="Gaia Vega",
        photometric_band="G",
        color_name="BP-RP",
        color_order=1,
        coefficients=(20.0, 0.0),
        calibrator_count=6,
        inlier_count=6,
        validation_count=2,
        fit_rms_mag=0.1,
        validation_rms_mag=0.1,
        residual_mad_mag=0.05,
        color_min=0.2,
        color_max=1.0,
        status="VALID",
    )
    calibrated_analysis = replace(analysis, photometric_calibration=photometric)

    monkeypatch.setattr(workflow, "solve_plate", lambda *_args, **_kwargs: plate_solution)
    monkeypatch.setattr(workflow, "fit_affine_wcs_from_matches", lambda *_args, **_kwargs: affine)
    monkeypatch.setattr(workflow, "recalibrate_frame_analysis", lambda *_args, **_kwargs: calibrated_analysis)

    result = workflow.run_auto_photometric_workflow(
        frame,
        catalog,
        initial_analysis=analysis,
    )

    assert result.status == "CALIBRATED"
    assert result.calibrated
    assert result.affine_wcs is affine
    assert result.photometric_status == "VALID"
    assert "Gaia Vega/G" in result.reason

    compact = result.as_dict()
    full = result.as_dict(include_all_sources=True)
    assert compact["analysis"]["detection"]["sources"] == []
    assert compact["analysis"]["detection"]["returned_count"] == 1
    assert compact["analysis"]["evidence_scope"] == "compact_auto_photometry"
    assert len(full["analysis"]["detection"]["sources"]) == 1
    assert full["analysis"]["evidence_scope"] == "full_source_evidence"
