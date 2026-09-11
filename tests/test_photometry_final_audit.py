from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rst19 import pipeline
from rst19.catalog import CatalogSource
from rst19.detection import Detection, DetectionResult
from rst19.matching import CatalogMatch
from rst19.models import FitsFrame
from rst19.photometry import (
    AbsoluteMagnitudeEstimate,
    PhotometricCalibration,
    absolute_magnitude_estimate_from_parallax,
    build_source_photometry,
    find_faintest_source,
    fit_photometric_calibration,
    instrumental_magnitude,
)
from rst19.wcs import TangentPlaneWCS


def _detection(
    detection_id: int,
    flux: float,
    *,
    snr: float = 20.0,
    flags: tuple[str, ...] = (),
    quality_passed: bool = True,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=20.0 + detection_id * 8.0,
        y=30.0,
        peak=flux,
        flux=flux,
        background=10.0,
        noise=2.0,
        snr=snr,
        fwhm=2.0,
        flags=flags,
        flux_error=1.0,
        flux_snr=snr,
        filter_snr=snr,
        fwhm_x=2.0,
        fwhm_y=2.0,
        ellipticity=0.0,
        sharpness=0.2,
        footprint_pixels=10,
        psf_support_pixels=5,
        quality_passed=quality_passed,
    )


def _calibration(
    *,
    system: str = "Gaia Vega",
    band: str = "G",
    color_name: str = "BP-RP",
    coefficients: tuple[float, ...] = (20.0, 0.0),
) -> PhotometricCalibration:
    return PhotometricCalibration(
        photometric_system=system,
        photometric_band=band,
        color_name=color_name,
        color_order=len(coefficients) - 1,
        coefficients=coefficients,
        calibrator_count=8,
        inlier_count=8,
        validation_count=2,
        fit_rms_mag=0.01,
        validation_rms_mag=0.02,
        residual_mad_mag=0.01,
        color_min=0.0 if len(coefficients) > 1 else None,
        color_max=2.0 if len(coefficients) > 1 else None,
        status="VALID",
    )


def _matched_catalog(
    detection: Detection,
    *,
    source_id: str = "gaia-1",
    color: float = 0.5,
    extinction_mag: float | None = 0.2,
    parallax_mas: float | None = 10.0,
    parallax_error_mas: float | None = 0.1,
    photometric_system: str = "Gaia Vega",
    photometric_band: str = "G",
) -> tuple[CatalogSource, CatalogMatch]:
    source = CatalogSource(
        source_id=source_id,
        ra_deg=10.0,
        dec_deg=20.0,
        magnitude=13.0,
        magnitude_error=0.02,
        color=color,
        color_name="BP-RP",
        catalog_name="Gaia DR3",
        photometric_system=photometric_system,
        photometric_band=photometric_band,
        parallax_mas=parallax_mas,
        parallax_error_mas=parallax_error_mas,
        extinction_mag=extinction_mag,
        extinction_error_mag=0.03 if extinction_mag is not None else None,
        extinction_band="G" if extinction_mag is not None else "unknown",
        extinction_system="Gaia" if extinction_mag is not None else "unknown",
        extinction_source=("Gaia DR3 GSP-Phot: ag_gspphot" if extinction_mag is not None else "unknown"),
    )
    match = CatalogMatch(
        detection_id=detection.detection_id,
        source_id=source_id,
        detection_x=detection.x,
        detection_y=detection.y,
        predicted_x=detection.x,
        predicted_y=detection.y,
        residual_px=0.1,
        catalog_magnitude=source.magnitude,
    )
    return source, match


def test_final_audit_keeps_m_inst_m_cal_and_M_as_separate_layers() -> None:
    detection = _detection(1, 100.0)
    source, match = _matched_catalog(detection)

    without_calibration = build_source_photometry(
        (detection,),
        matches=(match,),
        catalog=(source,),
    )[0]
    assert without_calibration.instrumental_magnitude == pytest.approx(
        instrumental_magnitude(100.0)
    )
    assert without_calibration.calibrated_magnitude is None
    assert without_calibration.absolute_magnitude is None

    with_calibration = build_source_photometry(
        (detection,),
        matches=(match,),
        catalog=(source,),
        photometric_calibration=_calibration(),
    )[0]
    assert with_calibration.instrumental_magnitude is not None
    assert with_calibration.calibrated_magnitude == pytest.approx(
        with_calibration.instrumental_magnitude + 20.0
    )
    assert with_calibration.photometric_system == "Gaia Vega"
    assert with_calibration.photometric_band == "G"
    assert with_calibration.absolute_magnitude is not None
    assert with_calibration.absolute_magnitude.status == "VALID"
    assert with_calibration.absolute_magnitude.value is not None


def test_final_audit_never_crosses_gaia_g_into_v() -> None:
    detection = _detection(1, 100.0)
    source, match = _matched_catalog(detection)

    # A V-labelled calibration is not allowed to consume a Gaia G reference
    # row, even if its numerical coefficients look plausible.
    row = build_source_photometry(
        (detection,),
        matches=(match,),
        catalog=(source,),
        photometric_calibration=_calibration(band="V"),
    )[0]
    assert row.status == "CALIBRATION_METADATA_MISMATCH"
    assert row.calibrated_magnitude is None
    assert row.absolute_magnitude is None
    assert "CALIBRATION_NOT_APPLIED" in row.flags


def test_final_audit_requires_parallax_quality_and_extinction_for_numeric_M() -> None:
    no_extinction = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=0.1,
        extinction_mag=None,
    )
    assert no_extinction.value is None
    assert no_extinction.status == "VALID_NO_EXTINCTION"
    assert "EXTINCTION_NOT_PROVIDED" in no_extinction.flags

    no_parallax_error = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=None,
        extinction_mag=0.2,
    )
    assert no_parallax_error.value is None
    assert no_parallax_error.status == "NO_PARALLAX_ERROR"

    low_quality_parallax = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=3.0,
        extinction_mag=0.2,
    )
    assert low_quality_parallax.value is None
    assert low_quality_parallax.status == "LOW_PARALLAX_SNR"

    invalid_extinction = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=0.1,
        extinction_mag=-0.2,
    )
    assert invalid_extinction.value is None
    assert invalid_extinction.status == "INVALID_EXTINCTION"

    invalid_parallax = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=0.0,
        parallax_error_mas=0.1,
        extinction_mag=0.2,
    )
    assert invalid_parallax.value is None
    assert invalid_parallax.status == "INVALID_PARALLAX"


def test_final_audit_faintest_source_uses_valid_m_cal_not_M_or_rejected_peaks() -> None:
    sources = (
        _detection(0, 100.0),
        # This source has lower flux but a larger colour term, so it is the
        # faintest in the calibrated G system despite not being faintest in
        # m_inst.
        _detection(1, 200.0),
        _detection(2, 1.0, snr=3.0),
        _detection(3, 0.1, flags=("MASKED",)),
        _detection(4, 0.01, quality_passed=False),
    )
    result = find_faintest_source(
        sources,
        min_snr=5.0,
        photometric_calibration=_calibration(coefficients=(20.0, 2.0)),
        source_colors={0: 0.0, 1: 2.0, 2: 0.0, 3: 2.0, 4: 2.0},
        source_absolute_magnitudes={
            0: AbsoluteMagnitudeEstimate(99.0, None, 10.0, 100.0, 0.2, "VALID"),
            1: AbsoluteMagnitudeEstimate(-99.0, None, 10.0, 100.0, 0.2, "VALID"),
        },
    )

    assert result is not None
    assert result.detection_id == 1
    assert result.calibrated_magnitude is not None
    assert result.absolute_magnitude is not None
    assert result.absolute_magnitude.value == -99.0


def test_final_audit_pipeline_keeps_M_empty_when_extinction_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wcs = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=1.0,
        crpix_x=64.0,
        crpix_y=64.0,
    )
    detections = tuple(_detection(index, 100.0 + index * 10.0) for index in range(8))
    catalog: list[CatalogSource] = []
    for detection in detections:
        ra, dec = wcs.pixel_to_world(detection.x, detection.y)
        instrumental = instrumental_magnitude(detection.flux)
        assert instrumental is not None
        catalog.append(
            CatalogSource(
                source_id=f"gaia-{detection.detection_id}",
                ra_deg=float(ra),
                dec_deg=float(dec),
                magnitude=instrumental + 20.0,
                magnitude_error=0.02,
                color=0.2 + detection.detection_id * 0.1,
                color_name="BP-RP",
                catalog_name="Gaia DR3",
                photometric_system="Gaia Vega",
                photometric_band="G",
                parallax_mas=10.0,
                parallax_error_mas=0.1,
                extinction_mag=None,
            )
        )
    detection_result = DetectionResult(
        image_shape=(128, 128),
        background=10.0,
        noise=2.0,
        threshold=8.0,
        candidate_count=len(detections),
        sources=detections,
        parameters={},
        quality_count=len(detections),
    )
    monkeypatch.setattr(pipeline, "detect_sources", lambda *args, **kwargs: detection_result)
    frame = FitsFrame(
        path=Path("synthetic-final-audit.fits"),
        header={"EXPOSURE": 1000},
        data=np.zeros((128, 128), dtype=np.int16),
        auxiliary=None,
        data_offset=0,
    )

    result = pipeline.analyze_frame(
        frame,
        catalog=tuple(catalog),
        wcs=wcs,
        fit_photometry=True,
        photometric_system="Gaia Vega",
        photometric_band="G",
        photometric_color_name="BP-RP",
        photometric_min_calibrators=3,
    )

    assert result.photometric_calibration is not None
    assert result.photometric_calibration.status == "VALID"
    assert result.source_photometry
    assert all(row.calibrated_magnitude is not None for row in result.source_photometry)
    assert all(row.absolute_magnitude is not None for row in result.source_photometry)
    assert all(
        row.absolute_magnitude is not None
        and row.absolute_magnitude.value is None
        and row.absolute_magnitude.status == "VALID_NO_EXTINCTION"
        for row in result.source_photometry
    )
    assert result.faintest is not None
    assert result.faintest.absolute_magnitude is not None
    assert result.faintest.absolute_magnitude.value is None
