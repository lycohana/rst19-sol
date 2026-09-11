from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rst19 import pipeline
from rst19.catalog import CatalogSource
from rst19.detection import Detection, DetectionResult
from rst19.models import FitsFrame
from rst19.photometry import instrumental_magnitude
from rst19.wcs import TangentPlaneWCS, fit_affine_wcs_from_matches


def test_analyze_frame_wires_catalog_matching_calibration_and_absolute_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wcs = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=1.0,
        crpix_x=64.0,
        crpix_y=64.0,
    )
    detections: list[Detection] = []
    catalog: list[CatalogSource] = []
    for detection_id in range(12):
        x = 35.0 + (detection_id % 4) * 18.0
        y = 38.0 + (detection_id // 4) * 18.0
        flux = 100.0 + detection_id * 8.0
        detection = Detection(
            detection_id=detection_id,
            x=x,
            y=y,
            peak=flux,
            flux=flux,
            background=10.0,
            noise=2.0,
            snr=20.0,
            fwhm=2.0,
            flags=(),
            flux_error=1.0,
            flux_snr=20.0,
            filter_snr=20.0,
            fwhm_x=2.0,
            fwhm_y=2.0,
            ellipticity=0.0,
            sharpness=0.2,
            footprint_pixels=10,
            psf_support_pixels=5,
            quality_passed=True,
        )
        detections.append(detection)
        ra, dec = wcs.pixel_to_world(x, y)
        color = 0.1 + detection_id * 0.2
        instrumental = instrumental_magnitude(flux)
        assert instrumental is not None
        catalog.append(
            CatalogSource(
                source_id=f"gaia-{detection_id}",
                ra_deg=float(ra),
                dec_deg=float(dec),
                magnitude=instrumental + 21.0 + 0.25 * color,
                magnitude_error=0.01,
                color=color,
                color_name="BP-RP",
                catalog_name="Gaia DR3",
                photometric_system="Gaia Vega",
                photometric_band="G",
                parallax_mas=10.0,
                parallax_error_mas=0.1,
                extinction_mag=0.2,
                extinction_error_mag=0.03,
            )
        )

    detections_tuple = tuple(detections)
    detection_result = DetectionResult(
        image_shape=(128, 128),
        background=10.0,
        noise=2.0,
        threshold=8.0,
        candidate_count=len(detections_tuple),
        sources=detections_tuple,
        parameters={},
        quality_count=len(detections_tuple),
    )
    monkeypatch.setattr(pipeline, "detect_sources", lambda *args, **kwargs: detection_result)
    frame = FitsFrame(
        path=Path("synthetic.fits"),
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
        photometric_color_order=1,
        photometric_min_calibrators=6,
    )

    assert result.matching is not None
    assert result.matching.matched_count == len(detections_tuple)
    assert result.photometric_calibration is not None
    assert result.photometric_calibration.status == "VALID"
    assert result.photometric_calibration.zero_point == pytest.approx(21.0, abs=1e-6)
    assert result.photometric_calibration.color_coefficient == pytest.approx(0.25, abs=1e-6)
    assert len(result.source_photometry) == len(detections_tuple)
    assert all(row.status == "CALIBRATED" for row in result.source_photometry)
    assert all(row.calibrated_magnitude is not None for row in result.source_photometry)
    assert all(row.absolute_magnitude is not None for row in result.source_photometry)
    assert all(row.absolute_magnitude.status == "VALID" for row in result.source_photometry if row.absolute_magnitude)

    assert result.matching is not None
    affine = fit_affine_wcs_from_matches(result.matching.matches, tuple(catalog), wcs, min_matches=6)
    refined = pipeline.recalibrate_frame_analysis(
        result,
        tuple(catalog),
        affine,
        match_radius_px=3.0,
    )
    assert refined.matching is not None
    assert refined.matching.matched_count == len(detections_tuple)
    assert refined.photometric_calibration is not None
    assert refined.photometric_calibration.status == "VALID"
    assert refined.faintest is not None
    assert refined.faintest.selection_scope == "CALIBRATED_MATCHES"
