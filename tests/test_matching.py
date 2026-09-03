from __future__ import annotations

import numpy as np
import pytest

from rst19.catalog import CatalogSource, load_catalog_csv
from rst19.detection import Detection
from rst19.matching import CatalogMatch, match_detections
from rst19.wcs import TangentPlaneWCS, fit_affine_wcs_from_matches
from rst19.wcs_validation import validate_frame_matches, write_wcs_validation_artifacts


def _detection(index: int, x: float, y: float) -> Detection:
    return Detection(index, x, y, 100.0, 200.0, 10.0, 2.0, 45.0, 2.0, ())


def test_tangent_plane_round_trip() -> None:
    wcs = TangentPlaneWCS(
        center_ra_deg=129.5,
        center_dec_deg=-1.8,
        pixel_scale_arcsec=10.0,
        crpix_x=512.0,
        crpix_y=256.0,
        rotation_deg=17.0,
        parity=-1,
    )
    x = np.array([100.0, 512.0, 800.0])
    y = np.array([50.0, 256.0, 700.0])

    ra, dec = wcs.pixel_to_world(x, y)
    round_trip_x, round_trip_y = wcs.world_to_pixel(ra, dec)

    np.testing.assert_allclose(round_trip_x, x, atol=1e-8)
    np.testing.assert_allclose(round_trip_y, y, atol=1e-8)


def test_match_detections_is_one_to_one() -> None:
    wcs = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=20.0,
        crpix_x=50.0,
        crpix_y=50.0,
    )
    pixels_x = np.array([20.0, 50.0, 80.0])
    pixels_y = np.array([30.0, 50.0, 70.0])
    ra, dec = wcs.pixel_to_world(pixels_x, pixels_y)
    catalog = tuple(
        CatalogSource(source_id=f"s{index}", ra_deg=float(ra[index]), dec_deg=float(dec[index]), magnitude=10.0 + index)
        for index in range(3)
    )
    detections = (
        _detection(0, 20.4, 30.1),
        _detection(1, 50.2, 49.8),
        _detection(2, 80.0, 70.3),
        _detection(3, 50.1, 50.2),
    )

    result = match_detections(detections, catalog, wcs, radius_px=1.0)

    assert result.matched_count == 3
    assert result.unmatched_detection_ids == (1,)
    assert result.rms_residual_px is not None and result.rms_residual_px < 0.5


def test_load_catalog_csv_supports_gaia_column_aliases(tmp_path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "source_id,ra,dec,phot_g_mean_mag,pmra,pmdec,ref_epoch\n"
        "123,10.0,20.0,12.5,100.0,-50.0,2016.0\n",
        encoding="utf-8",
    )

    catalog = load_catalog_csv(path)

    assert len(catalog) == 1
    assert catalog[0].source_id == "123"
    assert catalog[0].magnitude == 12.5
    propagated = catalog[0].at_epoch(2017.0)
    assert propagated.ra_deg != catalog[0].ra_deg
    assert propagated.dec_deg != catalog[0].dec_deg


def test_fit_affine_wcs_recovers_local_scale_rotation_and_rejects_outlier() -> None:
    reference = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=20.0,
        crpix_x=512.0,
        crpix_y=512.0,
    )
    true_matrix = np.array(
        (
            (-0.041, 0.012),
            (0.012, 0.041),
        ),
        dtype=np.float64,
    )
    true_offset = np.array((480.0, 535.0), dtype=np.float64)
    sky_points = np.array(
        (
            (-180.0, -120.0),
            (-100.0, 140.0),
            (-20.0, -40.0),
            (70.0, 180.0),
            (130.0, -160.0),
            (210.0, 40.0),
            (35.0, -210.0),
        ),
        dtype=np.float64,
    )
    catalog: list[CatalogSource] = []
    matches: list[CatalogMatch] = []
    for index, (east, north) in enumerate(sky_points):
        tangent_ra, tangent_dec = reference.pixel_to_world(
            reference.crpix_x + east / reference.pixel_scale_arcsec,
            reference.crpix_y + north / reference.pixel_scale_arcsec,
        )
        source_id = f"s{index}"
        catalog.append(CatalogSource(source_id, float(tangent_ra), float(tangent_dec)))
        pixel = true_matrix @ np.array((east, north), dtype=np.float64) + true_offset
        if index == 6:
            pixel += np.array((35.0, -28.0))
        matches.append(CatalogMatch(index, source_id, float(pixel[0]), float(pixel[1]), float(pixel[0]), float(pixel[1]), 0.0, None))

    calibration = fit_affine_wcs_from_matches(matches, catalog, reference, min_matches=5)

    np.testing.assert_allclose(calibration.matrix_px_per_arcsec, true_matrix, atol=1e-10)
    np.testing.assert_allclose(calibration.offset_px, true_offset, atol=1e-10)
    assert calibration.matched_count == 7
    assert calibration.inlier_count == 6
    assert calibration.all_max_residual_px > 30.0
    assert calibration.rms_residual_px < 1e-8
    assert calibration.validation_count == 6
    assert calibration.leave_one_out_rms_residual_px is not None and calibration.leave_one_out_rms_residual_px < 1e-8
    assert calibration.plate_scale_arcsec_per_pixel == pytest.approx(1.0 / np.linalg.svd(true_matrix, compute_uv=False).mean())


def test_fit_affine_wcs_requires_non_collinear_matches() -> None:
    reference = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=20.0,
        crpix_x=50.0,
        crpix_y=50.0,
    )
    catalog: list[CatalogSource] = []
    matches: list[CatalogMatch] = []
    for index, east in enumerate((-30.0, 0.0, 30.0, 60.0, 90.0, 120.0)):
        ra, dec = reference.pixel_to_world(reference.crpix_x + east / 20.0, reference.crpix_y)
        source_id = f"line{index}"
        catalog.append(CatalogSource(source_id, float(ra), float(dec)))
        matches.append(CatalogMatch(index, source_id, 100.0 + east, 200.0, 0.0, 0.0, 0.0, None))

    with pytest.raises(ValueError, match="共线"):
        fit_affine_wcs_from_matches(matches, catalog, reference, min_matches=6)


def test_validate_frame_matches_reports_each_frame_and_writes_artifacts(tmp_path) -> None:
    reference = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=20.0,
        crpix_x=512.0,
        crpix_y=512.0,
    )
    sky_points = np.array(
        (
            (-180.0, -120.0),
            (-100.0, 140.0),
            (-20.0, -40.0),
            (70.0, 180.0),
            (130.0, -160.0),
            (210.0, 40.0),
        ),
        dtype=np.float64,
    )
    catalog: list[CatalogSource] = []
    tangent_by_id: dict[str, tuple[float, float]] = {}
    for index, (east, north) in enumerate(sky_points):
        ra, dec = reference.pixel_to_world(
            reference.crpix_x + east / reference.pixel_scale_arcsec,
            reference.crpix_y + north / reference.pixel_scale_arcsec,
        )
        source_id = f"s{index}"
        catalog.append(CatalogSource(source_id, float(ra), float(dec)))
        tangent_by_id[source_id] = (float(east), float(north))

    frame_matches: list[list[CatalogMatch]] = []
    for frame_index, (matrix, offset) in enumerate(
        (
            (np.array(((-0.04, 0.01), (0.01, 0.04))), np.array((480.0, 535.0))),
            (np.array(((-0.04, 0.01), (0.01, 0.04))), np.array((480.4, 534.7))),
        )
    ):
        matches: list[CatalogMatch] = []
        for detection_id, source in enumerate(catalog):
            pixel = matrix @ np.asarray(tangent_by_id[source.source_id]) + offset
            matches.append(CatalogMatch(detection_id, source.source_id, float(pixel[0]), float(pixel[1]), float(pixel[0]), float(pixel[1]), 0.0, None))
        frame_matches.append(matches)

    report = validate_frame_matches(
        frame_matches,
        catalog,
        reference,
        frame_paths=("frame-01.fits", "frame-02.fits"),
        min_matches=6,
    )

    assert report.frame_count == 2
    assert report.validated_count == 2
    assert report.validation_ratio == pytest.approx(1.0)
    assert all(row.leave_one_out_rms_residual_px is not None for row in report.frame_rows)
    assert report.frame_rows[0].plate_scale_arcsec_per_pixel == pytest.approx(report.frame_rows[1].plate_scale_arcsec_per_pixel)

    output = write_wcs_validation_artifacts(report, tmp_path / "wcs")
    assert (output / "wcs_frame_validation.csv").is_file()
    assert (output / "wcs_validation_report.json").is_file()
    assert (output / "wcs_validation_residual.png").is_file()
    assert (output / "wcs_validation_scale.png").is_file()
