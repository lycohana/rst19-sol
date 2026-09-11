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


def test_affine_wcs_world_to_pixel_preserves_full_matrix_for_rematching() -> None:
    reference_wcs = TangentPlaneWCS(
        center_ra_deg=129.5,
        center_dec_deg=-1.8,
        pixel_scale_arcsec=8.5,
        crpix_x=1000.0,
        crpix_y=900.0,
    )
    tangent_points = np.asarray(
        ((-1200.0, -700.0), (-800.0, 1100.0), (-150.0, -250.0), (500.0, 900.0), (1200.0, -600.0), (1500.0, 800.0)),
        dtype=np.float64,
    )
    ra, dec = reference_wcs.pixel_to_world(
        reference_wcs.crpix_x + tangent_points[:, 0] / reference_wcs.pixel_scale_arcsec,
        reference_wcs.crpix_y + tangent_points[:, 1] / reference_wcs.pixel_scale_arcsec,
    )
    catalog = tuple(
        CatalogSource(
            source_id=f"affine-{index}",
            ra_deg=float(ra[index]),
            dec_deg=float(dec[index]),
            magnitude=10.0 + index,
        )
        for index in range(len(tangent_points))
    )
    matrix = np.asarray(((0.105, 0.004), (-0.003, 0.116)), dtype=np.float64)
    offset = np.asarray((1050.0, 875.0), dtype=np.float64)
    pixels = tangent_points @ matrix.T + offset
    detections = tuple(_detection(index, *pixels[index]) for index in range(len(pixels)))
    initial_matches = tuple(
        CatalogMatch(
            detection_id=index,
            source_id=f"affine-{index}",
            detection_x=float(pixels[index, 0]),
            detection_y=float(pixels[index, 1]),
            predicted_x=float(pixels[index, 0]),
            predicted_y=float(pixels[index, 1]),
            residual_px=0.0,
            catalog_magnitude=catalog[index].magnitude,
        )
        for index in range(len(pixels))
    )
    affine = fit_affine_wcs_from_matches(initial_matches, catalog, reference_wcs, min_matches=6)

    predicted_x, predicted_y = affine.world_to_pixel(ra, dec)
    np.testing.assert_allclose(predicted_x, pixels[:, 0], atol=1e-8)
    np.testing.assert_allclose(predicted_y, pixels[:, 1], atol=1e-8)
    rematched = match_detections(detections, catalog, affine, radius_px=1e-6)

    assert rematched.matched_count == len(detections)
    assert rematched.rms_residual_px is not None and rematched.rms_residual_px < 1e-8


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


def test_global_assignment_preserves_cardinality_in_a_competing_local_component() -> None:
    wcs = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=10.0,
        crpix_x=0.0,
        crpix_y=0.0,
    )
    catalog_pixels = np.array(((0.0, 0.0), (2.0, 0.0)), dtype=np.float64)
    ra, dec = wcs.pixel_to_world(catalog_pixels[:, 0], catalog_pixels[:, 1])
    catalog = tuple(
        CatalogSource(source_id=f"s{index}", ra_deg=float(ra[index]), dec_deg=float(dec[index]))
        for index in range(2)
    )
    detections = (
        _detection(0, 1.0, 0.0),
        _detection(1, 0.9, 0.9),
    )

    greedy = match_detections(detections, catalog, wcs, radius_px=1.3, assignment_mode="greedy")
    global_result = match_detections(detections, catalog, wcs, radius_px=1.3, assignment_mode="global")
    default_result = match_detections(detections, catalog, wcs, radius_px=1.3)

    assert greedy.matched_count == 1
    assert global_result.matched_count == 2
    assert default_result == global_result
    assert [(match.detection_id, match.source_id) for match in global_result.matches] == [(0, "s1"), (1, "s0")]
    assert global_result.assignment_mode == "global"


def test_match_detections_rejects_unknown_assignment_mode() -> None:
    wcs = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=10.0,
        crpix_x=0.0,
        crpix_y=0.0,
    )

    with pytest.raises(ValueError, match="assignment_mode"):
        match_detections((), (), wcs, assignment_mode="nearest")


def test_load_catalog_csv_supports_gaia_column_aliases(tmp_path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "source_id,ra,dec,phot_g_mean_mag,pmra,pmdec,ref_epoch,"
        "phot_g_mean_flux_over_error,phot_bp_rp_excess_factor,ruwe,"
        "duplicated_source,visibility_periods_used,phot_variable_flag\n"
        "123,10.0,20.0,12.5,100.0,-50.0,2016.0,80,1.18,1.05,false,12,NOT_AVAILABLE\n",
        encoding="utf-8",
    )

    catalog = load_catalog_csv(path)

    assert len(catalog) == 1
    assert catalog[0].source_id == "123"
    assert catalog[0].magnitude == 12.5
    assert catalog[0].catalog_name == "Gaia DR3"
    assert catalog[0].photometric_system == "Gaia Vega"
    assert catalog[0].photometric_band == "G"
    assert catalog[0].magnitude_source == "Gaia DR3 phot_g_mean_mag"
    assert catalog[0].phot_g_mean_flux_over_error == 80.0
    assert catalog[0].phot_bp_rp_excess_factor == 1.18
    assert catalog[0].ruwe == 1.05
    assert catalog[0].duplicated_source is False
    assert catalog[0].visibility_periods_used == 12
    assert catalog[0].phot_variable_flag == "NOT_AVAILABLE"
    propagated = catalog[0].at_epoch(2017.0)
    assert propagated.ra_deg != catalog[0].ra_deg
    assert propagated.dec_deg != catalog[0].dec_deg
    assert propagated.ruwe == catalog[0].ruwe


def test_load_catalog_csv_rejects_duplicate_source_ids(tmp_path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "source_id,ra,dec\n"
        "same,10.0,20.0\n"
        "same,11.0,21.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate source_id"):
        load_catalog_csv(path)


def test_load_catalog_csv_rejects_non_finite_optional_values(tmp_path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "source_id,ra,dec,phot_g_mean_mag,pmra\n"
        "s1,10.0,20.0,nan,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite magnitude"):
        load_catalog_csv(path)


def test_load_catalog_csv_rejects_conflicting_generic_and_gaia_magnitudes(tmp_path) -> None:
    path = tmp_path / "conflicting-magnitude.csv"
    path.write_text(
        "source_id,ra,dec,magnitude,phot_g_mean_mag\n"
        "s1,10.0,20.0,12.5,13.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="magnitude and phot_g_mean_mag conflict"):
        load_catalog_csv(path)


def test_load_catalog_csv_records_generic_magnitude_provenance(tmp_path) -> None:
    path = tmp_path / "generic-magnitude.csv"
    path.write_text(
        "source_id,ra,dec,magnitude\n"
        "s1,10.0,20.0,12.5\n",
        encoding="utf-8",
    )

    source = load_catalog_csv(path)[0]

    assert source.magnitude == 12.5
    assert source.magnitude_source == "CSV magnitude column"
    assert source.catalog_name == "unknown"
    assert source.photometric_band == "unknown"


def test_load_catalog_csv_infers_bailer_jones_distance_provenance(tmp_path) -> None:
    path = tmp_path / "bailer-jones-distance.csv"
    path.write_text(
        "source_id,ra,dec,phot_g_mean_mag,r_med_geo,r_lo_geo,r_hi_geo\n"
        "s1,10.0,20.0,12.5,100.0,95.0,106.0\n",
        encoding="utf-8",
    )

    source = load_catalog_csv(path)[0]

    assert source.distance_pc == 100.0
    assert source.distance_lower_pc == 95.0
    assert source.distance_upper_pc == 106.0
    assert source.distance_source == "Bailer-Jones Gaia DR3 geometric posterior"


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
