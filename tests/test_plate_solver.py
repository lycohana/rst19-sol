from __future__ import annotations

import math

import numpy as np

from rst19.catalog import CatalogSource
from rst19.detection import Detection
from rst19.plate_solver import PlateTransform, solve_plate
from rst19.wcs import TangentPlaneWCS


def _detection(index: int, x: float, y: float, *, snr: float = 100.0, flags: tuple[str, ...] = ()) -> Detection:
    return Detection(
        detection_id=index,
        x=float(x),
        y=float(y),
        peak=1000.0,
        flux=5000.0,
        background=10.0,
        noise=2.0,
        snr=snr,
        fwhm=2.0,
        flags=flags,
        flux_error=50.0,
        flux_snr=snr,
        filter_snr=snr,
        quality_passed=True,
    )


def _catalog_for_tangent_points(reference_wcs: TangentPlaneWCS, points: np.ndarray) -> tuple[CatalogSource, ...]:
    sources: list[CatalogSource] = []
    for index, (east, north) in enumerate(points):
        x = reference_wcs.crpix_x + float(east) / reference_wcs.pixel_scale_arcsec
        y = reference_wcs.crpix_y + float(north) / reference_wcs.pixel_scale_arcsec
        ra, dec = reference_wcs.pixel_to_world(x, y)
        sources.append(
            CatalogSource(
                source_id=f"gaia-{index}",
                ra_deg=float(ra),
                dec_deg=float(dec),
                magnitude=9.0 + index * 0.2,
                color=0.5 + index * 0.02,
                color_name="BP-RP",
                catalog_name="Gaia DR3",
                photometric_system="Gaia Vega",
                photometric_band="G",
            )
        )
    return tuple(sources)


def test_solve_plate_recovers_rotation_reflection_and_offset_from_star_pairs() -> None:
    reference_wcs = TangentPlaneWCS(
        center_ra_deg=129.5,
        center_dec_deg=-1.8,
        pixel_scale_arcsec=8.5,
        crpix_x=2000.0,
        crpix_y=2000.0,
    )
    tangent_points = np.asarray(
        (
            (-9000.0, -6500.0),
            (-6500.0, 8500.0),
            (-4200.0, 2200.0),
            (-1000.0, -9200.0),
            (1800.0, 7200.0),
            (4300.0, -3000.0),
            (6800.0, 6400.0),
            (9100.0, -7100.0),
            (10500.0, 1800.0),
            (11500.0, 9000.0),
        ),
        dtype=np.float64,
    )
    catalog = _catalog_for_tangent_points(reference_wcs, tangent_points)
    theta = math.radians(27.0)
    scale = 1.0 / 8.5
    # A reflection is intentional: the image-axis parity is unknown to the
    # solver and must be recovered from the pair geometry, not assumed.
    matrix = scale * np.asarray(
        ((math.cos(theta), math.sin(theta)), (math.sin(theta), -math.cos(theta))),
        dtype=np.float64,
    )
    known = PlateTransform(
        matrix_px_per_arcsec=(tuple(matrix[0]), tuple(matrix[1])),
        offset_px=(2100.0, 1840.0),
        plate_scale_arcsec_per_pixel=8.5,
        rotation_deg=27.0,
        parity=-1,
        anisotropy_ratio=1.0,
    )
    predicted = known.project_tangent(tangent_points)
    detections = tuple(
        _detection(index, x + (0.08 if index % 2 else -0.06), y + (0.05 if index % 3 else -0.04))
        for index, (x, y) in enumerate(predicted, start=1)
    )
    # Low-ranking clutter is present, but should not displace the high-quality
    # point set used for geometric hypotheses.
    detections += tuple(_detection(100 + index, 300.0 + index * 17.0, 3200.0 - index * 23.0, snr=8.0) for index in range(8))

    result = solve_plate(
        detections,
        catalog,
        reference_wcs,
        image_shape=(4000, 4000),
        scale_tolerance=0.04,
        min_pair_distance_px=40.0,
        match_radius_px=1.0,
        min_matches=8,
        min_coverage_area=0.01,
        max_rms_residual_px=0.5,
        max_leave_one_out_rms_px=1.0,
        max_image_points=24,
        max_catalog_points=24,
    )

    assert result.valid, result.as_dict()
    assert result.best is not None
    assert result.best.matched_count == 10
    assert result.best.transform.parity == -1
    assert abs(result.best.transform.plate_scale_arcsec_per_pixel - 8.5) < 0.02
    assert result.best.rms_residual_px < 0.5
    assert result.best.leave_one_out_rms_residual_px is not None
    assert result.best.leave_one_out_rms_residual_px < 1.0
    assert result.best.coverage_area > 0.01


def test_solve_plate_rejects_insufficient_or_unmatched_geometry() -> None:
    reference_wcs = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=8.5,
        crpix_x=1000.0,
        crpix_y=1000.0,
    )
    detections = tuple(_detection(index, 100.0 + index * 80.0, 200.0 + index * 35.0) for index in range(6))
    tangent_points = np.asarray(
        ((-8000.0, -6000.0), (-2000.0, 7000.0), (4000.0, -5000.0), (8500.0, 5000.0), (10500.0, 1000.0), (12000.0, -8000.0)),
        dtype=np.float64,
    )
    catalog = _catalog_for_tangent_points(reference_wcs, tangent_points)
    result = solve_plate(
        detections,
        catalog,
        reference_wcs,
        image_shape=(2000, 2000),
        scale_tolerance=0.02,
        min_pair_distance_px=20.0,
        match_radius_px=0.5,
        min_matches=6,
        max_rms_residual_px=0.5,
        max_image_points=12,
        max_catalog_points=12,
    )

    assert not result.valid
    assert result.status in {"NO_SOLUTION", "REJECTED"}
    assert result.best is None or result.best.matched_count < 6 or result.best.rms_residual_px > 0.5


def test_solve_plate_excludes_quality_rejected_and_structural_points() -> None:
    reference_wcs = TangentPlaneWCS(
        center_ra_deg=80.0,
        center_dec_deg=-10.0,
        pixel_scale_arcsec=8.5,
        crpix_x=100.0,
        crpix_y=100.0,
    )
    points = np.asarray(((-500.0, -400.0), (-100.0, 300.0), (400.0, -250.0)), dtype=np.float64)
    catalog = _catalog_for_tangent_points(reference_wcs, points)
    detections = tuple(
        _detection(index, 100.0 + east / 8.5, 100.0 + north / 8.5)
        for index, (east, north) in enumerate(points, start=1)
    )
    detections += (_detection(99, 50.0, 50.0, snr=1000.0, flags=("LINE_ARTIFACT",)),)
    result = solve_plate(
        detections,
        catalog,
        reference_wcs,
        min_matches=3,
        min_pair_distance_px=10.0,
        match_radius_px=1.0,
        max_image_points=4,
        max_catalog_points=4,
        image_shape=(300, 300),
        min_coverage_area=0.01,
        max_leave_one_out_rms_px=2.0,
    )
    assert result.image_points_considered == 3
    assert not result.valid
    assert result.status == "REJECTED"
    assert result.best is not None
    assert "留一验证无有效样本" in result.reason


def test_solve_plate_prefers_bright_sources_inside_wide_field_catalog() -> None:
    reference_wcs = TangentPlaneWCS(
        center_ra_deg=129.5,
        center_dec_deg=-1.8,
        pixel_scale_arcsec=8.5,
        crpix_x=1000.0,
        crpix_y=1000.0,
    )
    in_field_tangent = np.asarray(
        (
            (-6500.0, -5200.0),
            (-5200.0, 6100.0),
            (-3000.0, 2400.0),
            (-800.0, -6800.0),
            (1500.0, 7000.0),
            (3400.0, -2600.0),
            (5600.0, 4300.0),
            (7000.0, -1200.0),
        ),
        dtype=np.float64,
    )
    in_field = _catalog_for_tangent_points(reference_wcs, in_field_tangent)
    out_of_field_tangent = np.asarray(
        (
            (-18000.0, -14000.0),
            (-12000.0, 18000.0),
            (14000.0, -16000.0),
            (19000.0, 9000.0),
            (22000.0, -3000.0),
            (-21000.0, 7000.0),
            (11000.0, 21000.0),
            (-16000.0, -19000.0),
        ),
        dtype=np.float64,
    )
    out_of_field_sources: list[CatalogSource] = []
    for index, (east, north) in enumerate(out_of_field_tangent):
        ra, dec = reference_wcs.pixel_to_world(
            reference_wcs.crpix_x + east / reference_wcs.pixel_scale_arcsec,
            reference_wcs.crpix_y + north / reference_wcs.pixel_scale_arcsec,
        )
        out_of_field_sources.append(
            CatalogSource(
                source_id=f"outside-{index}",
                ra_deg=float(ra),
                dec_deg=float(dec),
                magnitude=5.0 + index * 0.1,
                color=0.8,
                color_name="BP-RP",
                catalog_name="Gaia DR3",
                photometric_system="Gaia Vega",
                photometric_band="G",
            )
        )
    out_of_field = tuple(out_of_field_sources)
    catalog = out_of_field + in_field
    matrix = np.asarray(((1.0 / 8.5, 0.0), (0.0, -1.0 / 8.5)), dtype=np.float64)
    transform = PlateTransform(
        matrix_px_per_arcsec=(tuple(matrix[0]), tuple(matrix[1])),
        offset_px=(1000.0, 1000.0),
        plate_scale_arcsec_per_pixel=8.5,
        rotation_deg=0.0,
        parity=-1,
        anisotropy_ratio=1.0,
    )
    detections = tuple(
        _detection(index, *transform.project_tangent(point), snr=100.0)
        for index, point in enumerate(in_field_tangent, start=1)
    )

    result = solve_plate(
        detections,
        catalog,
        reference_wcs,
        image_shape=(2000, 2000),
        scale_tolerance=0.04,
        min_pair_distance_px=20.0,
        match_radius_px=1.0,
        min_matches=6,
        min_coverage_area=0.01,
        max_rms_residual_px=0.5,
        max_leave_one_out_rms_px=1.0,
        max_image_points=16,
        max_catalog_points=8,
    )

    assert result.valid, result.as_dict()
    assert result.catalog_points_considered == 8
    assert result.best is not None
    assert all(str(match.source_id).startswith("gaia-") for match in result.best.matches)
