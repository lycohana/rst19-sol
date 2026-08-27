from __future__ import annotations

import numpy as np

from rst19.catalog import CatalogSource, load_catalog_csv
from rst19.detection import Detection
from rst19.matching import match_detections
from rst19.wcs import TangentPlaneWCS


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
