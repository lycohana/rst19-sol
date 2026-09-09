from __future__ import annotations

import json

import numpy as np

from rst19.detection import Detection
from rst19.models import FitsFrame
from rst19.pair_pixel_topology import (
    run_pair_pixel_topology_audit,
    write_pair_pixel_topology_artifacts,
)


def _source(
    detection_id: int,
    x: float,
    y: float,
    *,
    peak_x: int,
    peak_y: int,
    peak: float,
    codes: tuple[int, ...] = (),
) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=peak,
        flux=100.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=2.0,
        quality_passed=False,
        flags=(),
        flux_snr=10.0,
        peak_x=peak_x,
        peak_y=peak_y,
        repeated_code_values=codes,
    )


def test_pair_pixel_topology_separates_raw_and_special_value_connectivity(tmp_path, monkeypatch) -> None:
    image = np.full((40, 40), 10.0, dtype=np.float32)
    image[20, 20] = 130.0
    image[20, 21] = 99.0
    image[21, 19] = 80.0
    image[21, 18] = 70.0
    image[21, 17] = 50.0
    image[21, 16] = 40.0
    frame = FitsFrame(
        path=tmp_path / "synthetic.fits",
        header={},
        data=image,
        auxiliary=None,
        data_offset=0,
    )
    primary = _source(1, 21.0, 20.0, peak_x=21, peak_y=20, peak=99.0, codes=(99,))
    secondary = _source(2, 16.0, 21.0, peak_x=16, peak_y=21, peak=40.0, codes=(99,))

    result = run_pair_pixel_topology_audit(
        frame,
        primary,
        secondary,
        patch_padding_px=4,
        sigma_levels=(3.0, 5.0),
        raw_maximum_neighborhoods_px=(3,),
    )

    raw_rows = [row for row in result.component_rows if row.mode == "raw"]
    excluded_rows = [row for row in result.component_rows if row.mode == "special_excluded"]
    assert all(row.same_nonzero_component for row in raw_rows)
    assert all(not row.primary_peak_is_positive for row in excluded_rows)
    assert all(row.secondary_peak_is_positive for row in excluded_rows)
    assert result.maximum_rows[0].primary_peak_is_raw_local_maximum is False
    assert result.maximum_rows[0].secondary_peak_is_raw_local_maximum is False

    monkeypatch.setattr("rst19.pair_pixel_topology.read_fits", lambda _path: frame)
    output = write_pair_pixel_topology_artifacts(result, tmp_path / "out")
    assert (output / "pair_pixel_topology_components.csv").is_file()
    assert (output / "pair_pixel_topology_maxima.csv").is_file()
    assert (output / "pair_pixel_topology.json").is_file()
    assert (output / "pair_pixel_topology.svg").is_file()
    payload = json.loads((output / "pair_pixel_topology.json").read_text(encoding="utf-8"))
    assert payload["primary_peak"] == [21, 20]
    assert "连通块" in payload["conclusion"]


def test_pair_pixel_topology_rejects_same_peak() -> None:
    image = np.ones((10, 10), dtype=np.float32)
    frame = FitsFrame(path="same.fits", header={}, data=image, auxiliary=None, data_offset=0)
    primary = _source(1, 4.0, 4.0, peak_x=4, peak_y=4, peak=20.0)
    secondary = _source(2, 4.0, 4.0, peak_x=4, peak_y=4, peak=20.0)

    try:
        run_pair_pixel_topology_audit(frame, primary, secondary)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("expected identical peak coordinates to be rejected")
