from __future__ import annotations

import json

import numpy as np

from rst19.detection import Detection
from rst19.fits import FitsFrame
from rst19.pair_response_attribution import (
    run_pair_response_attribution,
    write_pair_response_attribution_artifacts,
)


def _source(
    detection_id: int,
    x: float,
    y: float,
    *,
    peak_x: int,
    peak_y: int,
    codes: tuple[int, ...] = (),
) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=100.0,
        flux=500.0,
        background=10.0,
        noise=2.0,
        snr=30.0,
        fwhm=2.0,
        flags=(),
        flux_snr=20.0,
        peak_x=peak_x,
        peak_y=peak_y,
        repeated_code_values=codes,
    )


def test_pair_response_attribution_separates_mask_modes_and_writes_artifacts(tmp_path) -> None:
    image = np.full((60, 60), 10.0, dtype=np.float64)
    image[30, 30] = 130.0
    image[30, 31] = 99.0
    image[30, 29] = -80.0
    image[31, 30] = 70.0
    frame = FitsFrame(path=tmp_path / "synthetic.fits", header={}, data=image, auxiliary=None, data_offset=0)
    primary = _source(1, 31.0, 30.0, peak_x=31, peak_y=30, codes=(99,))
    secondary = _source(2, 28.0, 30.0, peak_x=28, peak_y=30, codes=(99,))

    result = run_pair_response_attribution(
        frame,
        primary,
        secondary,
        patch_padding_px=8,
        psf_fwhm=2.0,
        negative_anomaly_threshold_adu=-50.0,
    )

    target_rows = {(row.mask_mode, row.target_label): row for row in result.target_rows}
    assert target_rows[("raw", "primary")].valid_at_target is True
    assert target_rows[("repeated_code_masked", "primary")].valid_at_target is False
    assert target_rows[("raw", "secondary")].valid_at_target is True
    assert target_rows[("negative_anomaly_masked", "secondary")].valid_at_target is True
    assert len(result.contribution_rows) == 6
    assert any(row.pixel_class == "repeated_code" and row.pixel_count > 0 for row in result.contribution_rows)
    assert any(row.pixel_class == "negative_anomaly" and row.pixel_count > 0 for row in result.contribution_rows)

    output = write_pair_response_attribution_artifacts(result, tmp_path / "out")
    assert (output / "pair_response_targets.csv").is_file()
    assert (output / "pair_response_maxima.csv").is_file()
    assert (output / "pair_response_contributions.csv").is_file()
    payload = json.loads((output / "pair_response_attribution.json").read_text(encoding="utf-8"))
    assert payload["repeated_code_values"] == [99]
    assert "Gaussian" in payload["conclusion"]


def test_pair_response_attribution_rejects_same_peak() -> None:
    image = np.ones((12, 12), dtype=np.float64)
    frame = FitsFrame(path="same.fits", header={}, data=image, auxiliary=None, data_offset=0)
    primary = _source(1, 4.0, 4.0, peak_x=4, peak_y=4)
    secondary = _source(2, 4.0, 4.0, peak_x=4, peak_y=4)

    try:
        run_pair_response_attribution(frame, primary, secondary)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("expected identical peak coordinates to be rejected")
