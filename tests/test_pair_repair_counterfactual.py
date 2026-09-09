from __future__ import annotations

import json

import numpy as np

from rst19.detection import Detection, DetectionResult
from rst19.fits import FitsFrame
from rst19.pair_repair_counterfactual import (
    _repair_pixels,
    run_pair_repair_counterfactual,
    write_pair_repair_counterfactual_artifacts,
)


def _source(detection_id: int, x: float, y: float, *, peak_x: int, peak_y: int) -> Detection:
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
    )


def test_repair_pixels_uses_non_special_local_median() -> None:
    patch = np.full((7, 7), 10.0, dtype=np.float64)
    patch[3, 3] = -100.0
    patch[3, 4] = 99.0
    selector = patch == -100.0
    exclusion = selector | (patch == 99.0)

    repaired, rows = _repair_pixels(
        patch,
        selector,
        exclusion=exclusion,
        radius=1,
        patch_x0=20,
        patch_y0=30,
        pixel_class="negative_anomaly",
    )

    assert repaired[3, 3] == 10.0
    assert len(rows) == 1
    assert rows[0].x == 23 and rows[0].y == 33
    assert rows[0].replacement_value_adu == 10.0


def test_pair_repair_counterfactual_runs_variants_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    image = np.full((80, 80), 10.0, dtype=np.float64)
    image[40, 40] = 130.0
    image[40, 41] = 99.0
    image[40, 39] = -100.0
    frame = FitsFrame(path=tmp_path / "synthetic.fits", header={}, data=image, auxiliary=None, data_offset=0)
    primary = _source(1, 41.0, 40.0, peak_x=41, peak_y=40)
    secondary = _source(2, 38.0, 40.0, peak_x=38, peak_y=40)

    calls: list[bool] = []

    def fake_detect(crop, **_kwargs):
        has_code = bool(np.any(crop == 99.0))
        has_negative = bool(np.any(crop <= -100.0))
        calls.append(has_code or has_negative)
        if has_code or has_negative:
            if has_code and not has_negative:
                sources = (
                    _source(0, 41.0 - 30.0, 40.0 - 32.0, peak_x=41 - 30, peak_y=40 - 32),
                )
            else:
                sources = (
                    _source(0, 41.0 - 30.0, 40.0 - 32.0, peak_x=41 - 30, peak_y=40 - 32),
                    _source(1, 38.0 - 30.0, 40.0 - 32.0, peak_x=38 - 30, peak_y=40 - 32),
                )
        else:
            sources = (
                _source(0, 41.0 - 30.0, 40.0 - 32.0, peak_x=41 - 30, peak_y=40 - 32),
            )
        return DetectionResult(
            image_shape=tuple(crop.shape),
            background=10.0,
            noise=2.0,
            threshold=18.0,
            candidate_count=len(sources),
            sources=sources,
            parameters={},
            quality_count=sum(source.quality_passed for source in sources),
        )

    monkeypatch.setattr("rst19.pair_repair_counterfactual.detect_sources", fake_detect)
    result = run_pair_repair_counterfactual(
        frame,
        primary,
        secondary,
        patch_padding_px=8,
        repeated_code_values=(99,),
        negative_anomaly_threshold_adu=-50.0,
    )

    assert calls == [True, True, True, False]
    summary = {row.variant: row for row in result.summaries}
    assert summary["raw"].target_window_candidate_count == 2
    assert summary["repair_negative_anomaly"].repaired_pixel_count == 1
    assert summary["repair_both"].repaired_pixel_count == 2
    output = write_pair_repair_counterfactual_artifacts(result, tmp_path / "out")
    assert (output / "pair_repair_summaries.csv").is_file()
    assert (output / "pair_repair_candidates.csv").is_file()
    assert (output / "pair_repair_replacements.csv").is_file()
    payload = json.loads((output / "pair_repair_counterfactual.json").read_text(encoding="utf-8"))
    assert payload["repeated_code_values"] == [99]
