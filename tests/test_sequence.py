from __future__ import annotations

import pytest

from rst19.detection import Detection
from rst19.sequence import track_detections


def _source(detection_id: int, x: float, y: float) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=100.0,
        flux=500.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=3.0,
        flags=(),
        flux_error=20.0,
        flux_snr=25.0,
        quality_passed=True,
    )


def test_track_detections_separates_static_and_moving_sources() -> None:
    static_positions = ((10.0, 10.0), (50.0, 10.0), (10.0, 50.0), (50.0, 50.0), (70.0, 70.0))
    frames = [
        tuple([_source(index, x, y) for index, (x, y) in enumerate(static_positions)] + [_source(5, 30.0, 20.0)]),
        tuple([_source(index, x + 0.2, y + 0.1) for index, (x, y) in enumerate(static_positions)] + [_source(5, 32.0, 20.0)]),
        tuple([_source(index, x - 0.1, y + 0.1) for index, (x, y) in enumerate(static_positions)] + [_source(5, 34.0, 20.0)]),
    ]

    result = track_detections(frames, min_presence=3, link_radius_px=3.0, motion_min_displacement_px=2.0)

    assert result.stable_source_count == len(static_positions)
    assert result.moving_track_count == 1
    assert result.transient_track_count == 0
    moving = next(track for track in result.tracks if track.classification == "moving")
    assert moving.presence == 3
    assert moving.displacement_px == pytest.approx(4.0, abs=0.2)


def test_track_detections_removes_global_translation_before_classifying() -> None:
    frames = [
        (_source(0, 10.0, 10.0), _source(1, 40.0, 20.0)),
        (_source(0, 11.0, 10.0), _source(1, 41.0, 20.0)),
        (_source(0, 12.0, 10.0), _source(1, 42.0, 20.0)),
    ]

    result = track_detections(frames, min_presence=3, link_radius_px=2.0, registration_radius_px=2.0)

    assert result.stable_source_count == 2
    assert result.moving_track_count == 0
    assert result.cumulative_shifts[-1] == (2.0, 0.0)
