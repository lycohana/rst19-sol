from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rst19.detection import Detection
from rst19.models import FitsFrame
from rst19.pipeline import FrameAnalysis
from rst19.sequence import detect_motion_features, track_detections
from rst19.detection import DetectionResult


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


def test_track_detections_does_not_call_a_last_point_outlier_motion() -> None:
    reference_positions = ((10.0, 10.0), (50.0, 10.0), (10.0, 50.0), (50.0, 50.0))
    frames = []
    for frame_index in range(5):
        target = (30.0, 20.0) if frame_index < 4 else (32.0, 20.0)
        positions = reference_positions + (target,)
        frames.append(tuple(_source(index, x, y) for index, (x, y) in enumerate(positions)))

    result = track_detections(frames, min_presence=5, link_radius_px=3.0, motion_min_displacement_px=2.0)

    assert result.moving_track_count == 0


def test_track_detections_does_not_reconnect_after_a_missed_frame() -> None:
    result = track_detections(
        [(_source(0, 10.0, 10.0),), (), (_source(1, 11.0, 10.0),)],
        min_presence=1,
        link_radius_px=3.0,
    )

    assert len(result.tracks) == 2
    assert [track.points[0].frame_index for track in result.tracks] == [0, 2]


def _feature_analysis(frame_index: int) -> FrameAnalysis:
    image = np.full((96, 96), 10.0, dtype=np.float32)
    for offset in range(42):
        x = 6 + 8 * frame_index + offset
        y = 15 + offset
        if x < image.shape[1] and y < image.shape[0]:
            image[y, x] = 300.0
    detection = DetectionResult(
        image_shape=image.shape,
        background=10.0,
        noise=2.0,
        threshold=40.0,
        candidate_count=0,
        sources=(),
        parameters={},
        quality_count=0,
    )
    frame = FitsFrame(Path(f"synthetic-{frame_index}.fits"), {}, image, None, 0)
    return FrameAnalysis(frame, detection, None, None)


def test_detect_motion_features_tracks_a_moving_streak_separately() -> None:
    tracks, threshold = detect_motion_features(
        tuple(_feature_analysis(index) for index in range(4)),
        ((0.0, 0.0),) * 4,
        psf_fwhm=3.0,
    )

    assert threshold == pytest.approx(100.0)
    assert len(tracks) == 1
    assert tracks[0].classification == "moving"
    assert tracks[0].presence == 4
    assert tracks[0].displacement_px == pytest.approx(24.0, abs=0.5)
