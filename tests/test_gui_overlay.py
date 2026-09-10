from __future__ import annotations

from pathlib import Path

from rst19.gui import moving_points_for_frame, sequence_frame_index_for_frame, stable_points_for_frame, trusted_points_for_frame
from rst19.sequence import FrameSequenceSummary, SequenceResult, SourceTrack, TrackPoint


def _sequence_result() -> SequenceResult:
    moving = SourceTrack(
        track_id=7,
        classification="moving",
        points=(
            TrackPoint(0, 10, 20.0, 30.0, 20.0, 30.0, 12.0),
            TrackPoint(1, 11, 21.0, 30.0, 21.0, 30.0, 11.0),
        ),
        displacement_px=1.0,
        speed_px_per_frame=1.0,
        fit_rms_px=0.1,
    )
    static = SourceTrack(
        track_id=8,
        classification="static",
        points=(TrackPoint(0, 20, 50.0, 60.0, 50.0, 60.0, 20.0),),
        displacement_px=0.0,
        speed_px_per_frame=0.0,
        fit_rms_px=None,
    )
    frames = tuple(FrameSequenceSummary(index, str(Path(f"frame-{index}.fits")), 1, 1, 1) for index in range(2))
    return SequenceResult(frames, ((0.0, 0.0), (0.0, 0.0)), (moving, static), 4.0, 2, 2.0, 0.75)


def test_motion_overlay_only_returns_current_points_from_moving_tracks() -> None:
    result = _sequence_result()

    frame_zero = moving_points_for_frame(result, 0)
    frame_one = moving_points_for_frame(result, 1)

    assert [(track.track_id, point.detection_id) for track, point in frame_zero] == [(7, 10)]
    assert [(track.track_id, point.detection_id) for track, point in frame_one] == [(7, 11)]
    assert moving_points_for_frame(None, 0) == ()


def test_stable_overlay_returns_static_and_persistent_points_only() -> None:
    result = _sequence_result()
    persistent = SourceTrack(
        track_id=9,
        classification="persistent",
        points=(TrackPoint(0, 30, 70.0, 80.0, 70.0, 80.0, 8.0),),
        displacement_px=0.1,
        speed_px_per_frame=0.1,
        fit_rms_px=0.1,
    )
    result = SequenceResult(
        result.frames,
        result.cumulative_shifts,
        result.tracks + (persistent,),
        result.link_radius_px,
        result.min_presence,
        result.motion_min_displacement_px,
        result.max_motion_fit_rms_px,
        result.motion_features,
        persistent_min_presence=1,
    )

    points = stable_points_for_frame(result, 0)

    assert [(track.track_id, point.detection_id) for track, point in points] == [(8, 20), (9, 30)]


def test_trusted_overlay_returns_quality_passed_points_including_moving_sources() -> None:
    result = _sequence_result()

    points = trusted_points_for_frame(result, 0)

    assert [(track.track_id, point.detection_id) for track, point in points] == [(7, 10), (8, 20)]
    assert trusted_points_for_frame(None, 0) == ()


def test_sequence_frame_index_matches_relative_cached_path() -> None:
    result = _sequence_result()
    selected = Path("C:/data/frame-1.fits")
    stored_frames = (
        FrameSequenceSummary(0, "frame-0.fits", 1, 1, 1),
        FrameSequenceSummary(1, "frame-1.fits", 1, 1, 1),
    )
    result = SequenceResult(
        stored_frames,
        result.cumulative_shifts,
        result.tracks,
        result.link_radius_px,
        result.min_presence,
        result.motion_min_displacement_px,
        result.max_motion_fit_rms_px,
        result.motion_features,
    )

    assert sequence_frame_index_for_frame(result, selected, (Path("C:/data/frame-0.fits"), selected)) == 1
