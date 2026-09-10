from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rst19.detection import Detection
from rst19.models import FitsFrame
from rst19.pipeline import FrameAnalysis
from rst19.sequence import (
    DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT,
    SourceTrack,
    TrackPoint,
    _candidate_consensus_tracks,
    _registered_coadd_reference,
    _registered_median_reference,
    _stack_faint_tracks,
    _stack_forced_frame_measure,
    _stack_forced_frame_measure_batch,
    _track_fast_candidate_movers,
    _temporal_psf_fwhm_bank,
    _temporal_reference_candidate_frames,
    audit_fixed_sentinel,
    audit_fixed_sentinel_impacts,
    detect_long_trails,
    detect_motion_features,
    detect_single_frame_long_trails,
    track_fast_point_movers,
    track_detections,
)
from rst19.detection import DetectionResult
import rst19.sequence as sequence_module


def test_audit_fixed_sentinel_counts_fixed_coordinates_and_excludes_auxiliary() -> None:
    images = []
    for frame_index in range(3):
        image = np.zeros((5, 6), dtype=np.int16)
        image[0, 0] = -1  # 比赛格式辅助字段，不应进入固定像素统计。
        image[1, 2] = -1  # 三帧均出现。
        if frame_index < 2:
            image[3, 4] = -1  # 只在前两帧出现，不是固定坐标。
        images.append(image)

    audit = audit_fixed_sentinel(images)

    assert audit.frame_count == 3
    assert audit.image_shape == (5, 6)
    assert audit.same_shape is True
    assert audit.per_frame_occurrences == (2, 2, 1)
    assert audit.total_occurrences == 5
    assert audit.fixed_coordinate_count == 1
    assert audit.fixed_coordinates == ((2, 1),)
    assert audit.as_dict()["fixed_coordinates"] == [[2, 1]]


def test_audit_fixed_sentinel_impacts_separates_full_candidates_from_returned_sources() -> None:
    image = np.zeros((24, 24), dtype=np.int16)
    image[5, 5] = -1
    image[9, 9] = -1
    frame = FitsFrame(Path("impact.fits"), {}, image, None, 0)
    quality_source = _source(7, 5.0, 5.0)
    rejected_source = Detection(
        detection_id=8,
        x=9.0,
        y=9.0,
        peak=80.0,
        flux=300.0,
        background=10.0,
        noise=2.0,
        snr=12.0,
        fwhm=3.0,
        flags=("MASKED",),
        flux_error=20.0,
        flux_snr=12.0,
        quality_passed=False,
    )
    detection = DetectionResult(
        image_shape=image.shape,
        background=10.0,
        noise=2.0,
        threshold=18.0,
        candidate_count=3,
        sources=(quality_source, rejected_source),
        parameters={"aperture_radius": 2},
        quality_count=1,
        candidate_peaks=np.asarray(((5, 5, 20.0), (9, 9, 15.0), (20, 20, 9.0)), dtype=np.float32),
    )
    analysis = FrameAnalysis(frame, detection, None, None)
    sentinel_audit = audit_fixed_sentinel((image,))

    impact = audit_fixed_sentinel_impacts((analysis,), sentinel_audit)

    assert impact.candidate_peak_data_available is True
    assert impact.candidate_peak_frame_count == 1
    assert impact.candidate_peak_count == 3
    assert impact.candidate_peak_affected_count == 2
    assert impact.affected_frame_count == 1
    assert impact.affected_returned_source_count == 2
    assert impact.affected_quality_source_count == 1
    assert impact.record_count == 2
    assert [record.detection_id for record in impact.records] == [7, 8]
    assert impact.records[1].flags == ("MASKED",)
    assert impact.as_dict()["record_count"] == 2


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


def test_track_detections_marks_stationary_half_sequence_as_persistent() -> None:
    source = _source(0, 10.0, 10.0)
    result = track_detections(
        [(source,), (source,), (source,), ()],
        min_presence=4,
        persistent_min_presence=3,
        link_radius_px=3.0,
    )

    assert result.stable_source_count == 0
    assert result.persistent_source_count == 1
    assert result.stable_field_candidate_count == 1
    assert result.tracks[0].classification == "persistent"
    assert result.persistent_min_presence == 3


def test_track_fast_point_movers_recovers_a_point_source_beyond_static_link_radius() -> None:
    frames = []
    for frame_index in range(15):
        frames.append(
            (
                _source(1000 + frame_index, 100.0, 100.0),
                _source(2000 + frame_index, 300.0, 300.0),
                _source(3000 + frame_index, 1795.0 + 8.0 * frame_index, 919.0 - 6.0 * frame_index, ),
            )
        )
    existing = []
    for detection_id, x, y in ((1000, 100.0, 100.0), (2000, 300.0, 300.0)):
        points = tuple(
            TrackPoint(frame_index, detection_id + frame_index, x, y, x, y, 25.0)
            for frame_index in range(15)
        )
        existing.append(SourceTrack(detection_id, "static", points, 0.0, 0.0, 0.0))

    tracks = track_fast_point_movers(
        frames,
        ((0.0, 0.0),) * 15,
        link_radius_px=4.0,
        min_presence=12,
        min_flux_snr=7.0,
        max_step_px=20.0,
        gate_px=4.5,
        max_fit_rms_px=1.5,
        existing_tracks=existing,
    )

    assert len(tracks) == 1
    moving = tracks[0]
    assert moving.classification == "moving"
    assert moving.evidence_level == "fast_point_motion"
    assert moving.presence == 15
    assert moving.speed_px_per_frame == pytest.approx(10.0, abs=0.01)
    assert moving.displacement_px == pytest.approx(140.0, abs=0.1)


def test_track_fast_point_movers_does_not_relabel_stationary_sources() -> None:
    frames = [tuple(_source(100 + frame_index, 50.0, 50.0) for frame_index in range(5)) for _ in range(5)]

    tracks = track_fast_point_movers(
        frames,
        ((0.0, 0.0),) * 5,
        min_presence=5,
        min_flux_snr=7.0,
        max_step_px=20.0,
        gate_px=4.5,
        max_fit_rms_px=1.5,
    )

    assert tracks == ()


def test_fast_candidate_movers_recovers_a_peak_omitted_from_source_working_set() -> None:
    analyses = []
    candidate_frames = []
    yy, xx = np.mgrid[:256, :256]
    for frame_index in range(15):
        x = 22.0 + 8.0 * frame_index
        y = 104.0 - 6.0 * frame_index
        image = 10.0 + 90.0 * np.exp(-0.5 * (((xx - x) / 1.6) ** 2 + ((yy - y) / 1.6) ** 2))
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=14.0,
            candidate_count=2,
            # 模拟 max_sources=1：真实移动点只在 candidate_peaks 中，
            # 不出现在 sources/quality_sources 工作集。
            sources=(),
            parameters={
                "mask_zero_pixels": 0,
                "saturation_level": -1.0,
                "min_psf_support_pixels": 3,
                "aperture_radius": 4,
                "min_fwhm": 0.8,
                "max_fwhm": 12.0,
                "max_ellipticity": 0.65,
            },
            quality_count=0,
            candidate_peaks=np.asarray(((round(x), round(y), 40.0),), dtype=np.float32),
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"fast-candidate-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )
        candidate_frames.append(detection.candidate_peaks)

    tracks = _track_fast_candidate_movers(
        candidate_frames,
        analyses,
        ((0.0, 0.0),) * 15,
        link_radius_px=4.0,
        min_presence=12,
        motion_min_displacement_px=2.0,
        min_filter_snr=7.0,
        max_step_px=20.0,
        gate_px=4.5,
        max_fit_rms_px=1.5,
    )

    assert len(tracks) == 1
    moving = tracks[0]
    assert moving.evidence_level == "fast_point_motion"
    assert moving.presence == 15
    assert moving.quality_presence == 15
    assert moving.points[0].candidate_snr == pytest.approx(40.0)
    assert moving.points[0].flux_snr is not None and moving.points[0].flux_snr >= 7.0
    assert moving.speed_px_per_frame == pytest.approx(10.0, abs=0.2)


def test_candidate_consensus_keeps_repeated_peak_separate_from_quality_track() -> None:
    candidates = [
        ((20, 20, 18.0), (60, 60, 22.0)),
        ((20, 20, 17.0), (60, 60, 21.0)),
        ((20, 20, 16.0), (60, 60, 20.0)),
        ((20, 20, 18.0), (60, 60, 23.0)),
    ]
    quality = _source(0, 20.0, 20.0)
    quality_result = track_detections(
        [(quality,), (quality,), (quality,), (quality,)],
        min_presence=4,
        persistent_min_presence=2,
        link_radius_px=2.0,
    )

    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        quality_result.tracks,
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
    )

    assert len(tracks) == 1
    assert tracks[0].classification == "persistent"
    assert tracks[0].evidence_level == "candidate_consensus"
    assert tracks[0].presence == 4
    assert tracks[0].quality_presence == 0
    assert tracks[0].points[0].candidate_snr == pytest.approx(22.0)


def test_candidate_consensus_rejects_a_repeated_single_pixel_spike() -> None:
    candidates = [
        ((20, 20, 18.0), (28, 20, 18.0)),
        ((20, 20, 18.0), (28, 20, 18.0)),
        ((20, 20, 18.0), (28, 20, 18.0)),
        ((20, 20, 18.0), (28, 20, 18.0)),
    ]
    analyses = []
    for frame_index in range(4):
        image = np.full((48, 48), 10.0, dtype=np.float32)
        image[20, 20] = 38.0
        image[19:22, 19:22] += 12.0
        image[20, 20] += 12.0
        image[20, 28] = 38.0
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=14.0,
            candidate_count=2,
            sources=(),
            parameters={
                "mask_zero_pixels": 0,
                "saturation_level": -1.0,
                "min_psf_support_pixels": 3,
                "aperture_radius": 4,
            },
            quality_count=0,
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"candidate-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )

    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        frame_analyses=analyses,
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
    )

    assert len(tracks) == 1
    assert tracks[0].points[0].x == pytest.approx(20.0)


def test_temporal_reference_channel_can_recover_a_lower_per_frame_response() -> None:
    candidates = [
        np.asarray([[30.0, 30.0, 8.0, 1.0, 20.0]], dtype=np.float32)
        for _ in range(4)
    ]
    analyses = []
    yy, xx = np.mgrid[:64, :64]
    source = 80.0 * np.exp(-0.5 * (((xx - 30.0) / 1.25) ** 2 + ((yy - 30.0) / 1.25) ** 2))
    for frame_index in range(4):
        image = (10.0 + source).astype(np.float32)
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=14.0,
            candidate_count=1,
            sources=(),
            parameters={
                "mask_zero_pixels": 0,
                "saturation_level": -1.0,
                "min_psf_support_pixels": 3,
                "aperture_radius": 4,
            },
            quality_count=0,
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"temporal-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )

    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        frame_analyses=analyses,
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
    )

    assert len(tracks) == 1
    assert tracks[0].evidence_level == "temporal_reference"
    assert tracks[0].points[0].candidate_snr == pytest.approx(8.0)


def test_temporal_reference_uses_a_small_psf_scale_bank() -> None:
    assert _temporal_psf_fwhm_bank(3.0) == pytest.approx((1.95, 3.0, 4.05))

    yy, xx = np.mgrid[:96, :96]
    source = 22.0 * np.exp(-0.5 * (((xx - 45.0) / 0.8) ** 2 + ((yy - 45.0) / 0.8) ** 2))
    image = (10.0 + source).astype(np.float32)
    analyses = []
    for frame_index in range(4):
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=15.0,
            candidate_count=0,
            sources=(),
            parameters={
                "mask_zero_pixels": 0,
                "saturation_level": -1.0,
                "min_psf_support_pixels": 3,
                "aperture_radius": 4,
            },
            quality_count=0,
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"temporal-multiscale-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )

    rows, proposal_count = _temporal_reference_candidate_frames(
        analyses,
        ((0.0, 0.0),) * 4,
        image,
        psf_fwhm=3.0,
        min_candidate_snr=15.0,
        temporal_multiscale=True,
    )

    assert proposal_count >= 1
    assert all(len(frame_rows) >= 1 for frame_rows in rows)


def test_temporal_reference_relocates_an_integer_prediction_to_the_local_psf_peak() -> None:
    yy, xx = np.mgrid[:96, :96]
    reference_source = 22.0 * np.exp(
        -0.5 * (((xx - 45.0) / 1.25) ** 2 + ((yy - 45.0) / 1.25) ** 2)
    )
    frame_source = 22.0 * np.exp(
        -0.5 * (((xx - 46.0) / 1.25) ** 2 + ((yy - 45.0) / 1.25) ** 2)
    )
    reference = (10.0 + reference_source).astype(np.float32)
    image = (10.0 + frame_source).astype(np.float32)
    analyses = []
    for frame_index in range(4):
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=15.0,
            candidate_count=0,
            sources=(),
            parameters={
                "mask_zero_pixels": 0,
                "saturation_level": -1.0,
                "min_psf_support_pixels": 3,
                "aperture_radius": 4,
            },
            quality_count=0,
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"temporal-local-search-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )

    rows, proposal_count = _temporal_reference_candidate_frames(
        analyses,
        ((0.0, 0.0),) * 4,
        reference,
        psf_fwhm=3.0,
        min_candidate_snr=15.0,
    )

    assert proposal_count >= 1
    assert all(len(frame_rows) >= 1 for frame_rows in rows)
    assert all(float(frame_rows[0, 0]) == pytest.approx(46.0) for frame_rows in rows)


def test_temporal_audit_tolerates_an_isolated_zero_pixel_in_the_aperture() -> None:
    candidates = [
        np.asarray([[30.0, 30.0, 8.0, 1.0, 20.0]], dtype=np.float32)
        for _ in range(4)
    ]
    yy, xx = np.mgrid[:64, :64]
    source = 80.0 * np.exp(-0.5 * (((xx - 30.0) / 1.25) ** 2 + ((yy - 30.0) / 1.25) ** 2))
    analyses = []
    for frame_index in range(4):
        image = (10.0 + source).astype(np.float32)
        if frame_index == 1:
            # 孔径内、但不在中心 3×3 内的孤立坏像素；不应使整帧失效。
            image[26, 30] = 0.0
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=14.0,
            candidate_count=1,
            sources=(),
            parameters={
                "mask_zero_pixels": 1,
                "saturation_level": -1.0,
                "min_psf_support_pixels": 3,
                "aperture_radius": 4,
            },
            quality_count=0,
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"zero-pixel-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )

    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        frame_analyses=analyses,
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
    )

    assert len(tracks) == 1
    assert tracks[0].presence == 4


def test_temporal_reference_channel_rejects_a_line_mask_overlap() -> None:
    candidates = [
        np.asarray([[30.0, 30.0, 8.0, 1.0, 20.0]], dtype=np.float32)
        for _ in range(4)
    ]
    analyses = []
    image = np.full((64, 64), 10.0, dtype=np.float32)
    image[29:32, 29:32] += 40.0
    for frame_index in range(4):
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=14.0,
            candidate_count=1,
            sources=(),
            parameters={
                "mask_zero_pixels": 0,
                "saturation_level": -1.0,
                "min_psf_support_pixels": 3,
                "aperture_radius": 4,
            },
            quality_count=0,
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"line-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )

    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        frame_analyses=analyses,
        line_artifact_frames=[np.asarray([[30, 30]], dtype=np.int32)] * 4,
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
    )

    assert tracks == ()


def test_temporal_reference_channel_tolerates_a_line_touching_one_long_sequence_frame() -> None:
    candidates = [
        np.asarray([[30.0, 30.0, 8.0, 1.0, 20.0]], dtype=np.float32)
        for _ in range(5)
    ]
    analyses = []
    image = np.full((64, 64), 10.0, dtype=np.float32)
    image[29:32, 29:32] += 40.0
    for frame_index in range(5):
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=14.0,
            candidate_count=1,
            sources=(),
            parameters={
                "mask_zero_pixels": 0,
                "saturation_level": -1.0,
                "min_psf_support_pixels": 3,
                "aperture_radius": 4,
            },
            quality_count=0,
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"one-line-touch-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )

    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 5,
        (),
        frame_analyses=analyses,
        line_artifact_frames=[np.asarray([[30, 30]], dtype=np.int32), (), (), (), ()],
        link_radius_px=2.0,
        min_presence=5,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
    )

    assert len(tracks) == 1
    assert tracks[0].presence == 5


def test_candidate_consensus_can_seed_from_a_later_frame_when_first_frame_is_missing() -> None:
    candidates = [
        np.empty((0, 3), dtype=np.float32),
        np.asarray([[30.0, 30.0, 8.0, 1.0, 20.0]], dtype=np.float32),
        np.asarray([[30.0, 30.0, 8.0, 1.0, 20.0]], dtype=np.float32),
        np.asarray([[30.0, 30.0, 8.0, 1.0, 20.0]], dtype=np.float32),
    ]
    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
    )

    assert len(tracks) == 1
    assert tracks[0].evidence_level == "temporal_reference"
    assert tracks[0].presence == 3
    assert [point.frame_index for point in tracks[0].points] == [1, 2, 3]


def test_temporal_reference_snr_can_be_lowered_without_lowering_normal_candidate_gate() -> None:
    candidates = [
        np.asarray([[30.0, 30.0, 8.0, 1.0, 12.5]], dtype=np.float32)
        for _ in range(4)
    ]

    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
        temporal_reference_min_snr=12.0,
    )

    assert len(tracks) == 1
    assert tracks[0].evidence_level == "temporal_reference"

    rejected = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
        temporal_reference_min_snr=13.0,
    )

    assert rejected == ()


def test_temporal_candidate_response_gate_is_independent_from_reference_gate() -> None:
    candidates = [
        np.asarray([[30.0, 30.0, 6.0, 1.0, 12.5]], dtype=np.float32)
        for _ in range(4)
    ]

    accepted = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=6.0,
        temporal_reference_min_snr=12.0,
    )
    rejected_at_default = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (),
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
        temporal_reference_min_snr=12.0,
    )

    assert len(accepted) == 1
    assert rejected_at_default == ()


def test_candidate_consensus_is_not_suppressed_by_a_short_quality_track() -> None:
    candidates = [
        np.asarray([[30.0, 30.0, 8.0, 1.0, 20.0]], dtype=np.float32)
        for _ in range(4)
    ]
    short_quality = SourceTrack(
        track_id=7,
        classification="transient",
        points=(
            TrackPoint(
                frame_index=0,
                detection_id=7,
                x=30.0,
                y=30.0,
                aligned_x=30.0,
                aligned_y=30.0,
                flux_snr=25.0,
                quality_passed=True,
            ),
        ),
        displacement_px=0.0,
        speed_px_per_frame=0.0,
        fit_rms_px=0.0,
    )
    tracks = _candidate_consensus_tracks(
        candidates,
        ((0.0, 0.0),) * 4,
        (short_quality,),
        link_radius_px=2.0,
        min_presence=4,
        persistent_min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
    )

    assert len(tracks) == 1
    assert tracks[0].evidence_level == "temporal_reference"


def test_registered_coadd_rejects_a_single_frame_bright_transient() -> None:
    analyses = []
    yy, xx = np.mgrid[:48, :48]
    stable = 20.0 * np.exp(-0.5 * (((xx - 15.0) / 1.25) ** 2 + ((yy - 15.0) / 1.25) ** 2))
    for frame_index in range(5):
        image = np.full((48, 48), 10.0, dtype=np.float32) + stable.astype(np.float32)
        if frame_index == 0:
            image[30, 30] += 100.0
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=14.0,
            candidate_count=0,
            sources=(),
            parameters={"mask_zero_pixels": 0, "saturation_level": -1.0},
            quality_count=0,
        )
        analyses.append(
            FrameAnalysis(
                FitsFrame(Path(f"coadd-{frame_index}.fits"), {}, image, None, 0),
                detection,
                None,
                None,
            )
        )

    coadd = _registered_coadd_reference(analyses, ((0.0, 0.0),) * 5, tile_rows=24)

    assert coadd is not None
    assert coadd[15, 15] > 20.0
    assert coadd[30, 30] == pytest.approx(10.0, abs=1e-5)


def test_analyze_sequence_rejects_unknown_temporal_proposal_mode() -> None:
    with pytest.raises(ValueError, match="temporal_proposal_mode"):
        sequence_module.analyze_sequence(("not-read.fits",), temporal_proposal_mode="unknown")


def test_analyze_sequence_rejects_temporal_reference_snr_above_candidate_gate() -> None:
    with pytest.raises(ValueError, match="temporal_reference_min_snr"):
        sequence_module.analyze_sequence(("not-read.fits",), temporal_reference_min_snr=16.0)


def test_analyze_sequence_rejects_temporal_candidate_snr_above_candidate_gate() -> None:
    with pytest.raises(ValueError, match="temporal_candidate_min_snr"):
        sequence_module.analyze_sequence(("not-read.fits",), temporal_candidate_min_snr=16.0)


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


def test_detect_motion_features_suppresses_a_static_streak_with_temporal_reference() -> None:
    analyses = []
    for frame_index in range(4):
        image = np.full((96, 96), 10.0, dtype=np.float32)
        for offset in range(42):
            image[15 + offset, 12 + offset] = 300.0
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
        frame = FitsFrame(Path(f"static-{frame_index}.fits"), {}, image, None, 0)
        analyses.append(FrameAnalysis(frame, detection, None, None))

    tracks, _threshold = detect_motion_features(tuple(analyses), ((0.0, 0.0),) * 4, psf_fwhm=3.0)

    assert tracks == ()


def test_detect_long_trails_exposes_single_frame_streak_as_candidate() -> None:
    updates: list[tuple[float, str]] = []
    tracks = detect_long_trails(
        _feature_analysis(0),
        psf_fwhm=3.0,
        progress=lambda value, label: updates.append((value, label)),
    )

    assert len(tracks) == 1
    assert tracks[0].classification == "candidate"
    assert tracks[0].presence == 1
    assert tracks[0].points[0].frame_index == 0
    assert updates[0] == (0.0, "准备线状残差筛选")
    assert updates[-1] == (100.0, "线状目标筛选完成")


def test_detect_single_frame_long_trails_uses_lightweight_baseline() -> None:
    updates: list[tuple[float, str]] = []

    tracks = detect_single_frame_long_trails(
        _feature_analysis(0).frame,
        psf_fwhm=3.0,
        progress=lambda value, label: updates.append((value, label)),
    )

    assert len(tracks) == 1
    assert tracks[0].classification == "candidate"
    assert tracks[0].presence == 1
    assert tracks[0].points[0].length_px > 30.0
    assert updates[0][1] == "估计单帧背景与噪声"
    assert updates[-1][0] == pytest.approx(100.0)


def test_detect_motion_features_can_explain_frame_level_filtering() -> None:
    audits = []
    tracks, _threshold = detect_motion_features(
        (_feature_analysis(0),),
        ((0.0, 0.0),),
        psf_fwhm=3.0,
        audit_sink=audits,
    )

    assert len(tracks) == 1
    assert len(audits) == 1
    audit = audits[0]
    assert audit.component_count >= 1
    assert audit.area_pass_count == 1
    assert audit.geometry_pass_count == 1
    assert audit.feature_count == 1
    assert audit.edge_rejected_count == 0


def test_analyze_sequence_reports_frame_and_stage_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((32, 32), 10.0, dtype=np.float32)
    detection = DetectionResult((32, 32), 10.0, 1.0, 14.0, 0, (), {}, 0)
    fake_frame = FitsFrame(Path("fake.fits"), {"EXPOSURE": 1500.0}, image, None, 0)
    fake_analysis = FrameAnalysis(fake_frame, detection, None, None)
    detector_calls: list[dict[str, object]] = []

    def fake_analyze_frame(_path: str | Path, **kwargs: object) -> FrameAnalysis:
        detector_calls.append(dict(kwargs))
        callback = kwargs["progress"]
        assert callable(callback)
        callback(50.0, "mock detection")  # type: ignore[operator]
        callback(100.0, "mock complete")  # type: ignore[operator]
        return fake_analysis

    def fake_track(*_args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            cumulative_shifts=((0.0, 0.0), (0.0, 0.0)),
            tracks=(),
            link_radius_px=float(kwargs["link_radius_px"]),
            min_presence=2,
            motion_min_displacement_px=float(kwargs["motion_min_displacement_px"]),
            max_motion_fit_rms_px=float(kwargs["max_motion_fit_rms_px"]),
        )

    monkeypatch.setattr(sequence_module, "analyze_frame", fake_analyze_frame)
    monkeypatch.setattr(sequence_module, "track_detections", fake_track)
    def fake_motion(*_args: object, **kwargs: object) -> tuple[tuple[object, ...], float]:
        callback = kwargs["progress"]
        assert callable(callback)
        callback(0.0, "mock residual start")  # type: ignore[operator]
        callback(100.0, "mock residual complete")  # type: ignore[operator]
        return (), 0.0

    monkeypatch.setattr(sequence_module, "detect_motion_features", fake_motion)
    stages: list[tuple[str, int, int]] = []
    details: list[tuple[int, int, float, str]] = []

    result = sequence_module.analyze_sequence(
        ("frame-1.fits", "frame-2.fits"),
        progress=lambda stage, index, total: stages.append((stage, index, total)),
        detail_progress=lambda index, total, value, label: details.append((index, total, value, label)),
    )

    assert stages == [
        ("prepare", 0, 2),
        ("frame", 1, 2),
        ("frame", 2, 2),
        ("sentinel-audit", 2, 2),
        ("registration", 2, 2),
        ("fast-point-motion", 0, 2),
        ("fast-point-motion", 2, 2),
        ("stack-faint", 2, 2),
        ("consensus", 2, 2),
        ("motion-detail", 0, 100),
        ("motion-detail", 100, 100),
        ("motion", 2, 2),
        ("complete", 2, 2),
    ]
    assert {item[:2] for item in details} == {(1, 2), (2, 2)}
    assert sum(item[2:] == (100.0, "mock complete") for item in details) == 2
    assert all(call["max_sources"] == 6000 for call in detector_calls)
    assert all(call["background_box_size"] == 256 for call in detector_calls)
    assert all(call["background_sample_limit"] == DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT for call in detector_calls)
    assert all(call["fast_sequence"] is True for call in detector_calls)
    assert all(call["refine_local_background"] is False for call in detector_calls)
    assert all(call["use_float32"] is True for call in detector_calls)
    assert result.source_working_limit == 6000
    assert result.background_sample_limit == DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT
    assert result.fast_sequence is True
    assert result.fixed_sentinel_audit is not None
    assert result.fixed_sentinel_audit.fixed_coordinate_count == 0
    assert result.fixed_sentinel_impact_audit is not None
    assert result.fixed_sentinel_impact_audit.affected_returned_source_count == 0

    detector_calls.clear()
    full_result = sequence_module.analyze_sequence(
        ("frame-1.fits", "frame-2.fits"),
        sequence_max_sources=None,
    )
    assert all(call.get("max_sources") is None for call in detector_calls)
    assert full_result.source_working_limit is None


def _faint_frame_analysis(
    frame_index: int,
    *,
    star: bool = True,
    noise: float = 3.0,
    background: float = 20.0,
) -> FrameAnalysis:
    rng = np.random.default_rng(10_000 + frame_index)
    image = rng.normal(background, noise, size=(48, 48)).astype(np.float32)
    if star:
        yy, xx = np.indices(image.shape, dtype=np.float64)
        sigma = 1.2
        image += (10.0 * np.exp(-0.5 * ((xx - 24.0) ** 2 + (yy - 24.0) ** 2) / sigma**2)).astype(np.float32)
    detection = DetectionResult(
        image_shape=image.shape,
        background=background,
        noise=noise,
        threshold=background + 4.0 * noise,
        candidate_count=0,
        sources=(),
        parameters={"mask_zero_pixels": 0, "saturation_level": -1.0, "aperture_radius": 4},
        quality_count=0,
    )
    frame = FitsFrame(Path(f"faint-{frame_index}.fits"), {}, image, None, 0)
    return FrameAnalysis(frame, detection, None, None)


def _faint_quality_source() -> Detection:
    return Detection(
        detection_id=1,
        x=24.0,
        y=24.0,
        peak=30.0,
        flux=90.0,
        background=20.0,
        noise=3.0,
        snr=10.0,
        fwhm=2.8,
        flags=(),
        flux_error=25.0,
        flux_snr=3.6,
        filter_snr=12.0,
        quality_passed=True,
        peak_x=24.0,
        peak_y=24.0,
    )


def test_stack_forced_frame_measure_batch_matches_scalar_on_interior_sources() -> None:
    rng = np.random.default_rng(77)
    image = rng.normal(20.0, 3.0, size=(96, 96)).astype(np.float32)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    for cx0, cy0, sigma, amplitude in ((24.0, 24.0, 1.2, 10.0), (50.0, 50.0, 1.2, 8.0), (60.0, 30.0, 1.2, 12.0)):
        image += (amplitude * np.exp(-0.5 * ((xx - cx0) ** 2 + (yy - cy0) ** 2) / sigma**2)).astype(np.float32)
    mask = np.zeros(image.shape, dtype=bool)
    xs = np.array([24.0, 50.0, 60.0])
    ys = np.array([24.0, 50.0, 30.0])

    batch = _stack_forced_frame_measure_batch(
        image,
        mask,
        xs,
        ys,
        aperture_radius=4,
        min_psf_support_pixels=3,
        global_background=20.0,
        global_noise=3.0,
    )

    for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
        scalar = _stack_forced_frame_measure(
            image,
            mask,
            x,
            y,
            aperture_radius=4,
            min_psf_support_pixels=3,
            psf_fwhm=2.0,
            global_background=20.0,
            global_noise=3.0,
        )
        assert batch[0][index] == pytest.approx(scalar[0], rel=1e-3)
        assert batch[1][index] == scalar[1]
        assert batch[2][index] == scalar[2]
        assert batch[3][index] == pytest.approx(scalar[3], rel=1e-3)
        assert batch[4][index] == pytest.approx(scalar[4], rel=1e-3)


def test_stack_forced_frame_measure_detects_faint_point_source() -> None:
    analysis = _faint_frame_analysis(0)
    flux_snr, support, center_valid, fwhm, ellipticity = _stack_forced_frame_measure(
        np.asarray(analysis.frame.data),
        np.zeros(analysis.frame.data.shape, dtype=bool),
        24.0,
        24.0,
        aperture_radius=4,
        min_psf_support_pixels=3,
        psf_fwhm=2.0,
        global_background=float(analysis.detection.background),
        global_noise=float(analysis.detection.noise),
    )

    assert flux_snr is not None and flux_snr > 3.0
    assert support >= 3
    assert center_valid is True
    assert fwhm is not None and 0.8 <= fwhm <= 12.0
    assert ellipticity is not None and ellipticity <= 0.65


def test_stack_faint_tracks_recovers_persistent_faint_source(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tuple(_faint_frame_analysis(index) for index in range(3))
    reference = np.full((48, 48), 20.0, dtype=np.float64)
    faint_source = _faint_quality_source()
    fake_detection = DetectionResult(
        image_shape=reference.shape,
        background=20.0,
        noise=1.5,
        threshold=26.0,
        candidate_count=1,
        sources=(faint_source,),
        parameters={},
        quality_count=1,
    )
    monkeypatch.setattr(sequence_module, "detect_sources", lambda *_args, **_kwargs: fake_detection)

    tracks = _stack_faint_tracks(
        frames,
        ((0.0, 0.0),) * len(frames),
        reference,
        threshold_sigma=4.0,
        min_distance=4,
        aperture_radius=4,
        psf_fwhm=2.0,
        min_flux_snr=5.0,
        min_psf_support_pixels=3,
        proposal_mode="gaussian",
        stack_frame_min_flux_snr=3.0,
        min_presence=3,
        link_radius_px=4.0,
        existing_tracks=(),
    )

    assert len(tracks) == 1
    assert tracks[0].evidence_level == "stack_faint"
    assert tracks[0].classification == "persistent"
    assert tracks[0].presence == 3


def test_stack_faint_tracks_deduplicates_explained_position(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tuple(_faint_frame_analysis(index) for index in range(3))
    reference = np.full((48, 48), 20.0, dtype=np.float64)
    fake_detection = DetectionResult(
        image_shape=reference.shape,
        background=20.0,
        noise=1.5,
        threshold=26.0,
        candidate_count=1,
        sources=(_faint_quality_source(),),
        parameters={},
        quality_count=1,
    )
    monkeypatch.setattr(sequence_module, "detect_sources", lambda *_args, **_kwargs: fake_detection)
    explained = SourceTrack(
        track_id=0,
        classification="static",
        points=(
            TrackPoint(
                frame_index=0,
                detection_id=9,
                x=24.0,
                y=24.0,
                aligned_x=24.0,
                aligned_y=24.0,
                flux_snr=20.0,
            ),
        ),
        displacement_px=0.0,
        speed_px_per_frame=0.0,
        fit_rms_px=0.1,
        evidence_level="quality",
    )

    tracks = _stack_faint_tracks(
        frames,
        ((0.0, 0.0),) * len(frames),
        reference,
        threshold_sigma=4.0,
        min_distance=4,
        aperture_radius=4,
        psf_fwhm=2.0,
        min_flux_snr=5.0,
        min_psf_support_pixels=3,
        proposal_mode="gaussian",
        stack_frame_min_flux_snr=3.0,
        min_presence=3,
        link_radius_px=4.0,
        existing_tracks=(explained,),
    )

    assert tracks == ()


def test_stack_faint_tracks_rejects_transient_source(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = (_faint_frame_analysis(0, star=True), _faint_frame_analysis(1, star=False), _faint_frame_analysis(2, star=False))
    reference = np.full((48, 48), 20.0, dtype=np.float64)
    fake_detection = DetectionResult(
        image_shape=reference.shape,
        background=20.0,
        noise=1.5,
        threshold=26.0,
        candidate_count=1,
        sources=(_faint_quality_source(),),
        parameters={},
        quality_count=1,
    )
    monkeypatch.setattr(sequence_module, "detect_sources", lambda *_args, **_kwargs: fake_detection)

    tracks = _stack_faint_tracks(
        frames,
        ((0.0, 0.0),) * len(frames),
        reference,
        threshold_sigma=4.0,
        min_distance=4,
        aperture_radius=4,
        psf_fwhm=2.0,
        min_flux_snr=5.0,
        min_psf_support_pixels=3,
        proposal_mode="gaussian",
        stack_frame_min_flux_snr=3.0,
        min_presence=3,
        link_radius_px=4.0,
        existing_tracks=(),
    )

    assert tracks == ()
