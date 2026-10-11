from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rst19.detection import Detection
from rst19.fits import read_fits
from rst19.models import FitsFrame
from rst19.pipeline import FrameAnalysis
from rst19.sequence import (
    DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT,
    MotionFeaturePoint,
    MotionFeatureTrack,
    SourceTrack,
    TrackPoint,
    _candidate_consensus_tracks,
    _coherent_raw_motion_point_ids,
    _finite_line_template,
    _fit_motion_feature_track,
    _motion_feature_pair_geometry_penalty,
    _observation_times_seconds,
    _estimate_translation,
    _global_frame_assignments,
    _rescue_collinear_motion_feature_groups,
    _registered_coadd_reference,
    _registered_median_reference,
    _single_frame_line_model_evidence,
    _single_frame_line_geometries_overlap,
    _deduplicate_single_frame_line_tracks,
    _single_frame_trail_shape_reason,
    _single_frame_trail_shape_is_supported,
    _stack_faint_tracks,
    _stack_forced_frame_measure,
    _stack_forced_frame_measure_batch,
    _fit_track_coordinates,
    _link_motion_evidence_tracks,
    _track_fast_candidate_movers,
    _temporal_psf_fwhm_bank,
    _temporal_reference_candidate_frames,
    _line_continuity_ratio,
    _line_artifact_mask_from_coordinates,
    _line_profile_metrics,
    _motion_feature_shape_is_supported,
    _motion_feature_track_shape_is_supported,
    _motion_feature_track_model_is_supported,
    _motion_feature_motion_axis_is_supported,
    _motion_feature_artifact_is_supported_as_moving,
    _motion_feature_artifact_overlap_is_compatible_with_motion,
    _motion_feature_temporal_continuity_is_supported,
    _motion_feature_internal_gap_bridge_is_supported,
    _motion_feature_edge_truncation_is_supported,
    _motion_feature_confirmation_presence,
    _motion_feature_rejection_reasons,
    _stitch_unassociated_single_frame_points,
    _motion_feature_exposure_is_supported,
    _motion_feature_exposure_metrics,
    _sequence_motion_candidate_min_length,
    _single_frame_sequence_line_model_is_review_supported,
    _deconvolved_exposure_motion_length,
    _exposure_aware_short_trail_min_area,
    audit_fixed_sentinel,
    audit_fixed_sentinel_impacts,
    detect_long_trails,
    detect_motion_features,
    detect_single_frame_long_trails,
    filter_single_frame_trail_blends,
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


def test_track_fit_uses_elapsed_seconds_but_keeps_legacy_interval_speed() -> None:
    points = tuple(
        TrackPoint(
            frame_index=index,
            detection_id=index,
            x=float(2 * index),
            y=0.0,
            aligned_x=float(2 * index),
            aligned_y=0.0,
            flux_snr=20.0,
        )
        for index in range(3)
    )
    displacement, speed_per_interval, fit_rms = _fit_track_coordinates(
        points,
        np.asarray(((0.0, 0.0), (2.0, 0.0), (6.0, 0.0)), dtype=np.float64),
        frame_times=(0.0, 1.0, 3.0),
    )

    # 实际轨迹是 2 px/s，首末间隔 3 s，因此位移为 6 px；公共字段仍
    # 用中位观测间隔 1.5 s 折回为 3 px/interval。
    assert displacement == pytest.approx(6.0)
    assert speed_per_interval == pytest.approx(3.0)
    assert fit_rms == pytest.approx(0.0)


def test_track_detections_removes_small_global_rotation_before_classifying() -> None:
    # 平移注册仍会让视场边缘留下随位置变化的亚像素/像素级残差；这类
    # 残差在 15 帧上可能恰好拟合成一条低 RMS 直线，但不应被标成 moving。
    reference = np.asarray(
        [
            (120.0, 120.0),
            (300.0, 120.0),
            (500.0, 120.0),
            (700.0, 120.0),
            (900.0, 120.0),
            (120.0, 320.0),
            (300.0, 320.0),
            (500.0, 320.0),
            (700.0, 320.0),
            (900.0, 320.0),
            (120.0, 520.0),
            (300.0, 520.0),
            (500.0, 520.0),
            (700.0, 520.0),
            (900.0, 520.0),
            (120.0, 720.0),
            (300.0, 720.0),
            (500.0, 720.0),
            (700.0, 720.0),
            (900.0, 720.0),
            (120.0, 900.0),
            (300.0, 900.0),
            (500.0, 900.0),
            (700.0, 900.0),
            (900.0, 900.0),
        ],
        dtype=np.float64,
    )
    center = np.asarray((510.0, 510.0), dtype=np.float64)
    frames = []
    for frame_index in range(15):
        angle = 0.00032 * frame_index
        matrix = np.asarray(
            (
                (np.cos(angle), -np.sin(angle)),
                (np.sin(angle), np.cos(angle)),
            ),
            dtype=np.float64,
        )
        positions = (reference - center) @ matrix.T + center + np.asarray((1.5 * frame_index, -0.8 * frame_index))
        frames.append(tuple(_source(index, x, y) for index, (x, y) in enumerate(positions)))

    result = track_detections(
        frames,
        min_presence=15,
        link_radius_px=4.0,
        registration_radius_px=8.0,
        motion_min_displacement_px=2.0,
    )

    assert result.registration_model == "translation+robust_affine"
    assert result.registration_affine_frame_count >= 10
    assert result.moving_track_count == 0
    assert result.effective_motion_min_displacement_px is not None


def test_estimate_translation_recovers_drift_beyond_local_registration_radius() -> None:
    previous = tuple(
        _source(index, 24.0 + (index % 6) * 40.0, 24.0 + (index // 6) * 60.0)
        for index in range(24)
    )
    current = tuple(
        _source(index, source.x - 2.0, source.y + 18.0)
        for index, source in enumerate(previous)
    )

    # 8 px remains the intended local association scale, but registration
    # must be able to search farther and select the coherent displacement
    # cluster instead of a wrong nearby source.
    assert _estimate_translation(previous, current, radius_px=8.0) == pytest.approx(
        (-2.0, 18.0),
        abs=1e-6,
    )


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


def test_link_motion_evidence_tracks_deduplicates_point_and_line_channels() -> None:
    point_track = SourceTrack(
        track_id=7813,
        classification="moving",
        points=tuple(
            TrackPoint(
                frame_index=index,
                detection_id=100 + index,
                x=10.0 + 10.0 * index,
                y=20.0 - 5.0 * index,
                aligned_x=10.0 + 10.0 * index,
                aligned_y=20.0 - 5.0 * index,
                flux_snr=20.0,
            )
            for index in range(3)
        ),
        displacement_px=22.36,
        speed_px_per_frame=11.18,
        fit_rms_px=0.0,
        evidence_level="fast_point_motion",
    )

    def line_point(frame_index: int, x: float, y: float) -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index=frame_index,
            x=x,
            y=y,
            aligned_x=x,
            aligned_y=y,
            residual_snr=30.0,
            area_pixels=40,
            length_px=12.0,
            width_px=3.0,
            angle_deg=-26.565,
            bbox=(int(x - 5), int(y - 5), int(x + 5), int(y + 5)),
            touches_edge=False,
            continuity_ratio=0.95,
            profile_fill_ratio=0.9,
        )

    matching_line = MotionFeatureTrack(
        track_id=2,
        classification="moving",
        points=tuple(
            line_point(index, 12.0 + 10.0 * index, 19.0 - 5.0 * index)
            for index in range(3)
        ),
        displacement_px=22.36,
        speed_px_per_frame=11.18,
        fit_rms_px=0.0,
    )
    unrelated_line = MotionFeatureTrack(
        track_id=4,
        classification="moving",
        points=tuple(
            line_point(index, 12.0 + 10.0 * index, 60.0 - 5.0 * index)
            for index in range(3)
        ),
        displacement_px=22.36,
        speed_px_per_frame=11.18,
        fit_rms_px=0.0,
    )

    links = _link_motion_evidence_tracks(
        (point_track,),
        (matching_line, unrelated_line),
    )

    assert len(links) == 1
    assert links[0].point_track_id == 7813
    assert links[0].line_track_id == 2
    assert links[0].overlap_frame_count == 3
    assert links[0].overlap_fraction == pytest.approx(1.0)
    assert links[0].median_perpendicular_distance_px == pytest.approx(0.0, abs=1e-6)
    assert links[0].direction_difference_deg == pytest.approx(0.0, abs=1e-3)


def test_global_frame_assignments_preserves_two_nearby_tracks() -> None:
    # 贪心最近邻会先让 track 11 占用 source 0，随后 track 10 失去唯一
    # 可用源；全局分配应选择两条轨迹都能延续的组合。
    assignments = _global_frame_assignments(
        ((10, 0.0, 0.0), (11, 1.0, 0.0)),
        np.asarray(((0.9, 0.0), (2.8, 0.0)), dtype=np.float64),
        link_radius_px=2.0,
    )

    assert {(track_index, source_index) for _, track_index, source_index in assignments} == {
        (10, 0),
        (11, 1),
    }
    assert [distance for distance, _, _ in assignments] == pytest.approx([0.9, 1.8])


def test_global_frame_assignments_can_remove_an_exposure_inconsistent_pair() -> None:
    assignments = _global_frame_assignments(
        ((10, 0.0, 0.0), (11, 1.0, 0.0)),
        np.asarray(((0.2, 0.0), (1.2, 0.0)), dtype=np.float64),
        link_radius_px=2.0,
        pair_allowed=lambda track_id, _source_index, _distance: track_id == 11,
    )

    assert len(assignments) == 1
    assert assignments[0][0] == pytest.approx(0.2)
    assert assignments[0][1:] == (11, 1)


def test_global_frame_assignments_can_prefer_soft_geometry_cost() -> None:
    assignments = _global_frame_assignments(
        ((10, 0.0, 0.0),),
        np.asarray(((2.0, 0.0), (0.0, 2.0)), dtype=np.float64),
        link_radius_px=3.0,
        pair_cost=lambda _track_id, source_index, _distance: (
            0.0 if source_index == 0 else 4.0
        ),
    )

    assert assignments[0][1:] == (10, 0)
    # 对外仍返回几何距离，而不是距离加软代价。
    assert assignments[0][0] == pytest.approx(2.0)


def test_motion_feature_pair_geometry_uses_direction_and_translation_endpoints() -> None:
    previous = MotionFeaturePoint(
        frame_index=0,
        x=100.0,
        y=100.0,
        aligned_x=100.0,
        aligned_y=100.0,
        residual_snr=20.0,
        area_pixels=100,
        length_px=20.0,
        width_px=3.0,
        angle_deg=179.0,
        bbox=(90, 90, 110, 110),
        touches_edge=False,
    )
    aligned = replace(
        previous,
        frame_index=1,
        x=110.0,
        aligned_x=110.0,
        angle_deg=1.0,
    )
    perpendicular = replace(aligned, angle_deg=90.0)

    aligned_penalty = _motion_feature_pair_geometry_penalty(
        previous,
        aligned,
        psf_fwhm=2.0,
    )
    perpendicular_penalty = _motion_feature_pair_geometry_penalty(
        previous,
        perpendicular,
        psf_fwhm=2.0,
    )

    # 179° 与 1° 对运动方向 0° 是同一条近似水平线，不应发生角度
    # 环绕的假不一致；垂直候选则应被软评分明显降权。
    assert aligned_penalty < 0.2
    assert perpendicular_penalty > aligned_penalty + 1.0


def test_motion_feature_pair_geometry_does_not_hard_reject_edge_truncation() -> None:
    previous = MotionFeaturePoint(
        0,
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        100,
        30.0,
        3.0,
        0.0,
        (0, 0, 16, 16),
        True,
    )
    current = replace(
        previous,
        frame_index=1,
        x=20.0,
        aligned_x=20.0,
        bbox=(4, 0, 24, 16),
        touches_edge=True,
        angle_deg=75.0,
    )

    penalty = _motion_feature_pair_geometry_penalty(
        previous,
        current,
        psf_fwhm=2.0,
    )

    assert np.isfinite(penalty)
    assert penalty >= 0.0


def test_motion_feature_pair_geometry_keeps_both_exposure_direction_branches() -> None:
    previous = MotionFeaturePoint(
        0,
        100.0,
        100.0,
        100.0,
        100.0,
        20.0,
        100,
        20.0,
        3.0,
        0.0,
        (90, 98, 110, 102),
        False,
    )
    forward = replace(previous, frame_index=1, x=110.0, aligned_x=110.0)
    backward = replace(previous, frame_index=1, x=90.0, aligned_x=90.0)
    perpendicular = replace(
        previous,
        frame_index=1,
        x=100.0,
        y=110.0,
        aligned_x=100.0,
        aligned_y=110.0,
        angle_deg=90.0,
    )

    forward_penalty = _motion_feature_pair_geometry_penalty(
        previous,
        forward,
        psf_fwhm=2.0,
        expected_step_px=10.0,
    )
    backward_penalty = _motion_feature_pair_geometry_penalty(
        previous,
        backward,
        psf_fwhm=2.0,
        expected_step_px=10.0,
    )
    perpendicular_penalty = _motion_feature_pair_geometry_penalty(
        previous,
        perpendicular,
        psf_fwhm=2.0,
        expected_step_px=10.0,
    )

    # 曝光内线段只有无方向轴，沿轴正向和反向都必须保留；垂直于线段
    # 的候选才应因方向和预期步长不一致而明显降权。
    assert forward_penalty < 0.2
    assert backward_penalty < 0.2
    assert perpendicular_penalty > forward_penalty + 1.0


@pytest.mark.parametrize("frame_count", [1, 2, 3, 4])
def test_track_detections_short_sequences_do_not_fail_default_presence(frame_count: int) -> None:
    frames = [(_source(0, 10.0 + frame_index, 10.0),) for frame_index in range(frame_count)]

    result = track_detections(frames, link_radius_px=3.0)

    assert result.tracks
    if frame_count < 3:
        assert result.moving_track_count == 0


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
        # 曝光内拖影轴应与相邻帧的中心位移方向一致；这里用接近 8:6
        # 的帧间位移模拟真实拖影方向，而不是用斜线配水平漂移。
        y = 15 + 6 * frame_index + offset
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
    assert tracks[0].displacement_px == pytest.approx(30.0, abs=0.5)


def test_detect_motion_features_passes_1500ms_step_to_first_line_association(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyses = []
    for frame_index in range(4):
        image = np.full((96, 96), 10.0, dtype=np.float32)
        start_x = 16 + 10 * frame_index
        image[39:42, start_x : start_x + 12] = 300.0
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
        frame = FitsFrame(
            Path(f"exposure-step-{frame_index}.fits"),
            {
                "DATE-OBS": f"2026-01-01T00:00:{frame_index * 2:02d}.000",
                "EXPOSURE": 1500.0,
            },
            image,
            None,
            frame_index,
        )
        analyses.append(FrameAnalysis(frame, detection, None, None))

    expected_steps: list[float | None] = []
    original_penalty = sequence_module._motion_feature_pair_geometry_penalty

    def record_expected_step(*args: object, **kwargs: object) -> float:
        expected_steps.append(kwargs.get("expected_step_px"))  # type: ignore[arg-type]
        return original_penalty(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        sequence_module,
        "_motion_feature_pair_geometry_penalty",
        record_expected_step,
    )

    tracks, _threshold = detect_motion_features(
        tuple(analyses),
        ((0.0, 0.0),) * 4,
        psf_fwhm=3.0,
        temporal_reference=np.full((96, 96), 10.0, dtype=np.float32),
        raw_fallback_workers=1,
    )

    finite_steps = [
        float(step)
        for step in expected_steps
        if step is not None and np.isfinite(float(step)) and float(step) > 0.0
    ]
    assert finite_steps
    # 线长约 12 px、时间差 2 s、曝光 1.5 s；这里验证的是首个
    # “没有历史速度”的配对也拿到了曝光推导步长，而不是断言某个
    # 特定候选最终一定获选。
    assert max(finite_steps) > 1.0
    moving = [track for track in tracks if track.classification == "moving"]
    assert len(moving) == 1
    assert moving[0].presence == 4
    assert moving[0].exposure_velocity_ratio == pytest.approx(1.5, abs=0.25)


def test_detect_motion_features_reconnects_two_short_line_fragments_across_two_missing_frames() -> None:
    frame_analyses = []
    present_frames = {4, 5, 6, 7, 10, 11, 12, 13}
    for frame_index in range(15):
        image = np.full((192, 192), 10.0, dtype=np.float32)
        if frame_index in present_frames:
            start_x = 20 + 9 * frame_index
            image[50:53, start_x : start_x + 14] = 300.0
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
        frame = FitsFrame(
            Path(f"gapped-exposure-line-{frame_index}.fits"),
            {
                "DATE-OBS": f"2026-01-01T00:00:{frame_index * 2:02d}",
                "EXPOSURE": 1500.0,
            },
            image,
            None,
            frame_index,
        )
        frame_analyses.append(FrameAnalysis(frame, detection, None, None))

    tracks, _threshold = detect_motion_features(
        tuple(frame_analyses),
        ((0.0, 0.0),) * 15,
        psf_fwhm=2.0,
        min_residual_adu=50.0,
        min_axis_ratio=2.5,
        raw_fallback_workers=1,
    )

    merged = [track for track in tracks if track.presence == len(present_frames)]
    assert len(merged) == 1
    # 宽筛允许跨两帧缺口关联，避免候选消失；本例还通过逐帧原图线模型、
    # 形态、常速度拟合和 1.5 s 曝光速度一致性，因此进入严格桥接分支。
    # 中间缺帧本身仍会保留在证据标签中，不能把它伪装成无缺口观测。
    assert merged[0].classification == "moving"
    assert "interior_gap_bridged" in merged[0].evidence_level
    assert tuple(point.frame_index for point in merged[0].points) == tuple(sorted(present_frames))
    assert merged[0].exposure_velocity_ratio == pytest.approx(2.0, abs=0.25)


def test_long_sequence_requires_half_frame_coverage_for_confirmed_line_motion() -> None:
    """15 帧中只出现 3--7 帧的共线结果必须留在 candidate 层。"""

    assert _motion_feature_confirmation_presence(4) == 3
    assert _motion_feature_confirmation_presence(7) == 3
    assert _motion_feature_confirmation_presence(8) == 4
    assert _motion_feature_confirmation_presence(15) == 8
    with pytest.raises(ValueError, match="frame_count"):
        _motion_feature_confirmation_presence(0)


def test_edge_truncated_line_can_use_one_frame_presence_relaxation_only_at_sequence_end() -> None:
    """连续从边缘进入并延伸到序列末端的线可解释一帧缺失。"""

    frame_shapes = ((128, 128),) * 15
    suffix_points = tuple(
        MotionFeaturePoint(
            frame_index=index,
            x=10.0 + index * 2.0,
            y=100.0 - index * 5.0,
            aligned_x=10.0 + index * 2.0,
            aligned_y=100.0 - index * 5.0,
            residual_snr=50.0,
            area_pixels=80,
            length_px=36.0,
            width_px=3.0,
            angle_deg=-32.0,
            bbox=(10 + index * 2, 127 if index == 8 else 80 - index * 3, 30 + index * 2, 127 if index == 8 else 100 - index * 3),
            touches_edge=index == 8,
            continuity_ratio=0.90,
            profile_fill_ratio=0.88,
        )
        for index in range(8, 15)
    )
    assert _motion_feature_edge_truncation_is_supported(suffix_points, frame_shapes) is True

    middle_points = tuple(
        replace(point, frame_index=point.frame_index - 4, bbox=(point.bbox[0], 10, point.bbox[2], 35), touches_edge=False)
        for point in suffix_points
    )
    assert _motion_feature_edge_truncation_is_supported(middle_points, frame_shapes) is False

    suffix_without_edge = tuple(
        replace(point, bbox=(point.bbox[0], 20, point.bbox[2], 50), touches_edge=False)
        for point in suffix_points
    )
    assert _motion_feature_edge_truncation_is_supported(suffix_without_edge, frame_shapes) is False


def test_exposure_line_speed_matches_inter_frame_speed_for_a_1500ms_source() -> None:
    """1.5 s 曝光形成的线长应与相邻帧位移给出同一速度量级。"""

    psf_fwhm = 2.0
    motion_px_per_frame = 9.0
    psf_sigma = psf_fwhm / 2.3548200450309493
    measured_length = np.sqrt(
        motion_px_per_frame**2 / 0.75 + 16.0 * psf_sigma**2
    )
    frame_analyses = tuple(
        FrameAnalysis(
            FitsFrame(
                Path(f"exposure-{frame_index}.fits"),
                {
                    "DATE-OBS": f"2026-01-01T00:00:{frame_index * 2:02d}",
                    "EXPOSURE": 1500.0,
                },
                np.zeros((32, 64), dtype=np.float32),
                None,
                frame_index,
            ),
            DetectionResult((32, 64), 0.0, 1.0, 5.0, 0, (), {}, 0),
            None,
            None,
        )
        for frame_index in range(4)
    )
    points = tuple(
        MotionFeaturePoint(
            frame_index,
            10.0 + motion_px_per_frame * frame_index,
            16.0,
            10.0 + motion_px_per_frame * frame_index,
            16.0,
            50.0,
            80,
            float(measured_length),
            2.0,
            0.0,
            (1, 1, 62, 30),
            False,
            0.95,
            0.95,
        )
        for frame_index in range(4)
    )

    ratio, log_mad, consistency = _motion_feature_exposure_metrics(
        points,
        frame_analyses,
        (0.0, 1.5, 3.0, 4.5),
        psf_fwhm=psf_fwhm,
    )

    assert _deconvolved_exposure_motion_length(measured_length, psf_fwhm) == pytest.approx(
        motion_px_per_frame,
        rel=1e-6,
    )
    assert ratio == pytest.approx(1.0, rel=1e-6)
    assert log_mad == pytest.approx(0.0, abs=1e-10)
    assert consistency == pytest.approx(1.0)
    assert _motion_feature_exposure_is_supported(ratio, log_mad, consistency) is True


def test_exposure_aware_sequence_length_floor_only_widens_1p5s_proposals() -> None:
    """1.5 s 只影响前端候选召回，不改变显式门槛或无元数据路径。"""

    assert _sequence_motion_candidate_min_length(
        psf_fwhm=2.0,
        min_length_px=None,
        exposure_s=(1.5,) * 15,
        metadata_supported=True,
    ) == pytest.approx(6.0)
    assert _sequence_motion_candidate_min_length(
        psf_fwhm=3.0,
        min_length_px=None,
        exposure_s=(1.5,) * 15,
        metadata_supported=True,
    ) == pytest.approx(7.5)
    assert _sequence_motion_candidate_min_length(
        psf_fwhm=2.0,
        min_length_px=None,
        exposure_s=(1.5,) * 14 + (None,),
        metadata_supported=False,
    ) == pytest.approx(8.0)
    assert _sequence_motion_candidate_min_length(
        psf_fwhm=2.0,
        min_length_px=11.0,
        exposure_s=(1.5,) * 15,
        metadata_supported=True,
    ) == pytest.approx(11.0)
    assert _sequence_motion_candidate_min_length(
        psf_fwhm=2.0,
        min_length_px=None,
        exposure_s=(1.0,) * 15,
        metadata_supported=True,
    ) == pytest.approx(8.0)


def test_single_frame_1p5s_header_uses_the_shared_proposal_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float | None] = {}

    def fake_extract(_analysis: object, **kwargs: object) -> tuple[tuple[MotionFeatureTrack, ...], float]:
        captured["min_length_px"] = float(kwargs["min_length_px"])
        return (), float(kwargs["min_residual_adu"])

    monkeypatch.setattr(sequence_module, "_extract_single_frame_trail_candidates", fake_extract)
    frame = FitsFrame(
        Path("single-frame-1p5s.fits"),
        {"EXPOSURE": 1500.0},
        np.zeros((32, 32), dtype=np.float32),
        None,
        0,
    )

    detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    assert captured["min_length_px"] == pytest.approx(6.0)


def test_known_1p5s_exposure_lowers_only_the_short_line_proposal_floor() -> None:
    assert _exposure_aware_short_trail_min_area(psf_fwhm=2.0, exposure_s=1.5) == 12
    assert _exposure_aware_short_trail_min_area(psf_fwhm=4.0, exposure_s=1.5) == 20
    assert _exposure_aware_short_trail_min_area(psf_fwhm=2.0, exposure_s=None) == 20
    assert _exposure_aware_short_trail_min_area(psf_fwhm=2.0, exposure_s=1.0) == 20


def test_sequence_exposure_aware_area_floor_reaches_the_main_line_path() -> None:
    """1.5 s 的 12--19 px 短线应先进入主路径，再由后续门复核。"""

    def make_analyses(exposure_ms: float | None) -> tuple[FrameAnalysis, ...]:
        analyses = []
        for frame_index in range(4):
            image = np.full((96, 96), 10.0, dtype=np.float32)
            start_x = 18 + 10 * frame_index
            image[40:42, start_x : start_x + 9] = 300.0
            header = {} if exposure_ms is None else {
                "DATE-OBS": f"2026-01-01T00:00:{frame_index * 2:02d}.000",
                "EXPOSURE": exposure_ms,
            }
            analyses.append(
                FrameAnalysis(
                    FitsFrame(Path(f"area-floor-{exposure_ms}-{frame_index}.fits"), header, image, None, 0),
                    DetectionResult(
                        image_shape=image.shape,
                        background=10.0,
                        noise=2.0,
                        threshold=40.0,
                        candidate_count=0,
                        sources=(),
                        parameters={},
                        quality_count=0,
                    ),
                    None,
                    None,
                )
            )
        return tuple(analyses)

    exposure_audits: list[MotionFrameAudit] = []
    detect_motion_features(
        make_analyses(1500.0),
        ((0.0, 0.0),) * 4,
        psf_fwhm=2.0,
        temporal_reference=np.full((96, 96), 10.0, dtype=np.float32),
        audit_sink=exposure_audits,
        raw_fallback_workers=1,
    )
    assert exposure_audits[0].short_min_pixels == 12
    assert exposure_audits[0].area_pass_count >= 1

    unknown_audits: list[MotionFrameAudit] = []
    detect_motion_features(
        make_analyses(None),
        ((0.0, 0.0),) * 4,
        psf_fwhm=2.0,
        temporal_reference=np.full((96, 96), 10.0, dtype=np.float32),
        audit_sink=unknown_audits,
        raw_fallback_workers=1,
    )
    assert unknown_audits[0].short_min_pixels == 20
    assert unknown_audits[0].area_pass_count == 0


def test_exposure_line_speed_rejects_a_static_or_wrongly_linked_line() -> None:
    """线长与帧间位移不在同一速度量级时不能升级为 moving。"""

    frame_analyses = tuple(
        FrameAnalysis(
            FitsFrame(
                Path(f"wrong-link-{frame_index}.fits"),
                {
                    "DATE-OBS": f"2026-01-01T00:00:{frame_index * 2:02d}",
                    "EXPOSURE": 1500.0,
                },
                np.zeros((32, 128), dtype=np.float32),
                None,
                frame_index,
            ),
            DetectionResult((32, 128), 0.0, 1.0, 5.0, 0, (), {}, 0),
            None,
            None,
        )
        for frame_index in range(4)
    )
    points = tuple(
        MotionFeaturePoint(
            frame_index,
            10.0 + 30.0 * frame_index,
            16.0,
            10.0 + 30.0 * frame_index,
            16.0,
            50.0,
            80,
            10.53,
            2.0,
            0.0,
            (1, 1, 120, 30),
            False,
            0.95,
            0.95,
        )
        for frame_index in range(4)
    )

    ratio, log_mad, consistency = _motion_feature_exposure_metrics(
        points,
        frame_analyses,
        (0.0, 1.5, 3.0, 4.5),
        psf_fwhm=2.0,
    )

    assert ratio == pytest.approx(0.288, abs=0.005)
    assert log_mad == pytest.approx(0.0, abs=1e-10)
    assert consistency == pytest.approx(0.0)
    assert _motion_feature_exposure_is_supported(ratio, log_mad, consistency) is False


def test_parallel_raw_single_frame_fallback_preserves_motion_result() -> None:
    frame_analyses = tuple(_feature_analysis(index) for index in range(4))
    shifts = ((0.0, 0.0),) * len(frame_analyses)

    serial_tracks, serial_threshold = detect_motion_features(
        frame_analyses,
        shifts,
        psf_fwhm=3.0,
        raw_fallback_workers=1,
    )
    parallel_tracks, parallel_threshold = detect_motion_features(
        frame_analyses,
        shifts,
        psf_fwhm=3.0,
        raw_fallback_workers=2,
    )

    assert parallel_threshold == pytest.approx(serial_threshold)
    assert parallel_tracks == serial_tracks


def test_motion_feature_raw_fallback_worker_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="raw_fallback_workers"):
        detect_motion_features(
            tuple(_feature_analysis(index) for index in range(2)),
            ((0.0, 0.0),) * 2,
            psf_fwhm=3.0,
            raw_fallback_workers=0,
        )


def test_line_artifact_mask_is_recorded_without_deleting_single_frame_line() -> None:
    image = np.full((64, 64), 10.0, dtype=np.float32)
    image[20:40, 30] = 300.0
    artifact_coordinates = np.column_stack(
        (np.full(20, 30, dtype=np.int32), np.arange(20, 40, dtype=np.int32))
    )
    assert _line_artifact_mask_from_coordinates(artifact_coordinates, image.shape).sum() == 20
    detection = DetectionResult(
        image_shape=image.shape,
        background=10.0,
        noise=2.0,
        threshold=40.0,
        candidate_count=0,
        sources=(),
        parameters={},
        quality_count=0,
        line_artifact_coordinates=artifact_coordinates,
    )
    frame = FitsFrame(Path("line-artifact-overlap.fits"), {}, image, None, 0)

    tracks = detect_long_trails(
        FrameAnalysis(frame, detection, None, None),
        psf_fwhm=2.0,
        residual_sigma=10.0,
        min_residual_adu=60.0,
        min_feature_area=20,
        min_axis_ratio=2.5,
        min_length_px=8.0,
        bridge_gaps=True,
    )

    assert tracks
    assert tracks[0].points[0].line_artifact_overlap_fraction > 0.9


def test_coherent_raw_motion_chain_keeps_moving_line_and_rejects_static_line() -> None:
    def point(frame_index: int, x: float, y: float) -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index,
            x,
            y,
            x,
            y,
            60.0,
            28,
            11.0,
            2.5,
            140.0,
            (int(x - 4), int(y - 3), int(x + 4), int(y + 3)),
            False,
            0.92,
            0.88,
        )

    moving = [
        [point(frame_index, 20.0 + 8.0 * frame_index, 70.0 - 5.0 * frame_index)]
        for frame_index in range(4)
    ]
    static = [[point(frame_index, 20.0, 70.0)] for frame_index in range(4)]

    moving_ids = _coherent_raw_motion_point_ids(moving, psf_fwhm=2.0)
    static_ids = _coherent_raw_motion_point_ids(static, psf_fwhm=2.0)

    assert len(moving_ids) == 4
    assert static_ids == set()


def test_observation_times_use_date_obs_and_fit_line_with_elapsed_seconds() -> None:
    headers = (
        "2026-03-30T16:32:05.413",
        "2026-03-30T16:32:06.886",
        "2026-03-30T16:32:08.415",
    )
    analyses = tuple(
        FrameAnalysis(
            FitsFrame(Path(f"time-{index}.fits"), {"DATE-OBS": value}, np.zeros((8, 8), dtype=np.float32), None, 0),
            DetectionResult((8, 8), 0.0, 1.0, 4.0, 0, (), {}, 0),
            None,
            None,
        )
        for index, value in enumerate(headers)
    )

    times = _observation_times_seconds(analyses)

    assert times == pytest.approx((0.0, 1.473, 3.002))

    speed_x, speed_y = 2.0, -1.0
    points = tuple(
        MotionFeaturePoint(
            frame_index=index,
            x=10.0 + speed_x * float(time),
            y=20.0 + speed_y * float(time),
            aligned_x=10.0 + speed_x * float(time),
            aligned_y=20.0 + speed_y * float(time),
            residual_snr=20.0,
            area_pixels=30,
            length_px=20.0,
            width_px=2.0,
            angle_deg=0.0,
            bbox=(0, 0, 1, 1),
            touches_edge=False,
        )
        for index, time in enumerate(times)
    )
    displacement, speed_per_frame, fit_rms = _fit_motion_feature_track(
        points,
        frame_times=times,
        reference_interval_s=float(np.median(np.diff(times))),
    )

    expected_speed_per_second = float(np.hypot(speed_x, speed_y))
    assert displacement == pytest.approx(expected_speed_per_second * 3.002, rel=1e-6)
    assert speed_per_frame == pytest.approx(expected_speed_per_second * 1.501, rel=1e-6)
    assert fit_rms == pytest.approx(0.0, abs=1e-9)


def test_detect_motion_features_rescues_a_long_trail_beyond_center_link_radius() -> None:
    analyses = []
    for frame_index, start_y in enumerate((420, 300, 180, 60)):
        image = np.full((512, 512), 10.0, dtype=np.float32)
        image[start_y : start_y + 60, 256] = 300.0
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
        frame = FitsFrame(Path(f"long-trail-{frame_index}.fits"), {}, image, None, 0)
        analyses.append(FrameAnalysis(frame, detection, None, None))

    tracks, _threshold = detect_motion_features(
        tuple(analyses),
        ((0.0, 0.0),) * 4,
        psf_fwhm=3.0,
    )

    moving = [track for track in tracks if track.classification == "moving"]
    assert len(moving) == 1
    assert moving[0].presence == 4
    assert moving[0].evidence_level == "collinear_trail"
    assert moving[0].geometry_rms_px == pytest.approx(moving[0].fit_rms_px)
    assert moving[0].displacement_px > 300.0


def test_detect_motion_features_recovers_a_short_continuous_streak() -> None:
    analyses = []
    for frame_index in range(4):
        image = np.full((96, 96), 10.0, dtype=np.float32)
        start_x = 16 + 10 * frame_index
        image[39:42, start_x : start_x + 12] = 300.0
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
        frame = FitsFrame(Path(f"short-moving-trail-{frame_index}.fits"), {}, image, None, 0)
        analyses.append(FrameAnalysis(frame, detection, None, None))

    tracks, _threshold = detect_motion_features(
        tuple(analyses),
        ((0.0, 0.0),) * 4,
        psf_fwhm=3.0,
        temporal_reference=np.full((96, 96), 10.0, dtype=np.float32),
    )

    moving = [track for track in tracks if track.classification == "moving"]
    assert len(moving) == 1
    assert moving[0].presence == 4
    assert all(point.length_px < 16.0 for point in moving[0].points)
    assert _motion_feature_track_shape_is_supported(moving[0].points) is True


def test_detect_motion_features_does_not_promote_a_static_adjacent_star_pair() -> None:
    """同一位置的两颗静态星不能仅因相邻而产生 moving 线状轨迹。"""

    yy, xx = np.mgrid[:96, :96]
    sigma = 2.0 / 2.354820045
    image = np.full((96, 96), 10.0, dtype=np.float32)
    for center_x, amplitude in ((42.0, 320.0), (50.6, 145.0)):
        image += (
            amplitude
            * np.exp(-0.5 * (((xx - center_x) ** 2 + (yy - 48.0) ** 2) / (sigma * sigma)))
        ).astype(np.float32)

    analyses = tuple(
        FrameAnalysis(
            FitsFrame(Path(f"static-adjacent-pair-{frame_index}.fits"), {}, image.copy(), None, 0),
            DetectionResult(
                image_shape=image.shape,
                background=10.0,
                noise=2.0,
                threshold=40.0,
                candidate_count=2,
                sources=(),
                parameters={},
                quality_count=0,
            ),
            None,
            None,
        )
        for frame_index in range(4)
    )

    features, _threshold = detect_motion_features(
        analyses,
        ((0.0, 0.0),) * 4,
        psf_fwhm=2.0,
        temporal_reference=image,
        multiscale_line_proposals=False,
    )

    assert not [track for track in features if track.classification == "moving"]


def test_detect_motion_features_uses_multiscale_spatial_proposals_for_low_peak_streak() -> None:
    """线段积分显著但逐像素低于残差门时，15 帧路线仍应能关联。"""

    analyses = []
    yy, xx = np.indices((96, 96), dtype=np.float64)
    psf_sigma = 3.0 / 2.354820045
    for frame_index in range(4):
        image = np.full((96, 96), 10.0, dtype=np.float32)
        start_x = 16 + 10 * frame_index
        # 每个 PSF 的峰值约为 4 ADU，低于本测试故意设置的 100 ADU
        # 连通域门；12 px 连续结构的沿线积分由多尺度提议器负责发现。
        for offset in np.arange(-6.0, 6.25, 0.5):
            source_x = start_x + 6.0 + offset
            image += (
                4.0
                * np.exp(
                    -0.5
                    * (((xx - source_x) ** 2 + (yy - 40.0) ** 2) / (psf_sigma**2))
                )
            ).astype(np.float32) * 0.5
        detection = DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=1.0,
            threshold=100.0,
            candidate_count=0,
            sources=(),
            parameters={},
            quality_count=0,
        )
        frame = FitsFrame(Path(f"low-peak-moving-trail-{frame_index}.fits"), {}, image, None, 0)
        analyses.append(FrameAnalysis(frame, detection, None, None))

    tracks, _threshold = detect_motion_features(
        tuple(analyses),
        ((0.0, 0.0),) * 4,
        psf_fwhm=3.0,
        temporal_reference=np.full((96, 96), 10.0, dtype=np.float32),
        multiscale_line_proposals=True,
    )

    moving = [track for track in tracks if track.classification == "moving"]
    assert len(moving) == 1
    assert moving[0].presence == 4
    assert "single_frame_shape" in moving[0].evidence_level


def test_detect_motion_features_bridges_one_pixel_gap_in_short_streak() -> None:
    analyses = []
    for frame_index in range(4):
        image = np.full((96, 96), 10.0, dtype=np.float32)
        start_x = 16 + 10 * frame_index
        image[39:42, start_x : start_x + 12] = 300.0
        image[40, start_x + 6] = 10.0
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
        frame = FitsFrame(Path(f"gapped-short-moving-trail-{frame_index}.fits"), {}, image, None, 0)
        analyses.append(FrameAnalysis(frame, detection, None, None))

    tracks, _threshold = detect_motion_features(
        tuple(analyses),
        ((0.0, 0.0),) * 4,
        psf_fwhm=3.0,
        temporal_reference=np.full((96, 96), 10.0, dtype=np.float32),
    )

    moving = [track for track in tracks if track.classification == "moving"]
    assert len(moving) == 1
    assert moving[0].presence == 4
    assert all(point.area_pixels == 35 for point in moving[0].points)


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

    # A fixed line must not be promoted to a cross-frame motion track, but the
    # raw single-frame evidence is retained for audit/display instead of being
    # silently discarded.
    assert all(track.classification == "candidate" for track in tracks)
    assert all(track.evidence_level == "single_frame_trail" for track in tracks)
    assert all(track.presence == 1 for track in tracks)
    assert not any(track.classification == "moving" for track in tracks)


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
    assert updates[-1] == (100.0, "单帧线状候选筛选完成")


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


def test_single_frame_route_records_linear_artifact_overlap_for_a_line() -> None:
    image = np.full((64, 64), 10.0, dtype=np.float32)
    image[12:52, 30] = 300.0
    frame = FitsFrame(Path("single-frame-line-artifact-audit.fits"), {}, image, None, 0)

    tracks = detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    assert tracks
    assert max(track.points[0].line_artifact_overlap_fraction for track in tracks) > 0.9
    assert all(
        _single_frame_trail_shape_is_supported(track.points[0]) is False
        for track in tracks
        if track.points[0].line_artifact_overlap_fraction is not None
        and track.points[0].line_artifact_overlap_fraction > 0.9
    )


def test_single_frame_default_wide_gate_keeps_a_short_exposure_streak() -> None:
    image = np.full((64, 64), 10.0, dtype=np.float32)
    for offset in range(22):
        if offset != 10:
            image[18 + offset, 18 + offset] = 300.0
    frame = FitsFrame(Path("short-exposure-streak.fits"), {}, image, None, 0)

    old_strict = detect_single_frame_long_trails(
        frame,
        psf_fwhm=2.0,
        residual_sigma=15.0,
        min_residual_adu=100.0,
        min_feature_area=40,
        min_axis_ratio=4.0,
    )
    # 缺口桥接现在是显式的保守选项；这个回归仍验证“宽松长度/轴比”
    # 可以找回带一个量化缺口的短拖影，但不把它误当成单帧默认口径。
    default_wide = detect_single_frame_long_trails(
        frame,
        psf_fwhm=2.0,
        bridge_gaps=True,
    )

    assert old_strict == ()
    assert len(default_wide) == 1
    assert default_wide[0].is_single_frame_motion_candidate is True
    assert default_wide[0].evidence_level == "single_frame_trail"
    assert default_wide[0].points[0].length_px >= 18.0
    assert default_wide[0].points[0].continuity_ratio is not None


def test_single_frame_default_does_not_bridge_an_adjacent_double_star() -> None:
    """默认单帧口径不能把两个 PSF 的空隙连成曝光线。"""

    size = 64
    yy, xx = np.mgrid[:size, :size]
    sigma = 2.0 / 2.354820045
    image = np.full((size, size), 10.0, dtype=np.float32)
    for center, amplitude in (((28.0, 28.0), 300.0), ((33.0, 35.0), 220.0)):
        cx, cy = center
        image += (
            amplitude
            * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / (sigma * sigma))
        ).astype(np.float32)
    frame = FitsFrame(Path("single-frame-adjacent-double-star.fits"), {}, image, None, 0)

    tracks = detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    assert tracks == ()


def test_single_frame_trail_path_skips_temporal_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((64, 64), 10.0, dtype=np.float32)
    image[20:44, 28:31] = 300.0
    frame = FitsFrame(Path("single-frame-no-temporal-reference.fits"), {}, image, None, 0)

    def fail_if_called(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("single-frame trail detection must not build a temporal reference")

    monkeypatch.setattr("rst19.sequence._registered_median_reference", fail_if_called)

    tracks = detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    assert tracks
    assert all(track.evidence_level == "single_frame_trail" for track in tracks)


def test_single_frame_trail_path_does_not_enter_multi_frame_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((64, 64), 10.0, dtype=np.float32)
    image[20:44, 28:31] = 300.0
    frame = FitsFrame(Path("single-frame-independent-route.fits"), {}, image, None, 0)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("single-frame trail detection must not enter detect_motion_features")

    monkeypatch.setattr("rst19.sequence.detect_motion_features", fail_if_called)

    tracks = detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    assert tracks
    assert all(track.classification == "candidate" for track in tracks)
    assert all(track.evidence_level == "single_frame_trail" for track in tracks)


def test_large_single_frame_route_keeps_multiscale_proposals_current_frame_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """大图会进入单帧宽筛，但仍不能触发任何 15 帧入口。"""

    image = np.full((512, 512), 10.0, dtype=np.float32)
    image[120:420, 256] = 300.0
    frame = FitsFrame(Path("large-single-frame-independent-route.fits"), {}, image, None, 0)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("large single-frame route must not enter sequence analysis")

    monkeypatch.setattr(sequence_module, "analyze_sequence", fail_if_called)
    monkeypatch.setattr(sequence_module, "detect_motion_features", fail_if_called)
    monkeypatch.setattr(sequence_module, "_registered_median_reference", fail_if_called)
    # 只替换当前帧宽筛的外部提案器，仍让 detect_single_frame_long_trails
    # 走真实的单帧连通域/模型路径；这样测试确实经过大图分支，但不需要
    # 在单元测试里重复扫描 512² 图像的所有方向和长度。
    monkeypatch.setattr(sequence_module, "detect_single_frame_trail_proposals", lambda *args, **kwargs: ())

    tracks = detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    assert tracks
    assert all(track.classification == "candidate" for track in tracks)
    assert all(track.evidence_level == "single_frame_trail" for track in tracks)


def test_sequence_singleton_is_not_labeled_as_single_frame_exposure() -> None:
    point = MotionFeaturePoint(
        3,
        24.0,
        28.0,
        24.0,
        28.0,
        18.0,
        32,
        22.0,
        2.0,
        135.0,
        (18, 20, 30, 36),
        False,
    )
    sequence_track = MotionFeatureTrack(
        1,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="center_motion",
    )

    assert sequence_track.is_single_frame_motion_candidate is False


def test_motion_feature_candidate_serializes_final_gate_reasons() -> None:
    point = MotionFeaturePoint(
        frame_index=4,
        x=843.5,
        y=1597.8,
        aligned_x=843.5,
        aligned_y=1597.8,
        residual_snr=180.1,
        area_pixels=30,
        length_px=13.0,
        width_px=2.7,
        angle_deg=0.0,
        bbox=(837, 1596, 850, 1600),
        touches_edge=False,
        continuity_ratio=0.9,
        profile_fill_ratio=0.9,
        line_artifact_overlap_fraction=0.0,
        single_frame_model="double",
    )
    reasons = _motion_feature_rejection_reasons(
        classification="candidate",
        evidence_level="center_motion",
        points=(point,),
        presence_supported=False,
        displacement_px=0.0,
        min_displacement_px=20.0,
        fit_rms_px=None,
        max_fit_rms_px=4.0,
        shape_supported=True,
        model_supported=False,
        motion_axis_supported=False,
        temporal_continuity_supported=False,
        exposure_supported=True,
        artifact_supported=True,
    )
    assert reasons == (
        "NO_CROSS_FRAME_ASSOCIATION",
        "INSUFFICIENT_FRAME_PRESENCE",
        "INSUFFICIENT_DISPLACEMENT",
        "NO_MOTION_FIT",
        "COMPACT_DOUBLE_MODEL_PRESENT",
        "RAW_LINE_MODEL_MAJORITY_NOT_MET",
        "MOTION_AXIS_MISMATCH",
        "FRAME_GAP_OR_NONCONTIGUOUS",
    )
    track = MotionFeatureTrack(
        7,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        rejection_reasons=reasons,
    )
    payload = track.as_dict()
    assert payload["rejection_reasons"] == list(reasons)


def test_motion_feature_rejection_reasons_are_empty_for_confirmed_track() -> None:
    assert _motion_feature_rejection_reasons(
        classification="moving",
        evidence_level="center_motion",
        points=(),
    ) == ()


def test_stitch_reconnects_strictly_predicted_short_single_frame_lines() -> None:
    def make_point(frame_index: int) -> MotionFeaturePoint:
        x = 20.0 + 7.0 * frame_index
        y = 50.0 - 4.0 * frame_index
        return MotionFeaturePoint(
            frame_index=frame_index,
            x=x,
            y=y,
            aligned_x=x,
            aligned_y=y,
            residual_snr=25.0,
            area_pixels=30,
            length_px=10.0,
            width_px=2.5,
            angle_deg=-29.745,
            bbox=(int(x - 5), int(y - 2), int(x + 5), int(y + 2)),
            touches_edge=False,
            continuity_ratio=0.9,
            profile_fill_ratio=0.9,
            line_artifact_overlap_fraction=0.0,
            line_template_snr=20.0,
            line_vs_point_delta_bic=20.0,
            line_vs_double_delta_bic=20.0,
            line_vs_background_delta_bic=20.0,
            line_vs_elliptical_delta_bic=20.0,
            single_frame_model="line",
            single_frame_model_margin_bic=20.0,
        )

    existing = [make_point(index) for index in (4, 5, 6, 7)]
    stitched, remaining = _stitch_unassociated_single_frame_points(
        (existing,),
        (make_point(2), make_point(0)),
        psf_fwhm=2.0,
        frame_times=np.arange(8, dtype=float),
    )

    assert remaining == ()
    assert [point.frame_index for point in stitched[0]] == [0, 2, 4, 5, 6, 7]


def test_motion_fit_sorts_reconnected_points_before_using_time_span() -> None:
    def make_point(frame_index: int) -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index=frame_index,
            x=float(frame_index * 4),
            y=20.0,
            aligned_x=float(frame_index * 4),
            aligned_y=20.0,
            residual_snr=12.0,
            area_pixels=40,
            length_px=12.0,
            width_px=2.5,
            angle_deg=0.0,
            bbox=(frame_index * 4 - 5, 18, frame_index * 4 + 5, 22),
            touches_edge=False,
            continuity_ratio=0.9,
            profile_fill_ratio=0.9,
            line_artifact_overlap_fraction=0.0,
        )

    # This is the order produced before the stitching output was normalized:
    # early observations were appended after the existing F04--F07 history.
    points = tuple(make_point(index) for index in (4, 5, 6, 7, 2, 0))
    displacement, speed, rms = _fit_motion_feature_track(
        points,
        frame_times=np.arange(8, dtype=float),
        reference_interval_s=1.0,
    )

    assert displacement == pytest.approx(28.0)
    assert speed == pytest.approx(4.0)
    assert rms == pytest.approx(0.0)


def test_single_frame_short_trail_gate_recovers_continuous_sub16px_line() -> None:
    image = np.full((64, 64), 10.0, dtype=np.float32)
    for offset in range(12):
        image[20 + offset, 29:32] = 300.0
    frame = FitsFrame(Path("short-sub16px-streak.fits"), {}, image, None, 0)

    tracks = detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    short_points = [point for track in tracks for point in track.points if point.length_px < 16.0]
    assert short_points
    assert any(_single_frame_trail_shape_is_supported(point) for point in short_points)


def test_single_frame_short_trail_gate_rejects_low_fill_pair() -> None:
    point = MotionFeaturePoint(
        0,
        32.0,
        32.0,
        32.0,
        32.0,
        80.0,
        24,
        12.0,
        3.0,
        0.0,
        (26, 30, 38, 34),
        False,
        0.92,
        0.70,
    )

    assert _single_frame_trail_shape_is_supported(point) is False


def test_single_frame_finite_exposure_model_distinguishes_line_point_and_double() -> None:
    size = 96
    yy, xx = np.mgrid[:size, :size]
    valid = np.ones((size, size), dtype=bool)
    sigma = 2.0 / 2.354820045
    rng = np.random.default_rng(7)

    line_image = 100.0 * _finite_line_template(
        xx,
        yy,
        center_x=48.0,
        center_y=48.0,
        length_px=18.0,
        angle_deg=35.0,
        psf_sigma=sigma,
    ) + rng.normal(0.0, 2.0, (size, size))
    line_evidence = _single_frame_line_model_evidence(
        line_image,
        valid,
        center_x=48.0,
        center_y=48.0,
        length_px=18.0,
        width_px=2.0,
        angle_deg=35.0,
        psf_fwhm=2.0,
        noise=2.0,
    )

    angle = np.deg2rad(35.0)
    double_image = rng.normal(0.0, 2.0, (size, size))
    for offset in (-9.0, 9.0):
        double_image += 100.0 * np.exp(
            -0.5
            * (
                (xx - (48.0 + offset * np.cos(angle))) ** 2
                + (yy - (48.0 + offset * np.sin(angle))) ** 2
            )
            / (sigma * sigma)
        )
    double_evidence = _single_frame_line_model_evidence(
        double_image,
        valid,
        center_x=48.0,
        center_y=48.0,
        length_px=18.0,
        width_px=2.0,
        angle_deg=35.0,
        psf_fwhm=2.0,
        noise=2.0,
    )

    point_image = 100.0 * np.exp(
        -0.5 * ((xx - 48.0) ** 2 + (yy - 48.0) ** 2) / (sigma * sigma)
    ) + rng.normal(0.0, 2.0, (size, size))
    point_evidence = _single_frame_line_model_evidence(
        point_image,
        valid,
        center_x=48.0,
        center_y=48.0,
        length_px=18.0,
        width_px=2.0,
        angle_deg=35.0,
        psf_fwhm=2.0,
        noise=2.0,
    )

    assert line_evidence["single_frame_model"] == "line"
    assert float(line_evidence["line_vs_point_delta_bic"]) > 6.0
    assert float(line_evidence["line_vs_double_delta_bic"]) > 6.0
    assert double_evidence["single_frame_model"] == "double"
    assert point_evidence["single_frame_model"] == "point"


def test_single_frame_model_rejects_brightness_weighted_double_star() -> None:
    """不等亮双星的加权质心也不能被有限曝光线模型吞掉。"""

    size = 96
    yy, xx = np.mgrid[:size, :size]
    sigma = 2.0 / 2.354820045
    angle = np.deg2rad(54.5)
    rng = np.random.default_rng(72600)
    image = rng.normal(23.5, 8.6, (size, size))
    for offset, amplitude in ((-4.3, 3990.0), (4.3, 1769.0)):
        image += amplitude * np.exp(
            -0.5
            * (
                (xx - (48.0 + offset * np.cos(angle))) ** 2
                + (yy - (48.0 + offset * np.sin(angle))) ** 2
            )
            / (sigma * sigma)
        )

    evidence = _single_frame_line_model_evidence(
        image,
        np.ones_like(image, dtype=bool),
        # 亮星把候选的 flux-weighted centre 拉向一侧；这里故意不把
        # 中心改回两颗星的几何中点，复现截图中的失败模式。
        center_x=48.0 - 0.8 * np.cos(angle),
        center_y=48.0 - 0.8 * np.sin(angle),
        length_px=17.8,
        width_px=3.9,
        angle_deg=54.5,
        psf_fwhm=2.0,
        noise=8.6,
    )

    assert evidence["single_frame_model"] == "double"
    assert float(evidence["line_vs_double_delta_bic"]) < 0.0


def test_real_f01_id72600_pair_is_not_a_single_frame_trail() -> None:
    """锁住用户截图对应的真实 FITS 反例，防止回归成一条橙线。"""

    fits_path = (
        Path(__file__).parents[1]
        / "doc"
        / "00-项目资料"
        / "原始数据"
        / "20260330163205413_9901.fits"
    )
    if not fits_path.exists():
        pytest.skip("比赛 FITS 数据未随工作区提供")

    source = read_fits(fits_path)
    # ID 72600 的实际位置约为 (2378, 3564)；只读一个 128 px 局部窗口，
    # 让回归测试验证当前单帧入口，而不是依赖整幅图的候选数量。
    x0, y0 = 2320, 3500
    cutout = np.asarray(source.data)[y0 : y0 + 128, x0 : x0 + 128]
    frame = FitsFrame(Path("user-id72600-cutout.fits"), source.header, cutout, None, 0)
    tracks = detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    assert not any(
        float(np.hypot(track.points[0].x - 58.0, track.points[0].y - 64.0)) < 25.0
        for track in tracks
        if track.points
    )


def test_single_frame_template_uses_an_active_set_for_non_negative_amplitudes() -> None:
    """负振幅活动集必须重新拟合背景，而不是事后截断系数。"""

    rng = np.random.default_rng(19)
    x = np.linspace(-1.0, 1.0, 80)
    template_a = np.exp(-((x + 0.08) / 0.24) ** 2)
    template_b = np.exp(-((x - 0.08) / 0.24) ** 2)
    data = 12.0 + 3.0 * x + 5.0 * template_b + rng.normal(0.0, 0.05, x.size)

    rss, snr, baseline = sequence_module._fit_single_frame_template(
        data,
        (template_a, template_b),
        noise=0.05,
        background_terms=(x,),
    )
    expected_design = np.column_stack((np.ones(x.size), x, template_b))
    expected_coefficients, *_ = np.linalg.lstsq(expected_design, data, rcond=None)
    expected_rss = float(np.sum(((data - expected_design @ expected_coefficients) / 0.05) ** 2))

    assert rss == pytest.approx(expected_rss, rel=1e-10, abs=1e-10)
    assert snr == 0.0
    assert baseline == pytest.approx(expected_coefficients[0])


def test_single_frame_model_keeps_a_static_elliptical_psf_out_of_strong_line() -> None:
    size = 96
    yy, xx = np.mgrid[:size, :size]
    angle = np.deg2rad(45.0)
    dx = xx - 48.0
    dy = yy - 48.0
    along = dx * np.cos(angle) + dy * np.sin(angle)
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    image = 250.0 * np.exp(-0.5 * ((along / 2.5) ** 2 + (across / 1.0) ** 2))
    evidence = _single_frame_line_model_evidence(
        image,
        np.ones_like(image, dtype=bool),
        center_x=48.0,
        center_y=48.0,
        length_px=15.5,
        width_px=2.9,
        angle_deg=45.0,
        psf_fwhm=2.0,
        noise=1.0,
    )

    assert evidence["single_frame_model"] == "elliptical_point"
    assert float(evidence["line_vs_elliptical_delta_bic"]) < 0.0


def test_single_frame_finite_exposure_model_rejects_static_double_grid() -> None:
    size = 64
    yy, xx = np.mgrid[:size, :size]
    sigma = 2.0 / 2.354820045
    angle = np.deg2rad(35.0)

    for separation_px, brightness_ratio in ((4.0, 1.0), (8.0, 1.0), (12.0, 1.0), (8.0, 4.0), (12.0, 4.0)):
        image = np.random.default_rng(int(separation_px * 10 + brightness_ratio)).normal(
            0.0,
            2.0,
            (size, size),
        )
        for offset, amplitude in (
            (-separation_px / 2.0, 100.0),
            (separation_px / 2.0, 100.0 / brightness_ratio),
        ):
            image += amplitude * np.exp(
                -0.5
                * (
                    (xx - (32.0 + offset * np.cos(angle))) ** 2
                    + (yy - (32.0 + offset * np.sin(angle))) ** 2
                )
                / (sigma * sigma)
            )
        evidence = _single_frame_line_model_evidence(
            image,
            np.ones((size, size), dtype=bool),
            center_x=32.0,
            center_y=32.0,
            length_px=max(8.0, separation_px),
            width_px=2.0,
            angle_deg=35.0,
            psf_fwhm=2.0,
            noise=2.0,
        )

        assert evidence["single_frame_model"] == "double"
        assert float(evidence["line_vs_double_delta_bic"]) < 0.0


def test_single_frame_model_does_not_bypass_sparse_pixel_shape_gate() -> None:
    point = MotionFeaturePoint(
        frame_index=0,
        x=32.0,
        y=32.0,
        aligned_x=32.0,
        aligned_y=32.0,
        residual_snr=80.0,
        area_pixels=24,
        length_px=12.0,
        width_px=3.0,
        angle_deg=0.0,
        bbox=(26, 30, 38, 34),
        touches_edge=False,
        continuity_ratio=0.34,
        profile_fill_ratio=0.34,
        line_template_snr=100.0,
        line_vs_point_delta_bic=100.0,
        line_vs_double_delta_bic=100.0,
        line_vs_background_delta_bic=100.0,
        single_frame_model="line",
        single_frame_model_margin_bic=100.0,
    )

    assert _single_frame_trail_shape_is_supported(point) is False


def test_sequence_review_model_gate_keeps_weak_line_for_cross_frame_linking() -> None:
    """弱线只能进入 15 帧关联，不能单独绕过强线门。"""

    point = MotionFeaturePoint(
        frame_index=0,
        x=32.0,
        y=32.0,
        aligned_x=32.0,
        aligned_y=32.0,
        residual_snr=12.0,
        area_pixels=24,
        length_px=12.0,
        width_px=1.8,
        angle_deg=0.0,
        bbox=(26, 31, 38, 33),
        touches_edge=False,
        continuity_ratio=0.52,
        profile_fill_ratio=0.51,
        line_template_snr=5.2,
        line_vs_point_delta_bic=2.4,
        line_vs_double_delta_bic=2.1,
        line_vs_background_delta_bic=2.2,
        line_vs_elliptical_delta_bic=2.0,
        single_frame_model="line",
        single_frame_model_margin_bic=2.0,
    )

    assert _single_frame_sequence_line_model_is_review_supported(point, psf_fwhm=2.0) is True
    # 它没有达到 6 的强模型差异，不应在单帧阶段被当作强线。
    assert sequence_module._single_frame_sequence_line_model_is_supported(point, psf_fwhm=2.0) is False


def test_sequence_review_model_gate_rejects_compact_source_winner() -> None:
    point = MotionFeaturePoint(
        0, 32.0, 32.0, 32.0, 32.0, 20.0, 30, 14.0, 2.0, 0.0,
        (25, 31, 39, 33), False, 0.9, 0.9,
        None, 5.0, 2.5, -4.0, 2.0, -4.0, "double", 6.0,
    )
    assert _single_frame_sequence_line_model_is_review_supported(point, psf_fwhm=2.0) is False


def test_sequence_track_model_gate_requires_multiple_review_points() -> None:
    def point(frame_index: int, model: str = "line") -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index, 20.0 + frame_index, 30.0,
            20.0 + frame_index, 30.0, 20.0, 30, 12.0, 1.8, 0.0,
            (14, 29, 28, 31), False, 0.52, 0.51,
            None, 5.2, 2.4, 2.1, 2.2, 2.0, model, 2.0,
        )

    assert _motion_feature_track_model_is_supported(tuple(point(i) for i in range(3)), psf_fwhm=2.0) is True
    assert _motion_feature_track_model_is_supported(
        (point(0), point(1), point(2, model="double")),
        psf_fwhm=2.0,
    ) is True
    assert _motion_feature_track_model_is_supported(
        (point(0, model="double"), point(1, model="double"), point(2, model="line")),
        psf_fwhm=2.0,
    ) is False


def test_single_frame_overlength_candidate_is_not_treated_as_model_supported() -> None:
    point = MotionFeaturePoint(
        frame_index=0,
        x=32.0,
        y=32.0,
        aligned_x=32.0,
        aligned_y=32.0,
        residual_snr=100.0,
        area_pixels=500,
        length_px=685.0,
        width_px=12.0,
        angle_deg=90.0,
        bbox=(25, 0, 39, 684),
        touches_edge=False,
        continuity_ratio=0.92,
        profile_fill_ratio=0.90,
        single_frame_model="not_fit",
    )

    assert _single_frame_trail_shape_is_supported(point) is False
    assert "超出局部有限线段模型范围" in _single_frame_trail_shape_reason(point)


def test_single_frame_finite_exposure_model_absorbs_a_local_background_gradient() -> None:
    size = 96
    yy, xx = np.mgrid[:size, :size]
    image = (
        5.0 * (xx - 48.0) / 48.0
        + 4.0 * (yy - 48.0) / 48.0
        + np.random.default_rng(10).normal(0.0, 2.0, (size, size))
    )

    evidence = _single_frame_line_model_evidence(
        image,
        np.ones((size, size), dtype=bool),
        center_x=48.0,
        center_y=48.0,
        length_px=18.0,
        width_px=2.0,
        angle_deg=35.0,
        psf_fwhm=2.0,
        noise=2.0,
    )

    assert evidence["single_frame_model"] == "background"
    assert float(evidence["line_vs_background_delta_bic"]) < 0.0


def test_single_frame_line_nms_deduplicates_overlapping_segments_only() -> None:
    first = (100.0, 100.0, 100.0, 2.0, 45.0)
    along_offset = 20.0 / np.sqrt(2.0)
    overlapping = (100.0 + along_offset, 100.0 + along_offset, 40.0, 2.0, 46.0)
    disjoint = (100.0 + 80.0 / np.sqrt(2.0), 100.0 + 80.0 / np.sqrt(2.0), 20.0, 2.0, 45.0)
    parallel_offset = (100.0, 108.0, 80.0, 2.0, 45.0)

    assert _single_frame_line_geometries_overlap(first, overlapping) is True
    assert _single_frame_line_geometries_overlap(overlapping, first) is True
    assert _single_frame_line_geometries_overlap(first, disjoint) is False
    assert _single_frame_line_geometries_overlap(first, parallel_offset) is False


def test_single_frame_track_nms_removes_component_pieces_but_keeps_short_neighbor() -> None:
    def make_track(track_id: int, x: float, y: float, length: float) -> MotionFeatureTrack:
        point = MotionFeaturePoint(
            frame_index=0,
            x=x,
            y=y,
            aligned_x=x,
            aligned_y=y,
            residual_snr=20.0,
            area_pixels=40,
            length_px=length,
            width_px=2.0,
            angle_deg=45.0,
            bbox=(0, 0, 200, 200),
            touches_edge=False,
            continuity_ratio=0.9,
            profile_fill_ratio=0.9,
            line_artifact_overlap_fraction=0.0,
            single_frame_model="line",
        )
        return MotionFeatureTrack(
            track_id=track_id,
            classification="candidate",
            points=(point,),
            displacement_px=0.0,
            speed_px_per_frame=0.0,
            fit_rms_px=None,
            evidence_level="single_frame_trail",
        )

    long_line = make_track(1, 100.0, 100.0, 100.0)
    overlapping_piece = make_track(
        2,
        100.0 + 20.0 / np.sqrt(2.0),
        100.0 + 20.0 / np.sqrt(2.0),
        40.0,
    )
    parallel_short_line = make_track(3, 100.0, 108.0, 18.0)

    deduplicated = _deduplicate_single_frame_line_tracks(
        (long_line, overlapping_piece, parallel_short_line)
    )

    assert tuple(track.track_id for track in deduplicated) == (1, 3)


def test_single_frame_default_gate_keeps_a_broad_short_diagonal() -> None:
    image = np.full((80, 80), 10.0, dtype=np.float32)
    image[30:37, 20:40] = 300.0
    frame = FitsFrame(Path("broad-short-diagonal.fits"), {}, image, None, 0)

    strict_axis = detect_single_frame_long_trails(
        frame,
        psf_fwhm=2.0,
        min_axis_ratio=3.0,
        min_length_px=16.0,
    )
    default_wide = detect_single_frame_long_trails(frame, psf_fwhm=2.0)

    assert strict_axis == ()
    assert len(default_wide) == 1
    assert default_wide[0].points[0].length_px >= 16.0
    assert default_wide[0].points[0].width_px > 0.0
    assert default_wide[0].points[0].continuity_ratio is not None


def test_line_continuity_ratio_separates_ridge_from_two_isolated_peaks() -> None:
    continuous = np.zeros((64, 64), dtype=np.float32)
    continuous[31:34, 16:49] = 100.0
    separated = np.zeros((64, 64), dtype=np.float32)
    separated[31:34, 16:20] = 100.0
    separated[31:34, 45:49] = 100.0

    continuous_ratio = _line_continuity_ratio(
        continuous,
        center_x=32.0,
        center_y=32.5,
        length_px=32.0,
        width_px=3.0,
        angle_deg=0.0,
        threshold_adu=50.0,
    )
    separated_ratio = _line_continuity_ratio(
        separated,
        center_x=32.0,
        center_y=32.5,
        length_px=32.0,
        width_px=3.0,
        angle_deg=0.0,
        threshold_adu=50.0,
    )

    assert continuous_ratio is not None
    assert separated_ratio is not None
    assert continuous_ratio >= 0.9
    assert separated_ratio <= 0.35

    _continuous_continuity, continuous_fill = _line_profile_metrics(
        continuous,
        center_x=32.0,
        center_y=32.5,
        length_px=32.0,
        width_px=3.0,
        angle_deg=0.0,
        threshold_adu=50.0,
    )
    _separated_continuity, separated_fill = _line_profile_metrics(
        separated,
        center_x=32.0,
        center_y=32.5,
        length_px=32.0,
        width_px=3.0,
        angle_deg=0.0,
        threshold_adu=50.0,
    )
    assert continuous_fill is not None and continuous_fill >= 0.9
    assert separated_fill is not None and separated_fill <= 0.35

    continuous_point = MotionFeaturePoint(
        0, 32.0, 32.5, 32.0, 32.5, 50.0, 99, 32.0, 3.0, 0.0,
        (16, 31, 48, 33), False, continuous_ratio, continuous_fill,
    )
    separated_point = MotionFeaturePoint(
        0, 32.0, 32.5, 32.0, 32.5, 50.0, 12, 32.0, 3.0, 0.0,
        (16, 31, 48, 33), False, separated_ratio, separated_fill,
    )
    assert _motion_feature_shape_is_supported(continuous_point) is True
    assert _motion_feature_shape_is_supported(separated_point) is False


def test_motion_feature_track_shape_gate_requires_majority_of_supported_frames() -> None:
    def point(frame_index: int, continuity: float, fill: float) -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index,
            32.0 + frame_index,
            32.0,
            32.0 + frame_index,
            32.0,
            60.0,
            120,
            40.0,
            3.0,
            0.0,
            (10, 30, 60, 34),
            False,
            continuity,
            fill,
        )

    mostly_continuous = tuple(point(index, 0.80, 0.75) for index in range(3))
    one_isolated_pair = (
        point(0, 0.80, 0.75),
        point(1, 0.80, 0.75),
        point(2, 0.20, 0.18),
    )
    isolated_pairs = tuple(point(index, 0.20, 0.18) for index in range(3))

    assert _motion_feature_track_shape_is_supported(mostly_continuous) is True
    assert _motion_feature_track_shape_is_supported(one_isolated_pair) is True
    assert _motion_feature_track_shape_is_supported(isolated_pairs) is False

    direction_break = tuple(
        MotionFeaturePoint(
            item.frame_index,
            item.x,
            item.y,
            item.aligned_x,
            item.aligned_y,
            item.residual_snr,
            item.area_pixels,
            item.length_px,
            item.width_px,
            40.0 if item.frame_index == 2 else 0.0,
            item.bbox,
            item.touches_edge,
            item.continuity_ratio,
            item.profile_fill_ratio,
        )
        for item in mostly_continuous
    )
    assert _motion_feature_track_shape_is_supported(direction_break) is False

    inconsistent_lengths = tuple(
        MotionFeaturePoint(
            item.frame_index,
            item.x,
            item.y,
            item.aligned_x,
            item.aligned_y,
            item.residual_snr,
            item.area_pixels,
            100.0 if item.frame_index >= 2 else 10.0,
            item.width_px,
            item.angle_deg,
            item.bbox,
            item.touches_edge,
            item.continuity_ratio,
            item.profile_fill_ratio,
        )
        for item in (*mostly_continuous, point(3, 0.80, 0.75))
    )
    assert _motion_feature_track_shape_is_supported(inconsistent_lengths) is False


def test_motion_feature_motion_axis_gate_rejects_perpendicular_line_segments() -> None:
    def point(frame_index: int, angle_deg: float) -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index,
            20.0 + 10.0 * frame_index,
            30.0,
            20.0 + 10.0 * frame_index,
            30.0,
            60.0,
            120,
            18.0,
            3.0,
            angle_deg,
            (20, 28, 38, 32),
            False,
            0.95,
            0.90,
        )

    aligned = tuple(point(index, 0.0) for index in range(3))
    perpendicular = tuple(point(index, 90.0) for index in range(3))

    assert _motion_feature_motion_axis_is_supported(aligned) is True
    assert _motion_feature_motion_axis_is_supported(perpendicular) is False


def test_motion_feature_artifact_overlap_veto_keeps_track_out_of_moving() -> None:
    def point(frame_index: int, overlap: float | None) -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index,
            20.0 + 10.0 * frame_index,
            30.0,
            20.0 + 10.0 * frame_index,
            30.0,
            60.0,
            120,
            18.0,
            3.0,
            0.0,
            (20, 28, 38, 32),
            False,
            0.95,
            0.90,
            overlap,
        )

    artifact_track = tuple(point(index, 1.0) for index in range(7))
    mixed_track = tuple(point(index, 1.0 if index == 0 else 0.0) for index in range(7))
    unknown_track = tuple(point(index, None) for index in range(7))

    assert _motion_feature_artifact_is_supported_as_moving(artifact_track) is False
    assert _motion_feature_artifact_is_supported_as_moving(mixed_track) is True
    assert _motion_feature_artifact_is_supported_as_moving(unknown_track) is True


def test_motion_feature_artifact_overlap_can_be_overridden_by_strong_exposure_motion() -> None:
    """真实拖影被通用直线掩膜命中时，完整跨帧证据不能被自我否决。"""

    points = tuple(
        MotionFeaturePoint(
            frame_index=index,
            x=100.0 + index * 40.0,
            y=200.0 - index * 30.0,
            aligned_x=100.0 + index * 40.0,
            aligned_y=200.0 - index * 30.0,
            residual_snr=120.0,
            area_pixels=150,
            length_px=60.0,
            width_px=3.5,
            angle_deg=-36.9,
            bbox=(0, 0, 200, 200),
            touches_edge=False,
            continuity_ratio=0.90,
            profile_fill_ratio=0.86,
            line_artifact_overlap_fraction=1.0,
        )
        for index in range(7)
    )

    assert _motion_feature_artifact_overlap_is_compatible_with_motion(
        points,
        displacement_px=340.6,
        fit_rms_px=2.22,
        max_fit_rms_px=4.0,
        presence_supported=True,
        min_displacement_px=20.0,
        shape_supported=True,
        motion_axis_supported=True,
        temporal_continuity_supported=True,
        exposure_supported=True,
    ) is True
    assert _motion_feature_artifact_overlap_is_compatible_with_motion(
        points,
        displacement_px=340.6,
        fit_rms_px=5.0,
        max_fit_rms_px=4.0,
        presence_supported=True,
        min_displacement_px=20.0,
        shape_supported=True,
        motion_axis_supported=True,
        temporal_continuity_supported=True,
        exposure_supported=True,
    ) is False


def test_motion_feature_continuity_gate_rejects_two_short_runs_but_keeps_edge_run() -> None:
    def point(frame_index: int) -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index,
            20.0 + 10.0 * frame_index,
            30.0,
            20.0 + 10.0 * frame_index,
            30.0,
            60.0,
            120,
            18.0,
            3.0,
            0.0,
            (20, 28, 38, 32),
            False,
            0.95,
            0.90,
        )

    split_runs = tuple(point(index) for index in (0, 1, 2, 3, 6, 7, 8, 9))
    edge_run = tuple(point(index) for index in range(8, 15))

    assert _motion_feature_temporal_continuity_is_supported(
        split_runs,
        min_presence=8,
    ) is False
    assert _motion_feature_temporal_continuity_is_supported(
        edge_run,
        min_presence=8,
        edge_truncated=True,
    ) is True


def test_motion_feature_internal_gap_bridge_only_accepts_two_strong_runs() -> None:
    def point(frame_index: int) -> MotionFeaturePoint:
        return MotionFeaturePoint(
            frame_index,
            20.0 + 10.0 * frame_index,
            30.0,
            20.0 + 10.0 * frame_index,
            30.0,
            60.0,
            120,
            18.0,
            3.0,
            0.0,
            (20, 28, 38, 32),
            False,
            0.95,
            0.90,
        )

    assert _motion_feature_internal_gap_bridge_is_supported(
        tuple(point(index) for index in (4, 5, 6, 7, 10, 11, 12, 13))
    ) is True
    assert _motion_feature_internal_gap_bridge_is_supported(
        tuple(point(index) for index in (0, 2, 4, 5, 6, 7, 10, 11, 12, 13))
    ) is True
    assert _motion_feature_internal_gap_bridge_is_supported(
        tuple(point(index) for index in (4, 5, 10, 11))
    ) is False
    assert _motion_feature_internal_gap_bridge_is_supported(
        tuple(point(index) for index in (4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 18))
    ) is False


def test_motion_feature_track_serializes_disjoint_observed_frame_ranges() -> None:
    points = tuple(
        MotionFeaturePoint(
            frame_index,
            20.0 + 8.0 * frame_index,
            30.0,
            20.0 + 8.0 * frame_index,
            30.0,
            60.0,
            120,
            18.0,
            3.0,
            0.0,
            (20, 28, 38, 32),
            False,
            0.95,
            0.90,
        )
        for frame_index in (4, 5, 6, 7, 10, 11, 12, 13)
    )
    track = MotionFeatureTrack(3, "candidate", points, 72.0, 8.0, 0.2)

    assert track.observed_frame_indices == (4, 5, 6, 7, 10, 11, 12, 13)
    assert track.observed_frame_ranges == ((4, 7), (10, 13))
    assert track.as_dict()["observed_frame_ranges"] == [[4, 7], [10, 13]]


def test_single_frame_blend_guard_does_not_suppress_shape_candidate_from_static_neighbors() -> None:
    image = np.full((80, 80), 10.0, dtype=np.float32)
    detection = DetectionResult(
        image_shape=image.shape,
        background=10.0,
        noise=2.0,
        threshold=18.0,
        candidate_count=2,
        sources=(_source(1, 29.0, 40.0), _source(2, 39.0, 40.0)),
        parameters={},
        quality_count=2,
    )
    analysis = FrameAnalysis(FitsFrame(Path("blend-guard.fits"), {}, image, None, 0), detection, None, None)
    point = MotionFeaturePoint(
        frame_index=0,
        x=34.0,
        y=40.0,
        aligned_x=34.0,
        aligned_y=40.0,
        residual_snr=100.0,
        area_pixels=40,
        length_px=18.0,
        width_px=4.0,
        angle_deg=0.0,
        bbox=(25, 37, 43, 43),
        touches_edge=False,
    )
    track = MotionFeatureTrack(1, "candidate", (point,), 0.0, 0.0, None)

    kept, rejected_count = filter_single_frame_trail_blends((track,), analysis)

    assert kept == (track,)
    assert rejected_count == 0


def test_stack_forced_measure_batch_marks_out_of_frame_coordinates_invalid() -> None:
    image = np.full((32, 32), 10.0, dtype=np.float32)
    mask = np.zeros_like(image, dtype=bool)

    flux_snr, support, center_valid, fwhm, ellipticity = _stack_forced_frame_measure_batch(
        image,
        mask,
        np.asarray((10.0, 40.0, -3.0)),
        np.asarray((10.0, 10.0, 5.0)),
        aperture_radius=4,
        min_psf_support_pixels=3,
        global_background=10.0,
        global_noise=1.0,
    )

    assert center_valid.tolist() == [True, False, False]
    assert np.isfinite(flux_snr[0])
    assert np.isnan(flux_snr[1:]).all()
    assert support[1:].tolist() == [0, 0]
    assert np.isnan(fwhm[1:]).all()
    assert np.isnan(ellipticity[1:]).all()


def test_single_frame_trails_can_explain_frame_level_filtering() -> None:
    audits = []
    tracks = detect_long_trails(
        _feature_analysis(0),
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
        assert float(kwargs["line_link_radius_px"]) > 0.0
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
        ("registration", 0, 2),
        ("registration", 2, 2),
        ("fast-point-motion", 0, 2),
        ("fast-point-motion", 2, 2),
        ("stack-faint", 0, 2),
        ("stack-faint", 2, 2),
        ("consensus", 2, 2),
        ("motion-detail", 0, 100),
        ("motion-detail", 100, 100),
        ("motion", 2, 2),
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
    assert result.motion_line_link_radius_px == 96.0
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
        motion_line_link_radius_px=64.0,
    )
    assert all(call.get("max_sources") is None for call in detector_calls)
    assert full_result.source_working_limit is None
    assert full_result.motion_line_link_radius_px == 64.0

    relative_result = sequence_module.analyze_sequence(
        ("frame-1.fits", "frame-2.fits"),
        relative_photometry=True,
    )
    assert relative_result.relative_photometry is not None
    assert relative_result.relative_photometry.status == "NO_VALID_REFERENCES"


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
    pad = 4 + 4
    cached_batch = _stack_forced_frame_measure_batch(
        image,
        mask,
        xs,
        ys,
        aperture_radius=4,
        min_psf_support_pixels=3,
        global_background=20.0,
        global_noise=3.0,
        padded_image=np.pad(image.astype(np.float64), pad, mode="constant", constant_values=np.nan),
        padded_mask=np.pad(mask, pad, mode="constant", constant_values=True),
    )

    for expected, actual in zip(batch, cached_batch, strict=True):
        if actual.dtype == bool or np.issubdtype(actual.dtype, np.integer):
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, equal_nan=True)

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
