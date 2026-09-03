from __future__ import annotations

from pathlib import Path

import numpy as np

from rst19.cache import (
    cache_key,
    clear_cache,
    load_analysis,
    load_sequence_result,
    save_analysis,
    save_sequence_result,
    sequence_cache_key,
)
from rst19.detection import Detection, DetectionResult
from rst19.models import FitsFrame
from rst19.photometry import FaintestSource
from rst19.pipeline import FrameAnalysis
from rst19.sequence import FixedSentinelAudit, FixedSentinelImpactAudit, FixedSentinelSourceImpact, FrameSequenceSummary, MotionFeaturePoint, MotionFeatureTrack, MotionFrameAudit, SequenceResult, SourceTrack, TrackPoint


def _analysis(frame: FitsFrame) -> FrameAnalysis:
    source = Detection(
        0,
        5.0,
        6.0,
        100.0,
        20.0,
        10.0,
        2.0,
        45.0,
        2.0,
        (),
        flux_error=1.5,
        flux_snr=13.333,
        filter_snr=22.0,
        fwhm_x=2.1,
        fwhm_y=1.9,
        ellipticity=0.095,
        sharpness=0.31,
        footprint_pixels=9,
        psf_support_pixels=7,
        quality_passed=True,
        peak_x=5.0,
        peak_y=6.0,
        centroid_shift_px=0.0,
        proposal_methods=("gaussian", "dog_narrow"),
        proposal_scales=(2.6, 3.0),
        proposal_snr=24.0,
        nearest_gaussian_px=8.25,
        deblend_delta_bic=14.5,
        deblend_component_snr=6.25,
    )
    detection = DetectionResult(
        image_shape=(12, 16),
        background=10.0,
        noise=2.0,
        threshold=18.0,
        candidate_count=1,
        sources=(source,),
        parameters={"threshold_sigma": 4.0, "max_sources": -1},
        quality_count=1,
    )
    faintest = FaintestSource(
        0,
        5.0,
        6.0,
        20.0,
        45.0,
        -3.252574989,
        None,
        (),
        flux_snr=13.333,
        flux_rate=10.0,
    )
    return FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=faintest)


def test_analysis_cache_round_trip_and_clear(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.fits"
    frame_path.write_bytes(b"frame")
    frame = FitsFrame(frame_path, {"BITPIX": 16}, np.zeros((12, 16), dtype=np.int16), None, 0)
    key = cache_key(frame, threshold_sigma=4.0, min_distance=4, aperture_radius=4, max_sources=None, zero_point=None)
    cache_dir = tmp_path / ".rst19-cache"
    analysis = _analysis(frame)

    save_analysis(cache_dir, key, analysis)
    restored = load_analysis(cache_dir, key, frame)

    assert restored is not None
    assert restored.detection.sources[0].flux == 20.0
    restored_source = restored.detection.sources[0]
    assert restored_source.flux_error == 1.5
    assert restored_source.flux_snr == 13.333
    assert restored_source.filter_snr == 22.0
    assert restored_source.footprint_pixels == 9
    assert restored_source.psf_support_pixels == 7
    assert restored_source.peak_x == 5.0
    assert restored_source.peak_y == 6.0
    assert restored_source.centroid_shift_px == 0.0
    assert restored_source.proposal_methods == ("gaussian", "dog_narrow")
    assert restored_source.proposal_scales == (2.6, 3.0)
    assert restored_source.proposal_snr == 24.0
    assert restored_source.nearest_gaussian_px == 8.25
    assert restored_source.deblend_delta_bic == 14.5
    assert restored_source.deblend_component_snr == 6.25
    assert restored.detection.star_count == 1
    assert restored.faintest is not None
    assert restored.faintest.detection_id == 0
    assert restored.faintest.flux_rate == 10.0
    assert clear_cache(cache_dir) == 1
    assert load_analysis(cache_dir, key, frame) is None


def test_cache_key_changes_when_scientific_parameters_change(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.fits"
    frame_path.write_bytes(b"frame")
    frame = FitsFrame(frame_path, {"BITPIX": 16}, np.zeros((4, 4), dtype=np.int16), None, 0)

    base = cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None)
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, psf_fwhm=4.0) != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, min_flux_snr=8.0) != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, min_psf_support_pixels=4) != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, reject_linear_artifacts=False) != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, proposal_mode="hybrid") != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, proposal_mode="hybrid", dog_threshold_sigma=7.0) != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, proposal_mode="ensemble") != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, proposal_mode="ensemble", starlet_threshold_sigma=8.0) != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, enable_local_deblend=True) != base


def test_clear_cache_does_not_remove_unrelated_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".rst19-cache"
    cache_dir.mkdir()
    (cache_dir / "result.json.gz").write_bytes(b"cache")
    (cache_dir / "partial.tmp").write_bytes(b"cache")
    (cache_dir / "legacy-result.json").write_bytes(b"legacy cache")
    keep = cache_dir / "README.txt"
    keep.write_text("keep", encoding="utf-8")

    assert clear_cache(cache_dir) == 3
    assert keep.read_text(encoding="utf-8") == "keep"


def test_sequence_cache_round_trip_and_parameter_key(tmp_path: Path) -> None:
    frame_paths = (tmp_path / "frame-01.fits", tmp_path / "frame-02.fits")
    for path in frame_paths:
        path.write_bytes(b"frame")
    result = SequenceResult(
        frames=(
            FrameSequenceSummary(0, str(frame_paths[0]), 10, 8, 3, timestamp="2026-01-01T00:00:00"),
            FrameSequenceSummary(1, str(frame_paths[1]), 11, 9, 4, timestamp="2026-01-01T00:00:01"),
        ),
        cumulative_shifts=((0.0, 0.0), (0.1, 0.2)),
        tracks=(
            SourceTrack(
                4,
                "moving",
                (
                    TrackPoint(0, 1, 10.0, 20.0, 10.0, 20.0, 12.0),
                    TrackPoint(1, 2, 11.0, 21.0, 10.9, 20.8, 13.0),
                ),
                1.2,
                1.2,
                0.1,
            ),
        ),
        link_radius_px=4.0,
        min_presence=2,
        motion_min_displacement_px=2.0,
        max_motion_fit_rms_px=0.75,
        motion_features=(
            MotionFeatureTrack(
                9,
                "candidate",
                (MotionFeaturePoint(0, 30.0, 40.0, 30.0, 40.0, 20.0, 50, 40.0, 3.0, 10.0, (10, 20, 50, 60), False),),
                0.0,
                0.0,
                None,
            ),
        ),
        motion_residual_threshold_adu=100.0,
        motion_reference_mode="temporal_median_small_shift",
        motion_frame_audits=(
            MotionFrameAudit(0, 100.0, 2.8, 180, 24, 8, 3, 2, 1, 1, 42.0),
        ),
        source_working_limit=6000,
        calculation_dtype="float32",
        background_model_mode="shared_sequence_pilot",
        persistent_min_presence=1,
        temporal_multiscale=True,
        temporal_min_psf_correlation=0.85,
        fixed_sentinel_audit=FixedSentinelAudit(
            sentinel_value=-1,
            frame_count=2,
            image_shape=(4096, 4096),
            same_shape=True,
            total_occurrences=6,
            per_frame_occurrences=(3, 3),
            fixed_coordinate_count=1,
            fixed_coordinates=((1274, 3467),),
        ),
        fixed_sentinel_impact_audit=FixedSentinelImpactAudit(
            frame_count=2,
            affected_frame_count=1,
            candidate_peak_data_available=True,
            candidate_peak_frame_count=2,
            candidate_peak_count=30,
            candidate_peak_affected_count=3,
            affected_returned_source_count=2,
            affected_quality_source_count=1,
            records=(
                FixedSentinelSourceImpact(
                    frame_index=0,
                    detection_id=12,
                    x=1274.0,
                    y=3467.0,
                    peak_x=1274.0,
                    peak_y=3467.0,
                    nearest_fixed_distance_px=0.0,
                    fixed_coordinate_count=2,
                    peak=3991.0,
                    flux_snr=10.85,
                    quality_passed=False,
                    flags=("NEGATIVE_OVERFLOW",),
                ),
            ),
            record_count=1,
        ),
    )
    cache_dir = tmp_path / ".rst19-cache"
    key = sequence_cache_key(frame_paths, parameters={"threshold_sigma": 4.0})

    save_sequence_result(cache_dir, key, result)
    restored = load_sequence_result(cache_dir, key)

    assert restored is not None
    assert restored.frames[1].timestamp == "2026-01-01T00:00:01"
    assert restored.tracks[0].points[1].aligned_y == 20.8
    assert restored.motion_features[0].points[0].bbox == (10, 20, 50, 60)
    assert restored.motion_reference_mode == "temporal_median_small_shift"
    assert restored.motion_frame_audits[0].geometry_pass_count == 2
    assert restored.motion_frame_audits[0].max_feature_residual_snr == 42.0
    assert restored.source_working_limit == 6000
    assert restored.calculation_dtype == "float32"
    assert restored.background_model_mode == "shared_sequence_pilot"
    assert restored.persistent_min_presence == 1
    assert restored.temporal_multiscale is True
    assert restored.temporal_min_psf_correlation == 0.85
    assert restored.fixed_sentinel_audit is not None
    assert restored.fixed_sentinel_audit.fixed_coordinate_count == 1
    assert restored.fixed_sentinel_audit.fixed_coordinates == ((1274, 3467),)
    assert restored.fixed_sentinel_impact_audit is not None
    assert restored.fixed_sentinel_impact_audit.candidate_peak_affected_count == 3
    assert restored.fixed_sentinel_impact_audit.records[0].flags == ("NEGATIVE_OVERFLOW",)
    assert sequence_cache_key(frame_paths, parameters={"threshold_sigma": 5.0}) != key
    assert clear_cache(cache_dir) == 1
    assert load_sequence_result(cache_dir, key) is None
