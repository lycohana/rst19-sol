from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rst19.experiments import (
    _gaussian_source,
    _inject_source_signal,
    _empirical_source,
    EmpiricalPSF,
    _local_patch_stats,
    _RealInjectionSite,
    _real_injection_special_mask,
    _fixed_position_multi_psf_fit,
    _free_position_multi_psf_fit,
    classify_source_feature,
    classify_source_diagnostic_subgroup,
    classify_source_proposal_subgroup,
    estimate_empirical_psf,
    run_feature_cross_audit,
    run_feature_audit,
    run_crowded_blend_audit,
    run_sequence_feature_persistence,
    run_source_pair_audit,
    run_local_multipsf_audit,
    run_local_free_multipsf_audit,
    run_local_free_multipsf_radius_sweep,
    run_temporal_code_audit,
    run_detection_psf_sweep,
    run_single_frame_trail_audit,
    run_detection_threshold_sweep,
    run_injection_recovery,
    run_pair_flux_ratio_audit,
    run_proposal_mode_comparison,
    run_real_background_injection,
    run_stratified_real_background_injection,
    summarize_feature_psf_similarity,
    summarize_feature_morphology,
    summarize_feature_spatial_distribution,
    summarize_sequence_feature_temporal_profiles,
    summarize_sequence_feature_proposal_methods,
    summarize_spatial_psf_similarity,
    summarize_source_features,
    summarize_source_proposal_methods,
    summarize_source_feature_subclasses,
    write_detection_source_artifacts,
    write_detection_source_cutouts,
    write_feature_cross_audit_artifacts,
    write_feature_audit_artifacts,
    write_crowded_blend_audit_artifacts,
    write_sequence_feature_persistence_artifacts,
    write_source_pair_audit_artifacts,
    write_local_multipsf_audit_artifacts,
    write_local_free_multipsf_audit_artifacts,
    write_local_free_multipsf_radius_sweep_artifacts,
    write_temporal_code_audit_artifacts,
    write_proposal_mode_comparison_artifacts,
    write_pair_flux_ratio_audit_artifacts,
    write_stratified_real_background_injection_artifacts,
    write_sequence_source_audit,
    write_single_frame_trail_artifacts,
)
from rst19.detection import Detection, DetectionResult
from rst19.models import FitsFrame
from rst19.pipeline import FrameAnalysis
from rst19.sequence import MotionFeaturePoint, MotionFeatureTrack


def test_injection_recovery_is_reproducible_and_reports_both_layers() -> None:
    first = run_injection_recovery(
        peak_levels=(12.0, 80.0),
        trials_per_level=1,
        sources_per_trial=6,
        image_shape=(128, 128),
        seed=19019,
    )
    second = run_injection_recovery(
        peak_levels=(12.0, 80.0),
        trials_per_level=1,
        sources_per_trial=6,
        image_shape=(128, 128),
        seed=19019,
    )

    assert first == second
    assert first[0].candidate_recall <= first[1].candidate_recall
    assert first[0].quality_recall <= first[1].quality_recall
    assert first[1].candidate_recall > 0.0
    assert first[1].quality_recall > 0.0


def test_feature_audit_separates_known_sources_from_negative_artifacts_and_writes_outputs(tmp_path) -> None:
    progress: list[tuple[int, int]] = []
    first = run_feature_audit(
        image_shape=(96, 96),
        background_box_size=32,
        progress=lambda index, total: progress.append((index, total)),
    )
    second = run_feature_audit(image_shape=(96, 96), background_box_size=32)

    assert first == second
    assert progress == [(index, 11) for index in range(1, 12)]
    by_scenario = {row.scenario: row for row in first}
    assert by_scenario["isolated_fwhm2"].truth_count == 1
    assert by_scenario["isolated_fwhm2"].candidate_true_hits == 1
    assert by_scenario["isolated_fwhm2"].quality_true_hits == 1
    assert by_scenario["pair_separation3"].truth_count == 2
    assert by_scenario["pair_separation3"].nearby_candidate_count <= 2
    assert by_scenario["single_pixel_spike"].truth_count == 0
    assert by_scenario["single_pixel_spike"].nearby_candidate_count >= 1
    assert by_scenario["single_pixel_spike"].nearby_quality_count == 0
    assert by_scenario["long_line_negative"].truth_count == 0
    assert by_scenario["long_line_negative"].nearby_candidate_count >= 1
    assert "LINE_ARTIFACT" in by_scenario["long_line_negative"].nearby_flags

    output = write_feature_audit_artifacts(first, tmp_path / "feature-audit")
    assert (output / "feature_audit.csv").is_file()
    assert (output / "feature_audit.json").is_file()
    assert (output / "feature_audit_recall.png").is_file()


def test_sequence_feature_persistence_separates_anchor_categories(monkeypatch, tmp_path) -> None:
    def make_source(
        detection_id: int,
        x: float,
        y: float,
        *,
        quality: bool,
        flags: tuple[str, ...],
        proposal_methods: tuple[str, ...],
    ) -> Detection:
        return Detection(
            detection_id=detection_id,
            x=x,
            y=y,
            peak=121.0,
            flux=100.0 if quality else 20.0,
            background=21.0,
            noise=5.0,
            snr=20.0 if quality else 4.0,
            fwhm=2.0,
            flags=flags,
            flux_snr=20.0 if quality else 2.0,
            filter_snr=20.0 if quality else 4.0,
            fwhm_x=2.0,
            fwhm_y=2.0,
            ellipticity=0.1,
            sharpness=0.2,
            footprint_pixels=8,
            psf_support_pixels=5 if quality else 1,
            quality_passed=quality,
            peak_x=float(round(x)),
            peak_y=float(round(y)),
            proposal_methods=proposal_methods,
        )

    dog_only = make_source(
        99,
        70.0,
        70.0,
        quality=True,
        flags=(),
        proposal_methods=("dog_narrow",),
    )
    assert classify_source_proposal_subgroup(dog_only) == "dog_only_no_deblend"
    assert (
        classify_source_proposal_subgroup(
            replace(dog_only, deblend_delta_bic=12.0, deblend_component_snr=5.0)
        )
        == "dog_only_deblend"
    )
    assert classify_source_diagnostic_subgroup(
        make_source(
            3,
            75.0,
            75.0,
            quality=False,
            flags=("SPIKE", "INSUFFICIENT_PSF_SUPPORT"),
            proposal_methods=("dog_narrow",),
        )
    ) == "spike_and_psf_support"
    assert classify_source_diagnostic_subgroup(dog_only) is None

    fake_analyses = {}
    for frame_index in range(3):
        shift = frame_index * 0.2
        sources = (
            make_source(
                0,
                20.0 + shift,
                20.0 + shift,
                quality=True,
                flags=(),
                proposal_methods=("gaussian", "dog_narrow", "dog_broad"),
            ),
            make_source(
                1,
                40.0 + shift,
                40.0 + shift,
                quality=False,
                flags=("LOW_FLUX_SNR",),
                proposal_methods=("gaussian",),
            ),
            make_source(
                2,
                60.0 + shift,
                60.0 + shift,
                quality=False,
                flags=("SPIKE", "INSUFFICIENT_PSF_SUPPORT"),
                proposal_methods=("dog_narrow",),
            ),
        )
        detection = DetectionResult(
            image_shape=(96, 96),
            background=21.0,
            noise=5.0,
            threshold=41.0,
            candidate_count=len(sources),
            sources=sources,
            parameters={},
            quality_count=1,
        )
        path = tmp_path / f"frame-{frame_index}.fits"
        fake_analyses[path] = FrameAnalysis(
            frame=FitsFrame(
                path=path,
                header={},
                data=np.zeros((96, 96), dtype=np.float32),
                auxiliary=None,
                data_offset=0,
            ),
            detection=detection,
            matching=None,
            faintest=None,
        )

    def fake_analyze(path, **_kwargs):
        return fake_analyses[Path(path)]

    monkeypatch.setattr("rst19.experiments.analyze_frame", fake_analyze)
    progress: list[tuple[int, int]] = []
    result = run_sequence_feature_persistence(
        tuple(fake_analyses),
        association_radius_px=1.0,
        progress=lambda index, total: progress.append((index, total)),
    )

    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert result.required_presence == 3
    assert len(result.frame_rows) == 3 * 9
    by_class = {row.feature_class: row for row in result.persistence_rows}
    assert by_class["compact_quality"].anchor_candidate_count == 1
    assert by_class["compact_quality"].quality_presence_ge_required_count == 1
    assert by_class["weak_or_background"].anchor_candidate_count == 1
    assert by_class["weak_or_background"].quality_presence_ge_required_count == 0
    assert by_class["spike_or_support"].anchor_candidate_count == 1
    assert by_class["spike_or_support"].quality_presence_ge_required_count == 0
    method_rows = [
        row
        for row in result.proposal_method_rows
        if row.frame_index == 1
    ]
    by_method_class = {row.feature_class: row for row in method_rows}
    assert by_method_class["compact_quality"].all_three_count == 1
    assert by_method_class["weak_or_background"].gaussian_only_count == 1
    assert by_method_class["spike_or_support"].dog_only_count == 1
    method_profiles = summarize_sequence_feature_proposal_methods(result.proposal_method_rows)
    by_method_profile = {str(row["feature_class"]): row for row in method_profiles}
    assert by_method_profile["compact_quality"]["all_three_count"] == 3
    assert by_method_profile["crowded_blend"]["candidate_count_total"] == 0
    subgroup_rows = {row.proposal_subgroup: row for row in result.source_subgroup_rows}
    assert subgroup_rows["all_three"].anchor_count == 1
    assert subgroup_rows["all_three"].candidate_presence_ge_required_count == 1
    assert subgroup_rows["all_three"].quality_presence_ge_required_count == 1
    diagnostic_rows = {
        (row.feature_class, row.diagnostic_subgroup): row
        for row in result.diagnostic_subgroup_rows
    }
    assert diagnostic_rows[("weak_or_background", "weak_low_flux_snr")].anchor_count == 1
    assert (
        diagnostic_rows[("weak_or_background", "weak_low_flux_snr")]
        .candidate_same_subgroup_presence_ge_required_count
        == 1
    )
    assert (
        diagnostic_rows[("spike_or_support", "spike_and_psf_support")]
        .candidate_same_subgroup_presence_all_frames_count
        == 1
    )
    diagnostic_sources = {
        row.detection_id: row for row in result.diagnostic_source_rows
    }
    assert set(diagnostic_sources) == {1, 2}
    assert diagnostic_sources[2].candidate_presence == 3
    assert diagnostic_sources[2].candidate_same_subgroup_presence == 3

    temporal_profiles = summarize_sequence_feature_temporal_profiles(result.frame_rows)
    by_temporal_class = {row.feature_class: row for row in temporal_profiles}
    assert by_temporal_class["compact_quality"].frame_count == 3
    assert by_temporal_class["compact_quality"].candidate_count_mean == 1.0
    assert by_temporal_class["compact_quality"].quality_fraction_weighted == 1.0
    assert by_temporal_class["weak_or_background"].quality_fraction_weighted == 0.0

    transition_rows = result.class_transition_rows
    compact_candidate_self = next(
        row
        for row in transition_rows
        if row.layer == "candidate"
        and row.anchor_feature_class == "compact_quality"
        and row.response_feature_class == "compact_quality"
    )
    compact_quality_self = next(
        row
        for row in transition_rows
        if row.layer == "quality"
        and row.anchor_feature_class == "compact_quality"
        and row.response_feature_class == "compact_quality"
    )
    assert compact_candidate_self.matched_count == 3
    assert compact_candidate_self.possible_match_count == 3
    assert compact_candidate_self.match_rate == 1.0
    assert compact_quality_self.matched_count == 3
    assert compact_quality_self.match_rate == 1.0

    output = write_sequence_feature_persistence_artifacts(result, tmp_path / "sequence-feature")
    assert (output / "sequence_feature_persistence.csv").is_file()
    assert (output / "sequence_feature_frame_summary.csv").is_file()
    assert (output / "sequence_feature_persistence.json").is_file()
    assert (output / "sequence_feature_persistence.png").is_file()
    assert (output / "sequence_feature_temporal_profile.csv").is_file()
    assert (output / "sequence_feature_class_transition.csv").is_file()
    assert (output / "sequence_feature_method_frame_summary.csv").is_file()
    assert (output / "sequence_feature_method_profile.csv").is_file()
    assert (output / "sequence_feature_source_subgroup_persistence.csv").is_file()
    assert (output / "sequence_feature_diagnostic_subgroup_persistence.csv").is_file()
    assert (output / "sequence_feature_diagnostic_sources.csv").is_file()


def test_fixed_position_multipsf_audit_compares_model_order_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    rng = np.random.default_rng(19019)
    image = rng.normal(21.0, 1.0, (96, 96)).astype(np.float32)
    _gaussian_source(image, 48.0, 48.0, 120.0, 2.0)
    _gaussian_source(image, 53.0, 50.0, 70.0, 2.0)
    mask = np.zeros(image.shape, dtype=bool)
    positions = ((48.0, 48.0), (53.0, 50.0))
    single = _fixed_position_multi_psf_fit(
        image,
        mask,
        positions[:1],
        patch_positions=positions,
        psf_fwhm=2.0,
    )
    pair = _fixed_position_multi_psf_fit(
        image,
        mask,
        positions,
        patch_positions=positions,
        psf_fwhm=2.0,
    )
    assert single is not None and pair is not None
    assert float(pair["bic"]) < float(single["bic"])
    assert float(pair["amplitudes"][0]) > 0.0
    assert float(pair["amplitudes"][1]) > 0.0
    assert float(pair["component_snr"][1]) > 5.0

    frame_path = tmp_path / "multipsf.fits"
    frame = FitsFrame(path=frame_path, header={}, data=image, auxiliary=None, data_offset=0)
    monkeypatch.setattr("rst19.experiments.read_fits", lambda _path: frame)
    progress: list[tuple[int, int]] = []
    result = run_local_multipsf_audit(
        (frame_path,),
        positions,
        mask_modes=("raw",),
        progress=lambda index, total: progress.append((index, total)),
    )
    assert progress == [(1, 1)]
    assert len(result.rows) == 2
    assert result.rows[1].delta_bic_from_single is not None
    assert result.rows[1].delta_bic_from_single > 0.0
    output = write_local_multipsf_audit_artifacts(result, tmp_path / "multipsf-artifacts")
    assert (output / "local_multipsf_audit.csv").is_file()
    assert (output / "local_multipsf_audit.json").is_file()
    assert (output / "local_multipsf_bic.png").is_file()


def test_free_position_multipsf_audit_refines_centers_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    rng = np.random.default_rng(19020)
    image = rng.normal(21.0, 0.6, (96, 96)).astype(np.float32)
    _gaussian_source(image, 48.35, 48.25, 150.0, 2.0)
    _gaussian_source(image, 53.15, 50.20, 90.0, 2.0)
    mask = np.zeros(image.shape, dtype=bool)
    initial_positions = ((48.0, 48.0), (53.0, 50.0))
    fit = _free_position_multi_psf_fit(
        image,
        mask,
        initial_positions,
        patch_positions=initial_positions,
        psf_fwhm=2.0,
        position_radius_px=1.25,
    )
    assert fit is not None
    assert bool(fit["optimizer_success"])
    fitted_positions = tuple(fit["fitted_positions"])
    assert np.linalg.norm(np.asarray(fitted_positions[0]) - np.asarray((48.35, 48.25))) < 0.4
    assert np.linalg.norm(np.asarray(fitted_positions[1]) - np.asarray((53.15, 50.20))) < 0.4
    assert float(fit["component_snr"][1]) > 5.0

    frame_path = tmp_path / "free-multipsf.fits"
    frame = FitsFrame(path=frame_path, header={}, data=image, auxiliary=None, data_offset=0)
    monkeypatch.setattr("rst19.experiments.read_fits", lambda _path: frame)
    progress: list[tuple[int, int]] = []
    result = run_local_free_multipsf_audit(
        (frame_path,),
        initial_positions,
        mask_modes=("raw",),
        position_radius_px=1.25,
        progress=lambda index, total: progress.append((index, total)),
    )
    assert progress == [(1, 1)]
    assert len(result.rows) == 2
    assert result.rows[1].optimizer_success
    assert result.rows[1].fitted_positions != result.rows[1].initial_positions
    output = write_local_free_multipsf_audit_artifacts(result, tmp_path / "free-multipsf-artifacts")
    assert (output / "local_free_multipsf_audit.csv").is_file()
    assert (output / "local_free_multipsf_audit.json").is_file()
    assert (output / "local_free_multipsf_bic.png").is_file()


def test_free_position_multipsf_radius_sweep_reports_boundary_sensitivity(monkeypatch, tmp_path) -> None:
    image = np.full((64, 64), 21.0, dtype=np.float32)
    _gaussian_source(image, 30.25, 30.0, 100.0, 2.0)
    _gaussian_source(image, 35.0, 32.0, 60.0, 2.0)
    frame_path = tmp_path / "free-radius.fits"
    frame = FitsFrame(path=frame_path, header={}, data=image, auxiliary=None, data_offset=0)
    monkeypatch.setattr("rst19.experiments.read_fits", lambda _path: frame)
    progress: list[tuple[int, int]] = []
    result = run_local_free_multipsf_radius_sweep(
        (frame_path,),
        ((30.0, 30.0), (35.0, 32.0)),
        position_radii_px=(0.5, 1.0),
        mask_modes=("raw",),
        progress=lambda index, total: progress.append((index, total)),
    )
    assert progress == [(1, 2), (2, 2)]
    assert result.position_radii_px == (0.5, 1.0)
    assert len(result.rows) == 4
    output = write_local_free_multipsf_radius_sweep_artifacts(result, tmp_path / "free-radius-artifacts")
    assert (output / "local_free_multipsf_radius_sweep.csv").is_file()
    assert (output / "local_free_multipsf_radius_sweep.json").is_file()
    assert (output / "local_free_multipsf_radius_sweep.png").is_file()


def test_source_pair_audit_keeps_raw_range_evidence_separate_from_quality(monkeypatch, tmp_path) -> None:
    image_a = np.full((96, 96), 21, dtype=np.int16)
    image_a[30, 20] = 120
    image_a[32, 25] = 80
    image_a[31, 24] = -31_000
    image_a[32, 24] = -1
    image_b = image_a.copy()
    image_b[30, 20] = 118
    image_b[32, 25] = 81
    paths = (tmp_path / "a.fits", tmp_path / "b.fits")
    frames = {
        path: FitsFrame(path=path, header={}, data=image, auxiliary=None, data_offset=0)
        for path, image in zip(paths, (image_a, image_b), strict=True)
    }

    def make_source(detection_id: int, x: float, y: float, *, quality: bool, flags: tuple[str, ...]) -> Detection:
        return Detection(
            detection_id=detection_id,
            x=x,
            y=y,
            peak=120.0 if detection_id == 1 else 80.0,
            flux=1000.0 if quality else 350.0,
            background=21.0,
            noise=5.0,
            snr=20.0 if quality else 12.0,
            fwhm=2.0,
            flags=flags,
            flux_snr=20.0 if quality else 8.0,
            filter_snr=20.0 if quality else 8.0,
            fwhm_x=2.0,
            fwhm_y=2.0,
            ellipticity=0.1,
            sharpness=0.2,
            footprint_pixels=8,
            psf_support_pixels=5,
            quality_passed=quality,
            peak_x=x,
            peak_y=y,
        )

    def fake_analyze(frame, **_kwargs):
        sources = (
            make_source(1, 20.0, 30.0, quality=True, flags=()),
            make_source(2, 25.0, 32.0, quality=False, flags=("NEGATIVE_OVERFLOW",)),
        )
        detection = DetectionResult(
            image_shape=frame.data.shape,
            background=21.0,
            noise=5.0,
            threshold=41.0,
            candidate_count=2,
            sources=sources,
            parameters={},
            quality_count=1,
        )
        return FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=None)

    monkeypatch.setattr("rst19.experiments.read_fits", lambda path: frames[Path(path)])
    monkeypatch.setattr("rst19.experiments.analyze_frame", fake_analyze)
    progress: list[tuple[int, int]] = []
    result = run_source_pair_audit(
        paths,
        (20.0, 30.0),
        (25.0, 32.0),
        progress=lambda index, total: progress.append((index, total)),
    )

    assert progress == [(1, 2), (2, 2)]
    assert result.frame_count == 2
    assert result.target_peak_distance_px == np.hypot(5.0, 2.0)
    assert result.rows[0].secondary_detection_id == 2
    assert result.rows[0].secondary_quality_passed is False
    assert result.rows[0].secondary_negative_overflow_count >= 1
    assert result.rows[0].secondary_fixed_minus_one_count >= 1
    assert result.rows[0].nearest_secondary_detection_id == 2
    assert result.rows[0].nearest_source_same_for_targets is False
    assert result.rows[0].nearest_secondary_quality_passed is False
    assert result.rows[0].secondary_aperture_pixel_count > 0
    assert result.rows[0].secondary_shared_aperture_pixel_count > 0
    assert result.rows[0].secondary_aperture_raw_sum_adu is not None
    assert result.rows[0].secondary_shared_aperture_raw_sum_adu is not None
    assert result.rows[0].secondary_aperture_background_adu == 21.0
    assert "同一检测源的情况有 0/2 帧" in result.conclusion
    assert "detector" in result.association_coordinate_system
    output = write_source_pair_audit_artifacts(result, tmp_path / "pair-audit")
    assert (output / "source_pair_audit.csv").is_file()
    assert (output / "source_pair_audit.json").is_file()
    assert (output / "source_pair_raw_codes.png").is_file()
    assert (output / "source_pair_psf_evidence.png").is_file()
    header = (output / "source_pair_audit.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "secondary_shared_aperture_net_fraction" in header
    payload = json.loads((output / "source_pair_audit.json").read_text(encoding="utf-8"))
    assert "固定 detector 坐标" in payload["note"]

    def fake_analyze_primary_only(frame, **_kwargs):
        source = make_source(1, 20.0, 30.0, quality=True, flags=())
        detection = DetectionResult(
            image_shape=frame.data.shape,
            background=21.0,
            noise=5.0,
            threshold=41.0,
            candidate_count=1,
            sources=(source,),
            parameters={},
            quality_count=1,
        )
        return FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=None)

    monkeypatch.setattr("rst19.experiments.analyze_frame", fake_analyze_primary_only)
    same_source = run_source_pair_audit(
        paths[:1],
        (20.0, 30.0),
        (25.0, 32.0),
    )
    assert same_source.rows[0].secondary_detection_id is None
    assert same_source.rows[0].nearest_source_same_for_targets is True
    assert same_source.rows[0].pair_delta_bic_raw is None
    assert same_source.rows[0].pair_component_snr_raw is None
    assert "同一检测源的情况有 1/1 帧" in same_source.conclusion

    registered = run_source_pair_audit(
        paths[:1],
        (19.0, 30.0),
        (24.0, 32.0),
        frame_shifts=((1.0, 0.0),),
    )
    assert registered.rows[0].frame_shift_x_px == 1.0
    assert registered.rows[0].frame_shift_y_px == 0.0
    assert registered.rows[0].primary_detection_id == 1
    assert "registered detector" in registered.association_coordinate_system
    registered_output = write_source_pair_audit_artifacts(registered, tmp_path / "registered-pair-audit")
    registered_payload = json.loads(
        (registered_output / "source_pair_audit.json").read_text(encoding="utf-8")
    )
    assert "registered detector 坐标" in registered_payload["note"]


def test_detection_threshold_sweep_keeps_candidate_and_quality_layers() -> None:
    rng = np.random.default_rng(19019)
    image = rng.normal(21.0, 5.0, (128, 128)).astype(np.float32)
    _gaussian_source(image, 64.0, 64.0, 128.0, 3.0)
    frame = FitsFrame(path=__file__, header={}, data=image, auxiliary=None, data_offset=0)

    rows = run_detection_threshold_sweep(frame, threshold_levels=(4.0, 12.0), min_distance=4, background_box_size=32)

    assert [row.threshold_sigma for row in rows] == [4.0, 12.0]
    assert all(row.candidate_count >= row.quality_count for row in rows)
    assert all(row.rejected_count == row.returned_count - row.quality_count for row in rows)
    assert all(row.background_adu > 0.0 and row.noise_adu > 0.0 for row in rows)


def test_detection_psf_sweep_keeps_requested_order_and_layers() -> None:
    rng = np.random.default_rng(19019)
    image = rng.normal(21.0, 5.0, (128, 128)).astype(np.float32)
    _gaussian_source(image, 64.0, 64.0, 128.0, 3.0)
    frame = FitsFrame(path=__file__, header={}, data=image, auxiliary=None, data_offset=0)

    rows = run_detection_psf_sweep(frame, psf_levels=(2.0, 3.0), background_box_size=32)

    assert [row.psf_fwhm_px for row in rows] == [2.0, 3.0]
    assert all(row.candidate_count >= row.quality_count for row in rows)
    assert all(row.rejected_count == row.returned_count - row.quality_count for row in rows)


def test_real_background_injection_is_reproducible_and_keeps_background_semantics() -> None:
    rng = np.random.default_rng(19019)
    image = rng.normal(21.0, 5.0, (256, 256)).astype(np.float32)
    frame = FitsFrame(path=__file__, header={}, data=image, auxiliary=None, data_offset=0)

    progress: list[tuple[int, int]] = []
    first = run_real_background_injection(
        frame,
        peak_levels=(16.0, 80.0),
        trials_per_level=1,
        sources_per_trial=2,
        background_box_size=32,
        seed=19019,
        progress=lambda index, total: progress.append((index, total)),
    )
    second = run_real_background_injection(
        frame,
        peak_levels=(16.0, 80.0),
        trials_per_level=1,
        sources_per_trial=2,
        background_box_size=32,
        seed=19019,
    )

    assert first == second
    assert all(row.injected_count == 2 for row in first)
    assert all(row.baseline_candidate_count >= row.baseline_quality_count for row in first)
    assert all(row.mean_candidate_count >= row.mean_quality_count for row in first)
    assert first[1].candidate_recall >= first[0].candidate_recall
    assert first[1].quality_recall >= first[0].quality_recall
    assert all(row.mean_background_candidate_count >= 0.0 for row in first)
    assert progress == [(1, 2), (2, 2)]


def test_stratified_real_background_injection_separates_conditions_and_writes_artifacts(tmp_path) -> None:
    rng = np.random.default_rng(19019)
    frame = FitsFrame(
        path=tmp_path / "stratified.fits",
        header={},
        data=rng.normal(21.0, 5.0, (128, 128)).astype(np.float32),
        auxiliary=None,
        data_offset=0,
    )
    progress: list[tuple[int, int]] = []
    rows = run_stratified_real_background_injection(
        frame,
        strata=("blank", "high_background", "edge"),
        peak_levels=(80.0,),
        trials_per_level=1,
        sources_per_trial=1,
        psf_fwhm=2.0,
        background_box_size=32,
        seed=19019,
        progress=lambda index, total: progress.append((index, total)),
    )

    assert [row.stratum for row in rows] == ["blank", "high_background", "edge"]
    assert all(row.injected_count == 1 for row in rows)
    assert all(0.0 <= row.candidate_recall <= 1.0 for row in rows)
    assert all(0.0 <= row.quality_recall <= 1.0 for row in rows)
    assert rows[0].ambiguous_injection_count == 0
    assert rows[0].unambiguous_injected_count == 1
    assert rows[0].site_condition_json.startswith("[")
    assert rows[1].local_noise_adu >= rows[0].local_noise_adu
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]

    output = write_stratified_real_background_injection_artifacts(rows, tmp_path / "artifacts")
    assert (output / "stratified_real_background_injection.csv").is_file()
    assert (output / "stratified_real_background_injection.json").is_file()
    assert (output / "stratified_real_background_injection.png").is_file()


def test_stratified_real_background_injection_reuses_run_immutable_selection_state(monkeypatch) -> None:
    image = np.full((96, 96), 21.0, dtype=np.int16)
    frame = FitsFrame(path="stratified.fits", header={}, data=image, auxiliary=None, data_offset=0)
    baseline_source = Detection(
        detection_id=1,
        x=20.0,
        y=20.0,
        peak=120.0,
        flux=800.0,
        background=21.0,
        noise=5.0,
        snr=20.0,
        fwhm=2.0,
        flags=(),
        quality_passed=True,
    )
    baseline = DetectionResult(
        image_shape=image.shape,
        background=21.0,
        noise=5.0,
        threshold=41.0,
        candidate_count=1,
        sources=(baseline_source,),
        parameters={"saturation_level": -1.0, "mask_zero_pixels": 0},
        quality_count=1,
    )
    empty = DetectionResult(
        image_shape=image.shape,
        background=21.0,
        noise=5.0,
        threshold=41.0,
        candidate_count=0,
        sources=(),
        parameters={},
        quality_count=0,
    )
    detect_calls = 0
    injected_image_ids: list[int] = []

    def fake_detect(*_args, **_kwargs):
        nonlocal detect_calls
        detect_calls += 1
        if detect_calls > 1:
            injected_image_ids.append(id(_args[0]))
        return baseline if detect_calls == 1 else empty

    special_mask_calls = 0

    def fake_special_mask(values):
        nonlocal special_mask_calls
        special_mask_calls += 1
        return np.zeros(values.shape, dtype=bool)

    seen_selection_state: list[tuple[object, object]] = []
    site = _RealInjectionSite(
        x=70.0,
        y=70.0,
        local_background_adu=21.0,
        local_noise_adu=5.0,
        upper_excess_adu=0.0,
        special_pixel_fraction=0.0,
        baseline_neighbor_count=0,
        nearest_baseline_source_px=None,
    )

    def fake_select(*_args, **kwargs):
        seen_selection_state.append((kwargs["_special_mask"], kwargs["_baseline_tree"]))
        return (site,)

    monkeypatch.setattr("rst19.experiments.detect_sources", fake_detect)
    monkeypatch.setattr("rst19.experiments._real_injection_special_mask", fake_special_mask)
    monkeypatch.setattr("rst19.experiments._select_stratified_real_injection_sites", fake_select)

    rows = run_stratified_real_background_injection(
        frame,
        strata=("blank", "edge"),
        peak_levels=(80.0,),
        trials_per_level=1,
        sources_per_trial=1,
    )

    assert len(rows) == 2
    assert detect_calls == 3
    assert special_mask_calls == 1
    assert len(seen_selection_state) == 2
    assert seen_selection_state[0][0] is seen_selection_state[1][0]
    assert seen_selection_state[0][1] is seen_selection_state[1][1]
    assert len(set(injected_image_ids)) == 1
    assert seen_selection_state[0][0].flags.writeable is False
    assert np.array_equal(frame.data, image)


def test_local_patch_stats_integer_pixels_match_float64_reference() -> None:
    image = np.arange(81, dtype=np.int16).reshape(9, 9)
    result = _local_patch_stats(image, 4.0, 4.0, radius=2)

    patch = image[2:7, 2:7].astype(np.float64)
    median = float(np.median(patch))
    mad_noise = max(1.4826 * float(np.median(np.abs(patch - median))), np.finfo(np.float64).eps)
    upper_excess = float(np.quantile(patch, 0.98) - median)

    assert result == (median, mad_noise, upper_excess)


def test_real_injection_special_mask_integer_pixels_matches_reference() -> None:
    image = np.array([[-32768, -30000, -1, 0, 32700, 32767]], dtype=np.int16)
    numeric = image.astype(np.float64)
    expected = numeric == -1.0
    expected |= numeric <= -0.9 * float(np.iinfo(image.dtype).max)
    expected |= numeric >= float(np.iinfo(image.dtype).max - 32)

    assert np.array_equal(_real_injection_special_mask(image), expected)


def test_proposal_mode_comparison_uses_same_injection_population_and_writes_artifacts(tmp_path) -> None:
    rng = np.random.default_rng(19019)
    image = rng.normal(21.0, 5.0, (192, 192)).astype(np.float32)
    frame = FitsFrame(path=tmp_path / "same-background.fits", header={}, data=image, auxiliary=None, data_offset=0)

    first = run_proposal_mode_comparison(
        frame,
        peak_levels=(16.0, 80.0),
        trials_per_level=1,
        sources_per_trial=3,
        background_box_size=32,
        seed=19019,
    )
    second = run_proposal_mode_comparison(
        frame,
        peak_levels=(16.0, 80.0),
        trials_per_level=1,
        sources_per_trial=3,
        background_box_size=32,
        seed=19019,
    )

    assert first == second
    assert all(row.injected_count == 3 for row in first)
    assert all(0.0 <= row.gaussian_candidate_recall <= 1.0 for row in first)
    assert all(0.0 <= row.hybrid_candidate_recall <= 1.0 for row in first)
    assert all(0.0 <= row.ensemble_candidate_recall <= 1.0 for row in first)
    assert all(0.0 <= row.gaussian_quality_recall <= 1.0 for row in first)
    assert all(0.0 <= row.hybrid_quality_recall <= 1.0 for row in first)
    assert all(0.0 <= row.ensemble_quality_recall <= 1.0 for row in first)
    assert first[1].gaussian_candidate_recall >= first[0].gaussian_candidate_recall
    assert first[1].hybrid_candidate_recall >= first[0].hybrid_candidate_recall
    assert first[1].ensemble_candidate_recall >= first[0].ensemble_candidate_recall

    output = write_proposal_mode_comparison_artifacts(first, tmp_path / "comparison")
    assert (output / "proposal_mode_comparison.csv").is_file()
    assert (output / "proposal_mode_comparison.png").is_file()

    paired = run_proposal_mode_comparison(
        frame,
        peak_levels=(80.0,),
        trials_per_level=1,
        sources_per_trial=4,
        pair_separation_px=6.0,
        background_box_size=32,
        seed=19019,
    )
    assert paired[0].injected_count == 4
    assert paired[0].pair_separation_px == 6.0


def test_pair_flux_ratio_audit_keeps_total_peak_fixed_and_reports_resolution(tmp_path) -> None:
    rng = np.random.default_rng(19019)
    image = rng.normal(21.0, 5.0, (192, 192)).astype(np.float32)
    frame = FitsFrame(path=tmp_path / "pair-ratio.fits", header={}, data=image, auxiliary=None, data_offset=0)

    progress: list[tuple[int, int]] = []
    rows = run_pair_flux_ratio_audit(
        frame,
        total_peak_levels=(80.0,),
        secondary_to_primary_ratios=(1.0, 0.25),
        trials_per_condition=1,
        pairs_per_trial=1,
        pair_separation_px=5.4,
        proposal_mode="gaussian",
        background_box_size=32,
        progress=lambda index, total: progress.append((index, total)),
        seed=19019,
    )
    second = run_pair_flux_ratio_audit(
        frame,
        total_peak_levels=(80.0,),
        secondary_to_primary_ratios=(1.0, 0.25),
        trials_per_condition=1,
        pairs_per_trial=1,
        pair_separation_px=5.4,
        proposal_mode="gaussian",
        background_box_size=32,
        seed=19019,
    )

    assert rows == second
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert all(row.pair_count == 1 and row.injected_count == 2 for row in rows)
    assert all(row.primary_peak_excess_adu + row.secondary_peak_excess_adu == 80.0 for row in rows)
    assert rows[0].secondary_peak_excess_adu == 40.0
    assert rows[1].secondary_peak_excess_adu == 16.0
    assert all(0.0 <= row.pair_candidate_resolution_recall <= 1.0 for row in rows)
    assert all(0.0 <= row.pair_quality_resolution_recall <= 1.0 for row in rows)
    output = write_pair_flux_ratio_audit_artifacts(rows, tmp_path / "pair-ratio-artifacts")
    assert (output / "pair_flux_ratio_audit.csv").is_file()
    assert (output / "pair_flux_ratio_audit.json").is_file()
    assert (output / "pair_flux_ratio_audit.png").is_file()


def test_pair_flux_ratio_audit_can_fix_discrete_integrated_signal(tmp_path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    primary_peak = _inject_source_signal(
        image,
        20.25,
        20.5,
        64.0,
        signal_normalization="integrated_excess",
        psf_model="gaussian",
        gaussian_fwhm=3.0,
        empirical_psf=None,
    )
    secondary_peak = _inject_source_signal(
        image,
        44.75,
        43.5,
        16.0,
        signal_normalization="integrated_excess",
        psf_model="gaussian",
        gaussian_fwhm=3.0,
        empirical_psf=None,
    )

    assert primary_peak != secondary_peak
    assert np.isclose(float(np.sum(image)), 80.0, rtol=0.0, atol=1e-5)

    rng = np.random.default_rng(19019)
    background = rng.normal(21.0, 5.0, (192, 192)).astype(np.float32)
    frame = FitsFrame(
        path=tmp_path / "pair-integrated.fits",
        header={},
        data=background,
        auxiliary=None,
        data_offset=0,
    )
    rows = run_pair_flux_ratio_audit(
        frame,
        total_peak_levels=(80.0,),
        secondary_to_primary_ratios=(0.25,),
        trials_per_condition=1,
        pairs_per_trial=1,
        pair_separation_px=5.4,
        proposal_mode="gaussian",
        background_box_size=32,
        signal_normalization="integrated_excess",
        seed=19019,
    )

    assert rows[0].signal_normalization == "integrated_excess"
    assert rows[0].total_peak_excess_adu is None
    assert rows[0].total_control_signal_adu == 80.0
    assert rows[0].primary_control_signal_adu == 64.0
    assert rows[0].secondary_control_signal_adu == 16.0
    assert rows[0].mean_injected_primary_peak_excess_adu > 0.0
    assert rows[0].mean_injected_secondary_peak_excess_adu > 0.0


def test_crowded_blend_audit_is_reproducible_and_reports_group_resolution(tmp_path) -> None:
    rng = np.random.default_rng(19019)
    image = rng.normal(21.0, 5.0, (192, 192)).astype(np.float32)
    frame = FitsFrame(path=tmp_path / "crowded-blend.fits", header={}, data=image, auxiliary=None, data_offset=0)
    progress: list[tuple[int, int]] = []
    kwargs = dict(
        total_control_levels=(96.0,),
        group_sizes=(2, 3),
        separations_px=(3.0,),
        trials_per_condition=1,
        groups_per_trial=1,
        psf_fwhm=2.0,
        proposal_mode="gaussian",
        background_box_size=32,
        signal_normalization="integrated_excess",
        seed=19019,
    )
    rows = run_crowded_blend_audit(frame, progress=lambda index, total: progress.append((index, total)), **kwargs)
    second = run_crowded_blend_audit(frame, **kwargs)

    assert rows == second
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert {row.group_size for row in rows} == {2, 3}
    assert all(row.injected_count == row.group_size for row in rows)
    assert all(0.0 <= row.group_candidate_resolution_recall <= 1.0 for row in rows)
    assert all(0.0 <= row.merged_candidate_fraction <= 1.0 for row in rows)
    output = write_crowded_blend_audit_artifacts(rows, tmp_path / "crowded-blend-artifacts")
    assert (output / "crowded_blend_audit.csv").is_file()
    assert (output / "crowded_blend_audit.json").is_file()
    assert (output / "crowded_blend_audit.png").is_file()


def test_empirical_psf_is_built_from_isolated_quality_sources_and_can_be_injected() -> None:
    image = np.full((96, 96), 21.0, dtype=np.float32)
    positions = ((20.0, 20.0), (48.0, 20.0), (76.0, 20.0))
    sources = []
    for index, (x, y) in enumerate(positions):
        _gaussian_source(image, x, y, 128.0, 3.0)
        sources.append(
            Detection(
                detection_id=index,
                x=x,
                y=y,
                peak=149.0,
                flux=1000.0,
                background=21.0,
                noise=5.0,
                snr=25.0,
                fwhm=3.0,
                flags=(),
                flux_snr=50.0,
                ellipticity=0.1,
            )
        )

    psf = estimate_empirical_psf(image, sources, support_radius=7, max_sources=3)
    assert psf is not None
    assert psf.source_count == 3
    assert psf.kernel.shape == (15, 15)
    assert psf.as_dict()["kernel_sum"] > 0.0
    similarity = summarize_feature_psf_similarity(image, sources, psf=psf, per_class=3)
    compact_similarity = next(row for row in similarity if row["feature_class"] == "compact_quality")
    assert compact_similarity["valid_count"] == 3
    assert float(compact_similarity["psf_correlation_median"]) > 0.95
    spatial = summarize_feature_spatial_distribution(sources, image.shape, grid_size=2)
    compact_cell = next(
        row
        for row in spatial
        if row["feature_class"] == "compact_quality" and row["grid_row"] == 0 and row["grid_column"] == 0
    )
    assert compact_cell["candidate_count"] == 1
    assert compact_cell["quality_count"] == 1
    before = image.copy()
    _empirical_source(image, 48.5, 70.25, 64.0, psf)
    assert np.max(image - before) > 0.0


def test_spatial_psf_similarity_leaves_out_samples_and_uses_local_templates() -> None:
    image = np.full((104, 104), 21.0, dtype=np.float32)
    positions = ((16.0, 16.0), (52.0, 16.0), (88.0, 16.0), (16.0, 88.0), (52.0, 88.0), (88.0, 88.0))
    sources = []
    for index, (x, y) in enumerate(positions):
        _gaussian_source(image, x, y, 128.0, 3.0)
        sources.append(
            Detection(
                detection_id=index,
                x=x,
                y=y,
                peak=149.0,
                flux=1000.0,
                background=21.0,
                noise=5.0,
                snr=25.0,
                fwhm=3.0,
                flags=(),
                flux_snr=50.0,
                ellipticity=0.1,
                quality_passed=True,
            )
        )

    global_psf = estimate_empirical_psf(image, sources, support_radius=7, max_sources=6)
    assert global_psf is not None
    rows = summarize_spatial_psf_similarity(
        image,
        sources,
        global_psf=global_psf,
        per_class=3,
        support_radius=7,
        grid_size=1,
    )
    compact = next(row for row in rows if row["feature_class"] == "compact_quality")
    assert compact["sample_count"] == 3
    assert compact["global_valid_count"] == 3
    assert compact["local_template_available_cell_count"] == 1
    assert compact["local_template_source_count_median"] == 3.0
    assert compact["local_valid_count"] == 3
    assert compact["local_fallback_count"] == 0
    assert compact["paired_correlation_count"] == 3
    assert float(compact["local_correlation_median"]) > 0.95


def test_empirical_psf_excludes_rejected_and_partially_masked_sources() -> None:
    image = np.full((160, 160), 21.0, dtype=np.float32)
    positions = ((20.0, 20.0), (80.0, 20.0), (140.0, 20.0), (20.0, 100.0), (80.0, 100.0))
    sources = []
    for index, (x, y) in enumerate(positions):
        _gaussian_source(image, x, y, 128.0, 3.0)
        flags: tuple[str, ...] = () if index < 3 else ("LOW_FLUX_SNR",)
        if index == 4:
            flags = ("PARTIAL_MASKED",)
        sources.append(
            Detection(
                detection_id=index,
                x=x,
                y=y,
                peak=149.0,
                flux=1000.0,
                background=21.0,
                noise=5.0,
                snr=25.0,
                fwhm=3.0,
                flags=flags,
                flux_snr=50.0,
                ellipticity=0.1,
                quality_passed=not flags,
            )
        )

    psf = estimate_empirical_psf(image, sources, support_radius=7, max_sources=10)

    assert psf is not None
    assert psf.source_count == 3


def test_empirical_psf_isolation_sees_rejected_neighbors_when_full_pool_is_passed() -> None:
    image = np.full((160, 160), 21.0, dtype=np.float32)
    good_positions = ((30.0, 30.0), (100.0, 30.0), (30.0, 100.0))
    rejected_position = (45.0, 30.0)
    sources = []
    for index, (x, y) in enumerate((*good_positions, rejected_position)):
        _gaussian_source(image, x, y, 128.0, 3.0)
        flags: tuple[str, ...] = () if index < 3 else ("LOW_FLUX_SNR",)
        sources.append(
            Detection(
                detection_id=index,
                x=x,
                y=y,
                peak=149.0,
                flux=1000.0,
                background=21.0,
                noise=5.0,
                snr=25.0,
                fwhm=3.0,
                flags=flags,
                flux_snr=50.0,
                ellipticity=0.1,
                quality_passed=not flags,
            )
        )

    # The rejected neighbor is within the empirical PSF isolation radius of
    # the first good source. Passing only quality_sources would hide it.
    assert estimate_empirical_psf(image, sources, support_radius=7, max_sources=10) is None
    quality_only = estimate_empirical_psf(image, sources[:3], support_radius=7, max_sources=10)
    assert quality_only is not None
    assert quality_only.source_count == 3
    quality_only_with_full_isolation = estimate_empirical_psf(
        image,
        sources[:3],
        support_radius=7,
        max_sources=10,
        isolation_sources=sources,
    )
    assert quality_only_with_full_isolation is None


def test_real_background_empirical_injection_passes_full_detection_pool_to_psf_builder(monkeypatch, tmp_path) -> None:
    image = np.full((64, 64), 21.0, dtype=np.int16)
    frame = FitsFrame(path=tmp_path / "frame.fits", header={}, data=image, auxiliary=None, data_offset=0)

    def make_source(detection_id: int, *, quality: bool) -> Detection:
        flags: tuple[str, ...] = () if quality else ("LOW_FLUX_SNR",)
        return Detection(
            detection_id=detection_id,
            x=20.0 + detection_id * 4.0,
            y=20.0,
            peak=120.0,
            flux=800.0,
            background=21.0,
            noise=5.0,
            snr=20.0,
            fwhm=2.0,
            flags=flags,
            flux_snr=20.0,
            ellipticity=0.1,
            quality_passed=quality,
        )

    sources = (make_source(0, quality=True), make_source(1, quality=False))
    baseline = DetectionResult(
        image_shape=image.shape,
        background=21.0,
        noise=5.0,
        threshold=41.0,
        candidate_count=len(sources),
        sources=sources,
        parameters={"saturation_level": 32735.0, "mask_zero_pixels": 0},
        quality_count=1,
    )
    seen_pools: list[tuple[Detection, ...]] = []

    monkeypatch.setattr("rst19.experiments.detect_sources", lambda *_args, **_kwargs: baseline)
    monkeypatch.setattr(
        "rst19.experiments._select_real_background_positions",
        lambda *_args, **_kwargs: ([(50.0, 50.0)], [(21.0, 5.0)]),
    )

    def fake_empirical_psf(image, passed_sources, **_kwargs):
        seen_pools.append(tuple(passed_sources))
        return EmpiricalPSF(
            kernel=np.ones((7, 7), dtype=np.float32),
            support_radius=3,
            source_count=1,
            median_fwhm_px=2.0,
            median_ellipticity=0.1,
        )

    monkeypatch.setattr("rst19.experiments.estimate_empirical_psf", fake_empirical_psf)
    rows = run_real_background_injection(
        frame,
        peak_levels=(16.0,),
        trials_per_level=1,
        sources_per_trial=1,
        psf_model="empirical",
    )

    assert len(seen_pools) == 1
    assert seen_pools[0] == sources
    assert rows[0].psf_source_count == 1


def test_single_frame_trail_audit_keeps_geometry_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    frame = FitsFrame(
        path=tmp_path / "frame.fits",
        header={"DATE-OBS": "2026-03-30T16:32:05.000"},
        data=np.full((64, 64), 21.0, dtype=np.float32),
        auxiliary=None,
        data_offset=0,
    )
    detection = DetectionResult(
        image_shape=(64, 64),
        background=21.0,
        noise=5.0,
        threshold=41.0,
        candidate_count=12,
        sources=(),
        quality_count=5,
        parameters={},
    )
    analysis = FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=None)
    point = MotionFeaturePoint(
        frame_index=0,
        x=32.0,
        y=30.0,
        aligned_x=32.0,
        aligned_y=30.0,
        residual_snr=120.0,
        area_pixels=80,
        length_px=48.0,
        width_px=4.0,
        angle_deg=-35.0,
        bbox=(10, 5, 54, 55),
        touches_edge=False,
    )
    trail = MotionFeatureTrack(7, "candidate", (point,), 0.0, 0.0, None)
    monkeypatch.setattr("rst19.experiments.read_fits", lambda _path: frame)
    monkeypatch.setattr("rst19.experiments.analyze_frame", lambda _frame, **_kwargs: analysis)
    monkeypatch.setattr("rst19.experiments.detect_long_trails", lambda _analysis, **_kwargs: (trail,))

    progress: list[tuple[int, int]] = []
    rows = run_single_frame_trail_audit(
        [tmp_path / "a.fits", tmp_path / "b.fits"],
        progress=lambda index, total: progress.append((index, total)),
    )

    assert progress == [(1, 2), (2, 2)]
    assert [row.trail_count for row in rows] == [1, 1]
    assert rows[0].max_length_px == 48.0
    assert rows[0].longest_angle_deg == -35.0
    assert rows[0].longest_bbox == "10,5,54,55"
    output = write_single_frame_trail_artifacts(rows, tmp_path / "artifacts")
    assert (output / "single_frame_trails.csv").is_file()
    assert (output / "single_frame_trail_counts.png").is_file()
    assert (output / "single_frame_trail_lengths.png").is_file()


def test_source_artifacts_preserve_snr_layers_and_overlapping_flags(tmp_path) -> None:
    sources = (
        Detection(
            detection_id=1,
            x=12.5,
            y=18.5,
            peak=80.0,
            flux=120.0,
            background=21.0,
            noise=5.0,
            snr=11.8,
            fwhm=3.0,
            flags=(),
            flux_error=12.0,
            flux_snr=10.0,
            filter_snr=12.0,
            fwhm_x=3.0,
            fwhm_y=3.1,
            ellipticity=0.03,
            sharpness=0.3,
            footprint_pixels=12,
            quality_passed=True,
        ),
        Detection(
            detection_id=2,
            x=28.0,
            y=31.0,
            peak=30.0,
            flux=10.0,
            background=21.0,
            noise=5.0,
            snr=1.8,
            fwhm=0.5,
            flags=("LOW_FLUX_SNR", "NARROW"),
            flux_error=9.0,
            flux_snr=1.1,
            filter_snr=4.2,
            fwhm_x=0.5,
            fwhm_y=0.5,
            ellipticity=0.0,
            sharpness=0.9,
            footprint_pixels=1,
            quality_passed=False,
        ),
    )
    detection = DetectionResult(
        image_shape=(64, 64),
        background=21.0,
        noise=5.0,
        threshold=20.0,
        candidate_count=2,
        sources=sources,
        parameters={"threshold_sigma": 4.0, "min_distance": 4, "psf_fwhm": 3.0, "min_flux_snr": 5.0},
        quality_count=1,
    )

    frame = FitsFrame(
        path=tmp_path / "frame.fits",
        header={},
        data=np.full((64, 64), 21.0, dtype=np.float32),
        auxiliary=None,
        data_offset=0,
    )
    output = write_detection_source_artifacts(detection, tmp_path / "source-audit", frame=frame)

    source_csv = (output / "source_catalog.csv").read_text(encoding="utf-8-sig")
    summary_csv = (output / "source_quality_summary.csv").read_text(encoding="utf-8-sig")
    assert "quality_class" in source_csv
    assert "feature_class" in source_csv
    assert "repeated_code_count" in source_csv
    assert "range_anomaly_pixel_count" in source_csv
    assert "repeated_code_values" in source_csv
    assert "compact_quality" in source_csv
    assert "spike_or_support" in source_csv
    assert (output / "source_feature_spatial.csv").is_file()
    assert "LOW_FLUX_SNR|NARROW" in source_csv
    assert "flag:LOW_FLUX_SNR,1" in summary_csv
    assert "flag:NARROW,1" in summary_csv
    assert "quality_density_cv" in summary_csv
    assert "repeated_high_code_values" in summary_csv
    assert "close_pair_guarded_source_count" in summary_csv
    assert "feature:compact_quality:candidate_count" in summary_csv
    assert "feature:spike_or_support:candidate_count" in summary_csv
    assert (output / "source_snr_rank.png").is_file()
    assert (output / "source_flux_snr_distribution.png").is_file()
    assert (output / "source_quality_flags.png").is_file()
    assert (output / "source_feature_summary.csv").is_file()
    assert (output / "source_feature_method_summary.csv").is_file()
    assert "dog_only_count" in (output / "source_feature_method_summary.csv").read_text(encoding="utf-8-sig")
    assert (output / "source_feature_morphology.csv").is_file()
    assert (output / "source_feature_subclass_summary.csv").is_file()
    assert (output / "source_feature_psf_spatial.csv").is_file()
    assert (output / "source_feature_classes.png").is_file()
    assert (output / "source_spatial_density.png").is_file()
    spatial_lines = (output / "source_spatial_grid.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(spatial_lines) == 16 * 16 + 1
    assert "quality_fraction" in spatial_lines[0]

    expanded_output = write_detection_source_artifacts(
        detection,
        tmp_path / "source-audit-expanded",
        frame=frame,
        spatial_psf_per_class=16,
        spatial_psf_grid_size=2,
    )
    expanded_rows = list(
        csv.DictReader(
            (expanded_output / "source_feature_psf_spatial.csv").open(
                encoding="utf-8-sig",
                newline="",
            )
        )
    )
    assert expanded_rows
    assert {row["grid_size"] for row in expanded_rows} == {"2"}
    assert {row["sample_count"] for row in expanded_rows} == {"0", "1"}


def test_source_feature_classes_are_primary_and_keep_overlapping_flags() -> None:
    compact = Detection(
        detection_id=1,
        x=10.0,
        y=10.0,
        peak=80.0,
        flux=120.0,
        background=21.0,
        noise=5.0,
        snr=12.0,
        fwhm=2.0,
        flags=(),
        flux_snr=10.0,
        quality_passed=True,
    )
    contaminated = Detection(
        detection_id=2,
        x=20.0,
        y=20.0,
        peak=30000.0,
        flux=100.0,
        background=21.0,
        noise=5.0,
        snr=20.0,
        fwhm=3.0,
        flags=("NEGATIVE_OVERFLOW", "LOW_FLUX_SNR", "SPIKE"),
        flux_snr=2.0,
        quality_passed=False,
    )
    assert classify_source_feature(compact) == "compact_quality"
    assert classify_source_feature(contaminated) == "range_anomaly"
    rows = summarize_source_features((compact, contaminated))
    by_class = {str(row["feature_class"]): row for row in rows}
    assert by_class["compact_quality"]["candidate_count"] == 1
    assert by_class["compact_quality"]["quality_count"] == 1
    assert by_class["range_anomaly"]["candidate_count"] == 1
    assert by_class["range_anomaly"]["rejected_count"] == 1
    assert "NEGATIVE_OVERFLOW" in str(by_class["range_anomaly"]["common_flags"])


def test_source_proposal_method_summary_separates_detector_agreement() -> None:
    def source(
        detection_id: int,
        methods: tuple[str, ...],
        quality_passed: bool,
        feature_flag: tuple[str, ...] = (),
    ) -> Detection:
        return Detection(
            detection_id=detection_id,
            x=float(detection_id),
            y=float(detection_id),
            peak=80.0,
            flux=120.0,
            background=21.0,
            noise=5.0,
            snr=12.0,
            fwhm=2.0,
            flags=feature_flag,
            flux_snr=10.0,
            quality_passed=quality_passed,
            proposal_methods=methods,
        )

    rows = summarize_source_proposal_methods(
        (
            source(1, ("gaussian", "dog_narrow", "dog_broad"), True),
            source(2, ("gaussian",), False, ("LOW_FLUX_SNR",)),
            source(3, ("dog_narrow",), False, ("UNRESOLVED_BLEND",)),
        )
    )
    by_class = {str(row["feature_class"]): row for row in rows}
    assert by_class["compact_quality"]["all_three_count"] == 1
    assert by_class["compact_quality"]["quality_count"] == 1
    assert by_class["weak_or_background"]["gaussian_only_count"] == 1
    assert by_class["crowded_blend"]["dog_only_count"] == 1
    assert by_class["crowded_blend"]["no_gaussian_quality_count"] == 0


def test_feature_morphology_summary_separates_snr_from_shape_and_flags() -> None:
    compact = Detection(
        detection_id=1,
        x=10.0,
        y=20.0,
        peak=80.0,
        flux=120.0,
        background=21.0,
        noise=5.0,
        snr=12.0,
        fwhm=2.0,
        flags=(),
        flux_snr=10.0,
        filter_snr=14.0,
        ellipticity=0.05,
        sharpness=0.2,
        footprint_pixels=9,
        psf_support_pixels=6,
        centroid_shift_px=0.15,
        quality_passed=True,
    )
    rejected_high_snr = Detection(
        detection_id=2,
        x=30.0,
        y=40.0,
        peak=30000.0,
        flux=900.0,
        background=21.0,
        noise=5.0,
        snr=20.0,
        fwhm=3.0,
        flags=("NEGATIVE_OVERFLOW", "LOW_FLUX_SNR", "CODE_PATTERN"),
        flux_snr=20.0,
        filter_snr=25.0,
        ellipticity=0.4,
        sharpness=0.8,
        footprint_pixels=2,
        psf_support_pixels=2,
        centroid_shift_px=1.2,
        quality_passed=False,
    )
    code_pattern_only = Detection(
        detection_id=3,
        x=50.0,
        y=60.0,
        peak=500.0,
        flux=300.0,
        background=21.0,
        noise=5.0,
        snr=11.0,
        fwhm=2.5,
        flags=("CODE_PATTERN",),
        flux_snr=11.0,
        quality_passed=False,
    )

    rows = summarize_feature_morphology((compact, rejected_high_snr, code_pattern_only))
    by_class = {str(row["feature_class"]): row for row in rows}
    range_row = by_class["range_anomaly"]
    compact_row = by_class["compact_quality"]
    assert range_row["candidate_count"] == 2
    assert range_row["high_flux_snr_count"] == 2
    assert range_row["high_flux_snr_rejected_count"] == 2
    assert range_row["high_flux_snr_rejection_fraction"] == 1.0
    assert range_row["range_anomaly_fraction"] == 1.0
    assert compact_row["quality_fraction"] == 1.0
    assert compact_row["fwhm_px_median"] == 2.0
    assert compact_row["x_px_median"] == 10.0
    assert compact_row["y_px_median"] == 20.0


def test_source_feature_subclasses_keep_mask_tokens_separate() -> None:
    def source(detection_id: int, flags: tuple[str, ...], quality_passed: bool, flux_snr: float) -> Detection:
        return Detection(
            detection_id=detection_id,
            x=float(detection_id),
            y=float(detection_id),
            peak=100.0,
            flux=200.0,
            background=21.0,
            noise=5.0,
            snr=20.0,
            fwhm=2.0,
            flags=flags,
            flux_snr=flux_snr,
            quality_passed=quality_passed,
        )

    rows = summarize_source_feature_subclasses(
        (
            source(1, ("EDGE",), False, 4.0),
            source(2, ("PARTIAL_MASKED",), True, 6.0),
            source(3, ("EDGE", "PARTIAL_MASKED"), False, 8.0),
            source(4, ("CODE_PATTERN", "NEGATIVE_OVERFLOW"), False, 20.0),
            source(5, ("EDGE", "MASKED"), False, 3.0),
        )
    )
    by_subclass = {str(row["subclass"]): row for row in rows}
    assert by_subclass["edge_only"]["candidate_count"] == 1
    assert by_subclass["edge_only"]["quality_count"] == 0
    assert by_subclass["partial_masked_only"]["candidate_count"] == 1
    assert by_subclass["partial_masked_only"]["quality_count"] == 1
    assert by_subclass["edge_partial_masked"]["candidate_count"] == 1
    assert by_subclass["edge_hard_masked"]["candidate_count"] == 1
    assert by_subclass["range_code_pattern"]["candidate_count"] == 1
    assert by_subclass["range_negative_overflow"]["candidate_count"] == 1
    assert by_subclass["range_code_pattern"]["high_snr_rejected_count"] == 1


def test_source_cutout_audit_writes_stratified_manifest_and_contact_sheet(tmp_path) -> None:
    frame = FitsFrame(
        path=tmp_path / "frame.fits",
        header={},
        data=np.full((96, 96), 21.0, dtype=np.float32),
        auxiliary=None,
        data_offset=0,
    )
    sources = (
        Detection(
            detection_id=1,
            x=20.2,
            y=30.1,
            peak=80.0,
            flux=120.0,
            background=21.0,
            noise=5.0,
            snr=11.8,
            fwhm=3.0,
            flags=(),
            flux_snr=5.4,
            peak_x=20.0,
            peak_y=30.0,
            centroid_shift_px=0.224,
            quality_passed=True,
        ),
        Detection(
            detection_id=2,
            x=70.0,
            y=40.0,
            peak=30.0,
            flux=10.0,
            background=21.0,
            noise=5.0,
            snr=1.8,
            fwhm=0.5,
            flags=("LOW_FLUX_SNR",),
            flux_snr=1.1,
            peak_x=70.0,
            peak_y=40.0,
            centroid_shift_px=0.0,
            quality_passed=False,
        ),
    )
    detection = DetectionResult(
        image_shape=(96, 96),
        background=21.0,
        noise=5.0,
        threshold=41.0,
        candidate_count=2,
        sources=sources,
        parameters={},
        quality_count=1,
    )

    output = write_detection_source_cutouts(frame, detection, tmp_path / "cutouts", per_group=2, radius_px=8, scale=2)

    manifest = (output / "source_cutout_manifest.csv").read_text(encoding="utf-8-sig")
    assert "quality_near_threshold" in manifest
    assert "rejected_low_flux_snr" in manifest
    assert "feature_compact_quality" in manifest
    assert "feature_weak_or_background" in manifest
    assert "peak_x" in manifest
    assert (output / "source_cutout_contact_sheet.png").is_file()
    assert len(list((output / "source_cutouts").glob("*.png"))) == 4


def test_temporal_code_audit_reports_low_variation_components_and_writes_outputs(tmp_path, monkeypatch) -> None:
    frame_paths = (tmp_path / "frame-1.fits", tmp_path / "frame-2.fits")
    image_a = np.zeros((8, 8), dtype=np.int16)
    image_b = image_a.copy()
    image_a[2, 3] = -1
    image_b[2, 3] = -1
    image_a[3, 5] = 3991
    image_b[3, 5] = 3992
    image_a[4, 5] = 3991
    image_b[4, 5] = 3993
    image_a[6, 6] = 20
    image_b[6, 6] = 21
    frames = {
        frame_paths[0]: FitsFrame(frame_paths[0], {}, image_a, None, 0),
        frame_paths[1]: FitsFrame(frame_paths[1], {}, image_b, None, 0),
    }
    monkeypatch.setattr("rst19.experiments.read_fits", lambda path: frames[Path(path)])
    progress: list[tuple[int, int]] = []

    result = run_temporal_code_audit(
        frame_paths,
        code_focus_xy=(5, 3),
        sentinel_focus_xy=(3, 2),
        progress=lambda index, total: progress.append((index, total)),
    )

    assert progress == [(1, 2), (2, 2)]
    assert result.frame_count == 2
    assert result.exact_stable_pixel_count == 61
    assert result.sentinel_exact_stable_pixel_count == 1
    assert result.sentinel_focus_component_size == 1
    assert result.code_pixel_count == 2
    assert result.code_component_count == 1
    assert result.code_components_ge_2 == 1
    assert result.code_focus_component_size == 2
    assert result.code_focus_component_coordinates == ((5, 3), (5, 4))
    assert result.code_focus_values_by_frame == (3991, 3992)
    assert result.sentinel_focus_values_by_frame == (-1, -1)

    output = write_temporal_code_audit_artifacts(result, tmp_path / "temporal-code-audit")
    assert (output / "temporal_code_audit.json").is_file()
    focus_series = (output / "temporal_code_focus_series.csv").read_text(encoding="utf-8-sig")
    assert "code_value" in focus_series
    assert "3991" in focus_series
    components = (output / "temporal_code_components.csv").read_text(encoding="utf-8-sig")
    assert "low_variation_code" in components
    assert "exact_stable_sentinel" in components


def test_feature_cross_audit_reproduces_snr_and_persistence_counterexamples(tmp_path) -> None:
    source = tmp_path / "source_feature_summary.csv"
    persistence = tmp_path / "sequence_feature_persistence.csv"
    transitions = tmp_path / "sequence_feature_class_transition.csv"
    source.write_text(
        "feature_class,feature_class_label,candidate_count,quality_count,rejected_count,quality_fraction,median_flux_snr,max_flux_snr,common_flags\n"
        "compact_quality,紧凑质量候选,10,10,0,1.0,12.3,100.0,\n"
        "spike_or_support,尖峰/PSF支持不足,10,0,10,0.0,1.9,133.8,SPIKE\n",
        encoding="utf-8-sig",
    )
    persistence.write_text(
        "feature_class,feature_class_label,anchor_candidate_count,anchor_quality_count,frame_count,required_presence,association_radius_px,candidate_median_presence,candidate_mean_presence,candidate_presence_ge_required_count,candidate_presence_all_frames_count,quality_median_presence,quality_mean_presence,quality_presence_ge_required_count,quality_presence_all_frames_count\n"
        "compact_quality,紧凑质量候选,10,10,15,12,1.0,15.0,14.0,9,8,15.0,13.0,8,7\n"
        "spike_or_support,尖峰/PSF支持不足,10,0,15,12,1.0,15.0,12.0,8,6,0.0,0.2,0,0\n",
        encoding="utf-8-sig",
    )
    transitions.write_text(
        "layer,anchor_feature_class,anchor_feature_class_label,response_feature_class,response_feature_class_label,anchor_count,frame_count,possible_match_count,matched_count,match_rate,mean_matches_per_anchor\n"
        "candidate,compact_quality,紧凑质量候选,compact_quality,紧凑质量候选,10,15,150,9,0.06,0.9\n"
        "candidate,compact_quality,紧凑质量候选,spike_or_support,尖峰/PSF支持不足,10,15,150,3,0.02,0.3\n"
        "candidate,spike_or_support,尖峰/PSF支持不足,compact_quality,紧凑质量候选,10,15,150,2,0.013333333333333334,0.2\n"
        "candidate,spike_or_support,尖峰/PSF支持不足,spike_or_support,尖峰/PSF支持不足,10,15,150,6,0.04,0.6\n"
        "quality,compact_quality,紧凑质量候选,compact_quality,紧凑质量候选,10,15,150,8,0.05333333333333334,0.8\n"
        "quality,compact_quality,紧凑质量候选,spike_or_support,尖峰/PSF支持不足,10,15,150,0,0.0,0.0\n"
        "quality,spike_or_support,尖峰/PSF支持不足,compact_quality,紧凑质量候选,0,15,0,0,,0.0\n"
        "quality,spike_or_support,尖峰/PSF支持不足,spike_or_support,尖峰/PSF支持不足,0,15,0,0,,0.0\n",
        encoding="utf-8-sig",
    )

    result = run_feature_cross_audit(source, persistence, class_transition_path=transitions)

    assert result.frame_count == 15
    assert result.required_presence == 12
    by_class = {row.feature_class: row for row in result.rows}
    assert by_class["compact_quality"].candidate_presence_fraction == 0.9
    assert by_class["compact_quality"].quality_presence_fraction == 0.8
    assert by_class["spike_or_support"].high_snr_rejected is True
    assert by_class["spike_or_support"].persistence_gap == 0.8
    assert by_class["compact_quality"].candidate_any_response_fraction == 0.08
    assert by_class["compact_quality"].candidate_same_class_response_fraction == 0.06
    assert by_class["compact_quality"].quality_same_class_response_fraction == 8 / 150
    assert by_class["compact_quality"].dominant_cross_class == "spike_or_support"
    assert "spike_or_support" in result.conclusion

    output = write_feature_cross_audit_artifacts(result, tmp_path / "feature-cross")
    assert (output / "feature_cross_audit.csv").is_file()
    assert (output / "feature_cross_audit.json").is_file()
    assert (output / "feature_cross_audit.png").is_file()
    assert "candidate_same_class_response_fraction" in (output / "feature_cross_audit.csv").read_text(encoding="utf-8-sig")


def test_sequence_source_audit_writes_missing_frame_cues_and_summary(tmp_path, monkeypatch) -> None:
    frame_paths = tuple(tmp_path / f"frame-{index}.fits" for index in range(4))
    frames = tuple(
        FitsFrame(path, {}, np.full((48, 48), 20.0 + index, dtype=np.float32), None, 0)
        for index, path in enumerate(frame_paths)
    )
    for path in frame_paths:
        path.write_bytes(b"placeholder")
    by_path = {frame.path: frame for frame in frames}
    monkeypatch.setattr("rst19.experiments.read_fits", lambda path: by_path[Path(path)])

    def point(frame_index: int, x: float = 20.0, y: float = 21.0, snr: float = 8.0) -> dict[str, object]:
        return {
            "frame_index": frame_index,
            "x": x,
            "y": y,
            "aligned_x": x,
            "aligned_y": y,
            "flux_snr": snr,
        }

    payload = {
        "frames": [{"path": str(path)} for path in frame_paths],
        "cumulative_shifts": [[0.0, 0.0]] * 4,
        "persistent_min_presence": 2,
        "tracks": [
            {"track_id": 1, "classification": "static", "displacement_px": 0.1, "fit_rms_px": 0.1, "points": [point(i) for i in range(4)]},
            {"track_id": 2, "classification": "persistent", "displacement_px": 0.2, "fit_rms_px": 0.2, "points": [point(0), point(1)]},
            {"track_id": 3, "classification": "transient", "displacement_px": 0.0, "fit_rms_px": None, "points": [point(2)]},
        ],
    }

    output = write_sequence_source_audit(payload, tmp_path / "sequence-audit", per_group=1, radius_px=6, scale=1)

    summary = (output / "source_track_audit_summary.json").read_text(encoding="utf-8")
    manifest = (output / "source_track_audit.csv").read_text(encoding="utf-8-sig")
    assert '"strict_static_count": 1' in summary
    assert '"persistent_count": 1' in summary
    assert '"sampled_track_count": 3' in summary
    assert "static_strict" in manifest
    assert "persistent" in manifest
    assert "transient_one_frame" in manifest
    assert (output / "source_track_contact_sheet.png").is_file()
    assert len(list((output / "source_track_cutouts").glob("*.png"))) == 3
