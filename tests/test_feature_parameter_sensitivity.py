from __future__ import annotations

import json

from rst19.feature_parameter_sensitivity import (
    FeatureParameterProfileRow,
    FeatureParameterRunRow,
    FeatureParameterSensitivityResult,
    FeatureParameterTransitionRow,
    _reciprocal_match_indices,
    write_feature_parameter_sensitivity_artifacts,
)
from rst19.detection import Detection


def _source(index: int, x: float, y: float, *, quality: bool = True) -> Detection:
    return Detection(
        detection_id=index,
        x=x,
        y=y,
        peak=100.0,
        flux=200.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=2.0,
        flags=(),
        flux_snr=10.0,
        filter_snr=12.0,
        quality_passed=quality,
        peak_x=x,
        peak_y=y,
    )


def test_reciprocal_match_is_one_to_one_and_radius_limited() -> None:
    anchor = (_source(1, 10.0, 10.0), _source(2, 30.0, 30.0))
    current = (_source(3, 10.4, 10.2), _source(4, 60.0, 60.0))
    assert _reciprocal_match_indices(anchor, current, radius_px=1.0) == {0: 0}


def test_writer_preserves_separate_run_and_profile_layers(tmp_path) -> None:
    run = FeatureParameterRunRow(
        source_path="frame.fits",
        parameter_family="threshold_sigma",
        parameter_value=4.0,
        threshold_sigma=4.0,
        min_flux_snr=5.0,
        candidate_count=10,
        returned_count=10,
        quality_count=7,
        rejected_count=3,
        background_adu=12.0,
        noise_adu=2.0,
    )
    profile = FeatureParameterProfileRow(
        source_path="frame.fits",
        parameter_family="threshold_sigma",
        parameter_value=4.0,
        threshold_sigma=4.0,
        min_flux_snr=5.0,
        feature_class="compact_quality",
        baseline_candidate_anchor_count=10,
        current_candidate_count=10,
        current_quality_count=7,
        current_quality_fraction=0.7,
        baseline_candidate_matched_count=10,
        baseline_same_class_matched_count=10,
        baseline_candidate_match_fraction=1.0,
        baseline_same_class_match_fraction=1.0,
        baseline_quality_anchor_count=7,
        baseline_quality_matched_count=7,
        baseline_quality_still_passed_count=7,
        baseline_quality_match_fraction=1.0,
        baseline_quality_still_passed_fraction=1.0,
        note="diagnostic",
    )
    transition = FeatureParameterTransitionRow(
        source_path="frame.fits",
        parameter_family="threshold_sigma",
        parameter_value=4.0,
        threshold_sigma=4.0,
        min_flux_snr=5.0,
        baseline_feature_class="compact_quality",
        current_feature_class="compact_quality",
        matched_count=10,
        current_quality_count=7,
    )
    result = FeatureParameterSensitivityResult(
        source_path="frame.fits",
        image_shape=(32, 32),
        baseline_threshold_sigma=4.0,
        baseline_min_flux_snr=5.0,
        association_radius_px=1.5,
        detector_parameters={"proposal_mode": "hybrid"},
        requested_threshold_levels=(4.0,),
        requested_flux_snr_levels=(5.0,),
        runs=(run,),
        profiles=(profile,),
        transitions=(transition,),
        observations={"interpretation": "diagnostic"},
    )
    output = write_feature_parameter_sensitivity_artifacts(result, tmp_path)
    assert (output / "feature_parameter_runs.csv").is_file()
    assert (output / "feature_parameter_profiles.csv").is_file()
    assert (output / "feature_parameter_transitions.csv").is_file()
    payload = json.loads((output / "feature_parameter_sensitivity.json").read_text(encoding="utf-8"))
    assert payload["runs"][0]["quality_count"] == 7
    assert payload["profiles"][0]["feature_class"] == "compact_quality"
    assert payload["transitions"][0]["matched_count"] == 10
