"""Freeze Python single-frame sequence stitching cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import MotionFeaturePoint, _stitch_unassociated_single_frame_points


def point(
    frame: int,
    x: float,
    y: float = 0.0,
    *,
    angle: float = 0.0,
    length: float = 12.0,
    model: str | None = "line",
    snr: float | None = 7.0,
    background_delta: float | None = 3.0,
    compact_delta: float | None = 3.0,
    continuity: float | None = 0.6,
    fill: float | None = 0.6,
    width: float = 2.0,
    artifact_overlap: float | None = None,
) -> MotionFeaturePoint:
    return MotionFeaturePoint(
        frame_index=frame,
        x=x,
        y=y,
        aligned_x=x,
        aligned_y=y,
        residual_snr=20.0,
        area_pixels=20,
        length_px=length,
        width_px=width,
        angle_deg=angle,
        bbox=(0, 0, 8, 8),
        touches_edge=False,
        continuity_ratio=continuity,
        profile_fill_ratio=fill,
        line_artifact_overlap_fraction=artifact_overlap,
        line_template_snr=snr,
        line_vs_point_delta_bic=compact_delta,
        line_vs_double_delta_bic=compact_delta,
        line_vs_background_delta_bic=background_delta,
        line_vs_elliptical_delta_bic=compact_delta,
        single_frame_model=model,
    )


def capture(
    label: str,
    tracks: list[list[MotionFeaturePoint]],
    points: list[MotionFeaturePoint],
    *,
    psf_fwhm: float = 2.0,
    frame_times: list[float] | list[str] | list[bool] | None = None,
    max_frame_gap: int | float = 3,
    max_prediction_error_px: float | None = None,
) -> dict[str, object]:
    if frame_times is None:
        highest = max(
            [item.frame_index for history in tracks for item in history]
            + [item.frame_index for item in points]
            + [0]
        )
        frame_times = [float(index) for index in range(highest + 1)]
    resolved_tracks, remaining = _stitch_unassociated_single_frame_points(
        tracks,
        points,
        psf_fwhm=psf_fwhm,
        frame_times=frame_times,
        max_frame_gap=max_frame_gap,
        max_prediction_error_px=max_prediction_error_px,
    )
    request: dict[str, object] = {
        "tracks": [[item.as_dict() for item in history] for history in tracks],
        "points": [item.as_dict() for item in points],
        "psf_fwhm": psf_fwhm,
        "frame_times": frame_times,
        "max_frame_gap": max_frame_gap,
        "max_prediction_error_px": max_prediction_error_px,
    }
    return {
        "label": label,
        "request": request,
        "expected": {
            "tracks": [[item.as_dict() for item in history] for history in resolved_tracks],
            "remaining_points": [item.as_dict() for item in remaining],
        },
    }


def main() -> None:
    chain = [point(5, 50), point(7, 70), point(8, 80)]
    linear = [point(4, 40), point(5, 50), point(6, 60)]
    cases = [
        capture("later_singletons_extend_prediction_chain", [chain], [point(2, 20), point(4, 40)]),
        capture("single_candidate_attaches_and_history_is_sorted", [linear], [point(3, 30)]),
        capture("track_with_fewer_than_three_points_is_ignored", [[point(4, 40), point(5, 50)]], [point(3, 30)]),
        capture("compact_point_model_is_not_stitched", [linear], [point(3, 30, model="point")]),
        capture("artifact_overlap_at_limit_is_not_stitched", [linear], [point(3, 30, artifact_overlap=0.75)]),
        capture("artifact_overlap_below_limit_can_stitch", [linear], [point(3, 30, artifact_overlap=0.749)]),
        capture("angle_limit_is_inclusive", [linear], [point(3, 30, angle=35.0)]),
        capture("angle_over_limit_is_rejected", [linear], [point(3, 30, angle=35.001)]),
        capture("prediction_error_at_limit_is_inclusive", [linear], [point(3, 38)], max_prediction_error_px=8.0),
        capture("prediction_error_over_limit_is_rejected", [linear], [point(3, 39)], max_prediction_error_px=8.0),
        capture("nearest_gap_limit_is_inclusive", [chain], [point(2, 20)]),
        capture("nearest_gap_over_limit_is_rejected", [chain], [point(1, 10)]),
        capture("point_in_existing_track_frame_is_rejected", [linear], [point(5, 50)]),
        capture("zero_velocity_track_cannot_predict", [[point(0, 40), point(1, 40), point(2, 40)]], [point(3, 40)]),
        capture("invalid_times_fall_back_to_frame_indices", [linear], [point(3, 30)], frame_times=[0.0, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0]),
        capture("irregular_times_drive_prediction", [[point(2, 25), point(4, 70), point(5, 100)]], [point(1, 10)], frame_times=[0, 2, 5, 9, 14, 20]),
        capture("minimum_length_ratio_is_inclusive", [[point(4, 40, length=20), point(5, 50, length=20), point(6, 60, length=20)]], [point(3, 30, length=7)]),
        capture("length_ratio_below_minimum_is_rejected", [[point(4, 40, length=20), point(5, 50, length=20), point(6, 60, length=20)]], [point(3, 30, length=6.99)]),
        capture("maximum_length_ratio_is_inclusive", [[point(4, 40, length=12), point(5, 50, length=12), point(6, 60, length=12)]], [point(3, 30, length=30)]),
        capture("length_ratio_over_maximum_is_rejected", [linear], [point(3, 30, length=30.01)]),
        capture("long_line_profile_uses_background_evidence", [linear], [point(3, 30, model="long_line_profile", compact_delta=None)]),
        capture("thin_elliptical_line_review_exception_stitches", [linear], [point(3, 30, model="elliptical_point", width=0.75, continuity=0.8, fill=0.8)]),
        capture("wide_elliptical_point_is_rejected", [linear], [point(3, 30, model="elliptical_point", width=1.01, continuity=0.8, fill=0.8)]),
        capture("missing_required_model_evidence_is_rejected", [linear], [point(3, 30, snr=None)]),
        capture("out_of_range_point_frame_is_rejected", [linear], [point(12, 120)], frame_times=[float(index) for index in range(10)]),
        capture("closest_prediction_wins_between_tracks", [linear, [point(4, 39), point(5, 49), point(6, 59)]], [point(3, 30.5)]),
        capture("configured_one_frame_gap_is_respected", [chain], [point(2, 20)], max_frame_gap=1),
        capture("floating_gap_limit_is_truncated_like_python_int", [chain], [point(4, 40)], max_frame_gap=1.9),
        capture("positive_fractional_gap_can_match_no_integer_frame", [chain], [point(4, 40)], max_frame_gap=0.9),
        capture("default_prediction_tolerance_scales_with_psf", [linear], [point(3, 39)], psf_fwhm=4.0),
        capture("empty_singleton_input_still_sorts_track", [[point(6, 60), point(4, 40), point(5, 50)]], []),
        capture("numeric_string_times_are_converted", [linear], [point(3, 30)], frame_times=["0", "1", "2", "3", "4", "5", "6"]),
        capture("boolean_times_fall_back_when_not_monotonic", [linear], [point(3, 30)], frame_times=[False, True, False, True, True, True, True]),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/single_frame_stitch.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
