"""Freeze Python collinear motion-feature rescue cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import MotionFeaturePoint, _rescue_collinear_motion_feature_groups


def point(frame: int, x: float, y: float, *, angle: float = 0.0, length: float = 40.0) -> MotionFeaturePoint:
    return MotionFeaturePoint(
        frame_index=frame,
        x=x,
        y=y,
        aligned_x=x,
        aligned_y=y,
        residual_snr=25.0,
        area_pixels=50,
        length_px=length,
        width_px=3.0,
        angle_deg=angle,
        bbox=(int(x - length / 2), int(y - 2), int(x + length / 2), int(y + 2)),
        touches_edge=False,
        continuity_ratio=0.9,
        profile_fill_ratio=0.9,
    )


def capture(
    label: str,
    tracks: list[list[MotionFeaturePoint]],
    *,
    psf_fwhm: float = 2.0,
    min_presence: int = 3,
    angle_tolerance_deg: float = 12.0,
    line_tolerance_px: float = 12.0,
    min_span_px: float = 64.0,
    frame_times: list[float] | None = None,
    reference_interval_s: float | None = None,
) -> dict[str, object]:
    groups = _rescue_collinear_motion_feature_groups(
        tracks,
        psf_fwhm=psf_fwhm,
        min_presence=min_presence,
        angle_tolerance_deg=angle_tolerance_deg,
        line_tolerance_px=line_tolerance_px,
        min_span_px=min_span_px,
        frame_times=frame_times,
        reference_interval_s=reference_interval_s,
    )
    return {
        "label": label,
        "request": {
            "tracks": [[item.as_dict() for item in history] for history in tracks],
            "psf_fwhm": psf_fwhm,
            "min_presence": min_presence,
            "angle_tolerance_deg": angle_tolerance_deg,
            "line_tolerance_px": line_tolerance_px,
            "min_span_px": min_span_px,
            "frame_times": frame_times,
            "reference_interval_s": reference_interval_s,
        },
        "expected": {"groups": [[item.as_dict() for item in group] for group in groups]},
    }


def main() -> None:
    line = [[point(0, 0, 0)], [point(1, 40, 0)], [point(2, 80, 0)]]
    two_lines = line + [[point(3, 0, 100)], [point(4, 40, 100)], [point(5, 80, 100)]]
    cases = [
        capture("three_long_singletons_form_rescued_track", line),
        capture("short_feature_below_default_length_is_ignored", [[point(0, 0, 0, length=31.99)], [point(1, 40, 0)], [point(2, 80, 0)]]),
        capture("minimum_feature_length_is_inclusive", [[point(0, 0, 0, length=32)], [point(1, 40, 0)], [point(2, 80, 0)]]),
        capture("multi_point_histories_are_not_rescue_singletons", [[point(0, 0, 0), point(1, 40, 0)], [point(1, 40, 0)], [point(2, 80, 0)]]),
        capture("min_presence_greater_than_available_returns_empty", line, min_presence=4),
        capture("two_disjoint_paths_are_rescued_iteratively", two_lines),
        capture("short_path_below_minimum_span_is_rejected", [[point(0, 0, 0)], [point(1, 25, 0)], [point(2, 50, 0)]], min_span_px=64),
        capture("psf_expands_minimum_length_gate", [[point(0, 0, 0, length=79.9)], [point(1, 40, 0, length=80)], [point(2, 80, 0, length=80)]], psf_fwhm=10),
        capture("psf_contributes_to_minimum_span", [[point(0, 0, 0, length=80)], [point(1, 20, 0, length=80)], [point(2, 40, 0, length=80)]], psf_fwhm=10, min_span_px=20),
        capture("line_tolerance_controls_perpendicular_residual", [[point(0, 0, 0)], [point(1, 40, 4)], [point(2, 80, 0)]], line_tolerance_px=1),
        capture("relaxed_line_tolerance_accepts_small_offset", [[point(0, 0, 0)], [point(1, 40, 4)], [point(2, 80, 0)]], line_tolerance_px=5),
        capture("angle_outlier_does_not_prevent_majority_group", [[point(0, 0, 0)], [point(1, 40, 0)], [point(2, 80, 0, angle=50)], [point(3, 120, 0)]], angle_tolerance_deg=12),
        capture("same_frame_alternative_uses_nearest_line_candidate", [[point(0, 0, 0)], [point(1, 40, 9)], [point(1, 40, 1)], [point(2, 80, 0)]]),
        capture("shuffled_track_order_preserves_selected_frame_order", [[point(2, 80, 0)], [point(0, 0, 0)], [point(1, 40, 0)]]),
        capture("irregular_frame_times_are_passed_to_geometry_refit", line, frame_times=[0, 2, 8], reference_interval_s=4),
        capture("missing_time_metadata_falls_back_to_frame_indices", line, frame_times=None),
        capture("line_angle_gate_rejects_wrong_axis", [[point(0, 0, 0, angle=90)], [point(1, 40, 0, angle=90)], [point(2, 80, 0, angle=90)]], angle_tolerance_deg=10),
        capture("all_available_points_are_removed_after_rescue", two_lines, min_presence=3),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/collinear_motion_rescue.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
