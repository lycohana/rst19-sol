"""Freeze Python best-collinear-motion-feature grouping cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _best_collinear_motion_feature_group


def point(frame: int, x: float, y: float, *, angle: float = 0.0, length: float = 12.0):
    return SimpleNamespace(
        frame_index=frame,
        aligned_x=x,
        aligned_y=y,
        angle_deg=angle,
        length_px=length,
    )


def capture(
    label: str,
    points,
    candidate_indices: list[int],
    *,
    min_length_px: float = 8.0,
    angle_tolerance_deg: float = 20.0,
    line_tolerance_px: float = 3.0,
    min_span_px: float = 8.0,
    frame_times: list[float] | None = None,
    reference_interval_s: float | None = None,
) -> dict[str, object]:
    result = _best_collinear_motion_feature_group(
        points,
        candidate_indices,
        min_length_px=min_length_px,
        angle_tolerance_deg=angle_tolerance_deg,
        line_tolerance_px=line_tolerance_px,
        min_span_px=min_span_px,
        frame_times=frame_times,
        reference_interval_s=reference_interval_s,
    )
    return {
        "label": label,
        "request": {
            "points": [vars(item) for item in points],
            "candidate_indices": candidate_indices,
            "min_length_px": min_length_px,
            "angle_tolerance_deg": angle_tolerance_deg,
            "line_tolerance_px": line_tolerance_px,
            "min_span_px": min_span_px,
            "frame_times": frame_times,
            "reference_interval_s": reference_interval_s,
        },
        "expected": {"group_indices": result},
    }


def main() -> None:
    cases = [
        capture("three_collinear_frames_form_group", [point(0, 0, 0), point(1, 10, 0), point(2, 20, 0)], [0, 1, 2]),
        capture("fewer_than_three_candidates_return_none", [point(0, 0, 0), point(1, 10, 0)], [0, 1]),
        capture("minimum_length_filters_short_candidate", [point(0, 0, 0, length=7), point(1, 10, 0), point(2, 20, 0), point(3, 30, 0)], [0, 1, 2, 3]),
        capture("angle_outlier_is_excluded_from_group", [point(0, 0, 0), point(1, 10, 0), point(2, 20, 0), point(3, 30, 0, angle=80)], [0, 1, 2, 3]),
        capture("perpendicular_outlier_is_excluded", [point(0, 0, 0), point(1, 10, 0), point(2, 20, 0), point(3, 15, 8)], [0, 1, 2, 3]),
        capture("nearest_same_frame_component_is_selected", [point(0, 0, 0), point(1, 10, 2), point(1, 10, 0.5), point(2, 20, 0)], [0, 1, 2, 3]),
        capture("decreasing_projection_is_valid", [point(0, 20, 0, angle=180), point(1, 10, 0, angle=180), point(2, 0, 0, angle=180)], [0, 1, 2]),
        capture("small_monotonic_backtrack_within_tolerance_is_valid", [point(0, 0, 0), point(1, 10, 0), point(2, 9, 0)], [0, 1, 2], line_tolerance_px=4),
        capture("large_monotonic_backtrack_is_rejected", [point(0, 0, 0), point(1, 10, 0), point(2, 6, 0)], [0, 1, 2], line_tolerance_px=4),
        capture("minimum_span_gate_rejects_short_path", [point(0, 0, 0), point(1, 4, 0), point(2, 7, 0)], [0, 1, 2], min_span_px=8),
        capture("crossing_frame_order_is_rejected", [point(0, 0, 0), point(1, 10, 0), point(2, 5, 0)], [0, 1, 2]),
        capture("exact_perpendicular_limit_is_inclusive", [point(0, 0, 3), point(1, 10, 3), point(2, 20, 0)], [0, 1, 2], line_tolerance_px=3),
        capture("exact_angle_limit_is_inclusive", [point(0, 0, 0), point(1, 10, 0, angle=20), point(2, 20, 0)], [0, 1, 2], angle_tolerance_deg=20),
        capture("shuffled_candidate_order_returns_frame_order", [point(0, 0, 0), point(1, 10, 0), point(2, 20, 0)], [2, 0, 1]),
        capture("larger_group_wins_over_shorter_alternative", [point(0, 0, 0), point(1, 10, 0), point(2, 20, 0), point(3, 30, 0), point(0, 0, 20, angle=90), point(1, 0, 30, angle=90), point(2, 0, 40, angle=90)], list(range(7))),
        capture("rotated_line_uses_undirected_angles", [point(0, 0, 0, angle=45), point(1, 10, 10, angle=45), point(2, 20, 20, angle=45)], [0, 1, 2]),
        capture("time_metadata_is_passed_to_geometry_refit", [point(0, 0, 0), point(1, 10, 0), point(2, 20, 0)], [0, 1, 2], frame_times=[0, 5, 20], reference_interval_s=10),
        capture("ties_keep_first_seed_pair_group", [point(0, 0, 0), point(1, 10, 0), point(2, 20, 0), point(0, 0, 20, angle=90), point(1, 0, 30, angle=90), point(2, 0, 40, angle=90)], list(range(6))),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/collinear_motion_group.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
