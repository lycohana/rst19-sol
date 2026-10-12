"""Freeze Python point-track/line-track evidence-link cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import (
    MotionFeaturePoint,
    MotionFeatureTrack,
    SourceTrack,
    TrackPoint,
    _link_motion_evidence_tracks,
)


def source_track(
    track_id: int,
    coords: list[tuple[float, float]],
    *,
    frames: list[int] | None = None,
    classification: str = "moving",
) -> SourceTrack:
    if frames is None:
        frames = list(range(len(coords)))
    points = tuple(
        TrackPoint(
            frame_index=frame,
            detection_id=track_id * 100 + index,
            x=x,
            y=y,
            aligned_x=x,
            aligned_y=y,
            flux_snr=20.0,
        )
        for index, (frame, (x, y)) in enumerate(zip(frames, coords, strict=True))
    )
    displacement = (
        ((coords[-1][0] - coords[0][0]) ** 2 + (coords[-1][1] - coords[0][1]) ** 2) ** 0.5
        if len(coords) >= 2
        else 0.0
    )
    return SourceTrack(track_id, classification, points, displacement, 1.0, 0.0, "fast_point_motion")


def line_track(
    track_id: int,
    coords: list[tuple[float, float]],
    *,
    frames: list[int] | None = None,
    classification: str = "moving",
) -> MotionFeatureTrack:
    if frames is None:
        frames = list(range(len(coords)))
    points = tuple(
        MotionFeaturePoint(
            frame_index=frame,
            x=x,
            y=y,
            aligned_x=x,
            aligned_y=y,
            residual_snr=30.0,
            area_pixels=40,
            length_px=12.0,
            width_px=3.0,
            angle_deg=0.0,
            bbox=(int(x - 5), int(y - 5), int(x + 5), int(y + 5)),
            touches_edge=False,
            continuity_ratio=0.95,
            profile_fill_ratio=0.9,
        )
        for frame, (x, y) in zip(frames, coords, strict=True)
    )
    displacement = (
        ((coords[-1][0] - coords[0][0]) ** 2 + (coords[-1][1] - coords[0][1]) ** 2) ** 0.5
        if len(coords) >= 2
        else 0.0
    )
    return MotionFeatureTrack(track_id, classification, points, displacement, 1.0, 0.0)


def capture(
    label: str,
    point_tracks: list[SourceTrack],
    line_tracks: list[MotionFeatureTrack],
    *,
    min_overlap_fraction: float = 0.5,
    max_perpendicular_distance_px: float = 10.0,
    max_direction_difference_deg: float = 25.0,
) -> dict[str, object]:
    links = _link_motion_evidence_tracks(
        point_tracks,
        line_tracks,
        min_overlap_fraction=min_overlap_fraction,
        max_perpendicular_distance_px=max_perpendicular_distance_px,
        max_direction_difference_deg=max_direction_difference_deg,
    )
    return {
        "label": label,
        "request": {
            "point_tracks": [track.as_dict() for track in point_tracks],
            "line_tracks": [track.as_dict() for track in line_tracks],
            "min_overlap_fraction": min_overlap_fraction,
            "max_perpendicular_distance_px": max_perpendicular_distance_px,
            "max_direction_difference_deg": max_direction_difference_deg,
        },
        "expected": {"links": [link.as_dict() for link in links]},
    }


def main() -> None:
    horizontal_points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    horizontal_lines = [(2.0, 0.0), (12.0, 0.0), (22.0, 0.0)]
    cases = [
        capture("longitudinal_centroid_offset_is_allowed", [source_track(10, horizontal_points)], [line_track(20, horizontal_lines)]),
        capture("perpendicular_offset_over_default_limit_is_rejected", [source_track(10, horizontal_points)], [line_track(20, [(2, 10.01), (12, 10.01), (22, 10.01)])]),
        capture("perpendicular_limit_is_inclusive", [source_track(10, horizontal_points)], [line_track(20, [(2, 10), (12, 10), (22, 10)])]),
        capture("direction_limit_is_inclusive", [source_track(10, horizontal_points)], [line_track(20, [(0, 0), (10, 10), (20, 20)])], max_direction_difference_deg=45.0),
        capture("direction_over_limit_is_rejected", [source_track(10, horizontal_points)], [line_track(20, [(0, 0), (10, 10), (20, 20)])], max_direction_difference_deg=44.9),
        capture("opposite_direction_is_rejected", [source_track(10, horizontal_points)], [line_track(20, [(20, 0), (10, 0), (0, 0)])]),
        capture("minimum_displacement_ratio_is_inclusive", [source_track(10, horizontal_points)], [line_track(20, [(0, 0), (3.5, 0), (7, 0)])]),
        capture("ratio_below_minimum_is_rejected", [source_track(10, horizontal_points)], [line_track(20, [(0, 0), (3.49, 0), (6.98, 0)])]),
        capture("maximum_displacement_ratio_is_inclusive", [source_track(10, horizontal_points)], [line_track(20, [(0, 0), (25, 0), (50, 0)])]),
        capture("ratio_above_maximum_is_rejected", [source_track(10, horizontal_points)], [line_track(20, [(0, 0), (25.01, 0), (50.02, 0)])]),
        capture("at_least_three_common_frames_are_required", [source_track(10, [(0, 0), (10, 0), (20, 0)], frames=[0, 1, 2])], [line_track(20, [(0, 0), (10, 0), (20, 0)], frames=[0, 1, 3])]),
        capture("overlap_fraction_ceiling_is_applied", [source_track(10, [(0, 0), (10, 0), (20, 0), (30, 0)], frames=[0, 1, 2, 4])], [line_track(20, [(0, 0), (10, 0), (20, 0), (30, 0)], frames=[0, 1, 2, 3])], min_overlap_fraction=0.8),
        capture("overlap_fraction_boundary_is_inclusive", [source_track(10, [(0, 0), (10, 0), (20, 0), (30, 0)], frames=[0, 1, 2, 4])], [line_track(20, [(0, 0), (10, 0), (20, 0), (30, 0)], frames=[0, 1, 2, 3])], min_overlap_fraction=0.75),
        capture("common_frame_fraction_uses_shorter_track", [source_track(10, [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)])], [line_track(20, [(2, 0), (12, 0), (22, 0)])]),
        capture("nonmoving_point_track_is_ignored", [source_track(10, horizontal_points, classification="static")], [line_track(20, horizontal_lines)]),
        capture("nonmoving_line_track_is_ignored", [source_track(10, horizontal_points)], [line_track(20, horizontal_lines, classification="candidate")]),
        capture("short_point_track_is_ignored", [source_track(10, [(0, 0)])], [line_track(20, horizontal_lines)]),
        capture("short_line_track_is_ignored", [source_track(10, horizontal_points)], [line_track(20, [(0, 0), (10, 0)])]),
        capture("zero_point_displacement_is_ignored", [source_track(10, [(0, 0), (0, 0), (0, 0)])], [line_track(20, horizontal_lines)]),
        capture("zero_line_displacement_is_ignored", [source_track(10, horizontal_points)], [line_track(20, [(0, 0), (0, 0), (0, 0)])]),
        capture("one_to_one_greedy_pairing_selects_better_link", [source_track(10, horizontal_points), source_track(11, [(0, 1), (10, 1), (20, 1)])], [line_track(20, horizontal_lines)]),
        capture("links_are_sorted_by_track_ids", [source_track(91, horizontal_points), source_track(4, [(0, 40), (10, 40), (20, 40)])], [line_track(81, horizontal_lines), line_track(3, [(2, 40), (12, 40), (22, 40)])]),
        capture("empty_moving_tracks_produce_no_links", [], [line_track(20, horizontal_lines)]),
        capture("duplicate_frames_keep_last_point_for_overlap", [source_track(10, [(0, 0), (10, 0), (20, 0), (30, 0)], frames=[0, 1, 1, 2])], [line_track(20, horizontal_lines)]),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/motion_evidence_link.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
