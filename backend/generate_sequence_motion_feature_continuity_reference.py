"""Freeze Python temporal-continuity and internal-gap bridge scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import (
    _motion_feature_internal_gap_bridge_is_supported,
    _motion_feature_temporal_continuity_is_supported,
)


def capture(label, indices, *, min_presence=3, edge_truncated=False, options=None):
    options = options or {}
    points = [{"frame_index": index} for index in indices]
    objects = [SimpleNamespace(**point) for point in points]
    expected = {
        "temporal_continuity_supported": _motion_feature_temporal_continuity_is_supported(
            objects, min_presence=min_presence, edge_truncated=edge_truncated
        ),
        "internal_gap_bridge_supported": _motion_feature_internal_gap_bridge_is_supported(
            objects,
            max_gap_frames=options.get("max_gap_frames", 2),
            min_run_length=options.get("min_run_length", 3),
        ),
    }
    return {
        "label": label,
        "request": {
            "points": points,
            "min_presence": min_presence,
            "edge_truncated": edge_truncated,
            "options": options,
        },
        "expected": expected,
    }


def main() -> None:
    cases = [
        capture("single_contiguous_run_meets_presence", [0, 1, 2, 3], min_presence=4),
        capture("single_contiguous_run_short_of_presence", [0, 1, 2, 3], min_presence=5),
        capture("edge_truncation_relaxes_one_frame", [0, 1, 2], min_presence=4, edge_truncated=True),
        capture("edge_relaxation_keeps_three_frame_floor", [0, 1], min_presence=2, edge_truncated=True),
        capture("interior_gap_splits_contiguous_confirmation", [0, 1, 4, 5], min_presence=3),
        capture("duplicate_observation_frames_count_once", [0, 1, 1, 2], min_presence=3),
        capture("empty_track_fails_both_gates", [], min_presence=3),
        capture("two_three_frame_runs_with_one_gap_bridge", [0, 1, 2, 4, 5, 6], min_presence=4),
        capture("maximum_two_frame_gap_can_bridge", [0, 1, 2, 5, 6, 7], min_presence=4),
        capture("gap_over_limit_cannot_bridge", [0, 1, 2, 6, 7, 8], min_presence=4),
        capture("only_one_strong_run_cannot_bridge", [0, 1, 2, 4, 5], min_presence=3),
        capture("two_runs_below_minimum_length_cannot_bridge", [0, 1, 3, 4], min_presence=3, options={"min_run_length": 3}),
        capture("third_strong_run_cannot_bridge", [0, 1, 2, 4, 5, 6, 8, 9, 10], min_presence=5),
        capture("one_nearby_leading_singleton_is_allowed", [0, 2, 3, 4, 6, 7, 8], min_presence=4),
        capture("two_leading_isolated_points_are_allowed", [0, 1, 3, 4, 5, 7, 8, 9], min_presence=4),
        capture("distant_leading_singleton_is_rejected", [0, 5, 6, 7, 9, 10, 11], min_presence=4),
        capture("interior_short_run_is_rejected", [0, 1, 2, 4, 6, 7, 8], min_presence=4, options={"max_gap_frames": 3}),
        capture("custom_gap_and_run_thresholds", [0, 1, 3, 4], min_presence=3, options={"max_gap_frames": 1, "min_run_length": 2}),
        capture("zero_gap_limit_disables_bridge", [0, 1, 2, 4, 5, 6], min_presence=4, options={"max_gap_frames": 0}),
        capture("one_frame_run_threshold_disables_bridge", [0, 1, 2, 4, 5, 6], min_presence=4, options={"min_run_length": 1}),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_motion_feature_continuity.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
