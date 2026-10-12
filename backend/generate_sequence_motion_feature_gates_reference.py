"""Freeze Python cross-frame motion-feature gate scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import (
    _motion_feature_artifact_is_supported_as_moving,
    _motion_feature_artifact_overlap_is_compatible_with_motion,
    _motion_feature_edge_truncation_is_supported,
    _motion_feature_motion_axis_is_supported,
    _motion_feature_track_shape_is_supported,
)


def point(index, **values):
    defaults = {
        "frame_index": index,
        "aligned_x": 20.0 + 2.0 * index,
        "aligned_y": 30.0,
        "angle_deg": 0.0,
        "length_px": 20.0,
        "continuity_ratio": 0.8,
        "profile_fill_ratio": 0.8,
        "line_artifact_overlap_fraction": None,
        "bbox": [30.0, 30.0, 40.0, 35.0],
    }
    defaults.update(values)
    return defaults


def points(**overrides):
    return [point(index, **overrides) for index in range(4)]


def capture(label, items, *, frame_shapes=None, options=None, compatibility=None):
    options = options or {}
    objects = [SimpleNamespace(**item) for item in items]
    expected = {
        "track_shape_supported": _motion_feature_track_shape_is_supported(
            objects,
            min_profile_fill_ratio=options.get("min_profile_fill_ratio", 0.45),
            min_continuity_ratio=options.get("min_continuity_ratio", 0.45),
            max_angle_deviation_deg=options.get("max_angle_deviation_deg", 15.0),
            max_length_deviation_fraction=options.get("max_length_deviation_fraction", 0.75),
        ),
        "motion_axis_supported": _motion_feature_motion_axis_is_supported(
            objects,
            max_deviation_deg=options.get("max_motion_axis_deviation_deg", 25.0),
            min_pass_fraction=options.get("min_motion_axis_pass_fraction", 0.60),
        ),
        "artifact_supported_as_moving": _motion_feature_artifact_is_supported_as_moving(
            objects,
            overlap_fraction=options.get("artifact_overlap_fraction", 0.90),
            min_clear_fraction=options.get("min_artifact_clear_fraction", 0.50),
        ),
        "artifact_overlap_compatible_with_motion": _motion_feature_artifact_overlap_is_compatible_with_motion(
            objects,
            displacement_px=compatibility["displacement_px"],
            fit_rms_px=compatibility["fit_rms_px"],
            max_fit_rms_px=compatibility["max_fit_rms_px"],
            presence_supported=compatibility["presence_supported"],
            min_displacement_px=compatibility["min_displacement_px"],
            shape_supported=compatibility["shape_supported"],
            motion_axis_supported=compatibility["motion_axis_supported"],
            temporal_continuity_supported=compatibility["temporal_continuity_supported"],
            exposure_supported=compatibility["exposure_supported"],
            model_supported=compatibility["model_supported"],
        ),
        "edge_truncation_supported": _motion_feature_edge_truncation_is_supported(
            objects,
            frame_shapes or [],
            edge_margin_px=options.get("edge_margin_px", 4.0),
        ),
    }
    return {
        "label": label,
        "request": {
            "points": items,
            "frame_shapes": frame_shapes or [],
            "options": options,
            "artifact_compatibility": compatibility,
        },
        "expected": expected,
    }


def compatibility(**overrides):
    values = {
        "displacement_px": 5.0,
        "fit_rms_px": 0.2,
        "max_fit_rms_px": 0.75,
        "presence_supported": True,
        "min_displacement_px": 2.0,
        "shape_supported": True,
        "motion_axis_supported": True,
        "temporal_continuity_supported": True,
        "exposure_supported": True,
        "model_supported": True,
    }
    values.update(overrides)
    return values


def edge_points(indices, edge_index=None, *, edge_side="left"):
    output = []
    for index in indices:
        bbox = [30.0, 30.0, 40.0, 35.0]
        if index == edge_index:
            bbox = [2.0, 30.0, 8.0, 35.0] if edge_side == "left" else [30.0, 2.0, 35.0, 8.0]
        output.append(point(index, bbox=bbox))
    return output


def main() -> None:
    stable = points()
    wrap = [point(i, angle_deg=angle) for i, angle in enumerate([179.0, 0.0, 1.0, 178.0])]
    unstable = [point(i, angle_deg=angle) for i, angle in enumerate([0.0, 45.0, 90.0, 135.0])]
    partial = [
        point(0, continuity_ratio=0.9, profile_fill_ratio=0.9),
        point(1, continuity_ratio=0.9, profile_fill_ratio=0.9),
        point(2, continuity_ratio=0.1, profile_fill_ratio=0.1),
        point(3, continuity_ratio=0.1, profile_fill_ratio=0.1),
    ]
    sparse = [point(i, continuity_ratio=0.9 if i == 0 else 0.1, profile_fill_ratio=0.9 if i == 0 else 0.1) for i in range(4)]
    irregular = [point(i, length_px=length) for i, length in enumerate([1.0, 1.0, 100.0, 100.0])]
    short = points(length_px=12.0, continuity_ratio=0.7, profile_fill_ratio=0.7)
    perpendicular = [point(i, angle_deg=90.0) for i in range(4)]
    stationary = [point(i, aligned_x=20.0, aligned_y=30.0) for i in range(4)]
    high_artifact = points(line_artifact_overlap_fraction=0.95)
    half_clear = [point(i, line_artifact_overlap_fraction=value) for i, value in enumerate([0.1, 0.95, 0.1, 0.95])]
    unknown_artifact = points()
    frames = [[100, 120] for _ in range(6)]
    cases = [
        capture("stable_shape_direction_and_clear_artifact_audit", stable, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("line_angle_wraparound_is_shortest_axis_difference", wrap, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("unstable_line_direction_rejects_track_shape", unstable, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("half_of_observations_can_support_shape", partial, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("one_supported_observation_is_not_enough", sparse, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("inconsistent_lengths_reject_track_shape", irregular, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("short_lines_require_stronger_profile_support", short, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("motion_axis_parallel_to_line_passes", stable, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("motion_axis_perpendicular_to_line_fails", perpendicular, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("subpixel_motion_steps_do_not_count_as_transitions", stationary, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("unknown_artifact_overlap_is_not_a_veto", unknown_artifact, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("known_high_overlap_majority_vetoes_moving", high_artifact, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility(presence_supported=False)),
        capture("half_clear_artifact_audit_passes", half_clear, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("artifact_veto_can_be_overridden_by_complete_motion_evidence", high_artifact, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("artifact_veto_requires_low_fit_residual", high_artifact, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility(fit_rms_px=1.0)),
        capture("artifact_veto_requires_minimum_displacement", high_artifact, frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility(displacement_px=1.0)),
        capture("suffix_edge_truncation_checks_first_observation", edge_points([2, 3, 4, 5], 2), frame_shapes=frames, compatibility=compatibility()),
        capture("prefix_edge_truncation_checks_last_observation", edge_points([0, 1, 2], 2, edge_side="top"), frame_shapes=[ [100, 120] for _ in range(5)], compatibility=compatibility()),
        capture("internal_observation_run_has_no_edge_exception", edge_points([1, 2, 3]), frame_shapes=[[100, 120] for _ in range(5)], compatibility=compatibility()),
        capture("internal_gap_disables_edge_exception", edge_points([0, 2, 3], 0), frame_shapes=[[100, 120] for _ in range(4)], compatibility=compatibility()),
        capture("caller_can_override_track_shape_and_axis_limits", stable, frame_shapes=[[100, 120] for _ in range(4)], options={"min_profile_fill_ratio": 0.75, "min_continuity_ratio": 0.75, "max_angle_deviation_deg": 10.0, "max_length_deviation_fraction": 0.5, "max_motion_axis_deviation_deg": 20.0, "min_motion_axis_pass_fraction": 0.75, "edge_margin_px": 3.0}, compatibility=compatibility()),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_motion_feature_gates.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
