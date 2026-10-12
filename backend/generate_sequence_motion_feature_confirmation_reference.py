"""Freeze Python motion-feature confirmation presence and reason scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import (
    _motion_feature_confirmation_presence,
    _motion_feature_rejection_reasons,
)


def capture(label, frame_count, *, fields=None, point_models=()):
    track = {
        "classification": "candidate",
        "evidence_level": "multi_frame",
        "points": [{"single_frame_model": model} for model in point_models],
    }
    track.update(fields or {})
    point_objects = [SimpleNamespace(**point) for point in track["points"]]
    reason_kwargs = {
        key: track.get(key, default)
        for key, default in {
            "presence_supported": None,
            "displacement_px": 0.0,
            "min_displacement_px": None,
            "fit_rms_px": None,
            "max_fit_rms_px": None,
            "shape_supported": None,
            "model_supported": None,
            "motion_axis_supported": None,
            "temporal_continuity_supported": None,
            "exposure_supported": None,
            "artifact_supported": None,
            "edge_truncated": False,
        }.items()
    }
    expected = {
        "required_presence": _motion_feature_confirmation_presence(frame_count),
        "rejection_reasons": list(
            _motion_feature_rejection_reasons(
                classification=track["classification"],
                evidence_level=track["evidence_level"],
                points=point_objects,
                **reason_kwargs,
            )
        ),
    }
    return {"label": label, "request": {"frame_count": frame_count, "track": track}, "expected": expected}


def main() -> None:
    passing = {
        "presence_supported": True,
        "displacement_px": 5.0,
        "min_displacement_px": 2.0,
        "fit_rms_px": 0.2,
        "max_fit_rms_px": 0.75,
        "shape_supported": True,
        "model_supported": True,
        "motion_axis_supported": True,
        "temporal_continuity_supported": True,
        "exposure_supported": True,
        "artifact_supported": True,
    }
    all_failed = {
        "presence_supported": False,
        "edge_truncated": True,
        "displacement_px": 1.0,
        "min_displacement_px": 2.0,
        "fit_rms_px": 1.0,
        "max_fit_rms_px": 0.75,
        "shape_supported": False,
        "model_supported": False,
        "motion_axis_supported": False,
        "temporal_continuity_supported": False,
        "exposure_supported": False,
        "artifact_supported": False,
    }
    cases = [
        capture("short_sequence_frame_count_1", 1, fields={"classification": "moving"}),
        capture("short_sequence_frame_count_2", 2, fields={"classification": "moving"}),
        capture("short_sequence_frame_count_3", 3, fields={"classification": "moving"}),
        capture("short_sequence_frame_count_7", 7, fields={"classification": "moving"}),
        capture("long_sequence_frame_count_8", 8, fields={"classification": "moving"}),
        capture("odd_sequence_rounds_required_coverage_up", 9, fields={"classification": "moving"}),
        capture("fifteen_frame_confirmation_requires_eight", 15, fields={"classification": "moving"}),
        capture("large_sequence_confirmation_scales_by_half", 101, fields={"classification": "moving"}),
        capture("single_frame_track_explains_missing_time_axis", 15, fields={"evidence_level": "single_frame_trail"}),
        capture("single_frame_track_with_point_keeps_two_reason_codes", 8, fields={"evidence_level": "single_frame_trail"}, point_models=(None,)),
        capture("missing_points_and_fit_report_basic_failures", 4),
        capture("insufficient_coverage_distinguishes_edge_relaxation", 15, fields={"presence_supported": False, "edge_truncated": True}, point_models=(None, None)),
        capture("insufficient_normal_coverage_uses_presence_reason", 4, fields={"presence_supported": False}, point_models=(None, None)),
        capture("displacement_and_missing_fit_are_reported", 8, fields={"displacement_px": 1.0, "min_displacement_px": 2.0}, point_models=(None, None)),
        capture("high_fit_residual_is_reported", 8, fields={"fit_rms_px": 1.0, "max_fit_rms_px": 0.75}, point_models=(None, None)),
        capture("compact_model_families_precede_line_majority_reason", 15, fields={"model_supported": False}, point_models=("double", "point", "elliptical_point", "line")),
        capture("all_failed_confirmation_gates_keep_stable_order", 15, fields=all_failed, point_models=("double", "point", "elliptical_point")),
        capture("candidate_without_explicit_failures_gets_generic_reason", 15, point_models=(None, None)),
        capture("fully_supported_moving_track_has_no_rejection_codes", 15, fields={"classification": "moving", **passing}, point_models=("line", "line")),
        capture("noncandidate_never_gets_confirmation_reasons", 15, fields={"classification": "rejected", **all_failed}, point_models=("double", "point")),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_motion_feature_confirmation.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
