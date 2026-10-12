"""Freeze Python line-shape classification and review-reason scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import (
    _motion_feature_shape_is_supported,
    _sequence_line_shape_is_supported,
    _single_frame_model_is_supported,
    _single_frame_trail_shape_is_supported,
    _single_frame_trail_shape_reason,
)


def point(**values):
    defaults = {
        "length_px": 12.0,
        "continuity_ratio": 0.9,
        "profile_fill_ratio": 0.9,
        "line_artifact_overlap_fraction": None,
        "single_frame_model": None,
        "line_template_snr": None,
        "line_vs_point_delta_bic": None,
        "line_vs_double_delta_bic": None,
        "line_vs_background_delta_bic": None,
        "line_vs_elliptical_delta_bic": None,
    }
    defaults.update(values)
    return defaults


def capture(label, values, options=None):
    options = options or {}
    candidate = SimpleNamespace(**point(**values))
    kwargs = {
        "min_length_px": options.get("min_length_px", 8.0),
        "min_profile_fill_ratio": options.get("min_profile_fill_ratio", 0.80),
        "min_continuity_ratio": options.get("min_continuity_ratio", 0.80),
    }
    motion_fill = options.get("motion_min_profile_fill_ratio", 0.45)
    motion_continuity = options.get("motion_min_continuity_ratio", 0.45)
    reason = _single_frame_trail_shape_reason(candidate, **kwargs)
    expected = {
        "single_frame_model_supported": _single_frame_model_is_supported(candidate),
        "motion_feature_shape_supported": _motion_feature_shape_is_supported(
            candidate,
            min_profile_fill_ratio=motion_fill,
            min_continuity_ratio=motion_continuity,
        ),
        "single_frame_supported": _single_frame_trail_shape_is_supported(candidate, **kwargs),
        "single_frame_reason": reason,
        "sequence_line_supported": _sequence_line_shape_is_supported(candidate),
    }
    return {
        "label": label,
        "request": {"point": point(**values), "options": options},
        "expected": expected,
    }


def line_model(**values):
    return {
        "single_frame_model": "line",
        "line_template_snr": 5.0,
        "line_vs_point_delta_bic": 6.0,
        "line_vs_double_delta_bic": 6.0,
        "line_vs_elliptical_delta_bic": 6.0,
        **values,
    }


def main() -> None:
    cases = [
        capture("length_gate_reports_first", {"length_px": 7.9, "continuity_ratio": None, "profile_fill_ratio": None}),
        capture("missing_continuity_is_explicit", {"continuity_ratio": None}),
        capture("missing_fill_is_explicit", {"profile_fill_ratio": None}),
        capture("artifact_overlap_keeps_review_candidate", {"line_artifact_overlap_fraction": 0.75}),
        capture("local_model_out_of_range_is_preserved", {"single_frame_model": "not_fit"}),
        capture("strong_line_model_still_requires_pixel_support", line_model(continuity_ratio=0.5, profile_fill_ratio=0.7)),
        capture("strong_short_line_model_rescues_shape_gate", line_model(continuity_ratio=0.6, profile_fill_ratio=0.6)),
        capture("elliptical_point_reason", {"single_frame_model": "elliptical_point"}),
        capture("single_point_reason", {"single_frame_model": "point"}),
        capture("double_point_reason", {"single_frame_model": "double"}),
        capture("background_reason", {"single_frame_model": "background"}),
        capture("unsupported_long_profile_reason", {"single_frame_model": "long_line_profile", "line_template_snr": 4.0, "line_vs_background_delta_bic": 6.0}),
        capture("short_shape_continuity_gate", {"continuity_ratio": 0.79, "profile_fill_ratio": 0.9}),
        capture("short_shape_fill_gate", {"continuity_ratio": 0.9, "profile_fill_ratio": 0.79}),
        capture("long_lines_use_general_shape_threshold", {"length_px": 20.0, "continuity_ratio": 0.45, "profile_fill_ratio": 0.45}),
        capture("sequence_model_rescue_uses_motion_shape_gate", line_model(continuity_ratio=0.5, profile_fill_ratio=0.5)),
        capture("artifact_below_review_threshold_does_not_reject", {"line_artifact_overlap_fraction": 0.7499}),
        capture("caller_can_override_shape_thresholds", {"length_px": 7.0, "continuity_ratio": 0.7, "profile_fill_ratio": 0.65}, {"min_length_px": 6.0, "min_profile_fill_ratio": 0.6, "min_continuity_ratio": 0.7, "motion_min_profile_fill_ratio": 0.6, "motion_min_continuity_ratio": 0.7}),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_trail_shape.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
