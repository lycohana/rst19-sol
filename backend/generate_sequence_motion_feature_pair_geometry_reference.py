"""Freeze Python line-candidate pair geometry penalty cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _motion_feature_pair_geometry_penalty


def point(x: float, y: float, angle: float | None, length: float, *, edge: bool = False):
    return SimpleNamespace(
        aligned_x=x,
        aligned_y=y,
        angle_deg=angle,
        length_px=length,
        touches_edge=edge,
    )


def capture(
    label: str,
    previous,
    current,
    *,
    psf_fwhm: float = 2.0,
    expected_step_px: float | None = None,
) -> dict[str, object]:
    request = {
        "previous": vars(previous),
        "current": vars(current),
        "psf_fwhm": psf_fwhm,
        "expected_step_px": expected_step_px,
    }
    expected = {
        "penalty": _motion_feature_pair_geometry_penalty(
            previous,
            current,
            psf_fwhm=psf_fwhm,
            expected_step_px=expected_step_px,
        )
    }
    return {"label": label, "request": request, "expected": expected}


def main() -> None:
    base = point(10.0, 10.0, 0.0, 8.0)
    cases = [
        capture("movement_below_identifiable_threshold", base, point(11.0, 10.0, 90.0, 8.0)),
        capture("movement_exactly_at_threshold_is_ignored", base, point(11.5, 10.0, 90.0, 8.0)),
        capture("aligned_direction_and_translated_endpoints_cost_zero", base, point(20.0, 10.0, 0.0, 8.0)),
        capture("perpendicular_lines_receive_capped_soft_cost", base, point(20.0, 10.0, 90.0, 8.0)),
        capture("line_angle_wraparound_uses_undirected_axis", point(10, 10, 179, 8), point(20, 10, 1, 8)),
        capture("edge_truncation_reduces_angle_cost", base, point(20, 10, 90, 8, edge=True)),
        capture("expected_step_aligned_forward_has_no_extra_cost", base, point(14, 10, 0, 8), expected_step_px=4),
        capture("expected_step_aligned_backward_is_supported", base, point(6, 10, 180, 8), expected_step_px=4),
        capture("expected_step_mismatch_is_a_soft_cost", base, point(20, 10, 0, 8), expected_step_px=4),
        capture("nonpositive_expected_step_is_ignored", base, point(20, 10, 0, 8), expected_step_px=0),
        capture("unequal_line_lengths_add_endpoint_cost", base, point(20, 10, 0, 14)),
        capture("reversed_line_endpoint_order_is_equivalent", point(10, 10, 180, 8), point(20, 10, 0, 8)),
        capture("both_edge_candidates_apply_one_weight", point(10, 10, 90, 8, edge=True), point(20, 10, 90, 8, edge=True)),
        capture("only_previous_edge_candidate_reduces_cost", point(10, 10, 90, 8, edge=True), point(20, 10, 90, 8)),
        capture("nonpositive_length_skips_endpoint_cost", point(10, 10, 90, 0), point(20, 10, 90, 8)),
        capture("diagonal_motion_uses_two_dimensional_angle", point(0, 0, 45, 10), point(10, 10, 45, 10)),
        capture("expected_step_uses_previous_line_both_directions", point(5, 5, 45, 10), point(12.071067811865476, 12.071067811865476, 45, 10), expected_step_px=10),
        capture("larger_psf_raises_minimum_motion_threshold", point(0, 0, 0, 8), point(2.5, 0, 90, 8), psf_fwhm=4),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_motion_feature_pair_geometry.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
