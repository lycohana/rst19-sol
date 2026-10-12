"""Freeze Python spatial PCA and time-aware trail-geometry fit cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _fit_collinear_motion_geometry


def capture(
    label: str,
    coordinates: list[tuple[int, float, float]],
    *,
    frame_times: list[float] | None = None,
    reference_interval_s: float | str | None = None,
) -> dict[str, object]:
    points = [
        SimpleNamespace(frame_index=index, aligned_x=x, aligned_y=y)
        for index, x, y in coordinates
    ]
    result = _fit_collinear_motion_geometry(
        points,
        frame_times=frame_times,
        reference_interval_s=reference_interval_s,
    )
    return {
        "label": label,
        "request": {
            "points": [vars(point) for point in points],
            "frame_times": frame_times,
            "reference_interval_s": reference_interval_s,
        },
        "expected": {
            "projected_span": result[0],
            "mean_speed_px_per_frame": result[1],
            "rms_px": result[2],
            "angle_deg": result[3],
        },
    }


def main() -> None:
    cases = [
        capture("too_few_points_returns_null_geometry", [(0, 0, 0)]),
        capture("horizontal_line_geometry", [(0, 0, 0), (1, 5, 0), (2, 10, 0)]),
        capture("vertical_line_geometry", [(0, 2, 0), (1, 2, 5), (2, 2, 10)]),
        capture("diagonal_line_geometry", [(0, 0, 0), (1, 3, 4), (2, 6, 8)]),
        capture("off_axis_residual_and_pca_span", [(0, 0, 0), (1, 4, 1), (2, 8, 0), (3, 12, 2)]),
        capture("two_point_geometry", [(0, 0, 0), (1, 3, 4)]),
        capture("isotropic_square_has_no_unique_axis", [(0, 0, 0), (1, 1, 0), (2, 1, 1), (3, 0, 1)]),
        capture("identical_centers_have_no_unique_axis", [(0, 4, 4), (1, 4, 4), (2, 4, 4)]),
        capture("irregular_times_scale_legacy_frame_speed", [(0, 0, 0), (1, 10, 0), (2, 25, 0)], frame_times=[100, 110, 125], reference_interval_s=10),
        capture("missing_times_use_frame_indices", [(0, 0, 0), (1, 4, 0), (2, 8, 0)]),
        capture("duplicate_times_use_frame_indices", [(0, 0, 0), (1, 4, 0), (2, 8, 0)], frame_times=[0, 2, 2]),
        capture("wrong_time_vector_length_uses_frame_indices", [(0, 0, 0), (1, 4, 0), (2, 8, 0)], frame_times=[0, 2]),
        capture("out_of_order_points_fall_back_to_frame_numbers", [(2, 8, 0), (0, 0, 0), (1, 4, 0)], frame_times=[0, 2, 4]),
        capture("invalid_reference_interval_falls_back_to_one", [(0, 0, 0), (1, 5, 0), (2, 10, 0)], reference_interval_s=0),
        capture("negative_reference_interval_falls_back_to_one", [(0, 0, 0), (1, 5, 0), (2, 10, 0)], reference_interval_s=-2),
        capture("numeric_string_reference_interval_converts", [(0, 0, 0), (1, 5, 0), (2, 10, 0)], reference_interval_s="2.5"),
        capture("duplicate_frame_points_keep_zero_duration", [(0, 0, 0), (0, 5, 0)]),
        capture("nearly_horizontal_angle_is_stable", [(0, 0, 0), (1, 5, 0.000001), (2, 10, 0)]),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/collinear_motion_geometry.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
