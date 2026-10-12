"""Freeze the Python line-profile metrics used by sequence trail analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _line_profile_metrics


def plane_json(values):
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in np.asarray(values, dtype=np.float32)
    ]


def capsule(center, length, half_width, angle_deg, *, shape=(48, 56), value=12.0):
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    angle = np.deg2rad(angle_deg)
    dx = xx - center[0]
    dy = yy - center[1]
    along = dx * np.cos(angle) + dy * np.sin(angle)
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    result = np.zeros(shape, dtype=np.float32)
    result[(np.abs(along) <= 0.5 * length) & (np.abs(across) <= half_width)] = value
    return result


def capture(label, residual, *, center, length, width, angle, threshold):
    expected = _line_profile_metrics(
        residual,
        center_x=center[0],
        center_y=center[1],
        length_px=length,
        width_px=width,
        angle_deg=angle,
        threshold_adu=threshold,
    )
    return {
        "label": label,
        "request": {
            "residual": plane_json(residual) if np.asarray(residual).ndim == 2 else np.asarray(residual).tolist(),
            "center_x": center[0],
            "center_y": center[1],
            "length_px": length,
            "width_px": width,
            "angle_deg": angle,
            "threshold_adu": threshold,
        },
        "expected": {
            "continuity_ratio": expected[0],
            "profile_fill_ratio": expected[1],
        },
    }


def main() -> None:
    continuous = capsule((28, 24), 28, 1.5, 0.0)
    split = np.zeros((48, 56), dtype=np.float32)
    split[24, 13:19] = 12.0
    split[24, 37:43] = 12.0
    one_pixel_gap = capsule((28, 24), 28, 1.5, 0.0)
    one_pixel_gap[24, 28] = 0.0
    diagonal = capsule((27, 24), 28, 1.5, 38.0)
    edge = capsule((2, 24), 22, 1.5, 0.0)
    invalid_sample = continuous.copy()
    invalid_sample[24, 27] = np.nan
    outside = np.zeros((48, 56), dtype=np.float32)
    cases = [
        capture("continuous_horizontal_trail", continuous, center=(28, 24), length=28, width=1.0, angle=0, threshold=5),
        capture("separated_point_pair_has_sparse_fill", split, center=(28, 24), length=30, width=1.0, angle=0, threshold=5),
        capture("single_pixel_gap_is_bridged", one_pixel_gap, center=(28, 24), length=28, width=1.0, angle=0, threshold=5),
        capture("diagonal_trail_uses_rotated_cross_sections", diagonal, center=(27, 24), length=28, width=2.0, angle=38, threshold=5),
        capture("edge_truncated_profile_ignores_constant_padding", edge, center=(2, 24), length=22, width=1.0, angle=0, threshold=5),
        capture("nonfinite_interpolation_is_not_valid_support", invalid_sample, center=(28, 24), length=28, width=1.0, angle=0, threshold=5),
        capture("entire_profile_outside_image_returns_null", outside, center=(-10, 24), length=8, width=1.0, angle=0, threshold=5),
        capture("one_dimensional_residual_returns_null", np.arange(10, dtype=np.float32), center=(0, 0), length=5, width=1, angle=0, threshold=1),
        capture("nonpositive_length_or_threshold_returns_null", continuous, center=(28, 24), length=0, width=1.0, angle=0, threshold=5),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_trail_metrics.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
