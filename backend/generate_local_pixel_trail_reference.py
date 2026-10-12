"""Freeze raw-pixel local trail-proposal cases from the Python implementation."""
from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
import sys

import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.single_frame_trails import detect_local_pixel_trail_proposals


def render(
    *,
    lines: list[tuple[float, float, float, float, float]],
    points: list[tuple[float, float, float]],
    fwhm: float,
    seed: int,
    size: int = 64,
    variable_fields: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((size, size), dtype=np.float64)
    rng = np.random.default_rng(seed)
    whitened = rng.normal(0.0, 1.0, (size, size))
    sigma = fwhm / 2.354820045
    for cx, cy, length, angle_deg, amplitude in lines:
        angle = np.deg2rad(angle_deg)
        dx = xx - cx
        dy = yy - cy
        along = dx * np.cos(angle) + dy * np.sin(angle)
        across = -dx * np.sin(angle) + dy * np.cos(angle)
        profile = 0.5 * (
            erf((along + length / 2.0) / (np.sqrt(2.0) * sigma))
            - erf((along - length / 2.0) / (np.sqrt(2.0) * sigma))
        )
        whitened += amplitude * profile * np.exp(-0.5 * (across / sigma) ** 2)
    for cx, cy, amplitude in points:
        whitened += amplitude * np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
    if variable_fields:
        background = 80.0 + 0.03 * xx - 0.02 * yy
        noise = 1.4 + 0.002 * xx + 0.001 * yy
    else:
        background = 80.0
        noise = 1.4
    image = background + noise * whitened
    valid = np.ones((size, size), dtype=bool)
    return image, valid, background, noise


def capture(
    label: str,
    *,
    lines: list[tuple[float, float, float, float, float]] | None = None,
    points: list[tuple[float, float, float]] | None = None,
    fwhm: float = 2.6,
    seed: int = 101,
    size: int = 64,
    variable_fields: bool = False,
    mask_slices: list[tuple[slice, slice]] | None = None,
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    image, valid, background, noise = render(
        lines=lines or [],
        points=points or [],
        fwhm=fwhm,
        seed=seed,
        size=size,
        variable_fields=variable_fields,
    )
    for y_slice, x_slice in mask_slices or []:
        valid[y_slice, x_slice] = False
    resolved_options = {
        "length_bank_px": [8.0, 10.0, 12.0, 16.0, 20.0, 24.0],
        "angle_step_deg": 15.0,
        "max_proposals": 8,
        "max_seeds": 64,
        "seed_threshold": 2.25,
        "sample_step_px": 1.5,
    }
    resolved_options.update(options or {})
    proposals = detect_local_pixel_trail_proposals(
        image,
        valid,
        background,
        noise,
        fwhm,
        **resolved_options,
    )
    request = {
        "image": image.tolist(),
        "valid_mask": valid.astype(np.uint8).tolist(),
        "background": background.tolist() if isinstance(background, np.ndarray) else background,
        "noise": noise.tolist() if isinstance(noise, np.ndarray) else noise,
        "psf_fwhm": fwhm,
        **resolved_options,
    }
    return {
        "label": label,
        "request": request,
        "expected": {"proposals": [proposal.as_dict() for proposal in proposals]},
    }


def capture_error(label: str, request: dict[str, object]) -> dict[str, object]:
    arguments = dict(request)
    image = arguments.pop("image")
    valid = arguments.pop("valid_mask")
    background = arguments.pop("background")
    noise = arguments.pop("noise")
    fwhm = arguments.pop("psf_fwhm")
    try:
        detect_local_pixel_trail_proposals(image, valid, background, noise, fwhm, **arguments)
    except (TypeError, ValueError) as error:
        return {"label": label, "request": request, "expected_error": str(error)}
    raise AssertionError(f"expected invalid request to fail: {label}")


def main() -> None:
    cases = [
        capture("isolated_horizontal_exposure_line", lines=[(32.0, 30.0, 18.0, 0.0, 7.0)], seed=11),
        capture("diagonal_line_refines_orientation", lines=[(31.0, 32.0, 20.0, 32.0, 7.5)], seed=12),
        capture("near_vertical_line_uses_unoriented_angle", lines=[(30.0, 32.0, 18.0, 83.0, 7.0)], seed=13),
        capture("reverse_axis_angle_is_equivalent", lines=[(32.0, 32.0, 20.0, 142.0, 7.0)], seed=14),
        capture("two_separated_lines_keep_distinct_geometry", lines=[(20.0, 22.0, 16.0, 15.0, 7.0), (44.0, 42.0, 18.0, 105.0, 7.5)], seed=15),
        capture("single_compact_point_is_rejected", points=[(32.0, 32.0, 14.0)], seed=16),
        capture("compact_double_source_is_negative_control", points=[(28.5, 31.5, 11.0), (35.5, 32.0, 10.0)], seed=17),
        capture("weak_extended_line_below_final_response", lines=[(32.0, 32.0, 16.0, 55.0, 3.6)], seed=18),
        capture("masked_line_core_uses_local_support_normalization", lines=[(32.0, 31.0, 20.0, 8.0, 8.0)], mask_slices=[(slice(29, 34), slice(30, 34))], seed=19),
        capture("edge_clipped_line_respects_coverage_gate", lines=[(8.0, 30.0, 20.0, 0.0, 8.0)], seed=20),
        capture("per_pixel_background_and_noise_fields", lines=[(34.0, 31.0, 19.0, 70.0, 8.0)], variable_fields=True, seed=21),
        capture("point_source_plus_extended_line", lines=[(32.0, 31.0, 18.0, 30.0, 7.5)], points=[(19.0, 45.0, 12.0)], seed=22),
        capture("cosmic_ray_near_line_does_not_replace_local_model", lines=[(31.0, 32.0, 20.0, 42.0, 7.0)], points=[(48.0, 15.0, 35.0)], seed=23),
        capture("empty_noise_field_returns_no_proposals", seed=24),
        capture("custom_longer_sampling_step", lines=[(32.0, 32.0, 20.0, 24.0, 8.0)], options={"sample_step_px": 2.0}, seed=25),
        capture("smaller_output_budget_preserves_best_candidate", lines=[(20.0, 20.0, 18.0, 20.0, 8.0), (44.0, 43.0, 18.0, 90.0, 8.0)], options={"max_proposals": 1}, seed=26),
        capture("default_length_bank_path", lines=[(32.0, 31.0, 24.0, 65.0, 8.0)], options={"length_bank_px": None}, seed=27),
        capture("seed_cap_keeps_local_spatial_coverage", lines=[(18.0, 18.0, 16.0, 0.0, 10.0), (262.0, 262.0, 16.0, 90.0, 10.0)], size=272, options={"length_bank_px": [8.0, 12.0, 16.0], "max_seeds": 2}, seed=28),
        capture("fwhm_changes_smoothing_and_template_width", lines=[(32.0, 32.0, 20.0, 120.0, 8.0)], fwhm=3.4, seed=29),
    ]
    base = cases[0]["request"]
    errors = []
    for label, key, value in [
        ("image_short_side_is_rejected", "image", base["image"][:7]),
        ("psf_below_discrete_sampling_floor_is_rejected", "psf_fwhm", 0.4),
        ("psf_larger_than_half_short_side_is_rejected", "psf_fwhm", 33.0),
        ("empty_valid_mask_is_rejected", "valid_mask", [[False] * 64 for _ in range(64)]),
        ("nonpositive_noise_is_rejected", "noise", 0.0),
        ("mask_shape_mismatch_is_rejected", "valid_mask", [row[:-1] for row in base["valid_mask"]]),
        ("background_field_shape_mismatch_is_rejected", "background", [[80.0] * 63 for _ in range(64)]),
        ("zero_angle_step_is_rejected", "angle_step_deg", 0.0),
        ("angle_step_above_180_is_rejected", "angle_step_deg", 181.0),
        ("nonnumeric_angle_step_is_rejected", "angle_step_deg", "invalid"),
        ("zero_output_budget_is_rejected", "max_proposals", 0),
        ("boolean_seed_budget_is_rejected", "max_seeds", True),
        ("zero_seed_threshold_is_rejected", "seed_threshold", 0.0),
        ("negative_sample_step_is_rejected", "sample_step_px", -1.0),
        ("empty_length_bank_is_rejected", "length_bank_px", []),
        ("nonpositive_length_is_rejected", "length_bank_px", [-2.0]),
        ("all_lengths_outside_image_are_rejected", "length_bank_px", [65.0]),
    ]:
        invalid = copy.deepcopy(base)
        invalid[key] = value
        errors.append(capture_error(label, invalid))

    source = ROOT / "src/rst19/single_frame_trails.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
        "errors": errors,
    }
    path = Path(__file__).parent / "tests/fixtures/local_pixel_trail_proposals.json"
    path.write_text(
        json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path} ({len(cases)} image cases)")


if __name__ == "__main__":
    main()
