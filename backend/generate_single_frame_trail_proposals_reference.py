from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.single_frame_trails import detect_single_frame_trail_proposals  # noqa: E402


def base_image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(100.0, 1.0, size=(96, 96)).astype(np.float32)


def add_line(
    image: np.ndarray,
    *,
    center: tuple[float, float],
    length: float,
    angle_deg: float,
    level: float = 3.2,
) -> None:
    yy, xx = np.indices(image.shape, dtype=np.float64)
    angle = np.deg2rad(angle_deg)
    sigma = 2.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    for offset in np.arange(-0.5 * length, 0.5 * length + 0.25, 0.5):
        sx = center[0] + offset * np.cos(angle)
        sy = center[1] + offset * np.sin(angle)
        image += (level * np.exp(-0.5 * (((xx - sx) ** 2 + (yy - sy) ** 2) / sigma**2))).astype(
            image.dtype,
            copy=False,
        ) * 0.5


def add_point(image: np.ndarray, center: tuple[float, float], amplitude: float) -> None:
    yy, xx = np.indices(image.shape, dtype=np.float64)
    sigma = 2.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    image += (amplitude * np.exp(-0.5 * (((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / sigma**2))).astype(
        image.dtype,
        copy=False,
    )


def request_for(
    image: np.ndarray,
    *,
    default_lengths: bool = False,
    valid_mask: np.ndarray | None = None,
    background: Any = 100.0,
    noise: Any = 1.0,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "image": image.astype(np.float64).tolist(),
        "valid_mask": (np.ones(image.shape, dtype=bool) if valid_mask is None else valid_mask).tolist(),
        "background": np.asarray(background).tolist() if np.ndim(background) else background,
        "noise": np.asarray(noise).tolist() if np.ndim(noise) else noise,
        "psf_fwhm": 2.0,
        "angle_step_deg": 10.0,
        "max_proposals": 128,
    }
    if not default_lengths:
        request["length_bank_px"] = [4.0, 6.0, 8.0, 12.0, 18.0, 28.0, 40.0]
    return request


def build_fixture() -> dict[str, Any]:
    single = base_image(1901)
    add_line(single, center=(28.0, 48.0), length=24.0, angle_deg=28.0)

    multi = base_image(1902)
    add_line(multi, center=(24.0, 24.0), length=8.0, angle_deg=23.0, level=4.0)
    add_line(multi, center=(70.0, 70.0), length=28.0, angle_deg=113.0)

    negative = base_image(1903)
    add_point(negative, (76.0, 20.0), 28.0)
    add_point(negative, (70.0, 72.0), 24.0)
    add_point(negative, (78.0, 72.0), 20.0)

    rng = np.random.default_rng(1904)
    yy, xx = np.indices((96, 96), dtype=np.float64)
    background = 100.0 + 0.015 * xx - 0.01 * yy
    noise = 0.8 + 0.004 * xx
    masked = (background + noise * rng.normal(size=(96, 96))).astype(np.float32)
    add_line(masked, center=(19.0, 42.0), length=18.0, angle_deg=43.0)
    valid_mask = np.ones(masked.shape, dtype=bool)
    valid_mask[:4, :] = False
    valid_mask[:, :5] = False
    valid_mask[36:41, 48:] = False

    case_inputs = [
        ("single_diagonal_line", request_for(single)),
        ("short_and_long_lines", request_for(multi)),
        ("point_and_double_negative_controls", request_for(negative)),
        (
            "spatial_background_noise_and_mask",
            request_for(masked, valid_mask=valid_mask, background=background, noise=noise),
        ),
        ("generated_default_length_bank", request_for(single, default_lengths=True)),
    ]
    cases = []
    for label, request in case_inputs:
        proposals = detect_single_frame_trail_proposals(
            request["image"],
            valid_mask=request["valid_mask"],
            background=request["background"],
            noise=request["noise"],
            psf_fwhm=request["psf_fwhm"],
            **{key: request[key] for key in ("length_bank_px", "angle_step_deg", "max_proposals") if key in request},
        )
        cases.append({"label": label, "request": request, "expected": {"proposals": [item.as_dict() for item in proposals]}})

    tiny = {
        "image": [[100.0] * 8 for _ in range(8)],
        "valid_mask": [[True] * 8 for _ in range(8)],
        "background": 100.0,
        "noise": 1.0,
        "psf_fwhm": 2.0,
    }
    errors = []
    for label, overrides in [
        ("angle_step_zero", {"angle_step_deg": 0.0}),
        ("proposal_limit_too_large", {"max_proposals": 100_001}),
        ("length_exceeds_image", {"length_bank_px": [8.0]}),
    ]:
        request = {**tiny, **overrides}
        try:
            detect_single_frame_trail_proposals(
                request["image"],
                valid_mask=request["valid_mask"],
                background=request["background"],
                noise=request["noise"],
                psf_fwhm=request["psf_fwhm"],
                **{key: request[key] for key in ("length_bank_px", "angle_step_deg", "max_proposals") if key in request},
            )
        except (TypeError, ValueError) as exc:
            errors.append({"label": label, "request": request, "expected_error": str(exc)})
        else:
            raise AssertionError(f"error fixture {label} did not fail")

    source = ROOT / "src" / "rst19" / "single_frame_trails.py"
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": cases,
        "errors": errors,
    }


if __name__ == "__main__":
    target = ROOT / "backend" / "tests" / "fixtures" / "single_frame_trail_proposals.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_fixture(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(target)
