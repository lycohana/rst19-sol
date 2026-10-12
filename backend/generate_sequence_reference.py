"""Freeze Python registered-median and robust-coadd reference scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _registered_coadd_reference, _registered_median_reference


def json_plane(values):
    if values is None:
        return None
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in np.asarray(values, dtype=np.float32)
    ]


def image(seed: int, shape: tuple[int, int] = (24, 32)) -> np.ndarray:
    return np.random.default_rng(seed).normal(100.0, 3.0, shape).astype(np.float32)


def capture(label, mode, images, shifts, *, parameters=None, options=None):
    parameters = parameters or [
        {"mask_zero_pixels": 0, "saturation_level": -1.0} for _ in images
    ]
    options = options or {}
    frame_analyses = [
        SimpleNamespace(
            frame=SimpleNamespace(data=np.asarray(values, dtype=np.float32)),
            detection=SimpleNamespace(parameters=parameter),
        )
        for values, parameter in zip(images, parameters, strict=True)
    ]
    expected = (
        _registered_median_reference(frame_analyses, shifts, **options)
        if mode == "median"
        else _registered_coadd_reference(frame_analyses, shifts, **options)
    )
    return {
        "label": label,
        "operation": f"sequence.reference.{mode}",
        "request": {
            "frames": [
                {
                    "image": json_plane(values),
                    "background": 100.0,
                    "noise": 3.0,
                    "parameters": parameter,
                }
                for values, parameter in zip(images, parameters, strict=True)
            ],
            "cumulative_shifts": [list(shift) for shift in shifts],
            "options": options,
        },
        "expected": {"reference": json_plane(expected)},
    }


def main() -> None:
    direct_images = [image(10 + index) for index in range(4)]
    direct_images[0][5, 7] = 0.0
    direct_images[1][8, 9] = 500.0
    direct_parameters = [
        {"mask_zero_pixels": 1, "saturation_level": -1.0},
        {"mask_zero_pixels": 0, "saturation_level": 200.0},
        {"mask_zero_pixels": 0, "saturation_level": -1.0},
        {"mask_zero_pixels": 0, "saturation_level": -1.0},
    ]
    cases = [
        capture(
            "direct_median_honors_zero_mask_saturation_and_auxiliary_row",
            "median",
            direct_images,
            [(0.0, 0.0), (0.5, -0.25), (-0.75, 0.0), (0.0, 0.0)],
            parameters=direct_parameters,
        )
    ]

    shifted_images = [image(30 + index) for index in range(4)]
    cases.append(
        capture(
            "large_shift_uses_bilinear_registered_median",
            "median",
            shifted_images,
            [(0.0, 0.0), (1.25, -0.5), (-1.0, 0.75), (0.0, 1.2)],
        )
    )

    coadd_images = [image(50 + index) for index in range(5)]
    coadd_images[0][12, 16] = 2000.0
    cases.append(
        capture(
            "robust_coadd_rejects_single_frame_cosmic_ray",
            "coadd",
            coadd_images,
            [(0.0, 0.0)] * len(coadd_images),
        )
    )

    shifted_coadd_images = [image(70 + index) for index in range(4)]
    shifted_coadd_images[2][10, 12] = -100.0
    cases.append(
        capture(
            "shifted_robust_coadd_clips_transient_and_resamples",
            "coadd",
            shifted_coadd_images,
            [(0.0, 0.0), (1.1, 0.2), (-0.9, 1.0), (0.0, -1.25)],
            options={"clip_sigma": 2.5, "tile_rows": 16},
        )
    )

    cases.append(
        capture(
            "shape_mismatch_returns_no_reference",
            "median",
            [image(90), image(91, (25, 32)), image(92)],
            [(0.0, 0.0)] * 3,
        )
    )
    cases.append(
        capture(
            "single_frame_returns_no_reference",
            "coadd",
            [image(100)],
            [(0.0, 0.0)],
        )
    )

    payload = {
        "source_hashes": {
            "sequence.py": hashlib.sha256((ROOT / "src/rst19/sequence.py").read_bytes()).hexdigest()
        },
        "cases": cases,
    }
    destination = ROOT / "backend/tests/fixtures/sequence_reference.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination} with {len(cases)} cases")


if __name__ == "__main__":
    main()
