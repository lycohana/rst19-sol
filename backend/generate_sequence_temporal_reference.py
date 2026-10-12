"""Freeze Python temporal-reference candidate proposals for native parity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _temporal_reference_candidate_frames


def json_plane(values: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in np.asarray(values, dtype=np.float32)
    ]


def scene(*, center: tuple[int, int] = (20, 20), sigma: float = 1.15, amplitude: float = 180.0, seed: int = 10):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:40, :40]
    x, y = center
    background = rng.normal(100.0, 2.0, (40, 40)).astype(np.float32)
    return (background + amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))).astype(np.float32)


def capture(label, reference, images, options, *, shifts=None, parameters=None):
    frame_count = len(images)
    shifts = shifts or [(0.0, 0.0)] * frame_count
    parameters = parameters or [
        {
            "mask_zero_pixels": 0,
            "saturation_level": -1.0,
            "min_psf_support_pixels": 3,
        }
        for _ in images
    ]
    frame_analyses = [
        SimpleNamespace(
            frame=SimpleNamespace(data=np.asarray(image, dtype=np.float32)),
            detection=SimpleNamespace(
                background=100.0,
                noise=4.0,
                parameters=parameter,
            ),
        )
        for image, parameter in zip(images, parameters, strict=True)
    ]
    candidate_frames, candidate_count = _temporal_reference_candidate_frames(
        frame_analyses,
        shifts,
        reference,
        psf_fwhm=options.get("psf_fwhm", 3.0),
        min_candidate_snr=options.get("min_candidate_snr", 15.0),
        temporal_multiscale=options.get("temporal_multiscale", False),
    )
    return {
        "label": label,
        "request": {
            "reference": None if reference is None else json_plane(reference),
            "frames": [
                {
                    "image": json_plane(image),
                    "background": 100.0,
                    "noise": 4.0,
                    "parameters": parameter,
                }
                for image, parameter in zip(images, parameters, strict=True)
            ],
            "cumulative_shifts": [list(shift) for shift in shifts],
            "options": options,
        },
        "expected": {
            "candidate_peak_frames": [rows.tolist() for rows in candidate_frames],
            "candidate_count": candidate_count,
        },
    }


def main() -> None:
    baseline_reference = scene()
    baseline_frames = [scene(amplitude=160 + index * 3, seed=30 + index) for index in range(4)]
    cases = [
        capture(
            "median_reference_proposes_repeated_psf_candidate",
            baseline_reference,
            baseline_frames,
            {"psf_fwhm": 3.0, "min_candidate_snr": 15.0},
        )
    ]

    shifts = [(0.0, 0.0), (0.6, 0.0), (-0.6, 0.0), (0.0, 0.6)]
    shifted_centers = [(20, 20), (21, 20), (19, 20), (20, 21)]
    shifted_frames = [
        scene(center=center, amplitude=160 + index * 2, seed=40 + index)
        for index, center in enumerate(shifted_centers)
    ]
    cases.append(
        capture(
            "fractional_registration_shift_selects_neighbor_pixel",
            baseline_reference,
            shifted_frames,
            {"psf_fwhm": 3.0, "min_candidate_snr": 15.0},
            shifts=shifts,
        )
    )

    narrow_reference = scene(sigma=0.55, amplitude=150.0, seed=50)
    narrow_frames = [scene(sigma=0.55, amplitude=150 + index * 2, seed=60 + index) for index in range(4)]
    cases.append(
        capture(
            "multiscale_reference_recovers_narrow_candidate",
            narrow_reference,
            narrow_frames,
            {"psf_fwhm": 3.0, "min_candidate_snr": 15.0, "temporal_multiscale": True},
        )
    )

    empty_reference = np.random.default_rng(70).normal(100.0, 2.0, (40, 40)).astype(np.float32)
    empty_frames = [scene(amplitude=0.0, seed=80 + index) for index in range(4)]
    cases.append(
        capture(
            "noise_only_reference_has_no_temporal_candidates",
            empty_reference,
            empty_frames,
            {"psf_fwhm": 3.0, "min_candidate_snr": 15.0},
        )
    )

    masked_frames = [scene(amplitude=160 + index * 3, seed=90 + index) for index in range(4)]
    masked_frames[0][20, 20] = 0.0
    masked_parameters = [
        {
            "mask_zero_pixels": 1,
            "saturation_level": -1.0,
            "min_psf_support_pixels": 3,
        }
        for _ in masked_frames
    ]
    cases.append(
        capture(
            "masked_raw_frame_is_skipped_but_other_frames_keep_candidate",
            baseline_reference,
            masked_frames,
            {"psf_fwhm": 3.0, "min_candidate_snr": 15.0},
            parameters=masked_parameters,
        )
    )

    saturated_frames = [scene(amplitude=420.0, seed=100 + index) for index in range(4)]
    saturated_parameters = [
        {
            "mask_zero_pixels": 0,
            "saturation_level": 180.0 if index == 0 else -1.0,
            "min_psf_support_pixels": 3,
        }
        for index in range(4)
    ]
    cases.append(
        capture(
            "saturated_raw_frame_is_rejected",
            scene(amplitude=420.0, seed=105),
            saturated_frames,
            {"psf_fwhm": 3.0, "min_candidate_snr": 15.0},
            parameters=saturated_parameters,
        )
    )

    border_reference = scene(center=(8, 20), amplitude=220.0, seed=110)
    border_frames = [scene(center=(8, 20), amplitude=200.0 + index * 3, seed=120 + index) for index in range(4)]
    cases.append(
        capture(
            "near_edge_candidate_respects_patch_bounds",
            border_reference,
            border_frames,
            {"psf_fwhm": 3.0, "min_candidate_snr": 15.0},
        )
    )

    invalid_reference = baseline_reference.copy()
    invalid_reference[5, 5] = np.nan
    cases.append(
        capture(
            "invalid_reference_pixel_is_masked",
            invalid_reference,
            baseline_frames,
            {"psf_fwhm": 3.0, "min_candidate_snr": 15.0},
        )
    )

    payload = {
        "source_hashes": {
            "sequence.py": hashlib.sha256((ROOT / "src/rst19/sequence.py").read_bytes()).hexdigest()
        },
        "cases": cases,
    }
    destination = ROOT / "backend/tests/fixtures/sequence_temporal_reference.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {destination} with {len(cases)} cases")


if __name__ == "__main__":
    main()
