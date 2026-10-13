"""Freeze an end-to-end pair-repair result from the Python implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.detection import Detection
from rst19.fits import FitsFrame
from rst19.pair_repair_counterfactual import run_pair_repair_counterfactual


def image_values() -> np.ndarray:
    yy, xx = np.indices((96, 96), dtype=np.float64)
    pattern = ((37 * xx.astype(np.int64) + 101 * yy.astype(np.int64)) % 17 - 8) * 0.25
    image = 100.0 + pattern
    sigma = 2.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    for cx, cy, amplitude in ((40.0, 40.0, 800.0), (50.0, 42.0, 720.0)):
        image += amplitude * np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / sigma**2))
    image[41, 44] = 900.0
    image[43, 46] = -250.0
    return image


def target(detection_id: int, x: float, y: float, peak_x: int, peak_y: int) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=800.0,
        flux=2000.0,
        background=100.0,
        noise=1.0,
        snr=100.0,
        fwhm=2.0,
        flags=(),
        peak_x=peak_x,
        peak_y=peak_y,
        repeated_code_values=(900,),
    )


def main() -> None:
    frame = FitsFrame(
        path=Path("<FRAME>"),
        header={},
        data=image_values(),
        auxiliary=None,
        data_offset=0,
    )
    result = run_pair_repair_counterfactual(
        frame,
        target(1, 40.0, 40.0, 40, 40),
        target(2, 50.0, 42.0, 50, 42),
        patch_padding_px=18,
        replacement_radius_px=1,
        target_match_radius_px=5.0,
        repeated_code_values=None,
        negative_anomaly_threshold_adu=-200.0,
        threshold_sigma=4.0,
        min_distance=4,
        aperture_radius=3,
        psf_fwhm=2.0,
        background_box_size=16,
        min_flux_snr=5.0,
        min_psf_support_pixels=3,
        proposal_mode="hybrid",
    )
    payload = result.as_dict()
    fixture = {
        "source_hashes": {
            "detection.py": hashlib.sha256((ROOT / "src/rst19/detection.py").read_bytes()).hexdigest(),
            "pair_repair_counterfactual.py": hashlib.sha256(
                (ROOT / "src/rst19/pair_repair_counterfactual.py").read_bytes()
            ).hexdigest(),
        },
        "expected": payload,
    }
    output = ROOT / "backend/tests/fixtures/pair_repair_counterfactual.json"
    output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
