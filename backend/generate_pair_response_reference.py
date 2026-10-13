"""Freeze an end-to-end pair-response result from the Python baseline."""
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
from rst19.pair_response_attribution import run_pair_response_attribution


def source(detection_id: int, x: float, peak_x: int, codes: tuple[int, ...]) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=30.0,
        peak=100.0,
        flux=500.0,
        background=10.0,
        noise=2.0,
        snr=30.0,
        fwhm=2.0,
        flags=(),
        flux_snr=20.0,
        peak_x=peak_x,
        peak_y=30,
        repeated_code_values=codes,
    )


def main() -> None:
    image = np.full((60, 60), 10.0, dtype=np.float64)
    image[30, 30] = 130.0
    image[30, 31] = 99.0
    image[30, 29] = -80.0
    image[31, 30] = 70.0
    frame = FitsFrame(
        path=Path("<FRAME>"),
        header={},
        data=image,
        auxiliary=None,
        data_offset=0,
    )
    result = run_pair_response_attribution(
        frame,
        source(1, 31.0, 31, (99,)),
        source(2, 28.0, 28, (99,)),
        patch_padding_px=8,
        psf_fwhm=2.0,
        negative_anomaly_threshold_adu=-50.0,
        maximum_neighborhood_px=3,
        maximum_count=12,
    )
    payload = {
        "source_hashes": {
            "detection.py": hashlib.sha256((ROOT / "src/rst19/detection.py").read_bytes()).hexdigest(),
            "pair_response_attribution.py": hashlib.sha256(
                (ROOT / "src/rst19/pair_response_attribution.py").read_bytes()
            ).hexdigest(),
        },
        "expected": result.as_dict(),
    }
    output = ROOT / "backend/tests/fixtures/pair_response_attribution.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
