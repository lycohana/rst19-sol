"""Freeze a representative pair-flux covariance result from the Python baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.pair_flux_covariance import (
    ForcedStabilityFluxRow,
    run_pair_flux_covariance_audit,
)


def rows() -> tuple[ForcedStabilityFluxRow, ...]:
    target_a = (10.0, 20.0, 30.0, 40.0, 50.0)
    target_b = (5.0, 10.0, 15.0, 20.0, 25.0)
    controls = {
        1: (8.0, 7.0, 6.0, 5.0, 4.0),
        2: (2.0, 9.0, 3.0, 8.0, 4.0),
        3: (4.0, 1.0, 5.0, 2.0, 6.0),
        4: (9.0, 3.0, 8.0, 2.0, 7.0),
    }
    output = [
        ForcedStabilityFluxRow(82931, index, value, value + 1.0)
        for index, value in enumerate(target_a, 1)
    ]
    output.extend(
        ForcedStabilityFluxRow(82934, index, value, value + 1.0)
        for index, value in enumerate(target_b, 1)
    )
    for detection_id, values in controls.items():
        output.extend(
            ForcedStabilityFluxRow(detection_id, index, value, value + 0.5)
            for index, value in enumerate(values, 1)
        )
    return tuple(output)


def main() -> None:
    result = run_pair_flux_covariance_audit(rows(), 82931, 82934)
    payload = {
        "source_hashes": {
            "pair_flux_covariance.py": hashlib.sha256(
                (ROOT / "src/rst19/pair_flux_covariance.py").read_bytes()
            ).hexdigest(),
        },
        "expected": result.as_dict(),
    }
    output = ROOT / "backend/tests/fixtures/pair_flux_covariance.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
