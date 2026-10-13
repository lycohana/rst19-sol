"""Compare Rust stack-forced photometry with its frozen Python implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.sequence import _stack_forced_frame_measure_batch  # noqa: E402


PYTHON_BASELINE_SHA256 = "51fcabdb952c8aefa836877da6032d40b8ed91390ce471ec2841137acae54088"


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _compare(actual: object, expected: object, path: str) -> int:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), (path, "keys")
        return sum(_compare(actual[key], value, f"{path}.{key}") for key, value in expected.items())
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), (path, "length")
        return sum(_compare(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(actual, expected)))
    if isinstance(expected, bool) or expected is None or isinstance(expected, int):
        assert actual == expected, (path, actual, expected)
        return 1
    assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (path, actual)
    tolerance = 2e-6 + abs(float(expected)) * 2e-7
    assert abs(float(actual) - float(expected)) <= tolerance, (path, actual, expected, tolerance)
    return 1


def _case() -> tuple[dict[str, object], dict[str, object]]:
    rng = np.random.default_rng(77)
    image = rng.normal(20.0, 3.0, size=(96, 96)).astype(np.float32)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    for cx, cy, sigma, amplitude in (
        (24.0, 24.0, 1.2, 10.0),
        (50.0, 50.0, 1.2, 8.0),
        (60.0, 30.0, 1.2, 12.0),
        (36.25, 70.75, 1.0, 9.0),
    ):
        image += (amplitude * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma**2)).astype(np.float32)
    # A bright annulus contaminant exercises the three-pass robust clipping.
    image[50, 56] = 240.0
    image[5, 5] = np.nan
    image[80, 80] = 200.0
    image[79:82, 79:82] = np.maximum(image[79:82, 79:82], 90.0)
    mask = np.zeros(image.shape, dtype=bool)
    mask[80, 80] = True
    mask[6, 6] = True
    xs = np.asarray([24.0, 50.0, 60.0, 36.25, 0.5, -0.5, 80.0, np.nan, 95.0])
    ys = np.asarray([24.0, 50.0, 30.0, 70.75, 4.0, 10.0, 80.0, 12.0, 95.0])
    parameters = {
        "aperture_radius": 4,
        "min_psf_support_pixels": 3,
        "global_background": 20.0,
        "global_noise": 3.0,
    }
    arrays = _stack_forced_frame_measure_batch(
        image,
        mask,
        xs,
        ys,
        padded_image=np.pad(image.astype(np.float64), 8, mode="constant", constant_values=np.nan),
        padded_mask=np.pad(mask, 8, mode="constant", constant_values=True),
        **parameters,
    )
    request = {
        "image": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in image.astype(np.float64).tolist()
        ],
        "mask": mask.tolist(),
        "xs": [_finite_or_none(value) for value in xs],
        "ys": [_finite_or_none(value) for value in ys],
        **parameters,
    }
    expected = {
        "flux_snr": [_finite_or_none(value) for value in arrays[0]],
        "support": [int(value) for value in arrays[1]],
        "center_valid": [bool(value) for value in arrays[2]],
        "fwhm": [_finite_or_none(value) for value in arrays[3]],
        "ellipticity": [_finite_or_none(value) for value in arrays[4]],
    }
    return request, expected


def _invoke(
    executable: Path,
    request: dict[str, object],
    operation: str,
    *,
    disable_external_python_search: bool,
) -> tuple[dict[str, object], float]:
    environment = os.environ.copy()
    if disable_external_python_search:
        environment["PYTHONPATH"] = ""
        environment["PYTHONHOME"] = ""
        environment["PATH"] = (
            str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32")
            if os.name == "nt"
            else "/usr/bin:/bin"
        )
    started = time.perf_counter()
    response = subprocess.run(
        [str(executable)],
        input=json.dumps({"operation": operation, **request}, allow_nan=False, separators=(",", ":")) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=60,
        check=False,
    )
    elapsed = time.perf_counter() - started
    events = [json.loads(line) for line in response.stdout.splitlines()]
    assert response.returncode == 0 and len(events) == 1 and events[0].get("ok") is True, (
        executable,
        response.returncode,
        response.stdout,
        response.stderr,
    )
    return events[0]["payload"], elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--disable-external-python-search", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_path = ROOT / "src/rst19/sequence.py"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert source_hash == PYTHON_BASELINE_SHA256, ("Python baseline changed", source_hash)
    request, expected = _case()
    endpoints = [("engine", args.engine.resolve())]
    if args.worker:
        endpoints.append(("desktop_worker", args.worker.resolve()))
    records = []
    for label, executable in endpoints:
        actual, elapsed = _invoke(
            executable,
            request,
            "sequence.stack_forced_measure_batch",
            disable_external_python_search=args.disable_external_python_search,
        )
        fields = _compare(actual, expected, label)
        records.append(
            {
                "endpoint": label,
                "all_fields_compared": True,
                "scalar_values_compared": fields,
                "elapsed_seconds_including_startup_and_json": elapsed,
                "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            }
        )
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "python_source": "src/rst19/sequence.py::_stack_forced_frame_measure_batch",
        "python_source_sha256": source_hash,
        "synthetic_frame_shape": [96, 96],
        "source_positions_checked": len(request["xs"]),
        "external_python_search_disabled": args.disable_external_python_search,
        "coverage": [
            "integer and subpixel centers",
            "robust local-annulus clipping",
            "masked center and pixels",
            "non-finite image pixel",
            "image boundary and off-frame coordinates",
        ],
        "records": records,
        "all_fields_compared": True,
    }
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
