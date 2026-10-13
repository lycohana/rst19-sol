"""Compare native photometric model-sensitivity results with Python."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from rst19.calibration_sensitivity import build_calibration_model_sensitivity
from rst19.catalog import CatalogSource
from rst19.detection import Detection
from rst19.matching import CatalogMatch
from rst19.photometry import instrumental_magnitude


def fixture() -> tuple[tuple[CatalogMatch, ...], tuple[Detection, ...], tuple[CatalogSource, ...]]:
    detections = tuple(
        Detection(
            detection_id=index,
            x=100.0 + index,
            y=200.0 + index,
            peak=200.0 + index,
            flux=100.0 + index * 8.0,
            background=10.0,
            noise=2.0,
            snr=30.0,
            flux_snr=30.0,
            fwhm=2.0,
            flags=(),
        )
        for index in range(12)
    )
    catalog = tuple(
        CatalogSource(
            str(index),
            10.0 + index * 0.01,
            20.0 + index * 0.01,
            magnitude=(
                instrumental_magnitude(detection.flux)
                + 20.0
                + 0.25 * (index / 5.0)
                + 0.04 * (index / 5.0) ** 2
            ),
            color=index / 5.0,
            color_name="BP-RP",
            photometric_system="Gaia Vega",
            photometric_band="G",
        )
        for index, detection in enumerate(detections)
    )
    matches = tuple(
        CatalogMatch(
            index,
            str(index),
            detection.x,
            detection.y,
            detection.x,
            detection.y,
            0.1,
            catalog[index].magnitude,
            catalog_color=catalog[index].color,
            catalog_color_name="BP-RP",
            photometric_system="Gaia Vega",
            photometric_band="G",
        )
        for index, detection in enumerate(detections)
    )
    return matches, detections, catalog


def compare(actual: Any, expected: Any, path: str = "result") -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: keys differ: {set(actual) ^ set(expected)}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: lengths differ: {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]")
        return
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        if not math.isclose(float(actual), float(expected), rel_tol=2e-10, abs_tol=2e-11):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def native(binary: Path, request: dict[str, Any]) -> tuple[dict[str, Any], int]:
    environment = {**os.environ, "PYTHONPATH": "", "PYTHONHOME": ""}
    environment["PATH"] = (
        str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32")
        if os.name == "nt"
        else "/usr/bin:/bin"
    )
    completed = subprocess.run(
        [str(binary)],
        input=json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=60,
    )
    lines = completed.stdout.splitlines()
    if not lines:
        raise RuntimeError(f"{binary.name} returned no JSON: {completed.stderr}")
    return json.loads(lines[-1]), completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    matches, detections, catalog = fixture()
    cases: list[dict[str, Any]] = []
    normal = [
        (
            "default-models",
            matches,
            detections,
            catalog,
            {
                "exposure_s": 1.0,
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "color_name": "BP-RP",
                "model_orders": [0, 1, 2],
                "min_snr": 5.0,
                "min_calibrators": 6,
            },
        ),
        (
            "unknown-photometric-metadata",
            matches,
            detections,
            catalog,
            {
                "exposure_s": 1.0,
                "photometric_system": "unknown",
                "photometric_band": "unknown",
                "color_name": "BP-RP",
                "model_orders": [0, 1, 2],
                "min_snr": 5.0,
                "min_calibrators": 6,
            },
        ),
        (
            "ordered-deduplicated-models-and-quality-rejection",
            matches,
            (replace(detections[0], flags=("EDGE",)), *detections[1:]),
            catalog,
            {
                "exposure_s": 1.5,
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "color_name": "BP-RP",
                "model_orders": [2, 1, 0, 1],
                "min_snr": 10.0,
                "min_calibrators": 6,
            },
        ),
    ]
    for name, current_matches, current_detections, current_catalog, options in normal:
        expected = build_calibration_model_sensitivity(
            current_matches,
            current_detections,
            current_catalog,
            **options,
        )
        request = {
            "operation": "photometry.calibration.sensitivity",
            "matches": [asdict(value) for value in current_matches],
            "detections": [asdict(value) for value in current_detections],
            "catalog": [asdict(value) for value in current_catalog],
            "options": options,
        }
        engine_event, engine_code = native(args.engine, request)
        worker_event, worker_code = native(args.worker, request)
        if engine_code != 0 or not engine_event.get("ok"):
            raise AssertionError(f"{name}: engine failed: {engine_event}")
        if worker_code != 0 or not worker_event.get("ok"):
            raise AssertionError(f"{name}: worker failed: {worker_event}")
        compare(engine_event["payload"], expected, name)
        compare(worker_event["payload"], expected, name + ".worker")
        compare(worker_event["payload"], engine_event["payload"], name + ".worker-engine")
        cases.append({"case": name, "status": expected["status"], "all_fields_match": True})

    errors: list[dict[str, Any]] = []
    for name, invalid in [
        ("invalid-model-order", {"model_orders": [3]}),
        ("too-few-calibrators", {"min_calibrators": 1}),
        ("nonpositive-exposure", {"exposure_s": 0.0}),
        ("nonpositive-min-snr", {"min_snr": 0.0}),
    ]:
        kwargs: dict[str, Any] = {
            "exposure_s": 1.0,
            "photometric_system": "Gaia Vega",
            "photometric_band": "G",
            "color_name": "BP-RP",
            "model_orders": [0, 1, 2],
            "min_snr": 5.0,
            "min_calibrators": 6,
        }
        kwargs.update(invalid)
        try:
            build_calibration_model_sensitivity(
                matches, detections, catalog, **kwargs
            )
        except ValueError as error:
            expected_error = str(error)
        else:
            raise AssertionError(f"{name}: Python accepted invalid options")
        request = {
            "operation": "photometry.calibration.sensitivity",
            "matches": [asdict(value) for value in matches],
            "detections": [asdict(value) for value in detections],
            "catalog": [asdict(value) for value in catalog],
            "options": kwargs,
        }
        engine_event, engine_code = native(args.engine, request)
        worker_event, worker_code = native(args.worker, request)
        if engine_code == 0 or engine_event.get("ok") or engine_event.get("error") != expected_error:
            raise AssertionError(f"{name}: engine error differs: {engine_event}")
        if worker_code == 0 or worker_event.get("ok") or worker_event.get("error") != expected_error:
            raise AssertionError(f"{name}: worker error differs: {worker_event}")
        errors.append({"case": name, "error_matches_python": True})

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        "cases": cases,
        "error_cases": errors,
        "numeric_tolerance": {"relative": 2e-10, "absolute": 2e-11},
        "all_fields_match": True,
    }
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
