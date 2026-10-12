"""Verify the Rust local-pixel trail proposal detector against Python fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from verify_detection import compare

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    engine = args.engine.resolve()
    fixture_path = Path(__file__).parent / "tests/fixtures/local_pixel_trail_proposals.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for name, expected in fixture["source_hashes"].items():
        actual = hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        assert actual == expected, f"Python baseline changed: {name}"
    records = []
    for case in fixture["cases"]:
        request = {"operation": "trails.local_pixel_proposals", **case["request"]}
        response = subprocess.run(
            [str(engine)],
            input=json.dumps(request, ensure_ascii=False, allow_nan=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        events = [json.loads(line) for line in response.stdout.splitlines()]
        assert response.returncode == 0 and len(events) == 1 and events[0]["ok"], (
            case["label"], response.returncode, response.stdout, response.stderr
        )
        compare(events[0]["payload"], case["expected"])
        records.append({"label": case["label"], "proposals": len(case["expected"]["proposals"]), "all_fields_compared": True})
    error_records = []
    for case in fixture["errors"]:
        request = {"operation": "trails.local_pixel_proposals", **case["request"]}
        response = subprocess.run(
            [str(engine)],
            input=json.dumps(request, ensure_ascii=False, allow_nan=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        events = [json.loads(line) for line in response.stdout.splitlines()]
        assert response.returncode != 0 and len(events) == 1 and not events[0]["ok"], (
            case["label"], response.returncode, response.stdout, response.stderr
        )
        assert events[0]["error"] == case["expected_error"], (
            case["label"], events[0]["error"], case["expected_error"]
        )
        error_records.append({"label": case["label"], "error_compared": True})
    report = {
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "python_source_hashes": fixture["source_hashes"],
        "native_requests_verified": len(records) + len(error_records),
        "records": records,
        "error_cases": error_records,
        "all_fields_compared": True,
        "scope": "single-frame local-pixel finite-line proposals with masked Gaussian whitening, spatially tiled peak seeds, length/angle center scans, local refinement, compact point/double negative controls, and geometric NMS; proposals are not motion classifications",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
