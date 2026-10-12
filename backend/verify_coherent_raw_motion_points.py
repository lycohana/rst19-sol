"""Verify the native coherent raw motion-point chain operation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from verify_detection import compare

ROOT = Path(__file__).resolve().parents[1]


def call(engine: Path, request: dict[str, object]) -> tuple[dict[str, object], int]:
    response = subprocess.run(
        [str(engine)],
        input=json.dumps(request, ensure_ascii=False, allow_nan=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    events = [json.loads(line) for line in response.stdout.splitlines()]
    assert len(events) == 1, (response.returncode, response.stdout, response.stderr)
    return events[0], response.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    engine = args.engine.resolve()
    fixture_path = Path(__file__).parent / "tests/fixtures/coherent_raw_motion_points.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for name, expected in fixture["source_hashes"].items():
        actual = hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        assert actual == expected, f"Python baseline changed: {name}"
    records = []
    for case in fixture["cases"]:
        request = {
            "operation": "sequence.coherent_raw_motion_point_indices",
            **case["request"],
        }
        event, return_code = call(engine, request)
        assert return_code == 0 and event["ok"], (case["label"], event, return_code)
        compare(event["payload"], case["expected"])
        records.append({"label": case["label"], "all_fields_compared": True})
    error_records = []
    for case in fixture["errors"]:
        request = {
            "operation": "sequence.coherent_raw_motion_point_indices",
            **case["request"],
        }
        event, return_code = call(engine, request)
        assert return_code != 0 and not event["ok"], (case["label"], event, return_code)
        assert event["error"] == case["expected_error"], (case["label"], event["error"])
        error_records.append({"label": case["label"], "error_compared": True})
    report = {
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "python_source_hashes": fixture["source_hashes"],
        "native_requests_verified": len(records) + len(error_records),
        "successful_cases": records,
        "error_cases": error_records,
        "all_fields_compared": True,
        "scope": "adjacent-frame raw line-proposal chains with elapsed-time and exposure-scaled displacement bounds, direction continuity, strong-shape admission, weak-point bridging, and exposure validation; outputs remain proposal evidence",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
