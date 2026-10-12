"""Compare supplemental fast quality-point tracks through the native JSONL process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from verify_detection import compare

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture_path = Path(__file__).parent / "tests/fixtures/sequence_fast.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for name, expected in fixture["source_hashes"].items():
        actual = hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        assert actual == expected, f"Python baseline changed: {name}"

    records = []
    for case in fixture["cases"]:
        request = {
            "operation": "sequence.fast_point_movers",
            "frames": case["frames"],
            "cumulative_shifts": case["cumulative_shifts"],
            "frame_times": case["frame_times"],
            "options": case["options"],
            "existing_tracks": case["existing_tracks"],
        }
        response = subprocess.run(
            [str(args.engine.resolve())],
            input=json.dumps(request, ensure_ascii=False, allow_nan=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        events = [json.loads(line) for line in response.stdout.splitlines()]
        assert response.returncode == 0 and len(events) == 1 and events[0]["ok"], (
            case["label"], response.returncode, response.stdout, response.stderr
        )
        compare(events[0]["payload"], case["expected"])
        records.append({"label": case["label"], "all_fields_compared": True})

    report = {
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "python_source_hashes": fixture["source_hashes"],
        "native_requests_verified": len(records),
        "records": records,
        "all_fields_compared": True,
        "scope": "supplemental association of high-SNR quality point detections, existing-track reservation, elapsed-time motion fit and one-frame-gap handling",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
