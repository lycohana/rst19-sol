"""Verify the native sequence per-frame motion-component operation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from verify_detection import compare


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "tests/fixtures/sequence_motion_components.json"


def invoke(engine: Path, request: dict[str, object]) -> tuple[dict[str, object], int]:
    response = subprocess.run(
        [str(engine)],
        input=json.dumps(request, ensure_ascii=False, allow_nan=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
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
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for name, expected in fixture["source_hashes"].items():
        actual = hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        assert actual == expected, f"Python baseline changed: {name}"

    successful = []
    for case in fixture["cases"]:
        event, return_code = invoke(
            engine,
            {"operation": "sequence.motion_feature_frame_components", **case["request"]},
        )
        assert return_code == 0 and event["ok"], (case["label"], event, return_code)
        compare(event["payload"], case["expected"])
        successful.append({"label": case["label"], "all_fields_compared": True})

    errors = []
    for case in fixture["errors"]:
        event, return_code = invoke(
            engine,
            {"operation": "sequence.motion_feature_frame_components", **case["request"]},
        )
        assert return_code != 0 and not event["ok"], (case["label"], event, return_code)
        assert event["error"] == case["expected_error"], (case["label"], event["error"])
        errors.append({"label": case["label"], "error_compared": True})

    report = {
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "python_source_hashes": fixture["source_hashes"],
        "native_requests_verified": len(successful) + len(errors),
        "successful_cases": successful,
        "error_cases": errors,
        "all_fields_compared": True,
        "scope": "single-frame threshold residual components, optional one-pixel gap grouping, PCA line geometry, edge and artifact audit; outputs remain unconfirmed image-plane proposals",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
