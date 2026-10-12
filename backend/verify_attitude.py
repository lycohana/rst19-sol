"""Compare auxiliary attitude audits through the native engine process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from verify_detection import compare
sys.path.insert(0, str(ROOT / "src"))
from rst19.attitude import (
    audit_auxiliary_boresight,
    summarize_auxiliary_boresight,
    write_auxiliary_boresight_artifacts,
)


def card(key: str, value: object) -> bytes:
    if isinstance(value, bool):
        encoded = "T" if value else "F"
    elif isinstance(value, str):
        encoded = "'" + value + "'"
    else:
        encoded = str(value)
    return f"{key:<8}= {encoded:>20}".ljust(80).encode("ascii")


def write_fits(path: Path, frame: dict[str, object]) -> None:
    header_values = frame["header"]
    cards = [
        card("SIMPLE", True),
        card("BITPIX", 16),
        card("NAXIS", 2),
        card("NAXIS1", 16),
        card("NAXIS2", 16),
    ]
    for key in ("AZIMUTH", "ELEVATIO", "DATE-OBS"):
        if key in header_values:
            cards.append(card(key, header_values[key]))
    cards.append(b"END".ljust(80))
    header = b"".join(cards).ljust(2880, b" ")
    data = bytearray(np.zeros(16 * 16, dtype=">i2").tobytes())
    values = frame["auxiliary_values"]
    if values is not None:
        auxiliary = bytearray(struct.pack("<26d", *values))
        for index in range(0, len(auxiliary), 2):
            auxiliary[index], auxiliary[index + 1] = auxiliary[index + 1], auxiliary[index]
        data[: len(auxiliary)] = auxiliary
    path.write_bytes(header + data)


def run_request(engine: Path, request: dict[str, object], expected: dict[str, object], label: str) -> dict[str, object]:
    response = subprocess.run(
        [str(engine.resolve())],
        input=json.dumps(request, ensure_ascii=False, allow_nan=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    event = json.loads(response.stdout)
    assert response.returncode == 0 and event["ok"], (label, event, response.stderr)
    compare(event["payload"], expected)
    return {"label": label, "operation": request["operation"], "all_fields_compared": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture_path = Path(__file__).parent / "tests/fixtures/attitude.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for name, digest in fixture["source_hashes"].items():
        actual = hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        assert actual == digest, f"Python baseline changed: {name}"
    records = []
    for case in fixture["cases"]:
        request = {
            "operation": "attitude.audit.frames",
            "frames": case["frames"],
            "camera_body_axis": case["camera_body_axis"],
        }
        records.append(run_request(args.engine, request, case["expected"], case["label"]))
    case = fixture["cases"][0]
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for index, frame in enumerate(case["frames"]):
            path = Path(directory) / f"frame-{index:02}.fits"
            write_fits(path, frame)
            paths.append(path)
        rows = audit_auxiliary_boresight(paths)
        expected = {
            "schema_version": 1,
            "summary": summarize_auxiliary_boresight(rows),
            "frames": [row.as_dict() for row in rows],
        }
        records.append(
            run_request(
                args.engine,
                {"operation": "attitude.audit.paths", "paths": [str(path) for path in paths]},
                expected,
                "synthetic_fits_paths",
            )
        )
        python_output = Path(directory) / "python-artifacts"
        native_output = Path(directory) / "native-artifacts"
        write_auxiliary_boresight_artifacts(rows, python_output)
        cli_result = subprocess.run(
            [
                str(args.cli.resolve()),
                str(paths[0].parent),
                "--out-dir",
                str(native_output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        assert cli_result.returncode == 0, (cli_result.stdout, cli_result.stderr)
        native_json = json.loads((native_output / "auxiliary_boresight_audit.json").read_text(encoding="utf-8"))
        compare(native_json, expected)
        python_csv = (python_output / "auxiliary_boresight_audit.csv").read_bytes()
        native_csv = (native_output / "auxiliary_boresight_audit.csv").read_bytes()
        assert native_csv == python_csv, "native attitude CSV differs from Python bytes"
        records.append(
            {
                "label": "native_cli_csv_json_artifacts",
                "operation": "rst19-aux-audit",
                "all_fields_compared": True,
                "csv_bytes_identical": True,
                "json_all_fields_compared": True,
            }
        )
    report = {
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "python_source_hashes": fixture["source_hashes"],
        "native_requests_verified": len(records),
        "records": records,
        "all_fields_compared": True,
        "scope": "auxiliary attitude quaternion/boresight audit, FITS path reads, and parity-checked CSV/JSON CLI artifacts; no WCS or physical pointing claim",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
