"""Compare the native FITS format audit and artifacts with the Python baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from verify_detection import compare
from verify_fits import card
from rst19.format_audit import run_format_audit, write_format_audit_artifacts


def auxiliary_bytes(values: list[float]) -> bytes:
    raw = bytearray(struct.pack("<26d", *values))
    for index in range(0, len(raw), 2):
        raw[index], raw[index + 1] = raw[index + 1], raw[index]
    return bytes(raw)


def write_frame(path: Path, timestamp: str, j2000: tuple[float, float, float], wgs84: tuple[float, float, float]) -> None:
    width, height = 104, 3
    pixels = np.arange(-(width * height) // 2, (width * height) // 2, dtype=np.int16).reshape(height, width)
    flat = pixels.reshape(-1)
    flat[120], flat[121], flat[122], flat[123] = -32768, 32767, -1, 0
    data = bytearray(pixels.astype(">i2").tobytes())
    aux = [0.0] * 26
    aux[:4] = [270.0, 0.0, 178.25, -33.5]
    aux[4:7] = j2000
    aux[7:10] = [1.0, 2.0, 3.0]
    aux[10] = 1.0
    aux[20:23] = wgs84
    aux[23:26] = [4.0, 5.0, 6.0]
    data[:208] = auxiliary_bytes(aux)
    header = b"".join([
        card("SIMPLE", "T"), card("BITPIX", "16"), card("NAXIS", "2"),
        card("NAXIS1", str(width)), card("NAXIS2", str(height)),
        card("EXPTIME", "2.5"), card("BSCALE", "2.0"), card("BZERO", "1.0"),
        card("BLANK", "-999"), card("DATE-OBS", f"'{timestamp}'"),
        card("CRPIX1", "8.0"), card("CRPIX2", "8.0"), card("CTYPE1", "'RA---TAN'"),
        b"END".ljust(80),
    ]).ljust(2880, b" ")
    path.write_bytes(header + data)


def run_worker(worker: Path, input_path: Path, output: Path, extreme_fraction: float) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    request = {
        "operation": "tool",
        "tool": "rst19-format-audit",
        "args": [str(input_path.resolve()), "--extreme-fraction", str(extreme_fraction),
                 "--out-dir", str(artifacts.resolve())],
        "out_dir": str(output.resolve()),
    }
    process = subprocess.run([str(worker.resolve())], input=json.dumps(request), capture_output=True,
                             text=True, encoding="utf-8", timeout=60)
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode == 0 and len(events) == 1 and events[0]["event"] == "done" \
        and events[0]["ok"], (process.returncode, process.stdout, process.stderr)
    return events[0]["payload"]["result"]


def verify(engine: Path, cli: Path, out: Path, real_frame: Path | None = None,
           worker: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "fits"
    inputs.mkdir()
    write_frame(inputs / "frame-a.fits", "2026-03-30T16:32:05+02:00", (100.0, 200.0, 300.0), (1000.0, 2000.0, 3000.0))
    write_frame(inputs / "frame-b.fits", "2026-03-30T14:32:07Z", (102.0, 204.0, 306.0), (1008.0, 2010.0, 3012.0))
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs.iterdir()}
    expected = run_format_audit(inputs, extreme_fraction=0.9).as_dict()
    request = {"operation": "audit.format_paths", "input_path": str(inputs.resolve()), "extreme_fraction": 0.9}
    process = subprocess.run([str(engine.resolve())], input=json.dumps(request), capture_output=True,
                             text=True, encoding="utf-8", timeout=60)
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode == 0 and len(events) == 1 and events[0]["ok"], (process.returncode, process.stdout, process.stderr)
    compare(events[0]["payload"], expected)

    python_out, native_out = out / "python-artifacts", out / "native-artifacts"
    write_format_audit_artifacts(run_format_audit(inputs, extreme_fraction=0.9), python_out)
    cli_run = subprocess.run([str(cli.resolve()), str(inputs.resolve()), "--extreme-fraction", "0.9",
                              "--out-dir", str(native_out)], capture_output=True, text=True,
                             encoding="utf-8", timeout=60)
    assert cli_run.returncode == 0, (cli_run.returncode, cli_run.stdout, cli_run.stderr)
    compare(json.loads((native_out / "format_audit.json").read_text(encoding="utf-8")), expected)
    for name in ("format_audit_frames.csv", "format_audit_velocity.csv"):
        assert (native_out / name).read_bytes() == (python_out / name).read_bytes(), name
    if worker is not None:
        worker_result = run_worker(worker, inputs, out / "worker-synthetic", 0.9)
        compare(worker_result, expected)
        for name in ("format_audit_frames.csv", "format_audit_velocity.csv"):
            assert ((out / "worker-synthetic" / "artifacts" / name).read_bytes()
                    == (python_out / name).read_bytes()), name
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs.iterdir()}
    assert before == after, "original FITS changed"
    real_record = None
    if real_frame is not None:
        real_before = hashlib.sha256(real_frame.read_bytes()).hexdigest()
        real_expected = run_format_audit(real_frame).as_dict()
        real_request = {"operation": "audit.format_paths", "input_path": str(real_frame.resolve()), "extreme_fraction": 0.9}
        real_process = subprocess.run([str(engine.resolve())], input=json.dumps(real_request), capture_output=True,
                                      text=True, encoding="utf-8", timeout=60)
        real_events = [json.loads(line) for line in real_process.stdout.splitlines()]
        assert real_process.returncode == 0 and len(real_events) == 1 and real_events[0]["ok"], (
            real_process.returncode, real_process.stdout, real_process.stderr
        )
        compare(real_events[0]["payload"], real_expected)
        real_python_out, real_native_out = out / "python-real-artifacts", out / "native-real-artifacts"
        write_format_audit_artifacts(run_format_audit(real_frame), real_python_out)
        real_cli = subprocess.run([str(cli.resolve()), str(real_frame.resolve()), "--out-dir", str(real_native_out)],
                                  capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert real_cli.returncode == 0, (real_cli.returncode, real_cli.stdout, real_cli.stderr)
        compare(json.loads((real_native_out / "format_audit.json").read_text(encoding="utf-8")), real_expected)
        for name in ("format_audit_frames.csv", "format_audit_velocity.csv"):
            assert (real_native_out / name).read_bytes() == (real_python_out / name).read_bytes(), name
        if worker is not None:
            real_worker_result = run_worker(worker, real_frame, out / "worker-real", 0.9)
            compare(real_worker_result, real_expected)
            for name in ("format_audit_frames.csv", "format_audit_velocity.csv"):
                assert ((out / "worker-real" / "artifacts" / name).read_bytes()
                        == (real_python_out / name).read_bytes()), name
        real_after = hashlib.sha256(real_frame.read_bytes()).hexdigest()
        assert real_before == real_after, "real FITS changed"
        real_record = {"path": str(real_frame.resolve()), "sha256": real_before, "all_fields_compared": True,
                       "csv_artifacts_byte_identical": True, "original_unchanged": True}
    report = {
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest() if worker is not None else None,
        "python_source_sha256": hashlib.sha256((ROOT / "src/rst19/format_audit.py").read_bytes()).hexdigest(),
        "frame_count": expected["summary"]["frame_count"],
        "velocity_rows": len(expected["velocity"]),
        "forward_and_backward_velocity_fields_compared": True,
        "result_json_fields_compared": True,
        "csv_artifacts_byte_identical": True,
        "desktop_worker_parity": worker is not None,
        "fits_sha256_before_after": before,
        "real_frame": real_record,
        "scope": "synthetic two-frame FITS storage and telemetry audit; not a physical-unit or WCS validation",
    }
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-frame", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.engine, args.cli, args.out, args.real_frame, args.worker),
                     ensure_ascii=False, indent=2))
