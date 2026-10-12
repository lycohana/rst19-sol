"""Exercise a packaged worker with no external Python on its search path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys


def make_fits(path: Path) -> None:
    def card(key: str, value: str) -> bytes:
        return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")
    header = b"".join([card("SIMPLE", "T"), card("BITPIX", "16"), card("NAXIS", "2"),
                       card("NAXIS1", "128"), card("NAXIS2", "128"), b"END".ljust(80)]).ljust(2880, b" ")
    rng = random.Random(19)
    pixels = []
    for y in range(128):
        for x in range(128):
            value = rng.gauss(1000, 8)
            for px, py in [(30, 40), (70, 80), (98, 50)]:
                value += 1000 * math.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * 1.4 ** 2))
            pixels.append(round(value))
    path.write_bytes(header + struct.pack(f">{len(pixels)}h", *pixels))


def make_feature_catalog(path: Path) -> None:
    path.write_bytes(
        b"\xef\xbb\xbf"
        + b"detection_id,feature_class,quality_passed,flags\n"
        + b"1,compact,true,wing|blend|wing\n"
        + b"2,compact,false,blend\n"
        + b"3,diffuse,yes,\n"
    )


def verify(package: Path, out: Path) -> dict:
    package = package.resolve()
    out.mkdir(parents=True, exist_ok=False)
    worker = package / "runtime" / ("rst19-worker.exe" if os.name == "nt" else "rst19-worker")
    native_worker = package / "runtime" / ("rst19-native-worker.exe" if os.name == "nt" else "rst19-native-worker")
    assert native_worker.is_file(), "Rust single-frame worker is missing from the package"
    environment = {**os.environ, "PYTHONPATH": "", "PYTHONHOME": "", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    environment["PATH"] = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32") if os.name == "nt" else "/usr/bin:/bin"
    catalog = subprocess.run([str(worker), "--catalog"], capture_output=True, text=True, encoding="utf-8",
                             env=environment, cwd=out, timeout=60, check=True)
    entry = json.loads(catalog.stdout)
    assert entry["ok"] and entry["payload"]["environment"]["frozen"]
    assert len(entry["payload"]["tools"]) == 78
    frame = out / "原始数据 with spaces.fits"
    make_fits(frame)
    original = hashlib.sha256(frame.read_bytes()).hexdigest()
    request = {"operation": "tool", "tool": "rst19", "args": [str(frame.resolve())], "out_dir": str((out / "job").resolve())}
    response = subprocess.run([str(native_worker)], input=json.dumps(request, ensure_ascii=False), capture_output=True,
                              text=True, encoding="utf-8", env=environment, cwd=out, timeout=60, check=True)
    events = [json.loads(line) for line in response.stdout.splitlines()]
    terminal = [event for event in events if event["event"] == "done"]
    assert len(terminal) == 1 and terminal[0]["ok"]
    payload = terminal[0]["payload"]
    assert payload["result"]["detection"]["quality_count"] >= 3
    assert Path(payload["preview"]).is_file()
    attitude_out = out / "attitude-job"
    attitude_request = {
        "operation": "tool",
        "tool": "rst19-aux-audit",
        "args": [str(frame.resolve()), "--out-dir", str(attitude_out.resolve())],
        "out_dir": str(attitude_out.resolve()),
    }
    attitude_response = subprocess.run(
        [str(native_worker)], input=json.dumps(attitude_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    attitude_events = [json.loads(line) for line in attitude_response.stdout.splitlines()]
    assert len(attitude_events) == 1 and attitude_events[0]["ok"]
    attitude = attitude_events[0]["payload"]
    assert attitude["tool"] == "rst19-aux-audit"
    assert attitude["result"]["summary"]["frame_count"] == 1
    assert attitude["result"]["summary"]["missing_auxiliary_count"] == 1
    assert Path(attitude["result_path"]).is_file()
    assert (attitude_out / "auxiliary_boresight_audit.csv").is_file()
    catalog = out / "source catalog.csv"
    make_feature_catalog(catalog)
    profile_out = out / "feature-flag-profile"
    profile_request = {
        "operation": "tool",
        "tool": "rst19-feature-flag-profile",
        "args": [
            str(catalog.resolve()), "--target-ids", "3", "1", "3", "missing",
            "--out-dir", str(profile_out.resolve()),
        ],
        "out_dir": str((out / "feature-flag-job").resolve()),
    }
    profile_response = subprocess.run(
        [str(native_worker)], input=json.dumps(profile_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    profile_events = [json.loads(line) for line in profile_response.stdout.splitlines()]
    assert len(profile_events) == 1 and profile_events[0]["ok"]
    profile = profile_events[0]["payload"]
    assert profile["tool"] == "rst19-feature-flag-profile"
    assert profile["result"]["targets"] == ["3", "1", "missing"]
    assert profile["result"]["rows"][0]["flag"] == "blend"
    assert profile["result"]["rows"][0]["flag_count"] == 2
    assert Path(profile["result_path"]).is_file()
    assert (profile_out / "feature_flag_profile.csv").is_file()
    assert (profile_out / "feature_flag_profile_targets.csv").is_file()
    preview_request = {
        "operation": "preview",
        "path": str(frame.resolve()),
        "mode": "noise",
        "out_dir": str((out / "preview-job").resolve()),
    }
    preview_response = subprocess.run(
        [str(native_worker)], input=json.dumps(preview_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    preview_events = [json.loads(line) for line in preview_response.stdout.splitlines()]
    assert len(preview_events) == 1 and preview_events[0]["ok"]
    assert Path(preview_events[0]["payload"]["preview"]).is_file()
    assert preview_events[0]["payload"]["width"] == 128
    assert hashlib.sha256(frame.read_bytes()).hexdigest() == original
    checksums = json.loads((package / "SHA256.json").read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        assert hashlib.sha256((package / relative).read_bytes()).hexdigest() == expected, relative
    report = {"package": str(package), "environment": entry["payload"]["environment"], "tool_count": 78,
              "native_backends": ["rst19 single-frame", "rst19-aux-audit", "rst19-feature-flag-profile"],
              "candidate_count": payload["result"]["detection"]["candidate_count"],
              "quality_count": payload["result"]["detection"]["quality_count"], "raw_fits_unchanged": True,
              "external_python_search_disabled": True, "checksum_files": len(checksums)}
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.out), ensure_ascii=False, indent=2))
