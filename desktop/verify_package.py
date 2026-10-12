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
        + b"detection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,psf_support_pixels,fwhm,ellipticity,sharpness,footprint_pixels\n"
        + b"1,compact,Compact,true,wing|blend|wing,12,5,2.5,0.1,0.3,20\n"
        + b"2,compact,Compact,false,blend,11,4,2.5,0.1,0.3,20\n"
        + b"3,diffuse,Diffuse,yes,,3,2,0.5,0.8,0.9,1\n"
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
    interaction_out = out / "feature-flag-interaction"
    interaction_request = {
        "operation": "tool",
        "tool": "rst19-feature-flag-interaction",
        "args": [
            str(catalog.resolve()), "--high-snr-threshold", "10", "--top-n", "2",
            "--out-dir", str(interaction_out.resolve()),
        ],
        "out_dir": str((out / "feature-flag-interaction-job").resolve()),
    }
    interaction_response = subprocess.run(
        [str(native_worker)], input=json.dumps(interaction_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    interaction_events = [json.loads(line) for line in interaction_response.stdout.splitlines()]
    assert len(interaction_events) == 1 and interaction_events[0]["ok"]
    interaction = interaction_events[0]["payload"]
    assert interaction["tool"] == "rst19-feature-flag-interaction"
    assert interaction["result"]["record_count"] == 3
    assert interaction["result"]["rows"][0]["interaction"] == "blend"
    assert interaction["result"]["rows"][0]["candidate_count"] == 2
    assert Path(interaction["result_path"]).is_file()
    assert (interaction_out / "feature_flag_interactions.csv").is_file()
    margin_out = out / "feature-gate-margin"
    margin_request = {
        "operation": "tool",
        "tool": "rst19-feature-gate-margin",
        "args": [str(catalog.resolve()), "--out-dir", str(margin_out.resolve())],
        "out_dir": str((out / "feature-gate-margin-job").resolve()),
    }
    margin_response = subprocess.run(
        [str(native_worker)], input=json.dumps(margin_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    margin_events = [json.loads(line) for line in margin_response.stdout.splitlines()]
    assert len(margin_events) == 1 and margin_events[0]["ok"]
    margin = margin_events[0]["payload"]
    assert margin["tool"] == "rst19-feature-gate-margin"
    assert margin["result"]["rows"][0]["feature_class"] == "compact"
    assert margin["result"]["rows"][0]["metric"] == "ellipticity"
    assert margin["result"]["rows"][0]["class_count"] == 2
    assert Path(margin["result_path"]).is_file()
    assert (margin_out / "feature_gate_margins.csv").is_file()
    effect_out = out / "feature-effect-size"
    effect_request = {
        "operation": "tool",
        "tool": "rst19-feature-effect-size",
        "args": [
            str(catalog.resolve()), "--reference-class", "compact",
            "--comparison-classes", "diffuse", "--features", "flux_snr", "fwhm",
            "--top-n", "2", "--out-dir", str(effect_out.resolve()),
        ],
        "out_dir": str((out / "feature-effect-size-job").resolve()),
    }
    effect_response = subprocess.run(
        [str(native_worker)], input=json.dumps(effect_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    effect_events = [json.loads(line) for line in effect_response.stdout.splitlines()]
    assert len(effect_events) == 1 and effect_events[0]["ok"]
    effect = effect_events[0]["payload"]
    assert effect["tool"] == "rst19-feature-effect-size"
    assert effect["result"]["comparison_classes"] == ["diffuse"]
    assert effect["result"]["top_features_by_class"]["diffuse"] == ["flux_snr", "fwhm"]
    assert Path(effect["result_path"]).is_file()
    assert (effect_out / "feature_effect_sizes.csv").is_file()
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
              "native_backends": [
                  "rst19 single-frame", "rst19-aux-audit", "rst19-feature-flag-profile",
                  "rst19-feature-flag-interaction", "rst19-feature-gate-margin",
                  "rst19-feature-effect-size",
              ],
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
