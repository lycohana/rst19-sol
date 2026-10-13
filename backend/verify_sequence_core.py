"""Compare Rust FITS-to-track orchestration with its Python constituent stages."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.detection import build_background_model  # noqa: E402
from rst19.fits import auxiliary_mask, read_fits  # noqa: E402
from rst19.pipeline import analyze_frame  # noqa: E402
from rst19.sequence import (  # noqa: E402
    _observation_times_seconds,
    track_detections,
    track_fast_point_movers,
)

FRAME_OPTIONS: dict[str, Any] = {
    "threshold_sigma": 4.0,
    "min_distance": 3,
    "max_sources": 6_000,
    "background_box_size": 256,
    "background_sample_limit": 4_096,
    "use_float32": True,
    "fast_sequence": True,
    "refine_local_background": False,
}


def card(name: str, value: str) -> bytes:
    return f"{name:<8}= {value:>20}".ljust(80).encode("ascii")


def make_fits(path: Path, image: np.ndarray, timestamp: str) -> None:
    height, width = image.shape
    header = b"".join(
        [
            card("SIMPLE", "T"),
            card("BITPIX", "16"),
            card("NAXIS", "2"),
            card("NAXIS1", str(width)),
            card("NAXIS2", str(height)),
            card("DATE-OBS", f"'{timestamp}'"),
            card("EXPTIME", "1.5"),
            b"END".ljust(80),
        ]
    ).ljust(2880, b" ")
    payload = header + np.asarray(image, dtype=">i2").tobytes()
    path.write_bytes(payload + bytes((-len(payload)) % 2880))


def synthetic_frames(directory: Path) -> list[Path]:
    rng = np.random.default_rng(1913)
    yy, xx = np.indices((128, 128), dtype=np.float64)
    paths = []
    static_stars = [(20, 20, 360), (43, 29, 270), (81, 22, 330), (30, 77, 310), (91, 86, 420)]
    for index in range(3):
        image = rng.normal(100.0, 2.0, size=(128, 128))
        for x, y, amplitude in static_stars:
            image += amplitude * np.exp(-0.5 * (((xx - x) / 1.25) ** 2 + ((yy - y) / 1.25) ** 2))
        image += 440 * np.exp(
            -0.5 * (((xx - (55 + index * 4)) / 1.25) ** 2 + ((yy - (57 + index * 2)) / 1.25) ** 2)
        )
        path = directory / f"frame-{index:02d}.fits"
        make_fits(path, np.rint(image).astype(np.int16), f"2026-07-03T12:00:{index * 2:02d}.000Z")
        paths.append(path)
    return paths


def compare(actual: Any, expected: Any, path: str = "result") -> None:
    if is_dataclass(expected):
        expected = asdict(expected)
    if isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
        if actual != expected:
            raise AssertionError(f"{path}: Rust returned {actual!r}; Python returned {expected!r}")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(
            actual, expected, rel_tol=5e-7, abs_tol=5e-7
        ):
            raise AssertionError(f"{path}: Rust returned {actual!r}; Python returned {expected!r}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length or type differs")
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            compare(got, wanted, f"{path}[{index}]")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: object keys differ")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    else:
        raise TypeError(f"unsupported baseline value at {path}: {type(expected)}")


def invoke(engine: Path, request: dict[str, Any]) -> dict[str, Any]:
    response = subprocess.run(
        [str(engine)],
        input=json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=300,
    )
    events = [json.loads(line) for line in response.stdout.splitlines()]
    if response.returncode != 0 or len(events) != 1 or events[0].get("ok") is not True:
        raise RuntimeError(f"Rust engine failed: exit={response.returncode}; {response.stdout}; {response.stderr}")
    return events[0]["payload"]


def verify(engine: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    paths = synthetic_frames(out)
    frames = [read_fits(path) for path in paths]
    shared_background = build_background_model(
        frames[0].data,
        mask=auxiliary_mask(frames[0].data.shape),
        background_box_size=FRAME_OPTIONS["background_box_size"],
        background_sample_limit=FRAME_OPTIONS["background_sample_limit"],
        refine_local_background=FRAME_OPTIONS["refine_local_background"],
        use_float32=FRAME_OPTIONS["use_float32"],
    )
    python_analyses = [
        analyze_frame(path, background_model=shared_background, **FRAME_OPTIONS)
        for path in paths
    ]
    frame_times = _observation_times_seconds(python_analyses).tolist()
    python_tracking_result = track_detections(
        [analysis.detection.quality_sources for analysis in python_analyses],
        frame_names=[str(path) for path in paths],
        frame_times=frame_times,
    )
    python_tracking = python_tracking_result.as_dict()
    effective_motion_gate = python_tracking_result.effective_motion_min_displacement_px or 2.0
    python_fast_point_tracks = track_fast_point_movers(
        [analysis.detection.quality_sources for analysis in python_analyses],
        python_tracking_result.cumulative_shifts,
        frame_times=frame_times,
        link_radius_px=python_tracking_result.link_radius_px,
        min_presence=python_tracking_result.min_presence,
        motion_min_displacement_px=effective_motion_gate,
        existing_tracks=python_tracking_result.tracks,
    )
    python_fast_point_rows = []
    for index, track in enumerate(python_fast_point_tracks):
        row = track.as_dict()
        row["track_id"] = len(python_tracking_result.tracks) + index
        python_fast_point_rows.append(row)
    request = {
        "operation": "sequence.analyze_core",
        "paths": [str(path) for path in paths],
        "frame_options": FRAME_OPTIONS,
    }
    actual = invoke(engine.resolve(), request)
    if actual["schema"] != "rst19.sequence.core.v1" or not actual["shared_background_model"]:
        raise AssertionError("Rust did not execute the sequence shared-background core")
    if len(actual["frame_analyses"]) != len(python_analyses):
        raise AssertionError("frame count differs")
    for index, (native, python) in enumerate(zip(actual["frame_analyses"], python_analyses)):
        compare(native, python.as_dict(), f"frame_analyses[{index}]")
    for key in (
        "frame_count",
        "stable_source_count",
        "persistent_source_count",
        "moving_track_count",
        "transient_track_count",
        "min_presence",
        "registration_model",
        "registration_affine_frame_count",
        "effective_motion_min_displacement_px",
        "cumulative_shifts",
        "tracks",
    ):
        compare(actual["tracking"][key], python_tracking[key], f"tracking.{key}")
    compare(actual["fast_point_tracks"], python_fast_point_rows, "fast_point_tracks")
    compare(actual["tracking"]["fast_point_motion_count"], len(python_fast_point_rows), "tracking.fast_point_motion_count")
    compare(actual["observation_time_axis"]["frame_times"], frame_times, "observation_time_axis.frame_times")
    disabled = invoke(engine.resolve(), {**request, "fast_point_motion": False})
    compare(disabled["fast_point_tracks"], [], "disabled.fast_point_tracks")
    compare(disabled["tracking"]["fast_point_motion_count"], 0, "disabled.tracking.fast_point_motion_count")
    report = {
        "engine": str(engine.resolve()),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "frame_count": len(paths),
        "frame_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "python_baseline_sha256": {
            module: hashlib.sha256((ROOT / "src" / "rst19" / module).read_bytes()).hexdigest()
            for module in ("detection.py", "fits.py", "pipeline.py", "sequence.py")
        },
        "per_frame_analysis_fields_compared": True,
        "quality_track_fields_compared": [
            "frame_count",
            "stable_source_count",
            "persistent_source_count",
            "moving_track_count",
            "transient_track_count",
            "min_presence",
            "registration_model",
            "registration_affine_frame_count",
            "effective_motion_min_displacement_px",
            "cumulative_shifts",
            "tracks",
        ],
        "fast_point_motion_track_count": len(python_fast_point_rows),
        "fast_point_motion_disable_verified": True,
        "observation_times_seconds": frame_times,
        "scope": "Rust FITS detection, shared first-frame background, time parsing, ordinary quality tracking, and supplemental fast-point quality tracking; temporal candidate recovery, trail auditing, stack recovery, and relative photometry remain outside this core operation",
    }
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.engine, arguments.out), ensure_ascii=False, indent=2))
