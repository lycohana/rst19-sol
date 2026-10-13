"""Compare native sequence-brief routes against the Python implementation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.sequence_brief import build_sequence_brief  # noqa: E402


def point(frame_index: int, x: float, y: float) -> dict[str, Any]:
    return {"frame_index": frame_index, "aligned_x": x, "aligned_y": y}


def cases() -> list[tuple[str, dict[str, Any]]]:
    frames = [
        {"timestamp": "2026-03-30T16:32:05.000+03:00"},
        {"timestamp": "2026-03-30T16:32:07.000Z"},
        {"timestamp": "2026-03-30T16:32:09.500"},
    ]
    line_points = [point(index, 100.0 + 10.0 * index, 200.0 - 4.0 * index) for index in range(3)]
    point_points = [point(index, 40.0 + 3.0 * index, 80.0 + 2.0 * index) for index in range(3)]
    return [
        (
            "offset-time-moving-and-mosaic",
            {
                "operation": "sequence.brief",
                "result": {
                    "frames": frames,
                    "cumulative_shifts": [[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]],
                    "motion_features": [
                        {"track_id": 9, "classification": "candidate", "points": [point(0, 4.0, 6.0)],
                         "speed_px_per_frame": 0.0, "fit_rms_px": None, "evidence_level": "single_frame_trail"},
                        {"track_id": 7, "classification": "moving", "points": line_points,
                         "speed_px_per_frame": 10.0, "fit_rms_px": 0.2},
                    ],
                    "tracks": [
                        {"track_id": 1, "classification": "moving", "points": point_points,
                         "speed_px_per_frame": 3.6, "fit_rms_px": 0.1},
                    ],
                    "stable_source_count": 1234,
                    "persistent_source_count": 56,
                },
                "mosaic": {
                    "output_shape": [4098, 4098],
                    "covered_pixel_count": 16_777_216,
                    "overlap_pixel_count": 16_777_000,
                },
            },
        ),
        (
            "empty-invalid-times",
            {
                "operation": "sequence.brief",
                "result": {
                    "frames": [{"timestamp": "bad"}, {"timestamp": None}, {}],
                    "cumulative_shifts": [],
                    "tracks": [],
                    "motion_features": [],
                },
            },
        ),
        (
            "frame-based-speed-fallback",
            {
                "operation": "sequence.brief",
                "result": {
                    "frames": [{"timestamp": "bad"}, {"timestamp": "still bad"}, {"timestamp": ""}],
                    "cumulative_shifts": [[-3.0, 4.0]],
                    "tracks": [
                        {"track_id": 42, "classification": "moving",
                         "points": [point(0, 1.0, 4.0), point(2, 7.0, -2.0)],
                         "speed_px_per_frame": 4.25, "fit_rms_px": None},
                    ],
                    "motion_features": [],
                    "stable_source_count": 10_001,
                    "persistent_source_count": 0,
                },
                "mosaic": {
                    "output_shape": [128, 256],
                    "covered_pixel_count": 1_001,
                    "overlap_pixel_count": 1,
                },
            },
        ),
    ]


def namespace(value: Any) -> Any:
    if isinstance(value, list):
        return [namespace(item) for item in value]
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    return value


def python_result(request: dict[str, Any]) -> dict[str, Any]:
    result = build_sequence_brief(namespace(request["result"]), namespace(request.get("mosaic")))
    return {"brief": result.as_dict(), "text": result.text()}


def compare(actual: Any, expected: Any, path: str, tolerance: float) -> None:
    if is_dataclass(expected):
        expected = asdict(expected)
    if isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
        if actual != expected:
            raise AssertionError(f"{path}: Rust returned {actual!r}; Python returned {expected!r}")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(
            actual, expected, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise AssertionError(f"{path}: Rust returned {actual!r}; Python returned {expected!r}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length or type differs")
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            compare(got, wanted, f"{path}[{index}]", tolerance)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: object keys differ")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}", tolerance)
    else:
        raise TypeError(f"unsupported baseline value at {path}: {type(expected)}")


def invoke(executable: Path, request: dict[str, Any], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable)],
        input=json.dumps(request, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        timeout=60,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{executable} exited {completed.returncode}: {completed.stderr}")
    event = json.loads(completed.stdout.splitlines()[-1])
    if not event.get("ok"):
        raise RuntimeError(event.get("error", completed.stderr))
    return event["payload"]


def verify(engine: Path, worker: Path, out: Path, tolerance: float = 5e-7) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "PYTHONPATH": "",
        "PYTHONHOME": "",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    env["PATH"] = (
        str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32")
        if os.name == "nt"
        else "/usr/bin:/bin"
    )
    report = {"routes": ["engine JSONL", "desktop worker JSONL"], "tolerance": tolerance, "cases": []}
    for name, request in cases():
        expected = python_result(request)
        for route, executable in (("engine", engine), ("worker", worker)):
            actual = invoke(executable, request, out, env)
            compare(actual, expected, f"{name}.{route}", tolerance)
        report["cases"].append(name)
    report["case_count"] = len(report["cases"])
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.engine.resolve(), args.worker.resolve(), args.out.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
