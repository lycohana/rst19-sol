"""Compare the Rust relative-photometry routes with the Python implementation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.relative_photometry import Observation, fit_relative_photometry  # noqa: E402


def row(frame: int, source: str, magnitude: float, error: float, x: float, y: float,
        *, quality: bool = True, moving: bool = False, variable: bool | None = None) -> dict:
    return asdict(Observation(frame, source, magnitude, error, x, y, quality, moving, variable))


def synthetic(frame_count: int, source_count: int, *, noise_seed: int | None = None) -> list[dict]:
    rng = np.random.default_rng(noise_seed)
    zero_points = np.linspace(-0.18, 0.21, frame_count)
    zero_points -= np.median(zero_points)
    source_magnitudes = np.linspace(10.1, 14.6, source_count)
    return [
        row(frame, f"s{source}", source_magnitudes[source] + zero_points[frame]
            + (float(rng.normal(0.0, 0.0035)) if noise_seed is not None else 0.0),
            0.01, 100.0 + 13.0 * source + 0.15 * frame,
            80.0 + 7.0 * source - 0.12 * frame)
        for frame in range(frame_count) for source in range(source_count)
    ]


def cases() -> list[tuple[str, object, dict]]:
    noisy = synthetic(15, 6, noise_seed=4)
    rejected = synthetic(8, 5)
    rejected[17]["instrumental_magnitude"] += 1.5
    rejected.extend([
        row(frame, "moving", 3.0 + frame, 0.01, 20.0 + frame, 30.0, moving=True)
        for frame in range(8)
    ])
    rejected.extend([
        row(0, "bad-quality", 1.0, 0.01, 20.0, 30.0, quality=False),
        row(0, "variable", 2.0, 0.01, 20.0, 30.0, variable=True),
    ])
    spatial_rng = np.random.default_rng(33)
    coords = spatial_rng.normal(size=(30, 2))
    center = np.mean(coords, axis=0)
    scale = np.std(coords, axis=0)
    spatial = [
        row(frame, f"s{source}",
            np.linspace(10.0, 14.0, 5)[source] + np.linspace(-0.12, 0.16, 6)[frame]
            - np.median(np.linspace(-0.12, 0.16, 6))
            + 0.08 * ((coords[frame * 5 + source, 0] - center[0]) / scale[0])
            - 0.05 * ((coords[frame * 5 + source, 1] - center[1]) / scale[1]),
            0.01, float(coords[frame * 5 + source, 0]), float(coords[frame * 5 + source, 1]))
        for frame in range(6) for source in range(5)
    ]
    quadratic = [
        row(frame, f"s{source}",
            np.linspace(10.0, 14.0, 5)[source] + np.linspace(-0.12, 0.16, 6)[frame]
            - np.median(np.linspace(-0.12, 0.16, 6))
            + 0.03 * ((coords[frame * 5 + source, 0] - center[0]) / scale[0]) ** 2
            + 0.01 * ((coords[frame * 5 + source, 0] - center[0]) / scale[0])
                     * ((coords[frame * 5 + source, 1] - center[1]) / scale[1])
            - 0.02 * ((coords[frame * 5 + source, 1] - center[1]) / scale[1]) ** 2,
            0.01, float(coords[frame * 5 + source, 0]), float(coords[frame * 5 + source, 1]))
        for frame in range(6) for source in range(5)
    ]
    flat = [dict(item, x=100.0, y=200.0) for item in synthetic(4, 4)]
    tiny = [row(0, "a", 10.0, 0.01, 0.0, 0.0), row(0, "b", 11.0, 0.01, 1.0, 1.0),
            row(1, "a", 10.1, 0.01, 0.0, 0.0), row(1, "b", 11.1, 0.01, 1.0, 1.0)]
    return [
        ("noisy-holdout", noisy, {"validation_fraction": 0.2, "random_state": 12}),
        ("filter-and-mad-rejection", rejected, {"validation_fraction": 0.0, "mad_threshold": 4.0}),
        ("linear-spatial", spatial, {"spatial_order": 1, "validation_fraction": 0.0}),
        ("quadratic-spatial", quadratic, {"spatial_order": 2, "validation_fraction": 0.0}),
        ("degenerate-spatial", flat, {"spatial_order": 1, "validation_fraction": 0.0}),
        ("zero-dof", tiny, {"validation_fraction": 0.0}),
        ("mapping-input", {"not_a_row": True}, {"validation_fraction": 0.0}),
    ]


def compare(actual, expected, path: str = "result", *, tolerance: float = 5e-7) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
        if actual != expected:
            raise AssertionError(f"{path}: Rust returned {actual!r}; Python returned {expected!r}")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(
                actual, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(f"{path}: Rust returned {actual!r}; Python returned {expected!r}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length/type differs")
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            compare(got, wanted, f"{path}[{index}]", tolerance=tolerance)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: keys differ ({getattr(actual, 'keys', lambda: [])()} vs {expected.keys()})")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}", tolerance=tolerance)
    else:
        raise TypeError(f"unsupported baseline value at {path}: {type(expected)}")


def invoke_jsonl(executable: Path, request: dict, cwd: Path, env: dict[str, str]) -> dict:
    completed = subprocess.run([str(executable)], input=json.dumps(request) + "\n", text=True,
                               capture_output=True, cwd=cwd, timeout=60, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{executable} exited {completed.returncode}: {completed.stderr}")
    event = json.loads(completed.stdout.splitlines()[-1])
    if not event.get("ok"):
        raise RuntimeError(event.get("error", completed.stderr))
    return event["payload"]


def cli_flags(options: dict) -> list[str]:
    flags = []
    for option, flag in (("spatial_order", "--spatial-order"),
                         ("validation_fraction", "--validation-fraction"),
                         ("random_state", "--random-state"),
                         ("mad_threshold", "--mad-threshold")):
        if option in options:
            flags.extend((flag, str(options[option])))
    return flags


def verify(engine: Path, worker: Path, cli: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONPATH": "", "PYTHONHOME": "", "PYTHONUTF8": "1",
           "PYTHONIOENCODING": "utf-8"}
    env["PATH"] = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32") if os.name == "nt" else "/usr/bin:/bin"
    report = {"routes": [], "tolerance": 5e-7, "case_count": 0}
    with tempfile.TemporaryDirectory(prefix="relative-photometry-") as temporary:
        scratch = Path(temporary)
        for name, observations, options in cases():
            expected = fit_relative_photometry(observations, **options).as_dict()
            request = {"operation": "photometry.relative.fit", "observations": observations,
                       "options": options}
            engine_result = invoke_jsonl(engine, request, scratch, env)
            compare(engine_result, expected, f"{name}.engine")
            worker_result = invoke_jsonl(worker, request, scratch, env)
            compare(worker_result, expected, f"{name}.worker-jsonl")

            payload_path = scratch / f"{name}.json"
            payload_path.write_text(json.dumps({"observations": observations}), encoding="utf-8")
            output_path = scratch / f"{name}-cli-result.json"
            tool_request = {
                "operation": "tool", "tool": "rst19-relative-photometry",
                "args": [str(payload_path), *cli_flags(options), "--out", str(output_path)],
                "out_dir": str(scratch / f"{name}-worker-out"),
            }
            tool_result = invoke_jsonl(worker, tool_request, scratch, env)
            compare(tool_result["result"], expected, f"{name}.worker-tool")
            if json.loads(output_path.read_text(encoding="utf-8")) != tool_result["result"]:
                raise AssertionError(f"{name}: desktop tool output file differs from its event payload")

            cli_output = scratch / f"{name}-standalone.json"
            completed = subprocess.run([str(cli), str(payload_path), *cli_flags(options), "--out", str(cli_output)],
                                       text=True, capture_output=True, cwd=scratch, env=env, timeout=60)
            if completed.returncode != 0:
                raise RuntimeError(f"{name}: CLI failed: {completed.stderr}")
            compare(json.loads(cli_output.read_text(encoding="utf-8")), expected, f"{name}.standalone-cli")
            report["routes"].extend(["engine JSONL", "desktop worker JSONL", "desktop tool",
                                     "standalone Rust CLI"])
            report["case_count"] += 1
    report["routes"] = list(dict.fromkeys(report["routes"]))
    report["cases"] = [name for name, _, _ in cases()]
    (out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.engine.resolve(), args.worker.resolve(), args.cli.resolve(), args.out.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
