"""Compare the Rust source-geometry audit with its Python baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.source_geometry_audit import (  # noqa: E402
    run_source_geometry_audit,
    write_source_geometry_artifacts,
)

FLOAT_FIELDS = {
    "x_min", "x_max", "y_min", "y_max", "pca_axis_ratio", "pca_orientation_deg",
    "perpendicular_residual_median_px", "perpendicular_residual_p90_px",
    "perpendicular_residual_max_px", "control_axis_ratio_median", "control_axis_ratio_p95",
    "control_axis_ratio_p99", "control_axis_ratio_max", "control_axis_ratio_upper_tail_fraction",
}
CSV_ARTIFACT = "source_geometry_summary.csv"
JSON_ARTIFACT = "source_geometry_audit.json"


def write_catalog(path: Path) -> None:
    fields = ("x", "y", "feature_class", "feature_class_label")
    rows = []
    for index in range(12):
        rows.append((10 + index % 4 * 25, 10 + index // 4 * 35, "compact_quality", "紧凑质量源"))
    for index in range(4):
        rows.append((20 + index * 10, 100 + index * 100, "linear_artifact", "线状候选"))
    for index in range(12):
        rows.append((8 + index % 3 * 31, 12 + index // 3 * 27, "same_size_target", "等样本目标"))
    rows.extend(((7, 3, "other_rejected", "拒绝"), (8, 4, "other_rejected", "拒绝")))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def invoke(executable: Path, request: dict[str, object]) -> dict:
    process = subprocess.run(
        [str(executable.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode == 0 and len(events) == 1 and events[0]["ok"], (
        process.returncode, events, process.stderr
    )
    return events[0]["payload"]


def invoke_error(executable: Path, request: dict[str, object]) -> str:
    process = subprocess.run(
        [str(executable.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode != 0 and len(events) == 1 and not events[0]["ok"], (
        process.returncode, events, process.stderr
    )
    return events[0]["error"]


def compare_value(expected: object, actual: object, location: str = "result") -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
        assert actual == expected, (location, actual, expected)
    elif isinstance(expected, float):
        assert isinstance(actual, (float, int)) and not isinstance(actual, bool), (location, actual)
        assert math.isclose(float(actual), expected, rel_tol=2e-12, abs_tol=2e-12), (
            location, actual, expected
        )
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), location
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare_value(left, right, f"{location}[{index}]")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), (
            location, actual.keys() if isinstance(actual, dict) else type(actual), expected.keys()
        )
        for key in expected:
            compare_value(expected[key], actual[key], f"{location}.{key}")
    else:
        raise TypeError(type(expected))


def compare_csv(expected_path: Path, actual_path: Path) -> None:
    with expected_path.open(encoding="utf-8-sig", newline="") as stream:
        expected_rows = list(csv.DictReader(stream))
    with actual_path.open(encoding="utf-8-sig", newline="") as stream:
        actual_rows = list(csv.DictReader(stream))
    assert len(expected_rows) == len(actual_rows)
    for row_index, (expected, actual) in enumerate(zip(expected_rows, actual_rows, strict=True)):
        assert expected.keys() == actual.keys()
        for key in expected:
            left, right = expected[key], actual[key]
            if key in FLOAT_FIELDS and left and right:
                assert math.isclose(float(left), float(right), rel_tol=2e-12, abs_tol=2e-12), (
                    row_index, key, left, right
                )
            else:
                assert left == right, (row_index, key, left, right)


def verify(worker: Path, engine: Path, cli: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rst19-source-geometry-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog with spaces.csv"
        write_catalog(catalog)
        input_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
        cases = []
        for name, targets, trials, seed in (
            ("line-class", ["linear_artifact"], 64, 19),
            ("default-classes", [], 8, 1909),
            ("full-population-sample", ["same_size_target"], 8, 19),
            ("max-u64-seed", ["linear_artifact"], 16, 2**64 - 1),
        ):
            expected_result = run_source_geometry_audit(
                catalog,
                target_classes=targets or None,
                control_trial_count=trials,
                random_seed=seed,
            )
            expected = expected_result.as_dict()
            python_out = root / f"{name}-python"
            write_source_geometry_artifacts(expected_result, python_out)
            worker_out = root / f"{name}-worker-artifacts"
            worker_request = {
                "operation": "tool",
                "tool": "rst19-source-geometry-audit",
                "args": [str(catalog), *[part for target in targets for part in ("--target-class", target)],
                         "--trials", str(trials), "--seed", str(seed), "--out-dir", str(worker_out)],
                "out_dir": str(root / f"{name}-worker-job"),
            }
            engine_request = {
                "operation": "audit.source_geometry",
                "catalog_path": str(catalog),
                "target_classes": targets or None,
                "control_trial_count": trials,
                "random_seed": seed,
            }
            worker_payload = invoke(worker, worker_request)
            engine_result = invoke(engine, engine_request)
            for label, actual in (("worker", worker_payload["result"]), ("engine", engine_result)):
                compare_value(expected, actual, f"{label}.{name}")
            for artifact_dir in (worker_out,):
                compare_csv(python_out / CSV_ARTIFACT, artifact_dir / CSV_ARTIFACT)
                compare_value(
                    json.loads((python_out / JSON_ARTIFACT).read_text(encoding="utf-8")),
                    json.loads((artifact_dir / JSON_ARTIFACT).read_text(encoding="utf-8")),
                    f"{name}.artifact_json",
                )

            cli_out = root / f"{name}-cli-artifacts"
            cli_args = [str(catalog), *[part for target in targets for part in ("--target-class", target)],
                        "--trials", str(trials), "--seed", str(seed), "--out-dir", str(cli_out)]
            cli_result = subprocess.run(
                [str(cli.resolve()), *cli_args], capture_output=True, text=True,
                encoding="utf-8", timeout=120,
            )
            assert cli_result.returncode == 0, (cli_result.stdout, cli_result.stderr)
            compare_csv(python_out / CSV_ARTIFACT, cli_out / CSV_ARTIFACT)
            compare_value(
                json.loads((python_out / JSON_ARTIFACT).read_text(encoding="utf-8")),
                json.loads((cli_out / JSON_ARTIFACT).read_text(encoding="utf-8")),
                f"{name}.cli_json",
            )
            cases.append({
                "case": name,
                "classes": len(expected["summaries"]),
                "trials_per_class": trials,
                "result_fields_match_with_float_tolerance": True,
                "csv_schema_and_values_match_with_float_tolerance": True,
                "json_fields_match_with_float_tolerance": True,
                "desktop_worker_and_standalone_cli_pass": True,
            })

        error_cases = []
        errors = (
            ("invalid-trials", {"control_trial_count": 0}, "control_trial_count 必须为正整数"),
            ("negative-trials", {"control_trial_count": -1}, "control_trial_count 必须为正整数"),
            ("unknown-class", {"target_classes": ["missing"]}, "source catalog has no target class: 'missing'"),
            ("missing-control", {"control_class": "missing"}, "source catalog has no control class: 'missing'"),
        )
        for name, overrides, expected_error in errors:
            request = {
                "operation": "audit.source_geometry",
                "catalog_path": str(catalog),
                "target_classes": ["linear_artifact"],
                "control_trial_count": 8,
                "random_seed": 19,
                **overrides,
            }
            try:
                run_source_geometry_audit(
                    catalog,
                    target_classes=request.get("target_classes"),
                    control_class=request.get("control_class", "compact_quality"),
                    control_trial_count=request["control_trial_count"],
                    random_seed=request["random_seed"],
                )
            except (OSError, ValueError) as error:
                assert str(error) == expected_error, (name, str(error), expected_error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {name}")
            assert invoke_error(engine, request) == expected_error, name
            worker_targets = request.get("target_classes") or []
            worker_args = [str(catalog)]
            worker_args.extend(
                part for target in worker_targets for part in ("--target-class", target)
            )
            worker_args.extend(("--trials", str(request["control_trial_count"])))
            worker_args.extend(("--seed", str(request["random_seed"])))
            if "control_class" in request:
                worker_args.extend(("--control-class", request["control_class"]))
            worker_request = {
                "operation": "tool",
                "tool": "rst19-source-geometry-audit",
                "args": worker_args,
                "out_dir": str(root / f"{name}-job"),
            }
            assert invoke_error(worker, worker_request) == expected_error, name
            error_cases.append({"case": name, "error_matches_python": True})

        assert hashlib.sha256(catalog.read_bytes()).hexdigest() == input_hash

    report = {
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "cases": cases,
        "error_cases": error_cases,
        "random_generator": "NumPy 1.26.4 SeedSequence + PCG64 XSL RR 128/64 + Generator.choice without replacement",
        "numeric_comparison": {"relative_tolerance": 2e-12, "absolute_tolerance": 2e-12},
        "all_inputs_unchanged": True,
    }
    (out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.worker, args.engine, args.cli, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
