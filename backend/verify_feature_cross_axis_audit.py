"""Compare the Rust cross-axis audit with the Python implementation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.feature_cross_axis_audit import (  # noqa: E402
    run_feature_cross_axis_audit,
    write_feature_cross_axis_artifacts,
)

INPUTS = {
    "catalog": (
        ["detection_id", "feature_class", "feature_class_label", "quality_passed"],
        [
            ["1", "compact_quality", "紧凑,清晰", "True"],
            ["2", "crowded_blend", "拥挤", "False"],
            ["3", "compact_quality", "紧凑,清晰", "yes"],
        ],
    ),
    "raw_summary": (
        [
            "detection_id", "core_fraction_median", "noise_median", "frames_support_ge3",
            "frames_flux_snr_ge5", "frames_negative_extreme_count_gt0",
            "frames_repeated_code_count_gt0",
        ],
        [
            ["1", "0.7", "8", "15", "15", "0", "0"],
            ["2", "0.2", "14", "0", "2", "1", "1"],
            ["3", "0.6", "9", "10", "12", "0", "0"],
        ],
    ),
    "peak_consistency": (
        [
            "detection_id", "feature_class", "feature_class_label", "quality_passed", "raw_valid",
            "raw_local_maximum_r1", "raw_local_max_gap_adu_r1",
        ],
        [
            ["1", "compact_quality", "紧凑,清晰", "True", "True", "True", "0"],
            ["2", "crowded_blend", "拥挤", "False", "True", "False", "100"],
            ["3", "compact_quality", "紧凑,清晰", "yes", "yes", "yes", "2.5"],
        ],
    ),
    "temporal_sources": (
        [
            "detection_id", "feature_class", "expected_frame_count", "frame_count", "coverage_fraction",
            "flux_snr_relative_mad", "flux_snr_positive_fraction", "flux_snr_ge5_fraction",
            "flux_snr_sign_flip_count",
        ],
        [
            ["1", "compact_quality", "15", "15", "1", "0.1", "1", "1", "0"],
            ["2", "crowded_blend", "15", "10", "0.6667", "0.3", "0.8", "0.1", "2"],
            ["3", "compact_quality", "15", "12", "0.8", "0.2", "0.9", "0.7", "1"],
        ],
    ),
}
CSV_ARTIFACTS = (
    "feature_cross_axis_sources.csv",
    "feature_cross_axis_summary.csv",
    "feature_cross_axis_patterns.csv",
)
JSON_ARTIFACT = "feature_cross_axis_audit.json"


def invoke(executable: Path, request: dict[str, object]) -> dict:
    process = subprocess.run(
        [str(executable.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=True,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(events) == 1 and events[0]["ok"], events
    return events[0]["payload"]


def invoke_error(executable: Path, request: dict[str, object]) -> str:
    process = subprocess.run(
        [str(executable.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode != 0 and len(events) == 1 and not events[0]["ok"], events
    return events[0]["error"]


def write_inputs(root: Path, *, tables: dict | None = None) -> dict[str, Path]:
    paths = {}
    for name, (fields, rows) in (tables or INPUTS).items():
        path = root / f"{name} 输入.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\r\n")
            writer.writerow(fields)
            writer.writerows(rows)
        paths[name] = path
    return paths


def make_requests(paths: dict[str, Path], output: Path, *, selected: bool) -> tuple[dict, dict]:
    args = [
        str(paths["catalog"]), str(paths["raw_summary"]), str(paths["peak_consistency"]),
        str(paths["temporal_sources"]),
    ]
    engine: dict[str, object] = {
        "operation": "audit.feature_cross_axis",
        "catalog_path": str(paths["catalog"]),
        "raw_summary_path": str(paths["raw_summary"]),
        "peak_consistency_path": str(paths["peak_consistency"]),
        "temporal_sources_path": str(paths["temporal_sources"]),
    }
    if selected:
        args += ["--detection-id", "2", "--detection-id", "1"]
        args += ["--feature-class", "crowded_blend", "--feature-class", "compact_quality"]
        engine["detection_ids"] = [2, 1]
        engine["feature_classes"] = ["crowded_blend", "compact_quality"]
    args += ["--out-dir", str(output)]
    worker = {
        "operation": "tool", "tool": "rst19-feature-cross-axis-audit", "args": args,
        "out_dir": str(output.with_name(output.name + "-job")),
    }
    return worker, engine


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    errors = []
    with tempfile.TemporaryDirectory(prefix="rst19-cross-axis-") as temporary:
        root = Path(temporary)
        paths = write_inputs(root)
        before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
        for case_name, selected in (("default-all-classes", False), ("ordered-source-and-class-selection", True)):
            python_result = run_feature_cross_axis_audit(
                paths["catalog"], paths["raw_summary"], paths["peak_consistency"],
                paths["temporal_sources"],
                **({"detection_ids": (2, 1), "feature_classes": ("crowded_blend", "compact_quality")} if selected else {}),
            )
            expected = python_result.as_dict()
            python_out = root / f"{case_name}-python"
            native_out = root / f"{case_name}-native"
            write_feature_cross_axis_artifacts(python_result, python_out)
            worker_request, engine_request = make_requests(paths, native_out, selected=selected)
            worker_payload = invoke(worker, worker_request)
            engine_payload = invoke(engine, engine_request)
            assert worker_payload["result"] == expected, case_name
            assert engine_payload == expected, case_name
            for artifact in CSV_ARTIFACTS:
                assert (native_out / artifact).read_bytes() == (python_out / artifact).read_bytes(), (case_name, artifact)
            assert json.loads((native_out / JSON_ARTIFACT).read_text(encoding="utf-8")) == json.loads(
                (python_out / JSON_ARTIFACT).read_text(encoding="utf-8")
            ), case_name
            cases.append({
                "case": case_name,
                "source_count": len(expected["source_rows"]),
                "class_count": len(expected["class_summaries"]),
                "pattern_count": len(expected["pattern_summaries"]),
                "result_fields_match": True,
                "csv_artifacts_byte_identical": True,
                "json_fields_match": True,
            })

        error_inputs = []
        temporal_mismatch = {name: (fields, [row[:] for row in rows]) for name, (fields, rows) in INPUTS.items()}
        temporal_mismatch["temporal_sources"][1].append(["4", "compact_quality", "15", "15", "1", "0.1", "1", "1", "0"])
        error_inputs.append(("temporal-id-set-mismatch", temporal_mismatch, None))
        bad_bool = {name: (fields, [row[:] for row in rows]) for name, (fields, rows) in INPUTS.items()}
        bad_bool["catalog"][1][0][3] = "maybe"
        error_inputs.append(("invalid-quality-boolean", bad_bool, None))
        class_mismatch = {name: (fields, [row[:] for row in rows]) for name, (fields, rows) in INPUTS.items()}
        class_mismatch["peak_consistency"][1][1][1] = "compact_quality"
        error_inputs.append(("peak-class-mismatch", class_mismatch, None))

        for name, tables, expected_fragment in error_inputs:
            case_root = root / name
            case_root.mkdir()
            error_paths = write_inputs(case_root, tables=tables)
            try:
                run_feature_cross_axis_audit(
                    error_paths["catalog"], error_paths["raw_summary"],
                    error_paths["peak_consistency"], error_paths["temporal_sources"],
                )
            except (OSError, ValueError, TypeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {name}")
            worker_request, engine_request = make_requests(error_paths, root / f"{name}-out", selected=False)
            assert invoke_error(engine, engine_request) == expected_error, name
            assert invoke_error(worker, worker_request) == expected_error, name
            if expected_fragment:
                assert expected_fragment in expected_error, expected_error
            errors.append({"case": name, "error_matches_python": True})

        assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()} == before

    report = {
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cases": cases,
        "error_cases": errors,
        "all_result_fields_match": True,
        "all_csv_artifacts_byte_identical": True,
        "all_json_fields_match": True,
        "all_inputs_unchanged": True,
    }
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.worker, arguments.engine, arguments.out), ensure_ascii=False, indent=2))
