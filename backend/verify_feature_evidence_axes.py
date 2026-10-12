"""Compare the Rust evidence-axes audit and artifacts with Python."""
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
from rst19.feature_evidence_axes import (  # noqa: E402
    run_feature_evidence_axes,
    write_feature_evidence_axes_artifacts,
)

FIELDS = [
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "candidate_presence_ge_required_count",
    "quality_response_ge_required_count",
    "spatial_local_correlation_median",
    "spatial_local_residual_median",
]
ROWS = [
    ["compact_quality", "紧凑,清晰", "100", "95", "85", "0.84", "0.54"],
    ["crowded_blend", "拥挤", "100", "60", "1", "0.17", "0.98"],
    ["masked_or_edge", "边缘/掩膜", "100", "80", "20", "0.67", "0.74"],
    ["linear_artifact", "线伪迹", "10", "2", "1", "0.40", "0.91"],
    ["other_rejected", "其它", "0", "0", "0", "", ""],
]
BOUNDARY_ROWS = [
    ["exact_aligned", "阈值,相等", "100", "80", "80", "0.80", "1.0"],
    ["strict_psf_edge", "PSF边界", "100", "80", "80", "0.85", "0.5"],
    ["quality_gap_edge", "持久边界", "100", "50", "5", "", ""],
    ["below_persistent", "稀疏", "100", "49", "30", "0.9", "0.1"],
    ["no_sample", "空", "0", "0", "0", "nan", "null"],
]


def invoke(executable: Path, request: dict[str, object]) -> dict:
    process = subprocess.run(
        [str(executable.resolve())],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(events) == 1 and events[0]["ok"], events
    return events[0]["payload"]


def invoke_error(executable: Path, request: dict[str, object]) -> str:
    process = subprocess.run(
        [str(executable.resolve())],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode != 0 and len(events) == 1 and not events[0]["ok"], events
    return events[0]["error"]


def write_matrix(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    case_reports = []
    with tempfile.TemporaryDirectory(prefix="rst19-feature-evidence-axes-") as temporary:
        root = Path(temporary)
        for name, rows in (("default", ROWS), ("boundaries", BOUNDARY_ROWS)):
            matrix = root / f"{name} matrix,证据.csv"
            write_matrix(matrix, rows)
            before = hashlib.sha256(matrix.read_bytes()).hexdigest()
            baseline = run_feature_evidence_axes(matrix)
            expected = baseline.as_dict()
            python_out = root / f"{name}-python"
            native_out = root / f"{name}-native"
            write_feature_evidence_axes_artifacts(baseline, python_out)
            worker_payload = invoke(
                worker,
                {
                    "operation": "tool",
                    "tool": "rst19-feature-evidence-axes",
                    "args": [str(matrix), "--out-dir", str(native_out)],
                    "out_dir": str(root / f"{name}-worker-job"),
                },
            )
            engine_payload = invoke(
                engine,
                {
                    "operation": "audit.feature_evidence_axes",
                    "matrix_path": str(matrix),
                },
            )
            assert worker_payload["result"] == expected, name
            assert engine_payload == expected, name
            for artifact in (
                "feature_evidence_axes.csv",
                "feature_evidence_axes_sensitivity.csv",
            ):
                assert (native_out / artifact).read_bytes() == (python_out / artifact).read_bytes(), artifact
            assert json.loads(
                (native_out / "feature_evidence_axes.json").read_text(encoding="utf-8")
            ) == expected
            assert hashlib.sha256(matrix.read_bytes()).hexdigest() == before
            case_reports.append(
                {
                    "case": name,
                    "row_count": len(expected["rows"]),
                    "sensitivity_row_count": len(expected["sensitivity_rows"]),
                    "result_fields_match": True,
                    "csv_artifacts_byte_identical": True,
                    "json_fields_match": True,
                    "input_unchanged": True,
                }
            )

        error_reports = []
        error_rows = [
            ("duplicate-class", ROWS + [ROWS[0]], None),
            ("quality-count-overflow", [ROWS[0], ["overflow", "x", "4", "1", "5", "0.2", "0.3"]], None),
            ("invalid-float", [ROWS[0], ["bad", "x", "4", "1", "1", "nope", "0.3"]], None),
        ]
        for name, rows, _unused in error_rows:
            matrix = root / f"{name}.csv"
            write_matrix(matrix, rows)
            request = {
                "operation": "audit.feature_evidence_axes",
                "matrix_path": str(matrix),
            }
            try:
                run_feature_evidence_axes(matrix)
            except (OSError, ValueError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {name}")
            assert invoke_error(engine, request) == expected_error, name
            worker_error = invoke_error(
                worker,
                {
                    "operation": "tool",
                    "tool": "rst19-feature-evidence-axes",
                    "args": [str(matrix), "--out-dir", str(root / f"{name}-out")],
                    "out_dir": str(root / f"{name}-job"),
                },
            )
            assert worker_error == expected_error, name
            error_reports.append({"case": name, "error_matches_python": True})

    report = {
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cases": case_reports,
        "error_cases": error_reports,
        "all_result_fields_match": True,
        "all_csv_artifacts_byte_identical": True,
        "all_inputs_unchanged": True,
    }
    (out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.worker, arguments.engine, arguments.out), ensure_ascii=False, indent=2))
