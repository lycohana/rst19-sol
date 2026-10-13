"""Compare the native class-conditioned injection audit with its Python baseline."""

from __future__ import annotations

import hashlib
import difflib
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.contaminated_pair_class import (  # noqa: E402
    run_feature_class_contaminated_pair_audit,
    write_feature_class_contaminated_pair_artifacts,
)

FRAME = ROOT / "backend/tests/fixtures/contaminated_pair_injection_parity.fits"
BASELINE_MODULE = ROOT / "src/rst19/contaminated_pair_class.py"
BASELINE_SHA256 = "6db47bcecbff2483f6ec9bf40f00a0e427db7ea8ab27e0e16e40aacec11291c0"
CATALOG = (
    "detection_id,feature_class,feature_class_label,x,y,filter_snr,flux_snr,peak,quality_passed,flags\n"
    "1,compact_quality,compact,64,112,20,10,100,true,\n"
    "2,compact_quality,compact,70,100,40,20,200,true,\n"
    "3,linear_artifact,line,80,120,100,50,300,false,LINE_ARTIFACT\n"
)
OPTIONS = (
    "--per-class",
    "2",
    "--separations-px",
    "6",
    "--ratios",
    "1",
    "--total-peak-adu",
    "2048",
    "--analysis-scope",
    "local_roi",
    "--roi-half-size-px",
    "20",
)


def compare(actual: object, expected: object, path: str = "result") -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        tolerance = 2e-5 + 2e-6 * max(abs(float(actual)), abs(float(expected)))
        assert abs(float(actual) - float(expected)) <= tolerance, (
            f"{path}: {actual!r} != {expected!r} (tolerance {tolerance})"
        )
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"{path}: array lengths differ"
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: object keys differ"
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def compare_csv_fields(actual_path: Path, expected_path: Path) -> None:
    with actual_path.open("r", encoding="utf-8-sig", newline="") as actual_stream:
        actual_rows = list(csv.reader(actual_stream))
    with expected_path.open("r", encoding="utf-8-sig", newline="") as expected_stream:
        expected_rows = list(csv.reader(expected_stream))
    assert len(actual_rows) == len(expected_rows), "injection CSV row count differs"
    for row_index, (actual_row, expected_row) in enumerate(zip(actual_rows, expected_rows)):
        assert len(actual_row) == len(expected_row), f"injection CSV row {row_index} width differs"
        for column_index, (actual_cell, expected_cell) in enumerate(zip(actual_row, expected_row)):
            try:
                actual_number = float(actual_cell)
                expected_number = float(expected_cell)
            except ValueError:
                assert actual_cell == expected_cell, (
                    f"injection CSV cell [{row_index},{column_index}] differs: "
                    f"{actual_cell!r} != {expected_cell!r}"
                )
            else:
                compare(actual_number, expected_number, f"injection_csv[{row_index},{column_index}]")


def verify(cli: Path) -> dict[str, object]:
    cli = cli.resolve()
    baseline_hash = hashlib.sha256(BASELINE_MODULE.read_bytes()).hexdigest()
    assert baseline_hash == BASELINE_SHA256, "Python baseline changed; refresh parity evidence before accepting migration"
    before = hashlib.sha256(FRAME.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="rst19-contaminated-pair-class-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        python_output = root / "python"
        rust_output = root / "rust"
        baseline = run_feature_class_contaminated_pair_audit(
            FRAME,
            catalog,
            per_class=2,
            separations_px=(6.0,),
            secondary_to_primary_ratios=(1.0,),
            total_peak_excess_adu=2048.0,
            analysis_scope="local_roi",
            roi_half_size_px=20,
        )
        expected = baseline.as_dict()
        write_feature_class_contaminated_pair_artifacts(baseline, python_output)
        environment = os.environ.copy()
        environment["PATH"] = ""
        process = subprocess.run(
            [
                str(cli),
                str(FRAME),
                "--source-catalog",
                str(catalog),
                *OPTIONS,
                "--out-dir",
                str(rust_output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            timeout=120,
        )
        assert process.returncode == 0, process.stderr
        actual = json.loads((rust_output / "contaminated_pair_class_audit.json").read_text(encoding="utf-8"))
        compare(actual, expected)
        artifacts = (
            "contaminated_pair_class_selection.csv",
            "contaminated_pair_class_summary.csv",
            "contaminated_pair_quality_reason_summary.csv",
        )
        equal_artifacts = {}
        for name in artifacts:
            rust_bytes = (rust_output / name).read_bytes()
            python_bytes = (python_output / name).read_bytes()
            same = rust_bytes == python_bytes
            equal_artifacts[name] = same
            if not same:
                actual_lines = rust_bytes.decode("utf-8-sig").splitlines()
                expected_lines = python_bytes.decode("utf-8-sig").splitlines()
                diff = "\n".join(
                    difflib.unified_diff(expected_lines, actual_lines, fromfile="python", tofile="rust")
                )
                raise AssertionError(f"CSV bytes differ: {name}\n{diff}")
        compare_csv_fields(
            rust_output / "contaminated_pair_injection.csv",
            python_output / "contaminated_pair_injection.csv",
        )
        compare(
            json.loads((rust_output / "contaminated_pair_injection.json").read_text(encoding="utf-8")),
            json.loads((python_output / "contaminated_pair_injection.json").read_text(encoding="utf-8")),
            "injection_artifact",
        )
        assert hashlib.sha256(FRAME.read_bytes()).hexdigest() == before
        return {
            "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
            "python_baseline_sha256": baseline_hash,
            "frame_sha256": before,
            "all_result_fields_match": True,
            "csv_artifacts_byte_identical": equal_artifacts,
            "injection_csv_fields_match_within_tolerance": True,
            "injection_json_fields_match": True,
            "source_fits_unchanged": True,
            "selected_anchor_count": len(actual["selected_rows"]),
            "injection_condition_count": len(actual["injection"]["rows"]),
            "class_summary_count": len(actual["class_summary"]),
        }


if __name__ == "__main__":
    executable = ROOT / "backend/target/debug/rst19-contaminated-pair-class.exe"
    if os.name != "nt":
        executable = executable.with_suffix("")
    if len(sys.argv) == 2:
        executable = Path(sys.argv[1])
    print(json.dumps(verify(executable), ensure_ascii=False, indent=2))
