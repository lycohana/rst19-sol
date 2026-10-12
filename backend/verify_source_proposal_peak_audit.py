"""Compare Rust source-proposal/raw-peak audit outputs with Python."""
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
from rst19.source_proposal_peak_audit import (  # noqa: E402
    run_source_proposal_peak_audit,
    write_source_proposal_peak_artifacts,
)

CSV_ARTIFACTS = (
    "source_proposal_peak_rows.csv",
    "source_proposal_peak_summary.csv",
    "source_proposal_peak_targets.csv",
)
JSON_ARTIFACT = "source_proposal_peak_audit.json"

CATALOG_ROWS = (
    (1, "compact_quality", "紧凑,候选", "True", "", "gaussian|dog_narrow|dog_broad", "10", "20"),
    (2, "compact_quality", "compact label", "True", "", "gaussian", "8", "11"),
    (3, "compact_quality", "compact label", "False", "UNRESOLVED_BLEND", "dog_narrow|dog_broad", "nan", "inf"),
    (4, "crowded_blend", "blend", "False", "UNRESOLVED_BLEND", "gaussian|dog_narrow", "-2", "2"),
    (5, "crowded_blend", "blend", "True", "", "", "None", "null"),
    (6, "crowded_blend", "blend", "False", "RANGE", "gaussian||dog_narrow|dog_narrow", "6.5", "7.5"),
)
PEAK_ROWS = (
    (1, "True", "True", "True", "0"),
    (2, "True", "False", "False", "4"),
    (3, "False", "False", "False", "7"),
    (4, "True", "False", "False", "9"),
    (5, "True", "True", "False", "0"),
    (6, "True", "False", "False", "2.5"),
)
CATALOG_FIELDS = (
    "detection_id", "feature_class", "feature_class_label", "quality_passed",
    "flags", "proposal_methods", "flux_snr", "filter_snr",
)
PEAK_FIELDS = (
    "detection_id", "raw_valid", "raw_local_maximum_r1",
    "raw_unique_local_maximum_r1", "raw_local_max_gap_adu_r1",
)


def write_csv(path: Path, fields: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def invoke(executable: Path, request: dict[str, object]) -> dict:
    process = subprocess.run(
        [str(executable.resolve())],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
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
        timeout=90,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode != 0 and len(events) == 1 and not events[0]["ok"], events
    return events[0]["error"]


def requests_for(catalog: Path, peak: Path, output: Path, targets: list[int]) -> tuple[dict, dict]:
    args = [str(catalog), str(peak)]
    for target_id in targets:
        args.extend(("--target-id", str(target_id)))
    args.extend(("--out-dir", str(output)))
    worker = {
        "operation": "tool",
        "tool": "rst19-source-proposal-peak-audit",
        "args": args,
        "out_dir": str(output.with_name(output.name + "-job")),
    }
    engine = {
        "operation": "audit.source_proposal_peak",
        "catalog_path": str(catalog),
        "peak_consistency_path": str(peak),
        "target_ids": targets,
    }
    return worker, engine


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rst19-source-proposal-peak-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        peak = root / "raw peak consistency.csv"
        write_csv(catalog, CATALOG_FIELDS, CATALOG_ROWS)
        write_csv(peak, PEAK_FIELDS, PEAK_ROWS)
        original_hashes = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (catalog, peak)
        }

        cases = []
        for case, target_ids in (("default-no-targets", []), ("ordered-deduplicated-targets", [4, 2, 4, 6])):
            expected = run_source_proposal_peak_audit(catalog, peak, target_ids=target_ids).as_dict()
            python_out = root / f"{case}-python"
            native_out = root / f"{case}-native"
            python_result = run_source_proposal_peak_audit(catalog, peak, target_ids=target_ids)
            write_source_proposal_peak_artifacts(python_result, python_out)
            worker_request, engine_request = requests_for(catalog, peak, native_out, target_ids)
            worker_payload = invoke(worker, worker_request)
            engine_result = invoke(engine, engine_request)
            if worker_payload["result"] != expected:
                for key in expected:
                    if worker_payload["result"].get(key) != expected.get(key):
                        print("worker field mismatch", case, key)
                        print("native:", worker_payload["result"].get(key))
                        print("python:", expected.get(key))
                raise AssertionError(case)
            if engine_result != expected:
                for key in expected:
                    if engine_result.get(key) != expected.get(key):
                        print("engine field mismatch", case, key)
                        print("native:", engine_result.get(key))
                        print("python:", expected.get(key))
                raise AssertionError(case)
            for artifact in CSV_ARTIFACTS:
                assert (native_out / artifact).read_bytes() == (python_out / artifact).read_bytes(), (case, artifact)
            assert json.loads((native_out / JSON_ARTIFACT).read_text(encoding="utf-8")) == json.loads(
                (python_out / JSON_ARTIFACT).read_text(encoding="utf-8")
            ), case
            cases.append({
                "case": case,
                "source_count": len(expected["rows"]),
                "class_method_group_count": len(expected["summaries"]),
                "target_count": len(expected["targets"]),
                "result_fields_match": True,
                "csv_artifacts_byte_identical": True,
                "json_fields_match": True,
            })

        invalid_inputs = (
            ("missing-catalog-columns", CATALOG_FIELDS[:-1], CATALOG_ROWS),
            ("missing-peak-columns", PEAK_FIELDS[:-1], PEAK_ROWS),
        )
        error_cases = []
        for case, fields, rows in invalid_inputs:
            path = root / f"{case}.csv"
            write_csv(path, fields, rows)
            cpath = path if "catalog" in case else catalog
            ppath = path if "peak" in case else peak
            try:
                run_source_proposal_peak_audit(cpath, ppath)
            except (OSError, ValueError, RuntimeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {case}")
            worker_request, engine_request = requests_for(cpath, ppath, root / f"{case}-out", [])
            assert invoke_error(engine, engine_request) == expected_error, case
            assert invoke_error(worker, worker_request) == expected_error, case
            error_cases.append({"case": case, "error_matches_python": True})

        catalog_text = catalog.read_text(encoding="utf-8-sig")
        peak_text = peak.read_text(encoding="utf-8-sig")
        invalid_catalogs = (
            ("invalid-quality-boolean", catalog_text.replace(",True,,gaussian", ",perhaps,,gaussian", 1)),
            ("duplicate-catalog-id", catalog_text.replace("2,compact_quality", "1,compact_quality", 1)),
            ("invalid-catalog-id", catalog_text.replace("1,compact_quality", "bad,compact_quality", 1)),
            ("empty-feature-class", catalog_text.replace("compact_quality,compact label", ",compact label", 1)),
        )
        for case, text in invalid_catalogs:
            path = root / f"{case}.csv"
            path.write_text(text, encoding="utf-8", newline="")
            try:
                run_source_proposal_peak_audit(path, peak)
            except (OSError, ValueError, RuntimeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {case}")
            worker_request, engine_request = requests_for(path, peak, root / f"{case}-out", [])
            assert invoke_error(engine, engine_request) == expected_error, case
            assert invoke_error(worker, worker_request) == expected_error, case
            error_cases.append({"case": case, "error_matches_python": True})

        invalid_peak = (
            ("invalid-raw-valid-boolean", peak_text.replace(",True,True,True,0", ",perhaps,True,True,0", 1)),
            ("duplicate-peak-id", peak_text.replace("2,True", "1,True", 1)),
            ("invalid-peak-id", peak_text.replace("1,True", "bad,True", 1)),
        )
        for case, text in invalid_peak:
            path = root / f"{case}.csv"
            path.write_text(text, encoding="utf-8", newline="")
            try:
                run_source_proposal_peak_audit(catalog, path)
            except (OSError, ValueError, RuntimeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {case}")
            worker_request, engine_request = requests_for(catalog, path, root / f"{case}-out", [])
            assert invoke_error(engine, engine_request) == expected_error, case
            assert invoke_error(worker, worker_request) == expected_error, case
            error_cases.append({"case": case, "error_matches_python": True})

        # ID-set mismatch and missing targets exercise the strict join contract.
        mismatch = root / "mismatched peaks.csv"
        write_csv(mismatch, PEAK_FIELDS, PEAK_ROWS[:-1])
        try:
            run_source_proposal_peak_audit(catalog, mismatch)
        except ValueError as error:
            expected_error = str(error)
        else:
            raise AssertionError("Python baseline unexpectedly accepted mismatched IDs")
        worker_request, engine_request = requests_for(catalog, mismatch, root / "mismatch-out", [])
        assert invoke_error(engine, engine_request) == expected_error
        assert invoke_error(worker, worker_request) == expected_error
        error_cases.append({"case": "mismatched-id-sets", "error_matches_python": True})

        try:
            run_source_proposal_peak_audit(catalog, peak, target_ids=(999,))
        except ValueError as error:
            expected_error = str(error)
        else:
            raise AssertionError("Python baseline unexpectedly accepted a missing target")
        worker_request, engine_request = requests_for(catalog, peak, root / "missing-target-out", [999])
        assert invoke_error(engine, engine_request) == expected_error
        assert invoke_error(worker, worker_request) == expected_error
        error_cases.append({"case": "missing-target", "error_matches_python": True})

        assert {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (catalog, peak)
        } == original_hashes

    report = {
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cases": cases,
        "error_cases": error_cases,
        "all_result_fields_match": True,
        "all_csv_artifacts_byte_identical": True,
        "all_json_fields_match": True,
        "all_inputs_unchanged": True,
    }
    (out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.worker, arguments.engine, arguments.out), ensure_ascii=False, indent=2))
