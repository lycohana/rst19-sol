"""Compare the Rust source-group audit with the Python implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.source_group_audit import (  # noqa: E402
    run_candidate_group_audit,
    write_candidate_group_artifacts,
)

CSV_ARTIFACTS = (
    "source_groups.csv",
    "source_group_class_summary.csv",
    "source_group_targets.csv",
)
JSON_ARTIFACT = "source_group_summary.json"
CATALOG = """\ufeffdetection_id,x,y,peak_x,peak_y,peak,flux_snr,filter_snr,fwhm,deblend_delta_bic,deblend_component_snr,quality_passed,feature_class,flags
1,0,0,0,0,3992,204.6,1195,2.59,-4.09,0.87,False,range_anomaly,CODE_PATTERN
2,2.738,0,4,0,569,115.5,300,2.44,-4.09,0.87,False,crowded_blend,UNRESOLVED_BLEND
3,20,0,20,0,250,12,20,2.0,,,True,compact_quality,
4,40,0,40,0,900,20,100,2.0,12,6,True,compact_quality,
5,42,0,42,0,400,10,60,2.0,,,True,compact_quality,
"""


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


def write_requests(
    catalog: Path,
    output: Path,
    target_ids: list[int],
    group_radius: float | None,
) -> tuple[dict[str, object], dict[str, object]]:
    args = [str(catalog), *(part for value in target_ids for part in ("--target-id", str(value)))]
    engine: dict[str, object] = {
        "operation": "audit.source_groups",
        "catalog_path": str(catalog),
        "target_ids": target_ids,
    }
    if group_radius is not None:
        args.extend(("--group-radius", str(group_radius)))
        engine["group_radius_px"] = group_radius
    args.extend(("--out-dir", str(output)))
    worker = {
        "operation": "tool",
        "tool": "rst19-source-group-audit",
        "args": args,
        "out_dir": str(output),
    }
    return worker, engine


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    error_cases = []
    with tempfile.TemporaryDirectory(prefix="rst19-source-groups-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_text(CATALOG, encoding="utf-8", newline="")
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()

        for case, target_ids, radius in (
            ("default-radius-with-transitive-components", [2, 1, 2], None),
            ("explicit-radius-and-isolated-targets", [4, 2, 1], 2.0),
        ):
            python_result = run_candidate_group_audit(
                catalog, target_ids=target_ids, **({} if radius is None else {"group_radius_px": radius}),
            )
            expected = python_result.as_dict()
            python_out = root / f"{case}-python"
            native_out = root / f"{case}-native"
            write_candidate_group_artifacts(python_result, python_out)
            worker_request, engine_request = write_requests(catalog, native_out, target_ids, radius)
            worker_payload = invoke(worker, worker_request)
            engine_payload = invoke(engine, engine_request)
            assert worker_payload["result"] == expected, case
            assert engine_payload == expected, case
            for artifact in CSV_ARTIFACTS:
                assert (native_out / artifact).read_bytes() == (python_out / artifact).read_bytes(), (case, artifact)
            assert json.loads((native_out / JSON_ARTIFACT).read_text(encoding="utf-8")) == json.loads(
                (python_out / JSON_ARTIFACT).read_text(encoding="utf-8")
            ), case
            cases.append({
                "case": case,
                "source_count": expected["source_count"],
                "group_count": expected["group_count"],
                "target_count": len(expected["targets"]),
                "result_fields_match": True,
                "csv_artifacts_byte_identical": True,
                "json_fields_match": True,
            })

        bad_catalogs = (
            ("invalid-quality-boolean", CATALOG.replace(",False,range_anomaly,", ",maybe,range_anomaly,")),
            ("duplicate-detection-id", CATALOG.replace("2,2.738,", "1,2.738,")),
            ("invalid-coordinate", CATALOG.replace("3,20,0,", "3,not-a-number,0,")),
        )
        for case, contents in bad_catalogs:
            path = root / f"{case}.csv"
            path.write_text(contents, encoding="utf-8", newline="")
            try:
                run_candidate_group_audit(path)
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {case}")
            worker_request, engine_request = write_requests(path, root / f"{case}-out", [], None)
            assert invoke_error(engine, engine_request) == expected_error, case
            assert invoke_error(worker, worker_request) == expected_error, case
            error_cases.append({"case": case, "error_matches_python": True})

        missing_target_request = {
            "operation": "audit.source_groups",
            "catalog_path": str(catalog),
            "target_ids": [99],
        }
        try:
            run_candidate_group_audit(catalog, target_ids=(99,))
        except ValueError as error:
            expected_error = str(error)
        else:
            raise AssertionError("Python baseline unexpectedly accepted a missing target")
        assert invoke_error(engine, missing_target_request) == expected_error
        worker_request, _ = write_requests(catalog, root / "missing-target-out", [99], None)
        assert invoke_error(worker, worker_request) == expected_error
        error_cases.append({"case": "missing-target", "error_matches_python": True})

        assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before

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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.worker, arguments.engine, arguments.out), ensure_ascii=False, indent=2))
