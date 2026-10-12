"""Compare the Rust source-separation audit with the Python implementation."""
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
from rst19.source_separation_audit import (  # noqa: E402
    run_source_separation_audit,
    write_source_separation_artifacts,
)

DEFAULT_RADII = [3.0, 4.0, 4.5, 5.0, 6.0]
CSV_ARTIFACTS = (
    "source_separation_thresholds.csv",
    "source_separation_classes.csv",
    "source_separation_targets.csv",
)
JSON_ARTIFACT = "source_separation_summary.json"
CATALOG = (
    "\ufeffdetection_id,x,y,peak_x,peak_y,quality_passed,feature_class\n"
    "1,0,0,0,0,True,compact_quality\n"
    "2,2.5,0,3,0,True,compact_quality\n"
    "3,10,0,10,0,False,weak_or_background\n"
    "4,10,5,10,5,False,shape_outlier\n"
)
TIED_CATALOG = (
    "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class\n"
    "1,0,0,0,0,True,compact_quality\n"
    "2,0,0,0,0,False,compact_quality\n"
    "3,1,0,1,0,True,linear_artifact\n"
    "4,0,1,0,1,False,linear_artifact\n"
)


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


def make_requests(
    catalog: Path,
    output: Path,
    *,
    radii: list[float] | None,
    target_ids: list[int],
) -> tuple[dict[str, object], dict[str, object]]:
    args = [str(catalog)]
    engine: dict[str, object] = {
        "operation": "audit.source_separation",
        "catalog_path": str(catalog),
        "target_ids": target_ids,
    }
    if radii is not None:
        args.extend(("--radii-px", *(str(radius) for radius in radii)))
        engine["radii_px"] = radii
    for target_id in target_ids:
        args.extend(("--target-id", str(target_id)))
    args.extend(("--out-dir", str(output)))
    worker = {
        "operation": "tool",
        "tool": "rst19-source-separation-audit",
        "args": args,
        "out_dir": str(output),
    }
    return worker, engine


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    error_cases = []
    with tempfile.TemporaryDirectory(prefix="rst19-source-separation-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_text(CATALOG, encoding="utf-8", newline="")
        tied_catalog = root / "tied coordinates.csv"
        tied_catalog.write_text(TIED_CATALOG, encoding="utf-8", newline="")
        before = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (catalog, tied_catalog)
        }

        for case, path, radii, target_ids in (
            ("default-radii-no-target", catalog, None, []),
            ("custom-unsorted-deduplicated-radii", catalog, [6.0, 3.0, 4.0, 3.0], [2, 1, 2]),
            ("duplicate-coordinate-nearest", tied_catalog, [1.0, 2.0], [1, 2]),
        ):
            kwargs = {"target_ids": target_ids}
            if radii is not None:
                kwargs["radii_px"] = radii
            python_result = run_source_separation_audit(path, **kwargs)
            expected = python_result.as_dict()
            python_out = root / f"{case}-python"
            native_out = root / f"{case}-native"
            write_source_separation_artifacts(python_result, python_out)
            worker_request, engine_request = make_requests(
                path, native_out, radii=radii, target_ids=target_ids,
            )
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
                "radius_count": len(expected["thresholds"]),
                "target_count": len(expected["targets"]),
                "result_fields_match": True,
                "csv_artifacts_byte_identical": True,
                "json_fields_match": True,
            })

        invalid_catalogs = (
            ("missing-required-fields", "detection_id,x,y\n1,0,0\n2,1,1\n", []),
            (
                "invalid-quality-boolean",
                CATALOG.replace(",True,compact_quality", ",perhaps,compact_quality", 1),
                [],
            ),
            (
                "duplicate-detection-id",
                CATALOG.replace("4,10,5,10,5", "1,10,5,10,5"),
                [],
            ),
        )
        for case, contents, target_ids in invalid_catalogs:
            path = root / f"{case}.csv"
            path.write_text(contents, encoding="utf-8", newline="")
            try:
                run_source_separation_audit(path, target_ids=target_ids)
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {case}")
            worker_request, engine_request = make_requests(
                path, root / f"{case}-out", radii=None, target_ids=target_ids,
            )
            assert invoke_error(engine, engine_request) == expected_error, case
            assert invoke_error(worker, worker_request) == expected_error, case
            error_cases.append({"case": case, "error_matches_python": True})

        try:
            run_source_separation_audit(catalog, target_ids=(99,))
        except ValueError as error:
            expected_error = str(error)
        else:
            raise AssertionError("Python baseline unexpectedly accepted a missing target")
        worker_request, engine_request = make_requests(
            catalog, root / "missing-target-out", radii=[3.0], target_ids=[99],
        )
        assert invoke_error(engine, engine_request) == expected_error
        assert invoke_error(worker, worker_request) == expected_error
        error_cases.append({"case": "missing-target", "error_matches_python": True})

        try:
            run_source_separation_audit(catalog, radii_px=(-1.0,))
        except ValueError as error:
            expected_error = str(error)
        else:
            raise AssertionError("Python baseline unexpectedly accepted a nonpositive radius")
        worker_request, engine_request = make_requests(
            catalog, root / "invalid-radius-out", radii=[-1.0], target_ids=[],
        )
        assert invoke_error(engine, engine_request) == expected_error
        assert invoke_error(worker, worker_request) == expected_error
        error_cases.append({"case": "nonpositive-radius", "error_matches_python": True})

        assert {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (catalog, tied_catalog)
        } == before

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
