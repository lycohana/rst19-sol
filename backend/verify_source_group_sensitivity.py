"""Compare the Rust source-group radius scan with the Python implementation."""
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
from rst19.source_group_sensitivity import (  # noqa: E402
    run_source_group_sensitivity,
    write_source_group_sensitivity_artifacts,
)

DEFAULT_RADII = (2.5, 3.0, 3.5, 4.0)
CSV_ARTIFACTS = (
    "source_group_radius_sensitivity.csv",
    "source_group_radius_targets.csv",
)
JSON_ARTIFACT = "source_group_radius_sensitivity.json"
CATALOG = (
    "\ufeffdetection_id,x,y,peak_x,peak_y,peak,flux_snr,filter_snr,"
    "deblend_delta_bic,deblend_component_snr,quality_passed,feature_class,flags\n"
    "1,0,0,0,0,3992,204.6,1195,-4.09,0.87,False,range_anomaly,CODE_PATTERN\n"
    "2,2.738,0,4,0,569,115.5,300,-4.09,0.87,False,crowded_blend,UNRESOLVED_BLEND\n"
    "3,20,0,20,0,250,12,20,,,True,compact_quality,\n"
    "4,40,0,40,0,900,20,100,12,6,True,compact_quality,\n"
    "5,42,0,42,0,400,10,60,,,True,compact_quality,\n"
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


def requests(
    catalog: Path,
    output: Path,
    *,
    radii: list[float] | None,
    target_ids: list[int],
    workers: int = 1,
) -> tuple[dict[str, object], dict[str, object]]:
    args = [str(catalog)]
    engine: dict[str, object] = {
        "operation": "audit.source_group_sensitivity",
        "catalog_path": str(catalog),
        "target_ids": target_ids,
        "workers": workers,
    }
    if radii is not None:
        for radius in radii:
            args.extend(("--group-radius", str(radius)))
        engine["group_radii_px"] = radii
    for target_id in target_ids:
        args.extend(("--target-id", str(target_id)))
    args.extend(("--workers", str(workers), "--out-dir", str(output)))
    worker = {
        "operation": "tool",
        "tool": "rst19-source-group-sensitivity",
        "args": args,
        "out_dir": str(output),
    }
    return worker, engine


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    error_cases = []
    with tempfile.TemporaryDirectory(prefix="rst19-source-group-sensitivity-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_text(CATALOG, encoding="utf-8", newline="")
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()

        for case, radii, target_ids, workers in (
            ("default-radii-no-target", None, [], 1),
            ("ordered-custom-radii-parallel-targets", [4.0, 2.5, 3.0], [2, 1, 2], 2),
        ):
            kwargs: dict[str, object] = {"target_ids": target_ids, "workers": workers}
            if radii is not None:
                kwargs["group_radii_px"] = tuple(radii)
            python_result = run_source_group_sensitivity(catalog, **kwargs)
            expected = python_result.as_dict()
            python_out = root / f"{case}-python"
            native_out = root / f"{case}-native"
            write_source_group_sensitivity_artifacts(python_result, python_out)
            worker_request, engine_request = requests(
                catalog, native_out, radii=radii, target_ids=target_ids, workers=workers,
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
                "radius_count": len(expected["rows"]),
                "target_count": len(expected["targets"]),
                "result_fields_match": True,
                "csv_artifacts_byte_identical": True,
                "json_fields_match": True,
            })

        invalid_cases = (
            ("duplicate-radii", [3.0, 3.0], [], 1),
            ("nonpositive-radius", [0.0], [], 1),
            ("nonpositive-workers", [3.0], [], 0),
            ("missing-target", [3.0], [99], 1),
        )
        for case, radii, target_ids, workers in invalid_cases:
            try:
                run_source_group_sensitivity(
                    catalog,
                    group_radii_px=tuple(radii),
                    target_ids=target_ids,
                    workers=workers,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {case}")
            worker_request, engine_request = requests(
                catalog, root / f"{case}-out", radii=radii, target_ids=target_ids, workers=workers,
            )
            assert invoke_error(engine, engine_request) == expected_error, case
            assert invoke_error(worker, worker_request) == expected_error, case
            error_cases.append({"case": case, "error_matches_python": True})

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
