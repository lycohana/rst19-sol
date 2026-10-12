"""Compare the Rust source-peak audit with the Python FITS implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.source_peak_consistency import (  # noqa: E402
    run_source_peak_consistency,
    write_source_peak_consistency_artifacts,
)

CSV_ARTIFACTS = (
    "source_peak_consistency.csv",
    "source_peak_consistency_class_summary.csv",
    "source_peak_consistency_targets.csv",
)
JSON_ARTIFACT = "source_peak_consistency.json"
CATALOG = (
    "detection_id,feature_class,feature_class_label,quality_passed,peak_x,peak_y\n"
    "1,compact_quality,compact,True,5,5\n"
    "2,range_anomaly,range,False,8,5\n"
    "3,crowded_blend,blend,False,12,5\n"
    "4,edge_candidate,edge,True,0,5\n"
    "5,auxiliary_row,aux,False,5,0\n"
    "6,plateau,plateau,True,15,5\n"
)


def write_fits(path: Path) -> None:
    image = np.full((20, 20), 10.0, dtype=np.float32)
    image[5, 5] = 100.0
    image[5, 7] = 30.0
    image[5, 8] = 20.0
    image[5, 9] = 90.0
    image[5, 12] = 80.0
    image[5, 15] = 80.0
    image[5, 0] = 20.0
    image[5, 1] = 30.0
    image[0, 5] = 1000.0
    image[4, 4] = np.nan
    image[6, 6] = np.inf
    cards = (
        "SIMPLE  =                    T",
        "BITPIX  =                  -32",
        "NAXIS   =                    2",
        "NAXIS1  =                   20",
        "NAXIS2  =                   20",
        "END",
    )
    header = "".join(card.ljust(80) for card in cards).encode("ascii")
    header += b" " * ((-len(header)) % 2880)
    pixels = image.astype(">f4", copy=False).tobytes(order="C")
    payload = header + pixels
    payload += b"\0" * ((-len(payload)) % 2880)
    path.write_bytes(payload)


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


def make_requests(
    catalog: Path,
    fits_path: Path,
    output: Path,
    *,
    target_ids: list[int],
) -> tuple[dict[str, object], dict[str, object]]:
    args = [str(catalog), str(fits_path)]
    for target_id in target_ids:
        args.extend(("--target-id", str(target_id)))
    args.extend(("--out-dir", str(output)))
    worker = {
        "operation": "tool",
        "tool": "rst19-source-peak-consistency",
        "args": args,
        "out_dir": str(output.with_name(output.name + "-job")),
    }
    engine = {
        "operation": "audit.source_peak_consistency",
        "catalog_path": str(catalog),
        "fits_path": str(fits_path),
        "target_ids": target_ids,
    }
    return worker, engine


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    error_cases = []
    with tempfile.TemporaryDirectory(prefix="rst19-source-peak-consistency-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        fits_path = root / "original frame.fits"
        catalog.write_text(CATALOG, encoding="utf-8", newline="")
        write_fits(fits_path)
        before = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (catalog, fits_path)
        }

        for case, target_ids in (("default-no-targets", []), ("ordered-deduplicated-targets-edge-and-plateau", [2, 3, 4, 2])):
            python_result = run_source_peak_consistency(catalog, fits_path, target_ids=target_ids)
            expected = python_result.as_dict()
            python_out = root / f"{case}-python"
            native_out = root / f"{case}-native"
            write_source_peak_consistency_artifacts(python_result, python_out)
            worker_request, engine_request = make_requests(
                catalog, fits_path, native_out, target_ids=target_ids,
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
                "source_count": len(expected["rows"]),
                "class_count": len(expected["class_summaries"]),
                "target_count": len(expected["targets"]),
                "image_shape": expected["image_shape"],
                "result_fields_match": True,
                "csv_artifacts_byte_identical": True,
                "json_fields_match": True,
            })

        invalid_catalogs = (
            (
                "missing-required-columns",
                "detection_id,feature_class,quality_passed,peak_x,peak_y\n1,compact,True,1,1\n",
                [],
            ),
            (
                "invalid-quality-boolean",
                CATALOG.replace("range_anomaly,range,False", "range_anomaly,range,perhaps", 1),
                [],
            ),
            (
                "duplicate-detection-id",
                CATALOG.replace("4,edge_candidate", "1,edge_candidate", 1),
                [],
            ),
            (
                "noninteger-peak-coordinate",
                CATALOG.replace("8,5\n3,crowded", "8.5,5\n3,crowded", 1),
                [],
            ),
            (
                "outside-image-coordinate",
                CATALOG.replace("4,edge_candidate,edge,True,0,5", "4,edge_candidate,edge,True,20,5"),
                [],
            ),
        )
        for case, contents, target_ids in invalid_catalogs:
            path = root / f"{case}.csv"
            path.write_text(contents, encoding="utf-8", newline="")
            try:
                run_source_peak_consistency(path, fits_path, target_ids=target_ids)
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {case}")
            worker_request, engine_request = make_requests(
                path, fits_path, root / f"{case}-out", target_ids=target_ids,
            )
            assert invoke_error(engine, engine_request) == expected_error, case
            assert invoke_error(worker, worker_request) == expected_error, case
            error_cases.append({"case": case, "error_matches_python": True})

        try:
            run_source_peak_consistency(catalog, fits_path, target_ids=(99,))
        except ValueError as error:
            expected_error = str(error)
        else:
            raise AssertionError("Python baseline unexpectedly accepted a missing target")
        worker_request, engine_request = make_requests(
            catalog, fits_path, root / "missing-target-out", target_ids=[99],
        )
        assert invoke_error(engine, engine_request) == expected_error
        assert invoke_error(worker, worker_request) == expected_error
        error_cases.append({"case": "missing-target", "error_matches_python": True})

        invalid_radius_request = {
            "operation": "audit.source_peak_consistency",
            "catalog_path": str(catalog),
            "fits_path": str(fits_path),
            "local_max_radii_px": [1, 2],
        }
        try:
            run_source_peak_consistency(catalog, fits_path, local_max_radii_px=(1, 2))
        except ValueError as error:
            expected_error = str(error)
        else:
            raise AssertionError("Python baseline unexpectedly accepted nonstable radii")
        assert invoke_error(engine, invalid_radius_request) == expected_error
        error_cases.append({"case": "nonstable-radii", "error_matches_python": True})

        assert {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (catalog, fits_path)
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
