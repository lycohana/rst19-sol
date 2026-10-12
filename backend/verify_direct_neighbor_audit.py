"""Compare the Rust direct-neighbor audit with its Python reference."""
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
from rst19.direct_neighbor_audit import (  # noqa: E402
    run_direct_neighbor_audit,
    write_direct_neighbor_artifacts,
)

FIELDS = [
    "detection_id",
    "x",
    "y",
    "peak_x",
    "peak_y",
    "peak",
    "flux_snr",
    "filter_snr",
    "fwhm",
    "deblend_delta_bic",
    "deblend_component_snr",
    "quality_passed",
    "feature_class",
    "flags",
]
ROWS = [
    ["1", "0", "0", "0", "0", "3992", "204.6", "1195", "2.59", "-4.09", "0.87", "False", "range_anomaly", "CODE_PATTERN"],
    ["2", "2.738", "0", "4", "0", "569", "115.5", "300", "2.44", "-4.09", "0.87", "False", "crowded_blend", "UNRESOLVED_BLEND"],
    ["3", "10", "0", "10", "0", "250", "12", "20", "2", "", "", "True", "compact_quality", ""],
    ["4", "20", "0", "20", "0", "900", "20", "100", "2", "12", "6", "True", "compact_quality", ""],
    ["5", "23", "0", "23", "0", "400", "10", "60", "2", "", "", "True", "compact_quality", ""],
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


def write_catalog(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)


def requests(
    catalog: Path,
    output: Path,
    *,
    target_ids=(1, 2),
    radius: float | None = 4.0,
) -> tuple[dict, dict]:
    args = [
        str(catalog),
        *[item for target in target_ids for item in ("--target-id", str(target))],
        "--psf-fwhm", "2",
        "--delta-bic-min", "10",
        "--component-snr-min", "5",
        "--out-dir", str(output),
    ]
    if radius is not None:
        args[1:1] = ["--pair-radius", str(radius)]
    worker = {
        "operation": "tool",
        "tool": "rst19-direct-neighbor-audit",
        "args": args,
        "out_dir": str(output.with_name(output.name + "-job")),
    }
    engine = {
        "operation": "audit.direct_neighbor_pairs",
        "catalog_path": str(catalog),
        "target_ids": list(target_ids),
        "psf_fwhm": 2.0,
        "pair_radius_px": radius,
        "delta_bic_min": 10.0,
        "component_snr_min": 5.0,
    }
    return worker, engine


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    errors = []
    with tempfile.TemporaryDirectory(prefix="rst19-direct-neighbor-") as temporary:
        root = Path(temporary)
        catalog = root / "候选 source catalog.csv"
        write_catalog(catalog, ROWS)
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()
        expected_result = run_direct_neighbor_audit(
            catalog,
            target_ids=(2, 1),
            psf_fwhm=2.0,
            pair_radius_px=4.0,
            delta_bic_min=10.0,
            component_snr_min=5.0,
        )
        expected = expected_result.as_dict()
        python_out = root / "python artifacts"
        native_out = root / "native artifacts"
        write_direct_neighbor_artifacts(expected_result, python_out)
        worker_request, engine_request = requests(
            catalog, native_out, target_ids=(2, 1)
        )
        worker_payload = invoke(worker, worker_request)
        engine_payload = invoke(engine, engine_request)
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        for artifact in (
            "direct_neighbor_pairs.csv",
            "direct_neighbor_summaries.csv",
            "direct_neighbor_targets.csv",
        ):
            assert (native_out / artifact).read_bytes() == (python_out / artifact).read_bytes(), artifact
        assert json.loads(
            (native_out / "direct_neighbor_audit.json").read_text(encoding="utf-8")
        ) == expected
        assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before
        cases.append(
            {
                "case": "direct-edges-and-reversed-target",
                "source_count": expected["source_count"],
                "pair_count": expected["direct_pair_count"],
                "unresolved_pair_count": expected["unresolved_pair_count"],
                "independent_pair_candidate_count": expected["independent_pair_candidate_count"],
                "result_fields_match": True,
                "csv_artifacts_byte_identical": True,
                "json_fields_match": True,
                "input_unchanged": True,
            }
        )

        default_out = root / "default-radius artifacts"
        default_result = run_direct_neighbor_audit(catalog)
        default_expected = default_result.as_dict()
        default_python_out = root / "default-radius python"
        write_direct_neighbor_artifacts(default_result, default_python_out)
        default_worker_request, default_engine_request = requests(
            catalog, default_out, target_ids=(), radius=None
        )
        default_worker = invoke(worker, default_worker_request)
        default_engine = invoke(engine, default_engine_request)
        assert default_worker["result"] == default_expected
        assert default_engine == default_expected
        assert default_expected["pair_radius_px"] == 3.0
        for artifact in (
            "direct_neighbor_pairs.csv",
            "direct_neighbor_summaries.csv",
            "direct_neighbor_targets.csv",
        ):
            assert (default_out / artifact).read_bytes() == (default_python_out / artifact).read_bytes(), artifact
        assert json.loads(
            (default_out / "direct_neighbor_audit.json").read_text(encoding="utf-8")
        ) == default_expected
        cases.append(
            {
                "case": "default-radius-without-target",
                "source_count": default_expected["source_count"],
                "pair_count": default_expected["direct_pair_count"],
                "pair_radius_px": default_expected["pair_radius_px"],
                "target_csv_is_bom_only": (default_out / "direct_neighbor_targets.csv").read_bytes() == b"\xef\xbb\xbf",
                "result_fields_match": True,
                "csv_artifacts_byte_identical": True,
                "json_fields_match": True,
            }
        )

        invalid_bool = [row[:] for row in ROWS]
        invalid_bool[0][11] = "maybe"
        invalid_float = [row[:] for row in ROWS]
        invalid_float[0][5] = "not-a-number"
        error_cases = [
            ("missing-target", ROWS, (1, 99), 4.0),
            ("duplicate-id", ROWS + [ROWS[0]], (1, 2), 4.0),
            ("invalid-bool", invalid_bool, (1, 2), 4.0),
            ("invalid-float", invalid_float, (1, 2), 4.0),
        ]
        for name, rows, target_ids, radius in error_cases:
            error_catalog = root / f"{name}.csv"
            write_catalog(error_catalog, rows)
            try:
                run_direct_neighbor_audit(
                    error_catalog,
                    target_ids=target_ids,
                    psf_fwhm=2.0,
                    pair_radius_px=radius,
                    delta_bic_min=10.0,
                    component_snr_min=5.0,
                )
            except (OSError, ValueError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {name}")
            worker_request, engine_request = requests(
                error_catalog, root / f"{name}-out", target_ids=target_ids, radius=radius
            )
            assert invoke_error(engine, engine_request) == expected_error, name
            assert invoke_error(worker, worker_request) == expected_error, name
            errors.append({"case": name, "error_matches_python": True})

    report = {
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cases": cases,
        "error_cases": errors,
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
