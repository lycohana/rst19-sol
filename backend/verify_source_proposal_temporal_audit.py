"""Compare Rust temporal source-proposal audit outputs with Python."""
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
from rst19.source_proposal_temporal_audit import (  # noqa: E402
    run_source_proposal_temporal_audit,
    write_source_proposal_temporal_artifacts,
)

CSV_ARTIFACTS = (
    "source_proposal_temporal_rows.csv",
    "source_proposal_temporal_summary.csv",
    "source_proposal_temporal_targets.csv",
)
JSON_ARTIFACT = "source_proposal_temporal_audit.json"
CATALOG_FIELDS = (
    "detection_id", "feature_class", "feature_class_label", "quality_passed", "flags", "proposal_methods",
)
CATALOG_ROWS = (
    (1, "compact_quality", "紧凑", "True", "", "gaussian|dog_narrow|dog_broad"),
    (2, "compact_quality", "紧凑", "True", "", "dog_narrow|dog_broad"),
    (3, "crowded_blend", "拥挤", "False", "UNRESOLVED_BLEND", "gaussian"),
    (4, "crowded_blend", "拥挤", "False", "UNRESOLVED_BLEND", "dog_narrow|dog_broad"),
)
DIAGNOSTIC_FIELDS = (
    "detection_id", "feature_class", "feature_class_label", "diagnostic_subgroup",
    "diagnostic_subgroup_label", "quality_passed", "proposal_methods", "candidate_presence",
    "candidate_same_subgroup_presence", "quality_presence", "quality_same_subgroup_presence",
    "frame_count", "required_presence", "association_radius_px",
)
DIAGNOSTIC_ROWS = (
    (3, "crowded_blend", "拥挤", "blend_unresolved", "未分辨", "False", "gaussian", 4, 3, 0, 0, 5, 4, 1.0),
    (4, "crowded_blend", "拥挤", "blend_unresolved", "未分辨", "False", "dog_narrow|dog_broad", 5, 5, 2, 1, 5, 4, 1.0),
)
SUBGROUP_FIELDS = (
    "feature_class", "feature_class_label", "proposal_subgroup", "proposal_subgroup_label",
    "anchor_count", "anchor_quality_count", "frame_count", "required_presence", "association_radius_px",
    "candidate_mean_presence", "candidate_presence_ge_required_count", "candidate_presence_all_frames_count",
    "quality_mean_presence", "quality_presence_ge_required_count", "quality_presence_all_frames_count",
)
SUBGROUP_ROWS = (
    ("compact_quality", "紧凑", "all_three", "三路", 1, 1, 5, 4, 1.0, 5, 1, 1, 5, 1, 1),
    ("compact_quality", "紧凑", "dog_only_no_deblend", "DoG", 1, 1, 5, 4, 1.0, 2, 0, 0, 1, 0, 0),
)


def write_csv(path: Path, fields: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def invoke(executable: Path, request: dict[str, object]) -> dict:
    process = subprocess.run(
        [str(executable.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=90, check=True,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(events) == 1 and events[0]["ok"], events
    return events[0]["payload"]


def invoke_error(executable: Path, request: dict[str, object]) -> str:
    process = subprocess.run(
        [str(executable.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode != 0 and len(events) == 1 and not events[0]["ok"], events
    return events[0]["error"]


def requests_for(catalog: Path, diagnostic: Path, subgroup: Path, output: Path, targets: list[int]) -> tuple[dict, dict]:
    args = [str(catalog), str(diagnostic), str(subgroup)]
    for target_id in targets:
        args.extend(("--target-id", str(target_id)))
    args.extend(("--out-dir", str(output)))
    worker = {
        "operation": "tool", "tool": "rst19-source-proposal-temporal-audit", "args": args,
        "out_dir": str(output.with_name(output.name + "-job")),
    }
    engine = {
        "operation": "audit.source_proposal_temporal", "catalog_path": str(catalog),
        "diagnostic_sources_path": str(diagnostic), "subgroup_persistence_path": str(subgroup),
        "target_ids": targets,
    }
    return worker, engine


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rst19-source-proposal-temporal-") as temporary:
        root = Path(temporary)
        catalog, diagnostic, subgroup = (root / name for name in (
            "source catalog.csv", "diagnostic sources.csv", "subgroup persistence.csv",
        ))
        write_csv(catalog, CATALOG_FIELDS, CATALOG_ROWS)
        write_csv(diagnostic, DIAGNOSTIC_FIELDS, DIAGNOSTIC_ROWS)
        write_csv(subgroup, SUBGROUP_FIELDS, SUBGROUP_ROWS)
        inputs = (catalog, diagnostic, subgroup)
        original_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}

        cases = []
        for case, targets in (("default-no-targets", []), ("ordered-deduplicated-targets", [1, 3, 1, 4])):
            expected_result = run_source_proposal_temporal_audit(
                catalog, diagnostic, subgroup, target_ids=targets,
            )
            expected = expected_result.as_dict()
            python_out, native_out = root / f"{case}-python", root / f"{case}-native"
            write_source_proposal_temporal_artifacts(expected_result, python_out)
            worker_request, engine_request = requests_for(catalog, diagnostic, subgroup, native_out, targets)
            worker_payload = invoke(worker, worker_request)
            engine_result = invoke(engine, engine_request)
            for name, native_result in (("worker", worker_payload["result"]), ("engine", engine_result)):
                if native_result != expected:
                    for key in expected:
                        if native_result.get(key) != expected.get(key):
                            print(name, case, key, "native:", native_result.get(key), "python:", expected.get(key))
                    raise AssertionError(f"{name} result mismatch: {case}")
            for artifact in CSV_ARTIFACTS:
                assert (native_out / artifact).read_bytes() == (python_out / artifact).read_bytes(), (case, artifact)
            assert json.loads((native_out / JSON_ARTIFACT).read_text(encoding="utf-8")) == json.loads(
                (python_out / JSON_ARTIFACT).read_text(encoding="utf-8")
            ), case
            cases.append({
                "case": case, "source_count": expected["source_row_count"],
                "class_method_group_count": len(expected["summary_rows"]),
                "target_count": len(expected["target_rows"]), "result_fields_match": True,
                "csv_artifacts_byte_identical": True, "json_fields_match": True,
            })

        error_cases = []

        def compare_error(case: str, cpath: Path = catalog, dpath: Path = diagnostic, spath: Path = subgroup,
                          targets: list[int] | None = None) -> None:
            try:
                run_source_proposal_temporal_audit(cpath, dpath, spath, target_ids=targets or [])
            except (OSError, ValueError, RuntimeError) as error:
                expected_error = str(error)
            else:
                raise AssertionError(f"Python baseline unexpectedly accepted {case}")
            worker_request, engine_request = requests_for(cpath, dpath, spath, root / f"{case}-out", targets or [])
            assert invoke_error(engine, engine_request) == expected_error, case
            assert invoke_error(worker, worker_request) == expected_error, case
            error_cases.append({"case": case, "error_matches_python": True})

        bad_catalog = root / "missing catalog column.csv"
        write_csv(bad_catalog, CATALOG_FIELDS[:-1], CATALOG_ROWS)
        compare_error("missing-catalog-column", cpath=bad_catalog)
        bad_diagnostic = root / "missing diagnostic column.csv"
        write_csv(bad_diagnostic, DIAGNOSTIC_FIELDS[:-1], tuple(row[:-1] for row in DIAGNOSTIC_ROWS))
        compare_error("missing-diagnostic-columns", dpath=bad_diagnostic)
        incomplete_optional = root / "incomplete optional metadata.csv"
        write_csv(incomplete_optional, DIAGNOSTIC_FIELDS[:-3] + ("frame_count",), tuple(row[:-3] + (row[-3],) for row in DIAGNOSTIC_ROWS))
        compare_error("incomplete-optional-metadata", dpath=incomplete_optional)
        mismatch = root / "missing noncompact diagnostic.csv"
        write_csv(mismatch, DIAGNOSTIC_FIELDS, DIAGNOSTIC_ROWS[:1])
        compare_error("noncompact-id-set-mismatch", dpath=mismatch)
        invalid_presence = root / "invalid presence.csv"
        bad_rows = (DIAGNOSTIC_ROWS[0][:-7] + (6,) + DIAGNOSTIC_ROWS[0][-6:], DIAGNOSTIC_ROWS[1])
        write_csv(invalid_presence, DIAGNOSTIC_FIELDS, bad_rows)
        compare_error("invalid-temporal-bounds", dpath=invalid_presence)
        compare_error("missing-target", targets=[999])

        assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs} == original_hashes

    report = {
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(), "cases": cases,
        "error_cases": error_cases, "all_result_fields_match": True,
        "all_csv_artifacts_byte_identical": True, "all_json_fields_match": True,
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
