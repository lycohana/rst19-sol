"""Verify Rust raw-evidence audit routes against the Python baseline."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_detection import compare
from rst19.feature_raw_evidence_audit import (
    RAW_EVIDENCE_METRICS,
    run_feature_raw_evidence_audit,
    write_feature_raw_evidence_artifacts,
)


CSV_ARTIFACT = "feature_raw_evidence_summary.csv"
JSON_ARTIFACT = "feature_raw_evidence_audit.json"
CUSTOM_CLASSES = ["linear_artifact", "compact_quality", "linear_artifact"]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_inputs(root: Path) -> tuple[Path, Path]:
    catalog = root / "源目录.csv"
    raw_summary = root / "原始抽样.csv"
    write_csv(
        catalog,
        ["detection_id", "feature_class", "feature_class_label"],
        [
            {"detection_id": 1, "feature_class": "compact_quality", "feature_class_label": "紧凑"},
            {"detection_id": 2, "feature_class": "compact_quality", "feature_class_label": "紧凑"},
            {"detection_id": 3, "feature_class": "linear_artifact", "feature_class_label": "线状"},
            {"detection_id": 4, "feature_class": "linear_artifact", "feature_class_label": "线状"},
            {"detection_id": 5, "feature_class": "weak_or_background", "feature_class_label": "弱"},
            {"detection_id": 6, "feature_class": "other_rejected", "feature_class_label": "其它拒绝"},
        ],
    )
    fields = ["detection_id", "origin", *RAW_EVIDENCE_METRICS]

    def row(
        detection_id: int,
        origin: str,
        core: str,
        noise: str,
        negative: str,
    ) -> dict[str, object]:
        values: dict[str, object] = {
            "detection_id": detection_id,
            "origin": origin,
            "core_fraction_median": core,
            "noise_median": noise,
            "zero_count_max": "0",
            "frames_zero_count_gt0": "0",
            "frames_negative_extreme_count_gt0": negative,
            "frames_repeated_code_count_gt0": "0",
            "frames_peak_snr_ge5": "15",
            "frames_flux_snr_ge5": "15",
            "frames_support_ge3": "15",
        }
        return values

    write_csv(
        raw_summary,
        fields,
        [
            row(1, "control", "0.8", "8.0", "0"),
            row(2, "control", "0.7", "10.0", "0"),
            row(3, "diagnostic", "0.2", "12.0", "3"),
            row(4, "diagnostic", "0.4", "14.0", "5"),
            row(5, "diagnostic", "0.5", "bad", "0"),
            row(6, "diagnostic", "0.9", "bad", "0"),
        ],
    )
    return catalog, raw_summary


def run_worker(worker: Path, args: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "operation": "tool",
        "tool": "rst19-feature-raw-evidence-audit",
        "args": args,
        "out_dir": str(out_dir.resolve()),
    }
    process = subprocess.run(
        [str(worker.resolve())],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode == 0 and len(events) == 1 and events[0]["event"] == "done" \
        and events[0]["ok"], (process.returncode, process.stdout, process.stderr)
    return events[0]["payload"]


def verify(engine: Path, cli: Path, worker: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    input_dir = out / "inputs"
    input_dir.mkdir()
    catalog, raw_summary = make_inputs(input_dir)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (catalog, raw_summary)
    }

    expected = run_feature_raw_evidence_audit(catalog, raw_summary).as_dict()
    engine_request = {
        "operation": "audit.feature_raw_evidence",
        "catalog_path": str(catalog),
        "raw_summary_path": str(raw_summary),
    }
    engine_process = subprocess.run(
        [str(engine.resolve())],
        input=json.dumps(engine_request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    engine_events = [json.loads(line) for line in engine_process.stdout.splitlines()]
    assert engine_process.returncode == 0 and len(engine_events) == 1 \
        and engine_events[0]["ok"], (
            engine_process.returncode, engine_process.stdout, engine_process.stderr
        )
    compare(engine_events[0]["payload"], expected)
    assert not any(
        row["feature_class"] == "weak_or_background" and row["metric"] == "noise_median"
        for row in expected["rows"]
    ), "non-numeric values should be excluded from finite metric summaries"
    custom_metrics = ["noise_median", "core_fraction_median", "noise_median"]
    custom_expected = run_feature_raw_evidence_audit(
        catalog,
        raw_summary,
        feature_classes=CUSTOM_CLASSES,
        metrics=custom_metrics,
    ).as_dict()
    custom_engine_request = {
        **engine_request,
        "feature_classes": CUSTOM_CLASSES,
        "metrics": custom_metrics,
    }
    custom_engine_process = subprocess.run(
        [str(engine.resolve())],
        input=json.dumps(custom_engine_request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    custom_engine_events = [json.loads(line) for line in custom_engine_process.stdout.splitlines()]
    assert custom_engine_process.returncode == 0 and len(custom_engine_events) == 1 \
        and custom_engine_events[0]["ok"], (
            custom_engine_process.returncode,
            custom_engine_process.stdout,
            custom_engine_process.stderr,
        )
    compare(custom_engine_events[0]["payload"], custom_expected)

    cli_expected = run_feature_raw_evidence_audit(
        catalog, raw_summary, feature_classes=CUSTOM_CLASSES
    ).as_dict()
    python_out = out / "python-artifacts"
    write_feature_raw_evidence_artifacts(
        run_feature_raw_evidence_audit(catalog, raw_summary, feature_classes=CUSTOM_CLASSES),
        python_out,
    )
    cli_out = out / "cli-artifacts"
    cli_args = [str(catalog), str(raw_summary)]
    for feature_class in CUSTOM_CLASSES:
        cli_args.extend(["--feature-class", feature_class])
    cli_args.extend(["--out-dir", str(cli_out.resolve())])
    cli_process = subprocess.run(
        [str(cli.resolve()), *cli_args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert cli_process.returncode == 0, (cli_process.stdout, cli_process.stderr)
    cli_json = json.loads((cli_out / JSON_ARTIFACT).read_text(encoding="utf-8"))
    compare(cli_json, cli_expected)

    worker_out = out / "worker-artifacts"
    worker_args = [str(catalog), str(raw_summary)]
    for feature_class in CUSTOM_CLASSES:
        worker_args.extend(["--feature-class", feature_class])
    worker_args.extend(["--out-dir", str(worker_out / "artifacts")])
    worker_result = run_worker(worker, worker_args, worker_out)
    compare(worker_result["result"], cli_expected)
    for name in (CSV_ARTIFACT,):
        expected_bytes = (python_out / name).read_bytes()
        assert (cli_out / name).read_bytes() == expected_bytes, f"CLI CSV differs: {name}"
        assert (worker_out / "artifacts" / name).read_bytes() == expected_bytes, (
            f"worker CSV differs: {name}"
        )
    compare(
        json.loads((python_out / JSON_ARTIFACT).read_text(encoding="utf-8")),
        cli_json,
    )
    compare(
        json.loads((worker_out / "artifacts" / JSON_ARTIFACT).read_text(encoding="utf-8")),
        cli_json,
    )

    invalid_raw = input_dir / "absent-id.csv"
    with raw_summary.open("r", encoding="utf-8-sig", newline="") as stream:
        invalid_rows = list(csv.DictReader(stream))
    invalid_rows[-1]["detection_id"] = "999"
    write_csv(invalid_raw, ["detection_id", "origin", *RAW_EVIDENCE_METRICS], invalid_rows)
    invalid_cli = subprocess.run(
        [str(cli.resolve()), str(catalog), str(invalid_raw), "--out-dir", str(out / "invalid-cli")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert invalid_cli.returncode == 2 and "absent from source catalog" in invalid_cli.stderr, (
        invalid_cli.returncode, invalid_cli.stderr
    )
    with raw_summary.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        duplicate_rows = list(reader)
        raw_fields = list(reader.fieldnames or [])
    duplicate_rows.append(dict(duplicate_rows[0]))
    duplicate_raw = input_dir / "duplicate-id.csv"
    write_csv(duplicate_raw, raw_fields, duplicate_rows)
    duplicate_cli = subprocess.run(
        [str(cli.resolve()), str(catalog), str(duplicate_raw), "--out-dir", str(out / "duplicate-cli")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert duplicate_cli.returncode == 2 and "duplicate detection_id=1" in duplicate_cli.stderr, (
        duplicate_cli.returncode, duplicate_cli.stderr
    )

    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (catalog, raw_summary)
    }
    assert before == after, "input CSV changed"
    return {
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "python_source_sha256": hashlib.sha256(
            (ROOT / "src/rst19/feature_raw_evidence_audit.py").read_bytes()
        ).hexdigest(),
        "default_feature_class_order_matches": True,
        "custom_repeated_feature_class_order_matches": True,
        "custom_metric_order_and_deduplication_match": True,
        "engine_cli_worker_json_fields_match": True,
        "csv_artifacts_byte_identical": True,
        "unknown_detection_id_rejected": True,
        "duplicate_detection_id_rejected": True,
        "inputs_sha256_before_after": before,
        "feature_class_count": len(cli_expected["feature_classes"]),
        "metric_row_count": len(cli_expected["rows"]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("worker", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.engine, args.cli, args.worker, args.out), ensure_ascii=False, indent=2))
