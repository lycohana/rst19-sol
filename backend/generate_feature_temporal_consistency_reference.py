"""Freeze temporal consistency summaries from the Python baseline."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.feature_temporal_consistency_audit import (
    run_feature_temporal_consistency_audit,
    write_feature_temporal_consistency_artifacts,
)


CATALOG_FIELDS = ["detection_id", "feature_class", "feature_class_label"]
METRIC_FIELDS = [
    "detection_id",
    "frame_index",
    "flux_snr_raw",
    "negative_extreme_count",
    "repeated_code_count",
    "zero_count",
]
CATALOG_ROWS = [
    ["1", "compact_quality", "紧凑"],
    ["2", "linear_artifact", "线状"],
]
METRIC_ROWS = []
for frame_index, flux in enumerate((10.0, 11.0, 9.0), start=1):
    METRIC_ROWS.append(["1", str(frame_index), str(flux), "0", "0", "0"])
for frame_index, flux in enumerate((5.0, -1.0, 5.0), start=1):
    METRIC_ROWS.append([
        "2",
        str(frame_index),
        str(flux),
        "1" if flux < 0 else "0",
        "1" if frame_index == 3 else "0",
        "1" if frame_index == 2 else "0",
    ])


def csv_text(fields: list[str], rows: list[list[str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(fields)
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    catalog_csv = csv_text(CATALOG_FIELDS, CATALOG_ROWS)
    metrics_csv = csv_text(METRIC_FIELDS, METRIC_ROWS)
    with tempfile.TemporaryDirectory(prefix="rst19-temporal-reference-") as temporary:
        catalog_path = Path(temporary) / "source_catalog.csv"
        metrics_path = Path(temporary) / "frame_metrics.csv"
        catalog_path.write_text(catalog_csv, encoding="utf-8-sig")
        metrics_path.write_text(metrics_csv, encoding="utf-8-sig")
        audit = run_feature_temporal_consistency_audit(
            catalog_path,
            metrics_path,
            expected_frame_count=3,
        )
        result = audit.as_dict()
        artifact_dir = Path(temporary) / "artifacts"
        write_feature_temporal_consistency_artifacts(audit, artifact_dir)
        artifacts = {
            name: (artifact_dir / name).read_bytes().decode("utf-8")
            for name in (
                "feature_temporal_consistency_sources.csv",
                "feature_temporal_consistency_summary.csv",
            )
        }
    result["catalog_path"] = "<CATALOG>"
    result["frame_metrics_path"] = "<FRAME_METRICS>"
    payload = {
        "source_hashes": {
            "feature_temporal_consistency_audit.py": hashlib.sha256(
                (ROOT / "src/rst19/feature_temporal_consistency_audit.py").read_bytes()
            ).hexdigest(),
            "feature_temporal_consistency_audit_cli.py": hashlib.sha256(
                (ROOT / "src/rst19/feature_temporal_consistency_audit_cli.py").read_bytes()
            ).hexdigest(),
        },
        "catalog_csv": catalog_csv,
        "frame_metrics_csv": metrics_csv,
        "artifacts": artifacts,
        "expected": result,
    }
    output = ROOT / "backend/tests/fixtures/feature_temporal_consistency.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
