"""Freeze feature spatial-context results and CSVs from the Python baseline."""
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

from rst19.feature_spatial_context import (  # noqa: E402
    run_feature_spatial_context,
    write_feature_spatial_context_artifacts,
)


FIELDS = [
    "detection_id",
    "x",
    "y",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "flags",
    "flux_snr",
    "filter_snr",
    "peak",
]
ROWS = [
    [1, 10, 10, "compact_quality", "紧凑质量源", "True", "", 8, 12, 80],
    [2, 12, 10, "compact_quality", "紧凑质量源", "False", "UNRESOLVED_BLEND", 20, 30, 200],
    [3, 90, 90, "compact_quality", "紧凑质量源", "False", "LOW_FLUX_SNR", 10, 15, 100],
    [4, 50, 50, "linear_artifact", "线状候选", "False", "LINE_ARTIFACT", 50, 60, 500],
]


def csv_text() -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(FIELDS)
    writer.writerows(ROWS)
    return stream.getvalue()


def main() -> None:
    catalog_csv = csv_text()
    with tempfile.TemporaryDirectory(prefix="rst19-spatial-context-reference-") as temporary:
        catalog_path = Path(temporary) / "source_catalog.csv"
        catalog_path.write_text(catalog_csv, encoding="utf-8-sig")
        result = run_feature_spatial_context(
            catalog_path,
            image_width=100,
            image_height=100,
            grid_size=2,
            edge_margin_px=5.0,
            high_flux_snr_threshold=10.0,
            top_n=1,
            target_ids=[2],
        )
        expected = result.as_dict()
        expected["catalog_path"] = "<CATALOG>"
        output_dir = Path(temporary) / "artifacts"
        write_feature_spatial_context_artifacts(result, output_dir)
        artifacts = {
            name: (output_dir / name).read_bytes().decode("utf-8")
            for name in (
                "feature_spatial_context_summary.csv",
                "feature_spatial_context_hotspots.csv",
                "feature_spatial_context_targets.csv",
            )
        }
    payload = {
        "source_hashes": {
            "feature_spatial_context.py": hashlib.sha256(
                (ROOT / "src/rst19/feature_spatial_context.py").read_bytes()
            ).hexdigest(),
            "feature_spatial_context_cli.py": hashlib.sha256(
                (ROOT / "src/rst19/feature_spatial_context_cli.py").read_bytes()
            ).hexdigest(),
        },
        "catalog_csv": catalog_csv,
        "artifacts": artifacts,
        "expected": expected,
    }
    output = ROOT / "backend/tests/fixtures/feature_spatial_context.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
