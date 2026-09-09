from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rst19.source_separation_audit import (
    run_source_separation_audit,
    write_source_separation_artifacts,
)


def _write_catalog(path: Path) -> None:
    fieldnames = (
        "detection_id",
        "x",
        "y",
        "peak_x",
        "peak_y",
        "quality_passed",
        "feature_class",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            (
                {"detection_id": 1, "x": 0, "y": 0, "peak_x": 0, "peak_y": 0, "quality_passed": "True", "feature_class": "compact_quality"},
                {"detection_id": 2, "x": 2.5, "y": 0, "peak_x": 3, "peak_y": 0, "quality_passed": "True", "feature_class": "compact_quality"},
                {"detection_id": 3, "x": 10, "y": 0, "peak_x": 10, "peak_y": 0, "quality_passed": "False", "feature_class": "weak_or_background"},
                {"detection_id": 4, "x": 10, "y": 5, "peak_x": 10, "peak_y": 5, "quality_passed": "False", "feature_class": "shape_outlier"},
            )
        )


def test_source_separation_reports_centroid_peak_and_target_position(tmp_path) -> None:
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)

    result = run_source_separation_audit(catalog, target_ids=(1, 2), radii_px=(3, 4, 6))

    assert result.source_count == 4
    assert result.quality_count == 2
    assert [row.radius_px for row in result.thresholds] == [3.0, 4.0, 6.0]
    assert result.thresholds[0].centroid_endpoint_count == 2
    assert result.thresholds[0].centroid_quality_endpoint_count == 2
    assert result.targets[0].centroid_nn_detection_id == 2
    assert result.targets[1].centroid_nn_detection_id == 1
    assert result.targets[0].centroid_nn_distance_px == pytest.approx(2.5)
    assert result.targets[0].peak_nn_distance_px == pytest.approx(3.0)
    assert result.targets[0].centroid_nn_percentile == pytest.approx(0.5)
    assert result.classes[0].feature_class == "compact_quality"

    output = write_source_separation_artifacts(result, tmp_path / "out")
    assert (output / "source_separation_thresholds.csv").is_file()
    assert (output / "source_separation_classes.csv").is_file()
    assert (output / "source_separation_targets.csv").is_file()
    payload = json.loads(
        (output / "source_separation_summary.json").read_text(encoding="utf-8")
    )
    assert payload["source_count"] == 4
    assert len(payload["targets"]) == 2
    assert "independent stars" in payload["conclusion"]


def test_source_separation_rejects_missing_target_and_duplicate_id(tmp_path) -> None:
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)
    with pytest.raises(ValueError, match="not in source catalog"):
        run_source_separation_audit(catalog, target_ids=(99,))

    duplicate = tmp_path / "duplicate.csv"
    _write_catalog(duplicate)
    text = duplicate.read_text(encoding="utf-8")
    duplicate.write_text(text.replace("4,10,5", "1,10,5"), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate detection_id"):
        run_source_separation_audit(duplicate)


def test_source_separation_requires_catalog_fields(tmp_path) -> None:
    catalog = tmp_path / "incomplete.csv"
    catalog.write_text("detection_id,x,y\n1,0,0\n2,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        run_source_separation_audit(catalog)
