from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rst19.pair_mechanism_context import (
    run_pair_mechanism_context,
    write_pair_mechanism_context_artifacts,
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
        "flags",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            (
                {
                    "detection_id": 1,
                    "x": 0,
                    "y": 0,
                    "peak_x": 0,
                    "peak_y": 0,
                    "quality_passed": "True",
                    "feature_class": "range_anomaly",
                    "flags": "CODE_PATTERN",
                },
                {
                    "detection_id": 2,
                    "x": 2.738,
                    "y": 0,
                    "peak_x": 4,
                    "peak_y": 0,
                    "quality_passed": "False",
                    "feature_class": "crowded_blend",
                    "flags": "UNRESOLVED_BLEND",
                },
                {
                    "detection_id": 3,
                    "x": 20,
                    "y": 0,
                    "peak_x": 20,
                    "peak_y": 0,
                    "quality_passed": "True",
                    "feature_class": "compact_quality",
                    "flags": "",
                },
            )
        )


def test_context_reports_target_and_writes_artifacts(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)

    result = run_pair_mechanism_context(catalog, target_ids=(1, 2))

    assert result.source_count == 3
    assert result.quality_count == 2
    assert len(result.pairs) == 1
    assert result.pairs[0].is_target_pair is True
    assert result.pairs[0].same_feature_class is False
    assert result.pairs[0].both_quality_passed is False
    assert result.target.in_geometry_context is True
    assert result.target.centroid_distance_px == pytest.approx(2.738)
    assert result.target.peak_distance_px == pytest.approx(4.0)

    output = write_pair_mechanism_context_artifacts(result, tmp_path / "out")
    assert (output / "pair_mechanism_context_pairs.csv").is_file()
    assert (output / "pair_mechanism_context_summary.csv").is_file()
    assert (output / "pair_mechanism_context_target.csv").is_file()
    payload = json.loads((output / "pair_mechanism_context.json").read_text(encoding="utf-8"))
    assert payload["target"]["in_geometry_context"] is True
    assert "不是物理双星真值" in payload["interpretation_guardrails"][0]


def test_context_rejects_invalid_target_and_geometry(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)

    with pytest.raises(ValueError, match="exactly two"):
        run_pair_mechanism_context(catalog, target_ids=(1,))
    with pytest.raises(ValueError, match="not in source catalog"):
        run_pair_mechanism_context(catalog, target_ids=(1, 99))
    with pytest.raises(ValueError, match="must not exceed"):
        run_pair_mechanism_context(
            catalog,
            target_ids=(1, 2),
            peak_distance_min_px=5,
            peak_distance_max_px=4,
        )


def test_context_requires_catalog_fields(tmp_path: Path) -> None:
    catalog = tmp_path / "incomplete.csv"
    catalog.write_text("detection_id,x,y\n1,0,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        run_pair_mechanism_context(catalog, target_ids=(1, 2))
