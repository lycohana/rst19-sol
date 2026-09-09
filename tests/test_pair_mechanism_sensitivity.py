from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rst19.pair_mechanism_sensitivity import (
    run_pair_mechanism_sensitivity,
    write_pair_mechanism_sensitivity_artifacts,
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
                    "quality_passed": "False",
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


def test_sensitivity_runs_declared_windows_and_writes_artifacts(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)

    result = run_pair_mechanism_sensitivity(
        catalog,
        target_ids=(1, 2),
        configurations=(
            ("narrow", 3.0, 3.5, 4.8),
            ("target", 3.5, 3.5, 4.8),
        ),
    )

    assert [row.configuration for row in result.rows] == ["narrow", "target"]
    assert all(row.pair_count == 1 for row in result.rows)
    assert all(row.cross_feature_class_count == 1 for row in result.rows)
    assert all(row.both_quality_count == 0 for row in result.rows)
    assert all(row.target_in_geometry_context for row in result.rows)

    output = write_pair_mechanism_sensitivity_artifacts(result, tmp_path / "out")
    assert (output / "pair_mechanism_sensitivity.csv").is_file()
    payload = json.loads((output / "pair_mechanism_sensitivity.json").read_text(encoding="utf-8"))
    assert payload["rows"][1]["configuration"] == "target"
    assert "不是物理双星真值" in payload["interpretation_guardrails"][0]


def test_sensitivity_rejects_invalid_configurations(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)

    with pytest.raises(ValueError, match="at least one"):
        run_pair_mechanism_sensitivity(catalog, (1, 2), configurations=())
    with pytest.raises(ValueError, match="duplicate configuration"):
        run_pair_mechanism_sensitivity(
            catalog,
            (1, 2),
            configurations=(
                ("same", 3.0, 3.5, 4.8),
                ("same", 3.5, 3.5, 4.8),
            ),
        )
    with pytest.raises(ValueError, match="non-numeric distance"):
        run_pair_mechanism_sensitivity(
            catalog,
            (1, 2),
            configurations=(("bad", 3.0, "x", 4.8),),
        )
