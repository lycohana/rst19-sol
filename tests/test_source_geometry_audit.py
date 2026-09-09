from __future__ import annotations

import csv

import pytest

from rst19.source_geometry_audit import (
    run_source_geometry_audit,
    write_source_geometry_artifacts,
)


def _write_catalog(path) -> None:
    fields = ["x", "y", "feature_class", "feature_class_label"]
    rows = []
    for index in range(12):
        rows.append(
            {
                "x": str(10 + (index % 4) * 25),
                "y": str(10 + (index // 4) * 35),
                "feature_class": "compact_quality",
                "feature_class_label": "紧凑质量源",
            }
        )
    for index in range(4):
        rows.append(
            {
                "x": str(20 + index * 10),
                "y": str(100 + index * 100),
                "feature_class": "linear_artifact",
                "feature_class_label": "线状候选",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_source_geometry_reports_line_shape_and_writes_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_source_geometry_audit(
        catalog,
        target_classes=("linear_artifact",),
        control_trial_count=64,
        random_seed=19,
    )

    summary = result.summaries[0]
    assert summary.feature_class == "linear_artifact"
    assert summary.candidate_count == 4
    assert summary.pca_axis_ratio > 10
    assert summary.perpendicular_residual_median_px == pytest.approx(0.0)
    assert summary.control_trial_count == 64
    assert summary.control_axis_ratio_p99 is not None
    assert summary.pca_axis_ratio > summary.control_axis_ratio_max
    assert summary.control_axis_ratio_upper_tail_fraction == 0.0

    output = write_source_geometry_artifacts(result, tmp_path / "out")
    assert (output / "source_geometry_audit.json").is_file()
    assert (output / "source_geometry_summary.csv").is_file()


def test_source_geometry_defaults_to_all_nonempty_classes(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_source_geometry_audit(catalog, control_trial_count=8)
    assert result.target_classes == ("compact_quality", "linear_artifact")
    assert result.summaries[0].control_trial_count == 0


def test_source_geometry_rejects_missing_control_or_invalid_trials(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    with pytest.raises(ValueError, match="control_trial_count"):
        run_source_geometry_audit(catalog, control_trial_count=0)
    with pytest.raises(ValueError, match="control class"):
        run_source_geometry_audit(catalog, control_class="missing", control_trial_count=8)
