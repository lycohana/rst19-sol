from __future__ import annotations

import csv

import pytest

from rst19.aperture_peak_ratio import (
    run_aperture_peak_ratio_audit,
    write_aperture_peak_ratio_artifacts,
)


def _write_catalog(path) -> None:
    rows = [
        {"detection_id": "82931", "feature_class": "range_anomaly", "peak": "4", "flux": "24", "proposal_methods": "gaussian|dog_narrow"},
        {"detection_id": "82932", "feature_class": "range_anomaly", "peak": "10", "flux": "20", "proposal_methods": "gaussian"},
        {"detection_id": "82934", "feature_class": "crowded_blend", "peak": "2", "flux": "50", "proposal_methods": "gaussian"},
        {"detection_id": "82935", "feature_class": "crowded_blend", "peak": "5", "flux": "5", "proposal_methods": "dog_narrow"},
        {"detection_id": "bad", "feature_class": "crowded_blend", "peak": "0", "flux": "99", "proposal_methods": "gaussian"},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_ratio_summary_is_class_conditioned_and_locates_targets(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_aperture_peak_ratio_audit(catalog)

    summaries = {row.feature_class: row for row in result.summaries}
    assert summaries["range_anomaly"].sample_count == 2
    assert summaries["range_anomaly"].ratio_median == pytest.approx(4.0)
    assert summaries["crowded_blend"].ratio_gt20_count == 1

    targets = {row.detection_id: row for row in result.target_rows}
    assert targets["82931"].flux_to_peak == pytest.approx(6.0)
    assert targets["82931"].class_percentile == pytest.approx(1.0)
    assert targets["82934"].class_percentile == pytest.approx(1.0)

    method_summaries = {
        (row.feature_class, row.method_group): row for row in result.method_summaries
    }
    assert method_summaries[("crowded_blend", "gaussian_only")].sample_count == 1
    assert method_summaries[("crowded_blend", "dog_only")].ratio_median == pytest.approx(1.0)


def test_ratio_writes_csv_and_json_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    output = write_aperture_peak_ratio_artifacts(
        run_aperture_peak_ratio_audit(catalog),
        tmp_path / "out",
    )
    assert (output / "aperture_peak_ratio_summary.csv").exists()
    assert (output / "aperture_peak_ratio_method_summary.csv").exists()
    assert (output / "aperture_peak_ratio_targets.csv").exists()
    assert (output / "aperture_peak_ratio.json").exists()
