from __future__ import annotations

import csv

import pytest

from rst19.feature_temporal_consistency_audit import (
    TEMPORAL_REQUIRED_FIELDS,
    run_feature_temporal_consistency_audit,
    write_feature_temporal_consistency_artifacts,
)


def _write_csv(path, fields, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path):
    catalog = tmp_path / "source_catalog.csv"
    _write_csv(
        catalog,
        ["detection_id", "feature_class", "feature_class_label"],
        [
            {"detection_id": "1", "feature_class": "compact_quality", "feature_class_label": "紧凑"},
            {"detection_id": "2", "feature_class": "linear_artifact", "feature_class_label": "线状"},
        ],
    )
    metrics = tmp_path / "frame_metrics.csv"
    fields = ["detection_id", "frame_index", *TEMPORAL_REQUIRED_FIELDS]
    rows = []
    for frame_index, flux in enumerate((10.0, 11.0, 9.0), start=1):
        rows.append(
            {
                "detection_id": "1",
                "frame_index": str(frame_index),
                "flux_snr_raw": str(flux),
                "negative_extreme_count": "0",
                "repeated_code_count": "0",
                "zero_count": "0",
            }
        )
    for frame_index, flux in enumerate((5.0, -1.0, 5.0), start=1):
        rows.append(
            {
                "detection_id": "2",
                "frame_index": str(frame_index),
                "flux_snr_raw": str(flux),
                "negative_extreme_count": "1" if flux < 0 else "0",
                "repeated_code_count": "1" if frame_index == 3 else "0",
                "zero_count": "1" if frame_index == 2 else "0",
            }
        )
    _write_csv(metrics, fields, rows)
    return catalog, metrics


def test_temporal_consistency_summarizes_source_and_class(tmp_path) -> None:
    catalog, metrics = _write_inputs(tmp_path)
    result = run_feature_temporal_consistency_audit(
        catalog,
        metrics,
        expected_frame_count=3,
    )

    assert result.feature_class_source == "catalog"
    assert result.selected_detection_ids == (1, 2)
    assert result.feature_classes == ("compact_quality", "linear_artifact")
    compact = next(row for row in result.source_rows if row.detection_id == 1)
    assert compact.frame_count == 3
    assert compact.coverage_fraction == pytest.approx(1.0)
    assert compact.flux_snr_median == pytest.approx(10.0)
    assert compact.flux_snr_relative_mad == pytest.approx(0.14826)
    assert compact.flux_snr_positive_fraction == 1.0
    assert compact.flux_snr_sign_flip_count == 0

    linear = next(row for row in result.source_rows if row.detection_id == 2)
    assert linear.flux_snr_positive_fraction == pytest.approx(2 / 3)
    assert linear.flux_snr_ge5_fraction == pytest.approx(2 / 3)
    assert linear.flux_snr_sign_flip_count == 2
    assert linear.zero_frame_fraction == pytest.approx(1 / 3)
    assert linear.negative_extreme_frame_fraction == pytest.approx(1 / 3)
    assert linear.repeated_code_frame_fraction == pytest.approx(1 / 3)

    summary = next(row for row in result.class_summaries if row.feature_class == "linear_artifact")
    assert summary.source_count == 1
    assert summary.source_any_sign_flip_fraction == 1.0
    assert summary.source_all_positive_fraction == 0.0
    assert summary.source_any_repeated_code_fraction == 1.0

    output = write_feature_temporal_consistency_artifacts(result, tmp_path / "out")
    assert (output / "feature_temporal_consistency_audit.json").is_file()
    assert (output / "feature_temporal_consistency_sources.csv").is_file()
    assert (output / "feature_temporal_consistency_summary.csv").is_file()


def test_temporal_consistency_rejects_duplicate_frame_or_missing_source(tmp_path) -> None:
    catalog, metrics = _write_inputs(tmp_path)
    fields = ["detection_id", "frame_index", *TEMPORAL_REQUIRED_FIELDS]
    rows = []
    for _ in range(2):
        rows.append(
            {
                "detection_id": "999",
                "frame_index": "1",
                "flux_snr_raw": "1",
                "negative_extreme_count": "0",
                "repeated_code_count": "0",
                "zero_count": "0",
            }
        )
    _write_csv(metrics, fields, rows)
    with pytest.raises(ValueError, match="absent from source catalog"):
        run_feature_temporal_consistency_audit(catalog, metrics)

    catalog, metrics = _write_inputs(tmp_path)
    with metrics.open("a", encoding="utf-8", newline="") as stream:
        stream.write("1,1,10,0,0,0\n")
    with pytest.raises(ValueError, match="duplicate detection_id=1, frame_index=1"):
        run_feature_temporal_consistency_audit(catalog, metrics)


def test_temporal_consistency_requires_explicit_local_class_source(tmp_path) -> None:
    catalog, metrics = _write_inputs(tmp_path)
    with metrics.open("r", encoding="utf-8-sig", newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    local_rows = []
    for row in original_rows:
        local_rows.append(
            {
                "detection_id": row["detection_id"],
                "frame_index": row["frame_index"],
                "feature_class": "range_anomaly" if row["detection_id"] == "1" else "crowded_blend",
                **{field: row[field] for field in TEMPORAL_REQUIRED_FIELDS},
            }
        )
    _write_csv(metrics, ["detection_id", "frame_index", "feature_class", *TEMPORAL_REQUIRED_FIELDS], local_rows)

    with pytest.raises(ValueError, match="feature_class mismatch"):
        run_feature_temporal_consistency_audit(catalog, metrics)

    result = run_feature_temporal_consistency_audit(
        catalog,
        metrics,
        feature_class_source="frame",
    )
    assert result.feature_class_source == "frame"
    assert result.feature_classes == ("crowded_blend", "range_anomaly")
    assert {row.feature_class for row in result.source_rows} == {"crowded_blend", "range_anomaly"}

    selected = run_feature_temporal_consistency_audit(
        catalog,
        metrics,
        detection_ids=(2,),
        feature_class_source="frame",
    )
    assert selected.selected_detection_ids == (2,)
    assert selected.feature_classes == ("crowded_blend",)


def test_temporal_consistency_accepts_pair_local_repeated_code_alias(tmp_path) -> None:
    catalog, metrics = _write_inputs(tmp_path)
    with metrics.open("r", encoding="utf-8-sig", newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    alias_rows = []
    for row in original_rows:
        alias_rows.append(
            {
                "detection_id": row["detection_id"],
                "frame_index": row["frame_index"],
                "flux_snr_raw": row["flux_snr_raw"],
                "negative_extreme_count": row["negative_extreme_count"],
                "repeated_3990_3993_count": row["repeated_code_count"],
                "zero_count": row["zero_count"],
            }
        )
    _write_csv(
        metrics,
        [
            "detection_id",
            "frame_index",
            "flux_snr_raw",
            "negative_extreme_count",
            "repeated_3990_3993_count",
            "zero_count",
        ],
        alias_rows,
    )

    result = run_feature_temporal_consistency_audit(catalog, metrics)

    linear = next(row for row in result.source_rows if row.detection_id == 2)
    assert linear.repeated_code_frame_fraction == pytest.approx(1 / 3)
