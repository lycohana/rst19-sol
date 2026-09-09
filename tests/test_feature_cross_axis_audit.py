from __future__ import annotations

import csv

import pytest

from rst19.feature_cross_axis_audit import (
    run_feature_cross_axis_audit,
    write_feature_cross_axis_artifacts,
)


def _write_csv(path, fields, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path):
    catalog = tmp_path / "catalog.csv"
    _write_csv(
        catalog,
        ["detection_id", "feature_class", "feature_class_label", "quality_passed"],
        [
            {
                "detection_id": "1",
                "feature_class": "compact_quality",
                "feature_class_label": "紧凑",
                "quality_passed": "True",
            },
            {
                "detection_id": "2",
                "feature_class": "crowded_blend",
                "feature_class_label": "拥挤",
                "quality_passed": "False",
            },
        ],
    )
    raw = tmp_path / "raw.csv"
    raw_fields = [
        "detection_id",
        "core_fraction_median",
        "noise_median",
        "frames_support_ge3",
        "frames_flux_snr_ge5",
        "frames_negative_extreme_count_gt0",
        "frames_repeated_code_count_gt0",
    ]
    _write_csv(
        raw,
        raw_fields,
        [
            {
                "detection_id": "1",
                "core_fraction_median": "0.7",
                "noise_median": "8",
                "frames_support_ge3": "15",
                "frames_flux_snr_ge5": "15",
                "frames_negative_extreme_count_gt0": "0",
                "frames_repeated_code_count_gt0": "0",
            },
            {
                "detection_id": "2",
                "core_fraction_median": "0.2",
                "noise_median": "14",
                "frames_support_ge3": "0",
                "frames_flux_snr_ge5": "2",
                "frames_negative_extreme_count_gt0": "1",
                "frames_repeated_code_count_gt0": "1",
            },
        ],
    )
    peak = tmp_path / "peak.csv"
    _write_csv(
        peak,
        [
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "raw_valid",
            "raw_local_maximum_r1",
            "raw_local_max_gap_adu_r1",
        ],
        [
            {
                "detection_id": "1",
                "feature_class": "compact_quality",
                "feature_class_label": "紧凑",
                "quality_passed": "True",
                "raw_valid": "True",
                "raw_local_maximum_r1": "True",
                "raw_local_max_gap_adu_r1": "0",
            },
            {
                "detection_id": "2",
                "feature_class": "crowded_blend",
                "feature_class_label": "拥挤",
                "quality_passed": "False",
                "raw_valid": "True",
                "raw_local_maximum_r1": "False",
                "raw_local_max_gap_adu_r1": "100",
            },
        ],
    )
    temporal = tmp_path / "temporal.csv"
    _write_csv(
        temporal,
        [
            "detection_id",
            "feature_class",
            "expected_frame_count",
            "frame_count",
            "coverage_fraction",
            "flux_snr_relative_mad",
            "flux_snr_positive_fraction",
            "flux_snr_ge5_fraction",
            "flux_snr_sign_flip_count",
        ],
        [
            {
                "detection_id": "1",
                "feature_class": "compact_quality",
                "expected_frame_count": "15",
                "frame_count": "15",
                "coverage_fraction": "1",
                "flux_snr_relative_mad": "0.1",
                "flux_snr_positive_fraction": "1",
                "flux_snr_ge5_fraction": "1",
                "flux_snr_sign_flip_count": "0",
            },
            {
                "detection_id": "2",
                "feature_class": "crowded_blend",
                "expected_frame_count": "15",
                "frame_count": "10",
                "coverage_fraction": "0.6667",
                "flux_snr_relative_mad": "0.3",
                "flux_snr_positive_fraction": "0.8",
                "flux_snr_ge5_fraction": "0.1",
                "flux_snr_sign_flip_count": "2",
            },
        ],
    )
    return catalog, raw, peak, temporal


def test_cross_axis_audit_aligns_sources_and_reports_conflict_pattern(tmp_path) -> None:
    inputs = _write_inputs(tmp_path)
    result = run_feature_cross_axis_audit(*inputs)

    assert result.selected_detection_ids == (1, 2)
    compact = next(row for row in result.source_rows if row.detection_id == 1)
    crowded = next(row for row in result.source_rows if row.detection_id == 2)
    assert compact.evidence_pattern == "raw_local_peak+value_domain_clean+support_seen"
    assert crowded.evidence_pattern == "raw_non_local_peak+value_domain_anomaly+no_support_seen"
    assert crowded.temporal_coverage_fraction == pytest.approx(2 / 3, rel=1e-3)
    summary = next(row for row in result.class_summaries if row.feature_class == "crowded_blend")
    assert summary.raw_non_local_maximum_fraction == 1.0
    assert summary.value_domain_anomaly_fraction == 1.0
    assert summary.temporal_any_sign_flip_fraction == 1.0

    output = write_feature_cross_axis_artifacts(result, tmp_path / "out")
    assert (output / "feature_cross_axis_audit.json").is_file()
    assert (output / "feature_cross_axis_sources.csv").is_file()
    assert (output / "feature_cross_axis_summary.csv").is_file()
    assert (output / "feature_cross_axis_patterns.csv").is_file()


def test_cross_axis_audit_rejects_temporal_id_set_mismatch(tmp_path) -> None:
    catalog, raw, peak, temporal = _write_inputs(tmp_path)
    with temporal.open("a", encoding="utf-8", newline="") as stream:
        stream.write("3,compact_quality,15,15,1,0.1,1,1,0\n")

    with pytest.raises(ValueError, match="detection_id set does not match"):
        run_feature_cross_axis_audit(catalog, raw, peak, temporal)


def test_cross_axis_audit_can_select_one_source(tmp_path) -> None:
    inputs = _write_inputs(tmp_path)
    result = run_feature_cross_axis_audit(inputs[0], inputs[1], inputs[2], inputs[3], detection_ids=(2,))

    assert result.selected_detection_ids == (2,)
    assert result.feature_classes == ("crowded_blend",)


def test_cross_axis_audit_rejects_peak_catalog_class_mismatch(tmp_path) -> None:
    catalog, raw, peak, temporal = _write_inputs(tmp_path)
    rows = []
    with peak.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["feature_class"] = "compact_quality"
    _write_csv(peak, list(rows[0]), rows)

    with pytest.raises(ValueError, match="feature_class mismatch"):
        run_feature_cross_axis_audit(catalog, raw, peak, temporal)


def test_cross_axis_audit_rejects_peak_catalog_label_mismatch(tmp_path) -> None:
    catalog, raw, peak, temporal = _write_inputs(tmp_path)
    rows = []
    with peak.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["feature_class_label"] = "错误标签"
    _write_csv(peak, list(rows[0]), rows)

    with pytest.raises(ValueError, match="feature_class_label mismatch"):
        run_feature_cross_axis_audit(catalog, raw, peak, temporal)
