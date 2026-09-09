from __future__ import annotations

import csv

import pytest

from rst19.feature_hard_negative import (
    run_feature_hard_negative_audit,
    write_feature_hard_negative_artifacts,
)


def _write_catalog(path) -> None:
    fields = [
        "detection_id",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flags",
        "flux_snr",
        "filter_snr",
        "peak",
        "fwhm",
        "ellipticity",
        "sharpness",
        "psf_support_pixels",
        "footprint_pixels",
    ]
    rows = [
        (1, "compact_quality", "True", "", 2.0),
        (2, "compact_quality", "False", "LOW_FLUX_SNR", 12.0),
        (3, "compact_quality", "False", "UNRESOLVED_BLEND", 20.0),
        (4, "range_anomaly", "False", "NEGATIVE_OVERFLOW", 50.0),
        (5, "other_rejected", "False", "OTHER", 100.0),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for detection_id, feature_class, quality, flags, flux_snr in rows:
            writer.writerow(
                {
                    "detection_id": detection_id,
                    "feature_class": feature_class,
                    "feature_class_label": feature_class,
                    "quality_passed": quality,
                    "flags": flags,
                    "flux_snr": flux_snr,
                    "filter_snr": flux_snr * 2,
                    "peak": flux_snr * 10,
                    "fwhm": 2.0,
                    "ellipticity": 0.1,
                    "sharpness": 0.3,
                    "psf_support_pixels": 5,
                    "footprint_pixels": 20,
                }
            )


def test_feature_hard_negative_ranks_rejected_sources_within_each_class(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_feature_hard_negative_audit(catalog, top_n=1)

    assert [summary.feature_class for summary in result.summaries] == [
        "compact_quality",
        "range_anomaly",
    ]
    assert [(row.feature_class, row.detection_id, row.metric_value) for row in result.rows] == [
        ("compact_quality", 3, 20.0),
        ("range_anomaly", 4, 50.0),
    ]
    compact = result.summaries[0]
    assert compact.class_count == 3
    assert compact.class_quality_count == 1
    assert compact.class_rejected_count == 2


def test_feature_hard_negative_can_include_other_rejected_and_writes_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_feature_hard_negative_audit(catalog, top_n=2, exclude_classes=())
    output = write_feature_hard_negative_artifacts(result, tmp_path / "out")

    assert "other_rejected" in {summary.feature_class for summary in result.summaries}
    assert (output / "feature_hard_negative.json").is_file()
    assert (output / "feature_hard_negative_summary.csv").is_file()
    assert (output / "feature_hard_negative_rows.csv").is_file()


def test_feature_hard_negative_rejects_invalid_arguments(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    with pytest.raises(ValueError, match="top_n"):
        run_feature_hard_negative_audit(catalog, top_n=0)
    with pytest.raises(ValueError, match="不支持的 hard-negative"):
        run_feature_hard_negative_audit(catalog, metric="snr")
