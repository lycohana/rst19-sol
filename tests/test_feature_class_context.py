from __future__ import annotations

import csv

import pytest

from rst19.feature_class_context import run_feature_class_context, write_feature_class_context_artifacts


def _write_catalog(path) -> None:
    fields = [
        "detection_id",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flags",
        "flux_snr",
        "filter_snr",
        "fwhm",
        "ellipticity",
        "sharpness",
        "psf_support_pixels",
        "footprint_pixels",
        "centroid_shift_px",
        "peak",
    ]
    rows = []
    for detection_id, flux_snr, quality in ((1, 5.0, "True"), (2, 10.0, "False"), (3, 15.0, "True")):
        rows.append(
            {
                "detection_id": str(detection_id),
                "feature_class": "compact_quality",
                "feature_class_label": "紧凑质量源",
                "quality_passed": quality,
                "flags": "" if quality == "True" else "LOW_FLUX_SNR",
                **{field: str(value) for field, value in {
                    "flux_snr": flux_snr,
                    "filter_snr": flux_snr * 2,
                    "fwhm": 2.0,
                    "ellipticity": 0.1,
                    "sharpness": 0.3,
                    "psf_support_pixels": 5,
                    "footprint_pixels": 20,
                    "centroid_shift_px": 0.1,
                    "peak": flux_snr * 10,
                }.items()},
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_feature_class_context_reports_within_class_percentiles_and_writes_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_feature_class_context(catalog, [2])

    assert result.targets[0].feature_class == "compact_quality"
    assert result.targets[0].class_count == 3
    assert result.targets[0].class_quality_count == 2
    flux = next(metric for metric in result.metrics if metric.metric == "flux_snr")
    assert flux.value == 10.0
    assert flux.class_median == 10.0
    assert flux.empirical_percentile == 2 / 3

    output = write_feature_class_context_artifacts(result, tmp_path / "out")
    assert (output / "feature_class_context.json").is_file()
    assert (output / "feature_class_context_targets.csv").is_file()
    assert (output / "feature_class_context_metrics.csv").is_file()


def test_feature_class_context_rejects_unknown_target(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    with pytest.raises(ValueError, match="找不到 detection_id"):
        run_feature_class_context(catalog, [99])
