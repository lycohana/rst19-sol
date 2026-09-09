from __future__ import annotations

import csv

from rst19.feature_gate_margin import run_feature_gate_margin, write_feature_gate_margin_artifacts


def _write_catalog(path) -> None:
    rows = [
        {
            "detection_id": "1",
            "feature_class": "compact_quality",
            "feature_class_label": "紧凑",
            "quality_passed": "True",
            "flux_snr": "12",
            "psf_support_pixels": "5",
            "fwhm": "2.5",
            "ellipticity": "0.1",
            "sharpness": "0.3",
            "footprint_pixels": "20",
        },
        {
            "detection_id": "2",
            "feature_class": "weak_or_background",
            "feature_class_label": "弱",
            "quality_passed": "False",
            "flux_snr": "3",
            "psf_support_pixels": "4",
            "fwhm": "2.5",
            "ellipticity": "0.1",
            "sharpness": "0.3",
            "footprint_pixels": "20",
        },
        {
            "detection_id": "3",
            "feature_class": "shape_outlier",
            "feature_class_label": "形状",
            "quality_passed": "False",
            "flux_snr": "25",
            "psf_support_pixels": "6",
            "fwhm": "3.5",
            "ellipticity": "0.7",
            "sharpness": "0.3",
            "footprint_pixels": "20",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_gate_margin_preserves_signed_boundary_direction(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_feature_gate_margin(catalog)
    rows = {(row.feature_class, row.metric): row for row in result.rows}

    assert rows[("compact_quality", "flux_snr")].median_margin == 7
    assert rows[("weak_or_background", "flux_snr")].negative_margin_count == 1
    assert rows[("shape_outlier", "ellipticity")].median_margin < 0
    assert rows[("shape_outlier", "flux_snr")].negative_margin_count == 0


def test_gate_margin_writes_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    output = write_feature_gate_margin_artifacts(run_feature_gate_margin(catalog), tmp_path / "out")
    assert (output / "feature_gate_margins.csv").exists()
    assert (output / "feature_gate_margins.json").exists()
