from __future__ import annotations

import csv

from rst19.feature_gate_route import run_feature_gate_route, write_feature_gate_route_artifacts


def _write_catalog(path) -> None:
    rows = [
        {
            "detection_id": "1",
            "feature_class": "compact_quality",
            "feature_class_label": "紧凑",
            "quality_passed": "True",
            "flags": "",
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
            "flags": "LOW_FLUX_SNR",
            "flux_snr": "3",
            "psf_support_pixels": "5",
            "fwhm": "2.5",
            "ellipticity": "0.1",
            "sharpness": "0.3",
            "footprint_pixels": "20",
        },
        {
            "detection_id": "3",
            "feature_class": "linear_artifact",
            "feature_class_label": "线状",
            "quality_passed": "False",
            "flags": "LINE_ARTIFACT",
            "flux_snr": "20",
            "psf_support_pixels": "5",
            "fwhm": "2.5",
            "ellipticity": "0.1",
            "sharpness": "0.3",
            "footprint_pixels": "20",
        },
        {
            "detection_id": "4",
            "feature_class": "crowded_blend",
            "feature_class_label": "拥挤",
            "quality_passed": "False",
            "flags": "LOW_FLUX_SNR|UNRESOLVED_BLEND",
            "flux_snr": "3",
            "psf_support_pixels": "2",
            "fwhm": "2.5",
            "ellipticity": "0.1",
            "sharpness": "0.3",
            "footprint_pixels": "20",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_gate_route_separates_numeric_and_structural_paths(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_feature_gate_route(catalog)
    sources = {row.detection_id: row for row in result.source_rows}

    assert sources["1"].route == "quality_passed"
    assert sources["2"].route == "rejected_numeric_only"
    assert sources["2"].numeric_failure_reasons == "flux_snr"
    assert sources["2"].structural_flags == ""
    assert sources["3"].route == "rejected_structural_only"
    assert sources["4"].route == "rejected_numeric_and_structural"
    assert sources["4"].numeric_failure_reasons == "flux_snr|psf_support_pixels"
    assert sources["4"].structural_flags == "UNRESOLVED_BLEND"


def test_gate_route_writes_source_and_summary_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    output = write_feature_gate_route_artifacts(run_feature_gate_route(catalog), tmp_path / "out")
    assert (output / "feature_gate_route_sources.csv").exists()
    assert (output / "feature_gate_route_summary.csv").exists()
    assert (output / "feature_gate_routes.json").exists()
