from __future__ import annotations

import csv

import pytest

from rst19.feature_effect_size import (
    run_feature_effect_size_audit,
    write_feature_effect_size_artifacts,
)


def _write_catalog(path) -> None:
    rows = [
        {"detection_id": "1", "feature_class": "compact_quality", "flux_snr": "10", "fwhm": "2"},
        {"detection_id": "2", "feature_class": "compact_quality", "flux_snr": "20", "fwhm": "2.2"},
        {"detection_id": "3", "feature_class": "compact_quality", "flux_snr": "30", "fwhm": "2.1"},
        {"detection_id": "4", "feature_class": "weak_or_background", "flux_snr": "1", "fwhm": "3"},
        {"detection_id": "5", "feature_class": "weak_or_background", "flux_snr": "2", "fwhm": "3.2"},
        {"detection_id": "6", "feature_class": "spike_or_support", "flux_snr": "5", "fwhm": "4"},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_feature_effect_size_preserves_direction_and_writes_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_feature_effect_size_audit(
        catalog,
        comparison_classes=("weak_or_background",),
        features=("flux_snr", "fwhm"),
        top_n=2,
    )

    rows = {(row.comparison_class, row.feature): row for row in result.rows}
    assert rows[("weak_or_background", "flux_snr")].auc_reference_greater == pytest.approx(1.0)
    assert rows[("weak_or_background", "fwhm")].auc_reference_greater == pytest.approx(0.0)
    assert result.top_features_by_class["weak_or_background"] == ("flux_snr", "fwhm")

    output = write_feature_effect_size_artifacts(result, tmp_path / "out")
    assert (output / "feature_effect_sizes.csv").exists()
    assert (output / "feature_effect_sizes.json").exists()


def test_feature_effect_size_rejects_missing_reference(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    with pytest.raises(ValueError, match="no feature_class"):
        run_feature_effect_size_audit(catalog, reference_class="missing", features=("flux_snr",))
