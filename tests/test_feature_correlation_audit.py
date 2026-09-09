from __future__ import annotations

import csv
import json

import pytest

from rst19.feature_correlation_audit import (
    run_feature_correlation_audit,
    write_feature_correlation_artifacts,
)


def _write_catalog(path) -> None:
    rows = [
        {"feature_class": "compact_quality", "feature_class_label": "紧凑", "a": "1", "b": "3", "c": "10"},
        {"feature_class": "compact_quality", "feature_class_label": "紧凑", "a": "2", "b": "2", "c": "20"},
        {"feature_class": "compact_quality", "feature_class_label": "紧凑", "a": "3", "b": "1", "c": "30"},
        {"feature_class": "weak_or_background", "feature_class_label": "弱", "a": "4", "b": "4", "c": "1"},
        {"feature_class": "weak_or_background", "feature_class_label": "弱", "a": "5", "b": "5", "c": "2"},
        {"feature_class": "weak_or_background", "feature_class_label": "弱", "a": "6", "b": "6", "c": "3"},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_feature_correlation_audit_keeps_sign_and_class_medians(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_feature_correlation_audit(catalog, features=("a", "b", "c"), strong_threshold=0.75)

    rows = {(row.feature_a, row.feature_b): row for row in result.correlation_rows}
    assert rows[("a", "b")].spearman_rho == pytest.approx(0.7714, abs=1e-3)
    assert rows[("a", "c")].spearman_rho == pytest.approx(-0.5429, abs=1e-3)
    assert rows[("b", "c")].spearman_rho == pytest.approx(-0.7714, abs=1e-3)
    assert rows[("b", "c")].strong_absolute_correlation is True

    medians = {row.feature_class: row for row in result.class_median_rows}
    assert medians["compact_quality"].medians == (2.0, 2.0, 20.0)
    assert medians["weak_or_background"].feature_class_label == "弱"
    within = {
        (row.feature_class, row.feature_a, row.feature_b): row
        for row in result.class_correlation_rows
    }
    assert within[("compact_quality", "a", "b")].spearman_rho == pytest.approx(-1.0)
    assert within[("weak_or_background", "a", "b")].spearman_rho == pytest.approx(1.0)

    output = write_feature_correlation_artifacts(result, tmp_path / "out")
    assert (output / "feature_correlations.csv").is_file()
    assert (output / "feature_class_medians.csv").is_file()
    assert (output / "feature_class_correlations.csv").is_file()
    payload = json.loads((output / "feature_correlation_audit.json").read_text(encoding="utf-8"))
    assert payload["features"] == ["a", "b", "c"]


def test_feature_correlation_audit_rejects_invalid_input(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    with pytest.raises(ValueError, match="at least two"):
        run_feature_correlation_audit(catalog, features=("a",))
    with pytest.raises(ValueError, match="strong_threshold"):
        run_feature_correlation_audit(catalog, features=("a", "b"), strong_threshold=0.0)
    with pytest.raises(ValueError, match="missing required columns"):
        run_feature_correlation_audit(catalog, features=("a", "missing"))
