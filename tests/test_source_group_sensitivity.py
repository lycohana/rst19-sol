from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rst19.source_group_sensitivity import (
    run_source_group_sensitivity,
    write_source_group_sensitivity_artifacts,
)


def _write_catalog(path: Path) -> None:
    fields = (
        "detection_id",
        "x",
        "y",
        "peak_x",
        "peak_y",
        "peak",
        "flux_snr",
        "filter_snr",
        "deblend_delta_bic",
        "deblend_component_snr",
        "quality_passed",
        "feature_class",
        "flags",
    )
    rows = (
        (1, 0.0, 0.0, 0, 0, 3992, 204.6, 1195, -4.09, 0.87, "False", "range_anomaly", "CODE_PATTERN"),
        (2, 2.738, 0.0, 4, 0, 569, 115.5, 300, -4.09, 0.87, "False", "crowded_blend", "UNRESOLVED_BLEND"),
        (3, 20.0, 0.0, 20, 0, 250, 12, 20, "", "", "True", "compact_quality", ""),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_radius_sensitivity_preserves_order_and_target_routing(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_source_group_sensitivity(
        catalog,
        group_radii_px=(2.5, 3.0, 4.0),
        target_ids=(1, 2),
        psf_fwhm=2.0,
        workers=2,
    )

    assert [row.group_radius_px for row in result.rows] == [2.5, 3.0, 4.0]
    assert [row.group_count for row in result.rows] == [3, 2, 2]
    assert [row.multi_member_group_count for row in result.rows] == [0, 1, 1]
    assert [target.classification for target in result.targets] == [
        "isolated",
        "isolated",
        "unresolved_group",
        "unresolved_group",
    ]
    assert result.targets[2].target_detection_ids == "1|2"


def test_radius_sensitivity_writes_artifacts_and_validates_radii(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_source_group_sensitivity(catalog, group_radii_px=(3.0,))
    output = write_source_group_sensitivity_artifacts(result, tmp_path / "out")

    assert (output / "source_group_radius_sensitivity.csv").is_file()
    assert (output / "source_group_radius_targets.csv").is_file()
    payload = json.loads(
        (output / "source_group_radius_sensitivity.json").read_text(encoding="utf-8")
    )
    assert payload["rows"][0]["group_radius_px"] == 3.0
    with pytest.raises(ValueError, match="unique"):
        run_source_group_sensitivity(catalog, group_radii_px=(3.0, 3.0))
    with pytest.raises(ValueError, match="finite"):
        run_source_group_sensitivity(catalog, group_radii_px=(float("nan"),))
