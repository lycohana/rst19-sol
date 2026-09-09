from __future__ import annotations

import csv

from rst19.feature_spatial_context import (
    run_feature_spatial_context,
    write_feature_spatial_context_artifacts,
)


def _write_catalog(path) -> None:
    fields = [
        "detection_id",
        "x",
        "y",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flags",
        "flux_snr",
        "filter_snr",
        "peak",
    ]
    rows = [
        (1, 10, 10, "compact_quality", "紧凑质量源", "True", "", 8, 12, 80),
        (2, 12, 10, "compact_quality", "紧凑质量源", "False", "UNRESOLVED_BLEND", 20, 30, 200),
        (3, 90, 90, "compact_quality", "紧凑质量源", "False", "LOW_FLUX_SNR", 10, 15, 100),
        (4, 50, 50, "linear_artifact", "线状候选", "False", "LINE_ARTIFACT", 50, 60, 500),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(fields, row)))


def test_spatial_context_reports_edge_hotspot_and_local_neighbors(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_feature_spatial_context(
        catalog,
        image_width=100,
        image_height=100,
        grid_size=2,
        edge_margin_px=5,
        high_flux_snr_threshold=10,
        top_n=1,
        target_ids=[2],
    )

    compact = next(row for row in result.summaries if row.feature_class == "compact_quality")
    assert compact.class_count == 3
    assert compact.class_rejected_count == 2
    assert compact.high_snr_rejected_count == 2
    assert compact.nonempty_grid_cells == 2
    assert compact.high_snr_hot_grid_row == 0
    assert compact.high_snr_hot_grid_column == 0
    assert compact.edge_count == 0

    target = next(row for row in result.targets if row.detection_id == 2)
    assert target.selection_reason == "top_metric|explicit"
    assert target.nearest_other_candidate_px == 2.0
    assert target.neighbor_count_radius_5px == 1
    assert target.neighbor_quality_count_radius_10px == 1
    assert "compact_quality:1" in target.neighbor_classes_radius_10px

    output = write_feature_spatial_context_artifacts(result, tmp_path / "out")
    assert (output / "feature_spatial_context.json").is_file()
    assert (output / "feature_spatial_context_summary.csv").is_file()
    assert (output / "feature_spatial_context_hotspots.csv").is_file()
    assert (output / "feature_spatial_context_targets.csv").is_file()


def test_spatial_context_rejects_unknown_target(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    try:
        run_feature_spatial_context(catalog, image_width=100, image_height=100, target_ids=[99])
    except ValueError as exc:
        assert "找不到 detection_id" in str(exc)
    else:
        raise AssertionError("expected unknown target to fail")


def test_spatial_context_uses_x_for_width_and_y_for_height(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_feature_spatial_context(
        catalog,
        image_width=200,
        image_height=100,
        grid_size=4,
        top_n=1,
        target_ids=[3],
    )

    target = next(row for row in result.targets if row.detection_id == 3)
    assert target.grid_row == 3
    assert target.grid_column == 1
