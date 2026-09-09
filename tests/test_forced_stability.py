from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rst19.forced_stability import (
    ForcedAnchor,
    ForcedFrameMetric,
    ForcedStabilityResult,
    _local_matched_filter_response,
    _local_search_offsets,
    _select_local_peak_indices,
    load_forced_anchors,
    summarize_forced_pair_relationships,
    summarize_forced_stability,
    write_forced_stability_artifacts,
)


def _write_source_catalog(path: Path) -> None:
    fields = (
        "detection_id",
        "x",
        "y",
        "peak_x",
        "peak_y",
        "flux_snr",
        "feature_class",
        "feature_class_label",
    )
    rows = (
        (1, 10.25, 20.75, 10.0, 21.0, 1.0, "compact_quality", "compact"),
        (2, 30.25, 40.75, 30.0, 41.0, 2.0, "compact_quality", "compact"),
        (3, 50.25, 60.75, 50.0, 61.0, 3.0, "compact_quality", "compact"),
        (4, 70.25, 80.75, 70.0, 81.0, 4.0, "range_anomaly", "range"),
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _metric(anchor: ForcedAnchor, frame_index: int, *, quality_like: bool, flux_snr: float) -> ForcedFrameMetric:
    return ForcedFrameMetric(
        detection_id=anchor.detection_id,
        feature_class=anchor.feature_class,
        feature_class_label=anchor.feature_class_label,
        frame_index=frame_index,
        predicted_x=anchor.x,
        predicted_y=anchor.y,
        flux_snr=flux_snr,
        support_3x3=5,
        center_valid=True,
        fwhm=2.0,
        ellipticity=0.1,
        quality_like=quality_like,
    )


def test_local_peak_selection_is_bounded_and_center_first_on_ties() -> None:
    offsets = _local_search_offsets(1)

    assert offsets.shape == (9, 2)
    assert tuple(offsets[0]) == (0, 0)
    assert {tuple(value) for value in offsets} == {
        (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    }

    flux_snr = np.asarray(
        [
            [5.0, 5.0, np.nan, 4.0, 5.0, 3.0, 2.0, 1.0, 0.0],
            [np.nan] * 9,
        ],
        dtype=np.float64,
    )
    indices = _select_local_peak_indices(flux_snr)

    # 第一行中心点与另一个偏移点并列，中心点优先；全无效行回退中心。
    assert indices.tolist() == [0, 0]

    with pytest.raises(ValueError, match="0 or 1"):
        _local_search_offsets(2)


def test_local_matched_response_follows_a_shifted_compact_source() -> None:
    yy, xx = np.mgrid[:64, :64]
    image = 10.0 + 100.0 * np.exp(
        -0.5 * (((xx - 33.0) / 1.2) ** 2 + ((yy - 30.0) / 1.2) ** 2)
    )
    offsets = _local_search_offsets(1)
    response = _local_matched_filter_response(
        image,
        np.zeros(image.shape, dtype=bool),
        32.0 + offsets[:, 0],
        30.0 + offsets[:, 1],
        global_background=10.0,
        global_noise=1.0,
        psf_fwhm=3.0,
    )

    best = int(_select_local_peak_indices(response[None, :])[0])
    assert tuple(offsets[best]) == (1, 0)


def test_load_forced_anchors_records_peak_or_centroid_explicitly(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_source_catalog(catalog)

    peaks = load_forced_anchors(catalog, max_per_class=2, anchor_coordinate="peak")
    centroids = load_forced_anchors(catalog, max_per_class=2, anchor_coordinate="centroid")
    peaks_with_control = load_forced_anchors(
        catalog,
        max_per_class=1,
        anchor_coordinate="peak",
        include_detection_ids=(2,),
    )

    peak_by_id = {anchor.detection_id: anchor for anchor in peaks}
    centroid_by_id = {anchor.detection_id: anchor for anchor in centroids}
    assert set(peak_by_id) == {1, 3, 4}
    assert set(centroid_by_id) == {1, 3, 4}
    assert (peak_by_id[1].x, peak_by_id[1].y) == (10.0, 21.0)
    assert (centroid_by_id[1].x, centroid_by_id[1].y) == (10.25, 20.75)
    assert {anchor.detection_id for anchor in peaks_with_control} == {1, 2, 4}

    with pytest.raises(ValueError, match="peak.*centroid"):
        load_forced_anchors(catalog, anchor_coordinate="unknown")
    with pytest.raises(ValueError, match="included detection_id"):
        load_forced_anchors(catalog, include_detection_ids=(999,))


def test_summarize_forced_stability_keeps_quality_like_as_a_diagnostic() -> None:
    anchors = (
        ForcedAnchor(1, "compact_quality", "compact", 10.0, 20.0, 10.0),
        ForcedAnchor(2, "compact_quality", "compact", 30.0, 40.0, 10.0),
    )
    metrics = tuple(
        metric
        for anchor in anchors
        for metric in (
            _metric(anchor, 1, quality_like=True, flux_snr=10.0),
            _metric(anchor, 2, quality_like=anchor.detection_id == 1, flux_snr=11.0),
            _metric(anchor, 3, quality_like=anchor.detection_id == 1, flux_snr=9.0),
        )
    )

    rows = summarize_forced_stability(anchors, metrics, frame_count=3)

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_count"] == 2
    assert row["frame_count"] == 3
    assert row["median_quality_like_presence"] == 2.0
    assert row["quality_like_presence_ge_12_count"] == 0
    assert row["median_flux_snr_across_frames"] == 10.0
    assert row["robust_flux_snr_cv_median"] == pytest.approx(0.14826)
    assert row["median_local_peak_flux_gain_fraction"] == 0.0
    assert row["median_modal_relocalization_fraction"] == 1.0

    with pytest.raises(ValueError, match="expected 3"):
        summarize_forced_stability(anchors, metrics[:-1], frame_count=3)


def test_pair_relationship_reports_inward_local_shifts() -> None:
    anchors = (
        ForcedAnchor(11, "range_anomaly", "range", 20.0, 30.0, 10.0),
        ForcedAnchor(12, "crowded_blend", "blend", 16.0, 30.0, 10.0),
    )
    metrics = (
        ForcedFrameMetric(
            detection_id=11,
            feature_class="range_anomaly",
            feature_class_label="range",
            frame_index=1,
            predicted_x=20.0,
            predicted_y=30.0,
            flux_snr=10.0,
            support_3x3=5,
            center_valid=True,
            fwhm=2.0,
            ellipticity=0.1,
            quality_like=True,
            local_peak_x=19.0,
            local_peak_y=30.0,
        ),
        ForcedFrameMetric(
            detection_id=12,
            feature_class="crowded_blend",
            feature_class_label="blend",
            frame_index=1,
            predicted_x=16.0,
            predicted_y=30.0,
            flux_snr=10.0,
            support_3x3=5,
            center_valid=True,
            fwhm=2.0,
            ellipticity=0.1,
            quality_like=True,
            local_peak_x=17.0,
            local_peak_y=30.0,
        ),
    )

    rows = summarize_forced_pair_relationships(anchors, metrics, (11, 12), frame_count=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["anchor_separation_px"] == pytest.approx(4.0)
    assert row["median_local_separation_px"] == pytest.approx(2.0)
    assert row["median_separation_change_px"] == pytest.approx(-2.0)
    assert row["local_separation_contract_fraction"] == 1.0
    assert row["both_shift_toward_each_other_fraction"] == 1.0
    assert row["local_offset_same_fraction"] == 0.0

    assert summarize_forced_pair_relationships(anchors, metrics, (11,), frame_count=1) == ()


def test_write_forced_stability_artifacts_persists_coordinate_definition(tmp_path: Path) -> None:
    anchor = ForcedAnchor(1, "compact_quality", "compact", 10.0, 20.0, 8.0)
    metric = _metric(anchor, 1, quality_like=True, flux_snr=8.0)
    result = ForcedStabilityResult(
        source_catalog=Path("source_catalog.csv"),
        persistence_json=Path("sequence.json"),
        anchor_coordinate="peak",
        included_detection_ids=(1,),
        frame_count=1,
        anchors=(anchor,),
        frame_metrics=(metric,),
        summary_rows=summarize_forced_stability((anchor,), (metric,), frame_count=1),
        aperture_radius=4,
        min_psf_support_pixels=3,
        min_flux_snr=5.0,
        min_fwhm=0.8,
        max_fwhm=12.0,
        max_ellipticity=0.65,
    )

    output = write_forced_stability_artifacts(result, tmp_path / "audit")
    payload = json.loads((output / "forced_stability.json").read_text(encoding="utf-8"))

    assert payload["parameters"]["anchor_coordinate"] == "peak"
    assert payload["parameters"]["included_detection_ids"] == [1]
    assert payload["parameters"]["local_peak_search_radius"] == 0
    assert "local_peak_selection" in payload["parameters"]
    assert "peak_x/peak_y" in payload["parameters"]["anchor_coordinate_definition"]
    assert (output / "forced_stability_frame_metrics.csv").is_file()
    assert (output / "forced_stability_summary.csv").is_file()
    assert not (output / "forced_stability_pair_summary.csv").exists()


def test_write_forced_stability_artifacts_emits_explicit_pair_tables(tmp_path: Path) -> None:
    anchors = (
        ForcedAnchor(11, "range_anomaly", "range", 20.0, 30.0, 8.0),
        ForcedAnchor(12, "crowded_blend", "blend", 16.0, 30.0, 7.0),
    )
    metrics = (
        ForcedFrameMetric(
            detection_id=11,
            feature_class="range_anomaly",
            feature_class_label="range",
            frame_index=1,
            predicted_x=20.0,
            predicted_y=30.0,
            flux_snr=8.0,
            support_3x3=5,
            center_valid=True,
            fwhm=2.0,
            ellipticity=0.1,
            quality_like=True,
            local_peak_x=19.0,
            local_peak_y=30.0,
        ),
        ForcedFrameMetric(
            detection_id=12,
            feature_class="crowded_blend",
            feature_class_label="blend",
            frame_index=1,
            predicted_x=16.0,
            predicted_y=30.0,
            flux_snr=7.0,
            support_3x3=5,
            center_valid=True,
            fwhm=2.0,
            ellipticity=0.1,
            quality_like=True,
            local_peak_x=17.0,
            local_peak_y=30.0,
        ),
    )
    result = ForcedStabilityResult(
        source_catalog=Path("source_catalog.csv"),
        persistence_json=Path("sequence.json"),
        anchor_coordinate="peak",
        included_detection_ids=(11, 12),
        frame_count=1,
        anchors=anchors,
        frame_metrics=metrics,
        summary_rows=summarize_forced_stability(anchors, metrics, frame_count=1),
        aperture_radius=4,
        min_psf_support_pixels=3,
        min_flux_snr=5.0,
        min_fwhm=0.8,
        max_fwhm=12.0,
        max_ellipticity=0.65,
    )

    output = write_forced_stability_artifacts(result, tmp_path / "pair-audit")

    assert (output / "forced_stability_pair_frame_metrics.csv").is_file()
    assert (output / "forced_stability_pair_summary.csv").is_file()
    with (output / "forced_stability_pair_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        pair_summary = list(csv.DictReader(stream))
    assert pair_summary[0]["both_shift_toward_each_other_fraction"] == "1.0"
    payload = json.loads((output / "forced_stability.json").read_text(encoding="utf-8"))
    assert payload["pair_relationship_summary"][0]["median_local_separation_px"] == 2.0
