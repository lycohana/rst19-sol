from __future__ import annotations

import csv

from rst19.feature_flag_interaction import run_feature_flag_interaction, write_feature_flag_interaction_artifacts


def _write_catalog(path) -> None:
    rows = [
        {"detection_id": "1", "quality_passed": "True", "flags": "PARTIAL_MASKED", "flux_snr": "8"},
        {"detection_id": "2", "quality_passed": "False", "flags": "LOW_FLUX_SNR|PARTIAL_MASKED", "flux_snr": "12"},
        {
            "detection_id": "3",
            "quality_passed": "False",
            "flags": "INSUFFICIENT_PSF_SUPPORT|PARTIAL_MASKED",
            "flux_snr": "20",
        },
        {"detection_id": "4", "quality_passed": "False", "flags": "LOW_FLUX_SNR", "flux_snr": "11"},
        {"detection_id": "5", "quality_passed": "True", "flags": "", "flux_snr": "30"},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_flag_interaction_separates_presence_from_exact_singleton(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_feature_flag_interaction(catalog, high_snr_threshold=10)
    rows = {row.interaction: row for row in result.rows}

    partial = rows["PARTIAL_MASKED"]
    assert partial.candidate_count == 3
    assert partial.quality_count == 1
    assert partial.exact_set_count == 1
    assert partial.exact_set_quality_count == 1

    masked_low = rows["LOW_FLUX_SNR+PARTIAL_MASKED"]
    assert masked_low.candidate_count == 1
    assert masked_low.quality_count == 0
    assert masked_low.exact_set_count == 1
    assert masked_low.high_snr_rejected_count == 1


def test_flag_interaction_writes_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    output = write_feature_flag_interaction_artifacts(
        run_feature_flag_interaction(catalog),
        tmp_path / "out",
    )
    assert (output / "feature_flag_interactions.csv").exists()
    assert (output / "feature_flag_interactions.json").exists()
