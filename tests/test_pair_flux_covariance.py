from __future__ import annotations

import json

from rst19.pair_flux_covariance import (
    ForcedStabilityFluxRow,
    load_forced_stability_flux_csv,
    run_pair_flux_covariance_audit,
    write_pair_flux_covariance_artifacts,
)


def _rows() -> tuple[ForcedStabilityFluxRow, ...]:
    target_a = (10.0, 20.0, 30.0, 40.0, 50.0)
    target_b = (5.0, 10.0, 15.0, 20.0, 25.0)
    controls = {
        1: (8.0, 7.0, 6.0, 5.0, 4.0),
        2: (2.0, 9.0, 3.0, 8.0, 4.0),
        3: (4.0, 1.0, 5.0, 2.0, 6.0),
        4: (9.0, 3.0, 8.0, 2.0, 7.0),
    }
    rows = [
        ForcedStabilityFluxRow(82931, index, a, a + 1.0)
        for index, a in enumerate(target_a, 1)
    ]
    rows.extend(
        ForcedStabilityFluxRow(82934, index, b, b + 1.0)
        for index, b in enumerate(target_b, 1)
    )
    for detection_id, values in controls.items():
        rows.extend(
            ForcedStabilityFluxRow(detection_id, index, value, value + 0.5)
            for index, value in enumerate(values, 1)
        )
    return tuple(rows)


def test_pair_flux_covariance_compares_raw_and_frame_normalized_controls() -> None:
    result = run_pair_flux_covariance_audit(_rows(), 82931, 82934)

    assert result.frame_count == 5
    assert result.control_source_count == 4
    assert len(result.metrics) == 2
    fixed = result.metrics[0]
    assert fixed.metric_name == "flux_snr"
    assert fixed.primary_pearson == 1.0
    assert fixed.primary_spearman == 1.0
    assert fixed.frame_median_normalized_pearson is not None
    assert fixed.control_pair_count == 6
    assert fixed.primary_fraction_median == 2.0 / 3.0
    assert fixed.primary_fraction_range == 0.0
    assert fixed.primary_fraction_mad_scaled == 0.0
    assert fixed.below_pair_total_median_primary_fraction_median == 2.0 / 3.0
    assert fixed.above_pair_total_median_primary_fraction_median == 2.0 / 3.0
    assert fixed.primary_fraction_shift_high_minus_low == 0.0
    assert fixed.control_fraction_shift_pair_count == 6
    assert fixed.control_fraction_shift_abs_median is not None
    assert fixed.control_fraction_shift_abs_p95 is not None
    assert fixed.absolute_fraction_shift_upper_tail_fraction == 1.0
    assert fixed.below_pair_total_median_frame_count == 2
    assert fixed.above_pair_total_median_frame_count == 3
    assert fixed.below_pair_total_median_pearson is None
    assert fixed.above_pair_total_median_pearson == 1.0
    assert "不是双星概率" in result.conclusion


def test_pair_flux_covariance_writes_artifacts_and_loader(tmp_path) -> None:
    csv_path = tmp_path / "forced.csv"
    csv_path.write_text(
        "detection_id,frame_index,flux_snr,local_peak_flux_snr\n"
        "82931,1,10,11\n82931,2,20,21\n82931,3,30,31\n"
        "82934,1,5,6\n82934,2,10,11\n82934,3,15,16\n"
        "1,1,8,9\n1,2,7,8\n1,3,6,7\n"
        "2,1,2,3\n2,2,9,10\n2,3,3,4\n"
        "3,1,4,5\n3,2,1,2\n3,3,5,6\n",
        encoding="utf-8",
    )
    rows = load_forced_stability_flux_csv(csv_path)
    result = run_pair_flux_covariance_audit(rows, 82931, 82934)
    output = write_pair_flux_covariance_artifacts(result, tmp_path / "out")

    assert (output / "pair_flux_covariance.csv").is_file()
    payload = json.loads((output / "pair_flux_covariance.json").read_text(encoding="utf-8"))
    assert payload["primary_detection_id"] == 82931
    assert payload["metrics"][0]["control_pair_count"] == 3


def test_pair_flux_covariance_rejects_duplicate_frame() -> None:
    rows = list(_rows())
    rows.append(rows[0])
    try:
        run_pair_flux_covariance_audit(rows, 82931, 82934)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate frame should be rejected")
