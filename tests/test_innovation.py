from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from rst19.innovation import (
    _absolute_magnitude_evidence,
    _compact_photometric_row,
    _difference_rows,
    _fit_constant_velocity,
    _motion_audit_rows,
    _motion_point_rows,
    _motion_rows,
    _point_motion_rows,
    _photometric_evidence_section,
    _row_evidence_level,
    _telemetry_consistency,
    _telemetry_position_chart,
    _telemetry_prediction,
    _vector_metrics,
    build_innovation_report_from_payload,
)
from rst19.photometric_report import build_photometric_report


def test_constant_velocity_fit_reports_small_sample_uncertainty() -> None:
    fit = _fit_constant_velocity(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.1, 1.9, 3.2],
        [0.0, 2.1, 3.9, 6.2],
        2.0,
    )

    assert fit["degrees_of_freedom"] == 2
    assert fit["t95_critical"] == pytest.approx(4.303)
    assert fit["speed_ci95_low_px_per_s"] < fit["speed_px_per_s"] < fit["speed_ci95_high_px_per_s"]
    assert fit["direction_ci95_half_width_deg"] > 0
    assert fit["predicted_x_ci95_half_width_px"] > 0
    assert fit["predicted_y_ci95_half_width_px"] > 0
    assert fit["forecast_elapsed_s"] == pytest.approx(5.0)


def test_constant_velocity_fit_does_not_invent_uncertainty_from_two_points() -> None:
    fit = _fit_constant_velocity([0.0, 2.0], [1.0, 5.0], [3.0, 3.0], 1.0)

    assert fit["degrees_of_freedom"] == 0
    assert fit["speed_px_per_s"] == pytest.approx(2.0)
    assert fit["predicted_x_px"] == pytest.approx(7.0)
    assert fit["speed_ci95_low_px_per_s"] is None
    assert fit["direction_ci95_half_width_deg"] is None
    assert fit["predicted_x_ci95_half_width_px"] is None


def test_motion_rows_uses_observation_time_for_pixel_speed() -> None:
    frame_rows = [
        {"timestamp": "2026-03-30T16:32:05.000", "frame_index": 0},
        {"timestamp": "2026-03-30T16:32:06.500", "frame_index": 1},
    ]
    payload = {
        "motion_features": [
            {
                "track_id": 3,
                "classification": "moving",
                "points": [
                    {"frame_index": 0, "aligned_x": 10.0, "aligned_y": 20.0, "length_px": 40.0, "width_px": 3.0, "residual_snr": 20.0},
                    {"frame_index": 1, "aligned_x": 25.0, "aligned_y": 20.0, "length_px": 41.0, "width_px": 3.0, "residual_snr": 22.0},
                ],
            }
        ]
    }

    rows = _motion_rows(payload, frame_rows)

    assert rows[0]["duration_s"] == pytest.approx(1.5)
    assert rows[0]["speed_px_per_s"] == pytest.approx(10.0)
    assert rows[0]["displacement_px"] == pytest.approx(15.0)
    assert rows[0]["direction_deg_image"] == pytest.approx(0.0)
    assert rows[0]["points"][0]["fit_residual_px"] == pytest.approx(0.0)
    assert rows[0]["points"][1]["fit_residual_px"] == pytest.approx(0.0)


def test_motion_rows_keeps_single_frame_candidate_without_fake_speed() -> None:
    rows = _motion_rows(
        {"motion_features": [{"track_id": 1, "classification": "candidate", "points": [{"frame_index": 0, "x": 2.0, "y": 3.0, "angle_deg": 90.0, "length_px": 30.0, "width_px": 2.0, "residual_snr": 10.0}]}]},
        [{"timestamp": "2026-03-30T16:32:05.000", "frame_index": 0}],
    )

    assert rows[0]["duration_s"] == pytest.approx(0.0)
    assert rows[0]["speed_px_per_s"] is None
    assert rows[0]["direction_deg_image"] == pytest.approx(90.0)


def test_point_motion_rows_uses_date_obs_and_keeps_evidence_level() -> None:
    rows = _point_motion_rows(
        {
            "tracks": [
                {
                    "track_id": 12,
                    "classification": "moving",
                    "evidence_level": "fast_point_motion",
                    "speed_px_per_frame": 15.0,
                    "points": [
                        {"frame_index": 0, "aligned_x": 10.0, "aligned_y": 20.0, "flux_snr": 18.0, "candidate_snr": 22.0},
                        {"frame_index": 1, "aligned_x": 25.0, "aligned_y": 20.0, "flux_snr": 20.0, "candidate_snr": 24.0},
                    ],
                },
                {
                    "track_id": 13,
                    "classification": "static",
                    "points": [{"frame_index": 0, "aligned_x": 4.0, "aligned_y": 5.0}],
                },
            ]
        },
        [
            {"timestamp": "2026-03-30T16:32:05.000", "frame_index": 0},
            {"timestamp": "2026-03-30T16:32:06.500", "frame_index": 1},
        ],
    )

    assert len(rows) == 1
    assert rows[0]["track_id"] == 12
    assert rows[0]["evidence_level"] == "fast_point_motion"
    assert rows[0]["speed_px_per_s"] == pytest.approx(10.0)
    assert rows[0]["displacement_px"] == pytest.approx(15.0)
    assert rows[0]["median_flux_snr"] == pytest.approx(19.0)
    assert rows[0]["points"][1]["fit_residual_px"] == pytest.approx(0.0)


def test_point_motion_rows_skips_velocity_fit_when_one_timestamp_is_missing() -> None:
    rows = _point_motion_rows(
        {
            "tracks": [
                {
                    "track_id": 21,
                    "classification": "moving",
                    "evidence_level": "fast_point_motion",
                    "points": [
                        {"frame_index": 0, "aligned_x": 10.0, "aligned_y": 20.0},
                        {"frame_index": 1, "aligned_x": 14.0, "aligned_y": 18.0},
                    ],
                }
            ]
        },
        [
            {"timestamp": "2026-03-30T16:32:05.000", "frame_index": 0},
            {"timestamp": None, "frame_index": 1},
        ],
    )

    assert len(rows) == 1
    assert rows[0]["kinematic_model"] is None
    assert rows[0]["speed_px_per_s"] is None
    assert rows[0]["points"][0]["fit_residual_px"] is None


def test_motion_point_rows_exposes_one_based_frame_number() -> None:
    rows = _motion_point_rows(
        [
            {
                "track_id": 7,
                "classification": "candidate",
                "points": [{"frame_index": 0, "x": 12.0, "y": 8.0}],
            }
        ]
    )

    assert rows[0]["frame_index"] == 0
    assert rows[0]["frame_number"] == 1


def test_motion_audit_rows_exposes_one_based_frame_number_and_filter_counts() -> None:
    rows = _motion_audit_rows(
        {
            "motion_frame_audits": [
                {
                    "frame_index": 8,
                    "component_count": 12,
                    "area_pass_count": 4,
                    "geometry_pass_count": 2,
                    "edge_rejected_count": 1,
                    "feature_count": 1,
                }
            ]
        }
    )

    assert rows[0]["frame_index"] == 8
    assert rows[0]["frame_number"] == 9
    assert rows[0]["geometry_pass_count"] == 2
    assert rows[0]["feature_count"] == 1


def test_difference_rows_preserves_raw_adu_units() -> None:
    rows = _difference_rows(
        [{"frame_index": 0, "timestamp": "2026-03-30T16:32:05.000", "background_rms_adu": 2.0}],
        [np.array([[10.0, 14.0], [8.0, np.nan]])],
        np.full((2, 2), 10.0),
    )

    assert rows[0]["residual_median_adu"] == pytest.approx(0.0)
    assert rows[0]["residual_rms_adu"] == pytest.approx(2.9652)
    assert rows[0]["residual_p99_abs_adu"] == pytest.approx(3.96, abs=0.01)
    assert rows[0]["positive_peak_sigma"] == pytest.approx(2.0)
    assert rows[0]["negative_peak_sigma"] == pytest.approx(-1.0)
    assert rows[0]["pixels_abs_gt_5sigma"] == 0
    assert rows[0]["valid_fraction"] == pytest.approx(0.75)


def test_vector_metrics_reports_position_and_velocity_directions() -> None:
    metrics = _vector_metrics(
        {
            "j2000_x": 3.0,
            "j2000_y": 4.0,
            "j2000_z": 0.0,
            "j2000_xv": 0.0,
            "j2000_yv": 1.0,
            "j2000_zv": 0.0,
        }
    )

    assert metrics["j2000_position_norm_raw"] == pytest.approx(5.0)
    assert metrics["j2000_position_azimuth_deg"] == pytest.approx(53.1301024)
    assert metrics["j2000_position_elevation_deg"] == pytest.approx(0.0)
    assert metrics["j2000_velocity_norm_raw"] == pytest.approx(1.0)
    assert metrics["j2000_velocity_azimuth_deg"] == pytest.approx(90.0)
    assert metrics["j2000_velocity_elevation_deg"] == pytest.approx(0.0)


def test_telemetry_consistency_uses_central_difference_for_interior_frames() -> None:
    rows = [
        {"timestamp": "2026-03-30T16:32:05.000", "j2000_x": 0.0, "j2000_y": 0.0, "j2000_z": 0.0, "j2000_xv": 10.0, "j2000_yv": 0.0, "j2000_zv": 0.0},
        {"timestamp": "2026-03-30T16:32:06.000", "j2000_x": 10.0, "j2000_y": 0.0, "j2000_z": 0.0, "j2000_xv": 10.0, "j2000_yv": 0.0, "j2000_zv": 0.0},
        {"timestamp": "2026-03-30T16:32:07.000", "j2000_x": 20.0, "j2000_y": 0.0, "j2000_z": 0.0, "j2000_xv": 10.0, "j2000_yv": 0.0, "j2000_zv": 0.0},
    ]

    summary = _telemetry_consistency(rows, "j2000")

    assert summary["valid_sample_count"] == 3
    assert summary["interior_sample_count"] == 1
    assert summary["position_rate_norm_m_per_s_median"] == pytest.approx(10.0)
    assert summary["interior_relative_error_assuming_m_per_s_median"] == pytest.approx(0.0)
    assert rows[1]["j2000_position_rate_norm_m_per_s"] == pytest.approx(10.0)
    assert rows[1]["j2000_velocity_consistency_relative_error_assuming_m_per_s"] == pytest.approx(0.0)


def test_telemetry_prediction_uses_last_state_and_declares_unit_assumption() -> None:
    rows = [
        {
            "frame_index": 0,
            "j2000_x": 100.0,
            "j2000_y": 200.0,
            "j2000_z": 300.0,
            "j2000_xv": 2.0,
            "j2000_yv": -1.0,
            "j2000_zv": 0.5,
        },
        {
            "frame_index": 1,
            "j2000_x": 110.0,
            "j2000_y": 195.0,
            "j2000_z": 302.5,
            "j2000_xv": 2.0,
            "j2000_yv": -1.0,
            "j2000_zv": 0.5,
        },
    ]

    prediction = _telemetry_prediction(rows, "j2000", 5.0)

    assert prediction["status"] == "available"
    assert prediction["frame_index"] == 1
    assert prediction["predicted_position_m_assuming_m_per_s"] == pytest.approx([120.0, 190.0, 305.0])
    assert prediction["unit_status"] == "速度单位待主办方确认"


def test_telemetry_prediction_is_unavailable_without_valid_last_state() -> None:
    prediction = _telemetry_prediction([{"j2000_x": 1.0}], "j2000", 5.0)

    assert prediction["status"] == "unavailable"
    assert prediction["predicted_position_m_assuming_m_per_s"] is None


def test_telemetry_position_chart_writes_observation_and_forecast_panels(tmp_path) -> None:
    rows = [
        {
            "j2000_x": 100.0,
            "j2000_y": 200.0,
            "wgs84_x": 500.0,
            "wgs84_y": 800.0,
        },
        {
            "j2000_x": 110.0,
            "j2000_y": 195.0,
            "wgs84_x": 510.0,
            "wgs84_y": 790.0,
        },
    ]
    predictions = {
        "j2000": {"predicted_position_m_assuming_m_per_s": [120.0, 190.0, 305.0]},
        "wgs84": {"predicted_position_m_assuming_m_per_s": [520.0, 780.0, 905.0]},
    }

    output = tmp_path / "telemetry_position.png"
    _telemetry_position_chart(output, rows, predictions)

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (1200, 680)


def _minimal_innovation_payload(**extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "frames": [],
        "cumulative_shifts": [],
        "motion_features": [],
    }
    payload.update(extra)
    return payload


def test_innovation_report_consumes_relative_scale_without_promoting_absolute_magnitude() -> None:
    payload = _minimal_innovation_payload(
        relative_photometry={
            "status": "VALID",
            "flags": ["HOLDOUT_SOURCE_BLOCK"],
            "reference_sample_count": 12,
            "frame_zero_points": {"0": 0.12, "1": -0.12},
            "relative_magnitudes": {"101": 3.4, "102": 4.1},
            "training_residual_rms": 0.018,
            "validation_residual_rms": 0.027,
            "validation_sample_count": 2,
        }
    )

    report = build_innovation_report_from_payload(payload)
    evidence = report["photometric_evidence"]
    relative = evidence["relative_photometry"]

    assert evidence["status"] == "AVAILABLE"
    assert evidence["evidence_sources"] == ["relative_photometry"]
    assert relative["usable"] is True
    assert relative["relative_magnitude_count"] == 2
    assert relative["relative_magnitudes"] == {"101": 3.4, "102": 4.1}
    assert relative["training_residual_rms_mag"] == pytest.approx(0.018)
    assert evidence["summary"]["reported_apparent_magnitude_count"] == 0
    assert evidence["summary"]["reported_absolute_magnitude_value_count"] == 0
    assert evidence["boundary"]["relative_magnitude"]["status"] == "CONSUMED_WITH_RELATIVE_LABEL"
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "NOT_INFERRED"
    assert relative["absolute_magnitude_claim"] == "NOT_DERIVED_FROM_RELATIVE_SCALE"

    # The additive section must remain strict-JSON serialisable for artifact export.
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_innovation_report_consumes_quality_report_and_preserves_false_valid_gate() -> None:
    quality_report = {
        "frame_count": 2,
        "source_count": 3,
        "level_counts": {
            "INSTRUMENTAL_ONLY": 1,
            "APPARENT_CALIBRATED": 1,
            "ABSOLUTE_ELIGIBLE": 1,
        },
        "false_valid": False,
        "false_valid_gate": {"passed": True, "violation_count": 0},
        "provenance": {"catalog": ["Gaia DR3"], "wcs": "astrometric-solution-v1"},
        "rows": [
            {
                "row_key": "0:10",
                "frame_id": "0",
                "source_id": "10",
                "observability_level": "APPARENT_CALIBRATED",
                "m_inst": 15.2,
                "m_cal": 13.7,
                "status": "CALIBRATED",
            },
            {
                "row_key": "0:11",
                "frame_id": "0",
                "source_id": "11",
                "observability_level": "ABSOLUTE_ELIGIBLE",
                "m_inst": 16.0,
                "m_cal": 14.5,
                "M": 4.2,
                "status": "ABSOLUTE_ELIGIBLE",
            },
        ],
    }

    evidence = _photometric_evidence_section({"photometric_quality_report": quality_report})

    assert evidence["status"] == "AVAILABLE"
    quality = evidence["photometric_quality"]
    assert quality["present"] is True
    assert quality["source"] == "photometric_quality_report"
    assert quality["row_count"] == 2
    assert quality["level_counts"]["ABSOLUTE_ELIGIBLE"] == 1
    assert quality["apparent_magnitude_count"] == 2
    assert quality["absolute_magnitude_value_count"] == 1
    assert quality["false_valid"] is False
    assert quality["false_valid_gate_passed"] is True
    assert quality["provenance"]["catalog"] == ["Gaia DR3"]
    assert evidence["summary"]["reported_absolute_magnitude_value_count"] == 1
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "DIAGNOSTIC_ONLY"
    assert evidence["boundary"]["absolute_magnitude"]["eligible_count"] == 0
    assert evidence["boundary"]["absolute_magnitude"]["reported_eligible_count"] == 1
    assert quality["absolute_magnitude_strict_count"] == 0
    assert quality["derived_level_counts"]["APPARENT_CALIBRATED"] == 2


def test_innovation_report_keeps_unqualified_nested_model_distance_as_diagnostic() -> None:
    evidence = _photometric_evidence_section(
        {
            "source_photometry": [
                {
                    "source_id": "gsp-1",
                    "status": "CALIBRATED",
                    "m_inst": 16.0,
                    "m_cal": 14.5,
                    "absolute_magnitude": {
                        "value": 4.2,
                        "status": "VALID_MODEL_DISTANCE",
                        "distance_source": "Gaia DR3 GSP-Phot",
                    },
                }
            ],
            "photometric_calibration": {"status": "VALID"},
        }
    )

    quality = evidence["photometric_quality"]
    assert quality["level_counts"]["APPARENT_CALIBRATED"] == 1
    assert quality["absolute_magnitude_value_count"] == 1
    assert quality["absolute_magnitude_strict_count"] == 0
    assert quality["absolute_magnitude_diagnostic_only_count"] == 1
    assert "IS_STRICT_MISSING" in quality["absolute_magnitude_strict_reason_counts"]
    assert "MODEL_DISTANCE_INTERVAL_MISSING" in quality["absolute_magnitude_strict_reason_counts"]


def test_innovation_report_rejects_quality_report_when_false_valid_gate_fails() -> None:
    evidence = _photometric_evidence_section(
        {
            "photometric_report": {
                "rows": [
                    {
                        "observability_level": "APPARENT_CALIBRATED",
                        "m_cal": 12.0,
                    }
                ],
                "false_valid_gate": {"passed": False},
            }
        }
    )

    assert evidence["status"] == "PRESENT_BUT_REJECTED"
    assert evidence["photometric_quality"]["false_valid"] is True
    assert evidence["boundary"]["apparent_magnitude"]["status"] == "REPORTED_BY_INPUT_ONLY"


def test_innovation_report_accepts_count_only_quality_summary() -> None:
    evidence = _photometric_evidence_section(
        {
            "photometric_quality": {
                "observability_counts": {
                    "APPARENT_CALIBRATED": 4,
                    "ABSOLUTE_ELIGIBLE": 1,
                }
            }
        }
    )

    assert evidence["status"] == "AVAILABLE"
    assert evidence["summary"]["reported_apparent_magnitude_count"] == 5
    assert evidence["summary"]["reported_absolute_eligible_count"] == 1
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "DIAGNOSTIC_ONLY"
    assert evidence["boundary"]["absolute_magnitude"]["strict_count"] == 0
    assert evidence["boundary"]["absolute_magnitude"]["reported_eligible_count"] == 1


def test_innovation_preserves_legacy_aggregate_absolute_counts_as_diagnostics() -> None:
    evidence = _photometric_evidence_section(
        {
            "photometric_quality": {
                "absolute_magnitude_value_count": 2,
                "absolute_eligible_count": 1,
            }
        }
    )

    quality = evidence["photometric_quality"]
    assert quality["absolute_magnitude_diagnostic_count"] == 2
    assert quality["absolute_magnitude_diagnostic_only_count"] == 2
    assert quality["absolute_magnitude_strict_count"] == 0
    assert quality["reported_absolute_eligible_count"] == 1
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "DIAGNOSTIC_ONLY"


def test_innovation_report_keeps_old_payload_compatible_with_structured_boundary() -> None:
    evidence = _photometric_evidence_section(_minimal_innovation_payload())

    assert evidence["status"] == "NOT_PRESENT"
    assert evidence["evidence_sources"] == []
    assert evidence["relative_photometry"]["present"] is False
    assert evidence["photometric_quality"]["present"] is False
    assert evidence["boundary"]["instrumental_magnitude"]["status"] == "INPUT_MEASUREMENT_ONLY"
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "NOT_INFERRED"


def test_innovation_report_accepts_result_like_payload_and_ignores_raw_detection_sources() -> None:
    class ResultLike:
        def as_dict(self) -> dict[str, object]:
            return _minimal_innovation_payload(
                detection={"sources": [{"peak": 999}]},
                source_photometry=[
                    {
                        "detection_id": 4,
                        "instrumental_magnitude": 17.0,
                        "status": "INSTRUMENTAL",
                    }
                ],
            )

    report = build_innovation_report_from_payload(ResultLike())  # type: ignore[arg-type]
    evidence = report["photometric_evidence"]

    assert evidence["status"] == "INSTRUMENTAL_ONLY"
    quality = evidence["photometric_quality"]
    assert quality["source"] == "source_photometry"
    assert quality["row_count"] == 1
    assert quality["instrumental_magnitude_count"] == 1
    assert quality["apparent_magnitude_count"] == 0
    assert quality["absolute_magnitude_value_count"] == 0
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "NOT_INFERRED"


def test_innovation_report_accepts_top_level_photometry_rows() -> None:
    evidence = _photometric_evidence_section(
        {
            "photometry": [
                {"source_id": "s1", "m_inst": 18.0, "status": "INSTRUMENTAL"},
            ]
        }
    )

    assert evidence["status"] == "INSTRUMENTAL_ONLY"
    assert evidence["photometric_quality"]["source"] == "photometry"
    assert evidence["photometric_quality"]["instrumental_magnitude_count"] == 1


def _strict_absolute_row(*, value_key: str = "value", band: str = "G") -> dict[str, object]:
    absolute: dict[str, object] = {
        value_key: 4.2,
        "status": "VALID_MODEL_DISTANCE",
        "is_strict": True,
        "distance_source": "Gaia DR3 GSP-Phot",
        "distance_pc": 100.0,
        "distance_lower_pc": 95.0,
        "distance_upper_pc": 106.0,
        "distance_interval_status": "PROVIDED",
        "extinction_mag": 0.2,
        "extinction_band": band,
        "extinction_system": "Gaia",
        "extinction_source": "Gaia DR3 GSP-Phot: ag_gspphot",
    }
    return {
        "source_id": "strict-source",
        "status": "CALIBRATED",
        "m_cal": 14.5,
        "photometric_system": "Gaia Vega",
        "photometric_band": band,
        "absolute_magnitude": absolute,
    }


def test_innovation_keeps_strict_and_diagnostic_absolute_values_separate() -> None:
    row = _strict_absolute_row()

    parsed = _absolute_magnitude_evidence(row)
    compact = _compact_photometric_row(row)
    evidence = _photometric_evidence_section({"source_photometry": [row]})

    assert parsed["diagnostic_value"] == pytest.approx(4.2)
    assert parsed["strict_value"] == pytest.approx(4.2)
    assert parsed["gate_status"] == "STRICT"
    assert parsed["band"] == "G"
    assert parsed["label"] == "M_G"
    assert parsed["system"] == "Gaia"
    assert parsed["strict_reasons"] == []
    assert compact["absolute_magnitude"] == pytest.approx(4.2)
    assert compact["absolute_magnitude_diagnostic"] == pytest.approx(4.2)
    assert compact["absolute_magnitude_strict"] == pytest.approx(4.2)
    assert compact["strict_M"] == pytest.approx(4.2)
    assert compact["absolute_magnitude_value_role"] == "STRICT"
    assert compact["absolute_magnitude_label"] == "M_G"
    assert compact["absolute_magnitude_gate_status"] == "STRICT"
    assert evidence["strict_absolute_magnitude"]["count"] == 1
    assert evidence["summary"]["reported_strict_absolute_magnitude_count"] == 1
    assert evidence["boundary"]["absolute_magnitude"]["strict_count"] == 1
    json.dumps(evidence, ensure_ascii=False, allow_nan=False)


def test_photometric_report_serialization_preserves_strict_absolute_evidence() -> None:
    row = _strict_absolute_row()
    row["m_inst"] = 15.0
    row["catalog_name"] = "Gaia DR3"
    report = build_photometric_report(source_rows=[row], require_wcs=False)

    evidence = _photometric_evidence_section(
        {"source_photometry": report.as_dict()["rows"]}
    )

    assert evidence["photometric_quality"]["absolute_magnitude_strict_count"] == 1
    assert evidence["boundary"]["absolute_magnitude"]["strict_count"] == 1
    assert evidence["boundary"]["absolute_magnitude"]["strict_band_counts"] == {"M_G": 1}


def test_innovation_accepts_explicit_flattened_strict_aliases() -> None:
    parsed = _absolute_magnitude_evidence(
        {
            "M": 4.2,
            "strict_M": 4.2,
            "absolute_magnitude_is_strict": True,
            "absolute_status": "VALID",
            "absolute_magnitude_source": "parallax",
            "absolute_magnitude_system": "Gaia Vega",
            "absolute_magnitude_band": "G",
            "distance_pc": 100.0,
            "distance_lower_pc": 95.0,
            "distance_upper_pc": 106.0,
            "distance_interval_status": "PROVIDED",
            "extinction_mag": 0.2,
            "extinction_band": "G",
            "extinction_system": "Gaia",
        }
    )

    assert parsed["diagnostic_value"] == pytest.approx(4.2)
    assert parsed["strict_candidate_value"] == pytest.approx(4.2)
    assert parsed["strict_value"] == pytest.approx(4.2)
    assert parsed["band"] == "G"
    assert parsed["system"] == "Gaia"
    assert parsed["distance_source"] == "parallax"


@pytest.mark.parametrize(
    ("value_key", "expected_band"),
    [("M_G", "G"), ("M_V", "V")],
)
def test_innovation_preserves_named_absolute_band_for_diagnostic_values(
    value_key: str,
    expected_band: str,
) -> None:
    row = {value_key: 4.2, "status": "VALID", "m_cal": 14.5}

    parsed = _absolute_magnitude_evidence(row)

    assert parsed["diagnostic_value"] == pytest.approx(4.2)
    assert parsed["strict_value"] is None
    assert parsed["band"] == expected_band
    assert parsed["label"] == value_key
    assert "IS_STRICT_MISSING" in parsed["strict_reasons"]
    assert "ABSOLUTE_BAND_REQUIRED" not in parsed["strict_reasons"]


def test_innovation_does_not_infer_absolute_band_from_catalog_band() -> None:
    parsed = _absolute_magnitude_evidence(
        {
            "M": 4.2,
            "status": "VALID",
            "catalog_magnitude": 14.5,
            "catalog_band": "G",
        }
    )

    assert parsed["diagnostic_value"] == pytest.approx(4.2)
    assert parsed["band"] is None
    assert parsed["label"] == "M"
    assert parsed["strict_value"] is None
    assert "ABSOLUTE_BAND_REQUIRED" in parsed["strict_reasons"]


def test_innovation_reports_model_distance_without_interval_as_diagnostic_only() -> None:
    row = _strict_absolute_row()
    absolute = row["absolute_magnitude"]
    assert isinstance(absolute, dict)
    absolute.update(
        {
            "status": "VALID_MODEL_DISTANCE_NO_INTERVAL",
            "is_strict": False,
            "distance_interval_status": "REQUIRED",
        }
    )

    parsed = _absolute_magnitude_evidence(row)
    evidence = _photometric_evidence_section({"source_photometry": [row]})

    assert parsed["diagnostic_value"] == pytest.approx(4.2)
    assert parsed["strict_value"] is None
    assert parsed["gate_status"] == "DIAGNOSTIC_ONLY"
    assert "IS_STRICT_FALSE" in parsed["strict_reasons"]
    assert "MODEL_DISTANCE_INTERVAL_MISSING" in parsed["strict_reasons"]
    assert evidence["photometric_quality"]["absolute_magnitude_strict_count"] == 0
    assert evidence["photometric_quality"]["absolute_magnitude_diagnostic_only_count"] == 1
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "DIAGNOSTIC_ONLY"


def test_innovation_explains_missing_distance_source_and_extinction_semantics() -> None:
    row = _strict_absolute_row()
    absolute = row["absolute_magnitude"]
    assert isinstance(absolute, dict)
    absolute.pop("distance_source")
    absolute.pop("extinction_band")
    absolute.pop("extinction_system")
    absolute["distance_interval_status"] = "DERIVED_FROM_PARALLAX_ERROR"
    absolute["status"] = "VALID"

    parsed = _absolute_magnitude_evidence(row)

    assert parsed["strict_value"] is None
    assert "DISTANCE_SOURCE_REQUIRED" in parsed["strict_reasons"]
    assert "EXTINCTION_SEMANTICS_REQUIRED" in parsed["strict_reasons"]
    assert "IS_STRICT" not in parsed["strict_reason"]


def test_innovation_does_not_trust_strict_marker_without_interval_bounds() -> None:
    row = _strict_absolute_row()
    absolute = row["absolute_magnitude"]
    assert isinstance(absolute, dict)
    absolute.pop("distance_lower_pc")
    absolute.pop("distance_upper_pc")

    parsed = _absolute_magnitude_evidence(row)

    assert parsed["is_strict"] is True
    assert parsed["distance_interval_status"] == "PROVIDED"
    assert parsed["strict_value"] is None
    assert "MODEL_DISTANCE_INTERVAL_MISSING" in parsed["strict_reasons"]


def test_innovation_explains_current_non_numeric_absolute_result_status() -> None:
    evidence = _photometric_evidence_section(
        {
            "source_photometry": [
                {
                    "source_id": "gsp-no-interval",
                    "m_cal": 14.5,
                    "status": "CALIBRATED",
                    "photometric_system": "Gaia Vega",
                    "photometric_band": "G",
                    "absolute_magnitude": {
                        "value": None,
                        "status": "MODEL_DISTANCE_INTERVAL_REQUIRED",
                        "distance_source": "Gaia DR3 GSP-Phot",
                        "distance_interval_status": "REQUIRED",
                        "distance_pc": 100.0,
                        "extinction_mag": 0.2,
                        "extinction_band": "G",
                        "extinction_system": "Gaia",
                    },
                }
            ]
        }
    )

    row = evidence["photometric_quality"]["sample_rows"][0]
    assert row["absolute_magnitude"] is None
    assert row["absolute_magnitude_gate_status"] == "NOT_AVAILABLE"
    assert "MODEL_DISTANCE_INTERVAL_MISSING" in row["absolute_magnitude_strict_reasons"]
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "PRESENT_BUT_INCOMPLETE"
    assert evidence["boundary"]["absolute_magnitude"]["strict_count"] == 0


def test_innovation_rejects_strict_absolute_result_when_quality_gate_fails() -> None:
    row = _strict_absolute_row()
    evidence = _photometric_evidence_section(
        {
            "photometric_quality_report": {
                "false_valid": True,
                "rows": [row],
            }
        }
    )

    assert evidence["status"] == "PRESENT_BUT_REJECTED"
    assert evidence["photometric_quality"]["absolute_magnitude_gate_status"] == "REJECTED"
    assert evidence["strict_absolute_magnitude"]["status"] == "NOT_AVAILABLE"
    assert evidence["strict_absolute_magnitude"]["count"] == 0
    assert evidence["boundary"]["absolute_magnitude"]["status"] == "REJECTED"
