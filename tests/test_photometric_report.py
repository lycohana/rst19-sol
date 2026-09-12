from __future__ import annotations

import json

import pytest

from rst19.photometric_report import (
    OBSERVABILITY_LEVELS,
    audit_photometry,
    build_photometric_report,
)
from rst19.photometric_report_cli import _photometry_input
from rst19.photometry import absolute_magnitude_estimate_from_distance


def _frame(frame_id: str, sources: list[dict[str, object]], *, wcs: object = None) -> dict[str, object]:
    row: dict[str, object] = {"frame_id": frame_id, "sources": sources}
    if wcs is not None:
        row["wcs"] = wcs
    return row


def test_observability_levels_and_counts_are_deterministic() -> None:
    report = build_photometric_report(
        frame_rows=[
            _frame(
                "f1",
                [
                    {"source_id": "inst", "m_inst": 12.0, "status": "INSTRUMENTAL_ONLY"},
                    {
                        "source_id": "relative",
                        "m_inst": 12.0,
                        "m_cal": 11.5,
                        "status": "RELATIVE_CALIBRATED",
                        "calibration_scope": "relative",
                    },
                    {
                        "source_id": "apparent",
                        "m_inst": 12.0,
                        "m_cal": 11.4,
                        "status": "APPARENT_CALIBRATED",
                        "catalog_name": "offline-catalog",
                        "photometric_system": "Gaia Vega",
                        "photometric_band": "G",
                    },
                    {
                        "source_id": "absolute",
                        "m_inst": 12.0,
                        "m_cal": 11.3,
                        "M": 3.2,
                        "status": "ABSOLUTE_ELIGIBLE",
                        "absolute_status": "VALID",
                        "catalog_name": "offline-catalog",
                        "photometric_system": "Gaia Vega",
                        "photometric_band": "G",
                        "parallax_mas": 10.0,
                        "extinction_mag": 0.2,
                        "extinction_band": "G",
                        "extinction_system": "Gaia",
                        "extinction_source": "Gaia DR3 GSP-Phot: ag_gspphot",
                    },
                ],
                wcs={"status": "VALID"},
            )
        ]
    )

    assert report.source_count == 4
    assert report.frame_count == 1
    assert report.level_counts == {
        "INSTRUMENTAL_ONLY": 1,
        "RELATIVE_CALIBRATED": 1,
        "APPARENT_CALIBRATED": 1,
        "ABSOLUTE_ELIGIBLE": 1,
        "GEOMETRY_UNAVAILABLE": 0,
    }
    assert tuple(report.level_counts) == OBSERVABILITY_LEVELS
    assert report.rows[0].observability_level == "INSTRUMENTAL_ONLY"
    assert report.rows[-1].observability_level == "ABSOLUTE_ELIGIBLE"
    assert report.to_json() == report.to_json()


def test_missing_wcs_catalog_parallax_and_extinction_degrade_separately() -> None:
    missing_wcs = build_photometric_report(
        frame_rows=[_frame("no-wcs", [{"source_id": "s0", "m_inst": 12.0}])]
    )
    assert missing_wcs.rows[0].observability_level == "GEOMETRY_UNAVAILABLE"
    assert missing_wcs.missing_inputs["wcs"] == 1

    missing_catalog = build_photometric_report(
        frame_rows=[
            _frame(
                "catalog-missing",
                [{"source_id": "s1", "m_inst": 12.0}],
                wcs={"status": "VALID"},
            )
        ]
    )
    assert missing_catalog.rows[0].observability_level == "INSTRUMENTAL_ONLY"
    assert missing_catalog.missing_inputs["catalog"] == 1

    missing_parallax = build_photometric_report(
        frame_rows=[
            _frame(
                "parallax-missing",
                [
                    {
                        "source_id": "s2",
                        "m_inst": 12.0,
                        "m_cal": 11.0,
                        "M": 3.0,
                        "status": "APPARENT_CALIBRATED",
                        "calibration_status": "VALID",
                        "catalog_name": "catalog",
                        "photometric_system": "G",
                        "photometric_band": "G",
                        "extinction_mag": 0.1,
                        "absolute_status": "NO_PARALLAX",
                    }
                ],
                wcs={"status": "VALID"},
            )
        ]
    )
    assert missing_parallax.rows[0].observability_level == "APPARENT_CALIBRATED"
    assert missing_parallax.missing_inputs["parallax"] == 1

    missing_extinction = build_photometric_report(
        source_rows=[
            {
                "source_id": "s3",
                "m_inst": 12.0,
                "m_cal": 11.0,
                "M": 3.0,
                "status": "APPARENT_CALIBRATED",
                "calibration_status": "VALID",
                "catalog_name": "catalog",
                "photometric_system": "G",
                "photometric_band": "G",
                "parallax_mas": 10.0,
                "absolute_status": "NO_EXTINCTION",
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )
    assert missing_extinction.rows[0].observability_level == "APPARENT_CALIBRATED"
    assert missing_extinction.missing_inputs["extinction"] == 1


def test_false_valid_gate_flags_calibrated_and_absolute_values_with_bad_status() -> None:
    report = audit_photometry(
        source_rows=[
                {
                    "source_id": "bad-cal",
                    "m_inst": 12.0,
                    "m_cal": 11.0,
                    "status": "INSTRUMENTAL_ONLY",
                    "wcs_available": True,
                    "unknown_future_field": {"ignored": True},
                },
            {
                "source_id": "bad-abs",
                "m_inst": 12.0,
                "m_cal": 11.0,
                "M": 3.0,
                "status": "APPARENT_CALIBRATED",
                "calibration_status": "VALID",
                "catalog_name": "catalog",
                "photometric_system": "G",
                "photometric_band": "G",
                "parallax_mas": 10.0,
                "extinction_mag": 0.1,
                "absolute_status": "REJECTED_QUALITY",
                "wcs_available": True,
            },
        ],
        require_wcs=True,
    )

    assert report.false_valid is True
    assert report.false_valid_gate.violation_count == 2
    assert "M_CAL_STATUS_INVALID" in report.false_valid_gate.flags
    assert "M_STATUS_INVALID" in report.false_valid_gate.flags
    assert report.rows[0].observability_level == "INSTRUMENTAL_ONLY"
    assert report.rows[1].observability_level == "APPARENT_CALIBRATED"
    assert "FALSE_VALID_M_CAL" in report.rows[0].flags
    assert "FALSE_VALID_M" in report.rows[1].flags
    with pytest.raises(ValueError, match="false-valid photometry gate"):
        build_photometric_report(
            source_rows=[{"m_inst": 1.0, "m_cal": 2.0, "status": "INSTRUMENTAL_ONLY"}],
            strict=True,
        )


def test_catalog_inconsistent_m_cal_is_retained_as_diagnostic_only() -> None:
    report = audit_photometry(
        source_rows=[
            {
                "source_id": "catalog-outlier",
                "m_inst": 12.0,
                "m_cal": 11.0,
                "status": "CATALOG_INCONSISTENT",
                "calibration_status": "VALID",
                "catalog_name": "Gaia DR3",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "photometric_consistent": False,
                "photometric_outlier_reason": "RESIDUAL_EXCEEDS_LIMIT",
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    row = report.rows[0]
    assert row.m_cal == pytest.approx(11.0)
    assert row.calibrated_magnitude_value_role == "DIAGNOSTIC_ONLY"
    assert row.observability_level == "INSTRUMENTAL_ONLY"
    assert report.false_valid is False
    assert report.false_valid_gate.violation_count == 0
    assert "M_CAL_DIAGNOSTIC_ONLY" in row.flags
    assert "FALSE_VALID_M_CAL" not in row.flags
    assert "M_CAL_DIAGNOSTIC_ONLY" in row.rejection_reasons


def test_null_unavailable_layers_are_not_reported_as_nonfinite_values() -> None:
    report = audit_photometry(
        source_rows=[
            {
                "source_id": "unavailable-layers",
                "m_inst": 12.0,
                "m_cal": None,
                "M": None,
                "absolute_magnitude": {"value": None, "status": "NO_PARALLAX"},
                "status": "INSTRUMENTAL_ONLY",
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    row = report.rows[0]
    assert row.m_cal is None
    assert row.absolute_magnitude is None
    assert row.observability_level == "INSTRUMENTAL_ONLY"
    assert report.false_valid is False
    assert report.false_valid_gate.violation_count == 0
    assert "NONFINITE_M_CAL" not in row.flags
    assert "NONFINITE_M" not in row.flags


def test_empty_and_unknown_rows_are_safe_and_json_serializable() -> None:
    empty = build_photometric_report()
    assert empty.source_count == 0
    assert empty.frame_count == 0
    assert empty.flags == ("NO_DATA",)
    json.loads(empty.to_json())

    unknown = build_photometric_report(source_rows=[{"future_field": "kept out"}, "not-a-row"])
    assert unknown.source_count == 1
    assert unknown.ignored_row_count == 1
    assert unknown.rows[0].observability_level == "GEOMETRY_UNAVAILABLE"
    payload = unknown.as_dict()
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_error_budget_and_provenance_are_aggregated_without_unknown_fields() -> None:
    report = build_photometric_report(
        frame_rows=[
            _frame(
                "f1",
                [
                    {
                        "source_id": "s1",
                        "m_inst": 12.0,
                        "m_inst_error": 0.10,
                        "m_cal": 11.0,
                        "m_cal_error": 0.20,
                        "status": "APPARENT_CALIBRATED",
                        "catalog_name": "offline-catalog",
                        "photometric_system": "Gaia Vega",
                        "photometric_band": "G",
                        "provenance": {"operator": "ignored-by-summary", "source_file": "frame.fits"},
                        "error_budget": {"background": {"sigma": 0.30}, "not-a-number": "bad"},
                    }
                ],
                wcs={"status": "VALID", "wcs_source": "offline-solution"},
            )
        ],
        calibration={"status": "VALID", "fit_rms_mag": 0.04, "validation_rms_mag": 0.05},
    )

    assert report.error_budget["m_inst"]["count"] == 1
    assert report.error_budget["m_cal"]["max"] == pytest.approx(0.20)
    assert report.error_budget["background"]["mean"] == pytest.approx(0.30)
    assert report.error_budget["calibration_fit"]["mean"] == pytest.approx(0.04)
    assert report.provenance["declared"]["catalog"] == ["offline-catalog"]
    assert report.provenance["declared"]["wcs"] == ["offline-solution"]
    assert "operator" not in report.provenance["declared"]


def test_dict_inputs_and_indexed_absolute_results_are_supported() -> None:
    report = build_photometric_report(
        frames={
            "f1": [
                {
                    "source_id": "s1",
                    "m_inst": 12.0,
                    "m_cal": 11.0,
                    "M": 3.0,
                    "status": "ABSOLUTE_ELIGIBLE",
                    "catalog": "offline-catalog",
                    "photometric_band": "G",
                }
            ]
        },
        absolute_results={
            "s1": {
                "value": 3.0,
                "status": "VALID",
                    "parallax_mas": 10.0,
                    "extinction_mag": 0.0,
                    "extinction_band": "G",
                    "extinction_system": "Gaia",
                    "extinction_source": "test-catalog",
            }
        },
        require_wcs=False,
    )

    assert report.frame_count == 1
    assert report.source_count == 1
    assert report.rows[0].observability_level == "ABSOLUTE_ELIGIBLE"
    assert report.rows[0].absolute_status == "VALID"


def test_positive_distance_without_source_is_not_strict_absolute() -> None:
    report = build_photometric_report(
        source_rows=[
            {
                "source_id": "unprovenanced-distance",
                "m_inst": 12.0,
                "m_cal": 11.0,
                "M": 3.0,
                "status": "ABSOLUTE_ELIGIBLE",
                "calibration_status": "VALID",
                "catalog_name": "Gaia DR3",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "distance_pc": 100.0,
                "extinction_mag": 0.2,
                "extinction_band": "G",
                "extinction_system": "Gaia",
                "extinction_source": "Gaia DR3 GSP-Phot: ag_gspphot",
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    row = report.rows[0]
    assert row.observability_level == "APPARENT_CALIBRATED"
    assert "DISTANCE_SOURCE_REQUIRED" in row.flags
    assert "distance_source" in row.missing_inputs
    assert report.false_valid is True
    assert "DISTANCE_SOURCE_REQUIRED" in report.false_valid_gate.flags


def test_model_distance_without_interval_is_retained_but_not_strict_absolute() -> None:
    report = build_photometric_report(
        source_rows=[
            {
                "source_id": "gsp-no-interval",
                "m_inst": 12.0,
                "m_cal": 11.0,
                "status": "CALIBRATED",
                "calibration_status": "VALID",
                "catalog_name": "Gaia DR3",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "absolute_magnitude": {
                    "value": 3.0,
                    "status": "VALID_MODEL_DISTANCE_NO_INTERVAL",
                    "distance_source": "Gaia DR3 GSP-Phot",
                    "distance_pc": 100.0,
                    "extinction_mag": 0.2,
                    "extinction_band": "G",
                    "extinction_system": "Gaia",
                    "extinction_source": "Gaia DR3 GSP-Phot: ag_gspphot",
                },
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    row = report.rows[0]
    assert row.observability_level == "APPARENT_CALIBRATED"
    assert row.absolute_status == "VALID_MODEL_DISTANCE_NO_INTERVAL"
    assert row.absolute_magnitude == pytest.approx(3.0)
    assert "MODEL_DISTANCE_INTERVAL_MISSING" in row.flags
    assert "distance_interval" in row.missing_inputs
    assert report.false_valid is False
    assert row.strict_absolute_magnitude is None
    assert row.absolute_magnitude_is_strict is False
    assert row.absolute_magnitude_value_role == "DIAGNOSTIC_ONLY"
    assert row.absolute_magnitude_band == "G"
    assert row.absolute_magnitude_system == "Gaia"
    assert row.absolute_magnitude_source == "Gaia DR3 GSP-Phot"


@pytest.mark.parametrize(
    ("source_id", "absolute_magnitude"),
    [
        (
            "parallax-path",
            {
                "value": 3.0,
                "status": "VALID",
                "distance_source": "parallax",
                "distance_pc": 100.0,
                "parallax_mas": 10.0,
                "extinction_mag": 0.2,
                "extinction_band": "G",
                "extinction_system": "Gaia",
                "extinction_source": "test-catalog",
            },
        ),
        (
            "gsp-phot-path",
            {
                "value": 3.0,
                "status": "VALID_MODEL_DISTANCE",
                "distance_source": "Gaia DR3 GSP-Phot",
                "distance_pc": 100.0,
                "distance_lower_pc": 95.0,
                "distance_upper_pc": 106.0,
                "extinction_mag": 0.2,
                "extinction_band": "G",
                "extinction_system": "Gaia",
                "extinction_source": "Gaia DR3 GSP-Phot: ag_gspphot",
            },
        ),
    ],
)
def test_consistent_parallax_and_gspphot_paths_can_be_strict_absolute(
    source_id: str,
    absolute_magnitude: dict[str, object],
) -> None:
    report = build_photometric_report(
        source_rows=[
            {
                "source_id": source_id,
                "m_inst": 12.0,
                "m_cal": 11.0,
                "status": "CALIBRATED",
                "calibration_status": "VALID",
                "catalog_name": "Gaia DR3",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "absolute_magnitude": absolute_magnitude,
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    assert report.rows[0].observability_level == "ABSOLUTE_ELIGIBLE"
    assert report.rows[0].absolute_status in {"VALID", "VALID_MODEL_DISTANCE"}
    assert report.false_valid is False
    assert report.rows[0].strict_absolute_magnitude == pytest.approx(3.0)
    assert report.rows[0].absolute_magnitude_is_strict is True
    assert report.rows[0].absolute_magnitude_value_role == "STRICT"
    assert report.rows[0].absolute_magnitude_band == "G"
    assert report.rows[0].absolute_magnitude_system == "Gaia"


def test_report_prefers_absolute_estimate_as_dict_over_dataclass_fields() -> None:
    estimate = absolute_magnitude_estimate_from_distance(
        13.0,
        distance_pc=100.0,
        distance_lower_pc=95.0,
        distance_upper_pc=106.0,
        distance_source="Gaia DR3 GSP-Phot",
        extinction_mag=0.2,
        extinction_band="G",
        extinction_system="Gaia",
        extinction_source="Gaia DR3 GSP-Phot: ag_gspphot",
    )
    assert estimate.is_strict is True

    report = build_photometric_report(
        source_rows=[
            {
                "source_id": "estimate-object",
                "m_inst": 12.0,
                "m_cal": 13.0,
                "status": "CALIBRATED",
                "calibration_status": "VALID",
                "catalog_name": "Gaia DR3",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "absolute_magnitude": estimate,
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    row = report.rows[0]
    assert row.absolute_magnitude == pytest.approx(estimate.value)
    assert row.strict_absolute_magnitude == pytest.approx(estimate.value)
    assert row.absolute_magnitude_is_strict is True
    assert row.absolute_magnitude_distance_interval_status == "PROVIDED"
    assert row.absolute_magnitude_distance_lower_pc == pytest.approx(95.0)
    assert row.absolute_magnitude_distance_upper_pc == pytest.approx(106.0)
    assert row.absolute_magnitude_extinction_band == "G"
    assert row.absolute_magnitude_extinction_system == "Gaia"
    assert row.absolute_magnitude_extinction_source == "Gaia DR3 GSP-Phot: ag_gspphot"


def test_model_distance_absolute_status_is_reported_as_eligible_with_provenance() -> None:
    report = build_photometric_report(
        source_rows=[
            {
                "source_id": "gsp-1",
                "m_inst": 12.0,
                "m_cal": 11.0,
                "M": 3.0,
                "status": "CALIBRATED",
                "calibration_status": "VALID",
                "catalog_name": "Gaia DR3",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "distance_source": "Gaia DR3 GSP-Phot",
                "distance_pc": 100.0,
                "extinction_mag": 0.2,
                "extinction_band": "G",
                "extinction_system": "Gaia",
                "extinction_source": "Gaia DR3 GSP-Phot: ag_gspphot",
                "absolute_magnitude": {
                    "value": 3.0,
                    "status": "VALID_MODEL_DISTANCE",
                    "distance_source": "Gaia DR3 GSP-Phot",
                    "distance_lower_pc": 95.0,
                    "distance_upper_pc": 106.0,
                    "extinction_band": "G",
                    "extinction_system": "Gaia",
                    "extinction_source": "Gaia DR3 GSP-Phot: ag_gspphot",
                },
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    assert report.rows[0].observability_level == "ABSOLUTE_ELIGIBLE"
    assert report.rows[0].absolute_status == "VALID_MODEL_DISTANCE"
    assert report.false_valid is False


def test_explicit_non_strict_absolute_value_is_retained_as_diagnostic_only() -> None:
    report = build_photometric_report(
        source_rows=[
            {
                "source_id": "old-cache",
                "m_inst": 12.0,
                "m_cal": 11.0,
                "status": "CALIBRATED",
                "calibration_status": "VALID",
                "catalog_name": "Gaia DR3",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "absolute_magnitude": {
                    "value": 3.0,
                    "status": "VALID",
                    "is_strict": False,
                    "distance_source": "parallax",
                    "parallax_mas": 10.0,
                    "distance_pc": 100.0,
                    "extinction_mag": 0.2,
                    "extinction_band": "G",
                    "extinction_system": "Gaia",
                    "extinction_source": "test-catalog",
                },
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    row = report.rows[0]
    assert row.absolute_magnitude == pytest.approx(3.0)
    assert row.strict_absolute_magnitude is None
    assert row.absolute_magnitude_is_strict is False
    assert row.absolute_magnitude_value_role == "DIAGNOSTIC_ONLY"
    assert "ABSOLUTE_NOT_STRICT" in row.flags
    assert "strict_absolute_result" in row.missing_inputs
    assert row.observability_level == "APPARENT_CALIBRATED"


def test_absolute_alias_keeps_v_band_identity_in_report() -> None:
    report = build_photometric_report(
        source_rows=[
            {
                "source_id": "v-source",
                "m_inst": 12.0,
                "m_cal": 11.0,
                "status": "CALIBRATED",
                "calibration_status": "VALID",
                "catalog_name": "Johnson standards",
                "photometric_system": "Johnson",
                "photometric_band": "V",
                "M_V": 4.2,
                "absolute_status": "VALID",
                "parallax_mas": 10.0,
                "extinction_mag": 0.1,
                "extinction_band": "V",
                "extinction_system": "Johnson",
                "extinction_source": "standard-field",
                "wcs_available": True,
            }
        ],
        require_wcs=True,
    )

    row = report.rows[0]
    assert row.absolute_magnitude == pytest.approx(4.2)
    assert row.absolute_magnitude_band == "V"
    assert row.absolute_magnitude_system == "Johnson"
    assert row.strict_absolute_magnitude == pytest.approx(4.2)
    assert row.as_dict()["strict_M"] == pytest.approx(4.2)


@pytest.mark.parametrize(
    ("fields", "expected_flag"),
    [
        ({}, "EXTINCTION_SEMANTICS_REQUIRED"),
        (
            {"extinction_band": "V", "extinction_system": "Johnson", "extinction_source": "test"},
            "EXTINCTION_BAND_MISMATCH",
        ),
        (
            {"extinction_band": "G", "extinction_system": "Gaia"},
            "EXTINCTION_SOURCE_REQUIRED",
        ),
    ],
)
def test_strict_absolute_report_requires_compatible_extinction_provenance(
    fields: dict[str, object],
    expected_flag: str,
) -> None:
    row = {
        "source_id": "unsafe-extinction",
        "m_inst": 12.0,
        "m_cal": 11.0,
        "M": 3.0,
        "status": "CALIBRATED",
        "calibration_status": "VALID",
        "catalog_name": "Gaia DR3",
        "photometric_system": "Gaia Vega",
        "photometric_band": "G",
        "parallax_mas": 10.0,
        "extinction_mag": 0.2,
        "absolute_status": "VALID",
        "wcs_available": True,
    }
    row.update(fields)

    report = build_photometric_report(source_rows=[row], require_wcs=True)

    assert report.rows[0].observability_level == "APPARENT_CALIBRATED"
    assert expected_flag in report.rows[0].flags
    assert report.false_valid is True


def test_report_cli_unwraps_automatic_photometry_envelope() -> None:
    payload = {
        "status": "CALIBRATED_PARTIAL",
        "calibrated": True,
        "frame_path": "frame.fits",
        "catalog_path": "gaia.csv",
        "affine_wcs": {"rms_residual_px": 0.5},
        "analysis": {
            "path": "frame.fits",
            "detection": {"sources": []},
            "photometric_calibration": {
                "status": "VALID",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
            },
            "source_photometry": [
                {
                    "detection_id": 7,
                    "source_id": "gaia-7",
                    "m_inst": -7.5,
                    "calibrated_magnitude": 12.4,
                    "catalog_magnitude": 12.5,
                    "photometric_system": "Gaia Vega",
                    "photometric_band": "G",
                    "status": "CALIBRATED",
                }
            ],
        },
    }

    report_input, calibration, raw_only = _photometry_input(payload)
    report = build_photometric_report(
        report_input,
        calibration=calibration,
        require_wcs=True,
    )

    assert raw_only is False
    assert report.source_count == 1
    assert report.rows[0].source_id == "gaia-7"
    assert report.rows[0].catalog_available is True
    assert report.rows[0].observability_level == "APPARENT_CALIBRATED"
