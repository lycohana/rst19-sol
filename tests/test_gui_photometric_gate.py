from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from rst19 import gui


def _calibration(*, system: str = "Gaia Vega", band: str = "G") -> SimpleNamespace:
    return SimpleNamespace(
        status="VALID",
        photometric_system=system,
        photometric_band=band,
    )


def _auto_result(
    *,
    calibrated: bool = True,
    status: str = "CALIBRATED",
    provenance: str = "COMPLETE",
    reason: str = "WCS 与光度验收通过",
) -> SimpleNamespace:
    return SimpleNamespace(
        calibrated=calibrated,
        status=status,
        catalog_provenance_status=provenance,
        reason=reason,
    )


def _row(*, status: str = "CALIBRATED", band: str = "G") -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        photometric_system="Gaia Vega",
        photometric_band=band,
        calibrated_magnitude=12.7,
        calibrated_magnitude_error=0.08,
        photometric_outlier_reason="RESIDUAL_EXCEEDS_LIMIT" if status == "CATALOG_INCONSISTENT" else None,
        flags=("PHOTOMETRIC_OUTLIER",) if status == "CATALOG_INCONSISTENT" else (),
    )


def _absolute(
    status: str = "VALID",
    value: float | None = 4.2,
    *,
    is_strict: object = True,
    interval_status: str = "PROVIDED",
    distance_source: str | None = "parallax",
    distance_lower_pc: float | None = 90.0,
    distance_upper_pc: float | None = 110.0,
    system: str | None = "Gaia Vega",
    band: str | None = "G",
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        value=value,
        error=0.12,
        distance_pc=100.0,
        distance_lower_pc=distance_lower_pc,
        distance_upper_pc=distance_upper_pc,
        distance_source=distance_source,
        distance_interval_status=interval_status,
        is_strict=is_strict,
        extinction_system=system,
        extinction_band=band,
    )


def _strict_kwargs(*, row: object | None = None, band: str = "G") -> dict[str, object]:
    calibration = _calibration(band=band)
    return {
        "system": calibration.photometric_system,
        "band": band,
        "row": row if row is not None else _row(band=band),
        "calibration": calibration,
        "status": calibration.status,
        "wcs_available": True,
        "wcs_verified": True,
        "auto_result": _auto_result(),
        "strict_gate": True,
    }


def test_strict_absolute_display_requires_auto_source_strict_and_distance_gates() -> None:
    valid = gui._gui_format_absolute_magnitude(
        _absolute(),
        **_strict_kwargs(),
    )
    assert "M_G" in valid
    assert "4.200" in valid

    cases = (
        (
            _absolute(),
            _auto_result(
                calibrated=False,
                status="WCS_REFINEMENT_REJECTED",
                reason="仿射 WCS 验收失败",
            ),
            "WCS_REFINEMENT_REJECTED",
        ),
        (
            _absolute(),
            _auto_result(
                status="CALIBRATED_PROVISIONAL_REFERENCE",
                provenance="PROVENANCE_UNKNOWN",
                reason="星表完整性未确认",
            ),
            "星表完整性未确认",
        ),
        (
            _absolute(
                status="VALID_MODEL_DISTANCE_NO_INTERVAL",
                is_strict=False,
                interval_status="NOT_AVAILABLE",
                distance_source="Gaia DR3 GSP-Phot",
            ),
            _auto_result(),
            "缺少距离误差区间",
        ),
        (
            _absolute(interval_status="PROVIDED", distance_lower_pc=None, distance_upper_pc=None),
            _auto_result(),
            "缺少完整距离区间数值",
        ),
    )
    for absolute, auto_result, reason in cases:
        kwargs = _strict_kwargs()
        kwargs["auto_result"] = auto_result
        rendered = gui._gui_format_absolute_magnitude(absolute, **kwargs)
        assert "不可用" in rendered
        assert "4.2" not in rendered
        assert reason in rendered


def test_partial_catalog_coverage_does_not_hide_individually_valid_magnitudes() -> None:
    auto = _auto_result(status="CALIBRATED_PARTIAL")
    kwargs = _strict_kwargs()
    kwargs["auto_result"] = auto
    assert "M_G = 4.200" in gui._gui_format_absolute_magnitude(_absolute(), **kwargs)
    assert "m_G [Gaia Vega/G] = 12.700" in gui._gui_format_calibrated_magnitude(12.7, **kwargs)
    faintest = SimpleNamespace(
        instrumental_magnitude=-7.0, calibrated_magnitude=12.7,
        photometric_system="Gaia Vega", photometric_band="G", calibration_status="VALID",
        selection_scope="CALIBRATED_MATCHES_PARTIAL",
    )
    title, value, allowed = gui._gui_primary_faintest_display(
        faintest, _calibration(), row=_row(), auto_result=auto, wcs_verified=True, strict_gate=True,
    )
    assert allowed and value == "12.70"
    assert "已标定子集最暗" in title
    for rejected_row in (_row(status="INSTRUMENTAL_ONLY"), _row(status="CATALOG_INCONSISTENT")):
        kwargs["row"] = rejected_row
        assert "不可用" in gui._gui_format_calibrated_magnitude(12.7, **kwargs)
        assert "不可用" in gui._gui_format_absolute_magnitude(_absolute(), **kwargs)


def test_gaia_extinction_provenance_alias_is_compatible_with_gaia_vega_magnitude() -> None:
    rendered = gui._gui_format_absolute_magnitude(_absolute(system="Gaia"), **_strict_kwargs())
    assert rendered.startswith("M_G = 4.200")


def test_legacy_absolute_payload_without_is_strict_is_diagnostic_only() -> None:
    legacy_payload = {
        "status": "VALID",
        "value": 4.2,
        "error": 0.12,
        "distance_source": "parallax",
        "distance_interval_status": "PROVIDED",
        "extinction_system": "Gaia Vega",
        "extinction_band": "G",
    }

    rendered = gui._gui_format_absolute_magnitude(
        legacy_payload,
        **_strict_kwargs(),
    )

    assert "不可用" in rendered
    assert "4.2" not in rendered
    assert "is_strict" in rendered


@pytest.mark.parametrize("row_status", ["CATALOG_INCONSISTENT", "REJECTED_QUALITY"])
def test_row_status_cannot_be_overridden_by_global_valid_calibration(row_status: str) -> None:
    row = _row(status=row_status)
    kwargs = _strict_kwargs(row=row)

    calibrated = gui._gui_format_calibrated_magnitude(
        12.7,
        status="VALID",
        system="Gaia Vega",
        band="G",
        row=row,
        calibration=_calibration(),
        wcs_available=True,
        wcs_verified=True,
        auto_result=_auto_result(),
        strict_gate=True,
        compact=True,
        include_provenance=True,
    )
    absolute = gui._gui_format_absolute_magnitude(
        _absolute(),
        **kwargs,
        compact=True,
        include_provenance=True,
    )

    assert "不可用" in calibrated
    assert row_status in calibrated
    assert "12.7" not in calibrated
    assert "不可用" in absolute
    assert row_status in absolute
    assert "4.2" not in absolute


def test_compact_photometry_keeps_dynamic_band_system_and_model_distance_source() -> None:
    row = _row(band="V")
    calibration = _calibration(system="Johnson", band="V")
    auto_result = _auto_result()
    calibrated = gui._gui_format_calibrated_magnitude(
        12.7,
        status="VALID",
        system="Johnson",
        band="V",
        row=row,
        calibration=calibration,
        wcs_available=True,
        wcs_verified=True,
        auto_result=auto_result,
        strict_gate=True,
        compact=True,
        include_provenance=True,
    )
    model_absolute = gui._gui_format_absolute_magnitude(
        _absolute(
            "VALID_MODEL_DISTANCE",
            distance_source="Gaia DR3 GSP-Phot",
            system="Johnson",
            band="V",
        ),
        system="Johnson",
        band="V",
        row=row,
        calibration=calibration,
        status="VALID",
        wcs_available=True,
        wcs_verified=True,
        auto_result=auto_result,
        strict_gate=True,
        compact=True,
        include_provenance=True,
    )

    assert calibrated == "m_V = 12.70 · Johnson/V"
    assert model_absolute == "M_V = 4.20 · Johnson/V · 模型距离：Gaia DR3 GSP-Phot"


def test_secondary_headline_uses_the_declared_band_and_not_a_hardcoded_g() -> None:
    source = Path(gui.__file__).read_text(encoding="utf-8")
    assert "m_G" not in source

    faintest = SimpleNamespace(
        instrumental_magnitude=19.1,
        calibrated_magnitude=12.7,
        calibration_status="VALID",
        photometric_system="Johnson",
        photometric_band="V",
    )
    title, value, calibrated = gui._gui_primary_faintest_display(
        faintest,
        _calibration(system="Johnson", band="V"),
        wcs_verified=True,
    )

    assert title == "m_V,cal · 已标定"
    assert value == "12.70"
    assert calibrated is True
