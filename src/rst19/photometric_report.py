"""Deterministic, standalone audit reports for photometric observability.

The existing photometry pipeline deliberately keeps instrumental magnitude
(``m_inst``), calibrated apparent magnitude (``m_cal``) and absolute
magnitude (``M``) separate.  This module is the reporting boundary for those
values.  It accepts plain mappings/sequences so it can audit cached JSON,
frame summaries, or rows produced by another process without importing the
GUI or changing the active pipeline.

The report is conservative by design:

* a value is never promoted merely because it is numerically present;
* a calibrated value needs a legal status;
* an absolute value also needs catalogue, a qualified parallax or explicitly
  sourced distance estimate, extinction and geometry evidence;
* missing inputs are retained as structured evidence instead of being
  replaced with zeroes or empty strings;
* unknown input fields are ignored, while unknown status claims are flagged.

``build_photometric_report`` is the main entry point.  ``make_photometric_report``
and ``audit_photometry`` are intentionally small aliases for callers that use
different naming conventions.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence


class ObservabilityLevel(str, Enum):
    """Stable names for the mutually exclusive report levels."""

    INSTRUMENTAL_ONLY = "INSTRUMENTAL_ONLY"
    RELATIVE_CALIBRATED = "RELATIVE_CALIBRATED"
    APPARENT_CALIBRATED = "APPARENT_CALIBRATED"
    ABSOLUTE_ELIGIBLE = "ABSOLUTE_ELIGIBLE"
    GEOMETRY_UNAVAILABLE = "GEOMETRY_UNAVAILABLE"


OBSERVABILITY_LEVELS: tuple[str, ...] = (
    "INSTRUMENTAL_ONLY",
    "RELATIVE_CALIBRATED",
    "APPARENT_CALIBRATED",
    "ABSOLUTE_ELIGIBLE",
    "GEOMETRY_UNAVAILABLE",
)

_LEVEL_RANK = {
    "INSTRUMENTAL_ONLY": 1,
    "RELATIVE_CALIBRATED": 2,
    "APPARENT_CALIBRATED": 3,
    "ABSOLUTE_ELIGIBLE": 4,
}

_MISSING = object()

_CALIBRATION_VALID_STATUSES = {
    "VALID",
    "VALID_NO_HOLDOUT",
    "CALIBRATED",
    "CALIBRATION_VALID",
    "RELATIVE_CALIBRATED",
    "APPARENT_CALIBRATED",
    "ABSOLUTE_ELIGIBLE",
}

_ABSOLUTE_VALID_STATUSES = {
    "VALID",
    "VALID_NO_HOLDOUT",
    "VALID_MODEL_DISTANCE",
    "VALID_MODEL_DISTANCE_NO_INTERVAL",
    "ABSOLUTE_VALID",
    "ABSOLUTE_ELIGIBLE",
    "CALIBRATED_ABSOLUTE",
}

_INVALID_STATUS_WORDS = {
    "INVALID",
    "REJECTED",
    "REJECT",
    "MISSING",
    "NONE",
    "UNKNOWN",
    "UNAVAILABLE",
    "NO_CALIBRATION",
    "CALIBRATION_INVALID",
    "CALIBRATION_MISSING_COLOR",
    "CALIBRATION_NOT_APPLIED",
    "LOW_PARALLAX_SNR",
    "NO_PARALLAX",
    "NO_EXTINCTION",
    "GEOMETRY_UNAVAILABLE",
    "NO_WCS",
}

_ROW_COLLECTION_KEYS = (
    "sources",
    "source_rows",
    "source_photometry",
    "source_results",
    "photometry",
    "photometry_rows",
    "photometry_results",
    "rows",
    "items",
)
_FRAME_COLLECTION_KEYS = ("frames", "frame_rows", "frame_results")
_RESULT_COLLECTION_KEYS = (
    "results",
    "rows",
    "items",
    "calibrations",
    "calibration_results",
    "absolute_results",
    "absolute_magnitudes",
)

_KNOWN_ROW_KEYS = {
    "frame_id",
    "frame",
    "source_id",
    "detection_id",
    "id",
    "m_inst",
    "instrumental_magnitude",
    "m_cal",
    "calibrated_magnitude",
    "apparent_magnitude",
    "m",
    "M",
    "absolute_magnitude",
    "status",
    "observability_level",
    "level",
    "wcs",
    "wcs_available",
    "has_wcs",
    "geometry",
    "geometry_status",
    "catalog",
    "catalog_name",
    "source_id",
    "parallax_mas",
    "extinction_mag",
    "calibration",
    "calibration_result",
    "absolute",
    "absolute_result",
    "absolute_magnitude",
    "error_budget",
    "provenance",
}

_PROVENANCE_ALIASES = {
    "catalog": ("catalog", "catalog_name", "catalog_id", "catalog_version"),
    "photometric_system": ("photometric_system", "system"),
    "photometric_band": ("photometric_band", "band"),
    "calibration": ("calibration_id", "calibration_version", "calibration_source"),
    "wcs": ("wcs_source", "wcs_version", "wcs_id"),
    "absolute": ("absolute_method", "distance_source", "absolute_source"),
    "input": ("source", "source_file", "input_file", "data_source"),
}

_ERROR_ALIASES = {
    "m_inst": ("m_inst_error", "instrumental_magnitude_error", "instrumental_error", "sigma_m_inst"),
    "m_cal": ("m_cal_error", "calibrated_magnitude_error", "apparent_magnitude_error", "sigma_m_cal"),
    "M": ("M_error", "absolute_magnitude_error", "sigma_M", "sigma_absolute_magnitude"),
    "zero_point": ("zero_point_error", "zp_error", "zero_point_sigma"),
    "color_term": ("color_term_error", "color_coefficient_error", "color_sigma"),
    "calibration_fit": ("fit_rms_mag", "fit_error_mag", "calibration_fit_error"),
    "calibration_validation": ("validation_rms_mag", "validation_error_mag"),
    "calibration_rms": ("rms", "residual_rms", "training_residual_rms"),
    "calibration_mad": ("mad", "residual_mad", "training_residual_mad"),
    "parallax": ("parallax_error_mas", "sigma_parallax_mas"),
    "extinction": ("extinction_error_mag", "sigma_extinction_mag"),
    "wcs": ("wcs_error_arcsec", "wcs_residual_arcsec", "rms_residual_px", "wcs_rms_px"),
}


def _norm_key(value: object) -> str:
    """Return a forgiving key form used only for input lookup."""

    return "".join(char if char.isalnum() else "_" for char in str(value).strip().lower()).strip("_")


def _normalised_mapping(value: Mapping[object, object]) -> dict[str, object]:
    return {_norm_key(key): item for key, item in value.items()}


def _as_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        converted = as_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return dict(attributes)
    return None


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _lookup(mapping: Mapping[str, object] | None, aliases: Iterable[str]) -> tuple[bool, object]:
    if mapping is None:
        return False, _MISSING
    values = _normalised_mapping(mapping)
    for alias in aliases:
        key = _norm_key(alias)
        if key in values:
            return True, values[key]
    return False, _MISSING


def _value(mapping: Mapping[str, object] | None, aliases: Iterable[str], default: object = None) -> object:
    found, item = _lookup(mapping, aliases)
    return item if found else default


def _nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return True


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool) or isinstance(value, Mapping) or _is_sequence(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_scalar(value: object) -> str | int | float | bool | None:
    """Keep provenance scalars JSON-safe and drop non-finite numbers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _present_number(mapping: Mapping[str, object] | None, aliases: Iterable[str]) -> tuple[bool, float | None]:
    found, value = _lookup(mapping, aliases)
    return found and value is not None and value != "", _number(value)


def _bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = _norm_status(value)
        if normalized in {"TRUE", "YES", "Y", "AVAILABLE", "VALID", "OK", "PRESENT"}:
            return True
        if normalized in {"FALSE", "NO", "N", "MISSING", "UNAVAILABLE", "INVALID", "ABSENT"}:
            return False
    return None


def _norm_status(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _status(mapping: Mapping[str, object] | None, aliases: Iterable[str] = ("status",)) -> str:
    value = _value(mapping, aliases, default=None)
    return _norm_status(value) if _nonempty(value) else ""


def _level(value: object) -> str | None:
    normalized = _norm_status(value)
    return normalized if normalized in OBSERVABILITY_LEVELS else None


def _looks_like_row(value: Mapping[object, object]) -> bool:
    keys = {_norm_key(key) for key in value}
    known = {_norm_key(key) for key in _KNOWN_ROW_KEYS}
    if keys.intersection(known):
        return True
    return any(_norm_key(key) in keys for key in _ROW_COLLECTION_KEYS)


def _row_list(value: object, *, id_aliases: Iterable[str] = ()) -> tuple[list[dict[str, object]], int]:
    """Coerce a dict/list input to rows and count malformed entries.

    A mapping with row-like keys is one row.  A mapping without row-like keys
    is treated as an ID-to-row mapping, sorted by the string form of its key.
    This makes dictionary input deterministic without rejecting harmless
    wrapper/unknown fields.
    """

    if value is None:
        return [], 0
    mapping = _as_mapping(value)
    if mapping is not None:
        for alias in id_aliases:
            found, nested = _lookup(mapping, (alias,))
            if found and _is_sequence(nested):
                return _row_list(nested, id_aliases=id_aliases)
        if _looks_like_row(mapping):
            return [mapping], 0
        for alias in _ROW_COLLECTION_KEYS + _RESULT_COLLECTION_KEYS:
            found, nested = _lookup(mapping, (alias,))
            if found and (_is_sequence(nested) or isinstance(nested, Mapping)):
                return _row_list(nested, id_aliases=id_aliases)
        rows: list[dict[str, object]] = []
        malformed = 0
        for key in sorted(mapping, key=lambda item: str(item)):
            raw_item = mapping[key]
            item = _as_mapping(raw_item)
            if item is None and _is_sequence(raw_item) and "frame_id" in {_norm_key(alias) for alias in id_aliases}:
                item = {"frame_id": key, "sources": raw_item}
            if item is None:
                malformed += 1
                continue
            if not _lookup(item, id_aliases)[0] and id_aliases:
                item = {id_aliases[0]: key, **item}
            rows.append(item)
        return rows, malformed
    if _is_sequence(value):
        rows = []
        malformed = 0
        for item in value:
            item_mapping = _as_mapping(item)
            if item_mapping is None:
                malformed += 1
            else:
                rows.append(item_mapping)
        return rows, malformed
    return [], 1


def _collection(mapping: Mapping[str, object]) -> tuple[bool, object]:
    for alias in _ROW_COLLECTION_KEYS:
        found, value = _lookup(mapping, (alias,))
        if found:
            return True, value
    return False, None


def _geometry_state(mapping: Mapping[str, object] | None, *, default: str = "unknown") -> str:
    """Return ``available``, ``unavailable`` or ``unknown`` for WCS geometry."""

    if mapping is None:
        return default

    for aliases in (("wcs_available", "has_wcs", "geometry_available"),):
        found, value = _lookup(mapping, aliases)
        if found:
            parsed = _bool_value(value)
            if parsed is not None:
                return "available" if parsed else "unavailable"

    found, value = _lookup(mapping, ("geometry_status", "wcs_status"))
    if found:
        status = _norm_status(value)
        if status in {"VALID", "AVAILABLE", "SOLVED", "OK", "CALIBRATED", "READY"}:
            return "available"
        if status:
            return "unavailable"

    found, value = _lookup(mapping, ("wcs", "wcs_solution", "world_coordinate_system"))
    if found:
        if isinstance(value, Mapping):
            nested_status = _status(value)
            if nested_status in {"VALID", "AVAILABLE", "SOLVED", "OK", "CALIBRATED", "READY"}:
                return "available"
            if nested_status:
                return "unavailable"
            return "available" if value else "unavailable"
        parsed = _bool_value(value)
        if parsed is not None:
            return "available" if parsed else "unavailable"
        return "available" if _nonempty(value) else "unavailable"

    found, value = _lookup(mapping, ("geometry", "sky_geometry"))
    if found:
        if isinstance(value, Mapping):
            return "available" if value else "unavailable"
        parsed = _bool_value(value)
        if parsed is not None:
            return "available" if parsed else "unavailable"

    return default


def _merge_context(
    frame: Mapping[str, object],
    source: Mapping[str, object],
    *,
    frame_id: str | None,
    geometry: str,
) -> dict[str, object]:
    """Inherit only known context fields; source values win over frame values."""

    context_aliases = (
        "wcs",
        "wcs_available",
        "has_wcs",
        "geometry",
        "geometry_status",
        "catalog",
        "catalog_name",
        "catalog_id",
        "photometric_system",
        "photometric_band",
        "band",
        "provenance",
        "calibration",
        "calibration_result",
        "absolute",
        "absolute_result",
        "absolute_magnitude",
        "error_budget",
    )
    merged: dict[str, object] = {}
    frame_values = _normalised_mapping(frame)
    source_values = _normalised_mapping(source)
    for alias in context_aliases:
        key = _norm_key(alias)
        if key in frame_values:
            merged[alias] = frame_values[key]
        if key in source_values:
            merged[alias] = source_values[key]
    merged.update(source)
    if frame_id is not None and not _lookup(merged, ("frame_id", "frame"))[0]:
        merged["frame_id"] = frame_id
    if geometry == "unavailable" and not any(
        _lookup(source, aliases)[0] for aliases in (("wcs", "wcs_available", "has_wcs", "geometry", "geometry_status"),)
    ):
        merged["__report_geometry_state"] = "unavailable"
    else:
        merged["__report_geometry_state"] = _geometry_state(source, default=geometry)
    merged["__report_frame_context"] = True
    return merged


def _normalise_frame_input(value: object) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    """Return ``(frame_rows, source_rows, malformed_count)``."""

    mapping = _as_mapping(value)
    if mapping is not None:
        for alias in _FRAME_COLLECTION_KEYS:
            found, nested = _lookup(mapping, (alias,))
            if found:
                value = nested
                break
    rows, malformed = _row_list(value, id_aliases=("frame_id", "frame", "id"))
    if not rows:
        return [], [], malformed

    has_nested_sources = any(_collection(row)[0] for row in rows)
    if not has_nested_sources and all(_looks_like_row(row) for row in rows):
        # A positional list of source rows is a common compact input form.
        direct = []
        for row in rows:
            copy = dict(row)
            copy["__report_geometry_state"] = _geometry_state(copy, default="unknown")
            copy["__report_frame_context"] = False
            direct.append(copy)
        return [], direct, malformed

    frames: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for index, frame in enumerate(rows):
        raw_id = _value(frame, ("frame_id", "frame", "id"), default=index)
        frame_id = str(raw_id) if _nonempty(raw_id) else str(index)
        has_sources, raw_sources = _collection(frame)
        frame_geometry = _geometry_state(frame, default="unavailable" if has_sources else "unknown")
        frame_record = dict(frame)
        frame_record["frame_id"] = frame_id
        frame_record["__report_geometry_state"] = frame_geometry
        frames.append(frame_record)
        if not has_sources:
            if _looks_like_row(frame):
                source = dict(frame)
                source["__report_geometry_state"] = frame_geometry
                source["__report_frame_context"] = True
                sources.append(source)
            continue
        nested, nested_bad = _row_list(raw_sources, id_aliases=("source_id", "detection_id", "id"))
        malformed += nested_bad
        for source in nested:
            sources.append(_merge_context(frame, source, frame_id=frame_id, geometry=frame_geometry))
    return frames, sources, malformed


def _normalise_source_input(value: object) -> tuple[list[dict[str, object]], int]:
    rows, malformed = _row_list(value, id_aliases=("source_id", "detection_id", "id"))
    result: list[dict[str, object]] = []
    for row in rows:
        copy = dict(row)
        copy["__report_geometry_state"] = _geometry_state(copy, default="unknown")
        copy["__report_frame_context"] = False
        result.append(copy)
    return result, malformed


def _normalise_results(
    value: object,
    *,
    kind: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]], int]:
    """Normalise global/per-source calibration or absolute result inputs."""

    if value is None:
        return [], {}, 0
    mapping = _as_mapping(value)
    if mapping is not None and not _looks_like_row(mapping):
        for alias in _RESULT_COLLECTION_KEYS:
            found, nested = _lookup(mapping, (alias,))
            if found:
                rows, index, malformed = _normalise_results(nested, kind=kind)
                # Keep useful top-level metadata as a global row when the
                # wrapper contains status/error/provenance fields.
                metadata = {
                    key: item
                    for key, item in mapping.items()
                    if _norm_key(key)
                    not in {_norm_key(alias_name) for alias_name in _RESULT_COLLECTION_KEYS}
                }
                if metadata and _looks_like_row(metadata):
                    rows.insert(0, metadata)
                return rows, index, malformed

    rows, malformed = _row_list(value, id_aliases=("source_id", "detection_id", "frame_id", "id"))
    global_rows: list[dict[str, object]] = []
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        source_id = _value(row, ("source_id", "source", "catalog_id"), default=None)
        detection_id = _value(row, ("detection_id", "id"), default=None)
        frame_id = _value(row, ("frame_id", "frame"), default=None)
        if source_id is None and detection_id is None and frame_id is None:
            global_rows.append(row)
            continue
        if source_id is not None:
            indexed[("source_id", str(source_id))] = row
        if detection_id is not None:
            indexed[("detection_id", str(detection_id))] = row
        if frame_id is not None:
            indexed[("frame_id", str(frame_id))] = row
    return global_rows, indexed, malformed


def _result_for(
    row: Mapping[str, object],
    *,
    global_rows: Sequence[Mapping[str, object]],
    indexed: Mapping[tuple[str, str], Mapping[str, object]],
    nested_aliases: Iterable[str],
) -> dict[str, object]:
    merged: dict[str, object] = {}
    for result in global_rows:
        merged.update(result)
    for aliases in (("frame_id", "frame"), ("source_id", "source", "catalog_id"), ("detection_id", "id")):
        value = _value(row, aliases, default=None)
        if value is not None:
            specific = indexed.get((aliases[0], str(value)))
            if specific is not None:
                merged.update(specific)
    for alias in nested_aliases:
        nested = _value(row, (alias,), default=None)
        nested_mapping = _as_mapping(nested)
        if nested_mapping is not None:
            merged.update(nested_mapping)
    return merged


def _unique_mappings(mappings: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Remove alias-index duplicates while preserving deterministic order."""

    result: list[dict[str, object]] = []
    seen: set[int] = set()
    for mapping in mappings:
        object_id = id(mapping)
        if object_id not in seen:
            seen.add(object_id)
            result.append(mapping)
    return result


def _catalog_present(row: Mapping[str, object]) -> bool:
    found, explicit = _lookup(row, ("has_catalog", "catalog_available", "catalog_match"))
    if found:
        parsed = _bool_value(explicit)
        if parsed is not None:
            return parsed
    found, catalog_value = _lookup(row, ("catalog", "catalog_source", "catalog_name", "catalog_version"))
    if found and isinstance(catalog_value, Mapping):
        catalog_status = _status(catalog_value)
        if catalog_status and _contains_invalid_status(catalog_status):
            return False
        return bool(catalog_value)
    for aliases in (
        ("catalog_id", "matched_source_id"),
        ("catalog", "catalog_name", "catalog_version"),
        ("catalog_magnitude", "reference_magnitude", "catalog_mag"),
        ("match", "catalog_match_result"),
    ):
        found, value = _lookup(row, aliases)
        if found and _nonempty(value):
            return True
    return False


def _calibration_mode(row: Mapping[str, object], calibration: Mapping[str, object]) -> str:
    value = _value(
        row,
        ("calibration_scope", "calibration_type", "calibration_mode", "mode", "photometric_scope"),
        default=None,
    )
    if value is None:
        value = _value(calibration, ("calibration_scope", "calibration_type", "calibration_mode", "mode"), default=None)
    normalized = _norm_status(value)
    if normalized in {"RELATIVE", "RELATIVE_CALIBRATION", "RELATIVE_CALIBRATED", "INSTRUMENTAL_RELATIVE"}:
        return "relative"
    if normalized in {"APPARENT", "APPARENT_CALIBRATION", "APPARENT_CALIBRATED", "STANDARD", "ABSOLUTE_FLUX"}:
        return "apparent"
    for mapping in (row, calibration):
        found, value = _lookup(mapping, ("relative", "is_relative"))
        if found and _bool_value(value) is True:
            return "relative"
        found, value = _lookup(mapping, ("apparent", "is_apparent"))
        if found and _bool_value(value) is True:
            return "apparent"
    if _catalog_present(row) or _catalog_present(calibration) or _nonempty(
        _value(calibration, ("photometric_system", "photometric_band", "band"), default=None)
    ):
        return "apparent"
    return "relative"


def _contains_invalid_status(status: str) -> bool:
    if not status:
        return False
    if status in _INVALID_STATUS_WORDS:
        return True
    if status in _CALIBRATION_VALID_STATUSES or status in _ABSOLUTE_VALID_STATUSES:
        return False
    tokens = set(status.split("_"))
    return bool(tokens.intersection({"INVALID", "REJECTED", "MISSING", "UNAVAILABLE", "NO"}))


def _calibration_status_valid(
    statuses: Sequence[str],
    *,
    explicit_level: str | None,
    calibration: Mapping[str, object],
) -> bool:
    if explicit_level in {"RELATIVE_CALIBRATED", "APPARENT_CALIBRATED", "ABSOLUTE_ELIGIBLE"} and not statuses:
        return True
    nonempty = [status for status in statuses if status]
    if not nonempty:
        return False
    has_valid_status = False
    for status in nonempty:
        if _contains_invalid_status(status):
            return False
        if status in _CALIBRATION_VALID_STATUSES:
            has_valid_status = True
            continue
        if "CALIBRAT" in status and "NOT" not in status:
            has_valid_status = True
            continue
        # A status that is neither a known calibrated status nor a known
        # invalid status is unsafe to use as an implicit promotion.
        return False
    if has_valid_status:
        return True
    # A boolean validity flag is accepted only when it is explicit.
    found, value = _lookup(calibration, ("valid", "is_valid"))
    return found and _bool_value(value) is True


def _absolute_status_valid(
    statuses: Sequence[str],
    *,
    explicit_level: str | None,
    absolute: Mapping[str, object],
) -> bool:
    nonempty = [status for status in statuses if status]
    if not nonempty:
        return False
    has_valid_status = False
    allowed_lower_layer = {
        "INSTRUMENTAL_ONLY",
        "RELATIVE_CALIBRATED",
        "APPARENT_CALIBRATED",
        "CALIBRATED",
    }
    for status in nonempty:
        # Check the allow-list before the generic token check.  Valid states
        # such as VALID_NO_HOLDOUT and
        # VALID_MODEL_DISTANCE_NO_INTERVAL contain the token ``NO`` but are
        # still explicit, meaningful states rather than rejection states.
        if status in _ABSOLUTE_VALID_STATUSES:
            has_valid_status = True
            continue
        if status in allowed_lower_layer:
            continue
        if _contains_invalid_status(status):
            return False
        if status == "VALID":
            has_valid_status = True
            continue
        if "ABSOLUTE" in status and "ELIGIB" in status:
            has_valid_status = True
            continue
        return False
    if has_valid_status:
        return True
    # A bare ``valid: true`` flag is not enough for an absolute result.  The
    # report boundary needs a named status so it can check that the distance
    # path and the status describe the same provenance.
    return False


_MODEL_DISTANCE_STATUSES = {
    "VALID_MODEL_DISTANCE",
    "VALID_MODEL_DISTANCE_NO_INTERVAL",
}


def _distance_source_kind(value: object) -> str | None:
    """Classify the small set of distance provenance labels we can audit.

    The report layer intentionally does not infer a provenance label from a
    positive ``distance_pc``.  Only recognizable parallax/model labels are
    accepted, and unknown labels remain unsafe for strict eligibility.
    """

    normalized = _norm_status(value)
    if not normalized:
        return None
    # Check model first because a fallback label such as
    # PARALLAX_QUALITY_FALLBACK_TO_MODEL_DISTANCE contains both words.
    if ("GSP" in normalized and "PHOT" in normalized) or (
        "MODEL" in normalized and "DISTANCE" in normalized
    ):
        return "model"
    if "PARALLAX" in normalized:
        return "parallax"
    return "unknown"


def _distance_evidence(
    row: Mapping[str, object],
    absolute: Mapping[str, object],
    statuses: Sequence[str],
    *,
    parallax: float | None,
    distance_pc: float | None,
) -> tuple[str | None, bool, str | None]:
    """Return ``(path, strict, issue)`` for the absolute-distance gate.

    A positive parallax field is itself an explicit direct-parallax path, so
    legacy rows that contain parallax but no redundant source label remain
    auditable.  A positive ``distance_pc`` on its own is deliberately not
    evidence of provenance.  Model distances require a model status and a
    complete two-sided interval before they can promote a row to the strict
    absolute level; the point estimate can still be retained by callers.
    """

    positive_parallax = parallax is not None and parallax > 0
    positive_distance = distance_pc is not None and distance_pc > 0

    source_values: list[object] = []
    for mapping in (row, absolute):
        found, value = _lookup(mapping, ("distance_source", "distance_method", "distance_origin"))
        if found and _nonempty(value):
            source_values.append(value)

    source_kinds = [_distance_source_kind(value) for value in source_values]
    if source_kinds:
        if any(kind in {None, "unknown"} for kind in source_kinds):
            return None, False, "DISTANCE_SOURCE_UNRECOGNIZED"
        if len(set(source_kinds)) != 1:
            return None, False, "DISTANCE_SOURCE_CONFLICT"
        path = source_kinds[0]
    elif positive_parallax:
        # The presence of a usable parallax field identifies the direct path.
        path = "parallax"
    elif positive_distance:
        return None, False, "DISTANCE_SOURCE_REQUIRED"
    else:
        return None, False, "DISTANCE_MISSING"

    has_model_status = any(status in _MODEL_DISTANCE_STATUSES for status in statuses)
    if path == "parallax":
        if has_model_status:
            return path, False, "DISTANCE_SOURCE_STATUS_MISMATCH"
        if not positive_parallax and not positive_distance:
            return path, False, "PARALLAX_VALUE_MISSING"
        return path, True, None

    # The only other recognized path is an explicitly sourced model distance.
    if not positive_distance:
        return path, False, "MODEL_DISTANCE_MISSING"
    if not has_model_status:
        return path, False, "DISTANCE_SOURCE_STATUS_MISMATCH"
    # This status is a legitimate model point estimate, but it explicitly
    # says that the distance interval is absent.  It must not be promoted to
    # the complete strict level.
    if "VALID_MODEL_DISTANCE_NO_INTERVAL" in statuses:
        return path, False, "MODEL_DISTANCE_INTERVAL_MISSING"

    lower_present, lower = _present_number(row, ("distance_lower_pc", "distance_gspphot_lower", "distance_lower"))
    if not lower_present:
        lower_present, lower = _present_number(
            absolute,
            ("distance_lower_pc", "distance_gspphot_lower", "distance_lower"),
        )
    upper_present, upper = _present_number(row, ("distance_upper_pc", "distance_gspphot_upper", "distance_upper"))
    if not upper_present:
        upper_present, upper = _present_number(
            absolute,
            ("distance_upper_pc", "distance_gspphot_upper", "distance_upper"),
        )
    interval_valid = (
        lower is not None
        and upper is not None
        and lower > 0
        and upper > 0
        and lower <= upper
        and distance_pc is not None
        and lower <= distance_pc <= upper
    )
    if not interval_valid:
        return path, False, "MODEL_DISTANCE_INTERVAL_MISSING"
    return path, True, None


def _canonical_photometric_band(value: object) -> str:
    """Normalize the small set of bands used by the strict report gate."""

    if value is None:
        return "unknown"
    text = str(value).strip()
    key = "".join(character for character in text.casefold() if character.isalnum())
    if key in {"g", "gaiag", "gaiadr3g"}:
        return "G"
    if key in {"v", "johnsonv", "johnsoncousinsv"}:
        return "V"
    if key in {"a0", "azero", "a05414nm", "5414nm"}:
        return "A0(541.4 nm)"
    if key in {"", "unknown", "unk", "na", "none"}:
        return "unknown"
    return text


def _canonical_photometric_system(value: object) -> str:
    """Normalize catalogue/system aliases without guessing arbitrary systems."""

    if value is None:
        return "unknown"
    text = str(value).strip()
    key = "".join(character for character in text.casefold() if character.isalnum())
    if key in {"gaia", "gaiavega", "gaiadr3", "gaiadr3vega"}:
        return "Gaia"
    if key in {"johnson", "johnsonv", "johnsoncousins", "johnsoncousinsv"}:
        return "Johnson"
    if key in {"monochromatic", "monochromatic5414nm", "a0"}:
        return "monochromatic"
    if key in {"", "unknown", "unk", "na", "none"}:
        return "unknown"
    return text


def _first_nonempty_value(
    mappings: Sequence[Mapping[str, object]], aliases: Iterable[str]
) -> object | None:
    for mapping in mappings:
        found, value = _lookup(mapping, aliases)
        if found and _nonempty(value):
            return value
    return None


def _extinction_evidence(
    row: Mapping[str, object],
    absolute: Mapping[str, object],
) -> tuple[bool, str | None, float | None]:
    """Validate extinction value, band, system and provenance as one unit.

    A numeric ``extinction_mag`` is not self-describing.  The strict report
    therefore requires a declared band, a compatible photometric system and a
    non-placeholder source.  The function intentionally does not infer
    ``A_G`` from ``A_V`` or ``A_0``.
    """

    mappings = (row, absolute)
    found, raw_value = _lookup(row, ("extinction_mag", "extinction", "A_V", "av"))
    if not found:
        found, raw_value = _lookup(absolute, ("extinction_mag", "extinction", "A_V", "av"))
    if not found:
        return False, "EXTINCTION_MISSING", None
    extinction = _number(raw_value)
    if extinction is None or extinction < 0:
        return False, "EXTINCTION_INVALID", extinction

    band = _first_nonempty_value(
        mappings,
        ("extinction_band", "ext_band", "extinction_passband"),
    )
    system = _first_nonempty_value(
        mappings,
        ("extinction_system", "extinction_photometric_system", "ext_system"),
    )
    source = _first_nonempty_value(
        mappings,
        ("extinction_source", "extinction_provenance", "ext_source"),
    )
    canonical_band = _canonical_photometric_band(band)
    canonical_system = _canonical_photometric_system(system)
    if canonical_band == "unknown" or canonical_system == "unknown":
        return False, "EXTINCTION_SEMANTICS_REQUIRED", extinction
    if not _nonempty(source) or _norm_key(source) in {"unknown", "unk", "na", "none"}:
        return False, "EXTINCTION_SOURCE_REQUIRED", extinction

    photometric_band = _first_nonempty_value(
        mappings,
        ("photometric_band", "band", "passband"),
    )
    photometric_system = _first_nonempty_value(
        mappings,
        ("photometric_system", "system"),
    )
    expected_band = _canonical_photometric_band(photometric_band)
    expected_system = _canonical_photometric_system(photometric_system)
    if expected_band != "unknown" and expected_band != canonical_band:
        return False, "EXTINCTION_BAND_MISMATCH", extinction
    if expected_system != "unknown" and expected_system != canonical_system:
        return False, "EXTINCTION_BAND_MISMATCH", extinction
    return True, None, extinction


def _error_values(mapping: Mapping[str, object] | None) -> dict[str, float]:
    if mapping is None:
        return {}
    result: dict[str, float] = {}
    for component, aliases in _ERROR_ALIASES.items():
        found, value = _lookup(mapping, aliases)
        if found:
            number = _number(value)
            if number is not None and number >= 0:
                result[component] = number
    nested = _value(mapping, ("error_budget", "errors", "uncertainty"), default=None)
    nested_mapping = _as_mapping(nested)
    if nested_mapping is not None:
        for key in sorted(nested_mapping, key=lambda item: str(item)):
            value = nested_mapping[key]
            if isinstance(value, Mapping):
                value = _value(value, ("value", "sigma", "error", "std", "rms"), default=None)
            number = _number(value)
            if number is not None and number >= 0:
                name = _norm_key(key) or "unknown"
                result.setdefault(name, number)
    return result


def _summarise_errors(values: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for component in sorted(values):
        finite = sorted(float(value) for value in values[component] if math.isfinite(float(value)) and value >= 0)
        if not finite:
            continue
        summary[component] = {
            "count": len(finite),
            "min": finite[0],
            "max": finite[-1],
            "mean": statistics.fmean(finite),
            "median": statistics.median(finite),
            "rms": math.sqrt(statistics.fmean(value * value for value in finite)),
        }
    return summary


def _add_provenance(target: dict[str, set[str]], mapping: Mapping[str, object] | None) -> None:
    if mapping is None:
        return
    nested_values = [_as_mapping(_value(mapping, ("provenance",), default=None))]
    for alias in ("wcs", "calibration", "calibration_result", "absolute", "absolute_result", "absolute_magnitude"):
        nested_values.append(_as_mapping(_value(mapping, (alias,), default=None)))
    mappings = (mapping, *(item for item in nested_values if item is not None))
    for current in mappings:
        for category, aliases in _PROVENANCE_ALIASES.items():
            found, value = _lookup(current, aliases)
            if not found:
                continue
            if _is_sequence(value):
                for item in value:
                    if _nonempty(item):
                        target[category].add(str(item))
            elif _nonempty(value):
                target[category].add(str(value))


def _provenance_summary(
    *,
    frame_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    absolute_rows: Sequence[Mapping[str, object]],
    geometry_counts: Mapping[str, int],
    malformed_count: int,
) -> dict[str, object]:
    values: dict[str, set[str]] = defaultdict(set)
    for mapping in tuple(frame_rows) + tuple(source_rows) + tuple(calibration_rows) + tuple(absolute_rows):
        _add_provenance(values, mapping)
    frame_ids = sorted(
        {
            str(item)
            for row in frame_rows
            if (item := _value(row, ("frame_id", "frame"), default=None)) is not None
        }
    )
    source_ids = sorted(
        {
            str(item)
            for row in source_rows
            if (item := _value(row, ("source_id", "detection_id", "id"), default=None)) is not None
        }
    )
    return {
        "input_shapes": {
            "frame_rows": len(frame_rows),
            "source_rows": len(source_rows),
            "calibration_results": len(calibration_rows),
            "absolute_results": len(absolute_rows),
            "malformed_rows_ignored": malformed_count,
        },
        "frame_ids": frame_ids,
        "source_ids": source_ids,
        "geometry": {key: int(geometry_counts.get(key, 0)) for key in ("available", "unavailable", "unknown")},
        "declared": {
            key: sorted(values[key])
            for key in sorted(set(_PROVENANCE_ALIASES).union(values))
        },
    }


@dataclass(frozen=True, slots=True)
class FalseValidViolation:
    """One value/status contradiction found by the false-valid gate."""

    row_key: str
    field: str
    code: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, object]:
        return {
            "row_key": self.row_key,
            "field": self.field,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class FalseValidGate:
    """Aggregate state of the strict magnitude/status consistency gate."""

    checked_rows: int
    violation_count: int
    error_count: int
    flag_count: int
    passed: bool
    flags: tuple[str, ...] = ()
    violations: tuple[FalseValidViolation, ...] = ()

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(violation.message for violation in self.violations if violation.severity == "error")

    def as_dict(self) -> dict[str, object]:
        return {
            "checked_rows": self.checked_rows,
            "violation_count": self.violation_count,
            "error_count": self.error_count,
            "flag_count": self.flag_count,
            "passed": self.passed,
            "flags": list(self.flags),
            "errors": list(self.errors),
            "violations": [violation.as_dict() for violation in self.violations],
        }


@dataclass(frozen=True, slots=True)
class PhotometricAuditRow:
    """Audited, JSON-friendly view of one frame/source photometry row."""

    row_key: str
    frame_id: str | None
    source_id: str | None
    observability_level: str
    status: str | None
    calibration_status: str | None
    absolute_status: str | None
    m_inst: float | None
    m_cal: float | None
    absolute_magnitude: float | None
    wcs_available: bool | None
    catalog_available: bool
    flags: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    provenance: dict[str, object] | None = None
    error_budget: dict[str, float] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "row_key": self.row_key,
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "observability_level": self.observability_level,
            "observability": self.observability_level,
            "level": self.observability_level,
            "status": self.status,
            "calibration_status": self.calibration_status,
            "absolute_status": self.absolute_status,
            "m_inst": self.m_inst,
            "instrumental_magnitude": self.m_inst,
            "m_cal": self.m_cal,
            "calibrated_magnitude": self.m_cal,
            "M": self.absolute_magnitude,
            "absolute_magnitude": self.absolute_magnitude,
            "wcs_available": self.wcs_available,
            "catalog_available": self.catalog_available,
            "flags": list(self.flags),
            "rejection_reasons": list(self.rejection_reasons),
            "missing_inputs": list(self.missing_inputs),
            "provenance": dict(self.provenance or {}),
            "error_budget": dict(self.error_budget or {}),
        }


@dataclass(frozen=True, slots=True)
class PhotometricQualityReport:
    """Complete deterministic photometric observability audit."""

    frame_count: int
    source_count: int
    level_counts: dict[str, int]
    rejection_reasons: dict[str, int]
    missing_inputs: dict[str, int]
    error_budget: dict[str, dict[str, float | int]]
    provenance: dict[str, object]
    rows: tuple[PhotometricAuditRow, ...]
    false_valid_gate: FalseValidGate
    flags: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    ignored_row_count: int = 0

    @property
    def observability_counts(self) -> dict[str, int]:
        return dict(self.level_counts)

    @property
    def reason_counts(self) -> dict[str, int]:
        return dict(self.rejection_reasons)

    @property
    def missing_input_counts(self) -> dict[str, int]:
        return dict(self.missing_inputs)

    @property
    def counts(self) -> dict[str, object]:
        return {
            "frames": self.frame_count,
            "sources": self.source_count,
            "observability": dict(self.level_counts),
        }

    @property
    def false_valid(self) -> bool:
        return not self.false_valid_gate.passed

    @classmethod
    def from_rows(cls, rows: object = None, **kwargs: object) -> "PhotometricQualityReport":
        """Construct a report through the main normalising entry point."""

        return build_photometric_report(rows, **kwargs)

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "source_count": self.source_count,
            "counts": self.counts,
            "level_counts": dict(self.level_counts),
            "observability_counts": dict(self.level_counts),
            "observability_levels": dict(self.level_counts),
            "reason_counts": dict(self.rejection_reasons),
            "missing_input_counts": dict(self.missing_inputs),
            "rejection_reasons": dict(self.rejection_reasons),
            "missing_inputs": dict(self.missing_inputs),
            "error_budget": {key: dict(value) for key, value in self.error_budget.items()},
            "provenance": dict(self.provenance),
            "false_valid_gate": self.false_valid_gate.as_dict(),
            "false_valid": self.false_valid,
            "flags": list(self.flags),
            "errors": list(self.errors),
            "ignored_row_count": self.ignored_row_count,
            "rows": [row.as_dict() for row in self.rows],
        }

    def to_json(self, *, sort_keys: bool = True) -> str:
        """Return strict JSON; non-finite numeric values never escape here."""

        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=sort_keys, allow_nan=False)

    def to_dict(self) -> dict[str, object]:
        """Alias for :meth:`as_dict` used by some JSON/report callers."""

        return self.as_dict()

    def as_json(self, *, sort_keys: bool = True) -> str:
        """Alias for :meth:`to_json`."""

        return self.to_json(sort_keys=sort_keys)


# Friendly aliases for callers that use the shorter report names.
PhotometricReport = PhotometricQualityReport
PhotometricAuditReport = PhotometricQualityReport
PhotometricInnovationQualityReport = PhotometricQualityReport
InnovationQualityReport = PhotometricQualityReport


def _row_key(row: Mapping[str, object], index: int) -> tuple[str, str | None, str | None]:
    frame = _value(row, ("frame_id", "frame"), default=None)
    source = _value(row, ("source_id", "detection_id", "id"), default=None)
    frame_id = str(frame) if _nonempty(frame) else None
    source_id = str(source) if _nonempty(source) else None
    return f"{frame_id or 'frame?'}:{source_id or f'row-{index}'}", frame_id, source_id


def _audit_row(
    row: Mapping[str, object],
    *,
    index: int,
    calibration: Mapping[str, object],
    absolute: Mapping[str, object],
    require_wcs: bool,
) -> tuple[PhotometricAuditRow, list[FalseValidViolation], str]:
    row_key, frame_id, source_id = _row_key(row, index)
    explicit_level = _level(_value(row, ("observability_level", "level"), default=None))
    row_status = _status(row, ("status", "state")) or None

    absolute_raw_found, absolute_raw = _lookup(row, ("absolute_magnitude", "absolute", "absolute_result"))
    absolute_value_mapping = dict(absolute)
    nested_absolute = _as_mapping(absolute_raw if absolute_raw_found else None)
    if nested_absolute is not None:
        absolute_value_mapping.update(nested_absolute)
    calibration_value_mapping = dict(calibration)
    nested_calibration = _as_mapping(_value(row, ("calibration", "calibration_result"), default=None))
    if nested_calibration is not None:
        calibration_value_mapping.update(nested_calibration)
    cal_status = (
        _status(row, ("calibration_status", "m_cal_status"))
        or _status(calibration_value_mapping, ("status", "state", "calibration_status"))
        or row_status
    ) or None
    abs_status = (
        _status(row, ("absolute_status", "M_status", "absolute_magnitude_status"))
        or _status(absolute_value_mapping, ("status", "state", "absolute_status"))
        or row_status
    ) or None

    m_inst_present, m_inst = _present_number(row, ("m_inst", "instrumental_magnitude", "instrumental_mag"))
    m_cal_present, m_cal = _present_number(
        row,
        ("m_cal", "calibrated_magnitude", "apparent_magnitude", "m_std", "standard_magnitude"),
    )
    if not m_cal_present:
        m_cal_present, m_cal = _present_number(
            calibration,
            ("m_cal", "calibrated_magnitude", "apparent_magnitude"),
        )
    absolute_present, absolute_magnitude = _present_number(
        row,
        ("M", "absolute_magnitude_value", "absolute_mag", "M_V", "m_abs"),
    )
    if not absolute_present and absolute_raw_found and not isinstance(absolute_raw, Mapping):
        absolute_present = absolute_raw is not None and absolute_raw != ""
        absolute_magnitude = _number(absolute_raw)
    if not absolute_present:
        absolute_present, absolute_magnitude = _present_number(
            absolute_value_mapping,
            ("value", "M", "absolute_magnitude", "absolute_magnitude_value"),
        )
    if not m_inst_present and m_inst is None and _lookup(row, ("m_inst", "instrumental_magnitude"))[0]:
        m_inst_present = True
    if not m_cal_present and _lookup(row, ("m_cal", "calibrated_magnitude"))[0]:
        m_cal_present = True
    if not absolute_present and _lookup(row, ("M", "absolute_magnitude_value"))[0]:
        absolute_present = True

    geometry = str(row.get("__report_geometry_state", _geometry_state(row, default="unknown")))
    if geometry not in {"available", "unavailable", "unknown"}:
        geometry = "unknown"
    explicit_geometry = _geometry_state(row, default="unknown")
    if explicit_geometry != "unknown":
        geometry = explicit_geometry
    geometry_available: bool | None = {"available": True, "unavailable": False}.get(geometry)
    catalog_available = (
        _catalog_present(row)
        or _catalog_present(calibration_value_mapping)
        or _catalog_present(absolute_value_mapping)
    )
    calibration_statuses = tuple(
        status for status in (cal_status, row_status, _status(calibration_value_mapping)) if status
    )
    absolute_statuses = tuple(
        status for status in (abs_status, row_status, _status(absolute_value_mapping)) if status
    )
    calibration_valid = m_cal is not None and _calibration_status_valid(
        calibration_statuses,
        explicit_level=explicit_level,
        calibration=calibration_value_mapping,
    )
    absolute_status_valid = absolute_magnitude is not None and _absolute_status_valid(
        absolute_statuses,
        explicit_level=explicit_level,
        absolute=absolute_value_mapping,
    )

    parallax_present, parallax = _present_number(row, ("parallax_mas", "parallax", "corrected_parallax_mas"))
    if not parallax_present:
        parallax_present, parallax = _present_number(
            absolute_value_mapping,
            ("parallax_mas", "parallax", "corrected_parallax_mas"),
        )
    distance_present, distance_pc = _present_number(row, ("distance_pc", "distance"))
    if not distance_present:
        distance_present, distance_pc = _present_number(absolute_value_mapping, ("distance_pc", "distance"))
    distance_path, distance_qualified, distance_issue = _distance_evidence(
        row,
        absolute_value_mapping,
        absolute_statuses,
        parallax=parallax,
        distance_pc=distance_pc,
    )
    extinction_ok, extinction_issue, extinction = _extinction_evidence(
        row,
        absolute_value_mapping,
    )

    violations: list[FalseValidViolation] = []
    if m_cal_present and m_cal is None:
        violations.append(
            FalseValidViolation(row_key, "m_cal", "NONFINITE_M_CAL", "m_cal is present but is not finite")
        )
    elif m_cal is not None and not calibration_valid:
        code = "M_CAL_STATUS_MISSING" if not calibration_statuses else "M_CAL_STATUS_INVALID"
        violations.append(
            FalseValidViolation(
                row_key,
                "m_cal",
                code,
                f"m_cal is present but calibration status is not valid: {','.join(calibration_statuses) or 'missing'}",
            )
        )
    if absolute_present and absolute_magnitude is None:
        violations.append(FalseValidViolation(row_key, "M", "NONFINITE_M", "M is present but is not finite"))
    elif absolute_magnitude is not None and not absolute_status_valid:
        code = "M_STATUS_MISSING" if not absolute_statuses else "M_STATUS_INVALID"
        violations.append(
            FalseValidViolation(
                row_key,
                "M",
                code,
                f"M is present but absolute status is not valid: {','.join(absolute_statuses) or 'missing'}",
            )
        )
    if absolute_magnitude is not None and absolute_status_valid:
        missing_absolute = []
        if not calibration_valid:
            missing_absolute.append("calibration")
        if not catalog_available:
            missing_absolute.append("catalog")
        # A model point estimate without an interval is intentionally kept as
        # an incomplete-but-meaningful result rather than called false-valid.
        # Other distance provenance failures are reported as a dedicated
        # violation below, so they are not hidden behind a generic parallax
        # message.
        incomplete_model_estimate = (
            distance_path == "model"
            and distance_issue == "MODEL_DISTANCE_INTERVAL_MISSING"
            and "VALID_MODEL_DISTANCE_NO_INTERVAL" in absolute_statuses
        )
        distance_provenance_issue = distance_issue not in {None, "DISTANCE_MISSING"}
        if not distance_qualified and not incomplete_model_estimate and not distance_provenance_issue:
            missing_absolute.append("parallax")
        if not extinction_ok:
            missing_absolute.append("extinction")
        if missing_absolute:
            violations.append(
                FalseValidViolation(
                    row_key,
                    "M",
                    "M_WITH_MISSING_INPUTS",
                    "M is present/marked valid but required inputs are missing: " + ",".join(missing_absolute),
                )
            )
        if not distance_qualified and distance_provenance_issue and not incomplete_model_estimate:
            violations.append(
                FalseValidViolation(
                    row_key,
                    "distance_source",
                    distance_issue or "DISTANCE_EVIDENCE_INVALID",
                    "M is present/marked valid but distance provenance is not strict: "
                    + (distance_issue or "unknown"),
                )
            )
    if m_cal is None and explicit_level in {"RELATIVE_CALIBRATED", "APPARENT_CALIBRATED", "ABSOLUTE_ELIGIBLE"}:
        violations.append(
            FalseValidViolation(
                row_key,
                "observability_level",
                "LEVEL_WITHOUT_M_CAL",
                "calibrated level claimed without m_cal",
            )
        )
    if explicit_level == "ABSOLUTE_ELIGIBLE" and absolute_magnitude is None:
        violations.append(
            FalseValidViolation(
                row_key,
                "observability_level",
                "LEVEL_WITHOUT_M",
                "absolute level claimed without M",
            )
        )

    missing: list[str] = []
    if not m_inst_present or m_inst is None:
        missing.append("m_inst")
    geometry_gate_failed = require_wcs and geometry != "available"
    if geometry_gate_failed:
        missing.append("wcs")
    if not catalog_available:
        missing.append("catalog")
    if not calibration_valid:
        missing.append("calibration")
    if not distance_qualified:
        if distance_issue == "DISTANCE_SOURCE_REQUIRED":
            missing.append("distance_source")
        elif distance_issue == "MODEL_DISTANCE_INTERVAL_MISSING":
            missing.append("distance_interval")
        elif distance_issue == "DISTANCE_MISSING":
            # Preserve the established report vocabulary for a row with no
            # distance evidence at all.
            missing.append("parallax")
        else:
            missing.append("distance")
    if not extinction_ok:
        missing.append("extinction")
    if absolute_magnitude is None:
        missing.append("absolute_result")
    row_photometric_identity = _value(row, ("photometric_system", "photometric_band", "band"), default=None)
    calibration_photometric_identity = _value(
        calibration_value_mapping,
        ("photometric_system", "photometric_band", "band"),
        default=None,
    )
    if calibration_valid and not _nonempty(row_photometric_identity) and not _nonempty(
        calibration_photometric_identity
    ):
        missing.append("photometric_system")

    source_flags: list[str] = []
    raw_flags = _value(row, ("flags", "flag"), default=())
    if isinstance(raw_flags, str):
        source_flags = [_norm_status(raw_flags)] if raw_flags.strip() else []
    elif _is_sequence(raw_flags):
        source_flags = [_norm_status(flag) for flag in raw_flags if _nonempty(flag)]
    elif _nonempty(raw_flags):
        source_flags = [_norm_status(raw_flags)]
    raw_reasons = _value(row, ("rejection_reasons", "rejection_reason", "reason"), default=())
    reasons: list[str] = [_norm_status(reason) for reason in raw_reasons] if _is_sequence(raw_reasons) else []
    if isinstance(raw_reasons, str) and raw_reasons.strip():
        reasons.append(_norm_status(raw_reasons))
    for item in missing:
        reasons.append("MISSING_" + item.upper())
    if distance_issue and distance_issue != "DISTANCE_MISSING":
        source_flags.append(distance_issue)
        reasons.append(distance_issue)
    if extinction_issue and extinction_issue != "EXTINCTION_MISSING":
        source_flags.append(extinction_issue)
        reasons.append(extinction_issue)
    if row_status and _contains_invalid_status(row_status):
        source_flags.append(row_status)
        reasons.append(row_status)
    quality_passed = _bool_value(_value(row, ("quality_passed", "quality_ok"), default=None))
    if quality_passed is False:
        source_flags.append("QUALITY_REJECTED")
        reasons.append("QUALITY_REJECTED")
    for violation in violations:
        source_flags.append(violation.code)
        reasons.append(violation.code)

    mode = _calibration_mode(row, calibration_value_mapping)
    apparent_valid = calibration_valid and m_cal is not None and catalog_available and mode == "apparent"
    relative_valid = calibration_valid and m_cal is not None
    geometry_sufficient_for_absolute = not require_wcs or geometry == "available"
    absolute_valid = (
        absolute_status_valid
        and absolute_magnitude is not None
        and apparent_valid
        and distance_qualified
        and extinction_ok
        and geometry_sufficient_for_absolute
    )

    # A frame collection without a WCS is an explicit geometry failure.  For
    # compact direct source rows, geometry is left unknown unless the caller
    # requests a WCS gate or the row explicitly declares WCS unavailable; this
    # keeps old instrumental-only JSON auditable.
    if geometry_gate_failed:
        level = "GEOMETRY_UNAVAILABLE"
    elif absolute_valid:
        level = "ABSOLUTE_ELIGIBLE"
    elif apparent_valid:
        level = "APPARENT_CALIBRATED"
    elif relative_valid:
        level = "RELATIVE_CALIBRATED"
    elif m_inst is not None:
        level = "INSTRUMENTAL_ONLY"
    else:
        level = "GEOMETRY_UNAVAILABLE"

    if explicit_level and explicit_level != "GEOMETRY_UNAVAILABLE" and level != "GEOMETRY_UNAVAILABLE":
        # Never upgrade an explicit claim beyond the evidence actually seen.
        if _LEVEL_RANK.get(explicit_level, 0) < _LEVEL_RANK.get(level, 0):
            level = explicit_level
    elif explicit_level == "GEOMETRY_UNAVAILABLE":
        level = explicit_level

    if level == "GEOMETRY_UNAVAILABLE":
        source_flags.append("GEOMETRY_UNAVAILABLE")
        reasons.append("GEOMETRY_UNAVAILABLE")
    if violations:
        source_flags.append("FALSE_VALID")
    if m_cal is not None and not calibration_valid:
        source_flags.append("FALSE_VALID_M_CAL")
    if absolute_magnitude is not None and not absolute_status_valid:
        source_flags.append("FALSE_VALID_M")

    unique_flags = tuple(sorted({flag for flag in source_flags if flag}))
    unique_reasons = tuple(sorted({reason for reason in reasons if reason}))
    unique_missing = tuple(sorted(set(missing)))
    nested_provenance: dict[str, object] = {}
    source_provenance = _as_mapping(_value(row, ("provenance",), default=None))
    if source_provenance is not None:
        for key in sorted(source_provenance, key=lambda item: str(item)):
            value = source_provenance[key]
            safe_value = _json_scalar(value)
            if safe_value is not None or value is None:
                nested_provenance[str(key)] = safe_value
    error_values = _error_values(row)
    return (
        PhotometricAuditRow(
            row_key=row_key,
            frame_id=frame_id,
            source_id=source_id,
            observability_level=level,
            status=row_status,
            calibration_status=cal_status,
            absolute_status=abs_status,
            m_inst=m_inst,
            m_cal=m_cal,
            absolute_magnitude=absolute_magnitude,
            wcs_available=geometry_available,
            catalog_available=catalog_available,
            flags=unique_flags,
            rejection_reasons=unique_reasons,
            missing_inputs=unique_missing,
            provenance=nested_provenance,
            error_budget=error_values,
        ),
        violations,
        geometry,
    )


def build_photometric_report(
    rows: object = None,
    *,
    frame_rows: object = None,
    frames: object = None,
    source_rows: object = None,
    sources: object = None,
    photometry_rows: object = None,
    calibration: object = None,
    calibration_results: object = None,
    calibration_rows: object = None,
    calibration_result: object = None,
    absolute: object = None,
    absolute_results: object = None,
    absolute_rows: object = None,
    absolute_result: object = None,
    require_wcs: bool | None = None,
    strict: bool = False,
    raise_on_false_valid: bool | None = None,
) -> PhotometricQualityReport:
    """Build a deterministic photometric observability audit.

    ``rows`` may be either a source-row list/dict or a frame-row list/dict.
    Explicit ``frame_rows`` and ``source_rows`` can be supplied together.
    Calibration and absolute results may be global mappings, keyed mappings,
    or lists with ``source_id``/``detection_id``/``frame_id`` fields.

    ``require_wcs`` defaults to ``True`` for nested frame inputs and ``False``
    for compact direct source rows.  Set it explicitly when the caller knows
    whether geometry is a required gate.  ``strict`` keeps the default report
    non-throwing but can be used by CI to turn false-valid values into a
    ``ValueError``.
    """

    if raise_on_false_valid is not None:
        strict = bool(raise_on_false_valid)

    all_frame_rows: list[dict[str, object]] = []
    all_source_rows: list[dict[str, object]] = []
    ignored = 0
    has_explicit_frame_input = frame_rows is not None or frames is not None

    if rows is not None:
        normalised_frames, normalised_sources, malformed = _normalise_frame_input(rows)
        all_frame_rows.extend(normalised_frames)
        all_source_rows.extend(normalised_sources)
        ignored += malformed
    for frame_input in (frame_rows, frames):
        if frame_input is None:
            continue
        normalised_frames, normalised_sources, malformed = _normalise_frame_input(frame_input)
        all_frame_rows.extend(normalised_frames)
        all_source_rows.extend(normalised_sources)
        ignored += malformed
    if source_rows is not None:
        normalised_sources, malformed = _normalise_source_input(source_rows)
        all_source_rows.extend(normalised_sources)
        ignored += malformed
    if sources is not None:
        normalised_sources, malformed = _normalise_source_input(sources)
        all_source_rows.extend(normalised_sources)
        ignored += malformed
    if photometry_rows is not None:
        normalised_sources, malformed = _normalise_source_input(photometry_rows)
        all_source_rows.extend(normalised_sources)
        ignored += malformed

    if require_wcs is None:
        require_wcs = bool(
            has_explicit_frame_input
            or any(
                row.get("__report_frame_context")
                or _geometry_state(row, default="unknown") != "unknown"
                for row in all_source_rows
            )
        )
    require_wcs = bool(require_wcs)

    calibration_input = calibration_results if calibration_results is not None else calibration_rows
    if calibration_input is None:
        calibration_input = calibration_result
    if calibration_input is None:
        calibration_input = calibration
    absolute_input = absolute_results if absolute_results is not None else absolute_rows
    if absolute_input is None:
        absolute_input = absolute_result
    if absolute_input is None:
        absolute_input = absolute

    calibration_globals, calibration_index, calibration_bad = _normalise_results(calibration_input, kind="calibration")
    absolute_globals, absolute_index, absolute_bad = _normalise_results(absolute_input, kind="absolute")
    ignored += calibration_bad + absolute_bad
    calibration_index_rows = _unique_mappings(calibration_index.values())
    absolute_index_rows = _unique_mappings(absolute_index.values())
    indexed_result_rows = _unique_mappings(calibration_index_rows + absolute_index_rows)

    audited_rows: list[PhotometricAuditRow] = []
    violations: list[FalseValidViolation] = []
    geometry_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    error_values: defaultdict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(all_source_rows):
        row_calibration = _result_for(
            row,
            global_rows=calibration_globals,
            indexed=calibration_index,
            nested_aliases=("calibration", "calibration_result"),
        )
        row_absolute = _result_for(
            row,
            global_rows=absolute_globals,
            indexed=absolute_index,
            nested_aliases=("absolute", "absolute_result", "absolute_magnitude"),
        )
        audited, row_violations, geometry = _audit_row(
            row,
            index=index,
            calibration=row_calibration,
            absolute=row_absolute,
            require_wcs=require_wcs,
        )
        audited_rows.append(audited)
        violations.extend(row_violations)
        geometry_counts[geometry] += 1
        for reason in audited.rejection_reasons:
            rejection_counts[reason] += 1
        for missing in audited.missing_inputs:
            missing_counts[missing] += 1
        for component, value in (audited.error_budget or {}).items():
            error_values[component].append(value)
        nested_groups = (
            ("calibration", "calibration_result"),
            ("absolute", "absolute_result", "absolute_magnitude"),
        )
        for nested_aliases in nested_groups:
            nested = _as_mapping(_value(row, nested_aliases, default=None))
            for component, value in _error_values(nested).items():
                error_values[component].append(value)

    # Global calibration/absolute uncertainty is input provenance, not a
    # repeated per-source measurement.  It is nevertheless included in the
    # same deterministic component summary.
    for mapping in tuple(calibration_globals) + tuple(absolute_globals):
        for component, value in _error_values(mapping).items():
            error_values[component].append(value)
    for mapping in indexed_result_rows:
        for component, value in _error_values(mapping).items():
            error_values[component].append(value)

    level_counts = {level: 0 for level in OBSERVABILITY_LEVELS}
    for row in audited_rows:
        level_counts[row.observability_level] = level_counts.get(row.observability_level, 0) + 1

    gate_flags = tuple(sorted({violation.code for violation in violations}))
    gate = FalseValidGate(
        checked_rows=len(audited_rows),
        violation_count=len(violations),
        error_count=sum(1 for violation in violations if violation.severity == "error"),
        flag_count=len(gate_flags),
        passed=not violations,
        flags=gate_flags,
        violations=tuple(violations),
    )
    report_flags = set(gate_flags)
    report_errors = tuple(violation.message for violation in violations if violation.severity == "error")
    if not audited_rows:
        report_flags.add("NO_DATA")
    if ignored:
        report_flags.add("MALFORMED_ROWS_IGNORED")
    if violations:
        report_flags.add("FALSE_VALID")

    frame_count = len(all_frame_rows)
    if frame_count == 0:
        frame_ids = {row.frame_id for row in audited_rows if row.frame_id is not None}
        frame_count = len(frame_ids)
    provenance = _provenance_summary(
        frame_rows=all_frame_rows,
        source_rows=all_source_rows,
        calibration_rows=tuple(calibration_globals) + tuple(calibration_index_rows),
        absolute_rows=tuple(absolute_globals) + tuple(absolute_index_rows),
        geometry_counts=geometry_counts,
        malformed_count=ignored,
    )
    report = PhotometricQualityReport(
        frame_count=frame_count,
        source_count=len(audited_rows),
        level_counts=level_counts,
        rejection_reasons={key: rejection_counts[key] for key in sorted(rejection_counts)},
        missing_inputs={key: missing_counts[key] for key in sorted(missing_counts)},
        error_budget=_summarise_errors(error_values),
        provenance=provenance,
        rows=tuple(audited_rows),
        false_valid_gate=gate,
        flags=tuple(sorted(report_flags)),
        errors=report_errors,
        ignored_row_count=ignored,
    )
    if strict and violations:
        raise ValueError("false-valid photometry gate failed: " + "; ".join(report_errors))
    return report


def make_photometric_report(*args: object, **kwargs: object) -> PhotometricQualityReport:
    """Alias for :func:`build_photometric_report`."""

    return build_photometric_report(*args, **kwargs)


def audit_photometry(*args: object, **kwargs: object) -> PhotometricQualityReport:
    """Alias for :func:`build_photometric_report`."""

    return build_photometric_report(*args, **kwargs)


def generate_photometric_report(*args: object, **kwargs: object) -> PhotometricQualityReport:
    """Alias for :func:`build_photometric_report`."""

    return build_photometric_report(*args, **kwargs)


def generate_photometric_quality_report(*args: object, **kwargs: object) -> PhotometricQualityReport:
    """Explicit-name alias for :func:`build_photometric_report`."""

    return build_photometric_report(*args, **kwargs)


def create_photometric_report(*args: object, **kwargs: object) -> PhotometricQualityReport:
    """Alias for :func:`build_photometric_report`."""

    return build_photometric_report(*args, **kwargs)


def build_report(*args: object, **kwargs: object) -> PhotometricQualityReport:
    """Short alias for :func:`build_photometric_report`."""

    return build_photometric_report(*args, **kwargs)


def build_photometric_quality_report(*args: object, **kwargs: object) -> PhotometricQualityReport:
    """Explicit-name alias for :func:`build_photometric_report`."""

    return build_photometric_report(*args, **kwargs)


__all__ = [
    "OBSERVABILITY_LEVELS",
    "ObservabilityLevel",
    "FalseValidGate",
    "FalseValidViolation",
    "PhotometricAuditReport",
    "PhotometricAuditRow",
    "PhotometricInnovationQualityReport",
    "InnovationQualityReport",
    "PhotometricQualityReport",
    "PhotometricReport",
    "audit_photometry",
    "build_photometric_report",
    "build_photometric_quality_report",
    "build_report",
    "create_photometric_report",
    "generate_photometric_report",
    "generate_photometric_quality_report",
    "make_photometric_report",
]
