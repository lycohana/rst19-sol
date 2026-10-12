"""Freeze Python DATE-OBS parsing and fallback scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _observation_times_seconds


def capture(label: str, headers: list[dict[str, object]]) -> dict[str, object]:
    analyses = [SimpleNamespace(frame=SimpleNamespace(header=header)) for header in headers]
    values = _observation_times_seconds(analyses)
    return {
        "label": label,
        "request": {"headers": headers},
        "expected": {"frame_times": values.tolist()},
    }


def main() -> None:
    cases = [
        capture("utc_timestamps_and_fractional_seconds", [
            {"DATE-OBS": "2024-02-29T23:59:59.250Z"},
            {"DATE-OBS": "2024-03-01T00:00:00.750Z"},
        ]),
        capture("timezone_offsets_normalize_to_utc", [
            {"DATE-OBS": "2024-03-01T00:30:00+02:00"},
            {"DATE-OBS": "2024-02-29T23:45:00+01:00"},
        ]),
        capture("naive_fits_timestamps_are_assumed_utc", [
            {"DATE-OBS": "2024-01-01 23:59:00"},
            {"DATE-OBS": "2024-01-02 00:01:00"},
        ]),
        capture("numeric_timezone_offset_without_colon", [
            {"DATE-OBS": "2024-01-01T10:00:00+0200"},
            {"DATE-OBS": "2024-01-01T09:00:30+0100"},
        ]),
        capture("date_only_values_parse_at_midnight", [
            {"DATE-OBS": "2024-01-01"},
            {"DATE-OBS": "2024-01-02"},
        ]),
        capture("missing_date_falls_back_to_frame_index", [
            {"DATE-OBS": "2024-01-01T00:00:00Z"},
            {},
            {"DATE-OBS": "2024-01-01T00:00:10Z"},
        ]),
        capture("null_date_falls_back_to_frame_index", [
            {"DATE-OBS": "2024-01-01T00:00:00Z"},
            {"DATE-OBS": None},
        ]),
        capture("invalid_timestamp_falls_back_to_frame_index", [
            {"DATE-OBS": "2024-01-01T00:00:00Z"},
            {"DATE-OBS": "not-a-timestamp"},
        ]),
        capture("duplicate_utc_time_falls_back_to_frame_index", [
            {"DATE-OBS": "2024-01-01T00:00:00Z"},
            {"DATE-OBS": "2024-01-01T00:00:00+00:00"},
        ]),
        capture("reverse_time_order_falls_back_to_frame_index", [
            {"DATE-OBS": "2024-01-01T00:00:10Z"},
            {"DATE-OBS": "2024-01-01T00:00:00Z"},
        ]),
        capture("invalid_calendar_date_falls_back_to_frame_index", [
            {"DATE-OBS": "2023-02-29T00:00:00Z"},
            {"DATE-OBS": "2023-03-01T00:00:00Z"},
        ]),
        capture("offset_crosses_year_boundary", [
            {"DATE-OBS": "2024-01-01T00:30:00+01:00"},
            {"DATE-OBS": "2023-12-31T23:45:00Z"},
        ]),
        capture("subsecond_offsets_preserve_millisecond_deltas", [
            {"DATE-OBS": "2024-06-01T12:00:00.125000Z"},
            {"DATE-OBS": "2024-06-01T12:00:00.375000Z"},
        ]),
        capture("offset_seconds_are_applied", [
            {"DATE-OBS": "2024-06-01T12:30:00+01:00:30"},
            {"DATE-OBS": "2024-06-01T12:30:31+01:00:00"},
        ]),
        capture("hour_and_minute_without_seconds_are_supported", [
            {"DATE-OBS": "2024-06-01T12:30.5Z"},
            {"DATE-OBS": "2024-06-01T12:31:00Z"},
        ]),
        capture("hour_only_timezone_offset_is_supported", [
            {"DATE-OBS": "2024-06-01T10:00:00+02"},
            {"DATE-OBS": "2024-06-01T08:01:00Z"},
        ]),
        capture("single_character_plus_separator_uses_python_iso_semantics", [
            {"DATE-OBS": "2024-06-01+10:00"},
            {"DATE-OBS": "2024-06-01T10:00:01"},
        ]),
        capture("alternate_ascii_datetime_separator_is_accepted", [
            {"DATE-OBS": "2024-06-01x10:00:00Z"},
            {"DATE-OBS": "2024-06-01T10:00:01Z"},
        ]),
        capture("overlong_fraction_truncates_to_microseconds", [
            {"DATE-OBS": "2024-06-01T12:00:00.123456789Z"},
            {"DATE-OBS": "2024-06-01T12:00:00.123457789Z"},
        ]),
        capture("invalid_timezone_offset_falls_back", [
            {"DATE-OBS": "2024-06-01T12:00:00+24:00"},
            {"DATE-OBS": "2024-06-01T13:00:00+24:00"},
        ]),
        capture("leap_second_falls_back", [
            {"DATE-OBS": "2024-06-01T12:00:60Z"},
            {"DATE-OBS": "2024-06-01T12:01:00Z"},
        ]),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_observation_time_axis.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
