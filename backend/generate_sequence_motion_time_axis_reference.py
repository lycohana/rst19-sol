"""Freeze Python frame-time normalization and reference-interval cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _normalise_motion_times, _reference_time_interval_seconds


def capture(label: str, frame_count: int, frame_times: list[object] | None) -> dict[str, object]:
    normalized = _normalise_motion_times(frame_times, frame_count)
    return {
        "label": label,
        "request": {"frame_count": frame_count, "frame_times": frame_times},
        "expected": {
            "frame_times": normalized.tolist(),
            "reference_interval_s": _reference_time_interval_seconds(normalized),
        },
    }


def main() -> None:
    cases = [
        capture("missing_times_fall_back_to_frame_index", 4, None),
        capture("valid_irregular_times_are_relative_to_first", 4, [100.0, 110.0, 125.0, 140.0]),
        capture("already_relative_times_keep_elapsed_steps", 4, [0.0, 2.0, 7.0, 8.0]),
        capture("single_frame_has_one_second_reference_fallback", 1, [25.0]),
        capture("nonmonotonic_times_fall_back_as_a_whole", 4, [0.0, 1.0, 1.0, 3.0]),
        capture("decreasing_times_fall_back_as_a_whole", 3, [10.0, 9.0, 12.0]),
        capture("wrong_length_falls_back_as_a_whole", 3, [0.0, 1.0]),
        capture("null_time_falls_back_as_a_whole", 3, [0.0, None, 2.0]),
        capture("numeric_strings_convert_to_float64", 3, ["12.5", "15", "20.25"]),
        capture("boolean_values_follow_numeric_array_conversion", 2, [False, True]),
        capture("empty_sequence_keeps_empty_fallback_axis", 0, None),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_motion_time_axis.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
