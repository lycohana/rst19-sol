"""Freeze Python exposure-speed consistency scenarios for line tracks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import (
    _motion_feature_exposure_is_supported,
    _motion_feature_exposure_metrics,
)


def measured_length(motion_length, fwhm=2.0):
    sigma = fwhm / 2.3548200450309493
    return math.sqrt(16.0 * sigma * sigma + motion_length * motion_length / 0.75)


def points_for(indices, positions, lengths, *, edge_indices=()):
    return [
        {
            "frame_index": index,
            "aligned_x": position,
            "aligned_y": 0.0,
            "length_px": measured_length(length),
            "touches_edge": index in edge_indices,
        }
        for index, position, length in zip(indices, positions, lengths, strict=True)
    ]


def headers(exposures, *, missing_date_indices=(), missing_exposure_indices=(), keys=None):
    result = []
    keys = keys or ["EXPTIME"] * len(exposures)
    for index, (exposure, key) in enumerate(zip(exposures, keys, strict=True)):
        header = {}
        if index not in missing_date_indices:
            header["DATE-OBS"] = f"2024-01-01T00:00:{index:02d}Z"
        if index not in missing_exposure_indices:
            header[key] = exposure
        result.append(header)
    return result


def capture(label, points, frame_headers, *, frame_times=None, psf_fwhm=2.0):
    point_objects = [SimpleNamespace(**point) for point in points]
    analyses = [
        SimpleNamespace(frame=SimpleNamespace(header=header)) for header in frame_headers
    ]
    metrics = _motion_feature_exposure_metrics(
        point_objects,
        analyses,
        frame_times,
        psf_fwhm=psf_fwhm,
    )
    expected = {
        "ratio": metrics[0],
        "log_mad": metrics[1],
        "in_range_fraction": metrics[2],
        "supported": _motion_feature_exposure_is_supported(*metrics),
    }
    return {
        "label": label,
        "request": {
            "points": points,
            "frame_headers": frame_headers,
            "frame_times": frame_times,
            "psf_fwhm": psf_fwhm,
        },
        "expected": expected,
    }


def main() -> None:
    consistent_points = points_for([0, 1, 2], [0, 10, 20], [1, 1, 1])
    three_step_points = points_for([0, 1, 2, 3, 4], [0, 10, 20, 30, 40], [1, 1, 1, 1, 1])
    inconsistent_fraction = points_for(
        [0, 1, 2, 3, 4], [0, 10, 20, 30, 40], [1, 0.1, 1, 1, 1]
    )
    high_mad = points_for(
        [0, 1, 2, 3, 4], [0, 10, 20, 30, 40], [0.5, 2, 0.5, 2, 0.5]
    )
    headers_3 = headers([1.0, 1.0, 1.0])
    headers_5 = headers([1.0] * 5)
    cases = [
        capture("complete_metadata_and_constant_speed", consistent_points, headers_3, frame_times=[0, 10, 20]),
        capture("missing_date_makes_physical_check_unknown", consistent_points, headers([1.0] * 3, missing_date_indices=(1,)), frame_times=[0, 10, 20]),
        capture("missing_exposure_makes_physical_check_unknown", consistent_points, headers([1.0] * 3, missing_exposure_indices=(2,)), frame_times=[0, 10, 20]),
        capture("one_observation_cannot_measure_exposure_speed", consistent_points[:1], headers_3, frame_times=[0, 10, 20]),
        capture("one_transition_is_below_two_transition_minimum", consistent_points[:2], headers_3, frame_times=[0, 10, 20]),
        capture("current_edge_truncation_transition_is_omitted", points_for([0, 1, 2], [0, 10, 20], [1, 1, 1], edge_indices=(1,)), headers_3, frame_times=[0, 10, 20]),
        capture("ratio_above_allowed_speed_band_is_rejected", points_for([0, 1, 2], [0, 10, 20], [4, 4, 4]), headers_3, frame_times=[0, 10, 20]),
        capture("ratio_below_allowed_speed_band_is_rejected", points_for([0, 1, 2], [0, 10, 20], [0.2, 0.2, 0.2]), headers_3, frame_times=[0, 10, 20]),
        capture("three_of_four_transitions_meets_range_fraction_boundary", inconsistent_fraction, headers_5, frame_times=[0, 10, 20, 30, 40]),
        capture("large_log_mad_rejects_inconsistent_exposure_speed", high_mad, headers_5, frame_times=[0, 10, 20, 30, 40]),
        capture("irregular_elapsed_times_use_real_seconds", points_for([0, 1, 2, 3], [0, 10, 25, 40], [1, 1, 1, 1]), headers([1.0] * 4), frame_times=[0, 10, 25, 40]),
        capture("invalid_times_fall_back_to_frame_numbers", consistent_points, headers_3, frame_times=[0, 10, 9]),
        capture("project_exposure_keyword_is_milliseconds", consistent_points, headers([1000.0] * 3, keys=["EXPOSURE"] * 3), frame_times=[0, 10, 20]),
        capture("explicit_exptime_milliseconds_are_normalized", points_for([0, 1, 2], [0, 20, 40], [1, 1, 1]), [{"DATE-OBS": f"2024-01-01T00:00:{i:02d}Z", "EXPTIME": 500.0, "EXPTIME_UNIT": "ms"} for i in range(3)], frame_times=[0, 10, 20], psf_fwhm=2.0),
        capture("present_but_unparseable_date_is_only_metadata_presence", consistent_points, [dict(header, **{"DATE-OBS": "not-a-timestamp"}) for header in headers_3], frame_times=[0, 10, 20]),
        capture("subpixel_motion_is_ignored", points_for([0, 1, 2], [0, 0.5, 1.0], [1, 1, 1]), headers_3, frame_times=[0, 10, 20]),
        capture("psf_wider_than_measured_trail_has_no_motion_length", points_for([0, 1, 2], [0, 10, 20], [1, 1, 1]), headers_3, frame_times=[0, 10, 20], psf_fwhm=10.0),
        capture("duplicate_frame_index_does_not_form_transition", points_for([0, 1, 1], [0, 10, 10], [1, 1, 1]), headers_3, frame_times=[0, 10, 20]),
    ]
    sequence_source = ROOT / "src/rst19/sequence.py"
    fits_source = ROOT / "src/rst19/fits.py"
    fixture = {
        "source_hashes": {
            sequence_source.name: hashlib.sha256(sequence_source.read_bytes()).hexdigest(),
            fits_source.name: hashlib.sha256(fits_source.read_bytes()).hexdigest(),
        },
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_motion_feature_exposure.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
