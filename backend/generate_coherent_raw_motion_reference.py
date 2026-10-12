"""Freeze Python coherent raw motion-point chain cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import MotionFeaturePoint, _coherent_raw_motion_point_ids


def point(
    x: float,
    y: float = 0.0,
    *,
    length: float = 20.0,
    continuity: float | None = 0.9,
    fill: float | None = 0.9,
    artifact: float | None = 0.0,
    model: str | None = None,
    snr: float | None = None,
    point_bic: float | None = None,
    double_bic: float | None = None,
    background_bic: float | None = None,
    elliptical_bic: float | None = None,
    frame: int = 0,
) -> MotionFeaturePoint:
    return MotionFeaturePoint(
        frame_index=frame,
        x=x,
        y=y,
        aligned_x=x,
        aligned_y=y,
        residual_snr=20.0,
        area_pixels=30,
        length_px=length,
        width_px=3.0,
        angle_deg=0.0,
        bbox=(int(x - length / 2), int(y - 2), int(x + length / 2), int(y + 2)),
        touches_edge=False,
        continuity_ratio=continuity,
        profile_fill_ratio=fill,
        line_artifact_overlap_fraction=artifact,
        line_template_snr=snr,
        line_vs_point_delta_bic=point_bic,
        line_vs_double_delta_bic=double_bic,
        line_vs_background_delta_bic=background_bic,
        line_vs_elliptical_delta_bic=elliptical_bic,
        single_frame_model=model,
    )


def chain(
    positions: list[float],
    *,
    length: float = 20.0,
    weak_indices: set[int] | None = None,
    model: str | None = None,
    artifact: float | None = 0.0,
    continuity: float = 0.9,
    fill: float = 0.9,
) -> list[list[MotionFeaturePoint]]:
    weak_indices = weak_indices or set()
    frames = []
    for frame, x in enumerate(positions):
        weak = frame in weak_indices
        frames.append([point(
            x,
            frame=frame,
            length=length,
            continuity=0.3 if weak else continuity,
            fill=0.3 if weak else fill,
            artifact=artifact,
            model=model,
            snr=8.0 if model in {"line", "long_line_profile"} else None,
            point_bic=8.0 if model == "line" else None,
            double_bic=8.0 if model == "line" else None,
            background_bic=8.0 if model in {"line", "long_line_profile"} else None,
            elliptical_bic=8.0 if model == "line" else None,
        )])
    return frames


def capture(
    label: str,
    frames: list[list[MotionFeaturePoint]],
    *,
    psf_fwhm: float = 2.0,
    min_presence: int = 3,
    frame_times: list[float] | None = None,
    exposure_times_s: list[float] | None = None,
) -> dict[str, object]:
    selected_ids = _coherent_raw_motion_point_ids(
        frames,
        psf_fwhm=psf_fwhm,
        min_presence=min_presence,
        frame_times=frame_times,
        exposure_times_s=exposure_times_s,
    )
    selected_indices = [
        [frame_index, point_index]
        for frame_index, frame in enumerate(frames)
        for point_index, item in enumerate(frame)
        if id(item) in selected_ids
    ]
    return {
        "label": label,
        "request": {
            "frames": [[item.as_dict() for item in frame] for frame in frames],
            "psf_fwhm": psf_fwhm,
            "min_presence": min_presence,
            "frame_times": frame_times,
            "exposure_times_s": exposure_times_s,
        },
        "expected": {"selected_indices": selected_indices},
    }


def bad_exposure(label: str, frames: list[list[MotionFeaturePoint]], values: list[float]) -> dict[str, object]:
    return {
        "label": label,
        "request": {
            "frames": [[item.as_dict() for item in frame] for frame in frames],
            "psf_fwhm": 2.0,
            "min_presence": 3,
            "exposure_times_s": values,
        },
        "expected_error": (
            "exposure_times_s length must match raw_feature_frames"
            if len(values) != len(frames)
            else "exposure_times_s must contain positive finite values"
        ),
    }


def main() -> None:
    strong = chain([0.0, 4.0, 8.0])
    model_line = chain([0.0, 4.0, 8.0], length=12.0, model="line", artifact=0.9)
    weak_shape = chain([0.0, 4.0, 8.0], length=12.0, continuity=0.79, fill=0.9)
    exposure_gate = chain([0.0, 30.0, 60.0], length=8.0)
    cases = [
        capture("three_strong_adjacent_lines_form_chain", strong),
        capture("sub_psf_static_shift_is_rejected", chain([0.0, 1.99, 3.98])),
        capture("direction_reversal_breaks_chain", chain([0.0, 4.0, 0.0, -4.0])),
        capture("two_strong_points_bridge_one_weak_shape_frame", chain([0.0, 4.0, 8.0], weak_indices={1})),
        capture("weak_only_chain_is_not_admitted", chain([0.0, 4.0, 8.0], weak_indices={0, 1, 2})),
        capture("single_strong_point_cannot_seed_chain", chain([0.0, 4.0, 8.0], weak_indices={1, 2})),
        capture("strong_local_line_model_rescues_artifact_overlap", model_line),
        capture("compact_point_model_is_not_a_line_proposal", chain([0.0, 4.0, 8.0], length=20.0, model="point")),
        capture("artifact_overlap_without_model_support_is_rejected", chain([0.0, 4.0, 8.0], length=20.0, artifact=0.9)),
        capture("short_unmodeled_line_needs_eighty_percent_shape", weak_shape),
        capture("strong_model_line_uses_sequence_shape_rescue_gate", chain([0.0, 4.0, 8.0], length=12.0, model="line", artifact=0.0, continuity=0.59, fill=0.9)),
        capture("long_line_profile_requires_strong_template_evidence", chain([0.0, 4.0, 8.0], length=140.0, model="long_line_profile")),
        capture("irregular_elapsed_times_scale_motion_gates", chain([0.0, 10.0, 40.0]), frame_times=[0.0, 2.0, 8.0], exposure_times_s=[1.0, 1.0, 1.0]),
        capture("invalid_time_axis_falls_back_to_frame_indices", chain([0.0, 10.0, 40.0]), frame_times=[0.0, 2.0, 2.0], exposure_times_s=[1.0, 1.0, 1.0]),
        capture("short_exposure_allows_large_line_step", exposure_gate, exposure_times_s=[0.1, 0.1, 0.1]),
        capture("long_exposure_does_not_inflate_line_step_gate", exposure_gate, exposure_times_s=[10.0, 10.0, 10.0]),
        capture("pairwise_line_length_expands_displacement_gate", chain([0.0, 50.0, 100.0], length=20.0)),
        capture("psf_fwhm_sets_the_minimum_nonzero_step", chain([0.0, 3.0, 6.0]), psf_fwhm=4.0),
        capture("nonuniform_frames_keep_all_independent_chains", [[point(0, frame=0), point(100, frame=0)], [point(4, frame=1), point(104, frame=1)], [point(8, frame=2), point(108, frame=2)]]),
        capture("minimum_presence_two_is_inclusive", chain([0.0, 4.0]), min_presence=2),
        capture("minimum_presence_above_frame_count_returns_empty", strong, min_presence=4),
        capture("empty_sequence_returns_empty_before_option_validation", [], min_presence=3),
        capture("invalid_time_length_uses_index_fallback", chain([0.0, 4.0, 8.0]), frame_times=[100.0]),
    ]
    errors = [
        bad_exposure("exposure_vector_length_mismatch_is_rejected", strong, [1.0, 1.0]),
        bad_exposure("nonpositive_exposure_is_rejected", strong, [1.0, 0.0, 1.0]),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
        "errors": errors,
    }
    path = Path(__file__).parent / "tests/fixtures/coherent_raw_motion_points.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases, {len(errors)} error cases)")


if __name__ == "__main__":
    main()
