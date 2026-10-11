"""Paired pixel-level ablation for sequence multiscale motion-line proposals."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detection import DetectionResult
from .fits import FitsFrame
from .pipeline import FrameAnalysis
from .sequence import MotionFeatureTrack, detect_motion_features


_FRAME_COUNT = 4
_IMAGE_SHAPE = (96, 96)
_BACKGROUND_ADU = 10.0
_NOISE_SIGMA_ADU = 1.0
_PSF_FWHM_PX = 3.0
_LINE_LENGTH_PX = 12.0
_LINE_SAMPLE_AMPLITUDE_ADU = 4.0
_FRAME_STEP_PX = 10.0
_TRUTH_Y_PX = 40.0
_TRUTH_START_X_PX = 16.0
_MATCH_RADIUS_PX = 7.0
_DETECTOR_OPTIONS = {
    "psf_fwhm": _PSF_FWHM_PX,
    "residual_sigma": 15.0,
    "min_residual_adu": 100.0,
    "min_feature_area": 40,
    "min_axis_ratio": 4.0,
    "min_length_px": 8.0,
    "bridge_gaps": True,
    "raw_fallback_workers": 1,
}


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _add_finite_exposure_trail(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length_px: float,
    amplitude_adu: float,
    psf_sigma_px: float,
) -> None:
    yy, xx = np.indices(image.shape, dtype=np.float64)
    offsets = np.arange(-0.5 * length_px, 0.5 * length_px + 0.25, 0.5)
    for offset in offsets:
        source_x = center_x + offset
        image += (
            amplitude_adu
            * np.exp(-0.5 * (((xx - source_x) ** 2 + (yy - center_y) ** 2) / (psf_sigma_px**2)))
        ).astype(np.float32) * 0.5


def _as_frame_analyses(images: Sequence[np.ndarray], prefix: str) -> tuple[FrameAnalysis, ...]:
    analyses = []
    for frame_index, image in enumerate(images):
        detection = DetectionResult(
            image_shape=image.shape,
            background=_BACKGROUND_ADU,
            noise=_NOISE_SIGMA_ADU,
            threshold=100.0,
            candidate_count=0,
            sources=(),
            parameters={},
            quality_count=0,
        )
        frame = FitsFrame(Path(f"{prefix}-{frame_index:02d}.fits"), {}, image, None, 0)
        analyses.append(FrameAnalysis(frame, detection, None, None))
    return tuple(analyses)


def _run_version(
    analyses: Sequence[FrameAnalysis],
    *,
    multiscale: bool,
) -> tuple[tuple[MotionFeatureTrack, ...], tuple[object, ...], float]:
    audits: list[object] = []
    started = time.perf_counter()
    tracks, threshold = detect_motion_features(
        analyses,
        ((0.0, 0.0),) * len(analyses),
        temporal_reference=np.full(_IMAGE_SHAPE, _BACKGROUND_ADU, dtype=np.float32),
        multiscale_line_proposals=multiscale,
        audit_sink=audits,  # type: ignore[arg-type]
        **_DETECTOR_OPTIONS,
    )
    return tracks, tuple(audits), time.perf_counter() - started


def _matched_frame_points(
    tracks: Sequence[MotionFeatureTrack],
    truth_centers: Sequence[tuple[float, float]],
    radius_px: float,
) -> tuple[dict[int, tuple[float, float]], list[tuple[float, float]]]:
    matched: dict[int, tuple[float, float]] = {}
    all_errors = []
    for frame_index, (truth_x, truth_y) in enumerate(truth_centers):
        nearby = sorted(
            (
                math.hypot(float(point.x) - truth_x, float(point.y) - truth_y),
                float(point.x),
                float(point.y),
            )
            for track in tracks
            if track.classification == "moving"
            for point in track.points
            if int(point.frame_index) == frame_index
        )
        if nearby and nearby[0][0] <= radius_px:
            error, x, y = nearby[0]
            matched[frame_index] = (x, y)
            all_errors.append((error, float(frame_index)))
    return matched, all_errors


def _moving_track_count_covering_truth(
    tracks: Sequence[MotionFeatureTrack],
    truth_centers: Sequence[tuple[float, float]],
    radius_px: float,
) -> int:
    """Count moving tracks with one matched point in every truth frame."""

    return sum(
        track.classification == "moving"
        and all(
            any(
                int(point.frame_index) == frame_index
                and math.hypot(
                    float(point.x) - truth_centers[frame_index][0],
                    float(point.y) - truth_centers[frame_index][1],
                ) <= radius_px
                for point in track.points
            )
            for frame_index in range(len(truth_centers))
        )
        for track in tracks
    )


def _save_case_sheet(
    path: Path,
    images: np.ndarray,
    truth_centers: Sequence[tuple[float, float]],
    base_tracks: Sequence[MotionFeatureTrack],
    multiscale_tracks: Sequence[MotionFeatureTrack],
) -> None:
    tile = 240
    image = Image.new("RGB", (tile * 2, tile * 2 + 46), (24, 31, 42))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("segoeui.ttf", 14)
        small = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        font = small = ImageFont.load_default()
    for frame_index, frame in enumerate(images):
        row, col = divmod(frame_index, 2)
        x0, y0 = col * tile, row * tile
        lo, hi = np.percentile(frame, [5, 99.8])
        gray = np.clip((frame - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        rendered = Image.fromarray(np.asarray(gray * 255.0, dtype=np.uint8), mode="L").convert("RGB")
        rendered = rendered.resize((tile - 18, tile - 34), Image.Resampling.NEAREST)
        image.paste(rendered, (x0 + 9, y0 + 22))
        draw.text((x0 + 8, y0 + 4), f"F{frame_index}", fill=(235, 241, 249), font=font)
        scale_x = rendered.width / frame.shape[1]
        scale_y = rendered.height / frame.shape[0]

        def marker(x: float, y: float, color: tuple[int, int, int], radius: int = 5) -> None:
            px, py = x0 + 9 + x * scale_x, y0 + 22 + y * scale_y
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=2)

        truth_x, truth_y = truth_centers[frame_index]
        marker(truth_x, truth_y, (255, 222, 80), 6)
        for track in base_tracks:
            for point in track.points:
                if int(point.frame_index) == frame_index:
                    marker(float(point.x), float(point.y), (78, 168, 255), 4)
        for track in multiscale_tracks:
            for point in track.points:
                if int(point.frame_index) == frame_index:
                    marker(float(point.x), float(point.y), (72, 224, 161), 4)
    draw.text(
        (10, tile * 2 + 12),
        "Yellow = truth | Blue = baseline | Green = multiscale proposals",
        fill=(235, 241, 249),
        font=small,
    )
    image.save(path)


def run_motion_proposal_ablation(out_dir: str | Path, *, seed: int = 19019) -> dict[str, object]:
    """Run paired no-injection and low-peak pixel-level motion proposal controls."""

    output = Path(out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    noise_frames = rng.normal(
        _BACKGROUND_ADU,
        _NOISE_SIGMA_ADU,
        size=(_FRAME_COUNT, *_IMAGE_SHAPE),
    ).astype(np.float32)
    control_frames = noise_frames.copy()
    injected_frames = noise_frames.copy()
    psf_sigma = _PSF_FWHM_PX / 2.354820045
    truth_centers: list[tuple[float, float]] = []
    for frame_index in range(_FRAME_COUNT):
        center_x = _TRUTH_START_X_PX + 6.0 + _FRAME_STEP_PX * frame_index
        truth_centers.append((center_x, _TRUTH_Y_PX))
        _add_finite_exposure_trail(
            injected_frames[frame_index],
            center_x=center_x,
            center_y=_TRUTH_Y_PX,
            length_px=_LINE_LENGTH_PX,
            amplitude_adu=_LINE_SAMPLE_AMPLITUDE_ADU,
            psf_sigma_px=psf_sigma,
        )

    input_path = output / "frozen_motion_frames.npz"
    np.savez_compressed(input_path, control_frames=control_frames, injected_frames=injected_frames)
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    run_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    positive_tracks: dict[bool, tuple[MotionFeatureTrack, ...]] = {}
    runtime_by_case: dict[str, float] = {}

    for scene_name, frames in (("no_injection_control", control_frames), ("low_peak_moving_trail", injected_frames)):
        analyses = _as_frame_analyses(frames, scene_name)
        scene_hash = _array_sha256(frames)
        for multiscale in (False, True):
            tracks, audits, runtime_seconds = _run_version(analyses, multiscale=multiscale)
            version = "B 多尺度运动线提案" if multiscale else "A 基础线提案"
            runtime_by_case[f"{scene_name}:{'B' if multiscale else 'A'}"] = runtime_seconds
            audit_by_frame = {int(item.frame_index): item for item in audits}
            for frame_index in range(_FRAME_COUNT):
                audit = audit_by_frame[frame_index]
                audit_rows.append(
                    {
                        "scene": scene_name,
                        "version": version,
                        "frame_index": frame_index,
                        "feature_count": int(audit.feature_count),
                        "component_count": int(audit.component_count),
                        "support_pixel_count": int(audit.support_pixel_count),
                        "threshold_adu": float(audit.threshold_adu),
                        "max_feature_residual_snr": audit.max_feature_residual_snr,
                    }
                )
            for track_index, track in enumerate(tracks, start=1):
                track_rows.append(
                    {
                        "scene": scene_name,
                        "version": version,
                        "track_index": track_index,
                        "classification": track.classification,
                        "presence": int(track.presence),
                        "point_count": len(track.points),
                        "evidence_level": track.evidence_level,
                        "median_residual_snr": float(np.median([point.residual_snr for point in track.points]))
                        if track.points
                        else None,
                    }
                )
            if scene_name == "no_injection_control":
                run_rows.append(
                    {
                        "scene": scene_name,
                        "version": version,
                        "frame_count": _FRAME_COUNT,
                        "input_array_sha256": scene_hash,
                        "proposal_feature_observations": sum(int(item.feature_count) for item in audits),
                        "moving_track_count": sum(track.classification == "moving" for track in tracks),
                        "target_recovered_frames": "",
                        "target_recovery": "",
                        "median_position_error_px": "",
                        "p90_position_error_px": "",
                        "runtime_seconds": runtime_seconds,
                    }
                )
            else:
                positive_tracks[multiscale] = tracks
                matched, errors = _matched_frame_points(tracks, truth_centers, _MATCH_RADIUS_PX)
                matched_errors = [error for error, _frame in errors]
                recovered = len(matched) == _FRAME_COUNT
                target_track_count = _moving_track_count_covering_truth(
                    tracks, truth_centers, _MATCH_RADIUS_PX
                )
                run_rows.append(
                    {
                        "scene": scene_name,
                        "version": version,
                        "frame_count": _FRAME_COUNT,
                        "input_array_sha256": scene_hash,
                        "proposal_feature_observations": sum(int(item.feature_count) for item in audits),
                        "moving_track_count": sum(track.classification == "moving" for track in tracks),
                        "target_recovered_frames": f"{len(matched)}/{_FRAME_COUNT}",
                        "target_recovery": bool(recovered and target_track_count == 1),
                        "median_position_error_px": float(np.median(matched_errors)) if matched_errors else None,
                        "p90_position_error_px": float(np.percentile(matched_errors, 90)) if matched_errors else None,
                        "runtime_seconds": runtime_seconds,
                    }
                )

    truth_rows = [
        {"target_id": "LOW-PEAK-01", "frame_index": frame_index, "x_px": x, "y_px": y, "visible": True}
        for frame_index, (x, y) in enumerate(truth_centers)
    ]
    summary_path = output / "summary.csv"
    audit_path = output / "frame_audit.csv"
    tracks_path = output / "track_results.csv"
    truth_path = output / "truth_trajectory.csv"
    _write_csv(summary_path, run_rows)
    _write_csv(audit_path, audit_rows)
    _write_csv(tracks_path, track_rows)
    _write_csv(truth_path, truth_rows)
    case_sheet_path = output / "motion_proposal_cases.png"
    _save_case_sheet(case_sheet_path, injected_frames, truth_centers, positive_tracks[False], positive_tracks[True])
    manifest = {
        "schema_version": 1,
        "experiment": "synthetic_pixel_level_motion_multiscale_proposal_ablation",
        "status": "completed",
        "created_local_time": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "seed": seed,
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "dtype": str(injected_frames.dtype),
            "shape_per_scene": list(injected_frames.shape),
            "control_array_sha256": _array_sha256(control_frames),
            "injected_array_sha256": _array_sha256(injected_frames),
            "paired_noise": True,
            "frames": _FRAME_COUNT,
        },
        "truth": {
            "target_id": "LOW-PEAK-01",
            "type": "synthetic finite-exposure horizontal Gaussian-PSF trail",
            "positions_px": [{"frame_index": index, "x_px": x, "y_px": y} for index, (x, y) in enumerate(truth_centers)],
            "line_length_px": _LINE_LENGTH_PX,
            "line_kernel_coefficient_adu": _LINE_SAMPLE_AMPLITUDE_ADU,
            "signal_definition": (
                "each 0.5 px line sample adds 0.5 times the kernel coefficient times a Gaussian PSF; "
                "the coefficient is not the sampled peak, which is recorded per frame below"
            ),
            "sampled_peak_pixel_increment_adu_by_frame": [
                float(np.max(injected_frames[index] - control_frames[index]))
                for index in range(_FRAME_COUNT)
            ],
            "frame_to_frame_step_px": _FRAME_STEP_PX,
        },
        "parameters": {
            "background_adu": _BACKGROUND_ADU,
            "noise_sigma_adu": _NOISE_SIGMA_ADU,
            "psf_fwhm_px": _PSF_FWHM_PX,
            "match_radius_px": _MATCH_RADIUS_PX,
            "same_detector_options": _DETECTOR_OPTIONS,
            "ablation_variable": "multiscale_line_proposals=False (A) vs True (B)",
            "temporal_reference": "constant shared background image, same for every frame and version",
            "association_and_shape_gates": "unchanged production detect_motion_features path",
        },
        "metrics": {
            "proposal_feature_observations": "sum of per-frame final MotionFrameAudit.feature_count; not a full-frame false-positive rate",
            "target_recovery": "one moving track has a proposal within the fixed 7 px truth tolerance in all 4 frames",
            "position_error": "distance to the injected trail center among matched moving points",
            "negative_control": "paired noise frames with no injected trail; count observed outputs but do not infer population precision",
        },
        "runtime": {"seconds_by_scene_and_version": runtime_by_case},
        "outputs": {
            "summary_csv": str(summary_path),
            "frame_audit_csv": str(audit_path),
            "track_results_csv": str(tracks_path),
            "truth_trajectory_csv": str(truth_path),
            "case_sheet": str(case_sheet_path),
        },
        "limitations": [
            "This is a small synthetic pixel-level mechanism test, not a real FITS motion-source recovery test.",
            "No crossing or crowded-trajectory identity test is included.",
            "The negative control is four paired noise frames and is not an estimate of full-frame false-positive rate.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["run_motion_proposal_ablation"]
