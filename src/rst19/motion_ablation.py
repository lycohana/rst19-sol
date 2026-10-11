"""Association-stage motion ablation over frozen cached quality detections."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from .detection import Detection
from .sequence import SourceTrack, track_detections, track_fast_point_movers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point_source(detection_id: int, x: float, y: float, *, flux_snr: float = 20.0) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=float(x),
        y=float(y),
        peak=100.0,
        flux=500.0,
        background=10.0,
        noise=2.0,
        snr=float(flux_snr),
        fwhm=2.0,
        flags=(),
        flux_error=20.0,
        flux_snr=float(flux_snr),
        filter_snr=float(flux_snr),
        quality_passed=True,
    )


def _load_frozen_cache(cache_path: Path) -> tuple[dict[str, object], list[list[Detection]], list[float], list[tuple[float, float]], list[list[tuple[float, float]]]]:
    with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
        envelope = json.load(stream)
    sequence = envelope["sequence"]
    frames = sequence["frames"]
    identities = envelope["input_identity"]["frames"]
    if len(frames) < 3 or len(frames) != len(identities):
        raise ValueError("sequence cache must contain matching frame summaries and input identities")

    frame_paths: list[Path] = []
    frame_sources: list[list[Detection]] = [[] for _ in frames]
    ids_seen: list[set[int]] = [set() for _ in frames]
    registered_background: list[list[tuple[float, float]]] = [[] for _ in frames]
    for identity in identities:
        path = Path(identity["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = _sha256(path)
        if actual_sha.lower() != str(identity.get("sha256", "")).lower():
            raise ValueError(f"cached input SHA-256 mismatch: {path}")
        frame_paths.append(path)
    for index, frame in enumerate(frames):
        if Path(frame["path"]).resolve() != frame_paths[index]:
            raise ValueError("frame order differs between sequence cache and input identity")

    quality_point_count = 0
    for track in sequence.get("tracks", []):
        if track.get("evidence_level", "quality") != "quality":
            continue
        for point in track.get("points", []):
            frame_index = int(point["frame_index"])
            detection_id = int(point["detection_id"])
            if not 0 <= frame_index < len(frames):
                raise ValueError("cached track point has an invalid frame index")
            if detection_id in ids_seen[frame_index]:
                raise ValueError(f"duplicate quality detection id in cached frame {frame_index}: {detection_id}")
            if not bool(point.get("quality_passed", False)):
                raise ValueError("cached quality track contains a point that failed its quality gate")
            flux_snr_value = point.get("flux_snr")
            if flux_snr_value is None or not math.isfinite(float(flux_snr_value)):
                raise ValueError("cached quality detection lacks a finite flux SNR")
            x, y = float(point["x"]), float(point["y"])
            detection = _point_source(detection_id, x, y, flux_snr=float(flux_snr_value))
            frame_sources[frame_index].append(detection)
            ids_seen[frame_index].add(detection_id)
            registered_background[frame_index].append((float(point["aligned_x"]), float(point["aligned_y"])))
            quality_point_count += 1
    for index, sources in enumerate(frame_sources):
        sources.sort(key=lambda item: item.detection_id)
        if len(sources) != int(frames[index]["quality_count"]):
            raise ValueError(
                f"cached quality points for frame {index} ({len(sources)}) do not match frame summary "
                f"({frames[index]['quality_count']})"
            )
    shifts = [(float(item[0]), float(item[1])) for item in sequence["cumulative_shifts"]]
    if len(shifts) != len(frames):
        raise ValueError("cached cumulative shift count differs from frame count")
    timestamps = [datetime.fromisoformat(str(frame["timestamp"])) for frame in frames]
    origin = timestamps[0]
    frame_times = [(stamp - origin).total_seconds() for stamp in timestamps]
    if any(right <= left for left, right in pairwise(frame_times)):
        raise ValueError("cached frame timestamps are not strictly increasing")
    metadata = {
        "envelope": envelope,
        "sequence": sequence,
        "frames": frames,
        "frame_paths": frame_paths,
        "quality_point_count": quality_point_count,
        "identities": identities,
    }
    return metadata, frame_sources, frame_times, shifts, registered_background


def _make_truth_paths(
    width: int,
    height: int,
    frame_count: int,
    frame_times: Sequence[float],
    cached_shifts: Sequence[tuple[float, float]],
    background_points: Sequence[Sequence[tuple[float, float]]],
    *,
    seed: int,
    avoid_radius_px: float = 18.0,
) -> list[dict[str, object]]:
    specifications = [
        {"target_id": "FAST-15", "vx_px_frame": 8.0, "vy_px_frame": -4.0, "missing_frames": [], "role": "fast positive, 15/15"},
        {"target_id": "FAST-GAP-12", "vx_px_frame": 6.0, "vy_px_frame": 2.0, "missing_frames": [3, 7, 11], "role": "fast positive with three separated gaps, 12/15"},
        {"target_id": "SLOW-BASE-15", "vx_px_frame": 2.0, "vy_px_frame": 0.5, "missing_frames": [], "role": "slow positive control for the 4 px baseline"},
        {"target_id": "BELOW-12", "vx_px_frame": -7.0, "vy_px_frame": -3.0, "missing_frames": [2, 5, 8, 11], "role": "presence negative control, 11/15"},
    ]
    trees = [cKDTree(np.asarray(points, dtype=np.float64)) if points else None for points in background_points]
    rng = random.Random(seed)
    accepted: list[dict[str, object]] = []
    for spec in specifications:
        vx, vy = float(spec["vx_px_frame"]), float(spec["vy_px_frame"])
        missing = {int(index) for index in spec["missing_frames"]}
        accepted_path = None
        for _attempt in range(50_000):
            start_x = rng.uniform(180.0, width - 180.0)
            start_y = rng.uniform(180.0, height - 180.0)
            registered = [
                (start_x + vx * frame, start_y + vy * frame)
                for frame in range(frame_count)
            ]
            if any(
                not (60.0 <= x + cached_shifts[i][0] <= width - 60.0 and 60.0 <= y + cached_shifts[i][1] <= height - 60.0)
                for i, (x, y) in enumerate(registered)
            ):
                continue
            contaminated = False
            for i, (x, y) in enumerate(registered):
                if i in missing:
                    continue
                if trees[i] is not None and trees[i].query((x, y), distance_upper_bound=avoid_radius_px)[0] < avoid_radius_px:
                    contaminated = True
                    break
            if contaminated:
                continue
            too_close = False
            for prior in accepted:
                prior_missing = set(prior["missing_frames"])
                for i, (x, y) in enumerate(registered):
                    if i in missing or i in prior_missing:
                        continue
                    px, py = prior["registered_positions"][i]
                    if math.hypot(x - float(px), y - float(py)) < 50.0:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue
            accepted_path = registered
            break
        if accepted_path is None:
            raise RuntimeError(f"could not find clean path for {spec['target_id']}")
        synthetic_id = 1_000_000 + len(accepted)
        expected = []
        for frame_index, (aligned_x, aligned_y) in enumerate(accepted_path):
            if frame_index in missing:
                continue
            shift_x, shift_y = cached_shifts[frame_index]
            expected.append({
                "frame_index": frame_index,
                "time_seconds": float(frame_times[frame_index]),
                "registered_x_px": aligned_x,
                "registered_y_px": aligned_y,
                "raw_x_px": aligned_x + shift_x,
                "raw_y_px": aligned_y + shift_y,
                "detection_id": synthetic_id,
            })
        accepted.append({
            **spec,
            "detection_id": synthetic_id,
            "expected_presence": len(expected),
            "registered_positions": accepted_path,
            "expected_observations": expected,
            "flux_snr": 20.0,
        })
    return accepted


def _append_truth_sources(
    frame_sources: Sequence[Sequence[Detection]], truth_paths: Sequence[dict[str, object]]
) -> list[list[Detection]]:
    result = [list(sources) for sources in frame_sources]
    for truth in truth_paths:
        for point in truth["expected_observations"]:
            result[int(point["frame_index"])].append(
                _point_source(
                    int(point["detection_id"]),
                    float(point["raw_x_px"]),
                    float(point["raw_y_px"]),
                    flux_snr=float(truth["flux_snr"]),
                )
            )
    for sources in result:
        sources.sort(key=lambda item: item.detection_id)
    return result


def _target_tracks(tracks: Sequence[SourceTrack], target_detection_id: int, expected_frames: set[int]) -> dict[str, object]:
    containing = [track for track in tracks if any(point.detection_id == target_detection_id for point in track.points)]
    best = max(containing, key=lambda track: sum(point.detection_id == target_detection_id for point in track.points), default=None)
    if best is None:
        return {
            "track_count": 0,
            "max_target_points_in_one_track": 0,
            "covered_expected_frames": 0,
            "exact_track_recovered": False,
            "best_track_classification": "",
            "mixed_with_other_injected_target": False,
            "mixed_with_background_source": False,
        }
    target_points = [point for point in best.points if point.detection_id == target_detection_id]
    other_ids = {point.detection_id for track in containing for point in track.points if point.detection_id != target_detection_id}
    target_frames = {point.frame_index for point in target_points}
    exact = target_frames == expected_frames and len(target_points) == len(expected_frames) and len(containing) == 1
    return {
        "track_count": len(containing),
        "max_target_points_in_one_track": len(target_points),
        "covered_expected_frames": len(target_frames & expected_frames),
        "exact_track_recovered": exact,
        "best_track_classification": best.classification,
        "mixed_with_other_injected_target": any(1_000_000 <= item < 1_000_100 for item in other_ids),
        "mixed_with_background_source": any(item < 1_000_000 for item in other_ids),
    }


def _plot_track_points(
    draw: ImageDraw.ImageDraw,
    tracks: Sequence[SourceTrack],
    detection_id: int,
    point_color: tuple[int, int, int],
    connect: bool,
    *,
    start: Sequence[float],
    vx: float,
    vy: float,
    norm: float,
    bottom: int,
    left: int,
    right: int,
    frame_count: int,
) -> None:
    points = [point for track in tracks for point in track.points if point.detection_id == detection_id]
    points.sort(key=lambda point: point.frame_index)
    drawn = []
    for point in points:
        frame = point.frame_index
        if frame >= frame_count:
            continue
        projected = ((point.aligned_x - start[0]) * vx + (point.aligned_y - start[1]) * vy) / norm
        px = left + (right - left) * frame / max(1, frame_count - 1)
        py = bottom - 15 - (projected + 12.0) / 160.0 * 92.0
        drawn.append((px, py))
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=point_color, outline=(255, 255, 255), width=1)
    if connect and len(drawn) > 1:
        draw.line(drawn, fill=point_color, width=2)


def _save_track_plot(path: Path, truth_paths: Sequence[dict[str, object]], baseline: Sequence[SourceTrack], fast: Sequence[SourceTrack], frame_count: int) -> None:
    width, height = 1050, max(640, 180 * len(truth_paths) + 70)
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("segoeui.ttf", 14)
        small = ImageFont.truetype("segoeui.ttf", 12)
    except OSError:
        font = small = ImageFont.load_default()
    left, right = 190, width - 40
    for index, truth in enumerate(truth_paths):
        top = 36 + index * 180
        bottom = top + 132
        draw.text((22, top + 4), str(truth["target_id"]), fill=(28, 39, 54), font=font)
        draw.text((22, top + 28), str(truth["role"]), fill=(85, 99, 117), font=small)
        draw.line((left, bottom, right, bottom), fill=(170, 180, 192), width=1)
        draw.line((left, top, left, bottom), fill=(170, 180, 192), width=1)
        for frame in range(frame_count):
            px = left + (right - left) * frame / max(1, frame_count - 1)
            draw.line((px, bottom - 3, px, bottom + 3), fill=(170, 180, 192), width=1)
            if index == len(truth_paths) - 1:
                draw.text((px - 5, bottom + 7), str(frame), fill=(91, 102, 118), font=small)
        expected = {int(point["frame_index"]): point for point in truth["expected_observations"]}
        projected_truth = []
        start = truth["registered_positions"][0]
        vx, vy = float(truth["vx_px_frame"]), float(truth["vy_px_frame"])
        norm = max(1e-9, math.hypot(vx, vy))
        for frame in range(frame_count):
            pos = truth["registered_positions"][frame]
            projected = ((pos[0] - start[0]) * vx + (pos[1] - start[1]) * vy) / norm
            px = left + (right - left) * frame / max(1, frame_count - 1)
            py = bottom - 15 - (projected + 12.0) / 160.0 * 92.0
            if frame in expected:
                projected_truth.append((px, py))
        if len(projected_truth) > 1:
            draw.line(projected_truth, fill=(250, 189, 45), width=3)

        detection_id = int(truth["detection_id"])
        _plot_track_points(
            draw, baseline, detection_id, (223, 105, 62), False,
            start=start, vx=vx, vy=vy, norm=norm, bottom=bottom,
            left=left, right=right, frame_count=frame_count,
        )
        _plot_track_points(
            draw, fast, detection_id, (24, 151, 108), True,
            start=start, vx=vx, vy=vy, norm=norm, bottom=bottom,
            left=left, right=right, frame_count=frame_count,
        )
        draw.text((left, top + 4), "Along-track displacement (px)", fill=(75, 89, 107), font=small)
    draw.text((left, height - 24), "Time: frame index | Yellow = truth | Orange = baseline | Green = supplement", fill=(65, 78, 96), font=small)
    image.save(path)


def run_motion_association_ablation(
    cache_path: str | Path,
    out_dir: str | Path,
    *,
    seed: int = 19019,
    min_presence: int | None = None,
    link_radius_px: float | None = None,
    registration_radius_px: float = 8.0,
    min_flux_snr: float = 7.0,
    max_step_px: float = 20.0,
    prediction_gate_px: float = 4.5,
    max_fit_rms_px: float = 1.5,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    cache = Path(cache_path).resolve()
    output = Path(out_dir).resolve()
    if not cache.is_file():
        raise FileNotFoundError(cache)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache_hash = _sha256(cache)
    metadata, background_sources, frame_times, cached_shifts, registered_background = _load_frozen_cache(cache)
    sequence = metadata["sequence"]
    frames = metadata["frames"]
    frame_paths = metadata["frame_paths"]
    actual_min_presence = int(sequence["min_presence"] if min_presence is None else min_presence)
    actual_link_radius = float(sequence["link_radius_px"] if link_radius_px is None else link_radius_px)
    if actual_min_presence < 3 or actual_min_presence > len(frames):
        raise ValueError("minimum presence must be within the number of cached frames")
    truth_paths = _make_truth_paths(
        int(frames[0]["width_px"]),
        int(frames[0]["height_px"]),
        len(frames),
        frame_times,
        cached_shifts,
        registered_background,
        seed=seed,
    )
    injected_sources = _append_truth_sources(background_sources, truth_paths)

    # Persist exactly the per-frame source records consumed by both association versions.
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["frame_index", "detection_id", "x_px", "y_px", "flux_snr", "quality_passed", "synthetic_target_id"])
    synthetic_by_detection = {int(item["detection_id"]): str(item["target_id"]) for item in truth_paths}
    for frame_index, sources in enumerate(injected_sources):
        for source in sources:
            writer.writerow([
                frame_index,
                source.detection_id,
                f"{source.x:.9f}",
                f"{source.y:.9f}",
                f"{float(source.flux_snr or 0.0):.6f}",
                int(bool(source.quality_passed)),
                synthetic_by_detection.get(source.detection_id, ""),
            ])
    frozen_csv = output / "frozen_frame_sources.csv.gz"
    frozen_csv.write_bytes(gzip.compress(buffer.getvalue().encode("utf-8"), compresslevel=6, mtime=0))
    (output / "truth_trajectories.json").write_text(
        json.dumps(truth_paths, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    if progress:
        progress("运行基础 4 px 最近邻关联")
    started = time.perf_counter()
    baseline = track_detections(
        injected_sources,
        frame_names=[str(path) for path in frame_paths],
        frame_times=frame_times,
        link_radius_px=actual_link_radius,
        min_presence=actual_min_presence,
        motion_min_displacement_px=float(sequence["motion_min_displacement_px"]),
        registration_radius_px=float(registration_radius_px),
        max_motion_fit_rms_px=float(sequence["max_motion_fit_rms_px"]),
        persistent_min_presence=(int(sequence["persistent_min_presence"]) if sequence.get("persistent_min_presence") else None),
    )
    baseline_seconds = time.perf_counter() - started
    if progress:
        progress("在同一逐帧检测列表上运行速度预测补充关联")
    started = time.perf_counter()
    fast_tracks = track_fast_point_movers(
        injected_sources,
        baseline.cumulative_shifts,
        frame_times=frame_times,
        link_radius_px=actual_link_radius,
        min_presence=actual_min_presence,
        motion_min_displacement_px=float(sequence["motion_min_displacement_px"]),
        min_flux_snr=float(min_flux_snr),
        max_step_px=float(max_step_px),
        gate_px=float(prediction_gate_px),
        max_fit_rms_px=float(max_fit_rms_px),
        existing_tracks=baseline.tracks,
    )
    fast_seconds = time.perf_counter() - started

    summary_rows: list[dict[str, object]] = []
    fast_summary_rows: list[dict[str, object]] = []
    for truth in truth_paths:
        target_id = str(truth["target_id"])
        detection_id = int(truth["detection_id"])
        expected_frames = {int(point["frame_index"]) for point in truth["expected_observations"]}
        base_metrics = _target_tracks(baseline.tracks, detection_id, expected_frames)
        fast_metrics = _target_tracks(fast_tracks, detection_id, expected_frames)
        combined_metrics = _target_tracks((*baseline.tracks, *fast_tracks), detection_id, expected_frames)
        row = {
            "target_id": target_id,
            "role": truth["role"],
            "expected_presence": truth["expected_presence"],
            "sequence_frames": len(frames),
            "baseline_tracks_containing_target": base_metrics["track_count"],
            "baseline_best_track_target_points": base_metrics["max_target_points_in_one_track"],
            "baseline_target_frame_coverage": base_metrics["covered_expected_frames"],
            "baseline_exact_track_recovered": base_metrics["exact_track_recovered"],
            "baseline_fragments": base_metrics["track_count"],
            "baseline_mixed_with_other_injected_target": base_metrics["mixed_with_other_injected_target"],
            "baseline_mixed_with_background_source": base_metrics["mixed_with_background_source"],
            "fast_supplement_tracks_containing_target": fast_metrics["track_count"],
            "fast_best_track_target_points": fast_metrics["max_target_points_in_one_track"],
            "fast_target_frame_coverage": fast_metrics["covered_expected_frames"],
            "fast_exact_track_recovered": fast_metrics["exact_track_recovered"],
            "fast_track_classification": fast_metrics["best_track_classification"],
            "fast_mixed_with_other_injected_target": fast_metrics["mixed_with_other_injected_target"],
            "fast_mixed_with_background_source": fast_metrics["mixed_with_background_source"],
            "combined_exact_track_recovered": combined_metrics["exact_track_recovered"],
            "expected_frame_indices": ",".join(str(i) for i in sorted(expected_frames)),
            "synthetic_detection_id": detection_id,
        }
        summary_rows.append(row)
        fast_summary_rows.append({
            "target_id": target_id,
            "expected_presence": truth["expected_presence"],
            "baseline_exact_track_recovered": base_metrics["exact_track_recovered"],
            "fast_exact_track_recovered": fast_metrics["exact_track_recovered"],
            "combined_exact_track_recovered": combined_metrics["exact_track_recovered"],
            "baseline_fragments": base_metrics["track_count"],
            "fast_supplement_track_count": fast_metrics["track_count"],
            "fast_mixed_with_background_source": fast_metrics["mixed_with_background_source"],
        })

    summary_path = output / "trajectory_results.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "motion_pairing.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fast_summary_rows[0]))
        writer.writeheader()
        writer.writerows(fast_summary_rows)

    _save_track_plot(output / "motion_tracks.png", truth_paths, baseline.tracks, fast_tracks, len(frames))
    registration_deltas = [
        math.hypot(baseline.cumulative_shifts[i][0] - cached_shifts[i][0], baseline.cumulative_shifts[i][1] - cached_shifts[i][1])
        for i in range(len(frames))
    ]
    manifest = {
        "schema_version": 1,
        "experiment": "association_stage_motion_constraint_ablation",
        "status": "completed",
        "created_local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_cache": {
            "path": str(cache),
            "sha256": cache_hash,
            "cache_version": metadata["envelope"].get("cache_version"),
            "sequence_cache_version": metadata["envelope"].get("sequence_cache_version"),
            "quality_track_points_rebuilt": metadata["quality_point_count"],
            "background_source_counts_by_frame": [len(sources) for sources in background_sources],
            "source_hashes_verified": len(metadata["identities"]),
            "source_inputs": [
                {"path": str(path), "sha256": identity["sha256"], "size_bytes": identity["size"]}
                for path, identity in zip(frame_paths, metadata["identities"])
            ],
        },
        "frozen_association_input": {
            "path": str(frozen_csv),
            "sha256": _sha256(frozen_csv),
            "truth_path_file": str(output / "truth_trajectories.json"),
            "truth_path_sha256": _sha256(output / "truth_trajectories.json"),
            "dtype": "Detection objects reconstructed from cached quality points; synthetic points appended",
            "coordinate_rule": "synthetic raw position = frozen registered truth position + cached cumulative shift",
            "all_quality_source_flux_snr_verified": True,
        },
        "frame_times_seconds": frame_times,
        "cached_cumulative_shifts_px": [list(shift) for shift in cached_shifts],
        "recomputed_registration_max_delta_px": max(registration_deltas, default=0.0),
        "truth_trajectories": [
            {key: value for key, value in truth.items() if key != "registered_positions"}
            for truth in truth_paths
        ],
        "versions": [
            {"id": "A", "name": "普通质量点源关联", "function": "track_detections", "link_radius_px": actual_link_radius},
            {"id": "B", "name": "A + 速度预测补充关联", "function": "track_fast_point_movers", "existing_tracks": "A tracks reserved; no already-linked multiframe source reused"},
        ],
        "parameters": {
            "link_radius_px": actual_link_radius,
            "min_presence_frames": actual_min_presence,
            "motion_min_displacement_px_from_cache": float(sequence["motion_min_displacement_px"]),
            "registration_radius_px": float(registration_radius_px),
            "max_motion_fit_rms_px_baseline": float(sequence["max_motion_fit_rms_px"]),
            "supplement_min_flux_snr": float(min_flux_snr),
            "supplement_max_step_px_per_reference_interval": float(max_step_px),
            "supplement_prediction_gate_px": float(prediction_gate_px),
            "supplement_max_fit_rms_px": float(max_fit_rms_px),
            "truth_coordinate_avoid_radius_px": 18.0,
            "truth_position_seed": seed,
            "truth_sample_types": "15/15 fast, 12/15 separated gaps, 15/15 slow baseline positive, 11/15 below-presence negative",
        },
        "association_metrics": {
            "exact_track": "one output track contains exactly the target's expected detection IDs on every expected frame, with no other frame for that target, and no additional track contains its ID",
            "fragment_count": "number of output tracks containing the target detection ID; baseline singletons count as fragments",
            "mixed_target_ids": "one track contains synthetic detection IDs from more than one known truth trajectory",
            "background_contamination": "one track containing a synthetic target ID also contains a cached background detection ID",
            "negative_control": "the 11/15 path should not produce a fast supplement because min_presence is 12; it is not a false-positive estimate",
        },
        "runtime": {"baseline_seconds": baseline_seconds, "supplement_seconds": fast_seconds},
        "outputs": {
            "trajectory_csv": str(summary_path),
            "pairing_csv": str(output / "motion_pairing.csv"),
            "frozen_sources_csv_gz": str(frozen_csv),
            "truth_json": str(output / "truth_trajectories.json"),
            "track_plot": str(output / "motion_tracks.png"),
        },
        "limitations": [
            "This is an association-stage test: artificial Detection records are appended to a frozen quality-source list.",
            "It does not test whether the pixel detector can discover these targets or whether the target passes single-frame image quality measurement.",
            "The background quality detections are reconstructed from the saved SequenceResult cache and revalidated against all 15 raw FITS SHA-256 identities.",
            "Track recovery applies only to the four injected paths and does not establish physical moving-object truth or full-sequence association precision.",
            "A crossing-track adversarial case is not included because the current supplement intentionally cannot replace already linked multiframe baseline tracks.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


__all__ = ["run_motion_association_ablation"]
