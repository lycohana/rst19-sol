"""Shared-input point-source ablation with per-target evidence exports."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import shutil
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from .detection import Detection, DetectionResult, detect_sources
from .experiments import _gaussian_source, _select_real_background_positions
from .fits import FitsFrame, auxiliary_mask, read_fits


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _match_one_to_one(
    truths: Sequence[dict[str, object]], detections: Sequence[Detection], radius_px: float
) -> tuple[dict[str, Detection], dict[str, int]]:
    """Greedily assign the shortest in-radius pairs, with each source used once."""

    if radius_px <= 0:
        raise ValueError("match radius must be positive")
    if not detections or not truths:
        return {}, {str(item["target_id"]): 0 for item in truths}
    points = np.asarray([(item.x, item.y) for item in detections], dtype=np.float64)
    tree = cKDTree(points)
    edges: list[tuple[float, str, int]] = []
    local_counts: dict[str, int] = {}
    for truth in truths:
        target_id = str(truth["target_id"])
        xy = (float(truth["x_px"]), float(truth["y_px"]))
        nearby = tree.query_ball_point(xy, r=radius_px)
        local_counts[target_id] = len(nearby)
        for index in nearby:
            detection = detections[index]
            edges.append((math.hypot(detection.x - xy[0], detection.y - xy[1]), target_id, index))
    assigned_truths: set[str] = set()
    assigned_detections: set[int] = set()
    matches: dict[str, Detection] = {}
    for _distance, target_id, index in sorted(edges):
        if target_id in assigned_truths or index in assigned_detections:
            continue
        matches[target_id] = detections[index]
        assigned_truths.add(target_id)
        assigned_detections.add(index)
    return matches, local_counts


def _detection_record(detection: Detection | None, x: float, y: float) -> dict[str, object]:
    if detection is None:
        return {"detection_id": None, "x_px": None, "y_px": None, "error_px": None, "quality_passed": None, "flags": ""}
    return {
        "detection_id": detection.detection_id,
        "x_px": float(detection.x),
        "y_px": float(detection.y),
        "error_px": float(math.hypot(detection.x - x, detection.y - y)),
        "quality_passed": bool(detection.quality_passed),
        "flags": "|".join(detection.flags),
    }


def _summary_row(
    level: float,
    version: str,
    matches: dict[str, Detection],
    truths: Sequence[dict[str, object]],
    candidate_counts: dict[str, int],
    run_seconds: float,
    full_candidates: int,
    full_quality: int,
) -> dict[str, object]:
    hit_count = len(matches)
    total = len(truths)
    low, high = _wilson(hit_count, total)
    errors = [
        math.hypot(float(matches[str(truth["target_id"])].x) - float(truth["x_px"]),
                   float(matches[str(truth["target_id"])].y) - float(truth["y_px"]))
        for truth in truths
        if str(truth["target_id"]) in matches
    ]
    local_duplicates = sum(max(0, count - 1) for count in candidate_counts.values())
    return {
        "signal_peak_excess_adu": level,
        "version": version,
        "matched_targets": hit_count,
        "target_total": total,
        "recall": hit_count / total if total else None,
        "wilson95_low_descriptive": low,
        "wilson95_high_descriptive": high,
        "unmatched_targets": total - hit_count,
        "median_position_error_px": float(np.median(errors)) if errors else None,
        "p90_position_error_px": float(np.percentile(errors, 90)) if errors else None,
        "extra_candidates_within_match_radius": local_duplicates,
        "full_frame_candidate_count": full_candidates,
        "full_frame_quality_count": full_quality,
        "detector_seconds": run_seconds,
    }


def _save_case_sheet(
    image_by_level: dict[float, np.ndarray],
    target_rows: Sequence[dict[str, object]],
    output_path: Path,
    level: float,
) -> None:
    selected = [row for row in target_rows if float(row["signal_peak_excess_adu"]) == level]
    tile = 124
    crop_radius = 28
    columns = 4
    rows = max(1, math.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * tile, rows * tile), (24, 31, 42))
    try:
        font = ImageFont.truetype("segoeui.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    image = image_by_level[level]
    height, width = image.shape
    for index, row in enumerate(selected):
        x = float(row["x_px"])
        y = float(row["y_px"])
        cx, cy = round(x), round(y)
        x0, x1 = max(0, cx - crop_radius), min(width, cx + crop_radius + 1)
        y0, y1 = max(0, cy - crop_radius), min(height, cy + crop_radius + 1)
        patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
        finite = patch[np.isfinite(patch)]
        lo, hi = (np.percentile(finite, [5, 99.7]) if finite.size else (0.0, 1.0))
        if not math.isfinite(float(hi)) or hi <= lo:
            hi = lo + 1.0
        gray = np.clip((patch - lo) / (hi - lo), 0.0, 1.0)
        gray_image = Image.fromarray(np.asarray(gray * 255, dtype=np.uint8), mode="L").convert("RGB")
        panel = Image.new("RGB", (tile - 8, tile - 8), (16, 20, 28))
        gray_image.thumbnail((tile - 16, tile - 32), Image.Resampling.NEAREST)
        panel.paste(gray_image, ((panel.width - gray_image.width) // 2, 2))
        panel_draw = ImageDraw.Draw(panel)
        scale_x = gray_image.width / max(1, x1 - x0)
        scale_y = gray_image.height / max(1, y1 - y0)

        tx = (x - x0) * scale_x + (panel.width - gray_image.width) / 2
        ty = (y - y0) * scale_y + 2
        panel_draw.ellipse((tx - 5, ty - 5, tx + 5, ty + 5), outline=(255, 222, 80), width=2)
        _draw_case_detection_marker(panel_draw, panel, gray_image, row, x0, y0, scale_x, scale_y, "gaussian", (78, 168, 255))
        _draw_case_detection_marker(panel_draw, panel, gray_image, row, x0, y0, scale_x, scale_y, "hybrid", (72, 224, 161))
        panel_draw.text((4, panel.height - 17), f"{row['target_id']} A:{int(bool(row['gaussian_candidate_hit']))} B:{int(bool(row['hybrid_candidate_hit']))} C:{int(bool(row['quality_gate_hit']))}", fill=(240, 244, 248), font=font)
        col, line = index % columns, index // columns
        sheet.paste(panel, (col * tile + 4, line * tile + 4))
    sheet.save(output_path)


def _draw_case_detection_marker(
    panel_draw: ImageDraw.ImageDraw,
    panel: Image.Image,
    gray_image: Image.Image,
    row: dict[str, object],
    x0: int,
    y0: int,
    scale_x: float,
    scale_y: float,
    record_prefix: str,
    color: tuple[int, int, int],
    radius: int = 4,
) -> None:
    px = row.get(f"{record_prefix}_x_px")
    py = row.get(f"{record_prefix}_y_px")
    if px in (None, "") or py in (None, ""):
        return
    mx = (float(px) - x0) * scale_x + (panel.width - gray_image.width) / 2
    my = (float(py) - y0) * scale_y + 2
    panel_draw.ellipse((mx - radius, my - radius, mx + radius, my + radius), outline=color, width=2)


def run_point_source_ablation(
    fits_path: str | Path,
    out_dir: str | Path,
    *,
    peak_levels_adu: Sequence[float] = (12.0, 24.0, 56.0),
    target_count: int = 8,
    seed: int = 19019,
    psf_fwhm_px: float = 2.0,
    injected_psf_fwhm_px: float | None = None,
    threshold_sigma: float = 4.0,
    min_distance_px: int = 4,
    aperture_radius_px: int = 4,
    background_box_size_px: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    match_radius_px: float | None = None,
    fixed_targets_manifest_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run Gaussian → DoG proposal → existing quality gate on frozen images."""

    source = Path(fits_path).resolve()
    output = Path(out_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frozen_dir = output / "frozen_inputs"
    frozen_dir.mkdir()

    frame: FitsFrame = read_fits(source)
    raw_image = np.asarray(frame.data)
    if raw_image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape={raw_image.shape}")
    levels = tuple(float(level) for level in peak_levels_adu)
    if not levels or any(not math.isfinite(level) or level <= 0 for level in levels):
        raise ValueError("peak levels must be finite and positive")
    if target_count < 1 or seed < 0:
        raise ValueError("target_count must be positive and seed non-negative")
    injected_fwhm = float(psf_fwhm_px if injected_psf_fwhm_px is None else injected_psf_fwhm_px)
    if psf_fwhm_px <= 0 or injected_fwhm <= 0 or threshold_sigma <= 0 or min_distance_px < 1:
        raise ValueError("detector and PSF parameters must be positive")
    if aperture_radius_px < 1 or background_box_size_px < 8 or min_flux_snr < 0 or min_psf_support_pixels < 1:
        raise ValueError("quality parameters are outside the supported range")
    actual_match_radius = max(3.0, float(match_radius_px if match_radius_px is not None else psf_fwhm_px))
    if match_radius_px is not None and match_radius_px <= 0:
        raise ValueError("match radius must be positive")

    common_options = {
        "mask": auxiliary_mask(raw_image.shape),
        "threshold_sigma": float(threshold_sigma),
        "min_distance": int(min_distance_px),
        "aperture_radius": int(aperture_radius_px),
        "psf_fwhm": float(psf_fwhm_px),
        "background_box_size": int(background_box_size_px),
        "min_flux_snr": float(min_flux_snr),
        "min_psf_support_pixels": int(min_psf_support_pixels),
        "reject_linear_artifacts": True,
    }
    control_path = frozen_dir / "control-float32.npy"
    np.save(control_path, np.asarray(raw_image, dtype=np.float32), allow_pickle=False)
    control_image = np.load(control_path, allow_pickle=False)
    if control_image.dtype != np.float32:
        raise RuntimeError("frozen control input is not float32")

    start_all = time.perf_counter()
    if progress:
        progress("无注入配对控制与固定位置筛选：Ensemble")
    started = time.perf_counter()
    control_result = detect_sources(
        control_image,
        proposal_mode="ensemble",
        **{**common_options, "mask": common_options["mask"].copy()},
    )
    control_seconds = time.perf_counter() - started
    raw_sha = _sha256_file(source)
    if fixed_targets_manifest_path is not None:
        fixed_manifest_path = Path(fixed_targets_manifest_path).resolve()
        fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8"))
        if fixed_manifest.get("source_fits", {}).get("sha256") != raw_sha:
            raise ValueError("fixed target manifest input SHA-256 does not match the current FITS")
        frozen_targets = fixed_manifest.get("targets")
        if not isinstance(frozen_targets, list) or len(frozen_targets) != target_count:
            raise ValueError("fixed target manifest target count differs from --targets")
        truths: list[dict[str, object]] = [
            {
                "target_id": str(item["target_id"]),
                "x_px": float(item["x_px"]),
                "y_px": float(item["y_px"]),
                "psf_fwhm_px": injected_fwhm,
                "local_background_adu": float(item["local_background_adu"]),
                "local_noise_adu": float(item["local_noise_adu"]),
                "control_candidate_within_match_radius": False,
                "truth_type": "synthetic Gaussian point-source injection",
            }
            for item in frozen_targets
        ]
        if len({str(item["target_id"]) for item in truths}) != len(truths):
            raise ValueError("fixed target manifest contains duplicate target IDs")
        sites = [(float(item["x_px"]), float(item["y_px"])) for item in truths]
        selection_method = f"reused frozen coordinates from {fixed_manifest_path}"
    else:
        rng = np.random.default_rng(seed)
        sites, local_stats = _select_real_background_positions(
            control_image,
            control_result.sources,
            rng,
            target_count,
            noise_adu=float(control_result.noise),
            margin=14,
            source_exclusion_radius_px=max(12.0, 2.0 * psf_fwhm_px + actual_match_radius),
        )
        truths = []
        for index, ((x, y), (local_background, local_noise)) in enumerate(zip(sites, local_stats), start=1):
            truths.append({
                "target_id": f"PSF-{index:03d}",
                "x_px": float(x),
                "y_px": float(y),
                "psf_fwhm_px": injected_fwhm,
                "local_background_adu": float(local_background),
                "local_noise_adu": float(local_noise),
                "control_candidate_within_match_radius": False,
                "truth_type": "synthetic Gaussian point-source injection",
            })
        selection_method = "Ensemble candidates plus local 98th-percentile blank patch screen; positions frozen before injected versions"
    if len(sites) != target_count:
        raise ValueError("selected site count differs from target count")
    for x, y in sites:
        if not (0 <= x < control_image.shape[1] and 0 <= y < control_image.shape[0]):
            raise ValueError("fixed target coordinate lies outside the image")
    control_matches, control_nearby = _match_one_to_one(truths, control_result.sources, actual_match_radius)
    if control_matches or any(control_nearby.values()):
        raise RuntimeError("a pre-injection candidate lies within the true-match radius; refusing a confounded run")

    target_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    run_artifacts: dict[str, object] = {}
    images_for_cases: dict[float, np.ndarray] = {}
    timings: dict[str, list[float]] = {"control_ensemble": [control_seconds], "gaussian": [], "hybrid": []}
    detector_parameters: dict[str, object] = {
        "control_ensemble": control_result.parameters,
    }
    for level in levels:
        if progress:
            progress(f"冻结 {level:g} ADU 注入图像并运行 Gaussian / Hybrid")
        input_image = np.asarray(control_image, dtype=np.float32).copy()
        level_truths: list[dict[str, object]] = []
        for truth in truths:
            _gaussian_source(input_image, float(truth["x_px"]), float(truth["y_px"]), level, injected_fwhm)
            level_truths.append({**truth, "signal_peak_excess_adu": level})
        image_path = frozen_dir / f"injected-peak-{level:g}-adu.npy"
        np.save(image_path, input_image, allow_pickle=False)
        frozen_image = np.load(image_path, allow_pickle=False)
        if frozen_image.dtype != np.float32 or frozen_image.shape != control_image.shape:
            raise RuntimeError("frozen injection image differs in dtype or shape from the shared control")
        images_for_cases[level] = frozen_image
        image_hash = _sha256_file(image_path)

        version_results: dict[str, tuple[DetectionResult, float]] = {}
        for mode in ("gaussian", "hybrid"):
            if progress:
                progress(f"{level:g} ADU：{mode} 检测")
            started = time.perf_counter()
            result = detect_sources(
                frozen_image,
                proposal_mode=mode,
                **{**common_options, "mask": common_options["mask"].copy()},
            )
            elapsed = time.perf_counter() - started
            timings[mode].append(elapsed)
            version_results[mode] = (result, elapsed)
            detector_parameters[mode] = result.parameters

        gaussian_result, gaussian_seconds = version_results["gaussian"]
        hybrid_result, hybrid_seconds = version_results["hybrid"]
        gaussian_matches, gaussian_nearby = _match_one_to_one(level_truths, gaussian_result.sources, actual_match_radius)
        hybrid_matches, hybrid_nearby = _match_one_to_one(level_truths, hybrid_result.sources, actual_match_radius)
        quality_detections = tuple(source for source in hybrid_result.sources if source.quality_passed)
        quality_matches, quality_nearby = _match_one_to_one(level_truths, quality_detections, actual_match_radius)
        summary_rows.extend([
            _summary_row(level, "A 基线 Gaussian 提案", gaussian_matches, level_truths, gaussian_nearby, gaussian_seconds,
                         gaussian_result.candidate_count, gaussian_result.star_count),
            _summary_row(level, "B Gaussian+DoG 多尺度提案", hybrid_matches, level_truths, hybrid_nearby, hybrid_seconds,
                         hybrid_result.candidate_count, hybrid_result.star_count),
            _summary_row(level, "C 对 B 候选应用现有质量门", quality_matches, level_truths, quality_nearby, hybrid_seconds,
                         hybrid_result.candidate_count, hybrid_result.star_count),
        ])
        for truth in level_truths:
            target_id = str(truth["target_id"])
            am = gaussian_matches.get(target_id)
            bm = hybrid_matches.get(target_id)
            cm = quality_matches.get(target_id)
            row = {
                **truth,
                "control_candidates_within_match_radius": control_nearby[target_id],
                "gaussian_candidate_hit": am is not None,
                "gaussian_nearby_candidate_count": gaussian_nearby[target_id],
                "hybrid_candidate_hit": bm is not None,
                "hybrid_nearby_candidate_count": hybrid_nearby[target_id],
                "quality_gate_hit": cm is not None,
                "quality_gate_loss_from_hybrid_candidate": bm is not None and cm is None,
                "quality_rejection_flags": "|".join(bm.flags) if bm is not None and cm is None else "",
                "hybrid_candidate_status": "recovered" if bm is not None else "missed",
                "quality_gate_status": "passed" if cm is not None else ("rejected" if bm is not None else "not reached"),
                **{f"gaussian_{key}": value for key, value in _detection_record(am, float(truth["x_px"]), float(truth["y_px"])).items()},
                **{f"hybrid_{key}": value for key, value in _detection_record(bm, float(truth["x_px"]), float(truth["y_px"])).items()},
                **{f"quality_{key}": value for key, value in _detection_record(cm, float(truth["x_px"]), float(truth["y_px"])).items()},
                "frozen_injection_image": str(image_path),
                "frozen_injection_image_sha256": image_hash,
            }
            target_rows.append(row)
        run_artifacts[str(level)] = {
            "frozen_image": str(image_path),
            "frozen_image_sha256": image_hash,
            "array_sha256": _sha256_array(frozen_image),
            "dtype": str(frozen_image.dtype),
            "shape": list(frozen_image.shape),
            "gaussian_detection_seconds": gaussian_seconds,
            "hybrid_detection_seconds": hybrid_seconds,
            "gaussian_candidate_count": gaussian_result.candidate_count,
            "gaussian_quality_count": gaussian_result.star_count,
            "hybrid_candidate_count": hybrid_result.candidate_count,
            "hybrid_quality_count": hybrid_result.star_count,
        }

    summary_path = output / "summary.csv"
    target_path = output / "target_results.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(target_path, target_rows)
    elapsed_all = time.perf_counter() - start_all
    manifest = {
        "schema_version": 1,
        "experiment": "point_source_detection_ablation",
        "status": "completed",
        "created_local_time": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "source_fits": {"path": str(source), "sha256": raw_sha, "size_bytes": source.stat().st_size},
        "source_image": {
            "shape": list(raw_image.shape),
            "dtype": str(raw_image.dtype),
            "array_sha256": _sha256_array(raw_image),
            "fits_header": {key: frame.header.get(key) for key in ("BITPIX", "BZERO", "BSCALE", "DATE-OBS", "EXPTIME", "EXPOSURE") if key in frame.header},
        },
        "shared_control": {
            "path": str(control_path),
            "file_sha256": _sha256_file(control_path),
            "array_sha256": _sha256_array(control_image),
            "dtype": str(control_image.dtype),
            "shape": list(control_image.shape),
            "selection_method": selection_method,
            "selection_seed": seed,
            "control_detector_seconds": control_seconds,
            "control_candidate_count": control_result.candidate_count,
            "control_quality_count": control_result.star_count,
            "background_candidate_collisions_within_match_radius": 0,
        },
        "targets": truths,
        "signal_levels_adu": list(levels),
        "versions": [
            {"id": "A", "name": "基线 Gaussian 提案", "candidate_proposal": "gaussian", "quality_gate": "measured and reported, not used for candidate recall"},
            {"id": "B", "name": "Gaussian + DoG 多尺度提案", "candidate_proposal": "hybrid", "quality_gate": "same existing gate, reported separately"},
            {"id": "C", "name": "B 候选 + 现有质量门", "candidate_pool": "same B result objects", "quality_gate": "Detection.quality_passed; rejected flags retained"},
        ],
        "parameters_requested": {
            "threshold_sigma": threshold_sigma,
            "psf_fwhm_px": psf_fwhm_px,
            "injected_psf_fwhm_px": injected_fwhm,
            "min_distance_px": min_distance_px,
            "aperture_radius_px": aperture_radius_px,
            "background_box_size_px": background_box_size_px,
            "min_flux_snr": min_flux_snr,
            "min_psf_support_pixels": min_psf_support_pixels,
            "reject_linear_artifacts": True,
            "match_radius_requested_px": match_radius_px,
            "match_radius_actual_px": actual_match_radius,
            "match_rule": "minimum-distance sorted greedy one-to-one candidate-truth assignment; maximum allowed distance is actual match radius",
            "signal_definition": "Gaussian peak excess above the original pixel value, ADU",
            "dtype_for_all_detector_versions": "float32",
            "background_handling": "original raw FITS values converted once to float32; no global background subtraction before injection; detector local background estimation unchanged",
            "psf_model": "discrete Gaussian kernel",
            "negative_control": "same float32 original image, detected before injection; all target sites must have zero candidates within match radius or the run aborts",
        },
        "detector_parameters_actual": detector_parameters,
        "injected_images": run_artifacts,
        "metrics": {
            "candidate_recall": "one-to-one matched injected truth targets / injected target count",
            "quality_recall": "one-to-one matches among the same version's quality-passed detections / injected target count",
            "quality_gate_loss": "B candidate matched but no quality-passed B candidate matches; divided by B candidate matched targets and separately by all injected targets",
            "position_error": "Euclidean distance between measured centroid and injected center; median and P90 among matched targets, with misses reported separately",
            "duplicates": "additional detector candidates inside a target's match radius after one-to-one assignment; not interpreted as false positives",
            "interval": "95% Wilson interval shown descriptively by target at each signal level; target positions share a single image and repeated levels, so it is not a population-level independent-sample interval",
        },
        "runtime": {"total_experiment_seconds": elapsed_all, "detector_seconds_by_version": timings},
        "outputs": {"summary_csv": str(summary_path), "target_csv": str(target_path), "case_sheet": str(output / "cases-weakest-level.png")},
        "limitations": [
            "The truth is an artificial Gaussian injection, not an independently catalogued physical star.",
            "No full-image precision or false-positive rate is reported because the real background is not fully labelled.",
            "All positions are on one FITS background; repeated signal levels and nearby pixels are not independent samples.",
            "Motion proposal and track association are not measured in this point-source experiment.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    _save_case_sheet(images_for_cases, target_rows, output / "cases-weakest-level.png", min(levels))
    return manifest


def write_ablation_html(
    run_dir: str | Path,
    report_path: str | Path,
    *,
    motion_run_dir: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir).resolve()
    report = Path(report_path).resolve()
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with (run_dir / "summary.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    with (run_dir / "target_results.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        targets = list(csv.DictReader(stream))
    report.parent.mkdir(parents=True, exist_ok=True)
    point_evidence_dir = report.parent / "evidence" / "point"
    point_evidence_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("manifest.json", "summary.csv", "target_results.csv", "cases-weakest-level.png"):
        shutil.copy2(run_dir / filename, point_evidence_dir / filename)

    def link(path: str | Path, label: str) -> str:
        target = Path(path).resolve()
        relative = os.path.relpath(target, report.parent).replace(os.sep, "/")
        return f'<a href="{quote(relative, safe="/._-")}">{html.escape(label)}</a>'

    def fmt(value: object, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    table_rows = []
    for row in summaries:
        recall = fmt(row["recall"])
        ci = f"[{fmt(row['wilson95_low_descriptive'])}, {fmt(row['wilson95_high_descriptive'])}]"
        table_rows.append(
            "<tr>"
            f"<td>{fmt(row['signal_peak_excess_adu'], 0)} ADU</td><td>{html.escape(row['version'])}</td>"
            f"<td>{row['matched_targets']}/{row['target_total']} ({recall})</td><td>{ci}</td>"
            f"<td>{fmt(row['median_position_error_px'])} / {fmt(row['p90_position_error_px'])}</td>"
            f"<td>{row['unmatched_targets']}</td><td>{fmt(row['detector_seconds'], 1)} s</td></tr>"
        )
    loss_by_level = []
    for level in payload["signal_levels_adu"]:
        rows = [row for row in targets if float(row["signal_peak_excess_adu"]) == float(level)]
        b = sum(row["hybrid_candidate_hit"] == "True" for row in rows)
        c = sum(row["quality_gate_hit"] == "True" for row in rows)
        loss = b - c
        loss_by_level.append(f"{level:g} ADU：B {b}/{len(rows)} → C {c}/{len(rows)}，质量门损失 {loss}/{b if b else 0} 候选命中。")
    source = payload["source_fits"]
    execution = payload.get("execution", {})
    actual_argv = " ".join(str(part) for part in execution.get("argv", []))
    level_args = " ".join(f"{float(level):g}" for level in payload["signal_levels_adu"])
    replay_command = (
        f'uv run rst19-ablation "{source["path"]}" --targets {len(payload["targets"])} '
        f'--peak-levels {level_args} --reuse-manifest "{run_dir / "manifest.json"}" '
        '--out-dir "tmp/ablation/replay" --report "deliverables/ablation/index.html"'
    )
    report.write_text(
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>rst19 统一条件点源消融实验</title><style>
:root{color-scheme:light;--ink:#17243a;--muted:#58677d;--line:#dce3eb;--accent:#245bb2;--soft:#f3f7fb;--ok:#176b4d;--warn:#895319}
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:var(--ink);font:16px/1.65 "Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1100px;margin:32px auto;padding:0 20px 56px}.hero,.card{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;margin:16px 0;box-shadow:0 8px 30px #1e35500a}
h1{font-size:30px;line-height:1.25;margin:8px 0 12px}h2{font-size:21px;margin:0 0 12px}h3{font-size:17px;margin:18px 0 8px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.08em}
.tag{display:inline-block;border-radius:999px;padding:3px 10px;background:#e4f4ed;color:var(--ok);font-size:13px;font-weight:700}.tag.warn{background:#fff1dc;color:var(--warn)}.muted{color:var(--muted)}
.flow{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;margin:18px 0}.flow span{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:10px 14px;font-weight:650}.flow b{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#f5f8fb;font-size:14px}td{font-size:14px}
.scroll{overflow:auto;border:1px solid var(--line);border-radius:10px}img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#151a22}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}.metric{background:var(--soft);border-radius:12px;padding:14px}.metric strong{display:block;font-size:22px}.casekey{display:flex;gap:15px;flex-wrap:wrap;font-size:14px}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}.blue{background:#4ea8ff}.green{background:#48e0a1}.yellow{background:#ffde50}
code{background:#edf1f5;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}a{color:var(--accent)}.note{border-left:4px solid #dca241;background:#fff8eb;padding:12px 14px;border-radius:6px}.good{border-left:4px solid #3b9773;background:#eff8f4;padding:12px 14px;border-radius:6px}
@media(max-width:700px){main{margin:12px auto;padding:0 12px 32px}.hero,.card{padding:17px}h1{font-size:25px}}
</style></head><body><main>
<header class="hero"><div class="eyebrow">RST19 · 可复现算法验证</div><h1>同一真值集下的点源检测消融</h1>
<p>这次完成了真实 FITS 背景上的一套点源注入对照：先固定注入位置和图像，再比较 Gaussian 基线、多尺度 DoG 提案，以及在同一 Hybrid 候选上应用现有质量门。</p>
<p><span class="tag">点源实验已运行</span> <span class="tag warn">运动关联待验证</span></p>
<p class="muted">真实输入：""" + html.escape(Path(source["path"]).name) + """ · SHA-256：<code>""" + html.escape(source["sha256"]) + """</code> · 运行时间：""" + fmt(payload["runtime"]["total_experiment_seconds"], 1) + """ 秒</p></header>
<section class="card"><h2>1. 本次完成了什么</h2><p>已实现共享输入的实验编排，冻结 float32 控制图、各信号档注入图、目标坐标、随机种子和生效参数；逐个目标导出 A/B/C 命中、质心偏差及质量拒绝原因。真实 FITS 运行完成，具体结果见下表。</p>
<div class="note"><b>范围边界：</b>这证明的是已知人工 Gaussian 注入目标在此图像和参数下的回收表现。它不证明真实恒星身份，也没有测量完整星图的精确率。运动提案与轨迹关联需要独立冻结逐帧检测和注入轨迹真值，本节不将其计作像素检测证据。</div></section>
<section class="card"><h2>2. 实验为什么公平</h2><div class="flow"><span>同一原始 FITS<br>同一 float32 控制图</span><b>→</b><span>固定目标 ID、坐标、PSF<br>分别生成冻结注入图</span><b>→</b><span>A / B / C 读取<br>完全相同的图像</span><b>→</b><span>同一半径<br>一对一匹配</span></div>
<ul><li>A：Gaussian 候选提案基线。</li><li>B：只把提案扩展为 Gaussian + DoG；候选阈值、FWHM、测量和背景规则不变。</li><li>C：不重跑检测，直接对同一 B 候选池应用现有 <code>quality_passed</code> 质量门；拒绝旗标保留。</li></ul>
<p>无注入控制在目标位置的匹配半径内有 <b>0</b> 个已有候选。实际匹配半径是 <b>""" + fmt(payload["parameters_requested"]["match_radius_actual_px"], 1) + " px</b>；规则为最近距离优先的一对一分配。" + """</p></section>
<section class="card"><h2>3. 主要结果：逐信号强度配对</h2><p class="muted">召回写成命中数／注入数。区间为逐目标 Wilson 95% 描述区间；同一图像上的位置和多个信号档相关，因此不能据此做独立样本总体推断。</p>
<div class="scroll"><table><thead><tr><th>注入强度</th><th>版本</th><th>命中／目标数（召回）</th><th>Wilson 95% 区间*</th><th>定位误差中位数 / P90（px）</th><th>未回收</th><th>单次检测耗时</th></tr></thead><tbody>""" + "".join(table_rows) + """</tbody></table></div>
<h3>质量门的配对变化</h3><p>""" + " ".join(loss_by_level) + """</p><p class="muted">额外候选仅统计目标匹配半径内一对一分配后剩余的候选，不称作假阳性。未标注背景中的其他候选也没有用于计算精确率或误检率。</p></section>
<section class="card"><h2>4. 具体注入目标对比</h2><div class="casekey"><span><i class="dot yellow"></i>注入真值</span><span><i class="dot blue"></i>A Gaussian</span><span><i class="dot green"></i>B Hybrid</span></div>
<p>下图展示最低信号档的每个固定注入点。标题里的 A/B/C 为该目标是否被回收（1=是，0=否）；C 表示质量门后的命中。</p>""" + f'<img src="{quote(os.path.relpath(point_evidence_dir / "cases-weakest-level.png", report.parent).replace(os.sep, "/"), safe="/._-")}" alt="最低注入强度下所有注入目标的检测前后对比">' + """
<p class="muted">真实目标的新增回收、仍漏检与质量误拒绝均可在逐目标 CSV 中按状态筛选；如果本轮未出现某类情况，则表格中对应分子为零，不补造示例。</p></section>
<section class="card"><h2>5. 这些结果能证明什么</h2><div class="grid"><div class="metric"><strong>模块实现</strong>固定输入、共享注入真值、候选到质量门的配对输出可运行并已落盘。</div><div class="metric"><strong>本轮实验结论</strong>只适用于所列强度、首帧真实背景、Gaussian 注入 PSF 和当前参数；观察到的增益或损失以表格为准。</div><div class="metric"><strong>未确认事项</strong>官方星表身份、真实目标总体完备率、整幅图误检率，以及运动轨迹关联效果。</div></div>
<p>由于同一组坐标重复测试多个强度，样本之间有关联。本轮是可复核的算法工程对照，不作为广泛天文总体结论或物理解释。</p></section>
<section class="card"><h2>6. 比赛如何呈现</h2><h3>7 分钟陈述可用的核心表</h3><p>展示上面的逐强度 A/B/C 召回表，并同时说出分母：例如“在同一真实背景上的 N 个已知注入目标中，基线回收 x 个，多尺度回收 y 个，质量门后 z 个”。再用案例图解释新增召回和质量损失。不要把候选总数说成真实星数。</p>
<h3>3 分钟程序演示建议</h3><ol><li>现场打开固定 FITS，展示常规识别入口和当前候选/质量层定义。</li><li>展示预先完成的逐目标消融报告，说明相同输入、位置和匹配半径。</li><li>现场只运行一次选定识别参数并展示输出；不要把完整多版本消融放入 3 分钟计时。</li></ol><p class="note">当前只实测了消融内的 detector 用时，尚未测 GUI 启动、载入 FITS 和现场操作总耗时。报告中的运行时间为本次离线实验时间，不代表比赛现场演示时长。</p></section>
<section class="card"><h2>7. 复现与原始证据</h2><p>本次实际命令：</p><pre><code>""" + html.escape(actual_argv) + """</code></pre><p>使用 manifest 中冻结坐标重放：</p><pre><code>""" + html.escape(replay_command) + """</code></pre>
<ul><li>""" + link(point_evidence_dir / "manifest.json", "完整配置、哈希、真实生效参数和耗时 manifest.json") + """</li><li>""" + link(point_evidence_dir / "summary.csv", "按版本和信号档汇总 summary.csv") + """</li><li>""" + link(point_evidence_dir / "target_results.csv", "逐目标命中、定位和拒绝原因 target_results.csv") + """</li><li>冻结 float32 图像：""" + link(run_dir / "frozen_inputs", "frozen_inputs 目录") + """（含无注入控制和各档注入 .npy 文件）</li><li>""" + link(point_evidence_dir / "cases-weakest-level.png", "最低信号档案例图") + """</li></ul>
<p class="muted">摘要和逐目标 CSV 随交付报告保存；较大的冻结图像保存在项目忽略的本地实验目录，未纳入 Git。重放命令会读取同一原始 FITS 并验证 SHA-256，然后从 manifest 直接复用坐标。逐档注入图像仍会重新生成并记录新的哈希，以检查输入是否可复现。</p>
<p class="muted">实验输出目录：<code>""" + html.escape(str(run_dir)) + """</code> · 报告生成时间：""" + html.escape(str(payload["created_local_time"])) + """</p></section>
</main></body></html>
""",
        encoding="utf-8",
    )
    if motion_run_dir is not None:
        motion_section = _render_motion_section(Path(motion_run_dir).resolve(), report.parent)
        document = report.read_text(encoding="utf-8")
        closing = "</main></body></html>"
        if closing not in document:
            raise RuntimeError("generated ablation report is missing its closing marker")
        document = document.replace("运动关联待验证", "运动关联补充实验已运行")
        document = document.replace(
            "本节不将其计作像素检测证据。",
            "本节不将其计作像素检测证据；关联层结果见第 8 节。",
        )
        document = document.replace(
            "官方星表身份、真实目标总体完备率、整幅图误检率，以及运动轨迹关联效果。",
            "官方星表身份、真实目标总体完备率、整幅图误检率，以及真实场景中的运动目标回收率。",
        )
        report.write_text(document.replace(closing, motion_section + closing), encoding="utf-8")
    return report


def _render_motion_section(run_dir: Path, report_parent: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evidence_dir = report_parent / "evidence" / "motion"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "manifest.json",
        "trajectory_results.csv",
        "motion_pairing.csv",
        "truth_trajectories.json",
        "frozen_frame_sources.csv.gz",
        "motion_tracks.png",
    ):
        shutil.copy2(run_dir / filename, evidence_dir / filename)
    with (run_dir / "trajectory_results.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    def link(path: str | Path, label: str) -> str:
        relative = os.path.relpath(Path(path).resolve(), report_parent).replace(os.sep, "/")
        return f'<a href="{quote(relative, safe="/._-")}">{html.escape(label)}</a>'

    table_rows = []
    for row in rows:
        presence = f'{row["expected_presence"]}/{row["sequence_frames"]}'
        table_rows.append(
            "<tr>"
            f'<td>{html.escape(row["target_id"])}</td><td>{html.escape({"FAST-15": "快速正例，15/15", "FAST-GAP-12": "快速正例，3 个分散缺帧，12/15", "SLOW-BASE-15": "慢速基线正例，15/15", "BELOW-12": "持续性负例，11/15"}.get(row["target_id"], row["role"]))}</td>'
            f'<td>{presence}</td><td>{row["baseline_exact_track_recovered"]}</td>'
            f'<td>{row["baseline_fragments"]}</td><td>{row["fast_exact_track_recovered"]}</td>'
            f'<td>{row["fast_best_track_target_points"]}/{row["expected_presence"]}</td>'
            f'<td>{row["fast_mixed_with_background_source"]}</td></tr>'
        )
    cache = manifest["input_cache"]
    execution = manifest.get("execution", {})
    actual_argv = " ".join(str(part) for part in execution.get("argv", []))
    values = manifest["parameters"]
    return (
        '<section class="card"><h2>8. 运动约束：独立的轨迹关联对照</h2>'
        '<p>该实验冻结同一逐帧质量源列表，A 使用普通最近邻轨迹关联，B 在保留 A 多帧轨迹的同时，对未被 A 串联的质量点运行速度预测补充关联。'
        '注入对象是逐帧 <code>Detection</code> 记录，不重新经过图像像素检测；因此它验证的是关联阶段，不是运动目标的像素检出能力。</p>'
        '<div class="note"><b>比较口径：</b>表中 A 的“精确轨迹”要求普通关联单独形成完整真值轨迹；B 的“精确补充轨迹”表示速度预测补充层形成完整轨迹。'
        'A 中遗留的单帧碎片仍可与 B 补充轨迹同时存在，故碎片数单列，不能把 B 解读为已清除重复轨迹或全序列精确率。</div>'
        '<div class="scroll"><table><thead><tr><th>固定真值轨迹</th><th>角色</th><th>有效观测／总帧数</th>'
        '<th>A 普通关联精确轨迹</th><th>A 关联碎片数</th><th>B 速度补充精确轨迹</th><th>B 最佳补充点数</th><th>B 是否关联到背景源</th></tr></thead><tbody>'
        + "".join(table_rows)
        + "</tbody></table></div>"
        + '<p>真值含 15/15 快速正例、带 3 个分散缺帧的 12/15 快速正例、15/15 慢速基线正例和 11/15 的持续性负例。'
        f'统一参数：普通关联半径 {values["link_radius_px"]:g} px、最少 {values["min_presence_frames"]} 帧；补充层 SNR ≥ {values["supplement_min_flux_snr"]:g}、'
        f'速度上限 {values["supplement_max_step_px_per_reference_interval"]:g} px/参考间隔、预测门 {values["supplement_prediction_gate_px"]:g} px、'
        f'拟合 RMS ≤ {values["supplement_max_fit_rms_px"]:g} px。缓存重算配准相对已保存平移的最大偏差为 '
        f'{manifest["recomputed_registration_max_delta_px"]:.4f} px；逐帧质量点数和 {cache["source_hashes_verified"]} 个原始 FITS SHA-256 已核验。</p>'
        + f'<img src="{quote(os.path.relpath(evidence_dir / "motion_tracks.png", report_parent).replace(os.sep, "/"), safe="/._-")}" alt="普通关联与速度补充关联的注入轨迹对比">'
        + '<p class="muted">橙点表示普通关联中目标的点，绿线表示速度预测补充轨迹，黄线表示注入真值。背景污染仅检查轨迹是否混入该冻结质量源列表中的其它检测 ID；11/15 负例用于验证 min_presence 门，不是全图误报率估计。</p>'
        + '<p>本次实际命令：</p><pre><code>' + html.escape(actual_argv) + '</code></pre><ul>'
        + f'<li>{link(evidence_dir / "manifest.json", "运动关联参数、来源哈希和运行耗时 manifest.json")}</li>'
        + f'<li>{link(evidence_dir / "trajectory_results.csv", "逐轨迹结果 trajectory_results.csv")}</li>'
        + f'<li>{link(evidence_dir / "motion_pairing.csv", "A/B 配对摘要 motion_pairing.csv")}</li>'
        + f'<li>{link(evidence_dir / "truth_trajectories.json", "冻结真值轨迹 truth_trajectories.json")}</li>'
        + f'<li>{link(evidence_dir / "frozen_frame_sources.csv.gz", "送入两种关联器的冻结逐帧源列表")}</li></ul>'
        + f'<p class="muted">缓存：<code>{html.escape(str(cache["path"]))}</code> · SHA-256：<code>{html.escape(cache["sha256"])}</code> · 实验目录：<code>{html.escape(str(run_dir))}</code></p></section>'
    )


__all__ = ["run_point_source_ablation", "write_ablation_html"]
