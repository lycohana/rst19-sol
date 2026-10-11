"""Paired crowded-pair and isolated-source control ablation on frozen FITS patches."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ablation import _match_one_to_one
from .detection import Detection, DetectionResult, detect_sources
from .experiments import _gaussian_source
from .fits import auxiliary_mask, read_fits


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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty evidence table: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _nearby_count(detections: tuple[Detection, ...], x: float, y: float, radius: float) -> int:
    return sum(math.hypot(float(item.x) - x, float(item.y) - y) <= radius for item in detections)


def _scene_metrics(
    *,
    scene: str,
    truths: list[dict[str, object]],
    version_sources: dict[str, tuple[Detection, ...]],
    candidate_counts: dict[str, int],
    quality_counts: dict[str, int],
    timings: dict[str, float],
    match_radius_px: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    injected_truths = [item for item in truths if bool(item["injected"])]
    negative_truths = [item for item in truths if not bool(item["injected"])]
    for version in ("A", "B", "C", "D"):
        all_sources = version_sources[version]
        injected_matches, injected_nearby = _match_one_to_one(injected_truths, all_sources, match_radius_px)
        quality_sources = tuple(item for item in all_sources if bool(item.quality_passed))
        quality_matches, quality_nearby = _match_one_to_one(injected_truths, quality_sources, match_radius_px)
        errors = [
            math.hypot(
                float(injected_matches[str(item["target_id"])].x) - float(item["x_px"]),
                float(injected_matches[str(item["target_id"])].y) - float(item["y_px"]),
            )
            for item in injected_truths
            if str(item["target_id"]) in injected_matches
        ]
        negative_candidate_hits = sum(
            _nearby_count(all_sources, float(item["x_px"]), float(item["y_px"]), match_radius_px) > 0
            for item in negative_truths
        )
        negative_quality_hits = sum(
            _nearby_count(quality_sources, float(item["x_px"]), float(item["y_px"]), match_radius_px) > 0
            for item in negative_truths
        )
        summary_rows.append({
            "anchor_id": str(injected_truths[0]["anchor_id"] if injected_truths else truths[0]["anchor_id"]),
            "scene": scene,
            "version": version,
            "version_name": {
                "A": "Gaussian 基线",
                "B": "Gaussian+DoG 多尺度提案",
                "C": "B 候选通过现有质量门",
                "D": "Gaussian+局部去混叠候选",
            }[version],
            "injected_truth_total": len(injected_truths),
            "candidate_matched_targets": len(injected_matches),
            "candidate_recall": len(injected_matches) / len(injected_truths) if injected_truths else None,
            "quality_matched_targets": len(quality_matches),
            "quality_recall": len(quality_matches) / len(injected_truths) if injected_truths else None,
            "quality_gate_lost_candidate_hits": len(injected_matches) - len(quality_matches),
            "median_position_error_px": float(np.median(errors)) if errors else None,
            "p90_position_error_px": float(np.percentile(errors, 90)) if errors else None,
            "negative_companion_site_candidates": negative_candidate_hits,
            "negative_companion_site_quality_hits": negative_quality_hits,
            "candidate_count": candidate_counts[version],
            "quality_count": quality_counts[version],
            "detector_seconds": timings[version],
            "match_radius_px": match_radius_px,
        })
        for truth in truths:
            target_id = str(truth["target_id"])
            match = injected_matches.get(target_id) if bool(truth["injected"]) else None
            quality_match = quality_matches.get(target_id) if bool(truth["injected"]) else None
            assigned_count = injected_nearby[target_id] if bool(truth["injected"]) else _nearby_count(
                all_sources, float(truth["x_px"]), float(truth["y_px"]), match_radius_px
            )
            q_count = quality_nearby[target_id] if bool(truth["injected"]) else _nearby_count(
                quality_sources, float(truth["x_px"]), float(truth["y_px"]), match_radius_px
            )
            target_rows.append({
                "scene": scene,
                "version": version,
                "anchor_id": truth["anchor_id"],
                "target_id": target_id,
                "role": truth["role"],
                "injected": bool(truth["injected"]),
                "truth_x_px": float(truth["x_px"]),
                "truth_y_px": float(truth["y_px"]),
                "candidate_nearby_count": assigned_count,
                "candidate_hit": match is not None if bool(truth["injected"]) else assigned_count > 0,
                "quality_nearby_count": q_count,
                "quality_hit": quality_match is not None if bool(truth["injected"]) else q_count > 0,
                "detection_id": match.detection_id if match is not None else None,
                "detection_x_px": float(match.x) if match is not None else None,
                "detection_y_px": float(match.y) if match is not None else None,
                "position_error_px": (
                    math.hypot(float(match.x) - float(truth["x_px"]), float(match.y) - float(truth["y_px"]))
                    if match is not None else None
                ),
                "quality_passed": bool(match.quality_passed) if match is not None else None,
                "flags": "|".join(match.flags) if match is not None else "",
                "frozen_input_sha256": truth["scene_array_sha256"],
            })
    return summary_rows, target_rows


def _save_crowded_case_sheet(
    output_path: Path,
    anchor_id: str,
    pair_image: np.ndarray,
    single_image: np.ndarray,
    target_rows: list[dict[str, object]],
) -> None:
    """Show one paired FITS patch with shared truth and A/B/D detections."""

    canvas = Image.new("RGB", (1080, 500), (239, 243, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        try:
            if candidate.is_file():
                font = ImageFont.truetype(str(candidate), 16)
                small = ImageFont.truetype(str(candidate), 13)
                break
        except OSError:
            continue
    scenes = (("pair", pair_image, "双源：弱伴星真实注入"), ("single", single_image, "单星：伴星位未注入"))
    versions = (("A", (55, 145, 235)), ("B", (35, 160, 118)), ("D", (222, 72, 73)))
    crop = (30, 30, 72, 66)
    scale = 10
    for index, (scene, values, label) in enumerate(scenes):
        left = 16 + index * 536
        draw.rounded_rectangle((left, 12, left + 520, 484), radius=12, fill=(255, 255, 255), outline=(205, 215, 228), width=2)
        draw.text((left + 14, 22), f"{anchor_id} · {label}", fill=(24, 38, 58), font=font)
        x0, y0, x1, y1 = crop
        patch = np.asarray(values[y0:y1, x0:x1], dtype=np.float32)
        lo, hi = np.percentile(patch[np.isfinite(patch)], (2, 99.7))
        if not np.isfinite(hi) or hi <= lo:
            hi = float(lo) + 1.0
        gray = np.clip((patch - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)
        panel = Image.fromarray(np.asarray(gray * 255, dtype=np.uint8), mode="L").convert("RGB")
        panel = panel.resize((panel.width * scale, panel.height * scale), Image.Resampling.NEAREST)
        image_left, image_top = left + 42, 58
        canvas.paste(panel, (image_left, image_top))
        draw.rectangle((image_left, image_top, image_left + panel.width, image_top + panel.height), outline=(80, 93, 108), width=1)

        def point(x: float, y: float) -> tuple[float, float]:
            return image_left + (x - x0) * scale, image_top + (y - y0) * scale

        primary = point(48, 48)
        companion = point(54, 48)
        draw.ellipse((primary[0] - 7, primary[1] - 7, primary[0] + 7, primary[1] + 7), outline=(255, 217, 59), width=3)
        if scene == "pair":
            draw.ellipse((companion[0] - 7, companion[1] - 7, companion[0] + 7, companion[1] + 7), outline=(192, 104, 221), width=3)
        else:
            draw.line((companion[0] - 6, companion[1] - 6, companion[0] + 6, companion[1] + 6), fill=(192, 104, 221), width=3)
            draw.line((companion[0] - 6, companion[1] + 6, companion[0] + 6, companion[1] - 6), fill=(192, 104, 221), width=3)
        for version, color in versions:
            rows = [row for row in target_rows if row["anchor_id"] == anchor_id and row["scene"] == scene and row["version"] == version]
            for row in rows:
                if row["detection_x_px"] is None:
                    continue
                center = point(float(row["detection_x_px"]), float(row["detection_y_px"]))
                radius = {"A": 4, "B": 7, "D": 10}[version]
                if version == "D" and row["role"] == "weak companion":
                    draw.line((center[0] - 6, center[1] - 6, center[0] + 6, center[1] + 6), fill=color, width=3)
                    draw.line((center[0] - 6, center[1] + 6, center[0] + 6, center[1] - 6), fill=color, width=3)
                else:
                    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline=color, width=2)
        legend_y = image_top + panel.height + 15
        draw.text((image_left, legend_y), "黄圈：主源真值  紫圈/叉：伴星真值或未注入位", fill=(48, 61, 78), font=small)
        draw.text((image_left, legend_y + 16), "A 蓝圈  ·  B 绿圈  ·  D 红圈/红叉（去混叠伴星未过质量门）", fill=(48, 61, 78), font=small)
    canvas.save(output_path)


def run_crowded_pair_ablation(
    fits_path: str | Path,
    fixed_targets_manifest_path: str | Path,
    out_dir: str | Path,
    *,
    anchor_count: int = 8,
    match_radius_px: float = 3.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Compare four detector variants on paired crowded-pair/single-source FITS patches.

    Each anchor yields a fixed 96x96 raw FITS patch. Pair and single-source
    scenes are derived from the same patch; the companion location in the
    single-source scene is a known not-injected negative site, not a claim that
    the astronomical field is physically empty.
    """

    source = Path(fits_path).resolve()
    fixed_manifest_path = Path(fixed_targets_manifest_path).resolve()
    output = Path(out_dir).resolve()
    if not source.is_file() or not fixed_manifest_path.is_file():
        raise FileNotFoundError(source if not source.is_file() else fixed_manifest_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment directory: {output}")
    if anchor_count < 1 or match_radius_px <= 0:
        raise ValueError("anchor count and match radius must be positive")
    source_sha = _sha256_file(source)
    fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8"))
    if fixed_manifest.get("source_fits", {}).get("sha256") != source_sha:
        raise ValueError("fixed anchor manifest input SHA-256 does not match the current FITS")
    fixed_targets = fixed_manifest.get("targets")
    if not isinstance(fixed_targets, list) or len(fixed_targets) < anchor_count:
        raise ValueError("fixed anchor manifest has fewer target sites than requested anchors")

    frame = read_fits(source)
    raw = np.asarray(frame.data, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape={raw.shape}")
    detector_options = {
        "threshold_sigma": 6.0,
        "min_distance": 4,
        "aperture_radius": 4,
        "psf_fwhm": 3.0,
        "background_box_size": 32,
        "min_flux_snr": 2.0,
        "min_psf_support_pixels": 3,
        "dog_threshold_sigma": 5.0,
    }
    primary_xy = (48.0, 48.0)
    companion_xy = (54.0, 48.0)
    eligible: list[dict[str, Any]] = []
    rejected_anchors: list[dict[str, object]] = []
    previous_centers: list[tuple[float, float]] = []
    started_all = time.perf_counter()

    # Screen baseline patches before creating output files. All known sites
    # must be free of A/B/D candidates in the corresponding no-injection image.
    for item in fixed_targets:
        if len(eligible) >= anchor_count:
            break
        x_center, y_center = float(item["x_px"]), float(item["y_px"])
        x_int, y_int = round(x_center), round(y_center)
        if x_int < 48 or y_int < 48 or x_int + 48 > raw.shape[1] or y_int + 48 > raw.shape[0]:
            rejected_anchors.append({"anchor_id": item.get("target_id"), "reason": "96x96 patch exceeds image edge"})
            continue
        if any(math.hypot(x_center - px, y_center - py) < 96.0 for px, py in previous_centers):
            rejected_anchors.append({"anchor_id": item.get("target_id"), "reason": "patch overlaps an earlier anchor"})
            continue
        patch = np.asarray(raw[y_int - 48:y_int + 48, x_int - 48:x_int + 48], dtype=np.float32).copy()
        if patch.shape != (96, 96) or not np.isfinite(patch).all():
            rejected_anchors.append({"anchor_id": item.get("target_id"), "reason": "patch shape or finite-value check failed"})
            continue
        baseline_results: dict[str, DetectionResult] = {}
        for name, mode, deblend in (("A", "gaussian", False), ("B", "hybrid", False), ("D", "gaussian", True)):
            result = detect_sources(
                patch,
                mask=auxiliary_mask(patch.shape),
                proposal_mode=mode,
                enable_local_deblend=deblend,
                **detector_options,
            )
            baseline_results[name] = result
        collisions = {
            name: {
                "primary": _nearby_count(result.sources, *primary_xy, match_radius_px),
                "companion": _nearby_count(result.sources, *companion_xy, match_radius_px),
            }
            for name, result in baseline_results.items()
        }
        if any(value for per_site in collisions.values() for value in per_site.values()):
            rejected_anchors.append({
                "anchor_id": item.get("target_id"),
                "reason": "baseline candidate within a known injection site",
                "baseline_collisions": collisions,
            })
            continue
        eligible.append({
            "anchor_id": str(item["target_id"]),
            "original_x_px": x_center,
            "original_y_px": y_center,
            "patch": patch,
            "baseline_results": baseline_results,
            "baseline_collisions": collisions,
        })
        previous_centers.append((x_center, y_center))
    if len(eligible) < anchor_count:
        raise RuntimeError(
            f"only {len(eligible)} fixed patches passed no-injection screening; {anchor_count} are required"
        )

    output.mkdir(parents=True, exist_ok=True)
    frozen_dir = output / "frozen_inputs"
    frozen_dir.mkdir()
    summary_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    anchor_records: list[dict[str, object]] = []
    selection_started = time.perf_counter()

    for anchor in eligible:
        anchor_id = str(anchor["anchor_id"])
        patch = np.asarray(anchor["patch"], dtype=np.float32)
        base_name = f"{anchor_id}-base-float32.npy"
        base_path = frozen_dir / base_name
        np.save(base_path, patch, allow_pickle=False)
        frozen_base = np.load(base_path, allow_pickle=False)
        base_hash = _sha256_file(base_path)
        pair_image = frozen_base.copy()
        _gaussian_source(pair_image, *primary_xy, 160.0, 1.4)
        _gaussian_source(pair_image, *companion_xy, 40.0, 0.8)
        single_image = frozen_base.copy()
        _gaussian_source(single_image, *primary_xy, 160.0, 1.4)
        scene_specs = (
            ("pair", pair_image, [
                {"target_id": f"{anchor_id}-PRIMARY", "role": "bright primary", "injected": True, "x_px": primary_xy[0], "y_px": primary_xy[1]},
                {"target_id": f"{anchor_id}-COMPANION", "role": "weak companion", "injected": True, "x_px": companion_xy[0], "y_px": companion_xy[1]},
            ]),
            ("single", single_image, [
                {"target_id": f"{anchor_id}-PRIMARY", "role": "bright primary", "injected": True, "x_px": primary_xy[0], "y_px": primary_xy[1]},
                {"target_id": f"{anchor_id}-NO-COMPANION", "role": "not-injected companion negative site", "injected": False, "x_px": companion_xy[0], "y_px": companion_xy[1]},
            ]),
        )
        anchor_records.append({
            "anchor_id": anchor_id,
            "original_fits_center_px": [anchor["original_x_px"], anchor["original_y_px"]],
            "crop_origin_px": [round(float(anchor["original_x_px"])) - 48, round(float(anchor["original_y_px"])) - 48],
            "patch_shape": list(frozen_base.shape),
            "base_array_sha256": _sha256_array(frozen_base),
            "base_file": str(base_path),
            "base_file_sha256": base_hash,
            "baseline_collisions": anchor["baseline_collisions"],
        })
        for version in ("A", "B", "D"):
            control = anchor["baseline_results"][version]
            control_rows.append({
                "anchor_id": anchor_id,
                "version": version,
                "primary_site_candidates": _nearby_count(control.sources, *primary_xy, match_radius_px),
                "companion_site_candidates": _nearby_count(control.sources, *companion_xy, match_radius_px),
                "candidate_count": control.candidate_count,
                "quality_count": control.star_count,
                "detector_seconds": None,
                "input": "same real FITS patch, no injection; pre-screen run",
            })
        for scene, image, scene_truths in scene_specs:
            scene_file = frozen_dir / f"{anchor_id}-{scene}-float32.npy"
            np.save(scene_file, np.asarray(image, dtype=np.float32), allow_pickle=False)
            frozen_image = np.load(scene_file, allow_pickle=False)
            scene_file_hash = _sha256_file(scene_file)
            array_hash = _sha256_array(frozen_image)
            truths = [
                {**truth, "anchor_id": anchor_id, "scene_array_sha256": array_hash}
                for truth in scene_truths
            ]
            options_for_run = {
                "mask": auxiliary_mask(frozen_image.shape),
                **detector_options,
            }
            detector_results: dict[str, tuple[DetectionResult, float]] = {}
            for version, mode, deblend in (("A", "gaussian", False), ("B", "hybrid", False), ("D", "gaussian", True)):
                if progress:
                    progress(f"{anchor_id} {scene} 场景：运行 {version}")
                started = time.perf_counter()
                result = detect_sources(
                    frozen_image,
                    proposal_mode=mode,
                    enable_local_deblend=deblend,
                    **{**options_for_run, "mask": options_for_run["mask"].copy()},
                )
                elapsed = time.perf_counter() - started
                detector_results[version] = (result, elapsed)
            # C consumes B's exact candidate records; only the quality-passed
            # subset changes, so this row has no second detection runtime.
            b_result, b_seconds = detector_results["B"]
            detector_results["C"] = (
                DetectionResult(
                    image_shape=b_result.image_shape,
                    background=b_result.background,
                    noise=b_result.noise,
                    threshold=b_result.threshold,
                    candidate_count=b_result.candidate_count,
                    sources=tuple(source for source in b_result.sources if source.quality_passed),
                    parameters=b_result.parameters,
                    quality_count=b_result.star_count,
                ),
                b_seconds,
            )
            version_sources: dict[str, tuple[Detection, ...]] = {}
            candidate_counts: dict[str, int] = {}
            quality_counts: dict[str, int] = {}
            timings: dict[str, float] = {}
            for version in ("A", "B", "C", "D"):
                result, elapsed = detector_results[version]
                version_sources[version] = tuple(result.sources)
                candidate_counts[version] = result.candidate_count
                quality_counts[version] = result.star_count
                timings[version] = elapsed
                run_rows.append({
                    "anchor_id": anchor_id,
                    "scene": scene,
                    "version": version,
                    "input_file": str(scene_file),
                    "input_file_sha256": scene_file_hash,
                    "input_array_sha256": array_hash,
                    "proposal_mode": "hybrid" if version in {"B", "C"} else "gaussian",
                    "enable_local_deblend": version == "D",
                    "candidate_count": result.candidate_count,
                    "returned_count": len(result.sources),
                    "quality_count": result.star_count,
                    "detector_seconds": elapsed,
                    "reused_detector_version": "B" if version == "C" else "",
                    "parameters_actual": result.parameters,
                })
            scene_summary, scene_target_rows = _scene_metrics(
                scene=scene,
                truths=truths,
                version_sources=version_sources,
                candidate_counts=candidate_counts,
                quality_counts=quality_counts,
                timings=timings,
                match_radius_px=match_radius_px,
            )
            summary_rows.extend(scene_summary)
            target_rows.extend(scene_target_rows)
            anchor_records[-1].setdefault("scenes", {})[scene] = {
                "file": str(scene_file),
                "file_sha256": scene_file_hash,
                "array_sha256": array_hash,
                "dtype": str(frozen_image.dtype),
                "truths": truths,
            }

    aggregate_rows: list[dict[str, object]] = []
    for scene in ("pair", "single"):
        for version in ("A", "B", "C", "D"):
            patch_rows = [row for row in summary_rows if row["scene"] == scene and row["version"] == version]
            injected_total = sum(int(row["injected_truth_total"]) for row in patch_rows)
            candidate_hits = sum(int(row["candidate_matched_targets"]) for row in patch_rows)
            quality_hits = sum(int(row["quality_matched_targets"]) for row in patch_rows)
            negative_total = sum(
                1 for row in target_rows
                if row["scene"] == scene and row["version"] == version and row["role"] == "not-injected companion negative site"
            )
            negative_hits = sum(int(row["negative_companion_site_candidates"]) for row in patch_rows)
            negative_quality_hits = sum(int(row["negative_companion_site_quality_hits"]) for row in patch_rows)
            errors = [
                float(row["position_error_px"])
                for row in target_rows
                if row["scene"] == scene and row["version"] == version and bool(row["injected"])
                and bool(row["candidate_hit"]) and row["position_error_px"] is not None
            ]
            seconds = [float(row["detector_seconds"]) for row in patch_rows]
            aggregate_rows.append({
                "scene": scene,
                "version": version,
                "version_name": patch_rows[0]["version_name"],
                "patch_count": len(patch_rows),
                "injected_truth_total": injected_total,
                "candidate_matched_targets": candidate_hits,
                "candidate_recall": candidate_hits / injected_total if injected_total else None,
                "quality_matched_targets": quality_hits,
                "quality_recall": quality_hits / injected_total if injected_total else None,
                "quality_gate_lost_candidate_hits": sum(int(row["quality_gate_lost_candidate_hits"]) for row in patch_rows),
                "median_position_error_px": float(np.median(errors)) if errors else None,
                "p90_position_error_px": float(np.percentile(errors, 90)) if errors else None,
                "negative_companion_sites_total": negative_total,
                "negative_companion_site_candidates": negative_hits,
                "negative_companion_site_fp_rate": negative_hits / negative_total if negative_total else None,
                "negative_companion_site_quality_hits": negative_quality_hits,
                "candidate_count_total": sum(int(row["candidate_count"]) for row in patch_rows),
                "quality_count_total": sum(int(row["quality_count"]) for row in patch_rows),
                "detector_seconds_total": sum(seconds),
                "median_detector_seconds_per_patch": float(np.median(seconds)) if seconds else None,
                "match_radius_px": match_radius_px,
            })
    _write_csv(output / "summary.csv", aggregate_rows)
    _write_csv(output / "anchor_summary.csv", summary_rows)
    _write_csv(output / "target_results.csv", target_rows)
    _write_csv(output / "control_summary.csv", control_rows)
    _write_csv(output / "detector_runs.csv", run_rows)
    first_anchor = eligible[0]
    first_anchor_id = str(first_anchor["anchor_id"])
    first_anchor_scenes = anchor_records[0]["scenes"]
    pair_image = np.load(Path(first_anchor_scenes["pair"]["file"]), allow_pickle=False)
    single_image = np.load(Path(first_anchor_scenes["single"]["file"]), allow_pickle=False)
    case_sheet_path = output / "crowded_pair_cases.png"
    _save_crowded_case_sheet(case_sheet_path, first_anchor_id, pair_image, single_image, target_rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment": "paired_real_fits_crowded_companion_ablation",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_fits": {"path": str(source), "sha256": source_sha, "size_bytes": source.stat().st_size},
        "fixed_anchor_manifest": {
            "path": str(fixed_manifest_path),
            "sha256": _sha256_file(fixed_manifest_path),
            "source_experiment": fixed_manifest.get("experiment"),
        },
        "patch_selection": {
            "requested_anchor_count": anchor_count,
            "accepted_anchor_count": len(eligible),
            "patch_shape": [96, 96],
            "site_screen": "A Gaussian, B hybrid, and D Gaussian+local-deblend all have zero returned candidates within 3 px of both prospective sites on the same no-injection patch",
            "accepted_anchors": anchor_records,
            "rejected_anchors": rejected_anchors,
        },
        "injections": {
            "primary": {"x_px": primary_xy[0], "y_px": primary_xy[1], "center_peak_excess_adu": 160.0, "fwhm_px": 1.4},
            "companion": {"x_px": companion_xy[0], "y_px": companion_xy[1], "center_peak_excess_adu": 40.0, "fwhm_px": 0.8},
            "pair_separation_px": math.hypot(companion_xy[0] - primary_xy[0], companion_xy[1] - primary_xy[1]),
            "negative_control": "single-source scenes contain the primary only; the companion coordinate is a known not-injected site, not a claim of physical blank sky",
            "all_version_inputs": "each detector variant loads the same frozen float32 NPY for a given anchor and scene",
        },
        "versions": [
            {"id": "A", "name": "Gaussian baseline proposals", "proposal_mode": "gaussian", "enable_local_deblend": False},
            {"id": "B", "name": "Gaussian + DoG multiscale proposals", "proposal_mode": "hybrid", "enable_local_deblend": False},
            {"id": "C", "name": "B candidates passed through existing quality gate", "candidate_pool": "same B Detection objects; quality_passed only"},
            {"id": "D", "name": "Gaussian proposals + local deblend", "proposal_mode": "gaussian", "enable_local_deblend": True},
        ],
        "parameters_requested": {
            **{key: value for key, value in detector_options.items() if key != "dog_threshold_sigma"},
            "dog_threshold_sigma": detector_options["dog_threshold_sigma"],
            "match_radius_px": match_radius_px,
            "injected_psf_fwhm_px": {"primary": 1.4, "companion": 0.8},
            "dtype": "float32",
        },
        "outputs": {
            "summary_csv": str(output / "summary.csv"),
            "anchor_summary_csv": str(output / "anchor_summary.csv"),
            "target_results_csv": str(output / "target_results.csv"),
            "control_summary_csv": str(output / "control_summary.csv"),
            "detector_runs_csv": str(output / "detector_runs.csv"),
            "case_sheet_png": str(case_sheet_path),
            "frozen_inputs_dir": str(frozen_dir),
        },
        "metrics": {
            "candidate_recall": "one-to-one matched injected sources / injected truth count, separately from quality-passed recall",
            "position_error": "Euclidean distance in pixels; median and P90 over matched injected truths only",
            "quality_gate_loss": "candidate hit minus matched quality-passed hit count; C uses the same B candidate pool",
            "false_split_control": "count of candidate and quality detections within 3 px of the not-injected companion site in paired single-source scenes",
            "matching": "minimum-distance sorted greedy one-to-one assignment, maximum radius 3 px",
        },
        "runtime": {"total_experiment_seconds": time.perf_counter() - started_all, "site_screen_seconds": selection_started - started_all},
        "limitations": [
            "Injected Gaussian sources are labelled synthetic truths on real FITS image patches; they are not independently catalogued physical stars.",
            "The not-injected companion coordinate is a local negative control only; it does not establish that the surrounding sky is physically empty.",
            "Eight patches from one FITS frame are paired measurements, not an independent survey population; no full-frame precision or physical completeness is estimated.",
            "D measures an optional single-frame deblend path. It is a separate crowded-pair condition and not a replacement for the A/B/C point-source ablation.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest
