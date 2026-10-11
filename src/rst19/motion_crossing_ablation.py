"""Frozen synthetic crossing-track experiment for association-stage behavior.

This experiment starts from explicit ``Detection`` records. It tests association
only; it does not exercise image detection, photometry, or physical-object
identification.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .detection import Detection
from .sequence import SourceTrack, track_detections, track_fast_point_movers

FRAME_COUNT = 15
MIN_PRESENCE = 12
LINK_RADIUS_PX = 4.0
REGISTRATION_RADIUS_PX = 8.0
MOTION_MIN_DISPLACEMENT_PX = 2.0
BASELINE_MAX_FIT_RMS_PX = 0.75
SUPPLEMENT_MIN_FLUX_SNR = 7.0
SUPPLEMENT_MAX_STEP_PX = 20.0
SUPPLEMENT_PREDICTION_GATE_PX = 4.5
SUPPLEMENT_MAX_FIT_RMS_PX = 1.5

_ANCHORS = tuple(
    (float(x), float(y))
    for y in (120, 320, 520, 720)
    for x in (120, 320, 520, 720, 920, 1120)
)
_SCENARIOS: dict[str, tuple[dict[str, Any], ...]] = {
    "slow_linked_identity_swap": (
        {
            "target_id": "SLOW-CROSS-A",
            "detection_id": 1_000_001,
            "start_registered_xy_px": (1500.0, 1800.0),
            "velocity_px_per_frame": (4.0, 0.0),
        },
        {
            "target_id": "SLOW-CROSS-B",
            "detection_id": 1_000_002,
            "start_registered_xy_px": (1552.0, 1800.0),
            "velocity_px_per_frame": (-4.0, 0.0),
        },
    ),
    "fast_orthogonal_supplement": (
        {
            "target_id": "FAST-CROSS-A",
            "detection_id": 1_000_011,
            "start_registered_xy_px": (1948.0, 2000.0),
            "velocity_px_per_frame": (8.0, 0.0),
        },
        {
            "target_id": "FAST-CROSS-B",
            "detection_id": 1_000_012,
            "start_registered_xy_px": (2000.0, 1948.0),
            "velocity_px_per_frame": (0.0, 8.0),
        },
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_detection(detection_id: int, x: float, y: float) -> Detection:
    """Return a fixed, quality-passed point record without pixel data."""

    return Detection(
        detection_id=int(detection_id),
        x=float(x),
        y=float(y),
        peak=100.0,
        flux=500.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=2.0,
        flags=(),
        flux_error=20.0,
        flux_snr=20.0,
        filter_snr=20.0,
        quality_passed=True,
    )


def _build_scenario(
    scenario: str,
) -> tuple[list[list[Detection]], list[dict[str, Any]]]:
    if scenario not in _SCENARIOS:
        raise ValueError(f"unknown crossing scenario: {scenario}")
    frames: list[list[Detection]] = []
    truth_rows: list[dict[str, Any]] = []
    specs = _SCENARIOS[scenario]
    for frame_index in range(FRAME_COUNT):
        sources = [
            _make_detection(detection_id, x, y)
            for detection_id, (x, y) in enumerate(_ANCHORS)
        ]
        for spec in specs:
            start_x, start_y = spec["start_registered_xy_px"]
            velocity_x, velocity_y = spec["velocity_px_per_frame"]
            x = float(start_x) + frame_index * float(velocity_x)
            y = float(start_y) + frame_index * float(velocity_y)
            detection_id = int(spec["detection_id"])
            sources.append(_make_detection(detection_id, x, y))
            truth_rows.append(
                {
                    "scenario": scenario,
                    "target_id": str(spec["target_id"]),
                    "detection_id": detection_id,
                    "frame_index": frame_index,
                    "x_px": f"{x:.9f}",
                    "y_px": f"{y:.9f}",
                    "role": "known synthetic crossing trajectory",
                }
            )
        sources.sort(key=lambda item: item.detection_id)
        if len({source.detection_id for source in sources}) != len(sources):
            raise ValueError(f"duplicate detection_id in {scenario}, frame {frame_index}")
        frames.append(sources)
    return frames, truth_rows


def _detection_csv_row(
    frame_index: int,
    source: Detection,
    target_by_detection_id: dict[int, str],
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "detection_id": int(source.detection_id),
        "x_px": f"{float(source.x):.9f}",
        "y_px": f"{float(source.y):.9f}",
        "peak_adu": f"{float(source.peak):.9f}",
        "flux_adu": f"{float(source.flux):.9f}",
        "background_adu": f"{float(source.background):.9f}",
        "noise_adu": f"{float(source.noise):.9f}",
        "snr": f"{float(source.snr):.9f}",
        "fwhm_px": "" if source.fwhm is None else f"{float(source.fwhm):.9f}",
        "flags_json": json.dumps(list(source.flags), ensure_ascii=False, separators=(",", ":")),
        "flux_error": "" if source.flux_error is None else f"{float(source.flux_error):.9f}",
        "flux_snr": "" if source.flux_snr is None else f"{float(source.flux_snr):.9f}",
        "filter_snr": "" if source.filter_snr is None else f"{float(source.filter_snr):.9f}",
        "quality_passed": int(bool(source.quality_passed)),
        "truth_target_id": target_by_detection_id.get(int(source.detection_id), ""),
    }


_FROZEN_FIELDS = (
    "frame_index",
    "detection_id",
    "x_px",
    "y_px",
    "peak_adu",
    "flux_adu",
    "background_adu",
    "noise_adu",
    "snr",
    "fwhm_px",
    "flags_json",
    "flux_error",
    "flux_snr",
    "filter_snr",
    "quality_passed",
    "truth_target_id",
)


def _write_and_reload_frozen_sources(
    path: Path,
    frames: list[list[Detection]],
    truth_rows: list[dict[str, Any]],
) -> tuple[list[list[Detection]], str, str]:
    target_by_detection_id = {
        int(spec["detection_id"]): str(spec["target_id"])
        for spec in _SCENARIOS[str(truth_rows[0]["scenario"])]
    }
    text_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(text_buffer, fieldnames=_FROZEN_FIELDS, lineterminator="\n")
    writer.writeheader()
    written_rows: list[dict[str, object]] = []
    for frame_index, sources in enumerate(frames):
        for source in sources:
            row = _detection_csv_row(frame_index, source, target_by_detection_id)
            writer.writerow(row)
            written_rows.append({key: str(row[key]) for key in _FROZEN_FIELDS})
    compressed = gzip.compress(text_buffer.getvalue().encode("utf-8"), compresslevel=6, mtime=0)
    path.write_bytes(compressed)
    frozen_hash = _sha256_bytes(compressed)

    rebuilt: list[list[Detection]] = [[] for _ in range(FRAME_COUNT)]
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _FROZEN_FIELDS:
            raise ValueError("frozen source CSV has an unexpected schema")
        canonical_rows: list[dict[str, object]] = []
        for row in reader:
            frame_index = int(row["frame_index"])
            if not 0 <= frame_index < FRAME_COUNT:
                raise ValueError("frozen source CSV has an invalid frame index")
            canonical_rows.append({key: row[key] for key in _FROZEN_FIELDS})
            rebuilt[frame_index].append(
                Detection(
                    detection_id=int(row["detection_id"]),
                    x=float(row["x_px"]),
                    y=float(row["y_px"]),
                    peak=float(row["peak_adu"]),
                    flux=float(row["flux_adu"]),
                    background=float(row["background_adu"]),
                    noise=float(row["noise_adu"]),
                    snr=float(row["snr"]),
                    fwhm=float(row["fwhm_px"]) if row["fwhm_px"] else None,
                    flags=tuple(json.loads(row["flags_json"])),
                    flux_error=float(row["flux_error"]) if row["flux_error"] else None,
                    flux_snr=float(row["flux_snr"]) if row["flux_snr"] else None,
                    filter_snr=float(row["filter_snr"]) if row["filter_snr"] else None,
                    quality_passed=bool(int(row["quality_passed"])),
                )
            )
    if canonical_rows != written_rows:
        raise ValueError("frozen source CSV rows do not match the generated Detection records")
    for sources in rebuilt:
        sources.sort(key=lambda item: item.detection_id)
    return rebuilt, frozen_hash, _fingerprint_frames(rebuilt)


def _fingerprint_frames(frames: list[list[Detection]]) -> str:
    rows = [
        [
            frame_index,
            int(source.detection_id),
            float(source.x),
            float(source.y),
            float(source.peak),
            float(source.flux),
            float(source.background),
            float(source.noise),
            float(source.snr),
            None if source.fwhm is None else float(source.fwhm),
            list(source.flags),
            None if source.flux_error is None else float(source.flux_error),
            None if source.flux_snr is None else float(source.flux_snr),
            None if source.filter_snr is None else float(source.filter_snr),
            bool(source.quality_passed),
        ]
        for frame_index, sources in enumerate(frames)
        for source in sources
    ]
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256_bytes(encoded)


def _track_record(track: SourceTrack, output_index: int, origin: str) -> dict[str, Any]:
    points = sorted(track.points, key=lambda point: (point.frame_index, point.detection_id))
    return {
        "track": track,
        "output_track_id": f"{origin}-{output_index:04d}",
        "origin_arm": origin,
        "source_track_id": track.track_id,
        "points": points,
    }


def _score_variant(
    scenario: str,
    version: str,
    records: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    input_sha256: str,
    registration_max_shift_px: float,
    runtime_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    truth_by_key = {
        (int(row["frame_index"]), int(row["detection_id"])): str(row["target_id"])
        for row in truth_rows
    }
    expected_by_target: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for key, target_id in truth_by_key.items():
        expected_by_target[target_id].add(key)

    track_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    track_keys_by_target: dict[str, list[set[tuple[int, int]]]] = defaultdict(list)
    covered_by_target: dict[str, set[tuple[int, int]]] = defaultdict(set)
    switches_by_track: dict[str, int] = {}
    mixed_track_count = 0
    background_contamination_count = 0
    assigned_truth_observations = 0
    assigned_truth_keys: set[tuple[int, int]] = set()

    for record in records:
        points = record["points"]
        labels = [truth_by_key.get((int(point.frame_index), int(point.detection_id)), "") for point in points]
        target_labels = [label for label in labels if label]
        unique_targets = sorted(set(target_labels))
        identity_switches = sum(left != right for left, right in zip(target_labels, target_labels[1:]))
        output_track_id = str(record["output_track_id"])
        switches_by_track[output_track_id] = identity_switches
        is_mixed = len(unique_targets) > 1
        has_background = any(not label for label in labels)
        if is_mixed:
            mixed_track_count += 1
        if unique_targets and has_background:
            background_contamination_count += 1

        track_keys: set[tuple[int, int]] = set()
        full_clean_targets: list[str] = []
        for point, label in zip(points, labels, strict=True):
            key = (int(point.frame_index), int(point.detection_id))
            track_keys.add(key)
            point_rows.append(
                {
                    "scenario": scenario,
                    "version": version,
                    "output_track_id": output_track_id,
                    "origin_arm": record["origin_arm"],
                    "frame_index": int(point.frame_index),
                    "detection_id": int(point.detection_id),
                    "truth_target_id": label,
                    "x_px": f"{float(point.x):.9f}",
                    "y_px": f"{float(point.y):.9f}",
                    "aligned_x_px": f"{float(point.aligned_x):.9f}",
                    "aligned_y_px": f"{float(point.aligned_y):.9f}",
                }
            )
            if label:
                covered_by_target[label].add(key)
                assigned_truth_keys.add(key)
                assigned_truth_observations += 1
        for target_id in expected_by_target:
            target_keys = {key for key in track_keys if truth_by_key.get(key) == target_id}
            if target_keys:
                track_keys_by_target[target_id].append(target_keys)
            if target_keys == expected_by_target[target_id] and len(points) == len(expected_by_target[target_id]):
                full_clean_targets.append(target_id)

        track = record["track"]
        track_rows.append(
            {
                "scenario": scenario,
                "version": version,
                "output_track_id": output_track_id,
                "origin_arm": record["origin_arm"],
                "source_track_id": record["source_track_id"],
                "classification": track.classification,
                "evidence_level": track.evidence_level,
                "point_count": len(points),
                "truth_target_ids_json": json.dumps(unique_targets, ensure_ascii=False, separators=(",", ":")),
                "full_clean_target_ids_json": json.dumps(full_clean_targets, ensure_ascii=False, separators=(",", ":")),
                "truth_observation_count": len(target_labels),
                "background_observation_count": len(labels) - len(target_labels),
                "identity_switch_count": identity_switches,
                "mixed_truth_ids": int(is_mixed),
                "background_contaminated": int(bool(unique_targets and has_background)),
                "frame_indices_csv": ",".join(str(int(point.frame_index)) for point in points),
                "detection_ids_csv": ",".join(str(int(point.detection_id)) for point in points),
                "displacement_px": f"{float(track.displacement_px):.9f}",
                "speed_px_per_frame": f"{float(track.speed_px_per_frame):.9f}",
                "fit_rms_px": "" if track.fit_rms_px is None else f"{float(track.fit_rms_px):.9f}",
            }
        )

    target_rows: list[dict[str, object]] = []
    total_fragments = 0
    full_clean_present_count = 0
    exclusive_full_count = 0
    covered_all_count = 0
    for target_id, expected_keys in sorted(expected_by_target.items()):
        track_key_lists = track_keys_by_target.get(target_id, [])
        clean_count = sum(
            1
            for record in records
            if { (int(point.frame_index), int(point.detection_id)) for point in record["points"] }
            == expected_keys
            and len(record["points"]) == len(expected_keys)
        )
        covered = len(covered_by_target.get(target_id, set()) & expected_keys)
        fragments = len(track_key_lists)
        total_fragments += fragments
        clean_present = clean_count > 0
        exclusive = clean_count == 1 and fragments == 1
        full_clean_present_count += int(clean_present)
        exclusive_full_count += int(exclusive)
        covered_all_count += int(covered == len(expected_keys))
        target_switches = sum(
            switches_by_track[str(record["output_track_id"])]
            for record in records
            if any(
                truth_by_key.get((int(point.frame_index), int(point.detection_id))) == target_id
                for point in record["points"]
            )
        )
        target_rows.append(
            {
                "scenario": scenario,
                "version": version,
                "target_id": target_id,
                "expected_observations": len(expected_keys),
                "covered_observations": covered,
                "observation_coverage_fraction": f"{covered / len(expected_keys):.6f}",
                "predicted_tracks_containing_target": fragments,
                "full_clean_track_count": clean_count,
                "full_clean_track_present": int(clean_present),
                "exclusive_full_track": int(exclusive),
                "identity_switches_on_containing_tracks": target_switches,
            }
        )

    expected_total = sum(len(keys) for keys in expected_by_target.values())
    summary = {
        "scenario": scenario,
        "version": version,
        "input_sha256": input_sha256,
        "target_count": len(expected_by_target),
        "truth_observations_total": expected_total,
        "truth_observations_covered_unique": len(assigned_truth_keys),
        "truth_observation_assignments_in_tracks": assigned_truth_observations,
        "duplicated_truth_observation_assignments": assigned_truth_observations - len(assigned_truth_keys),
        "targets_with_all_observations_covered": covered_all_count,
        "targets_with_full_clean_track_present": full_clean_present_count,
        "targets_with_exclusive_full_track": exclusive_full_count,
        "target_track_assignments_fragment_count_sum": total_fragments,
        "identity_switch_count": sum(switches_by_track.values()),
        "mixed_truth_track_count": mixed_track_count,
        "background_contaminated_target_track_count": background_contamination_count,
        "predicted_track_count": len(records),
        "registration_max_shift_px": f"{registration_max_shift_px:.9f}",
        "runtime_seconds": f"{runtime_seconds:.9f}",
    }
    return summary, track_rows, target_rows, point_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_crossing_plot(
    path: Path,
    truth_rows: list[dict[str, Any]],
    point_rows: list[dict[str, object]],
) -> None:
    """Draw truth paths beside baseline/supplement output using frozen records."""

    width, height, gap = 1160, 500, 22
    panel_width = (width - 3 * gap) // 2
    image = Image.new("RGB", (width, height), (239, 243, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("segoeui.ttf"),
    ):
        try:
            if font_path.is_file() or str(font_path) == "segoeui.ttf":
                font = ImageFont.truetype(str(font_path), 15)
                small_font = ImageFont.truetype(str(font_path), 12)
                break
        except OSError:
            continue
    panels = (
        ("slow_linked_identity_swap", "慢速直线交叉：A 已连成混合 ID 轨迹"),
        ("fast_orthogonal_supplement", "快速正交交叉：A 碎片，B 补充完整轨迹"),
    )
    truth_colors = ((245, 174, 54), (171, 84, 199))
    for panel_index, (scenario, title) in enumerate(panels):
        left = gap + panel_index * (panel_width + gap)
        top = gap
        draw.rounded_rectangle((left, top, left + panel_width, height - gap), radius=12, fill=(255, 255, 255), outline=(205, 215, 228), width=2)
        draw.text((left + 16, top + 12), title, fill=(24, 38, 58), font=font)
        plot_left, plot_top = left + 38, top + 70
        plot_width, plot_height = panel_width - 60, height - 130
        draw.rectangle((plot_left, plot_top, plot_left + plot_width, plot_top + plot_height), outline=(213, 221, 232), width=1)
        scenario_truth = [row for row in truth_rows if row["scenario"] == scenario]
        coords = [(float(row["x_px"]), float(row["y_px"])) for row in scenario_truth]
        min_x = min(x for x, _ in coords) - 12
        max_x = max(x for x, _ in coords) + 12
        min_y = min(y for _, y in coords) - 12
        max_y = max(y for _, y in coords) + 12
        scale = min(plot_width / max(1.0, max_x - min_x), plot_height / max(1.0, max_y - min_y))
        offset_x = plot_left + (plot_width - (max_x - min_x) * scale) / 2
        offset_y = plot_top + (plot_height - (max_y - min_y) * scale) / 2

        def map_point(x: float, y: float) -> tuple[float, float]:
            return offset_x + (x - min_x) * scale, offset_y + (y - min_y) * scale

        def draw_arrow(points: list[tuple[float, float]], index: int, color: tuple[int, int, int]) -> None:
            if index <= 0 or index >= len(points):
                return
            x0, y0 = points[index - 1]
            x1, y1 = points[index]
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 0.1:
                return
            ux, uy = dx / length, dy / length
            size = 7.0
            draw.line((x0, y0, x1, y1), fill=color, width=3)
            draw.line((x1, y1, x1 - size * ux + size * 0.55 * uy, y1 - size * uy - size * 0.55 * ux), fill=color, width=3)
            draw.line((x1, y1, x1 - size * ux - size * 0.55 * uy, y1 - size * uy + size * 0.55 * ux), fill=color, width=3)

        targets = sorted({str(row["target_id"]) for row in scenario_truth})
        for target_index, target_id in enumerate(targets):
            target_rows = sorted(
                (row for row in scenario_truth if row["target_id"] == target_id),
                key=lambda row: int(row["frame_index"]),
            )
            points = [map_point(float(row["x_px"]), float(row["y_px"])) for row in target_rows]
            color = truth_colors[target_index % len(truth_colors)]
            draw.line(points, fill=color, width=3)
            draw_arrow(points, len(points) // 2, color)
            for px, py in points:
                draw.ellipse((px - 2.5, py - 2.5, px + 2.5, py + 2.5), fill=color)

        legend_items = (
            (truth_colors[0], "真值 A"),
            (truth_colors[1], "真值 B"),
            ((207, 77, 77), "普通关联"),
            ((38, 150, 138), "速度补充"),
        )
        legend_x = left + 16
        for color, label in legend_items:
            draw.line((legend_x, top + 52, legend_x + 16, top + 52), fill=color, width=3)
            draw.text((legend_x + 20, top + 44), label, fill=(50, 62, 78), font=small_font)
            legend_x += 116

        arms = (
            ("A", (207, 77, 77)),
            ("B_supplement", (38, 150, 138)),
        )
        for arm, color in arms:
            track_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in point_rows:
                if row["scenario"] == scenario and row["origin_arm"] == arm and row["truth_target_id"]:
                    track_groups[str(row["output_track_id"])].append(row)
            arm_colors = ((213, 72, 79), (55, 113, 199), (142, 79, 171), (199, 119, 45))
            for track_index, rows in enumerate(track_groups.values()):
                rows.sort(key=lambda row: int(row["frame_index"]))
                points = [map_point(float(row["x_px"]), float(row["y_px"])) for row in rows]
                track_color = arm_colors[track_index % len(arm_colors)] if arm == "A" else color
                if len(points) >= 2:
                    draw.line(points, fill=track_color, width=2)
                    if arm == "A" and scenario == "slow_linked_identity_swap":
                        draw_arrow(points, 5, track_color)
                        draw_arrow(points, 11, track_color)
                    else:
                        draw_arrow(points, len(points) // 2, track_color)
                radius = 3.2 if arm == "A" else 2.2
                for px, py in points:
                    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=track_color)
    image.save(path)


def run_motion_crossing_ablation(out_dir: str | Path) -> dict[str, object]:
    """Run deterministic slow- and fast-crossing association comparisons.

    The A input and the input passed to the B supplement are each reloaded from
    the exact same frozen CSV. B is explicitly the union of A tracks and the
    supplemental tracks; supplement-only rows are also exported so its direct
    contribution stays visible.
    """

    output = Path(out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frozen_dir = output / "frozen_inputs"
    frozen_dir.mkdir()

    summary_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    target_summary_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    truth_rows_all: list[dict[str, Any]] = []
    input_records: dict[str, dict[str, Any]] = {}
    runtime_by_scenario: dict[str, dict[str, float]] = {}
    registration_by_scenario: dict[str, dict[str, Any]] = {}

    for scenario in _SCENARIOS:
        generated_frames, truth_rows = _build_scenario(scenario)
        truth_rows_all.extend(truth_rows)
        frozen_path = frozen_dir / f"{scenario}.csv.gz"
        frame_sources, frozen_hash, before_fingerprint = _write_and_reload_frozen_sources(
            frozen_path, generated_frames, truth_rows
        )
        if _fingerprint_frames(generated_frames) != before_fingerprint:
            raise ValueError(f"frozen Detection reload changed the generated inputs for {scenario}")
        source_count_by_frame = [len(sources) for sources in frame_sources]

        started = time.perf_counter()
        baseline = track_detections(
            frame_sources,
            frame_names=[f"synthetic:{scenario}:{index:02d}" for index in range(FRAME_COUNT)],
            frame_times=[float(index) for index in range(FRAME_COUNT)],
            link_radius_px=LINK_RADIUS_PX,
            min_presence=MIN_PRESENCE,
            motion_min_displacement_px=MOTION_MIN_DISPLACEMENT_PX,
            registration_radius_px=REGISTRATION_RADIUS_PX,
            max_motion_fit_rms_px=BASELINE_MAX_FIT_RMS_PX,
        )
        baseline_seconds = time.perf_counter() - started
        fingerprint_before_supplement = _fingerprint_frames(frame_sources)
        if fingerprint_before_supplement != before_fingerprint:
            raise ValueError(f"baseline association mutated the frozen Detection list for {scenario}")

        started = time.perf_counter()
        supplement = track_fast_point_movers(
            frame_sources,
            baseline.cumulative_shifts,
            frame_times=[float(index) for index in range(FRAME_COUNT)],
            link_radius_px=LINK_RADIUS_PX,
            min_presence=MIN_PRESENCE,
            motion_min_displacement_px=MOTION_MIN_DISPLACEMENT_PX,
            min_flux_snr=SUPPLEMENT_MIN_FLUX_SNR,
            max_step_px=SUPPLEMENT_MAX_STEP_PX,
            gate_px=SUPPLEMENT_PREDICTION_GATE_PX,
            max_fit_rms_px=SUPPLEMENT_MAX_FIT_RMS_PX,
            existing_tracks=baseline.tracks,
        )
        supplement_seconds = time.perf_counter() - started
        fingerprint_after_supplement = _fingerprint_frames(frame_sources)
        if fingerprint_after_supplement != before_fingerprint:
            raise ValueError(f"supplement association mutated the frozen Detection list for {scenario}")

        registration_max_shift = max(
            (math.hypot(float(dx), float(dy)) for dx, dy in baseline.cumulative_shifts),
            default=0.0,
        )
        if registration_max_shift > 1e-9:
            raise ValueError(f"static anchors failed to hold registration at zero for {scenario}")
        registration_by_scenario[scenario] = {
            "cumulative_shifts_px": [list(map(float, shift)) for shift in baseline.cumulative_shifts],
            "max_shift_px": registration_max_shift,
            "zero_verified": True,
            "static_anchor_count_per_frame": len(_ANCHORS),
        }

        baseline_records = [
            _track_record(track, index, "A") for index, track in enumerate(baseline.tracks)
        ]
        supplement_records = [
            _track_record(track, index, "B_supplement") for index, track in enumerate(supplement)
        ]
        union_records = baseline_records + supplement_records
        truth_by_id = {int(spec["detection_id"]): str(spec["target_id"]) for spec in _SCENARIOS[scenario]}
        input_records[scenario] = {
            "path": str(frozen_path),
            "sha256": frozen_hash,
            "detection_count": sum(source_count_by_frame),
            "detection_count_by_frame": source_count_by_frame,
            "same_input_sha256_for_A_and_B": True,
            "A_input_fingerprint": before_fingerprint,
            "B_input_fingerprint": fingerprint_before_supplement,
            "post_B_input_fingerprint": fingerprint_after_supplement,
            "A_and_B_same_Detection_list": (
                before_fingerprint == fingerprint_before_supplement == fingerprint_after_supplement
            ),
        }
        runtime_by_scenario[scenario] = {
            "A_baseline_seconds": baseline_seconds,
            "B_supplement_seconds": supplement_seconds,
        }

        for version, records, seconds in (
            ("A", baseline_records, baseline_seconds),
            ("B_supplement", supplement_records, supplement_seconds),
            ("B_union", union_records, baseline_seconds + supplement_seconds),
        ):
            summary, tracks_out, targets_out, points_out = _score_variant(
                scenario,
                version,
                records,
                truth_rows,
                frozen_hash,
                registration_max_shift,
                seconds,
            )
            summary["supplement_track_count"] = len(supplement)
            summary_rows.append(summary)
            track_rows.extend(tracks_out)
            target_summary_rows.extend(targets_out)
            point_rows.extend(points_out)

    truth_path = output / "truth.csv"
    _write_csv(truth_path, truth_rows_all)
    summary_path = output / "summary.csv"
    _write_csv(summary_path, summary_rows)
    track_path = output / "track_level.csv"
    _write_csv(track_path, track_rows)
    target_path = output / "target_summary.csv"
    _write_csv(target_path, target_summary_rows)
    points_path = output / "track_points.csv"
    _write_csv(points_path, point_rows)
    plot_path = output / "motion_crossing_tracks.png"
    _save_crossing_plot(plot_path, truth_rows_all, point_rows)

    artifacts = {
        "truth_csv": truth_path,
        "summary_csv": summary_path,
        "track_level_csv": track_path,
        "target_summary_csv": target_path,
        "track_points_csv": points_path,
        "motion_crossing_plot": plot_path,
    }
    artifacts.update(
        {f"frozen_input_{scenario}": Path(record["path"]) for scenario, record in input_records.items()}
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment": "synthetic_crossing_track_association_ablation",
        "status": "completed",
        "created_local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "association-stage synthetic Detection experiment; no image detector or physical identity is tested",
        "frame_count": FRAME_COUNT,
        "frame_times_seconds": [float(index) for index in range(FRAME_COUNT)],
        "scenarios": {
            name: {
                "targets": [
                    {
                        "target_id": spec["target_id"],
                        "detection_id": spec["detection_id"],
                        "start_registered_xy_px": list(spec["start_registered_xy_px"]),
                        "velocity_px_per_frame": list(spec["velocity_px_per_frame"]),
                        "expected_frames": list(range(FRAME_COUNT)),
                    }
                    for spec in specs
                ],
                "input": input_records[name],
                "registration": registration_by_scenario[name],
                "runtime_seconds": runtime_by_scenario[name],
            }
            for name, specs in _SCENARIOS.items()
        },
        "static_anchors": {
            "count_per_frame": len(_ANCHORS),
            "positions_px": [list(point) for point in _ANCHORS],
            "purpose": "hold the estimated global translation at exactly zero while crossing detections are associated",
        },
        "versions": [
            {"id": "A", "definition": "track_detections output with fixed link/registration parameters"},
            {
                "id": "B_supplement",
                "definition": (
                    "track_fast_point_movers on the same frozen Detection list; "
                    "A tracks are passed as existing_tracks"
                ),
            },
            {
                "id": "B_union",
                "definition": (
                    "A tracks concatenated with B_supplement tracks; "
                    "duplicate observations are retained and counted"
                ),
            },
        ],
        "parameters": {
            "link_radius_px": LINK_RADIUS_PX,
            "registration_radius_px": REGISTRATION_RADIUS_PX,
            "min_presence_frames": MIN_PRESENCE,
            "motion_min_displacement_px": MOTION_MIN_DISPLACEMENT_PX,
            "baseline_max_motion_fit_rms_px": BASELINE_MAX_FIT_RMS_PX,
            "supplement_min_flux_snr": SUPPLEMENT_MIN_FLUX_SNR,
            "supplement_max_step_px_per_frame": SUPPLEMENT_MAX_STEP_PX,
            "supplement_prediction_gate_px": SUPPLEMENT_PREDICTION_GATE_PX,
            "supplement_max_fit_rms_px": SUPPLEMENT_MAX_FIT_RMS_PX,
            "detection_dtype": "not applicable; this is a frozen Detection-record experiment without pixel arrays",
            "anchor_detection_snr": 20.0,
            "target_detection_snr": 20.0,
        },
        "metric_definitions": {
            "identity_switch": "adjacent truth-labelled observations in a predicted track have different target IDs",
            "mixed_truth_track": "one predicted track contains detections from more than one injected target ID",
            "fragment_count": (
                "number of output tracks containing at least one observation for the target; "
                "B_union retains A and supplement outputs"
            ),
            "full_clean_track_present": (
                "at least one output track contains exactly every expected target observation "
                "and no other observation"
            ),
            "exclusive_full_track": "exactly one target-containing track exists and it is a full clean track",
            "duplicate_observation_assignment": (
                "the same (frame_index, detection_id) appears in multiple output tracks"
            ),
            "truth_observation_coverage": (
                "unique expected (frame_index, detection_id) keys represented in output tracks "
                "divided by expected keys"
            ),
        },
        "limitations": [
            (
                "All Detection records are synthetic and quality_passed; no FITS pixels, detector "
                "proposals, photometry uncertainty, or quality gate are exercised."
            ),
            (
                "The two crossing scenarios establish deterministic behavior of these association "
                "functions only; they do not establish physical identity or sky-wide precision."
            ),
            (
                "B is an additive supplement and cannot replace or correct any A track; B_union "
                "intentionally preserves duplicate baseline singleton assignments."
            ),
            (
                "The cases use two clean targets and static anchors; crowded backgrounds, noise, "
                "missing frames, and detector misses are not part of this crossing experiment."
            ),
        ],
        "outputs": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in artifacts.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["run_motion_crossing_ablation"]
