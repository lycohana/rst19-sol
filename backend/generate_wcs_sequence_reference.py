"""Freeze prior-WCS fitting and sequence motion-metric reference cases."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.catalog import CatalogSource
from rst19.detection import Detection
from rst19.matching import CatalogMatch
from rst19.sequence import (
    TrackPoint,
    _derive_motion_displacement_floor,
    _estimate_translation,
    _fit_track_coordinates,
    _global_frame_assignments,
    track_detections,
)
from rst19.wcs import TangentPlaneWCS, fit_affine_wcs_from_matches
from rst19.wcs_validation import validate_frame_matches


def build_fit_case(reference, sky_points, matrix, offset, outlier=None):
    catalog = []
    matches = []
    for index, (east, north) in enumerate(sky_points):
        ra, dec = reference.pixel_to_world(
            reference.crpix_x + float(east) / reference.pixel_scale_arcsec,
            reference.crpix_y + float(north) / reference.pixel_scale_arcsec,
        )
        source_id = f"s{index}"
        catalog.append(CatalogSource(source_id, float(ra), float(dec)))
        pixel = matrix @ np.asarray((east, north), dtype=np.float64) + offset
        if outlier == index:
            pixel += np.asarray((35.0, -28.0), dtype=np.float64)
        matches.append(
            CatalogMatch(
                index,
                source_id,
                float(pixel[0]),
                float(pixel[1]),
                float(pixel[0]),
                float(pixel[1]),
                0.0,
                None,
            )
        )
    return catalog, matches


def source(detection_id, x, y, flux_snr=25.0):
    return Detection(
        detection_id=detection_id,
        x=float(x),
        y=float(y),
        peak=100.0,
        flux=500.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=3.0,
        flags=(),
        flux_error=20.0,
        flux_snr=flux_snr,
        quality_passed=True,
    )


def tracking_case(label, frames, options):
    result = track_detections(frames, **options)
    return {
        "label": label,
        "frames": [[row.as_dict() for row in frame] for frame in frames],
        "options": options,
        "expected": result.as_dict(),
    }


def main():
    reference_spec = {
        "center_ra_deg": 10.0,
        "center_dec_deg": 20.0,
        "pixel_scale_arcsec": 20.0,
        "crpix_x": 512.0,
        "crpix_y": 512.0,
        "rotation_deg": 0.0,
        "parity": 1,
    }
    reference = TangentPlaneWCS(**reference_spec)
    sky_points = np.asarray(
        [
            (-180.0, -120.0),
            (-100.0, 140.0),
            (-20.0, -40.0),
            (70.0, 180.0),
            (130.0, -160.0),
            (210.0, 40.0),
            (35.0, -210.0),
        ],
        dtype=np.float64,
    )
    matrix = np.asarray(((-0.041, 0.012), (0.012, 0.041)), dtype=np.float64)
    offset = np.asarray((480.0, 535.0), dtype=np.float64)
    catalog, matches = build_fit_case(reference, sky_points, matrix, offset, outlier=6)
    fit_cases = []
    for label, selected, min_matches in (
        ("outlier_clipped", matches, 5),
        ("insufficient", matches[:4], 6),
    ):
        try:
            expected = fit_affine_wcs_from_matches(selected, catalog, reference, min_matches=min_matches).as_dict()
            error = None
        except ValueError as exc:
            expected = None
            error = str(exc)
        fit_cases.append(
            {
                "label": label,
                "matches": [match.as_dict() for match in selected],
                "catalog": [source.as_dict() for source in catalog],
                "reference_wcs": reference_spec,
                "epoch": None,
                "fit_options": {"min_matches": min_matches},
                "expected": expected,
                "error": error,
            }
        )

    collinear_points = np.asarray([(value, 0.0) for value in (-30, 0, 30, 60, 90, 120)], dtype=np.float64)
    collinear_catalog, collinear_matches = build_fit_case(
        reference,
        collinear_points,
        np.asarray(((-0.04, 0.0), (0.0, 0.04))),
        np.asarray((100.0, 200.0)),
    )
    try:
        fit_affine_wcs_from_matches(collinear_matches, collinear_catalog, reference, min_matches=6)
    except ValueError as exc:
        collinear_error = str(exc)
    else:
        raise AssertionError("collinear fit unexpectedly succeeded")
    fit_cases.append(
        {
            "label": "collinear",
            "matches": [match.as_dict() for match in collinear_matches],
            "catalog": [source.as_dict() for source in collinear_catalog],
            "reference_wcs": reference_spec,
            "epoch": None,
            "fit_options": {"min_matches": 6},
            "expected": None,
            "error": collinear_error,
        }
    )

    frame_matches = []
    for frame_index, frame_offset in enumerate((offset, offset + np.asarray((0.4, -0.3)))):
        frame_catalog, frame_rows = build_fit_case(reference, sky_points[:6], matrix, frame_offset)
        assert [source.source_id for source in frame_catalog] == [source.source_id for source in catalog[:6]]
        frame_matches.append([match.as_dict() for match in frame_rows])
    validation_expected = validate_frame_matches(
        [
            [CatalogMatch(**match) for match in frame]
            for frame in frame_matches
        ]
        + [[]],
        catalog[:6],
        reference,
        frame_paths=("frame-01.fits", "frame-02.fits", "frame-03.fits"),
        min_matches=6,
    ).as_dict()
    validation_case = {
        "frame_matches": frame_matches + [[]],
        "catalog": [source.as_dict() for source in catalog[:6]],
        "reference_wcs": reference_spec,
        "frame_paths": ["frame-01.fits", "frame-02.fits", "frame-03.fits"],
        "epoch": None,
        "fit_options": {"min_matches": 6},
        "expected": validation_expected,
    }

    points = [
        TrackPoint(index, index, float(2 * index), 0.0, float(2 * index), 0.0, 20.0).as_dict()
        for index in range(3)
    ]
    motion_cases = []
    for label, selected, times in (
        ("elapsed_seconds", points, [0.0, 1.0, 3.0]),
        ("irregular_full_vector", [points[0], points[2]], [0.0, 1.0, 3.0]),
        ("invalid_time_fallback", points, [0.0, 4.0, 3.0]),
        ("single_observation", points[:1], None),
    ):
        point_rows = [
            {
                "frame_index": point["frame_index"],
                "aligned_x": point["aligned_x"],
                "aligned_y": point["aligned_y"],
            }
            for point in selected
        ]
        expected = _fit_track_coordinates(
            [TrackPoint(**point) for point in selected],
            np.asarray([[point["aligned_x"], point["aligned_y"]] for point in selected], dtype=np.float64),
            frame_times=times,
        )
        motion_cases.append(
            {
                "label": label,
                "points": point_rows,
                "frame_times": times,
                "expected": {
                    "displacement_px": expected[0],
                    "speed_px_per_frame": expected[1],
                    "fit_rms_px": expected[2],
                },
            }
        )

    floor_cases = []
    for label, metrics, affine_frame_count in (
        ("no_affine", [(15, 1.0, 1.0, 0.1)] * 40, 0),
        ("too_few_tracks", [(15, 1.0, 1.0, 0.1)] * 29, 2),
        (
            "robust_floor_and_gates",
            [(15, 1.0 + index * 0.01, 1.0, 0.1) for index in range(39)]
            + [(15, 100.0, 1.0, 0.1), (15, 50.0, 1.0, 1.0), (8, 40.0, 1.0, 0.1)],
            4,
        ),
    ):
        expected = _derive_motion_displacement_floor(
            metrics,
            required_presence=12,
            user_floor=2.0,
            max_motion_fit_rms_px=0.75,
            affine_frame_count=affine_frame_count,
        )
        floor_cases.append(
            {
                "label": label,
                "metrics": [
                    {
                        "presence": row[0],
                        "displacement_px": row[1],
                        "speed_px_per_frame": row[2],
                        "fit_rms_px": row[3],
                    }
                    for row in metrics
                ],
                "required_presence": 12,
                "user_floor": 2.0,
                "max_motion_fit_rms_px": 0.75,
                "affine_frame_count": affine_frame_count,
                "expected": expected,
            }
        )

    previous_grid = [
        source(index, 24.0 + (index % 6) * 40.0, 24.0 + (index // 6) * 60.0)
        for index in range(24)
    ]
    translation_inputs = [
        (
            "large_consensus_drift",
            previous_grid,
            [source(index, row.x - 2.0, row.y + 18.0) for index, row in enumerate(previous_grid)],
            8.0,
        ),
        (
            "empty_frames",
            [],
            previous_grid,
            8.0,
        ),
        (
            "no_neighbor_inside_search",
            [source(1, 0.0, 0.0)],
            [source(2, 64.0, 0.0)],
            8.0,
        ),
        (
            "two_displacement_clusters_fallback",
            [
                source(0, 0.0, 0.0),
                source(1, 100.0, 0.0),
                source(2, 0.0, 100.0),
                source(3, 100.0, 100.0),
            ],
            [
                source(0, 1.0, 2.0),
                source(1, 120.0, 20.0),
                source(2, 1.0, 102.0),
                source(3, 120.0, 120.0),
            ],
            8.0,
        ),
    ]
    translation_cases = []
    for label, previous, current, radius in translation_inputs:
        expected = _estimate_translation(previous, current, radius_px=radius)
        translation_cases.append(
            {
                "label": label,
                "previous": [row.as_dict() for row in previous],
                "current": [row.as_dict() for row in current],
                "radius_px": radius,
                "expected": {"translation_px": list(expected)},
            }
        )

    assignment_inputs = [
        (
            "two_tracks_global_solution",
            [(10, 0.0, 0.0), (11, 1.0, 0.0)],
            [(0.9, 0.0), (2.8, 0.0)],
            2.0,
            None,
            None,
        ),
        (
            "pair_gate",
            [(10, 0.0, 0.0), (11, 1.0, 0.0)],
            [(0.2, 0.0), (1.2, 0.0)],
            2.0,
            [[11, 0], [11, 1]],
            None,
        ),
        (
            "soft_geometry_cost",
            [(10, 0.0, 0.0)],
            [(2.0, 0.0), (0.0, 2.0)],
            3.0,
            None,
            [[10, 0, 0.0], [10, 1, 4.0]],
        ),
    ]
    assignment_cases = []
    for label, tracks, points, radius, allowed_pairs, pair_costs in assignment_inputs:
        allowed = None
        if allowed_pairs is not None:
            allowed_set = {(int(track), int(source_index)) for track, source_index in allowed_pairs}
            allowed = lambda track_id, source_index, _distance: (track_id, source_index) in allowed_set
        cost_map = None
        if pair_costs is not None:
            cost_map = {(int(track), int(source_index)): float(cost) for track, source_index, cost in pair_costs}
        cost = None if cost_map is None else lambda track_id, source_index, _distance: cost_map.get((track_id, source_index), 0.0)
        expected = _global_frame_assignments(
            tracks,
            np.asarray(points, dtype=np.float64),
            link_radius_px=radius,
            pair_allowed=allowed,
            pair_cost=cost,
        )
        assignment_cases.append(
            {
                "label": label,
                "track_positions": tracks,
                "current_points": points,
                "link_radius_px": radius,
                "allowed_pairs": allowed_pairs,
                "pair_costs": pair_costs,
                "expected": {"assignments": expected},
            }
        )

    tracking_cases = []
    static_positions = ((10.0, 10.0), (50.0, 10.0), (10.0, 50.0), (50.0, 50.0), (70.0, 70.0))
    tracking_cases.append(
        tracking_case(
            "static_and_moving",
            [
                [source(index, x, y) for index, (x, y) in enumerate(static_positions)] + [source(5, 30.0 + 2.0 * frame_index, 20.0)]
                for frame_index in range(3)
            ],
            {"min_presence": 3, "link_radius_px": 3.0, "motion_min_displacement_px": 2.0},
        )
    )
    tracking_cases.append(
        tracking_case(
            "missed_frame_splits_track",
            [[source(1, 10.0, 10.0)], [], [source(2, 11.0, 10.0)]],
            {"min_presence": 1, "link_radius_px": 3.0},
        )
    )
    grid = [(100.0 + column * 120.0, 100.0 + row * 120.0) for row in range(4) for column in range(4)]
    affine_frames = []
    center = np.asarray((280.0, 280.0), dtype=np.float64)
    for frame_index in range(5):
        angle = 0.00032 * frame_index
        matrix = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        positions = (np.asarray(grid) - center) @ matrix.T + center + np.asarray((0.4 * frame_index, -0.2 * frame_index))
        affine_frames.append([source(index, x, y) for index, (x, y) in enumerate(positions)])
    tracking_cases.append(
        tracking_case(
            "affine_drift_and_elapsed_times",
            affine_frames,
            {
                "frame_times": [0.0, 1.0, 2.4, 4.0, 5.5],
                "link_radius_px": 3.0,
                "registration_radius_px": 4.0,
                "motion_min_displacement_px": 1.5,
            },
        )
    )
    moving_outlier_frames = []
    for frame_index in range(4):
        angle = 0.00025 * frame_index
        matrix = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        positions = (np.asarray(grid) - center) @ matrix.T + center + np.asarray((0.3 * frame_index, -0.1 * frame_index))
        rows = [source(index, x, y) for index, (x, y) in enumerate(positions)]
        rows.append(source(100, 900.0 + 2.0 * frame_index, 920.0))
        moving_outlier_frames.append(rows)
    tracking_cases.append(
        tracking_case(
            "affine_ransac_rejects_moving_outlier",
            moving_outlier_frames,
            {"min_presence": 4, "link_radius_px": 4.0, "registration_radius_px": 4.0},
        )
    )

    source_hashes = {
        name: hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        for name in ("wcs.py", "wcs_validation.py", "sequence.py")
    }
    target = ROOT / "backend/tests/fixtures/wcs_sequence.json"
    target.write_text(
        json.dumps(
            {
                "source_hashes": source_hashes,
                "fit_cases": fit_cases,
                "validation_case": validation_case,
                "motion_cases": motion_cases,
                "floor_cases": floor_cases,
                "translation_cases": translation_cases,
                "assignment_cases": assignment_cases,
                "tracking_cases": tracking_cases,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
