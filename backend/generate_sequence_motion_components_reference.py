"""Capture the Python reference for native per-frame motion components."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rst19.sequence import _extract_motion_frame_components


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "tests/fixtures/sequence_motion_components.json"


def paint_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    value: float = 30.0,
    thickness: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    count = max(abs(x1 - x0), abs(y1 - y0)) + 1
    for index in range(count):
        fraction = index / max(1, count - 1)
        x = round(x0 + fraction * (x1 - x0))
        y = round(y0 + fraction * (y1 - y0))
        radius = thickness // 2
        image[
            max(0, y - radius) : min(image.shape[0], y + radius + 1),
            max(0, x - radius) : min(image.shape[1], x + radius + 1),
        ] = value


def normalize_axis_angle(value: float) -> float:
    """Canonicalize the unoriented PCA axis to [-90, 90) degrees."""
    return (float(value) + 90.0) % 180.0 - 90.0


def capture(label: str, image: np.ndarray, **changes: object) -> dict[str, object]:
    image = np.asarray(image, dtype=np.float32)
    valid = np.ones(image.shape, dtype=bool)
    artifact = np.zeros(image.shape, dtype=bool)
    parameters: dict[str, object] = {
        "frame_index": 2,
        "shift": [1.25, -0.75],
        "threshold_adu": 10.0,
        "residual_noise_adu": 5.0,
        "psf_fwhm": 2.0,
        "min_length_px": 8.0,
        "min_feature_area": 8,
        "min_axis_ratio": 4.0,
        "bridge_gaps": True,
        "short_min_pixels": 6,
    }
    parameters.update(changes)
    request: dict[str, object] = {
        "residual": image.tolist(),
        "valid_mask": valid.tolist(),
        "line_artifact_mask": artifact.tolist(),
        **parameters,
    }
    points, audit = _extract_motion_frame_components(
        image,
        valid,
        artifact,
        frame_index=int(parameters["frame_index"]),
        shift=tuple(parameters["shift"]),
        threshold=float(parameters["threshold_adu"]),
        residual_noise=float(parameters["residual_noise_adu"]),
        psf_fwhm=float(parameters["psf_fwhm"]),
        min_length=float(parameters["min_length_px"]),
        min_feature_area=int(parameters["min_feature_area"]),
        min_axis_ratio=float(parameters["min_axis_ratio"]),
        bridge_gaps=bool(parameters["bridge_gaps"]),
        short_min_pixels=int(parameters["short_min_pixels"]),
    )
    expected_points = [point.as_dict() for point in points]
    for point in expected_points:
        point["angle_deg"] = normalize_axis_angle(float(point["angle_deg"]))
    return {
        "label": label,
        "request": request,
        "expected": {"points": expected_points, "audit": audit.as_dict()},
    }


def main() -> None:
    cases: list[dict[str, object]] = []

    image = np.zeros((32, 40), dtype=np.float32)
    cases.append(capture("empty_support", image))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (8, 10), (23, 10), thickness=3)
    cases.append(capture("horizontal_short_trail", image))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (18, 5), (18, 25), thickness=3)
    cases.append(capture("vertical_long_trail", image))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (5, 5), (21, 21), thickness=3)
    cases.append(capture("diagonal_positive_axis", image))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (5, 24), (21, 8), thickness=3)
    cases.append(capture("diagonal_negative_axis", image))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (7, 14), (23, 14), thickness=3)
    image[14, 15] = 0.0
    cases.append(capture("single_pixel_gap_bridged", image, bridge_gaps=True))
    cases.append(capture("single_pixel_gap_not_bridged", image, bridge_gaps=False))

    image = np.zeros((32, 40), dtype=np.float32)
    image[12:19, 16:23] = 30.0
    cases.append(capture("compact_blob_rejected_by_axis_ratio", image))

    image = np.zeros((32, 40), dtype=np.float32)
    yy, xx = np.indices(image.shape)
    for center_x in (13.0, 21.0):
        image += (30.0 * np.exp(-0.5 * (((xx - center_x) ** 2 + (yy - 15.0) ** 2) / 1.2**2))).astype(np.float32)
    cases.append(capture("adjacent_double_source_component", image, threshold_adu=5.0, min_axis_ratio=2.0))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (5, 10), (9, 10), thickness=1)
    cases.append(capture("component_below_short_area_gate", image, short_min_pixels=8))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (7, 12), (17, 12), thickness=3)
    cases.append(capture("length_threshold_rejection", image, min_length_px=20.0))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (8, 12), (29, 12), thickness=1)
    cases.append(capture("long_line_fails_area_floor", image, min_feature_area=80))
    cases.append(capture("negative_area_floor_uses_psf_area_minimum", image, min_feature_area=-12))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (0, 12), (12, 12), thickness=3)
    cases.append(capture("short_edge_candidate_rejected", image))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (0, 12), (23, 12), thickness=3)
    cases.append(capture("long_edge_candidate_retained", image))

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (8, 15), (24, 15), thickness=3)
    valid = np.ones(image.shape, dtype=bool)
    valid[15, 15] = False
    artifact = np.zeros(image.shape, dtype=bool)
    artifact[14:17, 8:17] = True
    request_case = capture("masked_support_and_artifact_overlap", image)
    request_case["request"]["valid_mask"] = valid.tolist()
    request_case["request"]["line_artifact_mask"] = artifact.tolist()
    points, audit = _extract_motion_frame_components(
        image,
        valid,
        artifact,
        frame_index=2,
        shift=(1.25, -0.75),
        threshold=10.0,
        residual_noise=5.0,
        psf_fwhm=2.0,
        min_length=8.0,
        min_feature_area=8,
        min_axis_ratio=4.0,
        bridge_gaps=True,
        short_min_pixels=6,
    )
    expected_points = [point.as_dict() for point in points]
    for point in expected_points:
        point["angle_deg"] = normalize_axis_angle(float(point["angle_deg"]))
    request_case["expected"] = {"points": expected_points, "audit": audit.as_dict()}
    cases.append(request_case)

    image = np.zeros((32, 40), dtype=np.float32)
    paint_line(image, (5, 8), (17, 8), thickness=3, value=30.0)
    paint_line(image, (22, 22), (34, 22), thickness=3, value=45.0)
    cases.append(capture("two_disjoint_components_keep_scan_order", image))

    rng = np.random.default_rng(20261013)
    image = rng.normal(0.0, 2.0, size=(32, 40)).astype(np.float32)
    paint_line(image, (7, 18), (27, 18), thickness=3, value=25.0)
    cases.append(capture("deterministic_noise_background", image, threshold_adu=8.0, residual_noise_adu=4.0))

    image = np.zeros((24, 32), dtype=np.float32)
    paint_line(image, (6, 11), (22, 11), thickness=3, value=50.0)
    valid = np.ones(image.shape, dtype=bool)
    valid[:, :2] = False
    artifact = np.zeros(image.shape, dtype=bool)
    artifact[10:13, 6:14] = True
    points, audit = _extract_motion_frame_components(
        image,
        valid,
        artifact,
        frame_index=7,
        shift=(-0.5, 2.0),
        threshold=12.0,
        residual_noise=7.0,
        psf_fwhm=1.5,
        min_length=6.0,
        min_feature_area=6,
        min_axis_ratio=3.0,
        bridge_gaps=False,
        short_min_pixels=5,
    )
    expected_points = [point.as_dict() for point in points]
    for point in expected_points:
        point["angle_deg"] = normalize_axis_angle(float(point["angle_deg"]))
    cases.append(
        {
            "label": "custom_frame_shift_and_threshold",
            "request": {
                "residual": image.tolist(),
                "valid_mask": valid.tolist(),
                "line_artifact_mask": artifact.tolist(),
                "frame_index": 7,
                "shift": [-0.5, 2.0],
                "threshold_adu": 12.0,
                "residual_noise_adu": 7.0,
                "psf_fwhm": 1.5,
                "min_length_px": 6.0,
                "min_feature_area": 6,
                "min_axis_ratio": 3.0,
                "bridge_gaps": False,
                "short_min_pixels": 5,
            },
            "expected": {"points": expected_points, "audit": audit.as_dict()},
        }
    )

    error_base = cases[1]["request"]
    errors = [
        {
            "label": "mismatched_mask_shape",
            "request": {**error_base, "valid_mask": [[True]]},
            "expected_error": "residual, valid_mask, and line_artifact_mask shapes must match",
        },
        {
            "label": "non_boolean_validity_mask",
            "request": {**error_base, "valid_mask": [[1]]},
            "expected_error": "valid_mask values must be booleans",
        },
        {
            "label": "non_positive_psf",
            "request": {**error_base, "psf_fwhm": 0.0},
            "expected_error": "motion feature component thresholds must be finite and positive",
        },
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {"sequence.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "axis_angle_comparison": "angle_deg is canonicalized modulo 180 degrees because a PCA line axis is unoriented",
        "cases": cases,
        "errors": errors,
    }
    FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(cases)} motion-component cases and {len(errors)} validation cases to {FIXTURE}")


if __name__ == "__main__":
    main()
