"""Compare prior-WCS and sequence motion operations through the native process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from verify_detection import compare
sys.path.insert(0, str(ROOT / "src"))
from rst19.catalog import CatalogSource
from rst19.wcs import TangentPlaneWCS
from rst19.wcs_validation import run_sequence_wcs_validation


def write_fits(path: Path, image: np.ndarray) -> None:
    height, width = image.shape
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                  -32",
        "NAXIS   =                    2",
        f"NAXIS1  = {width:20d}",
        f"NAXIS2  = {height:20d}",
        "END",
    ]
    header = "".join(card.ljust(80)[:80] for card in cards).encode("ascii")
    header += b" " * ((-len(header)) % 2880)
    data = image.astype(">f4", copy=False).tobytes(order="C")
    data += b"\0" * ((-len(data)) % 2880)
    path.write_bytes(header + data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture_path = Path(__file__).parent / "tests/fixtures/wcs_sequence.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for name, digest in fixture["source_hashes"].items():
        actual = hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        assert actual == digest, f"Python baseline changed: {name}"

    requests = []
    for case in fixture["fit_cases"]:
        requests.append(
            (
                {
                    "operation": "wcs.affine.fit",
                    "matches": case["matches"],
                    "catalog": case["catalog"],
                    "reference_wcs": case["reference_wcs"],
                    "epoch": case["epoch"],
                    "fit_options": case["fit_options"],
                },
                case["expected"],
                case["error"] is not None,
                case["label"],
            )
        )
    case = fixture["validation_case"]
    requests.append(
        (
            {
                "operation": "wcs.validate.matches",
                **{key: value for key, value in case.items() if key != "expected"},
            },
            case["expected"],
            False,
            "per_frame_validation",
        )
    )
    for case in fixture["motion_cases"]:
        requests.append(
            (
                {
                    "operation": "sequence.track_motion.fit",
                    "points": case["points"],
                    "frame_times": case["frame_times"],
                },
                case["expected"],
                False,
                case["label"],
            )
        )
    for case in fixture["floor_cases"]:
        requests.append(
            (
                {
                    "operation": "sequence.motion_displacement_floor",
                    "metrics": case["metrics"],
                    "required_presence": case["required_presence"],
                    "user_floor": case["user_floor"],
                    "max_motion_fit_rms_px": case["max_motion_fit_rms_px"],
                    "affine_frame_count": case["affine_frame_count"],
                },
                {"effective_motion_min_displacement_px": case["expected"]},
                False,
                case["label"],
            )
        )
    for case in fixture["translation_cases"]:
        requests.append(
            (
                {
                    "operation": "sequence.translation.estimate",
                    "previous": case["previous"],
                    "current": case["current"],
                    "radius_px": case["radius_px"],
                },
                case["expected"],
                False,
                case["label"],
            )
        )
    for case in fixture["assignment_cases"]:
        requests.append(
            (
                {
                    "operation": "sequence.frame_assign",
                    "track_positions": case["track_positions"],
                    "current_points": case["current_points"],
                    "link_radius_px": case["link_radius_px"],
                    "allowed_pairs": case["allowed_pairs"],
                    "pair_costs": case["pair_costs"],
                },
                case["expected"],
                False,
                case["label"],
            )
        )

    for case in fixture["tracking_cases"]:
        requests.append(
            (
                {
                    "operation": "sequence.track_detections",
                    "frames": case["frames"],
                    "options": case["options"],
                },
                case["expected"],
                False,
                case["label"],
            )
        )

    records = []
    for request, expected, should_error, label in requests:
        response = subprocess.run(
            [str(args.engine.resolve())],
            input=json.dumps(request, ensure_ascii=False, allow_nan=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        event = json.loads(response.stdout)
        if should_error:
            assert response.returncode == 1 and not event["ok"], (label, event, response.stderr)
            assert event["error"] == next(
                case["error"] for case in fixture["fit_cases"] if case["label"] == label
            ), (label, event["error"])
        else:
            assert response.returncode == 0 and event["ok"], (label, event, response.stderr)
            compare(event["payload"], expected, label)
        records.append({"label": label, "operation": request["operation"], "all_fields_compared": True})

    sequence_options = {
        "threshold_sigma": 4.0,
        "psf_fwhm": 3.0,
        "min_distance": 3,
        "aperture_radius": 4,
        "background_box_size": 64,
        "min_flux_snr": 3.0,
        "max_sources": 200,
    }
    reference = TangentPlaneWCS(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        pixel_scale_arcsec=20.0,
        crpix_x=64.0,
        crpix_y=64.0,
    )
    star_pixels = [
        (18.0, 20.0), (48.0, 18.0), (91.0, 22.0), (112.0, 45.0),
        (23.0, 57.0), (61.0, 55.0), (101.0, 73.0), (31.0, 98.0),
        (70.0, 105.0), (111.0, 112.0), (45.0, 82.0), (83.0, 43.0),
    ]
    catalog = []
    for index, (x, y) in enumerate(star_pixels):
        ra, dec = reference.pixel_to_world(x, y)
        catalog.append(CatalogSource(f"s{index}", float(ra), float(dec)))
    rng = np.random.default_rng(19019)
    sequence_digests = []
    with tempfile.TemporaryDirectory(prefix="rst19-wcs-sequence-") as directory:
        paths = []
        for frame_index, (shift_x, shift_y) in enumerate(((0.0, 0.0), (0.4, -0.3))):
            image = rng.normal(100.0, 2.0, size=(128, 128)).astype(np.float32)
            for star_index, (x, y) in enumerate(star_pixels):
                center_x = x + shift_x
                center_y = y + shift_y
                sigma = 1.35
                amplitude = 250.0 + star_index * 12.0
                x0 = max(0, int(center_x) - 7)
                x1 = min(image.shape[1], int(center_x) + 8)
                y0 = max(0, int(center_y) - 7)
                y1 = min(image.shape[0], int(center_y) + 8)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                image[y0:y1, x0:x1] += amplitude * np.exp(
                    -0.5 * (((xx - center_x) / sigma) ** 2 + ((yy - center_y) / sigma) ** 2)
                ).astype(np.float32)
            path = Path(directory) / f"frame-{frame_index + 1:02}.fits"
            write_fits(path, image)
            paths.append(path.resolve())
            sequence_digests.append(
                {"frame": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        baseline = run_sequence_wcs_validation(
            paths,
            catalog,
            reference,
            detector_kwargs=sequence_options,
            match_radius_px=3.0,
            min_matches=6,
        )
        expected = baseline.as_dict()
        request = {
            "operation": "wcs.validate.sequence",
            "frame_paths": [str(path) for path in paths],
            "catalog": [source.as_dict() for source in catalog],
            "reference_wcs": reference.as_dict(),
            "options": sequence_options,
            "match_radius_px": 3.0,
            "min_matches": 6,
        }
        response = subprocess.run(
            [str(args.engine.resolve())],
            input=json.dumps(request, ensure_ascii=False, allow_nan=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        event = json.loads(response.stdout)
        assert response.returncode == 0 and event["ok"], (event, response.stderr)
        compare(event["payload"], expected, "sequence_file_pipeline")
        records.append(
            {
                "label": "sequence_file_pipeline",
                "operation": request["operation"],
                "frame_count": len(paths),
                "validated_count": expected["validated_count"],
                "all_fields_compared": True,
            }
        )

    report = {
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "python_source_hashes": fixture["source_hashes"],
        "native_requests_verified": len(records),
        "records": records,
        "synthetic_fits_sha256": sequence_digests,
        "all_fields_compared": True,
        "scope": "prior affine WCS fit/report; point-source sequence translation, affine registration, track construction/classification; time-aware motion floor; synthetic file-based WCS flow; no trail-feature, full desktop, or cross-platform acceptance",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
