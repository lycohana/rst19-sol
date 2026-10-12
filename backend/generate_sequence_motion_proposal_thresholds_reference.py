"""Freeze Python metadata-aware trail proposal threshold scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import (
    _exposure_aware_short_trail_min_area,
    _sequence_motion_candidate_min_length,
)


def capture(
    label: str,
    *,
    psf_fwhm: float,
    exposure_s: list[float | None],
    metadata_supported: bool = True,
    min_length_px: float | None = None,
    short_exposure_s: float | None = 1.5,
) -> dict[str, object]:
    request = {
        "psf_fwhm": psf_fwhm,
        "min_length_px": min_length_px,
        "exposure_s": exposure_s,
        "metadata_supported": metadata_supported,
        "short_exposure_s": short_exposure_s,
    }
    expected = {
        "min_length_px": _sequence_motion_candidate_min_length(
            psf_fwhm=psf_fwhm,
            min_length_px=min_length_px,
            exposure_s=exposure_s,
            metadata_supported=metadata_supported,
        ),
        "short_min_area": _exposure_aware_short_trail_min_area(
            psf_fwhm=psf_fwhm,
            exposure_s=short_exposure_s,
        ),
    }
    return {"label": label, "request": request, "expected": expected}


def main() -> None:
    cases = [
        capture("unknown_metadata_keeps_legacy_floors", psf_fwhm=2.0, exposure_s=[1.5] * 3, metadata_supported=False),
        capture("known_regime_reduces_line_floor", psf_fwhm=4.0, exposure_s=[1.5] * 3),
        capture("known_regime_keeps_floor_for_narrow_psf", psf_fwhm=2.0, exposure_s=[1.5] * 3),
        capture("variable_exposure_keeps_legacy_line_floor", psf_fwhm=4.0, exposure_s=[1.4, 1.6, 1.5]),
        capture("missing_exposure_keeps_legacy_line_floor", psf_fwhm=4.0, exposure_s=[1.5, None, 1.5]),
        capture("empty_exposures_keep_legacy_line_floor", psf_fwhm=4.0, exposure_s=[]),
        capture("out_of_regime_exposure_keeps_legacy_line_floor", psf_fwhm=4.0, exposure_s=[1.7] * 3),
        capture("lower_regime_boundary_is_inclusive", psf_fwhm=4.0, exposure_s=[1.35] * 3),
        capture("upper_regime_boundary_is_inclusive", psf_fwhm=4.0, exposure_s=[1.65] * 3),
        capture("spread_boundary_uses_median_tolerance", psf_fwhm=4.0, exposure_s=[1.475, 1.525]),
        capture("explicit_minimum_length_overrides_metadata", psf_fwhm=4.0, exposure_s=[1.5] * 3, min_length_px=3.25),
        capture("explicit_minimum_length_overrides_invalid_psf", psf_fwhm=0.0, exposure_s=[1.5] * 3, min_length_px=5.0),
        capture("nonpositive_psf_keeps_default_line_floor", psf_fwhm=0.0, exposure_s=[1.5] * 3),
        capture("short_area_uses_twelve_pixel_floor", psf_fwhm=2.0, exposure_s=[1.5] * 3, short_exposure_s=1.5),
        capture("short_area_rounds_footprint_up", psf_fwhm=3.0, exposure_s=[1.5] * 3, short_exposure_s=1.5),
        capture("short_area_stays_at_legacy_cap", psf_fwhm=4.0, exposure_s=[1.5] * 3, short_exposure_s=1.5),
        capture("short_area_rejects_lower_boundary_neighbor", psf_fwhm=2.0, exposure_s=[1.5] * 3, short_exposure_s=1.34),
        capture("short_area_accepts_exposure_boundary", psf_fwhm=2.0, exposure_s=[1.5] * 3, short_exposure_s=1.65),
        capture("missing_short_exposure_keeps_legacy_area", psf_fwhm=2.0, exposure_s=[1.5] * 3, short_exposure_s=None),
    ]
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    path = Path(__file__).parent / "tests/fixtures/sequence_motion_proposal_thresholds.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
