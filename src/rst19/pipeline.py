"""把单帧 FITS、星点检测和可选星表匹配串成可复用流程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .catalog import CatalogSource
from .detection import DetectionResult, detect_sources
from .fits import auxiliary_mask, read_fits
from .matching import MatchResult, match_detections
from .models import FitsFrame
from .photometry import FaintestSource, find_faintest_source
from .wcs import TangentPlaneWCS


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    frame: FitsFrame
    detection: DetectionResult
    matching: MatchResult | None
    faintest: FaintestSource | None

    def as_dict(self) -> dict[str, object]:
        header_keys = (
            "DATE-OBS",
            "EXPOSURE",
            "BITPIX",
            "NAXIS1",
            "NAXIS2",
            "AZIMUTH",
            "ELEVATIO",
        )
        return {
            "path": str(self.frame.path),
            "header": {key: self.frame.header[key] for key in header_keys if key in self.frame.header},
            "auxiliary": self.frame.auxiliary.as_dict() if self.frame.auxiliary else None,
            "detection": self.detection.as_dict(),
            "matching": self.matching.as_dict() if self.matching else None,
            "faintest_detected": self.faintest.as_dict() if self.faintest else None,
        }


def analyze_frame(
    path: str | Path | FitsFrame,
    *,
    catalog: Sequence[CatalogSource] | None = None,
    wcs: TangentPlaneWCS | None = None,
    threshold_sigma: float = 4.0,
    min_distance: int = 3,
    aperture_radius: int = 4,
    max_sources: int | None = None,
    match_radius_px: float = 3.0,
    epoch: float | None = None,
    zero_point: float | None = None,
    psf_fwhm: float = 3.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_fwhm: float = 0.8,
    max_fwhm: float = 12.0,
    max_ellipticity: float = 0.65,
    min_sharpness: float = 0.005,
    max_sharpness: float = 0.85,
    min_footprint_pixels: int = 2,
    gain_e_per_adu: float | None = None,
    read_noise_adu: float = 0.0,
    mask_zero_pixels: bool | None = None,
) -> FrameAnalysis:
    """分析单帧图像；提供 catalog 时必须同时提供先验 WCS。"""

    if catalog is not None and wcs is None:
        raise ValueError("catalog matching requires a TangentPlaneWCS")
    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    detection = detect_sources(
        frame.data,
        mask=auxiliary_mask(frame.data.shape),
        threshold_sigma=threshold_sigma,
        min_distance=min_distance,
        aperture_radius=aperture_radius,
        max_sources=max_sources,
        psf_fwhm=psf_fwhm,
        background_box_size=background_box_size,
        min_flux_snr=min_flux_snr,
        min_fwhm=min_fwhm,
        max_fwhm=max_fwhm,
        max_ellipticity=max_ellipticity,
        min_sharpness=min_sharpness,
        max_sharpness=max_sharpness,
        min_footprint_pixels=min_footprint_pixels,
        gain_e_per_adu=gain_e_per_adu,
        read_noise_adu=read_noise_adu,
        mask_zero_pixels=mask_zero_pixels,
    )
    matching = match_detections(detection.quality_sources, catalog, wcs, radius_px=match_radius_px, epoch=epoch) if catalog is not None else None
    exposure_ms = frame.header.get("EXPOSURE")
    exposure_s = float(exposure_ms) / 1000.0 if isinstance(exposure_ms, (int, float)) and float(exposure_ms) > 0 else 1.0
    faintest = find_faintest_source(
        detection.quality_sources,
        min_snr=min_flux_snr,
        zero_point=zero_point,
        exposure_s=exposure_s,
    )
    return FrameAnalysis(frame=frame, detection=detection, matching=matching, faintest=faintest)
