"""把单帧 FITS、星点检测和可选星表匹配串成可复用流程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
    background_sample_limit: int = 100_000,
    min_flux_snr: float = 5.0,
    min_fwhm: float = 0.8,
    max_fwhm: float = 12.0,
    max_ellipticity: float = 0.65,
    min_sharpness: float = 0.005,
    max_sharpness: float = 0.85,
    min_footprint_pixels: int = 2,
    min_psf_support_pixels: int = 3,
    gain_e_per_adu: float | None = None,
    read_noise_adu: float = 0.0,
    mask_zero_pixels: bool | None = None,
    allow_partial_zero_mask: bool | None = None,
    reject_linear_artifacts: bool = True,
    proposal_mode: str = "gaussian",
    dog_threshold_sigma: float | None = None,
    dog_min_peak_sigma: float = 2.0,
    dog_blend_radius_factor: float = 2.5,
    starlet_threshold_sigma: float | None = None,
    starlet_min_peak_sigma: float = 2.5,
    deblend_delta_bic_min: float = 10.0,
    deblend_component_snr_min: float = 5.0,
    deblend_primary_snr_min: float = 12.0,
    deblend_min_residual_sigma: float = 4.0,
    deblend_search_radius_factor: float = 2.0,
    enable_local_deblend: bool = False,
    refine_local_background: bool = True,
    use_float32: bool = False,
    fast_sequence: bool = False,
    background_model: tuple[object, object] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> FrameAnalysis:
    """分析单帧图像；提供 catalog 时必须同时提供先验 WCS。"""

    if catalog is not None and wcs is None:
        raise ValueError("catalog matching requires a TangentPlaneWCS")
    if progress is not None:
        progress(3.0, "读取 FITS")
    frame = path if isinstance(path, FitsFrame) else read_fits(path)

    def detection_progress(value: float, label: str) -> None:
        if progress is not None:
            progress(8.0 + float(value) * 0.86, label)

    detection = detect_sources(
        frame.data,
        mask=auxiliary_mask(frame.data.shape),
        threshold_sigma=threshold_sigma,
        min_distance=min_distance,
        aperture_radius=aperture_radius,
        max_sources=max_sources,
        psf_fwhm=psf_fwhm,
        background_box_size=background_box_size,
        background_sample_limit=background_sample_limit,
        min_flux_snr=min_flux_snr,
        min_fwhm=min_fwhm,
        max_fwhm=max_fwhm,
        max_ellipticity=max_ellipticity,
        min_sharpness=min_sharpness,
        max_sharpness=max_sharpness,
        min_footprint_pixels=min_footprint_pixels,
        min_psf_support_pixels=min_psf_support_pixels,
        gain_e_per_adu=gain_e_per_adu,
        read_noise_adu=read_noise_adu,
        mask_zero_pixels=mask_zero_pixels,
        allow_partial_zero_mask=allow_partial_zero_mask,
        reject_linear_artifacts=reject_linear_artifacts,
        proposal_mode=proposal_mode,
        dog_threshold_sigma=dog_threshold_sigma,
        dog_min_peak_sigma=dog_min_peak_sigma,
        dog_blend_radius_factor=dog_blend_radius_factor,
        starlet_threshold_sigma=starlet_threshold_sigma,
        starlet_min_peak_sigma=starlet_min_peak_sigma,
        deblend_delta_bic_min=deblend_delta_bic_min,
        deblend_component_snr_min=deblend_component_snr_min,
        deblend_primary_snr_min=deblend_primary_snr_min,
        deblend_min_residual_sigma=deblend_min_residual_sigma,
        deblend_search_radius_factor=deblend_search_radius_factor,
        enable_local_deblend=enable_local_deblend,
        refine_local_background=refine_local_background,
        use_float32=use_float32,
        fast_sequence=fast_sequence,
        background_model=background_model,  # type: ignore[arg-type]
        progress=detection_progress,
    )
    if progress is not None:
        progress(91.0, "完成星点质量筛选")
    matching = match_detections(detection.quality_sources, catalog, wcs, radius_px=match_radius_px, epoch=epoch) if catalog is not None else None
    exposure_ms = frame.header.get("EXPOSURE")
    exposure_s = float(exposure_ms) / 1000.0 if isinstance(exposure_ms, (int, float)) and float(exposure_ms) > 0 else 1.0
    faintest = find_faintest_source(
        detection.quality_sources,
        min_snr=min_flux_snr,
        zero_point=zero_point,
        exposure_s=exposure_s,
    )
    if progress is not None:
        progress(100.0, "单帧检测完成")
    return FrameAnalysis(frame=frame, detection=detection, matching=matching, faintest=faintest)
