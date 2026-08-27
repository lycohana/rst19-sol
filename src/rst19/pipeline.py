"""把单帧 FITS、星点检测和可选星表匹配串成可复用流程。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .catalog import CatalogSource
from .detection import DetectionResult, detect_sources
from .fits import auxiliary_mask, read_fits
from .matching import MatchResult, match_detections
from .models import FitsFrame
from .wcs import TangentPlaneWCS


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    frame: FitsFrame
    detection: DetectionResult
    matching: MatchResult | None

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
        }


def analyze_frame(
    path: str,
    *,
    catalog: Sequence[CatalogSource] | None = None,
    wcs: TangentPlaneWCS | None = None,
    threshold_sigma: float = 5.0,
    min_distance: int = 3,
    aperture_radius: int = 4,
    max_sources: int | None = None,
    match_radius_px: float = 3.0,
    epoch: float | None = None,
) -> FrameAnalysis:
    """分析单帧图像；提供 catalog 时必须同时提供先验 WCS。"""

    if catalog is not None and wcs is None:
        raise ValueError("catalog matching requires a TangentPlaneWCS")
    frame = read_fits(path)
    detection = detect_sources(
        frame.data,
        mask=auxiliary_mask(frame.data.shape),
        threshold_sigma=threshold_sigma,
        min_distance=min_distance,
        aperture_radius=aperture_radius,
        max_sources=max_sources,
    )
    matching = match_detections(detection.sources, catalog, wcs, radius_px=match_radius_px, epoch=epoch) if catalog is not None else None
    return FrameAnalysis(frame=frame, detection=detection, matching=matching)
