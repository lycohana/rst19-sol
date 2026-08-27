"""基于已有切平面 WCS 的一对一星表匹配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .catalog import CatalogSource
from .detection import Detection
from .wcs import TangentPlaneWCS


@dataclass(frozen=True, slots=True)
class CatalogMatch:
    detection_id: int
    source_id: str
    detection_x: float
    detection_y: float
    predicted_x: float
    predicted_y: float
    residual_px: float
    catalog_magnitude: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "source_id": self.source_id,
            "detection_x": self.detection_x,
            "detection_y": self.detection_y,
            "predicted_x": self.predicted_x,
            "predicted_y": self.predicted_y,
            "residual_px": self.residual_px,
            "catalog_magnitude": self.catalog_magnitude,
        }


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[CatalogMatch, ...]
    unmatched_detection_ids: tuple[int, ...]
    unmatched_catalog_ids: tuple[str, ...]
    max_residual_px: float | None
    rms_residual_px: float | None
    inlier_ratio: float
    radius_px: float

    @property
    def matched_count(self) -> int:
        return len(self.matches)

    def as_dict(self) -> dict[str, object]:
        return {
            "matched_count": self.matched_count,
            "unmatched_detection_ids": list(self.unmatched_detection_ids),
            "unmatched_catalog_ids": list(self.unmatched_catalog_ids),
            "max_residual_px": self.max_residual_px,
            "rms_residual_px": self.rms_residual_px,
            "inlier_ratio": self.inlier_ratio,
            "radius_px": self.radius_px,
            "matches": [match.as_dict() for match in self.matches],
        }


def match_detections(
    detections: Sequence[Detection],
    catalog: Sequence[CatalogSource],
    wcs: TangentPlaneWCS,
    *,
    radius_px: float = 3.0,
    epoch: float | None = None,
) -> MatchResult:
    """把图像检测源与 WCS 预测位置进行唯一、半径约束匹配。

    当前版本假设 WCS 初值已经存在；它不会把“最近邻最多”当作盲解算。
    传入 `epoch` 时，先按星表自行传播，再投影到图像坐标。
    """

    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if not detections or not catalog:
        return MatchResult(
            matches=(),
            unmatched_detection_ids=tuple(detection.detection_id for detection in detections),
            unmatched_catalog_ids=tuple(source.source_id for source in catalog),
            max_residual_px=None,
            rms_residual_px=None,
            inlier_ratio=0.0,
            radius_px=radius_px,
        )

    epoch_catalog = tuple(source.at_epoch(epoch) for source in catalog)
    ra = np.array([source.ra_deg for source in epoch_catalog], dtype=np.float64)
    dec = np.array([source.dec_deg for source in epoch_catalog], dtype=np.float64)
    predicted_x, predicted_y = wcs.world_to_pixel(ra, dec)
    predicted = np.column_stack((predicted_x, predicted_y))
    valid_catalog = np.isfinite(predicted).all(axis=1)
    valid_indices = np.flatnonzero(valid_catalog)
    if valid_indices.size == 0:
        return MatchResult(
            matches=(),
            unmatched_detection_ids=tuple(detection.detection_id for detection in detections),
            unmatched_catalog_ids=tuple(source.source_id for source in catalog),
            max_residual_px=None,
            rms_residual_px=None,
            inlier_ratio=0.0,
            radius_px=radius_px,
        )

    tree = cKDTree(predicted[valid_catalog])
    detection_points = np.array([(detection.x, detection.y) for detection in detections], dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for detection_index, point in enumerate(detection_points):
        for local_catalog_index in tree.query_ball_point(point, radius_px):
            catalog_index = int(valid_indices[local_catalog_index])
            residual = float(np.linalg.norm(point - predicted[catalog_index]))
            candidates.append((residual, detection_index, catalog_index))
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1], candidate[2]))

    assigned_detections: set[int] = set()
    assigned_catalog: set[int] = set()
    matches: list[CatalogMatch] = []
    for residual, detection_index, catalog_index in candidates:
        if detection_index in assigned_detections or catalog_index in assigned_catalog:
            continue
        assigned_detections.add(detection_index)
        assigned_catalog.add(catalog_index)
        detection = detections[detection_index]
        source = epoch_catalog[catalog_index]
        matches.append(
            CatalogMatch(
                detection_id=detection.detection_id,
                source_id=source.source_id,
                detection_x=detection.x,
                detection_y=detection.y,
                predicted_x=float(predicted[catalog_index, 0]),
                predicted_y=float(predicted[catalog_index, 1]),
                residual_px=residual,
                catalog_magnitude=source.magnitude,
            )
        )

    residuals = np.array([match.residual_px for match in matches], dtype=np.float64)
    unmatched_detection_ids = tuple(
        detection.detection_id for index, detection in enumerate(detections) if index not in assigned_detections
    )
    unmatched_catalog_ids = tuple(
        source.source_id for index, source in enumerate(epoch_catalog) if index not in assigned_catalog
    )
    return MatchResult(
        matches=tuple(sorted(matches, key=lambda match: match.detection_id)),
        unmatched_detection_ids=unmatched_detection_ids,
        unmatched_catalog_ids=unmatched_catalog_ids,
        max_residual_px=float(residuals.max()) if residuals.size else None,
        rms_residual_px=float(np.sqrt(np.mean(residuals**2))) if residuals.size else None,
        inlier_ratio=len(matches) / len(detections) if detections else 0.0,
        radius_px=radius_px,
    )
