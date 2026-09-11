"""基于已有切平面 WCS 的一对一星表匹配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .catalog import CatalogSource
from .detection import Detection
from .wcs import AffineWCSCalibration, TangentPlaneWCS


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
    catalog_magnitude_error: float | None = None
    catalog_color: float | None = None
    catalog_color_name: str | None = None
    photometric_system: str | None = None
    photometric_band: str | None = None

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
            "catalog_magnitude_error": self.catalog_magnitude_error,
            "catalog_color": self.catalog_color,
            "catalog_color_name": self.catalog_color_name,
            "photometric_system": self.photometric_system,
            "photometric_band": self.photometric_band,
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
    assignment_mode: str = "global"

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
            "assignment_mode": self.assignment_mode,
            "matches": [match.as_dict() for match in self.matches],
        }


def _greedy_assignment(candidates: Sequence[tuple[float, int, int]]) -> tuple[tuple[float, int, int], ...]:
    """按残差排序的历史基线分配。"""

    assigned_detections: set[int] = set()
    assigned_catalog: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for residual, detection_index, catalog_index in sorted(candidates, key=lambda candidate: candidate):
        if detection_index in assigned_detections or catalog_index in assigned_catalog:
            continue
        assigned_detections.add(detection_index)
        assigned_catalog.add(catalog_index)
        matches.append((residual, detection_index, catalog_index))
    return tuple(matches)


def _global_assignment(
    candidates: Sequence[tuple[float, int, int]],
    *,
    detection_count: int,
    catalog_count: int,
    radius_px: float,
) -> tuple[tuple[float, int, int], ...]:
    """在每个稀疏连通分量内求最大基数、最小残差的一对一匹配。

    候选边由半径门控产生。对一个分量构造带虚拟未匹配节点的方阵：
    有效边的代价是残差，未匹配代价高于任意有效边，非法边代价再高一档。
    因而优化顺序是先最大化匹配数量，再最小化总残差。按连通分量求解
    避免把整幅稀疏星场展开成一个巨大稠密矩阵。
    """

    if not candidates:
        return ()

    node_count = detection_count + catalog_count
    parent = list(range(node_count))
    rank = [0] * node_count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for _residual, detection_index, catalog_index in candidates:
        union(detection_index, detection_count + catalog_index)

    component_edges: dict[int, list[tuple[float, int, int]]] = {}
    for candidate in candidates:
        root = find(candidate[1])
        component_edges.setdefault(root, []).append(candidate)

    unmatched_penalty = radius_px + max(1.0, radius_px)
    invalid_penalty = 4.0 * unmatched_penalty
    matches: list[tuple[float, int, int]] = []
    for edges in component_edges.values():
        detection_indices = sorted({candidate[1] for candidate in edges})
        catalog_indices = sorted({candidate[2] for candidate in edges})
        detection_positions = {index: position for position, index in enumerate(detection_indices)}
        catalog_positions = {index: position for position, index in enumerate(catalog_indices)}
        detection_size = len(detection_indices)
        catalog_size = len(catalog_indices)
        size = detection_size + catalog_size
        cost = np.full((size, size), invalid_penalty, dtype=np.float64)
        cost[:detection_size, catalog_size:] = unmatched_penalty
        cost[detection_size:, :catalog_size] = unmatched_penalty
        cost[detection_size:, catalog_size:] = 0.0
        edge_residual: dict[tuple[int, int], float] = {}
        for residual, detection_index, catalog_index in edges:
            row = detection_positions[detection_index]
            column = catalog_positions[catalog_index]
            key = (detection_index, catalog_index)
            edge_residual[key] = min(residual, edge_residual.get(key, np.inf))
            cost[row, column] = edge_residual[key]

        row_indices, column_indices = linear_sum_assignment(cost)
        for row, column in zip(row_indices, column_indices):
            if row >= detection_size or column >= catalog_size:
                continue
            detection_index = detection_indices[row]
            catalog_index = catalog_indices[column]
            residual = edge_residual.get((detection_index, catalog_index))
            if residual is not None:
                matches.append((residual, detection_index, catalog_index))

    return tuple(sorted(matches, key=lambda candidate: (candidate[1], candidate[2], candidate[0])))


def match_detections(
    detections: Sequence[Detection],
    catalog: Sequence[CatalogSource],
    wcs: TangentPlaneWCS | AffineWCSCalibration,
    *,
    radius_px: float = 3.0,
    epoch: float | None = None,
    assignment_mode: str = "global",
) -> MatchResult:
    """把图像检测源与 WCS 预测位置进行唯一、半径约束匹配。

    当前版本假设 WCS 初值已经存在；它不会把“最近邻最多”当作盲解算。
    ``AffineWCSCalibration`` 可用于板解后的二次匹配，此时会保留拟合
    的完整仿射矩阵，而不是退回到只含平均尺度的等效 TAN 近似。
    ``global`` 在每个候选连通分量内先最大化一对一匹配数、再最小化总
    残差；``greedy`` 保留旧的按残差抢占基线，用于敏感性对照。
    传入 `epoch` 时，先按星表自行传播，再投影到图像坐标。
    """

    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if assignment_mode not in {"global", "greedy"}:
        raise ValueError("assignment_mode must be either 'global' or 'greedy'")
    if not detections or not catalog:
        return MatchResult(
            matches=(),
            unmatched_detection_ids=tuple(detection.detection_id for detection in detections),
            unmatched_catalog_ids=tuple(source.source_id for source in catalog),
            max_residual_px=None,
            rms_residual_px=None,
            inlier_ratio=0.0,
            radius_px=radius_px,
            assignment_mode=assignment_mode,
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
            assignment_mode=assignment_mode,
        )

    tree = cKDTree(predicted[valid_catalog])
    detection_points = np.array([(detection.x, detection.y) for detection in detections], dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for detection_index, point in enumerate(detection_points):
        for local_catalog_index in tree.query_ball_point(point, radius_px):
            catalog_index = int(valid_indices[local_catalog_index])
            residual = float(np.linalg.norm(point - predicted[catalog_index]))
            candidates.append((residual, detection_index, catalog_index))
    assigned_edges = (
        _global_assignment(
            candidates,
            detection_count=len(detections),
            catalog_count=len(epoch_catalog),
            radius_px=radius_px,
        )
        if assignment_mode == "global"
        else _greedy_assignment(candidates)
    )
    assigned_detections = {detection_index for _residual, detection_index, _catalog_index in assigned_edges}
    assigned_catalog = {catalog_index for _residual, _detection_index, catalog_index in assigned_edges}
    matches: list[CatalogMatch] = []
    for residual, detection_index, catalog_index in assigned_edges:
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
                catalog_magnitude_error=source.magnitude_error,
                catalog_color=source.color,
                catalog_color_name=source.color_name,
                photometric_system=source.photometric_system,
                photometric_band=source.photometric_band,
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
        assignment_mode=assignment_mode,
    )
