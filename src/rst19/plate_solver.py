"""基于星对几何的可解释视场解算。

这个模块解决的是现有 ``TangentPlaneWCS`` 的一个明确缺口：当辅助数据只
提供光轴中心和像元尺度先验，而图像轴向、旋转角和奇偶性未知时，不能先
用一个随意的旋转去做最近邻匹配。本实现用检测点与目录点的成对距离提出
相似变换候选，再用全局一对一匹配、仿射细化和留一验证验收候选。

它不是盲解算的物理真值证明，也不读取网络数据。调用方必须显式提供已经
下载或准备好的 ``CatalogSource``；结果中保留候选数、内点数、覆盖度和
失败原因，避免把两个巧合近邻包装成 WCS 成功。
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .catalog import CatalogSource
from .detection import Detection
from .matching import CatalogMatch
from .wcs import TangentPlaneWCS


_BAD_SOLVER_FLAGS = frozenset(
    {
        "EDGE",
        "MASKED",
        "SATURATED",
        "LINE_ARTIFACT",
        "CODE_PATTERN",
        "RANGE_ANOMALY",
        "UNRESOLVED_BLEND",
    }
)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _rank_detection(source: Detection) -> float:
    for value in (source.flux_snr, source.filter_snr, source.snr, source.peak):
        if value is not None and _finite(value):
            return float(value)
    return float("-inf")


def _rank_catalog(source: CatalogSource) -> float:
    if source.magnitude is None or not _finite(source.magnitude):
        return float("inf")
    return float(source.magnitude)


@dataclass(frozen=True, slots=True)
class PlateTransform:
    """切平面角秒到图像像素的二维仿射变换。"""

    matrix_px_per_arcsec: tuple[tuple[float, float], tuple[float, float]]
    offset_px: tuple[float, float]
    plate_scale_arcsec_per_pixel: float
    rotation_deg: float
    parity: int
    anisotropy_ratio: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix_px_per_arcsec, dtype=np.float64)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix_px_per_arcsec must be a finite 2x2 matrix")
        offset = np.asarray(self.offset_px, dtype=np.float64)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("offset_px must contain two finite values")
        if not _finite(self.plate_scale_arcsec_per_pixel) or self.plate_scale_arcsec_per_pixel <= 0:
            raise ValueError("plate_scale_arcsec_per_pixel must be positive")
        if self.parity not in (-1, 1):
            raise ValueError("parity must be either 1 or -1")

    def project_tangent(self, tangent_arcsec: np.ndarray) -> np.ndarray:
        values = np.asarray(tangent_arcsec, dtype=np.float64)
        if values.ndim == 1:
            if values.shape != (2,):
                raise ValueError("tangent point must contain two values")
            return values @ np.asarray(self.matrix_px_per_arcsec, dtype=np.float64).T + self.offset_px
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("tangent points must have shape (n, 2)")
        return values @ np.asarray(self.matrix_px_per_arcsec, dtype=np.float64).T + np.asarray(
            self.offset_px, dtype=np.float64
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "matrix_px_per_arcsec": [list(row) for row in self.matrix_px_per_arcsec],
            "offset_px": list(self.offset_px),
            "plate_scale_arcsec_per_pixel": self.plate_scale_arcsec_per_pixel,
            "rotation_deg": self.rotation_deg,
            "parity": self.parity,
            "anisotropy_ratio": self.anisotropy_ratio,
        }


@dataclass(frozen=True, slots=True)
class PlateSolveCandidate:
    """一个候选变换的完整验收证据。"""

    transform: PlateTransform
    matches: tuple[CatalogMatch, ...]
    rms_residual_px: float
    max_residual_px: float
    coverage_x: float
    coverage_y: float
    coverage_area: float
    leave_one_out_rms_residual_px: float | None
    leave_one_out_max_residual_px: float | None
    seed_image_pair: tuple[int, int]
    seed_catalog_pair: tuple[str, str]

    @property
    def matched_count(self) -> int:
        return len(self.matches)

    @property
    def score(self) -> tuple[int, float, float, float, float, int]:
        """按稳健几何支持、留一误差和覆盖度排序候选。

        宽匹配半径会把边缘近邻也纳入 ``matches``。如果只按匹配总数
        排序，几个残差较大的巧合近邻可能压过一个匹配数略少但几何更
        稳定的解。 ``_fit_affine`` 的最小剔除残差同样是 1 px，因此把
        1 px 内的支持数作为第一排序项；总匹配数只在几何质量相近时
        用作最后的次级依据。
        """

        tight_support = sum(float(match.residual_px) <= 1.0 for match in self.matches)
        leave_one_out = self.leave_one_out_rms_residual_px
        loo_score = (
            -float(leave_one_out)
            if leave_one_out is not None and math.isfinite(float(leave_one_out))
            else float("-inf")
        )
        return (
            int(tight_support),
            loo_score,
            -self.rms_residual_px,
            self.coverage_area,
            -self.max_residual_px,
            self.matched_count,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform.as_dict(),
            "matched_count": self.matched_count,
            "rms_residual_px": self.rms_residual_px,
            "max_residual_px": self.max_residual_px,
            "coverage_x": self.coverage_x,
            "coverage_y": self.coverage_y,
            "coverage_area": self.coverage_area,
            "leave_one_out_rms_residual_px": self.leave_one_out_rms_residual_px,
            "leave_one_out_max_residual_px": self.leave_one_out_max_residual_px,
            "seed_image_pair": list(self.seed_image_pair),
            "seed_catalog_pair": list(self.seed_catalog_pair),
            "matches": [match.as_dict() for match in self.matches],
        }


@dataclass(frozen=True, slots=True)
class PlateSolveResult:
    """星对解算结果；``status=VALID`` 才允许下游使用变换。"""

    status: str
    reason: str
    best: PlateSolveCandidate | None
    alternatives: tuple[PlateSolveCandidate, ...]
    image_points_considered: int
    catalog_points_considered: int
    pair_hypotheses: int
    unique_hypotheses: int
    acceptance: dict[str, object]

    @property
    def valid(self) -> bool:
        return self.status == "VALID" and self.best is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "valid": self.valid,
            "best": None if self.best is None else self.best.as_dict(),
            "alternatives": [candidate.as_dict() for candidate in self.alternatives],
            "image_points_considered": self.image_points_considered,
            "catalog_points_considered": self.catalog_points_considered,
            "pair_hypotheses": self.pair_hypotheses,
            "unique_hypotheses": self.unique_hypotheses,
            "acceptance": dict(self.acceptance),
        }


def _catalog_tangent_points(
    catalog: Sequence[CatalogSource],
    reference_wcs: TangentPlaneWCS,
    *,
    epoch: float | None,
) -> np.ndarray:
    values: list[tuple[float, float]] = []
    for source in catalog:
        at_epoch = source.at_epoch(epoch)
        east, north = reference_wcs.world_to_tangent_arcsec(at_epoch.ra_deg, at_epoch.dec_deg)
        values.append((float(east), float(north)))
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return result.reshape((-1, 2))


def _select_detections(
    detections: Sequence[Detection],
    *,
    max_points: int,
    quality_only: bool,
) -> tuple[Detection, ...]:
    selected: list[Detection] = []
    for source in detections:
        if not (_finite(source.x) and _finite(source.y)):
            continue
        if quality_only and not source.quality_passed:
            continue
        if _BAD_SOLVER_FLAGS.intersection(str(flag).upper() for flag in source.flags):
            continue
        selected.append(source)
    selected.sort(key=lambda source: (-_rank_detection(source), int(source.detection_id)))
    return tuple(selected[:max_points])


def _select_catalog(
    catalog: Sequence[CatalogSource],
    *,
    max_points: int,
) -> tuple[CatalogSource, ...]:
    selected = [source for source in catalog if source.magnitude is not None and _finite(source.magnitude)]
    selected.sort(key=lambda source: (_rank_catalog(source), str(source.source_id)))
    return tuple(selected[:max_points])


def _pair_rows(points: np.ndarray, *, min_distance: float) -> list[tuple[float, int, int]]:
    rows: list[tuple[float, int, int]] = []
    for left in range(len(points) - 1):
        delta = points[left + 1 :] - points[left]
        distances = np.hypot(delta[:, 0], delta[:, 1])
        for offset, distance in enumerate(distances, start=left + 1):
            if math.isfinite(float(distance)) and float(distance) >= min_distance:
                rows.append((float(distance), left, offset))
    return rows


def _similarity_matrix(source_vector: np.ndarray, target_vector: np.ndarray, *, reflected: bool) -> np.ndarray:
    sx, sy = (float(value) for value in source_vector)
    tx, ty = (float(value) for value in target_vector)
    norm2 = sx * sx + sy * sy
    if norm2 <= 0:
        raise ValueError("pair vector cannot be zero")
    if not reflected:
        c = (sx * tx + sy * ty) / norm2
        s = (sx * ty - sy * tx) / norm2
        return np.asarray(((c, -s), (s, c)), dtype=np.float64)
    c = (sx * tx - sy * ty) / norm2
    s = (sy * tx + sx * ty) / norm2
    return np.asarray(((c, s), (s, -c)), dtype=np.float64)


def _seed_transform(
    sky_a: np.ndarray,
    sky_b: np.ndarray,
    pixel_a: np.ndarray,
    pixel_b: np.ndarray,
    *,
    reflected: bool,
) -> PlateTransform:
    source_vector = sky_b - sky_a
    target_vector = pixel_b - pixel_a
    matrix = _similarity_matrix(source_vector, target_vector, reflected=reflected)
    offset = pixel_a - matrix @ sky_a
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    scale_px_per_arcsec = float(np.mean(singular_values))
    if scale_px_per_arcsec <= 0 or not np.all(np.isfinite(singular_values)):
        raise ValueError("seed transform has invalid scale")
    determinant = float(np.linalg.det(matrix))
    parity = 1 if determinant >= 0 else -1
    rotation_deg = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    anisotropy = float(singular_values[0] / singular_values[-1])
    return PlateTransform(
        matrix_px_per_arcsec=(
            (float(matrix[0, 0]), float(matrix[0, 1])),
            (float(matrix[1, 0]), float(matrix[1, 1])),
        ),
        offset_px=(float(offset[0]), float(offset[1])),
        plate_scale_arcsec_per_pixel=1.0 / scale_px_per_arcsec,
        rotation_deg=rotation_deg,
        parity=parity,
        anisotropy_ratio=anisotropy,
    )


def _matches_for_transform(
    transform: PlateTransform,
    image_points: Sequence[Detection],
    catalog: Sequence[CatalogSource],
    tangent_points: np.ndarray,
    *,
    radius_px: float,
    epoch: float | None,
) -> tuple[CatalogMatch, ...]:
    if not image_points or not catalog:
        return ()
    image_xy = np.asarray([(source.x, source.y) for source in image_points], dtype=np.float64)
    tree = cKDTree(image_xy)
    predicted = transform.project_tangent(tangent_points)
    distances, indices = tree.query(predicted, distance_upper_bound=float(radius_px))
    proposals: list[tuple[float, int, int]] = []
    for catalog_index, (distance, image_index) in enumerate(zip(distances, indices, strict=True)):
        if int(image_index) >= len(image_points) or not math.isfinite(float(distance)):
            continue
        proposals.append((float(distance), int(image_index), catalog_index))
    proposals.sort(key=lambda row: (row[0], image_points[row[1]].detection_id, str(catalog[row[2]].source_id)))
    used_image: set[int] = set()
    used_catalog: set[int] = set()
    matches: list[CatalogMatch] = []
    for residual, image_index, catalog_index in proposals:
        if image_index in used_image or catalog_index in used_catalog:
            continue
        detection = image_points[image_index]
        source = catalog[catalog_index].at_epoch(epoch)
        predicted_x, predicted_y = predicted[catalog_index]
        matches.append(
            CatalogMatch(
                detection_id=int(detection.detection_id),
                source_id=str(source.source_id),
                detection_x=float(detection.x),
                detection_y=float(detection.y),
                predicted_x=float(predicted_x),
                predicted_y=float(predicted_y),
                residual_px=float(residual),
                catalog_magnitude=(None if source.magnitude is None else float(source.magnitude)),
                catalog_magnitude_error=source.magnitude_error,
                catalog_color=source.color,
                catalog_color_name=source.color_name,
                photometric_system=source.photometric_system,
                photometric_band=source.photometric_band,
            )
        )
        used_image.add(image_index)
        used_catalog.add(catalog_index)
    return tuple(matches)


def _fit_affine(
    matches: Sequence[CatalogMatch],
    catalog_by_id: dict[str, CatalogSource],
    reference_wcs: TangentPlaneWCS,
    *,
    epoch: float | None,
    clip_sigma: float,
    max_iterations: int,
) -> tuple[PlateTransform, tuple[CatalogMatch, ...]] | None:
    if len(matches) < 3:
        return None
    sky_rows: list[tuple[float, float]] = []
    pixel_rows: list[tuple[float, float]] = []
    valid_matches: list[CatalogMatch] = []
    for match in matches:
        source = catalog_by_id.get(str(match.source_id))
        if source is None:
            continue
        source = source.at_epoch(epoch)
        east, north = reference_wcs.world_to_tangent_arcsec(source.ra_deg, source.dec_deg)
        values = np.asarray((east, north, match.detection_x, match.detection_y), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            continue
        sky_rows.append((float(east), float(north)))
        pixel_rows.append((float(match.detection_x), float(match.detection_y)))
        valid_matches.append(match)
    if len(valid_matches) < 3:
        return None
    sky = np.asarray(sky_rows, dtype=np.float64)
    pixels = np.asarray(pixel_rows, dtype=np.float64)
    design = np.column_stack((sky, np.ones(len(sky), dtype=np.float64)))
    active = np.ones(len(sky), dtype=bool)
    coefficients: np.ndarray | None = None
    for _ in range(max(1, max_iterations)):
        if int(active.sum()) < 3:
            return None
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(design[active], pixels[active], rcond=None)
        if rank < 3:
            return None
        predicted = design @ coefficients
        residuals = np.linalg.norm(predicted - pixels, axis=1)
        active_residuals = residuals[active]
        median = float(np.median(active_residuals))
        mad = float(np.median(np.abs(active_residuals - median)))
        cutoff = max(1.0, median + float(clip_sigma) * max(1e-6, 1.4826 * mad))
        new_active = residuals <= cutoff
        if int(new_active.sum()) < 3 or np.array_equal(new_active, active):
            active = new_active if int(new_active.sum()) >= 3 else active
            break
        active = new_active
    if coefficients is None:
        return None
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design[active], pixels[active], rcond=None)
    if rank < 3:
        return None
    matrix = coefficients[:2, :].T
    offset = coefficients[2, :]
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] <= 0 or not np.all(np.isfinite(singular_values)):
        return None
    determinant = float(np.linalg.det(matrix))
    parity = 1 if determinant >= 0 else -1
    rotation_deg = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    transform = PlateTransform(
        matrix_px_per_arcsec=(
            (float(matrix[0, 0]), float(matrix[0, 1])),
            (float(matrix[1, 0]), float(matrix[1, 1])),
        ),
        offset_px=(float(offset[0]), float(offset[1])),
        plate_scale_arcsec_per_pixel=float(1.0 / np.mean(singular_values)),
        rotation_deg=rotation_deg,
        parity=parity,
        anisotropy_ratio=float(singular_values[0] / singular_values[-1]),
    )
    retained = tuple(valid_matches[index] for index in np.flatnonzero(active))
    return transform, retained


def _leave_one_out(
    matches: Sequence[CatalogMatch],
    catalog_by_id: dict[str, CatalogSource],
    reference_wcs: TangentPlaneWCS,
    *,
    epoch: float | None,
) -> tuple[float | None, float | None]:
    residuals: list[float] = []
    for index in range(len(matches)):
        subset = tuple(match for position, match in enumerate(matches) if position != index)
        fitted = _fit_affine(
            subset,
            catalog_by_id,
            reference_wcs,
            epoch=epoch,
            clip_sigma=3.5,
            max_iterations=4,
        )
        if fitted is None:
            continue
        transform, _ = fitted
        source = catalog_by_id.get(str(matches[index].source_id))
        if source is None:
            continue
        source = source.at_epoch(epoch)
        east, north = reference_wcs.world_to_tangent_arcsec(source.ra_deg, source.dec_deg)
        predicted = transform.project_tangent(np.asarray((east, north), dtype=np.float64))
        residuals.append(float(np.hypot(predicted[0] - matches[index].detection_x, predicted[1] - matches[index].detection_y)))
    if not residuals:
        return None, None
    return float(np.sqrt(np.mean(np.square(residuals)))), float(np.max(residuals))


def _coverage(
    matches: Sequence[CatalogMatch],
    *,
    image_shape: tuple[int, int] | None,
) -> tuple[float, float, float]:
    if not matches:
        return 0.0, 0.0, 0.0
    xs = np.asarray([match.detection_x for match in matches], dtype=np.float64)
    ys = np.asarray([match.detection_y for match in matches], dtype=np.float64)
    if image_shape is None:
        width = max(float(np.ptp(xs)), 1.0)
        height = max(float(np.ptp(ys)), 1.0)
        return 1.0, 1.0, 1.0
    image_height, image_width = image_shape
    if image_width <= 1 or image_height <= 1:
        return 0.0, 0.0, 0.0
    coverage_x = float(np.clip(np.ptp(xs) / (image_width - 1), 0.0, 1.0))
    coverage_y = float(np.clip(np.ptp(ys) / (image_height - 1), 0.0, 1.0))
    return coverage_x, coverage_y, coverage_x * coverage_y


def _candidate_key(transform: PlateTransform) -> tuple[float, ...]:
    matrix = np.asarray(transform.matrix_px_per_arcsec, dtype=np.float64)
    return tuple(float(value) for value in np.round(matrix.reshape(-1), 8)) + tuple(
        float(value) for value in np.round(np.asarray(transform.offset_px), 2)
    )


def solve_plate(
    detections: Sequence[Detection],
    catalog: Sequence[CatalogSource],
    reference_wcs: TangentPlaneWCS,
    *,
    epoch: float | None = None,
    image_shape: tuple[int, int] | None = None,
    scale_tolerance: float = 0.03,
    min_pair_distance_px: float = 12.0,
    match_radius_px: float = 3.0,
    min_matches: int = 8,
    min_coverage_area: float = 0.02,
    max_rms_residual_px: float = 2.0,
    max_leave_one_out_rms_px: float = 3.0,
    max_image_points: int = 80,
    max_catalog_points: int = 160,
    max_pair_hypotheses: int = 20_000,
    quality_only: bool = True,
    max_iterations: int = 5,
) -> PlateSolveResult:
    """用星对距离提出并验收一个视场变换。

    ``reference_wcs`` 只用于把目录 RA/Dec 投影到光轴切平面；其旋转、
    parity、平移和局部尺度会被星对候选重新估计。默认 ``scale_tolerance``
    较窄，是为了防止密集星场中任意两对距离都能形成大量巧合候选；若设备
    焦距先验不确定，应先显式扩大范围并在结果中保留该敏感性分析。
    """

    if not (0.0 < scale_tolerance < 1.0):
        raise ValueError("scale_tolerance must be in (0, 1)")
    if min_pair_distance_px <= 0 or match_radius_px <= 0 or min_matches < 3:
        raise ValueError("pair distance, match radius and min_matches must be positive")
    if max_image_points < 2 or max_catalog_points < 2 or max_pair_hypotheses < 1:
        raise ValueError("point and hypothesis limits are too small")
    image_sources = _select_detections(detections, max_points=max_image_points, quality_only=quality_only)
    catalog_sources = _select_catalog(catalog, max_points=max_catalog_points)
    acceptance = {
        "min_matches": int(min_matches),
        "min_coverage_area": float(min_coverage_area),
        "max_rms_residual_px": float(max_rms_residual_px),
        "max_leave_one_out_rms_px": float(max_leave_one_out_rms_px),
        "scale_tolerance": float(scale_tolerance),
        "match_radius_px": float(match_radius_px),
    }
    if len(image_sources) < 2 or len(catalog_sources) < 2:
        return PlateSolveResult(
            status="INSUFFICIENT_POINTS",
            reason="筛选后检测点或星表点少于两个，无法提出星对",
            best=None,
            alternatives=(),
            image_points_considered=len(image_sources),
            catalog_points_considered=len(catalog_sources),
            pair_hypotheses=0,
            unique_hypotheses=0,
            acceptance=acceptance,
        )
    tangent = _catalog_tangent_points(catalog_sources, reference_wcs, epoch=epoch)
    pixel_points = np.asarray([(source.x, source.y) for source in image_sources], dtype=np.float64)
    image_pairs = _pair_rows(pixel_points, min_distance=float(min_pair_distance_px))
    catalog_pairs = _pair_rows(tangent, min_distance=0.0)
    catalog_pairs.sort(key=lambda row: row[0])
    catalog_distances = [row[0] for row in catalog_pairs]
    scale_min = float(reference_wcs.pixel_scale_arcsec) * (1.0 - float(scale_tolerance))
    scale_max = float(reference_wcs.pixel_scale_arcsec) * (1.0 + float(scale_tolerance))
    hypotheses = 0
    unique_keys: set[tuple[float, ...]] = set()
    candidates: list[PlateSolveCandidate] = []
    catalog_by_id = {str(source.source_id): source for source in catalog_sources}
    for image_distance, image_left, image_right in image_pairs:
        expected_sky_distance_min = image_distance * scale_min
        expected_sky_distance_max = image_distance * scale_max
        left = bisect.bisect_left(catalog_distances, expected_sky_distance_min)
        right = bisect.bisect_right(catalog_distances, expected_sky_distance_max)
        for catalog_distance, catalog_left, catalog_right in catalog_pairs[left:right]:
            for reverse in (False, True):
                if hypotheses >= max_pair_hypotheses:
                    break
                hypotheses += 1
                if reverse:
                    catalog_a, catalog_b = catalog_right, catalog_left
                else:
                    catalog_a, catalog_b = catalog_left, catalog_right
                try:
                    transform = _seed_transform(
                        tangent[catalog_a],
                        tangent[catalog_b],
                        pixel_points[image_left],
                        pixel_points[image_right],
                        reflected=False,
                    )
                    reflected_transform = _seed_transform(
                        tangent[catalog_a],
                        tangent[catalog_b],
                        pixel_points[image_left],
                        pixel_points[image_right],
                        reflected=True,
                    )
                except ValueError:
                    continue
                for seeded in (transform, reflected_transform):
                    key = _candidate_key(seeded)
                    if key in unique_keys:
                        continue
                    unique_keys.add(key)
                    matches = _matches_for_transform(
                        seeded,
                        image_sources,
                        catalog_sources,
                        tangent,
                        radius_px=match_radius_px,
                        epoch=epoch,
                    )
                    if len(matches) < 3:
                        continue
                    fitted = _fit_affine(
                        matches,
                        catalog_by_id,
                        reference_wcs,
                        epoch=epoch,
                        clip_sigma=3.5,
                        max_iterations=max_iterations,
                    )
                    if fitted is None:
                        continue
                    refined, refined_matches = fitted
                    refined_matches = _matches_for_transform(
                        refined,
                        image_sources,
                        catalog_sources,
                        tangent,
                        radius_px=match_radius_px,
                        epoch=epoch,
                    )
                    if len(refined_matches) >= len(matches):
                        matches = refined_matches
                    if len(matches) < 3:
                        continue
                    residuals = np.asarray([match.residual_px for match in matches], dtype=np.float64)
                    rms = float(np.sqrt(np.mean(np.square(residuals))))
                    maximum = float(np.max(residuals))
                    coverage_x, coverage_y, coverage_area = _coverage(matches, image_shape=image_shape)
                    loo_rms, loo_max = _leave_one_out(
                        matches,
                        catalog_by_id,
                        reference_wcs,
                        epoch=epoch,
                    )
                    candidates.append(
                        PlateSolveCandidate(
                            transform=refined,
                            matches=tuple(matches),
                            rms_residual_px=rms,
                            max_residual_px=maximum,
                            coverage_x=coverage_x,
                            coverage_y=coverage_y,
                            coverage_area=coverage_area,
                            leave_one_out_rms_residual_px=loo_rms,
                            leave_one_out_max_residual_px=loo_max,
                            seed_image_pair=(
                                int(image_sources[image_left].detection_id),
                                int(image_sources[image_right].detection_id),
                            ),
                            seed_catalog_pair=(
                                str(catalog_sources[catalog_a].source_id),
                                str(catalog_sources[catalog_b].source_id),
                            ),
                        )
                    )
            if hypotheses >= max_pair_hypotheses:
                break
        if hypotheses >= max_pair_hypotheses:
            break
    if not candidates:
        return PlateSolveResult(
            status="NO_SOLUTION",
            reason="没有候选变换形成至少三个唯一匹配点",
            best=None,
            alternatives=(),
            image_points_considered=len(image_sources),
            catalog_points_considered=len(catalog_sources),
            pair_hypotheses=hypotheses,
            unique_hypotheses=len(unique_keys),
            acceptance=acceptance,
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    best = candidates[0]
    # A candidate without a usable leave-one-out sample is not validated.
    # Treating ``None`` as success allowed a minimally constrained affine
    # fit (for example, only three non-collinear points) to enter the WCS /
    # photometric calibration path without an independent prediction check.
    loo_rms = best.leave_one_out_rms_residual_px
    loo_ok = (
        loo_rms is not None
        and math.isfinite(float(loo_rms))
        and float(loo_rms) <= max_leave_one_out_rms_px
    )
    accepted = (
        best.matched_count >= min_matches
        and best.rms_residual_px <= max_rms_residual_px
        and best.coverage_area >= min_coverage_area
        and loo_ok
    )
    if accepted:
        status = "VALID"
        reason = "通过唯一匹配、残差、覆盖度和留一验证"
    else:
        failed: list[str] = []
        if best.matched_count < min_matches:
            failed.append(f"匹配数 {best.matched_count} < {min_matches}")
        if best.rms_residual_px > max_rms_residual_px:
            failed.append(f"RMS {best.rms_residual_px:.3f}px > {max_rms_residual_px:.3f}px")
        if best.coverage_area < min_coverage_area:
            failed.append(f"覆盖度 {best.coverage_area:.4f} < {min_coverage_area:.4f}")
        if loo_rms is None:
            failed.append("留一验证无有效样本")
        elif not math.isfinite(float(loo_rms)):
            failed.append("留一残差无效")
        elif float(loo_rms) > max_leave_one_out_rms_px:
            failed.append("留一残差超过门槛")
        status = "REJECTED"
        reason = "；".join(failed) or "未通过严格验收"
    return PlateSolveResult(
        status=status,
        reason=reason,
        best=best,
        alternatives=tuple(candidates[1:6]),
        image_points_considered=len(image_sources),
        catalog_points_considered=len(catalog_sources),
        pair_hypotheses=hypotheses,
        unique_hypotheses=len(unique_keys),
        acceptance=acceptance,
    )


__all__ = [
    "PlateSolveCandidate",
    "PlateSolveResult",
    "PlateTransform",
    "solve_plate",
]
