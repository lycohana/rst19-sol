"""不依赖 Astropy 的切平面线性 WCS 基线。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .catalog import CatalogSource

if TYPE_CHECKING:
    from .matching import CatalogMatch


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def _wrap_delta_ra(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


class TangentPlaneWCS:
    """以光轴为切点的 TAN 近似模型。

    坐标约定明确写在接口里：图像 x 向右、y 向下；`crpix` 使用同一套
    0-based 像素坐标；`rotation_deg=0` 时天球东向对应图像 x；`parity`
    可设为 -1 处理 RA 轴反向。正式 WCS 仍需用真实匹配星验证。
    """

    def __init__(
        self,
        *,
        center_ra_deg: float,
        center_dec_deg: float,
        pixel_scale_arcsec: float,
        crpix_x: float,
        crpix_y: float,
        rotation_deg: float = 0.0,
        parity: int = 1,
    ) -> None:
        if pixel_scale_arcsec <= 0:
            raise ValueError("pixel_scale_arcsec must be positive")
        if parity not in (-1, 1):
            raise ValueError("parity must be either 1 or -1")
        if not -90.0 < center_dec_deg < 90.0:
            raise ValueError("center_dec_deg must be strictly between -90 and 90")
        self.center_ra_deg = float(center_ra_deg) % 360.0
        self.center_dec_deg = float(center_dec_deg)
        self.pixel_scale_arcsec = float(pixel_scale_arcsec)
        self.crpix_x = float(crpix_x)
        self.crpix_y = float(crpix_y)
        self.rotation_deg = float(rotation_deg)
        self.parity = int(parity)

    @property
    def scale_rad_per_pixel(self) -> float:
        return math.radians(self.pixel_scale_arcsec / 3600.0)

    def world_to_tangent_arcsec(
        self, ra_deg: float | np.ndarray, dec_deg: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """将天球坐标投影到光轴切平面的东、北坐标，单位为角秒。

        这里的东/北坐标不包含图像旋转、奇偶性和像元尺度，专门用于把
        星表匹配点拟合到图像像素平面。与 ``world_to_pixel`` 共用同一
        TAN 投影公式，避免校准阶段重复实现一套容易漂移的坐标变换。
        """

        ra = np.asarray(ra_deg, dtype=np.float64)
        dec = np.asarray(dec_deg, dtype=np.float64)
        ra0 = math.radians(self.center_ra_deg)
        dec0 = math.radians(self.center_dec_deg)
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        delta_ra = _wrap_delta_ra(ra_rad - ra0)
        sin_dec = np.sin(dec_rad)
        cos_dec = np.cos(dec_rad)
        sin_dec0 = math.sin(dec0)
        cos_dec0 = math.cos(dec0)
        denominator = sin_dec0 * sin_dec + cos_dec0 * cos_dec * np.cos(delta_ra)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = cos_dec * np.sin(delta_ra) / denominator
            eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * np.cos(delta_ra)) / denominator
        invalid = denominator <= 0
        xi = np.where(invalid, np.nan, np.degrees(xi) * 3600.0)
        eta = np.where(invalid, np.nan, np.degrees(eta) * 3600.0)
        return _scalar_or_array(xi), _scalar_or_array(eta)

    def world_to_pixel(
        self, ra_deg: float | np.ndarray, dec_deg: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        east_arcsec, north_arcsec = self.world_to_tangent_arcsec(ra_deg, dec_deg)
        east = np.asarray(east_arcsec, dtype=np.float64) / self.pixel_scale_arcsec
        north = np.asarray(north_arcsec, dtype=np.float64) / self.pixel_scale_arcsec
        theta = math.radians(self.rotation_deg)
        x_rot = east * math.cos(theta) - north * math.sin(theta)
        y_rot = east * math.sin(theta) + north * math.cos(theta)
        x = self.crpix_x + self.parity * x_rot
        y = self.crpix_y + y_rot
        return _scalar_or_array(x), _scalar_or_array(y)

    def pixel_to_world(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        theta = math.radians(self.rotation_deg)
        u = self.parity * (x_arr - self.crpix_x)
        v = y_arr - self.crpix_y
        east = u * math.cos(theta) + v * math.sin(theta)
        north = -u * math.sin(theta) + v * math.cos(theta)
        xi = east * self.scale_rad_per_pixel
        eta = north * self.scale_rad_per_pixel
        rho = np.hypot(xi, eta)
        c = np.arctan(rho)
        dec0 = math.radians(self.center_dec_deg)
        ra0 = math.radians(self.center_ra_deg)
        with np.errstate(divide="ignore", invalid="ignore"):
            dec = np.arcsin(np.cos(c) * math.sin(dec0) + eta * np.sin(c) * math.cos(dec0) / rho)
            ra = ra0 + np.arctan2(
                xi * np.sin(c),
                rho * math.cos(dec0) * np.cos(c) - eta * math.sin(dec0) * np.sin(c),
            )
        zero = rho == 0
        dec = np.where(zero, dec0, dec)
        ra = np.where(zero, ra0, ra)
        return _scalar_or_array(np.degrees(ra) % 360.0), _scalar_or_array(np.degrees(dec))

    def as_dict(self) -> dict[str, float | int]:
        return {
            "center_ra_deg": self.center_ra_deg,
            "center_dec_deg": self.center_dec_deg,
            "pixel_scale_arcsec": self.pixel_scale_arcsec,
            "crpix_x": self.crpix_x,
            "crpix_y": self.crpix_y,
            "rotation_deg": self.rotation_deg,
            "parity": self.parity,
        }


@dataclass(frozen=True, slots=True)
class AffineWCSCalibration:
    """由已匹配星表点估计的局部天空平面到像素仿射模型。

    ``matrix_px_per_arcsec`` 的输入是 ``(east, north)`` 角秒，输出是
    ``(x, y)`` 像素。它比只报告一个像元尺度更诚实：若图像存在轻微
    非正交、缩放不一致或残余剪切，矩阵和各向异性指标仍会保留这些
    证据，而不是强行压成一个看似精确的标量 WCS。
    """

    center_ra_deg: float
    center_dec_deg: float
    matrix_px_per_arcsec: tuple[tuple[float, float], tuple[float, float]]
    offset_px: tuple[float, float]
    plate_scale_arcsec_per_pixel: float
    rotation_deg: float
    parity: int
    anisotropy_ratio: float
    matched_count: int
    inlier_count: int
    rms_residual_px: float
    max_residual_px: float
    all_rms_residual_px: float
    all_max_residual_px: float
    condition_number: float
    inlier_source_ids: tuple[str, ...]
    validation_count: int = 0
    leave_one_out_rms_residual_px: float | None = None
    leave_one_out_max_residual_px: float | None = None

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / self.matched_count if self.matched_count else 0.0

    def refined_wcs(self) -> TangentPlaneWCS:
        """返回不含剪切项的等效 TAN WCS，供可视化和复投影使用。

        完整仿射矩阵仍保存在本对象中；该方法只提供一个兼容现有
        ``TangentPlaneWCS`` 接口的保守近似，不会丢失原始拟合证据。
        """

        return TangentPlaneWCS(
            center_ra_deg=self.center_ra_deg,
            center_dec_deg=self.center_dec_deg,
            pixel_scale_arcsec=self.plate_scale_arcsec_per_pixel,
            crpix_x=self.offset_px[0],
            crpix_y=self.offset_px[1],
            rotation_deg=self.rotation_deg,
            parity=self.parity,
        )

    def pixel_velocity_to_tangent_arcsec(self, dx_px: float, dy_px: float) -> tuple[float, float]:
        """把图像平面速度分量换算为东/北角秒每秒。"""

        matrix = np.asarray(self.matrix_px_per_arcsec, dtype=np.float64)
        if abs(float(np.linalg.det(matrix))) < 1e-15:
            raise ValueError("calibration matrix is singular")
        east, north = np.linalg.solve(matrix, np.array([float(dx_px), float(dy_px)], dtype=np.float64))
        return float(east), float(north)

    def angular_speed_arcsec_per_s(self, dx_px_per_s: float, dy_px_per_s: float) -> float:
        """返回角速度模长；输入必须是图像坐标的像素每秒。"""

        east, north = self.pixel_velocity_to_tangent_arcsec(dx_px_per_s, dy_px_per_s)
        return float(math.hypot(east, north))

    def as_dict(self) -> dict[str, object]:
        return {
            "center_ra_deg": self.center_ra_deg,
            "center_dec_deg": self.center_dec_deg,
            "matrix_px_per_arcsec": [list(row) for row in self.matrix_px_per_arcsec],
            "offset_px": list(self.offset_px),
            "plate_scale_arcsec_per_pixel": self.plate_scale_arcsec_per_pixel,
            "rotation_deg": self.rotation_deg,
            "parity": self.parity,
            "anisotropy_ratio": self.anisotropy_ratio,
            "matched_count": self.matched_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "rms_residual_px": self.rms_residual_px,
            "max_residual_px": self.max_residual_px,
            "all_rms_residual_px": self.all_rms_residual_px,
            "all_max_residual_px": self.all_max_residual_px,
            "condition_number": self.condition_number,
            "inlier_source_ids": list(self.inlier_source_ids),
            "validation_method": "leave_one_out",
            "validation_count": self.validation_count,
            "leave_one_out_rms_residual_px": self.leave_one_out_rms_residual_px,
            "leave_one_out_max_residual_px": self.leave_one_out_max_residual_px,
        }


def fit_affine_wcs_from_matches(
    matches: Sequence["CatalogMatch"],
    catalog: Sequence[CatalogSource],
    reference_wcs: TangentPlaneWCS,
    *,
    epoch: float | None = None,
    min_matches: int = 6,
    clip_sigma: float = 3.5,
    min_clip_residual_px: float = 1.0,
    max_iterations: int = 6,
) -> AffineWCSCalibration:
    """从先验匹配结果拟合局部仿射 WCS，并进行确定性的 MAD 剔除。

    这是 ``prior-WCS refinement``，不是盲解算：匹配本身仍由调用方用
    先验 WCS 和半径完成。这样可以避免把“密集星场中的最近邻巧合”误当
    作自动 plate solving。只有足够的、非共线的匹配点才允许进入校准。
    """

    if min_matches < 3:
        raise ValueError("min_matches must be at least 3")
    if clip_sigma <= 0 or min_clip_residual_px <= 0 or max_iterations <= 0:
        raise ValueError("clip_sigma, min_clip_residual_px and max_iterations must be positive")
    catalog_by_id = {source.source_id: source for source in catalog}
    rows: list[tuple[str, float, float, float, float]] = []
    for match in matches:
        source = catalog_by_id.get(str(match.source_id))
        if source is None:
            continue
        source = source.at_epoch(epoch)
        east, north = reference_wcs.world_to_tangent_arcsec(source.ra_deg, source.dec_deg)
        values = np.asarray((east, north, match.detection_x, match.detection_y), dtype=np.float64)
        if np.all(np.isfinite(values)):
            rows.append((source.source_id, float(east), float(north), float(match.detection_x), float(match.detection_y)))
    if len(rows) < min_matches:
        raise ValueError(f"至少需要 {min_matches} 个有效匹配点，当前只有 {len(rows)} 个")

    sky = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float64)
    pixels = np.asarray([[row[3], row[4]] for row in rows], dtype=np.float64)
    design = np.column_stack((sky, np.ones(len(rows), dtype=np.float64)))
    centered_sky = sky - np.mean(sky, axis=0)
    sky_singular_values = np.linalg.svd(centered_sky, compute_uv=False)
    if sky_singular_values[0] <= 0 or sky_singular_values[1] <= max(1e-6, sky_singular_values[0] * 1e-4):
        raise ValueError("匹配点在切平面上共线，无法同时估计尺度、旋转和位置")
    condition_design = np.column_stack(
        (centered_sky / max(1.0, float(sky_singular_values[0])), np.ones(len(rows), dtype=np.float64))
    )
    if np.linalg.matrix_rank(condition_design) < 3:
        raise ValueError("匹配点在切平面上共线，无法同时估计尺度、旋转和位置")
    condition_number = float(np.linalg.cond(condition_design))
    if not np.isfinite(condition_number) or condition_number > 1e10:
        raise ValueError("匹配点几何条件数过大，无法稳定估计仿射 WCS")

    def solve(mask: np.ndarray) -> np.ndarray:
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(design[mask], pixels[mask], rcond=None)
        if rank < 3:
            raise ValueError("有效匹配点不足以估计完整仿射模型")
        return coefficients.T

    inliers = np.ones(len(rows), dtype=bool)
    coefficients = solve(inliers)
    for _ in range(max_iterations):
        predicted = design @ coefficients.T
        residuals = np.linalg.norm(predicted - pixels, axis=1)
        active = residuals[inliers]
        median = float(np.median(active)) if active.size else float("inf")
        mad = float(np.median(np.abs(active - median))) if active.size else float("inf")
        cutoff = max(min_clip_residual_px, median + clip_sigma * 1.4826 * mad)
        new_inliers = residuals <= cutoff
        if int(new_inliers.sum()) < min_matches:
            break
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
        coefficients = solve(inliers)

    coefficients = solve(inliers)
    residuals = np.linalg.norm(design @ coefficients.T - pixels, axis=1)
    inlier_residuals = residuals[inliers]
    leave_one_out_residuals: list[float] = []
    for index in np.flatnonzero(inliers):
        leave_one_out_mask = inliers.copy()
        leave_one_out_mask[index] = False
        if np.linalg.matrix_rank(design[leave_one_out_mask]) < 3:
            continue
        leave_one_out_coefficients = solve(leave_one_out_mask)
        leave_one_out_prediction = design[index] @ leave_one_out_coefficients.T
        leave_one_out_residuals.append(float(np.linalg.norm(leave_one_out_prediction - pixels[index])))
    matrix = coefficients[:, :2]
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] <= 0 or not np.all(np.isfinite(singular_values)):
        raise ValueError("拟合出的仿射矩阵退化")
    plate_scale = float(1.0 / np.mean(singular_values))
    anisotropy = float(singular_values[0] / singular_values[-1])
    determinant = float(np.linalg.det(matrix))
    parity = 1 if determinant >= 0 else -1
    rotation = float(math.degrees(math.atan2(matrix[1, 0], matrix[1, 1])))
    return AffineWCSCalibration(
        center_ra_deg=reference_wcs.center_ra_deg,
        center_dec_deg=reference_wcs.center_dec_deg,
        matrix_px_per_arcsec=(
            (float(matrix[0, 0]), float(matrix[0, 1])),
            (float(matrix[1, 0]), float(matrix[1, 1])),
        ),
        offset_px=(float(coefficients[0, 2]), float(coefficients[1, 2])),
        plate_scale_arcsec_per_pixel=plate_scale,
        rotation_deg=rotation,
        parity=parity,
        anisotropy_ratio=anisotropy,
        matched_count=len(rows),
        inlier_count=int(inliers.sum()),
        rms_residual_px=float(np.sqrt(np.mean(inlier_residuals**2))),
        max_residual_px=float(np.max(inlier_residuals)),
        all_rms_residual_px=float(np.sqrt(np.mean(residuals**2))),
        all_max_residual_px=float(np.max(residuals)),
        condition_number=condition_number,
        inlier_source_ids=tuple(rows[index][0] for index in np.flatnonzero(inliers)),
        validation_count=len(leave_one_out_residuals),
        leave_one_out_rms_residual_px=(
            float(np.sqrt(np.mean(np.square(leave_one_out_residuals)))) if leave_one_out_residuals else None
        ),
        leave_one_out_max_residual_px=(float(np.max(leave_one_out_residuals)) if leave_one_out_residuals else None),
    )
