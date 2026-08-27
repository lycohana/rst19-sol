"""不依赖 Astropy 的切平面线性 WCS 基线。"""

from __future__ import annotations

import math

import numpy as np


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

    def world_to_pixel(
        self, ra_deg: float | np.ndarray, dec_deg: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
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
        scale = self.scale_rad_per_pixel
        east = xi / scale
        north = eta / scale
        theta = math.radians(self.rotation_deg)
        x_rot = east * math.cos(theta) - north * math.sin(theta)
        y_rot = east * math.sin(theta) + north * math.cos(theta)
        x = self.crpix_x + self.parity * x_rot
        y = self.crpix_y + y_rot
        invalid = denominator <= 0
        x = np.where(invalid, np.nan, x)
        y = np.where(invalid, np.nan, y)
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
