//! TAN projection and Hamilton-quaternion geometry, with the original image conventions.
use crate::{models::AuxiliaryData, Result};
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

pub fn quaternion_to_active_matrix(q: &[f64]) -> Result<[[f64; 3]; 3]> {
    if q.len() != 4 || q.iter().any(|v| !v.is_finite()) {
        return Err("quaternion must contain four finite values".into());
    }
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        return Err("quaternion norm must be positive".into());
    }
    let (x, y, z, w) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);
    Ok([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])
}
pub fn vector_to_radec_deg(v: &[f64]) -> Result<(f64, f64)> {
    if v.len() != 3 || v.iter().any(|n| !n.is_finite()) {
        return Err("direction vector must contain three finite values".into());
    }
    let norm = v.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        return Err("direction vector norm must be positive".into());
    }
    Ok((
        (v[1] / norm)
            .atan2(v[0] / norm)
            .to_degrees()
            .rem_euclid(360.0),
        (v[2] / norm).asin().to_degrees(),
    ))
}
pub fn auxiliary_boresight_radec(aux: &AuxiliaryData, axis: &[f64]) -> Result<(f64, f64)> {
    if axis.len() != 3 || axis.iter().any(|v| !v.is_finite()) {
        return Err("camera_body_axis must contain three finite values".into());
    }
    let norm = axis.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        return Err("camera_body_axis norm must be positive".into());
    }
    let matrix = quaternion_to_active_matrix(&aux.values[10..14])?;
    let direction: Vec<_> = matrix
        .iter()
        .map(|row| row.iter().zip(axis).map(|(m, a)| m * a / norm).sum())
        .collect();
    vector_to_radec_deg(&direction)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TangentPlaneWcs {
    pub center_ra_deg: f64,
    pub center_dec_deg: f64,
    pub pixel_scale_arcsec: f64,
    pub crpix_x: f64,
    pub crpix_y: f64,
    #[serde(default)]
    pub rotation_deg: f64,
    #[serde(default = "default_parity")]
    pub parity: i32,
}
fn default_parity() -> i32 {
    1
}
impl TangentPlaneWcs {
    pub fn validated(mut self) -> Result<Self> {
        if self.pixel_scale_arcsec <= 0.0 {
            return Err("pixel_scale_arcsec must be positive".into());
        }
        if self.parity != 1 && self.parity != -1 {
            return Err("parity must be either 1 or -1".into());
        }
        if !(self.center_dec_deg > -90.0 && self.center_dec_deg < 90.0) {
            return Err("center_dec_deg must be strictly between -90 and 90".into());
        }
        self.center_ra_deg = self.center_ra_deg.rem_euclid(360.0);
        Ok(self)
    }
    pub fn scale_rad_per_pixel(&self) -> f64 {
        (self.pixel_scale_arcsec / 3600.0).to_radians()
    }
    pub fn world_to_tangent_arcsec(&self, ra: f64, dec: f64) -> (f64, f64) {
        let ra0 = self.center_ra_deg.to_radians();
        let dec0 = self.center_dec_deg.to_radians();
        let dec = dec.to_radians();
        let delta = (ra.to_radians() - ra0 + PI).rem_euclid(TAU) - PI;
        let denominator = dec0.sin() * dec.sin() + dec0.cos() * dec.cos() * delta.cos();
        if denominator <= 0.0 {
            return (f64::NAN, f64::NAN);
        }
        (
            (dec.cos() * delta.sin() / denominator).to_degrees() * 3600.0,
            ((dec0.cos() * dec.sin() - dec0.sin() * dec.cos() * delta.cos()) / denominator)
                .to_degrees()
                * 3600.0,
        )
    }
    pub fn world_to_pixel(&self, ra: f64, dec: f64) -> (f64, f64) {
        let (east, north) = self.world_to_tangent_arcsec(ra, dec);
        let (e, n) = (
            east / self.pixel_scale_arcsec,
            north / self.pixel_scale_arcsec,
        );
        let theta = self.rotation_deg.to_radians();
        (
            self.crpix_x + self.parity as f64 * (e * theta.cos() - n * theta.sin()),
            self.crpix_y + e * theta.sin() + n * theta.cos(),
        )
    }
    pub fn pixel_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let theta = self.rotation_deg.to_radians();
        let u = self.parity as f64 * (x - self.crpix_x);
        let v = y - self.crpix_y;
        let xi = (u * theta.cos() + v * theta.sin()) * self.scale_rad_per_pixel();
        let eta = (-u * theta.sin() + v * theta.cos()) * self.scale_rad_per_pixel();
        let rho = xi.hypot(eta);
        let c = rho.atan();
        let dec0 = self.center_dec_deg.to_radians();
        let ra0 = self.center_ra_deg.to_radians();
        if rho == 0.0 {
            return (self.center_ra_deg, self.center_dec_deg);
        }
        let dec = (c.cos() * dec0.sin() + eta * c.sin() * dec0.cos() / rho).asin();
        let ra =
            ra0 + (xi * c.sin()).atan2(rho * dec0.cos() * c.cos() - eta * dec0.sin() * c.sin());
        (ra.to_degrees().rem_euclid(360.0), dec.to_degrees())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineProjection {
    pub center_ra_deg: f64,
    pub center_dec_deg: f64,
    pub matrix_px_per_arcsec: [[f64; 2]; 2],
    pub offset_px: [f64; 2],
}
impl AffineProjection {
    pub fn world_to_pixel(&self, ra: f64, dec: f64) -> Result<(f64, f64)> {
        let reference = TangentPlaneWcs {
            center_ra_deg: self.center_ra_deg,
            center_dec_deg: self.center_dec_deg,
            pixel_scale_arcsec: 1.0,
            crpix_x: 0.0,
            crpix_y: 0.0,
            rotation_deg: 0.0,
            parity: 1,
        }
        .validated()?;
        let (e, n) = reference.world_to_tangent_arcsec(ra, dec);
        let m = self.matrix_px_per_arcsec;
        Ok((
            m[0][0] * e + m[0][1] * n + self.offset_px[0],
            m[1][0] * e + m[1][1] * n + self.offset_px[1],
        ))
    }
    pub fn pixel_velocity_to_tangent_arcsec(&self, dx: f64, dy: f64) -> Result<(f64, f64)> {
        let [[a, b], [c, d]] = self.matrix_px_per_arcsec;
        let det = a * d - b * c;
        if det.abs() < 1e-15 {
            return Err("calibration matrix is singular".into());
        }
        Ok(((d * dx - b * dy) / det, (-c * dx + a * dy) / det))
    }
    pub fn angular_speed_arcsec_per_s(&self, dx: f64, dy: f64) -> Result<f64> {
        let (e, n) = self.pixel_velocity_to_tangent_arcsec(dx, dy)?;
        Ok(e.hypot(n))
    }
}
