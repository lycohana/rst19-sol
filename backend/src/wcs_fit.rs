//! Robust local affine WCS fitting and per-frame validation.
//!
//! This is prior-WCS refinement from a supplied catalog match set. It does not
//! perform blind plate solving or assign physical identities to detections.
use crate::{
    catalog::CatalogSource,
    coordinates::{AffineProjection, TangentPlaneWcs},
    matching::CatalogMatch,
    statistics, Result,
};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineWcsCalibration {
    pub center_ra_deg: f64,
    pub center_dec_deg: f64,
    pub matrix_px_per_arcsec: [[f64; 2]; 2],
    pub offset_px: [f64; 2],
    pub plate_scale_arcsec_per_pixel: f64,
    pub rotation_deg: f64,
    pub parity: i32,
    pub anisotropy_ratio: f64,
    pub matched_count: usize,
    pub inlier_count: usize,
    pub rms_residual_px: f64,
    pub max_residual_px: f64,
    pub all_rms_residual_px: f64,
    pub all_max_residual_px: f64,
    pub condition_number: f64,
    pub inlier_source_ids: Vec<String>,
    pub validation_count: usize,
    pub leave_one_out_rms_residual_px: Option<f64>,
    pub leave_one_out_max_residual_px: Option<f64>,
}
impl AffineWcsCalibration {
    pub fn inlier_ratio(&self) -> f64 {
        if self.matched_count == 0 {
            0.0
        } else {
            self.inlier_count as f64 / self.matched_count as f64
        }
    }
    pub fn projection(&self) -> AffineProjection {
        AffineProjection {
            center_ra_deg: self.center_ra_deg,
            center_dec_deg: self.center_dec_deg,
            matrix_px_per_arcsec: self.matrix_px_per_arcsec,
            offset_px: self.offset_px,
        }
    }
    pub fn refined_wcs(&self) -> TangentPlaneWcs {
        TangentPlaneWcs {
            center_ra_deg: self.center_ra_deg,
            center_dec_deg: self.center_dec_deg,
            pixel_scale_arcsec: self.plate_scale_arcsec_per_pixel,
            crpix_x: self.offset_px[0],
            crpix_y: self.offset_px[1],
            rotation_deg: self.rotation_deg,
            parity: self.parity,
        }
    }
    pub fn world_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Result<(f64, f64)> {
        self.projection().world_to_pixel(ra_deg, dec_deg)
    }
    pub fn pixel_velocity_to_tangent_arcsec(&self, dx_px: f64, dy_px: f64) -> Result<(f64, f64)> {
        self.projection()
            .pixel_velocity_to_tangent_arcsec(dx_px, dy_px)
    }
    pub fn angular_speed_arcsec_per_s(&self, dx_px_per_s: f64, dy_px_per_s: f64) -> Result<f64> {
        self.projection()
            .angular_speed_arcsec_per_s(dx_px_per_s, dy_px_per_s)
    }
    pub fn as_dict(&self) -> Value {
        json!({
            "center_ra_deg":self.center_ra_deg,
            "center_dec_deg":self.center_dec_deg,
            "matrix_px_per_arcsec":self.matrix_px_per_arcsec,
            "offset_px":self.offset_px,
            "plate_scale_arcsec_per_pixel":self.plate_scale_arcsec_per_pixel,
            "rotation_deg":self.rotation_deg,
            "parity":self.parity,
            "anisotropy_ratio":self.anisotropy_ratio,
            "matched_count":self.matched_count,
            "inlier_count":self.inlier_count,
            "inlier_ratio":self.inlier_ratio(),
            "rms_residual_px":self.rms_residual_px,
            "max_residual_px":self.max_residual_px,
            "all_rms_residual_px":self.all_rms_residual_px,
            "all_max_residual_px":self.all_max_residual_px,
            "condition_number":self.condition_number,
            "inlier_source_ids":self.inlier_source_ids,
            "validation_method":"leave_one_out",
            "validation_count":self.validation_count,
            "leave_one_out_rms_residual_px":self.leave_one_out_rms_residual_px,
            "leave_one_out_max_residual_px":self.leave_one_out_max_residual_px
        })
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default)]
pub struct AffineFitOptions {
    pub min_matches: usize,
    pub clip_sigma: f64,
    pub min_clip_residual_px: f64,
    pub max_iterations: usize,
}
impl Default for AffineFitOptions {
    fn default() -> Self {
        Self {
            min_matches: 6,
            clip_sigma: 3.5,
            min_clip_residual_px: 1.0,
            max_iterations: 6,
        }
    }
}

#[derive(Debug, Clone)]
struct FitRow {
    source_id: String,
    east: f64,
    north: f64,
    x: f64,
    y: f64,
}

fn sorted_singular_values(matrix: DMatrix<f64>) -> Vec<f64> {
    let mut values: Vec<_> = matrix
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .collect();
    values.sort_by(|a, b| b.total_cmp(a));
    values
}

fn matrix_rank(matrix: DMatrix<f64>) -> usize {
    let (rows, columns) = matrix.shape();
    let values = sorted_singular_values(matrix);
    let largest = values.first().copied().unwrap_or(0.0);
    let tolerance = f64::EPSILON * rows.max(columns) as f64 * largest;
    values.iter().filter(|value| **value > tolerance).count()
}

fn design_rows(rows: &[FitRow], selected: &[bool]) -> (DMatrix<f64>, DMatrix<f64>) {
    let count = selected.iter().filter(|value| **value).count();
    let mut design = Vec::with_capacity(count * 3);
    let mut pixels = Vec::with_capacity(count * 2);
    for (row, keep) in rows.iter().zip(selected) {
        if *keep {
            design.extend([row.east, row.north, 1.0]);
            pixels.extend([row.x, row.y]);
        }
    }
    (
        DMatrix::from_row_slice(count, 3, &design),
        DMatrix::from_row_slice(count, 2, &pixels),
    )
}

fn solve(rows: &[FitRow], selected: &[bool]) -> Result<[[f64; 3]; 2]> {
    let (design, pixels) = design_rows(rows, selected);
    if design.nrows() < 3 || matrix_rank(design.clone()) < 3 {
        return Err("有效匹配点不足以估计完整仿射模型".into());
    }
    let svd = design.clone().svd(true, true);
    let largest = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let cutoff = f64::EPSILON * design.nrows().max(design.ncols()) as f64 * largest;
    let coefficients = svd
        .solve(&pixels, cutoff)
        .map_err(|_| "有效匹配点不足以估计完整仿射模型".to_owned())?;
    Ok([
        [
            coefficients[(0, 0)],
            coefficients[(1, 0)],
            coefficients[(2, 0)],
        ],
        [
            coefficients[(0, 1)],
            coefficients[(1, 1)],
            coefficients[(2, 1)],
        ],
    ])
}

fn residuals(rows: &[FitRow], coefficients: &[[f64; 3]; 2]) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            let predicted_x =
                row.east * coefficients[0][0] + row.north * coefficients[0][1] + coefficients[0][2];
            let predicted_y =
                row.east * coefficients[1][0] + row.north * coefficients[1][1] + coefficients[1][2];
            (predicted_x - row.x).hypot(predicted_y - row.y)
        })
        .collect()
}

pub fn fit_affine_wcs_from_matches(
    matches: &[CatalogMatch],
    catalog: &[CatalogSource],
    reference_wcs: &TangentPlaneWcs,
    epoch: Option<f64>,
    options: AffineFitOptions,
) -> Result<AffineWcsCalibration> {
    if options.min_matches < 3 {
        return Err("min_matches must be at least 3".into());
    }
    if options.clip_sigma <= 0.0
        || options.min_clip_residual_px <= 0.0
        || options.max_iterations == 0
    {
        return Err("clip_sigma, min_clip_residual_px and max_iterations must be positive".into());
    }
    let reference_wcs = reference_wcs.clone().validated()?;
    // Python's dict comprehension deliberately gives the last duplicate ID precedence.
    let mut catalog_by_id: BTreeMap<String, CatalogSource> = BTreeMap::new();
    for source in catalog {
        catalog_by_id.insert(source.source_id.clone(), source.clone());
    }
    let mut rows = Vec::new();
    for matched in matches {
        let Some(source) = catalog_by_id.get(&matched.source_id) else {
            continue;
        };
        let source = source.clone().validated()?.at_epoch(epoch)?;
        let (east, north) = reference_wcs.world_to_tangent_arcsec(source.ra_deg, source.dec_deg);
        if [east, north, matched.detection_x, matched.detection_y]
            .iter()
            .all(|value| value.is_finite())
        {
            rows.push(FitRow {
                source_id: source.source_id,
                east,
                north,
                x: matched.detection_x,
                y: matched.detection_y,
            });
        }
    }
    if rows.len() < options.min_matches {
        return Err(format!(
            "至少需要 {} 个有效匹配点，当前只有 {} 个",
            options.min_matches,
            rows.len()
        ));
    }

    let mean_east = rows.iter().map(|row| row.east).sum::<f64>() / rows.len() as f64;
    let mean_north = rows.iter().map(|row| row.north).sum::<f64>() / rows.len() as f64;
    let centered_values: Vec<_> = rows
        .iter()
        .flat_map(|row| [row.east - mean_east, row.north - mean_north])
        .collect();
    let centered = DMatrix::from_row_slice(rows.len(), 2, &centered_values);
    let sky_singular_values = sorted_singular_values(centered);
    if sky_singular_values.len() < 2
        || sky_singular_values[0] <= 0.0
        || sky_singular_values[1] <= 1e-6f64.max(sky_singular_values[0] * 1e-4)
    {
        return Err("匹配点在切平面上共线，无法同时估计尺度、旋转和位置".into());
    }
    let scale = sky_singular_values[0].max(1.0);
    let condition_values: Vec<_> = rows
        .iter()
        .flat_map(|row| {
            [
                (row.east - mean_east) / scale,
                (row.north - mean_north) / scale,
                1.0,
            ]
        })
        .collect();
    let condition_design = DMatrix::from_row_slice(rows.len(), 3, &condition_values);
    if matrix_rank(condition_design.clone()) < 3 {
        return Err("匹配点在切平面上共线，无法同时估计尺度、旋转和位置".into());
    }
    let condition_values = sorted_singular_values(condition_design);
    let condition_number = condition_values[0] / condition_values[condition_values.len() - 1];
    if !condition_number.is_finite() || condition_number > 1e10 {
        return Err("匹配点几何条件数过大，无法稳定估计仿射 WCS".into());
    }

    let mut inliers = vec![true; rows.len()];
    let mut coefficients = solve(&rows, &inliers)?;
    for _ in 0..options.max_iterations {
        let all_residuals = residuals(&rows, &coefficients);
        let active: Vec<_> = all_residuals
            .iter()
            .zip(&inliers)
            .filter_map(|(residual, keep)| keep.then_some(*residual))
            .collect();
        let median = statistics::median(&active)?;
        let deviations: Vec<_> = active.iter().map(|value| (value - median).abs()).collect();
        let mad = statistics::median(&deviations)?;
        let cutoff = options
            .min_clip_residual_px
            .max(median + options.clip_sigma * 1.4826 * mad);
        let new_inliers: Vec<_> = all_residuals.iter().map(|value| *value <= cutoff).collect();
        if new_inliers.iter().filter(|keep| **keep).count() < options.min_matches
            || new_inliers == inliers
        {
            break;
        }
        inliers = new_inliers;
        coefficients = solve(&rows, &inliers)?;
    }
    coefficients = solve(&rows, &inliers)?;
    let all_residuals = residuals(&rows, &coefficients);
    let inlier_residuals: Vec<_> = all_residuals
        .iter()
        .zip(&inliers)
        .filter_map(|(residual, keep)| keep.then_some(*residual))
        .collect();

    let mut leave_one_out_residuals = Vec::new();
    for index in 0..rows.len() {
        if !inliers[index] {
            continue;
        }
        let mut selected = inliers.clone();
        selected[index] = false;
        let (loo_design, _) = design_rows(&rows, &selected);
        if matrix_rank(loo_design) < 3 {
            continue;
        }
        let loo_coefficients = solve(&rows, &selected)?;
        leave_one_out_residuals.push(residuals(&[rows[index].clone()], &loo_coefficients)[0]);
    }

    let matrix = DMatrix::from_row_slice(
        2,
        2,
        &[
            coefficients[0][0],
            coefficients[0][1],
            coefficients[1][0],
            coefficients[1][1],
        ],
    );
    let matrix_singular_values = sorted_singular_values(matrix);
    if matrix_singular_values[1] <= 0.0
        || matrix_singular_values
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("拟合出的仿射矩阵退化".into());
    }
    let determinant =
        coefficients[0][0] * coefficients[1][1] - coefficients[0][1] * coefficients[1][0];
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let rms = |values: &[f64]| {
        (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
    };

    Ok(AffineWcsCalibration {
        center_ra_deg: reference_wcs.center_ra_deg,
        center_dec_deg: reference_wcs.center_dec_deg,
        matrix_px_per_arcsec: [
            [coefficients[0][0], coefficients[0][1]],
            [coefficients[1][0], coefficients[1][1]],
        ],
        offset_px: [coefficients[0][2], coefficients[1][2]],
        plate_scale_arcsec_per_pixel: 1.0 / mean(&matrix_singular_values),
        rotation_deg: coefficients[1][0].atan2(coefficients[1][1]).to_degrees(),
        parity: if determinant >= 0.0 { 1 } else { -1 },
        anisotropy_ratio: matrix_singular_values[0] / matrix_singular_values[1],
        matched_count: rows.len(),
        inlier_count: inlier_residuals.len(),
        rms_residual_px: rms(&inlier_residuals),
        max_residual_px: inlier_residuals
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        all_rms_residual_px: rms(&all_residuals),
        all_max_residual_px: all_residuals
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        condition_number,
        inlier_source_ids: rows
            .iter()
            .zip(&inliers)
            .filter_map(|(row, keep)| keep.then_some(row.source_id.clone()))
            .collect(),
        validation_count: leave_one_out_residuals.len(),
        leave_one_out_rms_residual_px: (!leave_one_out_residuals.is_empty())
            .then(|| rms(&leave_one_out_residuals)),
        leave_one_out_max_residual_px: (!leave_one_out_residuals.is_empty()).then(|| {
            leave_one_out_residuals
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        }),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WcsFrameValidation {
    pub frame_index: usize,
    pub frame_path: String,
    pub status: String,
    pub matched_count: usize,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub rms_residual_px: Option<f64>,
    pub max_residual_px: Option<f64>,
    pub leave_one_out_rms_residual_px: Option<f64>,
    pub leave_one_out_max_residual_px: Option<f64>,
    pub plate_scale_arcsec_per_pixel: Option<f64>,
    pub rotation_deg: Option<f64>,
    pub parity: Option<i32>,
    pub anisotropy_ratio: Option<f64>,
    pub condition_number: Option<f64>,
    pub reason: Option<String>,
}
impl WcsFrameValidation {
    pub fn as_dict(&self) -> Value {
        json!({
            "frame_index":self.frame_index,
            "frame_number":self.frame_index + 1,
            "frame_path":self.frame_path,
            "status":self.status,
            "matched_count":self.matched_count,
            "inlier_count":self.inlier_count,
            "inlier_ratio":self.inlier_ratio,
            "rms_residual_px":self.rms_residual_px,
            "max_residual_px":self.max_residual_px,
            "leave_one_out_rms_residual_px":self.leave_one_out_rms_residual_px,
            "leave_one_out_max_residual_px":self.leave_one_out_max_residual_px,
            "plate_scale_arcsec_per_pixel":self.plate_scale_arcsec_per_pixel,
            "rotation_deg":self.rotation_deg,
            "parity":self.parity,
            "anisotropy_ratio":self.anisotropy_ratio,
            "condition_number":self.condition_number,
            "reason":self.reason
        })
    }
}

pub fn validate_frame_matches(
    frame_matches: &[Vec<CatalogMatch>],
    catalog: &[CatalogSource],
    reference_wcs: &TangentPlaneWcs,
    frame_paths: Option<&[String]>,
    epoch: Option<f64>,
    options: AffineFitOptions,
) -> Result<Value> {
    if options.min_matches < 3 {
        return Err("min_matches must be at least 3".into());
    }
    if frame_paths.is_some_and(|paths| paths.len() != frame_matches.len()) {
        return Err("frame_paths length must match frame_matches".into());
    }
    let mut rows = Vec::with_capacity(frame_matches.len());
    for (frame_index, matches) in frame_matches.iter().enumerate() {
        let frame_path = frame_paths
            .map(|paths| paths[frame_index].clone())
            .unwrap_or_else(|| format!("frame-{:02}", frame_index + 1));
        match fit_affine_wcs_from_matches(matches, catalog, reference_wcs, epoch, options) {
            Ok(calibration) => rows.push(WcsFrameValidation {
                frame_index,
                frame_path,
                status: "validated".into(),
                matched_count: calibration.matched_count,
                inlier_count: calibration.inlier_count,
                inlier_ratio: calibration.inlier_ratio(),
                rms_residual_px: Some(calibration.rms_residual_px),
                max_residual_px: Some(calibration.max_residual_px),
                leave_one_out_rms_residual_px: calibration.leave_one_out_rms_residual_px,
                leave_one_out_max_residual_px: calibration.leave_one_out_max_residual_px,
                plate_scale_arcsec_per_pixel: Some(calibration.plate_scale_arcsec_per_pixel),
                rotation_deg: Some(calibration.rotation_deg),
                parity: Some(calibration.parity),
                anisotropy_ratio: Some(calibration.anisotropy_ratio),
                condition_number: Some(calibration.condition_number),
                reason: None,
            }),
            Err(reason) => rows.push(WcsFrameValidation {
                frame_index,
                frame_path,
                status: if matches.len() < options.min_matches {
                    "insufficient"
                } else {
                    "failed"
                }
                .into(),
                matched_count: matches.len(),
                inlier_count: 0,
                inlier_ratio: 0.0,
                rms_residual_px: None,
                max_residual_px: None,
                leave_one_out_rms_residual_px: None,
                leave_one_out_max_residual_px: None,
                plate_scale_arcsec_per_pixel: None,
                rotation_deg: None,
                parity: None,
                anisotropy_ratio: None,
                condition_number: None,
                reason: Some(reason),
            }),
        }
    }
    validation_report(rows, catalog.len(), reference_wcs, options.min_matches)
}

fn validation_report(
    rows: Vec<WcsFrameValidation>,
    catalog_source_count: usize,
    reference_wcs: &TangentPlaneWcs,
    min_matches: usize,
) -> Result<Value> {
    let validated: Vec<_> = rows
        .iter()
        .filter(|row| row.status == "validated")
        .collect();
    let range = |select: fn(&WcsFrameValidation) -> Option<f64>| -> Result<Value> {
        let values: Vec<_> = validated.iter().filter_map(|row| select(row)).collect();
        if values.is_empty() {
            return Ok(json!({"min":null,"median":null,"max":null}));
        }
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(json!({"min":min,"median":statistics::median(&values)?,"max":max}))
    };
    let reference_wcs = reference_wcs.clone().validated()?;
    let frame_count = rows.len();
    Ok(json!({
        "schema_version":1,
        "method":"每帧独立局部仿射拟合 + MAD 离群剔除 + 匹配点留一验证",
        "catalog_source_count":catalog_source_count,
        "reference_wcs":{
            "center_ra_deg":reference_wcs.center_ra_deg,
            "center_dec_deg":reference_wcs.center_dec_deg,
            "pixel_scale_arcsec":reference_wcs.pixel_scale_arcsec,
            "crpix_x":reference_wcs.crpix_x,
            "crpix_y":reference_wcs.crpix_y,
            "rotation_deg":reference_wcs.rotation_deg,
            "parity":reference_wcs.parity
        },
        "min_matches":min_matches,
        "frame_count":frame_count,
        "validated_count":validated.len(),
        "validation_ratio":if frame_count == 0 {0.0} else {validated.len() as f64 / frame_count as f64},
        "summary":{
            "rms_residual_px":range(|row|row.rms_residual_px)?,
            "leave_one_out_rms_residual_px":range(|row|row.leave_one_out_rms_residual_px)?,
            "plate_scale_arcsec_per_pixel":range(|row|row.plate_scale_arcsec_per_pixel)?,
            "anisotropy_ratio":range(|row|row.anisotropy_ratio)?
        },
        "frames":rows.iter().map(WcsFrameValidation::as_dict).collect::<Vec<_>>(),
        "note":"validated 只表示该帧有足够的唯一匹配并通过本地仿射拟合；它不是全天空盲解算，也不等于官方逐星真值。应继续检查跨帧尺度、旋转、parity、空间残差和人工抽检。"
    }))
}

pub fn validate_sequence_paths(
    frame_paths: &[String],
    catalog: &[CatalogSource],
    reference_wcs: &TangentPlaneWcs,
    options: &crate::pipeline::FrameOptions,
    match_radius_px: f64,
    epoch: Option<f64>,
    fit_options: AffineFitOptions,
) -> Result<Value> {
    if frame_paths.is_empty() {
        return Err("frame_paths cannot be empty".into());
    }
    if match_radius_px <= 0.0 {
        return Err("match_radius_px must be positive".into());
    }
    let catalog: Vec<_> = catalog
        .iter()
        .cloned()
        .map(CatalogSource::validated)
        .collect::<Result<_>>()?;
    let reference_wcs = reference_wcs.clone().validated()?;
    let projection = crate::matching::Projection::Tan(reference_wcs.clone());
    let mut parameters = options.clone();
    parameters.match_radius_px = match_radius_px;
    parameters.epoch = epoch;
    let mut frame_matches = Vec::with_capacity(frame_paths.len());
    for path in frame_paths {
        let frame = crate::fits::read_fits(path, false)?;
        let analysis = crate::pipeline::analyze_frame(
            frame,
            Some(&catalog),
            Some(&projection),
            &parameters,
            None,
        )?;
        frame_matches.push(
            analysis
                .matching
                .map(|result| result.matches)
                .unwrap_or_default(),
        );
    }
    validate_frame_matches(
        &frame_matches,
        &catalog,
        &reference_wcs,
        Some(frame_paths),
        epoch,
        fit_options,
    )
}
