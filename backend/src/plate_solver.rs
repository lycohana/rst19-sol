//! Explicit local-catalog plate solving from star-pair geometry.
//!
//! This is not blind astrometric truth. It starts from a supplied field center,
//! pixel-scale prior, FITS image, and local catalog, then retains unique-match,
//! coverage, residual, and leave-one-out evidence for every accepted solution.

use crate::{
    catalog::CatalogSource,
    catalog_csv,
    coordinates::TangentPlaneWcs,
    detection::{self, DetectionOptions},
    fits,
    matching::CatalogMatch,
    measurement::Detection,
    Result,
};
use nalgebra::DMatrix;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

const BAD_SOLVER_FLAGS: [&str; 7] = [
    "EDGE",
    "MASKED",
    "SATURATED",
    "LINE_ARTIFACT",
    "CODE_PATTERN",
    "RANGE_ANOMALY",
    "UNRESOLVED_BLEND",
];

#[derive(Debug, Clone, Serialize)]
pub struct PlateTransform {
    pub matrix_px_per_arcsec: [[f64; 2]; 2],
    pub offset_px: [f64; 2],
    pub plate_scale_arcsec_per_pixel: f64,
    pub rotation_deg: f64,
    pub parity: i32,
    pub anisotropy_ratio: f64,
}

impl PlateTransform {
    fn project(&self, point: [f64; 2]) -> [f64; 2] {
        [
            self.matrix_px_per_arcsec[0][0] * point[0]
                + self.matrix_px_per_arcsec[0][1] * point[1]
                + self.offset_px[0],
            self.matrix_px_per_arcsec[1][0] * point[0]
                + self.matrix_px_per_arcsec[1][1] * point[1]
                + self.offset_px[1],
        ]
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PlateSolveCandidate {
    pub transform: PlateTransform,
    pub matched_count: usize,
    pub rms_residual_px: f64,
    pub max_residual_px: f64,
    pub coverage_x: f64,
    pub coverage_y: f64,
    pub coverage_area: f64,
    pub leave_one_out_rms_residual_px: Option<f64>,
    pub leave_one_out_max_residual_px: Option<f64>,
    pub seed_image_pair: [i64; 2],
    pub seed_catalog_pair: [String; 2],
    pub matches: Vec<CatalogMatch>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlateSolveResult {
    pub status: String,
    pub reason: String,
    pub valid: bool,
    pub best: Option<PlateSolveCandidate>,
    pub alternatives: Vec<PlateSolveCandidate>,
    pub image_points_considered: usize,
    pub catalog_points_considered: usize,
    pub pair_hypotheses: usize,
    pub unique_hypotheses: usize,
    pub acceptance: Value,
}

#[derive(Debug, Clone, Copy)]
pub struct PlateSolveOptions {
    pub epoch: Option<f64>,
    pub image_shape: Option<(usize, usize)>,
    pub scale_tolerance: f64,
    pub min_pair_distance_px: f64,
    pub match_radius_px: f64,
    pub min_matches: usize,
    pub min_coverage_area: f64,
    pub max_rms_residual_px: f64,
    pub max_leave_one_out_rms_px: f64,
    pub max_image_points: usize,
    pub max_catalog_points: usize,
    pub max_pair_hypotheses: usize,
    pub quality_only: bool,
    pub max_iterations: usize,
}

impl Default for PlateSolveOptions {
    fn default() -> Self {
        Self {
            epoch: None,
            image_shape: None,
            scale_tolerance: 0.03,
            min_pair_distance_px: 12.0,
            match_radius_px: 3.0,
            min_matches: 8,
            min_coverage_area: 0.02,
            max_rms_residual_px: 2.0,
            max_leave_one_out_rms_px: 3.0,
            max_image_points: 80,
            max_catalog_points: 160,
            max_pair_hypotheses: 20_000,
            quality_only: true,
            max_iterations: 5,
        }
    }
}

#[derive(Debug, Clone)]
struct Pair {
    distance: f64,
    left: usize,
    right: usize,
}

#[derive(Debug, Clone)]
struct FitRow {
    match_row: CatalogMatch,
    east: f64,
    north: f64,
}

fn finite(value: f64) -> bool {
    value.is_finite()
}

fn detection_rank(source: &Detection) -> f64 {
    [
        source.flux_snr,
        source.filter_snr,
        Some(source.snr),
        Some(source.peak),
    ]
    .into_iter()
    .flatten()
    .find(|value| value.is_finite())
    .unwrap_or(f64::NEG_INFINITY)
}

fn catalog_rank(source: &CatalogSource) -> f64 {
    source
        .magnitude
        .filter(|value| value.is_finite())
        .unwrap_or(f64::INFINITY)
}

fn select_detections(
    detections: &[Detection],
    max_points: usize,
    quality_only: bool,
) -> Vec<Detection> {
    let mut selected: Vec<_> = detections
        .iter()
        .filter(|source| {
            finite(source.x)
                && finite(source.y)
                && (!quality_only || source.quality_passed)
                && !source.flags.iter().any(|flag| {
                    BAD_SOLVER_FLAGS
                        .iter()
                        .any(|bad| flag.trim().eq_ignore_ascii_case(bad))
                })
        })
        .cloned()
        .collect();
    selected.sort_by(|a, b| {
        detection_rank(b)
            .total_cmp(&detection_rank(a))
            .then_with(|| a.detection_id.cmp(&b.detection_id))
    });
    selected.truncate(max_points);
    selected
}

fn tangent_point(
    source: &CatalogSource,
    reference_wcs: &TangentPlaneWcs,
    epoch: Option<f64>,
) -> Result<[f64; 2]> {
    let source = source.at_epoch(epoch)?;
    let (east, north) = reference_wcs.world_to_tangent_arcsec(source.ra_deg, source.dec_deg);
    Ok([east, north])
}

fn select_catalog(
    catalog: &[CatalogSource],
    reference_wcs: &TangentPlaneWcs,
    image_shape: Option<(usize, usize)>,
    epoch: Option<f64>,
    max_points: usize,
) -> Result<Vec<CatalogSource>> {
    let mut selected: Vec<_> = catalog
        .iter()
        .filter(|source| source.magnitude.is_some_and(f64::is_finite))
        .cloned()
        .collect();
    if let Some((image_height, image_width)) = image_shape {
        if image_height > 0 && image_width > 0 {
            let inner_radius = image_height.min(image_width) as f64
                * reference_wcs.pixel_scale_arcsec
                * 0.5
                * 1.05;
            let mut inner = Vec::new();
            for source in &selected {
                let [east, north] = tangent_point(source, reference_wcs, epoch)?;
                if finite(east) && finite(north) && east.hypot(north) <= inner_radius {
                    inner.push(source.clone());
                }
            }
            if inner.len() >= 8.min(max_points) {
                selected = inner;
            }
        }
    }
    selected.sort_by(|a, b| {
        catalog_rank(a)
            .total_cmp(&catalog_rank(b))
            .then_with(|| a.source_id.cmp(&b.source_id))
    });
    selected.truncate(max_points);
    Ok(selected)
}

fn catalog_tangent_points(
    catalog: &[CatalogSource],
    reference_wcs: &TangentPlaneWcs,
    epoch: Option<f64>,
) -> Result<Vec<[f64; 2]>> {
    catalog
        .iter()
        .map(|source| tangent_point(source, reference_wcs, epoch))
        .collect()
}

fn pair_rows(points: &[[f64; 2]], min_distance: f64) -> Vec<Pair> {
    let mut rows = Vec::new();
    for left in 0..points.len().saturating_sub(1) {
        for right in (left + 1)..points.len() {
            let distance =
                (points[right][0] - points[left][0]).hypot(points[right][1] - points[left][1]);
            if distance.is_finite() && distance >= min_distance {
                rows.push(Pair {
                    distance,
                    left,
                    right,
                });
            }
        }
    }
    rows
}

fn transform_from_matrix(matrix: [[f64; 2]; 2], offset: [f64; 2]) -> Option<PlateTransform> {
    if matrix
        .iter()
        .flatten()
        .chain(offset.iter())
        .any(|value| !finite(*value))
    {
        return None;
    }
    let dm = DMatrix::from_row_slice(
        2,
        2,
        &[matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]],
    );
    let mut singular: Vec<_> = dm
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .collect();
    singular.sort_by(|a, b| b.total_cmp(a));
    if singular.len() != 2 || singular[1] <= 0.0 || singular.iter().any(|v| !finite(*v)) {
        return None;
    }
    let scale = 1.0 / ((singular[0] + singular[1]) / 2.0);
    if !finite(scale) || scale <= 0.0 {
        return None;
    }
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    Some(PlateTransform {
        matrix_px_per_arcsec: matrix,
        offset_px: offset,
        plate_scale_arcsec_per_pixel: scale,
        rotation_deg: matrix[1][0].atan2(matrix[0][0]).to_degrees(),
        parity: if determinant >= 0.0 { 1 } else { -1 },
        anisotropy_ratio: singular[0] / singular[1],
    })
}

fn seed_transform(
    sky_a: [f64; 2],
    sky_b: [f64; 2],
    pixel_a: [f64; 2],
    pixel_b: [f64; 2],
    reflected: bool,
) -> Option<PlateTransform> {
    let sx = sky_b[0] - sky_a[0];
    let sy = sky_b[1] - sky_a[1];
    let tx = pixel_b[0] - pixel_a[0];
    let ty = pixel_b[1] - pixel_a[1];
    let norm2 = sx * sx + sy * sy;
    if norm2 <= 0.0 || !finite(norm2) {
        return None;
    }
    let matrix = if reflected {
        let c = (sx * tx - sy * ty) / norm2;
        let s = (sy * tx + sx * ty) / norm2;
        [[c, s], [s, -c]]
    } else {
        let c = (sx * tx + sy * ty) / norm2;
        let s = (sx * ty - sy * tx) / norm2;
        [[c, -s], [s, c]]
    };
    let offset = [
        pixel_a[0] - matrix[0][0] * sky_a[0] - matrix[0][1] * sky_a[1],
        pixel_a[1] - matrix[1][0] * sky_a[0] - matrix[1][1] * sky_a[1],
    ];
    transform_from_matrix(matrix, offset)
}

fn matches_for_transform(
    transform: &PlateTransform,
    image_points: &[Detection],
    catalog: &[CatalogSource],
    tangent_points: &[[f64; 2]],
    radius_px: f64,
    epoch: Option<f64>,
) -> Result<Vec<CatalogMatch>> {
    if image_points.is_empty() || catalog.is_empty() {
        return Ok(Vec::new());
    }
    let predicted: Vec<_> = tangent_points
        .iter()
        .map(|point| transform.project(*point))
        .collect();
    let mut proposals = Vec::<(f64, usize, usize)>::new();
    for (catalog_index, predicted_point) in predicted.iter().enumerate() {
        let nearest = image_points
            .iter()
            .enumerate()
            .map(|(image_index, source)| {
                let distance = (predicted_point[0] - source.x).hypot(predicted_point[1] - source.y);
                (distance, image_index)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        if let Some((distance, image_index)) = nearest {
            if distance.is_finite() && distance <= radius_px {
                proposals.push((distance, image_index, catalog_index));
            }
        }
    }
    proposals.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| {
                image_points[a.1]
                    .detection_id
                    .cmp(&image_points[b.1].detection_id)
            })
            .then_with(|| catalog[a.2].source_id.cmp(&catalog[b.2].source_id))
    });
    let mut used_image = HashSet::new();
    let mut used_catalog = HashSet::new();
    let mut matches = Vec::new();
    for (residual, image_index, catalog_index) in proposals {
        if !used_image.insert(image_index) {
            continue;
        }
        if !used_catalog.insert(catalog_index) {
            used_image.remove(&image_index);
            continue;
        }
        let detection = &image_points[image_index];
        let source = catalog[catalog_index].at_epoch(epoch)?;
        matches.push(CatalogMatch {
            detection_id: detection.detection_id,
            source_id: source.source_id,
            detection_x: detection.x,
            detection_y: detection.y,
            predicted_x: predicted[catalog_index][0],
            predicted_y: predicted[catalog_index][1],
            residual_px: residual,
            catalog_magnitude: source.magnitude,
            catalog_magnitude_error: source.magnitude_error,
            catalog_color: source.color,
            catalog_color_name: source.color_name,
            photometric_system: source.photometric_system,
            photometric_band: source.photometric_band,
        });
    }
    Ok(matches)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn solve_linear(rows: &[FitRow], active: &[bool]) -> Option<[[f64; 3]; 2]> {
    let selected: Vec<_> = rows
        .iter()
        .zip(active)
        .filter_map(|(row, keep)| keep.then_some(row))
        .collect();
    if selected.len() < 3 {
        return None;
    }
    let design_values: Vec<_> = selected
        .iter()
        .flat_map(|row| [row.east, row.north, 1.0])
        .collect();
    let pixel_values: Vec<_> = selected
        .iter()
        .flat_map(|row| [row.match_row.detection_x, row.match_row.detection_y])
        .collect();
    let design = DMatrix::from_row_slice(selected.len(), 3, &design_values);
    let pixels = DMatrix::from_row_slice(selected.len(), 2, &pixel_values);
    let svd = design.clone().svd(true, true);
    let largest = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let cutoff = f64::EPSILON * selected.len().max(3) as f64 * largest;
    if svd
        .singular_values
        .iter()
        .filter(|value| **value > cutoff)
        .count()
        < 3
    {
        return None;
    }
    let coefficients = svd.solve(&pixels, cutoff).ok()?;
    Some([
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

fn fit_residuals(rows: &[FitRow], coefficients: [[f64; 3]; 2]) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            let x =
                row.east * coefficients[0][0] + row.north * coefficients[0][1] + coefficients[0][2];
            let y =
                row.east * coefficients[1][0] + row.north * coefficients[1][1] + coefficients[1][2];
            (x - row.match_row.detection_x).hypot(y - row.match_row.detection_y)
        })
        .collect()
}

fn fit_affine(
    matches: &[CatalogMatch],
    catalog_by_id: &BTreeMap<String, CatalogSource>,
    reference_wcs: &TangentPlaneWcs,
    epoch: Option<f64>,
    clip_sigma: f64,
    max_iterations: usize,
) -> Result<Option<(PlateTransform, Vec<CatalogMatch>)>> {
    let mut rows = Vec::new();
    for matched in matches {
        let Some(source) = catalog_by_id.get(&matched.source_id) else {
            continue;
        };
        let [east, north] = tangent_point(source, reference_wcs, epoch)?;
        if [east, north, matched.detection_x, matched.detection_y]
            .iter()
            .all(|value| finite(*value))
        {
            rows.push(FitRow {
                match_row: matched.clone(),
                east,
                north,
            });
        }
    }
    if rows.len() < 3 {
        return Ok(None);
    }
    let mut active = vec![true; rows.len()];
    let Some(mut coefficients) = solve_linear(&rows, &active) else {
        return Ok(None);
    };
    for _ in 0..max_iterations.max(1) {
        let residuals = fit_residuals(&rows, coefficients);
        let active_residuals: Vec<_> = residuals
            .iter()
            .zip(&active)
            .filter_map(|(residual, keep)| keep.then_some(*residual))
            .collect();
        let Some(center) = median(&active_residuals) else {
            return Ok(None);
        };
        let deviations: Vec<_> = active_residuals
            .iter()
            .map(|residual| (residual - center).abs())
            .collect();
        let Some(mad) = median(&deviations) else {
            return Ok(None);
        };
        let cutoff = 1.0_f64.max(center + clip_sigma * (1e-6_f64).max(1.4826 * mad));
        let next: Vec<_> = residuals
            .iter()
            .map(|residual| *residual <= cutoff)
            .collect();
        if next.iter().filter(|keep| **keep).count() < 3 || next == active {
            if next.iter().filter(|keep| **keep).count() >= 3 {
                active = next;
            }
            break;
        }
        active = next;
        let Some(fitted) = solve_linear(&rows, &active) else {
            return Ok(None);
        };
        coefficients = fitted;
    }
    let Some(coefficients) = solve_linear(&rows, &active) else {
        return Ok(None);
    };
    let matrix = [
        [coefficients[0][0], coefficients[0][1]],
        [coefficients[1][0], coefficients[1][1]],
    ];
    let offset = [coefficients[0][2], coefficients[1][2]];
    let Some(transform) = transform_from_matrix(matrix, offset) else {
        return Ok(None);
    };
    let retained = rows
        .into_iter()
        .zip(active)
        .filter_map(|(row, keep)| keep.then_some(row.match_row))
        .collect();
    Ok(Some((transform, retained)))
}

fn leave_one_out(
    matches: &[CatalogMatch],
    catalog_by_id: &BTreeMap<String, CatalogSource>,
    reference_wcs: &TangentPlaneWcs,
    epoch: Option<f64>,
) -> Result<(Option<f64>, Option<f64>)> {
    let mut residuals = Vec::new();
    for held_index in 0..matches.len() {
        let subset: Vec<_> = matches
            .iter()
            .enumerate()
            .filter_map(|(index, matched)| (index != held_index).then_some(matched.clone()))
            .collect();
        let Some((transform, _)) =
            fit_affine(&subset, catalog_by_id, reference_wcs, epoch, 3.5, 4)?
        else {
            continue;
        };
        let matched = &matches[held_index];
        let Some(source) = catalog_by_id.get(&matched.source_id) else {
            continue;
        };
        let [east, north] = tangent_point(source, reference_wcs, epoch)?;
        let predicted = transform.project([east, north]);
        residuals
            .push((predicted[0] - matched.detection_x).hypot(predicted[1] - matched.detection_y));
    }
    if residuals.is_empty() {
        return Ok((None, None));
    }
    let rms =
        (residuals.iter().map(|value| value * value).sum::<f64>() / residuals.len() as f64).sqrt();
    let maximum = residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok((Some(rms), Some(maximum)))
}

fn coverage(matches: &[CatalogMatch], image_shape: Option<(usize, usize)>) -> (f64, f64, f64) {
    if matches.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let min_x = matches
        .iter()
        .map(|row| row.detection_x)
        .fold(f64::INFINITY, f64::min);
    let max_x = matches
        .iter()
        .map(|row| row.detection_x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = matches
        .iter()
        .map(|row| row.detection_y)
        .fold(f64::INFINITY, f64::min);
    let max_y = matches
        .iter()
        .map(|row| row.detection_y)
        .fold(f64::NEG_INFINITY, f64::max);
    let Some((height, width)) = image_shape else {
        return (1.0, 1.0, 1.0);
    };
    if width <= 1 || height <= 1 {
        return (0.0, 0.0, 0.0);
    }
    let coverage_x = ((max_x - min_x) / (width - 1) as f64).clamp(0.0, 1.0);
    let coverage_y = ((max_y - min_y) / (height - 1) as f64).clamp(0.0, 1.0);
    (coverage_x, coverage_y, coverage_x * coverage_y)
}

fn candidate_key(transform: &PlateTransform) -> Vec<i64> {
    transform
        .matrix_px_per_arcsec
        .iter()
        .flatten()
        .map(|value| (value * 1e8).round() as i64)
        .chain(
            transform
                .offset_px
                .iter()
                .map(|value| (value * 1e2).round() as i64),
        )
        .collect()
}

fn compare_candidates(a: &PlateSolveCandidate, b: &PlateSolveCandidate) -> Ordering {
    let tight_a = a
        .matches
        .iter()
        .filter(|matched| matched.residual_px <= 1.0)
        .count();
    let tight_b = b
        .matches
        .iter()
        .filter(|matched| matched.residual_px <= 1.0)
        .count();
    let loo_a = a.leave_one_out_rms_residual_px.unwrap_or(f64::NEG_INFINITY);
    let loo_b = b.leave_one_out_rms_residual_px.unwrap_or(f64::NEG_INFINITY);
    tight_b
        .cmp(&tight_a)
        .then_with(|| loo_b.total_cmp(&loo_a))
        .then_with(|| a.rms_residual_px.total_cmp(&b.rms_residual_px))
        .then_with(|| b.coverage_area.total_cmp(&a.coverage_area))
        .then_with(|| a.max_residual_px.total_cmp(&b.max_residual_px))
        .then_with(|| b.matched_count.cmp(&a.matched_count))
}

fn acceptance(options: PlateSolveOptions) -> Value {
    json!({
        "min_matches": options.min_matches,
        "min_coverage_area": options.min_coverage_area,
        "max_rms_residual_px": options.max_rms_residual_px,
        "max_leave_one_out_rms_px": options.max_leave_one_out_rms_px,
        "scale_tolerance": options.scale_tolerance,
        "match_radius_px": options.match_radius_px
    })
}

fn empty_result(
    status: &str,
    reason: &str,
    image_count: usize,
    catalog_count: usize,
    hypotheses: usize,
    unique: usize,
    acceptance: Value,
) -> PlateSolveResult {
    PlateSolveResult {
        status: status.into(),
        reason: reason.into(),
        valid: false,
        best: None,
        alternatives: Vec::new(),
        image_points_considered: image_count,
        catalog_points_considered: catalog_count,
        pair_hypotheses: hypotheses,
        unique_hypotheses: unique,
        acceptance,
    }
}

pub fn solve_plate(
    detections: &[Detection],
    catalog: &[CatalogSource],
    reference_wcs: &TangentPlaneWcs,
    options: PlateSolveOptions,
) -> Result<PlateSolveResult> {
    if !(0.0 < options.scale_tolerance && options.scale_tolerance < 1.0) {
        return Err("scale_tolerance must be in (0, 1)".into());
    }
    if options.min_pair_distance_px <= 0.0
        || options.match_radius_px <= 0.0
        || options.min_matches < 3
    {
        return Err("pair distance, match radius and min_matches must be positive".into());
    }
    if options.max_image_points < 2
        || options.max_catalog_points < 2
        || options.max_pair_hypotheses < 1
    {
        return Err("point and hypothesis limits are too small".into());
    }
    let reference_wcs = reference_wcs.clone().validated()?;
    let image_sources =
        select_detections(detections, options.max_image_points, options.quality_only);
    let catalog_sources = select_catalog(
        catalog,
        &reference_wcs,
        options.image_shape,
        options.epoch,
        options.max_catalog_points,
    )?;
    let acceptance = acceptance(options);
    if image_sources.len() < 2 || catalog_sources.len() < 2 {
        return Ok(empty_result(
            "INSUFFICIENT_POINTS",
            "筛选后检测点或星表点少于两个，无法提出星对",
            image_sources.len(),
            catalog_sources.len(),
            0,
            0,
            acceptance,
        ));
    }
    let tangent = catalog_tangent_points(&catalog_sources, &reference_wcs, options.epoch)?;
    let pixel_points: Vec<_> = image_sources
        .iter()
        .map(|source| [source.x, source.y])
        .collect();
    let image_pairs = pair_rows(&pixel_points, options.min_pair_distance_px);
    let mut catalog_pairs = pair_rows(&tangent, 0.0);
    catalog_pairs.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    let catalog_distances: Vec<_> = catalog_pairs.iter().map(|pair| pair.distance).collect();
    let scale_min = reference_wcs.pixel_scale_arcsec * (1.0 - options.scale_tolerance);
    let scale_max = reference_wcs.pixel_scale_arcsec * (1.0 + options.scale_tolerance);
    let mut hypotheses = 0;
    let mut unique_keys = HashSet::new();
    let mut candidates = Vec::new();
    let mut catalog_by_id = BTreeMap::new();
    for source in &catalog_sources {
        catalog_by_id.insert(source.source_id.clone(), source.clone());
    }
    'image_pairs: for image_pair in image_pairs {
        let minimum = image_pair.distance * scale_min;
        let maximum = image_pair.distance * scale_max;
        let left = catalog_distances.partition_point(|distance| *distance < minimum);
        let right = catalog_distances.partition_point(|distance| *distance <= maximum);
        for catalog_pair in &catalog_pairs[left..right] {
            for reverse in [false, true] {
                if hypotheses >= options.max_pair_hypotheses {
                    break 'image_pairs;
                }
                hypotheses += 1;
                let (catalog_a, catalog_b) = if reverse {
                    (catalog_pair.right, catalog_pair.left)
                } else {
                    (catalog_pair.left, catalog_pair.right)
                };
                for reflected in [false, true] {
                    let Some(seed) = seed_transform(
                        tangent[catalog_a],
                        tangent[catalog_b],
                        pixel_points[image_pair.left],
                        pixel_points[image_pair.right],
                        reflected,
                    ) else {
                        continue;
                    };
                    if !unique_keys.insert(candidate_key(&seed)) {
                        continue;
                    }
                    let matches = matches_for_transform(
                        &seed,
                        &image_sources,
                        &catalog_sources,
                        &tangent,
                        options.match_radius_px,
                        options.epoch,
                    )?;
                    if matches.len() < 3 {
                        continue;
                    }
                    let Some((refined, _retained_matches)) = fit_affine(
                        &matches,
                        &catalog_by_id,
                        &reference_wcs,
                        options.epoch,
                        3.5,
                        options.max_iterations,
                    )?
                    else {
                        continue;
                    };
                    let refined_matches = matches_for_transform(
                        &refined,
                        &image_sources,
                        &catalog_sources,
                        &tangent,
                        options.match_radius_px,
                        options.epoch,
                    )?;
                    let final_matches = if refined_matches.len() >= matches.len() {
                        refined_matches
                    } else {
                        matches
                    };
                    if final_matches.len() < 3 {
                        continue;
                    }
                    let residuals: Vec<_> =
                        final_matches.iter().map(|row| row.residual_px).collect();
                    let rms = (residuals.iter().map(|value| value * value).sum::<f64>()
                        / residuals.len() as f64)
                        .sqrt();
                    let maximum = residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let (coverage_x, coverage_y, coverage_area) =
                        coverage(&final_matches, options.image_shape);
                    let (loo_rms, loo_max) = leave_one_out(
                        &final_matches,
                        &catalog_by_id,
                        &reference_wcs,
                        options.epoch,
                    )?;
                    candidates.push(PlateSolveCandidate {
                        transform: refined,
                        matched_count: final_matches.len(),
                        rms_residual_px: rms,
                        max_residual_px: maximum,
                        coverage_x,
                        coverage_y,
                        coverage_area,
                        leave_one_out_rms_residual_px: loo_rms,
                        leave_one_out_max_residual_px: loo_max,
                        seed_image_pair: [
                            image_sources[image_pair.left].detection_id,
                            image_sources[image_pair.right].detection_id,
                        ],
                        seed_catalog_pair: [
                            catalog_sources[catalog_a].source_id.clone(),
                            catalog_sources[catalog_b].source_id.clone(),
                        ],
                        matches: final_matches,
                    });
                }
            }
        }
    }
    if candidates.is_empty() {
        return Ok(empty_result(
            "NO_SOLUTION",
            "没有候选变换形成至少三个唯一匹配点",
            image_sources.len(),
            catalog_sources.len(),
            hypotheses,
            unique_keys.len(),
            acceptance,
        ));
    }
    candidates.sort_by(compare_candidates);
    let best = candidates.remove(0);
    let loo_rms = best.leave_one_out_rms_residual_px;
    let loo_ok =
        loo_rms.is_some_and(|value| value.is_finite() && value <= options.max_leave_one_out_rms_px);
    let accepted = best.matched_count >= options.min_matches
        && best.rms_residual_px <= options.max_rms_residual_px
        && best.coverage_area >= options.min_coverage_area
        && loo_ok;
    let (status, reason) = if accepted {
        (
            "VALID".to_owned(),
            "通过唯一匹配、残差、覆盖度和留一验证".to_owned(),
        )
    } else {
        let mut failed = Vec::new();
        if best.matched_count < options.min_matches {
            failed.push(format!(
                "匹配数 {} < {}",
                best.matched_count, options.min_matches
            ));
        }
        if best.rms_residual_px > options.max_rms_residual_px {
            failed.push(format!(
                "RMS {:.3}px > {:.3}px",
                best.rms_residual_px, options.max_rms_residual_px
            ));
        }
        if best.coverage_area < options.min_coverage_area {
            failed.push(format!(
                "覆盖度 {:.4} < {:.4}",
                best.coverage_area, options.min_coverage_area
            ));
        }
        if loo_rms.is_none() {
            failed.push("留一验证无有效样本".into());
        } else if loo_rms.is_some_and(|value| !value.is_finite()) {
            failed.push("留一残差无效".into());
        } else if loo_rms.is_some_and(|value| value > options.max_leave_one_out_rms_px) {
            failed.push("留一残差超过门槛".into());
        }
        let reason = if failed.is_empty() {
            "未通过严格验收".into()
        } else {
            failed.join("；")
        };
        ("REJECTED".to_owned(), reason)
    };
    let alternatives = candidates.into_iter().take(5).collect();
    Ok(PlateSolveResult {
        status,
        reason,
        valid: accepted,
        best: Some(best),
        alternatives,
        image_points_considered: image_sources.len(),
        catalog_points_considered: catalog_sources.len(),
        pair_hypotheses: hypotheses,
        unique_hypotheses: unique_keys.len(),
        acceptance,
    })
}

#[derive(Debug, Clone)]
struct CliOptions {
    frame: PathBuf,
    catalog: PathBuf,
    center_ra: f64,
    center_dec: f64,
    pixel_scale_arcsec: f64,
    epoch: Option<f64>,
    threshold_sigma: f64,
    min_distance: usize,
    aperture_radius: usize,
    psf_fwhm: f64,
    max_detections: usize,
    solver: PlateSolveOptions,
    output: PathBuf,
}

fn parse_value<T: std::str::FromStr>(name: &str, value: String) -> Result<T> {
    value
        .parse()
        .map_err(|_| format!("{name} has an invalid value: {value}"))
}

impl CliOptions {
    fn parse(args: &[String], desktop_out_dir: Option<&Path>) -> Result<Self> {
        let mut frame = None;
        let mut catalog = None;
        let mut center_ra = None;
        let mut center_dec = None;
        let mut pixel_scale_arcsec = None;
        let mut epoch = None;
        let mut threshold_sigma = 8.0;
        let mut min_distance = 4;
        let mut aperture_radius = 4;
        let mut psf_fwhm = 2.0;
        let mut max_detections = 4000;
        let mut solver = PlateSolveOptions {
            scale_tolerance: 0.04,
            ..PlateSolveOptions::default()
        };
        let mut output = None;
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            if arg == "--help" || arg == "-h" {
                return Err("__help__".into());
            }
            if !arg.starts_with("--") {
                if frame.replace(PathBuf::from(arg)).is_some() {
                    return Err("only one FITS path is accepted".into());
                }
                index += 1;
                continue;
            }
            let (name, inline) = arg
                .split_once('=')
                .map(|(name, value)| (name, Some(value.to_owned())))
                .unwrap_or((arg.as_str(), None));
            let value = if let Some(value) = inline {
                value
            } else {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| format!("{name} requires a value"))?
            };
            match name {
                "--catalog" => catalog = Some(PathBuf::from(value)),
                "--center-ra" => center_ra = Some(parse_value(name, value)?),
                "--center-dec" => center_dec = Some(parse_value(name, value)?),
                "--pixel-scale-arcsec" => pixel_scale_arcsec = Some(parse_value(name, value)?),
                "--epoch" => epoch = Some(parse_value(name, value)?),
                "--threshold-sigma" => threshold_sigma = parse_value(name, value)?,
                "--min-distance" => min_distance = parse_value(name, value)?,
                "--aperture-radius" => aperture_radius = parse_value(name, value)?,
                "--psf-fwhm" => psf_fwhm = parse_value(name, value)?,
                "--max-detections" => max_detections = parse_value(name, value)?,
                "--max-image-points" => solver.max_image_points = parse_value(name, value)?,
                "--max-catalog-points" => solver.max_catalog_points = parse_value(name, value)?,
                "--scale-tolerance" => solver.scale_tolerance = parse_value(name, value)?,
                "--match-radius-px" => solver.match_radius_px = parse_value(name, value)?,
                "--min-matches" => solver.min_matches = parse_value(name, value)?,
                "--min-coverage-area" => solver.min_coverage_area = parse_value(name, value)?,
                "--max-rms-px" => solver.max_rms_residual_px = parse_value(name, value)?,
                "--out" => output = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown argument {arg}")),
            }
            index += 1;
        }
        let frame = frame.ok_or("a FITS path is required")?;
        let catalog = catalog.ok_or("--catalog is required")?;
        let output = output
            .or_else(|| desktop_out_dir.map(|dir| dir.join("plate-solution.json")))
            .ok_or("--out is required")?;
        if !frame.is_file() {
            return Err(format!("FITS does not exist: {}", frame.display()));
        }
        if !catalog.is_file() {
            return Err(format!("catalog does not exist: {}", catalog.display()));
        }
        let pixel_scale_arcsec = pixel_scale_arcsec.ok_or("--pixel-scale-arcsec is required")?;
        let mut solver = solver;
        solver.epoch = epoch;
        Ok(Self {
            frame,
            catalog,
            center_ra: center_ra.ok_or("--center-ra is required")?,
            center_dec: center_dec.ok_or("--center-dec is required")?,
            pixel_scale_arcsec,
            epoch,
            threshold_sigma,
            min_distance,
            aperture_radius,
            psf_fwhm,
            max_detections,
            solver,
            output,
        })
    }
}

fn solve_files(options: &CliOptions) -> Result<(Value, bool)> {
    let frame = fits::read_fits(&options.frame, false)?;
    let catalog = catalog_csv::load_catalog_csv(&options.catalog)?;
    let reference_wcs = TangentPlaneWcs {
        center_ra_deg: options.center_ra,
        center_dec_deg: options.center_dec,
        pixel_scale_arcsec: options.pixel_scale_arcsec,
        crpix_x: (frame.width - 1) as f64 / 2.0,
        crpix_y: (frame.height - 1) as f64 / 2.0,
        rotation_deg: 0.0,
        parity: 1,
    }
    .validated()?;
    let mut detect_options = DetectionOptions::default();
    detect_options.threshold_sigma = options.threshold_sigma;
    detect_options.min_distance = options.min_distance;
    detect_options.max_sources = Some(options.max_detections);
    detect_options.proposal_mode = "hybrid".into();
    detect_options.measurement.aperture_radius = options.aperture_radius;
    detect_options.measurement.psf_fwhm = options.psf_fwhm;
    detect_options.measurement.min_flux_snr = 5.0;
    let mask = fits::auxiliary_mask(frame.height, frame.width)?;
    let detection = detection::detect_sources(
        &frame.data,
        Some(&mask),
        frame.height,
        frame.width,
        &detect_options,
        None,
    )?;
    let result = solve_plate(
        &detection.sources,
        &catalog,
        &reference_wcs,
        PlateSolveOptions {
            epoch: options.epoch,
            image_shape: Some((frame.height, frame.width)),
            ..options.solver
        },
    )?;
    let payload = json!({
        "frame": frame.path.to_string_lossy(),
        "catalog": options.catalog.to_string_lossy(),
        "detection": {
            "candidate_count": detection.candidate_count,
            "returned_count": detection.sources.len(),
            "quality_count": detection.quality_count
        },
        "reference_wcs": {
            "center_ra_deg": reference_wcs.center_ra_deg,
            "center_dec_deg": reference_wcs.center_dec_deg,
            "pixel_scale_arcsec": reference_wcs.pixel_scale_arcsec,
            "crpix_x": reference_wcs.crpix_x,
            "crpix_y": reference_wcs.crpix_y,
            "rotation_deg": reference_wcs.rotation_deg,
            "parity": reference_wcs.parity
        },
        "plate_solution": result
    });
    if let Some(parent) = options
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut bytes = serde_json::to_vec_pretty(&payload).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    fs::write(&options.output, bytes)
        .map_err(|error| format!("cannot write {}: {error}", options.output.display()))?;
    eprintln!(
        "plate-solver: {}; {}; output={}",
        payload["plate_solution"]["status"]
            .as_str()
            .unwrap_or("ERROR"),
        payload["plate_solution"]["reason"].as_str().unwrap_or(""),
        options.output.display()
    );
    let valid = payload["plate_solution"]["valid"] == true;
    Ok((payload, valid))
}

pub fn run_desktop(args: &[String], out_dir: &Path) -> Result<Value> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let options = CliOptions::parse(args, Some(out_dir))?;
    solve_files(&options).map(|(payload, _)| payload)
}

const HELP: &str = "用检测点与本地 Gaia/星表 CSV 的星对几何候选估计图像 WCS\n\nUSAGE: rst19-plate-solver FRAME --catalog CSV --center-ra DEG --center-dec DEG --pixel-scale-arcsec ARCSEC --out JSON [OPTIONS]\n\nOptions:\n  --epoch YEAR\n  --threshold-sigma SIGMA    [default: 8]\n  --min-distance PX          [default: 4]\n  --aperture-radius PX       [default: 4]\n  --psf-fwhm PX              [default: 2]\n  --max-detections N         [default: 4000]\n  --max-image-points N       [default: 80]\n  --max-catalog-points N     [default: 160]\n  --scale-tolerance FRACTION [default: 0.04]\n  --match-radius-px PX       [default: 3]\n  --min-matches N            [default: 8]\n  --min-coverage-area FRACTION [default: 0.02]\n  --max-rms-px PX            [default: 2]\n  -h, --help";

pub fn run_cli(args: &[String]) -> i32 {
    let options = match CliOptions::parse(args, None) {
        Ok(options) => options,
        Err(error) if error == "__help__" => {
            println!("{HELP}");
            return 0;
        }
        Err(error) => {
            eprintln!("plate-solver: {error}\n\n{HELP}");
            return 2;
        }
    };
    match solve_files(&options) {
        Ok((_, true)) => 0,
        Ok((_, false)) => 3,
        Err(error) => {
            eprintln!("plate-solver: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use std::{fs, path::Path};

    fn detection(id: i64, x: f64, y: f64, snr: f64, flags: Vec<String>) -> Detection {
        Detection {
            detection_id: id,
            x,
            y,
            peak: 1000.0,
            flux: 5000.0,
            background: 10.0,
            noise: 2.0,
            snr,
            fwhm: Some(2.0),
            flags,
            flux_error: Some(50.0),
            flux_snr: Some(snr),
            filter_snr: Some(snr),
            fwhm_x: Some(2.0),
            fwhm_y: Some(2.0),
            ellipticity: Some(0.0),
            sharpness: Some(0.2),
            footprint_pixels: Some(20),
            psf_support_pixels: Some(10),
            quality_passed: true,
            peak_x: Some(x),
            peak_y: Some(y),
            centroid_shift_px: Some(0.0),
            proposal_methods: vec!["gaussian".into()],
            proposal_scales: vec![1.0],
            proposal_snr: Some(snr),
            nearest_gaussian_px: Some(0.0),
            deblend_delta_bic: None,
            deblend_component_snr: None,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: vec![],
        }
    }

    fn source(reference: &TangentPlaneWcs, id: usize, east: f64, north: f64) -> CatalogSource {
        let ra = reference.crpix_x + east / reference.pixel_scale_arcsec;
        let dec = reference.crpix_y + north / reference.pixel_scale_arcsec;
        let (ra_deg, dec_deg) = reference.pixel_to_world(ra, dec);
        serde_json::from_value(json!({
            "source_id": format!("gaia-{id}"),
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "magnitude": 9.0 + id as f64 * 0.2,
            "color": 0.5 + id as f64 * 0.02,
            "color_name": "BP-RP",
            "catalog_name": "Gaia DR3",
            "photometric_system": "Gaia Vega",
            "photometric_band": "G"
        }))
        .unwrap()
    }

    fn fits_card(key: &str, value: &str) -> [u8; 80] {
        let mut card = [b' '; 80];
        let text = format!("{key:<8}= {value:>20}");
        card[..text.len()].copy_from_slice(text.as_bytes());
        card
    }

    fn write_star_field_fits(path: &Path, centers: &[[f64; 2]]) {
        const WIDTH: usize = 512;
        const HEIGHT: usize = 512;
        let mut bytes = Vec::new();
        for (key, value) in [
            ("SIMPLE", "T".to_owned()),
            ("BITPIX", "16".to_owned()),
            ("NAXIS", "2".to_owned()),
            ("NAXIS1", WIDTH.to_string()),
            ("NAXIS2", HEIGHT.to_string()),
        ] {
            bytes.extend_from_slice(&fits_card(key, &value));
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let mut value = 1000.0 + ((x * 17 + y * 31) % 11) as f64 * 2.0 - 10.0;
                for center in centers {
                    let dx = x as f64 - center[0];
                    let dy = y as f64 - center[1];
                    value += 2400.0 * (-0.5 * (dx * dx + dy * dy) / 1.3_f64.powi(2)).exp();
                }
                bytes.extend_from_slice(&(value.round() as i16).to_be_bytes());
            }
        }
        bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn solves_rotation_reflection_offset_and_leave_one_out() {
        let reference = TangentPlaneWcs {
            center_ra_deg: 129.5,
            center_dec_deg: -1.8,
            pixel_scale_arcsec: 8.5,
            crpix_x: 2000.0,
            crpix_y: 2000.0,
            rotation_deg: 0.0,
            parity: 1,
        };
        let points = [
            [-9000.0, -6500.0],
            [-6500.0, 8500.0],
            [-4200.0, 2200.0],
            [-1000.0, -9200.0],
            [1800.0, 7200.0],
            [4300.0, -3000.0],
            [6800.0, 6400.0],
            [9100.0, -7100.0],
            [10500.0, 1800.0],
            [11500.0, 9000.0],
        ];
        let theta = 27.0 * PI / 180.0;
        let scale = 1.0 / 8.5;
        let matrix = [
            [scale * theta.cos(), scale * theta.sin()],
            [scale * theta.sin(), -scale * theta.cos()],
        ];
        let transform = transform_from_matrix(matrix, [2100.0, 1840.0]).unwrap();
        let catalog: Vec<_> = points
            .iter()
            .enumerate()
            .map(|(index, point)| source(&reference, index, point[0], point[1]))
            .collect();
        let detections: Vec<_> = points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let predicted = transform.project(*point);
                detection(
                    index as i64 + 1,
                    predicted[0] + if index % 2 == 0 { -0.06 } else { 0.08 },
                    predicted[1] + if index % 3 == 0 { -0.04 } else { 0.05 },
                    100.0,
                    vec![],
                )
            })
            .chain((0..8).map(|index| {
                detection(
                    100 + index,
                    300.0 + index as f64 * 17.0,
                    3200.0 - index as f64 * 23.0,
                    8.0,
                    vec![],
                )
            }))
            .collect();
        let result = solve_plate(
            &detections,
            &catalog,
            &reference,
            PlateSolveOptions {
                image_shape: Some((4000, 4000)),
                scale_tolerance: 0.04,
                min_pair_distance_px: 40.0,
                match_radius_px: 1.0,
                min_matches: 8,
                min_coverage_area: 0.01,
                max_rms_residual_px: 0.5,
                max_leave_one_out_rms_px: 1.0,
                max_image_points: 24,
                max_catalog_points: 24,
                ..PlateSolveOptions::default()
            },
        )
        .unwrap();
        assert!(
            result.valid,
            "{}",
            serde_json::to_string_pretty(&result).unwrap()
        );
        let best = result.best.unwrap();
        assert_eq!(best.matched_count, 10);
        assert_eq!(best.transform.parity, -1);
        assert!((best.transform.plate_scale_arcsec_per_pixel - 8.5).abs() < 0.02);
        assert!(best.rms_residual_px < 0.5);
        assert!(best.leave_one_out_rms_residual_px.unwrap() < 1.0);
        assert!(best.coverage_area > 0.01);
    }

    #[test]
    fn excludes_rejected_or_structural_detections_and_requires_loo() {
        let reference = TangentPlaneWcs {
            center_ra_deg: 80.0,
            center_dec_deg: -10.0,
            pixel_scale_arcsec: 8.5,
            crpix_x: 100.0,
            crpix_y: 100.0,
            rotation_deg: 0.0,
            parity: 1,
        };
        let points = [[-500.0, -400.0], [-100.0, 300.0], [400.0, -250.0]];
        let catalog: Vec<_> = points
            .iter()
            .enumerate()
            .map(|(index, point)| source(&reference, index, point[0], point[1]))
            .collect();
        let detections: Vec<_> = points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                detection(
                    index as i64 + 1,
                    100.0 + point[0] / 8.5,
                    100.0 + point[1] / 8.5,
                    100.0,
                    vec![],
                )
            })
            .chain([detection(
                99,
                50.0,
                50.0,
                1000.0,
                vec!["LINE_ARTIFACT".into()],
            )])
            .collect();
        let result = solve_plate(
            &detections,
            &catalog,
            &reference,
            PlateSolveOptions {
                image_shape: Some((300, 300)),
                min_pair_distance_px: 10.0,
                match_radius_px: 1.0,
                min_matches: 3,
                max_image_points: 4,
                max_catalog_points: 4,
                min_coverage_area: 0.01,
                max_leave_one_out_rms_px: 2.0,
                ..PlateSolveOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.image_points_considered, 3);
        assert!(!result.valid);
        assert_eq!(result.status, "REJECTED");
        assert!(result.best.is_some());
        assert!(result.reason.contains("留一验证无有效样本"));
    }

    #[test]
    fn empty_catalog_geometry_returns_no_solution_without_claiming_success() {
        let reference = TangentPlaneWcs {
            center_ra_deg: 10.0,
            center_dec_deg: 20.0,
            pixel_scale_arcsec: 8.5,
            crpix_x: 1000.0,
            crpix_y: 1000.0,
            rotation_deg: 0.0,
            parity: 1,
        };
        let result = solve_plate(
            &[detection(1, 100.0, 200.0, 20.0, vec![])],
            &[],
            &reference,
            PlateSolveOptions::default(),
        )
        .unwrap();
        assert_eq!(result.status, "INSUFFICIENT_POINTS");
        assert!(!result.valid);
        assert!(result.best.is_none());
    }

    #[test]
    fn native_cli_pipeline_detects_and_solves_a_synthetic_fits_field() {
        let root = std::env::temp_dir().join(format!(
            "rst19-plate-solver-e2e-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let fits_path = root.join("field.fits");
        let catalog_path = root.join("catalog.csv");
        let output_path = root.join("solution.json");
        let reference = TangentPlaneWcs {
            center_ra_deg: 129.5,
            center_dec_deg: -1.8,
            pixel_scale_arcsec: 8.5,
            crpix_x: 255.5,
            crpix_y: 255.5,
            rotation_deg: 0.0,
            parity: 1,
        };
        let points = [
            [-1100.0, -800.0],
            [-850.0, 900.0],
            [-500.0, 220.0],
            [-180.0, -1150.0],
            [220.0, 1000.0],
            [520.0, -350.0],
            [800.0, 750.0],
            [1080.0, -880.0],
            [1250.0, 200.0],
            [1400.0, 1100.0],
        ];
        let theta = 27.0 * PI / 180.0;
        let scale = 1.0 / 8.5;
        let matrix = [
            [scale * theta.cos(), scale * theta.sin()],
            [scale * theta.sin(), -scale * theta.cos()],
        ];
        let transform = transform_from_matrix(matrix, [256.0, 256.0]).unwrap();
        let centers: Vec<_> = points
            .iter()
            .map(|point| transform.project(*point))
            .collect();
        write_star_field_fits(&fits_path, &centers);

        let mut csv = String::from("source_id,ra_deg,dec_deg,magnitude\n");
        for (index, [east, north]) in points.iter().enumerate() {
            let (ra, dec) = reference.pixel_to_world(
                reference.crpix_x + east / reference.pixel_scale_arcsec,
                reference.crpix_y + north / reference.pixel_scale_arcsec,
            );
            csv.push_str(&format!(
                "star-{index},{ra:.12},{dec:.12},{:.2}\n",
                8.0 + index as f64 * 0.2
            ));
        }
        fs::write(&catalog_path, csv).unwrap();
        let args = vec![
            fits_path.to_string_lossy().into_owned(),
            "--catalog".into(),
            catalog_path.to_string_lossy().into_owned(),
            "--center-ra".into(),
            "129.5".into(),
            "--center-dec".into(),
            "-1.8".into(),
            "--pixel-scale-arcsec".into(),
            "8.5".into(),
            "--out".into(),
            output_path.to_string_lossy().into_owned(),
        ];
        let cli = CliOptions::parse(&args, None).unwrap();
        let (payload, valid) = solve_files(&cli).unwrap();
        assert!(valid, "{}", serde_json::to_string_pretty(&payload).unwrap());
        assert!(payload["detection"]["quality_count"].as_u64().unwrap() >= 8);
        assert!(
            payload["plate_solution"]["best"]["matched_count"]
                .as_u64()
                .unwrap()
                >= 8
        );
        assert_eq!(payload["plate_solution"]["best"]["transform"]["parity"], -1);
        assert!(output_path.is_file());
    }
}
