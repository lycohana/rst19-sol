//! Catalog propagation, WCS prediction and unique radius-gated association.
use crate::{
    assignment::{assign_candidates, CandidateEdge},
    catalog::CatalogSource,
    coordinates::{AffineProjection, TangentPlaneWcs},
    spatial::KdTree,
    statistics, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionPosition {
    pub detection_id: i64,
    pub x: f64,
    pub y: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Projection {
    Affine(AffineProjection),
    Tan(TangentPlaneWcs),
}
impl Projection {
    fn project(&self, ra: f64, dec: f64) -> Result<(f64, f64)> {
        match self {
            Self::Affine(w) => w.world_to_pixel(ra, dec),
            Self::Tan(w) => Ok(w.clone().validated()?.world_to_pixel(ra, dec)),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogMatch {
    pub detection_id: i64,
    pub source_id: String,
    pub detection_x: f64,
    pub detection_y: f64,
    pub predicted_x: f64,
    pub predicted_y: f64,
    pub residual_px: f64,
    pub catalog_magnitude: Option<f64>,
    pub catalog_magnitude_error: Option<f64>,
    pub catalog_color: Option<f64>,
    pub catalog_color_name: Option<String>,
    pub photometric_system: Option<String>,
    pub photometric_band: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub matched_count: usize,
    pub unmatched_detection_ids: Vec<i64>,
    pub unmatched_catalog_ids: Vec<String>,
    pub max_residual_px: Option<f64>,
    pub rms_residual_px: Option<f64>,
    pub inlier_ratio: f64,
    pub radius_px: f64,
    pub assignment_mode: String,
    pub matches: Vec<CatalogMatch>,
}
pub fn match_detections(
    detections: &[DetectionPosition],
    catalog: &[CatalogSource],
    wcs: &Projection,
    radius: f64,
    epoch: Option<f64>,
    mode: &str,
) -> Result<MatchResult> {
    if !radius.is_finite() || radius <= 0. || !["global", "greedy"].contains(&mode) {
        return Err("invalid matching radius or assignment mode".into());
    }
    let mut result = MatchResult {
        matched_count: 0,
        unmatched_detection_ids: detections.iter().map(|d| d.detection_id).collect(),
        unmatched_catalog_ids: catalog.iter().map(|s| s.source_id.clone()).collect(),
        max_residual_px: None,
        rms_residual_px: None,
        inlier_ratio: 0.,
        radius_px: radius,
        assignment_mode: mode.into(),
        matches: Vec::new(),
    };
    if detections.is_empty() || catalog.is_empty() {
        return Ok(result);
    }
    let catalog: Vec<_> = catalog
        .iter()
        .map(|s| s.clone().validated()?.at_epoch(epoch))
        .collect::<Result<_>>()?;
    let predicted: Vec<_> = catalog
        .iter()
        .map(|s| wcs.project(s.ra_deg, s.dec_deg))
        .collect::<Result<_>>()?;
    let indices: Vec<_> = predicted
        .iter()
        .enumerate()
        .filter(|(_, p)| p.0.is_finite() && p.1.is_finite())
        .map(|(i, _)| i)
        .collect();
    if indices.is_empty() {
        return Ok(result);
    }
    let points: Vec<_> = indices
        .iter()
        .map(|&i| [predicted[i].0, predicted[i].1])
        .collect();
    let tree = KdTree::new(&points)?;
    let mut candidates = Vec::new();
    for (i, d) in detections.iter().enumerate() {
        for (local, residual) in tree.within_radius([d.x, d.y], radius)? {
            candidates.push(CandidateEdge(residual, i, indices[local]));
        }
    }
    let assigned = assign_candidates(&candidates, detections.len(), catalog.len(), radius, mode)?;
    let mut used_d = BTreeSet::new();
    let mut used_c = BTreeSet::new();
    let mut residuals = Vec::new();
    for CandidateEdge(residual, i, j) in assigned {
        let d = &detections[i];
        let s = &catalog[j];
        used_d.insert(i);
        used_c.insert(j);
        residuals.push(residual);
        result.matches.push(CatalogMatch {
            detection_id: d.detection_id,
            source_id: s.source_id.clone(),
            detection_x: d.x,
            detection_y: d.y,
            predicted_x: predicted[j].0,
            predicted_y: predicted[j].1,
            residual_px: residual,
            catalog_magnitude: s.magnitude,
            catalog_magnitude_error: s.magnitude_error,
            catalog_color: s.color,
            catalog_color_name: s.color_name.clone(),
            photometric_system: s.photometric_system.clone(),
            photometric_band: s.photometric_band.clone(),
        });
    }
    result.matches.sort_by_key(|m| m.detection_id);
    result.matched_count = result.matches.len();
    result.unmatched_detection_ids = detections
        .iter()
        .enumerate()
        .filter(|(i, _)| !used_d.contains(i))
        .map(|(_, d)| d.detection_id)
        .collect();
    result.unmatched_catalog_ids = catalog
        .iter()
        .enumerate()
        .filter(|(i, _)| !used_c.contains(i))
        .map(|(_, s)| s.source_id.clone())
        .collect();
    if !residuals.is_empty() {
        result.max_residual_px = Some(residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        result.rms_residual_px = Some(
            (statistics::pairwise_sum(&residuals.iter().map(|r| r * r).collect::<Vec<_>>())
                / residuals.len() as f64)
                .sqrt(),
        );
    }
    result.inlier_ratio = result.matched_count as f64 / detections.len() as f64;
    Ok(result)
}
