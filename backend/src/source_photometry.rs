//! Per-source photometry evidence and scope-aware faintest-source selection.
use crate::{
    catalog::CatalogSource,
    matching::CatalogMatch,
    measurement::Detection,
    photometric_calibration::PhotometricCalibration,
    photometry::{
        catalog_estimate, instrumental_magnitude, metadata_matches, normalise_label,
        AbsoluteMagnitudeEstimate, CatalogAbsoluteOptions,
    },
    Result,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SourcePhotometry {
    pub detection_id: i64,
    pub source_id: Option<String>,
    pub instrumental_magnitude: Option<f64>,
    pub instrumental_magnitude_error: Option<f64>,
    pub calibrated_magnitude: Option<f64>,
    pub calibrated_magnitude_error: Option<f64>,
    pub catalog_magnitude: Option<f64>,
    pub catalog_magnitude_error: Option<f64>,
    pub color: Option<f64>,
    pub color_name: Option<String>,
    pub photometric_system: Option<String>,
    pub photometric_band: Option<String>,
    pub absolute_magnitude: Option<AbsoluteMagnitudeEstimate>,
    pub status: String,
    pub magnitude_source: Option<String>,
    pub flags: Vec<String>,
    pub photometric_residual_mag: Option<f64>,
    pub photometric_residual_limit_mag: Option<f64>,
    pub photometric_consistent: Option<bool>,
    pub photometric_outlier_reason: Option<String>,
    pub calibration_sample_role: Option<String>,
    pub catalog_mg_gspphot: Option<f64>,
    pub catalog_mg_gspphot_lower: Option<f64>,
    pub catalog_mg_gspphot_upper: Option<f64>,
    pub catalog_mg_gspphot_source: Option<String>,
}
impl SourcePhotometry {
    pub fn as_dict(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("source photometry serialization");
        value["absolute_magnitude"] = self
            .absolute_magnitude
            .as_ref()
            .map(AbsoluteMagnitudeEstimate::as_dict)
            .unwrap_or(Value::Null);
        value
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TableOptions {
    pub exposure_s: f64,
    pub parallax_zero_point_mas: f64,
    pub max_fractional_parallax_error: f64,
}
impl Default for TableOptions {
    fn default() -> Self {
        Self {
            exposure_s: 1.,
            parallax_zero_point_mas: 0.,
            max_fractional_parallax_error: 0.2,
        }
    }
}
fn truthy(value: Option<String>, fallback: Option<String>) -> Option<String> {
    value.filter(|s| !s.is_empty()).or(fallback)
}
pub fn build_table(
    sources: &[Detection],
    matches: &[CatalogMatch],
    catalog: &[CatalogSource],
    calibration: Option<&PhotometricCalibration>,
    o: &TableOptions,
) -> Result<Vec<SourcePhotometry>> {
    if !o.exposure_s.is_finite()
        || o.exposure_s <= 0.
        || !o.parallax_zero_point_mas.is_finite()
        || !o.max_fractional_parallax_error.is_finite()
        || o.max_fractional_parallax_error <= 0.
    {
        return Err("invalid source photometry options".into());
    }
    let catalogs: BTreeMap<_, _> = catalog.iter().map(|s| (&s.source_id, s)).collect();
    let matches: BTreeMap<_, _> = matches.iter().map(|m| (m.detection_id, m)).collect();
    let roles: BTreeMap<_, _> = calibration
        .map(|c| {
            c.calibration_sample_roles
                .iter()
                .map(|(id, role)| (id, role))
                .collect()
        })
        .unwrap_or_default();
    let mut rows = Vec::new();
    for d in sources {
        let snr = d.flux_snr.unwrap_or(d.snr);
        let instrumental = instrumental_magnitude(d.flux, o.exposure_s)?;
        let mut row = SourcePhotometry {
            detection_id: d.detection_id,
            instrumental_magnitude: instrumental,
            instrumental_magnitude_error: (instrumental.is_some() && snr.is_finite() && snr > 0.)
                .then(|| 1.0857362047581296 / snr),
            flags: d.flags.clone(),
            ..Default::default()
        };
        let m = matches.get(&d.detection_id).copied();
        let s = m.and_then(|m| catalogs.get(&m.source_id).copied());
        if let Some(s) = s {
            row.catalog_magnitude = s.magnitude;
            row.catalog_magnitude_error = s.magnitude_error;
            row.color = s.color;
            row.color_name = s.color_name.clone();
            row.photometric_system = s.photometric_system.clone();
            row.photometric_band = s.photometric_band.clone();
            row.magnitude_source = Some(s.magnitude_source.clone());
            row.catalog_mg_gspphot = s.mg_gspphot;
            row.catalog_mg_gspphot_lower = s.mg_gspphot_lower;
            row.catalog_mg_gspphot_upper = s.mg_gspphot_upper;
            row.catalog_mg_gspphot_source = s.mg_gspphot_source.clone();
        }
        if let Some(m) = m {
            row.source_id = Some(m.source_id.clone());
            row.catalog_magnitude = row.catalog_magnitude.or(m.catalog_magnitude);
            row.catalog_magnitude_error = row.catalog_magnitude_error.or(m.catalog_magnitude_error);
            row.color = row.color.or(m.catalog_color);
            row.color_name = truthy(row.color_name, m.catalog_color_name.clone());
            row.photometric_system = truthy(row.photometric_system, m.photometric_system.clone());
            row.photometric_band = truthy(row.photometric_band, m.photometric_band.clone());
            if calibration.is_some() {
                row.calibration_sample_role = roles.get(&m.source_id).map(|r| (*r).clone());
            }
        }
        if row.catalog_magnitude.is_some() && row.catalog_magnitude_error.is_none() {
            row.flags
                .push("CATALOG_MAGNITUDE_ERROR_NOT_PROVIDED".into());
        }
        if !d.quality_passed {
            row.status = "REJECTED_QUALITY".into();
            row.flags.push("QUALITY_REJECTED".into());
        } else if instrumental.is_none() {
            row.status = "NO_POSITIVE_FLUX".into();
            row.flags.push("NO_POSITIVE_FLUX".into());
        } else if m.is_none() {
            row.status = "INSTRUMENTAL_ONLY".into();
            row.flags.push("NO_CATALOG_MATCH".into());
        } else if calibration.is_none() {
            row.status = "CATALOG_MATCH_NO_CALIBRATION".into();
            row.flags.push("NO_PHOTOMETRIC_CALIBRATION".into());
        } else {
            let c = calibration.unwrap();
            let reference_system = s
                .map(|s| s.photometric_system.as_deref())
                .unwrap_or(row.photometric_system.as_deref());
            let reference_band = s
                .map(|s| s.photometric_band.as_deref())
                .unwrap_or(row.photometric_band.as_deref());
            let reference_color = s
                .map(|s| s.color_name.as_deref())
                .unwrap_or(row.color_name.as_deref());
            let metadata = metadata_matches(
                c.photometric_system.as_deref(),
                c.photometric_band.as_deref(),
                reference_system,
                reference_band,
            );
            let color_metadata = c.color_order == 0
                || (normalise_label(c.color_name.as_deref()).is_some()
                    && normalise_label(reference_color).is_some()
                    && normalise_label(c.color_name.as_deref())
                        == normalise_label(reference_color));
            if metadata && color_metadata {
                row.calibrated_magnitude = c.apply(instrumental.unwrap(), row.color);
            }
            if !metadata {
                row.status = "CALIBRATION_METADATA_MISMATCH".into();
            } else if c.color_order > 0 && row.color.is_none() {
                row.status = "CALIBRATION_MISSING_COLOR".into();
            } else if !color_metadata {
                row.status = "CALIBRATION_COLOR_METADATA_MISMATCH".into();
            } else if row.calibrated_magnitude.is_none() {
                row.status = "CALIBRATION_INVALID".into();
            } else {
                row.photometric_system = c.photometric_system.clone();
                row.photometric_band = c.photometric_band.clone();
                row.calibrated_magnitude_error =
                    c.magnitude_error(row.instrumental_magnitude_error, None);
                row.photometric_residual_limit_mag = c.max_source_residual_mag;
                if let Some(magnitude) = row.catalog_magnitude.filter(|v| v.is_finite()) {
                    let residual = magnitude - row.calibrated_magnitude.unwrap();
                    row.photometric_residual_mag = Some(residual);
                    let consistent = row
                        .photometric_residual_limit_mag
                        .is_none_or(|limit| residual.abs() <= limit);
                    row.photometric_consistent = Some(consistent);
                    if !consistent {
                        row.status = "CATALOG_INCONSISTENT".into();
                        row.photometric_outlier_reason = Some("RESIDUAL_EXCEEDS_LIMIT".into());
                        row.flags.extend([
                            "PHOTOMETRIC_OUTLIER".into(),
                            "CALIBRATION_NOT_APPLIED_TO_SELECTION".into(),
                        ]);
                    } else {
                        row.status = "CALIBRATED".into();
                    }
                } else {
                    row.status = "CALIBRATED".into();
                }
                if let Some(s) = s.filter(|_| row.photometric_consistent != Some(false)) {
                    let absolute = catalog_estimate(
                        s,
                        &CatalogAbsoluteOptions {
                            apparent_magnitude: row.calibrated_magnitude,
                            apparent_magnitude_error: row.calibrated_magnitude_error,
                            required_photometric_system: c.photometric_system.clone(),
                            required_photometric_band: c.photometric_band.clone(),
                            parallax_zero_point_mas: o.parallax_zero_point_mas,
                            max_fractional_parallax_error: o.max_fractional_parallax_error,
                        },
                    )?;
                    row.flags.extend(absolute.flags.iter().cloned());
                    row.absolute_magnitude = Some(absolute);
                }
            }
            if row.calibrated_magnitude.is_none() {
                row.flags.push("CALIBRATION_NOT_APPLIED".into());
            }
        }
        let mut unique = Vec::new();
        for flag in row.flags {
            if !unique.contains(&flag) {
                unique.push(flag);
            }
        }
        row.flags = unique;
        rows.push(row);
    }
    Ok(rows)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaintestSource {
    pub detection_id: i64,
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    pub snr: f64,
    pub instrumental_magnitude: f64,
    pub calibrated_magnitude: Option<f64>,
    pub flags: Vec<String>,
    pub flux_snr: Option<f64>,
    pub flux_rate: Option<f64>,
    pub instrumental_magnitude_error: Option<f64>,
    pub photometric_system: Option<String>,
    pub photometric_band: Option<String>,
    pub calibration_status: String,
    pub calibrated_magnitude_error: Option<f64>,
    pub absolute_magnitude: Option<AbsoluteMagnitudeEstimate>,
    pub selection_scope: String,
    pub eligible_candidate_count: usize,
    pub calibrated_candidate_count: usize,
}
impl FaintestSource {
    pub fn as_dict(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("faintest serialization");
        value["absolute_magnitude"] = self
            .absolute_magnitude
            .as_ref()
            .map(AbsoluteMagnitudeEstimate::as_dict)
            .unwrap_or(Value::Null);
        value
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FaintestOptions {
    pub min_snr: f64,
    pub zero_point: Option<f64>,
    pub exposure_s: f64,
    pub source_colors: Option<BTreeMap<i64, f64>>,
    pub source_absolute_magnitudes: Option<BTreeMap<i64, AbsoluteMagnitudeEstimate>>,
    pub source_photometry: Option<BTreeMap<i64, SourcePhotometry>>,
}
impl Default for FaintestOptions {
    fn default() -> Self {
        Self {
            min_snr: 5.,
            zero_point: None,
            exposure_s: 1.,
            source_colors: None,
            source_absolute_magnitudes: None,
            source_photometry: None,
        }
    }
}
pub fn find_faintest(
    sources: &[Detection],
    calibration: Option<&PhotometricCalibration>,
    o: &FaintestOptions,
) -> Result<Option<FaintestSource>> {
    if !o.exposure_s.is_finite() || o.exposure_s <= 0. {
        return Err("exposure_s must be positive".into());
    }
    let mut candidates = Vec::new();
    for d in sources {
        let snr = d.flux_snr.unwrap_or(d.snr);
        if !snr.is_finite()
            || snr < o.min_snr
            || d.flux <= 0.
            || !d.quality_passed
            || d.flags
                .iter()
                .any(|f| ["EDGE", "MASKED", "SATURATED"].contains(&f.as_str()))
        {
            continue;
        }
        let Some(magnitude) = instrumental_magnitude(d.flux, o.exposure_s)? else {
            continue;
        };
        let mut calibrated = o.zero_point.map(|z| magnitude + z);
        if let Some(c) = calibration {
            calibrated = c.apply(
                magnitude,
                o.source_colors
                    .as_ref()
                    .and_then(|m| m.get(&d.detection_id).copied()),
            );
            if let Some(rows) = &o.source_photometry {
                if rows.get(&d.detection_id).is_none_or(|r| {
                    r.status != "CALIBRATED" || r.photometric_consistent != Some(true)
                }) {
                    calibrated = None;
                }
            }
        }
        candidates.push((magnitude, calibrated, d));
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    let calibrated_count = candidates.iter().filter(|(_, c, _)| c.is_some()).count();
    let mut selected = None;
    for candidate in &candidates {
        if calibrated_count > 0 && candidate.1.is_none() {
            continue;
        }
        let score = if calibrated_count > 0 {
            candidate.1.unwrap()
        } else {
            candidate.0
        };
        if selected.is_none_or(|(old, old_d): (f64, &Detection)| {
            score > old || (score == old && candidate.2.detection_id > old_d.detection_id)
        }) {
            selected = Some((score, candidate.2));
        }
    }
    let source = selected.unwrap().1;
    let (magnitude, calibrated, _) = candidates
        .iter()
        .find(|(_, _, d)| std::ptr::eq(*d, source))
        .unwrap();
    let snr = source.flux_snr.unwrap_or(source.snr);
    let magnitude_error = (snr > 0.).then(|| 1.0857362047581296 / snr);
    let scope = if calibrated_count > 0 {
        if calibration.is_some() {
            if calibrated_count == candidates.len() {
                "CALIBRATED_MATCHES"
            } else {
                "CALIBRATED_MATCHES_PARTIAL"
            }
        } else {
            "ZERO_POINT_ADJUSTED"
        }
    } else if calibration.is_some() {
        "QUALITY_DETECTIONS_NO_USABLE_CALIBRATION"
    } else {
        "QUALITY_DETECTIONS"
    };
    Ok(Some(FaintestSource {
        detection_id: source.detection_id,
        x: source.x,
        y: source.y,
        flux: source.flux,
        snr: source.snr,
        instrumental_magnitude: *magnitude,
        calibrated_magnitude: *calibrated,
        flags: source.flags.clone(),
        flux_snr: source.flux_snr,
        flux_rate: Some(source.flux / o.exposure_s),
        instrumental_magnitude_error: magnitude_error,
        photometric_system: calibration.and_then(|c| c.photometric_system.clone()),
        photometric_band: calibration.and_then(|c| c.photometric_band.clone()),
        calibration_status: calibration
            .map(|c| c.status.clone())
            .unwrap_or("INSTRUMENTAL".into()),
        calibrated_magnitude_error: if calibrated.is_some() {
            calibration.and_then(|c| c.magnitude_error(magnitude_error, None))
        } else {
            None
        },
        absolute_magnitude: o
            .source_absolute_magnitudes
            .as_ref()
            .and_then(|m| m.get(&source.detection_id).cloned()),
        selection_scope: scope.into(),
        eligible_candidate_count: candidates.len(),
        calibrated_candidate_count: calibrated_count,
    }))
}
