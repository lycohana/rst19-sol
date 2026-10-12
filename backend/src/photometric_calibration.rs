//! Weighted zero point/color calibration with deterministic holdout and gates.
use crate::{
    catalog::CatalogSource,
    matching::CatalogMatch,
    measurement::Detection,
    photometry::{instrumental_magnitude, metadata_matches, normalise_label},
    statistics, Result,
};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

fn read_counts<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<(String, usize)>, D::Error> {
    #[derive(Deserialize)]
    struct Count {
        reason: String,
        count: usize,
    }
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Input {
        Pairs(Vec<(String, usize)>),
        Objects(Vec<Count>),
    }
    Ok(match Input::deserialize(d)? {
        Input::Pairs(v) => v,
        Input::Objects(v) => v.into_iter().map(|c| (c.reason, c.count)).collect(),
    })
}
fn read_roles<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<(String, String)>, D::Error> {
    #[derive(Deserialize)]
    struct Role {
        source_id: String,
        role: String,
    }
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Input {
        Pairs(Vec<(String, String)>),
        Objects(Vec<Role>),
    }
    Ok(match Input::deserialize(d)? {
        Input::Pairs(v) => v,
        Input::Objects(v) => v.into_iter().map(|c| (c.source_id, c.role)).collect(),
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotometricCalibration {
    pub photometric_system: Option<String>,
    pub photometric_band: Option<String>,
    pub color_name: Option<String>,
    pub color_order: usize,
    pub coefficients: Vec<f64>,
    pub calibrator_count: usize,
    pub inlier_count: usize,
    pub validation_count: usize,
    pub fit_rms_mag: Option<f64>,
    pub validation_rms_mag: Option<f64>,
    pub residual_mad_mag: Option<f64>,
    pub color_min: Option<f64>,
    pub color_max: Option<f64>,
    pub status: String,
    pub flags: Vec<String>,
    #[serde(default, deserialize_with = "read_counts")]
    pub catalog_filter_counts: Vec<(String, usize)>,
    pub validation_bias_mag: Option<f64>,
    pub max_fit_rms_mag: Option<f64>,
    pub max_validation_rms_mag: Option<f64>,
    pub max_residual_mad_mag: Option<f64>,
    pub max_validation_bias_mag: Option<f64>,
    pub min_validation_count: usize,
    pub max_source_residual_mag: Option<f64>,
    #[serde(default, deserialize_with = "read_roles")]
    pub calibration_sample_roles: Vec<(String, String)>,
}
impl PhotometricCalibration {
    pub fn apply(&self, instrumental: f64, color: Option<f64>) -> Option<f64> {
        if !["VALID", "VALID_NO_HOLDOUT"].contains(&self.status.as_str())
            || self.coefficients.is_empty()
            || normalise_label(self.photometric_system.as_deref()).is_none()
            || normalise_label(self.photometric_band.as_deref()).is_none()
            || !instrumental.is_finite()
        {
            return None;
        }
        let mut basis = vec![1.];
        if self.color_order > 0 {
            let c = color?;
            let (lo, hi) = (self.color_min?, self.color_max?);
            if !c.is_finite() || c < lo || c > hi {
                return None;
            }
            basis.push(c);
            if self.color_order >= 2 {
                basis.push(c * c);
            }
        }
        let coefficient_count = self.color_order.checked_add(1)?;
        let coefficients = self.coefficients.get(..coefficient_count)?;
        if basis.len() != coefficients.len() || coefficients.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(
            instrumental
                + basis
                    .iter()
                    .zip(coefficients)
                    .map(|(a, b)| a * b)
                    .sum::<f64>(),
        )
    }
    pub fn magnitude_error(&self, instrumental: Option<f64>, color: Option<f64>) -> Option<f64> {
        let error = instrumental.filter(|v| v.is_finite())?;
        let mut variance = error * error;
        if let Some(v) = self.residual_mad_mag {
            variance += v * v;
        }
        if self.color_order > 0 {
            if let Some((error, coefficient)) = color
                .filter(|v| v.is_finite())
                .zip(self.coefficients.get(1))
            {
                variance += (error * coefficient).powi(2);
            }
        }
        Some(variance.max(0.).sqrt())
    }
    pub fn as_dict(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("calibration serialization");
        let object = value.as_object_mut().unwrap();
        object.insert("zero_point".into(), json!(self.coefficients.first()));
        object.insert("color_coefficient".into(), json!(self.coefficients.get(1)));
        object.insert(
            "catalog_filter_counts".into(),
            json!(self
                .catalog_filter_counts
                .iter()
                .map(|(reason, count)| json!({"reason":reason,"count":count}))
                .collect::<Vec<_>>()),
        );
        object.insert(
            "calibration_sample_roles".into(),
            json!(self
                .calibration_sample_roles
                .iter()
                .map(|(source_id, role)| json!({"source_id":source_id,"role":role}))
                .collect::<Vec<_>>()),
        );
        value
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CalibrationOptions {
    pub exposure_s: f64,
    pub photometric_system: Option<String>,
    pub photometric_band: Option<String>,
    pub color_name: Option<String>,
    pub color_order: usize,
    pub min_snr: f64,
    pub min_calibrators: usize,
    pub max_match_residual_px: Option<f64>,
    pub min_catalog_flux_snr: Option<f64>,
    pub max_catalog_ruwe: Option<f64>,
    pub min_catalog_visibility_periods: Option<u64>,
    pub reject_duplicated_sources: bool,
    pub reject_variable_sources: bool,
    pub sigma_clip: f64,
    pub max_iterations: usize,
    pub validation_fraction: f64,
    pub max_fit_rms_mag: Option<f64>,
    pub max_validation_rms_mag: Option<f64>,
    pub max_residual_mad_mag: Option<f64>,
    pub max_validation_bias_mag: Option<f64>,
    pub min_validation_count: usize,
    pub max_source_residual_mag: Option<f64>,
}
impl Default for CalibrationOptions {
    fn default() -> Self {
        Self {
            exposure_s: 1.,
            photometric_system: Some("unknown".into()),
            photometric_band: Some("unknown".into()),
            color_name: Some("color".into()),
            color_order: 1,
            min_snr: 5.,
            min_calibrators: 3,
            max_match_residual_px: Some(3.),
            min_catalog_flux_snr: Some(20.),
            max_catalog_ruwe: Some(1.4),
            min_catalog_visibility_periods: Some(8),
            reject_duplicated_sources: true,
            reject_variable_sources: true,
            sigma_clip: 3.5,
            max_iterations: 5,
            validation_fraction: 0.2,
            max_fit_rms_mag: Some(0.25),
            max_validation_rms_mag: Some(0.35),
            max_residual_mad_mag: Some(0.25),
            max_validation_bias_mag: Some(0.20),
            min_validation_count: 2,
            max_source_residual_mag: Some(0.50),
        }
    }
}
impl CalibrationOptions {
    fn validate(&self) -> Result<()> {
        if !self.exposure_s.is_finite()
            || self.exposure_s <= 0.
            || !self.min_snr.is_finite()
            || self.min_snr <= 0.
            || self.color_order > 2
            || self.min_calibrators < self.color_order + 2
            || self.sigma_clip <= 0.
            || self.max_iterations < 1
            || !(0. ..0.5).contains(&self.validation_fraction)
        {
            return Err("invalid photometric fitting parameters".into());
        }
        if [
            self.max_fit_rms_mag,
            self.max_validation_rms_mag,
            self.max_residual_mad_mag,
            self.max_validation_bias_mag,
            self.max_source_residual_mag,
            self.min_catalog_flux_snr,
            self.max_catalog_ruwe,
        ]
        .into_iter()
        .flatten()
        .any(|v| !v.is_finite() || v <= 0.)
        {
            return Err("photometric quality limits must be positive or absent".into());
        }
        Ok(())
    }
    fn empty(
        &self,
        count: usize,
        filters: &BTreeMap<String, usize>,
        status: &str,
        flags: Vec<String>,
    ) -> PhotometricCalibration {
        PhotometricCalibration {
            photometric_system: self.photometric_system.clone(),
            photometric_band: self.photometric_band.clone(),
            color_name: self.color_name.clone(),
            color_order: self.color_order,
            calibrator_count: count,
            status: status.into(),
            flags,
            catalog_filter_counts: filters
                .iter()
                .filter(|(_, n)| **n > 0)
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            ..Default::default()
        }
    }
}
struct Row {
    id: String,
    instrumental: f64,
    magnitude: f64,
    color: f64,
    error: f64,
}
fn basis(color: f64, order: usize) -> Vec<f64> {
    let mut v = vec![1.];
    if order > 0 {
        v.push(color);
    }
    if order > 1 {
        v.push(color * color);
    }
    v
}
fn predict(color: f64, coefficients: &[f64], order: usize) -> f64 {
    basis(color, order)
        .iter()
        .zip(coefficients)
        .map(|(a, b)| a * b)
        .sum()
}
fn coefficients(rows: &[&Row], order: usize) -> Option<Vec<f64>> {
    let n = rows.len();
    let k = order + 1;
    if n < k {
        return None;
    }
    let data: Vec<_> = rows.iter().flat_map(|r| basis(r.color, order)).collect();
    let design = DMatrix::from_row_slice(n, k, &data);
    let svd = design.clone().svd(false, false);
    let tolerance =
        f64::EPSILON * n.max(k) as f64 * svd.singular_values.iter().copied().fold(0., f64::max);
    if svd
        .singular_values
        .iter()
        .filter(|v| **v > tolerance)
        .count()
        < k
    {
        return None;
    }
    let weighted = DMatrix::from_fn(n, k, |i, j| design[(i, j)] / rows[i].error.max(0.005));
    let target = DVector::from_iterator(
        n,
        rows.iter()
            .map(|r| (r.magnitude - r.instrumental) / r.error.max(0.005)),
    );
    let svd = weighted.svd(true, true);
    let cutoff =
        f64::EPSILON * n.max(k) as f64 * svd.singular_values.iter().copied().fold(0., f64::max);
    svd.solve(&target, cutoff)
        .ok()
        .map(|v| v.iter().copied().collect())
}
pub fn fit(
    matches: &[CatalogMatch],
    detections: &[Detection],
    catalog: &[CatalogSource],
    o: &CalibrationOptions,
) -> Result<PhotometricCalibration> {
    o.validate()?;
    let mut filters = BTreeMap::<String, usize>::new();
    if normalise_label(o.photometric_system.as_deref()).is_none()
        || normalise_label(o.photometric_band.as_deref()).is_none()
    {
        return Ok(o.empty(
            0,
            &filters,
            "INSUFFICIENT_CALIBRATORS",
            vec![
                "PHOTOMETRIC_METADATA_REQUIRED".into(),
                "INSUFFICIENT_CALIBRATORS".into(),
            ],
        ));
    }
    let dmap: BTreeMap<_, _> = detections.iter().map(|d| (d.detection_id, d)).collect();
    let cmap: BTreeMap<_, _> = catalog.iter().map(|s| (&s.source_id, s)).collect();
    let mut seen_d = BTreeSet::new();
    let mut seen_s = BTreeSet::new();
    let (mut missing_color, mut metadata_mismatch, mut color_mismatch, mut duplicate) =
        (false, false, false, false);
    let mut rows = Vec::new();
    for m in matches {
        if seen_d.contains(&m.detection_id) || seen_s.contains(&m.source_id) {
            duplicate = true;
            continue;
        }
        seen_d.insert(m.detection_id);
        seen_s.insert(m.source_id.clone());
        let (Some(d), Some(s)) = (dmap.get(&m.detection_id), cmap.get(&m.source_id)) else {
            continue;
        };
        let mut reasons = Vec::new();
        if o.reject_duplicated_sources && s.duplicated_source == Some(true) {
            reasons.push("DUPLICATED_SOURCE");
        }
        if o.reject_variable_sources
            && s.phot_variable_flag
                .as_ref()
                .is_some_and(|v| v.trim().to_uppercase() == "VARIABLE")
        {
            reasons.push("VARIABLE_SOURCE");
        }
        if s.phot_g_mean_flux_over_error
            .zip(o.min_catalog_flux_snr)
            .is_some_and(|(a, b)| a < b)
        {
            reasons.push("LOW_CATALOG_FLUX_SNR");
        }
        if s.ruwe.zip(o.max_catalog_ruwe).is_some_and(|(a, b)| a > b) {
            reasons.push("HIGH_RUWE");
        }
        if s.visibility_periods_used
            .zip(o.min_catalog_visibility_periods)
            .is_some_and(|(a, b)| a < b)
        {
            reasons.push("LOW_VISIBILITY_PERIODS");
        }
        if !reasons.is_empty() {
            for r in reasons {
                *filters.entry(r.into()).or_default() += 1;
            }
            continue;
        }
        if !metadata_matches(
            o.photometric_system.as_deref(),
            o.photometric_band.as_deref(),
            s.photometric_system.as_deref(),
            s.photometric_band.as_deref(),
        ) {
            metadata_mismatch = true;
            continue;
        }
        if o.color_order > 0 && s.color.is_none_or(|c| !c.is_finite()) {
            missing_color = true;
            *filters.entry("MISSING_COLOR".into()).or_default() += 1;
            continue;
        }
        if o.color_order > 0
            && (normalise_label(o.color_name.as_deref()).is_none()
                || normalise_label(s.color_name.as_deref())
                    != normalise_label(o.color_name.as_deref()))
        {
            color_mismatch = true;
            continue;
        }
        let snr = d.flux_snr.unwrap_or(d.snr);
        if !d.quality_passed
            || !snr.is_finite()
            || snr < o.min_snr
            || !d.flux.is_finite()
            || d.flux <= 0.
            || d.flags
                .iter()
                .any(|f| ["EDGE", "MASKED", "SATURATED"].contains(&f.as_str()))
        {
            continue;
        }
        if o.max_match_residual_px
            .is_some_and(|v| !m.residual_px.is_finite() || m.residual_px > v)
        {
            continue;
        }
        let Some(magnitude) = s
            .magnitude
            .or(m.catalog_magnitude)
            .filter(|v| v.is_finite())
        else {
            continue;
        };
        let Some(instrumental) = instrumental_magnitude(d.flux, o.exposure_s)? else {
            continue;
        };
        let measurement = 1.0857362047581296 / snr;
        let catalog_error = s
            .magnitude_error
            .or(m.catalog_magnitude_error)
            .unwrap_or(0.);
        rows.push(Row {
            id: s.source_id.clone(),
            instrumental,
            magnitude,
            color: if o.color_order > 0 {
                s.color.unwrap()
            } else {
                0.
            },
            error: (measurement * measurement + catalog_error * catalog_error).sqrt(),
        });
    }
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    let n = rows.len();
    if (metadata_mismatch || color_mismatch) && n >= o.min_calibrators {
        let mut flags = Vec::new();
        if metadata_mismatch {
            flags.push("PHOTOMETRIC_METADATA_MISMATCH".into());
        }
        if color_mismatch {
            flags.push("COLOR_METADATA_MISMATCH".into());
        }
        return Ok(o.empty(
            n,
            &filters,
            if metadata_mismatch {
                "PHOTOMETRIC_METADATA_MISMATCH"
            } else {
                "COLOR_METADATA_MISMATCH"
            },
            flags,
        ));
    }
    if duplicate && n >= o.min_calibrators {
        return Ok(o.empty(
            n,
            &filters,
            "DUPLICATE_MATCHES",
            vec!["DUPLICATE_MATCHES".into()],
        ));
    }
    if n < o.min_calibrators {
        let mut flags = vec!["INSUFFICIENT_CALIBRATORS".into()];
        for (active, flag) in [
            (missing_color && o.color_order > 0, "MISSING_COLOR"),
            (metadata_mismatch, "PHOTOMETRIC_METADATA_MISMATCH"),
            (color_mismatch, "COLOR_METADATA_MISMATCH"),
            (duplicate, "DUPLICATE_MATCHES"),
        ] {
            if active {
                flags.push(flag.into());
            }
        }
        return Ok(o.empty(n, &filters, "INSUFFICIENT_CALIBRATORS", flags));
    }
    let mut validation = BTreeSet::new();
    let min_holdout =
        (o.min_calibrators * 2).max(o.min_calibrators + o.min_validation_count.max(1));
    if o.validation_fraction > 0.
        && n >= min_holdout
        && n - o.min_calibrators >= o.min_validation_count.max(1)
    {
        let count = o
            .min_validation_count
            .max(1)
            .max((n as f64 * o.validation_fraction).round_ties_even() as usize)
            .min(n - o.min_calibrators);
        for i in 0..count {
            let position = if count == 1 {
                0
            } else if i + 1 == count {
                n - 1
            } else {
                (i as f64 * ((n - 1) as f64 / (count - 1) as f64)).floor() as usize
            };
            validation.insert(position);
        }
    }
    let mut training: Vec<_> = (0..n).filter(|i| !validation.contains(i)).collect();
    if training.len() < o.min_calibrators {
        training = (0..n).collect();
        validation.clear();
    }
    let mut active = vec![true; training.len()];
    let mut clipped = false;
    for _ in 0..o.max_iterations {
        let current: Vec<_> = training
            .iter()
            .zip(&active)
            .filter(|(_, a)| **a)
            .map(|(i, _)| &rows[*i])
            .collect();
        let Some(beta) = coefficients(&current, o.color_order) else {
            return Ok(o.empty(
                n,
                &filters,
                "INSUFFICIENT_COLOR_RANGE",
                vec!["INSUFFICIENT_COLOR_RANGE".into()],
            ));
        };
        let residuals: Vec<_> = training
            .iter()
            .map(|i| {
                rows[*i].magnitude
                    - rows[*i].instrumental
                    - predict(rows[*i].color, &beta, o.color_order)
            })
            .collect();
        let selected: Vec<_> = residuals
            .iter()
            .zip(&active)
            .filter(|(_, a)| **a)
            .map(|(v, _)| *v)
            .collect();
        let center = statistics::median(&selected)?;
        let mad = statistics::median(
            &selected
                .iter()
                .map(|v| (v - center).abs())
                .collect::<Vec<_>>(),
        )?;
        let scale = (1.4826 * mad).max(1e-6);
        let next: Vec<_> = residuals
            .iter()
            .map(|v| (v - center).abs() <= o.sigma_clip * scale)
            .collect();
        if next.iter().filter(|v| **v).count() < o.min_calibrators || next == active {
            break;
        }
        clipped = true;
        active = next;
    }
    let inliers: Vec<_> = training
        .iter()
        .zip(&active)
        .filter(|(_, a)| **a)
        .map(|(i, _)| &rows[*i])
        .collect();
    let Some(beta) = coefficients(&inliers, o.color_order) else {
        return Ok(o.empty(
            n,
            &filters,
            "INSUFFICIENT_INLIERS",
            vec!["INSUFFICIENT_INLIERS".into()],
        ));
    };
    if inliers.len() < o.min_calibrators {
        return Ok(o.empty(
            n,
            &filters,
            "INSUFFICIENT_INLIERS",
            vec!["INSUFFICIENT_INLIERS".into()],
        ));
    }
    let residuals: Vec<_> = inliers
        .iter()
        .map(|r| r.magnitude - r.instrumental - predict(r.color, &beta, o.color_order))
        .collect();
    let rms = |r: &[f64]| {
        (statistics::pairwise_sum(&r.iter().map(|v| v * v).collect::<Vec<_>>()) / r.len() as f64)
            .sqrt()
    };
    let fit_rms = rms(&residuals);
    let center = statistics::median(&residuals)?;
    let mad = 1.4826
        * statistics::median(
            &residuals
                .iter()
                .map(|v| (v - center).abs())
                .collect::<Vec<_>>(),
        )?;
    let validation_residuals: Vec<_> = validation
        .iter()
        .map(|i| {
            rows[*i].magnitude
                - rows[*i].instrumental
                - predict(rows[*i].color, &beta, o.color_order)
        })
        .collect();
    let validation_rms = (!validation.is_empty()).then(|| rms(&validation_residuals));
    let validation_bias = (!validation.is_empty()).then(|| {
        statistics::pairwise_sum(&validation_residuals) / validation_residuals.len() as f64
    });
    let mut flags = Vec::new();
    if clipped {
        flags.push("OUTLIERS_CLIPPED".into());
    }
    if validation.is_empty() {
        flags.push("NO_HOLDOUT_VALIDATION".into());
    }
    let mut gates = Vec::new();
    if o.max_fit_rms_mag.is_some_and(|v| fit_rms > v) {
        gates.push("FIT_RMS_EXCEEDS_LIMIT");
    }
    if o.max_residual_mad_mag.is_some_and(|v| mad > v) {
        gates.push("RESIDUAL_MAD_EXCEEDS_LIMIT");
    }
    if o.validation_fraction > 0. && o.min_validation_count > 0 {
        if validation.len() < o.min_validation_count {
            gates.push("INSUFFICIENT_HOLDOUT_VALIDATION");
        } else {
            if validation_rms
                .zip(o.max_validation_rms_mag)
                .is_some_and(|(a, b)| a > b)
            {
                gates.push("VALIDATION_RMS_EXCEEDS_LIMIT");
            }
            if validation_bias
                .zip(o.max_validation_bias_mag)
                .is_some_and(|(a, b)| a.abs() > b)
            {
                gates.push("VALIDATION_BIAS_EXCEEDS_LIMIT");
            }
        }
    }
    let status = if !gates.is_empty() {
        "PHOTOMETRY_REJECTED_QUALITY_GATE"
    } else if validation.is_empty() {
        "VALID_NO_HOLDOUT"
    } else {
        "VALID"
    };
    flags.extend(gates.into_iter().map(str::to_owned));
    let mut result = o.empty(n, &filters, status, flags);
    result.coefficients = beta;
    result.inlier_count = inliers.len();
    result.validation_count = validation.len();
    result.fit_rms_mag = Some(fit_rms);
    result.validation_rms_mag = validation_rms;
    result.residual_mad_mag = Some(mad);
    result.validation_bias_mag = validation_bias;
    if o.color_order > 0 {
        result.color_min = Some(rows.iter().map(|r| r.color).fold(f64::INFINITY, f64::min));
        result.color_max = Some(
            rows.iter()
                .map(|r| r.color)
                .fold(f64::NEG_INFINITY, f64::max),
        );
    }
    result.max_fit_rms_mag = o.max_fit_rms_mag;
    result.max_validation_rms_mag = o.max_validation_rms_mag;
    result.max_residual_mad_mag = o.max_residual_mad_mag;
    result.max_validation_bias_mag = o.max_validation_bias_mag;
    result.min_validation_count = o.min_validation_count;
    result.max_source_residual_mag = o.max_source_residual_mag;
    let active_by_index: BTreeMap<_, _> = training.iter().copied().zip(active).collect();
    result.calibration_sample_roles = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            (
                r.id.clone(),
                if validation.contains(&i) {
                    "validation"
                } else if active_by_index.get(&i) == Some(&true) {
                    "training_inlier"
                } else {
                    "training_outlier"
                }
                .into(),
            )
        })
        .collect();
    Ok(result)
}
