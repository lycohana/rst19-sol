//! Catalog source contracts and Julian-year proper-motion propagation.
use crate::Result;
use serde::{Deserialize, Serialize};

fn unknown() -> String {
    "unknown".into()
}
fn epoch2016() -> f64 {
    2016.
}
fn unknown_option() -> Option<String> {
    Some(unknown())
}
fn null_unknown<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_else(unknown))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSource {
    pub source_id: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    #[serde(default)]
    pub magnitude: Option<f64>,
    #[serde(default)]
    pub color: Option<f64>,
    #[serde(default)]
    pub pmra_mas_yr: f64,
    #[serde(default)]
    pub pmdec_mas_yr: f64,
    #[serde(default = "epoch2016")]
    pub ref_epoch: f64,
    #[serde(default = "unknown_option")]
    pub catalog_name: Option<String>,
    #[serde(default = "unknown_option")]
    pub photometric_system: Option<String>,
    #[serde(default = "unknown_option")]
    pub photometric_band: Option<String>,
    #[serde(default = "unknown", deserialize_with = "null_unknown")]
    pub magnitude_source: String,
    #[serde(default)]
    pub magnitude_error: Option<f64>,
    #[serde(default)]
    pub color_name: Option<String>,
    #[serde(default)]
    pub parallax_mas: Option<f64>,
    #[serde(default)]
    pub parallax_error_mas: Option<f64>,
    #[serde(default)]
    pub extinction_mag: Option<f64>,
    #[serde(default)]
    pub extinction_error_mag: Option<f64>,
    #[serde(default)]
    pub distance_pc: Option<f64>,
    #[serde(default)]
    pub distance_lower_pc: Option<f64>,
    #[serde(default)]
    pub distance_upper_pc: Option<f64>,
    #[serde(default)]
    pub distance_source: Option<String>,
    #[serde(default)]
    pub phot_g_mean_flux_over_error: Option<f64>,
    #[serde(default)]
    pub phot_bp_rp_excess_factor: Option<f64>,
    #[serde(default)]
    pub ruwe: Option<f64>,
    #[serde(default)]
    pub duplicated_source: Option<bool>,
    #[serde(default)]
    pub visibility_periods_used: Option<u64>,
    #[serde(default)]
    pub phot_variable_flag: Option<String>,
    #[serde(default = "unknown", deserialize_with = "null_unknown")]
    pub extinction_band: String,
    #[serde(default = "unknown", deserialize_with = "null_unknown")]
    pub extinction_system: String,
    #[serde(default = "unknown", deserialize_with = "null_unknown")]
    pub extinction_source: String,
    #[serde(default)]
    pub mg_gspphot: Option<f64>,
    #[serde(default)]
    pub mg_gspphot_lower: Option<f64>,
    #[serde(default)]
    pub mg_gspphot_upper: Option<f64>,
    #[serde(default)]
    pub mg_gspphot_source: Option<String>,
}
pub fn canonical_band(value: &str) -> String {
    let text = value.trim();
    let key: String = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    match key.as_str() {
        "" | "unknown" | "unk" | "na" | "none" => "unknown".into(),
        "g" | "gaiag" | "gaiadr3g" => "G".into(),
        "v" | "johnsonv" | "johnsoncousinsv" => "V".into(),
        "a0" | "azero" | "a05414nm" | "5414nm" => "A0(541.4 nm)".into(),
        _ => text.into(),
    }
}
pub fn canonical_system(value: &str) -> String {
    let text = value.trim();
    let key: String = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    match key.as_str() {
        "" | "unknown" | "unk" | "na" | "none" => "unknown".into(),
        "gaia" | "gaiavega" | "gaiadr3" | "gaiadr3vega" => "Gaia".into(),
        "johnson" | "johnsonv" | "johnsoncousins" | "johnsoncousinsv" => "Johnson".into(),
        "monochromatic" | "monochromatic5414nm" | "a0" => "monochromatic".into(),
        _ => text.into(),
    }
}
impl CatalogSource {
    pub fn extinction_compatibility(&self) -> &'static str {
        if self.extinction_mag.is_none() {
            return "not_applicable";
        }
        let a = canonical_band(&self.extinction_band);
        let b = canonical_band(self.photometric_band.as_deref().unwrap_or(""));
        if a == "unknown" || b == "unknown" {
            return "unknown";
        }
        if a != b {
            return "mismatch";
        }
        let a = canonical_system(&self.extinction_system);
        let b = canonical_system(self.photometric_system.as_deref().unwrap_or(""));
        if a == "unknown" || b == "unknown" {
            "unknown"
        } else if a == b {
            "compatible"
        } else {
            "mismatch"
        }
    }
    pub fn validated(mut self) -> Result<Self> {
        if self.source_id.is_empty() {
            return Err("catalog source_id cannot be empty".into());
        }
        if [
            Some(self.ra_deg),
            Some(self.dec_deg),
            self.magnitude,
            self.color,
            Some(self.pmra_mas_yr),
            Some(self.pmdec_mas_yr),
            Some(self.ref_epoch),
            self.magnitude_error,
            self.parallax_mas,
            self.parallax_error_mas,
            self.extinction_mag,
            self.extinction_error_mag,
            self.distance_pc,
            self.distance_lower_pc,
            self.distance_upper_pc,
            self.mg_gspphot,
            self.mg_gspphot_lower,
            self.mg_gspphot_upper,
            self.phot_g_mean_flux_over_error,
            self.phot_bp_rp_excess_factor,
            self.ruwe,
        ]
        .into_iter()
        .flatten()
        .any(|v| !v.is_finite())
        {
            return Err("catalog source contains a non-finite value".into());
        }
        if [
            self.magnitude_error,
            self.parallax_error_mas,
            self.extinction_error_mag,
            self.distance_pc,
            self.distance_lower_pc,
            self.distance_upper_pc,
            self.phot_g_mean_flux_over_error,
            self.phot_bp_rp_excess_factor,
            self.ruwe,
        ]
        .into_iter()
        .flatten()
        .any(|v| v < 0.)
        {
            return Err("catalog source contains a negative error or quality field".into());
        }
        if [
            self.distance_pc,
            self.distance_lower_pc,
            self.distance_upper_pc,
        ]
        .into_iter()
        .flatten()
        .any(|v| v <= 0.)
        {
            return Err("catalog distance must be positive".into());
        }
        if self
            .distance_lower_pc
            .zip(self.distance_upper_pc)
            .is_some_and(|(a, b)| a > b)
            || self
                .mg_gspphot_lower
                .zip(self.mg_gspphot_upper)
                .is_some_and(|(a, b)| a > b)
            || self
                .mg_gspphot
                .zip(self.mg_gspphot_lower)
                .is_some_and(|(a, b)| a < b)
            || self
                .mg_gspphot
                .zip(self.mg_gspphot_upper)
                .is_some_and(|(a, b)| a > b)
        {
            return Err("catalog interval bounds are inconsistent".into());
        }
        if [
            &self.distance_source,
            &self.mg_gspphot_source,
            &self.phot_variable_flag,
        ]
        .into_iter()
        .flatten()
        .any(|s| s.trim().is_empty())
        {
            return Err("catalog provenance cannot be empty when provided".into());
        }
        if !(-90. ..=90.).contains(&self.dec_deg) || !(-360. ..=360.).contains(&self.ra_deg) {
            return Err("catalog sky coordinate is out of range".into());
        }
        self.extinction_band = canonical_band(&self.extinction_band);
        self.extinction_system = canonical_system(&self.extinction_system);
        for text in [&mut self.magnitude_source, &mut self.extinction_source] {
            *text = text.trim().to_owned();
            if text.is_empty() {
                *text = unknown();
            }
        }
        if self.extinction_compatibility() == "mismatch" {
            return Err("catalog extinction semantics conflict with photometry".into());
        }
        Ok(self)
    }
    pub fn at_epoch(&self, epoch: Option<f64>) -> Result<Self> {
        let Some(epoch) = epoch else {
            return Ok(self.clone());
        };
        if epoch == self.ref_epoch {
            return Ok(self.clone());
        }
        let years = epoch - self.ref_epoch;
        let cos_dec = self.dec_deg.to_radians().cos().abs().max(1e-8);
        let mut result = self.clone();
        result.ra_deg =
            (self.ra_deg + self.pmra_mas_yr * years / cos_dec / 3_600_000.).rem_euclid(360.);
        result.dec_deg = self.dec_deg + self.pmdec_mas_yr * years / 3_600_000.;
        result.ref_epoch = epoch;
        result.validated()
    }
}
