//! Photometric arithmetic and provenance-bearing absolute magnitude gates.
use crate::{catalog::CatalogSource, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use unicode_casefold::UnicodeCaseFold;

pub fn normalise_label(value: Option<&str>) -> Option<String> {
    let text = value?
        .split(|c: char| c.is_whitespace() || ('\u{001c}'..='\u{001f}').contains(&c))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let folded: String = text.as_str().case_fold().collect();
    if [
        "",
        "unknown",
        "none",
        "null",
        "n/a",
        "na",
        "undefined",
        "?",
        "-",
        "—",
    ]
    .contains(&folded.as_str())
    {
        None
    } else {
        Some(folded)
    }
}
pub fn metadata_matches(
    es: Option<&str>,
    eb: Option<&str>,
    actual_s: Option<&str>,
    actual_b: Option<&str>,
) -> bool {
    let (es, eb) = (normalise_label(es), normalise_label(eb));
    es.is_some()
        && eb.is_some()
        && es == normalise_label(actual_s)
        && eb == normalise_label(actual_b)
}
fn declared(system: Option<&str>, band: Option<&str>) -> bool {
    normalise_label(system).is_some() && normalise_label(band).is_some()
}
pub fn instrumental_magnitude(flux: f64, exposure: f64) -> Result<Option<f64>> {
    if !flux.is_finite() || flux <= 0. || !exposure.is_finite() || exposure <= 0. {
        return Ok(None);
    }
    let rate = flux / exposure;
    if rate <= 0. {
        return Err("math domain error in flux rate".into());
    }
    Ok(Some(-2.5 * rate.log10()))
}
pub fn calibrated_apparent_magnitude(
    instrumental: f64,
    zero: f64,
    color: Option<f64>,
    coefficient: f64,
    extinction: f64,
) -> Result<Option<f64>> {
    if !instrumental.is_finite() {
        return Ok(None);
    }
    if [zero, coefficient, extinction]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err("photometric coefficients must be finite".into());
    }
    let term = match color {
        None if coefficient != 0. => {
            return Err("color is required when color coefficient is nonzero".into())
        }
        None => 0.,
        Some(v) if !v.is_finite() => return Err("color must be finite".into()),
        Some(v) => coefficient * v,
    };
    Ok(Some(instrumental + zero + term + extinction))
}
pub fn absolute_from_apparent(
    apparent: f64,
    distance: f64,
    extinction: f64,
) -> Result<Option<f64>> {
    if !apparent.is_finite() {
        return Ok(None);
    }
    if !distance.is_finite() || distance <= 0. || !extinction.is_finite() {
        return Err("invalid distance or extinction".into());
    }
    let ratio = distance / 10.;
    if ratio <= 0. {
        return Err("math domain error in distance modulus".into());
    }
    Ok(Some(apparent - 5. * ratio.log10() - extinction))
}
pub fn absolute_from_parallax(
    apparent: f64,
    parallax: f64,
    extinction: f64,
) -> Result<Option<f64>> {
    if !parallax.is_finite() || parallax <= 0. {
        return Err("parallax must be finite and positive".into());
    }
    absolute_from_apparent(apparent, 1000. / parallax, extinction)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsoluteMagnitudeEstimate {
    pub value: Option<f64>,
    pub error: Option<f64>,
    pub distance_pc: Option<f64>,
    pub corrected_parallax_mas: Option<f64>,
    pub extinction_mag: Option<f64>,
    pub status: String,
    pub flags: Vec<String>,
    pub distance_source: Option<String>,
    pub distance_lower_pc: Option<f64>,
    pub distance_upper_pc: Option<f64>,
    pub extinction_band: Option<String>,
    pub extinction_system: Option<String>,
    pub extinction_source: Option<String>,
}
impl AbsoluteMagnitudeEstimate {
    fn has(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }
    fn flags(&mut self, flags: &[&str]) {
        self.flags.extend(flags.iter().map(|s| (*s).into()));
    }
    fn fail(mut self, status: &str, flags: &[&str]) -> Self {
        self.status = status.into();
        self.flags(flags);
        self
    }
    pub fn distance_interval_status(&self) -> &'static str {
        for (flag, status) in [
            ("DISTANCE_INTERVAL_INVALID", "INVALID"),
            (
                "DISTANCE_INTERVAL_DERIVED_FROM_PARALLAX_ERROR",
                "DERIVED_FROM_PARALLAX_ERROR",
            ),
            ("DISTANCE_INTERVAL_USED", "PROVIDED"),
            ("DISTANCE_INTERVAL_ONE_SIDED", "ONE_SIDED"),
            ("DISTANCE_INTERVAL_REQUIRED", "REQUIRED"),
            ("DISTANCE_ERROR_NOT_PROVIDED", "NOT_PROVIDED"),
        ] {
            if self.has(flag) {
                return status;
            }
        }
        if self.distance_lower_pc.is_some() || self.distance_upper_pc.is_some() {
            "PRESENT_UNCLASSIFIED"
        } else {
            "NOT_AVAILABLE"
        }
    }
    pub fn error_status(&self) -> &'static str {
        if [
            "APPARENT_MAGNITUDE_ERROR_NOT_PROVIDED",
            "EXTINCTION_ERROR_NOT_PROVIDED",
            "DISTANCE_ERROR_NOT_PROVIDED",
            "DISTANCE_INTERVAL_REQUIRED",
            "PARALLAX_ERROR_REQUIRED",
            "PARALLAX_ZERO_POINT_ERROR_NOT_PROVIDED",
        ]
        .iter()
        .any(|f| self.has(f))
        {
            "INCOMPLETE"
        } else if self.error.is_some() {
            "AVAILABLE"
        } else {
            "NOT_AVAILABLE"
        }
    }
    pub fn is_strict(&self) -> bool {
        self.value.is_some()
            && ["VALID", "VALID_MODEL_DISTANCE"].contains(&self.status.as_str())
            && normalise_label(self.distance_source.as_deref()).is_some()
            && declared(
                self.extinction_system.as_deref(),
                self.extinction_band.as_deref(),
            )
            && ["PROVIDED", "DERIVED_FROM_PARALLAX_ERROR"]
                .contains(&self.distance_interval_status())
            && !self.has("EXTINCTION_SEMANTICS_REQUIRED")
            && !self.has("DISTANCE_SOURCE_REQUIRED")
    }
    pub fn as_dict(&self) -> Value {
        let lower = self
            .distance_pc
            .zip(self.distance_lower_pc)
            .filter(|(d, l)| d.is_finite() && l.is_finite())
            .map(|(d, l)| d - l)
            .filter(|v| *v >= 0.);
        let upper = self
            .distance_pc
            .zip(self.distance_upper_pc)
            .filter(|(d, u)| d.is_finite() && u.is_finite())
            .map(|(d, u)| u - d)
            .filter(|v| *v >= 0.);
        let error = match (lower, upper) {
            (Some(a), Some(b)) => Some(0.5 * (a + b)),
            (a, b) => a.or(b),
        };
        let mut value = serde_json::to_value(self).expect("absolute estimate serialization");
        let object = value.as_object_mut().unwrap();
        for (key, v) in [
            (
                "distance_interval_status",
                json!(self.distance_interval_status()),
            ),
            ("distance_error_pc", json!(error)),
            ("distance_error_lower_pc", json!(lower)),
            ("distance_error_upper_pc", json!(upper)),
            ("error_status", json!(self.error_status())),
            ("is_strict", json!(self.is_strict())),
            (
                "is_strict_with_uncertainty",
                json!(
                    self.is_strict() && self.error_status() == "AVAILABLE" && self.error.is_some()
                ),
            ),
        ] {
            object.insert(key.into(), v);
        }
        value
    }
}
fn default_fraction() -> f64 {
    0.2
}
#[derive(Debug, Clone, Deserialize)]
pub struct AbsoluteOptions {
    pub apparent_magnitude: f64,
    #[serde(default)]
    pub parallax_mas: Option<f64>,
    #[serde(default)]
    pub parallax_error_mas: Option<f64>,
    #[serde(default)]
    pub distance_pc: Option<f64>,
    #[serde(default)]
    pub distance_lower_pc: Option<f64>,
    #[serde(default)]
    pub distance_upper_pc: Option<f64>,
    #[serde(default)]
    pub extinction_mag: Option<f64>,
    #[serde(default)]
    pub apparent_magnitude_error: Option<f64>,
    #[serde(default)]
    pub extinction_error_mag: Option<f64>,
    #[serde(default)]
    pub parallax_zero_point_mas: f64,
    #[serde(default = "default_fraction")]
    pub max_fractional_parallax_error: f64,
    #[serde(default)]
    pub distance_source: Option<String>,
    #[serde(default)]
    pub extinction_band: Option<String>,
    #[serde(default)]
    pub extinction_system: Option<String>,
    #[serde(default)]
    pub extinction_source: Option<String>,
}
impl AbsoluteOptions {
    fn empty(&self) -> AbsoluteMagnitudeEstimate {
        AbsoluteMagnitudeEstimate {
            value: None,
            error: None,
            distance_pc: None,
            corrected_parallax_mas: None,
            extinction_mag: self.extinction_mag,
            status: String::new(),
            flags: Vec::new(),
            distance_source: None,
            distance_lower_pc: None,
            distance_upper_pc: None,
            extinction_band: self.extinction_band.clone(),
            extinction_system: self.extinction_system.clone(),
            extinction_source: self.extinction_source.clone(),
        }
    }
    fn validate_errors(&self) -> Result<()> {
        if [self.apparent_magnitude_error, self.extinction_error_mag]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite() || v < 0.)
        {
            Err("magnitude errors must be finite and nonnegative".into())
        } else {
            Ok(())
        }
    }
}
pub fn parallax_estimate(o: &AbsoluteOptions) -> Result<AbsoluteMagnitudeEstimate> {
    let parallax = o.parallax_mas.ok_or("missing parallax_mas")?;
    if !o.apparent_magnitude.is_finite()
        || !parallax.is_finite()
        || !o.parallax_zero_point_mas.is_finite()
        || !o.max_fractional_parallax_error.is_finite()
        || o.max_fractional_parallax_error <= 0.
    {
        return Err("invalid parallax estimate arguments".into());
    }
    o.validate_errors()?;
    let mut e = o.empty();
    if let Some(a) = o.extinction_mag {
        if !a.is_finite() {
            return Err("extinction must be finite".into());
        }
        if a < 0. {
            return Ok(e.fail("INVALID_EXTINCTION", &["NEGATIVE_EXTINCTION"]));
        }
    }
    let corrected = parallax - o.parallax_zero_point_mas;
    e.corrected_parallax_mas = Some(corrected);
    if corrected <= 0. {
        return Ok(e.fail("INVALID_PARALLAX", &["NON_POSITIVE_CORRECTED_PARALLAX"]));
    }
    let distance = 1000. / corrected;
    e.distance_pc = Some(distance);
    let Some(error) = o.parallax_error_mas else {
        e = e.fail(
            "NO_PARALLAX_ERROR",
            &["PARALLAX_ERROR_REQUIRED", "DISTANCE_ERROR_NOT_PROVIDED"],
        );
        if o.extinction_mag.is_none() {
            e.flags(&["EXTINCTION_NOT_PROVIDED"]);
        }
        return Ok(e);
    };
    if !error.is_finite() || error < 0. {
        return Err("parallax error must be finite and nonnegative".into());
    }
    if corrected + error > 0. && corrected - error > 0. {
        e.distance_lower_pc = Some(1000. / (corrected + error));
        e.distance_upper_pc = Some(1000. / (corrected - error));
    }
    let interval = if e.distance_lower_pc.is_some() && e.distance_upper_pc.is_some() {
        vec!["DISTANCE_INTERVAL_DERIVED_FROM_PARALLAX_ERROR"]
    } else {
        Vec::new()
    };
    if error / corrected > o.max_fractional_parallax_error || e.distance_upper_pc.is_none() {
        e = e.fail("LOW_PARALLAX_SNR", &["USE_DISTANCE_POSTERIOR"]);
        if o.extinction_mag.is_none() {
            e.flags(&["EXTINCTION_NOT_PROVIDED"]);
        }
        e.flags(&interval);
        return Ok(e);
    }
    if o.extinction_mag.is_none() {
        e = e.fail("VALID_NO_EXTINCTION", &["EXTINCTION_NOT_PROVIDED"]);
        e.flags(&interval);
        return Ok(e);
    }
    if !declared(o.extinction_system.as_deref(), o.extinction_band.as_deref()) {
        e = e.fail(
            "EXTINCTION_SEMANTICS_REQUIRED",
            &["EXTINCTION_SEMANTICS_REQUIRED"],
        );
        e.flags(&interval);
        return Ok(e);
    }
    e.value = absolute_from_apparent(o.apparent_magnitude, distance, o.extinction_mag.unwrap())?;
    e.flags(&interval);
    let mut variance = (5. / 10_f64.ln() * error / corrected).powi(2);
    let mut complete = true;
    for (value, flag) in [
        (
            o.apparent_magnitude_error,
            "APPARENT_MAGNITUDE_ERROR_NOT_PROVIDED",
        ),
        (o.extinction_error_mag, "EXTINCTION_ERROR_NOT_PROVIDED"),
    ] {
        if let Some(v) = value {
            variance += v * v;
        } else {
            e.flags(&[flag]);
            complete = false;
        }
    }
    if o.parallax_zero_point_mas != 0. {
        e.flags(&["PARALLAX_ZERO_POINT_CORRECTED"]);
    }
    e.error = (complete && variance > 0.).then(|| variance.sqrt());
    e.status = "VALID".into();
    e.distance_source = Some("parallax".into());
    Ok(e)
}
pub fn distance_estimate(o: &AbsoluteOptions) -> Result<AbsoluteMagnitudeEstimate> {
    let distance = o.distance_pc.ok_or("missing distance_pc")?;
    if !o.apparent_magnitude.is_finite() || !distance.is_finite() || distance <= 0. {
        return Err("invalid apparent magnitude or distance".into());
    }
    if [o.distance_lower_pc, o.distance_upper_pc]
        .into_iter()
        .flatten()
        .any(|v| !v.is_finite() || v <= 0.)
    {
        return Err("distance bounds must be finite and positive".into());
    }
    let mut e = o.empty();
    e.distance_pc = Some(distance);
    e.distance_lower_pc = o.distance_lower_pc;
    e.distance_upper_pc = o.distance_upper_pc;
    e.distance_source = o.distance_source.clone();
    if o.distance_lower_pc
        .zip(o.distance_upper_pc)
        .is_some_and(|(l, u)| l > u || distance < l || distance > u)
        || o.distance_lower_pc.is_some_and(|l| l > distance)
        || o.distance_upper_pc.is_some_and(|u| u < distance)
    {
        return Ok(e.fail("INVALID_DISTANCE_INTERVAL", &["DISTANCE_INTERVAL_INVALID"]));
    }
    let interval = match (o.distance_lower_pc, o.distance_upper_pc) {
        (Some(_), Some(_)) => vec!["DISTANCE_INTERVAL_USED"],
        (None, None) => vec!["DISTANCE_ERROR_NOT_PROVIDED", "DISTANCE_INTERVAL_REQUIRED"],
        _ => vec!["DISTANCE_INTERVAL_ONE_SIDED", "DISTANCE_INTERVAL_REQUIRED"],
    };
    o.validate_errors()?;
    if normalise_label(o.distance_source.as_deref()).is_none() {
        e = e.fail(
            "MODEL_DISTANCE_SOURCE_REQUIRED",
            &["DISTANCE_SOURCE_REQUIRED"],
        );
        e.flags(&interval);
        return Ok(e);
    }
    let Some(extinction) = o.extinction_mag else {
        e = e.fail(
            "MODEL_DISTANCE_NO_EXTINCTION",
            &["DISTANCE_MODEL_USED", "EXTINCTION_NOT_PROVIDED"],
        );
        e.flags(&interval);
        return Ok(e);
    };
    if !extinction.is_finite() {
        return Err("extinction must be finite".into());
    }
    if extinction < 0. {
        e = e.fail(
            "INVALID_EXTINCTION",
            &["NEGATIVE_EXTINCTION", "DISTANCE_MODEL_USED"],
        );
        e.flags(&interval);
        return Ok(e);
    }
    if !declared(o.extinction_system.as_deref(), o.extinction_band.as_deref()) {
        e = e.fail(
            "EXTINCTION_SEMANTICS_REQUIRED",
            &["DISTANCE_MODEL_USED", "EXTINCTION_SEMANTICS_REQUIRED"],
        );
        e.flags(&interval);
        return Ok(e);
    }
    let (Some(lower), Some(upper)) = (o.distance_lower_pc, o.distance_upper_pc) else {
        e = e.fail("MODEL_DISTANCE_INTERVAL_REQUIRED", &["DISTANCE_MODEL_USED"]);
        e.flags(&interval);
        return Ok(e);
    };
    e.value = absolute_from_apparent(o.apparent_magnitude, distance, extinction)?;
    e.flags(&["DISTANCE_MODEL_USED", "DISTANCE_INTERVAL_USED"]);
    let distance_log_error = 0.5 * (upper / lower).ln();
    let mut variance = (5. / 10_f64.ln() * distance_log_error).powi(2);
    let mut complete = true;
    for (value, flag) in [
        (
            o.apparent_magnitude_error,
            "APPARENT_MAGNITUDE_ERROR_NOT_PROVIDED",
        ),
        (o.extinction_error_mag, "EXTINCTION_ERROR_NOT_PROVIDED"),
    ] {
        if let Some(v) = value {
            variance += v * v;
        } else {
            e.flags(&[flag]);
            complete = false;
        }
    }
    e.error = (complete && variance > 0.).then(|| variance.sqrt());
    e.status = "VALID_MODEL_DISTANCE".into();
    Ok(e)
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CatalogAbsoluteOptions {
    pub apparent_magnitude: Option<f64>,
    pub apparent_magnitude_error: Option<f64>,
    pub required_photometric_system: Option<String>,
    pub required_photometric_band: Option<String>,
    pub parallax_zero_point_mas: f64,
    #[serde(default = "default_fraction")]
    pub max_fractional_parallax_error: f64,
}
impl Default for CatalogAbsoluteOptions {
    fn default() -> Self {
        Self {
            apparent_magnitude: None,
            apparent_magnitude_error: None,
            required_photometric_system: None,
            required_photometric_band: None,
            parallax_zero_point_mas: 0.,
            max_fractional_parallax_error: default_fraction(),
        }
    }
}
pub fn catalog_estimate(
    s: &CatalogSource,
    o: &CatalogAbsoluteOptions,
) -> Result<AbsoluteMagnitudeEstimate> {
    let magnitude = o.apparent_magnitude.or(s.magnitude);
    let args = AbsoluteOptions {
        apparent_magnitude: magnitude.unwrap_or(0.),
        parallax_mas: s.parallax_mas,
        parallax_error_mas: s.parallax_error_mas,
        distance_pc: s.distance_pc,
        distance_lower_pc: s.distance_lower_pc,
        distance_upper_pc: s.distance_upper_pc,
        extinction_mag: s.extinction_mag,
        apparent_magnitude_error: if o.apparent_magnitude.is_none()
            && o.apparent_magnitude_error.is_none()
        {
            s.magnitude_error
        } else {
            o.apparent_magnitude_error
        },
        extinction_error_mag: s.extinction_error_mag,
        parallax_zero_point_mas: o.parallax_zero_point_mas,
        max_fractional_parallax_error: o.max_fractional_parallax_error,
        distance_source: s.distance_source.clone(),
        extinction_band: Some(s.extinction_band.clone()),
        extinction_system: Some(s.extinction_system.clone()),
        extinction_source: Some(s.extinction_source.clone()),
    };
    let e = args.empty();
    if o.required_photometric_system.is_none() != o.required_photometric_band.is_none() {
        return Ok(e.fail(
            "PHOTOMETRIC_METADATA_REQUIRED",
            &["PHOTOMETRIC_SYSTEM_AND_BAND_REQUIRED"],
        ));
    }
    if o.required_photometric_system.is_some()
        && !metadata_matches(
            o.required_photometric_system.as_deref(),
            o.required_photometric_band.as_deref(),
            s.photometric_system.as_deref(),
            s.photometric_band.as_deref(),
        )
    {
        return Ok(e.fail(
            "PHOTOMETRIC_METADATA_MISMATCH",
            &["CATALOG_BAND_DOES_NOT_MATCH_CALIBRATION"],
        ));
    }
    if s.extinction_mag.is_some() {
        match s.extinction_compatibility() {
            "compatible" => {}
            "mismatch" => {
                return Ok(e.fail("EXTINCTION_BAND_MISMATCH", &["EXTINCTION_BAND_MISMATCH"]))
            }
            _ => {
                return Ok(e.fail(
                    "EXTINCTION_SEMANTICS_REQUIRED",
                    &["EXTINCTION_SEMANTICS_REQUIRED"],
                ))
            }
        }
    }
    if magnitude.is_none() {
        return Ok(e.fail("NO_APPARENT_MAGNITUDE", &["NO_CATALOG_MAGNITUDE"]));
    }
    let mut fallback = false;
    if s.parallax_mas.is_some() {
        let mut estimate = parallax_estimate(&args)?;
        if estimate.value.is_some() || s.distance_pc.is_none() {
            if estimate.distance_source.is_none() {
                estimate.distance_source = Some("parallax".into());
            }
            return Ok(estimate);
        }
        fallback = true;
    }
    if s.distance_pc.is_none() {
        return Ok(e.fail("NO_PARALLAX", &["NO_DISTANCE"]));
    }
    if s.distance_source.is_none() {
        let mut e = e;
        e.distance_pc = s.distance_pc;
        e.distance_lower_pc = s.distance_lower_pc;
        e.distance_upper_pc = s.distance_upper_pc;
        return Ok(e.fail(
            "MODEL_DISTANCE_SOURCE_REQUIRED",
            &["DISTANCE_SOURCE_REQUIRED"],
        ));
    }
    let mut e = distance_estimate(&args)?;
    if fallback {
        e.flags
            .insert(0, "PARALLAX_QUALITY_FALLBACK_TO_MODEL_DISTANCE".into());
    }
    Ok(e)
}

/// Structured native operation boundary; no Python runtime is involved.
pub fn evaluate(kind: &str, p: &Value, source: Option<&Value>) -> Result<Value> {
    let required = |key: &str| {
        p[key]
            .as_f64()
            .ok_or_else(|| format!("missing numeric {key}"))
    };
    let numeric = |key: &str, default: f64| -> Result<f64> {
        match p.get(key) {
            None => Ok(default),
            Some(v) => v.as_f64().ok_or_else(|| format!("invalid numeric {key}")),
        }
    };
    let optional = |key: &str| -> Result<Option<f64>> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| format!("invalid numeric {key}")),
        }
    };
    let value = match kind {
        "instrumental_magnitude" => {
            instrumental_magnitude(required("flux")?, numeric("exposure_s", 1.)?)?
        }
        "calibrated_apparent_magnitude" => calibrated_apparent_magnitude(
            required("instrumental")?,
            required("zero_point")?,
            optional("color")?,
            numeric("color_coefficient", 0.)?,
            numeric("extinction_mag", 0.)?,
        )?,
        "absolute_magnitude_from_apparent" => absolute_from_apparent(
            required("apparent_magnitude")?,
            required("distance_pc")?,
            numeric("extinction_mag", 0.)?,
        )?,
        "absolute_magnitude_from_parallax" => absolute_from_parallax(
            required("apparent_magnitude")?,
            required("parallax_mas")?,
            numeric("extinction_mag", 0.)?,
        )?,
        "absolute_magnitude_estimate_from_parallax"
        | "absolute_magnitude_estimate_from_distance" => {
            let options: AbsoluteOptions =
                serde_json::from_value(p.clone()).map_err(|e| e.to_string())?;
            return Ok(if kind.ends_with("parallax") {
                parallax_estimate(&options)?
            } else {
                distance_estimate(&options)?
            }
            .as_dict());
        }
        "catalog" => {
            let source: CatalogSource =
                serde_json::from_value(source.ok_or("missing catalog source")?.clone())
                    .map_err(|e| e.to_string())?;
            let options: CatalogAbsoluteOptions =
                serde_json::from_value(p.clone()).map_err(|e| e.to_string())?;
            return Ok(catalog_estimate(&source, &options)?.as_dict());
        }
        "inferred_zero_point_for_comparison" => {
            let instrumental = required("instrumental")?;
            let reference = required("reference_apparent_magnitude")?;
            (instrumental.is_finite() && reference.is_finite()).then_some(reference - instrumental)
        }
        _ => return Err("photometric operation has not been migrated".into()),
    };
    Ok(json!(value))
}
