//! Offline CSV aliases and explicit photometric/extinction provenance.
use crate::{
    catalog::{canonical_band, canonical_system, CatalogSource},
    Result,
};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

struct Row {
    entries: Vec<(String, Option<String>)>,
    number: usize,
}
impl Row {
    fn named(&self, names: &[&str]) -> Option<(String, String)> {
        let mut folded = BTreeMap::new();
        for (name, value) in &self.entries {
            folded.entry(name.trim().to_lowercase()).or_insert(value);
        }
        for &name in names {
            let exact = self
                .entries
                .iter()
                .find(|(key, _)| key == name)
                .and_then(|(_, v)| v.as_deref())
                .filter(|v| !v.trim().is_empty());
            let value = exact.or_else(|| {
                folded
                    .get(&name.trim().to_lowercase())
                    .and_then(|v| v.as_deref())
            });
            if let Some(value) = value.filter(|v| !v.trim().is_empty()) {
                return Some((name.trim().to_lowercase(), value.trim().into()));
            }
        }
        None
    }
    fn text(&self, names: &[&str]) -> Option<String> {
        self.named(names).map(|(_, v)| v)
    }
    fn number(&self, names: &[&str]) -> Result<Option<f64>> {
        self.text(names)
            .map(|v| {
                parse_number(&v).ok_or_else(|| {
                    format!(
                        "catalog row {}: invalid or non-finite {}",
                        self.number, names[0]
                    )
                })
            })
            .transpose()
    }
}
fn parse_number(value: &str) -> Option<f64> {
    let chars: Vec<_> = value.chars().collect();
    if chars.iter().enumerate().any(|(i, c)| {
        *c == '_'
            && (i == 0
                || i + 1 == chars.len()
                || !chars[i - 1].is_ascii_digit()
                || !chars[i + 1].is_ascii_digit())
    }) {
        return None;
    }
    value
        .replace('_', "")
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
}
fn numbers_equal(a: &str, b: &str) -> bool {
    match (parse_number(a), parse_number(b)) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-12_f64.max(1e-9 * a.abs().max(b.abs())),
        _ => a.trim() == b.trim(),
    }
}
fn extinction_kind(field: &str) -> &'static str {
    match field {
        "ag_gspphot" => "gaia_g",
        "azero_gspphot" | "a0_gspphot" | "a0" | "azero" => "a0",
        "a_v" | "av" => "johnson_v",
        _ => "unknown",
    }
}
fn extinction(row: &Row) -> Result<(Option<f64>, String, String, String)> {
    let fields = [
        "ag_gspphot",
        "azero_gspphot",
        "a0_gspphot",
        "a0",
        "azero",
        "a_v",
        "av",
        "a_band",
        "a_g",
        "ag",
        "extinction_mag",
    ];
    let present: Vec<_> = fields
        .iter()
        .filter_map(|field| row.text(&[field]).map(|v| (*field, v)))
        .collect();
    let kinds: BTreeSet<_> = present
        .iter()
        .map(|(field, _)| extinction_kind(field))
        .filter(|kind| *kind != "unknown")
        .collect();
    if kinds.len() > 1 {
        return Err(format!(
            "catalog row {}: conflicting extinction columns",
            row.number
        ));
    }
    if let Some((field, value)) = present.first() {
        if extinction_kind(field) != "unknown" {
            for (other, raw) in &present[1..] {
                if (extinction_kind(other) == "unknown" && !numbers_equal(value, raw))
                    || (extinction_kind(other) != "unknown"
                        && extinction_kind(other) != extinction_kind(field))
                {
                    return Err(format!(
                        "catalog row {}: conflicting extinction values",
                        row.number
                    ));
                }
            }
        }
    }
    let selected = present.first().map(|(field, _)| *field);
    let (inferred_band, inferred_system, inferred_source) = match selected {
        Some("ag_gspphot") => ("G", "Gaia", "Gaia DR3 GSP-Phot: ag_gspphot".into()),
        Some(field @ ("azero_gspphot" | "a0_gspphot")) => (
            "A0(541.4 nm)",
            "monochromatic",
            format!("Gaia DR3 GSP-Phot: {field}"),
        ),
        Some(field @ ("a0" | "azero")) => (
            "A0(541.4 nm)",
            "monochromatic",
            format!("CSV column: {field}"),
        ),
        Some(field @ ("a_v" | "av")) => ("V", "Johnson", format!("CSV column: {field}")),
        Some(field) => ("unknown", "unknown", format!("CSV column: {field}")),
        None => ("unknown", "unknown", "unknown".into()),
    };
    let band = row
        .text(&["extinction_band", "ext_band", "extinction_passband"])
        .map(|v| canonical_band(&v));
    let system = row
        .text(&[
            "extinction_system",
            "extinction_photometric_system",
            "ext_system",
        ])
        .map(|v| canonical_system(&v));
    if band
        .as_ref()
        .is_some_and(|v| inferred_band != "unknown" && *v != inferred_band)
        || system
            .as_ref()
            .is_some_and(|v| inferred_system != "unknown" && *v != inferred_system)
    {
        return Err(format!(
            "catalog row {}: explicit extinction semantics conflict with column",
            row.number
        ));
    }
    let source = row
        .text(&["extinction_source", "extinction_provenance", "ext_source"])
        .unwrap_or(inferred_source);
    let value = selected
        .map(|field| row.number(&[field]))
        .transpose()?
        .flatten();
    Ok((
        value,
        band.unwrap_or(inferred_band.into()),
        system.unwrap_or(inferred_system.into()),
        source,
    ))
}
pub fn load_catalog_csv(path: impl AsRef<std::path::Path>) -> Result<Vec<CatalogSource>> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    load_catalog_bytes(&bytes)
}
pub fn load_catalog_bytes(bytes: &[u8]) -> Result<Vec<CatalogSource>> {
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    // DictReader accepts a blank first record as an empty header. The CSV
    // parser skips blank records, so preserve that distinction explicitly.
    if bytes.first().is_some_and(|b| *b == b'\r' || *b == b'\n') {
        return if bytes.iter().all(|b| *b == b'\r' || *b == b'\n') {
            Ok(Vec::new())
        } else {
            Err("catalog row 2: missing source_id".into())
        };
    }
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let headers = reader.headers().map_err(|e| e.to_string())?.clone();
    if headers.is_empty() {
        return Err("catalog CSV has no header".into());
    }
    let fieldnames: BTreeSet<_> = headers.iter().map(|h| h.trim().to_lowercase()).collect();
    let mut sources = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, record) in reader.records().enumerate() {
        let record = record.map_err(|e| e.to_string())?;
        let mut entries: Vec<(String, Option<String>)> = Vec::new();
        for (i, h) in headers.iter().enumerate() {
            let value = record.get(i).map(str::to_owned);
            if let Some((_, v)) = entries.iter_mut().find(|(key, _)| key == h) {
                *v = value;
            } else {
                entries.push((h.into(), value));
            }
        }
        let row = Row {
            entries,
            number: index + 2,
        };
        let mut object = Map::new();
        let id = row
            .text(&["source_id", "sourceid", "id"])
            .ok_or_else(|| format!("catalog row {}: missing source_id", row.number))?;
        if !ids.insert(id.clone()) {
            return Err(format!("catalog row {}: duplicate source_id", row.number));
        }
        object.insert("source_id".into(), json!(id));
        for (field, names) in [
            ("ra_deg", &["ra_deg", "ra", "RA"][..]),
            ("dec_deg", &["dec_deg", "dec", "DEC"][..]),
        ] {
            let value = row
                .number(names)?
                .ok_or_else(|| format!("catalog row {}: missing {field}", row.number))?;
            object.insert(field.into(), json!(value));
        }
        let explicit = row.text(&[
            "magnitude_source",
            "mag_source",
            "magnitude_provenance",
            "mag_provenance",
        ]);
        let gaia = row.text(&["phot_g_mean_mag"]);
        let generic = row.text(&["magnitude", "apparent_magnitude", "mag"]);
        if gaia
            .as_ref()
            .zip(generic.as_ref())
            .is_some_and(|(a, b)| !numbers_equal(a, b))
        {
            return Err(format!(
                "catalog row {}: conflicting magnitudes",
                row.number
            ));
        }
        let is_gaia = gaia.is_some()
            && explicit.as_ref().is_none_or(|v| {
                v.to_lowercase().contains("gaia") || v.to_lowercase().contains("phot_g")
            });
        let raw = if is_gaia {
            gaia.as_ref()
        } else {
            generic.as_ref().or(gaia.as_ref())
        };
        let mag = raw
            .map(|v| parse_number(v).ok_or("invalid catalog magnitude"))
            .transpose()?;
        let provenance = explicit.unwrap_or_else(|| {
            if is_gaia {
                "Gaia DR3 phot_g_mean_mag".into()
            } else if generic.is_some() {
                "CSV magnitude column".into()
            } else {
                "unknown".into()
            }
        });
        object.insert("magnitude".into(), json!(mag));
        object.insert("magnitude_source".into(), json!(provenance));
        let numeric_fields: &[(&str, &[&str])] = &[
            (
                "magnitude_error",
                &[
                    "magnitude_error",
                    "mag_error",
                    "phot_g_mean_mag_error",
                    "phot_g_mean_mag_err",
                ],
            ),
            ("color", &["color", "bp_rp"]),
            ("pmra_mas_yr", &["pmra_mas_yr", "pmra"]),
            ("pmdec_mas_yr", &["pmdec_mas_yr", "pmdec"]),
            ("ref_epoch", &["ref_epoch", "epoch"]),
            ("parallax_mas", &["parallax_mas", "parallax", "plx"]),
            (
                "parallax_error_mas",
                &["parallax_error_mas", "parallax_error", "parallax_err"],
            ),
            (
                "extinction_error_mag",
                &[
                    "extinction_error_mag",
                    "ag_gspphot_error",
                    "azero_gspphot_error",
                    "a0_gspphot_error",
                    "a0_error",
                    "azero_error",
                    "extinction_error",
                    "a_band_error",
                    "a_v_error",
                    "av_error",
                ],
            ),
            (
                "distance_pc",
                &[
                    "distance_pc",
                    "distance_gspphot",
                    "distance",
                    "r_med_geo",
                    "r_med_photogeo",
                    "r_med_photogeometric",
                ],
            ),
            (
                "distance_lower_pc",
                &[
                    "distance_lower_pc",
                    "distance_gspphot_lower",
                    "distance_lower",
                    "r_lo_geo",
                    "r_lo_photogeo",
                    "r_lo_photogeometric",
                ],
            ),
            (
                "distance_upper_pc",
                &[
                    "distance_upper_pc",
                    "distance_gspphot_upper",
                    "distance_upper",
                    "r_hi_geo",
                    "r_hi_photogeo",
                    "r_hi_photogeometric",
                ],
            ),
            ("mg_gspphot", &["mg_gspphot", "mg_g", "gsp_phot_mg"]),
            (
                "mg_gspphot_lower",
                &["mg_gspphot_lower", "mg_g_lower", "gsp_phot_mg_lower"],
            ),
            (
                "mg_gspphot_upper",
                &["mg_gspphot_upper", "mg_g_upper", "gsp_phot_mg_upper"],
            ),
            (
                "phot_g_mean_flux_over_error",
                &["phot_g_mean_flux_over_error", "g_flux_over_error", "g_snr"],
            ),
            (
                "phot_bp_rp_excess_factor",
                &["phot_bp_rp_excess_factor", "bp_rp_excess_factor"],
            ),
            ("ruwe", &["ruwe"]),
        ];
        for (field, names) in numeric_fields {
            let mut value = row.number(names)?;
            if value.is_none() {
                value = match *field {
                    "pmra_mas_yr" | "pmdec_mas_yr" => Some(0.),
                    "ref_epoch" => Some(2016.),
                    _ => None,
                };
            }
            object.insert((*field).into(), json!(value));
        }
        for (field, names) in [
            (
                "catalog_name",
                &["catalog_name", "catalog", "catalogue"][..],
            ),
            (
                "photometric_system",
                &["photometric_system", "mag_system", "system"][..],
            ),
            (
                "photometric_band",
                &["photometric_band", "band", "passband"][..],
            ),
        ] {
            let value = row.text(names).unwrap_or_else(|| {
                if is_gaia {
                    match field {
                        "catalog_name" => "Gaia DR3",
                        "photometric_system" => "Gaia Vega",
                        _ => "G",
                    }
                    .into()
                } else {
                    "unknown".into()
                }
            });
            object.insert(field.into(), json!(value));
        }
        let color_name = row
            .text(&["color_name", "color_band"])
            .or_else(|| fieldnames.contains("bp_rp").then(|| "BP-RP".into()));
        object.insert("color_name".into(), json!(color_name));
        let distance_source = row
            .text(&["distance_source", "distance_method"])
            .or_else(|| {
                row.named(&[
                    "distance_pc",
                    "distance_gspphot",
                    "distance",
                    "r_med_geo",
                    "r_med_photogeo",
                    "r_med_photogeometric",
                ])
                .and_then(|(field, _)| match field.as_str() {
                    "distance_gspphot" => Some("Gaia DR3 GSP-Phot".into()),
                    "r_med_geo" => Some("Bailer-Jones Gaia DR3 geometric posterior".into()),
                    "r_med_photogeo" | "r_med_photogeometric" => {
                        Some("Bailer-Jones Gaia DR3 photogeometric posterior".into())
                    }
                    _ => None,
                })
            });
        object.insert("distance_source".into(), json!(distance_source));
        let mg_source = row
            .text(&[
                "mg_gspphot_source",
                "mg_g_source",
                "model_absolute_magnitude_source",
            ])
            .or_else(|| {
                row.named(&["mg_gspphot", "mg_g", "gsp_phot_mg"])
                    .filter(|(field, _)| field == "mg_gspphot")
                    .map(|_| "Gaia DR3 GSP-Phot: mg_gspphot".to_owned())
            });
        object.insert("mg_gspphot_source".into(), json!(mg_source));
        let duplicate = row
            .text(&["duplicated_source", "duplicate_source"])
            .map(|v| match v.to_lowercase().as_str() {
                "true" | "t" | "1" | "yes" | "y" => Ok(true),
                "false" | "f" | "0" | "no" | "n" => Ok(false),
                _ => Err("invalid catalog duplicated_source"),
            })
            .transpose()?;
        object.insert("duplicated_source".into(), json!(duplicate));
        let visibility = row.number(&["visibility_periods_used", "visibility_periods"])?;
        if visibility.is_some_and(|v| v < 0. || v.fract() != 0. || v >= u64::MAX as f64) {
            return Err("visibility_periods_used must be a nonnegative integer in range".into());
        }
        object.insert(
            "visibility_periods_used".into(),
            json!(visibility.map(|v| v as u64)),
        );
        object.insert(
            "phot_variable_flag".into(),
            json!(row.text(&["phot_variable_flag", "variable_flag"])),
        );
        let (value, band, system, source) = extinction(&row)?;
        object.insert("extinction_mag".into(), json!(value));
        object.insert("extinction_band".into(), json!(band));
        object.insert("extinction_system".into(), json!(system));
        object.insert("extinction_source".into(), json!(source));
        let source: CatalogSource =
            serde_json::from_value(Value::Object(object)).map_err(|e| e.to_string())?;
        sources.push(
            source
                .validated()
                .map_err(|e| format!("catalog row {}: {e}", row.number))?,
        );
    }
    Ok(sources)
}
