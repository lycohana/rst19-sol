//! Native Gaia DR3 TAP cone query and CatalogSource-compatible CSV export.

use csv::{ReaderBuilder, Terminator, WriterBuilder};
use reqwest::blocking::Client;
use roxmltree::{Document, Node};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const DEFAULT_GAIA_TAP_SYNC_URL: &str = "https://gea.esac.esa.int/tap-server/tap/sync";

const REQUIRED_COLUMNS: &[&str] = &[
    "source_id",
    "ra",
    "dec",
    "ref_epoch",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_bp_rp_excess_factor",
    "parallax",
    "parallax_error",
    "ruwe",
    "duplicated_source",
    "visibility_periods_used",
    "phot_variable_flag",
    "distance_gspphot",
    "distance_gspphot_lower",
    "distance_gspphot_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
];
const MODEL_COLUMNS: &[&str] = &["mg_gspphot", "mg_gspphot_lower", "mg_gspphot_upper"];
const NUMERIC_COLUMNS: &[&str] = &[
    "ra",
    "dec",
    "ref_epoch",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_bp_rp_excess_factor",
    "parallax",
    "parallax_error",
    "ruwe",
    "visibility_periods_used",
    "distance_gspphot",
    "distance_gspphot_lower",
    "distance_gspphot_upper",
    "mg_gspphot",
    "mg_gspphot_lower",
    "mg_gspphot_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
];
const OUTPUT_COLUMNS: &[&str] = &[
    "source_id",
    "ra_deg",
    "dec_deg",
    "magnitude",
    "magnitude_error",
    "color",
    "color_name",
    "pmra_mas_yr",
    "pmdec_mas_yr",
    "ref_epoch",
    "parallax_mas",
    "parallax_error_mas",
    "distance_pc",
    "distance_lower_pc",
    "distance_upper_pc",
    "distance_source",
    "mg_gspphot",
    "mg_gspphot_lower",
    "mg_gspphot_upper",
    "mg_gspphot_source",
    "extinction_mag",
    "extinction_error_mag",
    "extinction_band",
    "extinction_system",
    "extinction_source",
    "catalog_name",
    "photometric_system",
    "photometric_band",
    "magnitude_source",
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_bp_rp_excess_factor",
    "parallax",
    "parallax_error",
    "ruwe",
    "duplicated_source",
    "visibility_periods_used",
    "phot_variable_flag",
    "distance_gspphot",
    "distance_gspphot_lower",
    "distance_gspphot_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
];

#[derive(Debug, Clone)]
struct Options {
    ra: f64,
    dec: f64,
    radius: f64,
    out: Option<PathBuf>,
    limit: Option<usize>,
    min_g_mag: Option<f64>,
    max_g_mag: Option<f64>,
    include_model: bool,
    timeout: f64,
    endpoint: String,
    dry_run: bool,
}

fn number(raw: &str, name: &str) -> Result<f64, String> {
    let value = raw
        .parse::<f64>()
        .map_err(|_| format!("{name} must be a finite number; got {raw:?}"))?;
    if !value.is_finite() {
        return Err(format!("{name} must be a finite number; got {raw:?}"));
    }
    Ok(value)
}

fn fmt_float(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let scientific = format!("{value:.14e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent = exponent.parse::<i32>().unwrap_or(0);
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let mut digits = mantissa.trim_start_matches(['-', '+']).replace('.', "");
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    if (-4..15).contains(&exponent) {
        let decimal = exponent + 1;
        if decimal <= 0 {
            return format!("{sign}0.{}{}", "0".repeat((-decimal) as usize), digits);
        }
        if decimal as usize >= digits.len() {
            return format!(
                "{sign}{}{}",
                digits,
                "0".repeat(decimal as usize - digits.len())
            );
        }
        let point = decimal as usize;
        return format!("{sign}{}.{}", &digits[..point], &digits[point..]);
    }
    let fraction = digits.get(1..).unwrap_or("");
    let fraction = fraction.trim_end_matches('0');
    let signed_exp = if exponent >= 0 {
        format!("+{exponent}")
    } else {
        format!("-{:02}", exponent.unsigned_abs())
    };
    if fraction.is_empty() {
        format!("{sign}{}e{}", &digits[..1], signed_exp)
    } else {
        format!("{sign}{}.{}e{}", &digits[..1], fraction, signed_exp)
    }
}

pub fn build_adql(
    ra: f64,
    dec: f64,
    radius: f64,
    limit: Option<usize>,
    min_g_mag: Option<f64>,
    max_g_mag: Option<f64>,
    include_model: bool,
) -> Result<String, String> {
    if !ra.is_finite() || !(0.0..360.0).contains(&ra) {
        return Err(format!("ra must be in [0, 360) degrees; got {ra}"));
    }
    if !dec.is_finite() || !(-90.0..=90.0).contains(&dec) {
        return Err(format!("dec must be in [-90, 90] degrees; got {dec}"));
    }
    if !radius.is_finite() || !(0.0 < radius && radius <= 180.0) {
        return Err(format!(
            "radius must be greater than 0 and at most 180 degrees; got {radius}"
        ));
    }
    if limit == Some(0) {
        return Err("limit must be a positive integer".into());
    }
    for (name, value) in [("min_g_mag", min_g_mag), ("max_g_mag", max_g_mag)] {
        if let Some(value) = value {
            if !value.is_finite() || !(-50.0..=50.0).contains(&value) {
                return Err(format!(
                    "{name} must be in [-50, 50] magnitudes; got {value}"
                ));
            }
        }
    }
    if min_g_mag.zip(max_g_mag).is_some_and(|(min, max)| min > max) {
        return Err("min_g_mag cannot be greater than max_g_mag".into());
    }

    let columns = REQUIRED_COLUMNS.iter().map(|name| format!("g.{name}"));
    let selected = if include_model {
        columns
            .chain(
                MODEL_COLUMNS
                    .iter()
                    .map(|name| format!("ap.{name} AS {name}")),
            )
            .collect::<Vec<_>>()
            .join(",\n       ")
    } else {
        REQUIRED_COLUMNS.join(",\n       ")
    };
    let top = limit
        .map(|count| format!("TOP {count} "))
        .unwrap_or_default();
    let (from_clause, ra_column, dec_column, mag_column) = if include_model {
        (
            "FROM gaiadr3.gaia_source AS g\nLEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap\n  ON g.source_id = ap.source_id".to_owned(),
            "g.ra", "g.dec", "g.phot_g_mean_mag",
        )
    } else {
        (
            "FROM gaiadr3.gaia_source".to_owned(),
            "ra",
            "dec",
            "phot_g_mean_mag",
        )
    };
    let mut conditions = vec![format!(
        "1 = CONTAINS(\n  POINT('ICRS', {ra_column}, {dec_column}),\n  CIRCLE('ICRS', {}, {}, {})\n)",
        fmt_float(ra), fmt_float(dec), fmt_float(radius)
    )];
    if let Some(value) = min_g_mag {
        conditions.push(format!("{mag_column} >= {}", fmt_float(value)));
    }
    if let Some(value) = max_g_mag {
        conditions.push(format!("{mag_column} <= {}", fmt_float(value)));
    }
    Ok(format!(
        "SELECT {top}{selected}\n{from_clause}\nWHERE {}",
        conditions.join("\n  AND ")
    ))
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut values = HashMap::<String, String>::new();
    let mut flags = HashSet::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--help" || arg == "-h" {
            return Err("__help__".into());
        }
        if arg == "--dry-run" || arg == "--include-gspphot-model" {
            flags.insert(arg.clone());
            i += 1;
            continue;
        }
        if !arg.starts_with("--") {
            return Err(format!("unexpected argument: {arg}"));
        }
        let (name, inline) = arg
            .split_once('=')
            .map(|(a, b)| (a, Some(b.to_owned())))
            .unwrap_or((arg.as_str(), None));
        if ![
            "--ra",
            "--dec",
            "--radius",
            "--out",
            "--limit",
            "--min-g-mag",
            "--max-g-mag",
            "--timeout",
            "--endpoint",
        ]
        .contains(&name)
        {
            return Err(format!("unrecognized argument: {name}"));
        }
        let value = if let Some(value) = inline {
            value
        } else {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))?
        };
        values.insert(name.to_owned(), value);
        i += 1;
    }
    let required = |key: &str| values.get(key).ok_or_else(|| format!("{key} is required"));
    let ra = number(required("--ra")?, "ra")?;
    let dec = number(required("--dec")?, "dec")?;
    let radius = number(required("--radius")?, "radius")?;
    let limit = values
        .get("--limit")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("limit must be a positive integer; got {value:?}"))
        })
        .transpose()?;
    let min_g_mag = values
        .get("--min-g-mag")
        .map(|value| number(value, "min_g_mag"))
        .transpose()?;
    let max_g_mag = values
        .get("--max-g-mag")
        .map(|value| number(value, "max_g_mag"))
        .transpose()?;
    let timeout = values
        .get("--timeout")
        .map(|value| number(value, "timeout"))
        .transpose()?
        .unwrap_or(30.0);
    if timeout <= 0.0 {
        return Err(format!(
            "timeout must be greater than 0 seconds; got {timeout}"
        ));
    }
    let endpoint = values
        .get("--endpoint")
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| DEFAULT_GAIA_TAP_SYNC_URL.into());
    let url = reqwest::Url::parse(&endpoint)
        .map_err(|_| format!("endpoint must be an HTTP(S) URL; got {endpoint:?}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(format!("endpoint must be an HTTP(S) URL; got {endpoint:?}"));
    }
    build_adql(
        ra,
        dec,
        radius,
        limit,
        min_g_mag,
        max_g_mag,
        flags.contains("--include-gspphot-model"),
    )?;
    Ok(Options {
        ra,
        dec,
        radius,
        out: values.get("--out").map(PathBuf::from),
        limit,
        min_g_mag,
        max_g_mag,
        include_model: flags.contains("--include-gspphot-model"),
        timeout,
        endpoint,
        dry_run: flags.contains("--dry-run"),
    })
}

fn request_url(options: &Options, adql: &str) -> Result<reqwest::Url, String> {
    let mut url =
        reqwest::Url::parse(&options.endpoint).map_err(|e| format!("invalid TAP endpoint: {e}"))?;
    {
        let mut query = url.query_pairs_mut();
        query
            .append_pair("REQUEST", "doQuery")
            .append_pair("LANG", "ADQL")
            .append_pair("FORMAT", "csv")
            .append_pair("PHASE", "RUN")
            .append_pair("QUERY", adql);
        if let Some(limit) = options.limit {
            query.append_pair("MAXREC", &limit.to_string());
        }
    }
    Ok(url)
}

fn local_name<'a>(node: Node<'a, '_>) -> &'a str {
    node.tag_name().name()
}

fn parse_votable(text: &str) -> Result<Vec<HashMap<String, String>>, String> {
    let upper = text.to_ascii_uppercase();
    if upper.contains("<!DOCTYPE") || upper.contains("<!ENTITY") {
        return Err("Gaia VOTable response must not contain a DTD or entity declaration".into());
    }
    let doc =
        Document::parse(text).map_err(|e| format!("Gaia VOTable response is invalid XML: {e}"))?;
    let root = doc.root_element();
    if local_name(root) != "VOTABLE" {
        return Err("Gaia XML response is not a VOTable (possibly an HTML error page)".into());
    }
    let resources = root
        .descendants()
        .filter(|node| {
            local_name(*node) == "RESOURCE"
                && node
                    .attribute("type")
                    .unwrap_or("")
                    .eq_ignore_ascii_case("results")
        })
        .collect::<Vec<_>>();
    if resources.len() != 1 {
        return Err("Gaia VOTable must contain exactly one TAP results resource".into());
    }
    let resource = resources[0];
    let statuses = resource
        .descendants()
        .filter(|node| {
            local_name(*node) == "INFO"
                && node
                    .attribute("name")
                    .unwrap_or("")
                    .eq_ignore_ascii_case("QUERY_STATUS")
        })
        .collect::<Vec<_>>();
    if statuses.is_empty() {
        return Err("Gaia VOTable is missing TAP QUERY_STATUS".into());
    }
    for status in statuses {
        let value = status.attribute("value").unwrap_or("").to_ascii_uppercase();
        if value != "OK" {
            return Err(format!(
                "Gaia TAP QUERY_STATUS={value}: {}; result is not complete",
                status.text().unwrap_or("").trim()
            ));
        }
    }
    let tables = resource
        .children()
        .filter(|node| node.is_element() && local_name(*node) == "TABLE")
        .collect::<Vec<_>>();
    if tables.len() != 1 {
        return Err("Gaia VOTable must contain exactly one result table".into());
    }
    let table = tables[0];
    let headers = table
        .children()
        .filter(|node| node.is_element() && local_name(*node) == "FIELD")
        .map(|field| {
            field
                .attribute("name")
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        })
        .collect::<Vec<_>>();
    validate_headers(&headers, "VOTable")?;
    let data_nodes = table
        .descendants()
        .filter(|node| local_name(*node) == "TABLEDATA")
        .collect::<Vec<_>>();
    if data_nodes.len() != 1 {
        return Err("Gaia VOTable requires TABLEDATA; binary encodings are not supported".into());
    }
    let mut rows = Vec::new();
    for (row_index, row) in data_nodes[0]
        .children()
        .filter(|node| node.is_element())
        .enumerate()
    {
        if local_name(row) != "TR" {
            return Err(format!("Gaia VOTable row {}: expected TR", row_index + 1));
        }
        let cells = row
            .children()
            .filter(|node| node.is_element())
            .collect::<Vec<_>>();
        if cells.len() != headers.len() || cells.iter().any(|cell| local_name(*cell) != "TD") {
            return Err(format!(
                "Gaia VOTable row {}: column count/type mismatch",
                row_index + 1
            ));
        }
        rows.push(
            headers
                .iter()
                .zip(cells)
                .map(|(header, cell)| (header.clone(), cell.text().unwrap_or("").trim().to_owned()))
                .collect(),
        );
    }
    validate_rows(rows, "VOTable")
}

fn validate_headers(headers: &[String], format: &str) -> Result<(), String> {
    let mut seen = HashSet::<String>::new();
    for header in headers {
        if header.is_empty() {
            return Err(format!(
                "Gaia {format} response contains an empty column name"
            ));
        }
        if !seen.insert(header.clone()) {
            return Err(format!(
                "Gaia {format} response contains duplicate column: {header}"
            ));
        }
    }
    let missing = REQUIRED_COLUMNS
        .iter()
        .filter(|column| !seen.contains(**column))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Gaia {format} response is missing required columns: {}; available columns: {}",
            missing.join(", "),
            headers.join(", ")
        ));
    }
    Ok(())
}

fn validate_rows(
    rows: Vec<HashMap<String, String>>,
    format: &str,
) -> Result<Vec<HashMap<String, String>>, String> {
    let mut seen_ids = HashSet::new();
    for (index, row) in rows.iter().enumerate() {
        let row_number = index + 2;
        let id = row
            .get("source_id")
            .map(String::as_str)
            .unwrap_or("")
            .trim();
        if id.is_empty() {
            return Err(format!("Gaia {format} row {row_number}: missing source_id"));
        }
        if !seen_ids.insert(id.to_owned()) {
            return Err(format!(
                "Gaia {format} row {row_number}: duplicate source_id={id:?}"
            ));
        }
        for field in NUMERIC_COLUMNS {
            if let Some(value) = row.get(*field).filter(|value| !value.is_empty()) {
                let parsed = value.parse::<f64>().map_err(|_| {
                    format!("Gaia {format} row {row_number}: invalid numeric {field}={value:?}")
                })?;
                if !parsed.is_finite() {
                    return Err(format!(
                        "Gaia {format} row {row_number}: non-finite numeric {field}={value:?}"
                    ));
                }
            }
        }
        let ra = row
            .get("ra")
            .and_then(|value| value.parse::<f64>().ok())
            .ok_or_else(|| format!("Gaia {format} row {row_number}: missing ra"))?;
        let dec = row
            .get("dec")
            .and_then(|value| value.parse::<f64>().ok())
            .ok_or_else(|| format!("Gaia {format} row {row_number}: missing dec"))?;
        if !(0.0..=360.0).contains(&ra) {
            return Err(format!(
                "Gaia {format} row {row_number}: ra out of range: {ra}"
            ));
        }
        if !(-90.0..=90.0).contains(&dec) {
            return Err(format!(
                "Gaia {format} row {row_number}: dec out of range: {dec}"
            ));
        }
    }
    Ok(rows)
}

fn parse_response(bytes: &[u8]) -> Result<Vec<HashMap<String, String>>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| "Gaia response is not valid UTF-8".to_owned())?
        .trim_start_matches('\u{feff}');
    if text.trim_start().starts_with('<') {
        return parse_votable(text);
    }
    let mut reader = ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let raw_headers = reader
        .headers()
        .map_err(|e| format!("Gaia CSV response has no header: {e}"))?
        .clone();
    let headers = raw_headers
        .iter()
        .map(|raw| {
            raw.trim()
                .trim_start_matches('\u{feff}')
                .to_ascii_lowercase()
        })
        .collect::<Vec<_>>();
    validate_headers(&headers, "CSV")?;
    let mut rows = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let record = record.map_err(|e| format!("Gaia CSV row {}: {e}", index + 2))?;
        if record.len() > headers.len() {
            return Err(format!(
                "Gaia CSV row {} has more fields than its header",
                index + 2
            ));
        }
        let row = headers
            .iter()
            .enumerate()
            .map(|(column, header)| {
                (
                    header.clone(),
                    record.get(column).unwrap_or("").trim().to_owned(),
                )
            })
            .collect();
        rows.push(row);
    }
    validate_rows(rows, "CSV")
}

fn cell(row: &HashMap<String, String>, name: &str) -> String {
    row.get(name).cloned().unwrap_or_default()
}
fn numeric(row: &HashMap<String, String>, name: &str) -> Option<f64> {
    row.get(name)
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
}

fn normalized_output_row(row: &HashMap<String, String>) -> HashMap<String, String> {
    let mut output = row.clone();
    let mag = cell(row, "phot_g_mean_mag");
    let flux = numeric(row, "phot_g_mean_flux");
    let flux_error = numeric(row, "phot_g_mean_flux_error");
    let magnitude_error = match (flux, flux_error) {
        (Some(flux), Some(error)) if flux > 0.0 && error >= 0.0 => {
            fmt_float((2.5 / std::f64::consts::LN_10) * error / flux)
        }
        _ => String::new(),
    };
    let bp = numeric(row, "phot_bp_mean_mag");
    let rp = numeric(row, "phot_rp_mean_mag");
    let distance = cell(row, "distance_gspphot");
    let ag = cell(row, "ag_gspphot");
    let ag_error = match (
        numeric(row, "ag_gspphot_lower"),
        numeric(row, "ag_gspphot_upper"),
    ) {
        (Some(lower), Some(upper)) if upper >= lower => fmt_float(0.5 * (upper - lower)),
        _ => String::new(),
    };
    let mg = cell(row, "mg_gspphot");
    let fields = [
        ("ra_deg", cell(row, "ra")),
        ("dec_deg", cell(row, "dec")),
        ("magnitude", mag.clone()),
        ("magnitude_error", magnitude_error.clone()),
        ("phot_g_mean_mag_error", magnitude_error),
        (
            "magnitude_source",
            if mag.is_empty() {
                String::new()
            } else {
                "Gaia DR3 phot_g_mean_mag".into()
            },
        ),
        ("pmra_mas_yr", cell(row, "pmra")),
        ("pmdec_mas_yr", cell(row, "pmdec")),
        ("parallax_mas", cell(row, "parallax")),
        ("parallax_error_mas", cell(row, "parallax_error")),
        ("distance_pc", distance.clone()),
        ("distance_lower_pc", cell(row, "distance_gspphot_lower")),
        ("distance_upper_pc", cell(row, "distance_gspphot_upper")),
        (
            "distance_source",
            if distance.is_empty() {
                String::new()
            } else {
                "Gaia DR3 GSP-Phot".into()
            },
        ),
        ("mg_gspphot", mg.clone()),
        (
            "mg_gspphot_source",
            if mg.is_empty() {
                String::new()
            } else {
                "Gaia DR3 GSP-Phot: mg_gspphot".into()
            },
        ),
        ("extinction_mag", ag.clone()),
        ("extinction_error_mag", ag_error),
        (
            "extinction_band",
            if ag.is_empty() {
                "unknown".into()
            } else {
                "G".into()
            },
        ),
        (
            "extinction_system",
            if ag.is_empty() {
                "unknown".into()
            } else {
                "Gaia".into()
            },
        ),
        (
            "extinction_source",
            if ag.is_empty() {
                "unknown".into()
            } else {
                "Gaia DR3 GSP-Phot: ag_gspphot".into()
            },
        ),
        ("catalog_name", "Gaia DR3".into()),
        ("photometric_system", "Gaia Vega".into()),
        ("photometric_band", "G".into()),
        (
            "color",
            match (bp, rp) {
                (Some(bp), Some(rp)) => fmt_float(bp - rp),
                _ => String::new(),
            },
        ),
        (
            "color_name",
            if bp.is_some() && rp.is_some() {
                "BP-RP".into()
            } else {
                String::new()
            },
        ),
    ];
    for (key, value) in fields {
        output.insert(key.into(), value);
    }
    output
}

pub(crate) fn write_catalog(rows: &[HashMap<String, String>], output: &Path) -> Result<(), String> {
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut writer = WriterBuilder::new()
        .terminator(Terminator::Any(b'\n'))
        .from_path(output)
        .map_err(|e| e.to_string())?;
    writer
        .write_record(OUTPUT_COLUMNS)
        .map_err(|e| e.to_string())?;
    for row in rows {
        let normalized = normalized_output_row(row);
        writer
            .write_record(
                OUTPUT_COLUMNS
                    .iter()
                    .map(|column| normalized.get(*column).map(String::as_str).unwrap_or("")),
            )
            .map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn fetch(options: &Options, adql: &str) -> Result<Vec<HashMap<String, String>>, String> {
    let url = request_url(options, adql)?;
    let timeout = Duration::try_from_secs_f64(options.timeout).map_err(|_| {
        format!(
            "timeout is outside the supported duration range: {}",
            options.timeout
        )
    })?;
    let client = Client::builder()
        .timeout(timeout)
        .user_agent("rst19-gaia-remote/1.0")
        .build()
        .map_err(|e| format!("could not create Gaia HTTP client: {e}"))?;
    let response = client
        .get(url.clone())
        .header(
            reqwest::header::ACCEPT,
            "text/csv, text/plain;q=0.9, application/xml;q=0.8, */*;q=0.1",
        )
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                format!(
                    "Gaia TAP request timed out after {} seconds: {url}",
                    fmt_float(options.timeout)
                )
            } else {
                format!("Gaia TAP network error for {url}: {e}")
            }
        })?;
    let status = response.status();
    let body = response
        .bytes()
        .map_err(|e| format!("Gaia TAP response read failed: {e}"))?;
    if !status.is_success() {
        let preview = String::from_utf8_lossy(&body)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let preview = preview.chars().take(500).collect::<String>();
        return Err(format!(
            "Gaia TAP HTTP error {} for {url}: {preview}",
            status.as_u16()
        ));
    }
    let mut rows = parse_response(&body)?;
    if let Some(limit) = options.limit {
        rows.truncate(limit);
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_rows(
    ra: f64,
    dec: f64,
    radius: f64,
    limit: Option<usize>,
    min_g_mag: Option<f64>,
    max_g_mag: Option<f64>,
    include_model: bool,
    timeout: f64,
    endpoint: &str,
) -> Result<Vec<HashMap<String, String>>, String> {
    let options = Options {
        ra,
        dec,
        radius,
        out: None,
        limit,
        min_g_mag,
        max_g_mag,
        include_model,
        timeout,
        endpoint: endpoint.trim().to_owned(),
        dry_run: false,
    };
    let adql = build_adql(ra, dec, radius, limit, min_g_mag, max_g_mag, include_model)?;
    fetch(&options, &adql)
}

fn metadata(options: &Options, adql: &str, row_count: usize) -> Value {
    json!({
        "catalog":"Gaia DR3",
        "table": if options.include_model { "gaiadr3.gaia_source LEFT OUTER JOIN gaiadr3.astrophysical_parameters" } else { "gaiadr3.gaia_source" },
        "query_center_ra_deg":options.ra,
        "query_center_dec_deg":options.dec,
        "query_radius_deg":options.radius,
        "limit":options.limit,
        "min_g_mag":options.min_g_mag,
        "max_g_mag":options.max_g_mag,
        "row_count":row_count,
        "tap_maxrec":options.limit,
        "endpoint":options.endpoint,
        "magnitude_error_source":"local first-order approximation from phot_g_mean_flux_error; Gaia does not publish a symmetric phot_g_mean_mag_error column",
        "gspphot_fields":if options.include_model { "distance_gspphot/ag_gspphot and their 16th/84th percentile bounds are preserved when available; they are model-based single-star estimates; mg_gspphot and its 16th/84th percentile bounds are included from gaiadr3.astrophysical_parameters" } else { "distance_gspphot/ag_gspphot and their 16th/84th percentile bounds are preserved when available; they are model-based single-star estimates; mg_gspphot is omitted unless --include-gspphot-model is set" },
        "coverage_warning":if options.radius > 1.0 { Some("broad cone result requires completeness audit") } else { None },
        "query":adql,
    })
}

fn execute(options: &Options) -> Result<Value, String> {
    let adql = build_adql(
        options.ra,
        options.dec,
        options.radius,
        options.limit,
        options.min_g_mag,
        options.max_g_mag,
        options.include_model,
    )?;
    if options.dry_run {
        return Ok(json!({"query":adql,"network":false}));
    }
    let output = options
        .out
        .as_ref()
        .ok_or("--out is required unless --dry-run is used")?;
    let rows = fetch(options, &adql)?;
    write_catalog(&rows, output)?;
    let metadata_path = PathBuf::from(format!("{}.meta.json", output.to_string_lossy()));
    let mut bytes = serde_json::to_vec_pretty(&metadata(options, &adql, rows.len()))
        .map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    fs::write(&metadata_path, bytes).map_err(|e| e.to_string())?;
    Ok(
        json!({"csv":output.to_string_lossy(),"metadata":metadata_path.to_string_lossy(),"row_count":rows.len()}),
    )
}

fn help() -> &'static str {
    "按天空坐标查询公共 Gaia DR3 星表并导出 CSV\n\nUSAGE: rst19-gaia --ra DEG --dec DEG --radius DEG [OPTIONS]\n\nOptions:\n  --out PATH\n  --limit N\n  --min-g-mag MAG\n  --max-g-mag MAG\n  --include-gspphot-model\n  --timeout SECONDS  [default: 30]\n  --endpoint URL\n  --dry-run\n"
}

pub fn run_cli(args: &[String]) -> i32 {
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(error) if error == "__help__" => {
            print!("{}", help());
            return 0;
        }
        Err(error) => {
            eprintln!("gaia-remote: {error}");
            return 2;
        }
    };
    if options.radius > 1.0 && !options.dry_run {
        eprintln!("gaia-remote: warning: broad cone queries may be truncated by the TAP service; verify row count and magnitude distribution or query smaller tiles");
    }
    match execute(&options) {
        Ok(result) if options.dry_run => {
            println!("{}", result["query"].as_str().unwrap_or(""));
            0
        }
        Ok(result) => {
            eprintln!(
                "Gaia DR3: wrote {} rows to {} (query metadata: {})",
                result["row_count"],
                result["csv"].as_str().unwrap_or(""),
                result["metadata"].as_str().unwrap_or("")
            );
            0
        }
        Err(error) => {
            eprintln!("gaia-remote: {error}");
            2
        }
    }
}

pub fn run_desktop(args: &[String]) -> Result<Value, String> {
    let options = parse_options(args)?;
    execute(&options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn sample_csv() -> String {
        include_str!("../tests/fixtures/gaia_remote_parity/response.csv").to_owned()
    }

    #[test]
    fn adql_is_numeric_and_injection_safe() {
        let query = build_adql(12.5, -1.25, 0.4, Some(100), Some(5.0), Some(18.0), false).unwrap();
        assert!(query.starts_with("SELECT TOP 100 source_id,\n       ra,"));
        assert!(query.contains("CIRCLE('ICRS', 12.5, -1.25, 0.4)"));
        assert!(query.contains("phot_g_mean_mag >= 5"));
        let model_query = build_adql(12.5, -1.25, 0.4, None, None, None, true).unwrap();
        assert!(model_query.contains("LEFT OUTER JOIN gaiadr3.astrophysical_parameters"));
        assert!(model_query.contains("ap.mg_gspphot AS mg_gspphot"));
        assert!(build_adql(12.0, 0.0, 1.0, None, None, None, false).is_ok());
        assert!(build_adql(360.0, 0.0, 1.0, None, None, None, false).is_err());
        assert!(build_adql(12.0, 0.0, 1.0, Some(0), None, None, false).is_err());
    }

    #[test]
    fn adql_number_format_matches_python_significant_digit_style() {
        for (value, expected) in [
            (1.25, "1.25"),
            (-1.25, "-1.25"),
            (0.0001, "0.0001"),
            (-0.0000123, "-1.23e-05"),
            (1.0e15, "1e+15"),
            (-1.0e15, "-1e+15"),
        ] {
            assert_eq!(fmt_float(value), expected);
        }
    }

    #[test]
    fn csv_normalization_produces_catalog_aliases() {
        let rows = parse_response(sample_csv().as_bytes()).unwrap();
        assert_eq!(rows.len(), 1);
        let normalized = normalized_output_row(&rows[0]);
        assert_eq!(normalized["source_id"], "123");
        assert_eq!(normalized["magnitude"], "12.5");
        assert_eq!(normalized["distance_source"], "Gaia DR3 GSP-Phot");
        assert_eq!(normalized["mg_gspphot"], "1.5");
        assert_eq!(
            normalized["mg_gspphot_source"],
            "Gaia DR3 GSP-Phot: mg_gspphot"
        );
        assert_eq!(normalized["extinction_band"], "G");
        assert_eq!(normalized["color"], "1.3");
    }

    #[test]
    fn votable_tabledata_and_query_status_are_checked() {
        let headers = REQUIRED_COLUMNS
            .iter()
            .map(|name| format!("<FIELD name=\"{name}\" datatype=\"char\" arraysize=\"*\"/>"))
            .collect::<String>();
        let values = [
            "1",
            "2",
            "3",
            "2016",
            "0",
            "0",
            "12",
            "100",
            "1",
            "100",
            "13",
            "12",
            "1",
            "1",
            "0.1",
            "1",
            "false",
            "9",
            "NOT_AVAILABLE",
            "10",
            "9",
            "11",
            "0.2",
            "0.1",
            "0.3",
            "0.1",
        ];
        let cells = values
            .iter()
            .map(|value| format!("<TD>{value}</TD>"))
            .collect::<String>();
        let xml = format!("<VOTABLE><RESOURCE type=\"results\"><TABLE>{headers}<DATA><TABLEDATA><TR>{cells}</TR></TABLEDATA></DATA></TABLE><INFO name=\"QUERY_STATUS\" value=\"OK\"/></RESOURCE></VOTABLE>");
        assert_eq!(parse_votable(&xml).unwrap().len(), 1);
        assert!(
            parse_votable(&xml.replace("value=\"OK\"", "value=\"OVERFLOW\""))
                .unwrap_err()
                .contains("not complete")
        );
    }

    #[test]
    fn native_cli_queries_local_tap_and_writes_both_artifacts() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let body = sample_csv();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 8192];
            let count = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..count]);
            assert!(request.starts_with("GET /tap/sync?"));
            assert!(request.contains("REQUEST=doQuery"));
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/csv\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            stream.write_all(response.as_bytes()).unwrap();
        });
        let base = std::env::temp_dir().join(format!("rst19-gaia-native-{}", std::process::id()));
        let out = base.join("catalog.csv");
        let args = vec![
            "--ra".into(),
            "10".into(),
            "--dec".into(),
            "2".into(),
            "--radius".into(),
            "0.2".into(),
            "--endpoint".into(),
            format!("http://{address}/tap/sync"),
            "--out".into(),
            out.to_string_lossy().into_owned(),
        ];
        let value = run_desktop(&args).unwrap();
        server.join().unwrap();
        assert_eq!(value["row_count"], 1);
        let csv = fs::read_to_string(&out).unwrap();
        assert!(csv.starts_with("source_id,ra_deg,dec_deg,magnitude,"));
        assert!(csv.contains("123,45.5,-12.25,12.5,"));
        assert_eq!(
            fs::read(&out).unwrap(),
            include_bytes!("../tests/fixtures/gaia_remote_parity/reference.csv")
        );
        assert!(PathBuf::from(value["metadata"].as_str().unwrap()).is_file());
    }
}
