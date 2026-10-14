//! Native, auditable Gaia DR3 reference-catalog downloads.
//!
//! This is the Rust counterpart of `rst19.public_catalog`.  It keeps the
//! nominal camera field of view separate from a measured WCS and writes a
//! sidecar that records the returned coverage and truncation state.

use crate::{fits, gaia_tiled, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

pub const DEFAULT_CAMERA_FOV_WIDTH_DEG: f64 = 9.78;
pub const DEFAULT_CAMERA_FOV_HEIGHT_DEG: f64 = 9.78;
pub const DEFAULT_GAIA_MIN_G_MAG: f64 = 5.0;
pub const DEFAULT_GAIA_MAX_G_MAG: f64 = 13.5;
pub const DEFAULT_GAIA_TILE_RADIUS_DEG: f64 = 7.0;
pub const DEFAULT_GAIA_TILE_LIMIT: usize = 50_000;
pub const DEFAULT_GAIA_MAX_SUBDIVIDE_DEPTH: usize = 2;
pub const DEFAULT_GAIA_MAX_QUERIES: usize = 2_000;
pub const DEFAULT_GAIA_TIMEOUT_SECONDS: f64 = 30.0;

#[derive(Debug, Clone)]
pub struct PublicCatalogOptions {
    pub center_ra_deg: f64,
    pub center_dec_deg: f64,
    pub output_path: PathBuf,
    pub frame_path: Option<PathBuf>,
    pub search_radius_deg: Option<f64>,
    pub fov_width_deg: f64,
    pub fov_height_deg: f64,
    pub min_g_mag: Option<f64>,
    pub max_g_mag: Option<f64>,
    pub tile_radius_deg: f64,
    pub tile_limit: Option<usize>,
    pub max_subdivide_depth: usize,
    pub max_queries: usize,
    pub include_gspphot_model: bool,
    pub timeout_seconds: f64,
    pub endpoint: String,
}

impl Default for PublicCatalogOptions {
    fn default() -> Self {
        Self {
            center_ra_deg: 0.0,
            center_dec_deg: 0.0,
            output_path: PathBuf::new(),
            frame_path: None,
            search_radius_deg: None,
            fov_width_deg: DEFAULT_CAMERA_FOV_WIDTH_DEG,
            fov_height_deg: DEFAULT_CAMERA_FOV_HEIGHT_DEG,
            min_g_mag: Some(DEFAULT_GAIA_MIN_G_MAG),
            max_g_mag: Some(DEFAULT_GAIA_MAX_G_MAG),
            tile_radius_deg: DEFAULT_GAIA_TILE_RADIUS_DEG,
            tile_limit: Some(DEFAULT_GAIA_TILE_LIMIT),
            max_subdivide_depth: DEFAULT_GAIA_MAX_SUBDIVIDE_DEPTH,
            max_queries: DEFAULT_GAIA_MAX_QUERIES,
            include_gspphot_model: false,
            timeout_seconds: DEFAULT_GAIA_TIMEOUT_SECONDS,
            endpoint: crate::gaia_remote::DEFAULT_GAIA_TAP_SYNC_URL.into(),
        }
    }
}

/// Return the half diagonal used to cover a rectangular camera field.
///
/// This is only a spherical catalog-query radius. It is not a WCS plate scale
/// and does not assert that the corners have identical projection.
pub fn camera_footprint_radius_deg(width_deg: f64, height_deg: f64) -> Result<f64> {
    for (name, value) in [("width_deg", width_deg), ("height_deg", height_deg)] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be a positive number"));
        }
    }
    Ok(0.5 * width_deg.hypot(height_deg))
}

fn sha256_and_size(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut byte_count = 0_u64;
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        byte_count += count as u64;
    }
    Ok((format!("{:x}", hasher.finalize()), byte_count))
}

fn audit_path(output_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.meta.json", output_path.to_string_lossy()))
}

fn audit_value(
    options: &PublicCatalogOptions,
    output_path: &Path,
    radius: f64,
    tiled_result: &Value,
) -> Result<Value> {
    let (csv_sha256, csv_byte_count) = sha256_and_size(output_path)?;
    let row_count = tiled_result["rows"].as_array().map_or(0, Vec::len);
    let mut tiled_audit = tiled_result.clone();
    tiled_audit
        .as_object_mut()
        .ok_or("tiled result is not an object")?
        .remove("rows");
    let gspphot_fields = if options.include_gspphot_model {
        "distance_gspphot and ag_gspphot, with 16th/84th percentile bounds; these are Gaia DR3 GSP-Phot model estimates under a single-star assumption; mg_gspphot and its 16th/84th percentile bounds are included"
    } else {
        "distance_gspphot and ag_gspphot, with 16th/84th percentile bounds; these are Gaia DR3 GSP-Phot model estimates under a single-star assumption; mg_gspphot is not requested"
    };
    Ok(json!({
        "catalog":"Gaia DR3",
        "endpoint":options.endpoint,
        "table":if options.include_gspphot_model {"gaiadr3.gaia_source LEFT OUTER JOIN gaiadr3.astrophysical_parameters"} else {"gaiadr3.gaia_source"},
        "frame_path":options.frame_path.as_ref().map(|path| path.to_string_lossy()),
        "output_path":output_path.to_string_lossy(),
        "audit_path":audit_path(output_path).to_string_lossy(),
        "center_ra_deg":options.center_ra_deg,
        "center_dec_deg":options.center_dec_deg,
        "search_radius_deg":radius,
        "min_g_mag":options.min_g_mag,
        "max_g_mag":options.max_g_mag,
        "row_count":row_count,
        "csv_row_count":row_count,
        "csv_sha256":csv_sha256,
        "csv_byte_count":csv_byte_count,
        "complete":tiled_result["complete"],
        "initial_tile_count":tiled_result["initial_tile_count"],
        "queried_tile_count":tiled_result["queried_tile_count"],
        "duplicate_count":tiled_result["duplicate_count"],
        "outside_scope_count":tiled_result["outside_scope_count"],
        "tiled":tiled_audit,
        "provenance":{
            "source":"Gaia DR3 gaiadr3.gaia_source",
            "photometric_system":"Gaia Vega",
            "photometric_band":"G",
            "magnitude_error":"local first-order approximation from phot_g_mean_flux_error; Gaia does not publish a symmetric phot_g_mean_mag_error column",
            "gspphot_fields":gspphot_fields
        },
        "interpretation":"This is an external reference catalog for identity and apparent-magnitude calibration. It is not the camera-specific 450-750 nm response or an absolute photometric zero point."
    }))
}

/// Download a Gaia DR3 catalog centered on explicit sky coordinates, then
/// write the row CSV and the same compact completeness audit used by the
/// Python `PublicCatalogDownload.as_dict()` API.
pub fn download_public_gaia_catalog(options: &PublicCatalogOptions) -> Result<Value> {
    if !options.center_ra_deg.is_finite() || !(0.0..360.0).contains(&options.center_ra_deg) {
        return Err("center_ra_deg must be finite and in [0, 360)".into());
    }
    if !options.center_dec_deg.is_finite() || !(-90.0..=90.0).contains(&options.center_dec_deg) {
        return Err("center_dec_deg must be finite and in [-90, 90]".into());
    }
    let radius = match options.search_radius_deg {
        Some(value) => value,
        None => camera_footprint_radius_deg(options.fov_width_deg, options.fov_height_deg)?,
    };
    if !radius.is_finite() || radius <= 0.0 || radius > 180.0 {
        return Err("search_radius_deg must be a positive number at most 180 degrees".into());
    }
    if !options
        .output_path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("csv"))
    {
        return Err("output_path must use the .csv suffix".into());
    }
    if options.tile_limit == Some(0) {
        return Err("tile_limit must be a positive integer or None".into());
    }
    if options.max_queries == 0 {
        return Err("max_queries must be a positive integer".into());
    }
    if !options.timeout_seconds.is_finite() || options.timeout_seconds <= 0.0 {
        return Err("timeout must be greater than 0 seconds".into());
    }

    if let Some(parent) = options
        .output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut args = vec![
        "--ra".into(),
        options.center_ra_deg.to_string(),
        "--dec".into(),
        options.center_dec_deg.to_string(),
        "--radius".into(),
        radius.to_string(),
        "--out".into(),
        options.output_path.to_string_lossy().into_owned(),
        "--tile-radius".into(),
        options.tile_radius_deg.to_string(),
        "--max-depth".into(),
        options.max_subdivide_depth.to_string(),
        "--max-queries".into(),
        options.max_queries.to_string(),
        "--timeout".into(),
        options.timeout_seconds.to_string(),
        "--endpoint".into(),
        options.endpoint.clone(),
    ];
    if let Some(limit) = options.tile_limit {
        args.extend(["--limit".into(), limit.to_string()]);
    } else {
        args.push("--no-limit".into());
    }
    if let Some(magnitude) = options.min_g_mag {
        args.extend(["--min-g-mag".into(), magnitude.to_string()]);
    }
    if let Some(magnitude) = options.max_g_mag {
        args.extend(["--max-g-mag".into(), magnitude.to_string()]);
    }
    if options.include_gspphot_model {
        args.push("--include-gspphot-model".into());
    }

    let tiled = gaia_tiled::run_desktop(&args)?;
    let output_path = options
        .output_path
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if let Some(metadata_path) = tiled.get("metadata").and_then(Value::as_str) {
        if Path::new(metadata_path).exists() {
            fs::remove_file(metadata_path).map_err(|error| {
                format!("could not remove tiled intermediate metadata: {error}")
            })?;
        }
    }
    let tiled_result = tiled
        .get("result")
        .ok_or("Gaia tiled runner did not return a result")?;
    let audit = audit_value(options, &output_path, radius, tiled_result)?;
    let mut bytes = serde_json::to_vec_pretty(&audit).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    let metadata_path = audit_path(&output_path);
    if let Some(parent) = metadata_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&metadata_path, bytes).map_err(|error| error.to_string())?;
    Ok(audit)
}

/// Read the pointing center from the FITS auxiliary block before downloading.
pub fn download_public_gaia_catalog_for_frame(
    frame_path: &Path,
    output_path: &Path,
    mut options: PublicCatalogOptions,
) -> Result<Value> {
    let frame = fits::read_fits(frame_path, false).map_err(|error| error.to_string())?;
    let auxiliary = frame
        .auxiliary
        .as_ref()
        .ok_or("FITS has no usable auxiliary RA/DEC for a public catalog query")?;
    options.center_ra_deg = auxiliary.ra_deg();
    options.center_dec_deg = auxiliary.dec_deg();
    options.output_path = output_path.to_path_buf();
    options.frame_path = Some(frame.path);
    download_public_gaia_catalog(&options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    #[test]
    fn square_camera_query_radius_and_default_tile_cover_match_python() {
        let radius = camera_footprint_radius_deg(9.78, 9.78).unwrap();
        assert!((radius - 6.91550432000443).abs() < 1e-12);
        assert!(DEFAULT_GAIA_TILE_RADIUS_DEG > radius);
        let tiles = gaia_tiled::plan_gaia_tiles(
            129.533548,
            -1.845372,
            radius,
            DEFAULT_GAIA_TILE_RADIUS_DEG,
            "auto",
        )
        .unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(DEFAULT_GAIA_TILE_LIMIT, 50_000);
        assert!(camera_footprint_radius_deg(0.0, 1.0).is_err());
    }

    #[test]
    fn coordinate_download_writes_python_compatible_audit_without_rows_duplication() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let body = include_str!("../tests/fixtures/gaia_remote_parity/response.csv").to_owned();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 8192];
            let count = stream.read(&mut request).unwrap();
            assert!(String::from_utf8_lossy(&request[..count]).starts_with("GET /tap/sync?"));
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/csv\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rst19-public-catalog-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let output = root.join("gaia.csv");
        let options = PublicCatalogOptions {
            center_ra_deg: 45.5,
            center_dec_deg: -12.25,
            output_path: output.clone(),
            search_radius_deg: Some(0.1),
            tile_radius_deg: 1.0,
            endpoint: format!("http://{address}/tap/sync"),
            ..Default::default()
        };
        let audit = download_public_gaia_catalog(&options).unwrap();
        server.join().unwrap();
        let mut normalized_audit = audit.clone();
        normalized_audit["output_path"] = Value::String("<OUTPUT>".into());
        normalized_audit["audit_path"] = Value::String("<AUDIT>".into());
        normalized_audit["endpoint"] = Value::String("http://127.0.0.1:0/tap/sync".into());
        let python_reference: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/public_catalog_parity/reference.json"
        ))
        .unwrap();
        assert_eq!(normalized_audit, python_reference);
        assert_eq!(
            fs::read(&output).unwrap(),
            include_bytes!("../tests/fixtures/gaia_remote_parity/reference.csv")
        );
        assert_eq!(audit["catalog"], "Gaia DR3");
        assert_eq!(audit["center_ra_deg"], 45.5);
        assert_eq!(audit["center_dec_deg"], -12.25);
        assert_eq!(audit["search_radius_deg"], 0.1);
        assert_eq!(audit["complete"], true);
        assert_eq!(audit["row_count"], 1);
        assert_eq!(audit["csv_row_count"], audit["row_count"]);
        assert_eq!(
            audit["csv_byte_count"],
            fs::metadata(&output).unwrap().len()
        );
        assert_eq!(audit["csv_sha256"].as_str().unwrap().len(), 64);
        assert!(audit["tiled"].get("rows").is_none());
        assert_eq!(audit["frame_path"], Value::Null);
        let sidecar = PathBuf::from(audit["audit_path"].as_str().unwrap());
        let sidecar_value: Value = serde_json::from_slice(&fs::read(sidecar).unwrap()).unwrap();
        assert_eq!(sidecar_value, audit);
        assert!(!PathBuf::from(format!("{}.tiled.meta.json", output.display())).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gspphot_option_selects_join_and_preserves_model_columns() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let body = include_str!("../tests/fixtures/gaia_remote_parity/response.csv").to_owned();
        let (request_tx, request_rx) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 8192];
            let count = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..count]).to_string();
            request_tx
                .send(request.lines().next().unwrap_or_default().to_owned())
                .unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/csv\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rst19-public-gspphot-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let output = root.join("gaia.csv");
        let options = PublicCatalogOptions {
            center_ra_deg: 45.5,
            center_dec_deg: -12.25,
            output_path: output.clone(),
            search_radius_deg: Some(0.1),
            tile_radius_deg: 1.0,
            include_gspphot_model: true,
            endpoint: format!("http://{address}/tap/sync"),
            ..Default::default()
        };
        let audit = download_public_gaia_catalog(&options).unwrap();
        server.join().unwrap();
        let request_line = request_rx.recv().unwrap();
        let target = request_line.split_whitespace().nth(1).unwrap();
        let request_url = reqwest::Url::parse(&format!("http://127.0.0.1{target}")).unwrap();
        let adql = request_url
            .query_pairs()
            .find(|(key, _)| key == "QUERY")
            .map(|(_, value)| value.into_owned())
            .unwrap();
        assert!(adql.contains("LEFT OUTER JOIN gaiadr3.astrophysical_parameters"));
        assert_eq!(
            audit["table"],
            "gaiadr3.gaia_source LEFT OUTER JOIN gaiadr3.astrophysical_parameters"
        );
        assert!(audit["provenance"]["gspphot_fields"]
            .as_str()
            .unwrap()
            .contains("mg_gspphot and its 16th/84th percentile bounds are included"));
        assert!(fs::read_to_string(output).unwrap().contains("mg_gspphot"));
        fs::remove_dir_all(root).unwrap();
    }
}
