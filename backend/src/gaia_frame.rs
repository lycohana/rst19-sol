//! Native FITS-auxiliary-to-Gaia query workflow.

use crate::{
    fits, gaia_remote, gaia_tiled,
    public_catalog::{self, PublicCatalogOptions},
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const DEFAULT_FOV_WIDTH_DEG: f64 = 9.78;
const DEFAULT_FOV_HEIGHT_DEG: f64 = 9.78;
const DEFAULT_MIN_G_MAG: f64 = 5.0;
const DEFAULT_MAX_G_MAG: f64 = 13.5;
const DEFAULT_TILE_RADIUS_DEG: f64 = 7.0;
const DEFAULT_TILE_LIMIT: usize = 50_000;
const DEFAULT_MAX_DEPTH: usize = 2;
const DEFAULT_MAX_QUERIES: usize = 2_000;
const DEFAULT_TIMEOUT: f64 = 30.0;
const DEFAULT_ENDPOINT: &str = gaia_remote::DEFAULT_GAIA_TAP_SYNC_URL;

#[derive(Debug)]
struct Options {
    fits: PathBuf,
    output: Option<PathBuf>,
    radius: Option<String>,
    min_g: String,
    max_g: String,
    tile_radius: String,
    tile_limit: String,
    max_depth: String,
    max_queries: String,
    timeout: String,
    endpoint: String,
    include_model: bool,
    dry_run: bool,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut fits = None;
        let mut output = None;
        let mut radius = None;
        let mut min_g = DEFAULT_MIN_G_MAG.to_string();
        let mut max_g = DEFAULT_MAX_G_MAG.to_string();
        let mut tile_radius = DEFAULT_TILE_RADIUS_DEG.to_string();
        let mut tile_limit = DEFAULT_TILE_LIMIT.to_string();
        let mut max_depth = DEFAULT_MAX_DEPTH.to_string();
        let mut max_queries = DEFAULT_MAX_QUERIES.to_string();
        let mut timeout = DEFAULT_TIMEOUT.to_string();
        let mut endpoint = DEFAULT_ENDPOINT.to_owned();
        let mut include_model = false;
        let mut dry_run = false;
        let mut seen = std::collections::HashSet::new();
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            if arg == "--help" || arg == "-h" {
                return Err("__help__".into());
            }
            if arg == "--include-gspphot-model" {
                if !seen.insert(arg.clone()) {
                    return Err(format!("duplicate option {arg}"));
                }
                include_model = true;
                index += 1;
                continue;
            }
            if arg == "--dry-run" {
                if !seen.insert(arg.clone()) {
                    return Err(format!("duplicate option {arg}"));
                }
                dry_run = true;
                index += 1;
                continue;
            }
            if !arg.starts_with("--") {
                if fits.replace(PathBuf::from(arg)).is_some() {
                    return Err("only one FITS path is accepted".into());
                }
                index += 1;
                continue;
            }
            let (name, inline) = arg
                .split_once('=')
                .map(|(name, value)| (name, Some(value.to_owned())))
                .unwrap_or((arg.as_str(), None));
            if !seen.insert(name.to_owned()) {
                return Err(format!("duplicate option {name}"));
            }
            let value = if let Some(value) = inline {
                value
            } else {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| format!("{name} requires a value"))?
            };
            match name {
                "--out" => output = Some(PathBuf::from(value)),
                "--radius" => radius = Some(value),
                "--min-g" => min_g = value,
                "--max-g" => max_g = value,
                "--tile-radius" => tile_radius = value,
                "--tile-limit" => tile_limit = value,
                "--max-depth" => max_depth = value,
                "--max-queries" => max_queries = value,
                "--timeout" => timeout = value,
                "--endpoint" => endpoint = value,
                _ => return Err(format!("unknown argument {arg}")),
            }
            index += 1;
        }
        let fits = fits.ok_or("a FITS path is required")?;
        if !fits.is_file() {
            return Err(format!("FITS does not exist: {}", fits.display()));
        }
        if let Some(path) = &output {
            if !path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("csv"))
            {
                return Err("output path must use the .csv suffix".into());
            }
        } else if !dry_run {
            return Err("--out is required unless --dry-run is used".into());
        }
        Ok(Self {
            fits,
            output,
            radius,
            min_g,
            max_g,
            tile_radius,
            tile_limit,
            max_depth,
            max_queries,
            timeout,
            endpoint,
            include_model,
            dry_run,
        })
    }

    fn tiled_args(&self, center_ra: f64, center_dec: f64, out_dir: &Path) -> Vec<String> {
        let radius = self
            .radius
            .clone()
            .unwrap_or_else(|| camera_footprint_radius_deg().to_string());
        let output = self
            .output
            .clone()
            .unwrap_or_else(|| out_dir.join("gaia-frame.csv"));
        let mut args = vec![
            "--ra".into(),
            center_ra.to_string(),
            "--dec".into(),
            center_dec.to_string(),
            "--radius".into(),
            radius,
            "--out".into(),
            output.to_string_lossy().into_owned(),
            "--min-g-mag".into(),
            self.min_g.clone(),
            "--max-g-mag".into(),
            self.max_g.clone(),
            "--tile-radius".into(),
            self.tile_radius.clone(),
            "--limit".into(),
            self.tile_limit.clone(),
            "--max-depth".into(),
            self.max_depth.clone(),
            "--max-queries".into(),
            self.max_queries.clone(),
            "--timeout".into(),
            self.timeout.clone(),
            "--endpoint".into(),
            self.endpoint.clone(),
        ];
        if self.include_model {
            args.push("--include-gspphot-model".into());
        }
        if self.dry_run {
            args.push("--dry-run".into());
        }
        args
    }
}

pub fn camera_footprint_radius_deg() -> f64 {
    0.5 * DEFAULT_FOV_WIDTH_DEG.hypot(DEFAULT_FOV_HEIGHT_DEG)
}

pub fn run_desktop(args: &[String], default_out_dir: &Path) -> Result<Value, String> {
    let options = Options::parse(args)?;
    eprintln!("2.0% · 解析单帧 Gaia 星表任务参数");
    let fits_path = options
        .fits
        .canonicalize()
        .map_err(|error| format!("FITS cannot be resolved: {error}"))?;
    let frame = fits::read_fits(&fits_path, false).map_err(|error| error.to_string())?;
    eprintln!("8.0% · FITS 已载入，读取辅助 RA/Dec 指向坐标");
    let auxiliary = frame
        .auxiliary
        .as_ref()
        .ok_or("FITS has no usable auxiliary RA/DEC for a public catalog query")?;
    let center_ra = auxiliary.ra_deg();
    let center_dec = auxiliary.dec_deg();
    let radius = options
        .radius
        .as_deref()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| "radius must be numeric".to_owned())
        })
        .transpose()?
        .unwrap_or_else(camera_footprint_radius_deg);
    if options.dry_run {
        eprintln!("95.0% · 生成 Gaia 瓦片查询计划");
        let args = options.tiled_args(center_ra, center_dec, default_out_dir);
        let plan = gaia_tiled::run_desktop(&args)?;
        return Ok(json!({
            "dry_run":true,
            "frame_path":fits_path.to_string_lossy(),
            "center_ra_deg":center_ra,
            "center_dec_deg":center_dec,
            "search_radius_deg":radius,
            "plan":plan
        }));
    }
    let output_path = options.output.as_ref().ok_or("--out is required")?;
    let catalog_options = PublicCatalogOptions {
        center_ra_deg: center_ra,
        center_dec_deg: center_dec,
        output_path: output_path.clone(),
        frame_path: Some(fits_path.clone()),
        search_radius_deg: Some(radius),
        min_g_mag: Some(options.min_g.parse().map_err(|_| "min-g must be numeric")?),
        max_g_mag: Some(options.max_g.parse().map_err(|_| "max-g must be numeric")?),
        tile_radius_deg: options
            .tile_radius
            .parse()
            .map_err(|_| "tile-radius must be numeric")?,
        tile_limit: Some(
            options
                .tile_limit
                .parse()
                .map_err(|_| "tile-limit must be a positive integer")?,
        ),
        max_subdivide_depth: options
            .max_depth
            .parse()
            .map_err(|_| "max-depth must be a non-negative integer")?,
        max_queries: options
            .max_queries
            .parse()
            .map_err(|_| "max-queries must be a positive integer")?,
        include_gspphot_model: options.include_model,
        timeout_seconds: options
            .timeout
            .parse()
            .map_err(|_| "timeout must be numeric")?,
        endpoint: options.endpoint,
        ..Default::default()
    };
    eprintln!("18.0% · 坐标与搜索范围就绪，开始查询 Gaia DR3 瓦片");
    let audit = public_catalog::download_public_gaia_catalog(&catalog_options)?;
    eprintln!("99.0% · 星表与完整性审计已写入，整理单帧结果");
    Ok(json!({
        "csv":audit["output_path"],
        "audit":audit["audit_path"],
        "row_count":audit["row_count"],
        "complete":audit["complete"],
        "frame_path":fits_path.to_string_lossy(),
        "center_ra_deg":center_ra,
        "center_dec_deg":center_dec,
        "search_radius_deg":radius
    }))
}

pub fn run_cli(args: &[String]) -> i32 {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "USAGE: rst19-gaia-frame FITS --out PATH [OPTIONS]\n\nUses auxiliary FITS RA/DEC; default query radius covers the 9.78-degree square field.\nOptions: --radius DEG --min-g MAG --max-g MAG --tile-radius DEG --tile-limit N\n         --max-depth N --max-queries N --include-gspphot-model --timeout SECONDS\n         --endpoint URL --dry-run"
        );
        return 0;
    }
    match run_desktop(args, Path::new(".")) {
        Ok(result) if result["dry_run"] == true => {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            );
            0
        }
        Ok(result) if result["complete"] == true => {
            eprintln!(
                "Gaia DR3 complete: {} rows -> {} (audit: {})",
                result["row_count"], result["csv"], result["audit"]
            );
            0
        }
        Ok(result) => {
            eprintln!(
                "Gaia DR3 incomplete: {} rows -> {} (audit: {})",
                result["row_count"], result["csv"], result["audit"]
            );
            3
        }
        Err(error) => {
            eprintln!("gaia-frame: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    fn write_fits_with_auxiliary(path: &Path) {
        let mut header = Vec::new();
        for (key, value) in [
            ("SIMPLE", "T"),
            ("BITPIX", "-64"),
            ("NAXIS", "2"),
            ("NAXIS1", "16"),
            ("NAXIS2", "16"),
            ("END", ""),
        ] {
            let mut card = [b' '; 80];
            let content = if key == "END" {
                "END".to_owned()
            } else {
                format!("{key:<8}= {value:>20}")
            };
            card[..content.len()].copy_from_slice(content.as_bytes());
            header.extend_from_slice(&card);
        }
        header.resize(header.len().div_ceil(2880) * 2880, b' ');
        let mut values = [0.0_f64; 26];
        values[0] = 10.0;
        values[1] = 30.0;
        values[2] = 45.5;
        values[3] = -12.25;
        values[10] = 1.0;
        let mut data = vec![0_u8; 16 * 16 * 8];
        for (index, value) in values.iter().enumerate() {
            let mut bytes = value.to_le_bytes();
            for pair in bytes.as_chunks_mut::<2>().0 {
                pair.swap(0, 1);
            }
            data[index * 8..(index + 1) * 8].copy_from_slice(&bytes);
        }
        fs::write(path, [header, data].concat()).unwrap();
    }

    #[test]
    fn frame_dry_run_uses_auxiliary_pointing_and_conservative_default_radius() {
        let root =
            std::env::temp_dir().join(format!("rst19-gaia-frame-dry-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let fits = root.join("frame.fits");
        write_fits_with_auxiliary(&fits);
        let args = vec![fits.to_string_lossy().into_owned(), "--dry-run".into()];
        let result = run_desktop(&args, &root).unwrap();
        assert_eq!(result["dry_run"], true);
        assert_eq!(result["center_ra_deg"], 45.5);
        assert_eq!(result["center_dec_deg"], -12.25);
        assert!((result["search_radius_deg"].as_f64().unwrap() - 6.91550432000443).abs() < 1e-12);
        assert_eq!(result["plan"]["network"], false);
    }

    #[test]
    fn frame_query_writes_catalog_and_complete_audit_sidecar() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let body = include_str!("../tests/fixtures/gaia_remote_parity/response.csv").to_owned();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 8192];
            let count = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..count]);
            assert!(request.starts_with("GET /tap/sync?"));
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/csv\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            stream.write_all(response.as_bytes()).unwrap();
        });
        let root =
            std::env::temp_dir().join(format!("rst19-gaia-frame-run-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let fits = root.join("frame.fits");
        let output = root.join("catalog.csv");
        write_fits_with_auxiliary(&fits);
        let args = vec![
            fits.to_string_lossy().into_owned(),
            "--out".into(),
            output.to_string_lossy().into_owned(),
            "--endpoint".into(),
            format!("http://{address}/tap/sync"),
        ];
        let result = run_desktop(&args, &root).unwrap();
        server.join().unwrap();
        assert_eq!(result["complete"], true);
        assert_eq!(result["row_count"], 1);
        assert!(output.is_file());
        assert!(Path::new(result["audit"].as_str().unwrap()).is_file());
        let audit: Value =
            serde_json::from_slice(&fs::read(result["audit"].as_str().unwrap()).unwrap()).unwrap();
        assert_eq!(
            audit["frame_path"],
            fits.canonicalize().unwrap().to_string_lossy().as_ref()
        );
        assert_eq!(audit["center_ra_deg"], 45.5);
        assert_eq!(audit["csv_sha256"].as_str().unwrap().len(), 64);
        assert!(audit["tiled"]["tile_records"].as_array().unwrap().len() > 0);
        assert!(!audit["tiled"].as_object().unwrap().contains_key("rows"));
    }
}
