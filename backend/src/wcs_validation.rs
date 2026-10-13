//! Native desktop execution and artifact writing for prior-WCS sequence validation.
//!
//! The fit and per-frame image matching live in `wcs_fit`; this module adapts
//! the existing research-tool arguments and writes its JSON, CSV, and plots.

use crate::{catalog_csv, coordinates::TangentPlaneWcs, pipeline::FrameOptions, wcs_fit, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

const CSV_FIELDS: [&str; 17] = [
    "frame_index",
    "frame_number",
    "frame_path",
    "status",
    "matched_count",
    "inlier_count",
    "inlier_ratio",
    "rms_residual_px",
    "max_residual_px",
    "leave_one_out_rms_residual_px",
    "leave_one_out_max_residual_px",
    "plate_scale_arcsec_per_pixel",
    "rotation_deg",
    "parity",
    "anisotropy_ratio",
    "condition_number",
    "reason",
];

#[derive(Debug)]
struct CliOptions {
    data_dir: Option<PathBuf>,
    catalog_csv: Option<PathBuf>,
    out_dir: PathBuf,
    center_ra_deg: Option<f64>,
    center_dec_deg: Option<f64>,
    pixel_scale_arcsec: Option<f64>,
    crpix_x: f64,
    crpix_y: f64,
    rotation_deg: f64,
    parity: i32,
    match_radius_px: f64,
    min_matches: usize,
    epoch: Option<f64>,
    threshold_sigma: f64,
    min_distance: usize,
    aperture_radius: usize,
    psf_fwhm: f64,
    background_box_size: usize,
    min_flux_snr: f64,
    max_sources: Option<usize>,
    reject_linear_artifacts: bool,
}

impl CliOptions {
    fn new(default_out_dir: &Path) -> Self {
        Self {
            data_dir: None,
            catalog_csv: None,
            out_dir: default_out_dir.to_path_buf(),
            center_ra_deg: None,
            center_dec_deg: None,
            pixel_scale_arcsec: None,
            crpix_x: 2048.0,
            crpix_y: 2048.0,
            rotation_deg: 0.0,
            parity: 1,
            match_radius_px: 3.0,
            min_matches: 6,
            epoch: None,
            threshold_sigma: 4.0,
            min_distance: 4,
            aperture_radius: 4,
            psf_fwhm: 3.0,
            background_box_size: 128,
            min_flux_snr: 5.0,
            max_sources: None,
            reject_linear_artifacts: true,
        }
    }

    fn parse(args: &[String], default_out_dir: &Path) -> Result<Self> {
        let mut parsed = Self::new(default_out_dir);
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            let (name, inline) = arg
                .split_once('=')
                .map(|(name, value)| (name, Some(value)))
                .unwrap_or((arg.as_str(), None));
            if name == "--keep-linear-artifacts" {
                if inline.is_some() {
                    return Err("--keep-linear-artifacts does not take a value".into());
                }
                parsed.reject_linear_artifacts = false;
                index += 1;
                continue;
            }
            if !name.starts_with('-') {
                if parsed.data_dir.is_none() {
                    parsed.data_dir = Some(PathBuf::from(arg));
                    index += 1;
                    continue;
                }
                if parsed.catalog_csv.is_none() {
                    parsed.catalog_csv = Some(PathBuf::from(arg));
                    index += 1;
                    continue;
                }
                return Err("data_dir and catalog_csv are the only positional arguments".into());
            }
            let value = if let Some(value) = inline {
                value
            } else {
                index += 1;
                args.get(index)
                    .ok_or_else(|| format!("{name} requires a value"))?
            };
            macro_rules! parse_value {
                ($target:expr, $kind:ty) => {
                    $target = Some(
                        value
                            .parse::<$kind>()
                            .map_err(|_| format!("invalid value for {name}: {value}"))?,
                    )
                };
            }
            match name {
                "--out-dir" => parsed.out_dir = PathBuf::from(value),
                "--center-ra" => parse_value!(parsed.center_ra_deg, f64),
                "--center-dec" => parse_value!(parsed.center_dec_deg, f64),
                "--pixel-scale" => parse_value!(parsed.pixel_scale_arcsec, f64),
                "--crpix-x" => parsed.crpix_x = parse_arg(value, name)?,
                "--crpix-y" => parsed.crpix_y = parse_arg(value, name)?,
                "--rotation" => parsed.rotation_deg = parse_arg(value, name)?,
                "--parity" => parsed.parity = parse_arg(value, name)?,
                "--match-radius" => parsed.match_radius_px = parse_arg(value, name)?,
                "--min-matches" => parsed.min_matches = parse_arg(value, name)?,
                "--epoch" => parsed.epoch = Some(parse_arg(value, name)?),
                "--threshold-sigma" => parsed.threshold_sigma = parse_arg(value, name)?,
                "--min-distance" => parsed.min_distance = parse_arg(value, name)?,
                "--aperture-radius" => parsed.aperture_radius = parse_arg(value, name)?,
                "--psf-fwhm" => parsed.psf_fwhm = parse_arg(value, name)?,
                "--background-box-size" => parsed.background_box_size = parse_arg(value, name)?,
                "--min-flux-snr" => parsed.min_flux_snr = parse_arg(value, name)?,
                "--max-sources" => parsed.max_sources = Some(parse_arg(value, name)?),
                _ => return Err(format!("unknown argument {arg}")),
            }
            index += 1;
        }
        if parsed.data_dir.is_none() || parsed.catalog_csv.is_none() {
            return Err("data_dir and catalog_csv are required".into());
        }
        if parsed.center_ra_deg.is_none()
            || parsed.center_dec_deg.is_none()
            || parsed.pixel_scale_arcsec.is_none()
        {
            return Err("--center-ra, --center-dec, and --pixel-scale are required".into());
        }
        Ok(parsed)
    }
}

fn parse_arg<T: std::str::FromStr>(value: &str, name: &str) -> Result<T> {
    value
        .parse::<T>()
        .map_err(|_| format!("invalid value for {name}: {value}"))
}

fn sequence_paths(data_dir: &Path) -> Result<Vec<String>> {
    let entries = fs::read_dir(data_dir).map_err(|error| error.to_string())?;
    let mut paths = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "fits")
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no FITS files in {}", data_dir.display()));
    }
    Ok(paths
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect())
}

fn csv_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        _ => value.to_string(),
    }
}

fn write_report_csv(report: &Value, path: &Path) -> Result<()> {
    let rows = report["frames"]
        .as_array()
        .ok_or("WCS validation report is missing frame rows")?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(CSV_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let values = CSV_FIELDS
            .iter()
            .map(|field| csv_cell(&row[*field]))
            .collect::<Vec<_>>();
        writer
            .write_record(values)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

const WIDTH: usize = 640;
const HEIGHT: usize = 360;
const PLOT_LEFT: usize = 62;
const PLOT_RIGHT: usize = 20;
const PLOT_TOP: usize = 44;
const PLOT_BOTTOM: usize = 58;

fn glyph(character: char) -> [u8; 7] {
    match character.to_ascii_uppercase() {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'J' => [7, 2, 2, 2, 18, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 25, 21, 19, 19, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        '.' => [0, 0, 0, 0, 0, 6, 6],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        ':' => [0, 6, 6, 0, 6, 6, 0],
        '(' => [2, 4, 8, 8, 8, 4, 2],
        ')' => [8, 4, 2, 2, 2, 4, 8],
        _ => [0; 7],
    }
}

fn draw_text(image: &mut [u8], x: isize, y: isize, text: &str, color: [u8; 3]) {
    for (character_index, character) in text.chars().enumerate() {
        let rows = glyph(character);
        let origin_x = x + character_index as isize * 6;
        for (row_index, row) in rows.into_iter().enumerate() {
            for column in 0..5 {
                if row & (1 << (4 - column)) != 0 {
                    set_pixel(
                        image,
                        origin_x + column as isize,
                        y + row_index as isize,
                        color,
                    );
                }
            }
        }
    }
}

fn set_pixel(image: &mut [u8], x: isize, y: isize, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= WIDTH as isize || y >= HEIGHT as isize {
        return;
    }
    let index = (y as usize * WIDTH + x as usize) * 3;
    image[index..index + 3].copy_from_slice(&color);
}

fn draw_line(image: &mut [u8], start: (isize, isize), end: (isize, isize), color: [u8; 3]) {
    let (mut x, mut y) = start;
    let dx = (end.0 - x).abs();
    let sx = if x < end.0 { 1 } else { -1 };
    let dy = -(end.1 - y).abs();
    let sy = if y < end.1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        set_pixel(image, x, y, color);
        if x == end.0 && y == end.1 {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x += sx;
        }
        if doubled <= dx {
            error += dx;
            y += sy;
        }
    }
}

fn draw_marker(image: &mut [u8], x: isize, y: isize, color: [u8; 3]) {
    for offset_y in -2..=2 {
        for offset_x in -2..=2 {
            if offset_x * offset_x + offset_y * offset_y <= 4 {
                set_pixel(image, x + offset_x, y + offset_y, color);
            }
        }
    }
}

fn write_line_chart(path: &Path, rows: &[Value], series: &[(&str, [u8; 3])]) -> Result<()> {
    let mut pixels = vec![255_u8; WIDTH * HEIGHT * 3];
    let x0 = PLOT_LEFT as isize;
    let x1 = (WIDTH - PLOT_RIGHT) as isize;
    let y0 = PLOT_TOP as isize;
    let y1 = (HEIGHT - PLOT_BOTTOM) as isize;
    let values = rows
        .iter()
        .flat_map(|row| series.iter().filter_map(|(field, _)| row[*field].as_f64()))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let title = if series.len() > 1 {
        "WCS RESIDUAL RMS (PIXEL)"
    } else {
        "PLATE SCALE (ARCSEC/PIXEL)"
    };
    draw_text(&mut pixels, x0, 5, title, [35, 43, 52]);
    let (mut low, mut high) = if values.is_empty() {
        (0.0, 1.0)
    } else {
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (low, high)
    };
    if (high - low).abs() <= f64::EPSILON * low.abs().max(high.abs()).max(1.0) {
        let margin = low.abs().max(1.0) * 0.05;
        low -= margin;
        high += margin;
    } else {
        let margin = (high - low) * 0.05;
        low -= margin;
        high += margin;
    }
    for division in 0..=4 {
        let y = y0 + (y1 - y0) * division / 4;
        draw_line(&mut pixels, (x0, y), (x1, y), [225, 230, 236]);
        let tick = high - (high - low) * division as f64 / 4.0;
        draw_text(&mut pixels, 3, y - 3, &format!("{tick:.2}"), [80, 88, 98]);
    }
    draw_line(&mut pixels, (x0, y0), (x0, y1), [40, 48, 58]);
    draw_line(&mut pixels, (x0, y1), (x1, y1), [40, 48, 58]);
    for (index, (field, color)) in series.iter().enumerate() {
        let legend_x = x0 + 86 + index as isize * 185;
        let legend_y = 24;
        draw_line(
            &mut pixels,
            (legend_x, legend_y + 3),
            (legend_x + 13, legend_y + 3),
            *color,
        );
        let label = match *field {
            "rms_residual_px" => "FIT RMS",
            "leave_one_out_rms_residual_px" => "LEAVE ONE OUT",
            "plate_scale_arcsec_per_pixel" => "SCALE",
            _ => "SERIES",
        };
        draw_text(&mut pixels, legend_x + 18, legend_y, label, [55, 63, 73]);
    }
    if rows.len() == 1 {
        draw_text(&mut pixels, (x0 + x1) / 2 - 3, y1 + 8, "1", [80, 88, 98]);
    } else if !rows.is_empty() {
        draw_text(&mut pixels, x0 - 2, y1 + 8, "1", [80, 88, 98]);
        draw_text(
            &mut pixels,
            x1 - (rows.len().to_string().len() as isize * 6),
            y1 + 8,
            &rows.len().to_string(),
            [80, 88, 98],
        );
    }
    draw_text(
        &mut pixels,
        (x0 + x1) / 2 - 15,
        (HEIGHT - 11) as isize,
        "FRAME",
        [55, 63, 73],
    );
    if !rows.is_empty() {
        for (field, color) in series {
            let mut previous = None;
            for (index, row) in rows.iter().enumerate() {
                let Some(value) = row[*field].as_f64().filter(|value| value.is_finite()) else {
                    previous = None;
                    continue;
                };
                let x = if rows.len() == 1 {
                    (x0 + x1) / 2
                } else {
                    x0 + (x1 - x0) * index as isize / (rows.len() - 1) as isize
                };
                let y = y1
                    - (((value - low) / (high - low)).clamp(0.0, 1.0) * (y1 - y0) as f64).round()
                        as isize;
                if let Some(start) = previous {
                    draw_line(&mut pixels, start, (x, y), *color);
                }
                draw_marker(&mut pixels, x, y, *color);
                previous = Some((x, y));
            }
        }
    }
    let file = fs::File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = png::Encoder::new(file, WIDTH as u32, HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    writer
        .write_image_data(&pixels)
        .map_err(|error| error.to_string())
}

pub fn write_artifacts(report: &Value, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    write_report_csv(report, &out_dir.join("wcs_frame_validation.csv"))?;
    fs::write(
        out_dir.join("wcs_validation_report.json"),
        serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let rows = report["frames"]
        .as_array()
        .ok_or("WCS validation report is missing frame rows")?;
    write_line_chart(
        &out_dir.join("wcs_validation_residual.png"),
        rows,
        &[
            ("rms_residual_px", [79, 155, 131]),
            ("leave_one_out_rms_residual_px", [215, 148, 50]),
        ],
    )?;
    write_line_chart(
        &out_dir.join("wcs_validation_scale.png"),
        rows,
        &[("plate_scale_arcsec_per_pixel", [114, 185, 212])],
    )?;
    Ok(out_dir.to_path_buf())
}

pub fn run_desktop(args: &[String], default_out_dir: &Path) -> Result<Value> {
    let options = CliOptions::parse(args, default_out_dir)?;
    let data_dir = options.data_dir.as_deref().ok_or("data_dir is required")?;
    let catalog_path = options
        .catalog_csv
        .as_deref()
        .ok_or("catalog_csv is required")?;
    let frame_paths = sequence_paths(data_dir)?;
    let catalog = catalog_csv::load_catalog_csv(catalog_path)?;
    let reference_wcs = TangentPlaneWcs {
        center_ra_deg: options.center_ra_deg.ok_or("--center-ra is required")?,
        center_dec_deg: options.center_dec_deg.ok_or("--center-dec is required")?,
        pixel_scale_arcsec: options
            .pixel_scale_arcsec
            .ok_or("--pixel-scale is required")?,
        crpix_x: options.crpix_x,
        crpix_y: options.crpix_y,
        rotation_deg: options.rotation_deg,
        parity: options.parity,
    };
    let detector = json!({
        "threshold_sigma":options.threshold_sigma,
        "min_distance":options.min_distance,
        "aperture_radius":options.aperture_radius,
        "max_sources":options.max_sources,
        "psf_fwhm":options.psf_fwhm,
        "background_box_size":options.background_box_size,
        "min_flux_snr":options.min_flux_snr,
        "reject_linear_artifacts":options.reject_linear_artifacts,
    });
    let frame_options: FrameOptions =
        serde_json::from_value(detector).map_err(|error| error.to_string())?;
    let fit_options = wcs_fit::AffineFitOptions {
        min_matches: options.min_matches,
        ..wcs_fit::AffineFitOptions::default()
    };
    let report = wcs_fit::validate_sequence_paths(
        &frame_paths,
        &catalog,
        &reference_wcs,
        &frame_options,
        options.match_radius_px,
        options.epoch,
        fit_options,
    )?;
    let output = write_artifacts(&report, &options.out_dir)?;
    let result_path = output.join("wcs_validation_report.json");
    Ok(json!({
        "tool":"rst19-wcs-validation",
        "out_dir":output,
        "result_path":result_path,
        "result":report,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_the_public_tool_arguments_and_writes_csv_and_png_artifacts() {
        let args = vec![
            "observations".into(),
            "catalog.csv".into(),
            "--center-ra".into(),
            "10".into(),
            "--center-dec=20".into(),
            "--pixel-scale".into(),
            "1.5".into(),
            "--keep-linear-artifacts".into(),
        ];
        let options = CliOptions::parse(&args, Path::new("out")).unwrap();
        assert_eq!(options.data_dir.as_deref(), Some(Path::new("observations")));
        assert_eq!(
            options.catalog_csv.as_deref(),
            Some(Path::new("catalog.csv"))
        );
        assert_eq!(options.center_ra_deg, Some(10.0));
        assert_eq!(options.center_dec_deg, Some(20.0));
        assert_eq!(options.pixel_scale_arcsec, Some(1.5));
        assert!(!options.reject_linear_artifacts);

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rst19-wcs-validation-{stamp}"));
        let report = json!({
            "frames":[
                {"frame_index":0,"frame_number":1,"frame_path":"f1.fits","status":"validated","matched_count":8,"inlier_count":8,"inlier_ratio":1.0,"rms_residual_px":0.1,"max_residual_px":0.2,"leave_one_out_rms_residual_px":0.15,"leave_one_out_max_residual_px":0.3,"plate_scale_arcsec_per_pixel":1.5,"rotation_deg":0.0,"parity":1,"anisotropy_ratio":1.0,"condition_number":2.0,"reason":null},
                {"frame_index":1,"frame_number":2,"frame_path":"f2.fits","status":"insufficient","matched_count":2,"inlier_count":0,"inlier_ratio":0.0,"rms_residual_px":null,"max_residual_px":null,"leave_one_out_rms_residual_px":null,"leave_one_out_max_residual_px":null,"plate_scale_arcsec_per_pixel":null,"rotation_deg":null,"parity":null,"anisotropy_ratio":null,"condition_number":null,"reason":"too few"}
            ]
        });
        write_artifacts(&report, &root).unwrap();
        let csv = fs::read(root.join("wcs_frame_validation.csv")).unwrap();
        assert!(csv.starts_with(&[0xef, 0xbb, 0xbf]));
        assert!(csv.windows(9).any(|part| part == b"validated"));
        for name in ["wcs_validation_residual.png", "wcs_validation_scale.png"] {
            let bytes = fs::read(root.join(name)).unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        }
        let _ = fs::remove_dir_all(root);
    }
}
