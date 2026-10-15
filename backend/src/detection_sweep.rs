//! Native FITS detection threshold and PSF sensitivity sweeps.
//!
//! Every level is measured against the same image and auxiliary mask. Counts
//! describe detector candidates and quality-gated sources; they are not truth
//! recovery, precision, or physical classification metrics.
use crate::{
    detection::{self, DetectionOptions, DetectionResult},
    fits,
    measurement::MeasurementOptions,
    models::PixelData,
    Result,
};
use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const DEFAULT_THRESHOLDS: &[f64] = &[4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0];

#[derive(Clone, Debug, Serialize)]
struct SweepRow {
    #[serde(skip_serializing_if = "Option::is_none")]
    threshold_sigma: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psf_fwhm_px: Option<f64>,
    candidate_count: usize,
    returned_count: usize,
    quality_count: usize,
    rejected_count: usize,
    line_artifact_count: usize,
    background_adu: f64,
    noise_adu: f64,
}

impl SweepRow {
    fn from_detection(level: f64, psf: bool, result: DetectionResult) -> Self {
        let returned_count = result.sources.len();
        let line_artifact_count = result
            .sources
            .iter()
            .filter(|source| source.flags.iter().any(|flag| flag == "LINE_ARTIFACT"))
            .count();
        Self {
            threshold_sigma: (!psf).then_some(level),
            psf_fwhm_px: psf.then_some(level),
            candidate_count: result.candidate_count,
            returned_count,
            quality_count: result.quality_count,
            rejected_count: returned_count.saturating_sub(result.quality_count),
            line_artifact_count,
            background_adu: result.background,
            noise_adu: result.noise,
        }
    }

    fn x(&self, psf: bool) -> f64 {
        if psf {
            self.psf_fwhm_px.unwrap_or_default()
        } else {
            self.threshold_sigma.unwrap_or_default()
        }
    }

    fn csv_fields(&self, psf: bool) -> Vec<String> {
        let mut fields = vec![format!("{:?}", self.x(psf))];
        fields.extend([
            self.candidate_count.to_string(),
            self.returned_count.to_string(),
            self.quality_count.to_string(),
            self.rejected_count.to_string(),
            self.line_artifact_count.to_string(),
            format!("{:?}", self.background_adu),
            format!("{:?}", self.noise_adu),
        ]);
        fields
    }
}

#[derive(Debug)]
struct Options {
    input: PathBuf,
    thresholds: Vec<f64>,
    psf_fwhms: Option<Vec<f64>>,
    min_distance: usize,
    aperture_radius: usize,
    psf_fwhm: f64,
    background_box_size: usize,
    min_flux_snr: f64,
    min_psf_support_pixels: usize,
    max_sources: Option<usize>,
    reject_linear_artifacts: bool,
    out_dir: PathBuf,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            thresholds: DEFAULT_THRESHOLDS.to_vec(),
            psf_fwhms: None,
            min_distance: 4,
            aperture_radius: 4,
            psf_fwhm: 3.0,
            background_box_size: 128,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            max_sources: None,
            reject_linear_artifacts: true,
            out_dir: PathBuf::from("tmp/detection-sweep"),
        }
    }
}

fn parse_number<T: std::str::FromStr>(value: Option<&String>, name: &str) -> Result<T> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse()
        .map_err(|_| format!("invalid {name}"))
}

fn parse_levels(args: &[String], start: usize, flag: &str) -> Result<(Vec<f64>, usize)> {
    let mut values = Vec::new();
    let mut index = start;
    while index < args.len() && !args[index].starts_with('-') {
        let value = args[index]
            .parse::<f64>()
            .map_err(|_| format!("invalid value for {flag}"))?;
        values.push(value);
        index += 1;
    }
    if values.is_empty() {
        return Err(format!("{flag} requires at least one value"));
    }
    Ok((values, index))
}

fn parse_args(args: &[String]) -> Result<Options> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Err("help".into());
    }
    let mut options = Options::default();
    let mut index = 0;
    if args.first().is_some_and(|value| !value.starts_with('-')) {
        options.input = PathBuf::from(&args[0]);
        index = 1;
    }
    while index < args.len() {
        match args[index].as_str() {
            "--thresholds" => {
                let (values, next) = parse_levels(args, index + 1, "--thresholds")?;
                options.thresholds = values;
                index = next;
            }
            "--psf-fwhms" => {
                let (values, next) = parse_levels(args, index + 1, "--psf-fwhms")?;
                options.psf_fwhms = Some(values);
                index = next;
            }
            "--min-distance" => {
                options.min_distance = parse_number(args.get(index + 1), "--min-distance")?;
                index += 2;
            }
            "--aperture-radius" => {
                options.aperture_radius = parse_number(args.get(index + 1), "--aperture-radius")?;
                index += 2;
            }
            "--psf-fwhm" => {
                options.psf_fwhm = parse_number(args.get(index + 1), "--psf-fwhm")?;
                index += 2;
            }
            "--background-box-size" => {
                options.background_box_size =
                    parse_number(args.get(index + 1), "--background-box-size")?;
                index += 2;
            }
            "--min-flux-snr" => {
                options.min_flux_snr = parse_number(args.get(index + 1), "--min-flux-snr")?;
                index += 2;
            }
            "--min-psf-support-pixels" => {
                options.min_psf_support_pixels =
                    parse_number(args.get(index + 1), "--min-psf-support-pixels")?;
                index += 2;
            }
            "--max-sources" => {
                options.max_sources = Some(parse_number(args.get(index + 1), "--max-sources")?);
                index += 2;
            }
            "--out-dir" => {
                options.out_dir =
                    PathBuf::from(args.get(index + 1).ok_or("--out-dir requires a value")?);
                index += 2;
            }
            "--keep-linear-artifacts" => {
                options.reject_linear_artifacts = false;
                index += 1;
            }
            value if !value.starts_with('-') && options.input.as_os_str().is_empty() => {
                options.input = PathBuf::from(value);
                index += 1;
            }
            value => return Err(format!("unknown or unexpected argument: {value}")),
        }
    }
    if options.input.as_os_str().is_empty() {
        return Err("a FITS input path is required".into());
    }
    if options.min_distance == 0
        || options.aperture_radius == 0
        || !options.psf_fwhm.is_finite()
        || options.psf_fwhm <= 0.0
        || options.background_box_size < 16
        || !options.min_flux_snr.is_finite()
        || options.min_flux_snr <= 0.0
        || !(1..=9).contains(&options.min_psf_support_pixels)
    {
        return Err("detection sweep parameters must be positive and valid".into());
    }
    if options.thresholds.is_empty()
        || options
            .thresholds
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 100.0)
    {
        return Err("thresholds must be finite values between 0 and 100 sigma".into());
    }
    if options.psf_fwhms.as_ref().is_some_and(|values| {
        values.is_empty()
            || values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
    }) {
        return Err("psf_fwhms must be finite positive values".into());
    }
    Ok(options)
}

fn detector_options(options: &Options, threshold_sigma: f64, psf_fwhm: f64) -> DetectionOptions {
    DetectionOptions {
        measurement: MeasurementOptions {
            aperture_radius: options.aperture_radius,
            psf_fwhm,
            min_flux_snr: options.min_flux_snr,
            min_psf_support_pixels: options.min_psf_support_pixels,
            ..Default::default()
        },
        threshold_sigma,
        min_distance: options.min_distance,
        max_sources: options.max_sources,
        background_box_size: options.background_box_size,
        reject_linear_artifacts: options.reject_linear_artifacts,
        ..Default::default()
    }
}

fn detect_level(
    data: &PixelData,
    height: usize,
    width: usize,
    mask: &[bool],
    options: &Options,
    level: f64,
    psf: bool,
) -> Result<SweepRow> {
    let threshold = if psf { options.thresholds[0] } else { level };
    let psf_fwhm = if psf { level } else { options.psf_fwhm };
    let result = detection::detect_sources(
        data,
        Some(mask),
        height,
        width,
        &detector_options(options, threshold, psf_fwhm),
        None,
    )?;
    Ok(SweepRow::from_detection(level, psf, result))
}

fn run_levels(options: &Options) -> Result<(Vec<SweepRow>, bool)> {
    eprintln!("3.0% · 读取 FITS 图像并准备检测参数扫描");
    let frame = fits::read_fits(&options.input, false)?;
    let mask = fits::auxiliary_mask(frame.height, frame.width)?;
    let (levels, psf) = options
        .psf_fwhms
        .as_ref()
        .map(|values| (values.as_slice(), true))
        .unwrap_or((&options.thresholds, false));
    eprintln!("10.0% · 图像与掩膜就绪，共 {} 个参数点", levels.len());
    let mut rows = Vec::with_capacity(levels.len());
    for (index, &level) in levels.iter().enumerate() {
        rows.push(detect_level(
            &frame.data,
            frame.height,
            frame.width,
            &mask,
            options,
            level,
            psf,
        )?);
        let percent = 12.0 + 72.0 * (index + 1) as f32 / levels.len().max(1) as f32;
        let parameter = if psf { "PSF FWHM" } else { "检测阈值 σ" };
        eprintln!(
            "{percent:.1}% · 完成参数点 {}/{}, {parameter}={level}",
            index + 1,
            levels.len()
        );
    }
    Ok((rows, psf))
}

fn write_csv(path: &Path, rows: &[SweepRow], psf: bool) -> Result<()> {
    let mut output = File::create(path).map_err(|error| error.to_string())?;
    output
        .write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let level = if psf {
        "psf_fwhm_px"
    } else {
        "threshold_sigma"
    };
    let mut writer = WriterBuilder::new()
        .terminator(Terminator::CRLF)
        .from_writer(output);
    writer
        .write_record([
            level,
            "candidate_count",
            "returned_count",
            "quality_count",
            "rejected_count",
            "line_artifact_count",
            "background_adu",
            "noise_adu",
        ])
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row.csv_fields(psf))
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

fn set_pixel(pixels: &mut [u8], width: usize, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x >= width || y >= pixels.len() / 3 / width {
        return;
    }
    let offset = (y * width + x) * 3;
    pixels[offset..offset + 3].copy_from_slice(&color);
}

fn glyph(character: char) -> [u8; 7] {
    match character {
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
        'N' => [17, 25, 21, 19, 17, 17, 17],
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
        '.' => [0, 0, 0, 0, 0, 12, 12],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        _ => [0; 7],
    }
}

fn text_width(text: &str, scale: i32) -> i32 {
    text.chars().count() as i32 * 6 * scale - scale
}

fn draw_text(
    pixels: &mut [u8],
    width: usize,
    x: i32,
    y: i32,
    text: &str,
    color: [u8; 3],
    scale: i32,
) {
    let mut cursor_x = x;
    for character in text.chars() {
        for (row, bits) in glyph(character).into_iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) == 0 {
                    continue;
                }
                for sy in 0..scale {
                    for sx in 0..scale {
                        set_pixel(
                            pixels,
                            width,
                            cursor_x + column * scale + sx,
                            y + row as i32 * scale + sy,
                            color,
                        );
                    }
                }
            }
        }
        cursor_x += 6 * scale;
    }
}

fn line(
    pixels: &mut [u8],
    width: usize,
    start: (i32, i32),
    end: (i32, i32),
    color: [u8; 3],
    thickness: i32,
) {
    let (mut x0, mut y0) = start;
    let (x1, y1) = end;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        let radius = thickness / 2;
        for oy in -radius..=radius {
            for ox in -radius..=radius {
                set_pixel(pixels, width, x0 + ox, y0 + oy, color);
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x0 += sx;
        }
        if doubled <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn write_chart(path: &Path, rows: &[SweepRow], psf: bool) -> Result<()> {
    const WIDTH: usize = 960;
    const HEIGHT: usize = 540;
    const LEFT: i32 = 96;
    const RIGHT: i32 = WIDTH as i32 - 42;
    const TOP: i32 = 48;
    const BOTTOM: i32 = HEIGHT as i32 - 72;
    let mut pixels = vec![255; WIDTH * HEIGHT * 3];
    let x_values: Vec<_> = rows.iter().map(|row| row.x(psf)).collect();
    let x_min = x_values.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let raw_x_max = x_values.iter().copied().reduce(f64::max).unwrap_or(1.0);
    let x_max = if raw_x_max <= x_min {
        x_min + 1.0
    } else {
        raw_x_max
    };
    let y_max = rows
        .iter()
        .map(|row| row.candidate_count.max(row.quality_count) as f64)
        .reduce(f64::max)
        .unwrap_or(1.0)
        .max(1.0);
    let grid = [220, 216, 202];
    let axis = [98, 107, 125];
    for tick in 0..=5 {
        let y = TOP + (BOTTOM - TOP) * tick / 5;
        line(&mut pixels, WIDTH, (LEFT, y), (RIGHT, y), grid, 1);
        let label = format!("{:.0}", y_max * (5 - tick) as f64 / 5.0);
        draw_text(
            &mut pixels,
            WIDTH,
            LEFT - text_width(&label, 1) - 8,
            y - 3,
            &label,
            axis,
            1,
        );
    }
    line(&mut pixels, WIDTH, (LEFT, TOP), (LEFT, BOTTOM), axis, 2);
    line(&mut pixels, WIDTH, (LEFT, BOTTOM), (RIGHT, BOTTOM), axis, 2);
    let title = if psf {
        "DETECTION PSF FWHM SWEEP"
    } else {
        "DETECTION THRESHOLD SWEEP"
    };
    draw_text(&mut pixels, WIDTH, LEFT, 17, title, [32, 41, 61], 2);
    let legend_x = RIGHT - 205;
    for (index, (label, color)) in [
        ("CANDIDATE PEAKS", [215, 148, 50]),
        ("QUALITY SOURCES", [79, 155, 131]),
    ]
    .into_iter()
    .enumerate()
    {
        let y = 17 + index as i32 * 15;
        line(
            &mut pixels,
            WIDTH,
            (legend_x, y + 3),
            (legend_x + 16, y + 3),
            color,
            2,
        );
        draw_text(&mut pixels, WIDTH, legend_x + 22, y, label, [39, 49, 71], 1);
    }
    let point = |x: f64, y: f64| {
        let px = LEFT as f64 + (x - x_min) / (x_max - x_min) * (RIGHT - LEFT) as f64;
        let py = BOTTOM as f64 - y / y_max * (BOTTOM - TOP) as f64;
        (px.round() as i32, py.round() as i32)
    };
    for (key, color) in [(true, [215, 148, 50]), (false, [79, 155, 131])] {
        let values: Vec<_> = rows
            .iter()
            .map(|row| {
                (if key {
                    row.candidate_count
                } else {
                    row.quality_count
                }) as f64
            })
            .collect();
        let points: Vec<_> = x_values
            .iter()
            .zip(values)
            .map(|(&x, y)| point(x, y))
            .collect();
        for pair in points.windows(2) {
            line(&mut pixels, WIDTH, pair[0], pair[1], color, 3);
        }
        for (x, y) in points {
            for oy in -4..=4 {
                for ox in -4..=4 {
                    if ox * ox + oy * oy <= 16 {
                        set_pixel(&mut pixels, WIDTH, x + ox, y + oy, color);
                    }
                }
            }
        }
    }
    for &x in &x_values {
        let px = point(x, 0.0).0;
        let label = if (x - x.round()).abs() < 1e-8 {
            format!("{x:.0}")
        } else {
            format!("{x:?}")
        };
        draw_text(
            &mut pixels,
            WIDTH,
            px - text_width(&label, 1) / 2,
            BOTTOM + 9,
            &label,
            axis,
            1,
        );
    }
    let x_label = if psf {
        "PSF FWHM PX"
    } else {
        "THRESHOLD SIGMA"
    };
    draw_text(
        &mut pixels,
        WIDTH,
        (WIDTH as i32 - text_width(x_label, 1)) / 2,
        HEIGHT as i32 - 25,
        x_label,
        axis,
        1,
    );
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = png::Encoder::new(file, WIDTH as u32, HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .map_err(|error| error.to_string())?
        .write_image_data(&pixels)
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn write_artifacts(rows: &[SweepRow], out_dir: &Path, psf: bool) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = fs::canonicalize(out_dir).map_err(|error| error.to_string())?;
    let (csv_name, png_name) = if psf {
        ("detection_psf_sweep.csv", "detection_psf_sweep.png")
    } else {
        (
            "detection_threshold_sweep.csv",
            "detection_threshold_sweep.png",
        )
    };
    write_csv(&output.join(csv_name), rows, psf)?;
    write_chart(&output.join(png_name), rows, psf)?;
    Ok(output)
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let options = parse_args(args)?;
    let (rows, psf) = run_levels(&options)?;
    eprintln!("88.0% · 参数扫描计算完成，写出 CSV 与趋势图");
    let output = write_artifacts(&rows, &options.out_dir, psf)?;
    eprintln!("99.0% · 扫描结果和图表已保存");
    Ok(json!({
        "tool":"rst19-sweep",
        "mode":if psf { "psf_fwhm" } else { "threshold" },
        "result":rows,
        "out_dir":output.to_string_lossy(),
        "csv_path":output.join(if psf { "detection_psf_sweep.csv" } else { "detection_threshold_sweep.csv" }).to_string_lossy(),
        "chart_path":output.join(if psf { "detection_psf_sweep.png" } else { "detection_threshold_sweep.png" }).to_string_lossy()
    }))
}

pub fn run_cli(args: &[String]) -> Result<i32> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("真实 FITS 候选阈值/匹配滤波 PSF 敏感性扫描\n\nUsage: rst19-sweep FITS [--thresholds N ...] [--psf-fwhms N ...] [--min-distance N] [--aperture-radius N] [--psf-fwhm N] [--background-box-size N] [--min-flux-snr N] [--min-psf-support-pixels N] [--max-sources N] [--keep-linear-artifacts] [--out-dir DIR]");
        return Ok(0);
    }
    let response = run_desktop(args)?;
    let rows = response["result"].as_array().ok_or("sweep rows missing")?;
    println!(
        "detection sweep: {}",
        response["out_dir"].as_str().unwrap_or_default()
    );
    if let Some(first) = rows.first() {
        println!("levels={} first={first}", rows.len());
    } else {
        println!("levels=0");
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::{parse_args, run_desktop, write_artifacts, SweepRow};
    use serde_json::Value;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn fits_card(key: &str, value: &str) -> [u8; 80] {
        let mut card = [b' '; 80];
        let content = format!("{key:<8}= {value:>20}");
        card[..content.len()].copy_from_slice(content.as_bytes());
        card
    }

    fn write_synthetic_fits(path: &std::path::Path) {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        let mut bytes = Vec::new();
        for (key, value) in [
            ("SIMPLE", "T"),
            ("BITPIX", "16"),
            ("NAXIS", "2"),
            ("NAXIS1", "64"),
            ("NAXIS2", "64"),
        ] {
            bytes.extend_from_slice(&fits_card(key, value));
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let pattern = (((37 * x + 101 * y) % 17) as i32 - 8) as f64 * 0.5;
                let radius2 = (x as f64 - 32.0).powi(2) + (y as f64 - 35.0).powi(2);
                let value = (100.0 + pattern + 1200.0 * (-0.5 * radius2 / 2.0_f64.powi(2)).exp())
                    .round() as i16;
                bytes.extend_from_slice(&value.to_be_bytes());
            }
        }
        bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn parses_threshold_and_psf_modes_without_changing_defaults() {
        let args = vec![
            "frame.fits".into(),
            "--thresholds".into(),
            "4".into(),
            "8".into(),
        ];
        let options = parse_args(&args).unwrap();
        assert_eq!(options.thresholds, [4.0, 8.0]);
        assert_eq!(options.psf_fwhms, None);
        assert_eq!(options.min_distance, 4);

        let args = vec![
            "frame.fits".into(),
            "--psf-fwhms".into(),
            "2.5".into(),
            "3.5".into(),
        ];
        let options = parse_args(&args).unwrap();
        assert_eq!(options.psf_fwhms.unwrap(), [2.5, 3.5]);
    }

    #[test]
    fn artifacts_include_bom_csv_and_valid_png_in_both_modes() {
        let root = std::env::temp_dir().join(format!(
            "rst19-detection-sweep-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let rows = vec![
            SweepRow {
                threshold_sigma: Some(4.0),
                psf_fwhm_px: None,
                candidate_count: 12,
                returned_count: 10,
                quality_count: 6,
                rejected_count: 4,
                line_artifact_count: 1,
                background_adu: 42.0,
                noise_adu: 2.5,
            },
            SweepRow {
                threshold_sigma: Some(8.0),
                psf_fwhm_px: None,
                candidate_count: 7,
                returned_count: 7,
                quality_count: 5,
                rejected_count: 2,
                line_artifact_count: 0,
                background_adu: 42.0,
                noise_adu: 2.5,
            },
        ];
        let output = write_artifacts(&rows, &root, false).unwrap();
        let csv = fs::read(output.join("detection_threshold_sweep.csv")).unwrap();
        assert!(csv.starts_with(&[0xef, 0xbb, 0xbf]));
        assert!(String::from_utf8_lossy(&csv).contains("threshold_sigma,candidate_count"));
        let png = fs::read(output.join("detection_threshold_sweep.png")).unwrap();
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(png.len() > 1000);
        let psf_rows = rows
            .iter()
            .map(|row| SweepRow {
                threshold_sigma: None,
                psf_fwhm_px: Some(row.threshold_sigma.unwrap()),
                candidate_count: row.candidate_count,
                returned_count: row.returned_count,
                quality_count: row.quality_count,
                rejected_count: row.rejected_count,
                line_artifact_count: row.line_artifact_count,
                background_adu: row.background_adu,
                noise_adu: row.noise_adu,
            })
            .collect::<Vec<_>>();
        let psf_output = write_artifacts(&psf_rows, &output.join("psf"), true).unwrap();
        let psf_csv = fs::read(psf_output.join("detection_psf_sweep.csv")).unwrap();
        assert!(String::from_utf8_lossy(&psf_csv).contains("psf_fwhm_px,candidate_count"));
        let psf_png = fs::read(psf_output.join("detection_psf_sweep.png")).unwrap();
        assert!(psf_png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(psf_png.len() > 1000);
        let _decoded: Value = serde_json::from_slice(&serde_json::to_vec(&rows).unwrap()).unwrap();
        fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn threshold_and_psf_sweeps_run_over_fits_data_and_write_results() {
        let root = std::env::temp_dir().join(format!(
            "rst19-detection-sweep-fits-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let input = root.join("source.fits");
        write_synthetic_fits(&input);

        let threshold_out = root.join("threshold");
        let threshold = run_desktop(&[
            input.to_string_lossy().into_owned(),
            "--thresholds".into(),
            "4".into(),
            "8".into(),
            "--background-box-size".into(),
            "16".into(),
            "--out-dir".into(),
            threshold_out.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert_eq!(threshold["mode"], "threshold");
        assert_eq!(threshold["result"].as_array().unwrap().len(), 2);
        assert!(
            threshold["result"][0]["candidate_count"].as_u64().unwrap()
                >= threshold["result"][1]["candidate_count"].as_u64().unwrap()
        );
        assert!(threshold_out
            .join("detection_threshold_sweep.csv")
            .is_file());
        assert!(threshold_out
            .join("detection_threshold_sweep.png")
            .is_file());

        let psf_out = root.join("psf");
        let psf = run_desktop(&[
            input.to_string_lossy().into_owned(),
            "--thresholds".into(),
            "4".into(),
            "--psf-fwhms".into(),
            "2.5".into(),
            "3.5".into(),
            "--background-box-size".into(),
            "16".into(),
            "--out-dir".into(),
            psf_out.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert_eq!(psf["mode"], "psf_fwhm");
        assert_eq!(psf["result"].as_array().unwrap().len(), 2);
        assert!(psf_out.join("detection_psf_sweep.csv").is_file());
        assert!(psf_out.join("detection_psf_sweep.png").is_file());
        fs::remove_dir_all(root).unwrap();
    }
}
