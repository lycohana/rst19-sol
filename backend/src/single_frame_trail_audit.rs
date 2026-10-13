//! Native Rust implementation of the one-frame trail audit command.
use crate::{
    detection::DetectionOptions,
    fits,
    pipeline::{self, FrameOptions},
    sequence_motion_features, Result,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

const CSV_FIELDS: [&str; 16] = [
    "frame_number",
    "path",
    "timestamp",
    "candidate_count",
    "quality_count",
    "trail_count",
    "max_length_px",
    "max_width_px",
    "max_residual_snr",
    "edge_trail_count",
    "total_area_pixels",
    "longest_angle_deg",
    "longest_bbox",
    "accepted_trail_count",
    "double_star_count",
    "model_rejected_count",
];

#[derive(Clone, Debug)]
struct CliOptions {
    input_path: PathBuf,
    out_dir: PathBuf,
    threshold_sigma: f64,
    min_distance: usize,
    aperture_radius: usize,
    psf_fwhm: f64,
    background_box_size: usize,
    min_flux_snr: f64,
    max_sources: Option<usize>,
    residual_sigma: f64,
    min_residual_adu: f64,
    min_feature_area: usize,
    min_axis_ratio: f64,
}

#[derive(Clone, Debug)]
struct AuditRow {
    frame_number: usize,
    path: String,
    timestamp: Option<String>,
    candidate_count: usize,
    quality_count: usize,
    trail_count: usize,
    max_length_px: Option<f64>,
    max_width_px: Option<f64>,
    max_residual_snr: Option<f64>,
    edge_trail_count: usize,
    total_area_pixels: usize,
    longest_angle_deg: Option<f64>,
    longest_bbox: Option<String>,
    accepted_trail_count: usize,
    double_star_count: usize,
    model_rejected_count: usize,
}

impl AuditRow {
    fn cells(&self) -> [String; 16] {
        [
            self.frame_number.to_string(),
            self.path.clone(),
            self.timestamp.clone().unwrap_or_default(),
            self.candidate_count.to_string(),
            self.quality_count.to_string(),
            self.trail_count.to_string(),
            cell_number(self.max_length_px),
            cell_number(self.max_width_px),
            cell_number(self.max_residual_snr),
            self.edge_trail_count.to_string(),
            self.total_area_pixels.to_string(),
            cell_number(self.longest_angle_deg),
            self.longest_bbox.clone().unwrap_or_default(),
            self.accepted_trail_count.to_string(),
            self.double_star_count.to_string(),
            self.model_rejected_count.to_string(),
        ]
    }

    fn as_json(&self) -> Value {
        json!({
            "frame_number": self.frame_number,
            "path": self.path,
            "timestamp": self.timestamp,
            "candidate_count": self.candidate_count,
            "quality_count": self.quality_count,
            "trail_count": self.trail_count,
            "max_length_px": self.max_length_px,
            "max_width_px": self.max_width_px,
            "max_residual_snr": self.max_residual_snr,
            "edge_trail_count": self.edge_trail_count,
            "total_area_pixels": self.total_area_pixels,
            "longest_angle_deg": self.longest_angle_deg,
            "longest_bbox": self.longest_bbox,
            "accepted_trail_count": self.accepted_trail_count,
            "double_star_count": self.double_star_count,
            "model_rejected_count": self.model_rejected_count,
        })
    }
}

fn cell_number(value: Option<f64>) -> String {
    value
        .filter(|number| number.is_finite())
        .map_or_else(String::new, |number| number.to_string())
}

fn parse_value<T: std::str::FromStr>(
    args: &[String],
    index: &mut usize,
    inline: Option<&str>,
    name: &str,
) -> Result<T> {
    let value = if let Some(value) = inline {
        value
    } else {
        *index += 1;
        args.get(*index)
            .map(String::as_str)
            .ok_or_else(|| format!("{name} requires a value"))?
    };
    value
        .parse::<T>()
        .map_err(|_| format!("invalid value for {name}: {value}"))
}

fn parse_args(args: &[String]) -> Result<CliOptions> {
    let mut input_path = None;
    let mut out_dir = PathBuf::from("tmp").join("single-frame-trails");
    let mut threshold_sigma = 4.0_f64;
    let mut min_distance = 4usize;
    let mut aperture_radius = 4usize;
    let mut psf_fwhm = 3.0_f64;
    let mut background_box_size = 128usize;
    let mut min_flux_snr = 5.0_f64;
    let mut max_sources = None;
    let mut residual_sigma = 15.0_f64;
    let mut min_residual_adu = 100.0_f64;
    let mut min_feature_area = 40usize;
    let mut min_axis_ratio = 4.0_f64;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        if !name.starts_with('-') {
            if input_path.replace(PathBuf::from(arg)).is_some() {
                return Err("only one FITS file or directory is allowed".into());
            }
            index += 1;
            continue;
        }
        match name {
            "--out-dir" => {
                out_dir = PathBuf::from(parse_value::<String>(args, &mut index, inline, name)?)
            }
            "--threshold-sigma" => threshold_sigma = parse_value(args, &mut index, inline, name)?,
            "--min-distance" => min_distance = parse_value(args, &mut index, inline, name)?,
            "--aperture-radius" => aperture_radius = parse_value(args, &mut index, inline, name)?,
            "--psf-fwhm" => psf_fwhm = parse_value(args, &mut index, inline, name)?,
            "--background-box-size" => {
                background_box_size = parse_value(args, &mut index, inline, name)?
            }
            "--min-flux-snr" => min_flux_snr = parse_value(args, &mut index, inline, name)?,
            "--max-sources" => max_sources = Some(parse_value(args, &mut index, inline, name)?),
            "--residual-sigma" => residual_sigma = parse_value(args, &mut index, inline, name)?,
            "--min-residual-adu" => min_residual_adu = parse_value(args, &mut index, inline, name)?,
            "--min-feature-area" => min_feature_area = parse_value(args, &mut index, inline, name)?,
            "--min-axis-ratio" => min_axis_ratio = parse_value(args, &mut index, inline, name)?,
            _ => return Err(format!("unknown argument {arg}")),
        }
        index += 1;
    }
    let input_path = input_path.ok_or("an input FITS file or directory is required")?;
    if !threshold_sigma.is_finite()
        || threshold_sigma <= 0.0
        || min_distance == 0
        || aperture_radius == 0
        || !psf_fwhm.is_finite()
        || psf_fwhm <= 0.0
        || background_box_size < 16
        || !min_flux_snr.is_finite()
        || min_flux_snr <= 0.0
        || max_sources == Some(0)
        || !residual_sigma.is_finite()
        || residual_sigma <= 0.0
        || !min_residual_adu.is_finite()
        || min_residual_adu <= 0.0
        || min_feature_area < 3
        || !min_axis_ratio.is_finite()
        || min_axis_ratio <= 0.0
    {
        return Err("single-frame trail detection parameters must be positive".into());
    }
    Ok(CliOptions {
        input_path,
        out_dir,
        threshold_sigma,
        min_distance,
        aperture_radius,
        psf_fwhm,
        background_box_size,
        min_flux_snr,
        max_sources,
        residual_sigma,
        min_residual_adu,
        min_feature_area,
        min_axis_ratio,
    })
}

fn frame_paths(input: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = if input.is_dir() {
        fs::read_dir(input)
            .map_err(|error| error.to_string())?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "fits")
            })
            .collect::<Vec<_>>()
    } else {
        vec![input.to_path_buf()]
    };
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "directory contains no FITS files: {}",
            input.display()
        ));
    }
    Ok(paths)
}

fn header_timestamp(header: &serde_json::Map<String, Value>) -> Option<String> {
    header.get("DATE-OBS").and_then(|value| match value {
        Value::Null => None,
        Value::String(value) => Some(value.clone()),
        Value::Bool(value) => Some(if *value { "True" } else { "False" }.to_owned()),
        value => Some(value.to_string()),
    })
}

fn number(point: &Value, key: &str) -> Option<f64> {
    point
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

fn model_supported(point: &Value) -> bool {
    let model = point.get("single_frame_model").and_then(Value::as_str);
    let snr = number(point, "line_template_snr")
        .is_some_and(|value| value >= crate::single_frame_model::MODEL_MIN_SNR);
    match model {
        Some("long_line_profile") => {
            snr && number(point, "line_vs_background_delta_bic")
                .is_some_and(|value| value >= crate::single_frame_model::LONG_PROFILE_MIN_DELTA_BIC)
        }
        Some("line") => {
            snr && number(point, "line_vs_point_delta_bic")
                .is_some_and(|value| value >= crate::single_frame_model::MODEL_MIN_DELTA_BIC)
                && number(point, "line_vs_double_delta_bic")
                    .is_some_and(|value| value >= crate::single_frame_model::MODEL_MIN_DELTA_BIC)
                && number(point, "line_vs_elliptical_delta_bic")
                    .is_some_and(|value| value >= crate::single_frame_model::MODEL_MIN_DELTA_BIC)
        }
        _ => false,
    }
}

fn shape_supported(point: &Value) -> bool {
    let Some(length) = number(point, "length_px") else {
        return false;
    };
    if length < 8.0
        || number(point, "line_artifact_overlap_fraction").is_some_and(|value| value >= 0.75)
        || point["single_frame_model"] == "not_fit"
        || !model_supported(point)
    {
        return false;
    }
    let Some(continuity) = number(point, "continuity_ratio") else {
        return false;
    };
    let Some(fill) = number(point, "profile_fill_ratio") else {
        return false;
    };
    continuity >= 0.60 && fill >= 0.60
}

fn display_python_path(path: &Path) -> String {
    let value = path.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        value.replace('/', "\\")
    }
    #[cfg(not(windows))]
    {
        value
    }
}

fn make_row(
    frame_number: usize,
    path: &Path,
    analysis: &pipeline::FrameAnalysis,
    result: &Value,
) -> Result<AuditRow> {
    let points = result["points"]
        .as_array()
        .ok_or("single-frame component result is missing points")?;
    let accepted = points
        .iter()
        .filter(|point| shape_supported(point))
        .collect::<Vec<_>>();
    let longest = accepted.iter().copied().max_by(|left, right| {
        number(left, "length_px")
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&number(right, "length_px").unwrap_or(f64::NEG_INFINITY))
    });
    let bbox = longest
        .and_then(|point| point.get("bbox"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(",")
        });
    let accepted_length = |key: &str| longest.and_then(|point| number(point, key));
    Ok(AuditRow {
        frame_number,
        path: display_python_path(path),
        timestamp: header_timestamp(&analysis.frame.header),
        candidate_count: analysis.detection.candidate_count,
        quality_count: analysis.detection.quality_count,
        trail_count: points.len(),
        max_length_px: accepted_length("length_px"),
        max_width_px: accepted_length("width_px"),
        max_residual_snr: accepted
            .iter()
            .filter_map(|point| number(point, "residual_snr"))
            .max_by(f64::total_cmp),
        edge_trail_count: accepted
            .iter()
            .filter(|point| point["touches_edge"].as_bool().unwrap_or(false))
            .count(),
        total_area_pixels: accepted
            .iter()
            .filter_map(|point| point["area_pixels"].as_u64())
            .map(|value| value as usize)
            .sum(),
        longest_angle_deg: accepted_length("angle_deg"),
        longest_bbox: bbox,
        accepted_trail_count: accepted.len(),
        double_star_count: points
            .iter()
            .filter(|point| point["single_frame_model"] == "double")
            .count(),
        model_rejected_count: points.len().saturating_sub(accepted.len()),
    })
}

fn write_csv(rows: &[AuditRow], path: &Path) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(CSV_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row.cells())
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

const CHART_WIDTH: usize = 1200;
const CHART_HEIGHT: usize = 680;
const CHART_LEFT: isize = 90;
const CHART_TOP: isize = 75;
const CHART_RIGHT: isize = 45;
const CHART_BOTTOM: isize = 110;
type ChartSeries<'a> = (&'a str, [u8; 3], fn(&AuditRow) -> f64);

fn set_pixel(pixels: &mut [u8], x: isize, y: isize, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= CHART_WIDTH as isize || y >= CHART_HEIGHT as isize {
        return;
    }
    let offset = (y as usize * CHART_WIDTH + x as usize) * 3;
    pixels[offset..offset + 3].copy_from_slice(&color);
}

fn draw_line(pixels: &mut [u8], start: (isize, isize), end: (isize, isize), color: [u8; 3]) {
    let (mut x, mut y) = start;
    let dx = (end.0 - x).abs();
    let sx = if x < end.0 { 1 } else { -1 };
    let dy = -(end.1 - y).abs();
    let sy = if y < end.1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        set_pixel(pixels, x, y, color);
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

fn draw_text(pixels: &mut [u8], x: isize, y: isize, text: &str, color: [u8; 3]) {
    for (character_index, character) in text.chars().enumerate() {
        for (row_index, row) in glyph(character).into_iter().enumerate() {
            for column in 0..5 {
                if row & (1 << (4 - column)) != 0 {
                    set_pixel(
                        pixels,
                        x + character_index as isize * 6 + column,
                        y + row_index as isize,
                        color,
                    );
                }
            }
        }
    }
}

fn write_chart(
    path: &Path,
    rows: &[AuditRow],
    title: &str,
    y_label: &str,
    x_label: &str,
    series: &[ChartSeries<'_>],
) -> Result<()> {
    let mut pixels = vec![0_u8; CHART_WIDTH * CHART_HEIGHT * 3];
    for pixel in pixels.as_chunks_mut::<3>().0 {
        pixel.copy_from_slice(&[244, 240, 231]);
    }
    let left = CHART_LEFT;
    let right = CHART_WIDTH as isize - CHART_RIGHT;
    let top = CHART_TOP;
    let bottom = CHART_HEIGHT as isize - CHART_BOTTOM;
    let axis = [98, 107, 125];
    draw_text(&mut pixels, left, 22, title, [32, 41, 61]);
    draw_text(&mut pixels, 12, 48, y_label, [98, 107, 125]);
    draw_text(
        &mut pixels,
        left,
        CHART_HEIGHT as isize - 30,
        x_label,
        [98, 107, 125],
    );
    let mut legend_x = (CHART_WIDTH as isize - CHART_RIGHT - 390).max(left);
    for (label, color, _) in series {
        draw_line(&mut pixels, (legend_x, 50), (legend_x + 22, 50), *color);
        draw_text(&mut pixels, legend_x + 28, 46, label, [98, 107, 125]);
        legend_x += 28 + label.len() as isize * 6 + 20;
    }
    draw_line(&mut pixels, (left, top), (left, bottom), axis);
    draw_line(&mut pixels, (left, bottom), (right, bottom), axis);
    for grid in 1..=4 {
        let y = top + (bottom - top) * grid / 5;
        draw_line(&mut pixels, (left, y), (right, y), [222, 217, 207]);
    }
    let maximum = series
        .iter()
        .flat_map(|(_, _, value)| rows.iter().map(value))
        .filter(|value| value.is_finite())
        .fold(0.0f64, f64::max)
        .max(1.0);
    for (_, color, value_for_row) in series {
        let values = rows.iter().map(value_for_row).collect::<Vec<_>>();
        if values.len() > 1 {
            let mut previous = None;
            for (index, value) in values.iter().copied().enumerate() {
                if !value.is_finite() {
                    previous = None;
                    continue;
                }
                let x = left + (right - left) * index as isize / (values.len() - 1) as isize;
                let y =
                    bottom - ((value.max(0.0) / maximum) * (bottom - top) as f64).round() as isize;
                if let Some(start) = previous {
                    draw_line(&mut pixels, start, (x, y), *color);
                }
                previous = Some((x, y));
                for offset_y in -3..=3 {
                    for offset_x in -3..=3 {
                        if offset_x * offset_x + offset_y * offset_y <= 9 {
                            set_pixel(&mut pixels, x + offset_x, y + offset_y, *color);
                        }
                    }
                }
            }
        } else if let Some(value) = values.first().copied().filter(|value| value.is_finite()) {
            let y = bottom - ((value.max(0.0) / maximum) * (bottom - top) as f64).round() as isize;
            for offset in -4..=4 {
                set_pixel(&mut pixels, left + offset, y, *color);
                set_pixel(&mut pixels, left, y + offset, *color);
            }
        }
    }
    let file = fs::File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = png::Encoder::new(file, CHART_WIDTH as u32, CHART_HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .map_err(|error| error.to_string())?
        .write_image_data(&pixels)
        .map_err(|error| error.to_string())
}

fn geometry_count(row: &AuditRow) -> f64 {
    row.trail_count as f64
}

fn accepted_count(row: &AuditRow) -> f64 {
    row.accepted_trail_count as f64
}

fn maximum_length(row: &AuditRow) -> f64 {
    row.max_length_px.unwrap_or(0.0)
}

fn write_artifacts(rows: &[AuditRow], out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    write_csv(rows, &out_dir.join("single_frame_trails.csv"))?;
    write_chart(
        &out_dir.join("single_frame_trail_counts.png"),
        rows,
        "SINGLE FRAME TRAIL AUDIT GEOMETRY VS MODEL",
        "CANDIDATE COUNT",
        "FRAME NUMBER",
        &[
            ("GEOMETRY PROPOSALS", [184, 162, 122], geometry_count),
            ("ACCEPTED LINE EVIDENCE", [79, 155, 131], accepted_count),
        ],
    )?;
    write_chart(
        &out_dir.join("single_frame_trail_lengths.png"),
        rows,
        "SINGLE FRAME TRAIL AUDIT MAX LENGTH",
        "PIXEL",
        "FRAME NUMBER",
        &[("MAX LENGTH", [79, 155, 131], maximum_length)],
    )?;
    Ok(())
}

fn run_audit(options: &CliOptions) -> Result<Vec<AuditRow>> {
    let paths = frame_paths(&options.input_path)?;
    let detector_defaults = DetectionOptions::default();
    let detection = DetectionOptions {
        measurement: crate::measurement::MeasurementOptions {
            aperture_radius: options.aperture_radius,
            psf_fwhm: options.psf_fwhm,
            min_flux_snr: options.min_flux_snr,
            ..detector_defaults.measurement
        },
        threshold_sigma: options.threshold_sigma,
        min_distance: options.min_distance,
        max_sources: options.max_sources,
        background_box_size: options.background_box_size,
        reject_linear_artifacts: true,
        ..detector_defaults
    };
    let frame_options = FrameOptions {
        detection,
        ..FrameOptions::default()
    };
    let mut rows = Vec::with_capacity(paths.len());
    for (offset, path) in paths.iter().enumerate() {
        let frame = fits::read_fits(path, false)?;
        let analysis = pipeline::analyze_frame(frame, None, None, &frame_options, None)?;
        let trails = sequence_motion_features::single_frame_components_from_analysis(
            &analysis,
            options.psf_fwhm,
            options.residual_sigma,
            options.min_residual_adu,
            options.min_feature_area,
            options.min_axis_ratio,
        )?;
        rows.push(make_row(offset + 1, path, &analysis, &trails)?);
    }
    Ok(rows)
}

/// Run the Rust desktop command with the existing `rst19-trails` arguments.
pub fn run_desktop(args: &[String]) -> Result<Value> {
    let options = parse_args(args)?;
    let rows = run_audit(&options)?;
    write_artifacts(&rows, &options.out_dir)?;
    Ok(json!({
        "output": options.out_dir.to_string_lossy(),
        "frames": rows.iter().map(AuditRow::as_json).collect::<Vec<_>>(),
        "artifacts": [
            options.out_dir.join("single_frame_trails.csv"),
            options.out_dir.join("single_frame_trail_counts.png"),
            options.out_dir.join("single_frame_trail_lengths.png"),
        ],
    }))
}

const CLI_HELP: &str = "逐张 FITS 审计单图长线候选并写出表格/曲线\n\nUsage: rst19-trails <FITS_FILE_OR_DIR> [OPTIONS]\n\nOptions:\n  --threshold-sigma N\n  --min-distance PIXELS\n  --aperture-radius PIXELS\n  --psf-fwhm PIXELS\n  --background-box-size PIXELS\n  --min-flux-snr N\n  --max-sources N\n  --residual-sigma N\n  --min-residual-adu ADU\n  --min-feature-area PIXELS\n  --min-axis-ratio N\n  --out-dir DIR\n  -h, --help\n";

/// Run the standalone Rust command while preserving the existing CLI's text
/// summary and exit-code contract.
pub fn run_cli(args: &[String]) -> i32 {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        println!("{CLI_HELP}");
        return 0;
    }
    let result = match run_desktop(args) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("rst19-trails: {error}");
            return 2;
        }
    };
    let output = result["output"].as_str().unwrap_or_default();
    let display_output = fs::canonicalize(output)
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|_| output.to_owned());
    println!("single-frame trail audit: {display_output}");
    if let Some(frames) = result["frames"].as_array() {
        for frame in frames {
            let frame_number = frame["frame_number"].as_u64().unwrap_or_default();
            let trails = frame["trail_count"].as_u64().unwrap_or_default();
            let edge = frame["edge_trail_count"].as_u64().unwrap_or_default();
            let length = frame["max_length_px"]
                .as_f64()
                .map(|value| format!("{value:.1}px"))
                .unwrap_or_else(|| "—".to_owned());
            println!("frame={frame_number:02} trails={trails} max_length={length} edge={edge}");
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_preserves_python_cli_defaults_and_accepts_inline_values() {
        let parsed = parse_args(&["frame.fits".into(), "--min-distance=6".into()]).unwrap();
        assert_eq!(parsed.input_path, PathBuf::from("frame.fits"));
        assert_eq!(parsed.min_distance, 6);
        assert_eq!(parsed.threshold_sigma, 4.0);
        assert_eq!(parsed.residual_sigma, 15.0);
    }

    #[test]
    fn standalone_cli_help_documents_the_rust_command_and_controls() {
        assert!(CLI_HELP.contains("Usage: rst19-trails"));
        for option in [
            "--threshold-sigma",
            "--max-sources",
            "--min-axis-ratio",
            "--out-dir",
        ] {
            assert!(CLI_HELP.contains(option), "missing {option} in help");
        }
    }

    #[test]
    fn accepted_gate_rejects_double_and_accepts_supported_line_evidence() {
        let double = json!({
            "length_px":12.0,
            "continuity_ratio":1.0,
            "profile_fill_ratio":1.0,
            "line_artifact_overlap_fraction":0.0,
            "single_frame_model":"double",
            "line_template_snr":100.0,
            "line_vs_point_delta_bic":100.0,
            "line_vs_double_delta_bic":-100.0,
            "line_vs_elliptical_delta_bic":100.0
        });
        let line = json!({
            "length_px":20.0,
            "continuity_ratio":0.8,
            "profile_fill_ratio":0.8,
            "line_artifact_overlap_fraction":0.0,
            "single_frame_model":"line",
            "line_template_snr":100.0,
            "line_vs_point_delta_bic":100.0,
            "line_vs_double_delta_bic":100.0,
            "line_vs_elliptical_delta_bic":100.0
        });
        assert!(!shape_supported(&double));
        assert!(shape_supported(&line));
    }

    #[test]
    fn model_fit_fields_feed_the_same_single_frame_acceptance_gate() {
        let image = (0..64)
            .map(|y| {
                (0..64)
                    .map(|x| {
                        if (18..=46).contains(&x) {
                            (90.0 * (-0.5 * ((y as f64 - 32.0) / 1.25).powi(2)).exp()) as f32
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect::<Vec<Vec<f32>>>();
        let valid = vec![vec![true; 64]; 64];
        let mut point = json!({
            "length_px": 28.0,
            "width_px": 3.0,
            "angle_deg": 0.0,
            "area_pixels": 87,
            "touches_edge": false,
            "residual_snr": 90.0,
            "continuity_ratio": 0.9,
            "profile_fill_ratio": 0.9,
            "line_artifact_overlap_fraction": 0.0
        });
        let evidence = crate::single_frame_model::evaluate(
            &image, &valid, 32.0, 32.0, 28.0, 3.0, 0.0, 3.0, 1.0,
        );
        point
            .as_object_mut()
            .unwrap()
            .extend(evidence.as_object().unwrap().clone());
        assert_eq!(point["single_frame_model"], "line");
        assert!(shape_supported(&point));
    }

    #[test]
    fn windows_path_output_uses_python_path_separator_style() {
        let output = display_python_path(Path::new("doc/frames/image.fits"));
        #[cfg(windows)]
        assert_eq!(output, "doc\\frames\\image.fits");
        #[cfg(not(windows))]
        assert_eq!(output, "doc/frames/image.fits");
    }
}
