//! Native registered-image diagnostics and innovation-report artifact writers.
use crate::{fits, models::FitsFrame, source_artifacts::Chart, statistics, Result};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub struct ImagePlane {
    pub values: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

fn contributions(source: usize, target: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = source as f64 / target as f64;
    let support = scale.max(1.0);
    (0..target)
        .map(|output| {
            let center = (output as f64 + 0.5) * scale;
            let start = ((center - support + 0.5).floor() as isize).max(0) as usize;
            let end = ((center + support + 0.5).floor() as isize)
                .max(0)
                .min(source as isize) as usize;
            let mut weights = (start..end)
                .map(|index| {
                    (
                        index,
                        (1.0 - ((index as f64 - center + 0.5) / support).abs()).max(0.0),
                    )
                })
                .collect::<Vec<_>>();
            let sum = weights.iter().map(|(_, weight)| weight).sum::<f64>();
            if sum > 0.0 {
                for (_, weight) in &mut weights {
                    *weight /= sum;
                }
            }
            weights
        })
        .collect()
}

fn resize_f32(
    values: &[f32],
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<f32> {
    if width == target_width && height == target_height {
        return values.to_vec();
    }
    let xweights = contributions(width, target_width);
    let yweights = contributions(height, target_height);
    let mut horizontal = vec![0.0_f32; target_width * height];
    for y in 0..height {
        for x in 0..target_width {
            horizontal[y * target_width + x] = xweights[x]
                .iter()
                .map(|(sx, weight)| f64::from(values[y * width + sx]) * weight)
                .sum::<f64>() as f32;
        }
    }
    let mut output = vec![0.0_f32; target_width * target_height];
    for y in 0..target_height {
        for x in 0..target_width {
            output[y * target_width + x] = yweights[y]
                .iter()
                .map(|(sy, weight)| f64::from(horizontal[sy * target_width + x]) * weight)
                .sum::<f64>() as f32;
        }
    }
    output
}

fn shifted(plane: &ImagePlane, sx: f64, sy: f64) -> ImagePlane {
    let mut values = vec![f32::NAN; plane.values.len()];
    for y in 0..plane.height {
        for x in 0..plane.width {
            let px = x as f64 + sx;
            let py = y as f64 + sy;
            if px < 0.0
                || py < 0.0
                || px > (plane.width - 1) as f64
                || py > (plane.height - 1) as f64
            {
                continue;
            }
            let x0 = px.floor() as usize;
            let y0 = py.floor() as usize;
            let x1 = (x0 + 1).min(plane.width - 1);
            let y1 = (y0 + 1).min(plane.height - 1);
            let dx = px - x0 as f64;
            let dy = py - y0 as f64;
            let a = f64::from(plane.values[y0 * plane.width + x0]) * (1.0 - dx)
                + f64::from(plane.values[y0 * plane.width + x1]) * dx;
            let b = f64::from(plane.values[y1 * plane.width + x0]) * (1.0 - dx)
                + f64::from(plane.values[y1 * plane.width + x1]) * dx;
            values[y * plane.width + x] = (a * (1.0 - dy) + b * dy) as f32;
        }
    }
    ImagePlane {
        values,
        width: plane.width,
        height: plane.height,
    }
}

pub fn registered_stack(
    rows: &[Value],
    max_width: usize,
    normalize: bool,
) -> Result<Vec<ImagePlane>> {
    if rows.is_empty() {
        return Ok(vec![]);
    }
    if max_width == 0 {
        return Err("registered image width must be positive".into());
    }
    let first = fits::read_fits(
        Path::new(rows[0]["path"].as_str().ok_or("frame is missing path")?),
        false,
    )?;
    let width = max_width.min(first.width);
    let height = ((first.height as f64 * width as f64 / first.width as f64).round_ties_even()
        as usize)
        .max(1);
    let lows = rows
        .iter()
        .filter_map(|row| row["p01_adu"].as_f64())
        .collect::<Vec<_>>();
    let highs = rows
        .iter()
        .filter_map(|row| row["p998_adu"].as_f64())
        .collect::<Vec<_>>();
    let low = if lows.is_empty() {
        0.0
    } else {
        statistics::median(&lows)?
    };
    let high = if highs.is_empty() {
        low + 1.0
    } else {
        statistics::median(&highs)?
    };
    let high = if high <= low { low + 1.0 } else { high };
    let mut output = Vec::new();
    for row in rows {
        let owned;
        let frame = if row["path"] == rows[0]["path"] {
            &first
        } else {
            owned = fits::read_fits(
                Path::new(row["path"].as_str().ok_or("frame is missing path")?),
                false,
            )?;
            &owned
        };
        let mut valid = vec![0.0_f32; frame.data.len()];
        let mut source = vec![0.0_f32; frame.data.len()];
        for index in 0..source.len() {
            let value = frame.data.value(index) as f32;
            if value.is_finite() && !(index < frame.width && index < 104) {
                valid[index] = 1.0;
                source[index] = if normalize {
                    ((value - low as f32) / (high - low) as f32).clamp(0.0, 1.0)
                } else {
                    value
                };
            }
        }
        let mut thumbnail = resize_f32(&source, frame.width, frame.height, width, height);
        let coverage = resize_f32(&valid, frame.width, frame.height, width, height);
        for (pixel, coverage) in thumbnail.iter_mut().zip(coverage) {
            if coverage < 0.5 {
                *pixel = f32::NAN;
            }
        }
        let sx = row["cumulative_shift_x_px"].as_f64().unwrap_or(0.0) * width as f64
            / first.width as f64;
        let sy = row["cumulative_shift_y_px"].as_f64().unwrap_or(0.0) * height as f64
            / first.height as f64;
        output.push(shifted(
            &ImagePlane {
                values: thumbnail,
                width,
                height,
            },
            sx,
            sy,
        ));
    }
    Ok(output)
}

fn reference(stack: &[ImagePlane]) -> Result<Vec<f32>> {
    let Some(first) = stack.first() else {
        return Ok(vec![]);
    };
    if stack
        .iter()
        .any(|plane| plane.width != first.width || plane.height != first.height)
    {
        return Err("registered image shapes differ".into());
    }
    let mut output = Vec::with_capacity(first.values.len());
    for index in 0..first.values.len() {
        let values = stack
            .iter()
            .map(|plane| plane.values[index])
            .filter(|value| value.is_finite())
            .map(f64::from)
            .collect::<Vec<_>>();
        output.push(if values.is_empty() {
            f32::NAN
        } else {
            statistics::median(&values)? as f32
        });
    }
    Ok(output)
}

pub fn difference_rows(
    rows: &[Value],
    stack: &[ImagePlane],
    reference: &[f32],
) -> Result<Vec<Value>> {
    if rows.len() != stack.len()
        || stack
            .iter()
            .any(|plane| plane.values.len() != reference.len())
    {
        return Err("difference frames and reference do not match".into());
    }
    rows.iter().zip(stack).enumerate().map(|(index,(row,plane))| {
        let values = plane.values.iter().zip(reference).map(|(value,reference)|f64::from(*value)-f64::from(*reference)).filter(|value|value.is_finite()).collect::<Vec<_>>();
        let noise = row["background_rms_adu"].as_f64().filter(|value|*value != 0.0).unwrap_or(1.0).max(f64::EPSILON);
        if values.is_empty() { return Ok(json!({"frame_index":row["frame_index"].as_i64().unwrap_or(index as i64),"timestamp":row["timestamp"],"residual_median_adu":null,"residual_rms_adu":null,"residual_p99_abs_adu":null,"positive_peak_sigma":null,"negative_peak_sigma":null,"pixels_abs_gt_5sigma":0,"valid_fraction":0.0})); }
        let median = statistics::median(&values)?;
        let mad = statistics::median(&values.iter().map(|value|(value-median).abs()).collect::<Vec<_>>())?;
        let abs = values.iter().map(|value|value.abs()).collect::<Vec<_>>();
        Ok(json!({"frame_index":row["frame_index"].as_i64().unwrap_or(index as i64),"timestamp":row["timestamp"],"residual_median_adu":median,"residual_rms_adu":1.4826*mad,"residual_p99_abs_adu":statistics::percentile(&abs,99.0)?,"positive_peak_sigma":values.iter().copied().max_by(f64::total_cmp).map(|value|value/noise),"negative_peak_sigma":values.iter().copied().min_by(f64::total_cmp).map(|value|value/noise),"pixels_abs_gt_5sigma":abs.iter().filter(|value|**value >= 5.0*noise).count(),"valid_fraction":values.len() as f64/reference.len() as f64}))
    }).collect()
}

fn difference_chart(stack: &[ImagePlane], reference: &[f32]) -> Result<Chart> {
    let Some(first) = stack.first() else {
        return Ok(Chart::new(1, 1, [24, 32, 51]));
    };
    let mut signed = vec![0.0; reference.len()];
    for (index, reference) in reference.iter().enumerate() {
        let (mut positive, mut negative) = (0.0_f64, 0.0_f64);
        for plane in stack {
            let residual = f64::from(plane.values[index]) - f64::from(*reference);
            if residual.is_finite() {
                positive = positive.max(residual);
                negative = negative.min(residual);
            }
        }
        signed[index] = if positive >= negative.abs() {
            positive
        } else {
            negative
        };
    }
    let abs = signed.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let clip = statistics::percentile(&abs, 99.8)?.max(f64::EPSILON);
    let mut chart = Chart::new(first.width, first.height, [20, 28, 44]);
    for y in 0..first.height {
        for x in 0..first.width {
            let residual = signed[y * first.width + x];
            let strength = (residual.abs() / clip).clamp(0.0, 1.0);
            let colour = if residual >= 0.0 {
                [239.0, 111.0, 70.0]
            } else {
                [73.0, 145.0, 221.0]
            };
            let neutral = [20.0, 28.0, 44.0];
            chart.pixel(
                x as i32,
                y as i32,
                std::array::from_fn(|index| {
                    (neutral[index] + (colour[index] - neutral[index]) * strength).clamp(0.0, 255.0)
                        as u8
                }),
            );
        }
    }
    Ok(chart)
}

fn csv_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}
pub fn write_csv(path: &Path, rows: &[Value], excluded: &[&str]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    if rows.is_empty() {
        return Ok(());
    }
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut columns = Vec::new();
    for row in rows {
        if let Some(fields) = row.as_object() {
            for key in fields.keys() {
                if !excluded.contains(&key.as_str()) && !columns.contains(key) {
                    columns.push(key.clone());
                }
            }
        }
    }
    let mut writer = csv::Writer::from_writer(file);
    writer
        .write_record(&columns)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(columns.iter().map(|key| csv_text(&row[key])))
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

pub fn motion_point_rows(tracks: &[Value]) -> Vec<Value> {
    let mut output = Vec::new();
    for track in tracks {
        for point in track["points"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|point| point.is_object())
        {
            let index = point["frame_index"].as_i64().unwrap_or(0);
            let mut row = json!({"track_id":track["track_id"],"classification":track["classification"].as_str().unwrap_or("candidate"),"frame_index":index,"frame_number":index.saturating_add(1),
                "x_px":point["x"],"y_px":point["y"],"aligned_x_px":point["aligned_x"],"aligned_y_px":point["aligned_y"],"single_frame_model":point["single_frame_model"],"touches_edge":point["touches_edge"].as_bool().unwrap_or(false),"bbox":point["bbox"].as_array().map(|bbox|bbox.iter().map(csv_text).collect::<Vec<_>>().join(","))});
            for key in [
                "length_px",
                "width_px",
                "angle_deg",
                "residual_snr",
                "continuity_ratio",
                "profile_fill_ratio",
                "line_artifact_overlap_fraction",
                "line_template_snr",
                "line_vs_point_delta_bic",
                "line_vs_double_delta_bic",
                "line_vs_background_delta_bic",
                "line_vs_elliptical_delta_bic",
                "single_frame_model_margin_bic",
                "fit_residual_px",
            ] {
                row[key] = point[key].clone();
            }
            output.push(row);
        }
    }
    output
}

pub fn line_chart(
    path: &Path,
    title: &str,
    x: &[f64],
    series: &[(&str, Vec<f64>, [u8; 3])],
    include_zero: bool,
) -> Result<()> {
    let mut chart = Chart::new(1200, 680, [24, 32, 51]);
    chart.text(25, 20, title, [244, 240, 231]);
    let xs = x
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let ys = series
        .iter()
        .flat_map(|(_, values, _)| values.iter().copied())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let mut xmin = xs.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0);
    let mut xmax = xs.iter().copied().max_by(f64::total_cmp).unwrap_or(1.0);
    let mut ymin = ys.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0);
    let mut ymax = ys.iter().copied().max_by(f64::total_cmp).unwrap_or(1.0);
    if include_zero {
        ymin = ymin.min(0.0);
        ymax = ymax.max(0.0);
    }
    if xmax <= xmin {
        xmin -= 0.5;
        xmax += 0.5;
    }
    if ymax <= ymin {
        ymin -= 0.5;
        ymax += 0.5;
    }
    let padding = 0.05 * (ymax - ymin);
    ymin -= padding;
    ymax += padding;
    let xy = |x: f64, y: f64| {
        (
            (90.0 + 1060.0 * (x - xmin) / (xmax - xmin)).round() as i32,
            (610.0 - 520.0 * (y - ymin) / (ymax - ymin)).round() as i32,
        )
    };
    for grid in 0..=5 {
        let value = ymin + (ymax - ymin) * grid as f64 / 5.0;
        let y = xy(xmin, value).1;
        chart.line((90, y), (1150, y), [47, 57, 74]);
        chart.text(10, y - 3, &format!("{value:.3}"), [193, 206, 224]);
    }
    chart.line((90, 90), (90, 610), [193, 206, 224]);
    chart.line((90, 610), (1150, 610), [193, 206, 224]);
    for (index, (name, values, colour)) in series.iter().enumerate() {
        chart.text(100 + (index as i32 % 4) * 260, 650, name, *colour);
        let mut previous = None;
        for (x, y) in x.iter().zip(values) {
            if !x.is_finite() || !y.is_finite() {
                previous = None;
                continue;
            }
            let point = xy(*x, *y);
            if let Some(previous) = previous {
                chart.line(previous, point, *colour);
            }
            chart.rect(point.0 - 2, point.1 - 2, point.0 + 2, point.1 + 2, *colour);
            previous = Some(point);
        }
    }
    chart.text(90, 625, &format!("{xmin:.3}"), [193, 206, 224]);
    chart.text(1090, 625, &format!("{xmax:.3}"), [193, 206, 224]);
    chart.save(path)
}

fn trajectory_chart(path: &Path, title: &str, tracks: &[Value]) -> Result<()> {
    let mut chart = Chart::new(1200, 680, [24, 32, 51]);
    chart.text(25, 20, title, [244, 240, 231]);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for track in tracks {
        for point in track["points"].as_array().into_iter().flatten() {
            if let Some((x, y)) = point["aligned_x"]
                .as_f64()
                .or_else(|| point["x"].as_f64())
                .zip(point["aligned_y"].as_f64().or_else(|| point["y"].as_f64()))
            {
                xs.push(x);
                ys.push(y);
            }
        }
        if let Some((x, y)) = track["predicted_x_px"]
            .as_f64()
            .zip(track["predicted_y_px"].as_f64())
        {
            let ex = track["predicted_x_ci95_half_width_px"]
                .as_f64()
                .unwrap_or(0.0);
            let ey = track["predicted_y_ci95_half_width_px"]
                .as_f64()
                .unwrap_or(0.0);
            xs.extend([x - ex, x + ex]);
            ys.extend([y - ey, y + ey]);
        }
    }
    let xmin = xs.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0) - 5.0;
    let xmax = xs.iter().copied().max_by(f64::total_cmp).unwrap_or(1.0) + 5.0;
    let ymin = ys.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0) - 5.0;
    let ymax = ys.iter().copied().max_by(f64::total_cmp).unwrap_or(1.0) + 5.0;
    let xy = |x: f64, y: f64| {
        (
            (80.0 + 1060.0 * (x - xmin) / (xmax - xmin)).round() as i32,
            (70.0 + 540.0 * (y - ymin) / (ymax - ymin)).round() as i32,
        )
    };
    chart.text(
        25,
        640,
        "IMAGE PLANE PIXELS; FORECAST BARS = FITTED MEAN 95% CI",
        [193, 206, 224],
    );
    for track in tracks {
        let colour = if track["classification"] == "moving" {
            [255, 59, 212]
        } else {
            [239, 125, 69]
        };
        let mut previous = None;
        for point in track["points"].as_array().into_iter().flatten() {
            let Some((x, y)) = point["aligned_x"]
                .as_f64()
                .or_else(|| point["x"].as_f64())
                .zip(point["aligned_y"].as_f64().or_else(|| point["y"].as_f64()))
            else {
                continue;
            };
            let point = xy(x, y);
            if let Some(previous) = previous {
                chart.line(previous, point, colour);
            }
            chart.rect(point.0 - 3, point.1 - 3, point.0 + 3, point.1 + 3, colour);
            previous = Some(point);
        }
        if let Some((x, y)) = track["predicted_x_px"]
            .as_f64()
            .zip(track["predicted_y_px"].as_f64())
        {
            let point = xy(x, y);
            if let Some(previous) = previous {
                chart.line(previous, point, [114, 185, 212]);
            }
            let ex = track["predicted_x_ci95_half_width_px"]
                .as_f64()
                .unwrap_or(0.0);
            let ey = track["predicted_y_ci95_half_width_px"]
                .as_f64()
                .unwrap_or(0.0);
            chart.line(xy(x - ex, y), xy(x + ex, y), [114, 185, 212]);
            chart.line(xy(x, y - ey), xy(x, y + ey), [114, 185, 212]);
            chart.text(
                point.0 + 5,
                point.1 + 5,
                &format!("ID {} FORECAST", track["track_id"]),
                [114, 185, 212],
            );
        }
    }
    chart.save(path)
}

fn telemetry_position_chart(path: &Path, rows: &[Value], predictions: &Value) -> Result<()> {
    let mut chart = Chart::new(1200, 680, [24, 32, 51]);
    chart.text(
        25,
        20,
        "AUXILIARY POSITIONS: SEPARATE REFERENCE FRAMES",
        [244, 240, 231],
    );
    for (panel, prefix) in ["j2000", "wgs84"].iter().enumerate() {
        let observed = rows
            .iter()
            .filter_map(|row| {
                row[format!("{prefix}_x")]
                    .as_f64()
                    .zip(row[format!("{prefix}_y")].as_f64())
            })
            .collect::<Vec<_>>();
        let forecast = &predictions[*prefix]["predicted_position_m_assuming_m_per_s"];
        let predicted = forecast[0].as_f64().zip(forecast[1].as_f64());
        let mut bounds = observed.clone();
        if let Some(predicted) = predicted {
            bounds.push(predicted);
        }
        let mut xmin = bounds
            .iter()
            .map(|point| point.0)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let mut xmax = bounds
            .iter()
            .map(|point| point.0)
            .max_by(f64::total_cmp)
            .unwrap_or(1.0);
        let mut ymin = bounds
            .iter()
            .map(|point| point.1)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let mut ymax = bounds
            .iter()
            .map(|point| point.1)
            .max_by(f64::total_cmp)
            .unwrap_or(1.0);
        if xmax <= xmin {
            xmin -= 0.5;
            xmax += 0.5;
        }
        if ymax <= ymin {
            ymin -= 0.5;
            ymax += 0.5;
        }
        let pad_x = (xmax - xmin) * 0.05;
        let pad_y = (ymax - ymin) * 0.05;
        xmin -= pad_x;
        xmax += pad_x;
        ymin -= pad_y;
        ymax += pad_y;
        let left = 60 + panel as i32 * 590;
        let xy = |x: f64, y: f64| {
            (
                left + (500.0 * (x - xmin) / (xmax - xmin)).round() as i32,
                590 - (490.0 * (y - ymin) / (ymax - ymin)).round() as i32,
            )
        };
        chart.text(
            left,
            70,
            &format!("{}: X / M, Y / M", prefix.to_uppercase()),
            [114, 185, 212],
        );
        chart.line((left, 100), (left, 590), [193, 206, 224]);
        chart.line((left, 590), (left + 500, 590), [193, 206, 224]);
        let mut previous = None;
        for point in &observed {
            let point = xy(point.0, point.1);
            if let Some(previous) = previous {
                chart.line(previous, point, [79, 155, 131]);
            }
            chart.rect(
                point.0 - 3,
                point.1 - 3,
                point.0 + 3,
                point.1 + 3,
                [79, 155, 131],
            );
            previous = Some(point);
        }
        if let Some((x, y)) = predicted {
            let point = xy(x, y);
            if let Some(previous) = previous {
                chart.line(previous, point, [217, 148, 50]);
            }
            chart.rect(
                point.0 - 4,
                point.1 - 4,
                point.0 + 4,
                point.1 + 4,
                [217, 148, 50],
            );
            chart.text(point.0 + 6, point.1, "FORECAST", [217, 148, 50]);
        }
        chart.text(
            left,
            610,
            &format!("X RANGE {xmin:.2} .. {xmax:.2}"),
            [193, 206, 224],
        );
    }
    chart.text(
        25,
        650,
        "LAST STATE CONSTANT VELOCITY; VELOCITY UNIT UNCONFIRMED; NOT ORBIT PROPAGATION",
        [193, 206, 224],
    );
    chart.save(path)
}

fn motion_cutouts(report: &Value, output: &Path) -> Result<()> {
    let frames = report["frames"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut cache = std::collections::BTreeMap::<usize, FitsFrame>::new();
    let mut summaries = Vec::new();
    let mut crops = Vec::new();
    fs::create_dir_all(output.join("motion_cutouts")).map_err(|error| error.to_string())?;
    for track in report["motion"]["tracks"].as_array().into_iter().flatten() {
        let id = track["track_id"].as_i64().unwrap_or(0);
        for point in track["points"].as_array().into_iter().flatten() {
            let Some(index) = point["frame_index"]
                .as_u64()
                .map(|value| value as usize)
                .filter(|index| *index < frames.len())
            else {
                continue;
            };
            if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(index) {
                entry.insert(fits::read_fits(
                    Path::new(
                        frames[index]["path"]
                            .as_str()
                            .ok_or("cutout frame is missing path")?,
                    ),
                    false,
                )?);
            }
            let frame = &cache[&index];
            let x = point["x"].as_f64().unwrap_or(0.0);
            let y = point["y"].as_f64().unwrap_or(0.0);
            let length = point["length_px"].as_f64().unwrap_or(20.0);
            let width = point["width_px"].as_f64().unwrap_or(3.0);
            let bbox = point["bbox"]
                .as_array()
                .filter(|bbox| bbox.len() >= 4)
                .and_then(|bbox| {
                    bbox.iter()
                        .take(4)
                        .map(Value::as_f64)
                        .collect::<Option<Vec<_>>>()
                });
            let half = (length / 2.0).round_ties_even().max(16.0) as i64;
            let bbox = bbox.map_or(
                [
                    x.round_ties_even() as i64 - half,
                    y.round_ties_even() as i64 - half,
                    x.round_ties_even() as i64 + half,
                    y.round_ties_even() as i64 + half,
                ],
                |bbox| std::array::from_fn(|index| bbox[index] as i64),
            );
            let margin = (width * 3.0).round_ties_even().max(12.0) as i64;
            let left = (bbox[0] - margin).max(0) as usize;
            let top = (bbox[1] - margin).max(0) as usize;
            let right = (bbox[2] + margin + 1).max(0).min(frame.width as i64) as usize;
            let bottom = (bbox[3] + margin + 1).max(0).min(frame.height as i64) as usize;
            if right <= left || bottom <= top {
                continue;
            }
            let cw = right - left;
            let ch = bottom - top;
            let values = (top..bottom)
                .flat_map(|y| (left..right).map(move |x| frame.data.value(y * frame.width + x)))
                .collect::<Vec<_>>();
            let finite = values
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            let low = if finite.is_empty() {
                0.0
            } else {
                statistics::percentile(&finite, 1.0)?
            };
            let high = if finite.is_empty() {
                1.0
            } else {
                statistics::percentile(&finite, 99.7)?
            };
            let high = if high <= low { low + 1.0 } else { high };
            let mut crop = Chart::new(cw, ch, [24, 32, 51]);
            for cy in 0..ch {
                for cx in 0..cw {
                    let value = values[cy * cw + cx];
                    let shade = if value.is_finite() {
                        (((value - low) / (high - low)).clamp(0.0, 1.0) * 255.0).round_ties_even()
                            as u8
                    } else {
                        0
                    };
                    crop.pixel(cx as i32, cy as i32, [shade; 3]);
                }
            }
            let colour = if track["classification"] == "moving" {
                [255, 59, 212]
            } else {
                [239, 125, 69]
            };
            let angle = point["angle_deg"].as_f64().unwrap_or(0.0).to_radians();
            let dx = angle.cos() * length / 2.0;
            let dy = angle.sin() * length / 2.0;
            let cx = x - left as f64;
            let cy = y - top as f64;
            crop.line(
                ((cx - dx) as i32, (cy - dy) as i32),
                ((cx + dx) as i32, (cy + dy) as i32),
                colour,
            );
            crop.line(
                ((cx - 5.0) as i32, cy as i32),
                ((cx + 5.0) as i32, cy as i32),
                colour,
            );
            crop.line(
                (cx as i32, (cy - 5.0) as i32),
                (cx as i32, (cy + 5.0) as i32),
                colour,
            );
            crop.rect(3, 3, 190.min(cw.saturating_sub(3) as i32), 18, [24, 32, 51]);
            crop.text(
                6,
                5,
                &format!("TRAIL {id:04} F{:02}", index + 1),
                [244, 240, 231],
            );
            let filename = format!("motion_cutouts/track_{id:04}_frame_{:02}.png", index + 1);
            crop.save(&output.join(&filename))?;
            let mut row = point.clone();
            row["track_id"] = json!(id);
            row["classification"] = track["classification"].clone();
            row["timestamp"] = frames[index]["timestamp"].clone();
            row["x_px"] = json!(x);
            row["y_px"] = json!(y);
            for (key, value) in [
                ("crop_left", left),
                ("crop_top", top),
                ("crop_right", right),
                ("crop_bottom", bottom),
            ] {
                row[key] = json!(value);
            }
            row["path"] = json!(filename);
            summaries.push(row);
            crops.push((crop, cw, ch, id, index));
        }
    }
    write_csv(&output.join("motion_cutouts.csv"), &summaries, &[])?;
    if !crops.is_empty() {
        let mut sheet = Chart::new(960, crops.len().div_ceil(3) * 240, [24, 32, 51]);
        for (index, (crop, width, height, id, frame)) in crops.iter().enumerate() {
            // Keep original-pixel evidence in each full-resolution PNG; the contact view is a bounded preview.
            let x = (index % 3) * 320;
            let y = (index / 3) * 240;
            let scale = (300.0 / *width as f64).min(205.0 / *height as f64).min(1.0);
            let target_width = (*width as f64 * scale).round_ties_even().max(1.0) as usize;
            let target_height = (*height as f64 * scale).round_ties_even().max(1.0) as usize;
            let rgba = crop
                .rgb_data()
                .chunks_exact(3)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
                .collect::<Vec<_>>();
            let resized = crate::mosaic::resize_rgba_lanczos3(
                &rgba,
                *width,
                *height,
                target_width,
                target_height,
            )?;
            for row in 0..target_height {
                for column in 0..target_width {
                    let offset = (row * target_width + column) * 4;
                    sheet.pixel(
                        (x + column) as i32,
                        (y + 25 + row) as i32,
                        [resized[offset], resized[offset + 1], resized[offset + 2]],
                    );
                }
            }
            sheet.text(
                x as i32 + 8,
                y as i32 + 8,
                &format!("TRAIL {id:04} F{:02}", frame + 1),
                [244, 240, 231],
            );
        }
        sheet.save(&output.join("motion_contact_sheet.png"))?;
    }
    Ok(())
}

pub fn write_innovation_artifacts(report: &Value, output: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let output = fs::canonicalize(output).map_err(|error| error.to_string())?;
    let frames = report["frames"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let lines = report["motion"]["tracks"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let points = report["point_motion"]["tracks"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    write_csv(&output.join("frame_evidence.csv"), frames, &[])?;
    write_csv(
        &output.join("telemetry_prediction.csv"),
        &crate::innovation::telemetry_prediction_rows(
            &report["telemetry_predictions"]["predictions"],
        ),
        &[],
    )?;
    write_csv(&output.join("motion_evidence.csv"), lines, &["points"])?;
    write_csv(
        &output.join("motion_points_evidence.csv"),
        &motion_point_rows(lines),
        &[],
    )?;
    write_csv(
        &output.join("point_motion_evidence.csv"),
        points,
        &["points"],
    )?;
    write_csv(
        &output.join("point_motion_points_evidence.csv"),
        &motion_point_rows(points),
        &[],
    )?;
    write_csv(
        &output.join("motion_evidence_links.csv"),
        report["motion"]["evidence_links"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]),
        &[],
    )?;
    let audits = report["motion"]["frame_audits"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    write_csv(&output.join("motion_frame_audit.csv"), audits, &[])?;
    let ax = audits
        .iter()
        .enumerate()
        .map(|(index, row)| row["frame_number"].as_f64().unwrap_or(index as f64 + 1.0))
        .collect::<Vec<_>>();
    line_chart(
        &output.join("motion_frame_audit.png"),
        "MOTION FRAME AUDIT",
        &ax,
        &[
            (
                "components",
                audits
                    .iter()
                    .map(|row| row["component_count"].as_f64().unwrap_or(0.0))
                    .collect(),
                [217, 148, 50],
            ),
            (
                "features",
                audits
                    .iter()
                    .map(|row| row["feature_count"].as_f64().unwrap_or(0.0))
                    .collect(),
                [79, 155, 131],
            ),
        ],
        true,
    )?;
    motion_cutouts(report, &output)?;
    let stack = registered_stack(frames, 1200, false)?;
    let reference = reference(&stack)?;
    let differences = difference_rows(frames, &stack, &reference)?;
    write_csv(&output.join("difference_evidence.csv"), &differences, &[])?;
    difference_chart(&stack, &reference)?.save(&output.join("registered_difference.png"))?;
    let noise = differences
        .iter()
        .filter_map(|row| row["residual_rms_adu"].as_f64())
        .collect::<Vec<_>>();
    let mut rendered = report.clone();
    rendered["difference"] = json!({"frame_count":differences.len(),"residual_rms_median_adu":if noise.is_empty() { None } else { Some(statistics::median(&noise)?) },"note":"这是平移配准后相对中值参考的残差基线；未做 PSF matching 或 proper image subtraction。"});
    fs::write(
        output.join("innovation_report.json"),
        serde_json::to_vec_pretty(&rendered).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let x = frames
        .iter()
        .enumerate()
        .map(|(index, row)| row["frame_index"].as_f64().unwrap_or(index as f64) + 1.0)
        .collect::<Vec<_>>();
    line_chart(
        &output.join("quality_counts.png"),
        "FRAME QUALITY: LOG1P COUNTS",
        &x,
        &[
            (
                "candidate",
                frames
                    .iter()
                    .map(|row| row["candidate_count"].as_f64().unwrap_or(0.0).ln_1p())
                    .collect(),
                [217, 148, 50],
            ),
            (
                "quality / workset",
                frames
                    .iter()
                    .map(|row| row["quality_count"].as_f64().unwrap_or(0.0).ln_1p())
                    .collect(),
                [79, 155, 131],
            ),
        ],
        false,
    )?;
    line_chart(
        &output.join("registration_shift.png"),
        "REGISTRATION BASELINE: CUMULATIVE SHIFT PX",
        &x,
        &[(
            "shift norm",
            frames
                .iter()
                .map(|row| row["cumulative_shift_norm_px"].as_f64().unwrap_or(0.0))
                .collect(),
            [114, 185, 212],
        )],
        true,
    )?;
    line_chart(
        &output.join("telemetry_velocity.png"),
        "AUXILIARY TELEMETRY: VELOCITY NORM M/S?",
        &x,
        &[
            (
                "J2000",
                frames
                    .iter()
                    .map(|row| row["j2000_velocity_norm_raw"].as_f64().unwrap_or(f64::NAN))
                    .collect(),
                [199, 53, 158],
            ),
            (
                "WGS84",
                frames
                    .iter()
                    .map(|row| row["wgs84_velocity_norm_raw"].as_f64().unwrap_or(f64::NAN))
                    .collect(),
                [114, 185, 212],
            ),
        ],
        true,
    )?;
    trajectory_chart(
        &output.join("motion_trajectory.png"),
        "LINE MOTION: CONSTANT-VELOCITY OLS",
        lines,
    )?;
    trajectory_chart(
        &output.join("point_motion_trajectory.png"),
        "POINT MOTION: CONSTANT-VELOCITY OLS",
        points,
    )?;
    telemetry_position_chart(
        &output.join("telemetry_position.png"),
        frames,
        &report["telemetry_predictions"]["predictions"],
    )?;
    let normalized = registered_stack(frames, 1200, true)?;
    let composite = self::reference(&normalized)?;
    if let Some(first) = normalized.first() {
        let mut mosaic = Chart::new(first.width, first.height, [0; 3]);
        for y in 0..first.height {
            for x in 0..first.width {
                let value = composite[y * first.width + x];
                let shade = if value.is_finite() {
                    (value * 255.0).clamp(0.0, 255.0) as u8
                } else {
                    0
                };
                mosaic.pixel(x as i32, y as i32, [shade; 3]);
            }
        }
        mosaic.save(&output.join("registered_mosaic.png"))?;
    }
    Ok(output)
}
