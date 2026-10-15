//! Deterministic point-track cutouts measured from the original FITS frames.
use crate::{fits, models::FitsFrame, source_artifacts::Chart, statistics, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const BACKGROUND: [u8; 3] = [24, 32, 51];
const TEXT: [u8; 3] = [246, 241, 231];

#[derive(Serialize)]
struct AuditRow {
    group: String,
    group_label: String,
    track_id: u64,
    classification: String,
    presence: usize,
    frame_indices: String,
    min_flux_snr: Option<f64>,
    median_flux_snr: Option<f64>,
    max_flux_snr: Option<f64>,
    displacement_px: Option<f64>,
    fit_rms_px: Option<f64>,
    file: String,
}

fn points(track: &Value) -> Vec<&Value> {
    let mut points = track["points"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|point| point.is_object())
        .collect::<Vec<_>>();
    points.sort_by_key(|point| point["frame_index"].as_i64().unwrap_or(0));
    points
}

fn snr(track: &Value) -> f64 {
    points(track)
        .iter()
        .filter_map(|point| finite(point, "flux_snr"))
        .min_by(f64::total_cmp)
        .unwrap_or(f64::NEG_INFINITY)
}

fn finite(value: &Value, key: &str) -> Option<f64> {
    value[key].as_f64().filter(|number| number.is_finite())
}

fn tile(
    frame: &FitsFrame,
    x: f64,
    y: f64,
    median: f64,
    radius: usize,
    scale: usize,
    label: &str,
    colour: Option<[u8; 3]>,
) -> Result<Chart> {
    if !x.is_finite() || !y.is_finite() {
        return Err("sequence cutout coordinates must be finite".into());
    }
    let size = 2 * radius + 1;
    let left = x.round_ties_even() as i64 - radius as i64;
    let top = y.round_ties_even() as i64 - radius as i64;
    let mut values = vec![median as f32; size * size];
    for row in 0..size {
        for column in 0..size {
            let sx = left + column as i64;
            let sy = top + row as i64;
            if sx >= 0 && sy >= 0 && sx < frame.width as i64 && sy < frame.height as i64 {
                values[row * size + column] =
                    frame.data.value(sy as usize * frame.width + sx as usize) as f32;
            }
        }
    }
    let finite = values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let low = if finite.is_empty() {
        0.0
    } else {
        statistics::percentile(&finite, 2.0)?
    };
    let high = if finite.is_empty() {
        1.0
    } else {
        statistics::percentile(&finite, 99.5)?
    };
    let high = if high <= low { low + 1.0 } else { high };
    let mut tile = Chart::new(size * scale, size * scale, BACKGROUND);
    for row in 0..size {
        for column in 0..size {
            // The reference cutout performs normalization in float32.
            let value =
                ((values[row * size + column] - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
            let shade = if value.is_finite() {
                (value * 255.0).round_ties_even() as u8
            } else {
                0
            };
            tile.rect(
                (column * scale) as i32,
                (row * scale) as i32,
                ((column + 1) * scale - 1) as i32,
                ((row + 1) * scale - 1) as i32,
                [shade; 3],
            );
        }
    }
    tile.rect(0, 0, 50.min((size * scale - 1) as i32), 12, BACKGROUND);
    tile.text(1, 2, label, TEXT);
    if let Some(colour) = colour {
        let mx = (x - left as f64) * scale as f64;
        let my = (y - top as f64) * scale as f64;
        let r = 3usize.max(2 * scale) as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let distance = (dx * dx + dy * dy) as f64;
                if distance <= (r * r) as f64
                    && distance >= ((r - scale as i32).max(0)).pow(2) as f64
                {
                    tile.pixel(mx.round() as i32 + dx, my.round() as i32 + dy, colour);
                }
            }
        }
        tile.line(
            ((mx - 1.4 * r as f64) as i32, my as i32),
            ((mx + 1.4 * r as f64) as i32, my as i32),
            colour,
        );
        tile.line(
            (mx as i32, (my - 1.4 * r as f64) as i32),
            (mx as i32, (my + 1.4 * r as f64) as i32),
            colour,
        );
    } else {
        let end = (size * scale - 1) as i32;
        for (a, b) in [
            ((0, 0), (end, 0)),
            ((end, 0), (end, end)),
            ((end, end), (0, end)),
            ((0, end), (0, 0)),
        ] {
            tile.line(a, b, [105, 115, 134]);
        }
    }
    Ok(tile)
}

pub fn write_sequence_source_audit(
    sequence: &Value,
    output: &Path,
    per_group: usize,
    radius: usize,
    scale: usize,
) -> Result<PathBuf> {
    if per_group == 0 || radius < 3 || scale == 0 {
        return Err("sequence audit requires positive samples/scale and radius >= 3".into());
    }
    let frames = sequence["frames"]
        .as_array()
        .filter(|frames| !frames.is_empty())
        .ok_or("sequence audit requires nonempty frames")?;
    let tracks = sequence["tracks"]
        .as_array()
        .ok_or("sequence audit requires tracks")?;
    let frame_data = frames
        .iter()
        .map(|frame| {
            fits::read_fits(
                Path::new(
                    frame["path"]
                        .as_str()
                        .ok_or("sequence frame is missing path")?,
                ),
                false,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let medians = frame_data
        .iter()
        .map(|frame| {
            let values = (0..frame.data.len())
                .map(|index| frame.data.value(index) as f32)
                .filter(|value| value.is_finite())
                .map(f64::from)
                .collect::<Vec<_>>();
            if values.is_empty() {
                Ok(0.0)
            } else {
                statistics::median(&values)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let count = frames.len();
    let persistent = sequence["persistent_min_presence"]
        .as_u64()
        .unwrap_or(3usize.max(count.div_ceil(2)) as u64) as usize;
    let shifts = (0..count)
        .map(|index| {
            [
                sequence["cumulative_shifts"][index][0]
                    .as_f64()
                    .unwrap_or(0.0),
                sequence["cumulative_shifts"][index][1]
                    .as_f64()
                    .unwrap_or(0.0),
            ]
        })
        .collect::<Vec<_>>();
    fs::create_dir_all(output.join("source_track_cutouts")).map_err(|error| error.to_string())?;
    let output = fs::canonicalize(output).map_err(|error| error.to_string())?;
    let size = (2 * radius + 1) * scale;
    let width = count * size;
    let mut manifest = Vec::new();
    let mut sheets = Vec::new();
    for (group, label, colour) in [
        (
            "static_strict",
            "严格静态 · 至少约 80% 帧".to_string(),
            [79, 214, 170],
        ),
        (
            "persistent",
            format!("持续候选 · {persistent}/{count} 帧门槛"),
            [114, 185, 212],
        ),
        (
            "transient_one_frame",
            "瞬态 · 1 帧".to_string(),
            [217, 154, 78],
        ),
        (
            "transient_near_threshold",
            "瞬态 · 低于持续门槛".to_string(),
            [156, 139, 106],
        ),
    ] {
        let mut candidates = tracks
            .iter()
            .filter(|track| {
                let presence = points(track).len();
                presence > 0
                    && match group {
                        "static_strict" => track["classification"] == "static",
                        "persistent" => track["classification"] == "persistent",
                        "transient_one_frame" => {
                            track["classification"] == "transient" && presence == 1
                        }
                        _ => {
                            track["classification"] == "transient"
                                && presence > 1
                                && presence < persistent
                        }
                    }
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            snr(a).total_cmp(&snr(b)).then(
                a["track_id"]
                    .as_u64()
                    .unwrap_or(0)
                    .cmp(&b["track_id"].as_u64().unwrap_or(0)),
            )
        });
        let selected = if candidates.len() <= per_group {
            candidates.clone()
        } else {
            (0..per_group)
                .map(|index| {
                    candidates[if per_group == 1 {
                        0
                    } else {
                        (index as f64 * (candidates.len() - 1) as f64 / (per_group - 1) as f64)
                            .round_ties_even() as usize
                    }]
                })
                .collect()
        };
        if selected.is_empty() {
            continue;
        }
        let height = 25 + selected.len() * (size + 25);
        let mut sheet = Chart::new(width, height, BACKGROUND);
        sheet.text(
            3,
            5,
            &format!(
                "{group} sampled {}/{}; algorithmic class, not truth",
                selected.len(),
                candidates.len()
            ),
            [188, 224, 209],
        );
        for (row_index, track) in selected.iter().enumerate() {
            let points = points(track);
            let by_frame = points
                .iter()
                .filter_map(|point| {
                    point["frame_index"]
                        .as_u64()
                        .map(|index| (index as usize, *point))
                })
                .collect::<BTreeMap<_, _>>();
            let ax = statistics::median(
                &points
                    .iter()
                    .map(|point| {
                        finite(point, "aligned_x")
                            .or_else(|| finite(point, "x"))
                            .unwrap_or(0.0)
                    })
                    .collect::<Vec<_>>(),
            )?;
            let ay = statistics::median(
                &points
                    .iter()
                    .map(|point| {
                        finite(point, "aligned_y")
                            .or_else(|| finite(point, "y"))
                            .unwrap_or(0.0)
                    })
                    .collect::<Vec<_>>(),
            )?;
            let mut image = Chart::new(width, size + 25, BACKGROUND);
            let mut indices = Vec::new();
            let mut snrs = Vec::new();
            for (index, frame) in frame_data.iter().enumerate() {
                let point = by_frame.get(&index).copied();
                let x = point
                    .and_then(|point| finite(point, "x"))
                    .unwrap_or(ax + shifts[index][0]);
                let y = point
                    .and_then(|point| finite(point, "y"))
                    .unwrap_or(ay + shifts[index][1]);
                let value = point.and_then(|point| finite(point, "flux_snr"));
                if point.is_some() {
                    indices.push((index + 1).to_string());
                }
                if let Some(value) = value {
                    snrs.push(value);
                }
                let label = format!(
                    "F{:02} {}",
                    index + 1,
                    if point.is_none() {
                        "-".into()
                    } else {
                        value.map(|snr| format!("{snr:.0}")).unwrap_or("?".into())
                    }
                );
                image.paste(
                    &tile(
                        frame,
                        x,
                        y,
                        medians[index],
                        radius,
                        scale,
                        &label,
                        point.map(|_| colour),
                    )?,
                    (index * size) as i32,
                    25,
                );
            }
            let min = snrs.iter().copied().min_by(f64::total_cmp);
            let max = snrs.iter().copied().max_by(f64::total_cmp);
            let median = if snrs.is_empty() {
                None
            } else {
                Some(statistics::median(&snrs)?)
            };
            let id = track["track_id"].as_u64().unwrap_or(0);
            let format_snr = |value: Option<f64>| {
                value
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "nan".into())
            };
            image.text(
                3,
                6,
                &format!(
                    "{group} ID {id:05} {}/{count}F min/med/max SNR {}/{}/{} disp {:.2}px",
                    points.len(),
                    format_snr(min),
                    format_snr(median),
                    format_snr(max),
                    finite(track, "displacement_px").unwrap_or(0.0)
                ),
                TEXT,
            );
            let file = format!("source_track_cutouts/{group}_track{id:05}.png");
            image.save(&output.join(&file))?;
            sheet.paste(&image, 0, (25 + row_index * (size + 25)) as i32);
            manifest.push(AuditRow {
                group: group.into(),
                group_label: label.clone(),
                track_id: id,
                classification: track["classification"].as_str().unwrap_or("").into(),
                presence: points.len(),
                frame_indices: indices.join("|"),
                min_flux_snr: min,
                median_flux_snr: median,
                max_flux_snr: max,
                displacement_px: finite(track, "displacement_px"),
                fit_rms_px: finite(track, "fit_rms_px"),
                file,
            });
        }
        sheets.push((height, sheet));
    }
    let mut file = fs::File::create(output.join("source_track_audit.csv"))
        .map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut csv = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);
    csv.write_record([
        "group",
        "group_label",
        "track_id",
        "classification",
        "presence",
        "frame_indices",
        "min_flux_snr",
        "median_flux_snr",
        "max_flux_snr",
        "displacement_px",
        "fit_rms_px",
        "file",
    ])
    .map_err(|error| error.to_string())?;
    for row in &manifest {
        csv.serialize(row).map_err(|error| error.to_string())?;
    }
    csv.flush().map_err(|error| error.to_string())?;
    let class_count = |class: &str| {
        tracks
            .iter()
            .filter(|track| track["classification"] == class)
            .count()
    };
    let summary = json!({"frame_count":count,"strict_static_count":class_count("static"),"persistent_count":class_count("persistent"),"transient_count":class_count("transient"),"persistent_min_presence":persistent,"sampled_track_count":manifest.len(),"note":"抽样时序图用于人工检查点状持续性，不是逐星真值；缺帧位置以灰框显示。"});
    fs::write(
        output.join("source_track_audit_summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let mut contact = Chart::new(
        width.max(1),
        sheets
            .iter()
            .map(|(height, _)| height)
            .sum::<usize>()
            .max(40),
        BACKGROUND,
    );
    if sheets.is_empty() {
        contact.text(4, 12, "SOURCE TRACK AUDIT - no tracks selected", TEXT);
    }
    let mut offset = 0;
    for (height, sheet) in sheets {
        contact.paste(&sheet, 0, offset as i32);
        offset += height;
    }
    contact.save(&output.join("source_track_contact_sheet.png"))?;
    Ok(output)
}
