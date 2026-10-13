//! Source-coordinate PCA elongation audit with NumPy-compatible sampling.
//!
//! This diagnostic describes detector-coordinate distributions only. It does
//! not establish physical motion, artifact truth, or catalog identity.

use crate::Result;
use csv::{Terminator, WriterBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub const DEFAULT_CONTROL_CLASS: &str = "compact_quality";
const GUARDRAILS: [&str; 4] = [
    "PCA 细长性是 detector 坐标分布诊断，不是物理运动真值。",
    "control_axis_ratio_upper_tail_fraction 是有限次经验抽样的描述量，不是 p 值或 FDR。",
    "feature_class 和 compact_quality 都来自同一检测器源表，不能视为独立标注真值。",
    "线状候选仍需原始像素、线段模型、跨帧运动和官方真值联合核验。",
];
const CSV_FIELDS: [&str; 21] = [
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "pca_axis_ratio",
    "pca_orientation_deg",
    "perpendicular_residual_median_px",
    "perpendicular_residual_p90_px",
    "perpendicular_residual_max_px",
    "control_class",
    "control_sample_count",
    "control_trial_count",
    "control_axis_ratio_median",
    "control_axis_ratio_p95",
    "control_axis_ratio_p99",
    "control_axis_ratio_max",
    "control_axis_ratio_exceed_count",
    "control_axis_ratio_upper_tail_fraction",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SourceGeometrySummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub candidate_count: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub pca_axis_ratio: f64,
    pub pca_orientation_deg: f64,
    pub perpendicular_residual_median_px: f64,
    pub perpendicular_residual_p90_px: f64,
    pub perpendicular_residual_max_px: f64,
    pub control_class: String,
    pub control_sample_count: usize,
    pub control_trial_count: usize,
    pub control_axis_ratio_median: Option<f64>,
    pub control_axis_ratio_p95: Option<f64>,
    pub control_axis_ratio_p99: Option<f64>,
    pub control_axis_ratio_max: Option<f64>,
    pub control_axis_ratio_exceed_count: usize,
    pub control_axis_ratio_upper_tail_fraction: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SourceGeometryAuditResult {
    pub catalog_path: String,
    pub control_class: String,
    pub control_trial_count: usize,
    pub random_seed: u64,
    pub target_classes: Vec<String>,
    pub summaries: Vec<SourceGeometrySummary>,
}

impl SourceGeometryAuditResult {
    pub fn as_dict(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("serializable audit result");
        value["interpretation_guardrails"] = json!(GUARDRAILS);
        value
    }
}

#[derive(Clone, Debug)]
struct SourcePoint {
    x: f64,
    y: f64,
    label: String,
}

fn python_repr(value: &str) -> String {
    if !value.contains('\'') {
        format!("'{value}'")
    } else if !value.contains('"') {
        format!("\"{value}\"")
    } else {
        format!(
            "'{}'",
            value
                .replace('\\', "\\\\")
                .replace('\'', "\\'")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t")
        )
    }
}

fn read_catalog(path: &Path) -> Result<BTreeMap<String, Vec<SourcePoint>>> {
    let file = fs::File::open(path).map_err(|_| format!("无法读取源级 CSV：{}", path.display()))?;
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(file);
    let raw_headers = reader.headers().map_err(|error| error.to_string())?;
    let headers = raw_headers
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index == 0 {
                value.trim_start_matches('\u{feff}')
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let required = ["feature_class_label", "feature_class", "x", "y"];
    let missing = required
        .iter()
        .filter(|field| !headers.contains(field))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let index = |field: &str| headers.iter().position(|header| *header == field).unwrap();
    let x_index = index("x");
    let y_index = index("y");
    let class_index = index("feature_class");
    let label_index = index("feature_class_label");
    let mut grouped = BTreeMap::<String, Vec<SourcePoint>>::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let feature_class = record
            .get(class_index)
            .unwrap_or_default()
            .trim()
            .to_owned();
        if feature_class.is_empty() {
            return Err("source catalog contains a row without feature_class".into());
        }
        let parse_coordinate = |field_index: usize| -> Option<f64> {
            let value = record.get(field_index)?.trim();
            if value.is_empty() {
                return None;
            }
            let value = value.parse::<f64>().ok()?;
            value.is_finite().then_some(value)
        };
        let (Some(x), Some(y)) = (parse_coordinate(x_index), parse_coordinate(y_index)) else {
            return Err(format!(
                "feature_class={} contains non-finite x/y",
                python_repr(&feature_class)
            ));
        };
        let label = record.get(label_index).unwrap_or(&feature_class).trim();
        grouped
            .entry(feature_class.clone())
            .or_default()
            .push(SourcePoint {
                x,
                y,
                label: if label.is_empty() {
                    feature_class
                } else {
                    label.to_owned()
                },
            });
    }
    if grouped.is_empty() {
        return Err(format!("source catalog is empty: {}", path.display()));
    }
    Ok(grouped)
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn pca_geometry(points: &[(f64, f64)]) -> Result<(f64, f64, Vec<f64>)> {
    if points.len() < 2 {
        return Err("PCA 几何审计至少需要两个二维坐标".into());
    }
    let count = points.len() as f64;
    let mean_x = points.iter().map(|point| point.0).sum::<f64>() / count;
    let mean_y = points.iter().map(|point| point.1).sum::<f64>() / count;
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    let centered = points
        .iter()
        .map(|&(x, y)| (x - mean_x, y - mean_y))
        .collect::<Vec<_>>();
    for &(x, y) in &centered {
        xx += x * x;
        xy += x * y;
        yy += y * y;
    }
    let discriminant = (xx - yy).hypot(2.0 * xy);
    let major_eigenvalue = ((xx + yy + discriminant) / 2.0).max(0.0);
    let minor_eigenvalue = ((xx + yy - discriminant) / 2.0).max(0.0);
    let major = major_eigenvalue.sqrt();
    let minor = minor_eigenvalue.sqrt();
    let ratio = major / minor.max(f64::EPSILON);
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let angle_deg = angle.to_degrees();
    let orientation = ((angle_deg + 90.0) % 180.0) - 90.0;
    let residuals = centered
        .iter()
        .map(|&(x, y)| (-x * angle.sin() + y * angle.cos()).abs())
        .collect();
    Ok((ratio, orientation, residuals))
}

// NumPy 1.26.4's SeedSequence uses four uint32 pool words and the constants
// below. The fixed output preserves historical default_rng(seed).choice results.
fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(0x931e_8875);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> 16)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = 0xca01_f9dd_u32
        .wrapping_mul(x)
        .wrapping_sub(0x4973_f715_u32.wrapping_mul(y));
    result ^ (result >> 16)
}

fn seed_sequence_pool(seed: u128) -> [u32; 4] {
    let mut entropy = Vec::new();
    let mut remaining = seed;
    if remaining == 0 {
        entropy.push(0);
    } else {
        while remaining != 0 {
            entropy.push(remaining as u32);
            remaining >>= 32;
        }
    }
    let mut pool = [0_u32; 4];
    let mut hash_const = 0x43b0_d7e5;
    for (index, word) in pool.iter_mut().enumerate() {
        *word = hashmix(entropy.get(index).copied().unwrap_or(0), &mut hash_const);
    }
    for source in 0..pool.len() {
        for destination in 0..pool.len() {
            if source != destination {
                let mixed = hashmix(pool[source], &mut hash_const);
                pool[destination] = mix(pool[destination], mixed);
            }
        }
    }
    for &word in entropy.iter().skip(pool.len()) {
        for destination in &mut pool {
            let mixed = hashmix(word, &mut hash_const);
            *destination = mix(*destination, mixed);
        }
    }
    pool
}

#[derive(Clone, Debug)]
struct NumpyPcg64 {
    state: u128,
    increment: u128,
    has_uint32: bool,
    uint32_cache: u32,
}

impl NumpyPcg64 {
    fn new(seed: u128) -> Self {
        let pool = seed_sequence_pool(seed);
        let mut hash_const = 0x8b51_f9dd_u32;
        let mut words = [0_u32; 8];
        for (index, word) in words.iter_mut().enumerate() {
            let mut value = pool[index % pool.len()] ^ hash_const;
            hash_const = hash_const.wrapping_mul(0x58f3_8ded);
            value = value.wrapping_mul(hash_const);
            *word = value ^ (value >> 16);
        }
        let mut generated = [0_u64; 4];
        for index in 0..4 {
            generated[index] = words[index * 2] as u64 | ((words[index * 2 + 1] as u64) << 32);
        }
        let initial_state = ((generated[0] as u128) << 64) | generated[1] as u128;
        let initial_sequence = ((generated[2] as u128) << 64) | generated[3] as u128;
        let mut rng = Self {
            state: 0,
            increment: initial_sequence.wrapping_shl(1) | 1,
            has_uint32: false,
            uint32_cache: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initial_state);
        rng.step();
        rng
    }

    fn step(&mut self) {
        let multiplier = ((2_549_297_995_355_413_924_u128) << 64) | 4_865_540_595_714_422_341_u128;
        self.state = self
            .state
            .wrapping_mul(multiplier)
            .wrapping_add(self.increment);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        (high ^ low).rotate_right((high >> 58) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            self.uint32_cache
        } else {
            let value = self.next_u64();
            self.has_uint32 = true;
            self.uint32_cache = (value >> 32) as u32;
            value as u32
        }
    }

    fn bounded(&mut self, maximum: usize) -> usize {
        let maximum = maximum as u64;
        if maximum == 0 {
            return 0;
        }
        if maximum == u32::MAX as u64 {
            return self.next_u32() as usize;
        }
        if maximum == u64::MAX {
            return self.next_u64() as usize;
        }
        if maximum <= u32::MAX as u64 {
            let exclusive_range = maximum as u32 + 1;
            loop {
                let product = self.next_u32() as u64 * exclusive_range as u64;
                let leftover = product as u32;
                if leftover < exclusive_range {
                    let threshold = (u32::MAX - maximum as u32) % exclusive_range;
                    if leftover < threshold {
                        continue;
                    }
                }
                return (product >> 32) as usize;
            }
        }
        let exclusive_range = maximum + 1;
        loop {
            let product = self.next_u64() as u128 * exclusive_range as u128;
            let leftover = product as u64;
            if leftover < exclusive_range {
                let threshold = (u64::MAX - maximum) % exclusive_range;
                if leftover < threshold {
                    continue;
                }
            }
            return (product >> 64) as usize;
        }
    }
}

fn choice_indices(population: usize, sample_count: usize, rng: &mut NumpyPcg64) -> Vec<usize> {
    if population > 10_000 && sample_count > population / 50 {
        let mut indices = (0..population).collect::<Vec<_>>();
        let first = population.saturating_sub(sample_count).max(1);
        for index in (first..population).rev() {
            let chosen = rng.bounded(index);
            indices.swap(index, chosen);
        }
        return indices[population - sample_count..].to_vec();
    }

    // Floyd's algorithm, followed by NumPy's Fisher-Yates shuffle.
    let set_capacity = ((sample_count as f64 * 1.2) as usize).max(1);
    let mut mask = set_capacity as u64;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    let mask = mask as usize;
    let mut slots = vec![None; mask + 1];
    let mut indices = Vec::with_capacity(sample_count);
    for upper in (population - sample_count)..population {
        let value = rng.bounded(upper);
        let mut location = value & mask;
        while slots[location].is_some_and(|occupied| occupied != value) {
            location = (location + 1) & mask;
        }
        if slots[location].is_none() {
            slots[location] = Some(value);
            indices.push(value);
        } else {
            let mut fallback = upper & mask;
            while slots[fallback].is_some() {
                fallback = (fallback + 1) & mask;
            }
            slots[fallback] = Some(upper);
            indices.push(upper);
        }
    }
    for index in (1..sample_count).rev() {
        let chosen = rng.bounded(index);
        indices.swap(index, chosen);
    }
    indices
}

pub fn run(
    catalog_path: &Path,
    target_classes: Option<&[String]>,
    control_class: &str,
    control_trial_count: usize,
    random_seed: u64,
) -> Result<SourceGeometryAuditResult> {
    if control_trial_count < 1 {
        return Err("control_trial_count 必须为正整数".into());
    }
    let grouped = read_catalog(catalog_path)?;
    let control_rows = grouped.get(control_class).ok_or_else(|| {
        format!(
            "source catalog has no control class: {}",
            python_repr(control_class)
        )
    })?;
    let requested = match target_classes {
        Some(classes) => {
            let mut seen = HashSet::new();
            let values = classes
                .iter()
                .map(|class| class.trim())
                .filter(|class| !class.is_empty())
                .filter(|class| seen.insert((*class).to_owned()))
                .map(str::to_owned)
                .collect::<Vec<_>>();
            values
        }
        None => grouped
            .keys()
            .filter(|name| name.as_str() != "other_rejected")
            .cloned()
            .collect(),
    };
    if requested.is_empty() {
        return Err("target_classes 不能为空".into());
    }
    for name in &requested {
        if !grouped.contains_key(name) {
            return Err(format!(
                "source catalog has no target class: {}",
                python_repr(name)
            ));
        }
    }
    let control_points = control_rows
        .iter()
        .map(|point| (point.x, point.y))
        .collect::<Vec<_>>();
    let mut summaries = Vec::with_capacity(requested.len());
    for (class_index, feature_class) in requested.iter().enumerate() {
        let class_rows = &grouped[feature_class];
        let points = class_rows
            .iter()
            .map(|point| (point.x, point.y))
            .collect::<Vec<_>>();
        if points.len() < 2 {
            return Err(format!(
                "feature_class={} 至少需要两个坐标",
                python_repr(feature_class)
            ));
        }
        let (axis_ratio, orientation, mut residuals) = pca_geometry(&points)?;
        residuals.sort_by(f64::total_cmp);
        let sample_count = points.len().min(control_points.len());
        let mut control_ratios = Vec::new();
        if feature_class != control_class {
            if sample_count < 2 {
                return Err("control 类别至少需要两个坐标".into());
            }
            if sample_count > control_points.len() {
                return Err("control 类别坐标数量不足以构造等样本量对照".into());
            }
            let seed = u128::from(random_seed) + class_index as u128;
            let mut rng = NumpyPcg64::new(seed);
            control_ratios.reserve(control_trial_count);
            for _ in 0..control_trial_count {
                let selected = choice_indices(control_points.len(), sample_count, &mut rng);
                let sample = selected
                    .iter()
                    .map(|&index| control_points[index])
                    .collect::<Vec<_>>();
                control_ratios.push(pca_geometry(&sample)?.0);
            }
        }
        control_ratios.sort_by(f64::total_cmp);
        let exceed_count = control_ratios
            .iter()
            .filter(|&&ratio| ratio >= axis_ratio)
            .count();
        let control_axis_ratio_upper_tail_fraction = (!control_ratios.is_empty())
            .then_some(exceed_count as f64 / control_ratios.len() as f64);
        summaries.push(SourceGeometrySummary {
            feature_class: feature_class.clone(),
            feature_class_label: class_rows[0].label.clone(),
            candidate_count: points.len(),
            x_min: points
                .iter()
                .map(|point| point.0)
                .fold(f64::INFINITY, f64::min),
            x_max: points
                .iter()
                .map(|point| point.0)
                .fold(f64::NEG_INFINITY, f64::max),
            y_min: points
                .iter()
                .map(|point| point.1)
                .fold(f64::INFINITY, f64::min),
            y_max: points
                .iter()
                .map(|point| point.1)
                .fold(f64::NEG_INFINITY, f64::max),
            pca_axis_ratio: axis_ratio,
            pca_orientation_deg: orientation,
            perpendicular_residual_median_px: quantile(&residuals, 0.5),
            perpendicular_residual_p90_px: quantile(&residuals, 0.9),
            perpendicular_residual_max_px: residuals.last().copied().unwrap_or(0.0),
            control_class: control_class.to_owned(),
            control_sample_count: sample_count,
            control_trial_count: control_ratios.len(),
            control_axis_ratio_median: (!control_ratios.is_empty())
                .then(|| quantile(&control_ratios, 0.5)),
            control_axis_ratio_p95: (!control_ratios.is_empty())
                .then(|| quantile(&control_ratios, 0.95)),
            control_axis_ratio_p99: (!control_ratios.is_empty())
                .then(|| quantile(&control_ratios, 0.99)),
            control_axis_ratio_max: control_ratios.last().copied(),
            control_axis_ratio_exceed_count: exceed_count,
            control_axis_ratio_upper_tail_fraction,
        });
    }
    Ok(SourceGeometryAuditResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        control_class: control_class.to_owned(),
        control_trial_count,
        random_seed,
        target_classes: requested,
        summaries,
    })
}

fn csv_record(row: &SourceGeometrySummary) -> Vec<String> {
    vec![
        row.feature_class.clone(),
        row.feature_class_label.clone(),
        row.candidate_count.to_string(),
        row.x_min.to_string(),
        row.x_max.to_string(),
        row.y_min.to_string(),
        row.y_max.to_string(),
        row.pca_axis_ratio.to_string(),
        row.pca_orientation_deg.to_string(),
        row.perpendicular_residual_median_px.to_string(),
        row.perpendicular_residual_p90_px.to_string(),
        row.perpendicular_residual_max_px.to_string(),
        row.control_class.clone(),
        row.control_sample_count.to_string(),
        row.control_trial_count.to_string(),
        row.control_axis_ratio_median
            .map_or_else(String::new, |v| v.to_string()),
        row.control_axis_ratio_p95
            .map_or_else(String::new, |v| v.to_string()),
        row.control_axis_ratio_p99
            .map_or_else(String::new, |v| v.to_string()),
        row.control_axis_ratio_max
            .map_or_else(String::new, |v| v.to_string()),
        row.control_axis_ratio_exceed_count.to_string(),
        row.control_axis_ratio_upper_tail_fraction
            .map_or_else(String::new, |v| v.to_string()),
    ]
}

pub fn write_artifacts(result: &SourceGeometryAuditResult, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let json_path = out_dir.join("source_geometry_audit.json");
    fs::write(
        &json_path,
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    let csv_path = out_dir.join("source_geometry_summary.csv");
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(Vec::new());
    if result.summaries.is_empty() {
        writer.write_field("").map_err(|error| error.to_string())?;
        let mut bytes = writer.into_inner().map_err(|error| error.to_string())?;
        bytes.clear();
        bytes.extend_from_slice(b"\xef\xbb\xbf\r\n");
        fs::write(&csv_path, bytes).map_err(|error| error.to_string())?;
        return Ok(out_dir.to_path_buf());
    }
    writer
        .write_record(CSV_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in &result.summaries {
        writer
            .write_record(csv_record(row))
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(bytes.len() + 3);
    output.extend_from_slice(b"\xef\xbb\xbf");
    output.append(&mut bytes);
    fs::write(&csv_path, output).map_err(|error| error.to_string())?;
    Ok(out_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut positional = None;
    let mut targets = Vec::new();
    let mut control_class = DEFAULT_CONTROL_CLASS.to_owned();
    let mut trials = 5_000_usize;
    let mut seed = 1_909_u64;
    let mut output = PathBuf::from("tmp/source-geometry-audit");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        if matches!(
            name,
            "--target-class" | "--control-class" | "--trials" | "--seed" | "--out-dir"
        ) {
            let value = if let Some(value) = inline {
                value
            } else {
                index += 1;
                args.get(index).ok_or("option requires a value")?
            };
            match name {
                "--target-class" => targets.push(value.to_owned()),
                "--control-class" => control_class = value.to_owned(),
                "--trials" => {
                    let parsed = value.parse::<i64>().map_err(|error| error.to_string())?;
                    if parsed < 1 {
                        return Err("control_trial_count 必须为正整数".into());
                    }
                    trials =
                        usize::try_from(parsed).map_err(|_| "control_trial_count is too large")?;
                }
                "--seed" => {
                    seed = value.parse().map_err(|error: std::num::ParseIntError| {
                        if value.starts_with('-') {
                            "expected non-negative integer".to_owned()
                        } else {
                            error.to_string()
                        }
                    })?
                }
                "--out-dir" => output = PathBuf::from(value),
                _ => unreachable!(),
            }
        } else if name == "--" {
            index += 1;
            if let Some(value) = args.get(index) {
                positional = Some(PathBuf::from(value));
            }
            break;
        } else if name.starts_with('-') {
            return Err(format!("unknown argument {name}"));
        } else if positional.replace(PathBuf::from(name)).is_some() {
            return Err("expected exactly one source catalog path".into());
        }
        index += 1;
    }
    let catalog = positional.ok_or("expected exactly one source catalog path")?;
    let result = run(
        &catalog,
        (!targets.is_empty()).then_some(targets.as_slice()),
        &control_class,
        trials,
        seed,
    )?;
    write_artifacts(&result, &output)?;
    Ok(json!({
        "tool": "rst19-source-geometry-audit",
        "out_dir": output,
        "result_path": output.join("source_geometry_audit.json"),
        "result": result.as_dict(),
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        println!(
            "只读取 source_catalog，审计各特征类别的 detector 坐标细长性\n\nUsage: rst19-source-geometry-audit CATALOG [--target-class CLASS]... [--control-class CLASS] [--trials N] [--seed N] [--out-dir DIR]"
        );
        return Ok(());
    }
    let value = run_desktop(args)?;
    println!(
        "rst19-source-geometry-audit: 结果已写入 {}",
        value["out_dir"].as_str().unwrap_or_default()
    );
    let result: SourceGeometryAuditResult =
        serde_json::from_value(value["result"].clone()).map_err(|error| error.to_string())?;
    for row in result.summaries {
        let tail = row
            .control_axis_ratio_upper_tail_fraction
            .map_or_else(|| "-".to_owned(), |value| format!("{value:.4}"));
        println!(
            "{}: n={} axis_ratio={:.3} residual_p50/p90={:.3}/{:.3} control_tail={tail}",
            row.feature_class,
            row.candidate_count,
            row.pca_axis_ratio,
            row.perpendicular_residual_median_px,
            row.perpendicular_residual_p90_px,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg64_raw_stream_matches_numpy_1_26_4() {
        let mut rng = NumpyPcg64::new(19);
        assert_eq!(
            (0..5).map(|_| rng.next_u64()).collect::<Vec<_>>(),
            [
                7_754_620_058_831_902_006,
                17_079_274_849_526_874_851,
                5_052_009_576_229_543_119,
                1_107_701_232_231_750_491,
                5_728_515_699_266_668_138,
            ]
        );
    }

    #[test]
    fn choice_matches_numpy_floyd_and_tail_shuffle_paths() {
        let mut rng = NumpyPcg64::new(19);
        assert_eq!(choice_indices(12, 4, &mut rng), [3, 11, 5, 4]);
        let mut rng = NumpyPcg64::new(19);
        assert_eq!(
            choice_indices(12, 12, &mut rng),
            [9, 6, 0, 3, 11, 1, 5, 4, 2, 7, 10, 8]
        );
        let mut rng = NumpyPcg64::new(1909);
        assert_eq!(choice_indices(100, 2, &mut rng), [48, 38]);
        let mut rng = NumpyPcg64::new(123_456);
        assert_eq!(
            choice_indices(10_001, 250, &mut rng)[..10],
            [5_449, 9_400, 7_073, 9_410, 6_751, 3_633, 2_610, 9_325, 9_290, 655]
        );
    }

    #[test]
    fn large_integer_seed_uses_the_same_seed_sequence_words() {
        let mut rng = NumpyPcg64::new((1_u128 << 64) + 3);
        assert_eq!(rng.next_u64(), 13_362_612_774_902_755_871);
    }
}
