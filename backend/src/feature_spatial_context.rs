//! Spatial summaries for source-catalog feature classes and hard negatives.
//!
//! This audit reads an existing source CSV only. Its labels describe detector
//! rules and neighborhood structure, not physical source identity.

use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const SPATIAL_CONTEXT_METRIC_FIELDS: [&str; 3] = ["flux_snr", "filter_snr", "peak"];
const GUARDRAILS: [&str; 4] = [
    "空间集中只表示候选分布，不能单独证明固定坏区或伪影。",
    "edge_distance_px 使用 detector 像素边界，不是天球边界，也不等于有效孔径。",
    "高 SNR 落选计数是 hard-negative 诊断，不是误检率或真星概率。",
    "邻近候选可能来自同一混合结构；必须结合 PSF、值域、独占像素和星表/WCS。",
];

#[derive(Debug, Clone, Serialize)]
pub struct FeatureSpatialClassSummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub class_count: usize,
    pub class_quality_count: usize,
    pub class_rejected_count: usize,
    pub grid_size: usize,
    pub nonempty_grid_cells: usize,
    pub max_cell_count: usize,
    pub max_cell_share: f64,
    pub normalized_grid_entropy: f64,
    pub edge_margin_px: f64,
    pub edge_count: usize,
    pub edge_fraction: f64,
    pub high_flux_snr_threshold: f64,
    pub high_snr_rejected_count: usize,
    pub high_snr_max_cell_count: usize,
    pub high_snr_max_cell_share: f64,
    pub high_snr_hot_grid_row: Option<usize>,
    pub high_snr_hot_grid_column: Option<usize>,
    pub high_snr_hotspot_enrichment: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureSpatialHotspot {
    pub feature_class: String,
    pub feature_class_label: String,
    pub grid_row: usize,
    pub grid_column: usize,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub rejected_count: usize,
    pub high_snr_rejected_count: usize,
    pub candidate_share: f64,
    pub high_snr_share: f64,
    pub high_snr_enrichment: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureSpatialTarget {
    pub detection_id: i64,
    pub selection_reason: String,
    pub rank_within_class: Option<usize>,
    pub feature_class: String,
    pub feature_class_label: String,
    pub quality_passed: bool,
    pub flags: String,
    pub x: f64,
    pub y: f64,
    pub grid_row: usize,
    pub grid_column: usize,
    pub edge_distance_px: f64,
    pub is_edge: bool,
    pub flux_snr: Option<f64>,
    pub filter_snr: Option<f64>,
    pub class_count: usize,
    pub class_cell_count: usize,
    pub class_cell_share: f64,
    pub cell_high_snr_rejected_count: usize,
    pub cell_high_snr_share: f64,
    pub nearest_other_candidate_px: Option<f64>,
    pub neighbor_count_radius_5px: usize,
    pub neighbor_count_radius_10px: usize,
    pub neighbor_count_radius_20px: usize,
    pub neighbor_quality_count_radius_10px: usize,
    pub neighbor_classes_radius_10px: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureSpatialContextResult {
    pub catalog_path: String,
    pub image_width: usize,
    pub image_height: usize,
    pub grid_size: usize,
    pub edge_margin_px: f64,
    pub high_flux_snr_threshold: f64,
    pub metric: String,
    pub top_n: usize,
    pub excluded_classes: Vec<String>,
    pub summaries: Vec<FeatureSpatialClassSummary>,
    pub hotspots: Vec<FeatureSpatialHotspot>,
    pub targets: Vec<FeatureSpatialTarget>,
}

#[derive(Debug, Clone)]
pub struct FeatureSpatialContextOptions {
    pub image_width: usize,
    pub image_height: usize,
    pub grid_size: usize,
    pub edge_margin_px: f64,
    pub high_flux_snr_threshold: f64,
    pub top_n: usize,
    pub metric: String,
    pub target_ids: Vec<i64>,
    pub exclude_classes: Vec<String>,
}

impl Default for FeatureSpatialContextOptions {
    fn default() -> Self {
        Self {
            image_width: 4096,
            image_height: 4096,
            grid_size: 4,
            edge_margin_px: 16.0,
            high_flux_snr_threshold: 10.0,
            top_n: 5,
            metric: "flux_snr".into(),
            target_ids: Vec::new(),
            exclude_classes: vec!["other_rejected".into()],
        }
    }
}

impl FeatureSpatialContextResult {
    pub fn as_dict(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("finite spatial context result");
        value
            .as_object_mut()
            .expect("spatial context result object")
            .insert("interpretation_guardrails".into(), json!(GUARDRAILS));
        value
    }
}

#[derive(Debug, Clone)]
struct CatalogRow {
    detection_id: i64,
    x: f64,
    y: f64,
    feature_class: String,
    feature_class_label: String,
    quality_passed: bool,
    flags: String,
    flux_snr: Option<f64>,
    filter_snr: Option<f64>,
    peak: Option<f64>,
    grid_row: usize,
    grid_column: usize,
}

impl CatalogRow {
    fn metric(&self, metric: &str) -> Option<f64> {
        match metric {
            "flux_snr" => self.flux_snr,
            "filter_snr" => self.filter_snr,
            "peak" => self.peak,
            _ => None,
        }
    }
}

fn csv_reader(path: &Path) -> Result<csv::Reader<std::io::Cursor<Vec<u8>>>, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    Ok(csv::ReaderBuilder::new().from_reader(std::io::Cursor::new(bytes.to_vec())))
}

fn headers<R: std::io::Read>(
    reader: &mut csv::Reader<R>,
) -> Result<HashMap<String, usize>, String> {
    Ok(reader
        .headers()
        .map_err(|error| error.to_string())?
        .iter()
        .enumerate()
        .map(|(index, name)| (name.to_owned(), index))
        .collect())
}

fn field<'a>(record: &'a csv::StringRecord, index: &HashMap<String, usize>, name: &str) -> &'a str {
    index
        .get(name)
        .and_then(|column| record.get(*column))
        .unwrap_or_default()
}

fn finite_float(raw: &str) -> Option<f64> {
    if matches!(raw, "" | "nan" | "NaN" | "None") {
        return None;
    }
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn grid_index(value: f64, size: usize, grid_size: usize) -> usize {
    let cell = (value / size as f64 * grid_size as f64) as i64;
    cell.max(0).min(grid_size as i64 - 1) as usize
}

fn unique_strings(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .iter()
        .filter(|value| seen.insert((*value).clone()))
        .cloned()
        .collect()
}

fn normalized_entropy(cells: &BTreeMap<(usize, usize), usize>, grid_size: usize) -> f64 {
    let total: usize = cells.values().sum();
    if total == 0 || grid_size <= 1 {
        return 0.0;
    }
    let log_probability_sum = cells
        .values()
        .filter(|count| **count > 0)
        .map(|count| {
            let probability = *count as f64 / total as f64;
            probability * probability.ln()
        })
        .sum::<f64>();
    -log_probability_sum / ((grid_size * grid_size) as f64).ln()
}

fn required_catalog_rows(
    path: &Path,
    image_width: usize,
    image_height: usize,
    grid_size: usize,
) -> Result<Vec<CatalogRow>, String> {
    let mut reader =
        csv_reader(path).map_err(|_| format!("无法读取源级 CSV：{}", path.display()))?;
    let index = headers(&mut reader)?;
    let mut missing: Vec<_> = [
        "detection_id",
        "x",
        "y",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flags",
    ]
    .into_iter()
    .filter(|name| !index.contains_key(*name))
    .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!("源级 CSV 缺少列：{}", missing.join(", ")));
    }
    let mut rows = Vec::new();
    let mut seen_ids = HashSet::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let raw_id = field(&record, &index, "detection_id");
        let detection_id = raw_id
            .trim()
            .parse::<i64>()
            .map_err(|_| format!("源级 CSV 存在非法 detection_id：{raw_id:?}"))?;
        if !seen_ids.insert(detection_id) {
            return Err(format!("源级 CSV 存在重复 detection_id：{detection_id}"));
        }
        let raw_x = field(&record, &index, "x");
        let raw_y = field(&record, &index, "y");
        let x = finite_float(raw_x)
            .ok_or_else(|| format!("detection_id={detection_id} 缺少有限 x/y"))?;
        let y = finite_float(raw_y)
            .ok_or_else(|| format!("detection_id={detection_id} 缺少有限 x/y"))?;
        let feature_class = field(&record, &index, "feature_class").to_owned();
        if feature_class.is_empty() {
            return Err(format!("detection_id={detection_id} 缺少 feature_class"));
        }
        rows.push(CatalogRow {
            detection_id,
            x,
            y,
            feature_class,
            feature_class_label: field(&record, &index, "feature_class_label").to_owned(),
            quality_passed: field(&record, &index, "quality_passed")
                .trim()
                .eq_ignore_ascii_case("true"),
            flags: field(&record, &index, "flags").to_owned(),
            flux_snr: finite_float(field(&record, &index, "flux_snr")),
            filter_snr: finite_float(field(&record, &index, "filter_snr")),
            peak: finite_float(field(&record, &index, "peak")),
            grid_row: grid_index(y, image_height, grid_size),
            grid_column: grid_index(x, image_width, grid_size),
        });
    }
    if rows.is_empty() {
        return Err(format!("源级 CSV 为空：{}", path.display()));
    }
    Ok(rows)
}

pub fn run(
    catalog_path: &Path,
    options: &FeatureSpatialContextOptions,
) -> Result<FeatureSpatialContextResult, String> {
    let image_width = options.image_width;
    let image_height = options.image_height;
    let grid_size = options.grid_size;
    let edge_margin_px = options.edge_margin_px;
    let high_flux_snr_threshold = options.high_flux_snr_threshold;
    let top_n = options.top_n;
    let metric = options.metric.as_str();
    let target_ids = options.target_ids.as_slice();
    let exclude_classes = options.exclude_classes.as_slice();
    if image_width == 0 || image_height == 0 {
        return Err("image_width/image_height 必须为正整数".into());
    }
    if grid_size == 0 {
        return Err("grid_size 必须为正整数".into());
    }
    if !edge_margin_px.is_finite() || edge_margin_px < 0.0 {
        return Err("edge_margin_px 不能为负数".into());
    }
    if !high_flux_snr_threshold.is_finite() || high_flux_snr_threshold < 0.0 {
        return Err("high_flux_snr_threshold 不能为负数".into());
    }
    if top_n == 0 {
        return Err("top_n 必须为正整数".into());
    }
    if !SPATIAL_CONTEXT_METRIC_FIELDS.contains(&metric) {
        return Err(format!(
            "不支持的空间 hard-negative 指标：{metric:?}；可选项为 {}",
            SPATIAL_CONTEXT_METRIC_FIELDS.join(", ")
        ));
    }

    let rows = required_catalog_rows(catalog_path, image_width, image_height, grid_size)?;
    let excluded = unique_strings(exclude_classes);
    let requested_targets: Vec<i64> = {
        let mut seen = HashSet::new();
        target_ids
            .iter()
            .copied()
            .filter(|value| seen.insert(*value))
            .collect()
    };
    let by_id: BTreeMap<i64, usize> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| (row.detection_id, index))
        .collect();
    let mut missing_targets: Vec<i64> = requested_targets
        .iter()
        .filter(|target| !by_id.contains_key(target))
        .copied()
        .collect();
    missing_targets.sort_unstable();
    if !missing_targets.is_empty() {
        return Err(format!("源级 CSV 找不到 detection_id：{missing_targets:?}"));
    }

    let excluded_set: HashSet<&str> = excluded.iter().map(String::as_str).collect();
    let mut by_class: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        if !excluded_set.contains(row.feature_class.as_str()) {
            by_class
                .entry(row.feature_class.clone())
                .or_default()
                .push(index);
        }
    }

    let mut class_cells: BTreeMap<String, BTreeMap<(usize, usize), usize>> = BTreeMap::new();
    let mut class_high_cells: BTreeMap<String, BTreeMap<(usize, usize), usize>> = BTreeMap::new();
    let mut summaries = Vec::new();
    let mut hotspots = Vec::new();
    let mut top_rows: BTreeMap<i64, (Option<usize>, String)> = BTreeMap::new();

    for (feature_class, class_indices) in &by_class {
        let class_rows: Vec<&CatalogRow> =
            class_indices.iter().map(|index| &rows[*index]).collect();
        let mut cells = BTreeMap::<(usize, usize), usize>::new();
        for row in &class_rows {
            *cells.entry((row.grid_row, row.grid_column)).or_default() += 1;
        }
        let high_rows: Vec<&CatalogRow> = class_rows
            .iter()
            .copied()
            .filter(|row| {
                !row.quality_passed
                    && row
                        .flux_snr
                        .is_some_and(|value| value >= high_flux_snr_threshold)
            })
            .collect();
        let mut high_cells = BTreeMap::<(usize, usize), usize>::new();
        let mut high_cell_order = Vec::new();
        for row in &high_rows {
            let cell = (row.grid_row, row.grid_column);
            if !high_cells.contains_key(&cell) {
                high_cell_order.push(cell);
            }
            *high_cells.entry(cell).or_default() += 1;
        }
        let mut max_high_cell = None;
        let mut max_high_count = 0;
        for cell in high_cell_order {
            let count = high_cells[&cell];
            if count > max_high_count {
                max_high_count = count;
                max_high_cell = Some(cell);
            }
        }
        let rejected = class_rows.iter().filter(|row| !row.quality_passed).count();
        let edge_count = class_rows
            .iter()
            .filter(|row| edge_distance(row, image_width, image_height) <= edge_margin_px)
            .count();
        let total_high = high_rows.len();
        let max_cell_count = cells.values().copied().max().unwrap_or(0);
        let high_share = if total_high == 0 {
            0.0
        } else {
            max_high_count as f64 / total_high as f64
        };
        let high_enrichment = max_high_cell
            .filter(|cell| cells.get(cell).copied().unwrap_or(0) > 0 && !class_rows.is_empty())
            .map(|cell| high_share / (cells[&cell] as f64 / class_rows.len() as f64))
            .unwrap_or(0.0);
        let label = class_rows[0].feature_class_label.clone();
        summaries.push(FeatureSpatialClassSummary {
            feature_class: feature_class.clone(),
            feature_class_label: label.clone(),
            class_count: class_rows.len(),
            class_quality_count: class_rows.len() - rejected,
            class_rejected_count: rejected,
            grid_size,
            nonempty_grid_cells: cells.len(),
            max_cell_count,
            max_cell_share: max_cell_count as f64 / class_rows.len() as f64,
            normalized_grid_entropy: normalized_entropy(&cells, grid_size),
            edge_margin_px,
            edge_count,
            edge_fraction: edge_count as f64 / class_rows.len() as f64,
            high_flux_snr_threshold,
            high_snr_rejected_count: total_high,
            high_snr_max_cell_count: max_high_count,
            high_snr_max_cell_share: high_share,
            high_snr_hot_grid_row: max_high_cell.map(|cell| cell.0),
            high_snr_hot_grid_column: max_high_cell.map(|cell| cell.1),
            high_snr_hotspot_enrichment: high_enrichment,
        });

        for (cell, candidate_count) in &cells {
            let cell_quality_count = class_rows
                .iter()
                .filter(|row| row.quality_passed && (row.grid_row, row.grid_column) == *cell)
                .count();
            let high_count = high_cells.get(cell).copied().unwrap_or(0);
            let candidate_share = *candidate_count as f64 / class_rows.len() as f64;
            let cell_high_share = if total_high == 0 {
                0.0
            } else {
                high_count as f64 / total_high as f64
            };
            hotspots.push(FeatureSpatialHotspot {
                feature_class: feature_class.clone(),
                feature_class_label: label.clone(),
                grid_row: cell.0,
                grid_column: cell.1,
                candidate_count: *candidate_count,
                quality_count: cell_quality_count,
                rejected_count: candidate_count - cell_quality_count,
                high_snr_rejected_count: high_count,
                candidate_share,
                high_snr_share: cell_high_share,
                high_snr_enrichment: if candidate_share == 0.0 {
                    0.0
                } else {
                    cell_high_share / candidate_share
                },
            });
        }

        let mut ranked: Vec<(f64, i64)> = class_rows
            .iter()
            .filter(|row| !row.quality_passed)
            .filter_map(|row| row.metric(metric).map(|value| (value, row.detection_id)))
            .collect();
        ranked.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });
        for (rank, (_, detection_id)) in ranked.into_iter().take(top_n).enumerate() {
            top_rows.insert(detection_id, (Some(rank + 1), "top_metric".into()));
        }
        class_cells.insert(feature_class.clone(), cells);
        class_high_cells.insert(feature_class.clone(), high_cells);
    }

    for detection_id in &requested_targets {
        if let Some((_, reason)) = top_rows.get_mut(detection_id) {
            *reason = "top_metric|explicit".into();
        } else {
            top_rows.insert(*detection_id, (None, "explicit".into()));
        }
    }

    let targets = top_rows
        .iter()
        .map(|(detection_id, (rank, reason))| {
            let row = &rows[by_id[detection_id]];
            let feature_class = &row.feature_class;
            let excluded_target = excluded_set.contains(feature_class.as_str());
            let (class_count, class_cell_count, class_high_count, total_high) = if excluded_target {
                (1, 1, 0, 0)
            } else {
                let class_indices = &by_class[feature_class];
                let cells = &class_cells[feature_class];
                let high_cells = &class_high_cells[feature_class];
                let cell = (row.grid_row, row.grid_column);
                (
                    class_indices.len(),
                    cells.get(&cell).copied().unwrap_or(0),
                    high_cells.get(&cell).copied().unwrap_or(0),
                    high_cells.values().sum(),
                )
            };
            let nearest = rows
                .iter()
                .filter(|candidate| candidate.detection_id != *detection_id)
                .map(|candidate| (candidate.x - row.x).hypot(candidate.y - row.y))
                .min_by(f64::total_cmp);
            let neighbors: Vec<&CatalogRow> = rows
                .iter()
                .filter(|candidate| candidate.detection_id != *detection_id)
                .filter(|candidate| (candidate.x - row.x).hypot(candidate.y - row.y) <= 10.0)
                .collect();
            let neighbor_counts = |radius: f64| {
                rows.iter()
                    .filter(|candidate| candidate.detection_id != *detection_id)
                    .filter(|candidate| (candidate.x - row.x).hypot(candidate.y - row.y) <= radius)
                    .count()
            };
            let mut neighbor_classes = BTreeMap::<String, usize>::new();
            for neighbor in &neighbors {
                *neighbor_classes
                    .entry(neighbor.feature_class.clone())
                    .or_default() += 1;
            }
            let edge_distance = edge_distance(row, image_width, image_height);
            FeatureSpatialTarget {
                detection_id: *detection_id,
                selection_reason: reason.clone(),
                rank_within_class: *rank,
                feature_class: row.feature_class.clone(),
                feature_class_label: row.feature_class_label.clone(),
                quality_passed: row.quality_passed,
                flags: row.flags.clone(),
                x: row.x,
                y: row.y,
                grid_row: row.grid_row,
                grid_column: row.grid_column,
                edge_distance_px: edge_distance,
                is_edge: edge_distance <= edge_margin_px,
                flux_snr: row.flux_snr,
                filter_snr: row.filter_snr,
                class_count,
                class_cell_count,
                class_cell_share: class_cell_count as f64 / class_count as f64,
                cell_high_snr_rejected_count: class_high_count,
                cell_high_snr_share: if total_high == 0 {
                    0.0
                } else {
                    class_high_count as f64 / total_high as f64
                },
                nearest_other_candidate_px: nearest,
                neighbor_count_radius_5px: neighbor_counts(5.0),
                neighbor_count_radius_10px: neighbors.len(),
                neighbor_count_radius_20px: neighbor_counts(20.0),
                neighbor_quality_count_radius_10px: neighbors
                    .iter()
                    .filter(|neighbor| neighbor.quality_passed)
                    .count(),
                neighbor_classes_radius_10px: neighbor_classes
                    .iter()
                    .map(|(name, count)| format!("{name}:{count}"))
                    .collect::<Vec<_>>()
                    .join("|"),
            }
        })
        .collect();

    Ok(FeatureSpatialContextResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        image_width,
        image_height,
        grid_size,
        edge_margin_px,
        high_flux_snr_threshold,
        metric: metric.into(),
        top_n,
        excluded_classes: excluded,
        summaries,
        hotspots,
        targets,
    })
}

fn edge_distance(row: &CatalogRow, image_width: usize, image_height: usize) -> f64 {
    row.x
        .min(row.y)
        .min(image_width as f64 - 1.0 - row.x)
        .min(image_height as f64 - 1.0 - row.y)
}

fn python_float_string(value: f64) -> String {
    let mut raw = format!("{value:?}");
    let negative = raw.starts_with('-');
    if negative {
        raw.remove(0);
    }
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map(|(mantissa, exponent)| (mantissa, exponent.parse::<i32>().unwrap_or(0)))
        .unwrap_or((raw.as_str(), 0));
    let decimal_position = mantissa.find('.').unwrap_or(mantissa.len()) as i32 + exponent;
    let mut digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let leading_zeroes = digits.chars().take_while(|digit| *digit == '0').count();
    digits.drain(..leading_zeroes);
    let decimal_position = decimal_position - leading_zeroes as i32;
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    if digits.is_empty() {
        return if negative { "-0.0" } else { "0.0" }.into();
    }
    let sign = if negative { "-" } else { "" };
    if !(-3..=15).contains(&decimal_position) {
        let scientific_exponent = decimal_position - 1;
        let exponent_sign = if scientific_exponent >= 0 { "+" } else { "-" };
        let exponent_digits = format!("{:02}", scientific_exponent.abs());
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        return format!(
            "{sign}{}{fraction}e{exponent_sign}{exponent_digits}",
            &digits[..1]
        );
    }
    if decimal_position <= 0 {
        return format!(
            "{sign}0.{}{}",
            "0".repeat((-decimal_position) as usize),
            digits
        );
    }
    if decimal_position as usize >= digits.len() {
        return format!(
            "{sign}{}{}.0",
            digits,
            "0".repeat(decimal_position as usize - digits.len())
        );
    }
    format!(
        "{sign}{}.{}",
        &digits[..decimal_position as usize],
        &digits[decimal_position as usize..]
    )
}

fn csv_value(value: &Value, is_float: bool) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::Number(value) if is_float => python_float_string(value.as_f64().unwrap_or_default()),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                value.to_string()
            } else if let Some(value) = value.as_u64() {
                value.to_string()
            } else {
                python_float_string(value.as_f64().unwrap_or_default())
            }
        }
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn write_csv<T: Serialize>(
    path: &Path,
    columns: &[&str],
    float_columns: &[&str],
    rows: &[T],
) -> Result<(), String> {
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    if rows.is_empty() {
        file.write_all(b"\n").map_err(|error| error.to_string())?;
        return Ok(());
    }
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(columns)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let value = serde_json::to_value(row).map_err(|error| error.to_string())?;
        let object = value
            .as_object()
            .ok_or("cannot write a non-object CSV row")?;
        let record: Vec<String> = columns
            .iter()
            .map(|column| {
                object
                    .get(*column)
                    .map(|value| csv_value(value, float_columns.contains(column)))
                    .unwrap_or_default()
            })
            .collect();
        writer
            .write_record(record)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_artifacts(
    result: &FeatureSpatialContextResult,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    fs::write(
        output_dir.join("feature_spatial_context.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    write_csv(
        &output_dir.join("feature_spatial_context_summary.csv"),
        &[
            "feature_class",
            "feature_class_label",
            "class_count",
            "class_quality_count",
            "class_rejected_count",
            "grid_size",
            "nonempty_grid_cells",
            "max_cell_count",
            "max_cell_share",
            "normalized_grid_entropy",
            "edge_margin_px",
            "edge_count",
            "edge_fraction",
            "high_flux_snr_threshold",
            "high_snr_rejected_count",
            "high_snr_max_cell_count",
            "high_snr_max_cell_share",
            "high_snr_hot_grid_row",
            "high_snr_hot_grid_column",
            "high_snr_hotspot_enrichment",
        ],
        &[
            "max_cell_share",
            "normalized_grid_entropy",
            "edge_margin_px",
            "edge_fraction",
            "high_flux_snr_threshold",
            "high_snr_max_cell_share",
            "high_snr_hotspot_enrichment",
        ],
        &result.summaries,
    )?;
    write_csv(
        &output_dir.join("feature_spatial_context_hotspots.csv"),
        &[
            "feature_class",
            "feature_class_label",
            "grid_row",
            "grid_column",
            "candidate_count",
            "quality_count",
            "rejected_count",
            "high_snr_rejected_count",
            "candidate_share",
            "high_snr_share",
            "high_snr_enrichment",
        ],
        &["candidate_share", "high_snr_share", "high_snr_enrichment"],
        &result.hotspots,
    )?;
    write_csv(
        &output_dir.join("feature_spatial_context_targets.csv"),
        &[
            "detection_id",
            "selection_reason",
            "rank_within_class",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "flags",
            "x",
            "y",
            "grid_row",
            "grid_column",
            "edge_distance_px",
            "is_edge",
            "flux_snr",
            "filter_snr",
            "class_count",
            "class_cell_count",
            "class_cell_share",
            "cell_high_snr_rejected_count",
            "cell_high_snr_share",
            "nearest_other_candidate_px",
            "neighbor_count_radius_5px",
            "neighbor_count_radius_10px",
            "neighbor_count_radius_20px",
            "neighbor_quality_count_radius_10px",
            "neighbor_classes_radius_10px",
        ],
        &[
            "x",
            "y",
            "edge_distance_px",
            "flux_snr",
            "filter_snr",
            "class_cell_share",
            "cell_high_snr_share",
            "nearest_other_candidate_px",
        ],
        &result.targets,
    )?;
    Ok(output_dir.to_path_buf())
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    name: &str,
    inline: Option<&'a str>,
) -> Result<&'a str, String> {
    if let Some(value) = inline {
        Ok(value)
    } else {
        *index += 1;
        args.get(*index)
            .map(String::as_str)
            .ok_or_else(|| format!("{name} requires a value"))
    }
}

pub fn run_desktop(args: &[String]) -> Result<Value, String> {
    let mut catalog = None;
    let mut targets = Vec::new();
    let mut top_n = 5usize;
    let mut metric = "flux_snr".to_owned();
    let mut grid_size = 4usize;
    let mut image_width = 4096usize;
    let mut image_height = 4096usize;
    let mut edge_margin = 16.0f64;
    let mut high_flux_snr = 10.0f64;
    let mut output_dir = PathBuf::from("tmp/feature-spatial-context");
    let mut index = 0;
    while index < args.len() {
        let (name, inline) = args[index]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((args[index].as_str(), None));
        match name {
            "--target-id" | "--top-n" | "--metric" | "--grid-size" | "--image-width"
            | "--image-height" | "--edge-margin-px" | "--high-flux-snr" | "--out-dir" => {
                let raw = option_value(args, &mut index, name, inline)?;
                match name {
                    "--target-id" => targets.push(
                        raw.parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?,
                    ),
                    "--top-n" => {
                        top_n = raw
                            .parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?
                    }
                    "--metric" => metric = raw.into(),
                    "--grid-size" => {
                        grid_size = raw
                            .parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?
                    }
                    "--image-width" => {
                        image_width = raw
                            .parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?
                    }
                    "--image-height" => {
                        image_height = raw
                            .parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?
                    }
                    "--edge-margin-px" => {
                        edge_margin = raw
                            .parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?
                    }
                    "--high-flux-snr" => {
                        high_flux_snr = raw
                            .parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?
                    }
                    _ => output_dir = PathBuf::from(raw),
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ if catalog.is_none() => catalog = Some(PathBuf::from(name)),
            _ => return Err("only one source_catalog CSV input is accepted".into()),
        }
        index += 1;
    }
    let catalog = catalog.ok_or("a source catalog CSV is required")?;
    let excluded = vec!["other_rejected".to_owned()];
    let options = FeatureSpatialContextOptions {
        image_width,
        image_height,
        grid_size,
        edge_margin_px: edge_margin,
        high_flux_snr_threshold: high_flux_snr,
        top_n,
        metric,
        target_ids: targets,
        exclude_classes: excluded,
    };
    let result = run(&catalog, &options)?;
    let output_dir = write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool":"rst19-feature-spatial-context",
        "out_dir":output_dir,
        "result_path":output_dir.join("feature_spatial_context.json"),
        "result":result.as_dict()
    }))
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("rst19-feature-spatial-context CATALOG.csv [options]");
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "rst19-feature-spatial-context: 空间条件化审计已写入 {}",
        result["out_dir"].as_str().unwrap_or_default()
    );
    for summary in result["result"]["summaries"].as_array().unwrap() {
        let hot = if summary["high_snr_hot_grid_row"].is_null() {
            "-".into()
        } else {
            format!(
                "r{}c{}",
                summary["high_snr_hot_grid_row"], summary["high_snr_hot_grid_column"]
            )
        };
        println!(
            "{}: n={} edge={:.3} high_snr={} hot={}",
            summary["feature_class"].as_str().unwrap_or_default(),
            summary["class_count"],
            summary["edge_fraction"].as_f64().unwrap_or_default(),
            summary["high_snr_rejected_count"],
            hot
        );
    }
    Ok(())
}

pub fn handle_request(request: &Value) -> Result<Value, String> {
    let catalog = request["catalog_path"]
        .as_str()
        .ok_or("missing catalog_path")?;
    let image_width = request["image_width"].as_u64().unwrap_or(4096) as usize;
    let image_height = request["image_height"].as_u64().unwrap_or(4096) as usize;
    let grid_size = request["grid_size"].as_u64().unwrap_or(4) as usize;
    let edge_margin_px = request["edge_margin_px"].as_f64().unwrap_or(16.0);
    let high_flux_snr_threshold = request["high_flux_snr_threshold"].as_f64().unwrap_or(10.0);
    let top_n = request["top_n"].as_u64().unwrap_or(5) as usize;
    let metric = request["metric"].as_str().unwrap_or("flux_snr");
    let targets: Option<Vec<i64>> = request["target_ids"]
        .as_array()
        .map(|values| {
            values
                .iter()
                .map(|value| value.as_i64().ok_or("target_ids must contain integers"))
                .collect::<Result<_, _>>()
        })
        .transpose()?;
    let excluded: Vec<String> = request["exclude_classes"]
        .as_array()
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or("exclude_classes must contain strings")
                })
                .collect::<Result<_, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec!["other_rejected".into()]);
    let options = FeatureSpatialContextOptions {
        image_width,
        image_height,
        grid_size,
        edge_margin_px,
        high_flux_snr_threshold,
        top_n,
        metric: metric.into(),
        target_ids: targets.unwrap_or_default(),
        exclude_classes: excluded,
    };
    let result = run(Path::new(catalog), &options)?;
    let output = if let Some(path) = request.get("output_dir").and_then(Value::as_str) {
        let output_dir = write_artifacts(&result, Path::new(path))?;
        json!({"out_dir":output_dir, "result_path":output_dir.join("feature_spatial_context.json")})
    } else {
        Value::Null
    };
    Ok(json!({"result":result.as_dict(), "output":output}))
}
