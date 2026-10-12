use crate::{fits, measurement::Detection, models::FitsFrame, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Options {
    pub patch_padding_px: i64,
    pub sigma_levels: Vec<f64>,
    pub raw_maximum_neighborhoods_px: Vec<usize>,
    pub raw_maximum_sigma_level: f64,
    pub special_negative_threshold_adu: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            patch_padding_px: 10,
            sigma_levels: vec![3.0, 5.0, 8.0, 10.0],
            raw_maximum_neighborhoods_px: vec![3, 5, 7],
            raw_maximum_sigma_level: 3.0,
            special_negative_threshold_adu: -1000.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComponentRow {
    pub mode: String,
    pub sigma_level: f64,
    pub threshold_adu: f64,
    pub positive_pixel_count: usize,
    pub component_count: usize,
    pub primary_component_id: usize,
    pub secondary_component_id: usize,
    pub primary_component_size: usize,
    pub secondary_component_size: usize,
    pub same_nonzero_component: bool,
    pub primary_peak_value_adu: f64,
    pub secondary_peak_value_adu: f64,
    pub primary_peak_is_positive: bool,
    pub secondary_peak_is_positive: bool,
    pub special_pixel_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MaximumRow {
    pub neighborhood_px: usize,
    pub threshold_adu: f64,
    pub local_maximum_count: usize,
    pub primary_peak_is_raw_local_maximum: bool,
    pub secondary_peak_is_raw_local_maximum: bool,
    pub primary_nearest_raw_maximum_px: Option<f64>,
    pub secondary_nearest_raw_maximum_px: Option<f64>,
    pub top_raw_maxima_json: String,
}

#[derive(Clone, Debug, Serialize)]
struct RawMaximum {
    x: i64,
    y: i64,
    value_adu: f64,
}

#[derive(Clone, Debug)]
pub struct AuditResult {
    pub source_path: String,
    pub primary_detection_id: i64,
    pub secondary_detection_id: i64,
    pub primary_peak_x: i64,
    pub primary_peak_y: i64,
    pub secondary_peak_x: i64,
    pub secondary_peak_y: i64,
    pub patch_x0: i64,
    pub patch_y0: i64,
    pub patch_x1: i64,
    pub patch_y1: i64,
    pub common_background_adu: f64,
    pub common_noise_adu: f64,
    pub repeated_code_values: Vec<i64>,
    pub special_negative_threshold_adu: f64,
    pub brightest_raw_x: i64,
    pub brightest_raw_y: i64,
    pub brightest_raw_value_adu: f64,
    pub component_rows: Vec<ComponentRow>,
    pub maximum_rows: Vec<MaximumRow>,
    pub parameters: Value,
    pub conclusion: String,
    patch_width: usize,
    patch_height: usize,
    patch: Vec<f64>,
    special_excluded_3sigma: Vec<bool>,
}

impl AuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "source_path": self.source_path,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_peak": [self.primary_peak_x, self.primary_peak_y],
            "secondary_peak": [self.secondary_peak_x, self.secondary_peak_y],
            "patch_bounds": [self.patch_x0, self.patch_y0, self.patch_x1, self.patch_y1],
            "common_background_adu": self.common_background_adu,
            "common_noise_adu": self.common_noise_adu,
            "repeated_code_values": self.repeated_code_values,
            "special_negative_threshold_adu": self.special_negative_threshold_adu,
            "brightest_raw": {
                "x": self.brightest_raw_x,
                "y": self.brightest_raw_y,
                "value_adu": self.brightest_raw_value_adu,
            },
            "component_rows": self.component_rows,
            "maximum_rows": self.maximum_rows,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        })
    }
}

fn python_round_to_i64(value: f64) -> i64 {
    value.round_ties_even() as i64
}

fn peak_coordinate(detection: &Detection, x_axis: bool) -> i64 {
    let fallback = if x_axis { detection.x } else { detection.y };
    let coordinate = if x_axis {
        detection
            .peak_x
            .filter(|value| value.is_finite())
            .unwrap_or(fallback)
    } else {
        detection
            .peak_y
            .filter(|value| value.is_finite())
            .unwrap_or(fallback)
    };
    python_round_to_i64(coordinate)
}

fn normalize_options(options: &Options) -> Result<(Vec<f64>, Vec<usize>, f64, f64)> {
    if options.patch_padding_px < 1 {
        return Err("patch_padding_px must be positive".into());
    }
    let mut levels = options.sigma_levels.clone();
    levels.sort_by(f64::total_cmp);
    levels.dedup_by(|left, right| *left == *right);
    if levels.is_empty()
        || levels
            .iter()
            .any(|level| !level.is_finite() || *level <= 0.0)
    {
        return Err("sigma_levels must contain finite positive values".into());
    }
    let mut neighborhoods = options.raw_maximum_neighborhoods_px.clone();
    neighborhoods.sort_unstable();
    neighborhoods.dedup();
    if neighborhoods.is_empty()
        || neighborhoods
            .iter()
            .any(|size| *size < 3 || size.is_multiple_of(2))
    {
        return Err("raw_maximum_neighborhoods_px must contain odd integers >= 3".into());
    }
    let maximum_sigma = options.raw_maximum_sigma_level;
    if !maximum_sigma.is_finite() || maximum_sigma <= 0.0 {
        return Err("raw_maximum_sigma_level must be finite and positive".into());
    }
    let negative_threshold = options.special_negative_threshold_adu;
    if !negative_threshold.is_finite() {
        return Err("special_negative_threshold_adu must be finite".into());
    }
    Ok((levels, neighborhoods, maximum_sigma, negative_threshold))
}

fn component_labels(mask: &[bool], width: usize, height: usize) -> (Vec<usize>, usize, Vec<usize>) {
    let mut labels = vec![0; mask.len()];
    let mut sizes = vec![0];
    let mut count = 0;
    let mut pending = VecDeque::new();
    for start in 0..mask.len() {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        count += 1;
        labels[start] = count;
        pending.push_back(start);
        let mut size = 0;
        while let Some(index) = pending.pop_front() {
            size += 1;
            let x = index % width;
            let y = index / width;
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let neighbor = ny * width + nx;
                    if mask[neighbor] && labels[neighbor] == 0 {
                        labels[neighbor] = count;
                        pending.push_back(neighbor);
                    }
                }
            }
        }
        sizes.push(size);
    }
    (labels, count, sizes)
}

fn component_at(labels: &[usize], sizes: &[usize], index: usize) -> (usize, usize) {
    let id = labels[index];
    (id, sizes[id])
}

fn median_two(left: f64, right: f64) -> f64 {
    (left + right) / 2.0
}

fn nearest_distance(point_x: usize, point_y: usize, candidates: &[(usize, usize)]) -> Option<f64> {
    candidates
        .iter()
        .map(|(x, y)| (*x as f64 - point_x as f64).hypot(*y as f64 - point_y as f64))
        .min_by(f64::total_cmp)
}

pub fn run(
    frame: &FitsFrame,
    primary: &Detection,
    secondary: &Detection,
    options: &Options,
    repeated_code_values: Option<&[i64]>,
) -> Result<AuditResult> {
    if primary.detection_id == secondary.detection_id {
        return Err("primary and secondary detections must differ".into());
    }
    let (levels, neighborhoods, maximum_sigma, negative_threshold) = normalize_options(options)?;
    let primary_x = peak_coordinate(primary, true);
    let primary_y = peak_coordinate(primary, false);
    let secondary_x = peak_coordinate(secondary, true);
    let secondary_y = peak_coordinate(secondary, false);
    if (primary_x, primary_y) == (secondary_x, secondary_y) {
        return Err("primary and secondary peak coordinates must differ".into());
    }
    if frame.width == 0 || frame.height == 0 || frame.data.len() != frame.width * frame.height {
        return Err("expected a non-empty 2-D FITS image".into());
    }
    let width = frame.width as i64;
    let height = frame.height as i64;
    let padding = options.patch_padding_px;
    let x0 = primary_x
        .min(secondary_x)
        .saturating_sub(padding)
        .clamp(0, width);
    let y0 = primary_y
        .min(secondary_y)
        .saturating_sub(padding)
        .clamp(0, height);
    let x1 = primary_x
        .max(secondary_x)
        .saturating_add(padding)
        .saturating_add(1)
        .clamp(0, width);
    let y1 = primary_y
        .max(secondary_y)
        .saturating_add(padding)
        .saturating_add(1)
        .clamp(0, height);
    let patch_width = usize::try_from(x1 - x0).map_err(|_| "invalid pair patch width")?;
    let patch_height = usize::try_from(y1 - y0).map_err(|_| "invalid pair patch height")?;
    let mut patch = Vec::with_capacity(patch_width.saturating_mul(patch_height));
    for y in y0..y1 {
        for x in x0..x1 {
            patch.push(frame.data.value(y as usize * frame.width + x as usize));
        }
    }
    if patch.is_empty() || patch_width == 0 || patch_height == 0 {
        return Err("pair patch contains no finite pixels".into());
    }
    let finite: Vec<bool> = patch.iter().map(|value| value.is_finite()).collect();
    if !finite.iter().any(|value| *value) {
        return Err("pair patch contains no finite pixels".into());
    }
    let backgrounds = [primary.background, secondary.background];
    let noises = [primary.noise, secondary.noise];
    if backgrounds.iter().any(|value| !value.is_finite())
        || noises
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(
            "pair detections must contain finite positive noise and finite background".into(),
        );
    }
    let common_background = median_two(backgrounds[0], backgrounds[1]);
    let common_noise = median_two(noises[0], noises[1]);
    let mut repeated = BTreeSet::new();
    if let Some(values) = repeated_code_values {
        repeated.extend(values.iter().copied());
    }
    repeated.extend(primary.repeated_code_values.iter().copied());
    repeated.extend(secondary.repeated_code_values.iter().copied());
    let repeated: Vec<i64> = repeated.into_iter().collect();
    let special: Vec<bool> = patch
        .iter()
        .zip(&finite)
        .map(|(value, is_finite)| {
            !is_finite
                || repeated.iter().any(|code| *value == *code as f64)
                || (*is_finite && *value <= negative_threshold)
        })
        .collect();
    let point = |x: i64, y: i64| -> Result<(usize, usize, usize)> {
        let px = x - x0;
        let py = y - y0;
        if px < 0 || py < 0 || px >= patch_width as i64 || py >= patch_height as i64 {
            return Err("peak coordinate is outside the pair patch".into());
        }
        Ok((
            px as usize,
            py as usize,
            py as usize * patch_width + px as usize,
        ))
    };
    let (primary_px, primary_py, primary_index) = point(primary_x, primary_y)?;
    let (secondary_px, secondary_py, secondary_index) = point(secondary_x, secondary_y)?;
    let primary_peak_value = patch[primary_index];
    let secondary_peak_value = patch[secondary_index];

    let mut component_rows = Vec::with_capacity(2 * levels.len());
    for (mode, excluded) in [("raw", None), ("special_excluded", Some(&special))] {
        for &sigma_level in &levels {
            let threshold = common_background + sigma_level * common_noise;
            let positive: Vec<bool> = patch
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    finite[index]
                        && !excluded.is_some_and(|mask| mask[index])
                        && (*value - common_background) >= threshold - common_background
                })
                .collect();
            let (labels, count, sizes) = component_labels(&positive, patch_width, patch_height);
            let (primary_id, primary_size) = component_at(&labels, &sizes, primary_index);
            let (secondary_id, secondary_size) = component_at(&labels, &sizes, secondary_index);
            component_rows.push(ComponentRow {
                mode: mode.to_owned(),
                sigma_level,
                threshold_adu: threshold,
                positive_pixel_count: positive.iter().filter(|value| **value).count(),
                component_count: count,
                primary_component_id: primary_id,
                secondary_component_id: secondary_id,
                primary_component_size: primary_size,
                secondary_component_size: secondary_size,
                same_nonzero_component: primary_id > 0 && primary_id == secondary_id,
                primary_peak_value_adu: primary_peak_value,
                secondary_peak_value_adu: secondary_peak_value,
                primary_peak_is_positive: positive[primary_index],
                secondary_peak_is_positive: positive[secondary_index],
                special_pixel_count: excluded
                    .map(|mask| mask.iter().filter(|value| **value).count())
                    .unwrap_or(0),
            });
        }
    }

    let maximum_threshold = common_background + maximum_sigma * common_noise;
    let mut maximum_rows = Vec::with_capacity(neighborhoods.len());
    for &neighborhood in &neighborhoods {
        let half = neighborhood / 2;
        let mut local_maxima = vec![false; patch.len()];
        let mut points = Vec::new();
        for y in 0..patch_height {
            for x in 0..patch_width {
                let index = y * patch_width + x;
                if !finite[index] || patch[index] < maximum_threshold {
                    continue;
                }
                let x0 = x.saturating_sub(half);
                let y0 = y.saturating_sub(half);
                let x1 = x.saturating_add(half).min(patch_width - 1);
                let y1 = y.saturating_add(half).min(patch_height - 1);
                let mut maximum = f64::NEG_INFINITY;
                for neighbor_y in y0..=y1 {
                    for neighbor_x in x0..=x1 {
                        let neighbor = neighbor_y * patch_width + neighbor_x;
                        if finite[neighbor] {
                            maximum = maximum.max(patch[neighbor]);
                        }
                    }
                }
                if patch[index] == maximum {
                    local_maxima[index] = true;
                    points.push((x, y));
                }
            }
        }
        let mut ranked: Vec<RawMaximum> = points
            .iter()
            .map(|(x, y)| RawMaximum {
                x: *x as i64 + x0,
                y: *y as i64 + y0,
                value_adu: patch[y * patch_width + x],
            })
            .collect();
        ranked.sort_by(|left, right| right.value_adu.total_cmp(&left.value_adu));
        ranked.truncate(12);
        let top_raw_maxima_json =
            serde_json::to_string(&ranked).map_err(|error| error.to_string())?;
        maximum_rows.push(MaximumRow {
            neighborhood_px: neighborhood,
            threshold_adu: maximum_threshold,
            local_maximum_count: points.len(),
            primary_peak_is_raw_local_maximum: local_maxima[primary_index],
            secondary_peak_is_raw_local_maximum: local_maxima[secondary_index],
            primary_nearest_raw_maximum_px: nearest_distance(primary_px, primary_py, &points),
            secondary_nearest_raw_maximum_px: nearest_distance(secondary_px, secondary_py, &points),
            top_raw_maxima_json,
        });
    }

    let raw_rows: Vec<&ComponentRow> = component_rows
        .iter()
        .filter(|row| row.mode == "raw")
        .collect();
    let same_raw_count = raw_rows
        .iter()
        .filter(|row| row.same_nonzero_component)
        .count();
    let primary_local_max_count = maximum_rows
        .iter()
        .filter(|row| row.primary_peak_is_raw_local_maximum)
        .count();
    let secondary_local_max_count = maximum_rows
        .iter()
        .filter(|row| row.secondary_peak_is_raw_local_maximum)
        .count();
    let mut brightest_index = None;
    for (index, value) in patch.iter().enumerate() {
        if finite[index] && brightest_index.is_none_or(|current| *value > patch[current]) {
            brightest_index = Some(index);
        }
    }
    let brightest_index = brightest_index.ok_or("pair patch contains no finite pixels")?;
    let brightest_x = brightest_index % patch_width;
    let brightest_y = brightest_index / patch_width;
    let brightest_raw_x = brightest_x as i64 + x0;
    let brightest_raw_y = brightest_y as i64 + y0;
    let brightest_raw_value = patch[brightest_index];
    let conclusion = format!(
        "局部像素拓扑审计完成：raw 在 {}/{} 个阈值下将两个候选放在同一个非零正响应连通块；原始局部极大值口径下，主/副候选分别只有 {}/{}、{}/{} 次是局部极大值。该结果支持共享亮斑/结构混合或匹配响应重定位的优先解释，但不单独排除真实近邻 PSF，因为真实近邻也可能在阈值下连成一块。",
        same_raw_count,
        raw_rows.len(),
        primary_local_max_count,
        maximum_rows.len(),
        secondary_local_max_count,
        maximum_rows.len(),
    );
    let common_three_sigma = common_background + 3.0 * common_noise;
    let special_excluded_3sigma = patch
        .iter()
        .enumerate()
        .map(|(index, value)| finite[index] && *value >= common_three_sigma && !special[index])
        .collect();
    Ok(AuditResult {
        source_path: frame.path.to_string_lossy().into_owned(),
        primary_detection_id: primary.detection_id,
        secondary_detection_id: secondary.detection_id,
        primary_peak_x: primary_x,
        primary_peak_y: primary_y,
        secondary_peak_x: secondary_x,
        secondary_peak_y: secondary_y,
        patch_x0: x0,
        patch_y0: y0,
        patch_x1: x1,
        patch_y1: y1,
        common_background_adu: common_background,
        common_noise_adu: common_noise,
        repeated_code_values: repeated,
        special_negative_threshold_adu: negative_threshold,
        brightest_raw_x,
        brightest_raw_y,
        brightest_raw_value_adu: brightest_raw_value,
        component_rows,
        maximum_rows,
        parameters: json!({
            "patch_padding_px": options.patch_padding_px,
            "connectivity": 8,
            "sigma_levels": levels,
            "raw_maximum_neighborhoods_px": neighborhoods,
            "raw_maximum_sigma_level": maximum_sigma,
            "threshold_definition": "median(source backgrounds) + sigma * median(source noises)",
            "special_value_definition": "repeated code values or raw value <= special_negative_threshold_adu",
            "interpretation_boundary": "pixel topology diagnostic, not source truth or star probability",
        }),
        conclusion,
        patch_width,
        patch_height,
        patch,
        special_excluded_3sigma,
    })
}

fn f(value: f64) -> String {
    format!("{value:?}")
}
fn b(value: bool) -> String {
    if value { "True" } else { "False" }.to_owned()
}

fn write_csv(path: &Path, headers: &[&str], rows: Vec<Vec<String>>) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &AuditResult, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = out_dir.canonicalize().map_err(|error| error.to_string())?;
    write_csv(
        &output.join("pair_pixel_topology_components.csv"),
        &[
            "mode",
            "sigma_level",
            "threshold_adu",
            "positive_pixel_count",
            "component_count",
            "primary_component_id",
            "secondary_component_id",
            "primary_component_size",
            "secondary_component_size",
            "same_nonzero_component",
            "primary_peak_value_adu",
            "secondary_peak_value_adu",
            "primary_peak_is_positive",
            "secondary_peak_is_positive",
            "special_pixel_count",
        ],
        result
            .component_rows
            .iter()
            .map(|row| {
                vec![
                    row.mode.clone(),
                    f(row.sigma_level),
                    f(row.threshold_adu),
                    row.positive_pixel_count.to_string(),
                    row.component_count.to_string(),
                    row.primary_component_id.to_string(),
                    row.secondary_component_id.to_string(),
                    row.primary_component_size.to_string(),
                    row.secondary_component_size.to_string(),
                    b(row.same_nonzero_component),
                    f(row.primary_peak_value_adu),
                    f(row.secondary_peak_value_adu),
                    b(row.primary_peak_is_positive),
                    b(row.secondary_peak_is_positive),
                    row.special_pixel_count.to_string(),
                ]
            })
            .collect(),
    )?;
    write_csv(
        &output.join("pair_pixel_topology_maxima.csv"),
        &[
            "neighborhood_px",
            "threshold_adu",
            "local_maximum_count",
            "primary_peak_is_raw_local_maximum",
            "secondary_peak_is_raw_local_maximum",
            "primary_nearest_raw_maximum_px",
            "secondary_nearest_raw_maximum_px",
            "top_raw_maxima_json",
        ],
        result
            .maximum_rows
            .iter()
            .map(|row| {
                vec![
                    row.neighborhood_px.to_string(),
                    f(row.threshold_adu),
                    row.local_maximum_count.to_string(),
                    b(row.primary_peak_is_raw_local_maximum),
                    b(row.secondary_peak_is_raw_local_maximum),
                    row.primary_nearest_raw_maximum_px
                        .map(f)
                        .unwrap_or_default(),
                    row.secondary_nearest_raw_maximum_px
                        .map(f)
                        .unwrap_or_default(),
                    row.top_raw_maxima_json.clone(),
                ]
            })
            .collect(),
    )?;
    let json_bytes =
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    fs::write(output.join("pair_pixel_topology.json"), json_bytes)
        .map_err(|error| error.to_string())?;
    write_svg(&output.join("pair_pixel_topology.svg"), result)?;
    Ok(output)
}

fn percentile(values: &mut [f64], probability: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let position = (values.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let weight = position - lower as f64;
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}

fn write_svg(path: &Path, result: &AuditResult) -> Result<()> {
    let finite: Vec<f64> = result
        .patch
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return Err("cannot render an empty topology patch".into());
    }
    let mut lower_values = finite.clone();
    let mut upper_values = finite.clone();
    let mut lower = percentile(&mut lower_values, 0.01);
    let mut upper = percentile(&mut upper_values, 0.99);
    if !lower.is_finite() || !upper.is_finite() || upper <= lower {
        lower = finite.iter().copied().min_by(f64::total_cmp).unwrap();
        upper = finite.iter().copied().max_by(f64::total_cmp).unwrap();
    }
    upper = upper.max(lower + 1.0);
    let rows = result.patch_height;
    let columns = result.patch_width;
    let cell = 18_i64;
    let gap = 48_i64;
    let panel_width = columns as i64 * cell;
    let panel_height = rows as i64 * cell;
    let total_width = panel_width * 2 + gap * 3;
    let total_height = panel_height + 70;
    let primary = (
        result.primary_peak_x - result.patch_x0,
        result.primary_peak_y - result.patch_y0,
    );
    let secondary = (
        result.secondary_peak_x - result.patch_x0,
        result.secondary_peak_y - result.patch_y0,
    );
    let brightest = (
        result.brightest_raw_x - result.patch_x0,
        result.brightest_raw_y - result.patch_y0,
    );
    let color = |value: f64| -> String {
        let clipped = ((value - lower) / (upper - lower)).clamp(0.0, 1.0);
        let channel = python_round_to_i64(24.0 + 220.0 * clipped);
        format!("rgb({channel},{channel},{channel})")
    };
    let mut lines = vec![
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_width}\" height=\"{total_height}\" viewBox=\"0 0 {total_width} {total_height}\">"),
        "<rect width=\"100%\" height=\"100%\" fill=\"#0d131c\"/>".into(),
        format!("<text x=\"{gap}\" y=\"{}\" fill=\"#9aa7b8\" font-size=\"11\">P=primary peak · S=secondary peak · red cross=brightest raw pixel · patch origin=({},{})</text>", total_height - 18, result.patch_x0, result.patch_y0),
    ];
    let panel = |origin_x: i64, title: &str, positive_only: bool| {
        let mut panel_lines = vec![format!(
            "<text x=\"{origin_x}\" y=\"18\" fill=\"#d8dee9\" font-size=\"13\">{title}</text>"
        )];
        for row in 0..rows {
            for column in 0..columns {
                let index = row * columns + column;
                let value = result.patch[index];
                let visible =
                    value.is_finite() && (!positive_only || result.special_excluded_3sigma[index]);
                let fill = if visible {
                    color(value)
                } else {
                    "#121923".into()
                };
                panel_lines.push(format!(
                    "<rect x=\"{}\" y=\"{}\" width=\"18\" height=\"18\" fill=\"{fill}\" stroke=\"#283342\" stroke-width=\"0.35\"/>",
                    origin_x + column as i64 * cell,
                    30 + row as i64 * cell
                ));
            }
        }
        for (x, y, stroke, label) in [
            (primary.0, primary.1, "#f0b54d", "P"),
            (secondary.0, secondary.1, "#61d7d0", "S"),
        ] {
            if x >= 0 && x < columns as i64 && y >= 0 && y < rows as i64 {
                let cx = origin_x as f64 + x as f64 * cell as f64 + cell as f64 / 2.0;
                let cy = 30.0 + y as f64 * cell as f64 + cell as f64 / 2.0;
                panel_lines.push(format!("<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"7\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"2\"/>"));
                panel_lines.push(format!("<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{stroke}\" font-size=\"10\">{label}</text>", cx + 8.0, cy + 4.0));
            }
        }
        if brightest.0 >= 0
            && brightest.0 < columns as i64
            && brightest.1 >= 0
            && brightest.1 < rows as i64
        {
            let cx = origin_x as f64 + brightest.0 as f64 * cell as f64 + cell as f64 / 2.0;
            let cy = 30.0 + brightest.1 as f64 * cell as f64 + cell as f64 / 2.0;
            panel_lines.push(format!("<path d=\"M {:.1} {:.1} L {:.1} {:.1} M {:.1} {:.1} L {:.1} {:.1}\" stroke=\"#ff5b5b\" stroke-width=\"1.5\"/>", cx - 6.0, cy, cx + 6.0, cy, cx, cy - 6.0, cx, cy + 6.0));
        }
        panel_lines
    };
    lines.extend(panel(gap, "raw local patch", false));
    lines.extend(panel(
        gap * 2 + panel_width,
        "3σ after special-value exclusion",
        true,
    ));
    lines.push("</svg>".into());
    let svg = lines.join("\n");
    #[cfg(windows)]
    let svg = svg.replace('\n', "\r\n");
    fs::write(path, svg).map_err(|error| error.to_string())
}

fn parse_source_catalog(path: &Path) -> Result<Vec<Detection>> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read source catalog {}: {error}", path.display()))?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let required = [
        "background",
        "detection_id",
        "flux",
        "noise",
        "peak",
        "snr",
        "x",
        "y",
    ];
    let mut missing: Vec<_> = required
        .iter()
        .copied()
        .filter(|name| !headers.iter().any(|item| item == *name))
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let column = |name: &str| {
        headers
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (value == name).then_some(index))
            .last()
    };
    let index: std::collections::BTreeMap<&str, usize> = required
        .iter()
        .map(|name| (*name, column(name).unwrap()))
        .collect();
    let optional_index = |name: &str| column(name);
    let mut sources = Vec::new();
    for (row_number, record) in reader.records().enumerate() {
        let row = record.map_err(|error| error.to_string())?;
        let raw = |name: &str| row.get(index[name]).unwrap_or("").trim();
        let parse_float = |name: &str| -> Result<f64> {
            raw(name)
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("source catalog field {name:?} is required"))
        };
        let parse_optional_float = |name: &str| -> Result<Option<f64>> {
            let Some(column) = optional_index(name) else {
                return Ok(None);
            };
            let value = row.get(column).unwrap_or("").trim();
            if value.is_empty() {
                return Ok(None);
            }
            let number = value
                .parse::<f64>()
                .map_err(|_| format!("source catalog field {name:?} is not numeric: {value:?}"))?;
            Ok(number.is_finite().then_some(number))
        };
        let detection_id = raw("detection_id").parse::<i64>().map_err(|_| {
            format!(
                "invalid source catalog row {}: detection_id is required",
                row_number + 2
            )
        })?;
        let mut repeated_values = Vec::new();
        if let Some(column) = optional_index("repeated_code_values") {
            let value = row.get(column).unwrap_or("");
            for item in value
                .split('|')
                .map(str::trim)
                .filter(|item| !item.is_empty())
            {
                repeated_values.push(item.parse::<i64>().map_err(|_| format!("invalid source catalog row {}: source catalog list contains a non-integer value: {item:?}", row_number + 2))?);
            }
        }
        let flags: Vec<String> = optional_index("flags")
            .map(|column| {
                row.get(column)
                    .unwrap_or("")
                    .split('|')
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let quality_passed = optional_index("quality_passed")
            .map(|column| row.get(column).unwrap_or("").trim().to_ascii_lowercase())
            .unwrap_or_default();
        let quality_passed = match quality_passed.as_str() {
            "" | "true" | "1" | "yes" | "y" | "是" => true,
            "false" | "0" | "no" | "n" | "否" => false,
            _ => {
                return Err(format!(
                    "invalid source catalog row {}: source catalog boolean value is invalid",
                    row_number + 2
                ))
            }
        };
        let value = json!({
            "detection_id": detection_id,
            "x": parse_float("x")?,
            "y": parse_float("y")?,
            "peak": parse_float("peak")?,
            "flux": parse_float("flux")?,
            "background": parse_float("background")?,
            "noise": parse_float("noise")?,
            "snr": parse_float("snr")?,
            "fwhm": parse_optional_float("fwhm")?,
            "flags": flags,
            "quality_passed": quality_passed,
            "peak_x": parse_optional_float("peak_x")?,
            "peak_y": parse_optional_float("peak_y")?,
            "proposal_methods": [],
            "proposal_scales": [],
            "repeated_code_values": repeated_values,
        });
        sources.push(
            serde_json::from_value(value).map_err(|error| {
                format!("invalid source catalog row {}: {error}", row_number + 2)
            })?,
        );
    }
    Ok(sources)
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut frame_path = None;
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/pair-pixel-topology");
    let mut primary_id = None;
    let mut secondary_id = None;
    let mut options = Options::default();
    let mut index = 0;
    let take = |i: usize, option: &str| -> Result<&str> {
        args.get(i + 1)
            .map(String::as_str)
            .ok_or_else(|| format!("{option} requires a value"))
    };
    while index < args.len() {
        match args[index].as_str() {
            "--source-catalog" => {
                catalog_path = Some(PathBuf::from(take(index, "--source-catalog")?));
                index += 1;
            }
            "--primary-id" => {
                primary_id = Some(
                    take(index, "--primary-id")?
                        .parse::<i64>()
                        .map_err(|_| "--primary-id must be an integer")?,
                );
                index += 1;
            }
            "--secondary-id" => {
                secondary_id = Some(
                    take(index, "--secondary-id")?
                        .parse::<i64>()
                        .map_err(|_| "--secondary-id must be an integer")?,
                );
                index += 1;
            }
            "--patch-padding-px" => {
                options.patch_padding_px = take(index, "--patch-padding-px")?
                    .parse()
                    .map_err(|_| "--patch-padding-px must be an integer")?;
                index += 1;
            }
            "--raw-maximum-sigma-level" => {
                options.raw_maximum_sigma_level = take(index, "--raw-maximum-sigma-level")?
                    .parse()
                    .map_err(|_| "--raw-maximum-sigma-level must be numeric")?;
                index += 1;
            }
            "--special-negative-threshold-adu" => {
                options.special_negative_threshold_adu =
                    take(index, "--special-negative-threshold-adu")?
                        .parse()
                        .map_err(|_| "--special-negative-threshold-adu must be numeric")?;
                index += 1;
            }
            "--out-dir" => {
                output_dir = PathBuf::from(take(index, "--out-dir")?);
                index += 1;
            }
            "--sigma-levels" => {
                options.sigma_levels.clear();
                index += 1;
                while index < args.len() && !args[index].starts_with("--") {
                    options.sigma_levels.push(
                        args[index]
                            .parse()
                            .map_err(|_| "--sigma-levels must be numeric")?,
                    );
                    index += 1;
                }
                continue;
            }
            "--raw-maximum-neighborhoods-px" => {
                options.raw_maximum_neighborhoods_px.clear();
                index += 1;
                while index < args.len() && !args[index].starts_with("--") {
                    options.raw_maximum_neighborhoods_px.push(
                        args[index]
                            .parse()
                            .map_err(|_| "--raw-maximum-neighborhoods-px must be integer")?,
                    );
                    index += 1;
                }
                continue;
            }
            arg if arg.starts_with('-') => return Err(format!("unknown argument {arg}")),
            arg => {
                if frame_path.replace(PathBuf::from(arg)).is_some() {
                    return Err("only one FITS frame is allowed".into());
                }
            }
        }
        index += 1;
    }
    let frame_path = frame_path.ok_or("a FITS frame is required")?;
    let catalog_path = catalog_path.ok_or("--source-catalog is required")?;
    let primary_id = primary_id.ok_or("--primary-id is required")?;
    let secondary_id = secondary_id.ok_or("--secondary-id is required")?;
    let sources = parse_source_catalog(&catalog_path)?;
    let mut by_id = std::collections::BTreeMap::new();
    for source in sources {
        by_id.insert(source.detection_id, source);
    }
    let primary = by_id
        .get(&primary_id)
        .ok_or_else(|| format!("source catalog does not contain detection_id={primary_id}"))?;
    let secondary = by_id
        .get(&secondary_id)
        .ok_or_else(|| format!("source catalog does not contain detection_id={secondary_id}"))?;
    let frame = fits::read_fits(&frame_path, false)?;
    let result = run(&frame, primary, secondary, &options, None)?;
    let output = write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-pair-pixel-topology",
        "out_dir": output,
        "result_path": output.join("pair_pixel_topology.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(id: i64, peak_x: i64, peak_y: i64, peak: f64, codes: Vec<i64>) -> Detection {
        serde_json::from_value(json!({
            "detection_id": id,
            "x": peak_x,
            "y": peak_y,
            "peak": peak,
            "flux": 100.0,
            "background": 10.0,
            "noise": 2.0,
            "snr": 20.0,
            "flags": [],
            "quality_passed": false,
            "proposal_methods": [],
            "proposal_scales": [],
            "repeated_code_values": codes,
        }))
        .unwrap()
    }

    fn frame() -> FitsFrame {
        let mut pixels = vec![10_i16; 40 * 40];
        for (x, y, value) in [
            (20, 20, 130),
            (21, 20, 99),
            (19, 21, 80),
            (18, 21, 70),
            (17, 21, 50),
            (16, 21, 40),
        ] {
            pixels[y * 40 + x] = value;
        }
        FitsFrame {
            path: PathBuf::from("synthetic.fits"),
            header: serde_json::Map::new(),
            data: crate::models::PixelData::I16(pixels),
            auxiliary: None,
            data_offset: 0,
            width: 40,
            height: 40,
        }
    }

    #[test]
    fn raw_and_special_excluded_components_and_maxima_are_distinct() {
        let frame = frame();
        let primary = detection(1, 21, 20, 99.0, vec![99]);
        let secondary = detection(2, 16, 21, 40.0, vec![99]);
        let options = Options {
            patch_padding_px: 4,
            sigma_levels: vec![3.0, 5.0],
            raw_maximum_neighborhoods_px: vec![3],
            ..Options::default()
        };
        let result = run(&frame, &primary, &secondary, &options, None).unwrap();
        assert_eq!(
            [
                result.patch_x0,
                result.patch_y0,
                result.patch_x1,
                result.patch_y1
            ],
            [12, 16, 26, 26]
        );
        assert!(result
            .component_rows
            .iter()
            .filter(|row| row.mode == "raw")
            .all(|row| row.same_nonzero_component));
        assert!(result
            .component_rows
            .iter()
            .filter(|row| row.mode == "special_excluded")
            .all(|row| !row.primary_peak_is_positive));
        assert_eq!(result.maximum_rows[0].local_maximum_count, 1);
        assert_eq!(result.brightest_raw_value_adu, 130.0);
    }

    #[test]
    fn identical_peak_coordinates_are_rejected_and_rounding_ties_to_even() {
        let frame = frame();
        let primary = detection(1, 20, 20, 99.0, vec![]);
        let secondary = detection(2, 20, 20, 99.0, vec![]);
        assert!(run(&frame, &primary, &secondary, &Options::default(), None)
            .unwrap_err()
            .contains("must differ"));

        let primary = detection(1, 20, 20, 99.0, vec![]);
        let mut secondary = detection(2, 16, 21, 40.0, vec![]);
        secondary.peak_x = Some(20.5);
        secondary.peak_y = Some(20.0);
        assert!(run(&frame, &primary, &secondary, &Options::default(), None)
            .unwrap_err()
            .contains("must differ"));
    }
}
