//! Local Gaussian-response attribution for a nearby pair of detections.
//!
//! This audit compares raw, repeated-code-excluded, negative-anomaly-excluded,
//! and combined masked responses. It reports detector-level response evidence,
//! not physical-source counts or artifact probabilities.

use crate::{
    filtering, fits,
    models::FitsFrame,
    pair_repair_counterfactual::{self, TargetSource},
    Result,
};
use csv::{Terminator, WriterBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, HashMap},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const MASK_MODES: [&str; 4] = [
    "raw",
    "repeated_code_masked",
    "negative_anomaly_masked",
    "both_masked",
];
const PIXEL_CLASSES: [&str; 3] = ["ordinary", "repeated_code", "negative_anomaly"];
const FWHM_TO_SIGMA: f64 = 2.35482;
type LocalMaximumRanks = HashMap<(usize, usize), usize>;
const GUARDRAILS: [&str; 4] = [
    "Gaussian 响应是 detector-level 匹配响应，不是物理源数或星等真值。",
    "屏蔽后某个响应消失只说明该响应依赖当前值域/候选有效性，不单独证明噪点。",
    "像素贡献是带符号的局部核归因，不是因果概率或伪影率。",
    "普通像素、重复码和负异常的分类依赖当前阈值与原始存储语义。",
];

#[derive(Debug, Clone, Serialize)]
pub struct TargetRow {
    pub mask_mode: String,
    pub target_label: String,
    pub detection_id: i64,
    pub peak_x: usize,
    pub peak_y: usize,
    pub raw_value_adu: f64,
    pub valid_at_target: bool,
    pub gaussian_response_adu: f64,
    pub valid_weight: f64,
    pub response_local_maximum: bool,
    pub local_maximum_rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaximumRow {
    pub mask_mode: String,
    pub rank: usize,
    pub x: usize,
    pub y: usize,
    pub raw_value_adu: f64,
    pub gaussian_response_adu: f64,
    pub valid_weight: f64,
    pub is_primary_target: bool,
    pub is_secondary_target: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContributionRow {
    pub target_label: String,
    pub detection_id: i64,
    pub pixel_class: String,
    pub pixel_count: usize,
    pub kernel_weight_fraction: f64,
    pub signed_response_contribution_adu: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairResponseResult {
    pub source_path: String,
    pub primary_detection_id: i64,
    pub secondary_detection_id: i64,
    pub primary_peak: [usize; 2],
    pub secondary_peak: [usize; 2],
    pub patch_bounds: [usize; 4],
    pub common_background_adu: f64,
    pub common_noise_adu: f64,
    pub psf_fwhm_px: f64,
    pub repeated_code_values: Vec<i64>,
    pub negative_anomaly_threshold_adu: f64,
    pub target_rows: Vec<TargetRow>,
    pub maximum_rows: Vec<MaximumRow>,
    pub contribution_rows: Vec<ContributionRow>,
    pub parameters: Value,
    pub conclusion: String,
}

impl PairResponseResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "source_path": self.source_path,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_peak": self.primary_peak,
            "secondary_peak": self.secondary_peak,
            "patch_bounds": self.patch_bounds,
            "common_background_adu": self.common_background_adu,
            "common_noise_adu": self.common_noise_adu,
            "psf_fwhm_px": self.psf_fwhm_px,
            "repeated_code_values": self.repeated_code_values,
            "negative_anomaly_threshold_adu": self.negative_anomaly_threshold_adu,
            "target_rows": self.target_rows,
            "maximum_rows": self.maximum_rows,
            "contribution_rows": self.contribution_rows,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation_guardrails": GUARDRAILS,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Options {
    pub patch_padding_px: usize,
    pub psf_fwhm: f64,
    pub negative_anomaly_threshold_adu: f64,
    pub maximum_neighborhood_px: usize,
    pub maximum_count: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            patch_padding_px: 10,
            psf_fwhm: 2.0,
            negative_anomaly_threshold_adu: -1000.0,
            maximum_neighborhood_px: 3,
            maximum_count: 12,
        }
    }
}

fn validate(options: &Options) -> Result<()> {
    if options.patch_padding_px == 0 {
        return Err("patch_padding_px must be positive".into());
    }
    if !options.psf_fwhm.is_finite() || options.psf_fwhm <= 0.0 {
        return Err("psf_fwhm must be finite and positive".into());
    }
    if !options.negative_anomaly_threshold_adu.is_finite() {
        return Err("negative_anomaly_threshold_adu must be finite".into());
    }
    if options.maximum_neighborhood_px < 3 || options.maximum_neighborhood_px.is_multiple_of(2) {
        return Err("maximum_neighborhood_px must be an odd integer >= 3".into());
    }
    if options.maximum_count == 0 {
        return Err("maximum_count must be positive".into());
    }
    Ok(())
}

fn peak_coordinate(source: &TargetSource, x_axis: bool) -> Result<usize> {
    let coordinate = if x_axis {
        source.peak_x.unwrap_or(source.x)
    } else {
        source.peak_y.unwrap_or(source.y)
    };
    if !coordinate.is_finite() || coordinate < 0.0 || coordinate > usize::MAX as f64 {
        return Err("target peak coordinate must be finite and inside the image".into());
    }
    Ok(coordinate.round_ties_even() as usize)
}

fn code_pixel(value: f64, codes: &BTreeSet<i64>) -> bool {
    value.is_finite()
        && value.fract() == 0.0
        && value >= i64::MIN as f64
        && value <= i64::MAX as f64
        && codes.contains(&(value as i64))
}

fn pixel_class(value: f64, codes: &BTreeSet<i64>, negative_threshold: f64) -> usize {
    if code_pixel(value, codes) {
        1
    } else if value <= negative_threshold {
        2
    } else {
        0
    }
}

fn response_and_weight(
    residual: &[f64],
    excluded: &[bool],
    height: usize,
    width: usize,
    sigma: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let valid: Vec<f64> = excluded
        .iter()
        .map(|is_excluded| if *is_excluded { 0.0 } else { 1.0 })
        .collect();
    let weighted: Vec<f64> = residual
        .iter()
        .zip(&valid)
        .map(|(value, valid)| value * valid)
        .collect();
    let weighted = filtering::gaussian_filter(&weighted, height, width, sigma)?;
    let weights = filtering::gaussian_filter(&valid, height, width, sigma)?;
    let response = weighted
        .iter()
        .zip(&weights)
        .map(|(value, weight)| if *weight > 0.5 { value / weight } else { 0.0 })
        .collect();
    Ok((response, weights))
}

fn local_maxima(
    response: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    neighborhood_px: usize,
) -> Result<(Vec<bool>, LocalMaximumRanks)> {
    let maximum = filtering::maximum_filter(response, height, width, neighborhood_px / 2)?;
    let mut points: Vec<(usize, usize, f64)> = (0..response.len())
        .filter(|index| valid[*index] && response[*index] == maximum[*index])
        .map(|index| (index / width, index % width, response[index]))
        .collect();
    points.sort_by(|left, right| right.2.total_cmp(&left.2));
    let ranks = points
        .iter()
        .enumerate()
        .map(|(index, (y, x, _))| ((*y, *x), index + 1))
        .collect();
    let mut mask = vec![false; response.len()];
    for (y, x, _) in points {
        mask[y * width + x] = true;
    }
    Ok((mask, ranks))
}

struct ContributionInput<'a> {
    patch: &'a [f64],
    height: usize,
    width: usize,
    patch_x0: usize,
    patch_y0: usize,
    targets: [(&'a str, &'a TargetSource, usize, usize); 2],
    background: f64,
    sigma: f64,
    codes: &'a BTreeSet<i64>,
    negative_threshold: f64,
}

fn contribution_rows(input: ContributionInput<'_>) -> Result<Vec<ContributionRow>> {
    let ContributionInput {
        patch,
        height,
        width,
        patch_x0,
        patch_y0,
        targets,
        background,
        sigma,
        codes,
        negative_threshold,
    } = input;
    let support_radius = (4.0 * sigma).ceil().max(1.0) as isize;
    let mut rows = Vec::with_capacity(6);
    for (label, source, peak_x, peak_y) in targets {
        let local_x = peak_x as isize - patch_x0 as isize;
        let local_y = peak_y as isize - patch_y0 as isize;
        let mut counts = [0usize; 3];
        let mut weights = [0.0; 3];
        let mut values = [0.0; 3];
        let mut total_weight = 0.0;
        for dy in -support_radius..=support_radius {
            for dx in -support_radius..=support_radius {
                let x = local_x + dx;
                let y = local_y + dy;
                if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                    continue;
                }
                let weight = (-0.5 * (dx * dx + dy * dy) as f64 / sigma.powi(2)).exp();
                let value = patch[y as usize * width + x as usize];
                let class = pixel_class(value, codes, negative_threshold);
                counts[class] += 1;
                weights[class] += weight;
                values[class] += weight * (value - background);
                total_weight += weight;
            }
        }
        if total_weight <= 0.0 {
            return Err("Gaussian support contains no pixels".into());
        }
        for class in 0..PIXEL_CLASSES.len() {
            rows.push(ContributionRow {
                target_label: label.into(),
                detection_id: source.detection_id,
                pixel_class: PIXEL_CLASSES[class].into(),
                pixel_count: counts[class],
                kernel_weight_fraction: weights[class] / total_weight,
                signed_response_contribution_adu: values[class] / total_weight,
            });
        }
    }
    Ok(rows)
}

pub fn run(
    frame: &FitsFrame,
    primary: &TargetSource,
    secondary: &TargetSource,
    repeated_code_values: Option<&[i64]>,
    options: &Options,
) -> Result<PairResponseResult> {
    validate(options)?;
    if primary.detection_id == secondary.detection_id {
        return Err("primary and secondary detections must differ".into());
    }
    let height = frame.height;
    let width = frame.width;
    if height == 0 || width == 0 || frame.data.len() != height * width {
        return Err("expected a 2-D FITS image".into());
    }
    let primary_x = peak_coordinate(primary, true)?;
    let primary_y = peak_coordinate(primary, false)?;
    let secondary_x = peak_coordinate(secondary, true)?;
    let secondary_y = peak_coordinate(secondary, false)?;
    if (primary_x, primary_y) == (secondary_x, secondary_y) {
        return Err("primary and secondary peak coordinates must differ".into());
    }
    if [primary_x, secondary_x].iter().any(|x| *x >= width)
        || [primary_y, secondary_y].iter().any(|y| *y >= height)
    {
        return Err("target peak coordinate must be inside the image".into());
    }
    let x0 = primary_x
        .min(secondary_x)
        .saturating_sub(options.patch_padding_px);
    let y0 = primary_y
        .min(secondary_y)
        .saturating_sub(options.patch_padding_px);
    let x1 = primary_x
        .max(secondary_x)
        .saturating_add(options.patch_padding_px)
        .saturating_add(1)
        .min(width);
    let y1 = primary_y
        .max(secondary_y)
        .saturating_add(options.patch_padding_px)
        .saturating_add(1)
        .min(height);
    let patch_height = y1 - y0;
    let patch_width = x1 - x0;
    let mut patch = Vec::with_capacity(patch_height * patch_width);
    for y in y0..y1 {
        for x in x0..x1 {
            patch.push(frame.data.value(y * width + x));
        }
    }
    let finite: Vec<bool> = patch.iter().map(|value| value.is_finite()).collect();
    if !finite.iter().any(|value| *value) {
        return Err("pair patch contains no finite pixels".into());
    }
    let backgrounds = [primary.background, secondary.background];
    let noises = [primary.noise, secondary.noise];
    let [Some(primary_background), Some(secondary_background)] = backgrounds else {
        return Err(
            "pair detections must contain finite positive noise and finite background".into(),
        );
    };
    let [Some(primary_noise), Some(secondary_noise)] = noises else {
        return Err(
            "pair detections must contain finite positive noise and finite background".into(),
        );
    };
    if !primary_background.is_finite()
        || !secondary_background.is_finite()
        || !primary_noise.is_finite()
        || !secondary_noise.is_finite()
        || primary_noise <= 0.0
        || secondary_noise <= 0.0
    {
        return Err(
            "pair detections must contain finite positive noise and finite background".into(),
        );
    }
    let common_background = (primary_background + secondary_background) / 2.0;
    let common_noise = (primary_noise + secondary_noise) / 2.0;
    let sigma = options.psf_fwhm / FWHM_TO_SIGMA;
    let mut codes = BTreeSet::new();
    if let Some(values) = repeated_code_values {
        codes.extend(values.iter().copied());
    }
    codes.extend(primary.repeated_code_values.iter().copied());
    codes.extend(secondary.repeated_code_values.iter().copied());
    let repeated_code_values: Vec<_> = codes.iter().copied().collect();
    let code_mask: Vec<bool> = patch
        .iter()
        .map(|value| code_pixel(*value, &codes))
        .collect();
    let negative_mask: Vec<bool> = patch
        .iter()
        .zip(&finite)
        .map(|(value, finite)| *finite && *value <= options.negative_anomaly_threshold_adu)
        .collect();
    let residual: Vec<f64> = patch
        .iter()
        .zip(&finite)
        .map(|(value, finite)| {
            if *finite {
                *value - common_background
            } else {
                0.0
            }
        })
        .collect();
    let masks: Vec<Vec<bool>> = [
        finite.iter().map(|value| !*value).collect(),
        finite
            .iter()
            .enumerate()
            .map(|(i, value)| !*value || code_mask[i])
            .collect(),
        finite
            .iter()
            .enumerate()
            .map(|(i, value)| !*value || negative_mask[i])
            .collect(),
        finite
            .iter()
            .enumerate()
            .map(|(i, value)| !*value || code_mask[i] || negative_mask[i])
            .collect(),
    ]
    .into();
    let targets = [
        ("primary", primary, primary_x, primary_y),
        ("secondary", secondary, secondary_x, secondary_y),
    ];
    let target_coordinates = [(primary_x, primary_y), (secondary_x, secondary_y)];
    let mut target_rows = Vec::new();
    let mut maximum_rows = Vec::new();
    for (mode_index, mode) in MASK_MODES.iter().enumerate() {
        let excluded = &masks[mode_index];
        let (response, weights) =
            response_and_weight(&residual, excluded, patch_height, patch_width, sigma)?;
        let valid: Vec<_> = excluded.iter().map(|value| !*value).collect();
        let (local_maximum, ranks) = local_maxima(
            &response,
            &valid,
            patch_height,
            patch_width,
            options.maximum_neighborhood_px,
        )?;
        let mut maxima: Vec<(f64, usize, usize)> = (0..response.len())
            .filter(|index| local_maximum[*index])
            .map(|index| (response[index], index % patch_width, index / patch_width))
            .collect();
        maxima.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then(left.1.cmp(&right.1))
                .then(left.2.cmp(&right.2))
        });
        maxima.reverse();
        for (rank, (response_value, local_x, local_y)) in
            maxima.into_iter().take(options.maximum_count).enumerate()
        {
            let absolute_x = local_x + x0;
            let absolute_y = local_y + y0;
            maximum_rows.push(MaximumRow {
                mask_mode: (*mode).into(),
                rank: rank + 1,
                x: absolute_x,
                y: absolute_y,
                raw_value_adu: patch[local_y * patch_width + local_x],
                gaussian_response_adu: response_value,
                valid_weight: weights[local_y * patch_width + local_x],
                is_primary_target: (absolute_x, absolute_y) == target_coordinates[0],
                is_secondary_target: (absolute_x, absolute_y) == target_coordinates[1],
            });
        }
        for (label, source, peak_x, peak_y) in targets {
            let local_x = peak_x - x0;
            let local_y = peak_y - y0;
            let index = local_y * patch_width + local_x;
            target_rows.push(TargetRow {
                mask_mode: (*mode).into(),
                target_label: label.into(),
                detection_id: source.detection_id,
                peak_x,
                peak_y,
                raw_value_adu: patch[index],
                valid_at_target: valid[index],
                gaussian_response_adu: response[index],
                valid_weight: weights[index],
                response_local_maximum: local_maximum[index],
                local_maximum_rank: ranks.get(&(local_y, local_x)).copied(),
            });
        }
    }
    let contribution_rows = contribution_rows(ContributionInput {
        patch: &patch,
        height: patch_height,
        width: patch_width,
        patch_x0: x0,
        patch_y0: y0,
        targets,
        background: common_background,
        sigma,
        codes: &codes,
        negative_threshold: options.negative_anomaly_threshold_adu,
    })?;
    let raw_primary = target_rows
        .iter()
        .find(|row| row.mask_mode == "raw" && row.target_label == "primary")
        .unwrap();
    let raw_secondary = target_rows
        .iter()
        .find(|row| row.mask_mode == "raw" && row.target_label == "secondary")
        .unwrap();
    let code_primary = target_rows
        .iter()
        .find(|row| row.mask_mode == "repeated_code_masked" && row.target_label == "primary")
        .unwrap();
    let negative_secondary = target_rows
        .iter()
        .find(|row| row.mask_mode == "negative_anomaly_masked" && row.target_label == "secondary")
        .unwrap();
    let conclusion = format!(
        "局部 Gaussian 响应归因完成：raw、重复码留出和负异常留出分别改变了目标位置的有效性与局部极大值结构。主/副 raw 响应为 {:.3}/{:.3} ADU；重复码留出后主位置有效={}，负异常留出后副位置局部极大={}。这支持‘共享亮斑被值域缺口和匹配滤波重塑为两个响应’的优先解释，但不单独给出物理源数或噪点概率。",
        raw_primary.gaussian_response_adu,
        raw_secondary.gaussian_response_adu,
        if code_primary.valid_at_target {
            "True"
        } else {
            "False"
        },
        if negative_secondary.response_local_maximum {
            "True"
        } else {
            "False"
        },
    );
    Ok(PairResponseResult {
        source_path: frame.path.to_string_lossy().into_owned(),
        primary_detection_id: primary.detection_id,
        secondary_detection_id: secondary.detection_id,
        primary_peak: [primary_x, primary_y],
        secondary_peak: [secondary_x, secondary_y],
        patch_bounds: [x0, y0, x1, y1],
        common_background_adu: common_background,
        common_noise_adu: common_noise,
        psf_fwhm_px: options.psf_fwhm,
        repeated_code_values,
        negative_anomaly_threshold_adu: options.negative_anomaly_threshold_adu,
        target_rows,
        maximum_rows,
        contribution_rows,
        parameters: json!({
            "patch_padding_px": options.patch_padding_px,
            "gaussian_sigma_px": sigma,
            "maximum_neighborhood_px": options.maximum_neighborhood_px,
            "maximum_count": options.maximum_count,
            "mask_modes": MASK_MODES,
            "response_definition": "masked Gaussian convolution of (raw - common background), normalized by valid Gaussian weight",
            "contribution_definition": "discrete Gaussian weights within 4 sigma; signed raw ADU excess by mutually exclusive pixel class",
            "interpretation_boundary": "local response attribution; not source truth, FDR, precision, or physical artifact rate",
        }),
        conclusion,
    })
}

pub fn run_path(
    frame_path: &Path,
    catalog_path: &Path,
    primary_id: i64,
    secondary_id: i64,
    repeated_code_values: Option<&[i64]>,
    options: &Options,
) -> Result<PairResponseResult> {
    let (primary, secondary) =
        pair_repair_counterfactual::load_targets_csv(catalog_path, primary_id, secondary_id)?;
    let frame = fits::read_fits(frame_path, true)?;
    run(&frame, &primary, &secondary, repeated_code_values, options)
}

fn write_rows<T: Serialize>(path: &Path, headers: &[&str], rows: &[T]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer.serialize(row).map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_artifacts(result: &PairResponseResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    write_rows(
        &output_dir.join("pair_response_targets.csv"),
        &[
            "mask_mode",
            "target_label",
            "detection_id",
            "peak_x",
            "peak_y",
            "raw_value_adu",
            "valid_at_target",
            "gaussian_response_adu",
            "valid_weight",
            "response_local_maximum",
            "local_maximum_rank",
        ],
        &result.target_rows,
    )?;
    write_rows(
        &output_dir.join("pair_response_maxima.csv"),
        &[
            "mask_mode",
            "rank",
            "x",
            "y",
            "raw_value_adu",
            "gaussian_response_adu",
            "valid_weight",
            "is_primary_target",
            "is_secondary_target",
        ],
        &result.maximum_rows,
    )?;
    write_rows(
        &output_dir.join("pair_response_contributions.csv"),
        &[
            "target_label",
            "detection_id",
            "pixel_class",
            "pixel_count",
            "kernel_weight_fraction",
            "signed_response_contribution_adu",
        ],
        &result.contribution_rows,
    )?;
    fs::write(
        output_dir.join("pair_response_attribution.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::canonicalize(output_dir).map_err(|error| error.to_string())
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    name: &str,
    inline: Option<&'a str>,
) -> Result<&'a str> {
    if let Some(value) = inline {
        Ok(value)
    } else {
        *index += 1;
        args.get(*index)
            .map(String::as_str)
            .ok_or_else(|| format!("{name} requires a value"))
    }
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut frame = None;
    let mut catalog = None;
    let mut primary_id = None;
    let mut secondary_id = None;
    let mut repeated_codes = None;
    let mut output_dir = PathBuf::from("tmp/pair-response-attribution");
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let (name, inline) = args[index]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((args[index].as_str(), None));
        match name {
            "--source-catalog"
            | "--primary-id"
            | "--secondary-id"
            | "--patch-padding-px"
            | "--psf-fwhm"
            | "--repeated-code-values"
            | "--negative-anomaly-threshold-adu"
            | "--maximum-neighborhood-px"
            | "--maximum-count"
            | "--out-dir" => {
                let raw = option_value(args, &mut index, name, inline)?;
                let invalid =
                    |error: &dyn std::fmt::Display| format!("invalid value for {name}: {error}");
                match name {
                    "--source-catalog" => catalog = Some(PathBuf::from(raw)),
                    "--primary-id" => {
                        primary_id = Some(raw.parse::<i64>().map_err(|error| invalid(&error))?)
                    }
                    "--secondary-id" => {
                        secondary_id = Some(raw.parse::<i64>().map_err(|error| invalid(&error))?)
                    }
                    "--patch-padding-px" => {
                        options.patch_padding_px = raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--psf-fwhm" => {
                        options.psf_fwhm = raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--repeated-code-values" => {
                        repeated_codes = Some(
                            raw.split(',')
                                .map(str::trim)
                                .filter(|part| !part.is_empty())
                                .map(|part| part.parse::<i64>().map_err(|error| invalid(&error)))
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                    "--negative-anomaly-threshold-adu" => {
                        options.negative_anomaly_threshold_adu =
                            raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--maximum-neighborhood-px" => {
                        options.maximum_neighborhood_px =
                            raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--maximum-count" => {
                        options.maximum_count = raw.parse().map_err(|error| invalid(&error))?
                    }
                    _ => output_dir = PathBuf::from(raw),
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => {
                if frame.replace(PathBuf::from(name)).is_some() {
                    return Err("only one FITS frame is allowed".into());
                }
            }
        }
        index += 1;
    }
    let frame_path = frame.ok_or("an input FITS frame is required")?;
    let catalog_path = catalog.ok_or("--source-catalog is required")?;
    let primary_id = primary_id.ok_or("--primary-id is required")?;
    let secondary_id = secondary_id.ok_or("--secondary-id is required")?;
    let result = run_path(
        &frame_path,
        &catalog_path,
        primary_id,
        secondary_id,
        repeated_codes.as_deref(),
        &options,
    )?;
    let output_dir = write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-pair-response-attribution",
        "out_dir": output_dir,
        "result_path": output_dir.join("pair_response_attribution.json"),
        "result": result.as_dict(),
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("rst19-pair-response-attribution FRAME --source-catalog CSV --primary-id ID --secondary-id ID [options]");
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "pair response attribution: {}",
        result["out_dir"].as_str().unwrap_or_default()
    );
    println!(
        "{}",
        result["result"]["conclusion"].as_str().unwrap_or_default()
    );
    Ok(())
}

pub fn handle_request(request: &Value) -> Result<Value> {
    let frame_path = request["frame_path"].as_str().ok_or("missing frame_path")?;
    let catalog_path = request["source_catalog_path"]
        .as_str()
        .ok_or("missing source_catalog_path")?;
    let primary_id = request["primary_id"].as_i64().ok_or("missing primary_id")?;
    let secondary_id = request["secondary_id"]
        .as_i64()
        .ok_or("missing secondary_id")?;
    let repeated = request
        .get("repeated_code_values")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value::<Vec<i64>>)
        .transpose()
        .map_err(|error| error.to_string())?;
    let mut options = Options::default();
    if let Some(value) = request.get("options") {
        let mut merged = serde_json::to_value(&options).map_err(|error| error.to_string())?;
        if let (Some(destination), Some(source)) = (merged.as_object_mut(), value.as_object()) {
            for (key, value) in source {
                destination.insert(key.clone(), value.clone());
            }
        }
        options = serde_json::from_value(merged).map_err(|error| error.to_string())?;
    }
    let result = run_path(
        Path::new(frame_path),
        Path::new(catalog_path),
        primary_id,
        secondary_id,
        repeated.as_deref(),
        &options,
    )?;
    let output = if let Some(path) = request.get("output_dir").and_then(Value::as_str) {
        let output = write_artifacts(&result, Path::new(path))?;
        json!({"out_dir":output, "result_path":output.join("pair_response_attribution.json")})
    } else {
        Value::Null
    };
    Ok(json!({"result":result.as_dict(), "output":output}))
}
