//! Repeated, paired feature-class injection controls implemented in Rust.
//!
//! Positive controls measure recovery against injected coordinates. Negative
//! controls measure leakage around known structures; neither is physical truth.

use crate::{
    detection::{self, DetectionOptions},
    fits,
    measurement::{Detection, MeasurementOptions},
    models::PixelData,
    numpy_ziggurat,
    source_geometry_audit::NumpyPcg64,
    Result,
};
use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const HEIGHT: usize = 128;
const WIDTH: usize = 128;
const BACKGROUND_ADU: f64 = 21.0;
const NOISE_SIGMA_ADU: f64 = 5.0;
const PSF_FWHM: f64 = 2.0;
const SEED_STRIDE: u128 = 1_000_003;
const CSV_FIELDS: &[&str] = &[
    "feature_class",
    "scenario",
    "control_type",
    "trial_count",
    "truth_count",
    "candidate_true_hits",
    "quality_true_hits",
    "candidate_recall",
    "quality_recall",
    "candidate_recall_wilson95_low",
    "candidate_recall_wilson95_high",
    "quality_recall_wilson95_low",
    "quality_recall_wilson95_high",
    "mean_nearby_candidate_count",
    "max_nearby_candidate_count",
    "mean_nearby_quality_count",
    "max_nearby_quality_count",
    "mean_candidate_count",
    "mean_quality_count",
    "nearby_flags",
    "note",
];

pub const HELP: &str = "重复运行正/负特征类别注入控制并分开聚合回收与泄漏（Rust）\n\nUsage: rst19-feature-audit-replicates [--trials N] [--seed N] [--out-dir DIR]";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FeatureAuditRow {
    pub feature_class: String,
    pub scenario: String,
    pub control_type: String,
    pub truth_count: usize,
    pub candidate_true_hits: usize,
    pub quality_true_hits: usize,
    pub candidate_recall: Option<f64>,
    pub quality_recall: Option<f64>,
    pub nearby_candidate_count: usize,
    pub nearby_quality_count: usize,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub nearby_flags: String,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FeatureAuditAggregateRow {
    pub feature_class: String,
    pub scenario: String,
    pub control_type: String,
    pub trial_count: usize,
    pub truth_count: usize,
    pub candidate_true_hits: usize,
    pub quality_true_hits: usize,
    pub candidate_recall: Option<f64>,
    pub quality_recall: Option<f64>,
    pub candidate_recall_wilson95_low: Option<f64>,
    pub candidate_recall_wilson95_high: Option<f64>,
    pub quality_recall_wilson95_low: Option<f64>,
    pub quality_recall_wilson95_high: Option<f64>,
    pub mean_nearby_candidate_count: f64,
    pub max_nearby_candidate_count: usize,
    pub mean_nearby_quality_count: f64,
    pub max_nearby_quality_count: usize,
    pub mean_candidate_count: f64,
    pub mean_quality_count: f64,
    pub nearby_flags: String,
    pub note: String,
}

impl FeatureAuditAggregateRow {
    fn as_value(&self) -> Value {
        json!({
            "feature_class": self.feature_class,
            "scenario": self.scenario,
            "control_type": self.control_type,
            "trial_count": self.trial_count,
            "truth_count": self.truth_count,
            "candidate_true_hits": self.candidate_true_hits,
            "quality_true_hits": self.quality_true_hits,
            "candidate_recall": self.candidate_recall,
            "quality_recall": self.quality_recall,
            "candidate_recall_wilson95_low": self.candidate_recall_wilson95_low,
            "candidate_recall_wilson95_high": self.candidate_recall_wilson95_high,
            "quality_recall_wilson95_low": self.quality_recall_wilson95_low,
            "quality_recall_wilson95_high": self.quality_recall_wilson95_high,
            "mean_nearby_candidate_count": self.mean_nearby_candidate_count,
            "max_nearby_candidate_count": self.max_nearby_candidate_count,
            "mean_nearby_quality_count": self.mean_nearby_quality_count,
            "max_nearby_quality_count": self.max_nearby_quality_count,
            "mean_candidate_count": self.mean_candidate_count,
            "mean_quality_count": self.mean_quality_count,
            "nearby_flags": self.nearby_flags,
            "note": self.note,
        })
    }

    fn csv_record(&self) -> Vec<String> {
        vec![
            self.feature_class.clone(),
            self.scenario.clone(),
            self.control_type.clone(),
            self.trial_count.to_string(),
            self.truth_count.to_string(),
            self.candidate_true_hits.to_string(),
            self.quality_true_hits.to_string(),
            option_float(self.candidate_recall),
            option_float(self.quality_recall),
            option_float(self.candidate_recall_wilson95_low),
            option_float(self.candidate_recall_wilson95_high),
            option_float(self.quality_recall_wilson95_low),
            option_float(self.quality_recall_wilson95_high),
            python_float(self.mean_nearby_candidate_count),
            self.max_nearby_candidate_count.to_string(),
            python_float(self.mean_nearby_quality_count),
            self.max_nearby_quality_count.to_string(),
            python_float(self.mean_candidate_count),
            python_float(self.mean_quality_count),
            self.nearby_flags.clone(),
            self.note.clone(),
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureAuditReplicateResult {
    pub trial_count: usize,
    pub base_seed: u128,
    pub rows: Vec<FeatureAuditAggregateRow>,
    pub parameters: Value,
    pub note: String,
}

impl FeatureAuditReplicateResult {
    pub fn as_value(&self) -> Value {
        json!({
            "trial_count": self.trial_count,
            "base_seed": self.base_seed,
            "row_count": self.rows.len(),
            "positive_scenarios": self.rows.iter().filter(|row| row.control_type == "positive").count(),
            "negative_scenarios": self.rows.iter().filter(|row| row.control_type == "negative").count(),
            "parameters": self.parameters,
            "note": self.note,
            "rows": self.rows.iter().map(FeatureAuditAggregateRow::as_value).collect::<Vec<_>>(),
        })
    }

    fn json_bytes(&self) -> Result<Vec<u8>> {
        #[derive(Serialize)]
        struct Artifact<'a> {
            trial_count: usize,
            base_seed: u128,
            row_count: usize,
            positive_scenarios: usize,
            negative_scenarios: usize,
            parameters: &'a Value,
            note: &'a str,
            rows: &'a [FeatureAuditAggregateRow],
        }
        let positive_scenarios = self
            .rows
            .iter()
            .filter(|row| row.control_type == "positive")
            .count();
        let negative_scenarios = self.rows.len() - positive_scenarios;
        serde_json::to_vec_pretty(&Artifact {
            trial_count: self.trial_count,
            base_seed: self.base_seed,
            row_count: self.rows.len(),
            positive_scenarios,
            negative_scenarios,
            parameters: &self.parameters,
            note: &self.note,
            rows: &self.rows,
        })
        .map_err(|error| error.to_string())
    }
}

#[derive(Clone)]
struct Scenario {
    feature_class: &'static str,
    name: &'static str,
    control_type: &'static str,
    truth_positions: Vec<(f64, f64)>,
    feature_positions: Vec<(f64, f64)>,
    nearby_radius: f64,
    build: SceneBuilder,
    mask_outer_pixel: bool,
    saturation_level: Option<f64>,
    note: &'static str,
}

#[derive(Clone)]
enum SceneBuilder {
    Gaussian(Vec<(f64, f64, f64, f64)>),
    SinglePixelSpike,
    LongLine,
}

fn standard_scenarios() -> Vec<Scenario> {
    let center_x = WIDTH as f64 / 2.0;
    let center_y = HEIGHT as f64 / 2.0;
    let line_positions = if WIDTH > 80 {
        (18..HEIGHT - 18).map(|y| (64.0, y as f64)).collect()
    } else {
        (18..HEIGHT - 18).map(|y| (center_x, y as f64)).collect()
    };
    let center = vec![(center_x, center_y)];
    vec![
        Scenario {
            feature_class: "compact_quality",
            name: "isolated_fwhm2",
            control_type: "positive",
            truth_positions: center.clone(),
            feature_positions: center.clone(),
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![(center_x, center_y, 128.0, 2.0)]),
            mask_outer_pixel: false,
            saturation_level: None,
            note: "已知孤立 Gaussian PSF；检查标准紧凑源回收",
        },
        Scenario {
            feature_class: "weak_or_background",
            name: "weak_fwhm2_peak28",
            control_type: "positive",
            truth_positions: center.clone(),
            feature_positions: center.clone(),
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![(center_x, center_y, 28.0, 2.0)]),
            mask_outer_pixel: false,
            saturation_level: None,
            note: "已知弱 Gaussian PSF；检查候选层与质量层的差距",
        },
        Scenario {
            feature_class: "shape_outlier",
            name: "broad_fwhm6",
            control_type: "positive",
            truth_positions: center.clone(),
            feature_positions: center.clone(),
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![(center_x, center_y, 128.0, 6.0)]),
            mask_outer_pixel: false,
            saturation_level: None,
            note: "已知宽 PSF；不预设它一定是假源，观察模型失配代价",
        },
        Scenario {
            feature_class: "spike_or_support",
            name: "narrow_fwhm0.8",
            control_type: "positive",
            truth_positions: center.clone(),
            feature_positions: center.clone(),
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![(center_x, center_y, 160.0, 0.8)]),
            mask_outer_pixel: false,
            saturation_level: None,
            note: "已知欠采样窄源；检查尖峰/PSF 支持审计是否生效",
        },
        Scenario {
            feature_class: "crowded_blend",
            name: "pair_separation3",
            control_type: "positive",
            truth_positions: vec![(center_x - 1.5, center_y), (center_x + 1.5, center_y)],
            feature_positions: vec![(center_x - 1.5, center_y), (center_x + 1.5, center_y)],
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![
                (center_x - 1.5, center_y, 128.0, 2.0),
                (center_x + 1.5, center_y, 128.0, 2.0),
            ]),
            mask_outer_pixel: false,
            saturation_level: None,
            note: "已知 3 px 双源；低于/接近分辨尺度，允许合并结果",
        },
        Scenario {
            feature_class: "crowded_blend",
            name: "pair_separation6",
            control_type: "positive",
            truth_positions: vec![(center_x - 3.0, center_y), (center_x + 3.0, center_y)],
            feature_positions: vec![(center_x - 3.0, center_y), (center_x + 3.0, center_y)],
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![
                (center_x - 3.0, center_y, 128.0, 2.0),
                (center_x + 3.0, center_y, 128.0, 2.0),
            ]),
            mask_outer_pixel: false,
            saturation_level: None,
            note: "已知 6 px 双源；与当前双源细筛证据线对照",
        },
        Scenario {
            feature_class: "spike_or_support",
            name: "single_pixel_spike",
            control_type: "negative",
            truth_positions: vec![],
            feature_positions: center.clone(),
            nearby_radius: 5.0,
            build: SceneBuilder::SinglePixelSpike,
            mask_outer_pixel: false,
            saturation_level: None,
            note: "阴性样本：只有单像素尖峰，没有注入恒星",
        },
        Scenario {
            feature_class: "masked_or_edge",
            name: "masked_source",
            control_type: "positive",
            truth_positions: center.clone(),
            feature_positions: center.clone(),
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![(center_x, center_y, 128.0, 2.0)]),
            mask_outer_pixel: true,
            saturation_level: None,
            note: "已知 PSF 的孔径外圈叠加显式掩膜；检查有效性拒绝",
        },
        Scenario {
            feature_class: "masked_or_edge",
            name: "edge_source",
            control_type: "positive",
            truth_positions: vec![(3.0, center_y)],
            feature_positions: vec![(3.0, center_y)],
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![(3.0, center_y, 128.0, 2.0)]),
            mask_outer_pixel: false,
            saturation_level: None,
            note: "已知靠近边界的 PSF；检查不完整孔径不被当作普通星",
        },
        Scenario {
            feature_class: "range_anomaly",
            name: "saturated_source",
            control_type: "positive",
            truth_positions: center.clone(),
            feature_positions: center.clone(),
            nearby_radius: 5.0,
            build: SceneBuilder::Gaussian(vec![(center_x, center_y, 800.0, 2.0)]),
            mask_outer_pixel: false,
            saturation_level: Some(250.0),
            note: "已知高幅度 PSF，检测器显式设置饱和线；检查范围审计",
        },
        Scenario {
            feature_class: "linear_artifact",
            name: "long_line_negative",
            control_type: "negative",
            truth_positions: vec![],
            feature_positions: line_positions,
            nearby_radius: 4.0,
            build: SceneBuilder::LongLine,
            mask_outer_pixel: false,
            saturation_level: None,
            note: "阴性样本：长线残差，不注入恒星；检查线状源泄漏",
        },
    ]
}

fn build_image(seed: u128, scenario: &Scenario) -> Result<(Vec<f32>, Vec<bool>)> {
    let mut rng = NumpyPcg64::new(seed);
    let mut image = (0..HEIGHT * WIDTH)
        .map(|_| {
            (BACKGROUND_ADU + NOISE_SIGMA_ADU * numpy_ziggurat::standard_normal(&mut rng)) as f32
        })
        .collect::<Vec<_>>();
    match &scenario.build {
        SceneBuilder::Gaussian(sources) => {
            for &(x, y, peak, fwhm) in sources {
                inject_gaussian(&mut image, WIDTH, HEIGHT, x, y, peak, fwhm);
            }
        }
        SceneBuilder::SinglePixelSpike => {
            image[(HEIGHT / 2) * WIDTH + WIDTH / 2] += 400.0;
        }
        SceneBuilder::LongLine => {
            for y in 18..HEIGHT - 18 {
                let x = if WIDTH > 80 { 64 } else { WIDTH / 2 };
                image[y * WIDTH + x] += 80.0;
                image[y * WIDTH + x + 1] += 50.0;
            }
        }
    }
    let mut mask = fits::auxiliary_mask(HEIGHT, WIDTH)?;
    if scenario.mask_outer_pixel {
        mask[(HEIGHT / 2) * WIDTH + WIDTH / 2 + 3] = true;
    }
    Ok((image, mask))
}

fn inject_gaussian(
    image: &mut [f32],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
    peak: f64,
    fwhm: f64,
) {
    let sigma = (fwhm / 2.354_820_045).max(f64::EPSILON);
    let radius = (4.0 * sigma).ceil().max(4.0) as isize;
    let floor_x = x.floor() as isize;
    let floor_y = y.floor() as isize;
    let x_min = (floor_x - radius).max(0) as usize;
    let x_max = (floor_x + radius + 1).min(width as isize) as usize;
    let y_min = (floor_y - radius).max(0) as usize;
    let y_max = (floor_y + radius + 1).min(height as isize) as usize;
    for py in y_min..y_max {
        for px in x_min..x_max {
            let dx = px as f64 - x;
            let dy = py as f64 - y;
            let value = peak * (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
            let index = py * width + px;
            image[index] = (image[index] as f64 + value) as f32;
        }
    }
}

fn detect_scenario(seed: u128, scenario: &Scenario) -> Result<FeatureAuditRow> {
    let (image, mask) = build_image(seed, scenario)?;
    let mut options = DetectionOptions::default();
    options.threshold_sigma = 4.0;
    options.min_distance = 4;
    options.background_box_size = 32;
    options.background_sample_limit = 100_000;
    options.reject_linear_artifacts = true;
    options.proposal_mode = "hybrid".to_owned();
    options.measurement = MeasurementOptions {
        aperture_radius: 4,
        psf_fwhm: PSF_FWHM,
        min_flux_snr: 5.0,
        min_psf_support_pixels: 3,
        saturation_level: scenario.saturation_level,
        refine_local_background: true,
        ..MeasurementOptions::default()
    };
    let result = detection::detect_sources(
        &PixelData::F32(image),
        Some(&mask),
        HEIGHT,
        WIDTH,
        &options,
        None,
    )?;
    let nearby = result
        .sources
        .iter()
        .filter(|source| {
            scenario
                .feature_positions
                .iter()
                .any(|(x, y)| (source.x - x).hypot(source.y - y) <= scenario.nearby_radius)
        })
        .collect::<Vec<_>>();
    let mut flag_counts = BTreeMap::<String, usize>::new();
    for source in &nearby {
        for flag in &source.flags {
            *flag_counts.entry(flag.clone()).or_default() += 1;
        }
    }
    let mut flags = flag_counts.into_iter().collect::<Vec<_>>();
    flags.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let nearby_flags = flags
        .into_iter()
        .map(|(flag, count)| format!("{flag}:{count}"))
        .collect::<Vec<_>>()
        .join("|");
    let radius = 3.0_f64.max(PSF_FWHM);
    let candidate_hits = greedy_match_count(&scenario.truth_positions, &result.sources, radius);
    let quality_sources = result
        .sources
        .iter()
        .filter(|source| source.quality_passed)
        .cloned()
        .collect::<Vec<_>>();
    let quality_hits = greedy_match_count(&scenario.truth_positions, &quality_sources, radius);
    let truth_count = scenario.truth_positions.len();
    Ok(FeatureAuditRow {
        feature_class: scenario.feature_class.to_owned(),
        scenario: scenario.name.to_owned(),
        control_type: scenario.control_type.to_owned(),
        truth_count,
        candidate_true_hits: candidate_hits,
        quality_true_hits: quality_hits,
        candidate_recall: (truth_count > 0).then_some(candidate_hits as f64 / truth_count as f64),
        quality_recall: (truth_count > 0).then_some(quality_hits as f64 / truth_count as f64),
        nearby_candidate_count: nearby.len(),
        nearby_quality_count: nearby.iter().filter(|source| source.quality_passed).count(),
        candidate_count: result.candidate_count,
        quality_count: result.quality_count,
        nearby_flags,
        note: scenario.note.to_owned(),
    })
}

fn greedy_match_count(targets: &[(f64, f64)], detections: &[Detection], radius: f64) -> usize {
    let mut remaining = detections.iter().collect::<Vec<_>>();
    let mut matched = 0;
    for &(target_x, target_y) in targets {
        let mut best_index = None;
        let mut best_distance = radius;
        for (index, detection) in remaining.iter().enumerate() {
            let distance = (detection.x - target_x).hypot(detection.y - target_y);
            if distance <= best_distance {
                best_distance = distance;
                best_index = Some(index);
            }
        }
        if let Some(index) = best_index {
            remaining.remove(index);
            matched += 1;
        }
    }
    matched
}

fn wilson95_interval(successes: usize, trials: usize) -> Result<(f64, f64)> {
    if trials == 0 || successes > trials {
        return Err(format!(
            "invalid recall counts: successes={successes}, trials={trials}"
        ));
    }
    let z = 1.959_963_984_540_054_f64;
    let proportion = successes as f64 / trials as f64;
    let denominator = 1.0 + z * z / trials as f64;
    let center = (proportion + z * z / (2.0 * trials as f64)) / denominator;
    let half_width = z
        * (proportion * (1.0 - proportion) / trials as f64
            + z * z / (4.0 * (trials * trials) as f64))
            .sqrt()
        / denominator;
    Ok((
        if successes == 0 {
            0.0
        } else {
            (center - half_width).max(0.0)
        },
        if successes == trials {
            1.0
        } else {
            (center + half_width).min(1.0)
        },
    ))
}

fn merge_flags(rows: &[&FeatureAuditRow]) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for row in rows {
        let trial_flags = row
            .nearby_flags
            .split('|')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(|token| {
                token
                    .split_once(':')
                    .map_or(token, |(name, _)| name)
                    .to_owned()
            })
            .collect::<BTreeSet<_>>();
        for flag in trial_flags {
            *counts.entry(flag).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .map(|(flag, count)| format!("{flag}:{count}"))
        .collect::<Vec<_>>()
        .join("|")
}

fn aggregate_rows(
    grouped: BTreeMap<String, Vec<FeatureAuditRow>>,
    trial_count: usize,
) -> Result<Vec<FeatureAuditAggregateRow>> {
    let mut output = Vec::with_capacity(grouped.len());
    for (scenario, rows) in grouped {
        let first = rows
            .first()
            .ok_or_else(|| format!("scenario {scenario:?} has no replicate rows"))?;
        if rows.iter().any(|row| {
            row.feature_class != first.feature_class || row.control_type != first.control_type
        }) {
            return Err(format!(
                "scenario {scenario:?} changed feature class or control type across trials"
            ));
        }
        let truth_count = rows.iter().map(|row| row.truth_count).sum::<usize>();
        let candidate_true_hits = rows
            .iter()
            .map(|row| row.candidate_true_hits)
            .sum::<usize>();
        let quality_true_hits = rows.iter().map(|row| row.quality_true_hits).sum::<usize>();
        let (candidate_low, candidate_high) = if truth_count == 0 {
            (None, None)
        } else {
            let interval = wilson95_interval(candidate_true_hits, truth_count)?;
            (Some(interval.0), Some(interval.1))
        };
        let (quality_low, quality_high) = if truth_count == 0 {
            (None, None)
        } else {
            let interval = wilson95_interval(quality_true_hits, truth_count)?;
            (Some(interval.0), Some(interval.1))
        };
        let mean = |values: Vec<usize>| values.iter().sum::<usize>() as f64 / values.len() as f64;
        let nearby_candidate = rows
            .iter()
            .map(|row| row.nearby_candidate_count)
            .collect::<Vec<_>>();
        let nearby_quality = rows
            .iter()
            .map(|row| row.nearby_quality_count)
            .collect::<Vec<_>>();
        let candidate_count = rows
            .iter()
            .map(|row| row.candidate_count)
            .collect::<Vec<_>>();
        let quality_count = rows.iter().map(|row| row.quality_count).collect::<Vec<_>>();
        output.push(FeatureAuditAggregateRow {
            feature_class: first.feature_class.clone(),
            scenario: first.scenario.clone(),
            control_type: first.control_type.clone(),
            trial_count,
            truth_count,
            candidate_true_hits,
            quality_true_hits,
            candidate_recall: (truth_count > 0)
                .then_some(candidate_true_hits as f64 / truth_count as f64),
            quality_recall: (truth_count > 0)
                .then_some(quality_true_hits as f64 / truth_count as f64),
            candidate_recall_wilson95_low: candidate_low,
            candidate_recall_wilson95_high: candidate_high,
            quality_recall_wilson95_low: quality_low,
            quality_recall_wilson95_high: quality_high,
            mean_nearby_candidate_count: mean(nearby_candidate.iter().copied().collect()),
            max_nearby_candidate_count: *nearby_candidate.iter().max().unwrap_or(&0),
            mean_nearby_quality_count: mean(nearby_quality.iter().copied().collect()),
            max_nearby_quality_count: *nearby_quality.iter().max().unwrap_or(&0),
            mean_candidate_count: mean(candidate_count),
            mean_quality_count: mean(quality_count),
            nearby_flags: merge_flags(&rows.iter().collect::<Vec<_>>()),
            note: first.note.clone(),
        });
    }
    Ok(output)
}

pub fn run_feature_audit_replicates<F>(
    trials: usize,
    seed: u128,
    mut progress: F,
) -> Result<FeatureAuditReplicateResult>
where
    F: FnMut(usize, usize),
{
    if trials == 0 {
        return Err("trials must be positive".into());
    }
    let scenarios = standard_scenarios();
    let expected_scenarios = scenarios
        .iter()
        .map(|scenario| scenario.name)
        .collect::<Vec<_>>();
    let mut grouped = BTreeMap::<String, Vec<FeatureAuditRow>>::new();
    for trial_index in 0..trials {
        let trial_seed = seed
            .checked_add(trial_index as u128 * SEED_STRIDE)
            .ok_or_else(|| "trial seed overflow".to_owned())?;
        let mut observed = Vec::with_capacity(scenarios.len());
        for (scenario_index, scenario) in scenarios.iter().enumerate() {
            let scenario_seed = trial_seed
                .checked_add(scenario_index as u128 * 1009)
                .ok_or_else(|| "scenario seed overflow".to_owned())?;
            observed.push(detect_scenario(scenario_seed, scenario)?);
        }
        let observed_scenarios = observed
            .iter()
            .map(|row| row.scenario.as_str())
            .collect::<Vec<_>>();
        if observed_scenarios != expected_scenarios {
            return Err("native feature audit scenario order changed".into());
        }
        if observed_scenarios.iter().collect::<BTreeSet<_>>().len() != observed_scenarios.len() {
            return Err("native feature audit returned duplicate scenarios".into());
        }
        for row in observed {
            grouped.entry(row.scenario.clone()).or_default().push(row);
        }
        progress(trial_index + 1, trials);
    }
    let rows = aggregate_rows(grouped, trials)?;
    let parameters = json!({
        "trial_count": trials,
        "base_seed": seed,
        "seed_stride": SEED_STRIDE,
        "feature_audit_kwargs": {},
        "positive_recall_denominator": "sum(truth_count) across repeated known-source scenes",
        "negative_leakage_denominator": "trial count; no injected truth source exists",
        "interval_definition": "descriptive Wilson 95% interval for positive known-source hits only",
        "physical_truth_warning": true,
    });
    Ok(FeatureAuditReplicateResult {
        trial_count: trials,
        base_seed: seed,
        rows,
        parameters,
        note: "正样本的候选/质量回收率来自已知注入坐标；负样本只统计结构邻域泄漏。二者不合并为 accuracy、precision、真实伪影率或物理恒星概率。".into(),
    })
}

fn python_float(value: f64) -> String {
    let raw = format!("{value:?}");
    if let Some((mantissa, exponent)) = raw.split_once('e') {
        format!("{mantissa}e{:+03}", exponent.parse::<i32>().unwrap_or(0))
    } else {
        raw
    }
}

fn option_float(value: Option<f64>) -> String {
    value.map_or_else(String::new, python_float)
}

pub fn write_artifacts(result: &FeatureAuditReplicateResult, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .terminator(Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(CSV_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in &result.rows {
        writer
            .write_record(row.csv_record())
            .map_err(|error| error.to_string())?;
    }
    let mut csv_bytes = vec![0xef, 0xbb, 0xbf];
    csv_bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(out_dir.join("feature_audit_replicates.csv"), csv_bytes)
        .map_err(|error| error.to_string())?;
    fs::write(
        out_dir.join("feature_audit_replicates.json"),
        result.json_bytes()?,
    )
    .map_err(|error| error.to_string())?;
    Ok(out_dir.to_path_buf())
}

#[derive(Debug)]
struct CliOptions {
    trials: usize,
    seed: u128,
    out_dir: PathBuf,
}

fn parse_options(args: &[String]) -> Result<CliOptions> {
    let mut options = CliOptions {
        trials: 16,
        seed: 19_019,
        out_dir: PathBuf::from("tmp/feature-audit-replicates"),
    };
    let mut index = 0;
    while index < args.len() {
        let (name, inline_value) = args[index]
            .split_once('=')
            .map_or((args[index].as_str(), None), |(name, value)| {
                (name, Some(value.to_owned()))
            });
        if name == "--help" || name == "-h" {
            return Err(HELP.into());
        }
        let value = if let Some(value) = inline_value {
            value
        } else {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| format!("missing value for {name}"))?
        };
        match name {
            "--trials" => {
                options.trials = value
                    .parse()
                    .map_err(|_| "invalid --trials value".to_owned())?;
                if options.trials == 0 {
                    return Err("--trials must be positive".into());
                }
            }
            "--seed" => {
                options.seed = value
                    .parse()
                    .map_err(|_| "invalid non-negative --seed value".to_owned())?
            }
            "--out-dir" => options.out_dir = PathBuf::from(value),
            _ => return Err(format!("unknown option: {name}")),
        }
        index += 1;
    }
    Ok(options)
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let options = parse_options(args)?;
    let result = run_feature_audit_replicates(options.trials, options.seed, |_, _| {})?;
    let output = write_artifacts(&result, &options.out_dir)?;
    Ok(json!({"out_dir":output,"result":result.as_value()}))
}

pub fn run_cli(args: &[String]) -> i32 {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{HELP}");
        return 0;
    }
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("rst19-feature-audit-replicates: {error}");
            return 2;
        }
    };
    let result =
        match run_feature_audit_replicates(options.trials, options.seed, |completed, total| {
            eprintln!("feature-audit-replicates: trial {completed}/{total}");
        }) {
            Ok(result) => result,
            Err(error) => {
                eprintln!("rst19-feature-audit-replicates: 运行失败：{error}");
                return 2;
            }
        };
    match write_artifacts(&result, &options.out_dir) {
        Ok(output) => println!(
            "rst19-feature-audit-replicates: 结果已写入 {}",
            output.display()
        ),
        Err(error) => {
            eprintln!("rst19-feature-audit-replicates: 运行失败：{error}");
            return 2;
        }
    }
    for row in &result.rows {
        if row.truth_count > 0 {
            println!(
                "{}/{}: recall={:.4}/{:.4} candidate_ci=[{:.4},{:.4}] quality_ci=[{:.4},{:.4}]",
                row.feature_class,
                row.scenario,
                row.candidate_recall.unwrap_or(0.0),
                row.quality_recall.unwrap_or(0.0),
                row.candidate_recall_wilson95_low.unwrap_or(0.0),
                row.candidate_recall_wilson95_high.unwrap_or(0.0),
                row.quality_recall_wilson95_low.unwrap_or(0.0),
                row.quality_recall_wilson95_high.unwrap_or(0.0),
            );
        } else {
            println!(
                "{}/{}: leak_mean={:.2}/{:.2} leak_max={}/{}",
                row.feature_class,
                row.scenario,
                row.mean_nearby_candidate_count,
                row.mean_nearby_quality_count,
                row.max_nearby_candidate_count,
                row.max_nearby_quality_count,
            );
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        scenario: &str,
        control_type: &str,
        truth_count: usize,
        candidate_hits: usize,
        flags: &str,
    ) -> FeatureAuditRow {
        FeatureAuditRow {
            feature_class: if control_type == "positive" {
                "compact_quality"
            } else {
                "linear_artifact"
            }
            .into(),
            scenario: scenario.into(),
            control_type: control_type.into(),
            truth_count,
            candidate_true_hits: candidate_hits,
            quality_true_hits: 0,
            candidate_recall: (truth_count > 0)
                .then_some(candidate_hits as f64 / truth_count as f64),
            quality_recall: (truth_count > 0).then_some(0.0),
            nearby_candidate_count: 2,
            nearby_quality_count: 0,
            candidate_count: 3,
            quality_count: 0,
            nearby_flags: flags.into(),
            note: "test".into(),
        }
    }

    #[test]
    fn replicate_aggregates_separate_positive_recall_and_negative_leakage() {
        let rows = BTreeMap::from([
            (
                "known".to_owned(),
                vec![
                    row("known", "positive", 1, 1, ""),
                    row("known", "positive", 1, 1, ""),
                ],
            ),
            (
                "negative".to_owned(),
                vec![
                    row("negative", "negative", 0, 0, "LINE_ARTIFACT:1|SPIKE:2"),
                    row("negative", "negative", 0, 0, "LINE_ARTIFACT:3"),
                ],
            ),
        ]);
        let aggregate = aggregate_rows(rows, 2).unwrap();
        assert_eq!(aggregate[0].scenario, "known");
        assert_eq!(aggregate[0].candidate_recall, Some(1.0));
        assert_eq!(
            aggregate[0].candidate_recall_wilson95_low,
            Some(0.342_380_227_506_653_03)
        );
        assert_eq!(aggregate[1].candidate_recall, None);
        assert_eq!(aggregate[1].nearby_flags, "LINE_ARTIFACT:2|SPIKE:1");
    }

    #[test]
    fn seeded_single_trial_has_expected_scene_contract() {
        let scenarios = standard_scenarios();
        assert_eq!(scenarios.len(), 11);
        let rows = scenarios
            .iter()
            .enumerate()
            .map(|(index, scenario)| {
                detect_scenario(19_019 + index as u128 * 1009, scenario).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(rows[0].scenario, "isolated_fwhm2");
        assert_eq!(rows[0].truth_count, 1);
        assert_eq!(rows[9].scenario, "saturated_source");
        assert_eq!(rows[10].scenario, "long_line_negative");
    }

    #[test]
    fn seeded_trial_images_match_frozen_numpy_inputs() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/feature_audit_replicates_parity/frames-f32le.bin");
        let expected = fs::read(fixture).unwrap();
        let frame_bytes = HEIGHT * WIDTH * std::mem::size_of::<f32>();
        let scenarios = standard_scenarios();
        assert_eq!(expected.len(), scenarios.len() * frame_bytes);
        for (index, scenario) in scenarios.iter().enumerate() {
            let seed = 19_019_u128 + index as u128 * 1009;
            let (image, _) = build_image(seed, scenario).unwrap();
            let actual = image
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let start = index * frame_bytes;
            assert_eq!(
                actual.as_slice(),
                &expected[start..start + frame_bytes],
                "frozen input differs for {}",
                scenario.name
            );
        }
    }

    #[test]
    fn writer_uses_bom_and_python_field_order() {
        let result = FeatureAuditReplicateResult {
            trial_count: 1,
            base_seed: 19_019,
            rows: vec![FeatureAuditAggregateRow {
                feature_class: "compact_quality".into(),
                scenario: "known".into(),
                control_type: "positive".into(),
                trial_count: 1,
                truth_count: 1,
                candidate_true_hits: 1,
                quality_true_hits: 1,
                candidate_recall: Some(1.0),
                quality_recall: Some(1.0),
                candidate_recall_wilson95_low: Some(1.0),
                candidate_recall_wilson95_high: Some(1.0),
                quality_recall_wilson95_low: Some(1.0),
                quality_recall_wilson95_high: Some(1.0),
                mean_nearby_candidate_count: 1.0,
                max_nearby_candidate_count: 1,
                mean_nearby_quality_count: 1.0,
                max_nearby_quality_count: 1,
                mean_candidate_count: 1.0,
                mean_quality_count: 1.0,
                nearby_flags: String::new(),
                note: "test".into(),
            }],
            parameters: json!({}),
            note: "test".into(),
        };
        assert_eq!(result.as_value()["rows"][0]["scenario"], "known");
        assert!(result.rows[0].csv_record().starts_with(&[
            "compact_quality".into(),
            "known".into(),
            "positive".into(),
        ]));
    }
}
