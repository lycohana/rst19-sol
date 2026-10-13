//! Cross-check per-class source summaries against temporal persistence and
//! optional class-transition matrices.  All reported rates remain descriptive
//! detector evidence and are not physical-source probabilities.

use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    path::{Path, PathBuf},
};

const CSV_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "anchor_candidate_count",
    "anchor_quality_count",
    "median_flux_snr",
    "max_flux_snr",
    "required_presence",
    "frame_count",
    "candidate_presence_ge_required_count",
    "candidate_presence_fraction",
    "quality_presence_ge_required_count",
    "quality_presence_fraction",
    "persistence_gap",
    "high_snr_rejected",
    "interpretation",
    "candidate_any_response_fraction",
    "candidate_same_class_response_fraction",
    "quality_any_response_fraction",
    "quality_same_class_response_fraction",
    "dominant_cross_class",
    "dominant_cross_class_label",
];

#[derive(Debug, Clone, Serialize)]
pub struct FeatureCrossAuditRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub anchor_candidate_count: i64,
    pub anchor_quality_count: i64,
    pub median_flux_snr: Option<f64>,
    pub max_flux_snr: Option<f64>,
    pub required_presence: i64,
    pub frame_count: i64,
    pub candidate_presence_ge_required_count: i64,
    pub candidate_presence_fraction: Option<f64>,
    pub quality_presence_ge_required_count: i64,
    pub quality_presence_fraction: Option<f64>,
    pub persistence_gap: Option<f64>,
    pub high_snr_rejected: bool,
    pub interpretation: String,
    pub candidate_any_response_fraction: Option<f64>,
    pub candidate_same_class_response_fraction: Option<f64>,
    pub quality_any_response_fraction: Option<f64>,
    pub quality_same_class_response_fraction: Option<f64>,
    pub dominant_cross_class: Option<String>,
    pub dominant_cross_class_label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureCrossAuditResult {
    pub source_summary_path: String,
    pub persistence_path: String,
    pub frame_count: i64,
    pub required_presence: i64,
    pub high_snr_threshold: f64,
    pub rows: Vec<FeatureCrossAuditRow>,
    pub conclusion: String,
    pub class_transition_path: Option<String>,
}

impl FeatureCrossAuditResult {
    pub fn as_dict(&self) -> Value {
        json!(self)
    }
}

#[derive(Debug)]
struct CsvInput {
    rows: Vec<BTreeMap<String, String>>,
}

#[derive(Debug, Clone)]
struct TransitionMetric {
    matched: i64,
    possible: i64,
    response_label: String,
}

type TransitionKey = (String, String, String);

fn read_rows(path: &Path) -> Result<CsvInput> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| format!("cannot read CSV {}: {error}", path.display()))?;
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    if headers.is_empty() {
        return Err(format!("CSV has no header: {}", path.display()));
    }
    let columns: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index == 0 {
                value.trim_start_matches('\u{feff}').to_owned()
            } else {
                value.to_owned()
            }
        })
        .collect();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let mut row = BTreeMap::new();
        for (index, column) in columns.iter().enumerate() {
            row.insert(
                column.clone(),
                record.get(index).unwrap_or_default().to_owned(),
            );
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("CSV has no rows: {}", path.display()));
    }
    Ok(CsvInput { rows })
}

fn cell<'a>(row: &'a BTreeMap<String, String>, field: &str) -> &'a str {
    row.get(field).map(String::as_str).unwrap_or_default()
}

fn optional_float(value: &str, field: &str, feature_class: &str) -> Result<Option<f64>> {
    let text = value.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let number = text
        .parse::<f64>()
        .map_err(|_| format!("invalid {field} for {feature_class}: {value:?}"))?;
    Ok(number.is_finite().then_some(number))
}

fn required_int(value: &str, field: &str, feature_class: &str) -> Result<i64> {
    let number = optional_float(value, field, feature_class)?
        .filter(|number| {
            number.fract() == 0.0
                && *number >= i64::MIN as f64
                && *number < 9_223_372_036_854_775_808.0
        })
        .ok_or_else(|| format!("invalid integer {field} for {feature_class}: {value:?}"))?;
    Ok(number as i64)
}

fn keyed_rows(
    rows: &[BTreeMap<String, String>],
    path: &Path,
) -> Result<BTreeMap<String, BTreeMap<String, String>>> {
    let mut result = BTreeMap::new();
    for row in rows {
        let feature_class = cell(row, "feature_class").trim().to_owned();
        if feature_class.is_empty() {
            return Err(format!(
                "CSV row has empty feature_class: {}",
                path.display()
            ));
        }
        if result.insert(feature_class.clone(), row.clone()).is_some() {
            return Err(format!(
                "CSV has duplicate feature_class={feature_class}: {}",
                path.display()
            ));
        }
    }
    Ok(result)
}

fn transition_fraction(
    metrics: &BTreeMap<TransitionKey, TransitionMetric>,
    layer: &str,
    anchor: &str,
    response: &str,
) -> Option<f64> {
    let metric = metrics.get(&(layer.to_owned(), anchor.to_owned(), response.to_owned()))?;
    (metric.possible > 0).then_some(metric.matched as f64 / metric.possible as f64)
}

pub fn run_feature_cross_audit(
    source_summary_path: impl AsRef<Path>,
    persistence_path: impl AsRef<Path>,
    high_snr_threshold: f64,
    class_transition_path: Option<&Path>,
) -> Result<FeatureCrossAuditResult> {
    let source_path = source_summary_path.as_ref();
    let persistence_file = persistence_path.as_ref();
    if high_snr_threshold <= 0.0 {
        return Err("high_snr_threshold must be positive".into());
    }
    let source_rows = read_rows(source_path)?.rows;
    let persistence_rows = read_rows(persistence_file)?.rows;
    let source_by_class = keyed_rows(&source_rows, source_path)?;
    let persistence_by_class = keyed_rows(&persistence_rows, persistence_file)?;
    let source_classes: BTreeSet<_> = source_by_class.keys().cloned().collect();
    let persistence_classes: BTreeSet<_> = persistence_by_class.keys().cloned().collect();
    if source_classes != persistence_classes {
        let missing_in_persistence: Vec<_> = source_classes
            .difference(&persistence_classes)
            .cloned()
            .collect();
        let missing_in_source: Vec<_> = persistence_classes
            .difference(&source_classes)
            .cloned()
            .collect();
        return Err(format!(
            "feature classes do not match: missing_in_persistence={missing_in_persistence:?}, missing_in_source={missing_in_source:?}"
        ));
    }

    let first_persistence = &persistence_rows[0];
    let frame_count = required_int(
        cell(first_persistence, "frame_count"),
        "frame_count",
        "<header>",
    )?;
    let required_presence = required_int(
        cell(first_persistence, "required_presence"),
        "required_presence",
        "<header>",
    )?;
    if frame_count < 1 || required_presence < 1 || required_presence > frame_count {
        return Err("persistence frame_count/required_presence is invalid".into());
    }

    let mut source_anchor_counts = BTreeMap::new();
    for source_row in &source_rows {
        let feature_class = cell(source_row, "feature_class").trim().to_owned();
        let candidate_count = required_int(
            cell(source_row, "candidate_count"),
            "candidate_count",
            &feature_class,
        )?;
        let quality_count = required_int(
            cell(source_row, "quality_count"),
            "quality_count",
            &feature_class,
        )?;
        source_anchor_counts.insert(feature_class, (candidate_count, quality_count));
    }

    let mut transition_metrics: BTreeMap<TransitionKey, TransitionMetric> = BTreeMap::new();
    if let Some(transition_path) = class_transition_path {
        let transition_rows = read_rows(transition_path)?.rows;
        for row in &transition_rows {
            let layer = cell(row, "layer").trim().to_owned();
            let anchor_class = cell(row, "anchor_feature_class").trim().to_owned();
            let response_class = cell(row, "response_feature_class").trim().to_owned();
            if layer != "candidate" && layer != "quality" {
                return Err(format!("invalid transition layer: {layer:?}"));
            }
            if !source_classes.contains(&anchor_class) || !source_classes.contains(&response_class)
            {
                return Err(format!(
                    "transition feature class is not present in source summary: anchor={anchor_class:?}, response={response_class:?}"
                ));
            }
            let key = (layer.clone(), anchor_class.clone(), response_class.clone());
            let transition_context = format!("({layer}, {anchor_class}, {response_class})");
            let transition_frame_count =
                required_int(cell(row, "frame_count"), "frame_count", &transition_context)?;
            if transition_frame_count != frame_count {
                return Err(format!(
                    "transition frame_count={transition_frame_count} does not match persistence frame_count={frame_count}"
                ));
            }
            let expected_anchor_count = if layer == "candidate" {
                source_anchor_counts[&anchor_class].0
            } else {
                source_anchor_counts[&anchor_class].1
            };
            let anchor_count = required_int(
                cell(row, "anchor_count"),
                "anchor_count",
                &transition_context,
            )?;
            let possible = required_int(
                cell(row, "possible_match_count"),
                "possible_match_count",
                &transition_context,
            )?;
            let matched = required_int(
                cell(row, "matched_count"),
                "matched_count",
                &transition_context,
            )?;
            if anchor_count != expected_anchor_count {
                return Err(format!(
                    "transition anchor_count={anchor_count} does not match source summary {layer} anchor_count={expected_anchor_count} for {anchor_class}"
                ));
            }
            let expected_possible = anchor_count
                .checked_mul(frame_count)
                .ok_or("transition denominator overflow")?;
            if possible != expected_possible {
                return Err(format!(
                    "transition possible_match_count={possible} does not equal anchor_count*frame_count={expected_possible} for {transition_context}"
                ));
            }
            if matched < 0 || matched > possible {
                return Err(format!(
                    "transition matched_count is outside its denominator for {transition_context}"
                ));
            }
            let response_label = {
                let label = cell(row, "response_feature_class_label").trim();
                if label.is_empty() {
                    response_class.clone()
                } else {
                    label.to_owned()
                }
            };
            if transition_metrics
                .insert(
                    key.clone(),
                    TransitionMetric {
                        matched,
                        possible,
                        response_label,
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate class transition row: {key:?}"));
            }
        }

        let mut expected_keys = BTreeSet::new();
        for layer in ["candidate", "quality"] {
            for anchor in &source_classes {
                for response in &source_classes {
                    expected_keys.insert((layer.to_owned(), anchor.clone(), response.clone()));
                }
            }
        }
        let actual_keys: BTreeSet<_> = transition_metrics.keys().cloned().collect();
        if expected_keys != actual_keys {
            let missing: Vec<_> = expected_keys
                .difference(&actual_keys)
                .take(4)
                .cloned()
                .collect();
            let extra: Vec<_> = actual_keys
                .difference(&expected_keys)
                .take(4)
                .cloned()
                .collect();
            return Err(format!(
                "class transition matrix is incomplete or has unexpected rows: missing={missing:?}, extra={extra:?}"
            ));
        }
        for layer in ["candidate", "quality"] {
            for anchor_class in &source_classes {
                let anchor_index = if layer == "candidate" { 0 } else { 1 };
                let anchor_count = if anchor_index == 0 {
                    source_anchor_counts[anchor_class].0
                } else {
                    source_anchor_counts[anchor_class].1
                };
                let possible = anchor_count
                    .checked_mul(frame_count)
                    .ok_or("transition denominator overflow")?;
                let matched: i64 = source_classes
                    .iter()
                    .map(|response_class| {
                        transition_metrics[&(
                            layer.to_owned(),
                            anchor_class.clone(),
                            response_class.clone(),
                        )]
                            .matched
                    })
                    .sum();
                if matched > possible {
                    return Err(format!(
                        "class transition rows exceed one-to-one denominator for layer={layer}, anchor={anchor_class}"
                    ));
                }
            }
        }
    }

    let mut rows = Vec::new();
    for source_row in &source_rows {
        let feature_class = cell(source_row, "feature_class").trim().to_owned();
        let persistence_row = &persistence_by_class[&feature_class];
        let source_label = cell(source_row, "feature_class_label").trim();
        let label = if source_label.is_empty() {
            let persistence_label = cell(persistence_row, "feature_class_label").trim();
            if persistence_label.is_empty() {
                feature_class.clone()
            } else {
                persistence_label.to_owned()
            }
        } else {
            source_label.to_owned()
        };
        let anchor_candidates = source_anchor_counts[&feature_class].0;
        let anchor_quality = source_anchor_counts[&feature_class].1;
        let candidate_presence = required_int(
            cell(persistence_row, "candidate_presence_ge_required_count"),
            "candidate_presence_ge_required_count",
            &feature_class,
        )?;
        let quality_presence = required_int(
            cell(persistence_row, "quality_presence_ge_required_count"),
            "quality_presence_ge_required_count",
            &feature_class,
        )?;
        if [
            anchor_candidates,
            anchor_quality,
            candidate_presence,
            quality_presence,
        ]
        .into_iter()
        .any(|count| count < 0)
        {
            return Err(format!("negative count in feature class: {feature_class}"));
        }
        if anchor_quality > anchor_candidates {
            return Err(format!(
                "quality_count exceeds candidate_count: {feature_class}"
            ));
        }
        if candidate_presence > anchor_candidates || quality_presence > anchor_candidates {
            return Err(format!(
                "persistence count exceeds anchor count: {feature_class}"
            ));
        }
        let candidate_fraction =
            (anchor_candidates > 0).then_some(candidate_presence as f64 / anchor_candidates as f64);
        let quality_fraction =
            (anchor_candidates > 0).then_some(quality_presence as f64 / anchor_candidates as f64);
        let persistence_gap = candidate_fraction.zip(quality_fraction).map(|(c, q)| c - q);
        let median_flux_snr = optional_float(
            cell(source_row, "median_flux_snr"),
            "median_flux_snr",
            &feature_class,
        )?;
        let max_flux_snr = optional_float(
            cell(source_row, "max_flux_snr"),
            "max_flux_snr",
            &feature_class,
        )?;
        let high_snr_rejected =
            max_flux_snr.is_some_and(|value| value >= high_snr_threshold) && anchor_quality == 0;

        let mut candidate_any_response_fraction = None;
        let mut candidate_same_class_response_fraction = None;
        let mut quality_any_response_fraction = None;
        let mut quality_same_class_response_fraction = None;
        let mut dominant_cross_class = None;
        let mut dominant_cross_class_label = None;
        if class_transition_path.is_some() {
            let candidate_possible = transition_metrics[&(
                "candidate".to_owned(),
                feature_class.clone(),
                source_classes.first().cloned().unwrap_or_default(),
            )]
                .possible;
            if candidate_possible > 0 {
                let matched: i64 = source_classes
                    .iter()
                    .map(|response| {
                        transition_metrics[&(
                            "candidate".to_owned(),
                            feature_class.clone(),
                            response.clone(),
                        )]
                            .matched
                    })
                    .sum();
                candidate_any_response_fraction = Some(matched as f64 / candidate_possible as f64);
            }
            candidate_same_class_response_fraction = transition_fraction(
                &transition_metrics,
                "candidate",
                &feature_class,
                &feature_class,
            );
            let quality_possible = transition_metrics[&(
                "quality".to_owned(),
                feature_class.clone(),
                source_classes.first().cloned().unwrap_or_default(),
            )]
                .possible;
            if quality_possible > 0 {
                let matched: i64 = source_classes
                    .iter()
                    .map(|response| {
                        transition_metrics[&(
                            "quality".to_owned(),
                            feature_class.clone(),
                            response.clone(),
                        )]
                            .matched
                    })
                    .sum();
                quality_any_response_fraction = Some(matched as f64 / quality_possible as f64);
            }
            quality_same_class_response_fraction = transition_fraction(
                &transition_metrics,
                "quality",
                &feature_class,
                &feature_class,
            );
            let dominant = source_classes
                .iter()
                .filter(|response| *response != &feature_class)
                .map(|response| {
                    let metric = &transition_metrics[&(
                        "candidate".to_owned(),
                        feature_class.clone(),
                        response.clone(),
                    )];
                    (
                        metric.matched,
                        response.clone(),
                        metric.response_label.clone(),
                    )
                })
                .max_by(|left, right| (left.0, &left.1).cmp(&(right.0, &right.1)));
            if let Some((_matched, class_name, class_label)) = dominant.filter(|row| row.0 > 0) {
                dominant_cross_class = Some(class_name);
                dominant_cross_class_label = Some(class_label);
            }
        }

        let interpretation =
            if feature_class == "compact_quality" && anchor_quality == anchor_candidates {
                "当前形态与质量规则下稳定；仍需星表或注入确认物理身份"
            } else if feature_class == "crowded_blend" {
                "近邻/去混叠证据不足；低质量持久性不能等同于伪影"
            } else if high_snr_rejected
                && candidate_fraction.is_some_and(|value| value >= 0.7)
                && quality_fraction.is_some_and(|value| value <= 0.1)
            {
                "高显著且候选层可重复，但质量层拒绝；不能计为恒星"
            } else if high_snr_rejected {
                "高显著但质量层拒绝；优先复核形态、范围或线状结构"
            } else if candidate_fraction.is_some_and(|value| value >= 0.7)
                && quality_fraction.is_some_and(|value| value <= 0.1)
            {
                "候选层重复而质量层不重复；可能是弱源或固定结构"
            } else {
                "需结合原始像素、PSF、跨帧和星表证据"
            }
            .to_owned();

        rows.push(FeatureCrossAuditRow {
            feature_class,
            feature_class_label: label,
            anchor_candidate_count: anchor_candidates,
            anchor_quality_count: anchor_quality,
            median_flux_snr,
            max_flux_snr,
            required_presence,
            frame_count,
            candidate_presence_ge_required_count: candidate_presence,
            candidate_presence_fraction: candidate_fraction,
            quality_presence_ge_required_count: quality_presence,
            quality_presence_fraction: quality_fraction,
            persistence_gap,
            high_snr_rejected,
            interpretation,
            candidate_any_response_fraction,
            candidate_same_class_response_fraction,
            quality_any_response_fraction,
            quality_same_class_response_fraction,
            dominant_cross_class,
            dominant_cross_class_label,
        });
    }

    let high_snr_classes: Vec<_> = rows
        .iter()
        .filter(|row| row.high_snr_rejected)
        .map(|row| row.feature_class.as_str())
        .collect();
    let large_gap_classes: Vec<_> = rows
        .iter()
        .filter(|row| row.persistence_gap.is_some_and(|gap| gap >= 0.5))
        .map(|row| row.feature_class.as_str())
        .collect();
    let high_snr_text = if high_snr_classes.is_empty() {
        "无".to_owned()
    } else {
        high_snr_classes.join(",")
    };
    let large_gap_text = if large_gap_classes.is_empty() {
        "无".to_owned()
    } else {
        large_gap_classes.join(",")
    };
    let mut conclusion = format!(
        "交叉审计连接 {} 个特征类别；高 SNR 但质量层拒绝的类别为 {}，候选-质量持久性差距至少 0.5 的类别为 {}。SNR、候选持久性和质量持久性分别描述显著性、稳定性和点源约束，不能替代星表/WCS 身份确认。",
        rows.len(),
        high_snr_text,
        large_gap_text,
    );
    if class_transition_path.is_some() {
        conclusion.push_str(
            " 已连接类别条件转移；任意类别响应与同类响应被分别报告，仍不代表物理类别变化或逐星身份。",
        );
    }

    Ok(FeatureCrossAuditResult {
        source_summary_path: source_path.to_string_lossy().into_owned(),
        persistence_path: persistence_file.to_string_lossy().into_owned(),
        frame_count,
        required_presence,
        high_snr_threshold,
        rows,
        conclusion,
        class_transition_path: class_transition_path
            .map(|path| path.to_string_lossy().into_owned()),
    })
}

fn csv_value<T: ToString>(value: T) -> String {
    value.to_string()
}

fn optional_csv(value: Option<f64>) -> String {
    value.map(python_float).unwrap_or_default()
}

fn python_float(value: f64) -> String {
    let mut text = format!("{value:?}");
    if let Some((mantissa, exponent)) = text.split_once('e') {
        let parsed = exponent.parse::<i32>().unwrap_or(0);
        text = format!(
            "{mantissa}e{}{:02}",
            if parsed >= 0 { "+" } else { "-" },
            parsed.abs()
        );
    }
    text
}

fn row_csv(row: &FeatureCrossAuditRow) -> Vec<String> {
    vec![
        row.feature_class.clone(),
        row.feature_class_label.clone(),
        csv_value(row.anchor_candidate_count),
        csv_value(row.anchor_quality_count),
        optional_csv(row.median_flux_snr),
        optional_csv(row.max_flux_snr),
        csv_value(row.required_presence),
        csv_value(row.frame_count),
        csv_value(row.candidate_presence_ge_required_count),
        optional_csv(row.candidate_presence_fraction),
        csv_value(row.quality_presence_ge_required_count),
        optional_csv(row.quality_presence_fraction),
        optional_csv(row.persistence_gap),
        if row.high_snr_rejected {
            "True"
        } else {
            "False"
        }
        .into(),
        row.interpretation.clone(),
        optional_csv(row.candidate_any_response_fraction),
        optional_csv(row.candidate_same_class_response_fraction),
        optional_csv(row.quality_any_response_fraction),
        optional_csv(row.quality_same_class_response_fraction),
        row.dominant_cross_class.clone().unwrap_or_default(),
        row.dominant_cross_class_label.clone().unwrap_or_default(),
    ]
}

fn write_csv(path: &Path, rows: &[FeatureCrossAuditRow]) -> Result<()> {
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(CSV_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row_csv(row))
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(bytes.len() + 3);
    output.extend_from_slice(&[0xef, 0xbb, 0xbf]);
    output.extend_from_slice(&bytes);
    fs::write(path, output).map_err(|error| error.to_string())
}

const CHART_WIDTH: usize = 850;
const CHART_HEIGHT: usize = 500;

fn pixel(pixels: &mut [u8], x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= CHART_WIDTH as i32 || y >= CHART_HEIGHT as i32 {
        return;
    }
    let offset = (y as usize * CHART_WIDTH + x as usize) * 3;
    pixels[offset..offset + 3].copy_from_slice(&color);
}

fn line(pixels: &mut [u8], start: (i32, i32), end: (i32, i32), color: [u8; 3]) {
    let (mut x0, mut y0) = start;
    let (x1, y1) = end;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        pixel(pixels, x0, y0, color);
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

fn write_chart(path: &Path, rows: &[FeatureCrossAuditRow]) -> Result<()> {
    let mut pixels = vec![255u8; CHART_WIDTH * CHART_HEIGHT * 3];
    let left = 68;
    let right = CHART_WIDTH as i32 - 28;
    let top = 28;
    let bottom = CHART_HEIGHT as i32 - 62;
    let axis = [65, 72, 85];
    let grid = [225, 229, 235];
    for tick in 0..=5 {
        let y = bottom - ((bottom - top) * tick) / 5;
        line(&mut pixels, (left, y), (right, y), grid);
    }
    line(&mut pixels, (left, top), (left, bottom), axis);
    line(&mut pixels, (left, bottom), (right, bottom), axis);
    let x_for = |index: usize| {
        if rows.len() <= 1 {
            (left + right) / 2
        } else {
            left + ((right - left) as usize * index / (rows.len() - 1)) as i32
        }
    };
    let y_for = |value: Option<f64>| {
        let scaled = value.unwrap_or(0.0).clamp(0.0, 1.0);
        bottom - ((bottom - top) as f64 * scaled).round() as i32
    };
    for (values, color) in [
        (
            rows.iter()
                .map(|row| row.candidate_presence_fraction)
                .collect::<Vec<_>>(),
            [215, 148, 50],
        ),
        (
            rows.iter()
                .map(|row| row.quality_presence_fraction)
                .collect::<Vec<_>>(),
            [79, 155, 131],
        ),
    ] {
        for index in 1..values.len() {
            line(
                &mut pixels,
                (x_for(index - 1), y_for(values[index - 1])),
                (x_for(index), y_for(values[index])),
                color,
            );
        }
        for (index, value) in values.iter().enumerate() {
            let (cx, cy) = (x_for(index), y_for(*value));
            for dy in -3..=3 {
                for dx in -3..=3 {
                    if dx * dx + dy * dy <= 9 {
                        pixel(&mut pixels, cx + dx, cy + dy, color);
                    }
                }
            }
        }
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = png::Encoder::new(file, CHART_WIDTH as u32, CHART_HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .map_err(|error| error.to_string())?
        .write_image_data(&pixels)
        .map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &FeatureCrossAuditResult, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = fs::canonicalize(out_dir).map_err(|error| error.to_string())?;
    write_csv(&output.join("feature_cross_audit.csv"), &result.rows)?;
    let mut payload = result.as_dict();
    payload["note"] = json!(
        "该报告只连接单帧特征汇总和 15 帧响应持久性；质量邻域持久性不是特征类别的真阳性，高 SNR 反例也不能替代星表/WCS 身份核验。"
    );
    if result.class_transition_path.is_some() {
        payload["class_transition_note"] = json!(
            "类别转移字段来自注册坐标中的互相最近一对一响应；任意类别响应和同类响应是检测器稳定性描述，不是物理类别变化或逐星身份。"
        );
    }
    fs::write(
        output.join("feature_cross_audit.json"),
        serde_json::to_vec_pretty(&payload).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    write_chart(&output.join("feature_cross_audit.png"), &result.rows)?;
    Ok(output)
}

#[derive(Debug)]
struct CliOptions {
    source_summary: PathBuf,
    persistence: PathBuf,
    transition: Option<PathBuf>,
    high_snr_threshold: f64,
    out_dir: PathBuf,
}

fn parse_args(args: &[String]) -> Result<Option<CliOptions>> {
    let mut positional = Vec::new();
    let mut transition = None;
    let mut high_snr_threshold = 10.0;
    let mut out_dir = PathBuf::from("tmp/feature-cross-audit");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--help" || arg == "-h" {
            println!("Usage: rst19-feature-cross SOURCE_SUMMARY.csv PERSISTENCE.csv [OPTIONS]\n\n--class-transition-csv PATH --high-snr-threshold N --out-dir PATH");
            return Ok(None);
        }
        let (name, inline) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        match name {
            "--class-transition-csv" | "--high-snr-threshold" | "--out-dir" => {
                let value = if let Some(value) = inline {
                    value
                } else {
                    index += 1;
                    args.get(index).ok_or("option requires a value")?
                };
                match name {
                    "--class-transition-csv" => transition = Some(PathBuf::from(value)),
                    "--high-snr-threshold" => {
                        high_snr_threshold = value
                            .parse::<f64>()
                            .map_err(|error| format!("invalid high_snr_threshold: {error}"))?
                    }
                    _ => out_dir = PathBuf::from(value),
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => positional.push(PathBuf::from(name)),
        }
        index += 1;
    }
    if positional.len() != 2 {
        return Err("expected source summary and persistence CSV paths".into());
    }
    Ok(Some(CliOptions {
        source_summary: positional.remove(0),
        persistence: positional.remove(0),
        transition,
        high_snr_threshold,
        out_dir,
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    let Some(options) = parse_args(args)? else {
        return Ok(());
    };
    let result = run_feature_cross_audit(
        &options.source_summary,
        &options.persistence,
        options.high_snr_threshold,
        options.transition.as_deref(),
    )?;
    let output = write_artifacts(&result, &options.out_dir)?;
    println!("rst19-feature-cross: 审计已写入 {}", output.display());
    println!(
        "classes={} frames={} required_presence={} high_snr_threshold={}",
        result.rows.len(),
        result.frame_count,
        result.required_presence,
        result.high_snr_threshold
    );
    for row in &result.rows {
        let candidate = row
            .candidate_presence_fraction
            .map(|value| format!("{:.1}%", value * 100.0))
            .unwrap_or_else(|| "-".into());
        let quality = row
            .quality_presence_fraction
            .map(|value| format!("{:.1}%", value * 100.0))
            .unwrap_or_else(|| "-".into());
        let high_snr_rejected = if row.high_snr_rejected {
            "True"
        } else {
            "False"
        };
        println!(
            "{}: candidate={} quality={} high_snr_rejected={}",
            row.feature_class, candidate, quality, high_snr_rejected
        );
    }
    println!("{}", result.conclusion);
    Ok(())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let Some(options) = parse_args(args)? else {
        return Ok(json!({"tool":"rst19-feature-cross", "help":true}));
    };
    let result = run_feature_cross_audit(
        &options.source_summary,
        &options.persistence,
        options.high_snr_threshold,
        options.transition.as_deref(),
    )?;
    let output = write_artifacts(&result, &options.out_dir)?;
    Ok(json!({"tool":"rst19-feature-cross", "out_dir":output,
        "result_path":output.join("feature_cross_audit.json"), "result":result.as_dict()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "rst19-feature-cross-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn cross_audit_matches_python_counterexample_metrics_and_artifacts() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.csv");
        let persistence = root.join("persistence.csv");
        let transitions = root.join("transitions.csv");
        fs::write(
            &source,
            "\u{feff}feature_class,feature_class_label,candidate_count,quality_count,median_flux_snr,max_flux_snr\r\ncompact_quality,紧凑质量候选,10,10,12.3,100.0\r\nspike_or_support,尖峰/PSF支持不足,10,0,1.9,133.8\r\n",
        )
        .unwrap();
        fs::write(
            &persistence,
            "feature_class,feature_class_label,frame_count,required_presence,candidate_presence_ge_required_count,quality_presence_ge_required_count\r\ncompact_quality,紧凑质量候选,15,12,9,8\r\nspike_or_support,尖峰/PSF支持不足,15,12,8,0\r\n",
        )
        .unwrap();
        fs::write(
            &transitions,
            "layer,anchor_feature_class,anchor_feature_class_label,response_feature_class,response_feature_class_label,anchor_count,frame_count,possible_match_count,matched_count\r\ncandidate,compact_quality,紧凑质量候选,compact_quality,紧凑质量候选,10,15,150,9\r\ncandidate,compact_quality,紧凑质量候选,spike_or_support,尖峰/PSF支持不足,10,15,150,3\r\ncandidate,spike_or_support,尖峰/PSF支持不足,compact_quality,紧凑质量候选,10,15,150,2\r\ncandidate,spike_or_support,尖峰/PSF支持不足,spike_or_support,尖峰/PSF支持不足,10,15,150,6\r\nquality,compact_quality,紧凑质量候选,compact_quality,紧凑质量候选,10,15,150,8\r\nquality,compact_quality,紧凑质量候选,spike_or_support,尖峰/PSF支持不足,10,15,150,0\r\nquality,spike_or_support,尖峰/PSF支持不足,compact_quality,紧凑质量候选,0,15,0,0\r\nquality,spike_or_support,尖峰/PSF支持不足,spike_or_support,尖峰/PSF支持不足,0,15,0,0\r\n",
        )
        .unwrap();

        let result =
            run_feature_cross_audit(&source, &persistence, 10.0, Some(&transitions)).unwrap();
        assert_eq!(result.frame_count, 15);
        assert_eq!(result.required_presence, 12);
        assert_eq!(result.rows.len(), 2);
        let compact = &result.rows[0];
        assert_eq!(compact.candidate_presence_fraction, Some(0.9));
        assert_eq!(compact.quality_presence_fraction, Some(0.8));
        assert_eq!(compact.candidate_any_response_fraction, Some(0.08));
        assert_eq!(compact.candidate_same_class_response_fraction, Some(0.06));
        assert_eq!(
            compact.quality_same_class_response_fraction,
            Some(8.0 / 150.0)
        );
        assert_eq!(
            compact.dominant_cross_class.as_deref(),
            Some("spike_or_support")
        );
        let spike = &result.rows[1];
        assert!(spike.high_snr_rejected);
        assert_eq!(spike.persistence_gap, Some(0.8));
        assert!(result.conclusion.contains("spike_or_support"));

        let output = write_artifacts(&result, &root.join("out")).unwrap();
        let csv_bytes = fs::read(output.join("feature_cross_audit.csv")).unwrap();
        assert_eq!(&csv_bytes[..3], &[0xef, 0xbb, 0xbf]);
        assert!(
            String::from_utf8_lossy(&csv_bytes).contains("candidate_same_class_response_fraction")
        );
        assert!(fs::read(output.join("feature_cross_audit.png"))
            .unwrap()
            .starts_with(b"\x89PNG\r\n\x1a\n"));
        let json: Value =
            serde_json::from_slice(&fs::read(output.join("feature_cross_audit.json")).unwrap())
                .unwrap();
        assert_eq!(json["rows"][1]["high_snr_rejected"], true);
        assert!(json["note"]
            .as_str()
            .unwrap()
            .contains("不是特征类别的真阳性"));
        assert!(json.get("class_transition_note").is_some());

        let desktop_out = root.join("desktop-out");
        let desktop_args = vec![
            source.to_string_lossy().into_owned(),
            persistence.to_string_lossy().into_owned(),
            "--class-transition-csv".into(),
            transitions.to_string_lossy().into_owned(),
            "--out-dir".into(),
            desktop_out.to_string_lossy().into_owned(),
        ];
        let desktop = run_desktop(&desktop_args).unwrap();
        assert_eq!(desktop["tool"], "rst19-feature-cross");
        assert_eq!(
            desktop["result"]["rows"][0]["candidate_any_response_fraction"],
            0.08
        );
        assert!(desktop_out.join("feature_cross_audit.png").is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn transition_matrix_rejects_incomplete_or_inconsistent_inputs() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.csv");
        let persistence = root.join("persistence.csv");
        let transitions = root.join("transitions.csv");
        fs::write(
            &source,
            "feature_class,candidate_count,quality_count\na,1,1\n",
        )
        .unwrap();
        fs::write(&persistence, "feature_class,frame_count,required_presence,candidate_presence_ge_required_count,quality_presence_ge_required_count\na,2,2,1,1\n").unwrap();
        fs::write(&transitions, "layer,anchor_feature_class,response_feature_class,frame_count,anchor_count,possible_match_count,matched_count\ncandidate,a,a,2,1,2,1\n").unwrap();
        let error =
            run_feature_cross_audit(&source, &persistence, 10.0, Some(&transitions)).unwrap_err();
        assert!(error.contains("matrix is incomplete"));
        let _ = fs::remove_dir_all(root);
    }
}
