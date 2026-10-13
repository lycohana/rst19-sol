//! Compare low-order photometric color models over the same matched source set.
//!
//! This is a diagnostic of model sensitivity, not a probability or a claim that
//! catalog magnitudes equal a camera's unknown standard response.

use crate::{
    catalog::CatalogSource,
    matching::CatalogMatch,
    measurement::Detection,
    photometric_calibration::{self, CalibrationOptions, PhotometricCalibration},
    photometry, Result,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};

const MODEL_NAMES: [&str; 3] = [
    "zero_point_only",
    "linear_color_term",
    "quadratic_color_term",
];
const MODEL_LABELS: [&str; 3] = ["零阶零点", "一次颜色项", "二次颜色项"];
const VALID_STATUSES: [&str; 2] = ["VALID", "VALID_NO_HOLDOUT"];

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SensitivityOptions {
    pub exposure_s: f64,
    pub photometric_system: Option<String>,
    pub photometric_band: Option<String>,
    pub color_name: Option<String>,
    pub model_orders: Vec<i64>,
    pub min_snr: f64,
    pub min_calibrators: usize,
}

impl Default for SensitivityOptions {
    fn default() -> Self {
        Self {
            exposure_s: 1.0,
            photometric_system: Some("unknown".into()),
            photometric_band: Some("unknown".into()),
            color_name: Some("BP-RP".into()),
            model_orders: vec![0, 1, 2],
            min_snr: 5.0,
            min_calibrators: 6,
        }
    }
}

#[derive(Debug, Clone)]
struct MeasurementRow {
    detection_id: i64,
    source_id: String,
    x: f64,
    y: f64,
    instrumental_magnitude: f64,
    color: Option<f64>,
    flux_snr: f64,
}

#[derive(Debug, Clone)]
struct Variant {
    order: usize,
    calibration: Option<PhotometricCalibration>,
    failure_reason: Option<String>,
}

fn validate(options: &SensitivityOptions) -> Result<Vec<usize>> {
    if !options.exposure_s.is_finite() || options.exposure_s <= 0.0 {
        return Err("exposure_s must be positive".into());
    }
    if !options.min_snr.is_finite() || options.min_snr <= 0.0 {
        return Err("min_snr must be positive".into());
    }
    if options.min_calibrators < 2 {
        return Err("min_calibrators must be an integer >= 2".into());
    }
    if options.model_orders.is_empty() {
        return Err("model_orders must contain only 0, 1 or 2".into());
    }
    let mut seen = HashSet::new();
    let mut orders = Vec::new();
    for &order in &options.model_orders {
        if !(0..=2).contains(&order) {
            return Err("model_orders must contain only 0, 1 or 2".into());
        }
        let order = order as usize;
        if seen.insert(order) {
            orders.push(order);
        }
    }
    if orders.is_empty() {
        return Err("model_orders must contain only 0, 1 or 2".into());
    }
    Ok(orders)
}

fn matched_measurements(
    matches: &[CatalogMatch],
    detections: &[Detection],
    catalog: &[CatalogSource],
    options: &SensitivityOptions,
) -> Result<Vec<MeasurementRow>> {
    let detections_by_id = detections
        .iter()
        .map(|detection| (detection.detection_id, detection))
        .collect::<BTreeMap<_, _>>();
    let catalog_by_id = catalog
        .iter()
        .map(|source| (source.source_id.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    let mut seen_detection_ids = HashSet::new();
    let mut seen_source_ids = HashSet::new();
    let mut rows = Vec::new();

    for matched in matches {
        if seen_detection_ids.contains(&matched.detection_id)
            || seen_source_ids.contains(&matched.source_id)
        {
            continue;
        }
        let (Some(detection), Some(source)) = (
            detections_by_id.get(&matched.detection_id),
            catalog_by_id.get(matched.source_id.as_str()),
        ) else {
            continue;
        };
        let signal_snr = detection.flux_snr.unwrap_or(detection.snr);
        if !detection.quality_passed
            || !signal_snr.is_finite()
            || signal_snr < options.min_snr
            || !detection.flux.is_finite()
            || detection.flux <= 0.0
            || detection
                .flags
                .iter()
                .any(|flag| ["EDGE", "MASKED", "SATURATED"].contains(&flag.as_str()))
        {
            continue;
        }
        let Some(instrumental) =
            photometry::instrumental_magnitude(detection.flux, options.exposure_s)?
        else {
            continue;
        };
        let raw_color = source.color.or(matched.catalog_color);
        let color = raw_color.filter(|value| value.is_finite());
        rows.push(MeasurementRow {
            detection_id: detection.detection_id,
            source_id: source.source_id.clone(),
            x: detection.x,
            y: detection.y,
            instrumental_magnitude: instrumental,
            color,
            flux_snr: signal_snr,
        });
        seen_detection_ids.insert(matched.detection_id);
        seen_source_ids.insert(matched.source_id.clone());
    }
    rows.sort_by(|left, right| {
        left.detection_id
            .cmp(&right.detection_id)
            .then_with(|| left.source_id.cmp(&right.source_id))
    });
    Ok(rows)
}

fn fit_variants(
    matches: &[CatalogMatch],
    detections: &[Detection],
    catalog: &[CatalogSource],
    options: &SensitivityOptions,
    orders: &[usize],
) -> Vec<Variant> {
    orders
        .iter()
        .copied()
        .map(|order| {
            let fit_options = CalibrationOptions {
                exposure_s: options.exposure_s,
                photometric_system: options.photometric_system.clone(),
                photometric_band: options.photometric_band.clone(),
                color_name: options.color_name.clone(),
                color_order: order,
                min_snr: options.min_snr,
                min_calibrators: options.min_calibrators.max(order + 2),
                ..CalibrationOptions::default()
            };
            match photometric_calibration::fit(matches, detections, catalog, &fit_options) {
                Ok(calibration) => Variant {
                    order,
                    calibration: Some(calibration),
                    failure_reason: None,
                },
                Err(error) => Variant {
                    order,
                    calibration: None,
                    failure_reason: Some(error),
                },
            }
        })
        .collect()
}

fn model_dict(variant: &Variant) -> Value {
    let order = variant.order;
    let name = MODEL_NAMES[order];
    let label = MODEL_LABELS[order];
    let Some(calibration) = &variant.calibration else {
        return json!({
            "name":name,
            "label":label,
            "color_order":order,
            "status":"ERROR",
            "coefficients":[],
            "calibrator_count":0,
            "inlier_count":0,
            "validation_count":0,
            "fit_rms_mag":Value::Null,
            "validation_rms_mag":Value::Null,
            "residual_mad_mag":Value::Null,
            "color_min":Value::Null,
            "color_max":Value::Null,
            "flags":["MODEL_FIT_ERROR"],
            "usable":false,
            "failure_reason":variant.failure_reason.as_deref().unwrap_or("unknown model fit error"),
        });
    };
    let usable = VALID_STATUSES.contains(&calibration.status.as_str());
    let failure_reason = if !usable {
        Some(if calibration.flags.is_empty() {
            calibration.status.clone()
        } else {
            calibration.flags.join(";")
        })
    } else {
        None
    };
    let mut value = json!({
        "name":name,
        "label":label,
        "color_order":order,
        "status":calibration.status,
        "coefficients":calibration.coefficients,
        "calibrator_count":calibration.calibrator_count,
        "inlier_count":calibration.inlier_count,
        "validation_count":calibration.validation_count,
        "fit_rms_mag":calibration.fit_rms_mag,
        "validation_rms_mag":calibration.validation_rms_mag,
        "residual_mad_mag":calibration.residual_mad_mag,
        "color_min":calibration.color_min,
        "color_max":calibration.color_max,
        "flags":calibration.flags,
        "usable":usable,
    });
    if let Some(reason) = failure_reason {
        value["failure_reason"] = json!(reason);
    }
    value
}

fn faintest_rank(rows: &[Value], model_name: &str) -> Option<Value> {
    let mut eligible = rows
        .iter()
        .filter_map(|row| {
            let magnitude = row.get("models")?.get(model_name)?.as_f64()?;
            let detection_id = row.get("detection_id")?.as_i64()?;
            Some((magnitude, detection_id, row))
        })
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return None;
    }
    eligible.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| right.1.cmp(&left.1))
    });
    let (magnitude, _, winner) = eligible[eligible.len() - 1];
    Some(json!({
        "model":model_name,
        "detection_id":winner["detection_id"],
        "source_id":winner["source_id"],
        "magnitude":magnitude,
        "eligible_count":eligible.len(),
    }))
}

pub fn run(
    matches: &[CatalogMatch],
    detections: &[Detection],
    catalog: &[CatalogSource],
    options: &SensitivityOptions,
) -> Result<Value> {
    let orders = validate(options)?;
    let variants = fit_variants(matches, detections, catalog, options, &orders);
    let measurements = matched_measurements(matches, detections, catalog, options)?;
    let mut sources = Vec::with_capacity(measurements.len());
    for row in &measurements {
        let mut model_values = Map::new();
        let mut numeric = Vec::new();
        for variant in &variants {
            let Some(calibration) = &variant.calibration else {
                continue;
            };
            let Some(magnitude) = calibration.apply(row.instrumental_magnitude, row.color) else {
                continue;
            };
            let name = MODEL_NAMES[variant.order];
            model_values.insert(name.into(), json!(magnitude));
            numeric.push(magnitude);
        }
        numeric.sort_by(f64::total_cmp);
        let median = if numeric.is_empty() {
            None
        } else if numeric.len() % 2 == 1 {
            Some(numeric[numeric.len() / 2])
        } else {
            Some(0.5 * (numeric[numeric.len() / 2 - 1] + numeric[numeric.len() / 2]))
        };
        let min = numeric.first().copied();
        let max = numeric.last().copied();
        sources.push(json!({
            "detection_id":row.detection_id,
            "source_id":row.source_id,
            "x":row.x,
            "y":row.y,
            "instrumental_magnitude":row.instrumental_magnitude,
            "color":row.color,
            "flux_snr":row.flux_snr,
            "models":Value::Object(model_values),
            "valid_model_count":numeric.len(),
            "model_min_mag":min,
            "model_max_mag":max,
            "model_median_mag":median,
            "model_spread_mag":if numeric.len() >= 2 { max.zip(min).map(|(max,min)| max-min) } else { None },
        }));
    }
    let rankings = variants
        .iter()
        .filter_map(|variant| {
            let calibration = variant.calibration.as_ref()?;
            if !VALID_STATUSES.contains(&calibration.status.as_str()) {
                return None;
            }
            faintest_rank(&sources, MODEL_NAMES[variant.order])
        })
        .collect::<Vec<_>>();
    let mut winner_counts = BTreeMap::<String, usize>::new();
    for rank in &rankings {
        if let Some(source_id) = rank["source_id"].as_str() {
            *winner_counts.entry(source_id.to_owned()).or_default() += 1;
        }
    }
    let winner = winner_counts
        .iter()
        .max_by(|left, right| left.1.cmp(right.1).then_with(|| right.0.cmp(left.0)));
    let winner_source_id = winner.map(|(source_id, _)| source_id.clone());
    let winner_count = winner.map_or(0, |(_, count)| *count);
    let valid_model_count = rankings.len();
    let agreement_fraction =
        (valid_model_count > 0).then_some(winner_count as f64 / valid_model_count as f64);
    let models = variants.iter().map(model_dict).collect::<Vec<_>>();
    let usable_count = models
        .iter()
        .filter(|model| model["usable"].as_bool() == Some(true))
        .count();
    let status =
        if usable_count >= 2 && agreement_fraction.is_some_and(|fraction| fraction >= 2.0 / 3.0) {
            "MODEL_STABLE"
        } else if usable_count >= 2 {
            "MODEL_SENSITIVE"
        } else {
            "DIAGNOSTIC_ONLY"
        };
    Ok(json!({
        "schema_version":1,
        "status":status,
        "method":"low_order_color_model_sensitivity",
        "photometric_system":options.photometric_system,
        "photometric_band":options.photometric_band,
        "color_name":options.color_name,
        "models":models,
        "model_count_requested":orders.len(),
        "model_count_usable":usable_count,
        "matched_measurement_count":measurements.len(),
        "source_count_with_any_model":sources.iter().filter(|source| source["valid_model_count"].as_u64().unwrap_or(0)>0).count(),
        "sources":sources,
        "faintest_ranking":{
            "per_model":rankings,
            "winner_source_id":winner_source_id,
            "winner_count":winner_count,
            "valid_model_count":valid_model_count,
            "winner_agreement_fraction":agreement_fraction,
            "interpretation":"有效低阶颜色模型中最暗候选的确定性一致比例；不是后验概率、完备率或真星概率。",
        },
        "claim_boundary":{
            "can_claim":[
                "当前匹配参考星和颜色范围内的低阶光度模型敏感性",
                "模型阶数改变时逐源 m_cal 的范围",
                "最暗候选排序在有效模型之间是否一致",
            ],
            "cannot_claim":[
                "Gaia G 等于开运一号真实标准波段星等",
                "未知相机响应曲线已被恢复",
                "模型一致比例是统计置信度或真星概率",
                "没有距离和消光时的绝对星等 M",
            ],
        },
        "notes":[
            "零阶/一次/二次模型共享同一批检测、匹配和质量门控，但可用参考星数会因颜色缺失而不同。",
            "该实验量化颜色模型误差，不替代实验室响应曲线、标准星观测或 Gaia XP 合成测光。",
        ],
    }))
}
