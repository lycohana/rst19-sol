//! Source-class and spatial empirical-PSF audit summaries.
//!
//! These diagnostics describe compatibility with an empirical image kernel.
//! They do not alter the detector's quality decisions or identify physical
//! stars.

use crate::{
    detection::DetectionResult,
    empirical_psf::{self, EmpiricalPsf},
    fits,
    measurement::Detection,
    models::{FitsFrame, NumericPlane},
    source_classification::{self, SOURCE_FEATURE_CLASSES},
    Result,
};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

const SUPPORT_RADIUS: usize = 7;
const MAX_TEMPLATE_SOURCES: usize = 64;
const CORRELATION_THRESHOLD: f64 = 0.8;

pub struct SourcePsfTables {
    pub feature_rows: Vec<Map<String, Value>>,
    pub spatial_rows: Vec<Map<String, Value>>,
}

fn signal_snr(source: &Detection) -> f64 {
    match source.flux_snr {
        Some(value) if value.is_finite() => value,
        Some(_) => f64::NEG_INFINITY,
        None if source.snr.is_finite() => source.snr,
        None => f64::NEG_INFINITY,
    }
}

fn selected_by_class(
    sources: &[Detection],
    per_class: usize,
) -> Vec<(&'static str, &'static str, Vec<&Detection>)> {
    SOURCE_FEATURE_CLASSES
        .iter()
        .map(|&(feature_class, label)| {
            let mut members = sources
                .iter()
                .filter(|source| {
                    source_classification::classify_source_feature(source) == feature_class
                })
                .collect::<Vec<_>>();
            members.sort_by(|left, right| {
                signal_snr(left)
                    .total_cmp(&signal_snr(right))
                    .then_with(|| left.detection_id.cmp(&right.detection_id))
            });
            if members.len() > per_class {
                let total = members.len();
                members = (0..per_class)
                    .map(|index| {
                        let position = if per_class == 1 {
                            0
                        } else {
                            (index as f64 * (total - 1) as f64 / (per_class - 1) as f64)
                                .round_ties_even() as usize
                        };
                        members[position]
                    })
                    .collect();
            }
            (feature_class, label, members)
        })
        .collect()
}

fn quantile(values: &[f64], percentile: f64) -> Option<f64> {
    (!values.is_empty())
        .then(|| crate::statistics::percentile(values, percentile).ok())
        .flatten()
}

fn median(values: &[f64]) -> Option<f64> {
    quantile(values, 50.0)
}

fn value_object(value: Value) -> Map<String, Value> {
    value
        .as_object()
        .expect("source PSF summary is serialized as an object")
        .clone()
}

fn frame_pixels(frame: &FitsFrame) -> Vec<f64> {
    (0..frame.data.len())
        .map(|index| frame.data.value(index))
        .collect()
}

fn make_psf(
    image: NumericPlane<'_>,
    height: usize,
    width: usize,
    pool: &[Detection],
    isolation_sources: &[Detection],
    excluded_ids: &[i64],
) -> Result<Option<EmpiricalPsf>> {
    empirical_psf::estimate(
        image,
        height,
        width,
        pool,
        Some(isolation_sources),
        SUPPORT_RADIUS,
        MAX_TEMPLATE_SOURCES,
        excluded_ids,
    )
}

fn source_cell(
    source: &Detection,
    height: usize,
    width: usize,
    grid_size: usize,
) -> Option<(usize, usize)> {
    if !source.x.is_finite() || !source.y.is_finite() {
        return None;
    }
    let column = (source.x * grid_size as f64 / width as f64).trunc() as isize;
    let row = (source.y * grid_size as f64 / height as f64).trunc() as isize;
    Some((
        row.clamp(0, grid_size as isize - 1) as usize,
        column.clamp(0, grid_size as isize - 1) as usize,
    ))
}

fn local_psf_for_cell(
    cell: (usize, usize),
    frame: &FitsFrame,
    pixels: &[f64],
    sources: &[Detection],
    excluded_ids: &BTreeSet<i64>,
    grid_size: usize,
    cache: &mut BTreeMap<(usize, usize), Option<EmpiricalPsf>>,
) -> Result<Option<EmpiricalPsf>> {
    if let Some(value) = cache.get(&cell) {
        return Ok(value.clone());
    }
    let (row, column) = cell;
    let padding = (2 * SUPPORT_RADIUS + 4).max(18) as f64;
    let x_min = frame.width as f64 * column as f64 / grid_size as f64 - padding;
    let x_max = frame.width as f64 * (column + 1) as f64 / grid_size as f64 + padding;
    let y_min = frame.height as f64 * row as f64 / grid_size as f64 - padding;
    let y_max = frame.height as f64 * (row + 1) as f64 / grid_size as f64 + padding;
    let pool = sources
        .iter()
        .filter(|source| !excluded_ids.contains(&source.detection_id))
        .filter(|source| {
            source.x.is_finite()
                && source.y.is_finite()
                && x_min <= source.x
                && source.x < x_max
                && y_min <= source.y
                && source.y < y_max
        })
        .cloned()
        .collect::<Vec<_>>();
    let template = make_psf(
        NumericPlane::from(pixels),
        frame.height,
        frame.width,
        &pool,
        sources,
        &[],
    )?;
    cache.insert(cell, template.clone());
    Ok(template)
}

/// Build the two source-level PSF summary tables from one immutable frame.
pub fn summarize(
    frame: &FitsFrame,
    detection: &DetectionResult,
    per_class: usize,
    grid_size: usize,
) -> Result<SourcePsfTables> {
    if per_class == 0 || grid_size == 0 {
        return Err("spatial PSF sample count and grid size must be positive".into());
    }
    if frame.width != detection.image_shape[1]
        || frame.height != detection.image_shape[0]
        || frame.width.checked_mul(frame.height) != Some(frame.data.len())
    {
        return Err("source PSF frame and detection shapes do not match".into());
    }
    let pixels = frame_pixels(frame);
    let image = NumericPlane::from(pixels.as_slice());
    let global_psf = make_psf(
        image,
        frame.height,
        frame.width,
        &detection.sources,
        &detection.sources,
        &[],
    )?;
    let samples = selected_by_class(&detection.sources, per_class);
    let aux_mask = fits::auxiliary_mask(frame.height, frame.width)?;
    let mut feature_rows = Vec::with_capacity(SOURCE_FEATURE_CLASSES.len());

    for (feature_class, feature_label, selected) in &samples {
        let mut correlations = Vec::new();
        let mut residuals = Vec::new();
        let mut central_fractions = Vec::new();
        let mut amplitudes = Vec::new();
        for source in selected {
            if let Some(psf) = global_psf.as_ref() {
                if let Some((correlation, residual, central_fraction, amplitude)) =
                    empirical_psf::similarity_metrics_with_mask(
                        image,
                        frame.height,
                        frame.width,
                        source,
                        psf,
                        &aux_mask,
                    )?
                {
                    correlations.push(correlation);
                    residuals.push(residual);
                    central_fractions.push(central_fraction);
                    amplitudes.push(amplitude);
                }
            }
        }
        let threshold_count = correlations
            .iter()
            .filter(|&&value| value >= CORRELATION_THRESHOLD)
            .count();
        feature_rows.push(value_object(json!({
            "feature_class":feature_class,
            "feature_class_label":feature_label,
            "psf_model":if global_psf.is_some() { "empirical" } else { "unavailable" },
            "psf_source_count":global_psf.as_ref().map_or(0, |psf| psf.source_count),
            "correlation_threshold":CORRELATION_THRESHOLD,
            "sample_count":selected.len(),
            "valid_count":correlations.len(),
            "valid_fraction":if selected.is_empty() { None } else { Some(correlations.len() as f64 / selected.len() as f64) },
            "correlation_ge_threshold_count":threshold_count,
            "correlation_ge_threshold_fraction":if correlations.is_empty() { None } else { Some(threshold_count as f64 / correlations.len() as f64) },
            "psf_correlation_p10":quantile(&correlations, 10.0),
            "psf_correlation_median":median(&correlations),
            "psf_correlation_p90":quantile(&correlations, 90.0),
            "relative_residual_p10":quantile(&residuals, 10.0),
            "relative_residual_median":median(&residuals),
            "relative_residual_p90":quantile(&residuals, 90.0),
            "central_energy_fraction_p10":quantile(&central_fractions, 10.0),
            "central_energy_fraction_median":median(&central_fractions),
            "central_energy_fraction_p90":quantile(&central_fractions, 90.0),
            "fitted_amplitude_adu_median":median(&amplitudes),
            "sample_detection_ids":selected.iter().map(|source| source.detection_id.to_string()).collect::<Vec<_>>().join("|"),
        })));
    }

    let selected_ids = samples
        .iter()
        .flat_map(|(_, _, sources)| sources.iter().map(|source| source.detection_id))
        .collect::<BTreeSet<_>>();
    let mut local_cache = BTreeMap::<(usize, usize), Option<EmpiricalPsf>>::new();
    let mut spatial_rows = Vec::with_capacity(SOURCE_FEATURE_CLASSES.len());
    for (feature_class, feature_label, selected) in &samples {
        let mut global_correlations = Vec::new();
        let mut local_correlations = Vec::new();
        let mut global_residuals = Vec::new();
        let mut local_residuals = Vec::new();
        let mut paired_correlation_deltas = Vec::new();
        let mut paired_residual_deltas = Vec::new();
        let mut local_fallback_count = 0usize;
        let mut used_cells = BTreeSet::new();
        let mut sample_ids = Vec::new();
        for source in selected {
            sample_ids.push(source.detection_id.to_string());
            let global_metrics = if let Some(psf) = global_psf.as_ref() {
                empirical_psf::similarity_metrics_with_mask(
                    image,
                    frame.height,
                    frame.width,
                    source,
                    psf,
                    &aux_mask,
                )?
            } else {
                None
            };
            let cell = source_cell(source, frame.height, frame.width, grid_size);
            let local_psf = if let Some(cell) = cell {
                used_cells.insert(cell);
                local_psf_for_cell(
                    cell,
                    frame,
                    &pixels,
                    &detection.sources,
                    &selected_ids,
                    grid_size,
                    &mut local_cache,
                )?
            } else {
                None
            };
            if local_psf.is_none() {
                local_fallback_count += 1;
            }
            let local_metrics = if let Some(psf) = local_psf.as_ref() {
                empirical_psf::similarity_metrics_with_mask(
                    image,
                    frame.height,
                    frame.width,
                    source,
                    psf,
                    &aux_mask,
                )?
            } else {
                None
            };
            if let Some((correlation, residual, _, _)) = global_metrics {
                global_correlations.push(correlation);
                global_residuals.push(residual);
            }
            if let Some((correlation, residual, _, _)) = local_metrics {
                local_correlations.push(correlation);
                local_residuals.push(residual);
            }
            if let (
                Some((global_correlation, global_residual, _, _)),
                Some((local_correlation, local_residual, _, _)),
            ) = (global_metrics, local_metrics)
            {
                paired_correlation_deltas.push(local_correlation - global_correlation);
                paired_residual_deltas.push(local_residual - global_residual);
            }
        }
        let available_local_psfs = used_cells
            .iter()
            .filter_map(|cell| local_cache.get(cell).and_then(Option::as_ref))
            .collect::<Vec<_>>();
        let correlation_improved = paired_correlation_deltas
            .iter()
            .filter(|&&delta| delta > 0.0)
            .count();
        let residual_reduced = paired_residual_deltas
            .iter()
            .filter(|&&delta| delta < 0.0)
            .count();
        spatial_rows.push(value_object(json!({
            "feature_class":feature_class,
            "feature_class_label":feature_label,
            "grid_size":grid_size,
            "global_psf_source_count":global_psf.as_ref().map_or(0, |psf| psf.source_count),
            "local_template_cell_count":used_cells.len(),
            "local_template_available_cell_count":available_local_psfs.len(),
            "local_template_source_count_median":median(&available_local_psfs.iter().map(|psf| psf.source_count as f64).collect::<Vec<_>>()),
            "sample_count":selected.len(),
            "global_valid_count":global_correlations.len(),
            "local_valid_count":local_correlations.len(),
            "local_fallback_count":local_fallback_count,
            "correlation_threshold":CORRELATION_THRESHOLD,
            "global_correlation_median":median(&global_correlations),
            "local_correlation_median":median(&local_correlations),
            "paired_correlation_count":paired_correlation_deltas.len(),
            "local_correlation_improved_count":correlation_improved,
            "local_correlation_improved_fraction":if paired_correlation_deltas.is_empty() { None } else { Some(correlation_improved as f64 / paired_correlation_deltas.len() as f64) },
            "global_residual_median":median(&global_residuals),
            "local_residual_median":median(&local_residuals),
            "paired_residual_count":paired_residual_deltas.len(),
            "local_residual_reduced_count":residual_reduced,
            "local_residual_reduced_fraction":if paired_residual_deltas.is_empty() { None } else { Some(residual_reduced as f64 / paired_residual_deltas.len() as f64) },
            "median_correlation_delta_local_minus_global":median(&paired_correlation_deltas),
            "median_residual_delta_local_minus_global":median(&paired_residual_deltas),
            "sample_detection_ids":sample_ids.join("|"),
        })));
    }

    Ok(SourcePsfTables {
        feature_rows,
        spatial_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{detection::DetectionResult, models::PixelData};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    fn star(id: i64, x: f64, y: f64, snr: f64) -> Detection {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/source_classification_parity.json"
        ))
        .unwrap();
        let mut source: Detection = serde_json::from_value(fixture["sources"][0].clone()).unwrap();
        source.detection_id = id;
        source.x = x;
        source.y = y;
        source.peak_x = Some(x);
        source.peak_y = Some(y);
        source.background = 0.0;
        source.quality_passed = true;
        source.flags.clear();
        source.flux_snr = Some(snr);
        source.snr = snr;
        source.fwhm = Some(3.5);
        source.ellipticity = Some(0.1);
        source
    }

    #[test]
    fn source_psf_summaries_use_deterministic_samples_and_exclude_samples_from_local_templates() {
        let height = 400usize;
        let width = 400usize;
        let sources = vec![
            star(1, 130.0, 130.0, 20.0),
            star(2, 270.0, 130.0, 30.0),
            star(3, 130.0, 270.0, 40.0),
            star(4, 270.0, 270.0, 50.0),
        ];
        let centers = sources
            .iter()
            .map(|source| (source.x, source.y))
            .collect::<Vec<_>>();
        let mut pixels = vec![0.0f32; height * width];
        for y in 0..height {
            for x in 0..width {
                pixels[y * width + x] = centers
                    .iter()
                    .map(|&(cx, cy)| {
                        let dx = x as f64 - cx;
                        let dy = y as f64 - cy;
                        (1000.0 * (-(dx * dx + dy * dy) / (2.0 * 1.5f64.powi(2))).exp()) as f32
                    })
                    .sum();
            }
        }
        let frame = FitsFrame {
            path: "synthetic-psf.fits".into(),
            header: Map::new(),
            data: PixelData::F32(pixels),
            auxiliary: None,
            data_offset: 0,
            width,
            height,
        };
        let detection = DetectionResult {
            image_shape: [height, width],
            background: 0.0,
            noise: 1.0,
            threshold: 5.0,
            candidate_count: sources.len(),
            quality_count: sources.len(),
            sources: sources.clone(),
            parameters: json!({}),
            candidate_peaks: Vec::new(),
            line_artifact_coordinates: Vec::new(),
        };

        let result = summarize(&frame, &detection, 1, 2).unwrap();
        assert_eq!(result.feature_rows.len(), SOURCE_FEATURE_CLASSES.len());
        assert_eq!(result.spatial_rows.len(), SOURCE_FEATURE_CLASSES.len());
        let compact = result
            .feature_rows
            .iter()
            .find(|row| row["feature_class"] == "compact_quality")
            .unwrap();
        assert_eq!(compact["sample_count"], 1);
        assert_eq!(compact["sample_detection_ids"], "1");
        assert_eq!(compact["psf_model"], "empirical");
        assert_eq!(compact["psf_source_count"], 4);
        assert_eq!(compact["valid_count"], 1);
        let spatial = result
            .spatial_rows
            .iter()
            .find(|row| row["feature_class"] == "compact_quality")
            .unwrap();
        assert_eq!(spatial["sample_count"], 1);
        assert_eq!(spatial["local_fallback_count"], 1);
        assert_eq!(spatial["paired_correlation_count"], 0);
        assert!(detection.sources.iter().all(|source| source.quality_passed));

        let output = std::env::temp_dir().join(format!(
            "rst19-source-psf-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        crate::source_artifacts::write_detection_source_psf_tables(
            &frame, &detection, &output, 1, 2,
        )
        .unwrap();
        for name in [
            "source_feature_psf_similarity.csv",
            "source_feature_psf_spatial.csv",
        ] {
            let bytes = std::fs::read(output.join(name)).unwrap();
            assert!(bytes.starts_with(&[0xEF, 0xBB, 0xBF]));
            let mut reader = csv::ReaderBuilder::new().from_reader(bytes.as_slice());
            assert_eq!(reader.records().count(), SOURCE_FEATURE_CLASSES.len());
        }
        let _ = std::fs::remove_dir_all(output);
    }
}
