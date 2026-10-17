//! Full standalone exposure-line proposal profile, independent of sequence
//! budgets and the point-source detector's background/quality parameters.
use crate::{
    models::{FitsFrame, PixelData},
    sequence_motion_features, sequence_motion_recovery, sequence_motion_tracks,
    single_frame_trails, Result,
};
use serde_json::{json, Value};

const LOCAL_LENGTHS: &[f64] = &[8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0];
const REDUCED_LENGTHS: &[f64] = &[
    8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0,
];

struct Prepared {
    residual: Vec<Vec<f32>>,
    valid: Vec<Vec<bool>>,
    artifact: Vec<Vec<bool>>,
    background: f64,
    noise: f64,
    mask_zero: bool,
    saturation: Option<f64>,
}

fn prepare(frame: &FitsFrame, psf: f64) -> Result<Prepared> {
    let (height, width) = (frame.height, frame.width);
    if height.min(width) < 8 || frame.data.len() != height.saturating_mul(width) {
        return Err("invalid standalone exposure image shape".into());
    }
    let auxiliary = crate::fits::auxiliary_mask(height, width)?;
    let numeric = (0..frame.data.len())
        .map(|index| frame.data.value(index) as f32)
        .collect::<Vec<_>>();
    let base_mask = numeric
        .iter()
        .zip(&auxiliary)
        .map(|(value, mask)| *mask || !value.is_finite())
        .collect::<Vec<_>>();
    let float64 = matches!(frame.data, PixelData::F64(_));
    let statistics = |mask: &[bool]| {
        if float64 {
            let values = (0..frame.data.len())
                .map(|index| frame.data.value(index))
                .collect::<Vec<_>>();
            crate::statistics::sigma_clipped_stats(&values, Some(mask), 3.0, 4, 1_000_000)
        } else {
            crate::statistics32::sigma_clipped_stats(&numeric, Some(mask), 3.0, 4, 1_000_000)
        }
    };
    let (global_background, _) = statistics(&base_mask)?;
    let (mask, saturation, mask_zero) = if float64 {
        crate::detection::working_mask(
            &frame.data,
            Some(&auxiliary),
            global_background,
            None,
            None,
        )?
    } else {
        crate::detection::working_mask_float32(
            &frame.data,
            Some(&auxiliary),
            global_background,
            None,
            None,
        )?
    };
    let (background, noise) = statistics(&mask)?;
    let noise = noise.max(f64::from(f32::EPSILON));
    let valid = mask.iter().map(|masked| !*masked).collect::<Vec<_>>();
    let residual = numeric
        .iter()
        .zip(&valid)
        .map(|(value, good)| {
            if *good {
                *value - background as f32
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let artifact = crate::filtering32::linear_artifact_mask(
        &residual,
        &vec![noise as f32; residual.len()],
        &valid,
        height,
        width,
        4.0,
        psf,
    )?;
    Ok(Prepared {
        residual: residual.chunks(width).map(<[f32]>::to_vec).collect(),
        valid: valid.chunks(width).map(<[bool]>::to_vec).collect(),
        artifact: artifact.chunks(width).map(<[bool]>::to_vec).collect(),
        background,
        noise,
        mask_zero,
        saturation,
    })
}

fn lengths(min_length: f64) -> Vec<f64> {
    let mut result = LOCAL_LENGTHS.to_vec();
    result.push(min_length);
    result.sort_by(f64::total_cmp);
    result.dedup();
    result
}

fn local_options(min_length: f64, artifact: bool) -> Value {
    json!({"length_bank_px":lengths(min_length),"angle_step_deg":10.0,
        "max_proposals":if artifact {48} else {128},"max_seeds":if artifact {128} else {512},
        "seed_threshold":2.25,"sample_step_px":1.0})
}

fn reduced_plane(prepared: &Prepared, factor: usize) -> (Vec<f64>, Vec<bool>, usize, usize) {
    let height = prepared.residual.len() / factor;
    let width = prepared.residual[0].len() / factor;
    let mut image = Vec::with_capacity(height * width);
    let mut valid = Vec::with_capacity(height * width);
    let minimum = (0.75 * (factor * factor) as f64).ceil() as usize;
    for y in 0..height {
        for x in 0..width {
            let (mut sum, mut count) = (0.0, 0usize);
            for yy in y * factor..(y + 1) * factor {
                let mut row_sum = 0.0;
                for xx in x * factor..(x + 1) * factor {
                    if prepared.valid[yy][xx] {
                        row_sum += f64::from(prepared.residual[yy][xx]) + prepared.background;
                        count += 1;
                    }
                }
                sum += row_sum;
            }
            image.push(if count > 0 { sum / count as f64 } else { 0.0 });
            valid.push(count >= minimum.max(1));
        }
    }
    (image, valid, height, width)
}

fn proposal_array(result: Result<Value>, stage: &str, errors: &mut Vec<String>) -> Vec<Value> {
    match result {
        Ok(value) => value["proposals"].as_array().cloned().unwrap_or_default(),
        Err(error) => {
            errors.push(format!("{stage}: {error}"));
            vec![]
        }
    }
}

/// No model-proposal budget is applied here: the legacy standalone path
/// reviews every bounded proposal, including native-pixel 8/10/12/20px lines.
pub(crate) fn extract(
    frame: &FitsFrame,
    psf: f64,
    min_length: f64,
    short_min_area: usize,
    progress: &mut dyn FnMut(f32, &str),
) -> Result<Value> {
    progress(0.0, "按单帧原始口径估计全局背景、噪声与有效像素");
    let prepared = prepare(frame, psf)?;
    progress(0.10, "提取连通线形并复核局部单星、双星和线段模型");
    let mut components = sequence_motion_features::standalone_components_for_planes(
        &prepared.residual,
        &prepared.valid,
        &prepared.artifact,
        psf,
        prepared.noise,
        min_length,
        short_min_area,
    )?;
    let mut batches = Vec::<(String, Vec<Value>, f64, f64)>::new();
    let mut errors = Vec::new();
    let mut configs = Vec::new();
    if frame.height.min(frame.width) >= 512 {
        for (index, artifact_batch) in [false, true].into_iter().enumerate() {
            let stage = if artifact_batch {
                "artifact_pixels"
            } else {
                "native_pixels"
            };
            progress(
                if artifact_batch { 0.45 } else { 0.25 },
                if artifact_batch {
                    "以独立 128 种子预算复核伪影区域中的原图短线"
                } else {
                    "在原始像素上扫描完整短线长度档位、10°方向和512个种子"
                },
            );
            let options = local_options(min_length, artifact_batch);
            let mask = prepared
                .valid
                .iter()
                .zip(&prepared.artifact)
                .flat_map(|(valid, artifact)| {
                    valid
                        .iter()
                        .zip(artifact)
                        .map(move |(v, a)| *v && *a == artifact_batch)
                })
                .collect::<Vec<_>>();
            let proposals = if mask.iter().any(|value| *value) {
                let image = prepared
                    .residual
                    .iter()
                    .flatten()
                    .map(|value| f64::from(*value) + prepared.background)
                    .collect();
                proposal_array(
                    single_frame_trails::local_pixel_trail_proposals_for_plane(
                        image,
                        mask,
                        frame.height,
                        frame.width,
                        prepared.background,
                        prepared.noise,
                        psf,
                        &options,
                    ),
                    stage,
                    &mut errors,
                )
            } else {
                vec![]
            };
            configs.push(json!({"stage":stage,"options":options,"proposal_count":proposals.len(),"order":index}));
            batches.push((stage.into(), proposals, 1.0, 0.0));
        }
        progress(0.65, "补充降采样多尺度线段提案，随后回到原图复核");
        let factor = frame.height.min(frame.width).div_ceil(1024).max(1);
        let (image, mask, height, width) = reduced_plane(&prepared, factor);
        let mut lengths = REDUCED_LENGTHS
            .iter()
            .filter_map(|length| {
                let scaled = *length / factor as f64;
                (scaled < (height.min(width) as f64 - 1.0))
                    .then_some((scaled * 100.0).round_ties_even().max(200.0) / 100.0)
            })
            .collect::<Vec<_>>();
        lengths.sort_by(f64::total_cmp);
        lengths.dedup();
        let options = json!({"length_bank_px":lengths,"angle_step_deg":15.0,"max_proposals":256});
        let proposals = if lengths.is_empty() {
            vec![]
        } else {
            proposal_array(
                single_frame_trails::single_frame_trail_proposals_for_plane(
                    image,
                    mask,
                    height,
                    width,
                    prepared.background,
                    prepared.noise / factor as f64,
                    (psf / factor as f64).max(0.5),
                    &options,
                ),
                "reduced_pixels",
                &mut errors,
            )
        };
        configs.push(json!({"stage":"reduced_pixels","factor":factor,"options":options,"proposal_count":proposals.len()}));
        batches.push((
            "reduced_pixels".into(),
            proposals,
            factor as f64,
            0.5 * (factor - 1) as f64,
        ));
    }
    let points = components["points"]
        .as_array_mut()
        .ok_or("missing standalone connected components")?;
    let base_count = points.len();
    let mut accepted = points
        .iter()
        .filter(|point| {
            matches!(
                point["single_frame_model"].as_str(),
                Some("line" | "long_line_profile" | "elliptical_point" | "not_fit")
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let total = batches
        .iter()
        .map(|(_, proposals, _, _)| proposals.len())
        .sum::<usize>()
        .max(1);
    let mut completed = 0;
    let mut reviews = Vec::new();
    for (stage, proposals, scale, offset) in batches {
        let (mut duplicate, mut rejected, mut added) = (0usize, 0usize, 0usize);
        for proposal in proposals {
            completed += 1;
            progress(
                0.80 + 0.18 * completed as f32 / total as f32,
                &format!("原图模型复核 {completed}/{total} · {stage}"),
            );
            let number = |key: &str| {
                proposal[key]
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| format!("invalid trail {key}"))
            };
            let x = number("x")? * scale + offset;
            let y = number("y")? * scale + offset;
            let length = number("length")? * scale;
            let width = (number("width")? * scale).max(psf);
            let angle = number("angle")?.rem_euclid(180.0);
            if length < min_length {
                rejected += 1;
                continue;
            }
            let geometry =
                json!({"x":x,"y":y,"length_px":length,"width_px":width,"angle_deg":angle});
            if accepted.iter().any(|existing| {
                sequence_motion_tracks::line_geometries_overlap(&geometry, existing)
            }) {
                duplicate += 1;
                continue;
            }
            let mapped = json!({"x":x,"y":y,"length":length,"width":width,"angle":angle});
            let mut point = sequence_motion_recovery::score_proposal(
                &prepared.residual,
                &prepared.valid,
                &prepared.artifact,
                &mapped,
                psf,
                prepared.noise,
                0,
                [0.0, 0.0],
                [0, 0],
                [frame.height, frame.width],
            )?;
            if !sequence_motion_tracks::sequence_line_model_review_supported(&point, psf)? {
                rejected += 1;
                continue;
            }
            point["proposal_source"] = json!(stage);
            accepted.push(point.clone());
            points.push(point);
            added += 1;
        }
        reviews.push(json!({"stage":stage,"duplicate_count":duplicate,"model_rejected_count":rejected,"added_count":added}));
    }
    let added = points.len() - base_count;
    components["standalone_preprocessing"] = json!({"background":prepared.background,"noise":prepared.noise,
        "mask_zero_pixels":prepared.mask_zero,"saturation_level":prepared.saturation});
    components["proposal_profile"] = json!({"name":"single_frame_full","model_proposal_budget":null,
        "batches":configs,"reviews":reviews,"errors":errors,"added_count":added});
    progress(1.0, "完整单帧提案与原图复核完成");
    Ok(components)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_short_scan_keeps_single_frame_banks_and_budgets() {
        let ordinary = local_options(7.5, false);
        let artifact = local_options(7.5, true);
        assert_eq!(ordinary["angle_step_deg"], 10.0);
        assert_eq!(ordinary["sample_step_px"], 1.0);
        assert_eq!(ordinary["max_seeds"], 512);
        assert_eq!(ordinary["max_proposals"], 128);
        assert_eq!(artifact["max_seeds"], 128);
        assert!(ordinary["length_bank_px"]
            .as_array()
            .unwrap()
            .contains(&json!(10.0)));
        assert!(ordinary["length_bank_px"]
            .as_array()
            .unwrap()
            .contains(&json!(20.0)));
    }

    #[test]
    fn block_reduction_uses_valid_pixel_means_and_preserves_coordinate_offset() {
        let prepared = Prepared {
            residual: vec![vec![1.0, 3.0, 9.0, 11.0], vec![5.0, 7.0, 13.0, 15.0]],
            valid: vec![vec![true, true, false, true], vec![true, true, true, false]],
            artifact: vec![],
            background: 10.0,
            noise: 2.0,
            mask_zero: false,
            saturation: None,
        };
        let (image, valid, height, width) = reduced_plane(&prepared, 2);
        assert_eq!((height, width), (1, 2));
        assert_eq!(image, vec![14.0, 22.0]);
        assert_eq!(valid, vec![true, false]);
    }
}
