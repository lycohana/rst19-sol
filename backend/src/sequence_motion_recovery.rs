//! Bounded remeasurement of missing observations from original frame pixels.
use crate::{pipeline::FrameAnalysis, sequence_motion_tracks, sequence_trails, statistics, Result};
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn mask_zero_enabled(parameters: &Value) -> bool {
    parameters["mask_zero_pixels"]
        .as_bool()
        .unwrap_or_else(|| parameters["mask_zero_pixels"].as_f64().unwrap_or(0.0) != 0.0)
}

fn number(point: &Value, key: &str) -> Result<f64> {
    point[key]
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("missing finite {key}"))
}

fn bilinear(plane: &[Vec<f32>], x: f64, y: f64) -> Option<f32> {
    let height = plane.len();
    let width = plane.first()?.len();
    if x < 0.0 || y < 0.0 || x > (width - 1) as f64 || y > (height - 1) as f64 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    Some(
        ((1.0 - dy) * ((1.0 - dx) * f64::from(plane[y0][x0]) + dx * f64::from(plane[y0][x1]))
            + dy * ((1.0 - dx) * f64::from(plane[y1][x0]) + dx * f64::from(plane[y1][x1])))
            as f32,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn score_proposal(
    residual: &[Vec<f32>],
    valid: &[Vec<bool>],
    artifact: &[Vec<bool>],
    proposal: &Value,
    psf: f64,
    noise: f64,
    frame: usize,
    shift: [f64; 2],
    offset: [usize; 2],
    full_shape: [usize; 2],
) -> Result<Value> {
    let x = number(proposal, "x")?;
    let y = number(proposal, "y")?;
    let length = number(proposal, "length")?;
    let width = psf.max(number(proposal, "width")?);
    let angle = number(proposal, "angle")?.rem_euclid(180.0);
    let radians = angle.to_radians();
    let (sin, cos) = radians.sin_cos();
    let across = 1.0_f64.max(4.0_f64.min(0.5 * width + 1.0));
    let samples = 9usize.max(length.ceil() as usize + 1);
    let mut peak = f32::NEG_INFINITY;
    let mut artifact_count = 0usize;
    for i in 0..samples {
        for j in 0..5 {
            let along = -0.5 * length + length * i as f64 / (samples - 1) as f64;
            let cross = -across + 2.0 * across * j as f64 / 4.0;
            let sx = x + along * cos - cross * sin;
            let sy = y + along * sin + cross * cos;
            if let Some(value) = bilinear(residual, sx, sy) {
                peak = peak.max(value);
            }
            if sx >= 0.0
                && sy >= 0.0
                && sx <= (artifact[0].len() - 1) as f64
                && sy <= (artifact.len() - 1) as f64
                && artifact[(sy + 0.5).floor() as usize][(sx + 0.5).floor() as usize]
            {
                artifact_count += 1;
            }
        }
    }
    let (continuity, fill) = sequence_trails::line_profile_metrics_for_plane(
        residual,
        x,
        y,
        length,
        width,
        angle,
        (3.0 * noise).max(f64::EPSILON),
    );
    let global_x = x + offset[0] as f64;
    let global_y = y + offset[1] as f64;
    let half_width = (0.5 * width).max(0.5);
    let ex = 0.5 * length * cos.abs() + half_width * sin.abs() + 1.0;
    let ey = 0.5 * length * sin.abs() + half_width * cos.abs() + 1.0;
    let bbox = [
        (global_x - ex).floor().max(0.0) as usize,
        (global_y - ey).floor().max(0.0) as usize,
        (global_x + ex).ceil().min((full_shape[1] - 1) as f64) as usize,
        (global_y + ey).ceil().min((full_shape[0] - 1) as f64) as usize,
    ];
    let peak = if peak.is_finite() { peak } else { 0.0 };
    let mut point = json!({"frame_index":frame,"x":global_x,"y":global_y,
        "aligned_x":global_x-shift[0],"aligned_y":global_y-shift[1],"length_px":length,
        "width_px":width,"angle_deg":angle,"residual_snr":f64::from(peak)/noise,
        "area_pixels":1usize.max((length*width).round_ties_even() as usize),"bbox":bbox,
        "touches_edge":bbox[0]==0 || bbox[1]==0 || bbox[2]+1==full_shape[1] || bbox[3]+1==full_shape[0],
        "continuity_ratio":continuity,"profile_fill_ratio":fill,
        "line_artifact_overlap_fraction":artifact_count as f64/(samples*5) as f64,
        "observation_source":"detected"});
    let evidence = crate::single_frame_model::evaluate(
        residual, valid, x, y, length, width, angle, psf, noise,
    );
    if let (Some(object), Some(fields)) = (point.as_object_mut(), evidence.as_object()) {
        object.extend(fields.clone());
    }
    Ok(point)
}

/// Sequence-mode local multiscale proposals with separate ordinary/artifact budgets.
pub fn augment_multiscale_raw_components(
    components: &mut Value,
    analysis: &FrameAnalysis,
    reference: Option<&crate::sequence_temporal::RegisteredReference>,
    frame: usize,
    shift: [f64; 2],
    psf: f64,
    min_length: f64,
) -> Result<()> {
    let height = analysis.frame.height;
    let width = analysis.frame.width;
    let background = analysis.detection.background;
    let raw_noise = analysis.detection.noise.max(f64::from(f32::EPSILON));
    let noise = raw_noise
        * if reference.is_some() {
            std::f64::consts::SQRT_2
        } else {
            1.0
        };
    let mut valid = vec![vec![false; width]; height];
    let mut artifact = valid.clone();
    let mut residual = vec![vec![0.0_f32; width]; height];
    let mut raw_residual = residual.clone();
    let mask_zero = mask_zero_enabled(&analysis.detection.parameters);
    let saturation = analysis.detection.parameters["saturation_level"]
        .as_f64()
        .unwrap_or(-1.0);
    for y in 0..height {
        for x in 0..width {
            let value = analysis.frame.data.value(y * width + x) as f32;
            let good = value.is_finite()
                && !(y == 0 && x < 104)
                && (!mask_zero || value != 0.0)
                && (saturation <= 0.0 || f64::from(value) < saturation);
            valid[y][x] = good;
            if good {
                raw_residual[y][x] = value - background as f32;
                residual[y][x] = match reference {
                    Some(reference) => {
                        crate::sequence_motion_features::sample_registered_reference(
                            reference,
                            x as f64 - shift[0],
                            y as f64 - shift[1],
                        )
                        .filter(|value| value.is_finite())
                        .map_or(0.0, |reference| value - reference)
                    }
                    None => raw_residual[y][x],
                };
            }
        }
    }
    for &[x, y] in &analysis.detection.line_artifact_coordinates {
        if x < width && y < height {
            artifact[y][x] = true;
        }
    }
    let mut lengths = vec![min_length, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0];
    lengths.sort_by(f64::total_cmp);
    lengths.dedup();
    let mut supplemental = Vec::new();
    for (artifact_batch, budget, seeds, max_proposals) in [
        (false, 24usize, 256usize, 96usize),
        (true, 8usize, 64usize, 48usize),
    ] {
        let mask = valid
            .iter()
            .zip(&artifact)
            .flat_map(|(valid, artifact)| {
                valid
                    .iter()
                    .zip(artifact)
                    .map(|(valid, artifact)| *valid && (*artifact == artifact_batch))
            })
            .collect::<Vec<_>>();
        if !mask.iter().any(|valid| *valid) {
            continue;
        }
        let image = residual
            .iter()
            .flatten()
            .map(|value| f64::from(*value))
            .collect();
        let proposals = match crate::single_frame_trails::local_pixel_trail_proposals_for_plane(
            image,
            mask,
            height,
            width,
            0.0,
            noise,
            psf,
            &json!({
                "length_bank_px":lengths,"angle_step_deg":15.0,"max_proposals":max_proposals,
                "max_seeds":seeds,"seed_threshold":2.25,"sample_step_px":2.0,
            }),
        ) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let mut proposals = proposals["proposals"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if proposals.len() > budget {
            proposals.sort_by(|a, b| {
                b["response"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .total_cmp(&a["response"].as_f64().unwrap_or(0.0))
                    .then(
                        b["length"]
                            .as_f64()
                            .unwrap_or(0.0)
                            .total_cmp(&a["length"].as_f64().unwrap_or(0.0)),
                    )
                    .then(
                        a["y"]
                            .as_f64()
                            .unwrap_or(0.0)
                            .total_cmp(&b["y"].as_f64().unwrap_or(0.0)),
                    )
                    .then(
                        a["x"]
                            .as_f64()
                            .unwrap_or(0.0)
                            .total_cmp(&b["x"].as_f64().unwrap_or(0.0)),
                    )
            });
            proposals.truncate(budget);
        }
        for proposal in proposals {
            if number(&proposal, "length")? < min_length {
                continue;
            }
            let mut point = score_proposal(
                &residual,
                &valid,
                &artifact,
                &proposal,
                psf,
                noise,
                frame,
                shift,
                [0, 0],
                [height, width],
            )?;
            if supplemental
                .iter()
                .any(|existing| sequence_motion_tracks::line_geometries_overlap(&point, existing))
                || !sequence_motion_tracks::sequence_line_model_review_supported(&point, psf)?
            {
                continue;
            }
            point["area_pixels"] = json!((number(&point, "length_px")?
                * number(&point, "width_px")?.max(1.0))
            .round_ties_even() as usize);
            supplemental.push(point);
        }
    }
    let Some(points) = components["points"].as_array_mut() else {
        return Err("raw motion components require points".into());
    };
    let mut added = 0usize;
    for mut point in supplemental {
        if points
            .iter()
            .any(|existing| sequence_motion_tracks::line_geometries_overlap(&point, existing))
        {
            continue;
        }
        let evidence = crate::single_frame_model::evaluate(
            &raw_residual,
            &valid,
            number(&point, "x")?,
            number(&point, "y")?,
            number(&point, "length_px")?,
            number(&point, "width_px")?,
            number(&point, "angle_deg")?,
            psf,
            raw_noise,
        );
        if let (Some(object), Some(fields)) = (point.as_object_mut(), evidence.as_object()) {
            object.extend(fields.clone());
        }
        points.push(point);
        added += 1;
    }
    components["multiscale_added_count"] = json!(added);
    Ok(())
}

pub fn recover_predicted_roi_points(
    candidates: &mut Value,
    analyses: &[FrameAnalysis],
    shifts: &[[f64; 2]],
    times: &[f64],
    psf: f64,
) -> Result<()> {
    if analyses.len() != shifts.len()
        || analyses.len() != times.len()
        || times.iter().any(|value| !value.is_finite())
        || times.windows(2).any(|pair| pair[1] <= pair[0])
        || shifts.iter().flatten().any(|value| !value.is_finite())
        || psf <= 0.0
        || !psf.is_finite()
    {
        return Err("ROI recovery requires matching frames, finite shifts, increasing times, and positive PSF".into());
    }
    if analyses.len() < 2 {
        return Ok(());
    }
    let mut claimed = BTreeSet::new();
    let mut attempted = 0usize;
    let mut recovered = 0usize;
    let Some(tracks) = candidates["tracks"].as_array_mut() else {
        return Ok(());
    };
    for track in tracks {
        let Some(points) = track["points"].as_array_mut() else {
            continue;
        };
        points.sort_by_key(|point| point["frame_index"].as_u64().unwrap_or(0));
        if points.len() < 6 {
            continue;
        }
        let lengths = points
            .iter()
            .map(|point| number(point, "length_px"))
            .collect::<Result<Vec<_>>>()?;
        let median_length = statistics::median(&lengths)?;
        if median_length >= 32.0_f64.max(8.0 * psf) {
            continue;
        }
        let frames = points
            .iter()
            .map(|point| {
                point["frame_index"]
                    .as_u64()
                    .map(|v| v as usize)
                    .ok_or("invalid frame index".to_string())
            })
            .collect::<Result<BTreeSet<_>>>()?;
        if frames.last().is_none_or(|index| *index >= analyses.len()) {
            continue;
        }
        let mut runs = Vec::<Vec<usize>>::new();
        for frame in &frames {
            if runs
                .last()
                .is_some_and(|run| run.last() == Some(&frame.saturating_sub(1)))
            {
                runs.last_mut().unwrap().push(*frame);
            } else {
                runs.push(vec![*frame]);
            }
        }
        let strong = runs.iter().filter(|run| run.len() >= 3).collect::<Vec<_>>();
        if strong.len() < 2 {
            continue;
        }
        let first = *frames.first().unwrap();
        let last = *frames.last().unwrap();
        let mut missing = (first + 1..last)
            .filter(|index| !frames.contains(index))
            .map(|index| (index, false))
            .collect::<Vec<_>>();
        let mut gap_length = 0;
        let mut previous = None;
        let mut too_long = false;
        for (frame, _) in &missing {
            gap_length = if previous == Some(frame - 1) {
                gap_length + 1
            } else {
                1
            };
            too_long |= gap_length > 2;
            previous = Some(*frame);
        }
        if too_long {
            continue;
        }
        if strong.last().unwrap().len() >= 4 && last + 1 < analyses.len() {
            missing.push((last + 1, true));
        }
        let t = points
            .iter()
            .map(|point| times[point["frame_index"].as_u64().unwrap() as usize])
            .collect::<Vec<_>>();
        let mean_t = t.iter().sum::<f64>() / t.len() as f64;
        let denominator = t.iter().map(|v| (v - mean_t).powi(2)).sum::<f64>();
        if denominator <= 0.0 {
            continue;
        }
        let fit = |key: &str| -> Result<(f64, f64)> {
            let v = points
                .iter()
                .map(|point| number(point, key))
                .collect::<Result<Vec<_>>>()?;
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let slope = t
                .iter()
                .zip(v)
                .map(|(time, value)| (time - mean_t) * (value - mean))
                .sum::<f64>()
                / denominator;
            Ok((slope, mean - slope * mean_t))
        };
        let (vx, ix) = fit("aligned_x")?;
        let (vy, iy) = fit("aligned_y")?;
        let velocity_angle = vy.atan2(vx).to_degrees();
        for (frame, endpoint) in missing {
            attempted += 1;
            let analysis = &analyses[frame];
            let height = analysis.frame.height;
            let width = analysis.frame.width;
            let px = vx * times[frame] + ix + shifts[frame][0];
            let py = vy * times[frame] + iy + shifts[frame][1];
            let radius = 16usize.max((4.0 * psf + 12.0).ceil() as usize);
            let x0 = (px.floor() as isize - radius as isize).max(0) as usize;
            let y0 = (py.floor() as isize - radius as isize).max(0) as usize;
            if x0 >= width || y0 >= height {
                continue;
            }
            let x1 = width.min(x0 + 2 * radius + 1);
            let y1 = height.min(y0 + 2 * radius + 1);
            let background = analysis.detection.background;
            let noise = analysis.detection.noise.max(f64::from(f32::EPSILON));
            let mask_zero = mask_zero_enabled(&analysis.detection.parameters);
            let saturation = analysis.detection.parameters["saturation_level"]
                .as_f64()
                .unwrap_or(-1.0);
            let mut image = vec![vec![0.0_f32; x1 - x0]; y1 - y0];
            let mut valid = vec![vec![false; x1 - x0]; y1 - y0];
            let mut residual = image.clone();
            let mut artifact = valid.clone();
            for y in y0..y1 {
                for x in x0..x1 {
                    let value = analysis.frame.data.value(y * width + x) as f32;
                    let good = value.is_finite()
                        && !(y == 0 && x < 104)
                        && (!mask_zero || value != 0.0)
                        && (saturation <= 0.0 || f64::from(value) < saturation);
                    image[y - y0][x - x0] = if value.is_finite() { value } else { 0.0 };
                    valid[y - y0][x - x0] = good;
                    if good {
                        residual[y - y0][x - x0] = (f64::from(value) - background) as f32;
                    }
                }
            }
            for &[x, y] in &analysis.detection.line_artifact_coordinates {
                if x >= x0 && x < x1 && y >= y0 && y < y1 {
                    artifact[y - y0][x - x0] = true;
                }
            }
            let proposals = match crate::single_frame_trails::local_pixel_trail_proposals(&json!({
                "image":image,"valid_mask":valid,"background":background,"noise":noise,"psf_fwhm":psf,
                "length_bank_px":[6.0,8.0,10.0,12.0,16.0,20.0],"angle_step_deg":10.0,"max_proposals":12,"max_seeds":64,"seed_threshold":1.5,
            })) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let mut best: Option<(f64, Value)> = None;
            for proposal in proposals["proposals"].as_array().into_iter().flatten() {
                let x = number(proposal, "x")? + x0 as f64;
                let y = number(proposal, "y")? + y0 as f64;
                let error = (x - px).hypot(y - py);
                let length = number(proposal, "length")?;
                let angle_diff =
                    ((number(proposal, "angle")? - velocity_angle + 90.0).rem_euclid(180.0) - 90.0)
                        .abs();
                let claim = (
                    frame,
                    x.round_ties_even() as i64,
                    y.round_ties_even() as i64,
                );
                if error > 2.5
                    || angle_diff > 20.0
                    || !(0.5..=1.8).contains(&(length / median_length))
                    || claimed.contains(&claim)
                {
                    continue;
                }
                let mut point = score_proposal(
                    &residual,
                    &valid,
                    &artifact,
                    proposal,
                    psf,
                    noise,
                    frame,
                    shifts[frame],
                    [x0, y0],
                    [height, width],
                )?;
                let supported = if endpoint {
                    sequence_motion_tracks::model_is_supported(&point)
                } else {
                    sequence_motion_tracks::sequence_line_model_review_supported(&point, psf)?
                };
                if !supported
                    || point["line_artifact_overlap_fraction"]
                        .as_f64()
                        .unwrap_or(0.0)
                        >= 0.75
                {
                    continue;
                }
                point["observation_source"] = json!("recovered_by_predicted_roi");
                point["sequence_candidate_source"] = json!("single_frame_shape");
                if best.as_ref().is_none_or(|(score, _)| error < *score) {
                    best = Some((error, point));
                }
            }
            if let Some((_, point)) = best {
                claimed.insert((
                    frame,
                    number(&point, "x")?.round_ties_even() as i64,
                    number(&point, "y")?.round_ties_even() as i64,
                ));
                points.push(point);
                recovered += 1;
            }
        }
        points.sort_by_key(|point| point["frame_index"].as_u64().unwrap_or(0));
        track["point_count"] = json!(points.len());
    }
    candidates["predicted_roi_recovery_audit"] =
        json!({"attempted_frame_count":attempted,"recovered_point_count":recovered});
    Ok(())
}
