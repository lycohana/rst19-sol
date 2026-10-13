//! End-to-end point-source sequence registration, tracking, and classification.
use crate::{
    measurement::Detection,
    sequence_assignment::{global_frame_assignments, TrackPosition},
    sequence_motion::{
        derive_motion_displacement_floor, fit_track_motion, MotionMetric, TrackObservation,
    },
    sequence_registration::estimate_translation,
    Result,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const AFFINE_MIN_PAIRS: usize = 12;
const AFFINE_RANSAC_TRIALS: usize = 160;
const AFFINE_INLIER_RADIUS_PX: f64 = 1.5;
const AFFINE_MAX_SCALE_DEVIATION: f64 = 0.05;
const AFFINE_MAX_SKEW: f64 = 0.05;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TrackSequenceOptions {
    pub frame_names: Option<Vec<String>>,
    pub frame_times: Option<Vec<f64>>,
    pub link_radius_px: f64,
    pub min_presence: Option<usize>,
    pub motion_min_displacement_px: f64,
    pub registration_radius_px: f64,
    pub max_motion_fit_rms_px: f64,
    pub persistent_min_presence: Option<usize>,
}

impl Default for TrackSequenceOptions {
    fn default() -> Self {
        Self {
            frame_names: None,
            frame_times: None,
            link_radius_px: 4.0,
            min_presence: None,
            motion_min_displacement_px: 2.0,
            registration_radius_px: 8.0,
            max_motion_fit_rms_px: 0.75,
            persistent_min_presence: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackPoint {
    pub frame_index: usize,
    pub detection_id: i64,
    pub x: f64,
    pub y: f64,
    pub aligned_x: f64,
    pub aligned_y: f64,
    pub flux_snr: Option<f64>,
    pub quality_passed: bool,
    pub candidate_snr: Option<f64>,
}

#[derive(Debug, Clone)]
struct AffineFit {
    model: [[f64; 3]; 2],
    residual_median: f64,
}

#[derive(Debug, Clone)]
struct TrackMetrics {
    displacement: f64,
    speed: f64,
    fit_rms: Option<f64>,
}

fn fit_affine(reference: &[[f64; 2]], observed: &[[f64; 2]]) -> Option<[[f64; 3]; 2]> {
    if reference.len() != observed.len() || reference.len() < 3 {
        return None;
    }
    let count = reference.len() as f64;
    let center_x = reference.iter().map(|p| p[0]).sum::<f64>() / count;
    let center_y = reference.iter().map(|p| p[1]).sum::<f64>() / count;
    let output_center = [
        observed.iter().map(|p| p[0]).sum::<f64>() / count,
        observed.iter().map(|p| p[1]).sum::<f64>() / count,
    ];
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    let mut xz = [0.0; 2];
    let mut yz = [0.0; 2];
    for (input, output) in reference.iter().zip(observed) {
        let dx = input[0] - center_x;
        let dy = input[1] - center_y;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
        for axis in 0..2 {
            let dz = output[axis] - output_center[axis];
            xz[axis] += dx * dz;
            yz[axis] += dy * dz;
        }
    }
    let determinant = xx * yy - xy * xy;
    if !determinant.is_finite() || determinant <= f64::EPSILON * (xx * yy).abs().max(1.0) * 16.0 {
        return None;
    }
    let mut result = [[0.0; 3]; 2];
    for axis in 0..2 {
        result[axis][0] = (xz[axis] * yy - yz[axis] * xy) / determinant;
        result[axis][1] = (yz[axis] * xx - xz[axis] * xy) / determinant;
        result[axis][2] =
            output_center[axis] - result[axis][0] * center_x - result[axis][1] * center_y;
    }
    result
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(result)
}

fn affine_residual(model: &[[f64; 3]; 2], reference: [f64; 2], observed: [f64; 2]) -> f64 {
    let dx = model[0][0] * reference[0] + model[0][1] * reference[1] + model[0][2] - observed[0];
    let dy = model[1][0] * reference[0] + model[1][1] * reference[1] + model[1][2] - observed[1];
    dx.hypot(dy)
}

fn valid_affine(model: &[[f64; 3]; 2]) -> bool {
    let determinant = model[0][0] * model[1][1] - model[0][1] * model[1][0];
    determinant.is_finite()
        && determinant > 0.0
        && (model[0][0] - 1.0).abs() <= AFFINE_MAX_SCALE_DEVIATION
        && (model[1][1] - 1.0).abs() <= AFFINE_MAX_SCALE_DEVIATION
        && model[0][1].abs() <= AFFINE_MAX_SKEW
        && model[1][0].abs() <= AFFINE_MAX_SKEW
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

/// NumPy's legacy PCG64 stream and `Generator.choice(..., replace=False)`
/// sampler, kept local so sequence registration is reproducible without a
/// Python runtime or platform-specific random-number library.
struct NumpyPcg64 {
    state: u128,
    increment: u128,
    has_u32: bool,
    cached_u32: u32,
}

impl NumpyPcg64 {
    fn new(seed: u64) -> Self {
        const INIT_A: u32 = 0x43b0_d7e5;
        const MULT_A: u32 = 0x931e_8875;
        const INIT_B: u32 = 0x8b51_f9dd;
        const MULT_B: u32 = 0x58f3_8ded;
        const MIX_MULT_L: u32 = 0xca01_f9dd;
        const MIX_MULT_R: u32 = 0x4973_f715;

        fn hashmix(mut value: u32, hash: &mut u32) -> u32 {
            value ^= *hash;
            *hash = hash.wrapping_mul(MULT_A);
            value = value.wrapping_mul(*hash);
            value ^ (value >> 16)
        }

        fn mix(x: u32, y: u32) -> u32 {
            let value = MIX_MULT_L
                .wrapping_mul(x)
                .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
            value ^ (value >> 16)
        }

        let mut entropy = vec![seed as u32];
        if seed >> 32 != 0 {
            entropy.push((seed >> 32) as u32);
        }
        let mut pool = [0_u32; 4];
        let mut hash = INIT_A;
        for (index, value) in pool.iter_mut().enumerate() {
            *value = hashmix(entropy.get(index).copied().unwrap_or_default(), &mut hash);
        }
        let mut source = 0;
        while source < pool.len() {
            let source_value = pool[source];
            let mut destination = 0;
            while destination < pool.len() {
                if source != destination {
                    let mixed = hashmix(source_value, &mut hash);
                    pool[destination] = mix(pool[destination], mixed);
                }
                destination += 1;
            }
            source += 1;
        }
        for value in entropy.iter().skip(pool.len()) {
            for destination in &mut pool {
                let mixed = hashmix(*value, &mut hash);
                *destination = mix(*destination, mixed);
            }
        }

        let mut state_words = [0_u64; 4];
        let mut hash = INIT_B;
        for index in 0..state_words.len() * 2 {
            let mut value = pool[index % pool.len()] ^ hash;
            hash = hash.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash);
            value ^= value >> 16;
            let word = index / 2;
            if index.is_multiple_of(2) {
                state_words[word] = value as u64;
            } else {
                state_words[word] |= (value as u64) << 32;
            }
        }

        let init_state = ((state_words[0] as u128) << 64) | state_words[1] as u128;
        let init_sequence = ((state_words[2] as u128) << 64) | state_words[3] as u128;
        let mut generator = Self {
            state: 0,
            increment: (init_sequence << 1) | 1,
            has_u32: false,
            cached_u32: 0,
        };
        generator.advance();
        generator.state = generator.state.wrapping_add(init_state);
        generator.advance();
        generator
    }

    fn advance(&mut self) {
        const PCG64_MULTIPLIER: u128 =
            ((2549297995355413924_u128) << 64) | 4865540595714422341_u128;
        self.state = self
            .state
            .wrapping_mul(PCG64_MULTIPLIER)
            .wrapping_add(self.increment);
    }

    fn next_u64(&mut self) -> u64 {
        self.advance();
        let state = self.state;
        let value = ((state >> 64) as u64) ^ (state as u64);
        value.rotate_right((state >> 122) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if self.has_u32 {
            self.has_u32 = false;
            return self.cached_u32;
        }
        let value = self.next_u64();
        self.has_u32 = true;
        self.cached_u32 = (value >> 32) as u32;
        value as u32
    }

    fn bounded_inclusive(&mut self, maximum: usize) -> usize {
        if maximum == 0 {
            return 0;
        }
        let range = maximum as u32 + 1;
        loop {
            let product = (self.next_u32() as u64) * (range as u64);
            let leftover = product as u32;
            if leftover < range {
                let threshold = (u32::MAX - maximum as u32) % range;
                if leftover < threshold {
                    continue;
                }
            }
            return (product >> 32) as usize;
        }
    }

    fn choice_without_replacement(&mut self, population: usize, count: usize) -> Vec<usize> {
        let mut chosen = Vec::with_capacity(count);
        for upper in population - count..population {
            let value = self.bounded_inclusive(upper);
            let selected = if chosen.contains(&value) {
                upper
            } else {
                value
            };
            chosen.push(selected);
        }
        for index in (1..count).rev() {
            let other = self.bounded_inclusive(index);
            chosen.swap(index, other);
        }
        chosen
    }
}

fn fit_affine_ransac(
    reference: &[[f64; 2]],
    observed: &[[f64; 2]],
    seed: u64,
) -> Option<AffineFit> {
    if reference.len() != observed.len() || reference.len() < AFFINE_MIN_PAIRS {
        return None;
    }
    let mut best_mask: Option<Vec<bool>> = None;
    let mut best_count = 0usize;
    let mut best_median = f64::INFINITY;
    let mut rng = NumpyPcg64::new(seed);
    for _ in 0..AFFINE_RANSAC_TRIALS {
        let sample = rng.choice_without_replacement(reference.len(), 3);
        let [first, second, third] = [sample[0], sample[1], sample[2]];
        let a = reference[first];
        let b = reference[second];
        let c = reference[third];
        let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        let scale = (b[0] - a[0]).hypot(b[1] - a[1]) * (c[0] - a[0]).hypot(c[1] - a[1]);
        if !twice_area.is_finite() || twice_area.abs() <= f64::EPSILON * scale.max(1.0) * 16.0 {
            continue;
        }
        let sample_reference = sample.iter().map(|&i| reference[i]).collect::<Vec<_>>();
        let sample_observed = sample.iter().map(|&i| observed[i]).collect::<Vec<_>>();
        let Some(model) = fit_affine(&sample_reference, &sample_observed) else {
            continue;
        };
        if !valid_affine(&model) {
            continue;
        }
        let residuals = reference
            .iter()
            .zip(observed)
            .map(|(r, o)| affine_residual(&model, *r, *o))
            .collect::<Vec<_>>();
        let mask = residuals
            .iter()
            .map(|value| *value <= AFFINE_INLIER_RADIUS_PX)
            .collect::<Vec<_>>();
        let count = mask.iter().filter(|value| **value).count();
        if count < AFFINE_MIN_PAIRS {
            continue;
        }
        let mut inlier_residuals = residuals
            .iter()
            .zip(&mask)
            .filter_map(|(value, yes)| (*yes).then_some(*value))
            .collect::<Vec<_>>();
        let median_residual = median(&mut inlier_residuals).unwrap_or(f64::INFINITY);
        if count > best_count || (count == best_count && median_residual < best_median) {
            best_count = count;
            best_median = median_residual;
            best_mask = Some(mask);
        }
    }
    let mut inliers = best_mask?;
    let mut reference_inliers = reference
        .iter()
        .zip(&inliers)
        .filter_map(|(p, yes)| (*yes).then_some(*p))
        .collect::<Vec<_>>();
    let mut observed_inliers = observed
        .iter()
        .zip(&inliers)
        .filter_map(|(p, yes)| (*yes).then_some(*p))
        .collect::<Vec<_>>();
    let mut model = fit_affine(&reference_inliers, &observed_inliers)?;
    for _ in 0..6 {
        if !valid_affine(&model) {
            return None;
        }
        let residuals = reference
            .iter()
            .zip(observed)
            .map(|(r, o)| affine_residual(&model, *r, *o))
            .collect::<Vec<_>>();
        let mut selected = residuals
            .iter()
            .zip(&inliers)
            .filter_map(|(value, yes)| (*yes).then_some(*value))
            .collect::<Vec<_>>();
        let med = median(&mut selected).unwrap_or(f64::INFINITY);
        let mut deviations = residuals
            .iter()
            .zip(&inliers)
            .filter_map(|(value, yes)| (*yes).then_some((value - med).abs()))
            .collect::<Vec<_>>();
        let mad = median(&mut deviations).unwrap_or(f64::INFINITY);
        let cutoff = AFFINE_INLIER_RADIUS_PX.max(3.0f64.min(med + 3.0 * 1.4826 * mad));
        let updated = residuals
            .iter()
            .map(|value| *value <= cutoff)
            .collect::<Vec<_>>();
        if updated.iter().filter(|value| **value).count() < AFFINE_MIN_PAIRS {
            break;
        }
        if updated == inliers {
            break;
        }
        inliers = updated;
        reference_inliers = reference
            .iter()
            .zip(&inliers)
            .filter_map(|(p, yes)| (*yes).then_some(*p))
            .collect();
        observed_inliers = observed
            .iter()
            .zip(&inliers)
            .filter_map(|(p, yes)| (*yes).then_some(*p))
            .collect();
        model = fit_affine(&reference_inliers, &observed_inliers)?;
    }
    let inlier_count = inliers.iter().filter(|value| **value).count();
    if !valid_affine(&model)
        || inlier_count < AFFINE_MIN_PAIRS
        || inlier_count as f64 / (reference.len() as f64) < 0.5
    {
        return None;
    }
    let mut residuals = reference
        .iter()
        .zip(observed)
        .zip(&inliers)
        .filter_map(|((r, o), yes)| (*yes).then_some(affine_residual(&model, *r, *o)))
        .collect::<Vec<_>>();
    Some(AffineFit {
        model,
        residual_median: median(&mut residuals)?,
    })
}

fn estimate_affine_models(
    tracks: &[Vec<TrackPoint>],
    frame_count: usize,
) -> (Vec<[[f64; 3]; 2]>, usize, Option<f64>) {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut models = vec![identity; frame_count];
    let mut residuals = Vec::new();
    let mut fitted_count = 0usize;
    for (frame_index, model_slot) in models.iter_mut().enumerate().skip(1) {
        let mut reference = Vec::new();
        let mut observed = Vec::new();
        for history in tracks {
            let reference_point = history.iter().find(|point| point.frame_index == 0);
            let observed_point = history
                .iter()
                .find(|point| point.frame_index == frame_index);
            if let (Some(a), Some(b)) = (reference_point, observed_point) {
                reference.push([a.aligned_x, a.aligned_y]);
                observed.push([b.aligned_x, b.aligned_y]);
            }
        }
        if let Some(fit) = fit_affine_ransac(&reference, &observed, 20_260_913 + frame_index as u64)
        {
            *model_slot = fit.model;
            residuals.push(fit.residual_median);
            fitted_count += 1;
        }
    }
    (models, fitted_count, median(&mut residuals))
}

fn inverse_affine(point: [f64; 2], model: &[[f64; 3]; 2]) -> [f64; 2] {
    let determinant = model[0][0] * model[1][1] - model[0][1] * model[1][0];
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return point;
    }
    let x = point[0] - model[0][2];
    let y = point[1] - model[1][2];
    [
        (model[1][1] * x - model[0][1] * y) / determinant,
        (-model[1][0] * x + model[0][0] * y) / determinant,
    ]
}

fn track_point(source: &Detection, frame_index: usize, aligned: [f64; 2]) -> TrackPoint {
    TrackPoint {
        frame_index,
        detection_id: source.detection_id,
        x: source.x,
        y: source.y,
        aligned_x: aligned[0],
        aligned_y: aligned[1],
        flux_snr: source.flux_snr,
        quality_passed: source.quality_passed,
        candidate_snr: None,
    }
}

fn track_to_json(
    track_id: usize,
    classification: &str,
    points: &[TrackPoint],
    metric: &TrackMetrics,
) -> Value {
    json!({
        "track_id":track_id,
        "classification":classification,
        "presence":points.len(),
        "displacement_px":metric.displacement,
        "speed_px_per_frame":metric.speed,
        "fit_rms_px":metric.fit_rms,
        "quality_presence":points.iter().filter(|point| point.quality_passed).count(),
        "evidence_level":"quality",
        "points":points,
    })
}

/// Port of Python `track_detections`: translation registration, consecutive
/// one-to-one assignment, robust affine drift correction, and motion classes.
pub fn track_detections(
    frames: &[Vec<Detection>],
    options: &TrackSequenceOptions,
) -> Result<Value> {
    if frames.is_empty() {
        return Err("frame_sources cannot be empty".into());
    }
    if options.link_radius_px <= 0.0 || options.registration_radius_px <= 0.0 {
        return Err("link and registration radii must be positive".into());
    }
    if options.max_motion_fit_rms_px <= 0.0 {
        return Err("max_motion_fit_rms_px must be positive".into());
    }
    let frame_count = frames.len();
    let required = frame_count.min(
        options
            .min_presence
            .unwrap_or(3usize.max((4 * frame_count).div_ceil(5))),
    );
    if required < 1 || required > frame_count {
        return Err("min_presence must be within the frame count".into());
    }
    let persistent_required = options
        .persistent_min_presence
        .unwrap_or(required.min(3usize.max(frame_count.div_ceil(2))));
    if persistent_required < 1 || persistent_required > required {
        return Err("persistent_min_presence must be within 1..min_presence".into());
    }
    let names = options
        .frame_names
        .clone()
        .unwrap_or_else(|| (0..frame_count).map(|i| i.to_string()).collect());
    if names.len() != frame_count {
        return Err("frame_names length must match frame_sources".into());
    }
    let frame_times = match options.frame_times.as_ref() {
        Some(times)
            if times.len() == frame_count
                && times.iter().all(|v| v.is_finite())
                && times.windows(2).all(|w| w[1] > w[0]) =>
        {
            times.iter().map(|v| v - times[0]).collect::<Vec<_>>()
        }
        _ => (0..frame_count).map(|i| i as f64).collect(),
    };
    let mut shifts = vec![[0.0, 0.0]; frame_count];
    for frame_index in 1..frame_count {
        let delta = estimate_translation(
            &frames[frame_index - 1],
            &frames[frame_index],
            options.registration_radius_px,
        )?;
        shifts[frame_index] = [
            shifts[frame_index - 1][0] + delta[0],
            shifts[frame_index - 1][1] + delta[1],
        ];
    }
    let aligned: Vec<Vec<[f64; 2]>> = frames
        .iter()
        .enumerate()
        .map(|(frame_index, sources)| {
            sources
                .iter()
                .map(|source| {
                    [
                        source.x - shifts[frame_index][0],
                        source.y - shifts[frame_index][1],
                    ]
                })
                .collect()
        })
        .collect();
    let mut tracks: Vec<Vec<TrackPoint>> = frames[0]
        .iter()
        .zip(&aligned[0])
        .map(|(source, point)| vec![track_point(source, 0, *point)])
        .collect();
    for frame_index in 1..frame_count {
        if frames[frame_index].is_empty() {
            continue;
        }
        let eligible = tracks
            .iter()
            .enumerate()
            .filter_map(|(track_index, history)| {
                history
                    .last()
                    .filter(|point| point.frame_index + 1 == frame_index)
                    .map(|last| TrackPosition {
                        track_id: track_index as i64,
                        x: last.aligned_x,
                        y: last.aligned_y,
                    })
            })
            .collect::<Vec<_>>();
        let assignments = global_frame_assignments(
            &eligible,
            &aligned[frame_index],
            options.link_radius_px,
            None,
            None,
        )?;
        let mut assigned_tracks = vec![false; tracks.len()];
        let mut assigned_sources = vec![false; frames[frame_index].len()];
        for assignment in assignments {
            let track_index = assignment.track_id as usize;
            if track_index >= tracks.len()
                || assignment.source_index >= frames[frame_index].len()
                || assigned_tracks[track_index]
                || assigned_sources[assignment.source_index]
            {
                continue;
            }
            tracks[track_index].push(track_point(
                &frames[frame_index][assignment.source_index],
                frame_index,
                aligned[frame_index][assignment.source_index],
            ));
            assigned_tracks[track_index] = true;
            assigned_sources[assignment.source_index] = true;
        }
        for source_index in 0..frames[frame_index].len() {
            if !assigned_sources[source_index] {
                tracks.push(vec![track_point(
                    &frames[frame_index][source_index],
                    frame_index,
                    aligned[frame_index][source_index],
                )]);
            }
        }
    }
    let (models, affine_count, affine_residual_median) =
        estimate_affine_models(&tracks, frame_count);
    let mut metrics = Vec::with_capacity(tracks.len());
    let mut motion_metrics = Vec::with_capacity(tracks.len());
    for history in &tracks {
        let observations = history
            .iter()
            .map(|point| {
                let corrected = inverse_affine(
                    [point.aligned_x, point.aligned_y],
                    &models[point.frame_index],
                );
                TrackObservation {
                    frame_index: point.frame_index as i64,
                    aligned_x: corrected[0],
                    aligned_y: corrected[1],
                }
            })
            .collect::<Vec<_>>();
        let fit = fit_track_motion(&observations, Some(&frame_times))?;
        metrics.push(TrackMetrics {
            displacement: fit.displacement_px,
            speed: fit.speed_px_per_frame,
            fit_rms: fit.fit_rms_px,
        });
        motion_metrics.push(MotionMetric {
            presence: history.len(),
            displacement_px: fit.displacement_px,
            speed_px_per_frame: fit.speed_px_per_frame,
            fit_rms_px: fit.fit_rms_px,
        });
    }
    let effective_floor = derive_motion_displacement_floor(
        &motion_metrics,
        required,
        options.motion_min_displacement_px,
        options.max_motion_fit_rms_px,
        affine_count,
    )?;
    let mut rendered_tracks = Vec::with_capacity(tracks.len());
    let mut stable_count = 0usize;
    let mut persistent_count = 0usize;
    let mut moving_count = 0usize;
    let mut transient_count = 0usize;
    for (track_id, (history, metric)) in tracks.iter().zip(&metrics).enumerate() {
        let classification = if history.len() >= required
            && metric.displacement >= effective_floor
            && metric
                .fit_rms
                .is_some_and(|rms| rms <= options.max_motion_fit_rms_px)
        {
            if frame_count >= 3 {
                moving_count += 1;
                "moving"
            } else {
                stable_count += 1;
                "static"
            }
        } else if history.len() >= required {
            stable_count += 1;
            "static"
        } else if history.len() >= persistent_required
            && metric.displacement < effective_floor
            && metric
                .fit_rms
                .is_some_and(|rms| rms <= options.max_motion_fit_rms_px)
        {
            persistent_count += 1;
            "persistent"
        } else {
            transient_count += 1;
            "transient"
        };
        rendered_tracks.push(track_to_json(track_id, classification, history, metric));
    }
    let frame_json = frames.iter().enumerate().map(|(i, sources)| json!({
        "frame_index":i, "path":names[i], "candidate_count":sources.len(), "returned_count":sources.len(), "quality_count":sources.len(),
        "timestamp":Value::Null, "exposure_ms":Value::Null, "auxiliary":{}, "width_px":Value::Null, "height_px":Value::Null,
    })).collect::<Vec<_>>();
    Ok(json!({
        "frame_count":frame_count,
        "stable_source_count":stable_count,
        "persistent_source_count":persistent_count,
        "stable_field_candidate_count":stable_count+persistent_count,
        "candidate_consensus_count":0,
        "temporal_reference_count":0,
        "stack_faint_count":0,
        "moving_track_count":moving_count,
        "motion_evidence_link_count":0,
        "motion_evidence_group_count":moving_count,
        "fast_point_motion_count":0,
        "transient_track_count":transient_count,
        "link_radius_px":options.link_radius_px,
        "min_presence":required,
        "motion_min_displacement_px":options.motion_min_displacement_px,
        "max_motion_fit_rms_px":options.max_motion_fit_rms_px,
        "motion_line_link_radius_px":96.0,
        "track_association_method":"local_component_hungarian+exposure_velocity_gate+line_geometry_soft_cost+motion_axis_gate+contiguous_run_gate",
        "registration_model":if affine_count > 0 { "translation+robust_affine" } else { "translation" },
        "registration_affine_frame_count":affine_count,
        "registration_residual_median_px":affine_residual_median,
        "effective_motion_min_displacement_px":effective_floor,
        "detector_parameters":{},
        "motion_residual_threshold_adu":100.0,
        "motion_reference_mode":"per_frame_background",
        "motion_frame_audits":[],
        "source_working_limit":Value::Null,
        "calculation_dtype":"float64",
        "background_sample_limit":Value::Null,
        "fast_sequence":false,
        "background_model_mode":"per_frame_local",
        "persistent_min_presence":persistent_required,
        "temporal_reference_candidate_count":0,
        "temporal_coadd_candidate_count":0,
        "temporal_proposal_mode":"median",
        "temporal_candidate_min_snr":7.5,
        "temporal_reference_min_snr":15.0,
        "temporal_multiscale":false,
        "temporal_min_psf_correlation":0.8,
        "candidate_consensus_audit":{},
        "stack_faint_candidate_count":0,
        "stack_reference_mode":"median",
        "stack_threshold_sigma":4.0,
        "stack_min_flux_snr":5.0,
        "stack_frame_min_flux_snr":3.0,
        "stack_min_presence":0,
        "fixed_sentinel_audit":Value::Null,
        "fixed_sentinel_impact_audit":Value::Null,
        "relative_photometry":Value::Null,
        "frames":frame_json,
        "cumulative_shifts":shifts,
        "tracks":rendered_tracks,
        "motion_features":[],
        "motion_evidence_links":[],
    }))
}

#[cfg(test)]
mod numpy_rng_tests {
    use super::NumpyPcg64;

    #[test]
    fn pcg64_raw_stream_matches_numpy() {
        let mut rng = NumpyPcg64::new(20_260_914);
        let raw = (0..5).map(|_| rng.next_u64()).collect::<Vec<_>>();
        assert_eq!(
            raw,
            [
                3_830_804_745_680_267_115,
                8_470_884_136_868_310_837,
                14_068_450_233_165_282_280,
                16_984_099_281_438_361_369,
                11_701_897_475_996_835_555,
            ]
        );
    }

    #[test]
    fn choice_without_replacement_matches_numpy() {
        let mut rng = NumpyPcg64::new(20_260_914);
        assert_eq!(rng.choice_without_replacement(26, 3), [23, 15, 5]);
    }
}
