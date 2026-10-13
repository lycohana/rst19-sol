//! Sequence-wide fixed-sentinel diagnostics. These reports preserve raw data
//! quality evidence and never modify detection masks or track classifications.
use crate::{fits, models::FitsFrame, pipeline::FrameAnalysis, spatial::KdTree, Result};
use serde_json::{json, Value};

pub fn audit_fixed_sentinel(
    frames: &[FitsFrame],
    sentinel_value: i64,
    max_coordinates: usize,
) -> Result<Value> {
    let Some(first) = frames.first() else {
        return Ok(json!({
            "sentinel_value":sentinel_value,
            "frame_count":0,
            "image_shape":null,
            "same_shape":true,
            "total_occurrences":0,
            "per_frame_occurrences":[],
            "fixed_coordinate_count":0,
            "fixed_coordinates":[],
            "coordinates_truncated":false,
        }));
    };
    let first_shape = [first.height, first.width];
    let first_size = first
        .height
        .checked_mul(first.width)
        .ok_or("image shape overflow in fixed-sentinel audit")?;
    if first.data.len() != first_size {
        return Err("fixed-sentinel image shape does not match its data".into());
    }
    let mut same_shape = true;
    let mut fixed_mask: Option<Vec<bool>> = None;
    let mut per_frame_occurrences = Vec::with_capacity(frames.len());
    let mut total_occurrences = 0usize;
    for frame in frames {
        let size = frame
            .height
            .checked_mul(frame.width)
            .ok_or("image shape overflow in fixed-sentinel audit")?;
        if frame.data.len() != size {
            return Err("fixed-sentinel image shape does not match its data".into());
        }
        let frame_shape = [frame.height, frame.width];
        if frame_shape != first_shape {
            same_shape = false;
        }
        let auxiliary = fits::auxiliary_mask(frame.height, frame.width)?;
        let mut current = Vec::with_capacity(size);
        let mut frame_count = 0usize;
        for (index, masked) in auxiliary.into_iter().enumerate() {
            let matches = !masked && frame.data.value(index) == sentinel_value as f64;
            current.push(matches);
            frame_count += usize::from(matches);
        }
        total_occurrences += frame_count;
        per_frame_occurrences.push(frame_count);
        if same_shape {
            if let Some(fixed) = fixed_mask.as_mut() {
                for (kept, found) in fixed.iter_mut().zip(current) {
                    *kept &= found;
                }
            } else {
                fixed_mask = Some(current);
            }
        } else {
            fixed_mask = None;
        }
    }

    let mut fixed_coordinates = Vec::<[usize; 2]>::new();
    let mut fixed_coordinate_count = 0usize;
    if same_shape {
        if let Some(mask) = fixed_mask {
            for (index, fixed) in mask.into_iter().enumerate() {
                if !fixed {
                    continue;
                }
                fixed_coordinate_count += 1;
                if fixed_coordinates.len() < max_coordinates {
                    fixed_coordinates.push([index % first.width, index / first.width]);
                }
            }
        }
    }
    Ok(json!({
        "sentinel_value":sentinel_value,
        "frame_count":frames.len(),
        "image_shape":first_shape,
        "same_shape":same_shape,
        "total_occurrences":total_occurrences,
        "per_frame_occurrences":per_frame_occurrences,
        "fixed_coordinate_count":fixed_coordinate_count,
        "fixed_coordinates":fixed_coordinates,
        "coordinates_truncated":fixed_coordinate_count > fixed_coordinates.len(),
    }))
}

pub struct FixedSentinelImpactAccumulator {
    frame_count: usize,
    fixed_tree: Option<KdTree>,
    affected_frame_count: usize,
    candidate_peak_data_available: bool,
    candidate_peak_frame_count: usize,
    candidate_peak_count: usize,
    candidate_peak_affected_count: usize,
    affected_returned_source_count: usize,
    affected_quality_source_count: usize,
    records: Vec<Value>,
    record_count: usize,
    max_records: usize,
}

impl FixedSentinelImpactAccumulator {
    pub fn new(frame_count: usize, fixed_audit: &Value, max_records: usize) -> Result<Self> {
        let fixed_coordinates =
            if let Some(coordinates) = fixed_audit["fixed_coordinates"].as_array() {
                coordinates
                    .iter()
                    .map(|coordinate| {
                        let pair = coordinate
                            .as_array()
                            .filter(|pair| pair.len() == 2)
                            .ok_or("fixed-sentinel coordinates must contain x/y pairs")?;
                        let x = pair[0]
                            .as_u64()
                            .ok_or("fixed-sentinel x coordinate must be non-negative")?
                            as f64;
                        let y = pair[1]
                            .as_u64()
                            .ok_or("fixed-sentinel y coordinate must be non-negative")?
                            as f64;
                        Ok([x, y])
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
        let fixed_tree = (!fixed_coordinates.is_empty())
            .then(|| KdTree::new(&fixed_coordinates))
            .transpose()?;
        Ok(Self {
            frame_count,
            fixed_tree,
            affected_frame_count: 0,
            candidate_peak_data_available: false,
            candidate_peak_frame_count: 0,
            candidate_peak_count: 0,
            candidate_peak_affected_count: 0,
            affected_returned_source_count: 0,
            affected_quality_source_count: 0,
            records: Vec::new(),
            record_count: 0,
            max_records,
        })
    }

    pub fn add_frame(&mut self, frame_index: usize, analysis: &FrameAnalysis) -> Result<()> {
        let parameters = &analysis.detection.parameters;
        let aperture_radius = parameters
            .get("aperture_radius")
            .and_then(Value::as_f64)
            .filter(|radius| radius.is_finite() && *radius > 0.0)
            .unwrap_or(4.0);
        let mut frame_affected = false;

        self.candidate_peak_data_available = true;
        self.candidate_peak_frame_count += 1;
        self.candidate_peak_count += analysis.detection.candidate_peaks.len();
        if let Some(tree) = self.fixed_tree.as_ref() {
            for candidate in &analysis.detection.candidate_peaks {
                let point = [candidate[0], candidate[1]];
                if tree
                    .nearest(point)
                    .is_some_and(|(_, distance)| distance <= aperture_radius)
                {
                    self.candidate_peak_affected_count += 1;
                    frame_affected = true;
                }
            }
            for source in &analysis.detection.sources {
                let peak_x = source
                    .peak_x
                    .filter(|value| value.is_finite())
                    .unwrap_or(source.x);
                let peak_y = source
                    .peak_y
                    .filter(|value| value.is_finite())
                    .unwrap_or(source.y);
                let peak_position = [peak_x, peak_y];
                let nearby = tree.within_radius(peak_position, aperture_radius)?;
                if nearby.is_empty() {
                    continue;
                }
                let nearest_distance = tree
                    .nearest(peak_position)
                    .map(|(_, distance)| distance)
                    .unwrap_or(f64::INFINITY);
                self.record_count += 1;
                self.affected_returned_source_count += 1;
                self.affected_quality_source_count += usize::from(source.quality_passed);
                frame_affected = true;
                if self.records.len() < self.max_records {
                    self.records.push(json!({
                        "frame_index":frame_index,
                        "detection_id":source.detection_id,
                        "x":source.x,
                        "y":source.y,
                        "peak_x":peak_x,
                        "peak_y":peak_y,
                        "nearest_fixed_distance_px":nearest_distance,
                        "fixed_coordinate_count":nearby.len(),
                        "peak":source.peak,
                        "flux_snr":source.flux_snr,
                        "quality_passed":source.quality_passed,
                        "flags":source.flags,
                    }));
                }
            }
        }
        self.affected_frame_count += usize::from(frame_affected);
        Ok(())
    }

    pub fn finish(self) -> Value {
        json!({
            "frame_count":self.frame_count,
            "affected_frame_count":self.affected_frame_count,
            "candidate_peak_data_available":self.candidate_peak_data_available,
            "candidate_peak_frame_count":self.candidate_peak_frame_count,
            "candidate_peak_count":self.candidate_peak_count,
            "candidate_peak_affected_count":self.candidate_peak_affected_count,
            "affected_returned_source_count":self.affected_returned_source_count,
            "affected_quality_source_count":self.affected_quality_source_count,
            "record_count":self.record_count,
            "records_truncated":self.record_count > self.records.len(),
            "records":self.records,
        })
    }
}
