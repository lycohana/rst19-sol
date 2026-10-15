use eframe::egui::{self, Color32, Pos2, Rect, Vec2};
use serde_json::Value;

pub struct LineGeometry {
    pub endpoints: [Pos2; 2],
    pub width: f32,
    pub indicator_enlarged: bool,
}

fn clip_line(points: [Pos2; 2], rect: Rect) -> Option<[Pos2; 2]> {
    if !rect.is_positive() {
        return None;
    }
    let delta = points[1] - points[0];
    let mut lower = 0.0_f32;
    let mut upper = 1.0_f32;
    for (p, q) in [
        (-delta.x, points[0].x - rect.min.x),
        (delta.x, rect.max.x - points[0].x),
        (-delta.y, points[0].y - rect.min.y),
        (delta.y, rect.max.y - points[0].y),
    ] {
        if p.abs() <= f32::EPSILON {
            if q < 0.0 {
                return None;
            }
        } else if p < 0.0 {
            lower = lower.max(q / p);
        } else {
            upper = upper.min(q / p);
        }
        if lower > upper {
            return None;
        }
    }
    Some([points[0] + delta * lower, points[0] + delta * upper])
}

/// Geometry is in original FITS pixels. Apply the image transform exactly
/// once and clip the whole segment, including when its center is off screen.
pub fn line_geometry(
    point: &Value,
    image: Rect,
    viewport: Rect,
    scale: f32,
    zoom: f32,
    single_frame: bool,
) -> Option<LineGeometry> {
    let finite = |key: &str| point[key].as_f64().filter(|value| value.is_finite());
    let x = finite("x")? as f32;
    let y = finite("y")? as f32;
    let length = finite("length_px")? as f32;
    let width = finite("width_px")? as f32;
    let angle = finite("angle_deg")?.to_radians() as f32;
    if ![x, y, length, width, angle, scale, zoom]
        .iter()
        .all(|value| value.is_finite())
        || length <= 0.0
        || width <= 0.0
        || scale <= 0.0
    {
        return None;
    }
    let center = image.min + Vec2::new(x, y) * scale;
    let stroke_width = (width * scale).clamp(2.0, 10.0);
    let measured_half_length = 0.5 * length * scale;
    let half_length = if single_frame && zoom < 2.5 {
        measured_half_length.max(6.0_f32.max(3.0 * stroke_width))
    } else {
        measured_half_length
    };
    let offset = Vec2::new(angle.cos(), angle.sin()) * half_length;
    let endpoints = clip_line(
        [center - offset, center + offset],
        viewport.intersect(image),
    )?;
    Some(LineGeometry {
        endpoints,
        width: stroke_width,
        indicator_enlarged: half_length > measured_half_length,
    })
}

pub fn distance_to_line(pointer: Pos2, line: &LineGeometry) -> f32 {
    let delta = line.endpoints[1] - line.endpoints[0];
    let t = ((pointer - line.endpoints[0]).dot(delta) / delta.length_sq().max(f32::EPSILON))
        .clamp(0.0, 1.0);
    pointer.distance(line.endpoints[0] + delta * t)
}

/// Project observed positions through the selected raw frame's registration
/// shift. Missing observations split the path rather than inventing a link.
pub fn projected_track_segments(
    track: &Value,
    sequence: &Value,
    frame_index: usize,
    image: Rect,
    viewport: Rect,
    scale: f32,
) -> Vec<LineGeometry> {
    let shift = &sequence["cumulative_shifts"][frame_index];
    let Some(sx) = shift[0].as_f64().filter(|value| value.is_finite()) else {
        return vec![];
    };
    let Some(sy) = shift[1].as_f64().filter(|value| value.is_finite()) else {
        return vec![];
    };
    let Some(points) = track["points"].as_array() else {
        return vec![];
    };
    let mut observed = points
        .iter()
        .filter_map(|point| {
            let frame = point["frame_index"].as_u64()?;
            let x = point["aligned_x"]
                .as_f64()
                .filter(|value| value.is_finite())?;
            let y = point["aligned_y"]
                .as_f64()
                .filter(|value| value.is_finite())?;
            let position = image.min + Vec2::new((x + sx) as f32, (y + sy) as f32) * scale;
            (position.x.is_finite() && position.y.is_finite()).then_some((frame, position))
        })
        .collect::<Vec<_>>();
    observed.sort_by_key(|(frame, _)| *frame);
    observed
        .windows(2)
        .filter_map(|pair| {
            if pair[1].0 != pair[0].0 + 1 {
                return None;
            }
            let endpoints = clip_line([pair[0].1, pair[1].1], viewport.intersect(image))?;
            Some(LineGeometry {
                endpoints,
                width: 2.0,
                indicator_enlarged: false,
            })
        })
        .collect()
}

pub fn draw_line(
    painter: &egui::Painter,
    line: &LineGeometry,
    color: Color32,
    solid: bool,
    label: &str,
) {
    let shadow = Color32::from_rgb(5, 9, 18);
    let stroke = |endpoints, width, color| {
        painter.line_segment(endpoints, (width, color));
    };
    if solid {
        stroke(line.endpoints, line.width + 3.0, shadow);
        stroke(line.endpoints, line.width, color);
    } else {
        let delta = line.endpoints[1] - line.endpoints[0];
        let length = delta.length();
        let direction = delta / length.max(f32::EPSILON);
        let mut start = 0.0;
        while start < length {
            let dash = [
                line.endpoints[0] + direction * start,
                line.endpoints[0] + direction * (start + 6.0).min(length),
            ];
            stroke(dash, line.width + 3.0, shadow);
            stroke(dash, line.width, color);
            start += 10.0;
        }
    }
    if label.is_empty() {
        return;
    }
    let anchor = line.endpoints[0].lerp(line.endpoints[1], 0.5) + Vec2::new(5.0, -6.0);
    painter.text(
        anchor + Vec2::splat(1.0),
        egui::Align2::LEFT_BOTTOM,
        label,
        egui::FontId::monospace(11.0),
        shadow,
    );
    painter.text(
        anchor,
        egui::Align2::LEFT_BOTTOM,
        label,
        egui::FontId::monospace(11.0),
        color,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn exposure_line_uses_one_pixel_transform_and_retains_offscreen_centers() {
        let image = Rect::from_min_size(Pos2::new(-20.0, 10.0), Vec2::splat(200.0));
        let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.0));
        let point = json!({"x":20.0,"y":20.0,"length_px":80.0,"width_px":3.0,"angle_deg":0.0});
        let line = line_geometry(&point, image, viewport, 0.5, 3.0, true).unwrap();
        assert_eq!(
            line.endpoints,
            [Pos2::new(0.0, 20.0), Pos2::new(10.0, 20.0)]
        );
        assert!(distance_to_line(Pos2::new(7.0, 21.0), &line) < 1.01);
        assert!(!line.indicator_enlarged);
    }

    #[test]
    fn overview_indicator_preserves_direction_and_measurements() {
        let point = json!({"x":50.0,"y":50.0,"length_px":10.0,"width_px":3.0,"angle_deg":90.0});
        let original = point.clone();
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.0));
        let line = line_geometry(&point, rect, rect, 0.5, 1.0, true).unwrap();
        assert!(line.indicator_enlarged);
        assert!((line.endpoints[1].x - line.endpoints[0].x).abs() < 0.001);
        assert!((line.endpoints[1].y - line.endpoints[0].y - 12.0).abs() < 0.001);
        assert_eq!(point, original);
        let native = line_geometry(&point, rect, rect, 0.5, 3.0, true).unwrap();
        assert!((native.endpoints[1].distance(native.endpoints[0]) - 5.0).abs() < 0.001);
    }

    #[test]
    fn clipping_rejects_nonintersecting_segments_and_invalid_geometry() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.0));
        assert!(clip_line([Pos2::new(-10.0, 1.0), Pos2::new(1.0, -10.0)], rect).is_none());
        let invalid = json!({"x":50.0,"y":50.0,"length_px":10.0,"width_px":3.0});
        assert!(line_geometry(&invalid, rect, rect, 1.0, 1.0, true).is_none());
    }

    #[test]
    fn sequence_paths_use_selected_frame_shift_and_do_not_bridge_missing_frames() {
        let track = json!({"points":[
            {"frame_index":0,"x":1.0,"y":2.0,"aligned_x":1.0,"aligned_y":2.0},
            {"frame_index":1,"x":14.0,"y":26.0,"aligned_x":4.0,"aligned_y":6.0},
            {"frame_index":3,"x":90.0,"y":95.0,"aligned_x":6.0,"aligned_y":8.0}]});
        let sequence = json!({"cumulative_shifts":[[0,0],[10,20],[30,40],[84,87]]});
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(300.0));
        let segments = projected_track_segments(&track, &sequence, 2, rect, rect, 2.0);
        assert_eq!(segments.len(), 1);
        assert_eq!(
            segments[0].endpoints,
            [Pos2::new(62.0, 84.0), Pos2::new(68.0, 92.0)]
        );
    }
}
