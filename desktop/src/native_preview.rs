use image::{imageops, RgbImage};
use rst19_core::{fits, models::FitsFrame};
use serde_json::{json, Value};
use std::{fs, path::Path};

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let position = (sorted.len().saturating_sub(1) as f64) * fraction;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

pub fn render_preview(path: &Path, out_dir: &Path, mode: &str) -> Result<Value, String> {
    let frame = fits::read_fits(path, false)?;
    render_frame_preview(&frame, out_dir, mode)
}

fn render_frame_preview(frame: &FitsFrame, out_dir: &Path, mode: &str) -> Result<Value, String> {
    let auxiliary_mask = fits::auxiliary_mask(frame.height, frame.width)?;
    let mut valid = (0..frame.data.len())
        .filter_map(|index| {
            let value = (frame.data.value(index) as f32) as f64;
            (!auxiliary_mask[index] && value.is_finite()).then_some(value)
        })
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return Err("图像没有可预览的有限像素".into());
    }
    valid.sort_by(f64::total_cmp);
    let q005 = percentile(&valid, 0.005);
    let q25 = percentile(&valid, 0.25);
    let median = percentile(&valid, 0.5);
    let q75 = percentile(&valid, 0.75);
    let q998 = percentile(&valid, 0.998);
    let noise = ((q75 - q25) / 1.3489795).max(1e-6);
    let (low, high, gamma) = match mode {
        "noise" => (q005.max(median - 2.0 * noise), median + 7.0 * noise, 0.82),
        "enhanced" => (
            q005.max(median - noise),
            q998.max(median + 12.0 * noise),
            0.7,
        ),
        _ => (q005, q998, 1.0),
    };
    let scale = (high - low).max(1.0);
    let mut rgb = Vec::with_capacity(frame.data.len().saturating_mul(3));
    for (index, masked) in auxiliary_mask.iter().enumerate() {
        let value = (frame.data.value(index) as f32) as f64;
        let intensity = (if *masked || value.is_nan() || value == f64::NEG_INFINITY {
            0.0
        } else if value == f64::INFINITY {
            255.0
        } else {
            (((value - low) / scale).clamp(0.0, 1.0).powf(gamma) * 255.0).round()
        }) as u8;
        rgb.extend_from_slice(&[intensity, intensity, intensity]);
    }
    let width = u32::try_from(frame.width).map_err(|error| error.to_string())?;
    let height = u32::try_from(frame.height).map_err(|error| error.to_string())?;
    let image = RgbImage::from_raw(width, height, rgb).ok_or("无法创建预览图像")?;
    let image = if mode == "enhanced" {
        imageops::blur(&image, 0.35)
    } else {
        image
    };
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let target = out_dir.join("preview.png");
    image.save(&target).map_err(|error| error.to_string())?;
    let mut payload = json!({
        "preview": target,
        "width": frame.width,
        "height": frame.height,
        "path": frame.path,
        "header": frame.header,
        "auxiliary": frame.auxiliary.as_ref().map(|value| value.as_dict()),
    });
    if !payload.is_object() {
        payload = json!({});
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_linear_interpolation() {
        assert_eq!(percentile(&[0.0, 10.0, 20.0, 30.0], 0.25), 7.5);
        assert_eq!(percentile(&[5.0], 0.5), 5.0);
    }
}
