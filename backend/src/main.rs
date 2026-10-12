use rst19_core::{
    coordinates::TangentPlaneWcs,
    detection::{self, DetectionOptions},
    filtering::MergedProposal,
    fits,
    measurement::{self, MeasurementContext, MeasurementOptions},
    models::PixelData,
    psf, statistics, Result,
};
use serde_json::{json, Value};
use std::io::{self, BufRead};

fn numbers(value: &Value) -> Result<Vec<f64>> {
    value
        .as_array()
        .ok_or("expected numeric array")?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| "expected finite JSON number".into())
        })
        .collect()
}
fn execute(request: Value) -> Result<Value> {
    match request["operation"].as_str().ok_or("missing operation")? {
        "spatial.compatible_query" => {
            let points = request["points"]
                .as_array()
                .ok_or("missing spatial points")?
                .iter()
                .map(|p| {
                    let p = numbers(p)?;
                    if p.len() != 2 {
                        return Err("spatial point needs two values".into());
                    }
                    Ok([p[0], p[1]])
                })
                .collect::<Result<Vec<_>>>()?;
            let tree = rst19_core::compatible_spatial::CompatibleKdTree::new(&points)?;
            let result = request["queries"]
                .as_array()
                .ok_or("missing spatial queries")?
                .iter()
                .map(|q| {
                    let q = numbers(q)?;
                    if q.len() != 2 {
                        return Err("spatial query needs two values".into());
                    }
                    Ok(tree.nearest([q[0], q[1]]))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(json!({"nearest":result}))
        }
        "fits.inspect" => {
            let path = request["path"].as_str().ok_or("missing FITS path")?;
            let scaled = request["apply_scaling"].as_bool().unwrap_or(false);
            let frame = fits::read_fits(path, scaled)?;
            let samples: Vec<_> = request["indices"]
                .as_array()
                .map(|indices| indices.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_else(|| (0..frame.data.len().min(32) as u64).collect());
            if samples.iter().any(|i| *i >= frame.data.len() as u64) {
                return Err("pixel index out of bounds".into());
            }
            Ok(
                json!({"path":frame.path,"header":frame.header,"image_shape":[frame.height,frame.width],
                "dtype":if scaled {"float64"} else {frame.data.dtype()},"data_offset":frame.data_offset,
                "auxiliary":frame.auxiliary.as_ref().map(|a| a.as_dict()),
                "exposure_seconds":fits::exposure_seconds(&frame.header,1.0)?,"exposure_ms":fits::exposure_milliseconds(&frame.header)?,
                "samples":samples.iter().map(|i| frame.data.exact_value(*i as usize)).collect::<Vec<_>>()}),
            )
        }
        "detection.array" | "detection.fits" => {
            let (data, height, width, default_mask) = if request["operation"] == "detection.fits" {
                let frame =
                    fits::read_fits(request["path"].as_str().ok_or("missing FITS path")?, false)?;
                let mask = if request["mask_auxiliary"].as_bool().unwrap_or(true) {
                    Some(fits::auxiliary_mask(frame.height, frame.width)?)
                } else {
                    None
                };
                (frame.data, frame.height, frame.width, mask)
            } else {
                let height = request["height"].as_u64().ok_or("missing image height")? as usize;
                let width = request["width"].as_u64().ok_or("missing image width")? as usize;
                let values = request["image"].as_array().ok_or("missing image pixels")?;
                let floating: Result<Vec<_>> = values
                    .iter()
                    .map(|v| {
                        if v.is_null() {
                            Ok(f64::NAN)
                        } else {
                            v.as_f64().ok_or("invalid pixel".into())
                        }
                    })
                    .collect();
                let data = match request["dtype"].as_str().unwrap_or("float64") {
                    "float64" | "f64" | ">f8" => PixelData::F64(floating?),
                    "float32" | "f32" | ">f4" => {
                        PixelData::F32(floating?.into_iter().map(|v| v as f32).collect())
                    }
                    "int16" | "i16" | ">i2" => PixelData::I16(
                        values
                            .iter()
                            .map(|v| {
                                v.as_i64()
                                    .and_then(|n| i16::try_from(n).ok())
                                    .ok_or("invalid int16 pixel".into())
                            })
                            .collect::<Result<Vec<_>>>()?,
                    ),
                    "int32" | "i32" | ">i4" => PixelData::I32(
                        values
                            .iter()
                            .map(|v| {
                                v.as_i64()
                                    .and_then(|n| i32::try_from(n).ok())
                                    .ok_or("invalid int32 pixel".into())
                            })
                            .collect::<Result<Vec<_>>>()?,
                    ),
                    "int64" | "i64" | ">i8" => PixelData::I64(
                        values
                            .iter()
                            .map(|v| v.as_i64().ok_or("invalid int64 pixel".into()))
                            .collect::<Result<Vec<_>>>()?,
                    ),
                    "uint8" | "u8" => PixelData::U8(
                        values
                            .iter()
                            .map(|v| {
                                v.as_u64()
                                    .and_then(|n| u8::try_from(n).ok())
                                    .ok_or("invalid uint8 pixel".into())
                            })
                            .collect::<Result<Vec<_>>>()?,
                    ),
                    _ => return Err("unsupported image dtype".into()),
                };
                (data, height, width, None)
            };
            let provided: Option<Vec<bool>> = request["mask"]
                .as_array()
                .map(|m| {
                    m.iter()
                        .map(|v| v.as_bool().ok_or("invalid mask value".into()))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
                .or(default_mask);
            let options: DetectionOptions = serde_json::from_value(
                request.get("options").cloned().unwrap_or_else(|| json!({})),
            )
            .map_err(|e| e.to_string())?;
            let model = if request["background_model"].is_object() {
                Some((
                    numbers(&request["background_model"]["background"])?,
                    numbers(&request["background_model"]["noise"])?,
                ))
            } else {
                None
            };
            let result = detection::detect_sources(
                &data,
                provided.as_deref(),
                height,
                width,
                &options,
                model
                    .as_ref()
                    .map(|(b, n)| detection::BackgroundModel {
                        background: b.as_slice(),
                        noise: n.as_slice(),
                        height: request["background_model"]["height"]
                            .as_u64()
                            .unwrap_or(height as u64) as usize,
                        width: request["background_model"]["width"]
                            .as_u64()
                            .unwrap_or(width as u64) as usize,
                    })
                    .as_ref(),
            )?;
            Ok(result.as_dict())
        }
        "measurement.source" => {
            let image = numbers(&request["image"])?;
            let height = request["height"].as_u64().ok_or("missing height")? as usize;
            let width = request["width"].as_u64().ok_or("missing width")? as usize;
            let invalid: Vec<_> = request["mask"]
                .as_array()
                .map(|m| m.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
                .unwrap_or_else(|| vec![false; image.len()]);
            let line = request["line_mask"].as_array().map(|m| {
                m.iter()
                    .map(|v| v.as_bool().unwrap_or(false))
                    .collect::<Vec<_>>()
            });
            let background = numbers(&request["background"])?;
            let noise = numbers(&request["noise"])?;
            let options: MeasurementOptions = serde_json::from_value(
                request.get("options").cloned().unwrap_or_else(|| json!({})),
            )
            .map_err(|e| e.to_string())?;
            let proposal: MergedProposal =
                serde_json::from_value(request["proposal"].clone()).map_err(|e| e.to_string())?;
            let context = MeasurementContext {
                image: image.as_slice().into(),
                mask: &invalid,
                line_mask: line.as_deref(),
                background_map: background.as_slice().into(),
                noise_map: noise.as_slice().into(),
                height,
                width,
            };
            let result = measurement::source_from_peak(
                &context,
                [proposal.x, proposal.y],
                &options,
                &proposal,
                request["filter_snr"].as_f64().unwrap_or(proposal.score),
            )?;
            serde_json::to_value(result).map_err(|e| e.to_string())
        }
        "psf.pair" => {
            let image = numbers(&request["image"])?;
            let invalid: Vec<_> = request["mask"]
                .as_array()
                .map(|m| m.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
                .unwrap_or_else(|| vec![false; image.len()]);
            let p = numbers(&request["primary"])?;
            let s = numbers(&request["secondary"])?;
            if p.len() != 2 || s.len() != 2 {
                return Err("PSF position requires two coordinates".into());
            }
            let result = psf::pair_psf_evidence(
                &image,
                &invalid,
                request["height"].as_u64().ok_or("missing height")? as usize,
                request["width"].as_u64().ok_or("missing width")? as usize,
                [p[0], p[1]],
                [s[0], s[1]],
                request["psf_fwhm"].as_f64().unwrap_or(3.0),
            )?;
            Ok(json!({"delta_bic":result.0,"component_snr":result.1}))
        }
        "statistics.sigma_clipped" => {
            let values = numbers(&request["values"])?;
            let mask: Option<Vec<bool>> = request["mask"]
                .as_array()
                .map(|m| m.iter().map(|v| v.as_bool().unwrap_or(false)).collect());
            let (center, noise) = statistics::sigma_clipped_stats(
                &values,
                mask.as_deref(),
                request["sigma"].as_f64().unwrap_or(3.0),
                request["iterations"].as_u64().unwrap_or(4) as usize,
                request["sample_limit"].as_u64().unwrap_or(1_000_000) as usize,
            )?;
            Ok(json!({"center":center,"noise":noise}))
        }
        "coordinates.tan" => {
            let wcs: TangentPlaneWcs =
                serde_json::from_value(request["wcs"].clone()).map_err(|e| e.to_string())?;
            let wcs = wcs.validated()?;
            let pairs = request["points"]
                .as_array()
                .ok_or("missing coordinate points")?;
            let mut result = Vec::new();
            for pair in pairs {
                let v = numbers(pair)?;
                if v.len() != 2 {
                    return Err("coordinate needs two values".into());
                }
                let point = match request["direction"].as_str().unwrap_or("world_to_pixel") {
                    "world_to_pixel" => wcs.world_to_pixel(v[0], v[1]),
                    "pixel_to_world" => wcs.pixel_to_world(v[0], v[1]),
                    "world_to_tangent_arcsec" => wcs.world_to_tangent_arcsec(v[0], v[1]),
                    _ => return Err("unknown coordinate direction".into()),
                };
                result.push(point);
            }
            Ok(json!({"points":result,"wcs":wcs}))
        }
        _ => Err("operation has not been migrated to the native engine".into()),
    }
}
fn main() {
    let result = io::stdin()
        .lock()
        .lines()
        .next()
        .ok_or("missing JSON request".to_owned())
        .and_then(|s| s.map_err(|e| e.to_string()))
        .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
        .and_then(execute);
    let (event, code) = match result {
        Ok(payload) => (json!({"event":"done","ok":true,"payload":payload}), 0),
        Err(error) => (json!({"event":"done","ok":false,"error":error}), 1),
    };
    println!("{event}");
    std::process::exit(code);
}
