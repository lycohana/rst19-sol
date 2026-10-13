//! Native contract for the original `rst19` single-frame command.
use crate::{
    catalog_csv,
    coordinates::TangentPlaneWcs,
    fits,
    matching::Projection,
    pipeline::{self, FrameOptions},
    Result,
};
use serde_json::{json, Map, Value};
use std::{io::Write, path::PathBuf};
#[derive(Clone, Copy)]
enum Kind {
    Float,
    Integer,
    Text,
    Flag,
}
// Original argparse option spellings; defaults remain in FrameOptions except
// the CLI-specific hybrid proposal default and two inverse boolean flags.
const FLOATS: &[&str] = &[
    "center-ra",
    "center-dec",
    "pixel-scale-arcsec",
    "rotation-deg",
    "crpix-x",
    "crpix-y",
    "epoch",
    "match-radius-px",
    "threshold-sigma",
    "psf-fwhm",
    "dog-threshold-sigma",
    "dog-min-peak-sigma",
    "starlet-threshold-sigma",
    "starlet-min-peak-sigma",
    "min-flux-snr",
    "min-fwhm",
    "max-fwhm",
    "max-ellipticity",
    "min-sharpness",
    "max-sharpness",
    "gain-e-per-adu",
    "read-noise-adu",
    "zero-point",
    "parallax-zero-point-mas",
    "max-fractional-parallax-error",
];
const INTEGERS: &[&str] = &[
    "parity",
    "min-distance",
    "aperture-radius",
    "max-sources",
    "background-box-size",
    "min-footprint-pixels",
    "min-psf-support-pixels",
    "photometric-color-order",
    "photometric-min-calibrators",
];
const TEXTS: &[&str] = &[
    "catalog",
    "proposal-mode",
    "photometric-system",
    "photometric-band",
    "photometric-color-name",
    "json-out",
];
const FLAGS: &[&str] = &[
    "enable-local-deblend",
    "keep-zero-pixels",
    "keep-linear-artifacts",
    "fit-photometry",
];
fn option(name: &str) -> Result<(&'static str, Kind)> {
    let mut found = Vec::new();
    for (names, kind) in [
        (FLOATS, Kind::Float),
        (INTEGERS, Kind::Integer),
        (TEXTS, Kind::Text),
        (FLAGS, Kind::Flag),
    ] {
        for &candidate in names {
            if candidate == name {
                return Ok((candidate, kind));
            }
            if candidate.starts_with(name) {
                found.push((candidate, kind));
            }
        }
    }
    if found.len() == 1 {
        Ok(found[0])
    } else {
        Err(format!("unknown or ambiguous option --{name}"))
    }
}
pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        let command = std::env::current_exe()
            .ok()
            .and_then(|path| {
                path.file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "rst19".into());
        println!("读取 rst19 FITS，检测星点并按先验 WCS 匹配离线星表\n\nUsage: {command} FITS_PATH [OPTIONS]\n\n-h, --help");
        for names in [FLOATS, INTEGERS, TEXTS, FLAGS] {
            for name in names {
                println!("  --{name}");
            }
        }
        return Ok(());
    }
    let mut fields = Map::new();
    let mut path = None;
    let mut positional_only = false;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--" && !positional_only {
            positional_only = true;
            i += 1;
            continue;
        }
        if !positional_only && arg.starts_with("--") {
            let split = arg[2..].split_once('=');
            let name = split.map(|(n, _)| n).unwrap_or(&arg[2..]);
            let (name, kind) = option(name)?;
            let value = match kind {
                Kind::Flag => {
                    if split.is_some() {
                        return Err(format!("--{name} does not take a value"));
                    }
                    Value::Bool(true)
                }
                _ => {
                    let raw = if let Some((_, v)) = split {
                        v
                    } else {
                        i += 1;
                        args.get(i)
                            .ok_or_else(|| format!("--{name} requires a value"))?
                    };
                    match kind {
                        Kind::Float => {
                            let value = raw
                                .parse::<f64>()
                                .map_err(|_| format!("invalid float for --{name}"))?;
                            if !value.is_finite() {
                                return Err(format!("--{name} must be finite"));
                            }
                            json!(value)
                        }
                        Kind::Integer => json!(raw
                            .parse::<i64>()
                            .map_err(|_| format!("invalid integer for --{name}"))?),
                        Kind::Text => json!(raw),
                        Kind::Flag => unreachable!(),
                    }
                }
            };
            fields.insert(name.replace('-', "_"), value);
        } else if !positional_only && arg.starts_with('-') {
            return Err(format!("unknown argument {arg}"));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err("only one FITS path is allowed".into());
        }
        i += 1;
    }
    let path = path.ok_or("missing FITS path")?;
    let frame = fits::read_fits(&path, false)?;
    let catalog_path = fields.get("catalog").and_then(Value::as_str);
    let catalog = catalog_path
        .map(catalog_csv::load_catalog_csv)
        .transpose()?;
    let triple = ["center_ra", "center_dec", "pixel_scale_arcsec"]
        .map(|k| fields.get(k).and_then(Value::as_f64));
    let wcs = if catalog.is_none() && triple.iter().all(Option::is_none) {
        None
    } else {
        if catalog.is_none() {
            return Err("提供 WCS 参数时必须同时提供 --catalog".into());
        }
        let [Some(ra), Some(dec), Some(scale)] = triple else {
            return Err(
                "--catalog 需要同时提供 --center-ra、--center-dec 和 --pixel-scale-arcsec".into(),
            );
        };
        let parity = fields.get("parity").and_then(Value::as_i64).unwrap_or(1);
        if ![-1, 1].contains(&parity) {
            return Err("parity must be either -1 or 1".into());
        }
        Some(
            TangentPlaneWcs {
                center_ra_deg: ra,
                center_dec_deg: dec,
                pixel_scale_arcsec: scale,
                crpix_x: fields
                    .get("crpix_x")
                    .and_then(Value::as_f64)
                    .unwrap_or(frame.width as f64 / 2.),
                crpix_y: fields
                    .get("crpix_y")
                    .and_then(Value::as_f64)
                    .unwrap_or(frame.height as f64 / 2.),
                rotation_deg: fields
                    .get("rotation_deg")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.),
                parity: parity as i32,
            }
            .validated()?,
        )
    };
    fields.entry("proposal_mode").or_insert(json!("hybrid"));
    if !["gaussian", "hybrid", "ensemble"].contains(&fields["proposal_mode"].as_str().unwrap_or(""))
    {
        return Err("invalid proposal mode".into());
    }
    if fields
        .get("photometric_color_order")
        .and_then(Value::as_i64)
        .is_some_and(|v| !(0..=2).contains(&v))
    {
        return Err("photometric color order must be 0, 1 or 2".into());
    }
    if fields.get("keep_zero_pixels") == Some(&Value::Bool(true)) {
        fields.insert("mask_zero_pixels".into(), json!(false));
    }
    if fields.get("keep_linear_artifacts") == Some(&Value::Bool(true)) {
        fields.insert("reject_linear_artifacts".into(), json!(false));
    }
    let output = fields
        .get("json_out")
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let options: FrameOptions =
        serde_json::from_value(Value::Object(fields)).map_err(|e| e.to_string())?;
    let projection = wcs.clone().map(Projection::Tan);
    let mut value = pipeline::analyze_frame(
        frame,
        catalog.as_deref(),
        projection.as_ref(),
        &options,
        None,
    )?
    .as_dict();
    if let Some(wcs) = wcs {
        value["wcs"] = serde_json::to_value(wcs).map_err(|e| e.to_string())?;
    }
    let mut rendered = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    rendered.push('\n');
    if let Some(output) = output {
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(output, rendered).map_err(|e| e.to_string())?;
    } else {
        std::io::stdout()
            .lock()
            .write_all(rendered.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
