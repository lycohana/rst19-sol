//! Native single-frame analysis and re-calibration without repeating detection.
use crate::{
    catalog::CatalogSource,
    detection::{self, BackgroundModel, DetectionOptions, DetectionResult},
    fits,
    matching::{self, DetectionPosition, MatchResult, Projection},
    models::FitsFrame,
    photometric_calibration::{self, CalibrationOptions, PhotometricCalibration},
    source_photometry::{self, FaintestOptions, FaintestSource, SourcePhotometry, TableOptions},
    Result,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    pub frame: FitsFrame,
    pub detection: DetectionResult,
    pub matching: Option<MatchResult>,
    pub faintest: Option<FaintestSource>,
    pub photometric_calibration: Option<PhotometricCalibration>,
    pub source_photometry: Vec<SourcePhotometry>,
}
impl FrameAnalysis {
    pub fn as_dict(&self) -> Value {
        let header: serde_json::Map<_, _> = [
            "DATE-OBS", "EXPTIME", "EXPOSURE", "BITPIX", "NAXIS1", "NAXIS2", "AZIMUTH", "ELEVATIO",
        ]
        .iter()
        .filter_map(|key| {
            self.frame
                .header
                .get(*key)
                .map(|value| ((*key).to_owned(), value.clone()))
        })
        .collect();
        json!({"path":self.frame.path.to_string_lossy(),"header":header,"auxiliary":self.frame.auxiliary.as_ref().map(|a|a.as_dict()),"detection":self.detection.as_dict(),"matching":self.matching.as_ref(),"faintest_detected":self.faintest.as_ref().map(FaintestSource::as_dict),"photometric_calibration":self.photometric_calibration.as_ref().map(PhotometricCalibration::as_dict),"source_photometry":self.source_photometry.iter().map(SourcePhotometry::as_dict).collect::<Vec<_>>()})
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FrameOptions {
    #[serde(flatten)]
    pub detection: DetectionOptions,
    pub match_radius_px: f64,
    pub epoch: Option<f64>,
    pub zero_point: Option<f64>,
    pub fit_photometry: bool,
    pub photometric_system: Option<String>,
    pub photometric_band: Option<String>,
    pub photometric_color_name: Option<String>,
    pub photometric_color_order: usize,
    pub photometric_min_calibrators: usize,
    pub parallax_zero_point_mas: f64,
    pub max_fractional_parallax_error: f64,
}
impl Default for FrameOptions {
    fn default() -> Self {
        Self {
            detection: DetectionOptions::default(),
            match_radius_px: 3.,
            epoch: None,
            zero_point: None,
            fit_photometry: false,
            photometric_system: None,
            photometric_band: None,
            photometric_color_name: None,
            photometric_color_order: 1,
            photometric_min_calibrators: 6,
            parallax_zero_point_mas: 0.,
            max_fractional_parallax_error: 0.2,
        }
    }
}
fn nonempty(value: Option<String>) -> Option<String> {
    value.filter(|s| !s.is_empty())
}
fn finish(
    analysis: &mut FrameAnalysis,
    catalog: Option<&[CatalogSource]>,
    wcs: Option<&Projection>,
    o: &FrameOptions,
    color_is_resolved: bool,
) -> Result<()> {
    let quality: Vec<_> = analysis
        .detection
        .sources
        .iter()
        .filter(|s| s.quality_passed)
        .cloned()
        .collect();
    let positions: Vec<_> = quality
        .iter()
        .map(|s| DetectionPosition {
            detection_id: s.detection_id,
            x: s.x,
            y: s.y,
        })
        .collect();
    analysis.matching = match catalog {
        Some(c) => Some(matching::match_detections(
            &positions,
            c,
            wcs.ok_or("catalog matching requires a WCS")?,
            o.match_radius_px,
            o.epoch,
            "global",
        )?),
        None => None,
    };
    let exposure = fits::exposure_seconds(&analysis.frame.header, 1.)?;
    let mut colors = BTreeMap::new();
    analysis.photometric_calibration = None;
    if o.fit_photometry {
        if let (Some(m), Some(catalog)) = (&analysis.matching, catalog) {
            let by_id: BTreeMap<_, _> = catalog.iter().map(|s| (&s.source_id, s)).collect();
            for m in &m.matches {
                if let Some(color) = by_id.get(&m.source_id).and_then(|s| s.color) {
                    colors.insert(m.detection_id, color);
                }
            }
            let first = catalog.first();
            let options = CalibrationOptions {
                exposure_s: exposure,
                photometric_system: nonempty(o.photometric_system.clone())
                    .or_else(|| first.and_then(|s| s.photometric_system.clone()))
                    .or_else(|| first.is_none().then(|| "unknown".into())),
                photometric_band: nonempty(o.photometric_band.clone())
                    .or_else(|| first.and_then(|s| s.photometric_band.clone()))
                    .or_else(|| first.is_none().then(|| "unknown".into())),
                color_name: if color_is_resolved {
                    o.photometric_color_name.clone()
                } else {
                    o.photometric_color_name
                        .clone()
                        .or_else(|| first.and_then(|s| s.color_name.clone()))
                },
                color_order: o.photometric_color_order,
                min_calibrators: o.photometric_min_calibrators,
                ..Default::default()
            };
            analysis.photometric_calibration = Some(photometric_calibration::fit(
                &m.matches, &quality, catalog, &options,
            )?);
        }
    }
    analysis.source_photometry = if catalog.is_some() || o.fit_photometry {
        source_photometry::build_table(
            &analysis.detection.sources,
            analysis
                .matching
                .as_ref()
                .map(|m| m.matches.as_slice())
                .unwrap_or(&[]),
            catalog.unwrap_or(&[]),
            analysis.photometric_calibration.as_ref(),
            &TableOptions {
                exposure_s: exposure,
                parallax_zero_point_mas: o.parallax_zero_point_mas,
                max_fractional_parallax_error: o.max_fractional_parallax_error,
            },
        )?
    } else {
        Vec::new()
    };
    let absolute = analysis
        .source_photometry
        .iter()
        .filter_map(|r| r.absolute_magnitude.clone().map(|a| (r.detection_id, a)))
        .collect();
    let rows = analysis.photometric_calibration.as_ref().map(|_| {
        analysis
            .source_photometry
            .iter()
            .cloned()
            .map(|r| (r.detection_id, r))
            .collect()
    });
    analysis.faintest = source_photometry::find_faintest(
        &quality,
        analysis.photometric_calibration.as_ref(),
        &FaintestOptions {
            min_snr: o.detection.measurement.min_flux_snr,
            zero_point: o.zero_point,
            exposure_s: exposure,
            source_colors: Some(colors),
            source_absolute_magnitudes: Some(absolute),
            source_photometry: rows,
        },
    )?;
    Ok(())
}
pub fn analyze_frame(
    frame: FitsFrame,
    catalog: Option<&[CatalogSource]>,
    wcs: Option<&Projection>,
    o: &FrameOptions,
    background: Option<&BackgroundModel<'_>>,
) -> Result<FrameAnalysis> {
    let mut progress = |_: f32, _: &str| {};
    analyze_frame_with_progress(frame, catalog, wcs, o, background, &mut progress)
}

pub fn analyze_frame_with_progress(
    frame: FitsFrame,
    catalog: Option<&[CatalogSource]>,
    wcs: Option<&Projection>,
    o: &FrameOptions,
    background: Option<&BackgroundModel<'_>>,
    progress: &mut dyn FnMut(f32, &str),
) -> Result<FrameAnalysis> {
    if catalog.is_some() && wcs.is_none() {
        return Err("catalog matching requires a WCS".into());
    }
    progress(0.05, "准备图像掩膜");
    let mask = fits::auxiliary_mask(frame.height, frame.width)?;
    progress(0.12, "掩膜完成，开始候选检测与质量筛选");
    let detection = detection::detect_sources(
        &frame.data,
        Some(&mask),
        frame.height,
        frame.width,
        &o.detection,
        background,
    )?;
    progress(0.78, "候选检测完成，整理检测结果");
    let mut analysis = FrameAnalysis {
        frame,
        detection,
        matching: None,
        faintest: None,
        photometric_calibration: None,
        source_photometry: Vec::new(),
    };
    progress(0.82, "计算星表匹配与光度信息");
    finish(&mut analysis, catalog, wcs, o, false)?;
    progress(0.97, "单帧分析结果已整理");
    Ok(analysis)
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RecalibrationOptions {
    pub match_radius_px: f64,
    pub epoch: Option<f64>,
    pub photometric_system: Option<String>,
    pub photometric_band: Option<String>,
    pub photometric_color_name: Option<String>,
    pub photometric_color_order: Option<usize>,
    pub photometric_min_calibrators: usize,
    pub parallax_zero_point_mas: f64,
    pub max_fractional_parallax_error: f64,
}
impl Default for RecalibrationOptions {
    fn default() -> Self {
        Self {
            match_radius_px: 3.,
            epoch: None,
            photometric_system: None,
            photometric_band: None,
            photometric_color_name: None,
            photometric_color_order: None,
            photometric_min_calibrators: 6,
            parallax_zero_point_mas: 0.,
            max_fractional_parallax_error: 0.2,
        }
    }
}
pub fn recalibrate_frame_analysis(
    analysis: &mut FrameAnalysis,
    catalog: &[CatalogSource],
    wcs: &Projection,
    o: &RecalibrationOptions,
) -> Result<()> {
    if catalog.is_empty() {
        return Err("catalog must not be empty".into());
    }
    let previous = analysis.photometric_calibration.as_ref();
    let first = catalog.first();
    let system = nonempty(o.photometric_system.clone()).or_else(|| {
        if let Some(p) = previous {
            p.photometric_system.clone()
        } else {
            first.and_then(|s| s.photometric_system.clone())
        }
    });
    let band = nonempty(o.photometric_band.clone()).or_else(|| {
        if let Some(p) = previous {
            p.photometric_band.clone()
        } else {
            first.and_then(|s| s.photometric_band.clone())
        }
    });
    let color = nonempty(o.photometric_color_name.clone()).or_else(|| {
        if let Some(p) = previous {
            p.color_name.clone()
        } else {
            first.and_then(|s| s.color_name.clone())
        }
    });
    let options = FrameOptions {
        match_radius_px: o.match_radius_px,
        epoch: o.epoch,
        fit_photometry: true,
        photometric_system: Some(nonempty(system).unwrap_or("unknown".into())),
        photometric_band: Some(nonempty(band).unwrap_or("unknown".into())),
        photometric_color_name: color,
        photometric_color_order: o
            .photometric_color_order
            .or_else(|| previous.map(|p| p.color_order))
            .unwrap_or(1),
        photometric_min_calibrators: o.photometric_min_calibrators,
        parallax_zero_point_mas: o.parallax_zero_point_mas,
        max_fractional_parallax_error: o.max_fractional_parallax_error,
        detection: DetectionOptions {
            measurement: crate::measurement::MeasurementOptions {
                min_flux_snr: analysis.detection.parameters["min_flux_snr"]
                    .as_f64()
                    .unwrap_or(5.),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    finish(analysis, Some(catalog), Some(wcs), &options, true)
}
