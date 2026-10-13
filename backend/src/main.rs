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
use std::collections::{BTreeMap, BTreeSet};
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

fn nonnegative_seed(value: Option<&Value>) -> Result<u64> {
    let Some(value) = value else {
        return Ok(1909);
    };
    value
        .as_u64()
        .or_else(|| value.to_string().parse::<u64>().ok())
        .ok_or_else(|| "expected non-negative integer".into())
}

fn execute(request: Value) -> Result<Value> {
    match request["operation"].as_str().ok_or("missing operation")? {
        "catalog.merge_csvs" => {
            let input_paths: Vec<String> = serde_json::from_value(
                request
                    .get("input_paths")
                    .cloned()
                    .ok_or("missing input_paths")?,
            )
            .map_err(|error| error.to_string())?;
            let paths = input_paths
                .into_iter()
                .map(std::path::PathBuf::from)
                .collect::<Vec<_>>();
            let output_path = request["output_path"]
                .as_str()
                .ok_or("missing output_path")?;
            rst19_core::catalog_merge::merge_catalog_csvs(&paths, std::path::Path::new(output_path))
        }
        "audit.feature_evidence_axes" => {
            let matrix_path = request["matrix_path"]
                .as_str()
                .ok_or("missing matrix_path")?;
            Ok(
                rst19_core::feature_evidence_axes::run(std::path::Path::new(matrix_path))?
                    .as_dict(),
            )
        }
        "audit.direct_neighbor_pairs" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing catalog_path")?;
            let target_ids: Vec<i64> = request["target_ids"]
                .as_array()
                .map(|values| {
                    values
                        .iter()
                        .map(|value| value.as_i64().ok_or("target_ids must contain integers"))
                        .collect::<std::result::Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            Ok(rst19_core::direct_neighbor_audit::run(
                std::path::Path::new(catalog_path),
                &target_ids,
                request["psf_fwhm"].as_f64().unwrap_or(2.0),
                request["pair_radius_px"].as_f64(),
                request["delta_bic_min"].as_f64().unwrap_or(10.0),
                request["component_snr_min"].as_f64().unwrap_or(5.0),
            )?
            .as_dict())
        }
        "audit.pair_repair_counterfactual" => {
            rst19_core::pair_repair_counterfactual::handle_request(&request)
        }
        "audit.pair_response_attribution" => {
            rst19_core::pair_response_attribution::handle_request(&request)
        }
        "audit.pair_flux_covariance" => rst19_core::pair_flux_covariance::handle_request(&request),
        "audit.feature_temporal_consistency" => {
            rst19_core::feature_temporal_consistency_audit::handle_request(&request)
        }
        "audit.feature_spatial_context" => {
            rst19_core::feature_spatial_context::handle_request(&request)
        }
        "audit.feature_evidence_matrix" => {
            rst19_core::feature_evidence_matrix::handle_request(&request)
        }
        "audit.feature_psf_leaveout" => rst19_core::feature_psf_leaveout::handle_request(&request),
        "audit.feature_gate_temporal_route" => {
            rst19_core::feature_gate_temporal_route::handle_request(&request)
        }
        "audit.innovation_package" => rst19_core::innovation_package::handle_request(&request),
        "audit.temporal_codes" => rst19_core::temporal_code::handle_request(&request),
        "audit.feature_cross_axis" => {
            let path = |name: &str| -> Result<std::path::PathBuf> {
                request[name]
                    .as_str()
                    .map(std::path::PathBuf::from)
                    .ok_or_else(|| format!("missing {name}"))
            };
            let feature_classes = request
                .get("feature_classes")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<String>>)
                .transpose()
                .map_err(|error| error.to_string())?;
            let detection_ids = request
                .get("detection_ids")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<i64>>)
                .transpose()
                .map_err(|error| error.to_string())?;
            Ok(rst19_core::feature_cross_axis_audit::run(
                &path("catalog_path")?,
                &path("raw_summary_path")?,
                &path("peak_consistency_path")?,
                &path("temporal_sources_path")?,
                feature_classes.as_deref(),
                detection_ids.as_deref(),
            )?
            .as_dict())
        }
        "feedback.append" => {
            let path = request["path"].as_str().ok_or("missing path")?;
            let payload = request.get("feedback").ok_or("missing feedback")?;
            let feedback = rst19_core::feedback::from_value(payload)?;
            let written = rst19_core::feedback::append_manual_feedback(
                std::path::Path::new(path),
                &feedback,
            )?;
            Ok(json!({"path":written, "record":feedback.as_dict()}))
        }
        "feedback.load" => {
            let path = request["path"].as_str().ok_or("missing path")?;
            let records = rst19_core::feedback::load_manual_feedback(std::path::Path::new(path))?;
            Ok(json!({"records":records.iter().map(|record| record.as_dict()).collect::<Vec<_>>()}))
        }
        "audit.source_groups" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let target_ids = request
                .get("target_ids")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<i64>>)
                .transpose()
                .map_err(|error| error.to_string())?
                .unwrap_or_default();
            Ok(rst19_core::source_group_audit::run(
                std::path::Path::new(catalog_path),
                &target_ids,
                request
                    .get("psf_fwhm")
                    .and_then(Value::as_f64)
                    .unwrap_or(2.0),
                request.get("group_radius_px").and_then(Value::as_f64),
                request
                    .get("delta_bic_min")
                    .and_then(Value::as_f64)
                    .unwrap_or(10.0),
                request
                    .get("component_snr_min")
                    .and_then(Value::as_f64)
                    .unwrap_or(5.0),
            )?
            .as_dict())
        }
        "audit.source_group_sensitivity" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let group_radii_px = request
                .get("group_radii_px")
                .filter(|value| !value.is_null())
                .map(numbers)
                .transpose()?
                .unwrap_or_else(|| {
                    rst19_core::source_group_sensitivity::DEFAULT_GROUP_RADII_PX.to_vec()
                });
            let target_ids = request
                .get("target_ids")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<i64>>)
                .transpose()
                .map_err(|error| error.to_string())?
                .unwrap_or_default();
            Ok(rst19_core::source_group_sensitivity::run(
                std::path::Path::new(catalog_path),
                &group_radii_px,
                &target_ids,
                request
                    .get("psf_fwhm")
                    .and_then(Value::as_f64)
                    .unwrap_or(2.0),
                request
                    .get("delta_bic_min")
                    .and_then(Value::as_f64)
                    .unwrap_or(10.0),
                request
                    .get("component_snr_min")
                    .and_then(Value::as_f64)
                    .unwrap_or(5.0),
                request.get("workers").and_then(Value::as_i64).unwrap_or(1),
            )?
            .as_dict())
        }
        "audit.source_separation" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let radii_px = request
                .get("radii_px")
                .filter(|value| !value.is_null())
                .map(numbers)
                .transpose()?
                .unwrap_or_else(|| rst19_core::source_separation_audit::DEFAULT_RADII_PX.to_vec());
            let target_ids = request
                .get("target_ids")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<i64>>)
                .transpose()
                .map_err(|error| error.to_string())?
                .unwrap_or_default();
            Ok(rst19_core::source_separation_audit::run(
                std::path::Path::new(catalog_path),
                &target_ids,
                &radii_px,
            )?
            .as_dict())
        }
        "audit.source_peak_consistency" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let fits_path = request["fits_path"]
                .as_str()
                .ok_or("missing source FITS path")?;
            let targets = request
                .get("target_ids")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<i64>>)
                .transpose()
                .map_err(|error| error.to_string())?
                .unwrap_or_default();
            let radii = request
                .get("local_max_radii_px")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<usize>>)
                .transpose()
                .map_err(|error| error.to_string())?
                .unwrap_or_else(|| {
                    rst19_core::source_peak_consistency::DEFAULT_LOCAL_MAX_RADII.to_vec()
                });
            Ok(rst19_core::source_peak_consistency::run(
                std::path::Path::new(catalog_path),
                std::path::Path::new(fits_path),
                &targets,
                &radii,
            )?
            .as_dict())
        }
        "audit.source_proposal_peak" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let peak_path = request["peak_consistency_path"]
                .as_str()
                .ok_or("missing peak consistency path")?;
            let target_ids = request
                .get("target_ids")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<i64>>)
                .transpose()
                .map_err(|error| error.to_string())?
                .unwrap_or_default();
            Ok(rst19_core::source_proposal_peak_audit::run(
                std::path::Path::new(catalog_path),
                std::path::Path::new(peak_path),
                &target_ids,
            )?
            .as_dict())
        }
        "audit.source_proposal_temporal" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let diagnostic_path = request["diagnostic_sources_path"]
                .as_str()
                .ok_or("missing diagnostic sources path")?;
            let subgroup_path = request["subgroup_persistence_path"]
                .as_str()
                .ok_or("missing subgroup persistence path")?;
            let target_ids = request
                .get("target_ids")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<i64>>)
                .transpose()
                .map_err(|error| error.to_string())?
                .unwrap_or_default();
            Ok(rst19_core::source_proposal_temporal_audit::run(
                std::path::Path::new(catalog_path),
                std::path::Path::new(diagnostic_path),
                std::path::Path::new(subgroup_path),
                &target_ids,
            )?
            .as_dict())
        }
        "audit.source_geometry" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let target_classes = request
                .get("target_classes")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<String>>)
                .transpose()
                .map_err(|error| error.to_string())?;
            let control_class = request["control_class"]
                .as_str()
                .unwrap_or(rst19_core::source_geometry_audit::DEFAULT_CONTROL_CLASS);
            let trials = request
                .get("control_trial_count")
                .map(|value| value.as_u64().ok_or("control_trial_count 必须为正整数"))
                .transpose()?
                .map(|value| usize::try_from(value).map_err(|_| "control_trial_count is too large"))
                .transpose()?
                .unwrap_or(5000);
            let seed = nonnegative_seed(request.get("random_seed"))?;
            let result = rst19_core::source_geometry_audit::run(
                std::path::Path::new(catalog_path),
                target_classes.as_deref(),
                control_class,
                trials,
                seed,
            )?;
            Ok(result.as_dict())
        }
        "attitude.audit.frames" => {
            let frames: Vec<rst19_core::attitude::AttitudeFrameInput> =
                serde_json::from_value(request["frames"].clone()).map_err(|e| e.to_string())?;
            let axis = request
                .get("camera_body_axis")
                .filter(|value| !value.is_null())
                .map(numbers)
                .transpose()?;
            rst19_core::attitude::audit_frames(&frames, axis.as_deref())
        }
        "attitude.audit.paths" => {
            let paths: Vec<String> =
                serde_json::from_value(request["paths"].clone()).map_err(|e| e.to_string())?;
            let axis = request
                .get("camera_body_axis")
                .filter(|value| !value.is_null())
                .map(numbers)
                .transpose()?;
            rst19_core::attitude::audit_paths(&paths, axis.as_deref())
        }
        "audit.feature_flag_profile" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let target_ids: Vec<String> = request
                .get("target_ids")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| error.to_string())?
                .unwrap_or_else(|| {
                    ["82931", "82934", "44132"]
                        .into_iter()
                        .map(str::to_owned)
                        .collect()
                });
            Ok(rst19_core::feature_flag_profile::run(
                std::path::Path::new(catalog_path),
                &target_ids,
            )?
            .as_dict())
        }
        "audit.feature_flag_interaction" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let threshold = request
                .get("high_snr_threshold")
                .and_then(Value::as_f64)
                .unwrap_or(10.0);
            Ok(rst19_core::feature_flag_interaction::run(
                std::path::Path::new(catalog_path),
                threshold,
            )?
            .as_dict())
        }
        "audit.feature_gate_margin" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let thresholds: rst19_core::feature_gate_margin::FeatureGateMarginThresholds =
                serde_json::from_value(
                    request
                        .get("thresholds")
                        .cloned()
                        .unwrap_or_else(|| json!({})),
                )
                .map_err(|error| error.to_string())?;
            Ok(rst19_core::feature_gate_margin::run(
                std::path::Path::new(catalog_path),
                thresholds,
            )?
            .as_dict())
        }
        "audit.feature_gate_route" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            Ok(rst19_core::feature_gate_route::run(std::path::Path::new(catalog_path))?.as_dict())
        }
        "audit.feature_effect_size" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let reference_class = request["reference_class"]
                .as_str()
                .unwrap_or("compact_quality");
            let comparison_classes = request
                .get("comparison_classes")
                .filter(|value| !value.is_null())
                .map(|value| {
                    serde_json::from_value::<Vec<String>>(value.clone())
                        .map_err(|error| error.to_string())
                })
                .transpose()?;
            let features = request
                .get("features")
                .map(|value| {
                    serde_json::from_value::<Vec<String>>(value.clone())
                        .map_err(|error| error.to_string())
                })
                .transpose()?
                .unwrap_or_else(|| {
                    rst19_core::feature_effect_size::DEFAULT_EFFECT_FEATURES
                        .iter()
                        .map(|feature| (*feature).to_owned())
                        .collect()
                });
            let top_n = request.get("top_n").and_then(Value::as_i64).unwrap_or(5);
            if top_n < 1 {
                return Err("top_n must be positive".into());
            }
            Ok(rst19_core::feature_effect_size::run(
                std::path::Path::new(catalog_path),
                reference_class,
                comparison_classes.as_deref(),
                &features,
                top_n as usize,
            )?
            .as_dict())
        }
        "audit.feature_hard_negative" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let top_n = request.get("top_n").and_then(Value::as_u64).unwrap_or(5);
            let metric = request["metric"].as_str().unwrap_or("flux_snr");
            let exclude_classes = request
                .get("exclude_classes")
                .filter(|value| !value.is_null())
                .map(|value| {
                    serde_json::from_value::<Vec<String>>(value.clone())
                        .map_err(|error| error.to_string())
                })
                .transpose()?
                .unwrap_or_else(|| vec!["other_rejected".to_owned()]);
            Ok(rst19_core::feature_hard_negative::run(
                std::path::Path::new(catalog_path),
                top_n as usize,
                metric,
                &exclude_classes,
            )?
            .as_dict())
        }
        "audit.feature_correlation" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let features = request
                .get("features")
                .filter(|value| !value.is_null())
                .map(|value| {
                    serde_json::from_value::<Vec<String>>(value.clone())
                        .map_err(|error| error.to_string())
                })
                .transpose()?
                .unwrap_or_else(|| {
                    rst19_core::feature_correlation_audit::DEFAULT_CORRELATION_FEATURES
                        .iter()
                        .map(|feature| (*feature).to_owned())
                        .collect()
                });
            let threshold = request
                .get("strong_threshold")
                .and_then(Value::as_f64)
                .unwrap_or(0.45);
            Ok(rst19_core::feature_correlation_audit::run(
                std::path::Path::new(catalog_path),
                &features,
                threshold,
            )?
            .as_dict())
        }
        "audit.feature_transition_flow" => {
            let csv_path = request["csv_path"]
                .as_str()
                .ok_or("missing sequence feature transition CSV path")?;
            Ok(
                rst19_core::feature_transition_flow::summarize_sequence_feature_transition_flow(
                    std::path::Path::new(csv_path),
                )?
                .as_dict(),
            )
        }
        "audit.feature_class_context" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let target_ids: Vec<i64> = serde_json::from_value(
                request
                    .get("target_ids")
                    .cloned()
                    .ok_or("missing target detection IDs")?,
            )
            .map_err(|error| error.to_string())?;
            let metric_fields = request
                .get("metric_fields")
                .map(|value| {
                    serde_json::from_value::<Vec<String>>(value.clone())
                        .map_err(|error| error.to_string())
                })
                .transpose()?
                .unwrap_or_else(|| {
                    rst19_core::feature_class_context::CONTEXT_METRIC_FIELDS
                        .iter()
                        .map(|metric| (*metric).to_owned())
                        .collect()
                });
            Ok(rst19_core::feature_class_context::run(
                std::path::Path::new(catalog_path),
                &target_ids,
                &metric_fields,
            )?
            .as_dict())
        }
        "audit.aperture_peak_ratio" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let targets = request
                .get("target_ids")
                .filter(|value| !value.is_null())
                .map(|value| {
                    serde_json::from_value::<Vec<String>>(value.clone())
                        .map_err(|error| error.to_string())
                })
                .transpose()?
                .unwrap_or_else(|| ["82931", "82934"].into_iter().map(str::to_owned).collect());
            Ok(
                rst19_core::aperture_peak_ratio::run(std::path::Path::new(catalog_path), &targets)?
                    .as_dict(),
            )
        }
        "audit.format_paths" => {
            let input_path = request["input_path"]
                .as_str()
                .or_else(|| request["path"].as_str())
                .ok_or("missing input_path")?;
            let extreme_fraction = request
                .get("extreme_fraction")
                .and_then(Value::as_f64)
                .unwrap_or(0.9);
            rst19_core::format_audit::run_format_audit(
                std::path::Path::new(input_path),
                extreme_fraction,
            )
        }
        "sequence.feature_parameter_compare" => {
            let runs: Vec<rst19_core::sequence_feature_comparison::RunPath> =
                serde_json::from_value(request.get("runs").cloned().ok_or("missing runs")?)
                    .map_err(|error| error.to_string())?;
            rst19_core::sequence_feature_comparison::compare(&runs)
        }
        "audit.feature_raw_evidence" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let raw_summary_path = request["raw_summary_path"]
                .as_str()
                .ok_or("missing raw source summary path")?;
            let feature_classes = request
                .get("feature_classes")
                .filter(|value| !value.is_null())
                .map(|value| {
                    serde_json::from_value::<Vec<String>>(value.clone())
                        .map_err(|error| error.to_string())
                })
                .transpose()?;
            let metrics = request
                .get("metrics")
                .filter(|value| !value.is_null())
                .map(|value| {
                    serde_json::from_value::<Vec<String>>(value.clone())
                        .map_err(|error| error.to_string())
                })
                .transpose()?;
            rst19_core::feature_raw_evidence_audit::run(
                std::path::Path::new(catalog_path),
                std::path::Path::new(raw_summary_path),
                feature_classes,
                metrics,
            )
        }
        "audit.pair_mechanism_context" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let target_ids: Vec<i64> = serde_json::from_value(
                request
                    .get("target_ids")
                    .cloned()
                    .ok_or("missing target detection IDs")?,
            )
            .map_err(|error| error.to_string())?;
            let max_centroid = request
                .get("max_centroid_distance_px")
                .and_then(Value::as_f64)
                .unwrap_or(3.5);
            let peak_min = request
                .get("peak_distance_min_px")
                .and_then(Value::as_f64)
                .unwrap_or(3.5);
            let peak_max = request
                .get("peak_distance_max_px")
                .and_then(Value::as_f64)
                .unwrap_or(4.8);
            Ok(rst19_core::pair_mechanism_context::run(
                std::path::Path::new(catalog_path),
                &target_ids,
                max_centroid,
                peak_min,
                peak_max,
            )?
            .as_dict())
        }
        "audit.pair_mechanism_sensitivity" => {
            let catalog_path = request["catalog_path"]
                .as_str()
                .ok_or("missing source catalog path")?;
            let target_ids: Vec<i64> = serde_json::from_value(
                request
                    .get("target_ids")
                    .cloned()
                    .ok_or("missing target detection IDs")?,
            )
            .map_err(|error| error.to_string())?;
            let configurations = request
                .get("configurations")
                .filter(|value| !value.is_null())
                .map(|value| {
                    serde_json::from_value::<Vec<(String, f64, f64, f64)>>(value.clone())
                        .map(|values| {
                            values
                                .into_iter()
                                .map(|(name, centroid, peak_min, peak_max)| {
                                    rst19_core::pair_mechanism_sensitivity::Configuration {
                                        name,
                                        max_centroid_distance_px: centroid,
                                        peak_distance_min_px: peak_min,
                                        peak_distance_max_px: peak_max,
                                    }
                                })
                                .collect()
                        })
                        .map_err(|error| error.to_string())
                })
                .transpose()?
                .unwrap_or_else(rst19_core::pair_mechanism_sensitivity::default_configurations);
            Ok(rst19_core::pair_mechanism_sensitivity::run(
                std::path::Path::new(catalog_path),
                &target_ids,
                &configurations,
            )?
            .as_dict())
        }
        "audit.pair_pixel_topology" => {
            let frame_path = request["frame_path"]
                .as_str()
                .ok_or("missing FITS frame path")?;
            let primary: rst19_core::measurement::Detection = serde_json::from_value(
                request
                    .get("primary_detection")
                    .cloned()
                    .ok_or("missing primary detection")?,
            )
            .map_err(|error| error.to_string())?;
            let secondary: rst19_core::measurement::Detection = serde_json::from_value(
                request
                    .get("secondary_detection")
                    .cloned()
                    .ok_or("missing secondary detection")?,
            )
            .map_err(|error| error.to_string())?;
            let options: rst19_core::pair_pixel_topology::Options = serde_json::from_value(
                request.get("options").cloned().unwrap_or_else(|| json!({})),
            )
            .map_err(|error| error.to_string())?;
            let repeated: Option<Vec<i64>> = request
                .get("repeated_code_values")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| error.to_string())?;
            let frame = rst19_core::fits::read_fits(frame_path, false)?;
            Ok(rst19_core::pair_pixel_topology::run(
                &frame,
                &primary,
                &secondary,
                &options,
                repeated.as_deref(),
            )?
            .as_dict())
        }
        "sequence.track_detections" => {
            let frames: Vec<Vec<rst19_core::measurement::Detection>> =
                serde_json::from_value(request["frames"].clone()).map_err(|e| e.to_string())?;
            let options: rst19_core::sequence::TrackSequenceOptions =
                serde_json::from_value(request.get("options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            rst19_core::sequence::track_detections(&frames, &options)
        }
        "sequence.analyze_core" => rst19_core::sequence_pipeline::handle_request(&request),
        "sequence.fast_point_movers" => {
            let frames: Vec<Vec<rst19_core::measurement::Detection>> =
                serde_json::from_value(request["frames"].clone()).map_err(|e| e.to_string())?;
            let shifts: Vec<[f64; 2]> =
                serde_json::from_value(request["cumulative_shifts"].clone())
                    .map_err(|e| e.to_string())?;
            let times: Option<Vec<f64>> = request
                .get("frame_times")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| e.to_string())?;
            let options: rst19_core::sequence_fast::FastPointOptions =
                serde_json::from_value(request.get("options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            let existing_tracks: Vec<Value> = request
                .get("existing_tracks")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or_default();
            rst19_core::sequence_fast::track_fast_point_movers(
                &frames,
                &shifts,
                times.as_deref(),
                &options,
                &existing_tracks,
            )
        }
        "sequence.fast_candidate_associate" => {
            let frames: Vec<Vec<rst19_core::sequence_fast_candidates::FastCandidateRecord>> =
                serde_json::from_value(request["frames"].clone()).map_err(|e| e.to_string())?;
            let times: Option<Vec<f64>> = request
                .get("frame_times")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| e.to_string())?;
            let options: rst19_core::sequence_fast_candidates::FastCandidateOptions =
                serde_json::from_value(request.get("options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            rst19_core::sequence_fast_candidates::associate_fast_candidates(
                &frames,
                times.as_deref(),
                &options,
            )
        }
        "sequence.candidate_consensus" => {
            let candidates: Vec<Value> =
                serde_json::from_value(request["candidate_peak_frames"].clone())
                    .map_err(|e| e.to_string())?;
            let shifts: Vec<[f64; 2]> =
                serde_json::from_value(request["cumulative_shifts"].clone())
                    .map_err(|e| e.to_string())?;
            let quality_tracks: Vec<Value> = request
                .get("quality_tracks")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or_default();
            let times: Option<Vec<f64>> = request
                .get("frame_times")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| e.to_string())?;
            let options = request.get("options").cloned().unwrap_or(json!({}));
            rst19_core::sequence_consensus::candidate_consensus(
                &candidates,
                &shifts,
                &quality_tracks,
                times.as_deref(),
                options
                    .get("link_radius_px")
                    .and_then(Value::as_f64)
                    .unwrap_or(4.0),
                options
                    .get("min_presence")
                    .and_then(Value::as_u64)
                    .unwrap_or(1) as usize,
                options
                    .get("persistent_min_presence")
                    .and_then(Value::as_u64)
                    .unwrap_or(1) as usize,
                options
                    .get("motion_min_displacement_px")
                    .and_then(Value::as_f64)
                    .unwrap_or(2.0),
                options
                    .get("max_motion_fit_rms_px")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.75),
                options
                    .get("min_candidate_snr")
                    .and_then(Value::as_f64)
                    .unwrap_or(15.0),
                options
                    .get("temporal_candidate_min_snr")
                    .and_then(Value::as_f64),
                options
                    .get("temporal_reference_min_snr")
                    .and_then(Value::as_f64),
                options
                    .get("temporal_min_psf_correlation")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.8),
                request.get("image_support"),
            )
        }
        "sequence.stack_forced_measure_batch" => {
            rst19_core::sequence_photometry::handle_request(&request)
        }
        "sequence.brief" => rst19_core::sequence_brief::handle_request(&request),
        "photometry.relative.fit" => rst19_core::relative_photometry::handle_request(&request),
        "sequence.temporal_reference_candidates" => {
            rst19_core::sequence_temporal::temporal_reference_candidates(&request)
        }
        "sequence.line_profile_metrics" => {
            rst19_core::sequence_trails::line_profile_metrics(&request)
        }
        "sequence.trail_shape_evidence" => {
            rst19_core::sequence_trails::trail_shape_evidence(&request)
        }
        "sequence.motion_feature_track_gates" => {
            rst19_core::sequence_trails::motion_feature_track_gates(&request)
        }
        "sequence.motion_feature_continuity_gates" => {
            rst19_core::sequence_trails::motion_feature_continuity_gates(&request)
        }
        "sequence.motion_feature_exposure_evidence" => {
            rst19_core::sequence_trails::motion_feature_exposure_evidence(&request)
        }
        "sequence.motion_feature_confirmation_evidence" => {
            rst19_core::sequence_trails::motion_feature_confirmation_evidence(&request)
        }
        "sequence.motion_proposal_thresholds" => {
            rst19_core::sequence_trails::motion_proposal_thresholds(&request)
        }
        "sequence.motion_time_axis" => rst19_core::sequence_trails::motion_time_axis(&request),
        "sequence.observation_time_axis" => {
            rst19_core::sequence_trails::observation_time_axis(&request)
        }
        "sequence.motion_feature_pair_geometry" => {
            rst19_core::sequence_trails::motion_feature_pair_geometry_evidence(&request)
        }
        "sequence.fit_collinear_motion_geometry" => {
            rst19_core::sequence_trails::fit_collinear_motion_geometry(&request)
        }
        "sequence.best_collinear_motion_feature_group" => {
            rst19_core::sequence_trails::best_collinear_motion_feature_group(&request)
        }
        "sequence.stitch_unassociated_single_frame_points" => {
            rst19_core::sequence_trails::stitch_unassociated_single_frame_points(&request)
        }
        "sequence.link_motion_evidence_tracks" => {
            rst19_core::sequence_trails::link_motion_evidence_tracks(&request)
        }
        "sequence.rescue_collinear_motion_feature_groups" => {
            rst19_core::sequence_trails::rescue_collinear_motion_feature_groups(&request)
        }
        "sequence.motion_feature_frame_components" => {
            rst19_core::sequence_motion_features::motion_feature_frame_components(&request)
        }
        "sequence.coherent_raw_motion_point_indices" => {
            rst19_core::sequence_trails::coherent_raw_motion_point_indices(&request)
        }
        "trails.local_pixel_proposals" => {
            rst19_core::single_frame_trails::local_pixel_trail_proposals(&request)
        }
        "sequence.reference.median" => {
            rst19_core::sequence_temporal::registered_median_reference(&request)
        }
        "sequence.reference.coadd" => {
            rst19_core::sequence_temporal::registered_coadd_reference(&request)
        }
        "sequence.frame_assign" => {
            let raw_tracks: Vec<(i64, f64, f64)> =
                serde_json::from_value(request["track_positions"].clone())
                    .map_err(|e| e.to_string())?;
            let tracks: Vec<_> = raw_tracks
                .iter()
                .map(
                    |(track_id, x, y)| rst19_core::sequence_assignment::TrackPosition {
                        track_id: *track_id,
                        x: *x,
                        y: *y,
                    },
                )
                .collect();
            let current: Vec<[f64; 2]> = serde_json::from_value(request["current_points"].clone())
                .map_err(|e| e.to_string())?;
            let allowed_pairs: Option<BTreeSet<(i64, usize)>> = request
                .get("allowed_pairs")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<(i64, usize)>>)
                .transpose()
                .map_err(|e| e.to_string())?
                .map(|pairs| pairs.into_iter().collect());
            let costs: Option<BTreeMap<(i64, usize), f64>> = request
                .get("pair_costs")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value::<Vec<(i64, usize, f64)>>)
                .transpose()
                .map_err(|e| e.to_string())?
                .map(|items| {
                    items
                        .into_iter()
                        .map(|(track_id, source_index, cost)| ((track_id, source_index), cost))
                        .collect()
                });
            let allowed = |track_id, source_index, _distance| {
                allowed_pairs
                    .as_ref()
                    .is_none_or(|pairs| pairs.contains(&(track_id, source_index)))
            };
            let pair_cost = |track_id, source_index, _distance| {
                costs
                    .as_ref()
                    .and_then(|values| values.get(&(track_id, source_index)))
                    .copied()
                    .unwrap_or(0.0)
            };
            let result = rst19_core::sequence_assignment::global_frame_assignments(
                &tracks,
                &current,
                request["link_radius_px"].as_f64().unwrap_or(4.0),
                allowed_pairs
                    .as_ref()
                    .map(|_| &allowed as &dyn Fn(i64, usize, f64) -> bool),
                costs
                    .as_ref()
                    .map(|_| &pair_cost as &dyn Fn(i64, usize, f64) -> f64),
            )?;
            Ok(json!({
                "assignments":result.iter().map(|item|Value::Array(vec![
                    json!(item.distance_px),
                    json!(item.track_id),
                    json!(item.source_index),
                ])).collect::<Vec<_>>()
            }))
        }
        "sequence.translation.estimate" => {
            let previous: Vec<rst19_core::measurement::Detection> =
                serde_json::from_value(request["previous"].clone()).map_err(|e| e.to_string())?;
            let current: Vec<rst19_core::measurement::Detection> =
                serde_json::from_value(request["current"].clone()).map_err(|e| e.to_string())?;
            let delta = rst19_core::sequence_registration::estimate_translation(
                &previous,
                &current,
                request["radius_px"].as_f64().unwrap_or(8.0),
            )?;
            Ok(json!({"translation_px":delta}))
        }
        "sequence.track_motion.fit" => {
            let points: Vec<rst19_core::sequence_motion::TrackObservation> =
                serde_json::from_value(request["points"].clone()).map_err(|e| e.to_string())?;
            let frame_times: Option<Vec<f64>> = request
                .get("frame_times")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| e.to_string())?;
            Ok(
                rst19_core::sequence_motion::fit_track_motion(&points, frame_times.as_deref())?
                    .as_dict(),
            )
        }
        "sequence.motion_displacement_floor" => {
            let metrics: Vec<rst19_core::sequence_motion::MotionMetric> =
                serde_json::from_value(request["metrics"].clone()).map_err(|e| e.to_string())?;
            let floor = rst19_core::sequence_motion::derive_motion_displacement_floor(
                &metrics,
                request["required_presence"]
                    .as_u64()
                    .ok_or("missing required_presence")? as usize,
                request["user_floor"].as_f64().ok_or("missing user_floor")?,
                request["max_motion_fit_rms_px"]
                    .as_f64()
                    .ok_or("missing max_motion_fit_rms_px")?,
                request["affine_frame_count"]
                    .as_u64()
                    .ok_or("missing affine_frame_count")? as usize,
            )?;
            Ok(json!({"effective_motion_min_displacement_px":floor}))
        }
        "wcs.affine.fit" => {
            let matches: Vec<rst19_core::matching::CatalogMatch> =
                serde_json::from_value(request["matches"].clone()).map_err(|e| e.to_string())?;
            let catalog: Vec<rst19_core::catalog::CatalogSource> =
                serde_json::from_value(request["catalog"].clone()).map_err(|e| e.to_string())?;
            let reference_wcs: TangentPlaneWcs = serde_json::from_value(
                request.get("reference_wcs").cloned().unwrap_or(Value::Null),
            )
            .map_err(|e| e.to_string())?;
            let options: rst19_core::wcs_fit::AffineFitOptions =
                serde_json::from_value(request.get("fit_options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            Ok(rst19_core::wcs_fit::fit_affine_wcs_from_matches(
                &matches,
                &catalog,
                &reference_wcs,
                request.get("epoch").and_then(Value::as_f64),
                options,
            )?
            .as_dict())
        }
        "wcs.affine.refined" => {
            let calibration: rst19_core::wcs_fit::AffineWcsCalibration =
                serde_json::from_value(request["calibration"].clone())
                    .map_err(|e| e.to_string())?;
            Ok(json!({"wcs":calibration.refined_wcs().validated()?}))
        }
        "wcs.validate.matches" => {
            let frame_matches: Vec<Vec<rst19_core::matching::CatalogMatch>> =
                serde_json::from_value(request["frame_matches"].clone())
                    .map_err(|e| e.to_string())?;
            let catalog: Vec<rst19_core::catalog::CatalogSource> =
                serde_json::from_value(request["catalog"].clone()).map_err(|e| e.to_string())?;
            let reference_wcs: TangentPlaneWcs = serde_json::from_value(
                request.get("reference_wcs").cloned().unwrap_or(Value::Null),
            )
            .map_err(|e| e.to_string())?;
            let options: rst19_core::wcs_fit::AffineFitOptions =
                serde_json::from_value(request.get("fit_options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            let frame_paths: Option<Vec<String>> = request
                .get("frame_paths")
                .filter(|v| !v.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| e.to_string())?;
            rst19_core::wcs_fit::validate_frame_matches(
                &frame_matches,
                &catalog,
                &reference_wcs,
                frame_paths.as_deref(),
                request.get("epoch").and_then(Value::as_f64),
                options,
            )
        }
        "wcs.validate.sequence" => {
            let frame_paths: Vec<String> =
                serde_json::from_value(request.get("frame_paths").cloned().unwrap_or(Value::Null))
                    .map_err(|e| e.to_string())?;
            let catalog: Vec<rst19_core::catalog::CatalogSource> =
                serde_json::from_value(request["catalog"].clone()).map_err(|e| e.to_string())?;
            let reference_wcs: TangentPlaneWcs = serde_json::from_value(
                request.get("reference_wcs").cloned().unwrap_or(Value::Null),
            )
            .map_err(|e| e.to_string())?;
            let options: rst19_core::pipeline::FrameOptions =
                serde_json::from_value(request.get("options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            let fit_options: rst19_core::wcs_fit::AffineFitOptions =
                serde_json::from_value(request.get("fit_options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            rst19_core::wcs_fit::validate_sequence_paths(
                &frame_paths,
                &catalog,
                &reference_wcs,
                &options,
                request["match_radius_px"].as_f64().unwrap_or(3.0),
                request.get("epoch").and_then(Value::as_f64),
                fit_options,
            )
        }
        "photometry.table" | "photometry.faintest" => {
            let sources: Vec<rst19_core::measurement::Detection> =
                serde_json::from_value(request["sources"].clone()).map_err(|e| e.to_string())?;
            let calibration: Option<rst19_core::photometric_calibration::PhotometricCalibration> =
                serde_json::from_value(request.get("calibration").cloned().unwrap_or(Value::Null))
                    .map_err(|e| e.to_string())?;
            let options = request.get("options").cloned().unwrap_or(json!({}));
            if request["operation"] == "photometry.table" {
                let matches: Vec<rst19_core::matching::CatalogMatch> =
                    serde_json::from_value(request.get("matches").cloned().unwrap_or(json!([])))
                        .map_err(|e| e.to_string())?;
                let catalog: Vec<rst19_core::catalog::CatalogSource> =
                    serde_json::from_value(request.get("catalog").cloned().unwrap_or(json!([])))
                        .map_err(|e| e.to_string())?;
                let options = serde_json::from_value(options).map_err(|e| e.to_string())?;
                Ok(Value::Array(
                    rst19_core::source_photometry::build_table(
                        &sources,
                        &matches,
                        &catalog,
                        calibration.as_ref(),
                        &options,
                    )?
                    .iter()
                    .map(|r| r.as_dict())
                    .collect(),
                ))
            } else {
                let options = serde_json::from_value(options).map_err(|e| e.to_string())?;
                Ok(rst19_core::source_photometry::find_faintest(
                    &sources,
                    calibration.as_ref(),
                    &options,
                )?
                .map(|r| r.as_dict())
                .unwrap_or(Value::Null))
            }
        }
        "pipeline.analyze" => {
            let frame =
                fits::read_fits(request["path"].as_str().ok_or("missing FITS path")?, false)?;
            let model = match request.get("background_model") {
                None | Some(Value::Null) => None,
                Some(value) if value.is_object() => Some((
                    numbers(&value["background"])?,
                    numbers(&value["noise"])?,
                    value["height"].as_u64().unwrap_or(frame.height as u64) as usize,
                    value["width"].as_u64().unwrap_or(frame.width as u64) as usize,
                )),
                _ => return Err("invalid background model".into()),
            };
            let background = model
                .as_ref()
                .map(|(b, n, h, w)| detection::BackgroundModel {
                    background: b,
                    noise: n,
                    height: *h,
                    width: *w,
                });
            let catalog: Option<Vec<rst19_core::catalog::CatalogSource>> =
                if let Some(path) = request["catalog_path"].as_str() {
                    Some(rst19_core::catalog_csv::load_catalog_csv(path)?)
                } else {
                    serde_json::from_value(request.get("catalog").cloned().unwrap_or(Value::Null))
                        .map_err(|e| e.to_string())?
                };
            let wcs: Option<rst19_core::matching::Projection> =
                serde_json::from_value(request.get("wcs").cloned().unwrap_or(Value::Null))
                    .map_err(|e| e.to_string())?;
            let options =
                serde_json::from_value(request.get("options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            Ok(rst19_core::pipeline::analyze_frame(
                frame,
                catalog.as_deref(),
                wcs.as_ref(),
                &options,
                background.as_ref(),
            )?
            .as_dict())
        }
        "photometry.calibration.fit" => {
            let matches: Vec<rst19_core::matching::CatalogMatch> =
                serde_json::from_value(request["matches"].clone()).map_err(|e| e.to_string())?;
            let detections: Vec<rst19_core::measurement::Detection> =
                serde_json::from_value(request["detections"].clone()).map_err(|e| e.to_string())?;
            let catalog: Vec<rst19_core::catalog::CatalogSource> =
                serde_json::from_value(request["catalog"].clone()).map_err(|e| e.to_string())?;
            let options =
                serde_json::from_value(request.get("options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            Ok(
                rst19_core::photometric_calibration::fit(
                    &matches,
                    &detections,
                    &catalog,
                    &options,
                )?
                .as_dict(),
            )
        }
        "photometry.calibration.sensitivity" => {
            let matches: Vec<rst19_core::matching::CatalogMatch> =
                serde_json::from_value(request["matches"].clone()).map_err(|e| e.to_string())?;
            let detections: Vec<rst19_core::measurement::Detection> =
                serde_json::from_value(request["detections"].clone()).map_err(|e| e.to_string())?;
            let catalog: Vec<rst19_core::catalog::CatalogSource> =
                serde_json::from_value(request["catalog"].clone()).map_err(|e| e.to_string())?;
            let options: rst19_core::calibration_sensitivity::SensitivityOptions =
                serde_json::from_value(request.get("options").cloned().unwrap_or(json!({})))
                    .map_err(|e| e.to_string())?;
            rst19_core::calibration_sensitivity::run(&matches, &detections, &catalog, &options)
        }
        "photometry.calibration.apply" | "photometry.calibration.error" => {
            let calibration: rst19_core::photometric_calibration::PhotometricCalibration =
                serde_json::from_value(request["calibration"].clone())
                    .map_err(|e| e.to_string())?;
            let value = if request["operation"] == "photometry.calibration.apply" {
                calibration.apply(
                    request["instrumental"]
                        .as_f64()
                        .ok_or("missing instrumental magnitude")?,
                    request["color"].as_f64(),
                )
            } else {
                calibration.magnitude_error(
                    request["instrumental_error"].as_f64(),
                    request["color_error"].as_f64(),
                )
            };
            Ok(json!({"value":value}))
        }
        "photometry.evaluate" => rst19_core::photometry::evaluate(
            request["kind"].as_str().ok_or("missing photometric kind")?,
            &request["parameters"],
            request.get("source"),
        ),
        "catalog.load" => {
            let sources = rst19_core::catalog_csv::load_catalog_csv(
                request["path"].as_str().ok_or("missing catalog path")?,
            )?;
            Ok(json!({"sources":sources}))
        }
        "catalog.propagate" => {
            let source: rst19_core::catalog::CatalogSource =
                serde_json::from_value(request["source"].clone()).map_err(|e| e.to_string())?;
            let source = source.validated()?.at_epoch(request["epoch"].as_f64())?;
            Ok(
                json!({"extinction_compatibility":source.extinction_compatibility(),"source":source}),
            )
        }
        "matching.catalog" => {
            let detections: Vec<rst19_core::matching::DetectionPosition> =
                serde_json::from_value(request["detections"].clone()).map_err(|e| e.to_string())?;
            let catalog = if let Some(path) = request["catalog_path"].as_str() {
                rst19_core::catalog_csv::load_catalog_csv(path)?
            } else {
                let sources: Vec<rst19_core::catalog::CatalogSource> =
                    serde_json::from_value(request["catalog"].clone())
                        .map_err(|e| e.to_string())?;
                sources
                    .into_iter()
                    .map(rst19_core::catalog::CatalogSource::validated)
                    .collect::<Result<Vec<_>>>()?
            };
            let wcs = serde_json::from_value(request["wcs"].clone()).map_err(|e| e.to_string())?;
            let result = rst19_core::matching::match_detections(
                &detections,
                &catalog,
                &wcs,
                request["radius"].as_f64().unwrap_or(3.),
                request["epoch"].as_f64(),
                request["mode"].as_str().unwrap_or("global"),
            )?;
            serde_json::to_value(result).map_err(|e| e.to_string())
        }
        "assignment.solve" => {
            let cost = numbers(&request["cost"])?;
            let result = rst19_core::assignment::linear_sum_assignment(
                &cost,
                request["rows"].as_u64().ok_or("missing rows")? as usize,
                request["columns"].as_u64().ok_or("missing columns")? as usize,
            )?;
            Ok(json!({"assignment": result}))
        }
        "matching.assign" => {
            let candidates: Vec<rst19_core::assignment::CandidateEdge> =
                serde_json::from_value(request["edges"].clone()).map_err(|e| e.to_string())?;
            let result = rst19_core::assignment::assign_candidates(
                &candidates,
                request["detection_count"]
                    .as_u64()
                    .ok_or("missing detection_count")? as usize,
                request["catalog_count"]
                    .as_u64()
                    .ok_or("missing catalog_count")? as usize,
                request["radius"].as_f64().unwrap_or(3.),
                request["mode"].as_str().unwrap_or("global"),
            )?;
            Ok(json!({"matches": result}))
        }
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
