//! Sensitivity of detector-level source grouping to the chosen centroid radius.
//!
//! Every radius reuses the same source catalog and evidence thresholds. The
//! scan describes how the candidate grouping changes; it does not estimate
//! optical resolution or physical stellar identity.

use crate::{source_group_audit, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub const DEFAULT_GROUP_RADII_PX: &[f64] = &[2.5, 3.0, 3.5, 4.0];

#[derive(Debug, Clone, Serialize)]
pub struct SourceGroupSensitivityRow {
    pub group_radius_px: f64,
    pub source_count: usize,
    pub quality_count: usize,
    pub group_count: usize,
    pub multi_member_group_count: usize,
    pub unresolved_group_count: usize,
    pub independent_group_candidate_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceGroupSensitivityTargetRow {
    pub group_radius_px: f64,
    pub target_detection_ids: String,
    pub group_id: usize,
    pub group_detection_ids: String,
    pub classification: String,
    pub member_count: usize,
    pub centroid_span_px: f64,
    pub peak_span_px: f64,
    pub feature_classes: String,
    pub flags: String,
    pub independent_psf_evidence: bool,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct SourceGroupSensitivityResult {
    pub catalog_path: String,
    pub target_detection_ids: String,
    pub rows: Vec<SourceGroupSensitivityRow>,
    pub targets: Vec<SourceGroupSensitivityTargetRow>,
    pub parameters: Value,
    pub conclusion: String,
}

impl SourceGroupSensitivityResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "target_detection_ids": self.target_detection_ids,
            "rows": self.rows,
            "targets": self.targets,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "组半径是 detector-level 敏感性参数，不是光学分辨率或物理星间距阈值。",
                "半径改变会改变比较总体；不同半径的组数不能拼接或相加。",
                "目标在某个半径下 isolated 只表示没有进入该半径近邻组，不表示已确认独立恒星。",
            ],
        })
    }
}

fn validate_radii(group_radii_px: &[f64]) -> Result<Vec<f64>> {
    if group_radii_px.is_empty() {
        return Err("group_radii_px must contain at least one radius".into());
    }
    if group_radii_px
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("group radii must be finite and positive".into());
    }
    for (index, value) in group_radii_px.iter().enumerate() {
        if group_radii_px[..index].contains(value) {
            return Err("group radii must be unique".into());
        }
    }
    Ok(group_radii_px.to_vec())
}

fn validate_workers(workers: i64) -> Result<usize> {
    if workers <= 0 {
        return Err("workers must be positive".into());
    }
    Ok(workers as usize)
}

fn deduplicate_targets(target_ids: &[i64]) -> Vec<i64> {
    let mut values = Vec::new();
    for value in target_ids {
        if !values.contains(value) {
            values.push(*value);
        }
    }
    values
}

fn audit_radii(
    catalog_csv: &Path,
    radii: &[f64],
    target_ids: &[i64],
    psf_fwhm: f64,
    delta_bic_min: f64,
    component_snr_min: f64,
    workers: usize,
) -> Result<Vec<source_group_audit::CandidateGroupAuditResult>> {
    let worker_count = workers.min(radii.len());
    let chunk_size = radii.len().div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker_index in 0..worker_count {
            let start = worker_index * chunk_size;
            let end = (start + chunk_size).min(radii.len());
            if start >= end {
                continue;
            }
            handles.push(scope.spawn(move || -> Result<Vec<_>> {
                (start..end)
                    .map(|index| {
                        source_group_audit::run(
                            catalog_csv,
                            target_ids,
                            psf_fwhm,
                            Some(radii[index]),
                            delta_bic_min,
                            component_snr_min,
                        )
                        .map(|result| (index, result))
                    })
                    .collect()
            }));
        }
        let mut ordered: Vec<Option<source_group_audit::CandidateGroupAuditResult>> =
            std::iter::repeat_with(|| None).take(radii.len()).collect();
        for handle in handles {
            let chunk = handle
                .join()
                .map_err(|_| "source group radius worker panicked")??;
            for (index, result) in chunk {
                ordered[index] = Some(result);
            }
        }
        ordered
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| "source group radius worker omitted a result".into())
            })
            .collect()
    })
}

fn general(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let scientific = format!("{value:.5e}");
    let (raw_mantissa, raw_exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent = raw_exponent.parse::<i32>().unwrap_or(0);
    if !(-4..6).contains(&exponent) {
        let mantissa = raw_mantissa
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned();
        return format!(
            "{mantissa}e{}{:02}",
            if exponent >= 0 { "+" } else { "-" },
            exponent.abs()
        );
    }
    let decimals = (5 - exponent).max(0) as usize;
    let fixed = format!("{value:.*}", decimals);
    if fixed.contains('.') {
        fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        fixed
    }
}

pub fn run(
    catalog_csv: &Path,
    group_radii_px: &[f64],
    target_ids: &[i64],
    psf_fwhm: f64,
    delta_bic_min: f64,
    component_snr_min: f64,
    workers: i64,
) -> Result<SourceGroupSensitivityResult> {
    let radii = validate_radii(group_radii_px)?;
    let targets = deduplicate_targets(target_ids);
    let workers = validate_workers(workers)?;
    let audits = audit_radii(
        catalog_csv,
        &radii,
        &targets,
        psf_fwhm,
        delta_bic_min,
        component_snr_min,
        workers,
    )?;

    let rows: Vec<_> = radii
        .iter()
        .zip(&audits)
        .map(|(radius, result)| SourceGroupSensitivityRow {
            group_radius_px: *radius,
            source_count: result.source_count,
            quality_count: result.quality_count,
            group_count: result.group_count,
            multi_member_group_count: result.multi_member_group_count,
            unresolved_group_count: result.unresolved_group_count,
            independent_group_candidate_count: result.independent_group_candidate_count,
        })
        .collect();
    let mut target_rows = Vec::new();
    for (radius, result) in radii.iter().zip(&audits) {
        target_rows.extend(
            result
                .targets
                .iter()
                .map(|target| SourceGroupSensitivityTargetRow {
                    group_radius_px: *radius,
                    target_detection_ids: target.target_detection_ids.clone(),
                    group_id: target.group_id,
                    group_detection_ids: target.group_detection_ids.clone(),
                    classification: target.classification.clone(),
                    member_count: target.member_count,
                    centroid_span_px: target.centroid_span_px,
                    peak_span_px: target.peak_span_px,
                    feature_classes: target.feature_classes.clone(),
                    flags: target.flags.clone(),
                    independent_psf_evidence: target.independent_psf_evidence,
                    reason: target.reason.clone(),
                }),
        );
    }
    let descriptions = rows
        .iter()
        .map(|row| {
            format!(
                "{}px→{}个多成员组",
                general(row.group_radius_px),
                row.multi_member_group_count
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let target_description = if targets.is_empty() {
        "未指定目标".to_owned()
    } else {
        targets
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join("|")
    };
    let conclusion = format!(
        "在同一源表和证据线下扫描组半径 {descriptions}；目标 {target_description} 的变化只反映 detector-level 近邻分流，不能解释为物理分辨率、星数或双星概率。"
    );
    Ok(SourceGroupSensitivityResult {
        catalog_path: catalog_csv.to_string_lossy().into_owned(),
        target_detection_ids: targets
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join("|"),
        rows,
        targets: target_rows,
        parameters: json!({
            "group_radii_px": radii,
            "psf_fwhm": psf_fwhm,
            "delta_bic_min": delta_bic_min,
            "component_snr_min": component_snr_min,
            "workers": workers,
            "comparison_boundary": "same source catalog, same quality and PSF evidence thresholds",
        }),
        conclusion,
    })
}

const ROW_FIELDS: &[&str] = &[
    "group_radius_px",
    "source_count",
    "quality_count",
    "group_count",
    "multi_member_group_count",
    "unresolved_group_count",
    "independent_group_candidate_count",
];
const TARGET_FIELDS: &[&str] = &[
    "group_radius_px",
    "target_detection_ids",
    "group_id",
    "group_detection_ids",
    "classification",
    "member_count",
    "centroid_span_px",
    "peak_span_px",
    "feature_classes",
    "flags",
    "independent_psf_evidence",
    "reason",
];

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) => number
            .as_i64()
            .map(|value| value.to_string())
            .or_else(|| number.as_u64().map(|value| value.to_string()))
            .or_else(|| number.as_f64().map(|value| format!("{value:?}")))
            .unwrap_or_default(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn write_records(path: &Path, headers: &[&str], rows: &[Value]) -> Result<()> {
    if rows.is_empty() {
        fs::write(path, []).map_err(|error| error.to_string())?;
        return Ok(());
    }
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let values: Vec<_> = headers
            .iter()
            .map(|header| csv_value(&row[*header]))
            .collect();
        writer
            .write_record(values)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_artifacts(result: &SourceGroupSensitivityResult, output: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let rows: Vec<_> = result
        .rows
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| error.to_string())?;
    let targets: Vec<_> = result
        .targets
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| error.to_string())?;
    write_records(
        &output.join("source_group_radius_sensitivity.csv"),
        ROW_FIELDS,
        &rows,
    )?;
    write_records(
        &output.join("source_group_radius_targets.csv"),
        TARGET_FIELDS,
        &targets,
    )?;
    fs::write(
        output.join("source_group_radius_sensitivity.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut positional = Vec::new();
    let mut radii = Vec::new();
    let mut target_ids = Vec::new();
    let mut psf_fwhm = 2.0;
    let mut delta_bic_min = 10.0;
    let mut component_snr_min = 5.0;
    let mut workers = 1;
    let mut output = PathBuf::from("tmp/source-group-radius-sensitivity");
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let (name, inline) = argument.split_once('=').unwrap_or((argument, ""));
        let option = match name {
            "--group-radius"
            | "--target-id"
            | "--psf-fwhm"
            | "--delta-bic-min"
            | "--component-snr-min"
            | "--workers"
            | "--out-dir" => Some(name),
            _ => None,
        };
        if let Some(option) = option {
            let value = if !inline.is_empty() {
                inline
            } else {
                index += 1;
                args.get(index).ok_or("option requires a value")?
            };
            match option {
                "--group-radius" => {
                    radii.push(value.parse::<f64>().map_err(|error| error.to_string())?)
                }
                "--target-id" => {
                    target_ids.push(value.parse::<i64>().map_err(|error| error.to_string())?)
                }
                "--psf-fwhm" => {
                    psf_fwhm = value.parse::<f64>().map_err(|error| error.to_string())?
                }
                "--delta-bic-min" => {
                    delta_bic_min = value.parse::<f64>().map_err(|error| error.to_string())?
                }
                "--component-snr-min" => {
                    component_snr_min = value.parse::<f64>().map_err(|error| error.to_string())?
                }
                "--workers" => workers = value.parse::<i64>().map_err(|error| error.to_string())?,
                _ => output = PathBuf::from(value),
            }
        } else if name.starts_with('-') {
            return Err(format!("unknown argument {name}"));
        } else {
            positional.push(PathBuf::from(name));
        }
        index += 1;
    }
    if positional.len() != 1 {
        return Err("expected one source catalog path".into());
    }
    let radii = if radii.is_empty() {
        DEFAULT_GROUP_RADII_PX.to_vec()
    } else {
        radii
    };
    let result = run(
        &positional[0],
        &radii,
        &target_ids,
        psf_fwhm,
        delta_bic_min,
        component_snr_min,
        workers,
    )?;
    write_artifacts(&result, &output)?;
    Ok(
        json!({"tool":"rst19-source-group-sensitivity", "out_dir":output,
        "result_path":output.join("source_group_radius_sensitivity.json"), "result":result.as_dict()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn catalog(root: &Path) -> PathBuf {
        let path = root.join("catalog.csv");
        fs::write(
            &path,
            concat!(
                "detection_id,x,y,peak_x,peak_y,peak,flux_snr,filter_snr,",
                "deblend_delta_bic,deblend_component_snr,quality_passed,feature_class,flags\r\n",
                "1,0,0,0,0,3992,204.6,1195,-4.09,0.87,False,range_anomaly,CODE_PATTERN\r\n",
                "2,2.738,0,4,0,569,115.5,300,-4.09,0.87,False,crowded_blend,UNRESOLVED_BLEND\r\n",
                "3,20,0,20,0,250,12,20,,,True,compact_quality,\r\n"
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn radius_scan_keeps_order_targets_and_artifacts_aligned() {
        let root = std::env::temp_dir().join(format!(
            "rst19-source-group-sensitivity-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = catalog(&root);
        let result = run(&catalog, &[4.0, 2.5, 3.0], &[2, 1, 2], 2.0, 10.0, 5.0, 2).unwrap();
        assert_eq!(
            result
                .rows
                .iter()
                .map(|row| row.group_count)
                .collect::<Vec<_>>(),
            [2, 3, 2]
        );
        assert_eq!(result.target_detection_ids, "2|1");
        assert_eq!(result.targets.len(), 4);
        assert_eq!(result.targets[3].classification, "unresolved_group");
        assert!(result.conclusion.contains("4px→1个多成员组"));
        let output = root.join("out");
        write_artifacts(&result, &output).unwrap();
        assert!(output.join("source_group_radius_sensitivity.csv").is_file());
        assert!(output.join("source_group_radius_targets.csv").is_file());
        assert!(output
            .join("source_group_radius_sensitivity.json")
            .is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invalid_radii_and_worker_counts_are_rejected() {
        assert_eq!(
            validate_radii(&[]).unwrap_err(),
            "group_radii_px must contain at least one radius"
        );
        assert_eq!(
            validate_radii(&[3.0, 3.0]).unwrap_err(),
            "group radii must be unique"
        );
        assert_eq!(
            validate_radii(&[f64::NAN]).unwrap_err(),
            "group radii must be finite and positive"
        );
        assert_eq!(validate_workers(0).unwrap_err(), "workers must be positive");
        assert_eq!(general(10.0), "10");
        assert_eq!(general(1.0e6), "1e+06");
        assert_eq!(general(1.0e-5), "1e-05");
    }
}
