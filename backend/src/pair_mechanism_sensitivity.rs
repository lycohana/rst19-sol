use crate::{pair_mechanism_context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

const GUARDRAILS: [&str; 4] = [
    "窗口敏感性只描述候选目录中的比较总体，不是物理双星真值。",
    "放宽窗口会改变比较对象，不能把不同配置的 pair 数直接比较成概率。",
    "both_quality_count 是当前质量规则的通过数，不是可靠星数或噪点数。",
    "目标 pair 的 baseline/injected/new 反事实仍需单独审计。",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Configuration {
    pub name: String,
    pub max_centroid_distance_px: f64,
    pub peak_distance_min_px: f64,
    pub peak_distance_max_px: f64,
}

pub fn default_configurations() -> Vec<Configuration> {
    [
        ("centroid_3.0", 3.0, 3.5, 4.8),
        ("centroid_3.5", 3.5, 3.5, 4.8),
        ("centroid_4.0", 4.0, 3.5, 4.8),
        ("peak_3.0_5.0", 3.5, 3.0, 5.0),
        ("peak_3.5_5.5", 3.5, 3.5, 5.5),
    ]
    .into_iter()
    .map(|(name, centroid, peak_min, peak_max)| Configuration {
        name: name.into(),
        max_centroid_distance_px: centroid,
        peak_distance_min_px: peak_min,
        peak_distance_max_px: peak_max,
    })
    .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SensitivityRow {
    pub configuration: String,
    pub max_centroid_distance_px: f64,
    pub peak_distance_min_px: f64,
    pub peak_distance_max_px: f64,
    pub pair_count: usize,
    pub same_feature_class_count: usize,
    pub cross_feature_class_count: usize,
    pub both_quality_count: usize,
    pub both_quality_fraction: f64,
    pub target_in_geometry_context: bool,
    pub target_centroid_distance_px: f64,
    pub target_peak_distance_px: f64,
    pub target_feature_class_a: String,
    pub target_feature_class_b: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityResult {
    pub catalog_path: String,
    pub target_ids: [i64; 2],
    pub rows: Vec<SensitivityRow>,
    pub parameters: Value,
    pub conclusion: String,
}

impl SensitivityResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "target_ids": self.target_ids,
            "rows": self.rows,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation_guardrails": GUARDRAILS,
        })
    }
}

fn normalize(configurations: &[Configuration]) -> Result<Vec<Configuration>> {
    if configurations.is_empty() {
        return Err("configurations must contain at least one window".into());
    }
    let mut normalized = Vec::with_capacity(configurations.len());
    let mut seen = std::collections::HashSet::new();
    for config in configurations {
        let mut value = config.clone();
        value.name = value.name.trim().to_owned();
        if value.name.is_empty() {
            return Err("configuration name must not be empty".into());
        }
        if !seen.insert(value.name.clone()) {
            return Err(format!("duplicate configuration name: {}", value.name));
        }
        for distance in [
            value.max_centroid_distance_px,
            value.peak_distance_min_px,
            value.peak_distance_max_px,
        ] {
            if !distance.is_finite() || distance <= 0.0 {
                return Err(format!(
                    "configuration {:?} distances must be finite and positive",
                    value.name
                ));
            }
        }
        if value.peak_distance_min_px > value.peak_distance_max_px {
            return Err(format!(
                "configuration {:?} peak minimum exceeds maximum",
                value.name
            ));
        }
        normalized.push(value);
    }
    Ok(normalized)
}

pub fn run(
    catalog_path: &Path,
    target_ids: &[i64],
    configurations: &[Configuration],
) -> Result<SensitivityResult> {
    let mut ids = target_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != 2 {
        return Err("target_ids must contain exactly two distinct detection IDs".into());
    }
    let target_ids = [ids[0], ids[1]];
    let configurations = normalize(configurations)?;
    let mut rows = Vec::with_capacity(configurations.len());
    for config in &configurations {
        let context = pair_mechanism_context::run(
            catalog_path,
            &target_ids,
            config.max_centroid_distance_px,
            config.peak_distance_min_px,
            config.peak_distance_max_px,
        )?;
        let pair_count = context.pairs.len();
        let same_feature_class_count = context
            .pairs
            .iter()
            .filter(|row| row.same_feature_class)
            .count();
        let both_quality_count = context
            .pairs
            .iter()
            .filter(|row| row.both_quality_passed)
            .count();
        rows.push(SensitivityRow {
            configuration: config.name.clone(),
            max_centroid_distance_px: config.max_centroid_distance_px,
            peak_distance_min_px: config.peak_distance_min_px,
            peak_distance_max_px: config.peak_distance_max_px,
            pair_count,
            same_feature_class_count,
            cross_feature_class_count: pair_count - same_feature_class_count,
            both_quality_count,
            both_quality_fraction: if pair_count > 0 {
                both_quality_count as f64 / pair_count as f64
            } else {
                0.0
            },
            target_in_geometry_context: context.target.in_geometry_context,
            target_centroid_distance_px: context.target.centroid_distance_px,
            target_peak_distance_px: context.target.peak_distance_px,
            target_feature_class_a: context.target.feature_class_a,
            target_feature_class_b: context.target.feature_class_b,
        });
    }
    let conclusion = format!(
        "完成 {} 组几何窗口敏感性审计；结果只用于检查比较总体随窗口变化的方向，不能转换为噪点率、双星概率或物理恒星数。",
        rows.len()
    );
    let configs_json: Vec<Value> = configurations
        .iter()
        .map(|config| {
            json!({
                "name": config.name,
                "max_centroid_distance_px": config.max_centroid_distance_px,
                "peak_distance_min_px": config.peak_distance_min_px,
                "peak_distance_max_px": config.peak_distance_max_px,
            })
        })
        .collect();
    Ok(SensitivityResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        target_ids,
        rows,
        parameters: json!({
            "configuration_count": configurations.len(),
            "configurations": configs_json,
            "interpretation_boundary": "descriptive candidate-catalog sensitivity only; no physical star probability, precision, FDR, or truth",
        }),
        conclusion,
    })
}

fn f(value: f64) -> String {
    format!("{value:?}")
}
fn b(value: bool) -> String {
    if value { "True" } else { "False" }.to_owned()
}

pub fn write_artifacts(result: &SensitivityResult, out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record([
            "configuration",
            "max_centroid_distance_px",
            "peak_distance_min_px",
            "peak_distance_max_px",
            "pair_count",
            "same_feature_class_count",
            "cross_feature_class_count",
            "both_quality_count",
            "both_quality_fraction",
            "target_in_geometry_context",
            "target_centroid_distance_px",
            "target_peak_distance_px",
            "target_feature_class_a",
            "target_feature_class_b",
        ])
        .map_err(|error| error.to_string())?;
    for row in &result.rows {
        writer
            .write_record([
                row.configuration.clone(),
                f(row.max_centroid_distance_px),
                f(row.peak_distance_min_px),
                f(row.peak_distance_max_px),
                row.pair_count.to_string(),
                row.same_feature_class_count.to_string(),
                row.cross_feature_class_count.to_string(),
                row.both_quality_count.to_string(),
                f(row.both_quality_fraction),
                b(row.target_in_geometry_context),
                f(row.target_centroid_distance_px),
                f(row.target_peak_distance_px),
                row.target_feature_class_a.clone(),
                row.target_feature_class_b.clone(),
            ])
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(out_dir.join("pair_mechanism_sensitivity.csv"), bytes)
        .map_err(|error| error.to_string())?;
    let json_bytes =
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    fs::write(out_dir.join("pair_mechanism_sensitivity.json"), json_bytes)
        .map_err(|error| error.to_string())
}

fn parse_configuration(raw: &str) -> Result<Configuration> {
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        return Err("--configuration format is name,centroid_max,peak_min,peak_max".into());
    }
    let parse = |text: &str| {
        text.parse::<f64>()
            .map_err(|_| "--configuration distances must be numeric".to_owned())
    };
    Ok(Configuration {
        name: parts[0].to_owned(),
        max_centroid_distance_px: parse(parts[1])?,
        peak_distance_min_px: parse(parts[2])?,
        peak_distance_max_px: parse(parts[3])?,
    })
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog = None;
    let mut output_dir = PathBuf::from("tmp/pair-mechanism-sensitivity");
    let mut target_ids = Vec::new();
    let mut configurations = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let value = |option: &str| -> Result<&str> {
            args.get(index + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("{option} requires a value"))
        };
        match args[index].as_str() {
            "--target-id" => {
                target_ids.push(
                    value("--target-id")?
                        .parse()
                        .map_err(|_| "--target-id must be an integer")?,
                );
                index += 1;
            }
            "--configuration" => {
                configurations.push(parse_configuration(value("--configuration")?)?);
                index += 1;
            }
            "--out-dir" => {
                output_dir = PathBuf::from(value("--out-dir")?);
                index += 1;
            }
            arg if arg.starts_with('-') => return Err(format!("unknown argument {arg}")),
            arg => {
                if catalog.replace(PathBuf::from(arg)).is_some() {
                    return Err("only one source catalog is allowed".into());
                }
            }
        }
        index += 1;
    }
    let catalog = catalog.ok_or("a source catalog CSV is required")?;
    if configurations.is_empty() {
        configurations = default_configurations();
    }
    let result = run(&catalog, &target_ids, &configurations)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-pair-mechanism-sensitivity",
        "out_dir": output_dir,
        "result_path": output_dir.join("pair_mechanism_sensitivity.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitivity_executes_declared_windows_and_keeps_counts_descriptive() {
        let root =
            std::env::temp_dir().join(format!("rst19-pair-sensitivity-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(&catalog, "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class,flags\n1,0,0,0,0,false,range_anomaly,CODE_PATTERN\n2,2.738,0,4,0,false,crowded_blend,UNRESOLVED_BLEND\n3,20,0,20,0,true,compact_quality,\n").unwrap();
        let configs = vec![
            Configuration {
                name: "narrow".into(),
                max_centroid_distance_px: 3.0,
                peak_distance_min_px: 3.5,
                peak_distance_max_px: 4.8,
            },
            Configuration {
                name: "target".into(),
                max_centroid_distance_px: 3.5,
                peak_distance_min_px: 3.5,
                peak_distance_max_px: 4.8,
            },
        ];
        let result = run(&catalog, &[1, 2], &configs).unwrap();
        assert_eq!(
            result
                .rows
                .iter()
                .map(|row| row.pair_count)
                .collect::<Vec<_>>(),
            [1, 1]
        );
        assert!(result.rows.iter().all(|row| row.target_in_geometry_context));
        assert!(result.as_dict()["interpretation_guardrails"][0]
            .as_str()
            .unwrap()
            .contains("不是物理双星真值"));
    }
}
