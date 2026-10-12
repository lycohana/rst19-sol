use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

const DEFAULT_TARGETS: [&str; 3] = ["82931", "82934", "44132"];
const INTERPRETATION_BOUNDARY: &str = "flags are overlapping detector-level audit labels; flag fractions are not mutually exclusive probabilities, precision, FDR, or star probabilities";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureFlagProfileRow {
    pub feature_class: String,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub flag: String,
    pub flag_count: usize,
    pub flag_fraction: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureFlagProfileTarget {
    pub detection_id: String,
    pub feature_class: String,
    pub quality_passed: bool,
    pub flags: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureFlagProfileResult {
    pub catalog_path: String,
    pub targets: Vec<String>,
    pub rows: Vec<FeatureFlagProfileRow>,
    pub target_rows: Vec<FeatureFlagProfileTarget>,
}

impl FeatureFlagProfileResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "targets": self.targets,
            "rows": self.rows,
            "target_rows": self.target_rows,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

fn parse_quality(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes"
    )
}

fn parse_flags(value: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    value
        .split('|')
        .map(str::trim)
        .filter(|flag| !flag.is_empty() && seen.insert((*flag).to_owned()))
        .map(str::to_owned)
        .collect()
}

fn value_at(record: &csv::StringRecord, index: usize) -> &str {
    // Python's DictReader fills a short row with None; str(None) is "None".
    record.get(index).unwrap_or("None")
}

pub fn run(catalog_path: &Path, target_ids: &[String]) -> Result<FeatureFlagProfileResult> {
    let bytes = fs::read(catalog_path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let required = ["detection_id", "feature_class", "quality_passed", "flags"];
    let mut indices = BTreeMap::new();
    for name in required {
        if let Some(index) = headers
            .iter()
            .enumerate()
            .filter_map(|(index, header)| (header == name).then_some(index))
            .last()
        {
            indices.insert(name, index);
        }
    }
    let mut missing = required
        .iter()
        .filter(|name| !indices.contains_key(**name))
        .copied()
        .collect::<Vec<_>>();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }

    let mut records = Vec::new();
    for row in reader.records() {
        let row = row.map_err(|error| error.to_string())?;
        records.push((
            value_at(&row, indices["detection_id"]).trim().to_owned(),
            value_at(&row, indices["feature_class"]).trim().to_owned(),
            parse_quality(value_at(&row, indices["quality_passed"])),
            parse_flags(value_at(&row, indices["flags"])),
        ));
    }
    if records.is_empty() {
        return Err("source catalog is empty".into());
    }

    let mut targets = Vec::new();
    let mut seen_targets = HashSet::new();
    for value in target_ids {
        let value = value.trim().to_owned();
        if !value.is_empty() && seen_targets.insert(value.clone()) {
            targets.push(value);
        }
    }

    let mut class_records: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        if !record.1.is_empty() {
            class_records
                .entry(record.1.clone())
                .or_default()
                .push(index);
        }
    }
    let mut rows = Vec::new();
    for (feature_class, members) in &class_records {
        let candidate_count = members.len();
        let quality_count = members.iter().filter(|index| records[**index].2).count();
        let mut counts = BTreeMap::<String, usize>::new();
        for index in members {
            for flag in &records[*index].3 {
                *counts.entry(flag.clone()).or_default() += 1;
            }
        }
        if counts.is_empty() {
            counts.insert("<none>".into(), candidate_count);
        }
        let mut counts = counts.into_iter().collect::<Vec<_>>();
        counts.sort_by(|(left_flag, left_count), (right_flag, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_flag.cmp(right_flag))
        });
        rows.extend(
            counts
                .into_iter()
                .map(|(flag, flag_count)| FeatureFlagProfileRow {
                    feature_class: feature_class.clone(),
                    candidate_count,
                    quality_count,
                    flag,
                    flag_count,
                    flag_fraction: flag_count as f64 / candidate_count as f64,
                }),
        );
    }

    let wanted = targets.iter().collect::<HashSet<_>>();
    let mut target_rows = records
        .iter()
        .filter(|record| wanted.contains(&record.0))
        .map(|record| FeatureFlagProfileTarget {
            detection_id: record.0.clone(),
            feature_class: record.1.clone(),
            quality_passed: record.2,
            flags: record.3.join("|"),
        })
        .collect::<Vec<_>>();
    target_rows.sort_by(|left, right| left.detection_id.cmp(&right.detection_id));

    Ok(FeatureFlagProfileResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        targets,
        rows,
        target_rows,
    })
}

fn csv_cell(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Bool(value)) => if *value { "True" } else { "False" }.to_owned(),
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
    }
}

fn write_csv(output: &Path, columns: &[&str], rows: &[Value]) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(columns)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(columns.iter().map(|column| csv_cell(row.get(*column))))
            .map_err(|error| error.to_string())?;
    }
    let csv_bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut output_bytes = Vec::with_capacity(csv_bytes.len() + 3);
    output_bytes.extend_from_slice(&[0xef, 0xbb, 0xbf]);
    output_bytes.extend_from_slice(&csv_bytes);
    fs::write(output, output_bytes).map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &FeatureFlagProfileResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    let profile_rows = result
        .rows
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let target_rows = result
        .target_rows
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    write_csv(
        &output_dir.join("feature_flag_profile.csv"),
        &[
            "feature_class",
            "candidate_count",
            "quality_count",
            "flag",
            "flag_count",
            "flag_fraction",
        ],
        &profile_rows,
    )?;
    write_csv(
        &output_dir.join("feature_flag_profile_targets.csv"),
        &["detection_id", "feature_class", "quality_passed", "flags"],
        &target_rows,
    )?;
    fs::write(
        output_dir.join("feature_flag_profile.json"),
        serde_json::to_string_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/feature-flag-profile");
    let mut target_ids = DEFAULT_TARGETS
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--out-dir" {
            index += 1;
            output_dir = PathBuf::from(args.get(index).ok_or("--out-dir requires a value")?);
        } else if let Some(value) = arg.strip_prefix("--out-dir=") {
            output_dir = PathBuf::from(value);
        } else if arg == "--target-ids" {
            target_ids.clear();
            index += 1;
            while index < args.len() && !args[index].starts_with('-') {
                target_ids.push(args[index].clone());
                index += 1;
            }
            continue;
        } else if let Some(value) = arg.strip_prefix("--target-ids=") {
            target_ids.clear();
            if !value.is_empty() {
                target_ids.push(value.to_owned());
            }
        } else if arg.starts_with('-') {
            return Err(format!("unknown argument {arg}"));
        } else if catalog_path.replace(PathBuf::from(arg)).is_some() {
            return Err("only one source catalog is allowed".into());
        }
        index += 1;
    }
    let catalog_path = catalog_path.ok_or("a source catalog CSV is required")?;
    let result = run(&catalog_path, &target_ids)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-flag-profile",
        "out_dir": output_dir,
        "result_path": output_dir.join("feature_flag_profile.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_overlapping_flags_and_keeps_target_order_contract() {
        let dir = std::env::temp_dir().join(format!("rst19-flag-profile-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let catalog = dir.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,feature_class,quality_passed,flags\n1,compact,true,wing|blend|wing\n2,compact,0,blend\n3,extended,yes,\n4,extended,no,\n",
        )
        .unwrap();
        let targets = vec!["3".into(), "1".into(), "3".into(), "99".into()];
        let result = run(&catalog, &targets).unwrap();
        assert_eq!(result.targets, vec!["3", "1", "99"]);
        assert_eq!(result.rows[0].feature_class, "compact");
        assert_eq!(result.rows[0].flag, "blend");
        assert_eq!(result.rows[0].flag_count, 2);
        assert_eq!(result.rows[0].quality_count, 1);
        assert_eq!(result.rows[0].flag_fraction, 2.0 / 2.0);
        assert_eq!(result.target_rows[0].detection_id, "1");
        assert_eq!(
            result
                .rows
                .iter()
                .find(|row| row.feature_class == "extended")
                .unwrap()
                .flag,
            "<none>"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
