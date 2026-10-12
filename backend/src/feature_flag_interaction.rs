use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

const INTERPRETATION_BOUNDARY: [&str; 4] = [
    "quality_fraction describes the current detector quality rule, not physical truth",
    "flags overlap; singleton and pair rows are not mutually exclusive probabilities",
    "exact_set_count means the complete row flag set equals this singleton or pair",
    "high SNR rejection is a hard-negative diagnostic, not a false-positive rate",
];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureFlagInteractionRow {
    pub kind: String,
    pub interaction: String,
    pub flag_a: String,
    pub flag_b: String,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub quality_fraction: f64,
    pub exact_set_count: usize,
    pub exact_set_quality_count: usize,
    pub high_snr_count: usize,
    pub high_snr_rejected_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureFlagInteractionResult {
    pub catalog_path: String,
    pub record_count: usize,
    pub high_snr_threshold: f64,
    pub rows: Vec<FeatureFlagInteractionRow>,
}

impl FeatureFlagInteractionResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "record_count": self.record_count,
            "high_snr_threshold": self.high_snr_threshold,
            "rows": self.rows,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

#[derive(Default)]
struct Counts {
    candidate: usize,
    quality: usize,
    exact: usize,
    exact_quality: usize,
    high_snr: usize,
    high_snr_rejected: usize,
}

struct CatalogRecord {
    detection_id: String,
    quality: bool,
    flags: Vec<String>,
    flux_snr: Option<f64>,
}

fn parse_quality(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes"
    )
}

fn parse_flags(value: &str) -> Vec<String> {
    let mut flags = value
        .split('|')
        .map(str::trim)
        .filter(|flag| !flag.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    flags.sort();
    flags.dedup();
    flags
}

fn parse_finite_float(value: &str) -> Option<f64> {
    let raw = value.trim();
    if raw.is_empty() {
        return None;
    }
    raw.parse::<f64>().ok().filter(|number| number.is_finite())
}

fn field(record: &csv::StringRecord, index: usize) -> &str {
    record.get(index).unwrap_or("None")
}

fn read_catalog(path: &Path) -> Result<Vec<CatalogRecord>> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let required = ["detection_id", "quality_passed", "flags", "flux_snr"];
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
        records.push(CatalogRecord {
            detection_id: field(&row, indices["detection_id"]).trim().to_owned(),
            quality: parse_quality(field(&row, indices["quality_passed"])),
            flags: parse_flags(field(&row, indices["flags"])),
            flux_snr: parse_finite_float(field(&row, indices["flux_snr"])),
        });
    }
    if records.is_empty() {
        return Err("source catalog is empty".into());
    }
    Ok(records)
}

pub fn run(catalog_path: &Path, high_snr_threshold: f64) -> Result<FeatureFlagInteractionResult> {
    if !high_snr_threshold.is_finite() {
        return Err("high_snr_threshold must be finite".into());
    }
    let records = read_catalog(catalog_path)?;
    let mut stats: BTreeMap<Vec<String>, Counts> = BTreeMap::new();
    let mut seen_ids = HashSet::new();

    for record in &records {
        if record.detection_id.is_empty() {
            return Err("source catalog contains an empty detection_id".into());
        }
        if !seen_ids.insert(&record.detection_id) {
            return Err(format!(
                "source catalog contains duplicate detection_id: {}",
                record.detection_id
            ));
        }
        if record.flags.is_empty() {
            continue;
        }
        let high_snr = record
            .flux_snr
            .is_some_and(|value| value >= high_snr_threshold);
        let mut keys = record
            .flags
            .iter()
            .map(|flag| vec![flag.clone()])
            .collect::<Vec<_>>();
        for left in 0..record.flags.len() {
            for right in (left + 1)..record.flags.len() {
                keys.push(vec![
                    record.flags[left].clone(),
                    record.flags[right].clone(),
                ]);
            }
        }
        for key in keys {
            let counts = stats.entry(key.clone()).or_default();
            counts.candidate += 1;
            counts.quality += usize::from(record.quality);
            counts.exact += usize::from(record.flags == key);
            counts.exact_quality += usize::from(record.flags == key && record.quality);
            counts.high_snr += usize::from(high_snr);
            counts.high_snr_rejected += usize::from(high_snr && !record.quality);
        }
    }

    let mut rows = stats
        .into_iter()
        .map(|(key, counts)| {
            let kind = if key.len() == 1 { "single" } else { "pair" };
            FeatureFlagInteractionRow {
                kind: kind.into(),
                interaction: key.join("+"),
                flag_a: key[0].clone(),
                flag_b: key.get(1).cloned().unwrap_or_default(),
                candidate_count: counts.candidate,
                quality_count: counts.quality,
                quality_fraction: counts.quality as f64 / counts.candidate as f64,
                exact_set_count: counts.exact,
                exact_set_quality_count: counts.exact_quality,
                high_snr_count: counts.high_snr,
                high_snr_rejected_count: counts.high_snr_rejected,
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        (left.kind != "single")
            .cmp(&(right.kind != "single"))
            .then_with(|| right.candidate_count.cmp(&left.candidate_count))
            .then_with(|| left.interaction.cmp(&right.interaction))
    });
    Ok(FeatureFlagInteractionResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        record_count: records.len(),
        high_snr_threshold,
        rows,
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

pub fn write_artifacts(
    result: &FeatureFlagInteractionResult,
    output_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    let rows = result
        .rows
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let columns = [
        "kind",
        "interaction",
        "flag_a",
        "flag_b",
        "candidate_count",
        "quality_count",
        "quality_fraction",
        "exact_set_count",
        "exact_set_quality_count",
        "high_snr_count",
        "high_snr_rejected_count",
    ];
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(columns)
        .map_err(|error| error.to_string())?;
    for row in &rows {
        writer
            .write_record(columns.iter().map(|column| csv_cell(row.get(*column))))
            .map_err(|error| error.to_string())?;
    }
    let csv_bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut bom_csv = Vec::with_capacity(csv_bytes.len() + 3);
    bom_csv.extend_from_slice(&[0xef, 0xbb, 0xbf]);
    bom_csv.extend_from_slice(&csv_bytes);
    fs::write(output_dir.join("feature_flag_interactions.csv"), bom_csv)
        .map_err(|error| error.to_string())?;
    fs::write(
        output_dir.join("feature_flag_interactions.json"),
        serde_json::to_string_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/feature-flag-interaction");
    let mut high_snr_threshold = 10.0;
    let mut top_n = 20_i64;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--out-dir" {
            index += 1;
            output_dir = PathBuf::from(args.get(index).ok_or("--out-dir requires a value")?);
        } else if let Some(value) = arg.strip_prefix("--out-dir=") {
            output_dir = PathBuf::from(value);
        } else if arg == "--high-snr-threshold" {
            index += 1;
            high_snr_threshold = args
                .get(index)
                .ok_or("--high-snr-threshold requires a value")?
                .parse()
                .map_err(|error| format!("invalid high SNR threshold: {error}"))?;
        } else if let Some(value) = arg.strip_prefix("--high-snr-threshold=") {
            high_snr_threshold = value
                .parse()
                .map_err(|error| format!("invalid high SNR threshold: {error}"))?;
        } else if arg == "--top-n" {
            index += 1;
            top_n = args
                .get(index)
                .ok_or("--top-n requires a value")?
                .parse()
                .map_err(|error| format!("invalid top N: {error}"))?;
        } else if let Some(value) = arg.strip_prefix("--top-n=") {
            top_n = value
                .parse()
                .map_err(|error| format!("invalid top N: {error}"))?;
        } else if arg.starts_with('-') {
            return Err(format!("unknown argument {arg}"));
        } else if catalog_path.replace(PathBuf::from(arg)).is_some() {
            return Err("only one source catalog is allowed".into());
        }
        index += 1;
    }
    if top_n < 1 {
        return Err("top_n must be positive".into());
    }
    let catalog_path = catalog_path.ok_or("a source catalog CSV is required")?;
    let result = run(&catalog_path, high_snr_threshold)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-flag-interaction",
        "out_dir": output_dir,
        "result_path": output_dir.join("feature_flag_interactions.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_single_pair_exact_sets_and_high_snr_rejections() {
        let dir =
            std::env::temp_dir().join(format!("rst19-flag-interaction-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let catalog = dir.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,quality_passed,flags,flux_snr\n1,true,A|B,12\n2,false,B|A,11\n3,false,A|B|C,20\n4,true,C,3\n5,true,,15\n",
        )
        .unwrap();
        let result = run(&catalog, 10.0).unwrap();
        assert_eq!(result.record_count, 5);
        assert_eq!(result.rows[0].interaction, "A");
        assert_eq!(result.rows[0].candidate_count, 3);
        let pair = result
            .rows
            .iter()
            .find(|row| row.interaction == "A+B")
            .unwrap();
        assert_eq!(pair.candidate_count, 3);
        assert_eq!(pair.exact_set_count, 2);
        assert_eq!(pair.high_snr_rejected_count, 2);
        let _ = fs::remove_dir_all(dir);
    }
}
