use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

const REQUIRED_FIELDS: [&str; 9] = [
    "layer",
    "anchor_feature_class",
    "anchor_feature_class_label",
    "response_feature_class",
    "response_feature_class_label",
    "anchor_count",
    "frame_count",
    "possible_match_count",
    "matched_count",
];

const FEATURE_CLASS_ORDER: [&str; 9] = [
    "range_anomaly",
    "linear_artifact",
    "masked_or_edge",
    "crowded_blend",
    "spike_or_support",
    "weak_or_background",
    "shape_outlier",
    "compact_quality",
    "other_rejected",
];

const INTERPRETATION_GUARDRAILS: [&str; 4] = [
    "q_any 是首帧类别在后续帧得到任意邻域响应的比例，不是逐星身份概率。",
    "q_same 是发生响应后的条件同类率，不是类别分类准确率。",
    "最大跨类流向描述算法类别重写，不证明物理结构发生了变化。",
    "输入是已生成的类别转移 CSV；本后处理不重新读取 FITS 或运行检测器。",
];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SequenceFeatureTransitionFlowRow {
    pub layer: String,
    pub anchor_feature_class: String,
    pub anchor_feature_class_label: String,
    pub anchor_count: i64,
    pub frame_count: i64,
    pub possible_match_count: i64,
    pub matched_count: i64,
    pub same_class_match_count: i64,
    pub q_any: Option<f64>,
    pub q_same: Option<f64>,
    pub mean_matches_per_anchor: Option<f64>,
    pub top_non_same_feature_class: Option<String>,
    pub top_non_same_feature_class_label: Option<String>,
    pub top_non_same_matched_count: i64,
    pub top_non_same_fraction: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SequenceFeatureTransitionFlowResult {
    pub source_csv: String,
    pub input_row_count: usize,
    pub included_group_count: usize,
    pub skipped_zero_anchor_group_count: usize,
    pub rows: Vec<SequenceFeatureTransitionFlowRow>,
    pub interpretation_guardrails: Vec<String>,
}

impl SequenceFeatureTransitionFlowResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "source_csv": self.source_csv,
            "input_row_count": self.input_row_count,
            "included_group_count": self.included_group_count,
            "skipped_zero_anchor_group_count": self.skipped_zero_anchor_group_count,
            "rows": self.rows,
            "interpretation_guardrails": self.interpretation_guardrails,
        })
    }
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn number(value: &str, field: &str, path: &Path) -> Result<f64> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("{}: 字段 {field:?} 不是数字", path_text(path)))
}

fn integer(value: &str, field: &str, path: &Path) -> Result<i64> {
    let parsed = number(value, field, path)?;
    if !parsed.is_finite() || parsed.fract() != 0.0 {
        return Err(format!("{}: 字段 {field:?} 不是整数", path_text(path)));
    }
    if parsed < i64::MIN as f64 || parsed > i64::MAX as f64 {
        return Err(format!("{}: 字段 {field:?} 超出整数范围", path_text(path)));
    }
    Ok(parsed as i64)
}

fn class_order(class: &str) -> (usize, &str) {
    (
        FEATURE_CLASS_ORDER
            .iter()
            .position(|known| *known == class)
            .unwrap_or(FEATURE_CLASS_ORDER.len()),
        class,
    )
}

fn layer_order(layer: &str) -> (usize, &str) {
    let rank = match layer {
        "candidate" => 0,
        "quality" => 1,
        _ => 2,
    };
    (rank, layer)
}

type CsvRow = HashMap<String, String>;

fn read_rows(path: &Path) -> Result<Vec<CsvRow>> {
    let bytes = fs::read(path).map_err(|_| format!("无法读取类别转移 CSV：{}", path_text(path)))?;
    let content = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(content);
    let mut records = reader.records();
    let Some(header_record) = records.next() else {
        return Err(format!(
            "{}: 缺少字段 {}",
            path_text(path),
            REQUIRED_FIELDS.join(", ")
        ));
    };
    let headers: Vec<String> = header_record
        .map_err(|error| format!("{}: {error}", path_text(path)))?
        .iter()
        .map(str::to_owned)
        .collect();
    let header_set: std::collections::HashSet<&str> = headers.iter().map(String::as_str).collect();
    let mut missing: Vec<&str> = REQUIRED_FIELDS
        .iter()
        .copied()
        .filter(|field| !header_set.contains(field))
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!(
            "{}: 缺少字段 {}",
            path_text(path),
            missing.join(", ")
        ));
    }

    let mut rows = Vec::new();
    for record in records {
        let record = record.map_err(|error| format!("{}: {error}", path_text(path)))?;
        let any_value = record.iter().any(|value| !value.trim().is_empty());
        if !any_value {
            continue;
        }
        let mut row = HashMap::new();
        for (header, value) in headers.iter().zip(record.iter()) {
            // Python's DictReader keeps the last value when a header is repeated.
            row.insert(header.clone(), value.to_owned());
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{}: 类别转移 CSV 为空", path_text(path)));
    }
    Ok(rows)
}

fn row_value<'a>(row: &'a CsvRow, field: &str) -> &'a str {
    row.get(field).map(String::as_str).unwrap_or("")
}

fn trimmed_required(row: &CsvRow, field: &str, path: &Path) -> Result<String> {
    let value = row_value(row, field).trim();
    if value.is_empty() {
        return Err(format!("{}: {field} 不能为空", path_text(path)));
    }
    Ok(value.to_owned())
}

pub fn summarize_sequence_feature_transition_flow(
    csv_path: &Path,
) -> Result<SequenceFeatureTransitionFlowResult> {
    let source_rows = read_rows(csv_path)?;
    let mut groups: Vec<((String, String), Vec<CsvRow>)> = Vec::new();
    let mut group_indices = HashMap::<(String, String), usize>::new();
    for row in source_rows {
        let layer = trimmed_required(&row, "layer", csv_path)?;
        let anchor_class = trimmed_required(&row, "anchor_feature_class", csv_path)?;
        let key = (layer, anchor_class);
        if let Some(index) = group_indices.get(&key) {
            groups[*index].1.push(row);
        } else {
            let index = groups.len();
            group_indices.insert(key.clone(), index);
            groups.push((key, vec![row]));
        }
    }

    let mut result_rows = Vec::new();
    let mut skipped_zero_anchor = 0usize;
    for ((layer, anchor_class), group) in &groups {
        let first = &group[0];
        let anchor_count = integer(row_value(first, "anchor_count"), "anchor_count", csv_path)?;
        let frame_count = integer(row_value(first, "frame_count"), "frame_count", csv_path)?;
        let possible_match_count = integer(
            row_value(first, "possible_match_count"),
            "possible_match_count",
            csv_path,
        )?;
        if anchor_count <= 0 {
            skipped_zero_anchor += 1;
            continue;
        }
        if frame_count <= 0 || possible_match_count < 0 {
            return Err(format!(
                "{}: {layer}/{anchor_class} 的元数据不合法",
                path_text(csv_path)
            ));
        }

        let mut response_counts = HashMap::<String, i64>::new();
        let mut response_labels = HashMap::<String, String>::new();
        let mut matched_count = 0i64;
        for row in group {
            let row_anchor_count =
                integer(row_value(row, "anchor_count"), "anchor_count", csv_path)?;
            let row_frame_count = integer(row_value(row, "frame_count"), "frame_count", csv_path)?;
            let row_possible = integer(
                row_value(row, "possible_match_count"),
                "possible_match_count",
                csv_path,
            )?;
            if (row_anchor_count, row_frame_count, row_possible)
                != (anchor_count, frame_count, possible_match_count)
            {
                return Err(format!(
                    "{}: {layer}/{anchor_class} 的元数据在响应类别之间不一致",
                    path_text(csv_path)
                ));
            }
            let response_class = trimmed_required(row, "response_feature_class", csv_path)?;
            let matched = integer(row_value(row, "matched_count"), "matched_count", csv_path)?;
            if matched < 0 {
                return Err(format!("{}: matched_count 不能为负数", path_text(csv_path)));
            }
            let total = response_counts.entry(response_class.clone()).or_default();
            *total = total
                .checked_add(matched)
                .ok_or("matched_count sum exceeds supported integer range")?;
            response_labels
                .entry(response_class.clone())
                .or_insert_with(|| {
                    let label = row_value(row, "response_feature_class_label").trim();
                    if label.is_empty() {
                        response_class
                    } else {
                        label.to_owned()
                    }
                });
            matched_count = matched_count
                .checked_add(matched)
                .ok_or("matched_count sum exceeds supported integer range")?;
        }

        let same_class_match_count = response_counts.get(anchor_class).copied().unwrap_or(0);
        let mut non_same: Vec<(String, i64)> = response_counts
            .iter()
            .filter(|(feature_class, count)| feature_class.as_str() != anchor_class && **count > 0)
            .map(|(feature_class, count)| (feature_class.clone(), *count))
            .collect();
        non_same.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| class_order(&left.0).cmp(&class_order(&right.0)))
        });
        let (top_class, top_count, top_label, top_fraction) =
            if let Some((feature_class, count)) = non_same.first() {
                (
                    Some(feature_class.clone()),
                    *count,
                    response_labels.get(feature_class).cloned(),
                    (matched_count != 0).then_some(*count as f64 / matched_count as f64),
                )
            } else {
                (None, 0, None, None)
            };
        let anchor_label = row_value(first, "anchor_feature_class_label").trim();
        result_rows.push(SequenceFeatureTransitionFlowRow {
            layer: layer.clone(),
            anchor_feature_class: anchor_class.clone(),
            anchor_feature_class_label: if anchor_label.is_empty() {
                anchor_class.clone()
            } else {
                anchor_label.to_owned()
            },
            anchor_count,
            frame_count,
            possible_match_count,
            matched_count,
            same_class_match_count,
            q_any: (possible_match_count != 0)
                .then_some(matched_count as f64 / possible_match_count as f64),
            q_same: (matched_count != 0)
                .then_some(same_class_match_count as f64 / matched_count as f64),
            mean_matches_per_anchor: Some(matched_count as f64 / anchor_count as f64),
            top_non_same_feature_class: top_class,
            top_non_same_feature_class_label: top_label,
            top_non_same_matched_count: top_count,
            top_non_same_fraction: top_fraction,
        });
    }

    result_rows.sort_by(|left, right| {
        layer_order(&left.layer)
            .cmp(&layer_order(&right.layer))
            .then_with(|| {
                class_order(&left.anchor_feature_class)
                    .cmp(&class_order(&right.anchor_feature_class))
            })
    });
    Ok(SequenceFeatureTransitionFlowResult {
        source_csv: path_text(csv_path),
        input_row_count: groups.iter().map(|(_, rows)| rows.len()).sum(),
        included_group_count: result_rows.len(),
        skipped_zero_anchor_group_count: skipped_zero_anchor,
        rows: result_rows,
        interpretation_guardrails: INTERPRETATION_GUARDRAILS
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
    })
}

fn python_float(value: f64) -> String {
    let raw = format!("{value:?}");
    if let Some((mantissa, exponent)) = raw.split_once('e') {
        let exponent = exponent.parse::<i32>().unwrap_or(0);
        format!("{mantissa}e{exponent:+03}")
    } else {
        raw
    }
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(String::new, python_float)
}

pub fn write_sequence_feature_transition_flow_artifacts(
    result: &SequenceFeatureTransitionFlowResult,
    output_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    let headers = [
        "layer",
        "anchor_feature_class",
        "anchor_feature_class_label",
        "anchor_count",
        "frame_count",
        "possible_match_count",
        "matched_count",
        "same_class_match_count",
        "q_any",
        "q_same",
        "mean_matches_per_anchor",
        "top_non_same_feature_class",
        "top_non_same_feature_class_label",
        "top_non_same_matched_count",
        "top_non_same_fraction",
    ];
    let rows: Vec<Vec<String>> = result
        .rows
        .iter()
        .map(|row| {
            vec![
                row.layer.clone(),
                row.anchor_feature_class.clone(),
                row.anchor_feature_class_label.clone(),
                row.anchor_count.to_string(),
                row.frame_count.to_string(),
                row.possible_match_count.to_string(),
                row.matched_count.to_string(),
                row.same_class_match_count.to_string(),
                optional_float(row.q_any),
                optional_float(row.q_same),
                optional_float(row.mean_matches_per_anchor),
                row.top_non_same_feature_class.clone().unwrap_or_default(),
                row.top_non_same_feature_class_label
                    .clone()
                    .unwrap_or_default(),
                row.top_non_same_matched_count.to_string(),
                optional_float(row.top_non_same_fraction),
            ]
        })
        .collect();
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(
        output_dir.join("sequence_feature_transition_flow.csv"),
        bytes,
    )
    .map_err(|error| error.to_string())?;
    let json_bytes =
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    fs::write(
        output_dir.join("sequence_feature_transition_flow.json"),
        json_bytes,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut csv_path = None;
    let mut output_dir = PathBuf::from("tmp/sequence-feature-transition-flow");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline_value) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        match name {
            "--out-dir" => {
                let value = if let Some(value) = inline_value {
                    value
                } else {
                    index += 1;
                    args.get(index).ok_or("--out-dir requires a value")?
                };
                output_dir = PathBuf::from(value);
            }
            "--" => {
                index += 1;
                while index < args.len() {
                    if csv_path.replace(PathBuf::from(&args[index])).is_some() {
                        return Err("only one transition CSV is allowed".into());
                    }
                    index += 1;
                }
                break;
            }
            _ if arg.starts_with('-') => return Err(format!("unknown argument {arg}")),
            _ => {
                if csv_path.replace(PathBuf::from(arg)).is_some() {
                    return Err("only one transition CSV is allowed".into());
                }
            }
        }
        index += 1;
    }
    let csv_path = csv_path.ok_or("a sequence feature transition CSV is required")?;
    let result = summarize_sequence_feature_transition_flow(&csv_path)?;
    write_sequence_feature_transition_flow_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-transition-flow",
        "out_dir": output_dir,
        "result_path": output_dir.join("sequence_feature_transition_flow.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "rst19-feature-transition-flow-{}-{label}",
            std::process::id()
        ))
    }

    #[test]
    fn transition_flow_sorts_classes_and_computes_ratios() {
        let root = temp_path("sort");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("transitions.csv");
        fs::write(
            &path,
            "layer,anchor_feature_class,anchor_feature_class_label,response_feature_class,response_feature_class_label,anchor_count,frame_count,possible_match_count,matched_count\n\
             quality,weak_or_background,弱,weak_or_background,弱,4,3,12,2\n\
             candidate,compact_quality,紧凑,weak_or_background,背景,4,3,12,2\n\
             candidate,compact_quality,ignored,linear_artifact,线伪迹,4,3,12,2\n\
             candidate,compact_quality,ignored,compact_quality,紧凑,4,3,12,3\n",
        )
        .unwrap();
        let result = summarize_sequence_feature_transition_flow(&path).unwrap();
        assert_eq!(result.rows.len(), 2);
        let first = &result.rows[0];
        assert_eq!(first.anchor_feature_class, "compact_quality");
        assert_eq!(first.matched_count, 7);
        assert_eq!(first.same_class_match_count, 3);
        assert_eq!(
            first.top_non_same_feature_class.as_deref(),
            Some("linear_artifact")
        );
        assert_eq!(first.q_any, Some(7.0 / 12.0));
        assert_eq!(first.q_same, Some(3.0 / 7.0));
        assert_eq!(first.top_non_same_fraction, Some(2.0 / 7.0));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn bom_empty_rows_and_zero_anchor_groups_match_python_reader_semantics() {
        let root = temp_path("bom");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("transitions.csv");
        fs::write(
            &path,
            "\u{feff}layer,anchor_feature_class,anchor_feature_class_label,response_feature_class,response_feature_class_label,anchor_count,frame_count,possible_match_count,matched_count\r\n\r\nquality,other_rejected,拒绝,other_rejected,拒绝,0,0,0,0\r\ncandidate,compact_quality,紧凑,compact_quality,紧凑,2,2,0,0\r\n",
        )
        .unwrap();
        let result = summarize_sequence_feature_transition_flow(&path).unwrap();
        assert_eq!(result.input_row_count, 2);
        assert_eq!(result.included_group_count, 1);
        assert_eq!(result.skipped_zero_anchor_group_count, 1);
        assert_eq!(result.rows[0].q_any, None);
        assert_eq!(result.rows[0].q_same, None);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn inconsistent_response_metadata_is_rejected() {
        let root = temp_path("inconsistent");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("transitions.csv");
        fs::write(
            &path,
            "layer,anchor_feature_class,anchor_feature_class_label,response_feature_class,response_feature_class_label,anchor_count,frame_count,possible_match_count,matched_count\n\
             candidate,compact_quality,紧凑,compact_quality,紧凑,4,3,12,3\n\
             candidate,compact_quality,紧凑,weak_or_background,背景,5,3,15,2\n",
        )
        .unwrap();
        let error = summarize_sequence_feature_transition_flow(&path).unwrap_err();
        assert!(error.contains("元数据在响应类别之间不一致"));
        let _ = fs::remove_dir_all(root);
    }
}
