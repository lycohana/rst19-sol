use crate::Result;
use csv::{Terminator, WriterBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

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

const INTERPRETATION_GUARDRAILS: [&str; 3] = [
    "configuration 是调用者提供的标签；JSON 本身未声明 min_flux_snr 等命令行参数。",
    "candidate persistence 不是一对一身份确认，仍可能被邻近源偶然命中。",
    "quality persistence 只描述通过当前质量层的响应，不能单独替代星表真值。",
];

#[derive(Clone, Debug, Deserialize)]
pub struct RunPath {
    pub configuration: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SequenceFeatureRunSummary {
    configuration: String,
    json_path: String,
    source_path: String,
    frame_count: i64,
    unique_frame_count: usize,
    required_presence: i64,
    association_radius_px: f64,
    first_frame_candidate_count: i64,
    first_frame_quality_count: i64,
    first_frame_quality_fraction: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SequenceFeatureParameterRow {
    configuration: String,
    feature_class: String,
    feature_class_label: String,
    anchor_candidate_count: i64,
    candidate_presence_ge_required_count: i64,
    candidate_presence_all_frames_count: i64,
    candidate_persistence_fraction: Option<f64>,
    anchor_quality_count: i64,
    quality_presence_ge_required_count: i64,
    quality_presence_all_frames_count: i64,
    quality_persistence_fraction: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SequenceFeatureTransitionComparisonRow {
    configuration: String,
    layer: String,
    anchor_feature_class: String,
    response_feature_class: String,
    anchor_count: i64,
    frame_count: i64,
    possible_match_count: i64,
    matched_count: i64,
    match_rate: Option<f64>,
    mean_matches_per_anchor: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct ComparisonResult {
    runs: Vec<SequenceFeatureRunSummary>,
    rows: Vec<SequenceFeatureParameterRow>,
    transitions: Vec<SequenceFeatureTransitionComparisonRow>,
    observations: Value,
}

impl ComparisonResult {
    fn as_dict(&self) -> Value {
        json!({
            "runs": &self.runs,
            "rows": &self.rows,
            "transitions": &self.transitions,
            "observations": &self.observations,
        })
    }
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn py_string(value: Option<&Value>, default: &str) -> String {
    match value {
        None => default.to_owned(),
        Some(Value::Null) => "None".to_owned(),
        Some(Value::Bool(value)) => {
            if *value {
                "True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        Some(value) => value.to_string(),
    }
}

fn number(value: Option<&Value>, field: &str, path: &Path) -> Result<f64> {
    let parsed = match value {
        Some(Value::Number(value)) => value.as_f64(),
        Some(Value::String(value)) => value.trim().parse::<f64>().ok(),
        Some(Value::Bool(value)) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    };
    parsed.ok_or_else(|| format!("{}: 字段 {field:?} 不是数字", path_text(path)))
}

fn integer(value: Option<&Value>, field: &str, path: &Path) -> Result<i64> {
    let parsed = number(value, field, path)?;
    if !parsed.is_finite() || parsed.fract() != 0.0 {
        return Err(format!("{}: 字段 {field:?} 不是整数", path_text(path)));
    }
    if parsed < i64::MIN as f64 || parsed >= 9_223_372_036_854_775_808.0 {
        return Err(format!("{}: 字段 {field:?} 超出整数范围", path_text(path)));
    }
    Ok(parsed as i64)
}

fn fraction(numerator: i64, denominator: i64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn class_order(feature_class: &str) -> (usize, &str) {
    (
        FEATURE_CLASS_ORDER
            .iter()
            .position(|known| *known == feature_class)
            .unwrap_or(FEATURE_CLASS_ORDER.len()),
        feature_class,
    )
}

fn read_json(path: &Path) -> Result<Value> {
    let source = fs::read_to_string(path)
        .map_err(|_| format!("无法读取序列审计 JSON：{}", path_text(path)))?;
    let payload: Value = serde_json::from_str(&source)
        .map_err(|error| format!("序列审计 JSON 无法解析：{}: {error}", path_text(path)))?;
    if !payload.is_object() {
        return Err(format!("序列审计 JSON 顶层必须是对象：{}", path_text(path)));
    }
    Ok(payload)
}

fn load_run(
    configuration: &str,
    path: &Path,
) -> Result<(
    SequenceFeatureRunSummary,
    Vec<SequenceFeatureParameterRow>,
    Vec<SequenceFeatureTransitionComparisonRow>,
)> {
    let payload = read_json(path)?;
    let frame_count = integer(payload.get("frame_count"), "frame_count", path)?;
    let required_presence = integer(payload.get("required_presence"), "required_presence", path)?;
    let association_radius_px = number(
        payload.get("association_radius_px"),
        "association_radius_px",
        path,
    )?;
    if frame_count < 1 || required_presence < 1 || association_radius_px <= 0.0 {
        return Err(format!(
            "{}: frame_count/required_presence/association_radius_px 不合法",
            path_text(path)
        ));
    }

    let frame_rows = payload
        .get("frame_rows")
        .and_then(Value::as_array)
        .filter(|rows| !rows.is_empty())
        .ok_or_else(|| format!("{}: 缺少非空 frame_rows", path_text(path)))?;
    let mut frame_indices = BTreeSet::new();
    for row in frame_rows {
        let Some(row) = row.as_object() else {
            continue;
        };
        if row.contains_key("frame_index") {
            frame_indices.insert(integer(row.get("frame_index"), "frame_index", path)?);
        }
    }
    if frame_indices.len() as i64 != frame_count {
        return Err(format!(
            "{}: metadata frame_count={}，但 frame_rows 的唯一帧数为 {}",
            path_text(path),
            frame_count,
            frame_indices.len()
        ));
    }
    let mut first_frame_rows = Vec::new();
    let mut object_rows = Vec::new();
    for row in frame_rows {
        if let Some(row) = row.as_object() {
            let index = integer(row.get("frame_index"), "frame_index", path)?;
            object_rows.push(row);
            if index == 1 {
                first_frame_rows.push(row);
            }
        }
    }
    if first_frame_rows.is_empty() {
        first_frame_rows = object_rows;
    }
    let first = first_frame_rows
        .first()
        .ok_or_else(|| format!("{}: frame_rows 没有可用记录", path_text(path)))?;
    let first_candidate_count = integer(first.get("candidate_count"), "candidate_count", path)?;
    let first_quality_count = integer(first.get("quality_count"), "quality_count", path)?;
    let source_path = py_string(first.get("path"), "");

    let persistence_rows = payload
        .get("persistence_rows")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{}: 缺少 persistence_rows", path_text(path)))?;
    let mut rows = Vec::with_capacity(persistence_rows.len());
    for item in persistence_rows {
        let item = item
            .as_object()
            .ok_or_else(|| format!("{}: persistence_rows 中存在非对象记录", path_text(path)))?;
        let feature_class = py_string(item.get("feature_class"), "");
        if feature_class.is_empty() {
            return Err(format!(
                "{}: persistence_rows 存在空 feature_class",
                path_text(path)
            ));
        }
        let anchor_candidate_count = integer(
            item.get("anchor_candidate_count"),
            "anchor_candidate_count",
            path,
        )?;
        let candidate_presence_ge_required_count = integer(
            item.get("candidate_presence_ge_required_count"),
            "candidate_presence_ge_required_count",
            path,
        )?;
        let candidate_presence_all_frames_count = integer(
            item.get("candidate_presence_all_frames_count"),
            "candidate_presence_all_frames_count",
            path,
        )?;
        let anchor_quality_count = integer(
            item.get("anchor_quality_count"),
            "anchor_quality_count",
            path,
        )?;
        let quality_presence_ge_required_count = integer(
            item.get("quality_presence_ge_required_count"),
            "quality_presence_ge_required_count",
            path,
        )?;
        let quality_presence_all_frames_count = integer(
            item.get("quality_presence_all_frames_count"),
            "quality_presence_all_frames_count",
            path,
        )?;
        rows.push(SequenceFeatureParameterRow {
            configuration: configuration.to_owned(),
            feature_class_label: py_string(item.get("feature_class_label"), &feature_class),
            candidate_persistence_fraction: fraction(
                candidate_presence_ge_required_count,
                anchor_candidate_count,
            ),
            quality_persistence_fraction: fraction(
                quality_presence_ge_required_count,
                anchor_quality_count,
            ),
            feature_class,
            anchor_candidate_count,
            candidate_presence_ge_required_count,
            candidate_presence_all_frames_count,
            anchor_quality_count,
            quality_presence_ge_required_count,
            quality_presence_all_frames_count,
        });
    }

    let transition_payload = match payload.get("class_transition_rows") {
        None => &[][..],
        Some(Value::Array(rows)) => rows.as_slice(),
        _ => {
            return Err(format!(
                "{}: class_transition_rows 必须是列表",
                path_text(path)
            ))
        }
    };
    let mut transitions = Vec::with_capacity(transition_payload.len());
    for item in transition_payload {
        let item = item.as_object().ok_or_else(|| {
            format!(
                "{}: class_transition_rows 中存在非对象记录",
                path_text(path)
            )
        })?;
        let optional_number = |field: &str| -> Result<Option<f64>> {
            match item.get(field) {
                None | Some(Value::Null) => Ok(None),
                value => number(value, field, path).map(Some),
            }
        };
        transitions.push(SequenceFeatureTransitionComparisonRow {
            configuration: configuration.to_owned(),
            layer: py_string(item.get("layer"), ""),
            anchor_feature_class: py_string(item.get("anchor_feature_class"), ""),
            response_feature_class: py_string(item.get("response_feature_class"), ""),
            anchor_count: integer(item.get("anchor_count"), "anchor_count", path)?,
            frame_count: integer(item.get("frame_count"), "frame_count", path)?,
            possible_match_count: integer(
                item.get("possible_match_count"),
                "possible_match_count",
                path,
            )?,
            matched_count: integer(item.get("matched_count"), "matched_count", path)?,
            match_rate: optional_number("match_rate")?,
            mean_matches_per_anchor: optional_number("mean_matches_per_anchor")?,
        });
    }
    let summary = SequenceFeatureRunSummary {
        configuration: configuration.to_owned(),
        json_path: path_text(path),
        source_path,
        frame_count,
        unique_frame_count: frame_indices.len(),
        required_presence,
        association_radius_px,
        first_frame_candidate_count: first_candidate_count,
        first_frame_quality_count: first_quality_count,
        first_frame_quality_fraction: fraction(first_quality_count, first_candidate_count),
    };
    Ok((summary, rows, transitions))
}

pub fn compare(runs: &[RunPath]) -> Result<Value> {
    if runs.len() < 2 {
        return Err("至少需要两组序列审计 JSON 才能进行比较".into());
    }
    let mut summaries = Vec::with_capacity(runs.len());
    let mut rows = Vec::new();
    let mut transitions = Vec::new();
    for run in runs {
        let (summary, mut run_rows, mut run_transitions) = load_run(&run.configuration, &run.path)?;
        summaries.push(summary);
        rows.append(&mut run_rows);
        transitions.append(&mut run_transitions);
    }
    let reference = &summaries[0];
    for summary in &summaries[1..] {
        if summary.frame_count != reference.frame_count {
            return Err("各运行的 frame_count 不一致，不能比较".into());
        }
        if summary.required_presence != reference.required_presence {
            return Err("各运行的 required_presence 不一致，不能比较".into());
        }
        if (summary.association_radius_px - reference.association_radius_px).abs() > 1e-9 {
            return Err("各运行的 association_radius_px 不一致，不能比较".into());
        }
    }
    rows.sort_by(|left, right| {
        left.configuration
            .cmp(&right.configuration)
            .then_with(|| class_order(&left.feature_class).cmp(&class_order(&right.feature_class)))
    });
    transitions.sort_by(|left, right| {
        left.configuration
            .cmp(&right.configuration)
            .then_with(|| left.layer.cmp(&right.layer))
            .then_with(|| {
                class_order(&left.anchor_feature_class)
                    .cmp(&class_order(&right.anchor_feature_class))
            })
            .then_with(|| {
                class_order(&left.response_feature_class)
                    .cmp(&class_order(&right.response_feature_class))
            })
    });
    let mut quality_persistence = Map::new();
    for summary in &summaries {
        let mut by_class = Map::new();
        for row in rows
            .iter()
            .filter(|row| row.configuration == summary.configuration)
        {
            by_class.insert(
                row.feature_class.clone(),
                json!(row.quality_persistence_fraction),
            );
        }
        quality_persistence.insert(summary.configuration.clone(), Value::Object(by_class));
    }
    let mut first_frame_counts = Map::new();
    for summary in &summaries {
        first_frame_counts.insert(
            summary.configuration.clone(),
            json!({
                "candidate_count": summary.first_frame_candidate_count,
                "quality_count": summary.first_frame_quality_count,
                "quality_fraction": summary.first_frame_quality_fraction,
            }),
        );
    }
    let observations = json!({
        "same_sequence_geometry": true,
        "same_frame_count": reference.frame_count,
        "same_required_presence": reference.required_presence,
        "same_association_radius_px": reference.association_radius_px,
        "first_frame_counts": first_frame_counts,
        "quality_persistence_by_configuration": quality_persistence,
        "interpretation_guardrails": INTERPRETATION_GUARDRAILS,
    });
    Ok(ComparisonResult {
        runs: summaries,
        rows,
        transitions,
        observations,
    }
    .as_dict())
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

fn write_csv(path: &Path, headers: &[&str], rows: Vec<Vec<String>>) -> Result<()> {
    use std::io::Write;
    if rows.is_empty() {
        let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
        file.write_all(&[0xef, 0xbb, 0xbf, b'\n'])
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    let mut file = writer.into_inner().map_err(|error| error.to_string())?;
    file.flush().map_err(|error| error.to_string())?;
    Ok(())
}

fn csv_rows<T>(rows: &[T], serialize: impl Fn(&T) -> Vec<String>) -> Vec<Vec<String>> {
    rows.iter().map(serialize).collect()
}

pub fn write_artifacts(result: &Value, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let summaries: Vec<SequenceFeatureRunSummary> =
        serde_json::from_value(result["runs"].clone()).map_err(|error| error.to_string())?;
    let rows: Vec<SequenceFeatureParameterRow> =
        serde_json::from_value(result["rows"].clone()).map_err(|error| error.to_string())?;
    let transitions: Vec<SequenceFeatureTransitionComparisonRow> =
        serde_json::from_value(result["transitions"].clone()).map_err(|error| error.to_string())?;
    let run_headers = [
        "configuration",
        "json_path",
        "source_path",
        "frame_count",
        "unique_frame_count",
        "required_presence",
        "association_radius_px",
        "first_frame_candidate_count",
        "first_frame_quality_count",
        "first_frame_quality_fraction",
    ];
    let run_rows = csv_rows(&summaries, |row| {
        vec![
            row.configuration.clone(),
            row.json_path.clone(),
            row.source_path.clone(),
            row.frame_count.to_string(),
            row.unique_frame_count.to_string(),
            row.required_presence.to_string(),
            python_float(row.association_radius_px),
            row.first_frame_candidate_count.to_string(),
            row.first_frame_quality_count.to_string(),
            optional_float(row.first_frame_quality_fraction),
        ]
    });
    write_csv(
        &out_dir.join("sequence_feature_parameter_runs.csv"),
        &run_headers,
        run_rows,
    )?;
    let row_headers = [
        "configuration",
        "feature_class",
        "feature_class_label",
        "anchor_candidate_count",
        "candidate_presence_ge_required_count",
        "candidate_presence_all_frames_count",
        "candidate_persistence_fraction",
        "anchor_quality_count",
        "quality_presence_ge_required_count",
        "quality_presence_all_frames_count",
        "quality_persistence_fraction",
    ];
    let row_values = csv_rows(&rows, |row| {
        vec![
            row.configuration.clone(),
            row.feature_class.clone(),
            row.feature_class_label.clone(),
            row.anchor_candidate_count.to_string(),
            row.candidate_presence_ge_required_count.to_string(),
            row.candidate_presence_all_frames_count.to_string(),
            optional_float(row.candidate_persistence_fraction),
            row.anchor_quality_count.to_string(),
            row.quality_presence_ge_required_count.to_string(),
            row.quality_presence_all_frames_count.to_string(),
            optional_float(row.quality_persistence_fraction),
        ]
    });
    write_csv(
        &out_dir.join("sequence_feature_parameter_profiles.csv"),
        &row_headers,
        row_values,
    )?;
    let transition_headers = [
        "configuration",
        "layer",
        "anchor_feature_class",
        "response_feature_class",
        "anchor_count",
        "frame_count",
        "possible_match_count",
        "matched_count",
        "match_rate",
        "mean_matches_per_anchor",
    ];
    let transition_values = csv_rows(&transitions, |row| {
        vec![
            row.configuration.clone(),
            row.layer.clone(),
            row.anchor_feature_class.clone(),
            row.response_feature_class.clone(),
            row.anchor_count.to_string(),
            row.frame_count.to_string(),
            row.possible_match_count.to_string(),
            row.matched_count.to_string(),
            optional_float(row.match_rate),
            optional_float(row.mean_matches_per_anchor),
        ]
    });
    write_csv(
        &out_dir.join("sequence_feature_parameter_transitions.csv"),
        &transition_headers,
        transition_values,
    )?;
    fs::write(
        out_dir.join("sequence_feature_parameter_comparison.json"),
        serde_json::to_vec_pretty(result).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(out_dir.to_owned())
}

pub fn parse_runs(values: &[String]) -> Result<Vec<RunPath>> {
    let mut result = Vec::with_capacity(values.len());
    let mut names = BTreeSet::new();
    for value in values {
        let (name, raw_path) = value
            .split_once('=')
            .ok_or_else(|| format!("--run 格式应为 NAME=JSON，收到：{value:?}"))?;
        let name = name.trim();
        let raw_path = raw_path.trim();
        if name.is_empty() || raw_path.is_empty() {
            return Err(format!("--run 格式应为 NAME=JSON，收到：{value:?}"));
        }
        if !names.insert(name.to_owned()) {
            return Err(format!("重复的运行标签：{name}"));
        }
        result.push(RunPath {
            configuration: name.to_owned(),
            path: PathBuf::from(raw_path),
        });
    }
    Ok(result)
}

fn parse_args(args: &[String]) -> Result<(Vec<String>, PathBuf)> {
    let mut runs = Vec::new();
    let mut output = PathBuf::from("tmp/sequence-feature-parameter-comparison");
    let mut index = 0;
    while index < args.len() {
        let value = &args[index];
        if value == "--run" {
            index += 1;
            runs.push(args.get(index).ok_or("--run requires NAME=JSON")?.clone());
        } else if let Some(value) = value.strip_prefix("--run=") {
            runs.push(value.to_owned());
        } else if value == "--out-dir" {
            index += 1;
            output = PathBuf::from(args.get(index).ok_or("--out-dir requires a path")?);
        } else if let Some(value) = value.strip_prefix("--out-dir=") {
            output = PathBuf::from(value);
        } else {
            return Err(format!("unknown argument {value}"));
        }
        index += 1;
    }
    Ok((runs, output))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (run_args, out_dir) = parse_args(args)?;
    let result = compare(&parse_runs(&run_args)?)?;
    let output = write_artifacts(&result, &out_dir)?;
    Ok(json!({
        "tool": "rst19-feature-sequence-compare",
        "out_dir": output,
        "result_path": output.join("sequence_feature_parameter_comparison.json"),
        "result": result,
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("比较已生成的序列特征持久性 JSON，不重新运行 FITS 检测\n\nUsage: rst19-feature-sequence-compare --run NAME=JSON --run NAME=JSON [--out-dir PATH]");
        return Ok(());
    }
    let (run_args, out_dir) = parse_args(args)?;
    let result = compare(&parse_runs(&run_args)?)?;
    let output = write_artifacts(&result, &out_dir)?;
    println!(
        "rst19-feature-sequence-compare: 比较表已写入 {}",
        path_text(&output)
    );
    for summary in result["runs"].as_array().ok_or("runs missing")? {
        println!(
            "{}: frames={} candidate={} quality={}",
            summary["configuration"].as_str().unwrap_or_default(),
            summary["unique_frame_count"],
            summary["first_frame_candidate_count"],
            summary["first_frame_quality_count"],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repeated_run_arguments_and_preserves_order() {
        let args = vec!["SNR9=last.json".to_owned(), "SNR3=first.json".to_owned()];
        let runs = parse_runs(&args).unwrap();
        assert_eq!(runs[0].configuration, "SNR9");
        assert_eq!(runs[1].path, PathBuf::from("first.json"));
        assert!(parse_runs(&["same=a.json".into(), "same=b.json".into()]).is_err());
    }

    #[test]
    fn class_order_keeps_known_classes_before_unknown_values() {
        assert_eq!(
            class_order("compact_quality").cmp(&class_order("unknown")),
            std::cmp::Ordering::Less
        );
    }
}
