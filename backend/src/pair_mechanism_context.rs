use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const CONTEXT_GUARDRAILS: [&str; 4] = [
    "近邻对来自当前候选目录，不是物理双星真值。",
    "feature_class 是 detector-level 规则标签，不是物理类别。",
    "几何条件内没有双质量通过，只说明当前候选层/质量层的局部背景，不是噪点率。",
    "跨类别或共享机制只能触发联合 PSF、值域、独占像素和星表/WCS 复核。",
];

#[derive(Clone, Debug)]
struct Source {
    detection_id: i64,
    x: f64,
    y: f64,
    peak_x: f64,
    peak_y: f64,
    quality_passed: bool,
    feature_class: String,
    flags: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairRow {
    pub detection_id_a: i64,
    pub detection_id_b: i64,
    pub centroid_distance_px: f64,
    pub peak_distance_px: f64,
    pub feature_class_a: String,
    pub feature_class_b: String,
    pub same_feature_class: bool,
    pub quality_passed_a: bool,
    pub quality_passed_b: bool,
    pub both_quality_passed: bool,
    pub flags_a: String,
    pub flags_b: String,
    pub is_target_pair: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SummaryRow {
    pub feature_class_a: String,
    pub feature_class_b: String,
    pub pair_count: usize,
    pub cross_class_count: usize,
    pub both_quality_count: usize,
    pub both_quality_fraction: f64,
    pub centroid_distance_median_px: Option<f64>,
    pub peak_distance_median_px: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TargetRow {
    pub detection_id_a: i64,
    pub detection_id_b: i64,
    pub centroid_distance_px: f64,
    pub peak_distance_px: f64,
    pub feature_class_a: String,
    pub feature_class_b: String,
    pub same_feature_class: bool,
    pub quality_passed_a: bool,
    pub quality_passed_b: bool,
    pub both_quality_passed: bool,
    pub flags_a: String,
    pub flags_b: String,
    pub in_geometry_context: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextResult {
    pub catalog_path: String,
    pub source_count: usize,
    pub quality_count: usize,
    pub target_ids: [i64; 2],
    pub pairs: Vec<PairRow>,
    pub summaries: Vec<SummaryRow>,
    pub target: TargetRow,
    pub parameters: Value,
    pub conclusion: String,
}

impl ContextResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "source_count": self.source_count,
            "quality_count": self.quality_count,
            "target_ids": self.target_ids,
            "pairs": self.pairs,
            "summaries": self.summaries,
            "target": self.target,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation_guardrails": CONTEXT_GUARDRAILS,
        })
    }
}

fn read_catalog(path: &Path) -> Result<Vec<Source>> {
    let bytes = fs::read(path).map_err(|error| format!("无法读取源级 CSV：{path:?}: {error}"))?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let required = [
        "detection_id",
        "feature_class",
        "peak_x",
        "peak_y",
        "quality_passed",
        "x",
        "y",
    ];
    let mut missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|name| !headers.iter().any(|header| header == *name))
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!("catalog missing fields: {}", missing.join(", ")));
    }
    let column = |name: &str| {
        headers
            .iter()
            .enumerate()
            .filter_map(|(index, header)| (header == name).then_some(index))
            .last()
            .unwrap()
    };
    let indices: BTreeMap<&str, usize> =
        required.iter().map(|name| (*name, column(name))).collect();
    let flags_index = headers
        .iter()
        .enumerate()
        .filter_map(|(index, header)| (header == "flags").then_some(index))
        .last();
    let mut sources = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for record in reader.records() {
        let row = record.map_err(|error| error.to_string())?;
        let value = |name: &str| row.get(indices[name]).unwrap_or("").trim();
        let detection_id = value("detection_id").parse::<i64>().map_err(|_| {
            format!(
                "detection_id must be an integer, got {:?}",
                value("detection_id")
            )
        })?;
        if !seen.insert(detection_id) {
            return Err(format!("duplicate detection_id: {detection_id}"));
        }
        let feature_class = value("feature_class").to_owned();
        if feature_class.is_empty() {
            return Err(format!(
                "feature_class is empty for detection_id {detection_id}"
            ));
        }
        let parse_float = |field: &str| -> Result<f64> {
            let raw = value(field);
            let number = raw
                .parse::<f64>()
                .map_err(|_| format!("{field} must be numeric, got {raw:?}"))?;
            if !number.is_finite() {
                return Err(format!("{field} must be finite"));
            }
            Ok(number)
        };
        let raw_bool = value("quality_passed");
        let quality_passed = match raw_bool.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => true,
            "false" | "0" | "no" | "n" => false,
            _ => return Err(format!("quality_passed must be boolean, got {raw_bool:?}")),
        };
        sources.push(Source {
            detection_id,
            x: parse_float("x")?,
            y: parse_float("y")?,
            peak_x: parse_float("peak_x")?,
            peak_y: parse_float("peak_y")?,
            quality_passed,
            feature_class,
            flags: flags_index
                .map(|index| row.get(index).unwrap_or("None").trim().to_owned())
                .unwrap_or_default(),
        });
    }
    if sources.is_empty() {
        return Err("catalog contains no sources".into());
    }
    Ok(sources)
}

fn validate_geometry(centroid_max: f64, peak_min: f64, peak_max: f64) -> Result<()> {
    for (name, value) in [
        ("max_centroid_distance_px", centroid_max),
        ("peak_distance_min_px", peak_min),
        ("peak_distance_max_px", peak_max),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be finite and positive"));
        }
    }
    if peak_min > peak_max {
        return Err("peak_distance_min_px must not exceed peak_distance_max_px".into());
    }
    Ok(())
}

fn pair_row(first: &Source, second: &Source, target_ids: [i64; 2]) -> PairRow {
    let (a, b) = if first.detection_id <= second.detection_id {
        (first, second)
    } else {
        (second, first)
    };
    PairRow {
        detection_id_a: a.detection_id,
        detection_id_b: b.detection_id,
        centroid_distance_px: (a.x - b.x).hypot(a.y - b.y),
        peak_distance_px: (a.peak_x - b.peak_x).hypot(a.peak_y - b.peak_y),
        feature_class_a: a.feature_class.clone(),
        feature_class_b: b.feature_class.clone(),
        same_feature_class: a.feature_class == b.feature_class,
        quality_passed_a: a.quality_passed,
        quality_passed_b: b.quality_passed,
        both_quality_passed: a.quality_passed && b.quality_passed,
        flags_a: a.flags.clone(),
        flags_b: b.flags.clone(),
        is_target_pair: [a.detection_id, b.detection_id] == target_ids,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

pub fn run(
    catalog_path: &Path,
    target_ids: &[i64],
    max_centroid_distance_px: f64,
    peak_distance_min_px: f64,
    peak_distance_max_px: f64,
) -> Result<ContextResult> {
    let mut ids = target_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != 2 {
        return Err("target_ids must contain exactly two distinct detection IDs".into());
    }
    let target_ids = [ids[0], ids[1]];
    validate_geometry(
        max_centroid_distance_px,
        peak_distance_min_px,
        peak_distance_max_px,
    )?;
    let sources = read_catalog(catalog_path)?;
    let by_id: BTreeMap<i64, usize> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| (source.detection_id, index))
        .collect();
    let missing: Vec<i64> = target_ids
        .iter()
        .copied()
        .filter(|id| !by_id.contains_key(id))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "target detection IDs not in source catalog: {missing:?}"
        ));
    }

    // A sorted x sweep avoids checking distant sparse catalog rows while retaining
    // every pair within the inclusive radius. Dense neighborhoods necessarily emit
    // quadratic output in both this implementation and the Python baseline.
    let mut x_order: Vec<usize> = (0..sources.len()).collect();
    x_order.sort_by(|a, b| sources[*a].x.total_cmp(&sources[*b].x));
    let mut pairs = Vec::new();
    for left in 0..x_order.len() {
        let first = &sources[x_order[left]];
        for &right_index in &x_order[left + 1..] {
            let second = &sources[right_index];
            if second.x - first.x > max_centroid_distance_px {
                break;
            }
            if (second.y - first.y).abs() > max_centroid_distance_px {
                continue;
            }
            let row = pair_row(first, second, target_ids);
            if row.centroid_distance_px <= max_centroid_distance_px
                && row.peak_distance_px >= peak_distance_min_px
                && row.peak_distance_px <= peak_distance_max_px
            {
                pairs.push(row);
            }
        }
    }
    pairs.sort_by(|a, b| {
        a.centroid_distance_px
            .total_cmp(&b.centroid_distance_px)
            .then_with(|| a.detection_id_a.cmp(&b.detection_id_a))
            .then_with(|| a.detection_id_b.cmp(&b.detection_id_b))
    });

    let mut groups: BTreeMap<(String, String), Vec<&PairRow>> = BTreeMap::new();
    for row in &pairs {
        groups
            .entry((row.feature_class_a.clone(), row.feature_class_b.clone()))
            .or_default()
            .push(row);
    }
    let summaries = groups
        .into_iter()
        .map(|((feature_class_a, feature_class_b), rows)| {
            let pair_count = rows.len();
            let both_quality_count = rows.iter().filter(|row| row.both_quality_passed).count();
            let cross_class_count = rows.iter().filter(|row| !row.same_feature_class).count();
            let mut centroid: Vec<f64> = rows.iter().map(|row| row.centroid_distance_px).collect();
            let mut peak: Vec<f64> = rows.iter().map(|row| row.peak_distance_px).collect();
            SummaryRow {
                feature_class_a,
                feature_class_b,
                pair_count,
                cross_class_count,
                both_quality_count,
                both_quality_fraction: both_quality_count as f64 / pair_count as f64,
                centroid_distance_median_px: Some(median(&mut centroid)),
                peak_distance_median_px: Some(median(&mut peak)),
            }
        })
        .collect();
    let target_pair = pair_row(
        &sources[by_id[&target_ids[0]]],
        &sources[by_id[&target_ids[1]]],
        target_ids,
    );
    let target = TargetRow {
        detection_id_a: target_pair.detection_id_a,
        detection_id_b: target_pair.detection_id_b,
        centroid_distance_px: target_pair.centroid_distance_px,
        peak_distance_px: target_pair.peak_distance_px,
        feature_class_a: target_pair.feature_class_a,
        feature_class_b: target_pair.feature_class_b,
        same_feature_class: target_pair.same_feature_class,
        quality_passed_a: target_pair.quality_passed_a,
        quality_passed_b: target_pair.quality_passed_b,
        both_quality_passed: target_pair.both_quality_passed,
        flags_a: target_pair.flags_a,
        flags_b: target_pair.flags_b,
        in_geometry_context: pairs.iter().any(|row| row.is_target_pair),
    };
    let same_count = pairs.iter().filter(|row| row.same_feature_class).count();
    let cross_count = pairs.len() - same_count;
    let both_quality_count = pairs.iter().filter(|row| row.both_quality_passed).count();
    let conclusion = format!(
        "几何条件内找到 {} 对候选近邻，其中同类别 {} 对、跨类别 {} 对，双质量通过 {} 对。目标 pair 的类别为 {}+{}，{}该背景窗口内。该结果只表示候选目录中的机制一致性背景，不是物理双星或噪点真值。",
        pairs.len(),
        same_count,
        cross_count,
        both_quality_count,
        target.feature_class_a,
        target.feature_class_b,
        if target.in_geometry_context { "在" } else { "不在" },
    );
    Ok(ContextResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        source_count: sources.len(),
        quality_count: sources
            .iter()
            .filter(|source| source.quality_passed)
            .count(),
        target_ids,
        pairs,
        summaries,
        target,
        parameters: json!({
            "max_centroid_distance_px": max_centroid_distance_px,
            "peak_distance_min_px": peak_distance_min_px,
            "peak_distance_max_px": peak_distance_max_px,
            "pair_definition": "unordered candidate pair; measured centroid and integer peak geometry",
            "interpretation_boundary": "candidate-catalog mechanism context only; no physical star probability, precision, FDR, or pair truth",
        }),
        conclusion,
    })
}

fn write_csv(path: &Path, header: &[&str], rows: Vec<Vec<String>>) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(header)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn f(value: f64) -> String {
    format!("{value:?}")
}
fn b(value: bool) -> String {
    if value { "True" } else { "False" }.to_owned()
}

pub fn write_artifacts(result: &ContextResult, out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    write_csv(
        &out_dir.join("pair_mechanism_context_pairs.csv"),
        &[
            "detection_id_a",
            "detection_id_b",
            "centroid_distance_px",
            "peak_distance_px",
            "feature_class_a",
            "feature_class_b",
            "same_feature_class",
            "quality_passed_a",
            "quality_passed_b",
            "both_quality_passed",
            "flags_a",
            "flags_b",
            "is_target_pair",
        ],
        result
            .pairs
            .iter()
            .map(|row| {
                vec![
                    row.detection_id_a.to_string(),
                    row.detection_id_b.to_string(),
                    f(row.centroid_distance_px),
                    f(row.peak_distance_px),
                    row.feature_class_a.clone(),
                    row.feature_class_b.clone(),
                    b(row.same_feature_class),
                    b(row.quality_passed_a),
                    b(row.quality_passed_b),
                    b(row.both_quality_passed),
                    row.flags_a.clone(),
                    row.flags_b.clone(),
                    b(row.is_target_pair),
                ]
            })
            .collect(),
    )?;
    write_csv(
        &out_dir.join("pair_mechanism_context_summary.csv"),
        &[
            "feature_class_a",
            "feature_class_b",
            "pair_count",
            "cross_class_count",
            "both_quality_count",
            "both_quality_fraction",
            "centroid_distance_median_px",
            "peak_distance_median_px",
        ],
        result
            .summaries
            .iter()
            .map(|row| {
                vec![
                    row.feature_class_a.clone(),
                    row.feature_class_b.clone(),
                    row.pair_count.to_string(),
                    row.cross_class_count.to_string(),
                    row.both_quality_count.to_string(),
                    f(row.both_quality_fraction),
                    row.centroid_distance_median_px.map(f).unwrap_or_default(),
                    row.peak_distance_median_px.map(f).unwrap_or_default(),
                ]
            })
            .collect(),
    )?;
    write_csv(
        &out_dir.join("pair_mechanism_context_target.csv"),
        &[
            "detection_id_a",
            "detection_id_b",
            "centroid_distance_px",
            "peak_distance_px",
            "feature_class_a",
            "feature_class_b",
            "same_feature_class",
            "quality_passed_a",
            "quality_passed_b",
            "both_quality_passed",
            "flags_a",
            "flags_b",
            "in_geometry_context",
        ],
        vec![vec![
            result.target.detection_id_a.to_string(),
            result.target.detection_id_b.to_string(),
            f(result.target.centroid_distance_px),
            f(result.target.peak_distance_px),
            result.target.feature_class_a.clone(),
            result.target.feature_class_b.clone(),
            b(result.target.same_feature_class),
            b(result.target.quality_passed_a),
            b(result.target.quality_passed_b),
            b(result.target.both_quality_passed),
            result.target.flags_a.clone(),
            result.target.flags_b.clone(),
            b(result.target.in_geometry_context),
        ]],
    )?;
    let bytes = serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    fs::write(out_dir.join("pair_mechanism_context.json"), bytes).map_err(|error| error.to_string())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog = None;
    let mut output_dir = PathBuf::from("tmp/pair-mechanism-context");
    let mut target_ids = Vec::new();
    let mut centroid_max = 3.5;
    let mut peak_min = 3.5;
    let mut peak_max = 4.8;
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
            "--max-centroid-distance" => {
                centroid_max = value("--max-centroid-distance")?
                    .parse()
                    .map_err(|_| "--max-centroid-distance must be numeric")?;
                index += 1;
            }
            "--peak-distance-min" => {
                peak_min = value("--peak-distance-min")?
                    .parse()
                    .map_err(|_| "--peak-distance-min must be numeric")?;
                index += 1;
            }
            "--peak-distance-max" => {
                peak_max = value("--peak-distance-max")?
                    .parse()
                    .map_err(|_| "--peak-distance-max must be numeric")?;
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
    let result = run(&catalog, &target_ids, centroid_max, peak_min, peak_max)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-pair-mechanism-context",
        "out_dir": output_dir,
        "result_path": output_dir.join("pair_mechanism_context.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_context_orders_pairs_and_summarizes_groups() {
        let root = std::env::temp_dir().join(format!("rst19-pair-context-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(&catalog, "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class,flags\n1,0,0,0,0,true,range_anomaly,CODE_PATTERN\n2,2.738,0,4,0,false,crowded_blend,UNRESOLVED_BLEND\n3,20,0,20,0,true,compact_quality,\n").unwrap();
        let result = run(&catalog, &[2, 1, 2], 3.5, 3.5, 4.8).unwrap();
        assert_eq!(result.source_count, 3);
        assert_eq!(result.quality_count, 2);
        assert_eq!(result.pairs.len(), 1);
        assert_eq!(
            [
                result.pairs[0].detection_id_a,
                result.pairs[0].detection_id_b
            ],
            [1, 2]
        );
        assert!(result.target.in_geometry_context);
        assert!((result.target.centroid_distance_px - 2.738).abs() < 1e-12);
        assert_eq!(result.summaries[0].pair_count, 1);
    }

    #[test]
    fn flags_column_is_optional() {
        let root = std::env::temp_dir().join(format!(
            "rst19-pair-context-no-flags-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class\n1,0,0,0,0,true,compact\n2,1,0,4,0,false,blend\n",
        )
        .unwrap();
        let result = run(&catalog, &[1, 2], 2.0, 3.0, 5.0).unwrap();
        assert_eq!(result.target.flags_a, "");
        assert_eq!(result.target.flags_b, "");
    }
}
