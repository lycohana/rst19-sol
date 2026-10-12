//! Auditable merge of layered Gaia CSV catalogs.
//!
//! Rows are keyed by source_id; duplicate rows are collapsed while preserving
//! the richer representation and filling its missing fields. Completeness is
//! inherited only when every input sidecar binds its CSV, all footprints agree,
//! and no critical source fields conflict.

use crate::{catalog_csv, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};

const CATALOG_COMPAT_COLUMNS: &[&str] = &[
    "source_id",
    "ra_deg",
    "dec_deg",
    "magnitude",
    "magnitude_error",
    "color",
    "color_name",
    "pmra_mas_yr",
    "pmdec_mas_yr",
    "ref_epoch",
    "parallax_mas",
    "parallax_error_mas",
    "distance_pc",
    "distance_lower_pc",
    "distance_upper_pc",
    "distance_source",
    "mg_gspphot",
    "mg_gspphot_lower",
    "mg_gspphot_upper",
    "mg_gspphot_source",
    "extinction_mag",
    "extinction_error_mag",
    "extinction_band",
    "extinction_system",
    "extinction_source",
    "catalog_name",
    "photometric_system",
    "photometric_band",
    "magnitude_source",
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_bp_rp_excess_factor",
    "parallax",
    "parallax_error",
    "ruwe",
    "duplicated_source",
    "visibility_periods_used",
    "phot_variable_flag",
    "distance_gspphot",
    "distance_gspphot_lower",
    "distance_gspphot_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
];

const CRITICAL_FIELDS: &[&str] = &[
    "ra",
    "dec",
    "ra_deg",
    "dec_deg",
    "magnitude",
    "magnitude_source",
    "phot_g_mean_mag",
    "parallax",
    "parallax_mas",
    "distance_pc",
    "distance_gspphot",
    "mg_gspphot",
    "mg_gspphot_lower",
    "mg_gspphot_upper",
    "extinction_mag",
    "ag_gspphot",
];

#[derive(Debug)]
struct CsvInput {
    rows: Vec<BTreeMap<String, String>>,
    fields: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct Footprint {
    ra: f64,
    dec: f64,
    radius: f64,
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

fn sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn read_csv(path: &Path) -> Result<CsvInput> {
    if !path.is_file() {
        return Err(format!("星表文件不存在：{}", path.display()));
    }
    if !path
        .extension()
        .is_some_and(|extension| extension.to_string_lossy().eq_ignore_ascii_case("csv"))
    {
        return Err(format!("星表必须是 CSV：{}", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let headers = reader
        .headers()
        .map_err(|error| format!("星表不是有效 UTF-8 CSV：{} ({error})", path.display()))?
        .clone();
    let raw_fields: Vec<String> = headers.iter().map(str::to_owned).collect();
    let fields: Vec<String> = raw_fields
        .iter()
        .map(|name| name.trim().to_owned())
        .filter(|name| !name.is_empty())
        .collect();
    let source_index = raw_fields
        .iter()
        .rposition(|name| name.trim().eq_ignore_ascii_case("source_id"))
        .ok_or_else(|| format!("星表缺少 source_id 列：{}", path.display()))?;

    let mut unique_fields = Vec::with_capacity(fields.len() + 1);
    unique_fields.push("source_id".to_owned());
    for field in &fields {
        if !unique_fields.contains(field) {
            unique_fields.push(field.clone());
        }
    }

    let mut rows = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let record = record.map_err(|error| error.to_string())?;
        let mut row = BTreeMap::new();
        for (column, raw_name) in raw_fields.iter().enumerate() {
            let name = raw_name.trim();
            if !name.is_empty() {
                row.insert(
                    name.to_owned(),
                    record.get(column).unwrap_or("").trim().to_owned(),
                );
            }
        }
        let source_id = record.get(source_index).unwrap_or("").trim().to_owned();
        if source_id.is_empty() {
            return Err(format!(
                "{} 第 {} 行缺少 source_id",
                path.display(),
                index + 2
            ));
        }
        row.insert("source_id".to_owned(), source_id);
        rows.push(row);
    }
    Ok(CsvInput {
        rows,
        fields: unique_fields,
    })
}

fn sidecar_path(path: &Path) -> Option<PathBuf> {
    [".meta.json", ".tiled.meta.json"]
        .into_iter()
        .find_map(|suffix| {
            let candidate = PathBuf::from(format!("{}{suffix}", path.display()));
            candidate.is_file().then_some(candidate)
        })
}

fn read_audit(path: &Path) -> (Option<Value>, Option<PathBuf>, Option<String>) {
    let Some(audit_path) = sidecar_path(path) else {
        return (
            None,
            None,
            Some("缺少 .meta.json 或 .tiled.meta.json".into()),
        );
    };
    let bytes = match fs::read(&audit_path) {
        Ok(bytes) => bytes,
        Err(_) => {
            return (
                None,
                Some(audit_path),
                Some("审计文件不可读取：OSError".into()),
            )
        }
    };
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text,
        Err(_) => {
            return (
                None,
                Some(audit_path),
                Some("审计文件不可读取：UnicodeDecodeError".into()),
            )
        }
    };
    let payload: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(_) => {
            return (
                None,
                Some(audit_path),
                Some("审计文件不可读取：JSONDecodeError".into()),
            )
        }
    };
    if !payload.is_object() {
        return (
            None,
            Some(audit_path),
            Some("审计文件不是 JSON 对象".into()),
        );
    }
    (Some(payload), Some(audit_path), None)
}

fn audit_value(payload: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        let value = payload.get(*key)?;
        let number = match value {
            Value::Number(number) => number.as_f64(),
            Value::String(text) => text.trim().parse::<f64>().ok(),
            _ => None,
        }?;
        number.is_finite().then_some(number)
    })
}

fn footprint(payload: &Value) -> Option<Footprint> {
    Some(Footprint {
        ra: audit_value(payload, &["center_ra_deg", "query_center_ra_deg"])?.rem_euclid(360.0),
        dec: audit_value(payload, &["center_dec_deg", "query_center_dec_deg"])?,
        radius: audit_value(payload, &["search_radius_deg", "query_radius_deg"])?,
    })
}

fn audit_binding_error(
    rows: &[BTreeMap<String, String>],
    payload: Option<&Value>,
    digest: &str,
) -> Option<String> {
    let payload = payload?;
    if payload.get("complete") != Some(&Value::Bool(true)) {
        return None;
    }
    let declared_sha = payload
        .get("csv_sha256")
        .or_else(|| payload.get("sha256"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let Some(declared_sha) = declared_sha else {
        return Some("complete=true 审计缺少 csv_sha256，无法绑定当前 CSV".into());
    };
    if !declared_sha.trim().eq_ignore_ascii_case(digest) {
        return Some("complete=true 审计的 csv_sha256 与当前 CSV 不一致".into());
    }
    let count_value = payload
        .get("csv_row_count")
        .or_else(|| payload.get("row_count"));
    let count = count_value.and_then(|value| match value {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.trim().parse::<i64>().ok(),
        Value::Bool(value) => Some(i64::from(*value)),
        _ => None,
    });
    let Some(count) = count else {
        return Some("complete=true 审计缺少 csv_row_count，无法核对当前 CSV 行数".into());
    };
    if count != rows.len() as i64 {
        return Some(format!(
            "complete=true 审计的 csv_row_count={count} 与当前 CSV 行数={} 不一致",
            rows.len()
        ));
    }
    None
}

fn footprint_equal(left: Footprint, right: Footprint) -> bool {
    (left.ra - right.ra).abs() <= 1e-6
        && (left.dec - right.dec).abs() <= 1e-6
        && (left.radius - right.radius).abs() <= 1e-6
}

fn richness(row: &BTreeMap<String, String>) -> usize {
    row.values()
        .filter(|value| !value.trim().is_empty())
        .count()
}

fn numeric_equal(left: &str, right: &str) -> bool {
    match (left.parse::<f64>(), right.parse::<f64>()) {
        (Ok(a), Ok(b)) if a.is_finite() && b.is_finite() => {
            (a - b).abs() <= 1e-12_f64.max(1e-9 * a.abs().max(b.abs()))
        }
        _ => left == right,
    }
}

fn values_equal(field: &str, left: &str, right: &str) -> bool {
    left == right
        || (CRITICAL_FIELDS.contains(&field.trim().to_lowercase().as_str())
            && numeric_equal(left, right))
}

fn merge_row(
    rows_by_id: &mut BTreeMap<String, BTreeMap<String, String>>,
    incoming: BTreeMap<String, String>,
    conflict_ids: &mut BTreeSet<String>,
) -> bool {
    let source_id = incoming.get("source_id").cloned().unwrap_or_default();
    let Some(existing) = rows_by_id.remove(&source_id) else {
        rows_by_id.insert(source_id, incoming);
        return false;
    };
    let (mut base, supplement) = if richness(&incoming) > richness(&existing) {
        (incoming, existing)
    } else {
        (existing, incoming)
    };
    let mut conflict = false;
    for (field, value) in supplement {
        let value = value.trim();
        let current = base.get(&field).map(String::as_str).unwrap_or("").trim();
        if current.is_empty() && !value.is_empty() {
            base.insert(field, value.to_owned());
        } else if !current.is_empty() && !value.is_empty() && !values_equal(&field, current, value)
        {
            conflict = true;
        }
    }
    rows_by_id.insert(source_id.clone(), base);
    if conflict {
        conflict_ids.insert(source_id);
    }
    true
}

/// Merge two or more Gaia CSV files and emit the merged CSV plus integrity sidecar.
pub fn merge_catalog_csvs(input_paths: &[PathBuf], output_path: &Path) -> Result<Value> {
    let paths = input_paths
        .iter()
        .map(|path| absolute_path(path))
        .collect::<Result<Vec<_>>>()?;
    if paths.len() < 2 {
        return Err("至少需要两份星表才能执行分层合并".into());
    }
    let output = absolute_path(output_path)?;
    if !output
        .extension()
        .is_some_and(|extension| extension.to_string_lossy().eq_ignore_ascii_case("csv"))
    {
        return Err("output_path 必须使用 .csv 后缀".into());
    }
    if paths.contains(&output) {
        return Err("输出文件不能覆盖输入星表".into());
    }

    let mut rows_by_id = BTreeMap::<String, BTreeMap<String, String>>::new();
    let mut fields = CATALOG_COMPAT_COLUMNS
        .iter()
        .map(|field| (*field).to_owned())
        .collect::<Vec<_>>();
    let mut field_set = fields.iter().cloned().collect::<HashSet<_>>();
    let mut conflict_ids = BTreeSet::new();
    let mut duplicate_count = 0usize;
    let mut inputs = Vec::new();
    let mut footprints = Vec::new();
    let mut input_audits_complete = true;
    let mut input_audit_bindings_valid = true;

    for path in &paths {
        let input = read_csv(path)?;
        catalog_csv::load_catalog_csv(path).map_err(|error| {
            format!("星表无法按 CatalogSource 读取：{}：{error}", path.display())
        })?;
        for field in input.fields {
            if field_set.insert(field.clone()) {
                fields.push(field);
            }
        }
        let (audit, audit_path, mut audit_error) = read_audit(path);
        let digest = sha256(path)?;
        let binding_error = audit_binding_error(&input.rows, audit.as_ref(), &digest);
        if let Some(error) = &binding_error {
            audit_error = Some(match audit_error {
                Some(previous) => format!("{previous}; {error}"),
                None => error.clone(),
            });
            input_audit_bindings_valid = false;
        }
        let audit_complete = audit.as_ref().is_some_and(|payload| {
            payload.get("complete") == Some(&Value::Bool(true)) && audit_error.is_none()
        });
        input_audits_complete &= audit_complete;
        let input_footprint = audit.as_ref().and_then(footprint);
        if let Some(value) = input_footprint {
            footprints.push(value);
        }
        let footprint_value = input_footprint.map(|value| {
            json!({
                "center_ra_deg": value.ra,
                "center_dec_deg": value.dec,
                "radius_deg": value.radius,
            })
        });
        inputs.push(json!({
            "path": path.to_string_lossy(),
            "sha256": digest,
            "row_count": input.rows.len(),
            "audit_path": audit_path.as_ref().map(|value| value.to_string_lossy().into_owned()),
            "audit_complete": audit_complete,
            "audit_error": audit_error,
            "audit_binding_valid": binding_error.is_none(),
            "footprint": footprint_value,
            "min_g_mag": audit.as_ref().and_then(|value| value.get("min_g_mag")).cloned().unwrap_or(Value::Null),
            "max_g_mag": audit.as_ref().and_then(|value| value.get("max_g_mag")).cloned().unwrap_or(Value::Null),
        }));
        for row in input.rows {
            duplicate_count += usize::from(merge_row(&mut rows_by_id, row, &mut conflict_ids));
        }
    }

    let footprint_consistent = !footprints.is_empty()
        && footprints.len() == paths.len()
        && footprints
            .iter()
            .skip(1)
            .all(|value| footprint_equal(footprints[0], *value));
    let complete = input_audits_complete
        && input_audit_bindings_valid
        && footprint_consistent
        && conflict_ids.is_empty();

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::Any(b'\n'))
        .from_path(&output)
        .map_err(|error| error.to_string())?;
    writer
        .write_record(&fields)
        .map_err(|error| error.to_string())?;
    for row in rows_by_id.values() {
        writer
            .write_record(
                fields
                    .iter()
                    .map(|field| row.get(field).map(String::as_str).unwrap_or("")),
            )
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    drop(writer);

    let merged_sources = catalog_csv::load_catalog_csv(&output).map_err(|error| {
        format!(
            "合并结果无法按 CatalogSource 读取：{}：{error}",
            output.display()
        )
    })?;
    if merged_sources.len() != rows_by_id.len() {
        return Err("合并结果行数与唯一 source_id 数量不一致".into());
    }

    let audit_path = PathBuf::from(format!("{}.meta.json", output.display()));
    let conflicting_source_ids = conflict_ids.iter().cloned().collect::<Vec<_>>();
    let mut payload = json!({
        "schema_version": 1,
        "catalog": "Gaia DR3 merged layers",
        "output_path": output.to_string_lossy(),
        "audit_path": audit_path.to_string_lossy(),
        "input_paths": paths.iter().map(|path| path.to_string_lossy().into_owned()).collect::<Vec<_>>(),
        "row_count": rows_by_id.len(),
        "duplicate_count": duplicate_count,
        "conflict_count": conflict_ids.len(),
        "conflicting_source_ids": conflicting_source_ids,
        "complete": complete,
        "footprint_consistent": footprint_consistent,
        "input_audits_complete": input_audits_complete,
        "input_audit_bindings_valid": input_audit_bindings_valid,
        "interpretation": "A merged catalog is eligible for formal downstream provenance only when complete is true. Duplicate source_id rows are collapsed; field conflicts and missing input audits keep it incomplete.",
        "csv_row_count": rows_by_id.len(),
        "csv_sha256": sha256(&output)?,
        "csv_byte_count": fs::metadata(&output).map_err(|error| error.to_string())?.len(),
        "inputs": inputs,
    });
    if !complete {
        let mut reasons = Vec::new();
        if !input_audits_complete {
            reasons.push("至少一份输入缺少 complete=true 审计".to_owned());
        }
        if !input_audit_bindings_valid {
            reasons.push("至少一份 complete=true 审计未与当前 CSV 的 SHA-256/行数绑定".to_owned());
        }
        if !footprint_consistent {
            reasons.push("输入天空覆盖不一致或缺少覆盖字段".to_owned());
        }
        if !conflict_ids.is_empty() {
            reasons.push(format!(
                "{} 个 source_id 存在关键字段冲突",
                conflict_ids.len()
            ));
        }
        payload["incomplete_reasons"] = json!(reasons);
    }
    let mut audit_file = fs::File::create(&audit_path).map_err(|error| error.to_string())?;
    audit_file
        .write_all(
            serde_json::to_string_pretty(&payload)
                .map_err(|error| error.to_string())?
                .as_bytes(),
        )
        .and_then(|_| audit_file.write_all(b"\n"))
        .map_err(|error| error.to_string())?;
    Ok(payload)
}

/// Desktop adapter for the existing rst19-gaia-merge argument contract.
pub fn run_desktop(args: &[String], out_dir: &Path) -> Result<Value> {
    let mut catalogs = Vec::new();
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--out" {
            index += 1;
            output = args.get(index).map(PathBuf::from);
        } else if let Some(value) = argument.strip_prefix("--out=") {
            output = Some(PathBuf::from(value));
        } else if argument.starts_with('-') {
            return Err(format!("未知参数：{argument}"));
        } else {
            catalogs.push(PathBuf::from(argument));
        }
        index += 1;
    }
    let output = output.ok_or("缺少 --out 参数")?;
    let result = merge_catalog_csvs(&catalogs, &output)?;
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let result_path = out_dir.join("gaia_merge_result.json");
    fs::write(
        &result_path,
        serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(json!({
        "tool": "rst19-gaia-merge",
        "out_dir": out_dir.to_string_lossy(),
        "result_path": result_path.to_string_lossy(),
        "result": result,
    }))
}

/// Console adapter preserving the legacy exit code (3 means incomplete provenance).
pub fn run_cli(args: &[String]) -> Result<i32> {
    if args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("用法: rst19-gaia-merge <catalog.csv> <catalog.csv> [<catalog.csv> ...] --out <merged.csv>");
        println!("按 source_id 合并 Gaia 层，并继承输入完整性审计。");
        return Ok(0);
    }
    let mut catalogs = Vec::new();
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--out" {
            index += 1;
            output = args.get(index).map(PathBuf::from);
        } else if let Some(value) = argument.strip_prefix("--out=") {
            output = Some(PathBuf::from(value));
        } else if argument.starts_with('-') {
            return Err(format!("未知参数：{argument}"));
        } else {
            catalogs.push(PathBuf::from(argument));
        }
        index += 1;
    }
    let output = output.ok_or("--out 为必需参数")?;
    let payload = merge_catalog_csvs(&catalogs, &output)?;
    let complete = payload["complete"].as_bool().unwrap_or(false);
    eprintln!(
        "Gaia merged {}: {} unique rows -> {} (duplicates={}, conflicts={}; audit: {})",
        if complete { "complete" } else { "incomplete" },
        payload["row_count"].as_u64().unwrap_or(0),
        payload["output_path"].as_str().unwrap_or(""),
        payload["duplicate_count"].as_u64().unwrap_or(0),
        payload["conflict_count"].as_u64().unwrap_or(0),
        payload["audit_path"].as_str().unwrap_or(""),
    );
    Ok(if complete { 0 } else { 3 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rst19-catalog-merge-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn merge_collapses_ids_and_marks_conflict_incomplete() {
        let root = temp_dir();
        let first = root.join("first.csv");
        let second = root.join("second.csv");
        fs::write(
            &first,
            "source_id,ra_deg,dec_deg,magnitude\n1,129.5,-1.8,6.2\n",
        )
        .unwrap();
        fs::write(
            &second,
            "source_id,ra_deg,dec_deg,magnitude,parallax_mas\n1,129.5,-1.8,6.9,2.3\n2,129.7,-1.6,15,1.2\n",
        )
        .unwrap();
        let result = merge_catalog_csvs(&[first, second], &root.join("merged.csv")).unwrap();
        assert_eq!(result["row_count"], 2);
        assert_eq!(result["duplicate_count"], 1);
        assert_eq!(result["conflict_count"], 1);
        assert_eq!(result["complete"], false);
        assert_eq!(result["conflicting_source_ids"][0], "1");
        let _ = fs::remove_dir_all(root);
    }
}
