//! Native compressed caches bound to input contents and scientific context.
//! The native namespace preserves existing legacy caches during migration.
use crate::{detection::DetectionResult, models::FitsFrame, pipeline::FrameAnalysis, Result};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::UNIX_EPOCH,
};
use unicode_casefold::UnicodeCaseFold;

const CACHE_VERSION: u64 = 1;
const MAX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

fn path_text(path: &Path) -> String {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix("\\\\?\\UNC\\") {
        format!("\\\\{rest}")
    } else {
        text.strip_prefix("\\\\?\\").unwrap_or(&text).to_string()
    }
}

fn modified_ns(metadata: &fs::Metadata) -> Result<u64> {
    let duration = metadata
        .modified()
        .map_err(|error| error.to_string())?
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?;
    u64::try_from(duration.as_nanos())
        .map_err(|_| "file modification timestamp is out of range".into())
}

pub fn file_identity(path: &Path) -> Result<Value> {
    let path = fs::canonicalize(path).map_err(|error| error.to_string())?;
    let before = fs::metadata(&path).map_err(|error| error.to_string())?;
    if !before.is_file() {
        return Err("cache input must be a regular file".into());
    }
    let mut stream = fs::File::open(&path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = stream
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let after = fs::metadata(&path).map_err(|error| error.to_string())?;
    if before.len() != after.len() || modified_ns(&before)? != modified_ns(&after)? {
        return Err("cache input changed while hashing".into());
    }
    Ok(
        json!({"path":path_text(&path),"size":after.len(),"mtime_ns":modified_ns(&after)?,"sha256":format!("{:x}",digest.finalize())}),
    )
}

fn directory_files(path: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(path).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        if kind.is_dir() {
            directory_files(&entry.path(), output)?;
        } else if entry.path().is_file() {
            output.push(entry.path());
        }
    }
    Ok(())
}

pub fn directory_identity(path: &Path) -> Result<Value> {
    let path = fs::canonicalize(path).map_err(|error| error.to_string())?;
    let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
    if !metadata.is_dir() {
        return Err("cache input must be a directory".into());
    }
    let mut children = Vec::new();
    directory_files(&path, &mut children)?;
    children.sort_by_key(|child| {
        child
            .strip_prefix(&path)
            .unwrap_or(child)
            .to_string_lossy()
            .replace('\\', "/")
    });
    let files = children.iter().map(|child| Ok(json!({"relative_path":child.strip_prefix(&path).map_err(|error|error.to_string())?.to_string_lossy().replace('\\',"/"),"file":file_identity(child)?}))).collect::<Result<Vec<_>>>()?;
    Ok(json!({"path":path_text(&path),"mtime_ns":modified_ns(&metadata)?,"files":files}))
}

fn path_identity(path: &Path) -> Result<Value> {
    if path.is_file() {
        Ok(json!({"kind":"file","file":file_identity(path)?}))
    } else if path.is_dir() {
        Ok(json!({"kind":"directory","directory":directory_identity(path)?}))
    } else {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|error| error.to_string())?
                .join(path)
        };
        Ok(json!({"kind":"missing","path":path_text(&absolute)}))
    }
}

fn path_parameter(key: &str) -> bool {
    let key = key.trim().to_lowercase();
    matches!(
        key.as_str(),
        "path"
            | "file"
            | "filename"
            | "directory"
            | "dir"
            | "data_dir"
            | "catalog_path"
            | "catalog_file"
            | "catalog_csv"
            | "wcs_path"
            | "extinction_map_path"
            | "input_directory"
    ) || ["_path", "_file", "_dir", "_directory"]
        .iter()
        .any(|suffix| key.ends_with(suffix))
}

pub fn canonical_identity(value: &Value) -> Result<Value> {
    canonical(value, 0)
}

fn canonical(value: &Value, depth: usize) -> Result<Value> {
    if depth > 32 {
        return Err("cache identity nesting is too deep".into());
    }
    match value {
        Value::Array(items) => Ok(
            json!({"kind":"list","items":items.iter().map(|item|canonical(item,depth+1)).collect::<Result<Vec<_>>>()?}),
        ),
        Value::Object(fields) => {
            let mut items = fields
                .iter()
                .map(|(key, item)| {
                    let identity = if path_parameter(key) {
                        item.as_str()
                            .map(|path| path_identity(Path::new(path)))
                            .unwrap_or_else(|| canonical(item, depth + 1))
                    } else {
                        canonical(item, depth + 1)
                    }?;
                    Ok((
                        serde_json::to_string(key).map_err(|error| error.to_string())?,
                        json!([key, identity]),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            items.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(
                json!({"kind":"mapping","items":items.into_iter().map(|(_,item)|item).collect::<Vec<_>>()}),
            )
        }
        _ => Ok(value.clone()),
    }
}

pub fn sequence_input_identity(paths: &[PathBuf]) -> Result<Value> {
    let frames = paths
        .iter()
        .map(|path| file_identity(path))
        .collect::<Result<Vec<_>>>()?;
    let known = frames
        .iter()
        .map(|frame| {
            (
                frame["path"].as_str().unwrap_or("").to_string(),
                frame.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let directories = paths
        .iter()
        .map(|path| {
            fs::canonicalize(path)
                .map_err(|error| error.to_string())
                .and_then(|path| {
                    path.parent()
                        .map(Path::to_path_buf)
                        .ok_or("input file has no directory".into())
                })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let directories = directories.iter().map(|directory| {
        let metadata = fs::metadata(directory).map_err(|error|error.to_string())?;
        let mut entries = fs::read_dir(directory).map_err(|error|error.to_string())?.map(|entry|entry.map_err(|error|error.to_string())).collect::<Result<Vec<_>>>()?;
        entries.sort_by_key(|entry|entry.file_name().to_string_lossy().case_fold().collect::<String>());
        let mut files = Vec::new();
        for entry in entries {
            let path = entry.path();
            if !path.is_file() || !path.extension().and_then(|extension|extension.to_str()).is_some_and(|extension|matches!(extension.to_lowercase().as_str(),"fits"|"fit"|"fts")) { continue; }
            let resolved = fs::canonicalize(&path).map_err(|error|error.to_string())?;
            let identity = known.get(&path_text(&resolved)).cloned().map(Ok).unwrap_or_else(||file_identity(&path))?;
            files.push(json!({"name":entry.file_name().to_string_lossy(),"file":identity}));
        }
        Ok(json!({"path":path_text(directory),"mtime_ns":modified_ns(&metadata)?,"files":files}))
    }).collect::<Result<Vec<_>>>()?;
    Ok(json!({"identity_version":1,"kind":"sequence","frames":frames,"directories":directories}))
}

pub fn cache_key(path: &Path, parameters: &Value, scientific_context: &Value) -> Result<String> {
    key(
        &json!({"native_cache_version":CACHE_VERSION,"kind":"frame-analysis","input":{"identity_version":1,"kind":"frame","file":file_identity(path)?},"parameters":canonical_identity(parameters)?,"scientific_context":canonical_identity(scientific_context)?}),
    )
}

pub fn sequence_cache_key(
    paths: &[PathBuf],
    parameters: &Value,
    scientific_context: &Value,
) -> Result<String> {
    key(
        &json!({"native_cache_version":CACHE_VERSION,"kind":"sequence","input":sequence_input_identity(paths)?,"parameters":canonical_identity(parameters)?,"scientific_context":canonical_identity(scientific_context)?}),
    )
}

fn key(descriptor: &Value) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(descriptor).map_err(|error| error.to_string())?)
    ))
}

fn cache_path(directory: &Path, key: &str, sequence: bool) -> Result<PathBuf> {
    if key.is_empty()
        || key.chars().any(|character| {
            !character.is_ascii_alphanumeric() && character != '-' && character != '_'
        })
    {
        return Err("cache key must be a safe filename component".into());
    }
    Ok(directory.join(format!(
        "{key}.native{}.json.gz",
        if sequence { ".sequence" } else { "" }
    )))
}

fn save(path: &Path, payload: &Value) -> Result<()> {
    let directory = path.parent().ok_or("cache path has no parent")?;
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let temporary = directory.join(format!(
        ".native-cache-{}-{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        serde_json::to_writer(&mut encoder, payload).map_err(|error| error.to_string())?;
        encoder.flush().map_err(|error| error.to_string())?;
        encoder
            .finish()
            .map_err(|error| error.to_string())?
            .sync_all()
            .map_err(|error| error.to_string())?;
        fs::rename(&temporary, path).map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn load(path: &Path) -> Option<Value> {
    let file = fs::File::open(path).ok()?;
    let mut decoder = GzDecoder::new(file).take(MAX_CACHE_BYTES + 1);
    let mut data = Vec::new();
    decoder.read_to_end(&mut data).ok()?;
    if data.len() as u64 > MAX_CACHE_BYTES {
        return None;
    }
    serde_json::from_slice(&data).ok()
}

pub fn save_analysis(
    directory: &Path,
    cache_key: &str,
    analysis: &FrameAnalysis,
) -> Result<PathBuf> {
    let path = cache_path(directory, cache_key, false)?;
    let payload = json!({"native_cache_version":CACHE_VERSION,"cache_kind":"frame-analysis","cache_key":cache_key,"input_identity":{"identity_version":1,"kind":"frame","file":file_identity(&analysis.frame.path)?},"analysis":analysis.as_dict(),"candidate_peaks":analysis.detection.candidate_peaks,"line_artifact_coordinates":analysis.detection.line_artifact_coordinates});
    save(&path, &payload)?;
    Ok(path)
}

pub fn load_analysis(directory: &Path, cache_key: &str, frame: FitsFrame) -> Option<FrameAnalysis> {
    let payload = load(&cache_path(directory, cache_key, false).ok()?)?;
    let identity =
        json!({"identity_version":1,"kind":"frame","file":file_identity(&frame.path).ok()?});
    if payload["native_cache_version"] != CACHE_VERSION
        || payload["cache_kind"] != "frame-analysis"
        || payload["cache_key"] != cache_key
        || payload["input_identity"] != identity
    {
        return None;
    }
    let analysis = &payload["analysis"];
    if [
        "detection",
        "matching",
        "faintest_detected",
        "photometric_calibration",
        "source_photometry",
    ]
    .iter()
    .any(|field| analysis.get(field).is_none())
    {
        return None;
    }
    let detection = &analysis["detection"];
    let sources: Vec<crate::measurement::Detection> =
        serde_json::from_value(detection["sources"].clone()).ok()?;
    let quality_count = usize::try_from(detection["quality_count"].as_u64()?).ok()?;
    let candidate_count = usize::try_from(detection["candidate_count"].as_u64()?).ok()?;
    if quality_count
        != sources
            .iter()
            .filter(|source| source.quality_passed)
            .count()
        || candidate_count < sources.len()
        || !detection["parameters"].is_object()
    {
        return None;
    }
    let image_shape = serde_json::from_value(detection["image_shape"].clone()).ok()?;
    if image_shape != [frame.height, frame.width] {
        return None;
    }
    Some(FrameAnalysis {
        frame,
        detection: DetectionResult {
            image_shape,
            background: detection["background"].as_f64()?,
            noise: detection["noise"].as_f64()?,
            threshold: detection["threshold"].as_f64()?,
            candidate_count,
            sources,
            quality_count,
            parameters: detection["parameters"].clone(),
            candidate_peaks: serde_json::from_value(payload["candidate_peaks"].clone()).ok()?,
            line_artifact_coordinates: serde_json::from_value(
                payload["line_artifact_coordinates"].clone(),
            )
            .ok()?,
        },
        matching: serde_json::from_value(analysis["matching"].clone()).ok()?,
        faintest: serde_json::from_value(analysis["faintest_detected"].clone()).ok()?,
        photometric_calibration: serde_json::from_value(
            analysis["photometric_calibration"].clone(),
        )
        .ok()?,
        source_photometry: serde_json::from_value(analysis["source_photometry"].clone()).ok()?,
    })
}

fn frame_paths(sequence: &Value) -> Result<Vec<PathBuf>> {
    sequence["frames"]
        .as_array()
        .ok_or("sequence cache requires frame summaries")?
        .iter()
        .map(|frame| {
            frame["path"]
                .as_str()
                .map(PathBuf::from)
                .ok_or_else(|| "sequence cache frame is missing path".into())
        })
        .collect()
}

pub fn save_sequence_result(
    directory: &Path,
    cache_key: &str,
    sequence: &Value,
) -> Result<PathBuf> {
    if !sequence.is_object() || sequence.get("relative_photometry").is_none() {
        return Err("sequence cache requires complete photometry metadata".into());
    }
    let paths = frame_paths(sequence)?;
    let path = cache_path(directory, cache_key, true)?;
    save(
        &path,
        &json!({"native_cache_version":CACHE_VERSION,"cache_kind":"sequence","cache_key":cache_key,"input_identity":sequence_input_identity(&paths)?,"sequence":sequence}),
    )?;
    Ok(path)
}

pub fn load_sequence_result(directory: &Path, cache_key: &str, paths: &[PathBuf]) -> Option<Value> {
    let payload = load(&cache_path(directory, cache_key, true).ok()?)?;
    if payload["native_cache_version"] != CACHE_VERSION
        || payload["cache_kind"] != "sequence"
        || payload["cache_key"] != cache_key
        || payload["input_identity"] != sequence_input_identity(paths).ok()?
    {
        return None;
    }
    let sequence = &payload["sequence"];
    let stored = frame_paths(sequence).ok()?;
    if stored.len() != paths.len() || sequence.get("relative_photometry").is_none() {
        return None;
    }
    for (stored, expected) in stored.iter().zip(paths) {
        if fs::canonicalize(stored).ok()? != fs::canonicalize(expected).ok()? {
            return None;
        }
    }
    Some(sequence.clone())
}

/// Remove only native cache payloads; raw inputs and legacy caches are retained.
pub fn clear_cache(directory: &Path) -> Result<usize> {
    if !directory.is_dir() {
        return Ok(0);
    }
    let mut removed = 0;
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_file()
            && (name.ends_with(".native.json.gz") || name.ends_with(".native.sequence.json.gz"))
        {
            fs::remove_file(entry.path()).map_err(|error| error.to_string())?;
            removed += 1;
        }
    }
    Ok(removed)
}
