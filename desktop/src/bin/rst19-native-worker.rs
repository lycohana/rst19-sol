#[path = "../native_preview.rs"]
mod native_preview;

use serde_json::{json, Value};
use std::{
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
};

fn output_path(args: &mut Vec<String>, out_dir: &Path) -> PathBuf {
    if let Some(index) = args.iter().position(|arg| arg == "--json-out") {
        if let Some(path) = args.get(index + 1) {
            return PathBuf::from(path);
        }
    }
    if let Some(path) = args.iter().find_map(|arg| arg.strip_prefix("--json-out=")) {
        return PathBuf::from(path);
    }
    let path = out_dir.join("result.json");
    args.push("--json-out".into());
    args.push(path.to_string_lossy().into_owned());
    path
}

fn string_args(request: &Value) -> Result<Vec<String>, String> {
    request["args"]
        .as_array()
        .ok_or("args must be a string array")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "args must be a string array".to_owned())
        })
        .collect()
}

fn run(request: Value) -> Result<Value, String> {
    if request["operation"] == "preview" {
        let path = PathBuf::from(request["path"].as_str().ok_or("missing FITS path")?);
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        return native_preview::render_preview(
            &path,
            &out_dir,
            request["mode"].as_str().unwrap_or("raw"),
        );
    }
    if request["operation"] != "tool" {
        return Err("Rust desktop worker only accepts preview or native tools".into());
    }
    if request["tool"] == "rst19-feature-flag-profile" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_flag_profile::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-aux-audit" {
        return rst19_core::attitude::run_desktop(&string_args(&request)?);
    }
    if request["tool"] != "rst19" {
        return Err("Rust desktop worker does not support this tool".into());
    }
    let out_dir = PathBuf::from(
        request["out_dir"]
            .as_str()
            .ok_or("missing output directory")?,
    );
    fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
    fs::write(
        out_dir.join("request.json"),
        serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let mut args = string_args(&request)?;
    let input_path = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .map(PathBuf::from)
        .ok_or("missing FITS path")?;
    let result_path = output_path(&mut args, &out_dir);
    rst19_core::cli::run(&args)?;
    let result: Value =
        serde_json::from_slice(&fs::read(&result_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("Rust single-frame result is invalid JSON: {error}"))?;
    let mode = request["mode"].as_str().unwrap_or("raw");
    let preview = native_preview::render_preview(&input_path, &out_dir, mode)?;
    let mut payload = json!({
        "tool": "rst19",
        "out_dir": out_dir,
        "result_path": result_path,
        "result": result,
    });
    if let (Some(target), Some(source)) = (payload.as_object_mut(), preview.as_object()) {
        target.extend(
            source
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    Ok(payload)
}

fn main() {
    let request = io::stdin()
        .lock()
        .lines()
        .next()
        .ok_or_else(|| "missing JSON request".to_owned())
        .and_then(|line| line.map_err(|error| error.to_string()))
        .and_then(|line| serde_json::from_str(&line).map_err(|error| error.to_string()))
        .and_then(run);
    let (event, exit_code) = match request {
        Ok(payload) => (json!({"event":"done","ok":true,"payload":payload}), 0),
        Err(error) => (json!({"event":"done","ok":false,"error":error}), 1),
    };
    println!("{event}");
    std::process::exit(exit_code);
}

#[cfg(test)]
mod tests {
    use super::output_path;
    use std::path::Path;

    #[test]
    fn json_output_path_is_added_or_respected() {
        let mut args = vec!["frame.fits".into()];
        assert_eq!(
            output_path(&mut args, Path::new("job")),
            Path::new("job/result.json")
        );
        assert_eq!(args[1], "--json-out");

        let mut args = vec![
            "frame.fits".into(),
            "--json-out".into(),
            "custom.json".into(),
        ];
        assert_eq!(
            output_path(&mut args, Path::new("job")),
            Path::new("custom.json")
        );
    }
}
