use crate::{innovation, innovation_artifacts, Result};
use serde_json::{json, Value};
use std::path::PathBuf;

pub const HELP: &str = "从序列 JSON 生成观测量、运动拟合、遥测、测光和原图证据\n\nUsage: rst19-innovation SEQUENCE_JSON [--out-dir PATH]";

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut input = None;
    let mut output = PathBuf::from("tmp/innovation");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if let Some(value) = arg.strip_prefix("--out-dir=") {
            output = PathBuf::from(value);
        } else if arg == "--out-dir" {
            index += 1;
            output = PathBuf::from(args.get(index).ok_or("--out-dir requires a path")?);
        } else if arg.starts_with('-') {
            return Err(format!("unknown innovation argument {arg}"));
        } else if input.replace(PathBuf::from(arg)).is_some() {
            return Err("only one sequence JSON is allowed".into());
        }
        index += 1;
    }
    let input = input.ok_or("missing sequence JSON")?;
    let report = innovation::build_innovation_report(&input)?;
    let output = innovation_artifacts::write_innovation_artifacts(&report, &output)?;
    Ok(
        json!({"output":output,"frames":report["relation"]["frame_count"],"duration_s":report["relation"]["duration_s"],"motion":report["motion"]["moving_count"],"candidates":report["motion"]["candidate_count"]}),
    )
}

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "innovation report: {}",
        result["output"].as_str().unwrap_or("")
    );
    println!(
        "frames={} duration_s={} motion={} candidates={}",
        result["frames"], result["duration_s"], result["motion"], result["candidates"]
    );
    Ok(())
}
