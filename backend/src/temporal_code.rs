//! Multi-frame fixed-code and low-variation pixel audit.
use crate::{fits, models::PixelData, Result};
use csv::{Terminator, WriterBuilder};
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const BOUNDARY: &str = "低变化/相邻特殊值是固定结构候选，不是已由 FITS 格式说明确认的坏点或物理编码；需主办方格式、暗场/坏点资料和注入验证。";

#[derive(Clone, Debug)]
struct Options {
    low_variation_span_adu: i32,
    code_min_adu: i32,
    code_max_adu: i32,
    sentinel_value: i32,
    code_focus_xy: (i32, i32),
    sentinel_focus_xy: (i32, i32),
    out_dir: PathBuf,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            low_variation_span_adu: 2,
            code_min_adu: 3990,
            code_max_adu: 3993,
            sentinel_value: -1,
            code_focus_xy: (1274, 3466),
            sentinel_focus_xy: (1271, 3465),
            out_dir: PathBuf::from("tmp/temporal-code-audit"),
        }
    }
}

#[derive(Debug)]
struct ComponentSummary {
    count: usize,
    at_least_two: usize,
    largest: usize,
    focus_size: Option<usize>,
    focus_coordinates: Vec<[usize; 2]>,
    rows: Vec<Value>,
}

fn integer_pixels(data: &PixelData, path: &Path) -> Result<Vec<i32>> {
    let values = match data {
        PixelData::U8(values) => values.iter().map(|&v| v as i32).collect(),
        PixelData::I16(values) => values.iter().map(|&v| v as i32).collect(),
        PixelData::I32(values) => values.clone(),
        PixelData::I64(values) => values.iter().map(|&v| v as i32).collect(),
        PixelData::F32(_) | PixelData::F64(_) => {
            return Err(format!(
                "temporal code audit expects integer FITS pixels: {}",
                path.display()
            ))
        }
    };
    Ok(values)
}

fn component_summary(
    mask: &[bool],
    width: usize,
    height: usize,
    kind: &str,
    focus: Option<(i32, i32)>,
) -> ComponentSummary {
    let mut visited = vec![false; mask.len()];
    let mut rows = Vec::new();
    let mut focus_size = None;
    let mut focus_coordinates = Vec::new();
    let mut largest = 0;
    let mut at_least_two = 0;
    let mut focus_label = None;
    if let Some((x, y)) = focus {
        if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
            let index = y as usize * width + x as usize;
            if mask[index] {
                // Component IDs are assigned in the same row-major order as scipy.ndimage.label.
                focus_label = Some(index);
            }
        }
    }

    let mut component_id = 0;
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        component_id += 1;
        visited[start] = true;
        let mut queue = vec![start];
        let mut head = 0;
        let mut x0 = start % width;
        let mut x1 = x0;
        let mut y0 = start / width;
        let mut y1 = y0;
        let mut contains_focus = false;
        while head < queue.len() {
            let index = queue[head];
            head += 1;
            let x = index % width;
            let y = index / width;
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            contains_focus |= focus_label == Some(index);
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i64 + dx as i64;
                    let ny = y as i64 + dy as i64;
                    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                        continue;
                    }
                    let next = ny as usize * width + nx as usize;
                    if mask[next] && !visited[next] {
                        visited[next] = true;
                        queue.push(next);
                    }
                }
            }
        }
        let size = queue.len();
        largest = largest.max(size);
        at_least_two += usize::from(size >= 2);
        if contains_focus {
            focus_size = Some(size);
            focus_coordinates = queue
                .iter()
                .map(|&index| [index % width, index / width])
                .collect();
        }
        rows.push(json!({
            "kind":kind,
            "component_id":component_id,
            "size":size,
            "x0":x0,
            "y0":y0,
            "x1":x1,
            "y1":y1,
            "is_focus_component":contains_focus
        }));
    }
    ComponentSummary {
        count: component_id,
        at_least_two,
        largest,
        focus_size,
        focus_coordinates,
        rows,
    }
}

fn focus_value(values: &[i32], width: usize, height: usize, point: (i32, i32)) -> Option<i32> {
    let (x, y) = point;
    if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
        None
    } else {
        Some(values[y as usize * width + x as usize])
    }
}

fn point_value(point: Option<(i32, i32)>) -> Value {
    point.map_or(Value::Null, |(x, y)| json!([x, y]))
}

fn analyze(paths: &[PathBuf], options: &Options) -> Result<Value> {
    if paths.is_empty() {
        return Err("paths cannot be empty".into());
    }
    if options.low_variation_span_adu < 0 {
        return Err("low_variation_span_adu cannot be negative".into());
    }
    if options.code_min_adu > options.code_max_adu {
        return Err("code_min_adu cannot exceed code_max_adu".into());
    }

    let mut first: Option<Vec<i32>> = None;
    let mut minimum: Vec<i32> = Vec::new();
    let mut maximum: Vec<i32> = Vec::new();
    let mut width = 0;
    let mut height = 0;
    let mut code_focus_values = Vec::new();
    let mut sentinel_focus_values = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        let frame = fits::read_fits(path, false)?;
        if frame.width.checked_mul(frame.height) != Some(frame.data.len()) {
            return Err(format!("invalid image shape: {}", path.display()));
        }
        let pixels = integer_pixels(&frame.data, path)?;
        if let Some(first_values) = &first {
            if frame.width != width || frame.height != height {
                return Err("all FITS images must have the same shape".into());
            }
            for (position, value) in pixels.iter().copied().enumerate() {
                minimum[position] = minimum[position].min(value);
                maximum[position] = maximum[position].max(value);
            }
            let _ = first_values;
        } else {
            width = frame.width;
            height = frame.height;
            minimum.clone_from(&pixels);
            maximum.clone_from(&pixels);
            first = Some(pixels.clone());
        }
        code_focus_values.push(focus_value(
            &pixels,
            frame.width,
            frame.height,
            options.code_focus_xy,
        ));
        sentinel_focus_values.push(focus_value(
            &pixels,
            frame.width,
            frame.height,
            options.sentinel_focus_xy,
        ));
        eprintln!("temporal code audit: frame {}/{}", index + 1, paths.len());
    }
    let first = first.ok_or("temporal code audit did not read any image")?;
    let span = minimum
        .iter()
        .zip(&maximum)
        .map(|(&low, &high)| high.wrapping_sub(low))
        .collect::<Vec<_>>();
    let exact_stable = span.iter().map(|&value| value == 0).collect::<Vec<_>>();
    let low_variation = span
        .iter()
        .map(|&value| value <= options.low_variation_span_adu)
        .collect::<Vec<_>>();
    let code_mask = first
        .iter()
        .zip(&low_variation)
        .map(|(&value, &stable)| {
            stable && value >= options.code_min_adu && value <= options.code_max_adu
        })
        .collect::<Vec<_>>();
    let sentinel_mask = first
        .iter()
        .zip(&exact_stable)
        .map(|(&value, &stable)| stable && value == options.sentinel_value)
        .collect::<Vec<_>>();
    let code_components = component_summary(
        &code_mask,
        width,
        height,
        "low_variation_code",
        Some(options.code_focus_xy),
    );
    let sentinel_components = component_summary(
        &sentinel_mask,
        width,
        height,
        "exact_stable_sentinel",
        Some(options.sentinel_focus_xy),
    );
    let mut value_counts = BTreeMap::<i32, usize>::new();
    for (&value, &stable) in first.iter().zip(&exact_stable) {
        if stable {
            *value_counts.entry(value).or_default() += 1;
        }
    }
    let mut stable_counts = value_counts.into_iter().collect::<Vec<_>>();
    stable_counts.sort_by(|(value_a, count_a), (value_b, count_b)| {
        count_b.cmp(count_a).then_with(|| value_a.cmp(value_b))
    });
    let exact_stable_value_counts = stable_counts
        .iter()
        .map(|(value, count)| (value.to_string(), json!(count)))
        .collect::<Map<_, _>>();

    let focus_values = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let code = code_focus_values[index];
            let sentinel = sentinel_focus_values[index];
            json!({
                "frame_index":index+1,
                "path":path.to_string_lossy(),
                "code_value":code,
                "sentinel_value":sentinel,
                "code_in_range":code.is_some_and(|value| value >= options.code_min_adu && value <= options.code_max_adu),
                "sentinel_exact":sentinel == Some(options.sentinel_value)
            })
        })
        .collect::<Vec<_>>();
    let mut components = code_components.rows.clone();
    components.extend(sentinel_components.rows.clone());
    Ok(json!({
        "frame_count":paths.len(),
        "image_shape":[height,width],
        "paths":paths.iter().map(|path|path.to_string_lossy()).collect::<Vec<_>>(),
        "focus_values_by_frame":focus_values,
        "exact_stable_pixel_count":exact_stable.iter().filter(|&&value|value).count(),
        "exact_stable_value_counts":exact_stable_value_counts,
        "low_variation_span_adu":options.low_variation_span_adu,
        "low_variation_pixel_count":low_variation.iter().filter(|&&value|value).count(),
        "focus_code_range_adu":[options.code_min_adu,options.code_max_adu],
        "focus_code_pixel_count":code_mask.iter().filter(|&&value|value).count(),
        "focus_code_component_count":code_components.count,
        "focus_code_components_ge_2":code_components.at_least_two,
        "focus_code_largest_component":code_components.largest,
        "focus_code_xy":point_value(Some(options.code_focus_xy)),
        "focus_code_component_size":code_components.focus_size,
        "focus_code_component_coordinates":code_components.focus_coordinates,
        "sentinel":{
            "value":options.sentinel_value,
            "exact_stable_pixel_count":sentinel_mask.iter().filter(|&&value|value).count(),
            "component_count":sentinel_components.count,
            "largest_component":sentinel_components.largest,
            "focus_xy":point_value(Some(options.sentinel_focus_xy)),
            "focus_component_size":sentinel_components.focus_size,
            "focus_component_coordinates":sentinel_components.focus_coordinates
        },
        "components":components,
        "interpretation_boundary":BOUNDARY
    }))
}

fn parse_args(args: &[String]) -> Result<(PathBuf, Options)> {
    let mut input = None;
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let mut value = |flag: &str| -> Result<String> {
            if let Some((_, inline)) = arg.split_once('=') {
                Ok(inline.to_owned())
            } else {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| format!("{flag} requires a value"))
            }
        };
        macro_rules! number {
            ($flag:literal, $field:expr, $type:ty, $message:literal) => {
                if arg == $flag || arg.starts_with(concat!($flag, "=")) {
                    $field = value($flag)?.parse::<$type>().map_err(|_| $message)?;
                    index += 1;
                    continue;
                }
            };
        }
        number!(
            "--low-variation-span-adu",
            options.low_variation_span_adu,
            i32,
            "invalid --low-variation-span-adu"
        );
        number!(
            "--code-min-adu",
            options.code_min_adu,
            i32,
            "invalid --code-min-adu"
        );
        number!(
            "--code-max-adu",
            options.code_max_adu,
            i32,
            "invalid --code-max-adu"
        );
        number!(
            "--sentinel-value",
            options.sentinel_value,
            i32,
            "invalid --sentinel-value"
        );
        number!(
            "--focus-x",
            options.code_focus_xy.0,
            i32,
            "invalid --focus-x"
        );
        number!(
            "--focus-y",
            options.code_focus_xy.1,
            i32,
            "invalid --focus-y"
        );
        number!(
            "--sentinel-focus-x",
            options.sentinel_focus_xy.0,
            i32,
            "invalid --sentinel-focus-x"
        );
        number!(
            "--sentinel-focus-y",
            options.sentinel_focus_xy.1,
            i32,
            "invalid --sentinel-focus-y"
        );
        if arg == "--out-dir" || arg.starts_with("--out-dir=") {
            options.out_dir = PathBuf::from(value("--out-dir")?);
            index += 1;
            continue;
        }
        if arg == "--help" || arg == "-h" {
            return Err("help requested".into());
        }
        if arg.starts_with('-') {
            return Err(format!("unknown argument {arg}"));
        }
        if input.replace(PathBuf::from(arg)).is_some() {
            return Err(format!("unexpected positional argument {arg}"));
        }
        index += 1;
    }
    Ok((input.ok_or("input directory is required")?, options))
}

fn input_paths(input: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = if input.is_dir() {
        fs::read_dir(input)
            .map_err(|error| error.to_string())?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.is_file()
                    && path.extension().is_some_and(|extension| {
                        extension.to_string_lossy().eq_ignore_ascii_case("fits")
                    })
            })
            .collect::<Vec<_>>()
    } else {
        return Err(format!(
            "input directory does not exist: {}",
            input.display()
        ));
    };
    paths.sort();
    if paths.is_empty() {
        return Err(format!("数据目录没有 FITS 文件：{}", input.display()));
    }
    Ok(paths)
}

fn write_csv(path: &Path, headers: &[&str], rows: &[Vec<String>], bom: bool) -> Result<()> {
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    if bom {
        file.write_all(&[0xEF, 0xBB, 0xBF])
            .map_err(|error| error.to_string())?;
    }
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
    Ok(())
}

fn csv_value(value: &Option<i32>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn point_csv(point: (i32, i32)) -> String {
    format!("{},{}", point.0, point.1)
}

fn write_artifacts(result: &Value, out_dir: &Path, options: &Options) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = fs::canonicalize(out_dir).map_err(|error| error.to_string())?;
    let json_path = output.join("temporal_code_audit.json");
    let mut json_file = File::create(json_path).map_err(|error| error.to_string())?;
    json_file
        .write_all(
            serde_json::to_string_pretty(result)
                .map_err(|error| error.to_string())?
                .as_bytes(),
        )
        .map_err(|error| error.to_string())?;
    json_file
        .write_all(b"\n")
        .map_err(|error| error.to_string())?;

    let paths = result["paths"].as_array().ok_or("paths missing")?;
    let focus_rows = paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let values = &result["focus_values_by_frame"][index];
            vec![
                (index + 1).to_string(),
                path.as_str().unwrap_or_default().to_owned(),
                point_csv(options.code_focus_xy),
                csv_value(&values["code_value"].as_i64().map(|value| value as i32)),
                values["code_in_range"]
                    .as_bool()
                    .map_or_else(String::new, |value| {
                        if value { "True" } else { "False" }.to_owned()
                    }),
                point_csv(options.sentinel_focus_xy),
                csv_value(&values["sentinel_value"].as_i64().map(|value| value as i32)),
                values["sentinel_exact"]
                    .as_bool()
                    .map_or_else(String::new, |value| {
                        if value { "True" } else { "False" }.to_owned()
                    }),
            ]
        })
        .collect::<Vec<_>>();
    let focus_path = output.join("temporal_code_focus_series.csv");
    write_csv(
        &focus_path,
        &[
            "frame_index",
            "path",
            "code_focus_xy",
            "code_value",
            "code_in_range",
            "sentinel_focus_xy",
            "sentinel_value",
            "sentinel_exact",
        ],
        &focus_rows,
        true,
    )?;

    let component_rows = result["components"]
        .as_array()
        .ok_or("components missing")?
        .iter()
        .map(|row| {
            vec![
                row["kind"].as_str().unwrap_or_default().to_owned(),
                row["component_id"].to_string(),
                row["size"].to_string(),
                row["x0"].to_string(),
                row["y0"].to_string(),
                row["x1"].to_string(),
                row["y1"].to_string(),
                row["is_focus_component"]
                    .as_bool()
                    .map_or_else(String::new, |value| {
                        if value { "True" } else { "False" }.to_owned()
                    }),
            ]
        })
        .collect::<Vec<_>>();
    let component_path = output.join("temporal_code_components.csv");
    write_csv(
        &component_path,
        &[
            "kind",
            "component_id",
            "size",
            "x0",
            "y0",
            "x1",
            "y1",
            "is_focus_component",
        ],
        &component_rows,
        true,
    )?;
    Ok(output)
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (input, options) = parse_args(args)?;
    let paths = input_paths(&input)?;
    let result = analyze(&paths, &options)?;
    let output = write_artifacts(&result, &options.out_dir, &options)?;
    Ok(json!({
        "tool":"rst19-temporal-codes",
        "result":result,
        "out_dir":output.to_string_lossy()
    }))
}

pub fn run_cli(args: &[String]) -> Result<i32> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("15 帧 detector 固定值/低变化像素审计\n\nUsage: rst19-temporal-codes DATA_DIR [--low-variation-span-adu N] [--code-min-adu N] [--code-max-adu N] [--sentinel-value N] [--focus-x X] [--focus-y Y] [--sentinel-focus-x X] [--sentinel-focus-y Y] [--out-dir DIR]");
        return Ok(0);
    }
    let response = run_desktop(args)?;
    let result = &response["result"];
    println!(
        "rst19-temporal-codes: 审计已写入 {}",
        response["out_dir"].as_str().unwrap_or_default()
    );
    println!(
        "frames={} exact_stable={} low_variation={} code_pixels={} code_components={} sentinel_pixels={}",
        result["frame_count"],
        result["exact_stable_pixel_count"],
        result["low_variation_pixel_count"],
        result["focus_code_pixel_count"],
        result["focus_code_component_count"],
        result["sentinel"]["exact_stable_pixel_count"],
    );
    Ok(0)
}

pub fn handle_request(request: &Value) -> Result<Value> {
    let input = request
        .get("data_dir")
        .and_then(Value::as_str)
        .ok_or("missing data_dir")?;
    let mut args = vec![input.to_owned()];
    let options = [
        ("low_variation_span_adu", "--low-variation-span-adu"),
        ("code_min_adu", "--code-min-adu"),
        ("code_max_adu", "--code-max-adu"),
        ("sentinel_value", "--sentinel-value"),
        ("focus_x", "--focus-x"),
        ("focus_y", "--focus-y"),
        ("sentinel_focus_x", "--sentinel-focus-x"),
        ("sentinel_focus_y", "--sentinel-focus-y"),
    ];
    for (key, flag) in options {
        if let Some(value) = request.get(key).filter(|value| !value.is_null()) {
            args.push(flag.to_owned());
            args.push(value.to_string().trim_matches('"').to_owned());
        }
    }
    if let Some(out_dir) = request.get("output_dir").and_then(Value::as_str) {
        args.push("--out-dir".into());
        args.push(out_dir.to_owned());
    }
    run_desktop(&args)
}
