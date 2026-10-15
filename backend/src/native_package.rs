//! Build and verify native distributions without an interpreter or a build venv.
use crate::{cache, fits, Result};
use flate2::{write::DeflateEncoder, Compression};
use serde_json::{json, Map, Value};
use std::{
    fs,
    io::{Seek, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub const HELP:&str="Build: cargo run --locked --manifest-path backend/Cargo.toml --bin rst19-package -- build [--skip-build] [--compact]\nVerify: rst19-package verify PACKAGE_DIR [--out DIR]";
const FONT_URL:&str="https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf";
const LICENSE_URL: &str = "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/LICENSE";
fn executable(name: &str) -> String {
    format!("{name}{}", if cfg!(windows) { ".exe" } else { "" })
}
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("backend parent")
        .to_owned()
}
fn run_cargo(root: &Path, args: &[&str]) -> Result<()> {
    let status = Command::new("cargo")
        .args(args)
        .current_dir(root)
        .status()
        .map_err(|e| format!("cargo could not start: {e}"))?;
    if !status.success() {
        return Err(format!("cargo failed with {status}"));
    }
    Ok(())
}
fn download(path: &Path, url: &str) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    let bytes = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|e| e.to_string())?
        .get(url)
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .bytes()
        .map_err(|e| e.to_string())?;
    if bytes.is_empty() {
        return Err("font download was empty".into());
    }
    fs::write(path, bytes).map_err(|e| e.to_string())
}
fn relative(path: &Path, base: &Path) -> Result<String> {
    Ok(path
        .strip_prefix(base)
        .map_err(|e| e.to_string())?
        .to_string_lossy()
        .replace('\\', "/"))
}
fn files(dir: &Path, result: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let ty = entry.file_type().map_err(|e| e.to_string())?;
        if ty.is_symlink() {
            return Err("package cannot contain symbolic links".into());
        }
        if ty.is_dir() {
            files(&entry.path(), result)?;
        } else if ty.is_file() {
            result.push(entry.path());
        }
    }
    Ok(())
}
fn hashes(package: &Path) -> Result<Value> {
    let mut paths = Vec::new();
    files(package, &mut paths)?;
    paths.sort();
    let mut hashes = Map::new();
    for path in paths {
        if path.file_name().is_some_and(|n| n == "SHA256.json") {
            continue;
        }
        hashes.insert(
            relative(&path, package)?,
            cache::file_identity(&path)?["sha256"].clone(),
        );
    }
    Ok(Value::Object(hashes))
}
fn u16le(out: &mut Vec<u8>, n: u16) {
    out.extend(n.to_le_bytes());
}
fn u32le(out: &mut Vec<u8>, n: u32) {
    out.extend(n.to_le_bytes());
}
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0xedb88320 } else { 0 };
        }
    }
    !crc
}
/// ZIP32 with each native executable's Unix permission bits retained.
pub fn archive(package: &Path, path: &Path) -> Result<()> {
    let parent = package.parent().ok_or("package has no parent")?;
    let mut paths = Vec::new();
    files(package, &mut paths)?;
    paths.sort();
    if paths.len() > u16::MAX as usize {
        return Err("package has too many files".into());
    }
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let mut central = Vec::new();
    for source in &paths {
        let name = relative(source, parent)?;
        let bytes = fs::read(source).map_err(|e| e.to_string())?;
        let offset = u32::try_from(file.stream_position().map_err(|e| e.to_string())?)
            .map_err(|_| "package exceeds ZIP32 limits")?;
        let mut compressor = DeflateEncoder::new(Vec::new(), Compression::default());
        compressor.write_all(&bytes).map_err(|e| e.to_string())?;
        let compressed = compressor.finish().map_err(|e| e.to_string())?;
        let uncompressed = u32::try_from(bytes.len()).map_err(|_| "file exceeds ZIP32 limits")?;
        let compressed_len =
            u32::try_from(compressed.len()).map_err(|_| "compressed file exceeds ZIP32 limits")?;
        let crc = crc32(&bytes);
        let mut header = Vec::new();
        u32le(&mut header, 0x04034b50);
        u16le(&mut header, 20);
        u16le(&mut header, 0x800);
        u16le(&mut header, 8);
        u16le(&mut header, 0);
        u16le(&mut header, 33);
        u32le(&mut header, crc);
        u32le(&mut header, compressed_len);
        u32le(&mut header, uncompressed);
        u16le(&mut header, name.len() as u16);
        u16le(&mut header, 0);
        header.extend(name.bytes());
        file.write_all(&header).map_err(|e| e.to_string())?;
        file.write_all(&compressed).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        let mode = {
            use std::os::unix::fs::PermissionsExt;
            source
                .metadata()
                .map_err(|e| e.to_string())?
                .permissions()
                .mode()
        };
        #[cfg(not(unix))]
        let mode = if source.extension().is_some_and(|e| e == "exe") {
            0o100755
        } else {
            0o100644
        };
        u32le(&mut central, 0x02014b50);
        u16le(&mut central, (3 << 8) | 20);
        u16le(&mut central, 20);
        u16le(&mut central, 0x800);
        u16le(&mut central, 8);
        u16le(&mut central, 0);
        u16le(&mut central, 33);
        u32le(&mut central, crc);
        u32le(&mut central, compressed_len);
        u32le(&mut central, uncompressed);
        u16le(&mut central, name.len() as u16);
        for _ in 0..4 {
            u16le(&mut central, 0);
        }
        u32le(&mut central, mode << 16);
        u32le(&mut central, offset);
        central.extend(name.bytes());
    }
    let offset = u32::try_from(file.stream_position().map_err(|e| e.to_string())?)
        .map_err(|_| "package exceeds ZIP32 limits")?;
    let size = central.len() as u32;
    file.write_all(&central).map_err(|e| e.to_string())?;
    let mut end = Vec::new();
    u32le(&mut end, 0x06054b50);
    u16le(&mut end, 0);
    u16le(&mut end, 0);
    u16le(&mut end, paths.len() as u16);
    u16le(&mut end, paths.len() as u16);
    u32le(&mut end, size);
    u32le(&mut end, offset);
    u16le(&mut end, 0);
    file.write_all(&end).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}
fn worker(package: &Path, request: Option<&Value>, cwd: &Path) -> Result<Value> {
    let path = package
        .join("runtime")
        .join(executable("rst19-native-worker"));
    let mut command = Command::new(path);
    command
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if request.is_none() {
        command.arg("--catalog");
    }
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    if let Some(request) = request {
        writeln!(
            child.stdin.take().ok_or("missing worker input")?,
            "{request}"
        )
        .map_err(|e| e.to_string())?;
    } else {
        drop(child.stdin.take());
    }
    let result = child.wait_with_output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&result.stdout);
    let event = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|v| v["event"] == "done")
        .ok_or_else(|| {
            format!(
                "worker returned no terminal result: {}",
                String::from_utf8_lossy(&result.stderr)
            )
        })?;
    if !result.status.success() || event["ok"] != true {
        return Err(format!("native package worker failed: {}", event["error"]));
    }
    Ok(event["payload"].clone())
}
fn fits_card(key: &str, value: &str) -> [u8; 80] {
    let mut card = [b' '; 80];
    let text = format!("{key:<8}= {value:>20}");
    card[..text.len()].copy_from_slice(text.as_bytes());
    card
}
fn smoke_fits(path: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    for (key, value) in [
        ("SIMPLE", "T"),
        ("BITPIX", "16"),
        ("NAXIS", "2"),
        ("NAXIS1", "96"),
        ("NAXIS2", "96"),
        ("DATE-OBS", "'2026-10-15T00:00:00'"),
        ("EXPTIME", "1"),
    ] {
        bytes.extend(fits_card(key, value));
    }
    let mut end = [b' '; 80];
    end[..3].copy_from_slice(b"END");
    bytes.extend(end);
    bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
    for y in 0..96 {
        for x in 0..96 {
            let value = 100.
                + (((x * 31 + y * 17) % 7) as f64 - 3.)
                + 250.
                    * (-0.5 * ((x as f64 - 35.).powi(2) + (y as f64 - 40.).powi(2))
                        / 1.2f64.powi(2))
                    .exp();
            bytes.extend((value.round() as i16).to_be_bytes());
        }
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    fs::write(path, bytes).map_err(|e| e.to_string())
}
pub fn verify(package: &Path, out: &Path) -> Result<Value> {
    let package = package.canonicalize().map_err(|e| e.to_string())?;
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let out = out.canonicalize().map_err(|e| e.to_string())?;
    if out.starts_with(&package) {
        return Err("verification outputs must be outside the package".into());
    }
    let expected: Value =
        serde_json::from_slice(&fs::read(package.join("SHA256.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let actual = hashes(&package)?;
    if actual != expected {
        return Err("package SHA-256 manifest does not match its files".into());
    }
    for name in actual.as_object().ok_or("invalid hash manifest")?.keys() {
        let lowered = name.to_ascii_lowercase();
        if lowered.ends_with(".py")
            || lowered.ends_with(".pyc")
            || lowered.contains("python")
            || (lowered.contains("numpy") && !lowered.ends_with(".txt"))
            || (lowered.contains("scipy") && !lowered.ends_with(".txt"))
            || lowered.contains("site-packages")
        {
            return Err(format!("non-native runtime artifact in package: {name}"));
        }
    }
    let catalog = worker(&package, None, &out)?;
    let tools = catalog["tools"]
        .as_array()
        .ok_or("worker catalog missing tools")?;
    if tools.len() != 71 || tools.iter().any(|t| t["native"] != true) {
        return Err("package must expose all 71 native tools".into());
    }
    let input = out.join("frames");
    fs::create_dir_all(&input).map_err(|e| e.to_string())?;
    let first = input.join("frame-00.fits");
    smoke_fits(&first)?;
    fs::copy(&first, input.join("frame-01.fits")).map_err(|e| e.to_string())?;
    fits::read_fits(&first, false)?;
    let before = cache::file_identity(&first)?["sha256"].clone();
    let make = |tool: &str, args: Vec<String>, folder: &str| json!({"operation":"tool","tool":tool,"args":args,"out_dir":out.join(folder)});
    let single = worker(
        &package,
        Some(&make(
            "rst19",
            vec![first.to_string_lossy().into_owned()],
            "single",
        )),
        &out,
    )?;
    let sequence = worker(
        &package,
        Some(&make(
            "rst19-sequence",
            vec![
                input.to_string_lossy().into_owned(),
                "--no-stack-faint".into(),
            ],
            "sequence",
        )),
        &out,
    )?;
    let result_path = sequence["result_path"]
        .as_str()
        .ok_or("sequence output path missing")?;
    let innovation = worker(
        &package,
        Some(&make(
            "rst19-innovation",
            vec![result_path.into()],
            "innovation",
        )),
        &out,
    )?;
    let injection = worker(
        &package,
        Some(&make(
            "rst19-injection",
            vec![
                "--trials".into(),
                "1".into(),
                "--sources".into(),
                "2".into(),
                "--peak-levels".into(),
                "128".into(),
            ],
            "injection",
        )),
        &out,
    )?;
    let pair = worker(
        &package,
        Some(&make(
            "rst19-pair-audit",
            vec![
                input.to_string_lossy().into_owned(),
                "--primary-x".into(),
                "35".into(),
                "--primary-y".into(),
                "40".into(),
                "--secondary-x".into(),
                "40".into(),
                "--secondary-y".into(),
                "40".into(),
            ],
            "pair",
        )),
        &out,
    )?;
    let point = worker(
        &package,
        Some(&make(
            "rst19-ablation",
            vec![
                first.to_string_lossy().into_owned(),
                "--targets".into(),
                "2".into(),
                "--peak-levels".into(),
                "128".into(),
                "--report".into(),
                out.join("ablation.html").to_string_lossy().into_owned(),
            ],
            "point-ablation",
        )),
        &out,
    )?;
    let proposal = worker(
        &package,
        Some(&make(
            "rst19-motion-proposal-ablation",
            vec![
                "--point-run-dir".into(),
                out.join("point-ablation").to_string_lossy().into_owned(),
                "--report".into(),
                out.join("ablation.html").to_string_lossy().into_owned(),
            ],
            "motion-proposal",
        )),
        &out,
    )?;
    if before != cache::file_identity(&first)?["sha256"] {
        return Err("package modified input FITS".into());
    }
    let report = json!({"status":"passed","implementation":"Rust","platform":std::env::consts::OS,"package":package,"checked_file_count":actual.as_object().map_or(0,|v|v.len()),"native_tool_count":tools.len(),"python_runtime_files":0,"checks":{"catalog":true,"single_frame":single,"sequence":sequence,"innovation":innovation,"injection":injection,"pair_audit":pair,"point_ablation":point,"motion_proposal_ablation":proposal,"original_fits_unchanged":true},"limitations":["Synthetic fixtures verify process, calculation and artifact integration; they do not establish real-sky physical identity.","Native window visual acceptance is checked separately on an available GUI host."]});
    fs::write(
        out.join("verification.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(report)
}
pub fn build(skip: bool, compact: bool) -> Result<Value> {
    let root = root();
    if !skip {
        run_cargo(
            &root,
            &[
                "build",
                "-j4",
                "--locked",
                "--release",
                "--manifest-path",
                "backend/Cargo.toml",
                "--bins",
            ],
        )?;
        run_cargo(
            &root,
            &[
                "build",
                "-j4",
                "--locked",
                "--release",
                "--manifest-path",
                "desktop/Cargo.toml",
                "--bins",
            ],
        )?;
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    let package = root.join("dist").join(format!(
        "RST19-{}-{}-{stamp}",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    fs::create_dir_all(package.join("runtime")).map_err(|e| e.to_string())?;
    fs::create_dir_all(package.join("assets")).map_err(|e| e.to_string())?;
    for (name, destination) in [
        ("rst19-desktop", package.join(executable("rst19-desktop"))),
        (
            "rst19-native-worker",
            package
                .join("runtime")
                .join(executable("rst19-native-worker")),
        ),
    ] {
        fs::copy(
            root.join("desktop/target/release").join(executable(name)),
            destination,
        )
        .map_err(|e| format!("missing native desktop executable {name}: {e}"))?;
    }
    let bin_dir = root.join("backend/target/release");
    if compact {
        fs::copy(
            bin_dir.join(executable("rst19-engine")),
            package.join("runtime").join(executable("rst19-engine")),
        )
        .map_err(|e| e.to_string())?;
    } else {
        let metadata = Command::new("cargo")
            .args([
                "metadata",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                "backend/Cargo.toml",
            ])
            .current_dir(&root)
            .output()
            .map_err(|e| e.to_string())?;
        if !metadata.status.success() {
            return Err("cargo metadata failed".into());
        }
        let metadata: Value =
            serde_json::from_slice(&metadata.stdout).map_err(|e| e.to_string())?;
        for target in metadata["packages"][0]["targets"]
            .as_array()
            .ok_or("native binary metadata missing")?
        {
            if target["kind"]
                .as_array()
                .is_some_and(|k| k.contains(&json!("bin")))
            {
                let name = target["name"].as_str().ok_or("binary name missing")?;
                if name == "rst19-package" {
                    continue;
                }
                fs::copy(
                    bin_dir.join(executable(name)),
                    package.join("runtime").join(executable(name)),
                )
                .map_err(|e| format!("missing native CLI {name}: {e}"))?;
            }
        }
    }
    let assets = root.join("desktop/.assets");
    fs::create_dir_all(&assets).map_err(|e| e.to_string())?;
    let font = assets.join("NotoSansSC-Regular-2.004.otf");
    let license = assets.join("NotoSansSC-LICENSE-2.004.txt");
    download(&font, FONT_URL)?;
    download(&license, LICENSE_URL)?;
    let font_bytes = fs::read(&font).map_err(|e| e.to_string())?;
    if !font_bytes.starts_with(b"OTTO") {
        return Err("expected the static OpenType CJK regular font".into());
    }
    fs::copy(&font, package.join("assets/NotoSansSC-Regular.otf")).map_err(|e| e.to_string())?;
    fs::copy(&license, package.join("assets/NotoSansSC-LICENSE.txt")).map_err(|e| e.to_string())?;
    for file in [
        "README.md",
        "desktop/README.md",
        "backend/third_party/NUMPY-1.26.4-BSD-3-CLAUSE.txt",
    ] {
        let source = root.join(file);
        if source.is_file() {
            let name = if file == "README.md" {
                "PROJECT-README.md"
            } else {
                source
                    .file_name()
                    .and_then(|v| v.to_str())
                    .ok_or("invalid documentation name")?
            };
            fs::copy(&source, package.join(name)).map_err(|e| e.to_string())?;
        }
    }
    let migration: Value = serde_json::from_slice(
        &fs::read(root.join("backend/migration.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(package.join("environment.json"),serde_json::to_vec_pretty(&json!({"runtime":"Rust","core_version":env!("CARGO_PKG_VERSION"),"os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"python_runtime_required":false,"native_tools":71,"compact":compact,"font":{"url":FONT_URL,"license":LICENSE_URL,"sha256":cache::file_identity(&font)?["sha256"]},"migration":migration})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    fs::write(
        package.join("SHA256.json"),
        serde_json::to_vec_pretty(&hashes(&package)?).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let verification = root
        .join("tmp")
        .join(format!("package-verification-{stamp}"));
    verify(&package, &verification)?;
    let archive_path = package.with_extension("zip");
    archive(&package, &archive_path)?;
    Ok(
        json!({"status":"passed","package":package,"archive":archive_path,"verification":verification.join("verification.json")}),
    )
}
pub fn run(args: &[String]) -> Result<Value> {
    match args.first().map(String::as_str) {
        Some("build") => {
            for arg in &args[1..] {
                if !["--skip-build", "--compact"].contains(&arg.as_str()) {
                    return Err(format!("unknown build option {arg}"));
                }
            }
            build(
                args.iter().any(|v| v == "--skip-build"),
                args.iter().any(|v| v == "--compact"),
            )
        }
        Some("verify") => {
            let package = args.get(1).ok_or("missing package directory")?;
            let mut out = root().join("tmp/package-verification");
            if args.len() > 2 {
                if args.get(2).map(String::as_str) != Some("--out") || args.len() != 4 {
                    return Err("verify accepts PACKAGE_DIR [--out DIR]".into());
                }
                out = PathBuf::from(&args[3]);
            }
            verify(Path::new(package), &out)
        }
        _ => Err(HELP.into()),
    }
}
