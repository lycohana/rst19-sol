use rst19_core::{photometric_report, Result};

const HELP: &str = "审计 rst19 星等 JSON：区分仪器、相对、表观和绝对星等证据\n\nUsage: rst19-photometric-report INPUT [--out PATH] [--require-wcs|--no-require-wcs] [--strict]\n";

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{HELP}");
        return Ok(());
    }
    let writes_file = args.iter().any(|arg| arg == "--out");
    let (report, raw_only) = photometric_report::run_file_args(&args)?;
    if raw_only {
        eprintln!("photometric-report: input has no source_photometry rows; raw detection.sources were intentionally not treated as calibrated photometry");
    }
    if !writes_file {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|error| error.to_string())?
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("photometric-report: {error}");
        std::process::exit(2);
    }
}
