use rst19_core::{pair_support_audit, Result};

const HELP: &str = "在局部裁剪和多种留出掩膜下检查一对近邻候选的独立像素支持\n\nUsage: rst19-pair-support-audit DATA_DIR --source-catalog PATH --primary-id ID --secondary-id ID [OPTIONS]\n";

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{HELP}");
        return Ok(());
    }
    let result = pair_support_audit::run_desktop(&args)?;
    println!(
        "{}",
        serde_json::to_string(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rst19-pair-support-audit: {error}");
        std::process::exit(2);
    }
}
