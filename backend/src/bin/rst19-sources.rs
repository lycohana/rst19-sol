fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", rst19_core::source_audit::HELP);
        return;
    }
    let trail_catalog = match rst19_core::trail_catalog::take_flag(&args) {
        Ok((enabled, _)) => enabled,
        Err(error) => {
            eprintln!("rst19-sources: {error}");
            std::process::exit(2);
        }
    };
    match rst19_core::source_audit::run_desktop(&args) {
        Ok(summary) => {
            println!("source audit: {}", summary["output"].as_str().unwrap_or(""));
            println!(
                "candidate={} returned={} quality={} rejected={}",
                summary["candidate_count"].as_u64().unwrap_or_default(),
                summary["returned_count"].as_u64().unwrap_or_default(),
                summary["quality_count"].as_u64().unwrap_or_default(),
                summary["rejected_count"].as_u64().unwrap_or_default(),
            );
            let out_dir = summary["output"].as_str().unwrap_or_default();
            if trail_catalog {
                if let Some(line) =
                    rst19_core::trail_catalog::line_from_file(std::path::Path::new(out_dir))
                {
                    println!("{line}");
                }
            }
        }
        Err(error) => {
            eprintln!("rst19-sources: {error}");
            std::process::exit(2);
        }
    }
}
