use rst19_core::multipsf_audit;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", multipsf_audit::HELP);
        return ExitCode::SUCCESS;
    }
    match multipsf_audit::run_desktop(&args) {
        Ok(result) => {
            println!(
                "{}",
                result["conclusion"]
                    .as_str()
                    .unwrap_or("multi-PSF audit complete")
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rst19-multipsf-audit: {error}");
            ExitCode::from(2)
        }
    }
}
