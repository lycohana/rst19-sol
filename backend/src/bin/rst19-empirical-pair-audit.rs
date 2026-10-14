use rst19_core::empirical_pair_audit;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", empirical_pair_audit::HELP);
        return ExitCode::SUCCESS;
    }
    match empirical_pair_audit::run_desktop(&args) {
        Ok(result) => {
            println!(
                "{}",
                result["conclusion"]
                    .as_str()
                    .unwrap_or("empirical pair audit complete")
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rst19-empirical-pair-audit: {error}");
            ExitCode::from(2)
        }
    }
}
