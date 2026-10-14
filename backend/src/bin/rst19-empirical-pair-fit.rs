use rst19_core::empirical_pair_fit;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", empirical_pair_fit::HELP);
        return ExitCode::SUCCESS;
    }
    match empirical_pair_fit::run_desktop(&args) {
        Ok(result) => {
            println!(
                "{}",
                result["conclusion"]
                    .as_str()
                    .unwrap_or("empirical pair fit audit complete")
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rst19-empirical-pair-fit: {error}");
            ExitCode::from(2)
        }
    }
}
