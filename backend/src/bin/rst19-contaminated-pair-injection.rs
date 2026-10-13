use std::process::ExitCode;

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match rst19_core::contaminated_pair_injection::run_cli(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rst19-contaminated-pair-injection: {error}");
            ExitCode::from(2)
        }
    }
}
