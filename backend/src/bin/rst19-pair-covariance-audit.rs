fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = rst19_core::pair_flux_covariance::run_cli(&args) {
        eprintln!("rst19-pair-covariance-audit: {error}");
        std::process::exit(2);
    }
}
