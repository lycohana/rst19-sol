fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    std::process::exit(rst19_core::feature_audit_replicates::run_cli(&args));
}
