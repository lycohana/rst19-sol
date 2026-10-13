fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = rst19_core::feature_evidence_matrix::run_cli(&args) {
        eprintln!("rst19-feature-evidence: {error}");
        std::process::exit(2);
    }
}
