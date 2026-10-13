fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = rst19_core::feature_cross_audit::run_cli(&args) {
        eprintln!("rst19-feature-cross: {error}");
        std::process::exit(2);
    }
}
