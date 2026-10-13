fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = rst19_core::feature_spatial_context::run_cli(&args) {
        eprintln!("rst19-feature-spatial-context: {error}");
        std::process::exit(2);
    }
}
