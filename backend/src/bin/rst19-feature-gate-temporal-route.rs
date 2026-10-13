fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = rst19_core::feature_gate_temporal_route::run_cli(&args) {
        eprintln!("rst19-feature-gate-temporal-route: {error}");
        std::process::exit(2);
    }
}
