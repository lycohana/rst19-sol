fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    std::process::exit(rst19_core::crowded_pair_ablation::run_cli(&args));
}
