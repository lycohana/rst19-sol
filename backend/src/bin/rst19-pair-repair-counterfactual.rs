fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = rst19_core::pair_repair_counterfactual::run_cli(&args) {
        eprintln!("rst19-pair-repair-counterfactual: audit failed: {error}");
        std::process::exit(2);
    }
}
