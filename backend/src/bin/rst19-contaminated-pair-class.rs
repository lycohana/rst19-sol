fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    std::process::exit(rst19_core::contaminated_pair_class::run_cli(&args));
}
