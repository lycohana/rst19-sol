fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", rst19_core::sequence_feature::HELP);
        return;
    }
    std::process::exit(rst19_core::sequence_feature::run_cli(&args));
}
