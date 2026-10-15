fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|v| v == "--help" || v == "-h") {
        println!("{}", rst19_core::source_pair_audit::HELP);
        return;
    }
    match rst19_core::source_pair_audit::run_desktop(&args) {
        Ok(result) => println!("{result}"),
        Err(error) => {
            eprintln!("rst19-pair-audit: {error}");
            std::process::exit(2);
        }
    }
}
