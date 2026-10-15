fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|v| v == "--help" || v == "-h") {
        println!("{}", rst19_core::motion_proposal_ablation::HELP);
        return;
    }
    match rst19_core::motion_proposal_ablation::run_desktop(&args) {
        Ok(result) => println!("{result}"),
        Err(error) => {
            eprintln!("rst19-motion-proposal-ablation: {error}");
            std::process::exit(2);
        }
    }
}
