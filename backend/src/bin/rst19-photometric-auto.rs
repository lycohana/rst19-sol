fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(rst19_core::photometric_workflow::run_cli(&args));
}
