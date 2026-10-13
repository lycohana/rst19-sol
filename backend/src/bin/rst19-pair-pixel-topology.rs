fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match rst19_core::pair_pixel_topology::run_cli(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("rst19-pair-pixel-topology: {error}");
            std::process::exit(2);
        }
    }
}
