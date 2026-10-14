use std::env;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match rst19_core::free_multipsf_audit::run_desktop(&args) {
        Ok(result) => {
            println!("rst19-free-multipsf-audit: 审计已写入 {}", result["output"]);
            println!("{}", result["conclusion"].as_str().unwrap_or_default());
        }
        Err(error) if error == rst19_core::free_multipsf_audit::HELP => {
            println!("{error}");
        }
        Err(error) => {
            eprintln!("rst19-free-multipsf-audit: {error}");
            std::process::exit(2);
        }
    }
}
