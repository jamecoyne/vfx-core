//! Lists the effect crates this crate depends on, for the catalog (`src/catalog.rs`).

use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("run by Cargo"));
    println!("cargo:rerun-if-changed=Cargo.toml");
    let effects = vfx_build::effect_dependencies(&dir).unwrap_or_else(|e| panic!("{e}"));
    let entries: String = effects
        .iter()
        .map(|(name, _)| format!("        Entry::plugin(&{name}::VFX_PLUGIN),\n"))
        .collect();
    let code = format!("fn effect_crates() -> Vec<Entry> {{\n    vec![\n{entries}    ]\n}}\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("run by Cargo"));
    std::fs::write(out.join("effect_crates.rs"), code).expect("write effect_crates.rs");
}
