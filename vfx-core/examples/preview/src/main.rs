//! The repository's preview: every effect built here, one at a time.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    vfx_preview::run(vfx_preview::catalog::repository())
}

/// The web entry point, called by wasm-bindgen's generated JavaScript.
#[cfg(target_arch = "wasm32")]
fn main() {
    vfx_preview::run(vfx_preview::catalog::repository());
}
