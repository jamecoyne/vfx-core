fn main() {
    vfx_build::configure();
    // segcam-face loads the DirectML.dll beside the plugin rather than the host's.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-cdylib=/DELAYLOAD:DirectML.dll");
    }
}
