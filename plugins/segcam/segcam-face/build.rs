fn main() {
    // The delay-load helper, for plugins that delay-load DirectML (src/directml.rs).
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-lib=delayimp");
        // Tests load the build's DirectML.dll before ONNX Runtime asks for it (tests/runtime.rs).
        println!("cargo:rustc-link-arg-tests=/DELAYLOAD:DirectML.dll");
    }
}
