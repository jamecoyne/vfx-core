//! The build script every vfx-core effect crate runs:
//!
//! ```text
//! // build.rs
//! fn main() {
//!     vfx_build::configure();
//! }
//! ```
//!
//! It reads the crate's `[package.metadata.vfx]` and
//! - sets `VFX_NAME`, `VFX_ID` and the rest as compile-time variables, which the export
//!   macros (`vfx_plugin::export!`) read;
//! - links the macOS plugin binary as a Mach-O bundle, which hosts load;
//! - with the `ae` feature on, writes the After Effects PiPL resource
//!   (`$OUT_DIR/<file>.rsrc`, which `cargo vfx` puts in the bundle) and the `PIPL_*`
//!   variables the after-effects crate reads.

use std::path::{Path, PathBuf};

/// One effect crate's `[package.metadata.vfx]`.
#[derive(Clone, Debug)]
pub struct Metadata {
    pub name: String,
    pub short_name: String,
    pub category: String,
    pub file: String,
    pub id: String,
    pub ae_match_name: String,
    pub ffgl_code: String,
    pub windows_installer_id: Option<String>,
    /// DLLs from the build's output to ship beside the plugin on Windows.
    pub windows_dlls: Vec<String>,
    /// "gpu" (an `Effect`) or "cpu" (a `CpuEffect`).
    pub processor: String,
    pub version: String,
    pub description: String,
}

impl Metadata {
    /// Reads `<dir>/Cargo.toml`.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let path = dir.join("Cargo.toml");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let doc: toml::Value = text
            .parse()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let package = doc.get("package").ok_or("no [package]")?;
        let meta = package
            .get("metadata")
            .and_then(|m| m.get("vfx"))
            .ok_or_else(|| format!("{} has no [package.metadata.vfx]", path.display()))?;
        let field = |key: &str| {
            meta.get(key)
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        };
        let need =
            |key: &str| field(key).ok_or_else(|| format!("[package.metadata.vfx] needs {key}"));
        let package_field = |key: &str| {
            package
                .get(key)
                .and_then(toml::Value::as_str)
                .unwrap_or("")
                .to_owned()
        };
        let name = need("name")?;
        let m = Self {
            short_name: field("short-name").unwrap_or_else(|| name.clone()),
            name,
            category: need("category")?,
            file: need("file")?,
            id: need("id")?,
            ae_match_name: need("ae-match-name")?,
            ffgl_code: need("ffgl-code")?,
            windows_installer_id: field("windows-installer-id"),
            windows_dlls: meta
                .get("windows-dlls")
                .and_then(toml::Value::as_array)
                .map(|dlls| {
                    dlls.iter()
                        .filter_map(toml::Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            processor: field("processor").unwrap_or_else(|| "gpu".into()),
            version: package_field("version"),
            description: package_field("description"),
        };
        m.check()?;
        Ok(m)
    }

    fn check(&self) -> Result<(), String> {
        if self.ffgl_code.len() != 4 || !self.ffgl_code.is_ascii() {
            return Err(format!(
                "ffgl-code {:?} must be 4 ASCII characters",
                self.ffgl_code
            ));
        }
        if self.short_name.len() > 16 {
            return Err(format!(
                "{:?} is over FFGL's 16 bytes: set a short-name",
                self.short_name
            ));
        }
        if self.file.contains(char::is_whitespace) || self.file.is_empty() {
            return Err(format!("file {:?} must be one word", self.file));
        }
        if !matches!(self.processor.as_str(), "gpu" | "cpu") {
            return Err(format!(
                "processor {:?} must be \"gpu\" or \"cpu\"",
                self.processor
            ));
        }
        Ok(())
    }

    /// Major, minor and patch of the crate's version.
    pub fn version_parts(&self) -> [u32; 3] {
        let mut parts = self.version.split('.').map(|p| p.parse().unwrap_or(0));
        [(); 3].map(|_| parts.next().unwrap_or(0))
    }
}

/// Run from an effect crate's build script.
pub fn configure() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("run from a build script"));
    println!(
        "cargo:rerun-if-changed={}",
        dir.join("Cargo.toml").display()
    );
    let meta = Metadata::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    for (key, value) in [
        ("NAME", &meta.name),
        ("SHORT_NAME", &meta.short_name),
        ("CATEGORY", &meta.category),
        ("FILE", &meta.file),
        ("ID", &meta.id),
        ("AE_MATCH_NAME", &meta.ae_match_name),
        ("FFGL_CODE", &meta.ffgl_code),
        ("PROCESSOR", &meta.processor),
    ] {
        println!("cargo:rustc-env=VFX_{key}={value}");
    }
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // Hosts load Mach-O bundles (MH_BUNDLE), not dylibs, which is what cdylib links.
    if target_os == "macos" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-bundle");
    }
    if std::env::var_os("CARGO_FEATURE_AE").is_some() {
        after_effects(&meta, &target_os);
    }
}

/// The PiPL resource After Effects reads, and the variables the after-effects crate's
/// `define_effect!` compiles in. They must agree, so both come from here.
fn after_effects(meta: &Metadata, target_os: &str) {
    use pipl::{OutFlags, OutFlags2, PIPLType, Property, Stage};
    let leak = |s: &str| -> &'static str { Box::leak(s.to_owned().into_boxed_str()) };
    let [major, minor, patch] = meta.version_parts();
    let cpu = meta.processor == "cpu";
    // Sequence data is a Rust handle the crate restores from flattened bytes, so After
    // Effects must flatten it before saving. CPU effects may read earlier frames.
    let mut flags = OutFlags::DeepColorAware | OutFlags::SequenceDataNeedsFlattening;
    if cpu {
        flags |= OutFlags::WideTimeInput;
    }
    // 32 bpc float requires SmartFX. Each render leases its own instance, so Multi-Frame
    // Rendering is safe. GPU-native rendering exists for Metal, so macOS GPU effects only.
    let mut flags2 = OutFlags2::FloatColorAware
        | OutFlags2::SupportsSmartRender
        | OutFlags2::SupportsThreadedRendering
        | OutFlags2::SupportsGetFlattenedSequenceData;
    if !cpu && target_os == "macos" {
        flags2 |= OutFlags2::SupportsGpuRenderF32;
    }
    let properties = || {
        vec![
            Property::Kind(PIPLType::AEEffect),
            Property::Name(leak(&meta.name)),
            Property::Category(leak(&meta.category)),
            Property::CodeWin64X86("EffectMain"),
            Property::CodeMacIntel64("EffectMain"),
            Property::CodeMacARM64("EffectMain"),
            Property::AE_PiPL_Version { major: 2, minor: 0 },
            // After Effects disables Multi-Frame Rendering below 13.25.
            Property::AE_Effect_Spec_Version {
                major: 13,
                minor: 29,
            },
            Property::AE_Effect_Version {
                version: major,
                subversion: minor,
                bugversion: patch,
                stage: Stage::Develop,
                build: 0,
            },
            Property::AE_Effect_Info_Flags(0),
            Property::AE_Effect_Global_OutFlags(OutFlags::from_bits_retain(flags.bits())),
            Property::AE_Effect_Global_OutFlags_2(OutFlags2::from_bits_retain(flags2.bits())),
            Property::AE_Effect_Match_Name(leak(&meta.ae_match_name)),
        ]
    };
    pipl::plugin_build(properties());
    let bytes = pipl::build_pipl(properties()).expect("build the PiPL");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("run from a build script"))
        .join(format!("{}.rsrc", meta.file));
    pipl::produce_resource(&bytes, out.to_str());
    println!("cargo:rustc-env=PIPL_AE_RESERVED=0");
    println!("cargo:rustc-env=PIPL_SUPPORT_URL=");
}
