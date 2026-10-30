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

use std::collections::BTreeMap;
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
    /// The crate's own features `cargo vfx` turns on, by OS (`macos`, `windows`, `linux`) or
    /// for `all`, from `[package.metadata.vfx.features]`: an effect leaves out a heavy native
    /// library where it isn't used.
    pub features: BTreeMap<String, Vec<String>>,
}

/// The keys `[package.metadata.vfx.features]` takes.
pub const FEATURE_KEYS: [&str; 4] = ["all", "macos", "windows", "linux"];

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
        let mut features = BTreeMap::new();
        if let Some(table) = meta.get("features") {
            let table = table
                .as_table()
                .ok_or("[package.metadata.vfx] features must be a table")?;
            for (key, list) in table {
                let names = list
                    .as_array()
                    .and_then(|a| a.iter().map(|v| v.as_str().map(str::to_owned)).collect())
                    .ok_or_else(|| format!("features.{key} must be a list of feature names"))?;
                features.insert(key.clone(), names);
            }
        }
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
            features,
        };
        m.check()?;
        Ok(m)
    }

    /// What each host requires of the identity, so a mistake fails the build rather than
    /// showing up in a host.
    pub fn check(&self) -> Result<(), String> {
        let fail = |what: String| Err(format!("[package.metadata.vfx] {what}"));
        // After Effects keeps names, categories and match names to 31 bytes.
        for (key, value) in [
            ("name", &self.name),
            ("category", &self.category),
            ("ae-match-name", &self.ae_match_name),
        ] {
            if value.is_empty() || value.len() > 31 {
                return fail(format!(
                    "{key} {value:?} must be 1 to 31 bytes (After Effects)"
                ));
            }
        }
        if self.ffgl_code.len() != 4 || !self.ffgl_code.chars().all(|c| c.is_ascii_graphic()) {
            return fail(format!(
                "ffgl-code {:?} must be 4 printable ASCII characters",
                self.ffgl_code
            ));
        }
        if self.short_name.is_empty() || self.short_name.len() > 16 {
            return fail(format!(
                "{:?} is over FFGL's 16 bytes: set a short-name",
                self.short_name
            ));
        }
        let word = |c: char| c.is_ascii_alphanumeric() || c == '-' || c == '_';
        if self.file.is_empty() || !self.file.chars().all(word) {
            return fail(format!(
                "file {:?} must be letters, digits, - and _ (it names bundles and DLLs)",
                self.file
            ));
        }
        let parts: Vec<&str> = self.id.split('.').collect();
        if parts.len() < 2 || parts.iter().any(|p| p.is_empty() || !p.chars().all(word)) {
            return fail(format!(
                "id {:?} must be reverse-DNS, like com.example.MyEffect",
                self.id
            ));
        }
        if let Some(guid) = &self.windows_installer_id
            && !is_guid(guid)
        {
            return fail(format!(
                "windows-installer-id {guid:?} must be a GUID, like 2FFBD5EA-ABD9-4A8F-9ACC-24CF0C2C2E3E"
            ));
        }
        if !matches!(self.processor.as_str(), "gpu" | "cpu") {
            return fail(format!(
                "processor {:?} must be \"gpu\" or \"cpu\"",
                self.processor
            ));
        }
        if let Some(key) = self
            .features
            .keys()
            .find(|k| !FEATURE_KEYS.contains(&k.as_str()))
        {
            return fail(format!(
                "features.{key}: the keys are {}",
                FEATURE_KEYS.join(", ")
            ));
        }
        Ok(())
    }

    /// The features `cargo vfx` turns on when building for `os` (`macos`, `windows` or
    /// `linux`).
    pub fn features_for(&self, os: &str) -> Vec<String> {
        ["all", os]
            .iter()
            .filter_map(|k| self.features.get(*k))
            .flatten()
            .cloned()
            .collect()
    }

    /// Major, minor and patch of the crate's version.
    pub fn version_parts(&self) -> [u32; 3] {
        let mut parts = self.version.split('.').map(|p| p.parse().unwrap_or(0));
        [(); 3].map(|_| parts.next().unwrap_or(0))
    }
}

fn is_guid(text: &str) -> bool {
    let groups: Vec<&str> = text.split('-').collect();
    groups.iter().map(|g| g.len()).eq([8, 4, 4, 4, 12])
        && groups
            .iter()
            .all(|g| g.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Checks that no two effects in a workspace share an identity a host tells plugins apart
/// by. `effects` is each crate's package name and metadata.
pub fn check_workspace(effects: &[(String, Metadata)]) -> Result<(), Vec<String>> {
    type Field = fn(&Metadata) -> Option<&str>;
    let mut errors = Vec::new();
    let fields: [(&str, Field); 6] = [
        ("name", |m| Some(&m.name)),
        ("file", |m| Some(&m.file)),
        ("id", |m| Some(&m.id)),
        ("ae-match-name", |m| Some(&m.ae_match_name)),
        ("ffgl-code", |m| Some(&m.ffgl_code)),
        ("windows-installer-id", |m| {
            m.windows_installer_id.as_deref()
        }),
    ];
    for (key, get) in fields {
        let mut seen: BTreeMap<String, &str> = BTreeMap::new();
        for (package, meta) in effects {
            let Some(value) = get(meta) else { continue };
            // Hosts compare some of these without case (file systems, GUIDs).
            if let Some(other) = seen.insert(value.to_lowercase(), package) {
                errors.push(format!("{other} and {package} share {key} {value:?}"));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// The path dependencies of the crate in `manifest_dir` that are effect crates, in the order
/// its `[dependencies]` lists them: each as the name code refers to it by, and its metadata.
/// A host that links effects in (the preview) builds its list from this.
pub fn effect_dependencies(manifest_dir: &Path) -> Result<Vec<(String, Metadata)>, String> {
    let path = manifest_dir.join("Cargo.toml");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let doc: toml::Value = text
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut found = Vec::new();
    let Some(deps) = doc.get("dependencies").and_then(toml::Value::as_table) else {
        return Ok(found);
    };
    for (name, dep) in deps {
        let Some(dir) = dep.get("path").and_then(toml::Value::as_str) else {
            continue;
        };
        let dir = manifest_dir.join(dir);
        let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default();
        if manifest.contains("[package.metadata.vfx]") {
            found.push((name.replace('-', "_"), Metadata::read(&dir)?));
        }
    }
    Ok(found)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> Metadata {
        Metadata {
            name: "Test Effect".into(),
            short_name: "Test Effect".into(),
            category: "Tests".into(),
            file: "TestEffect".into(),
            id: "com.example.TestEffect".into(),
            ae_match_name: "EXAMPLE TestEffect".into(),
            ffgl_code: "TEST".into(),
            windows_installer_id: Some("2FFBD5EA-ABD9-4A8F-9ACC-24CF0C2C2E3E".into()),
            processor: "gpu".into(),
            version: "1.2.3".into(),
            description: String::new(),
            features: BTreeMap::new(),
            windows_dlls: Vec::new(),
        }
    }

    #[test]
    fn identities_hosts_would_reject_fail_the_build() {
        assert_eq!(meta().check(), Ok(()));
        let broken: [fn(&mut Metadata); 8] = [
            |m| m.ffgl_code = "TOOLONG".into(),
            |m| m.short_name = "Seventeen bytes!!".into(),
            |m| m.file = "Has Space".into(),
            |m| m.id = "NoDots".into(),
            |m| m.ae_match_name = "x".repeat(32),
            |m| m.windows_installer_id = Some("not-a-guid".into()),
            |m| m.processor = "npu".into(),
            |m| {
                m.features.insert("solaris".into(), vec!["x".into()]);
            },
        ];
        for (i, f) in broken.iter().enumerate() {
            let mut m = meta();
            f(&mut m);
            assert!(m.check().is_err(), "case {i} passed");
        }
    }

    #[test]
    fn features_by_os() {
        let mut m = meta();
        m.features.insert("all".into(), vec!["a".into()]);
        m.features.insert("windows".into(), vec!["w".into()]);
        assert_eq!(m.features_for("windows"), ["a", "w"]);
        assert_eq!(m.features_for("macos"), ["a"]);
    }

    #[test]
    fn workspace_identities_are_unique() {
        let a = ("a".to_owned(), meta());
        let mut other = meta();
        other.name = "Other".into();
        other.file = "Other".into();
        other.id = "com.example.Other".into();
        other.ffgl_code = "OTHR".into();
        other.windows_installer_id = None;
        // Same match name, differing only in case.
        other.ae_match_name = "example testeffect".into();
        let errors = check_workspace(&[a.clone(), ("b".into(), other.clone())]).unwrap_err();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("ae-match-name"));
        other.ae_match_name = "EXAMPLE Other".into();
        assert_eq!(check_workspace(&[a, ("b".into(), other)]), Ok(()));
    }

    #[test]
    fn the_preview_links_every_effect_crate() {
        let preview = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/preview");
        let names: Vec<String> = effect_dependencies(&preview)
            .unwrap()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert!(names.contains(&"vfx_chain".to_owned()), "{names:?}");
        assert!(names.contains(&"segcam_motion".to_owned()), "{names:?}");
    }
}
