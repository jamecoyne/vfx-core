//! `cargo vfx check`: what would go wrong in a host, found before building. Each crate's own
//! identity is checked by its build script (`vfx_build::Metadata::check`); every command
//! checks that no two effects share one (`vfx_build::check_workspace`). This adds what needs
//! the whole repository or this machine: the preview offering every effect, and installed
//! After Effects plugins that already use an effect's match name.

use std::fs;
use std::path::Path;

use crate::{Effect, MEDIACORE};

/// Problems, one per line; empty when there are none.
pub fn check(all: &[Effect], root: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    let preview = root.join("vfx-core/examples/preview");
    match vfx_build::effect_dependencies(&preview) {
        Ok(linked) => {
            for e in all {
                if !linked.iter().any(|(name, _)| *name == e.lib()) {
                    problems.push(format!(
                        "the preview doesn't offer {}: add it to vfx-core/examples/preview/Cargo.toml",
                        e.package
                    ));
                }
            }
        }
        Err(err) => problems.push(format!("the preview's effects: {err}")),
    }
    if cfg!(target_os = "macos") {
        for e in all {
            problems.extend(installed_clashes(Path::new(MEDIACORE), e));
        }
    }
    problems
}

/// Installed After Effects plugins other than this effect's own bundle whose resources
/// name its match name: After Effects would load only one of them.
pub fn installed_clashes(dir: &Path, e: &Effect) -> Vec<String> {
    let own = format!("{}.plugin", e.meta.file);
    let needle = e.meta.ae_match_name.as_bytes();
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".plugin") || name == own {
            continue;
        }
        let resources = entry.path().join("Contents/Resources");
        let Ok(files) = fs::read_dir(&resources) else {
            continue;
        };
        let names_it = files.flatten().any(|f| {
            f.path().extension().is_some_and(|x| x == "rsrc")
                && fs::read(f.path()).is_ok_and(|bytes| contains(&bytes, needle))
        });
        if names_it {
            found.push(format!(
                "{} is installed and uses {}'s After Effects match name {:?}: remove it, or After Effects loads only one",
                entry.path().display(),
                e.package,
                e.meta.ae_match_name
            ));
        }
    }
    found
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use vfx_build::Metadata;

    fn effect() -> Effect {
        Effect {
            package: "vfx-chain".into(),
            meta: Metadata {
                name: "VFX Preview".into(),
                short_name: "VFX Preview".into(),
                category: "vfx-core".into(),
                file: "VfxPreview".into(),
                id: "com.vfx-core.VfxPreview".into(),
                ae_match_name: "VFXPreview".into(),
                ffgl_code: "VFXP".into(),
                windows_installer_id: None,
                processor: "gpu".into(),
                version: "1.0.0".into(),
                description: String::new(),
                features: BTreeMap::new(),
                windows_dlls: Vec::new(),
            },
        }
    }

    #[test]
    fn an_old_bundle_with_the_same_match_name_is_found() {
        let dir = std::env::temp_dir().join(format!("cargo-vfx-check-{}", std::process::id()));
        let bundle = |name: &str, rsrc: &[u8]| {
            let res = dir.join(name).join("Contents/Resources");
            fs::create_dir_all(&res).unwrap();
            fs::write(res.join("x.rsrc"), rsrc).unwrap();
        };
        bundle("VfxPreview.plugin", b"..VFXPreview..");
        bundle("VfxAdobe.plugin", b"..VFXPreview..");
        bundle("Other.plugin", b"..SomethingElse..");
        let found = installed_clashes(&dir, &effect());
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("VfxAdobe.plugin"));
    }

    #[test]
    fn the_repository_passes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let all = crate::workspace_in(&root).unwrap().0;
        let problems: Vec<String> = check(&all, &root)
            .into_iter()
            .filter(|p| !p.contains("is installed"))
            .collect();
        assert!(problems.is_empty(), "{problems:?}");
    }
}
