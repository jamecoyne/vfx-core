//! `cargo vfx new NAME`: an effect crate in `effects/NAME`, ready to build for every host and
//! to try in the preview. It starts as a CPU effect that inverts the picture by an amount;
//! `effects/vfx-chain` shows a GPU one. The crate joins the workspace and the preview's
//! dependencies, so nothing else needs editing.

use std::fs;
use std::path::Path;

/// Writes the crate and registers it. `name` is the package name, like `my-glow`.
pub fn new(root: &Path, name: &str) -> Result<(), String> {
    if name.is_empty()
        || !name.starts_with(|c: char| c.is_ascii_lowercase())
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(format!(
            "`{name}`: name the crate in lower case, digits and hyphens, like my-glow"
        ));
    }
    let dir = root.join("effects").join(name);
    if dir.exists() {
        return Err(format!("{} already exists", dir.display()));
    }
    let id = Identity::from(name);
    fs::create_dir_all(dir.join("src")).map_err(|e| e.to_string())?;
    let write = |file: &str, text: String| {
        fs::write(dir.join(file), text).map_err(|e| format!("{file}: {e}"))
    };
    write("Cargo.toml", id.manifest(name))?;
    write("build.rs", BUILD_RS.to_owned())?;
    write("src/lib.rs", id.lib())?;

    let workspace = root.join("Cargo.toml");
    let text = fs::read_to_string(&workspace).map_err(|e| e.to_string())?;
    let text = add_member(&text, &format!("effects/{name}"))?;
    fs::write(&workspace, text).map_err(|e| e.to_string())?;

    let preview = root.join("examples/preview/Cargo.toml");
    let text = fs::read_to_string(&preview).map_err(|e| e.to_string())?;
    let text = add_effect_dependency(&text, name)?;
    fs::write(&preview, text).map_err(|e| e.to_string())?;

    println!("created {} ({})", dir.display(), id.name);
    println!("  cargo vfx preview -p {name}     try it");
    println!("  cargo vfx build -p {name}       the plugins");
    Ok(())
}

/// A new effect's names and IDs, made from its crate name. Change them before the first
/// release: hosts find saved effects by them.
struct Identity {
    /// "My Glow"
    name: String,
    /// "MyGlow"
    file: String,
    ffgl_code: String,
    guid: String,
}

impl Identity {
    fn from(crate_name: &str) -> Self {
        let words: Vec<String> = crate_name
            .split('-')
            .filter(|w| !w.is_empty())
            .map(|w| {
                let mut c = w.chars();
                c.next()
                    .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                    .unwrap_or_default()
            })
            .collect();
        let name: String = words.join(" ").chars().take(31).collect();
        let file: String = words.concat().chars().take(31).collect();
        let mut ffgl_code: String = file
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .take(4)
            .collect::<String>()
            .to_ascii_uppercase();
        while ffgl_code.len() < 4 {
            ffgl_code.push('X');
        }
        Self {
            name,
            file,
            ffgl_code,
            guid: guid(crate_name),
        }
    }

    fn manifest(&self, crate_name: &str) -> String {
        let Self {
            name,
            file,
            ffgl_code,
            guid,
        } = self;
        let short_name: String = name.chars().take(16).collect();
        format!(
            r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"
publish = false
description = "Inverts the picture by an amount."

# The plugin's identity in every host. Hosts find a plugin in saved projects by its IDs
# (id, ae-match-name, ffgl-code): change them now, then keep them across versions.
[package.metadata.vfx]
name = "{name}"
short-name = "{short_name}"
category = "vfx-core"
file = "{file}"
id = "com.example.{file}"
ae-match-name = "EXAMPLE {file}"
ffgl-code = "{ffgl_code}"
processor = "cpu"
windows-installer-id = "{guid}"

# The host targets. `cargo vfx build` switches them on and bundles the one binary for each.
[features]
ae = ["vfx-plugin/ae"]
ofx = ["vfx-plugin/ofx"]
ffgl = ["vfx-plugin/ffgl"]

# The rlib is the effect (for the preview app, tests and tools); the cdylib is the plugin.
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
vfx-core = {{ path = "../.." }}
vfx-plugin = {{ path = "../../wrappers/plugin" }}

[build-dependencies]
vfx-build = {{ path = "../../wrappers/build" }}
"#
        )
    }

    fn lib(&self) -> String {
        let Self { name, file, .. } = self;
        format!(
            r#"//! {name}: a vfx-core effect, made by `cargo vfx new`. It inverts the picture by an
//! amount, on the CPU, at whatever depth the host works in.

use vfx_core::cpu::{{Argb, Depth, FrameSource, ImageMut, ViewMut}};
use vfx_core::{{Context, CpuEffect, ParamSpec, ParamValues}};

vfx_plugin::export!(cpu: {file});

const PARAMS: &[ParamSpec] = &[ParamSpec::float("amount", 0.0, 1.0, 1.0).label("Amount")];

#[derive(Default)]
pub struct {file};

impl CpuEffect for {file} {{
    fn params(&self) -> &[ParamSpec] {{
        PARAMS
    }}

    fn render(
        &mut self,
        frames: &mut dyn FrameSource,
        count: usize,
        output: ImageMut<'_>,
        params: &ParamValues,
        _ctx: &Context,
    ) {{
        let amount = params.float("amount");
        match output {{
            ImageMut::U8(out) => invert(frames, count, out, amount),
            ImageMut::U16(out) => invert(frames, count, out, amount),
            ImageMut::F32(out) => invert(frames, count, out, amount),
        }}
    }}
}}

fn invert<C: Depth>(frames: &mut dyn FrameSource, count: usize, mut out: ViewMut<C>, amount: f32) {{
    frames.with_frame(count - 1, &mut |image| {{
        let Some(input) = image.and_then(C::view) else {{
            return;
        }};
        let mix = |c: C| {{
            let v = c.to_unit();
            C::from_unit(v + (1.0 - 2.0 * v) * amount)
        }};
        for y in out.rows() {{
            for (o, i) in out.row_mut(y).iter_mut().zip(input.row(y)) {{
                *o = Argb {{
                    alpha: i.alpha,
                    red: mix(i.red),
                    green: mix(i.green),
                    blue: mix(i.blue),
                }};
            }}
        }}
    }});
}}
"#
        )
    }
}

const BUILD_RS: &str = "fn main() {\n    vfx_build::configure();\n}\n";

/// A GUID for the Windows installer, from the name and the time: unique enough for one.
fn guid(seed: &str) -> String {
    use std::hash::{BuildHasher, Hasher};
    let state = std::collections::hash_map::RandomState::new();
    let mut words = [0u64; 2];
    for (i, w) in words.iter_mut().enumerate() {
        let mut h = state.build_hasher();
        h.write(seed.as_bytes());
        h.write_usize(i);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        *w = h.finish();
    }
    let hex = format!("{:016X}{:016X}", words[0], words[1]);
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// `text` (the workspace manifest) with `member` added to `[workspace] members`.
fn add_member(text: &str, member: &str) -> Result<String, String> {
    let start = text
        .find("members = [")
        .ok_or("the workspace Cargo.toml has no `members = [`")?;
    let end = start + text[start..].find(']').ok_or("unclosed members list")?;
    let list = &text[start..end];
    if list.contains(&format!("\"{member}\"")) {
        return Ok(text.to_owned());
    }
    let sep = if list.trim_end().ends_with('[') {
        ""
    } else {
        ", "
    };
    Ok(format!("{}{sep}\"{member}\"{}", &text[..end], &text[end..]))
}

/// `text` (the preview's manifest) with a path dependency on `effects/<name>`, after the last
/// effect crate it already lists.
fn add_effect_dependency(text: &str, name: &str) -> Result<String, String> {
    let line = format!("{name} = {{ path = \"../../effects/{name}\" }}\n");
    if text.contains(&line) {
        return Ok(text.to_owned());
    }
    let marker = "path = \"../../effects/";
    let last = text
        .match_indices(marker)
        .last()
        .ok_or("the preview's Cargo.toml lists no effect crate")?
        .0;
    let after = last + text[last..].find('\n').ok_or("no newline")? + 1;
    Ok(format!("{}{line}{}", &text[..after], &text[after..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_from_a_crate_name() {
        let id = Identity::from("my-glow");
        assert_eq!((id.name.as_str(), id.file.as_str()), ("My Glow", "MyGlow"));
        assert_eq!(id.ffgl_code, "MYGL");
        assert_eq!(Identity::from("ab").ffgl_code, "ABXX");
        let parts: Vec<usize> = id.guid.split('-').map(str::len).collect();
        assert_eq!(parts, [8, 4, 4, 4, 12]);
    }

    #[test]
    fn registers_in_the_workspace_and_the_preview() {
        let ws = "[workspace]\nmembers = [\".\", \"effects/a\"]\n";
        assert_eq!(
            add_member(ws, "effects/b").unwrap(),
            "[workspace]\nmembers = [\".\", \"effects/a\", \"effects/b\"]\n"
        );
        let preview = "[dependencies]\nx = 1\na = { path = \"../../effects/a\" }\ny = 2\n";
        assert_eq!(
            add_effect_dependency(preview, "b").unwrap(),
            "[dependencies]\nx = 1\na = { path = \"../../effects/a\" }\nb = { path = \"../../effects/b\" }\ny = 2\n"
        );
    }

    /// The new crate's identity passes the build script's checks.
    #[test]
    fn a_new_identity_builds() {
        let dir = std::env::temp_dir().join(format!("cargo-vfx-new-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("Cargo.toml"),
            Identity::from("my-glow").manifest("my-glow"),
        )
        .unwrap();
        let read = vfx_build::Metadata::read(&dir);
        let _ = fs::remove_dir_all(&dir);
        assert!(read.is_ok(), "{read:?}");
    }
}
