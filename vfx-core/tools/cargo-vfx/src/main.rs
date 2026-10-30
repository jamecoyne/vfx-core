//! `cargo vfx`: builds vfx-core effect crates into plugins for every host.
//!
//! An effect crate is any workspace package with `[package.metadata.vfx]`. Each one builds
//! to a single binary per architecture with every host target's entry points, which this
//! tool copies into each host's bundle layout:
//!
//! ```text
//! cargo vfx list                               the effects and their identities
//! cargo vfx build [-p EFFECT]... [--hosts ae,ofx,ffgl] [--debug] [--native]
//! cargo vfx package [-p EFFECT]...             macOS: signed (if asked) installers
//! cargo vfx install [-p EFFECT]... [--hosts ffgl]
//! cargo vfx preview [-p EFFECT] [-- ARGS]      the preview app, opened on the effect
//! cargo vfx env -p EFFECT                      the identity as KEY="value" lines
//! cargo vfx check                              problems a host would hit, before building
//! cargo vfx new NAME                           a new effect crate in vfx-core/effects/NAME
//! ```
//!
//! Every command first checks that no two effects share an identity. Builds turn on the
//! features an effect lists for the OS in `[package.metadata.vfx.features]`.
//!
//! Output goes to `target/vfx-plugin/`. `--native` builds only this machine's
//! architecture on macOS (quicker, for tests); without it macOS bundles are universal.
//! Signing is opt-in, through `SIGN_IDENTITY`, `NOTARY_PROFILE` and `INSTALLER_IDENTITY`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use vfx_build::Metadata;

mod check;
mod new;

const USAGE: &str = "usage: cargo vfx <list|build|package|install|preview|env|check> [-p EFFECT]... \
[--hosts ae,ofx,ffgl] [--debug] [--native] [-- ARGS]\n       cargo vfx new NAME";

const MEDIACORE: &str = "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore";
const MAC_OFX: &str = "/Library/OFX/Plugins";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Host {
    Ae,
    Ofx,
    Ffgl,
}

impl Host {
    const ALL: [Host; 3] = [Host::Ae, Host::Ofx, Host::Ffgl];

    fn feature(self) -> &'static str {
        match self {
            Host::Ae => "ae",
            Host::Ofx => "ofx",
            Host::Ffgl => "ffgl",
        }
    }

    fn parse(text: &str) -> Result<Self, String> {
        Host::ALL
            .into_iter()
            .find(|h| h.feature() == text)
            .ok_or_else(|| format!("unknown host `{text}` (ae, ofx or ffgl)"))
    }
}

struct Effect {
    package: String,
    meta: Metadata,
}

impl Effect {
    /// The library's file stem: `vfx-chain` builds `libvfx_chain.dylib`.
    fn lib(&self) -> String {
        self.package.replace('-', "_")
    }
}

struct Options {
    command: String,
    packages: Vec<String>,
    hosts: Vec<Host>,
    release: bool,
    native: bool,
    rest: Vec<String>,
}

fn parse_args() -> Result<Options, String> {
    // Cargo runs `cargo-vfx vfx …` for `cargo vfx …`.
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "vfx") {
        args.remove(0);
    }
    let mut it = args.into_iter();
    let command = it.next().ok_or(USAGE)?;
    let mut o = Options {
        command,
        packages: Vec::new(),
        hosts: Host::ALL.to_vec(),
        release: true,
        native: false,
        rest: Vec::new(),
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-p" | "--package" => o.packages.push(it.next().ok_or("-p needs a package")?),
            "--hosts" => {
                let list = it.next().ok_or("--hosts needs a list")?;
                o.hosts = list.split(',').map(Host::parse).collect::<Result<_, _>>()?;
            }
            "--debug" => o.release = false,
            "--native" => o.native = true,
            "--" => o.rest.extend(it.by_ref()),
            "-h" | "--help" => return Err(USAGE.into()),
            name if o.command == "new" && !name.starts_with('-') => o.rest.push(name.to_owned()),
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    Ok(o)
}

/// The workspace's effect crates, and its root and target directory.
fn workspace() -> Result<(Vec<Effect>, PathBuf, PathBuf), String> {
    workspace_in(Path::new("."))
}

/// [`workspace`] for the workspace containing `dir`, its identities checked.
fn workspace_in(dir: &Path) -> Result<(Vec<Effect>, PathBuf, PathBuf), String> {
    let out = Command::new(env_or("CARGO", "cargo"))
        .current_dir(dir)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("cargo metadata: {e}"))?;
    let mut effects = Vec::new();
    for p in json["packages"]
        .as_array()
        .ok_or("cargo metadata: no packages")?
    {
        if p["metadata"]["vfx"].is_null() {
            continue;
        }
        let manifest = PathBuf::from(p["manifest_path"].as_str().unwrap_or_default());
        let dir = manifest.parent().unwrap_or(Path::new(".")).to_path_buf();
        effects.push(Effect {
            package: p["name"].as_str().unwrap_or_default().to_owned(),
            meta: Metadata::read(&dir)?,
        });
    }
    effects.sort_by(|a, b| a.package.cmp(&b.package));
    let identities: Vec<(String, Metadata)> = effects
        .iter()
        .map(|e| (e.package.clone(), e.meta.clone()))
        .collect();
    vfx_build::check_workspace(&identities).map_err(|errors| errors.join("\n"))?;
    let root = PathBuf::from(json["workspace_root"].as_str().unwrap_or("."));
    let target = PathBuf::from(json["target_directory"].as_str().unwrap_or("target"));
    Ok((effects, root, target))
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let status = cmd.status().map_err(|e| format!("{cmd:?}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{cmd:?} failed ({status})"))
    }
}

fn main() -> ExitCode {
    match parse_args().and_then(|o| dispatch(&o)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cargo vfx: {e}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(o: &Options) -> Result<(), String> {
    if o.command == "new" {
        let [name] = o.rest.as_slice() else {
            return Err("usage: cargo vfx new NAME".into());
        };
        let (_, root, _) = workspace()?;
        return new::new(&root, name);
    }
    let (all, root, target) = workspace()?;
    let chosen: Vec<&Effect> = if o.packages.is_empty() {
        all.iter().collect()
    } else {
        o.packages
            .iter()
            .map(|name| {
                all.iter().find(|e| &e.package == name).ok_or_else(|| {
                    let names: Vec<&str> = all.iter().map(|e| e.package.as_str()).collect();
                    format!("no effect crate `{name}` (effects: {})", names.join(", "))
                })
            })
            .collect::<Result<_, _>>()?
    };
    let out = target.join("vfx-plugin");
    match o.command.as_str() {
        "list" => {
            for e in &all {
                let m = &e.meta;
                println!(
                    "{:16} {:22} {} ({} effect), file {}, OpenFX {}, AE {:?}, FFGL {}",
                    e.package,
                    m.name,
                    m.version,
                    m.processor,
                    m.file,
                    m.id,
                    m.ae_match_name,
                    m.ffgl_code
                );
            }
            Ok(())
        }
        "env" => {
            let [e] = chosen.as_slice() else {
                return Err("env takes one -p EFFECT".into());
            };
            print_env(e);
            Ok(())
        }
        "build" => {
            for e in &chosen {
                build(e, o, &target, &out)?;
            }
            Ok(())
        }
        "package" => {
            for e in &chosen {
                warn_installed(e, o);
                package(e, o, &target, &out)?;
            }
            Ok(())
        }
        "install" => {
            for e in &chosen {
                warn_installed(e, o);
                install(e, o, &target, &out)?;
            }
            Ok(())
        }
        "preview" => preview(&chosen, &root, o),
        "check" => {
            let problems = check::check(&all, &root);
            for p in &problems {
                eprintln!("{p}");
            }
            if problems.is_empty() {
                println!("{} effects: no problems", all.len());
                Ok(())
            } else {
                Err(format!("{} problems", problems.len()))
            }
        }
        other => Err(format!("unknown command `{other}`\n{USAGE}")),
    }
}

/// Warns about installed After Effects plugins that would clash with this one.
fn warn_installed(e: &Effect, o: &Options) {
    if cfg!(target_os = "macos") && o.hosts.contains(&Host::Ae) {
        for clash in check::installed_clashes(Path::new(MEDIACORE), e) {
            eprintln!("warning: {clash}");
        }
    }
}

/// The identity as KEY="value" lines, for shell scripts (the Windows installer).
fn print_env(e: &Effect) {
    let m = &e.meta;
    let pairs = [
        ("PLUGIN_NAME", m.name.as_str()),
        ("PLUGIN_SHORT_NAME", &m.short_name),
        ("PLUGIN_CATEGORY", &m.category),
        ("PLUGIN_FILE", &m.file),
        ("PLUGIN_ID", &m.id),
        ("AE_MATCH_NAME", &m.ae_match_name),
        ("FFGL_ID", &m.ffgl_code),
        (
            "WINDOWS_INSTALLER_ID",
            m.windows_installer_id.as_deref().unwrap_or(""),
        ),
        ("PLUGIN_VERSION", &m.version),
        ("PLUGIN_CRATE", &e.package),
    ];
    for (k, v) in pairs {
        println!("{k}=\"{v}\"");
    }
}

/// The OS a target triple builds for (`None`: this machine's).
fn target_os(triple: Option<&str>) -> &'static str {
    match triple {
        Some(t) if t.contains("apple") => "macos",
        Some(t) if t.contains("windows") => "windows",
        Some(_) => "linux",
        None if cfg!(target_os = "macos") => "macos",
        None if cfg!(target_os = "windows") => "windows",
        None => "linux",
    }
}

/// The plugin binary for one target triple (`None`: the host's own).
fn cargo_build(e: &Effect, o: &Options, triple: Option<&str>) -> Result<(), String> {
    let mut features: Vec<String> = o.hosts.iter().map(|h| h.feature().to_owned()).collect();
    features.extend(e.meta.features_for(target_os(triple)));
    let mut cmd = Command::new(env_or("CARGO", "cargo"));
    cmd.args([
        "build",
        "-p",
        &e.package,
        "--lib",
        "--features",
        &features.join(","),
    ]);
    if o.release {
        cmd.arg("--release");
    }
    if let Some(t) = triple {
        cmd.args(["--target", t]);
    }
    run(&mut cmd)
}

fn profile(o: &Options) -> &'static str {
    if o.release { "release" } else { "debug" }
}

/// Builds the effect and wraps the binary into each host's bundle. Returns the bundles.
fn build(e: &Effect, o: &Options, target: &Path, out: &Path) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(out).map_err(|err| err.to_string())?;
    if cfg!(target_os = "macos") {
        build_mac(e, o, target, out)
    } else if cfg!(target_os = "windows") {
        build_windows(e, o, target, out)
    } else {
        Err("plugins build on macOS and Windows".into())
    }
}

fn build_mac(e: &Effect, o: &Options, target: &Path, out: &Path) -> Result<Vec<PathBuf>, String> {
    let m = &e.meta;
    let archs: Vec<&str> = if o.native {
        vec![if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            "x86_64"
        }]
    } else {
        vec!["aarch64", "x86_64"]
    };
    let mut binaries = Vec::new();
    let mut rsrc = None;
    for arch in &archs {
        let triple = format!("{arch}-apple-darwin");
        cargo_build(e, o, Some(&triple))?;
        let dir = target.join(&triple).join(profile(o));
        binaries.push(dir.join(format!("lib{}.dylib", e.lib())));
        if o.hosts.contains(&Host::Ae) {
            rsrc = Some(newest_rsrc(&dir, &e.package, &m.file)?);
        }
    }
    let binary = out.join(format!(".{}.universal", m.file));
    run(Command::new("lipo")
        .arg("-create")
        .args(&binaries)
        .arg("-output")
        .arg(&binary))?;

    let mut bundles = Vec::new();
    for host in &o.hosts {
        let (bundle, exe, id, kind) = match host {
            Host::Ae => (
                format!("{}.plugin", m.file),
                m.file.clone(),
                format!("{}.adobe", m.id),
                "eFKT",
            ),
            Host::Ofx => (
                format!("{}.ofx.bundle", m.file),
                format!("{}.ofx", m.file),
                m.id.clone(),
                "BNDL",
            ),
            Host::Ffgl => (
                format!("{}.bundle", m.file),
                m.file.clone(),
                format!("{}.ffgl", m.id),
                "BNDL",
            ),
        };
        let bundle = out.join(bundle);
        let _ = fs::remove_dir_all(&bundle);
        let contents = bundle.join("Contents");
        fs::create_dir_all(contents.join("MacOS")).map_err(|err| err.to_string())?;
        fs::copy(&binary, contents.join("MacOS").join(&exe)).map_err(|err| err.to_string())?;
        let signature = if *host == Host::Ae {
            fs::create_dir_all(contents.join("Resources")).map_err(|err| err.to_string())?;
            let rsrc = rsrc.as_ref().expect("built with the ae feature");
            fs::copy(
                rsrc,
                contents.join("Resources").join(format!("{}.rsrc", m.file)),
            )
            .map_err(|err| err.to_string())?;
            fs::write(contents.join("PkgInfo"), "eFKTFXTC").map_err(|err| err.to_string())?;
            "\n  <key>CFBundleSignature</key><string>FXTC</string>"
        } else {
            ""
        };
        let plist = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>{exe}</string>
  <key>CFBundleIdentifier</key><string>{id}</string>
  <key>CFBundleName</key><string>{name}</string>
  <key>CFBundlePackageType</key><string>{kind}</string>{signature}
  <key>CFBundleVersion</key><string>{version}</string>
  <key>CFBundleShortVersionString</key><string>{version}</string>
</dict>
</plist>
"#,
            name = xml(&m.name),
            version = m.version,
        );
        fs::write(contents.join("Info.plist"), plist).map_err(|err| err.to_string())?;
        println!("{}", bundle.display());
        bundles.push(bundle);
    }
    let _ = fs::remove_file(&binary);
    Ok(bundles)
}

fn xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The PiPL resource the effect's build script wrote: the newest one for this crate.
fn newest_rsrc(profile_dir: &Path, package: &str, file: &str) -> Result<PathBuf, String> {
    let build = profile_dir.join("build");
    let prefix = format!("{package}-");
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = fs::read_dir(&build)
        .map_err(|e| format!("{}: {e}", build.display()))?
        .filter_map(Result::ok)
        .filter(|d| d.file_name().to_string_lossy().starts_with(&prefix))
        .map(|d| d.path().join("out").join(format!("{file}.rsrc")))
        .filter_map(|p| Some((p.metadata().ok()?.modified().ok()?, p)))
        .collect();
    found.sort();
    found
        .pop()
        .map(|(_, p)| p)
        .ok_or_else(|| format!("no {file}.rsrc under {}", build.display()))
}

fn build_windows(
    e: &Effect,
    o: &Options,
    target: &Path,
    out: &Path,
) -> Result<Vec<PathBuf>, String> {
    let m = &e.meta;
    cargo_build(e, o, None)?;
    let dll = target.join(profile(o)).join(format!("{}.dll", e.lib()));
    let mut bundles = Vec::new();
    for host in &o.hosts {
        let path = match host {
            Host::Ae => out.join(format!("{}.aex", m.file)),
            // Same layout as on macOS, with the DLL renamed .ofx under Contents/Win64.
            Host::Ofx => {
                let bundle = out.join(format!("{}.ofx.bundle", m.file));
                let _ = fs::remove_dir_all(&bundle);
                let dir = bundle.join("Contents").join("Win64");
                fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
                fs::copy(&dll, dir.join(format!("{}.ofx", m.file)))
                    .map_err(|err| err.to_string())?;
                for name in &m.windows_dlls {
                    fs::copy(target.join(profile(o)).join(name), dir.join(name))
                        .map_err(|err| format!("{name}: {err}"))?;
                }
                println!("{}", bundle.display());
                bundles.push(bundle);
                continue;
            }
            Host::Ffgl => out.join(format!("{}.dll", m.file)),
        };
        fs::copy(&dll, &path).map_err(|err| err.to_string())?;
        println!("{}", path.display());
        bundles.push(path);
    }
    Ok(bundles)
}

/// macOS installers for the system-wide hosts, signed and notarized when asked.
fn package(e: &Effect, o: &Options, target: &Path, out: &Path) -> Result<(), String> {
    if !cfg!(target_os = "macos") {
        return Err(
            "package builds macOS installers; on Windows use vfx-core/wrappers/package-win.sh"
                .into(),
        );
    }
    let release = Options {
        release: true,
        native: false,
        command: o.command.clone(),
        packages: Vec::new(),
        hosts: o.hosts.clone(),
        rest: Vec::new(),
    };
    let m = &e.meta;
    fs::create_dir_all(out).map_err(|err| err.to_string())?;
    for bundle in build_mac(e, &release, target, out)? {
        sign_and_notarize(&bundle, out)?;
        let name = bundle
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let (suffix, location) = if name.ends_with(".plugin") {
            ("adobe", MEDIACORE)
        } else if name.ends_with(".ofx.bundle") {
            ("ofx", MAC_OFX)
        } else {
            continue; // FFGL installs per user, by copying.
        };
        let pkg = out.join(format!("{}-{suffix}.pkg", m.file));
        pkgbuild(
            &bundle,
            &pkg,
            &format!("{}.{suffix}", m.id),
            &m.version,
            location,
        )?;
        println!("{}", pkg.display());
    }
    Ok(())
}

fn sign_and_notarize(bundle: &Path, out: &Path) -> Result<(), String> {
    let identity = std::env::var("SIGN_IDENTITY")
        .ok()
        .filter(|s| !s.is_empty());
    let notary = std::env::var("NOTARY_PROFILE")
        .ok()
        .filter(|s| !s.is_empty());
    if let Some(id) = &identity {
        run(Command::new("codesign")
            .args([
                "--force",
                "--options",
                "runtime",
                "--timestamp",
                "--sign",
                id,
            ])
            .arg(bundle))?;
        run(Command::new("codesign")
            .args(["--verify", "--strict", "--verbose=2"])
            .arg(bundle))?;
    }
    if let Some(profile) = notary {
        if identity.is_none() {
            return Err(
                "NOTARY_PROFILE needs SIGN_IDENTITY: Apple only notarizes signed code".into(),
            );
        }
        let zip = out.join(".notarize.zip");
        run(Command::new("ditto")
            .args(["-c", "-k", "--keepParent"])
            .arg(bundle)
            .arg(&zip))?;
        run(Command::new("xcrun")
            .args(["notarytool", "submit"])
            .arg(&zip)
            .args(["--keychain-profile", &profile, "--wait"]))?;
        run(Command::new("xcrun")
            .args(["stapler", "staple"])
            .arg(bundle))?;
        let _ = fs::remove_file(zip);
    }
    Ok(())
}

/// A component package that installs `bundle` into `location`, never relocated to a copy
/// with the same bundle ID elsewhere on disk.
fn pkgbuild(
    bundle: &Path,
    pkg: &Path,
    id: &str,
    version: &str,
    location: &str,
) -> Result<(), String> {
    let stage = std::env::temp_dir().join(format!("vfx-pkg-{}", std::process::id()));
    let _ = fs::remove_dir_all(&stage);
    fs::create_dir_all(&stage).map_err(|e| e.to_string())?;
    // The payload root's mode becomes the install folder's.
    run(Command::new("chmod").arg("755").arg(&stage))?;
    let name = bundle.file_name().unwrap_or_default();
    // No extended attributes, or the payload gains AppleDouble ._ files.
    run(Command::new("ditto")
        .args(["--norsrc", "--noextattr", "--noqtn"])
        .arg(bundle)
        .arg(stage.join(name)))?;
    let components = stage.with_extension("plist");
    run(Command::new("pkgbuild")
        .args(["--analyze", "--root"])
        .arg(&stage)
        .arg(&components))?;
    run(Command::new("plutil")
        .args(["-replace", "0.BundleIsRelocatable", "-bool", "NO"])
        .arg(&components))?;
    let mut cmd = Command::new("pkgbuild");
    cmd.env("COPYFILE_DISABLE", "1")
        .arg("--root")
        .arg(&stage)
        .arg("--component-plist")
        .arg(&components)
        .args([
            "--install-location",
            location,
            "--identifier",
            id,
            "--version",
            version,
        ]);
    if let Ok(identity) = std::env::var("INSTALLER_IDENTITY")
        && !identity.is_empty()
    {
        cmd.args(["--sign", &identity]);
    }
    let result = run(cmd.arg(pkg));
    let _ = fs::remove_dir_all(&stage);
    let _ = fs::remove_file(&components);
    result
}

/// Resolume plugins install per user; After Effects and OpenFX plugins go in system
/// folders, which need the user's password, so for those this prints the command.
fn install(e: &Effect, o: &Options, target: &Path, out: &Path) -> Result<(), String> {
    let m = &e.meta;
    if o.hosts.contains(&Host::Ffgl) {
        let ffgl = Options {
            hosts: vec![Host::Ffgl],
            command: o.command.clone(),
            packages: Vec::new(),
            rest: Vec::new(),
            ..*o
        };
        let bundle = build(e, &ffgl, target, out)?.remove(0);
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .ok_or("no home directory")?;
        let mut installed = false;
        for app in ["Resolume Avenue", "Resolume Arena"] {
            let app_dir = home.join("Documents").join(app);
            if !app_dir.is_dir() {
                continue;
            }
            let dir = app_dir.join("Extra Effects");
            fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
            let dest = dir.join(bundle.file_name().unwrap_or_default());
            let _ = fs::remove_dir_all(&dest);
            let _ = fs::remove_file(&dest);
            copy_all(&bundle, &dest)?;
            println!("installed {}", dest.display());
            installed = true;
        }
        if !installed {
            println!(
                "no Resolume Documents folder found; the bundle is {}",
                bundle.display()
            );
        }
    }
    let system: Vec<Host> = o
        .hosts
        .iter()
        .copied()
        .filter(|h| *h != Host::Ffgl)
        .collect();
    if !system.is_empty() && cfg!(target_os = "macos") {
        let pkgs = Options {
            hosts: system.clone(),
            command: o.command.clone(),
            packages: Vec::new(),
            rest: Vec::new(),
            ..*o
        };
        package(e, &pkgs, target, out)?;
        println!("These install into system folders and need your password:");
        for host in system {
            let suffix = if host == Host::Ae { "adobe" } else { "ofx" };
            let pkg = out.join(format!("{}-{suffix}.pkg", m.file));
            println!("  sudo installer -pkg '{}' -target /", pkg.display());
        }
    } else if !system.is_empty() {
        println!(
            "On Windows, install with vfx-core/wrappers/package-win.sh's installer, or copy the bundles from {}",
            out.display()
        );
    }
    Ok(())
}

fn copy_all(from: &Path, to: &Path) -> Result<(), String> {
    if from.is_dir() {
        fs::create_dir_all(to).map_err(|e| e.to_string())?;
        for entry in fs::read_dir(from).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            copy_all(&entry.path(), &to.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        fs::copy(from, to).map(|_| ()).map_err(|e| e.to_string())
    }
}

/// The preview app, opened on the effect. It links every effect in its catalog.
fn preview(chosen: &[&Effect], root: &Path, o: &Options) -> Result<(), String> {
    let mut cmd = Command::new(env_or("CARGO", "cargo"));
    cmd.current_dir(root)
        .args(["run", "-p", "vfx-preview", "--release", "--"]);
    if let [e] = chosen {
        cmd.args(["--effect", &slug(&e.meta.short_name)]);
    }
    run(cmd.args(&o.rest))
}

/// How the preview names an effect on its command line: "segcam Motion (Rust)" is
/// `segcam-motion-rust`.
fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_and_hosts() {
        assert_eq!(slug("segcam Motion (Rust)"), "segcam-motion-rust");
        assert_eq!(slug("VFX Preview"), "vfx-preview");
        assert_eq!(Host::parse("ofx"), Ok(Host::Ofx));
        assert!(Host::parse("vst").is_err());
    }
}
