//! `cargo vfx check --quick` and `--full`: the checks to run while iterating, and the ones to
//! run before a pull request. `--full` is CI's steps for this OS, the mini host harnesses and the
//! preview; `--quick` is format, the core clippy and the tests of the packages a branch changes.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use crate::env_or;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Quick,
    Full,
}

/// CI's core crates: plain Rust and wgpu, built and tested on every OS.
const CORE: [&str; 7] = [
    "vfx-core",
    "vfx-host",
    "vfx-build",
    "vfx-chain",
    "segcam-motion",
    "segcam-face",
    "cargo-vfx",
];
const WRAPPERS: [&str; 4] = ["vfx-ae", "vfx-ofx", "vfx-ffgl", "vfx-ffgl-common"];
const HOST_FEATURES: &str = "vfx-chain/ae,vfx-chain/ofx,vfx-chain/ffgl,segcam-motion/ae,segcam-motion/ofx,segcam-motion/ffgl";
const WASM: &str = "wasm32-unknown-unknown";

/// Runs the tier's steps one after another, each logged to `target/checks/NAME.log`.
/// `packages` replaces the changed packages that `--quick` tests.
pub fn run(tier: Tier, packages: &[String], root: &Path, target: &Path) -> Result<(), String> {
    if tier == Tier::Full && !packages.is_empty() {
        return Err("--full checks every package; -p goes with --quick".into());
    }
    let logs = target.join("checks");
    fs::create_dir_all(&logs).map_err(|e| format!("{}: {e}", logs.display()))?;
    let steps = match tier {
        Tier::Quick => quick(packages, root)?,
        Tier::Full => full(root, &logs),
    };
    let start = Instant::now();
    let mut failed = Vec::new();
    for (name, mut cmd) in steps {
        print!("{name:16} ");
        std::io::stdout().flush().ok();
        let log = logs.join(format!("{name}.log"));
        let file = File::create(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        let err = file.try_clone().map_err(|e| e.to_string())?;
        let step = Instant::now();
        let status = cmd.current_dir(root).stdout(file).stderr(err).status();
        let secs = step.elapsed().as_secs();
        match status {
            Ok(s) if s.success() => println!("ok      {secs:>5} s"),
            other => {
                println!("FAILED  {secs:>5} s  {}", log.display());
                if let Err(e) = other {
                    println!("  {cmd:?}: {e}");
                }
                let text = fs::read_to_string(&log).unwrap_or_default();
                let lines: Vec<&str> = text.lines().collect();
                for line in &lines[lines.len().saturating_sub(30)..] {
                    println!("  {line}");
                }
                failed.push(name);
            }
        }
    }
    println!("{:16} {:>13} s", "total", start.elapsed().as_secs());
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!("failed: {}", failed.join(", ")))
    }
}

fn quick(packages: &[String], root: &Path) -> Result<Vec<(String, Command)>, String> {
    let changed = if packages.is_empty() {
        changed_packages(root)?
    } else {
        packages.to_vec()
    };
    let mut steps = vec![fmt(), ("clippy-core".into(), clippy(&CORE, &[]))];
    if changed.is_empty() {
        println!("tests: no package changed");
    } else {
        println!("tests: {}", changed.join(", "));
        let names: Vec<&str> = changed.iter().map(String::as_str).collect();
        steps.push(("tests".into(), test(&names, &[])));
    }
    Ok(steps)
}

fn full(root: &Path, logs: &Path) -> Vec<(String, Command)> {
    let mut steps = vec![
        fmt(),
        ("clippy-core".into(), clippy(&CORE, &[])),
        ("tests-core".into(), test(&CORE, &[])),
        (
            "clippy-noort".into(),
            clippy(&["segcam-face"], &["--no-default-features"]),
        ),
        (
            "tests-noort".into(),
            test(&["segcam-face"], &["--no-default-features"]),
        ),
    ];
    // CI builds the plugins and runs the mini hosts on macOS and Windows only.
    if cfg!(any(target_os = "macos", windows)) {
        let mut plugin_crates = WRAPPERS.to_vec();
        plugin_crates.extend(["vfx-plugin", "vfx-chain", "segcam-motion"]);
        steps.push(("tests-wrappers".into(), test(&WRAPPERS, &[])));
        steps.push((
            "clippy-wrappers".into(),
            clippy(&plugin_crates, &["--features", HOST_FEATURES]),
        ));
        let segcam_dir = std::env::var_os("SEGCAM_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| logs.join("segcam-ofx"));
        for (name, script) in [
            ("ofx-minihost", "vfx-core/wrappers/ofx/minihost/run.sh"),
            (
                "ofx-segcam",
                "plugins/segcam/segcam-motion/ofx-tests/run.sh",
            ),
            ("ffgl-minihost", "vfx-core/wrappers/ffgl/minihost/run.sh"),
            (
                "ffgl-segcam",
                "plugins/segcam/segcam-motion/ffgl-tests/run.sh",
            ),
        ] {
            let mut cmd = bash();
            cmd.arg(root.join(script))
                .env("SEGCAM_TEST_DIR", &segcam_dir);
            if cfg!(windows) && std::env::var_os("CC").is_none() {
                cmd.env("CC", "clang");
            }
            steps.push((name.into(), cmd));
        }
    }
    steps.push(("clippy-preview".into(), clippy(&["vfx-preview"], &[])));
    steps.push(("tests-preview".into(), test(&["vfx-preview"], &[])));
    if wasm_installed() {
        let mut cmd = cargo(&["check", "-p", "vfx-preview", "--target", WASM]);
        // sccache on Windows passes web-sys's arguments on one command line, which is too long.
        cmd.env_remove("RUSTC_WRAPPER");
        steps.push(("wasm-preview".into(), cmd));
    } else {
        println!("wasm-preview: skipped, `rustup target add {WASM}` adds it");
    }
    steps
}

fn cargo(args: &[&str]) -> Command {
    let mut cmd = Command::new(env_or("CARGO", "cargo"));
    cmd.args(args);
    cmd
}

fn fmt() -> (String, Command) {
    ("fmt".into(), cargo(&["fmt", "--all", "--", "--check"]))
}

fn clippy(packages: &[&str], extra: &[&str]) -> Command {
    let mut cmd = cargo(&["clippy"]);
    for p in packages {
        cmd.args(["-p", p]);
    }
    cmd.args(extra)
        .args(["--all-targets", "--", "-D", "warnings"]);
    cmd
}

fn test(packages: &[&str], extra: &[&str]) -> Command {
    let mut cmd = cargo(&["test"]);
    for p in packages {
        cmd.args(["-p", p]);
    }
    cmd.args(extra);
    cmd
}

/// bash for the harness scripts. On Windows that is Git for Windows' own: a plain `bash`
/// would find WSL's in System32 first.
fn bash() -> Command {
    if cfg!(windows)
        && let Ok(out) = Command::new("git").arg("--exec-path").output()
    {
        // The exec path is `<git>/mingw64/libexec/git-core`.
        let exec = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        if let Some(git) = exec.ancestors().nth(3) {
            let bash = git.join("bin/bash.exe");
            if bash.is_file() {
                return Command::new(bash);
            }
        }
    }
    Command::new("bash")
}

fn wasm_installed() -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .stderr(Stdio::null())
        .output()
        .is_ok_and(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .any(|l| l.trim() == WASM)
        })
}

/// Packages with files changed since the branch left `origin/main` (or `main`), committed or not.
/// A change to the workspace manifest, the lock file or `.cargo/` counts as the core crates.
fn changed_packages(root: &Path) -> Result<Vec<String>, String> {
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let base = ["origin/main", "main"]
        .into_iter()
        .find_map(|b| git(&["merge-base", "HEAD", b]))
        .unwrap_or_default();
    let mut files = String::new();
    if !base.trim().is_empty() {
        files += &git(&["diff", "--name-only", base.trim()]).unwrap_or_default();
    }
    files += &git(&["ls-files", "--others", "--exclude-standard"]).unwrap_or_default();

    let out = Command::new(env_or("CARGO", "cargo"))
        .current_dir(root)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("cargo metadata: {e}"))?;
    let root = Path::new(json["workspace_root"].as_str().unwrap_or("."));
    let dirs: Vec<(PathBuf, String)> = json["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| {
            let manifest = Path::new(p["manifest_path"].as_str()?);
            let dir = manifest.parent()?.strip_prefix(root).ok()?.to_path_buf();
            Some((dir, p["name"].as_str()?.to_owned()))
        })
        .collect();
    let mut names = Vec::new();
    for file in files.lines().map(Path::new) {
        let owner = dirs
            .iter()
            .filter(|(dir, _)| file.starts_with(dir))
            .max_by_key(|(dir, _)| dir.components().count());
        match owner {
            Some((_, name)) => names.push(name.clone()),
            None if file == Path::new("Cargo.toml")
                || file == Path::new("Cargo.lock")
                || file.starts_with(".cargo") =>
            {
                names.extend(CORE.iter().map(|s| s.to_string()))
            }
            None => {}
        }
    }
    names.sort();
    names.dedup();
    Ok(names)
}
