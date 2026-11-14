//! The one error type of `cargo vfx`. `Display` is what the tool prints after `cargo vfx: `.

use std::io;
use std::path::PathBuf;
use std::process::ExitStatus;

use crate::{USAGE, bench};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{usage}", usage = USAGE)]
    Usage,
    #[error("unknown argument `{0}`\n{usage}", usage = USAGE)]
    UnknownArgument(String),
    #[error("unknown command `{0}`\n{usage}", usage = USAGE)]
    UnknownCommand(String),
    #[error("{flag} needs {what}")]
    MissingValue {
        flag: &'static str,
        what: &'static str,
    },
    #[error("unknown host `{0}` (ae, ofx or ffgl)")]
    UnknownHost(String),
    #[error("usage: cargo vfx new NAME")]
    NewUsage,
    #[error("env takes one -p EFFECT")]
    EnvNeedsOneEffect,
    #[error("no effect crate `{name}` (effects: {available})")]
    UnknownEffect { name: String, available: String },
    #[error("{0} problems")]
    Problems(usize),
    #[error("failed: {}", .0.join(", "))]
    StepsFailed(Vec<String>),
    #[error("--full checks every package; -p goes with --quick")]
    FullWithPackages,

    #[error("plugins build on macOS and Windows")]
    UnsupportedOs,
    #[error("package builds macOS installers; on Windows use vfx-core/wrappers/package-win.sh")]
    PackageOnWindows,
    #[error("NOTARY_PROFILE needs SIGN_IDENTITY: Apple only notarizes signed code")]
    NotaryNeedsSignIdentity,
    #[error("NOTARY_PROFILE needs INSTALLER_IDENTITY: Apple only notarizes signed installers")]
    NotaryNeedsInstallerIdentity,
    #[error("no home directory")]
    NoHome,
    #[error("{0} has no build script")]
    NoBuildScript(String),
    #[error("no {}", .0.display())]
    MissingFile(PathBuf),
    #[error("{}: the DXC download doesn't match its sha256", .0.display())]
    DxcChecksum(PathBuf),

    #[error(transparent)]
    Metadata(#[from] vfx_build::MetadataError),
    #[error("{}", .0.join("\n"))]
    Identities(Vec<String>),
    #[error("cargo metadata: {0}")]
    CargoMetadataRun(#[source] io::Error),
    #[error("cargo metadata: {0}")]
    CargoMetadataJson(#[source] serde_json::Error),
    #[error("cargo metadata: no packages")]
    CargoMetadataNoPackages,
    #[error("{0}")]
    CargoMetadataFailed(String),

    #[error("{command}: {source}")]
    Spawn {
        command: String,
        #[source]
        source: io::Error,
    },
    #[error("{command} failed ({status})")]
    CommandFailed { command: String, status: ExitStatus },
    #[error("{}: {source}", path.display())]
    File {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error("`{0}`: name the crate in lower case, digits and hyphens, like my-glow")]
    InvalidCrateName(String),
    #[error("{} already exists", .0.display())]
    AlreadyExists(PathBuf),
    #[error("the workspace Cargo.toml has no `members = [`")]
    NoMembersList,
    #[error("unclosed members list")]
    UnclosedMembersList,
    #[error("the preview's Cargo.toml lists no effect crate")]
    NoPreviewEffects,

    #[error("{usage}", usage = bench::USAGE)]
    BenchUsage,
    #[error("{0} needs a value\n{usage}", usage = bench::USAGE)]
    BenchMissingValue(String),
    #[error("unknown preset `{0}`\n{usage}", usage = bench::USAGE)]
    UnknownPreset(String),
    #[error("unknown depth `{0}`\n{usage}", usage = bench::USAGE)]
    UnknownDepth(String),
    #[error("--runs needs a count\n{usage}", usage = bench::USAGE)]
    InvalidRuns,
    #[error("unknown argument `{0}`\n{usage}", usage = bench::USAGE)]
    UnknownBenchArgument(String),
    #[error("{preset}: no face models, so this row would time no faces:\n{log}")]
    NoFaceModels { preset: &'static str, log: String },
    #[error("{preset}: the CLI named no path:\n{log}")]
    NoPathNamed { preset: &'static str, log: String },
    #[error("the CLI has no {0}")]
    CliPipeMissing(&'static str),
    #[error("reading the CLI's frames: {0}")]
    CliFrames(#[source] io::Error),
    #[error("{command} failed ({status}):\n{log}")]
    CliFailed {
        command: String,
        status: ExitStatus,
        log: String,
    },
    #[error("{}: {source}", path.display())]
    Reference {
        path: PathBuf,
        #[source]
        source: bench::HeaderMismatch,
    },
    #[error("ffprobe: {0} (the bench needs ffmpeg on the PATH)")]
    FfprobeSpawn(#[source] io::Error),
    #[error("ffprobe {clip}: {stderr}")]
    FfprobeFailed { clip: String, stderr: String },
    #[error("ffprobe: {0}")]
    FfprobeJson(#[source] serde_json::Error),
    #[error("{0}: no video stream")]
    NoVideoStream(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn file(path: impl Into<PathBuf>) -> impl FnOnce(io::Error) -> Self {
        let path = path.into();
        move |source| Self::File { path, source }
    }
}
