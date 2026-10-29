# vfx-core

A toolchain for video effects written once in Rust and shipped to many hosts. An effect
is one crate: it implements vfx-core's `Effect` (GPU, on [wgpu](https://wgpu.rs)) or
`CpuEffect` (frames in RAM, with earlier frames if it wants them), declares its
parameters, and gives its identity in its `Cargo.toml`. The host targets turn it into
plugins, and `cargo vfx` builds and packages them:

| Host target | Crate | Hosts | Checked with the generic target |
|---|---|---|---|
| After Effects | `wrappers/adobe` (`vfx-ae`) | After Effects 2026 (Premiere untested) | builds and unit tests; checked in After Effects with the earlier per-effect wrappers |
| OpenFX | `wrappers/ofx` (`vfx-ofx`) | DaVinci Resolve, Nuke, Natron… | both effects pass their mini-host checks; Resolve 21 and Natron 2.5 were checked with the earlier wrappers |
| FFGL | `wrappers/ffgl` (`vfx-ffgl`) | Resolume Avenue / Arena | both effects pass the mini host (GPU sharing, RAM and software GL); Avenue was checked with the earlier wrappers |
| Preview app | `examples/preview` (`vfx-preview`) | desktop and browser (WebGPU) | shows any one effect, with expressions and MP4 export; the web build deploys to [GitHub Pages](https://jamecoyne.github.io/vfx-core/) |

The effects in this repository:

| Effect | Crate | What |
|---|---|---|
| VFX Preview | `effects/vfx-chain` | colour adjust, vignette, a blend with a second layer and Gaussian blur, chained, on the GPU |
| segcam Motion (Rust) | `effects/segcam-motion` | segcam's motion segmentation on the CPU: overlay, mattes and the swarm collage; byte-identical to segcam's C++ effect |
| core effects | `vfx-core` (`src/effects`) | colour adjust, blur, vignette, blend, flip, feedback trail: building blocks, and tryable alone in the preview |

Licensed MIT OR Apache-2.0.

## An effect crate

`effects/segcam-motion` and `effects/vfx-chain` are complete examples. The parts:

```toml
# Cargo.toml
[package.metadata.vfx]
name = "VFX Preview"                # in each host's effect list
short-name = "VFX Preview"          # optional: where only 16 bytes fit (FFGL)
category = "vfx-core"               # After Effects category and OpenFX grouping
file = "VfxPreview"                 # bundles, DLLs, installers and logs (no spaces)
id = "com.vfx-core.VfxPreview"      # OpenFX identifier; macOS bundle IDs start with it
ae-match-name = "VFXPreview"
ffgl-code = "VFXP"                  # exactly 4 characters
processor = "gpu"                   # "gpu" (Effect) or "cpu" (CpuEffect)
windows-installer-id = "…a GUID…"

[features]
ae = ["vfx-plugin/ae"]
ofx = ["vfx-plugin/ofx"]
ffgl = ["vfx-plugin/ffgl"]

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
vfx-core = { path = "../.." }
vfx-plugin = { path = "../../wrappers/plugin" }

[build-dependencies]
vfx-build = { path = "../../wrappers/build" }
```

```rust
// build.rs
fn main() {
    vfx_build::configure();
}

// lib.rs
vfx_plugin::export!(gpu: VfxPreview);    // or `cpu: SegcamMotion`
```

Hosts find a plugin in saved projects by `id`, `ae-match-name` and `ffgl-code`, so pick
them once and keep them. A parameter's `name` is its saved identifier too (After Effects'
disk IDs are hashes of it); its `label` is what hosts show and can change.

The target crates read the effect's `ParamSpec`s and map them onto each host: groups,
sliders with soft ranges, menus, checkboxes, colours (split into red, green and blue for
FFGL), points (in each host's coordinates), and the effect's second layer as a clip or
layer picker. Parameters that need a second layer or a growing output are left out where
a host has neither. An effect's `region` tells hosts how far it reads past each pixel and
whether its output grows; `Context` gives it the time, frame number and render scale.

## Layout

```
src/                    vfx-core: Effect and CpuEffect, typed params, Frame, EffectChain, WGSL passes
  cpu.rs                frames in RAM for CPU effects: ARGB at 8/16/32 bits, strided views, history
  effects/              the core effects
tests/                  headless GPU tests for the core
effects/
  vfx-chain/            VFX Preview, the chain of core effects
    examples/ae_reference.rs   its After Effects render path without After Effects
  segcam-motion/        segcam Motion, ported from C++ (own README)
    ae-tests/ ofx-tests/ ffgl-tests/   host checks against the C++ effect and its CLI
wrappers/
  common/               vfx-host: the registered plugin, the GPU, the render pool, the log
  plugin/               vfx-plugin: an effect crate's one `export!`
  build/                vfx-build: the effect crate's build script (identity, PiPL)
  adobe/                vfx-ae: After Effects SmartFX, Metal interop
    ae-tests/           harness that renders VFX Preview inside After Effects
  ofx/                  vfx-ofx: OpenFX, hand-written bindings to the OpenFX 1.4 C API
    minihost/           a small OpenFX host in C for testing without an application
  ffgl/                 vfx-ffgl: FFGL 2.1, OpenGL + IOSurface / Vulkan interop
    minihost/           a small FFGL host in C with a headless OpenGL context
  ffgl-common/          FFGL bindings, GL transfers and pixel repacking
  package-win.sh        an effect's Windows installer (Inno Setup)
tools/cargo-vfx/        cargo vfx: builds, packages and installs effect crates
examples/preview/       the preview: a library host (`vfx_preview::run`) and the repository's app
  src/catalog.rs        the effects it offers
  web/                  index.html and build.sh for the WebGPU build
.github/workflows/      ci.yml (tests, every plugin), pages.yml (web preview), release.yml
```

## Building and running

Needs a current stable Rust (egui needs 1.95+; CI uses latest stable). macOS bundles are
universal, so add both targets once:

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin wasm32-unknown-unknown
```

`cargo vfx` runs from the repository (an alias in `.cargo/config.toml`). Without `-p`, it
acts on every effect crate; output goes to `target/vfx-plugin/`.

| What | Command | Result |
|---|---|---|
| The effects | `cargo vfx list` | each effect crate and its identity |
| Plugins | `cargo vfx build [-p segcam-motion] [--hosts ae,ofx,ffgl] [--native]` | `<file>.plugin`, `.ofx.bundle` and `.bundle` on macOS (one universal binary; `--native` builds this Mac's architecture only); `.aex`, `.ofx.bundle` and `.dll` on Windows |
| macOS installers | `cargo vfx package` | `<file>-adobe.pkg` (Adobe's MediaCore folder) and `<file>-ofx.pkg` (`/Library/OFX/Plugins`) |
| Install | `cargo vfx install [--hosts ffgl]` | copies FFGL bundles into Resolume's Extra Effects; for After Effects and OpenFX, prints the `sudo installer` commands |
| Windows installer | `EFFECT=segcam-motion wrappers/package-win.sh` | `<file>-<version>-Windows-Setup.exe` and a `.zip` (needs [Inno Setup](https://jrsoftware.org/isinfo.php)) |
| Preview (desktop) | `cargo vfx preview [-p segcam-motion] [-- clip.mp4]` | the preview, opened on the effect; drop an image or video on it |
| Preview (web) | `examples/preview/web/build.sh --serve` | http://localhost:8123, or `?effect=segcam-motion` to open on an effect (needs `wasm-bindgen-cli` at the lockfile's version) |
| Tests | `cargo test -p vfx-core -p vfx-host -p vfx-chain -p segcam-motion` | headless GPU and CPU tests |

`cargo vfx package` signs and notarizes only when `SIGN_IDENTITY`, `NOTARY_PROFILE` and
`INSTALLER_IDENTITY` are set; without them the packages are unsigned (fine locally).
Each plugin writes one-line notes about which render path ran to
`~/Library/Logs/<file>-AfterEffects.log`, `<file>-OpenFX.log` or `<file>-FFGL.log`
(on Windows, the same names in `%TEMP%`).

## The preview

The preview hosts one effect at a time: test patterns, images, video files or a webcam
go in, a panel made from the effect's parameters drives it, every float and integer
parameter has an expression box (`sin(time/4)`, `bass * 2`, face tracking), and a video
renders to MP4 through it. It links the effects it offers, so the same build runs on the
desktop and in the browser. `examples/preview/src/catalog.rs` lists the repository's:

```rust
Entry::plugin(&segcam_motion::VFX_PLUGIN)     // an effect crate, as its plugins see it
```

Another crate's effect can use the preview as a library:
`vfx_preview::run(vec![Entry::plugin(&my_effect::VFX_PLUGIN)])`.

## Releases

`.github/workflows/release.yml` builds every effect's plugins on GitHub's macOS and
Windows runners. Push a version tag and it makes a draft release with the files attached;
check it on GitHub, then publish it:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

Running the workflow by hand (Actions → release → Run workflow) builds the same files
as the run's artifacts, without a release. For each effect:

| File | What |
|---|---|
| `<file>-<version>-macOS-AfterEffects.pkg` | After Effects and Premiere Pro installer |
| `<file>-<version>-macOS-OpenFX.pkg` | OpenFX installer (Resolve, Nuke, Natron) |
| `<file>-<version>-macOS-Resolume.zip` | the FFGL bundle, for Extra Effects |
| `<file>-<version>-Windows-Setup.exe` | Windows installer: OpenFX and Resolume, After Effects optional |
| `<file>-<version>-Windows.zip` | the same Windows files, to copy by hand |

macOS signing and notarization run when these repository secrets are set (Settings →
Secrets and variables → Actions); without them the packages are unsigned:

- `MACOS_CERTIFICATES_P12`: a base64 `.p12` holding the Developer ID Application and
  Developer ID Installer certificates, and `MACOS_CERTIFICATES_PASSWORD`, its password.
- `APPLE_ID`, `APPLE_TEAM_ID` and `APPLE_APP_PASSWORD` (an app-specific password), for
  notarization.

The Windows installer is unsigned, so Windows SmartScreen asks before running it.

## Testing

Unit tests cover the core, each effect, and each target's parameter mapping and pixel
conversions. Beyond those, each host has a harness, run against the effects:

- **After Effects**: `python3 wrappers/adobe/ae-tests/run.py` builds fixture projects
  for VFX Preview in the running After Effects (8, 16, 32 bpc; CPU and Metal), renders
  them with `aerender` (MFR on and off), and compares against `ae_reference`, the same
  render path without After Effects. `python3 effects/segcam-motion/ae-tests/run.py`
  renders segcam beside segcam's C++ effect. Both test the *installed* plugins and need
  After Effects, so they aren't in CI.
- **OpenFX**: `wrappers/ofx/minihost/run.sh` loads VFX Preview into a C host built on the
  real OpenFX headers and checks pixels, depths, row order, premultiplication, render
  scale, regions of definition and interest. `effects/segcam-motion/ofx-tests/run.sh`
  renders segcam through the same host's suites at every depth, row order and alpha
  convention, and compares 36 sequences with its CLI. Both run in CI on macOS and
  Windows (from Git Bash with `CC` and `OFX_INCLUDE` set).
- **FFGL**: `wrappers/ffgl/minihost/run.sh` does the same for FFGL with a headless OpenGL
  context: metadata, pixels, padded textures, GL state left behind, and frame times.
  `EFFECT=<crate>` checks another effect; `effects/segcam-motion/ffgl-tests/run.sh` adds
  segcam's cases and compares 960 frames with its CLI. `FFGL_MINIHOST_SOFTWARE=1` uses
  Apple's software renderer, as on macOS CI runners; `VFX_FFGL_PATH=ram` forces the RAM
  path.

## Design notes

- **One binary per effect, every host's entry points in it.** `cargo vfx` builds an
  effect crate once per architecture with the host features on, then copies the binary
  into each host's bundle layout. Each export macro registers the crate's `VFX_PLUGIN`
  static, which the target crates read.
- **GPU effects: one device per plugin, instances pooled.** Building pipelines is the
  expensive part of a render, so prepared instances are kept per (device, size) and
  leased per render. Concurrent renders (After Effects MFR, OpenFX hosts) each lease
  their own.
- **CPU effects: host frames in place.** After Effects' buffers are read at their own
  depth without copies; OpenFX frames are converted one at a time; Resolume and the
  preview keep a history of recent frames, since they hand over one at a time.
- **No OpenGL backend for plugins.** Headless devices use Vulkan, Metal or DX12. wgpu's
  OpenGL backend releases whatever context is current when it starts, which on Windows
  is the FFGL host's. The preview creates its own instance.
- **GPU-native paths where hosts allow it.** After Effects with Metal acceleration hands
  over `MTLBuffer`s: the plugin opens wgpu on After Effects' own device and wraps them.
  Resolume's GL textures are drawn into IOSurfaces that wgpu's Metal device shares, or
  Vulkan images shared with OpenGL on Windows. These paths avoid any CPU copy; everything
  else (and any setup failure) goes through RAM, and a GPU failure passes the frame
  through unchanged rather than failing the host's render.
- **Hand-written host bindings.** OpenFX and FFGL are small C ABIs, transcribed from
  their headers (BSD-licensed; FFGL's notice is kept in `wrappers/ffgl-common/src/ffi.rs`).
  The `ffgl-core` crate was avoided because it's GPL-3.0.
- **Host quirks found by testing**, now handled and covered by tests or the harnesses:
  After Effects needs PiPL spec version 13.25+ for MFR and sequence-data flattening flags
  for saved projects; it trims an effect's input to mask bounds; Resolve 21 rejects the
  OpenFX plugin-description property; FFGL textures are premultiplied and bottom-up.

## Known gaps

- The generic target crates have been checked in the OpenFX and FFGL mini hosts, by unit
  tests and, for After Effects 26.5 on macOS, by both effects' harnesses. Resolve, Natron
  and Resolume were last checked with the earlier per-effect wrappers.
- Windows: the `.aex` builds in CI but hasn't been tested in After Effects; OpenFX hasn't
  been tried in Resolve on Windows; FFGL's Windows paths pass the mini host on an RX 5700
  XT but haven't been tried in Resolume on Windows with VFX Preview.
- Premiere Pro is untested. Signing and notarization have never been run.
- OpenFX GPU rendering (Resolve's Metal extension) and an FFGL thumbnail aren't done.
- Hosts with no target yet: TouchDesigner (a Custom TOP), and EffectCraft, whose only
  plugin API is CPU WebAssembly (a fit for CPU effects).

## Notes for agents working on this repo

- Effect logic belongs in an effect crate (`effects/`) or the core (`src/effects`). A
  target crate (`wrappers/`) only translates parameters and moves pixels, for any effect;
  what two targets share belongs in `wrappers/common`.
- Read the README in each target or effect (`wrappers/ffgl/`, `wrappers/adobe/ae-tests/`,
  `effects/segcam-motion/`) before changing it, and run its harness afterwards. The After
  Effects harnesses test the *installed* plugins, so `cargo vfx package` and reinstall
  first (installing to system folders needs the user's password).
- Keep labels under 16 bytes: FFGL hosts cut longer ones, and each effect crate has a test.
  Never rename a parameter's `name` once shipped; change its `label` instead.
- To make an effect tryable in the preview, add one entry to `examples/preview/src/catalog.rs`.
- `cargo fmt --all -- --check` and clippy with `-D warnings` (see `ci.yml` for the crates
  and features) must pass; CI runs latest stable, whose clippy can be stricter.
- wgpu is pinned to `=30.0.1`; the Metal interop uses `wgpu-hal` types and must be checked
  when it moves. The web build needs wgpu's `fragile-send-sync-non-atomic-wasm` feature
  (set in `examples/preview/Cargo.toml`) because effects are `Send`.
