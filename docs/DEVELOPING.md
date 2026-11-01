# Developing with vfx-core

An effect is one crate: it implements vfx-core's `Effect` (GPU, on [wgpu](https://wgpu.rs)) or
`CpuEffect` (frames in RAM, with earlier frames if it wants them), declares its parameters, and
gives its identity in its `Cargo.toml`. The host targets turn it into plugins, and `cargo vfx`
builds and packages them:

| Host target | Crate | Hosts | Checked with the generic target |
|---|---|---|---|
| After Effects | `vfx-core/wrappers/adobe` (`vfx-ae`) | After Effects 2026 (Premiere untested) | builds and unit tests; checked in After Effects with the earlier per-effect wrappers |
| OpenFX | `vfx-core/wrappers/ofx` (`vfx-ofx`) | DaVinci Resolve, Nuke, Natron… | both effects pass their mini-host checks; Resolve 21 and Natron 2.5 were checked with the earlier wrappers |
| FFGL | `vfx-core/wrappers/ffgl` (`vfx-ffgl`) | Resolume Avenue / Arena | both effects pass the mini host (GPU sharing, RAM and software GL); Avenue was checked with the earlier wrappers |
| Preview app | `vfx-core/examples/preview` (`vfx-preview`) | desktop and browser (WebGPU) | shows any one effect, with expressions and MP4 export; the web build deploys to [GitHub Pages](https://jamecoyne.github.io/vfx-core/) |

## An effect crate

`cargo vfx new my-glow` makes one in `vfx-core/effects/my-glow`: a CPU effect to start from,
already in the workspace and the preview. `plugins/segcam/segcam-motion` and
`vfx-core/effects/vfx-chain` are
complete examples. The parts:

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

# optional: the crate's own features `cargo vfx` turns on per OS (all, macos, windows, linux)
features = { windows = ["onnx"], linux = ["onnx"] }

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
them once and keep them. The build script checks them against each host's limits (lengths,
characters, a GUID), and every `cargo vfx` command checks that no two effects share one. A parameter's `name` is its saved identifier too (After Effects'
disk IDs are hashes of it); its `label` is what hosts show and can change.

The target crates read the effect's `ParamSpec`s and map them onto each host: groups,
sliders with soft ranges, menus, checkboxes, colours (split into red, green and blue for
FFGL), points (in each host's coordinates), and the effect's second layer as a clip or
layer picker. Parameters that need a second layer or a growing output are left out where
a host has neither. An effect's `region` tells hosts how far it reads past each pixel and
whether its output grows; `Context` gives it the time, frame number and render scale.
`vfx_host::controls::layout` decides the layout for every target, so it is tested once.

A `CpuEffect` that keeps work between renders says so with `keeps_state`. OpenFX, FFGL
and the preview then keep one per instance (`vfx_host::CpuInstance`, which also starts it
over after a panic, a seek back or a resize); After Effects never does. Either way every
frame must match a fresh effect's, even after earlier frames change upstream or a host
misses one: test it with `vfx_host::contract::check`, as `plugins/segcam/segcam-motion` does.

## Layout

One Cargo workspace (`Cargo.toml` at the root) in two folders: `vfx-core/`, the toolchain,
and `plugins/`, effects built with it. Nothing in `vfx-core/` depends on `plugins/` except the
preview's list of effects.

```
vfx-core/
  src/                  vfx-core: Effect and CpuEffect, typed params, Frame, EffectChain, WGSL passes
    cpu.rs              frames in RAM for CPU effects: ARGB at 8/16/32 bits, strided views, history
    effects/            the core effects
  tests/                headless GPU tests for the core
  effects/
    vfx-chain/          VFX Preview, the chain of core effects: the demo effect
      examples/ae_reference.rs   its After Effects render path without After Effects
  wrappers/
    common/             vfx-host: the registered plugin, the GPU, the render pool, the log,
                        control layout, kept CPU effects and their contract
    plugin/             vfx-plugin: an effect crate's one `export!`
    build/              vfx-build: the effect crate's build script (identity and its checks, PiPL)
    adobe/              vfx-ae: After Effects SmartFX, Metal interop
      ae-tests/         harness that renders VFX Preview inside After Effects
    ofx/                vfx-ofx: OpenFX, hand-written bindings to the OpenFX 1.4 C API
      minihost/         a small OpenFX host in C for testing without an application
    ffgl/               vfx-ffgl: FFGL 2.1, OpenGL + IOSurface / Vulkan interop
      minihost/         a small FFGL host in C with a headless OpenGL context
    ffgl-common/        FFGL bindings, GL transfers and pixel repacking
    package-win.sh      an effect's Windows installer (Inno Setup)
  tools/cargo-vfx/      cargo vfx: builds, packages and installs effect crates
  examples/preview/     the preview: a library host (`vfx_preview::run`) and the repository's app
    src/catalog.rs      the effects it offers (effect crates listed by build.rs)
    web/                index.html and build.sh for the WebGPU build
plugins/
  segcam/
    segcam-motion/      segcam Motion, ported from C++ (own README)
      ae-tests/ ofx-tests/ ffgl-tests/   host checks against the C++ effect and its CLI
    segcam-face/        faces, eyes, mouths and bodies for segcam (own README)
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
| A new effect | `cargo vfx new my-glow` | `vfx-core/effects/my-glow`, added to the workspace and the preview |
| Checks | `cargo vfx check` | effects missing from the preview, and installed After Effects plugins that share a match name; `--quick` and `--full` run the test tiers ([Testing](#testing)) |
| Plugins | `cargo vfx build [-p segcam-motion] [--hosts ae,ofx,ffgl] [--native]` | `<file>.plugin`, `.ofx.bundle` and `.bundle` on macOS (one universal binary; `--native` builds this Mac's architecture only); `.aex`, `.ofx.bundle` and `.dll` on Windows |
| macOS installers | `cargo vfx package` | `<file>-adobe.pkg` (Adobe's MediaCore folder) and `<file>-ofx.pkg` (`/Library/OFX/Plugins`) |
| Install | `cargo vfx install [--hosts ffgl]` | copies FFGL bundles into Resolume's Extra Effects; for After Effects and OpenFX, prints the `sudo installer` commands |
| Windows installer | `EFFECT=segcam-motion vfx-core/wrappers/package-win.sh` | `<file>-<version>-Windows-Setup.exe` and a `.zip` (needs [Inno Setup](https://jrsoftware.org/isinfo.php)) |
| Preview (desktop) | `cargo vfx preview [-p segcam-motion] [-- clip.mp4]` | the preview, opened on the effect; drop an image or video on it |
| Preview (web) | `vfx-core/examples/preview/web/build.sh --serve` | http://localhost:8123, or `?effect=segcam-motion` to open on an effect (needs `wasm-bindgen-cli` at the lockfile's version) |
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
desktop and in the browser. Its build script lists every effect crate among its
dependencies, as `Entry::plugin(&segcam_motion::VFX_PLUGIN)` (the crate as its plugins see
it); `vfx-core/examples/preview/src/catalog.rs` adds the core effects.

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

Two tiers run the checks locally, each step logged to `target/checks/<step>.log`:

| Tier | Command | Runs |
|---|---|---|
| Quick, while iterating | `cargo vfx check --quick [-p PACKAGE]...` | format, the core crates' clippy, and the tests of the packages changed since `origin/main` (committed or not; `-p` picks others). A change to `Cargo.toml`, `Cargo.lock` or `.cargo/` tests the core crates |
| Full, before a pull request | `cargo vfx check --full` | CI's steps for this OS, the OpenFX and FFGL mini hosts with segcam's full matrices, and the preview's clippy, tests and wasm check |

The full tier needs what the harnesses need: `OFX_INCLUDE` (Resolve's headers are the default on
macOS) and, on Windows, Git for Windows (its bash runs the scripts) and `CC` (default `clang`).
`segcam-motion`'s harnesses also take `--quick`, a smaller matrix for iterating on the plugin
targets; the full tier runs the whole one.

The test profile optimises `segcam-motion` and `vfx-core`, so their pixel tests take seconds;
overflow checks and debug assertions stay on. Several clones or worktrees on one machine can
share compiled dependencies through [sccache](https://github.com/mozilla/sccache):
`cargo install sccache`, then set `RUSTC_WRAPPER=sccache` in your shell (not in the repository's
config); the tiers use it when it is set. On Windows it can't build `web-sys` for wasm (the command line gets too long), so the full tier
runs the wasm check without it; unset `RUSTC_WRAPPER` for the web build too.

Unit tests cover the core, each effect, and each target's parameter mapping and pixel
conversions. Each host also has a harness that runs the effects:

- After Effects. `python3 vfx-core/wrappers/adobe/ae-tests/run.py` builds fixture projects for
  VFX Preview in the running After Effects (8, 16 and 32 bpc, CPU and Metal), renders them with
  `aerender` with MFR on and off, and compares them with `ae_reference`, the same render path
  without After Effects. `python3 plugins/segcam/segcam-motion/ae-tests/run.py` renders segcam
  next to segcam's C++ effect. Both test the *installed* plugins and need After Effects, so they
  aren't in CI.
- OpenFX. `vfx-core/wrappers/ofx/minihost/run.sh` loads VFX Preview into a small C host built on
  the real OpenFX headers and checks pixels, depths, row order, premultiplication, render scale,
  and regions of definition and interest. `plugins/segcam/segcam-motion/ofx-tests/run.sh` renders
  segcam through the same host at every depth, row order and alpha convention, and compares 36
  sequences with its CLI. Both run in CI on macOS and Windows (from Git Bash, with `CC` and
  `OFX_INCLUDE` set).
- FFGL. `vfx-core/wrappers/ffgl/minihost/run.sh` does the same for FFGL with a headless OpenGL
  context, checking metadata, pixels, padded textures, the GL state left behind, and frame times.
  `EFFECT=<crate>` checks another effect. `plugins/segcam/segcam-motion/ffgl-tests/run.sh` adds
  segcam's cases and compares 960 frames with its CLI. `FFGL_MINIHOST_SOFTWARE=1` uses Apple's
  software renderer, as the macOS CI runners do, and `VFX_FFGL_PATH=ram` forces the RAM path.

## Design notes

- Every host's entry points live in one binary per effect. `cargo vfx` builds an effect crate
  once per architecture with the host features on, then copies the binary into each host's
  bundle layout. Each export macro registers the crate's `VFX_PLUGIN` static, which the target
  crates read.
- GPU effects get one device per plugin, with instances pooled. Building pipelines is the
  expensive part of a render, so prepared instances are kept per device and size and lent out
  per render. Concurrent renders (After Effects MFR, OpenFX hosts) each borrow their own.
- CPU effects read host frames where they are. After Effects' buffers are read at their own
  depth without copies. OpenFX frames are converted one at a time. Resolume and the preview hand
  over one frame at a time, so they keep a history of recent frames.
- Plugins never use wgpu's OpenGL backend. Headless devices use Vulkan, Metal or DX12, because
  the OpenGL backend releases whatever context is current when it starts, and on Windows that's
  the FFGL host's. The preview creates its own instance.
- Frames stay on the GPU where a host allows it. After Effects with Metal acceleration hands over
  `MTLBuffer`s, and the plugin opens wgpu on After Effects' own device and wraps them. Resolume's
  GL textures are drawn into IOSurfaces that wgpu's Metal device shares, or into Vulkan images
  shared with OpenGL on Windows. Everything else, and any setup failure, goes through RAM. If the
  GPU fails, the frame passes through unchanged instead of failing the host's render.
- The OpenFX and FFGL bindings are written by hand. Both are small C ABIs, transcribed from their
  BSD licensed headers; FFGL's notice is kept in `vfx-core/wrappers/ffgl-common/src/ffi.rs`. The
  `ffgl-core` crate is GPL 3.0, so it isn't used.
- Testing found these host quirks, which are now handled and covered by tests or the harnesses.
  After Effects needs PiPL spec version 13.25 or later for MFR, and sequence data flattening
  flags for saved projects. It trims an effect's input to the mask bounds. Resolve 21 rejects the
  OpenFX plugin description property. FFGL textures are premultiplied and bottom up.

## Known gaps

- After Effects on Windows: the `.aex` builds in CI but hasn't been tried in After Effects.
- After Effects 26.5 on macOS passes both effects' harnesses. Natron was last checked with the
  earlier per effect wrappers.
- Premiere Pro is untested. Signing and notarization have never run.
- OpenFX GPU rendering (Resolve's Metal extension) and an FFGL thumbnail aren't done.
- Hosts with no target yet: TouchDesigner (a Custom TOP), and EffectCraft, whose only plugin API
  is CPU WebAssembly, a fit for CPU effects.
