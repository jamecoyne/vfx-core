# Developing with vfx-core

An effect is one crate. It implements vfx-core's `Effect` (GPU, on [wgpu](https://wgpu.rs)) or
`CpuEffect` (frames in RAM, with earlier frames if it wants them). It declares its parameters and
gives its identity in its `Cargo.toml`. The host targets turn it into plugins. `cargo vfx` builds and
packages them:

| Host target | Crate | Hosts | Checked with the generic target |
|---|---|---|---|
| After Effects | `vfx-core/wrappers/adobe` (`vfx-ae`) | After Effects 2026 (Premiere untested) | builds and unit tests; checked in After Effects with the earlier per-effect wrappers |
| OpenFX | `vfx-core/wrappers/ofx` (`vfx-ofx`) | DaVinci Resolve, Nuke, Natron… | VFX Preview (`plugins/vfx-chain`) and segcam Motion pass their mini host checks; Resolve 21 and Natron 2.5 were checked with the earlier wrappers |
| FFGL | `vfx-core/wrappers/ffgl` (`vfx-ffgl`) | Resolume Avenue / Arena | VFX Preview and segcam Motion pass the mini host (GPU sharing, RAM and software GL); Avenue was checked with the earlier wrappers |
| Preview app | `vfx-core/hosts/preview` (`vfx-preview`) | desktop and browser (WebGPU, or WebGL2 without it) | shows any one effect, with expressions and MP4 export; the web build deploys to [GitHub Pages](https://jamecoyne.github.io/vfx-core/) |

## An effect crate

`cargo vfx new my-glow` makes one in `plugins/my-glow`. It is a CPU effect to start from, and it is
already in the workspace and the preview. `plugins/segcam/segcam-motion` and `plugins/vfx-chain` are
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
windows-installer-id = "…a GUID…"  # the Windows installer's AppId, when this effect leads it

# optional: the crate's own features `cargo vfx` turns on per OS (all, macos, windows, linux)
features = { windows = ["onnx"], linux = ["onnx"] }

[features]
ae = ["vfx-plugin/ae"]
ofx = ["vfx-plugin/ofx"]
ffgl = ["vfx-plugin/ffgl"]

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
vfx-core = { path = "../../vfx-core" }
vfx-plugin = { path = "../../vfx-core/wrappers/plugin" }

[build-dependencies]
vfx-build = { path = "../../vfx-core/wrappers/build" }
```

```rust
// build.rs
fn main() {
    vfx_build::configure();
}

// lib.rs
vfx_plugin::export!(gpu: VfxPreview);    // or `cpu: SegcamMotion`
```

Hosts find a plugin in saved projects by `id`, `ae-match-name` and `ffgl-code`. Pick them once and
keep them. The build script checks them against each host's limits: lengths, characters and a GUID.
Every `cargo vfx` command checks that no two effects share one.

A parameter's `name` is also its saved identifier: After Effects' disk IDs are hashes of it. Its
`label` is what hosts show, and it can change.

The target crates read the effect's `ParamSpec`s and map them onto each host. That covers groups,
sliders with soft ranges, menus, checkboxes, colours, points and the effect's second layer. Colours
split into red, green and blue for FFGL. Points use each host's coordinates. The second layer
becomes a clip or layer picker. Parameters that need a second layer or a growing output are left out
on a host that has neither.

An effect's `region` tells hosts how far it reads past each pixel, and whether its output grows.
`Context` gives it the time, the frame number and the render scale.
`vfx_host::controls::layout` decides the layout for every target, so it is tested once.

A `CpuEffect` with slow setup can do that work in `values_known`. OpenFX and FFGL call it before the
first render.

A `CpuEffect` that keeps work between renders returns `true` from `keeps_state`. OpenFX, FFGL and
the preview then keep one per instance (`vfx_host::CpuInstance`). `CpuInstance` also rebuilds the effect
after a panic. After Effects never keeps one.

When a host's frames stop following on, it calls `clear_history`. That happens on a seek back, a clip
loop or a resize. An effect that holds GPU resources implements `clear_history`, so those resources
survive. Without it, the host drops the effect and makes a new one at that point.

Either way, every frame must match a fresh effect's. That holds even after earlier frames change
upstream, or after a host misses one. Test it with `vfx_host::contract::check`.

`frames_held` is how many of the `frames_before` a host that hands frames over one at a time must
hold in full. An effect that keeps state can work from what it kept and return fewer.

`report` returns where a `CpuEffect`'s last render drew, where its face models run, and any named
readings. Implement it when the work runs somewhere a host can't see, or when the effect finds
things a host could use. The preview's bar shows the first two. Its expressions match
readings by name.

### Pipelines per device

A CPU effect that draws on the GPU builds its pipelines and shaders once per device, with
`vfx_core::device::DeviceBuild`. The build runs on its own thread, so a render never waits for it.

```rust
static PIPELINES: LazyLock<DeviceBuild<Pipelines>> =
    LazyLock::new(|| DeviceBuild::new(device::slot(), Pipelines::build));
```

In a render:

- A live render calls `try_wait`. It never blocks. It returns `NoDevice::Opening` while the build
  runs, and the `Arc<Pipelines>` once it is done.
- An export calls `wait`, which blocks until the build is done.
- A build that fails returns `NoDevice::Unusable` with its error. It is not retried on that device.

What an instance keeps on the device, such as textures and buffers, goes in
`PIPELINES.user_with(state, release, on_last)`. `lock` gives access to it.

When the slot loses a device it opened, `DeviceBuild` drops what it built on that device. It also
calls `release` on the state of every instance that isn't rendering right then, and doesn't wait
for those instances to render. D3D12 needs this: it won't open a new device on an adapter until
the removed device is fully released.

On a device the app shares, what was built is dropped when the last instance is.

State that belongs to one render or one frame size goes in vfx-host's `pipeline` pool instead,
which GPU effects already use for theirs. segcam Motion's `gpu.rs` shows `DeviceBuild` in use.

## What a host implements

This is the core API for 1.0. A PR that adds a trait method, a `Context` field or a `Gpu` function
adds a row here, and says so in its description. Every new trait method has a default and `Context::at` fills every new field, so old targets
keep compiling. Once 1.0 is released, nothing here is removed or renamed.

| Item | Where |
|---|---|
| `Effect` (GPU): `params`, `prepare`, `render`, `layer_inputs`, `layer_name`, `render_layers`, `region` | vfx-core |
| `CpuEffect`: `params`, `frames_before` (the frame rate) and `render`. `render` gets its output as a `FrameOut`, which is a texture (`GpuFrame`) or pixels | vfx-core |
| `RenderError`, what `render` returns when it fails: `DeviceLost`, `DeviceOpening`, `OutOfMemory`, `Timeout` (the GPU didn't answer in time; the device is kept), `Gpu` with wgpu's error, and `Internal` for the effect's own errors | vfx-core |
| `CpuEffect::keeps_state` | vfx-core |
| `DeviceFaults::error_count` and `DeviceFaults::last_error` replace `errors`. They give how many errors a device has reported and the most recent one, which `RenderError::Gpu` carries | vfx-core |
| `DeviceFaults`, `Gpu::faults`, `Gpu::shared_faults`, `Gpu::watch_faults` (the error and loss callbacks of a device, which `Gpu::share` takes over; each device has its own record), `poll_device` (a poll that returns device loss as an error) | vfx-core |
| `CpuEffect::clear_history` (forget earlier frames, keeping GPU resources; default `false`, and the host rebuilds the effect) | vfx-core |
| `CpuEffect::frames_held` (how many of the frames before a render a live host holds in full) | vfx-core |
| `CpuEffect::values_known` (the values a host holds, ahead of a render) | vfx-core |
| `RenderError::needs_device`: true when the device was lost or the next one is still opening. Editors use it to wait and render the frame again | vfx-core |
| `FrameOut::with_pixels` and `GpuFrame::write`: an effect that draws in RAM uses them to upload its pixels into a texture output | vfx-core |
| `CpuEffect::report`, `Report`, `Reading` (where the last render's work ran and what it found; default empty) | vfx-core |
| `CpuEffect::device_lost`, `CpuInstance::device_lost` (the host's shared device was lost: the effect lets go of everything on it, so the host can open a new one; default does nothing) | vfx-core |
| `Context`: `time`, `frame`, `scale`, `premultiplied` | vfx-core |
| `Context::frame_rate` | vfx-core |
| `Region`, `Processor`, `EffectChain`, `Frame`, `Readback` | vfx-core |
| `ReadbackRing` reads small GPU results back through three mapped buffers. `newest_finished` is for live callers and never blocks. `wait` is for exports, native only. A result is at most two submissions old, and never older than one already returned. `ReadbackError`, `readback::Finished` | vfx-core |
| `FrameSource`: `with_frame`; `History` | vfx-core |
| `FrameSource::held`; `History::push_slot(before, held)` | vfx-core |
| `FrameSource::live` (true for `History`) | vfx-core |
| `FrameSource::fetch_budget` (the distinct earlier frames one render may ask for, at least 2, plus up to as many again that one analysis of the current frame needs and nothing was kept from, for a host that holds every frame it hands out until the render returns; default `None`, as many as it needs) | vfx-core |
| `DeviceFaults::same_device` (whether two records are one device, across wgpu instances), `ReopenBackoff` (when to try for a new device after a loss) | vfx-core |
| `FrameSource::with_frame_if_wanted`, `Image::byte_rows`, `View::byte_rows` (a frame's bytes as the source holds them, so an effect can tell it hasn't changed before the source converts it; by default the bytes `with_frame` shows) | vfx-core |
| `Gpu::from_parts` | vfx-core |
| Native only: `device::slot` and `DeviceSlot`, the one device every effect in a plugin file draws on. `try_wait` is for live callers and never blocks. `wait` is for exports. While a host shares a device, the slot hands out that one | vfx-core |
| Native only: `DeviceSlot::on_loss` and `LossSubscription`. The slot calls each listener when it loses a device it opened, so whatever holds that device can release it. Once everything has, the slot reopens by itself | vfx-core |
| Native only: `Device`, `NoDevice`, and `device::open`, which opens a device for a `DeviceRequest` and blocks. A request names backends in order (`Backend`) and an adapter (`AdapterChoice`: high performance, low power, LUID or name). `with_env` applies `WGPU_BACKEND`, `WGPU_ADAPTER_NAME` and `WGPU_POWER_PREF` | vfx-core |
| Native only: `device::adapters`, the adapters in the order `open` tries them, for a host with its own instance. `device::d3d12_instance` keeps the D3D12 adapters for the life of the process, so opening a device never releases one. `device::luid`, `adapter_name_from_env` | vfx-core |
| Native only: `device::DeviceBuild`, what an effect builds once per device (its pipelines and shaders), on a thread of its own. `try_wait` never blocks and returns `NoDevice::Opening` while it builds. `wait` is for exports. `declining_software` refuses software adapters. `NoDevice::Unusable` holds a failed build's error | vfx-core |
| Native only: `device::BuildUser`, one instance's handle on a `DeviceBuild`. `DeviceBuild::user` makes one; when the last one drops, a value built on a host's device is released. `DeviceBuild::user_with` also stores the instance's state, reached with `BuildUser::lock`, and releases it when the slot loses the device, without waiting for a render | vfx-core |
| `d3d12_device_lock`: hold it while making or dropping a D3D12 device. segcam-face re-exports it | vfx-core |
| `Gpu::share`, `Gpu::shared` | vfx-core |
| `Gpu::unshare` (a host whose device was lost stops sharing it before dropping it, which D3D12 needs to offer the adapter again) | vfx-core |
| `Gpu::memory_hints` (allocator block sizes for a device a host or effect opens) | vfx-core |
| `Gpu::backend_options` (wgpu backend options for every instance: on Windows, D3D12 compiles with the `dxcompiler.dll` beside the binary or in `Common Files/vfx-core/dxc-<tag>`, else FXC) | vfx-core |
| Pixels: `Channel`, `Depth`, `View`, `ViewMut`, `Image`, `ImageMut`, `Argb` | vfx-core |
| Parameters: `ParamSpec`, `ParamKind`, `ParamValue`, `ParamValues`, `ParamError`, `Requires`, `Unit` | vfx-core |
| `contract::check` and `Clip` (a kept render equals a fresh one; under a fetch budget, in order, and after a jump within `held_catch_up_renders` for a held frame and `stream_catch_up_renders` for a stream) | vfx-host |
| `CpuInstance` and `CpuRenderError` (one kept effect per host instance) | vfx-host |
| `device` (for exports, waits) and `try_device` (for live renders, never blocks): the core slot's device, for GPU effects that go through RAM. `pool_key`. `release_own_device`, which a target calls after it shares a device | vfx-host |
| `pipeline::run` returns a `Result` and fails on a validation error in its own calls (`pipeline::checked`). `pipeline::evict_lost` | vfx-host |
| `remove_d3d12_device` and `RemoveDeviceError`: removes a D3D12 device through D3D12's own call, for device loss tests | vfx-host |
| `wait_for_next_device` and `NextDevice`: an editor waits for the slot's next device after a render failed because there was none. Once a wait has timed out, later renders don't wait again until a device is back | vfx-host |
| `CpuInstance::frames_held` | vfx-host |
| `CpuInstance::values_known` | vfx-host |
| `CpuInstance::report` | vfx-host |
| `plugin::register`, `plugin`, `gpu` | vfx-host |
| `controls::layout` | vfx-host |
| `by_rows` (a frame in row bands over a work queue) | vfx-host |

A target implements `FrameSource` for its frames. It fills a `Context` for each render. It keeps a
`CpuInstance` if it keeps effects, and it lays out parameters with `controls::layout`. The browser
items (`share`, `shared`) are the only parts that depend on the platform.

## Layout

The Cargo workspace (`Cargo.toml` at the root) has two folders. `vfx-core/` is the toolchain.
`plugins/` holds every effect crate, and all of them build with it.

Nothing in `vfx-core/` depends on `plugins/`. The exceptions are the preview's list of effects and,
on Windows and Linux, its face tracker (segcam-face, behind the preview's `tracker` feature).

```
vfx-core/
  src/                  vfx-core: Effect and CpuEffect, typed params, Frame, EffectChain, WGSL passes
    cpu.rs              frames in RAM for CPU effects: ARGB at 8/16/32 bits, strided views, history
    effects/            the core effects
  tests/                headless GPU tests for the core
  wrappers/
    common/             vfx-host: the registered plugin, the GPU, the render pool, the log,
                        control layout, kept CPU effects and their contract
    plugin/             vfx-plugin: an effect crate's one `export!`
    build/              vfx-build: the effect crate's build script (identity and its checks, PiPL)
    adobe/              vfx-ae: After Effects SmartFX, Metal interop
      ae-tests/         harness that renders VFX Preview inside After Effects
    ofx/                vfx-ofx: OpenFX, bindings generated from the OpenFX 1.5.1 headers
      minihost/         a small OpenFX host in C for testing without an application
    ffgl/               vfx-ffgl: FFGL 2.1, OpenGL + IOSurface / Vulkan interop
      minihost/         a small FFGL host in C with a headless OpenGL context
    ffgl-common/        FFGL bindings, GL transfers and pixel repacking
    package-win.sh      the Windows release: one installer (Inno Setup) and segcam's zips
    package-mac.sh      the macOS release: one installer (productbuild) and segcam's zips
  tools/cargo-vfx/      cargo vfx: builds, packages and installs effect crates
  hosts/preview/        the preview: a library host (`vfx_preview::run`) and the repository's app
    src/catalog.rs      the effects it offers (effect crates listed by build.rs)
    web/                index.html and build.sh for the browser build (WebGPU, or WebGL2 without it)
plugins/
  colour-adjust/        Colour Adjust: the core ColorAdjust effect as a plugin (own README)
  gaussian-blur/        Gaussian Blur (own README)
  vignette/             Vignette (own README)
  blend/                Blend (own README)
  vfx-chain/            VFX Preview, the four core effects above as one chain: the sample plugin
    examples/ae_reference.rs   its After Effects render path without After Effects
  segcam/
    segcam-motion/      segcam, ported from C++ (own README)
      ae-tests/ ofx-tests/ ffgl-tests/   host checks against the C++ effect and its CLI
    segcam-face/        faces, eyes, mouths and bodies for segcam (own README)
.github/workflows/      ci.yml (tests, every plugin), pages.yml (web preview), release.yml
```

## Building and running

Needs a current stable Rust (egui needs 1.95+; CI uses latest stable). The desktop preview needs
`ffmpeg` and `ffprobe` on PATH, for video files, webcams and Record, and `ffplay` for sound. They
aren't bundled: get them from [ffmpeg.org](https://ffmpeg.org/download.html), `brew install ffmpeg`
or `winget install ffmpeg`.
macOS bundles are universal, so add the targets once:

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin wasm32-unknown-unknown
```

`cargo vfx` runs from the repository (an alias in `.cargo/config.toml`). Without `-p`, it
acts on every effect crate; output goes to `target/vfx-plugin/`.

| What | Command | Result |
|---|---|---|
| The effects | `cargo vfx list` | each effect crate and its identity |
| A new effect | `cargo vfx new my-glow` | `plugins/my-glow`, added to the workspace and the preview |
| segcam's benchmark | `cargo vfx bench clip.mp4 [--preset overlay] [--depth 8] [--runs 2] [--cpu] [--reference DIR]` | the offline rows of `plugins/segcam/BENCHMARK.md`, timed on the release CLI; needs ffmpeg |
| Checks | `cargo vfx check` | effects missing from the preview, and installed After Effects plugins that share a match name; `--quick` and `--full` run the test tiers ([Testing](#testing)) |
| Plugins | `cargo vfx build [-p segcam-motion] [--hosts ae,ofx,ffgl] [--native]` | `<file>.plugin`, `.ofx.bundle` and `.bundle` on macOS (one universal binary; `--native` builds this Mac's architecture only); `.aex`, `.ofx.bundle` and `.dll` on Windows |
| macOS installers | `cargo vfx package` | `<file>-adobe.pkg` (Adobe's MediaCore folder) and `<file>-ofx.pkg` (`/Library/OFX/Plugins`) for each effect |
| macOS release | `vfx-core/wrappers/package-mac.sh` | `segcam-<version>-macOS.pkg` with every effect, and segcam's zips |
| Install | `cargo vfx install [--hosts ffgl]` | copies FFGL bundles into Resolume's Extra Effects; for After Effects and OpenFX, prints the `sudo installer` commands |
| Windows release | `vfx-core/wrappers/package-win.sh` | `segcam-<version>-Windows-Setup.exe` with every effect, and segcam's zips (needs [Inno Setup](https://jrsoftware.org/isinfo.php)) |
| Preview (desktop) | `cargo vfx preview [-p segcam-motion] [-- clip.mp4]` | the preview, opened on the effect; drop an image or video on it |
| Preview (web) | `vfx-core/hosts/preview/web/build.sh --serve` | http://localhost:8123, or `?effect=segcam` to open on an effect (needs `wasm-bindgen-cli` at the lockfile's version) |
| Tests | `cargo test -p vfx-core -p vfx-host -p vfx-chain -p colour-adjust -p gaussian-blur -p vignette -p blend -p segcam-motion` | headless GPU and CPU tests |

`cargo vfx package` signs and notarizes only when `SIGN_IDENTITY`, `NOTARY_PROFILE` and
`INSTALLER_IDENTITY` are set; without them the packages are unsigned (fine locally). With
all three, the bundles and the `.pkg` installers are both notarized and stapled. To sign
with your own Developer ID certificates (`security find-identity -v` lists them), store
notarization credentials once, then package:

```bash
xcrun notarytool store-credentials vfx-notary --apple-id YOUR_APPLE_ID --team-id TEAM_ID
SIGN_IDENTITY="Developer ID Application: NAME (TEAM_ID)" \
INSTALLER_IDENTITY="Developer ID Installer: NAME (TEAM_ID)" \
NOTARY_PROFILE=vfx-notary cargo vfx package
```

Each plugin writes a line of notes about which render path ran to
`~/Library/Logs/<file>-AfterEffects.log`, `<file>-OpenFX.log` or `<file>-FFGL.log`
(on Windows, the same names in `%TEMP%`).

## The preview

The preview runs one effect on a test pattern, image, video, webcam or Windows Spout source. It
builds the control panel from the effect's parameters. Every float and integer parameter has an
expression box (`sin(time/4)`, `bass * 2`, face tracking). Record saves the live output as a video.

The preview links the effects it offers, so the same build runs on the desktop and in the browser.
Its build script lists every effect crate as a dependency. For segcam Motion, the entry is
`Entry::plugin(&segcam_motion::VFX_PLUGIN)`, where `segcam_motion` is the crate's name in Rust.
`vfx-core/hosts/preview/src/catalog.rs` adds the core effects.

Another crate's effect can use the preview as a library:
`vfx_preview::run(vec![Entry::plugin(&my_effect::VFX_PLUGIN)])`.

[The preview's README](../vfx-core/hosts/preview/README.md) lists its keys, flags and Spout
details. The browser build uses WebGPU where the browser has it. Otherwise it falls back to WebGL2.
Record needs WebGPU in the browser.

On the desktop, a lost GPU device is reopened in place. In the browser, the page says to reload.

On Windows the preview draws with D3D12 by default, which avoids the memory growth seen with
NVIDIA's Vulkan driver on the setups tested, and falls back to Vulkan, then GL, when D3D12 can't
open. `WGPU_BACKEND=vulkan` picks Vulkan instead; `WGPU_POWER_PREF=low` the low power adapter.

## Releases

`.github/workflows/release.yml` builds every effect's plugins on GitHub's macOS and
Windows runners. Push a version tag and it makes a draft release with the files attached;
check it on GitHub, then publish it:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

Running the workflow by hand (Actions → release → Run workflow) builds the same files
as the run's artifacts, without a release. The files are named after segcam:

| File | What |
|---|---|
| `segcam-<version>-macOS.pkg` | macOS installer for After Effects and OpenFX: segcam, the other effects as options |
| `segcam-<version>-Windows-Setup.exe` | Windows installer for OpenFX, Resolume and After Effects: segcam, the other effects as options |
| `segcam-<version>-<macOS or Windows>-OpenFX.zip` | segcam's OpenFX bundle, to copy by hand |
| `segcam-<version>-<macOS or Windows>-Resolume.zip` | segcam's FFGL plugin, for Extra Effects |
| `segcam-<version>-<macOS or Windows>-AfterEffects.zip` | segcam's After Effects plugin |

Both installers remove v0.1.0's `SegcamMotionRust` files, except the macOS Resolume bundle,
which no installer replaces. The Windows installer keeps segcam's v0.1.0 AppId
(`windows-installer-id`), so it replaces that install in Apps.

Every Windows plugin loads `dxcompiler.dll` from `Common Files/vfx-core/dxc-<tag>`, where
the Setup puts one copy for all its plugins (`<tag>` is `vfx-core/wrappers/dxc-tag`), or
else from its own folder, where the zips put it with DXC's licenses. Without it wgpu compiles
the D3D12 shaders with FXC, which takes seconds before an effect draws on the GPU.
`vfx-core/wrappers/fetch-dxc.sh` downloads it from a pinned [DXC
release](https://github.com/microsoft/DirectXShaderCompiler/releases), checked by sha256,
rather than taking whatever Windows SDK the machine has. `package-win.sh` stops when it
can't fetch it; `cargo vfx build`, `install` and `preview` put it next to what they build
too and go on without it, so local builds offline still work. To move to a newer DXC, change
`dxc-tag` and the zip and sha256 in the script.

macOS signing and notarization run when the repository owner sets these secrets (Settings,
Secrets and variables, Actions). A fork sets its own to sign its own builds. Without them the
packages are unsigned:

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

The test profile optimises `segcam-motion` and `vfx-core`, so their pixel tests take seconds.
Overflow checks and debug assertions stay on. Several clones or worktrees on one machine can
share compiled dependencies through [sccache](https://github.com/mozilla/sccache):
`cargo install sccache`, then set `RUSTC_WRAPPER=sccache` in your shell (not in the repository's
config); the tiers use it when it is set. On Windows, sccache can't build `web-sys` for wasm,
because the command line gets too long, and the full tier runs the wasm check without it. Unset
`RUSTC_WRAPPER` for the web build too.

Unit tests cover the core, each effect, and each target's parameter mapping and pixel
conversions. Each host also has a harness that runs the effects:

- After Effects. `python3 vfx-core/wrappers/adobe/ae-tests/run.py` builds fixture projects for
  VFX Preview in the running After Effects (8, 16 and 32 bpc, CPU and Metal), renders them with
  `aerender` with MFR on and off, and compares them with `ae_reference`, the same render path
  without After Effects (macOS only). `python3 plugins/segcam/segcam-motion/ae-tests/run.py`
  renders segcam next to segcam's C++ effect, on macOS and Windows (`python` or `py` there).
  Both test the *installed* plugins and need After Effects, so they aren't in CI.
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

- One binary per effect holds every host's entry points. `cargo vfx` builds an effect crate once per
  architecture, with the host features on. It then copies the binary into each host's bundle layout.
  Each export macro registers the crate's `VFX_PLUGIN` static, which the target crates read.
- GPU effects get one device per plugin, with instances pooled. Building pipelines is the expensive
  part of a render, so prepared instances are kept per device and size, and each render borrows one.
  Concurrent renders (After Effects MFR, OpenFX hosts) each borrow their own.
- CPU effects read host frames where they are. After Effects' buffers are read at their own depth,
  without copies. OpenFX frames are converted one at a time. Resolume and the preview hand over one
  frame at a time and keep a history of recent frames, in full as far as the effect's
  `frames_held` asks.
- Plugins never use wgpu's OpenGL backend. Headless devices use Vulkan, Metal or DX12. The OpenGL
  backend releases whatever context is current when it starts. On Windows, that context belongs to
  the FFGL host. The preview creates its own instance.
- Frames stay on the GPU where a host allows it. With Metal acceleration, After Effects hands over
  `MTLBuffer`s. The plugin opens wgpu on After Effects' own device and wraps them. Resolume's GL
  textures are drawn into IOSurfaces that wgpu's Metal device shares. On Windows they go into Vulkan
  images shared with OpenGL. Everything else, and any setup failure, goes through RAM. If the GPU
  fails, a GPU effect's frame passes through unchanged instead of failing the host's render.
- When a CPU effect's render returns a `RenderError`, what happens depends on the host:
  - Live hosts (the preview, FFGL) show the input for that frame.
  - Editors (OpenFX, After Effects) check `RenderError::needs_device`. If the render failed
    because there was no device, they wait for the slot's next one, for as long as
    `DeviceSlot::wait` does, and render the frame again. The second render runs as an export
    render, so the effect also waits for its own pipelines.
  - If the frame fails again, the render fails. OpenFX returns `kOfxStatFailed` and shows one
    error through the message suite. After Effects returns an error code.
  - Once a wait has timed out, later renders that find no device fail right away, and OpenFX
    doesn't show another message. That lasts until the slot has a device again, so one outage shows
    one message.
- A CPU effect that draws on the GPU can leave its output there. On Windows, the FFGL target shares
  the device of its Vulkan images (`Gpu::share`) before the effect's first render. It hands each
  render a texture on that device as its output (`FrameOut::Texture`), which is shown with no trip
  through RAM. An effect that draws in RAM uploads into it. The input still comes through RAM, for
  the effect's work on the CPU.
- The OpenFX and FFGL bindings are bindgen's output, committed so a build needs neither
  bindgen nor libclang: `vfx-core/wrappers/ofx/src/sys.rs` from the OpenFX 1.5.1 headers, and
  `vfx-core/wrappers/ffgl-common/src/sys.rs` from `FFGL.h` in Resolume's SDK, with its licence.
  Each folder's `bindings.sh` regenerates its file from the pinned source, and with `--check`
  fails when the file is out of date. The FFGL one runs on Windows. The `ffgl-core` crate is
  GPL 3.0, so it isn't used.
- Testing found these host quirks. Tests or the harnesses now cover each one.
  After Effects needs PiPL spec version 13.25 or later for MFR. It also needs sequence data
  flattening flags for saved projects. It trims an effect's input to the mask bounds. Resolve 21
  rejects the OpenFX plugin description property. FFGL textures are premultiplied and bottom up.

## Known gaps

- After Effects 26.5 on macOS passes the VFX Preview and segcam Motion harnesses. On Windows,
  After Effects 2026 passes the segcam Motion harness and renders segcam by hand, through `aerender`
  and the render queue; the VFX Preview harness is macOS only. Natron was last checked with the
  earlier wrappers, which were one per effect.
- Premiere Pro 2026 on Windows loads the same `.aex` and plays and exports segcam, but has no
  SmartFX: CPU effects get no earlier frames there, so segcam shows faces only. Signing and
  notarization have never run.
- OpenFX frames cross through RAM. Effects can still draw on the GPU, but taking the host's GPU frames
  (Resolve's Metal extension) isn't done, and neither is an FFGL thumbnail.
- Hosts with no target yet: TouchDesigner (a Custom TOP), and EffectCraft, whose only plugin API
  is CPU WebAssembly, a fit for CPU effects.
