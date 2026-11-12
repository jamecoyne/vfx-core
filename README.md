## Segcam

<!-- Hero: a short GIF of segcam's window swarm goes here. -->

[Download](https://github.com/jamecoyne/vfx-core/releases) ·
[Try segcam in your browser](https://jamecoyne.github.io/vfx-core/?effect=segcam) ·
[Developer guide](docs/DEVELOPING.md) ·
[Report a bug](https://github.com/jamecoyne/vfx-core/issues)

Segcam finds what moves in a shot, and faces and bodies. It draws them as labelled boxes, as mattes
that cut out the moving parts, or as Swarm. Swarm is a collage of frozen tiles, drawn as macOS
style windows that trail the subject. segcam started as a macOS app. Downloads for
[v0.1.2](https://github.com/jamecoyne/vfx-core/releases/tag/v0.1.2):

| Host | Plugin | macOS (signed and notarized) | Windows |
|---|---|---|---|
| After Effects | After Effects | [installer][mac-pkg] or [zip][mac-ae] | [installer][win-setup] or [zip][win-ae] |
| DaVinci Resolve | OpenFX | [installer][mac-pkg] or [zip][mac-ofx] | [installer][win-setup] or [zip][win-ofx] |
| Resolume Avenue and Arena | FFGL | [installer][mac-pkg] or [zip][mac-ffgl] | [installer][win-setup] or [zip][win-ffgl] |

[mac-pkg]: https://github.com/jamecoyne/vfx-core/releases/download/v0.1.2/segcam-0.1.2-macOS.pkg
[mac-ae]: https://github.com/jamecoyne/vfx-core/releases/download/v0.1.2/segcam-0.1.2-macOS-AfterEffects.zip
[mac-ofx]: https://github.com/jamecoyne/vfx-core/releases/download/v0.1.2/segcam-0.1.2-macOS-OpenFX.zip
[mac-ffgl]: https://github.com/jamecoyne/vfx-core/releases/download/v0.1.2/segcam-0.1.2-macOS-Resolume.zip
[win-setup]: https://github.com/jamecoyne/vfx-core/releases/download/v0.1.2/segcam-0.1.2-Windows-Setup.exe
[win-ae]: https://github.com/jamecoyne/vfx-core/releases/download/v0.1.2/segcam-0.1.2-Windows-AfterEffects.zip
[win-ofx]: https://github.com/jamecoyne/vfx-core/releases/download/v0.1.2/segcam-0.1.2-Windows-OpenFX.zip
[win-ffgl]: https://github.com/jamecoyne/vfx-core/releases/download/v0.1.2/segcam-0.1.2-Windows-Resolume.zip

Start with the installer for your platform. It covers Resolve, Resolume and After Effects, and
checks the ones it finds. Each zip holds one app's files and a `README.txt` naming the folder to
copy them into. The Windows files are unsigned, so SmartScreen asks before running the installer. Restart the
app after installing.

It runs on integrated graphics from 2020 or later (Intel UHD 770 or Iris Xe, AMD Radeon 680M, any
Apple M1 or newer) with a 6 core CPU.

[segcam's controls, install steps and limits](plugins/segcam/README.md)

# Installation


# vfx-core

VFX-core is our cross platform library and build chain for creating video effects. The app formally targets and has been tested against After Effects, DaVinci Resolve and, Resolume, but should be supported in any application that reads OpenFX, FFGL, webGL, or adobe plugins. VFX-core also includes a preview app to view or demo your tool.


## Creating your own effect with the VFX-core crate

```bash
cargo vfx new my-effect                   # a CPU effect to start from, already in the preview
cargo vfx preview -p my-effect           # open it in the preview; drop a video on it
cargo vfx build -p my-effect               # plugins for every host, in target/vfx-plugin/
```

An effect is one crate. It implements `Effect` (GPU, on [wgpu](https://wgpu.rs)) or `CpuEffect`
(frames in RAM), declares its parameters, and vfx-core turns it into a plugin for each host. The
[developer guide](docs/DEVELOPING.md) covers the crate, the layout, building, releases, testing and
the design. Coding agents start at [AGENTS.md](AGENTS.md).

## Additional effects in this repo

We've also developed a few other example effects using vfx core! Give them a try!

| Effect | What it does |
|---|---|
| [segcam](plugins/segcam/segcam-motion) | boxes, mattes and the window swarm, with faces and bodies |
| [Colour Adjust](plugins/colour-adjust) | brightness, contrast and saturation, on the GPU |
| [Gaussian Blur](plugins/gaussian-blur) | separable blur, horizontal, vertical or both, on the GPU |
| [Vignette](plugins/vignette) | fades towards a colour away from a centre, on the GPU |
| [Blend](plugins/blend) | a second layer over the input: normal, add, multiply or screen, on the GPU |
| [VFX Preview](plugins/vfx-chain) | the four above as one chain, with an optional linear light stage |
| [core effects](vfx-core/src/effects) | the building blocks behind the four, and Flip and Feedback Trail, which the preview offers too |

Each effect is one plugin crate under [`plugins/`](plugins). The toolchain, the host targets and the
preview app are in [`vfx-core/`](vfx-core), and the preview is at
[`vfx-core/hosts/preview`](vfx-core/hosts/preview).

## Credits

[Made by LO.FI.SCI.FI and friends.](https://github.com/jamecoyne/vfx-core/graphs/contributors)
Face detection uses YuNet, MediaPipe Face Mesh and PicoDet ([models and licenses](plugins/segcam/segcam-face/models/README.md)).

Licensed under MIT or Apache 2.0, whichever you prefer.
