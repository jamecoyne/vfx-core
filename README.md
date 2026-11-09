## segcam

Segcam finds what moves in a shot, and faces and bodies. It draws them as labelled boxes, as mattes
that cut out the moving parts, or as Swarm. Swarm is a collage of frozen tiles, drawn as macOS
style windows that trail the subject. segcam started as a macOS app. The plugins:

| Host | Plugin | macOS | Windows |
|---|---|---|---|
| After Effects | After Effects | installer | zip |
| DaVinci Resolve | OpenFX | installer | zip |
| Resolume Avenue and Arena | FFGL | zip | zip |

On Windows, copy the plugin into the folder its zip's readme names, then restart the host.

It runs on integrated graphics from 2020 or later (Intel UHD 770 or Iris Xe, AMD Radeon 680M, any
Apple M1 or newer) with a 6 core CPU. Swarm draws on the GPU when there is one.

[segcam's controls, install steps and limits](plugins/segcam/README.md)

# Installation

## Effects in this repository

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

# vfx-core

VFX-core is our cross platform library and build chain for creating video effects. The app formally targets and has been tested against After Effects, DaVinci Resolve and, Resolume, but should be supported in any application that reads OpenFX, FFGL, webGL, or adobe plugins. VFX-core also includes a preview app to view or demo your tool.

<!-- Hero: a short GIF of segcam's window swarm goes here. -->

[Download](https://github.com/jamecoyne/vfx-core/releases) ·
[Try segcam in your browser](https://jamecoyne.github.io/vfx-core/?effect=segcam-motion) ·
[Developer guide](docs/DEVELOPING.md) ·
[Report a bug](https://github.com/jamecoyne/vfx-core/issues)

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

## Credits

Made by LO.FI.SCI.FI and friends [contributors](https://github.com/jamecoyne/vfx-core/graphs/contributors).
Face detection uses YuNet, MediaPipe Face Mesh and PicoDet ([models and licenses](plugins/segcam/segcam-face/models/README.md)).

Licensed under MIT or Apache 2.0, whichever you prefer.
