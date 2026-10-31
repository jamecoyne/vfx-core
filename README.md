# vfx-core

Video effects written once in Rust and shipped as plugins for After Effects, DaVinci Resolve and
Resolume, with a preview app that also runs in the browser.

<!-- Hero: a short GIF of segcam's window swarm goes here. -->

[Download](https://github.com/jamecoyne/vfx-core/releases) ·
[Try it in your browser](https://jamecoyne.github.io/vfx-core/) ·
[Developer guide](docs/DEVELOPING.md) ·
[Report a bug](https://github.com/jamecoyne/vfx-core/issues)

> The segcam plugin is on its way. Until the release, try segcam in the browser preview or build it
> from source.

## segcam

segcam picks out what moves in a shot, plus faces and bodies, and turns them into labelled boxes,
mattes that cut the moving parts out, or Swarm: a collage of frozen tiles drawn as macOS windows
that trail the subject. It started as a macOS app. The plugins:

| Host | Plugin | macOS | Windows |
|---|---|---|---|
| After Effects | After Effects | installer | zip |
| DaVinci Resolve | OpenFX | installer | zip |
| Resolume Avenue and Arena | FFGL | zip | zip |

On Windows, copy the plugin into the folder its zip's readme names, then restart the host.

It runs on integrated graphics from 2020 or later (Intel UHD 770 or Iris Xe, AMD Radeon 680M, any
Apple M1 or newer) with a 6 core CPU. Swarm draws on the GPU when there is one.

[segcam's controls, install steps and limits](plugins/segcam/README.md)

## Effects in this repository

| Effect | What it does |
|---|---|
| [segcam Motion](plugins/segcam/segcam-motion) | segcam: boxes, mattes and the window swarm, with faces and bodies |
| [VFX Preview](vfx-core/effects/vfx-chain) | colour adjust, vignette, a blend with a second layer and blur, on the GPU |
| [core effects](vfx-core/src/effects) | colour, blur, vignette, blend, flip, feedback trail: building blocks to try in the preview |

## Make your own effect

You need a current stable Rust. From a clone of this repository:

```bash
cargo vfx new my-glow                    # a CPU effect to start from, already in the preview
cargo vfx preview -p my-glow             # open it in the preview; drop a video on it
cargo vfx build -p my-glow               # plugins for every host, in target/vfx-plugin/
```

An effect is one crate. It implements `Effect` (GPU, on [wgpu](https://wgpu.rs)) or `CpuEffect`
(frames in RAM), declares its parameters, and vfx-core turns it into a plugin for each host. The
[developer guide](docs/DEVELOPING.md) covers the crate, the layout, building, releases, testing and
the design. Coding agents start at [AGENTS.md](AGENTS.md).

## Credits

Made by jamecoyne and its [contributors](https://github.com/jamecoyne/vfx-core/graphs/contributors).
Face detection uses YuNet, MediaPipe Face Mesh and PicoDet ([models and licenses](plugins/segcam/segcam-face/models/README.md)).

Licensed under MIT or Apache 2.0, whichever you prefer.
