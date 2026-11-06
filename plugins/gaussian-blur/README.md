# Gaussian Blur

A separable Gaussian blur, horizontal, vertical or both, on the GPU. It is the core
[`GaussianBlur`](../../vfx-core/src/effects/blur.rs) as a plugin for After Effects, OpenFX
(Resolve, Nuke, Natron), Resolume and the preview.

| Parameter | Range | Default |
|---|---|---|
| Radius | 0 to 256 pixels | 4 |
| Direction | Both, Horizontal, Vertical | Both |
| Blur Edges | on or off | off |

At a radius of 0 the frame passes through exactly. Hosts hand over straight alpha, so the blur runs
on premultiplied colour and divides the alpha out afterwards. Transparent pixels don't bleed colour
into soft edges.

Blur Edges lets the blur spread past the frame's edge, which needs a host with regions of definition.
Hosts without them leave the control out.

Build and check it from the repository root:

```bash
cargo vfx build -p gaussian-blur
cargo test -p gaussian-blur
```

The [developer guide](../../docs/DEVELOPING.md) covers the layout, the host targets and the checks.
