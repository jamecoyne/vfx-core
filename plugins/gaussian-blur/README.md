# Gaussian Blur

Blurs the picture horizontally, vertically or both, on the GPU. For After Effects, OpenFX
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

## Building

From the repository root:

```bash
cargo vfx build -p gaussian-blur
cargo test -p gaussian-blur
```

[Developer guide](../../docs/DEVELOPING.md) · [Source](../../vfx-core/src/effects/blur.rs)
