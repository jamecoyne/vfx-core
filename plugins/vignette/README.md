# Vignette

Darkens the picture towards its edges, or fades it to any colour away from a centre, on the GPU.
For After Effects, OpenFX (Resolve, Nuke, Natron), Resolume and the preview.

| Parameter | Range | Default |
|---|---|---|
| Center | a point in the frame | the middle |
| Radius | 0 to 2 | 0.75 |
| Softness | 0 to 2 | 0.5 |
| Color | RGBA | black, opaque |
| Amount | 0 to 1 | 1 |

Distance is measured from the centre and divided by half the frame's diagonal, so a radius of 1
reaches the corners whatever the frame size or aspect. The fade reaches the colour at the radius and
starts `Softness` inside it. A radius of 0 covers the whole frame.

## Building

From the repository root:

```bash
cargo vfx build -p vignette
cargo test -p vignette
```

[Developer guide](../../docs/DEVELOPING.md) · [Source](../../vfx-core/src/effects/vignette.rs)
