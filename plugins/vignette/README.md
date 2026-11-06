# Vignette

Fades the picture towards a colour, away from a centre, on the GPU. The default colour is black, so the
edges darken. It works in After Effects, OpenFX (Resolve, Nuke, Natron), Resolume and the preview.

| Parameter | Range | Default |
|---|---|---|
| Center | a point in the frame | the middle |
| Radius | 0 to 2 | 0.75 |
| Softness | 0 to 2 | 0.5 |
| Color | RGBA | black, opaque |
| Amount | 0 to 1 | 1 |

Distance from the centre is divided by half the frame's diagonal. So a radius of 1 reaches the
corners, whatever the frame size or aspect. The fade reaches the colour at the radius, and it starts
`Softness` inside that radius. A radius of 0 covers the whole frame.

## Building

From the repository root:

```bash
cargo vfx build -p vignette
cargo test -p vignette
```

[Developer guide](../../docs/DEVELOPING.md) · [Source](../../vfx-core/src/effects/vignette.rs)
