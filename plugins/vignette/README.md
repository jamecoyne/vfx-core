# Vignette

Fades towards a colour away from a centre, on the GPU. It is the core
[`Vignette`](../../vfx-core/src/effects/vignette.rs) as a plugin for After Effects, OpenFX
(Resolve, Nuke, Natron), Resolume and the preview.

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

Build and check it from the repository root:

```bash
cargo vfx build -p vignette
cargo test -p vignette
```

The [developer guide](../../docs/DEVELOPING.md) covers the layout, the host targets and the checks.
