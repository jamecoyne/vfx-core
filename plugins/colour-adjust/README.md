# Colour Adjust

Changes the picture's brightness, contrast and saturation, on the GPU. For After Effects, OpenFX
(Resolve, Nuke, Natron), Resolume and the preview.

| Parameter | Range | Default |
|---|---|---|
| Brightness | -1 to 1 | 0 |
| Contrast | 0 to 4 | 1 |
| Saturation | 0 to 4 | 1 |

Values are used as they are, with nothing clamped, so a float target keeps values above 1 or below
0. Saturation uses Rec. 709 luma on whatever colour space the host sends, so for a linear chain,
convert to linear first.

## Building

It is the core [`ColorAdjust`](../../vfx-core/src/effects/color_adjust.rs) as a plugin. Build and
check it from the repository root:

```bash
cargo vfx build -p colour-adjust
cargo test -p colour-adjust
```

The [developer guide](../../docs/DEVELOPING.md) covers the layout, the host targets and the checks.
