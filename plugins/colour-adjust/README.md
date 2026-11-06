# Colour Adjust

Changes the picture's brightness, contrast and saturation, on the GPU. For After Effects, OpenFX
(Resolve, Nuke, Natron), Resolume and the preview.

| Parameter | Range | Default |
|---|---|---|
| Brightness | -1 to 1 | 0 |
| Contrast | 0 to 4 | 1 |
| Saturation | 0 to 4 | 1 |

Float output is not clamped to 0 to 1, so values above 1 or below 0 survive in float targets.

Saturation uses Rec. 709 luma on whatever colour space the host sends. For a linear chain, convert to linear first.

## Building

From the repository root:

```bash
cargo vfx build -p colour-adjust
cargo test -p colour-adjust
```

[Developer guide](../../docs/DEVELOPING.md) · [Source](../../vfx-core/src/effects/color_adjust.rs)
