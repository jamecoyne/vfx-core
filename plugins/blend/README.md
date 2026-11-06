# Blend

Lays a second layer over the picture with Normal, Add, Multiply or Screen, on the GPU. For After
Effects, OpenFX (Resolve, Nuke, Natron), Resolume and the preview.

The second layer is picked in the host, as a clip or layer.

| Parameter | Range | Default |
|---|---|---|
| Mode | Normal, Add, Multiply, Screen | Normal |
| Amount | 0 to 1 | 1 |
| Linearize Layer | on or off | off |

Amount scales the layer's alpha. A fully transparent layer leaves the input unchanged. Input and
output use straight alpha.

Linearize Layer decodes the layer from sRGB first, for a chain that works in linear light. The input
has already been converted by an earlier stage of the chain, so only the layer needs it.

## Building

From the repository root:

```bash
cargo vfx build -p blend
cargo test -p blend
```

[Developer guide](../../docs/DEVELOPING.md) · [Source](../../vfx-core/src/effects/blend.rs)
