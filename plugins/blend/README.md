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

Linearize Layer decodes the layer from sRGB first. Turn it on in a chain that works in linear light.
The input is already linear from an earlier stage, so only the layer needs the conversion.

## Building

From the repository root:

```bash
cargo vfx build -p blend
cargo test -p blend
```

[Developer guide](../../docs/DEVELOPING.md) · [Source](../../vfx-core/src/effects/blend.rs)
