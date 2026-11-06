# Blend

Blends a second layer over the input, on the GPU. It is the core
[`Blend`](../../vfx-core/src/effects/blend.rs) as a plugin for After Effects, OpenFX (Resolve, Nuke,
Natron), Resolume and the preview.

The second layer is picked in the host, as a clip or layer.

| Parameter | Range | Default |
|---|---|---|
| Mode | Normal, Add, Multiply, Screen | Normal |
| Amount | 0 to 1 | 1 |
| Linearize Layer | on or off | off |

The layer's alpha times Amount decides how much of the blended colour lays over the input, so a
transparent layer leaves the input as it is. Straight alpha in and out.

Linearize Layer decodes the layer from sRGB first, for a chain that works in linear light. The input
has already been converted by an earlier stage of the chain, so only the layer needs it.

Build and check it from the repository root:

```bash
cargo vfx build -p blend
cargo test -p blend
```

The [developer guide](../../docs/DEVELOPING.md) covers the layout, the host targets and the checks.
