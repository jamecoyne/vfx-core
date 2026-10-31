# ADR 4: Swarm draws on the GPU by default

Status: proposed. Date: 2026-10-30. Decides: Strongtut.

## Context

Drawing the windows was most of a Swarm render on the CPU. A wgpu compute pass draws them per
pixel, replaying the CPU's windows in order, within one level of the CPU path.

Measured offline at 1080p:

| | CPU | GPU |
|---|---|---|
| RTX 4070, Swarm drawing, float | 15.3 ms | 3.6 ms |
| RTX 4070, Swarm drawing, 8 bit | 21.5 ms | 1.35 ms |
| Intel UHD 770, Swarm render, 8 bit | 25.8 ms | 15.6 ms |
| Apple M4, Swarm render, 8 bit, frames handed over as Resolume does | 20.2 ms | 5.1 ms |

With few small windows the CPU is faster, since the GPU path pays to upload and read back the frame.

## Decision

Draw Swarm on the GPU by default wherever a hardware GPU exists. Each render picks CPU or GPU by the
window pixels it draws, from measured switch points. Software adapters are left out. The CPU path
stays the reference and the fallback. Overlay stays on the CPU.

## Options considered

| Option | For | Against |
|---|---|---|
| GPU by default, picked per render | fastest in every case measured | two paths to keep checked |
| GPU opt in | CPU stays the default | most users never get the speed |
| CPU only | one path | Swarm stays the slowest mode |

## Consequences

- Output is within one level of the CPU, not byte for byte; the docs say so.
- The switch points come from an RTX 4070 and need tuning per kind of GPU.
- `SEGCAM_GPU=0` forces the CPU for tests; users never need it.
- Keeping frames on the host's GPU would remove the upload and read back, and move Overlay to the
  GPU too. That comes after 1.0.
