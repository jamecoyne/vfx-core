# segcam roadmap

Goal: segcam runs reliably as a plugin in Resolve, Resolume and After Effects, then a release, then
an announcement. Requirements and their status are in [SPEC.md](SPEC.md).

## To 1.0

1. Land the work as small PRs: Windows fixes, Resolve speed (F4, F6), faster Swarm, Swarm on the
   GPU (F8, F9).
2. Faces every 2nd frame (F7) and the Deliver check (F5).
3. Reliability: long runs with no stalls, the 30 minute Resolume run (N8), After Effects on
   Windows.
4. The shared benchmark, with an offline bench in the repo, run on every contributor's machine.
5. Release dry run (R1 to R5), then `v1.0.0`.
6. Announce.

## After 1.0

- Reusing vfx-core for new effects.
- The web, with NicoWasHere's SYNUS as a test case.
- A live only build option for effects that keep feedback, so they skip the After Effects and
  OpenFX builds (NicoWasHere's idea).
- Keeping work between renders in After Effects.
- Frames staying on the host's GPU, which also moves Overlay to the GPU.
- Windows code signing.
