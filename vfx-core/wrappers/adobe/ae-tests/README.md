# After Effects test harness

Renders the plugin inside After Effects and checks the frames. It covers
the paths that unit tests can't reach: After Effects' parameter handling, SmartFX
rectangles, bit depths, downsampling, masks, Multi-Frame Rendering and the
GPU-native Metal path.

```bash
python3 vfx-core/wrappers/adobe/ae-tests/run.py
```

## What it needs

- After Effects 2026 with the plugin installed (`cargo vfx package -p vfx-chain`, then
  install `target/vfx-plugin/VfxPreview-adobe.pkg`). The harness tests whatever build is installed, not the source tree.
- `ffmpeg` and `ffprobe` on the `PATH`, and `cargo`.
- A clean After Effects project, or a moment to answer its save prompt. The harness
  builds its fixtures in the running After Effects and asks before closing your project;
  Cancel stops the run.

A full run takes under a minute with After Effects already open (157 checks).
Narrow it with `--cases blur,hdr` and `--variants 32bpc-gpu`.

## How it works

1. Writes two 8-bit test images (gradients, a checker, alpha ramps, a soft hole, and
   magenta under transparent pixels so colour bleed shows).
2. Builds one project per variant in After Effects: 8, 16 and 32 bpc on the CPU path,
   and 32 bpc with Metal GPU acceleration, which uses the GPU-native path. Each case
   in `cases.py` is a composition with its own render queue items.
3. Renders each project with `aerender`, so opening a saved project is tested too.
   The 32 bpc GPU project renders twice, with Multi-Frame Rendering on and off.
4. Runs the plugin's pipeline directly on the same input pixels
   (`vfx-core/effects/vfx-chain/examples/ae_reference.rs`) and compares.

| Check | Passes when |
|---|---|
| reference | After Effects is within 2 levels of the plugin's pipeline run directly, quantised the way each bit depth writes it (32 bpc premultiplies before clipping) |
| gpu vs cpu | the GPU-native and CPU paths differ by at most 1 level |
| mfr on vs off | frames are identical with Multi-Frame Rendering on and off |
| half | Half resolution is within a mean of 4 levels of Full, downsampled |
| path | the plugin log (`~/Library/Logs/VfxPreview-AfterEffects.log`) shows the expected render path |
| per-case checks | `checks.py`: blur past masks, Blur Edges, symmetry, 32 bpc headroom |

Results are in `target/ae-tests/`: `report.txt`, `report.json`, and for each failure
the After Effects frame, the expected frame and an amplified difference in `failures/`.
The exit status is non-zero when anything fails.

## Limits

- After Effects' output templates write 8-bit TIFFs, and neither scripts nor `aerender`
  can change the bit depth. So 32 bpc values above 1.0 are checked indirectly: the `hdr`
  case raises them and a second instance brings them back.
- Reference cases use a layer that lines up with the composition. Offsets, masks and
  growth are checked by their geometry instead.
- `aerender` can't run on GitHub's runners, which have no After Effects licence. Run it
  locally or on a self-hosted runner.

## Adding a case

Add a dict to `CASES` in `cases.py`. Set `reference=True` when the source is the
pattern and the result should match the pipeline exactly. Otherwise write a check in
`checks.py`. Parameter names are the ones shown in Effect Controls.
