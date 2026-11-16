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
   (`plugins/vfx-chain/examples/ae_reference.rs`) and compares.

| Check | Passes when |
|---|---|
| reference | After Effects is within 2 levels of the plugin's pipeline run directly, quantised the way each bit depth writes it (32 bpc premultiplies before clipping) |
| gpu vs cpu | the GPU-native and CPU paths differ by at most 1 level |
| mfr on vs off | frames are identical with Multi-Frame Rendering on and off |
| half | Half resolution is within a mean of 4 levels of Full, downsampled |
| path | the plugin log (`~/Library/Logs/VfxPreview-AfterEffects.log`) shows the expected render path |
| per-case checks | `checks.py`: blur past masks, Blur Edges, symmetry, 32 bpc headroom |

Results go to `$AE_TEST_OUT`: `report.txt`, `report.json`, and for each failure
the After Effects frame, the expected frame and an amplified difference in `failures/`.
The exit status is non-zero when anything fails.

## Output

Both After Effects harnesses, this one and segcam's
(`plugins/segcam/segcam-motion/ae-tests/run.py`), write frames to `$AE_TEST_OUT`:

- It has no default. Set it, or pass `--out`, to a folder on a data drive.
- A folder on the system disk is refused. A segcam run at 1080p writes gigabytes of frames.

At 8 and 16 bpc, frames are compressed 8-bit PNGs, through the template After Effects
ships hidden as `_HIDDEN X-Factor 8 Premul`. A PNG holds straight colour whatever the
template says. The harness premultiplies it as it reads, rounding to nearest, which gives
back the TIFF template's bytes exactly.

At 32 bpc, frames stay uncompressed TIFFs. A 32 bpc value above 1.0 under partial alpha
reaches the TIFF through premultiplying and is clipped after. The PNG's straight colour
clips it first, so the `hdr`, blend and linear references would fail. Where an After
Effects lacks the PNG template, the harness says so and writes TIFFs at every depth. Frames that pass their checks are deleted. Only the ones that fail are kept, with
the frame they were compared against.

## Limits

- A script can't set an output module's format or depth (`setSettings` answers that both
  are read-only), and `aerender` can't either. Only a template chooses them, and no
  template After Effects 2026 ships writes a compressed float format with alpha. So the
  frames are 8-bit at every depth, and the 16 and 32 bpc checks compare After Effects'
  dithered 8-bit output. They don't prove 16-bit or float values. 32 bpc values above 1.0
  are checked indirectly: the `hdr` case raises them and a second instance brings them back.
- Reference cases use a layer that lines up with the composition. Offsets, masks and
  growth are checked by their geometry instead.
- `aerender` can't run on GitHub's runners, which have no After Effects licence. Run it
  locally or on a self-hosted runner.

## Adding a case

Add a dict to `CASES` in `cases.py`. Set `reference=True` when the source is the
pattern and the result should match the pipeline exactly. Otherwise write a check in
`checks.py`. Parameter names are the ones shown in Effect Controls.
