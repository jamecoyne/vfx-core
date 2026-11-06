# segcam-face

Faces, eyes, mouths and bodies for segcam's effects: Apple's Vision on macOS, ONNX Runtime on other platforms. No host code.

| finds | model, outside macOS | licence |
| --- | --- | --- |
| faces | YuNet | MIT |
| eyes, mouths | MediaPipe Face Mesh V2 | Apache-2.0 |
| bodies | PicoDet-S 416 | Apache-2.0 |

```rust
let mut segmenter = segcam_face::Segmenter::best().expect("a device");
let image = Image { pixels: Pixels::Bgra8(&frame), width, height, stride: width };
let parts = Parts { face: true, eyes: true, mouth: true, body: false };
let segments = segmenter.find(&image, parts);
```

macOS uses Vision, as the macOS app does. Windows runs the models on a GPU through DirectML, or on the
CPU if that's more than half again as fast in a short trial, since in a render the CPU kernels
share the cores. Among GPUs that time close it takes the lowest numbered, and an integrated GPU
beside a discrete one gets no trial. `SEGCAM_FACE_DEVICE` skips the trial, with `cpu` or a DXGI
adapter's index. Linux runs them on the CPU. On the CPU they use 2 threads.
For DirectML in a plugin, add `cargo:rustc-link-arg-cdylib=/DELAYLOAD:DirectML.dll` to its build
script and ship `DirectML.dll` next to the plugin binary. Needed for Windows 10: the older
DirectML isn't compatible with newer ONNX Runtime, so it has to be bundled to avoid falling back
to CPU.

`cargo test -p segcam-face` passes on Windows (CPU, and DirectML on NVIDIA and AMD), Linux and
macOS. On Windows they also run on DirectML where it loads; set `SEGCAM_REQUIRE_DIRECTML=1` on a
machine with a GPU to make a failure to load an error. Cargo runs them from
`target/<profile>/deps`, so copy `DirectML.dll` there from `target/<profile>`.
