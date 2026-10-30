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

macOS uses Vision, like segcam 1.1. Windows runs the models on its fastest GPU through
DirectML, or on the CPU if that's faster. Linux runs them on the CPU.
For DirectML in a plugin, add `cargo:rustc-link-arg-cdylib=/DELAYLOAD:DirectML.dll` to its build
script and ship `DirectML.dll` next to the plugin binary. Needed for Windows 10: the older
DirectML isn't compatible with newer ONNX Runtime, so it has to be bundled to avoid falling back
to CPU.

`cargo test -p segcam-face` passes on Windows (CPU, and DirectML on NVIDIA and AMD), Linux and
macOS. On Windows the tests fail if a GPU is detected but DirectML doesn't load; Cargo runs them
from `target/<profile>/deps`, so copy `DirectML.dll` there from `target/<profile>`.
