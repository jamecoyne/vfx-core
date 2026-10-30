# Models

| File | Model | Licence | Source |
|---|---|---|---|
| `yunet.onnx` | YuNet face detector, `face_detection_yunet_2023mar.onnx` | MIT, see `YUNET-LICENSE` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `face_mesh.onnx` | MediaPipe Face Mesh V2 (478 landmarks), converted to ONNX | Apache-2.0, code and weights | [MediaPipe](https://developers.google.com/mediapipe); conversion from [QtMeshEditor](https://github.com/fernandotonon/QtMeshEditor/blob/master/scripts/export-facecap-onnx.py) |
| `picodet.onnx` | PicoDet-S 416, LCNet backbone, COCO; `picodet_s_416_coco_lcnet.onnx` | Apache-2.0, see `PICODET-LICENSE` | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8.1/configs/picodet) |

The Apache-2.0 text is in the repository's `LICENSE-APACHE`.
