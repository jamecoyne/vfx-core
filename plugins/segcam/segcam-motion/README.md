# segcam

segcam finds what **Detect** asks for (what moves, what is bright, or faces and bodies) and
draws it as labelled boxes, mattes or a collage of windows. To install it, read the
[segcam README](../README.md). How it works and how to check it are in
[DEVELOPING.md](DEVELOPING.md).

## Output modes

- **Overlay**: the layer, each moving region stroked as a box with its ID chip
  (`motion#412.1` is the biggest moving thing in frame 412). Boxes, labels and the box
  colour can be switched.
- **Motion matte**: the layer cut out by the motion mask itself, softened from the grid.
- **Box matte**: the layer cut out by the segments' boxes.
- **Swarm**: a desktop collage. Every segment of every recent frame becomes a frozen tile of the
  frame it was cut from, drawn as a macOS window (title bar, buttons, rounded corners and shadow),
  newest on top. A window stays until **Swarm max tiles** newer ones push it out. **Trail frames**
  is how far back windows can come from.

**Mirror** flips the picture and the boxes for a selfie camera.

**Detect** picks Motion, Threshold or Faces only. Threshold finds areas brighter than **Level**, or
darker with **Invert**, and **Auto level** picks the level for each frame. Its segments are labelled
`blob#…`. Faces only runs no motion or threshold detection: the segments are the ticked face parts,
Motion matte comes out empty and Box matte cuts by the face boxes.

**Swarm backdrop** puts the windows on a transparent canvas (the default), the source frame, or
black.

**Face**, **Eyes**, **Mouth** and **Body** enable detection of those, with Apple's Vision on macOS
and ONNX Runtime models on Windows and Linux. Resolume and the preview never wait for faces: they
draw without them until the models are up, and each render then takes the newest faces there are.
Other hosts may be exporting, and their renders wait for the models and find each frame's own
faces. The browser has no faces yet. In live playback, on a slow GPU, faces and bodies are found
less often to leave the GPU more time for the picture; exports still find every frame's faces.

## Parameters

| | Choices or range | Default |
| --- | --- | --- |
| Output | Overlay, Motion matte, Box matte, Swarm | Swarm |
| Detect | Motion, Threshold, Faces only | Faces only (Motion in the browser) |
| Mirror | on or off | off |
| Sensitivity (Motion) | 0 to 100 percent | 86.4 |
| Adaptation (Motion) | 0.1 to 50 percent | 6 |
| Background (Motion) | 1 to 120 analyses | 64 |
| Minimum size (Motion) | 0 to 25 percent | 0.25 |
| Max segments (Motion) | 1 to 256 | 16 |
| Grid width (Motion) | 32 to 1024 | 192 |
| Level (Threshold) | 0 to 255 | 170 |
| Auto level (Threshold) | on or off | off |
| Invert (Threshold) | on or off | off |
| Face, Body (Faces) | on or off each | off |
| Eyes, Mouth (Faces) | on or off each | on |
| Boxes (Overlay) | on or off | on |
| Labels (Overlay) | on or off | on |
| Box color (Overlay) | red, green, blue | 0.25, 1.0, 0.4 |
| Trail frames (Swarm) | 1 to 120 | 60 |
| Swarm max tiles (Swarm) | 1 to 1000 | 100 |
| Swarm backdrop (Swarm) | Transparent, Source, Black | Transparent |
| Live windows (Swarm) | on or off | off |

The group is in brackets. Some sliders show a narrower range than the limit.
