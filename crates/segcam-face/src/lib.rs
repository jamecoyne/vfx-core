//! Face, eye, mouth and body detection for segcam's effects. Callers pass in [`Image`]s and get
//! [`Segment`]s back.
//!
//! On macOS they come from Apple's Vision, as in segcam 1.1. Other platforms get YuNet faces,
//! Face Mesh V2 eyes and mouths (or YuNet's landmarks without it) and PicoDet-S bodies, on ONNX
//! Runtime ([`runtime`], with the `onnx` feature). [`Segmenter::best`] picks the right one.

#[cfg(all(windows, feature = "onnx"))]
pub mod directml;
pub mod mesh;
pub mod picodet;
#[cfg(feature = "onnx")]
pub mod runtime;
#[cfg(target_os = "macos")]
pub mod vision;
pub mod yunet;

pub use picodet::Body;
pub use yunet::Face;

/// The bundled YuNet model, `face_detection_yunet_2023mar.onnx`.
#[cfg(feature = "onnx")]
pub const YUNET_MODEL: &[u8] = include_bytes!("../models/yunet.onnx");

/// The bundled MediaPipe Face Mesh V2 landmark model, converted to ONNX.
#[cfg(feature = "onnx")]
pub const FACE_MESH_MODEL: &[u8] = include_bytes!("../models/face_mesh.onnx");

/// The bundled PicoDet-S 416 person detector, `picodet_s_416_coco_lcnet.onnx`.
#[cfg(feature = "onnx")]
pub const PICODET_MODEL: &[u8] = include_bytes!("../models/picodet.onnx");

/// Four-channel pixels, top row first.
#[derive(Clone, Copy, Debug)]
pub enum Pixels<'a> {
    Rgba8(&'a [u8]),
    Bgra8(&'a [u8]),
    /// RGBA from 0 to 1.
    RgbaF32(&'a [f32]),
}

#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub pixels: Pixels<'a>,
    pub width: usize,
    pub height: usize,
    /// Pixels from the start of one row to the next.
    pub stride: usize,
}

impl Image<'_> {
    /// Whether the size is non-zero and the pixels cover it.
    pub fn is_usable(&self) -> bool {
        let channels = match self.pixels {
            Pixels::Rgba8(data) | Pixels::Bgra8(data) => data.len(),
            Pixels::RgbaF32(data) => data.len(),
        };
        self.width > 0
            && self.height > 0
            && self.stride >= self.width
            && channels >= ((self.height - 1) * self.stride + self.width) * 4
    }

    /// Red, green and blue from 0 to 255 at a pixel inside a usable image.
    #[inline]
    pub fn rgb(&self, x: usize, y: usize) -> [f32; 3] {
        let i = (y * self.stride + x) * 4;
        match self.pixels {
            Pixels::Rgba8(d) => [d[i], d[i + 1], d[i + 2]].map(f32::from),
            Pixels::Bgra8(d) => [d[i + 2], d[i + 1], d[i]].map(f32::from),
            Pixels::RgbaF32(d) => [d[i], d[i + 1], d[i + 2]].map(|v| v.clamp(0.0, 1.0) * 255.0),
        }
    }
}

/// A rectangle normalized to the frame: origin top-left, y down.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    fn iou(&self, other: &Rect) -> f64 {
        let w = (self.x + self.w).min(other.x + other.w) - self.x.max(other.x);
        let h = (self.y + self.h).min(other.y + other.h) - self.y.max(other.y);
        if w <= 0.0 || h <= 0.0 {
            return 0.0;
        }
        let overlap = w * h;
        overlap / (self.w * self.h + other.w * other.h - overlap)
    }

    fn clamped(self) -> Rect {
        let (x0, y0) = (self.x.clamp(0.0, 1.0), self.y.clamp(0.0, 1.0));
        let (x1, y1) = (
            (self.x + self.w).clamp(0.0, 1.0),
            (self.y + self.h).clamp(0.0, 1.0),
        );
        Rect {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }
}

/// What a segment is. Eyes are named by the side of the image they are on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Face,
    EyeLeft,
    EyeRight,
    Mouth,
    Body,
}

/// A found face, face part or body. The parts of one face share its number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub kind: Kind,
    pub number: u64,
    pub rect: Rect,
    pub score: f32,
}

/// Which segments to find.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Parts {
    pub face: bool,
    pub eyes: bool,
    pub mouth: bool,
    pub body: bool,
}

/// Vision on macOS, or the models on ONNX Runtime.
pub enum Segmenter {
    #[cfg(target_os = "macos")]
    Vision(vision::VisionSegmenter),
    #[cfg(feature = "onnx")]
    Ort(Box<runtime::OrtSegmenter>),
}

impl Segmenter {
    /// Vision on macOS, or the models on the fastest device, or `None` if neither loads (or,
    /// without the `onnx` feature, outside macOS).
    pub fn best() -> Option<Self> {
        #[cfg(target_os = "macos")]
        if vision::VisionSegmenter::available() {
            return Some(Self::Vision(vision::VisionSegmenter::default()));
        }
        #[cfg(feature = "onnx")]
        return runtime::fastest(true, true).map(|(segmenter, _)| Self::Ort(Box::new(segmenter)));
        #[cfg(not(feature = "onnx"))]
        None
    }

    /// The `parts` in `image`: each face and then its parts, then the bodies, each with a new number.
    pub fn find(&mut self, image: &Image, parts: Parts) -> &[Segment] {
        match self {
            #[cfg(target_os = "macos")]
            Self::Vision(segmenter) => segmenter.find(image, parts),
            #[cfg(feature = "onnx")]
            Self::Ort(segmenter) => segmenter.find(image, parts),
            #[cfg(not(any(target_os = "macos", feature = "onnx")))]
            _ => unreachable!("without a backend, best() makes no segmenter: {image:?} {parts:?}"),
        }
    }

    /// Numbers faces after `last` from now on, so numbers stay unique across segmenters.
    pub fn continue_after(&mut self, last: u64) {
        match self {
            #[cfg(target_os = "macos")]
            Self::Vision(segmenter) => segmenter.continue_after(last),
            #[cfg(feature = "onnx")]
            Self::Ort(segmenter) => segmenter.continue_after(last),
            #[cfg(not(any(target_os = "macos", feature = "onnx")))]
            _ => unreachable!("without a backend, best() makes no segmenter: {last}"),
        }
    }
}

/// Finds faces in a frame, best first.
pub trait Detector {
    fn detect(&mut self, image: &Image) -> Vec<Face>;
}

/// Face landmarks normalized to the frame, indexed as MediaPipe Face Mesh's.
pub trait Landmarker {
    fn landmarks(&mut self, image: &Image, face: &Face) -> Option<Vec<(f64, f64)>>;
}

/// Finds people in a frame, best first.
pub trait BodyDetector: Send {
    fn detect(&mut self, image: &Image) -> Vec<Body>;
}

/// Turns detections into segments, sized to match segcam 1.1.
pub struct FaceSegmenter<D, M, B> {
    pub detector: D,
    pub mesh: Option<M>,
    pub bodies: Option<B>,
    next_number: u64,
    segments: Vec<Segment>,
}

/// Face boxes are square, as tall as YuNet's box and moved down by this fraction of its height.
const FACE_DROP: f64 = 0.09;
/// The smallest eye or mouth box, as a fraction of the frame.
const MIN_PART: f64 = 0.02;

// Eye and mouth sizes relative to the face when there is no mesh, fitted to segcam 1.1.
const EYE_WIDTH: f64 = 0.26;
const EYE_HEIGHT: f64 = 0.18;
const MOUTH_WIDTH: f64 = 1.35;
const MOUTH_HEIGHT: f64 = 0.2;
/// The mouth box sits below the line between the mouth corners by this fraction of its height.
const MOUTH_DROP: f64 = 0.1;
/// Eye and mouth boxes are the mesh's outlines scaled by these factors.
const EYE_OUTLINE_SCALE: f64 = 1.7;
const LIP_OUTLINE_SCALE: f64 = 1.27;

impl<D: Detector, M: Landmarker, B: BodyDetector> FaceSegmenter<D, M, B> {
    pub fn new(detector: D, mesh: Option<M>, bodies: Option<B>) -> Self {
        Self {
            detector,
            mesh,
            bodies,
            next_number: 0,
            segments: Vec::new(),
        }
    }

    /// Numbers faces after `last` from now on, so numbers stay unique across segmenters.
    pub fn continue_after(&mut self, last: u64) {
        self.next_number = self.next_number.max(last);
    }

    /// The `parts` in `image`: each face and then its parts, then the bodies, each with a new number.
    pub fn find(&mut self, image: &Image, parts: Parts) -> &[Segment] {
        self.segments.clear();
        let mut bodies = self.bodies.take();
        let found = std::thread::scope(|scope| {
            let finding = bodies
                .as_mut()
                .filter(|_| parts.body)
                .map(|b| scope.spawn(|| b.detect(image)));
            if parts.face || parts.eyes || parts.mouth {
                for face in &self.detector.detect(image) {
                    self.push_face(image, face, parts);
                }
            }
            finding
                .map(|f| f.join().unwrap_or_default())
                .unwrap_or_default()
        });
        self.bodies = bodies;
        for body in found {
            self.next_number += 1;
            self.push(Kind::Body, body.rect, body.score);
        }
        &self.segments
    }

    fn push_face(&mut self, image: &Image, face: &Face, parts: Parts) {
        self.next_number += 1;
        let r = face.rect;
        if parts.face {
            let w = r.h * image.height as f64 / image.width.max(1) as f64;
            let square = Rect {
                x: r.x + (r.w - w) / 2.0,
                y: r.y + r.h * FACE_DROP,
                w,
                h: r.h,
            };
            self.push(Kind::Face, square.clamped(), face.score);
        }
        let outline = if parts.eyes || parts.mouth {
            self.mesh.as_mut().and_then(|m| m.landmarks(image, face))
        } else {
            None
        };
        if let Some(points) = outline {
            let bound = |indices: &[usize], scale: f64| {
                part_around(indices.iter().map(|&i| points[i]), scale)
            };
            if parts.eyes {
                let left = bound(&mesh::LEFT_EYE, EYE_OUTLINE_SCALE);
                let right = bound(&mesh::RIGHT_EYE, EYE_OUTLINE_SCALE);
                self.push(Kind::EyeLeft, left, face.score);
                self.push(Kind::EyeRight, right, face.score);
            }
            if parts.mouth {
                let mouth = bound(&mesh::OUTER_LIPS, LIP_OUTLINE_SCALE);
                self.push(Kind::Mouth, mouth, face.score);
            }
            return;
        }

        let [left_eye, right_eye, _nose, mouth_left, mouth_right] = face.landmarks;
        if parts.eyes {
            let (w, h) = (r.w * EYE_WIDTH, r.h * EYE_HEIGHT);
            self.push(Kind::EyeLeft, part(left_eye, w, h), face.score);
            self.push(Kind::EyeRight, part(right_eye, w, h), face.score);
        }
        if parts.mouth {
            let (w, h) = (
                (mouth_right.0 - mouth_left.0).abs() * MOUTH_WIDTH,
                r.h * MOUTH_HEIGHT,
            );
            let centre = (
                (mouth_left.0 + mouth_right.0) / 2.0,
                (mouth_left.1 + mouth_right.1) / 2.0 + h * MOUTH_DROP,
            );
            self.push(Kind::Mouth, part(centre, w, h), face.score);
        }
    }

    fn push(&mut self, kind: Kind, rect: Rect, score: f32) {
        self.segments.push(Segment {
            kind,
            number: self.next_number,
            rect,
            score,
        });
    }
}

/// The bounding box of `points`, scaled about its centre.
fn part_around(points: impl Iterator<Item = (f64, f64)>, scale: f64) -> Rect {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for (x, y) in points {
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    part(
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        (x1 - x0) * scale,
        (y1 - y0) * scale,
    )
}

/// A `w` x `h` box around `centre`, at least [`MIN_PART`] on each side and clamped to the frame.
fn part(centre: (f64, f64), w: f64, h: f64) -> Rect {
    let (w, h) = (w.max(MIN_PART), h.max(MIN_PART));
    Rect {
        x: centre.0 - w / 2.0,
        y: centre.1 - h / 2.0,
        w,
        h,
    }
    .clamped()
}
