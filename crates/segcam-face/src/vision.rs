//! Faces, eyes, mouths and bodies through Apple's Vision, as segcam 1.1 finds them.

use std::ffi::c_void;
use std::ptr;

use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::msg_send;
use objc2::rc::{Allocated, Retained, autoreleasepool};
use objc2::runtime::{AnyClass, AnyObject};

use crate::{Image, Kind, MIN_PART, Parts, Rect, Segment};

/// 1.1 gives Vision 1280x720 camera frames, so wider frames are scaled down to this.
const MAX_WIDTH: usize = 1280;
/// 1.1 pads each box by this fraction of its size on each side.
const FACE_PADDING: f64 = 0.05;
const EYE_PADDING: f64 = 0.4;
const MOUTH_PADDING: f64 = 0.15;
/// `'BGRA'`.
const PIXEL_FORMAT_BGRA: u32 = 0x4247_5241;

#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGSize {
    w: f64,
    h: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGRect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

// SAFETY: these match CoreGraphics' layouts and type encodings.
unsafe impl Encode for CGPoint {
    const ENCODING: Encoding = Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl RefEncode for CGPoint {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}
unsafe impl Encode for CGSize {
    const ENCODING: Encoding = Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl Encode for CGRect {
    const ENCODING: Encoding = Encoding::Struct("CGRect", &[CGPoint::ENCODING, CGSize::ENCODING]);
}

/// CoreVideo's opaque pixel buffer.
#[repr(C)]
struct CVBuffer {
    _private: [u8; 0],
}

// SAFETY: matches CoreVideo's `CVBufferRef` encoding.
unsafe impl RefEncode for CVBuffer {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("__CVBuffer", &[]));
}

#[link(name = "Vision", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    fn CVPixelBufferCreateWithBytes(
        allocator: *const c_void,
        width: usize,
        height: usize,
        format: u32,
        base: *mut c_void,
        bytes_per_row: usize,
        release: *const c_void,
        release_context: *const c_void,
        attributes: *const c_void,
        out: *mut *mut CVBuffer,
    ) -> i32;
    fn CVPixelBufferRelease(buffer: *mut CVBuffer);
}

/// Turns Vision's faces, landmarks and people into segments.
#[derive(Default)]
pub struct VisionSegmenter {
    next_number: u64,
    segments: Vec<Segment>,
    bgra: Vec<u8>,
}

impl VisionSegmenter {
    /// Whether Vision can run here.
    pub fn available() -> bool {
        AnyClass::get(c"VNDetectFaceLandmarksRequest").is_some()
    }

    /// Numbers faces after `last` from now on, so numbers stay unique across segmenters.
    pub fn continue_after(&mut self, last: u64) {
        self.next_number = self.next_number.max(last);
    }

    /// The `parts` in `image`: each face and then its parts, then the bodies, each with a new number.
    pub fn find(&mut self, image: &Image, parts: Parts) -> &[Segment] {
        self.segments.clear();
        let faces = parts.face || parts.eyes || parts.mouth;
        if !image.is_usable() || !(faces || parts.body) {
            return &self.segments;
        }
        let (width, height) = self.fill_bgra(image);
        let found = detect(&mut self.bgra, width, height, faces, parts.body);
        for face in &found.faces {
            self.push_face(face, parts);
        }
        for &(rect, score) in &found.bodies {
            self.next_number += 1;
            self.push(Kind::Body, rect, score);
        }
        &self.segments
    }

    fn fill_bgra(&mut self, image: &Image) -> (usize, usize) {
        let step = image.width.div_ceil(MAX_WIDTH).max(1);
        let (width, height) = (image.width / step, image.height / step);
        self.bgra.clear();
        self.bgra.reserve(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                let [r, g, b] = image.rgb(x * step, y * step);
                self.bgra.extend([b as u8, g as u8, r as u8, 255]);
            }
        }
        (width, height)
    }

    fn push_face(&mut self, face: &VisionFace, parts: Parts) {
        self.next_number += 1;
        if parts.face {
            self.push(Kind::Face, padded(face.rect, FACE_PADDING), face.score);
        }
        let regions = [
            (parts.eyes, Kind::EyeLeft, &face.left_eye, EYE_PADDING),
            (parts.eyes, Kind::EyeRight, &face.right_eye, EYE_PADDING),
            (parts.mouth, Kind::Mouth, &face.outer_lips, MOUTH_PADDING),
        ];
        for (wanted, kind, points, padding) in regions {
            if wanted && !points.is_empty() {
                self.push(kind, padded_part(points, padding), face.score);
            }
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

/// `r` grown by `padding` of its size on each side, clamped to the frame.
fn padded(r: Rect, padding: f64) -> Rect {
    Rect {
        x: r.x - r.w * padding,
        y: r.y - r.h * padding,
        w: r.w * (1.0 + 2.0 * padding),
        h: r.h * (1.0 + 2.0 * padding),
    }
    .clamped()
}

/// The bounding box of `points`, padded, then grown to the minimum part size about its centre.
fn padded_part(points: &[(f64, f64)], padding: f64) -> Rect {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for &(x, y) in points {
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    let bounds = Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
    .clamped();
    let r = padded(bounds, padding);
    let (w, h) = (r.w.max(MIN_PART), r.h.max(MIN_PART));
    Rect {
        x: r.x - (w - r.w) / 2.0,
        y: r.y - (h - r.h) / 2.0,
        w,
        h,
    }
    .clamped()
}

struct VisionFace {
    rect: Rect,
    score: f32,
    left_eye: Vec<(f64, f64)>,
    right_eye: Vec<(f64, f64)>,
    outer_lips: Vec<(f64, f64)>,
}

#[derive(Default)]
struct Found {
    faces: Vec<VisionFace>,
    bodies: Vec<(Rect, f32)>,
}

/// Vision's box, flipped to a top-left origin.
fn flipped(b: CGRect) -> Rect {
    Rect {
        x: b.x,
        y: 1.0 - b.y - b.h,
        w: b.w,
        h: b.h,
    }
}

/// Runs 1.1's requests on one frame: face landmarks and whole-body human rectangles.
fn detect(bgra: &mut [u8], width: usize, height: usize, faces: bool, bodies: bool) -> Found {
    let (Some(face_class), Some(body_class), Some(handler_class)) = (
        AnyClass::get(c"VNDetectFaceLandmarksRequest"),
        AnyClass::get(c"VNDetectHumanRectanglesRequest"),
        AnyClass::get(c"VNImageRequestHandler"),
    ) else {
        return Found::default();
    };
    if width == 0 || height == 0 {
        return Found::default();
    }
    autoreleasepool(|_| {
        // SAFETY: the pixel buffer borrows `bgra`, which outlives the requests, and every result
        // is read before the buffer is released.
        unsafe {
            let mut buffer: *mut CVBuffer = ptr::null_mut();
            let status = CVPixelBufferCreateWithBytes(
                ptr::null(),
                width,
                height,
                PIXEL_FORMAT_BGRA,
                bgra.as_mut_ptr().cast(),
                width * 4,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut buffer,
            );
            if status != 0 || buffer.is_null() {
                return Found::default();
            }
            let face_request: Retained<AnyObject> = msg_send![face_class, new];
            let body_request: Retained<AnyObject> = msg_send![body_class, new];
            let _: () = msg_send![&body_request, setUpperBodyOnly: false];
            let handler: Allocated<AnyObject> = msg_send![handler_class, alloc];
            let options: Retained<AnyObject> =
                msg_send![AnyClass::get(c"NSDictionary").unwrap(), dictionary];
            let handler: Retained<AnyObject> =
                msg_send![handler, initWithCVPixelBuffer: buffer, options: &*options];
            let requests: Retained<AnyObject> =
                msg_send![AnyClass::get(c"NSMutableArray").unwrap(), array];
            if faces {
                let _: () = msg_send![&requests, addObject: &*face_request];
            }
            if bodies {
                let _: () = msg_send![&requests, addObject: &*body_request];
            }
            let mut error: *mut AnyObject = ptr::null_mut();
            let ok: bool = msg_send![&handler, performRequests: &*requests, error: &mut error];
            let found = if ok {
                Found {
                    faces: if faces {
                        read_faces(&face_request, width as f64, height as f64)
                    } else {
                        Vec::new()
                    },
                    bodies: if bodies {
                        read_bodies(&body_request)
                    } else {
                        Vec::new()
                    },
                }
            } else {
                Found::default()
            };
            CVPixelBufferRelease(buffer);
            found
        }
    })
}

unsafe fn read_bodies(request: &AnyObject) -> Vec<(Rect, f32)> {
    // SAFETY: results is an NSArray of VNHumanObservation.
    unsafe {
        let results: Option<Retained<AnyObject>> = msg_send![request, results];
        let Some(results) = results else {
            return Vec::new();
        };
        let count: usize = msg_send![&results, count];
        (0..count)
            .filter_map(|i| {
                let body: Option<Retained<AnyObject>> = msg_send![&results, objectAtIndex: i];
                let body = body?;
                let b: CGRect = msg_send![&body, boundingBox];
                let score: f32 = msg_send![&body, confidence];
                Some((flipped(b), score))
            })
            .collect()
    }
}

unsafe fn read_faces(request: &AnyObject, width: f64, height: f64) -> Vec<VisionFace> {
    // SAFETY: results is an NSArray of VNFaceObservation, and each landmark region holds
    // `pointCount` CGPoints.
    unsafe {
        let results: Option<Retained<AnyObject>> = msg_send![request, results];
        let Some(results) = results else {
            return Vec::new();
        };
        let count: usize = msg_send![&results, count];
        (0..count)
            .filter_map(|i| {
                let face: Option<Retained<AnyObject>> = msg_send![&results, objectAtIndex: i];
                let face = face?;
                let b: CGRect = msg_send![&face, boundingBox];
                let score: f32 = msg_send![&face, confidence];
                let landmarks: Option<Retained<AnyObject>> = msg_send![&face, landmarks];
                let region = |name: &str| -> Vec<(f64, f64)> {
                    let Some(landmarks) = &landmarks else {
                        return Vec::new();
                    };
                    let region: Option<Retained<AnyObject>> = match name {
                        "leftEye" => msg_send![landmarks, leftEye],
                        "rightEye" => msg_send![landmarks, rightEye],
                        _ => msg_send![landmarks, outerLips],
                    };
                    let Some(region) = region else {
                        return Vec::new();
                    };
                    let n: usize = msg_send![&region, pointCount];
                    let size = CGSize {
                        w: width,
                        h: height,
                    };
                    let points: *const CGPoint = msg_send![&region, pointsInImageOfSize: size];
                    if points.is_null() {
                        return Vec::new();
                    }
                    std::slice::from_raw_parts(points, n)
                        .iter()
                        .map(|p| (p.x / width, 1.0 - p.y / height))
                        .collect()
                };
                Some(VisionFace {
                    rect: flipped(b),
                    score,
                    left_eye: region("leftEye"),
                    right_eye: region("rightEye"),
                    outer_lips: region("outerLips"),
                })
            })
            .collect()
    }
}
