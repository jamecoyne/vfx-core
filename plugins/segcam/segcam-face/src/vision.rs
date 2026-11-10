//! Faces, eyes, mouths and bodies through Apple's Vision, as segcam 1.1 finds them.

use std::cmp::Ordering;
use std::ptr::{self, NonNull};

use objc2::msg_send;
use objc2::rc::{Allocated, Retained, autoreleasepool};
use objc2::runtime::{AnyClass, AnyObject};
use objc2_core_foundation::{CFRetained, CGPoint, CGRect, CGSize};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreateWithBytes, kCVPixelFormatType_32BGRA, kCVReturnSuccess,
};
// Links Vision, whose classes are looked up by name below.
use objc2_vision as _;

use crate::{Image, Kind, MIN_PART, Parts, Rect, Segment};

/// 1.1 gives Vision 1280x720 camera frames, so wider frames are scaled down to this.
const MAX_WIDTH: usize = 1280;
/// 1.1 pads each box by this fraction of its size on each side.
const FACE_PADDING: f64 = 0.05;
const EYE_PADDING: f64 = 0.4;
const MOUTH_PADDING: f64 = 0.15;

/// Turns Vision's faces, landmarks and people into segments.
#[derive(Default)]
pub struct VisionSegmenter {
    segments: Vec<Segment>,
    bgra: Vec<u8>,
}

impl VisionSegmenter {
    /// Whether Vision can run here.
    pub fn available() -> bool {
        AnyClass::get(c"VNDetectFaceLandmarksRequest").is_some()
    }

    /// The `parts` in `image`: each face and then its parts, then the bodies.
    pub fn find(&mut self, image: &Image, parts: Parts) -> &[Segment] {
        self.segments.clear();
        let faces = parts.face || parts.eyes || parts.mouth;
        if !image.is_usable() || !(faces || parts.body) {
            return &self.segments;
        }
        let (width, height) = self.fill_bgra(image);
        let mut found = detect(&mut self.bgra, width, height, faces, parts.body);
        // Vision's order changes between runs on the same frame, and IDs and stacking follow it.
        found.faces.sort_by(|a, b| biggest_first(&a.rect, &b.rect));
        found.bodies.sort_by(biggest_first);
        for face in &found.faces {
            self.push_face(face, parts);
        }
        for &rect in &found.bodies {
            self.push(Kind::Body, rect);
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
        if parts.face {
            self.push(Kind::Face, padded(face.rect, FACE_PADDING));
            if let Some(pushed) = self.segments.last_mut() {
                pushed.detected = Some(face.rect);
            }
        }
        let regions = [
            (parts.eyes, Kind::EyeLeft, &face.left_eye, EYE_PADDING),
            (parts.eyes, Kind::EyeRight, &face.right_eye, EYE_PADDING),
            (parts.mouth, Kind::Mouth, &face.outer_lips, MOUTH_PADDING),
        ];
        for (wanted, kind, points, padding) in regions {
            if wanted && !points.is_empty() {
                self.push(kind, padded_part(points, padding));
            }
        }
    }

    fn push(&mut self, kind: Kind, rect: Rect) {
        self.segments.push(Segment {
            kind,
            rect,
            detected: None,
            landmarks: None,
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
    let r = padded(Rect::bounding(points.iter().copied()).clamped(), padding);
    let (w, h) = (r.w.max(MIN_PART), r.h.max(MIN_PART));
    Rect {
        x: r.x - (w - r.w) / 2.0,
        y: r.y - (h - r.h) / 2.0,
        w,
        h,
    }
    .clamped()
}

/// Biggest first, as the motion segments are. Ties go by position.
fn biggest_first(a: &Rect, b: &Rect) -> Ordering {
    (b.w * b.h)
        .total_cmp(&(a.w * a.h))
        .then(a.x.total_cmp(&b.x))
        .then(a.y.total_cmp(&b.y))
}

struct VisionFace {
    rect: Rect,
    left_eye: Vec<(f64, f64)>,
    right_eye: Vec<(f64, f64)>,
    outer_lips: Vec<(f64, f64)>,
}

#[derive(Default)]
struct Found {
    faces: Vec<VisionFace>,
    bodies: Vec<Rect>,
}

/// Vision's box, flipped to a top-left origin.
fn flipped(b: CGRect) -> Rect {
    Rect {
        x: b.origin.x,
        y: 1.0 - b.origin.y - b.size.height,
        w: b.size.width,
        h: b.size.height,
    }
}

/// A BGRA pixel buffer over `bgra`, or `None` if `bgra` is short.
///
/// # Safety
///
/// `bgra` must outlive the buffer and all that retain it, and stay unwritten.
unsafe fn pixel_buffer(
    bgra: &mut [u8],
    width: usize,
    height: usize,
) -> Option<CFRetained<CVPixelBuffer>> {
    if bgra.len() < width.checked_mul(height)?.checked_mul(4)? {
        return None;
    }
    let mut buffer = ptr::null_mut();
    // SAFETY: `bgra` holds `height` rows of `width * 4` bytes, checked above, and
    // outlives the buffer. With no release callback, CoreVideo leaves the memory
    // to the caller.
    let status = unsafe {
        CVPixelBufferCreateWithBytes(
            None,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            NonNull::from(bgra).cast(),
            width * 4,
            None,
            ptr::null_mut(),
            None,
            NonNull::from(&mut buffer),
        )
    };
    if status != kCVReturnSuccess {
        return None;
    }
    // SAFETY: a Create function returns the buffer retained once, for the caller.
    NonNull::new(buffer).map(|buffer| unsafe { CFRetained::from_raw(buffer) })
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
        // SAFETY: `bgra` outlives the pool, so it outlives the buffer and anything
        // that retains it, and nothing writes to it until `detect` returns.
        let Some(buffer) = (unsafe { pixel_buffer(bgra, width, height) }) else {
            return Found::default();
        };
        // SAFETY: the selectors and types match Vision's headers, and every result
        // is read before the pool drains.
        unsafe {
            let face_request: Retained<AnyObject> = msg_send![face_class, new];
            let body_request: Retained<AnyObject> = msg_send![body_class, new];
            let _: () = msg_send![&body_request, setUpperBodyOnly: false];
            let handler: Allocated<AnyObject> = msg_send![handler_class, alloc];
            let options: Retained<AnyObject> =
                msg_send![AnyClass::get(c"NSDictionary").unwrap(), dictionary];
            let handler: Retained<AnyObject> =
                msg_send![handler, initWithCVPixelBuffer: &*buffer, options: &*options];
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
            if ok {
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
            }
        }
    })
}

unsafe fn read_bodies(request: &AnyObject) -> Vec<Rect> {
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
                Some(flipped(b))
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
                    let size = CGSize { width, height };
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
                    left_eye: region("leftEye"),
                    right_eye: region("rightEye"),
                    outer_lips: region("outerLips"),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_video::{
        CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight, CVPixelBufferGetWidth,
    };

    #[test]
    fn pixel_buffer_wraps_bgra() {
        let mut bgra = vec![0u8; 4 * 2 * 4];
        // SAFETY: `bgra` outlives `buffer` and is not written while it lives.
        let buffer = unsafe { pixel_buffer(&mut bgra, 4, 2) }.unwrap();
        assert_eq!(CVPixelBufferGetWidth(&buffer), 4);
        assert_eq!(CVPixelBufferGetHeight(&buffer), 2);
        assert_eq!(CVPixelBufferGetBytesPerRow(&buffer), 16);
    }

    #[test]
    fn pixel_buffer_refuses_short_bgra() {
        let mut bgra = vec![0u8; 4 * 2 * 4 - 1];
        // SAFETY: no buffer is made.
        assert!(unsafe { pixel_buffer(&mut bgra, 4, 2) }.is_none());
    }

    #[test]
    fn biggest_first_ignores_the_order_given() {
        let rect = |x, w| Rect {
            x,
            y: 0.1,
            w,
            h: 0.2,
        };
        let (big, left, right) = (rect(0.5, 0.4), rect(0.1, 0.2), rect(0.3, 0.2));
        for mut rects in [[big, left, right], [right, big, left], [left, right, big]] {
            rects.sort_by(biggest_first);
            assert_eq!(rects, [big, left, right]);
        }
    }
}
