//! The portrait the face tests paste into frames.

use segcam_motion::Argb;
use vfx_host::contract::Clip;

pub const W: i32 = 320;
pub const H: i32 = 180;

/// A portrait to paste into frames: RGB pixels, width and height.
pub struct Portrait {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// The portrait `SEGCAM_FACE_PORTRAIT` names, or None with a note when it isn't set.
pub fn portrait() -> Option<Portrait> {
    let Ok(path) = std::env::var("SEGCAM_FACE_PORTRAIT") else {
        eprintln!("skipped: set SEGCAM_FACE_PORTRAIT to a P6 PPM of a face");
        return None;
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut fields = Vec::new();
    let mut at = 0;
    while fields.len() < 4 {
        while bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        let from = at;
        while !bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        fields.push(std::str::from_utf8(&bytes[from..at]).unwrap().to_string());
    }
    assert_eq!(fields[0], "P6", "{path}: not a binary PPM");
    let (width, height): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    assert!(height <= H as usize, "{path}: taller than {H}");
    let rgb = bytes[at + 1..].to_vec();
    assert_eq!(rgb.len(), width * height * 3, "{path}: size");
    Some(Portrait { rgb, width, height })
}

/// A frame with the portrait centred on (cx, cy) over a gradient, and a bright square at
/// `square`'s x, if any.
pub fn frame(p: &Portrait, cx: i32, cy: i32, square: Option<i32>) -> Vec<Argb<u8>> {
    let (left, top) = (cx - p.width as i32 / 2, cy - p.height as i32 / 2);
    (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            if square.is_some_and(|s| (s..s + 24).contains(&x) && (130..154).contains(&y)) {
                return Argb {
                    alpha: 255,
                    red: 245,
                    green: 245,
                    blue: 245,
                };
            }
            let (px, py) = (x - left, y - top);
            if (0..p.width as i32).contains(&px) && (0..p.height as i32).contains(&py) {
                let at = (py as usize * p.width + px as usize) * 3;
                return Argb {
                    alpha: 255,
                    red: p.rgb[at],
                    green: p.rgb[at + 1],
                    blue: p.rgb[at + 2],
                };
            }
            Argb {
                alpha: 255,
                red: (90 + y / 5) as u8,
                green: (100 + y / 5) as u8,
                blue: 120,
            }
        })
        .collect()
}

/// The face drifting right, with a square crossing below it in some frames, so frames differ in
/// how many motion segments come before the faces. An even count, so the last frame is odd and
/// the contract's missing frames include the one before it.
pub fn drifting(p: &Portrait) -> Clip {
    let frames = (0..10)
        .map(|f: i32| {
            let square = (2..7).contains(&f).then_some(20 + f * 30);
            frame(p, 120 + f * 6, 80, square)
        })
        .collect();
    Clip {
        width: W,
        height: H,
        frames,
        frame_rate: None,
    }
}
