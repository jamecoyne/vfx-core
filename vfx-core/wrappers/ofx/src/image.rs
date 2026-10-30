//! Moves pixels between OpenFX images and the straight-alpha RGBA f32 canvases that
//! the pipeline renders. OpenFX images have their origin at the bottom left and may
//! store rows bottom-up or top-down (negative row bytes). Canvases are top-down, like
//! the other wrappers, so a canvas row is `canvas.y2 - 1 - y` for pixel row `y`.

/// A pixel rectangle, OpenFX style: x1/y1 inclusive, x2/y2 exclusive, y up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl Rect {
    pub fn width(&self) -> usize {
        (self.x2 - self.x1).max(0) as usize
    }

    pub fn height(&self) -> usize {
        (self.y2 - self.y1).max(0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    pub fn grow(&self, x: i32, y: i32) -> Rect {
        Rect {
            x1: self.x1 - x,
            y1: self.y1 - y,
            x2: self.x2 + x,
            y2: self.y2 + y,
        }
    }

    pub fn intersect(&self, o: &Rect) -> Rect {
        Rect {
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
            x2: self.x2.min(o.x2),
            y2: self.y2.min(o.y2),
        }
    }

    pub fn union(&self, o: &Rect) -> Rect {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        Rect {
            x1: self.x1.min(o.x1),
            y1: self.y1.min(o.y1),
            x2: self.x2.max(o.x2),
            y2: self.y2.max(o.y2),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    Byte,
    Short,
    Float,
}

impl Depth {
    fn bytes_per_channel(self) -> usize {
        match self {
            Depth::Byte => 1,
            Depth::Short => 2,
            Depth::Float => 4,
        }
    }
}

/// An RGBA OpenFX image as the host describes it.
#[derive(Clone, Copy, Debug)]
pub struct Image {
    pub data: *mut u8,
    pub bounds: Rect,
    /// Bytes from one row to the next row up. May be negative.
    pub row_bytes: isize,
    pub depth: Depth,
    pub premultiplied: bool,
}

impl Image {
    /// Byte offset of pixel (x, y), which must lie inside `bounds`.
    fn offset(&self, x: i32, y: i32) -> isize {
        let bpp = 4 * self.depth.bytes_per_channel() as isize;
        (y - self.bounds.y1) as isize * self.row_bytes + (x - self.bounds.x1) as isize * bpp
    }
}

/// Reads `image` into a straight-alpha canvas covering `canvas`. Pixels outside the
/// image's bounds are transparent.
///
/// # Safety
/// `image.data` must point to `bounds` worth of pixels laid out as described.
pub unsafe fn decode(image: &Image, canvas: Rect) -> Vec<f32> {
    let width = canvas.width();
    let mut out = vec![0.0f32; width * canvas.height() * 4];
    let area = canvas.intersect(&image.bounds);
    for y in area.y1..area.y2 {
        let row = (canvas.y2 - 1 - y) as usize;
        for x in area.x1..area.x2 {
            // SAFETY: (x, y) is inside the image's bounds.
            let mut px = unsafe { read_pixel(image, x, y) };
            let alpha = px[3];
            if image.premultiplied && alpha > 0.0 {
                for c in &mut px[..3] {
                    *c /= alpha;
                }
            }
            let i = (row * width + (x - canvas.x1) as usize) * 4;
            out[i..i + 4].copy_from_slice(&px);
        }
    }
    out
}

/// Writes the `window` part of a canvas covering `canvas` into `image`, premultiplying
/// if the image is premultiplied. Integer depths clamp to 0..=1; floats don't.
///
/// # Safety
/// `image.data` must point to writable pixels covering `window`, which must lie inside
/// both `canvas` and the image's bounds.
pub unsafe fn encode(pixels: &[f32], canvas: Rect, image: &Image, window: Rect) {
    let width = canvas.width();
    for y in window.y1..window.y2 {
        let row = (canvas.y2 - 1 - y) as usize;
        for x in window.x1..window.x2 {
            let i = (row * width + (x - canvas.x1) as usize) * 4;
            let mut px = [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]];
            if image.depth != Depth::Float {
                for c in &mut px {
                    *c = c.clamp(0.0, 1.0);
                }
            }
            if image.premultiplied {
                for c in 0..3 {
                    px[c] *= px[3];
                }
            }
            // SAFETY: (x, y) is inside the window, which the caller keeps inside bounds.
            unsafe { write_pixel(image, x, y, px) };
        }
    }
}

unsafe fn read_pixel(image: &Image, x: i32, y: i32) -> [f32; 4] {
    // SAFETY: the caller checked (x, y) is inside the image.
    let p = unsafe { image.data.offset(image.offset(x, y)) };
    let mut px = [0.0f32; 4];
    for (c, v) in px.iter_mut().enumerate() {
        // SAFETY: four channels of `depth` follow `p`. Reads are unaligned-safe.
        *v = unsafe {
            match image.depth {
                Depth::Byte => f32::from(*p.add(c)) / 255.0,
                Depth::Short => f32::from(p.cast::<u16>().add(c).read_unaligned()) / 65535.0,
                Depth::Float => p.cast::<f32>().add(c).read_unaligned(),
            }
        };
    }
    px
}

unsafe fn write_pixel(image: &Image, x: i32, y: i32, px: [f32; 4]) {
    // SAFETY: the caller checked (x, y) is inside the image.
    let p = unsafe { image.data.offset(image.offset(x, y)) };
    for (c, v) in px.into_iter().enumerate() {
        // SAFETY: four writable channels of `depth` follow `p`.
        unsafe {
            match image.depth {
                Depth::Byte => *p.add(c) = (v * 255.0).round() as u8,
                Depth::Short => p
                    .cast::<u16>()
                    .add(c)
                    .write_unaligned((v * 65535.0).round() as u16),
                Depth::Float => p.cast::<f32>().add(c).write_unaligned(v),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x1: i32, y1: i32, x2: i32, y2: i32) -> Rect {
        Rect { x1, y1, x2, y2 }
    }

    /// A 2x2 byte image whose rows are stored bottom-up (positive row bytes): the
    /// bottom row has red 10, 20 and the top row red 30, 40.
    fn bottom_up(data: &mut [u8; 16]) -> Image {
        *data = [
            10, 0, 0, 255, 20, 0, 0, 255, // y = 0 (bottom)
            30, 0, 0, 255, 40, 0, 0, 255, // y = 1 (top)
        ];
        Image {
            data: data.as_mut_ptr(),
            bounds: rect(0, 0, 2, 2),
            row_bytes: 8,
            depth: Depth::Byte,
            premultiplied: false,
        }
    }

    fn red(canvas: &[f32], width: usize, col: usize, row: usize) -> u8 {
        (canvas[(row * width + col) * 4] * 255.0).round() as u8
    }

    #[test]
    fn canvas_rows_run_top_down() {
        let mut data = [0; 16];
        let img = bottom_up(&mut data);
        let canvas = unsafe { decode(&img, img.bounds) };
        assert_eq!(red(&canvas, 2, 0, 0), 30, "canvas row 0 is the top row");
        assert_eq!(red(&canvas, 2, 1, 1), 20);
    }

    #[test]
    fn negative_row_bytes_store_rows_top_down() {
        // Same picture, but the data pointer is at the bottom row and rows go up in
        // memory by -8 bytes: the top row comes first in memory.
        let mut data = [
            30u8, 0, 0, 255, 40, 0, 0, 255, // y = 1 (top)
            10, 0, 0, 255, 20, 0, 0, 255, // y = 0 (bottom)
        ];
        let img = Image {
            data: unsafe { data.as_mut_ptr().add(8) },
            bounds: rect(0, 0, 2, 2),
            row_bytes: -8,
            depth: Depth::Byte,
            premultiplied: false,
        };
        let canvas = unsafe { decode(&img, img.bounds) };
        assert_eq!(red(&canvas, 2, 0, 0), 30);
        assert_eq!(red(&canvas, 2, 1, 1), 20);
    }

    #[test]
    fn larger_canvas_is_transparent_outside_the_image() {
        let mut data = [0; 16];
        let img = bottom_up(&mut data);
        let canvas = unsafe { decode(&img, rect(-1, -1, 3, 3)) };
        assert_eq!(canvas.len(), 4 * 4 * 4);
        assert_eq!(&canvas[0..4], &[0.0; 4]);
        // Pixel (0, 1) lands at canvas column 1, row 3 - 1 - 1 = 1.
        assert_eq!(red(&canvas, 4, 1, 1), 30);
    }

    #[test]
    fn round_trip_through_a_window() {
        let mut data = [0; 16];
        let img = bottom_up(&mut data);
        let canvas = unsafe { decode(&img, img.bounds) };
        let mut out = [0u8; 16];
        let dst = Image {
            data: out.as_mut_ptr(),
            ..img
        };
        unsafe { encode(&canvas, img.bounds, &dst, rect(1, 0, 2, 2)) };
        assert_eq!(out[4], 20, "right column written");
        assert_eq!(out[12], 40);
        assert_eq!(out[0], 0, "left column outside the window untouched");
    }

    #[test]
    fn premultiplied_images_are_unpremultiplied_and_back() {
        // Half-transparent pixel, premultiplied: stored red 0.4 means straight red 0.8.
        let mut data: Vec<u8> = [0.4f32, 0.0, 0.0, 0.5]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let img = Image {
            data: data.as_mut_ptr(),
            bounds: rect(0, 0, 1, 1),
            row_bytes: 16,
            depth: Depth::Float,
            premultiplied: true,
        };
        let canvas = unsafe { decode(&img, img.bounds) };
        assert!((canvas[0] - 0.8).abs() < 1e-6);
        let mut out = vec![0u8; 16];
        let dst = Image {
            data: out.as_mut_ptr(),
            ..img
        };
        unsafe { encode(&canvas, img.bounds, &dst, img.bounds) };
        assert_eq!(out, data);
    }

    #[test]
    fn short_scales_by_65535_and_floats_stay_unclamped() {
        let mut data: Vec<u8> = [65535u16, 0, 32768, 65535]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let img = Image {
            data: data.as_mut_ptr(),
            bounds: rect(0, 0, 1, 1),
            row_bytes: 8,
            depth: Depth::Short,
            premultiplied: false,
        };
        let canvas = unsafe { decode(&img, img.bounds) };
        assert_eq!(canvas[0], 1.0);
        assert!((canvas[2] - 0.5).abs() < 1e-4);

        let mut out = vec![0u8; 16];
        let float = Image {
            data: out.as_mut_ptr(),
            bounds: rect(0, 0, 1, 1),
            row_bytes: 16,
            depth: Depth::Float,
            premultiplied: false,
        };
        unsafe { encode(&[2.5, -0.5, 0.0, 1.0], float.bounds, &float, float.bounds) };
        let r = f32::from_ne_bytes(out[0..4].try_into().unwrap());
        assert_eq!(r, 2.5);
    }
}
