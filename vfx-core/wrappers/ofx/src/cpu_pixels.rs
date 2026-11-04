//! For CPU effects: OpenFX images (RGBA, rows from the bottom) to and from top-down ARGB
//! canvases at the effect's depth, keeping the host's alpha convention.

use vfx_core::cpu::{Argb, Channel};
use vfx_host::by_rows;

use crate::image::{Depth, Image, Rect};

fn channel_bytes(depth: Depth) -> usize {
    match depth {
        Depth::Byte => 1,
        Depth::Short => 2,
        Depth::Float => 4,
    }
}
unsafe fn address(image: &Image, x: i32, y: i32) -> *mut u8 {
    // SAFETY: caller supplies coordinates inside the host image.
    unsafe {
        image.data.offset(
            (y - image.bounds.y1) as isize * image.row_bytes
                + (x - image.bounds.x1) as isize * (4 * channel_bytes(image.depth)) as isize,
        )
    }
}
/// A host image shared by the threads converting its rows; each writes rows of its own.
struct Shared<'a>(&'a Image);

// SAFETY: threads read the image, and write only disjoint rows of it.
unsafe impl Sync for Shared<'_> {}

impl Shared<'_> {
    fn image(&self) -> &Image {
        self.0
    }
}

/// The bytes of `pixels` pixels of host row `y`, from column `x`.
///
/// # Safety
/// The pixels must lie inside the live host image, and no other slice of them be in use.
#[expect(
    clippy::mut_from_ref,
    reason = "an Image describes host memory it doesn't own; rows are handed out one per thread"
)]
unsafe fn row_bytes(image: &Image, x: i32, y: i32, pixels: usize) -> &mut [u8] {
    let len = pixels * 4 * channel_bytes(image.depth);
    // SAFETY: the caller confines the row to the host image and uses it alone.
    unsafe { std::slice::from_raw_parts_mut(address(image, x, y), len) }
}

/// One host row's bytes (`depth`, RGBA, native byte order) into `row`, through `byte` for
/// 8-bit channels.
fn decode_row<C: Channel>(src: &[u8], depth: Depth, row: &mut [Argb<C>], byte: &[C; 256]) {
    let argb = |[r, g, b, a]: [C; 4]| Argb {
        alpha: a,
        red: r,
        green: g,
        blue: b,
    };
    match depth {
        Depth::Byte => {
            for (px, c) in row.iter_mut().zip(src.as_chunks::<4>().0) {
                *px = argb(c.map(|v| byte[usize::from(v)]));
            }
        }
        Depth::Short => {
            for (px, c) in row.iter_mut().zip(src.as_chunks::<8>().0) {
                let channels = c.as_chunks::<2>().0;
                *px = argb(std::array::from_fn(|k| {
                    C::from_unit(f32::from(u16::from_ne_bytes(channels[k])) / 65535.0)
                }));
            }
        }
        Depth::Float => {
            for (px, c) in row.iter_mut().zip(src.as_chunks::<16>().0) {
                let channels = c.as_chunks::<4>().0;
                *px = argb(std::array::from_fn(|k| {
                    C::from_unit(f32::from_ne_bytes(channels[k]))
                }));
            }
        }
    }
}

/// `row` into one host row's bytes (`depth`, RGBA, native byte order).
fn encode_row<C: Channel>(row: &[Argb<C>], depth: Depth, dst: &mut [u8]) {
    let rgba = |v: &Argb<C>| [v.red, v.green, v.blue, v.alpha].map(Channel::to_unit);
    match depth {
        Depth::Byte => {
            for (v, d) in row.iter().zip(dst.as_chunks_mut::<4>().0) {
                *d = rgba(v).map(|c| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
        Depth::Short => {
            for (v, d) in row.iter().zip(dst.as_chunks_mut::<8>().0) {
                for (out, c) in d.as_chunks_mut::<2>().0.iter_mut().zip(rgba(v)) {
                    *out = ((c.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_ne_bytes();
                }
            }
        }
        Depth::Float => {
            for (v, d) in row.iter().zip(dst.as_chunks_mut::<16>().0) {
                for (out, c) in d.as_chunks_mut::<4>().0.iter_mut().zip(rgba(v)) {
                    *out = c.to_ne_bytes();
                }
            }
        }
    }
}

/// The host's pixels on the canvas, top row first; canvas pixels outside the image are clear.
///
/// # Safety
/// `image` must describe a live host image.
#[cfg(test)]
pub unsafe fn decode<C: Channel>(image: &Image, canvas: Rect) -> Vec<Argb<C>> {
    let mut out = Vec::new();
    // SAFETY: as for this function.
    unsafe { decode_into(image, canvas, &mut out) };
    out
}

/// [`decode`] into `out`, reusing its allocation.
///
/// # Safety
/// `image` must describe a live host image.
pub unsafe fn decode_into<C: Channel>(image: &Image, canvas: Rect, out: &mut Vec<Argb<C>>) {
    let r = canvas.intersect(&image.bounds);
    out.clear();
    out.resize(canvas.width() * canvas.height(), Argb::default());
    let byte: [C; 256] = std::array::from_fn(|v| C::from_unit(v as f32 / 255.0));
    let shared = Shared(image);
    let skip = (r.x1 - canvas.x1).max(0) as usize;
    by_rows(out, canvas.width(), |row, pixels| {
        let y = canvas.y2 - 1 - row as i32;
        if (r.y1..r.y2).contains(&y) {
            let pixels = &mut pixels[skip..skip + r.width()];
            let image = shared.image();
            // SAFETY: the intersection confines the row to the host image, read by this
            // thread alone.
            let src = unsafe { row_bytes(image, r.x1, y, pixels.len()) };
            decode_row(src, image.depth, pixels, &byte);
        }
    });
}

/// Writes the canvas's `window` into the host image.
///
/// # Safety
/// `image` must describe a live host image, and `window` lie inside both it and `canvas`.
pub unsafe fn encode<C: Channel>(pixels: &[Argb<C>], canvas: Rect, image: &Image, window: Rect) {
    let shared = Shared(image);
    let skip = (window.x1 - canvas.x1) as usize;
    let mut rows: Vec<i32> = (window.y1..window.y2).collect();
    by_rows(&mut rows, 1, |_, y| {
        let y = y[0];
        let start = (canvas.y2 - 1 - y) as usize * canvas.width() + skip;
        let image = shared.image();
        // SAFETY: the caller confines the window to both canvas and output bounds, and each
        // row is written by one thread.
        let dst = unsafe { row_bytes(image, window.x1, y, window.width()) };
        encode_row(&pixels[start..start + window.width()], image.depth, dst);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A canvas covering `canvas`, from `image`; transparent where it doesn't reach.
    ///
    /// # Safety
    /// `image` must describe a live host image.
    unsafe fn decode_reference<C: Channel>(image: &Image, canvas: Rect) -> Vec<Argb<C>> {
        let mut out = vec![Argb::default(); canvas.width() * canvas.height()];
        let r = canvas.intersect(&image.bounds);
        for y in r.y1..r.y2 {
            for x in r.x1..r.x2 {
                // SAFETY: intersection confines the pointer and four channels to the host image.
                let p = unsafe { address(image, x, y) };
                let read = |i| unsafe {
                    match image.depth {
                        Depth::Byte => C::from_unit(f32::from(*p.add(i)) / 255.0),
                        Depth::Short => C::from_unit(
                            f32::from(p.add(i * 2).cast::<u16>().read_unaligned()) / 65535.0,
                        ),
                        Depth::Float => C::from_unit(p.add(i * 4).cast::<f32>().read_unaligned()),
                    }
                };
                out[(canvas.y2 - 1 - y) as usize * canvas.width() + (x - canvas.x1) as usize] =
                    Argb {
                        alpha: read(3),
                        red: read(0),
                        green: read(1),
                        blue: read(2),
                    };
            }
        }
        out
    }
    /// Writes the `window` part of a canvas covering `canvas` into `image`.
    ///
    /// # Safety
    /// `image` must describe a live host image, and `window` lie inside both it and `canvas`.
    unsafe fn encode_reference<C: Channel>(
        pixels: &[Argb<C>],
        canvas: Rect,
        image: &Image,
        window: Rect,
    ) {
        for y in window.y1..window.y2 {
            for x in window.x1..window.x2 {
                let v = pixels
                    [(canvas.y2 - 1 - y) as usize * canvas.width() + (x - canvas.x1) as usize];
                // SAFETY: caller confines the window to both canvas and output bounds.
                let p = unsafe { address(image, x, y) };
                for (i, c) in [v.red, v.green, v.blue, v.alpha].into_iter().enumerate() {
                    let c = c.to_unit();
                    // SAFETY: each write addresses one channel of the host pixel, possibly unaligned.
                    unsafe {
                        match image.depth {
                            Depth::Byte => *p.add(i) = (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                            Depth::Short => p
                                .add(i * 2)
                                .cast::<u16>()
                                .write_unaligned((c.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16),
                            Depth::Float => p.add(i * 4).cast::<f32>().write_unaligned(c),
                        }
                    }
                }
            }
        }
    }

    /// The conversions themselves, on bytes alone.
    #[test]
    fn rows_convert_without_a_host() {
        let byte: [f32; 256] = std::array::from_fn(|v| v as f32 / 255.0);
        let mut row = [Argb::<f32>::default(); 2];
        let shorts: Vec<u8> = [0u16, 65535, 32768, 65535, 65535, 0, 0, 13107]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        decode_row(&shorts, Depth::Short, &mut row, &byte);
        assert_eq!((row[0].red, row[0].green, row[0].alpha), (0.0, 1.0, 1.0));
        assert_eq!((row[1].red, row[1].alpha), (1.0, 0.2));
        let mut back = vec![0u8; shorts.len()];
        encode_row(&row, Depth::Short, &mut back);
        assert_eq!(back, shorts);
        let mut floats = vec![0u8; 32];
        encode_row(&row, Depth::Float, &mut floats);
        let mut again = [Argb::<f32>::default(); 2];
        decode_row(&floats, Depth::Float, &mut again, &byte);
        assert_eq!(again, row);
    }

    /// Row conversions match the per-pixel ones.
    #[test]
    fn row_conversions_match_the_per_pixel_ones() {
        let (w, h) = (37usize, 5usize);
        for depth in [Depth::Byte, Depth::Short, Depth::Float] {
            let bytes = 4 * channel_bytes(depth);
            let mut host: Vec<u8> = (0..w * h * bytes).map(|i| (i * 7919 % 256) as u8).collect();
            if depth == Depth::Float {
                for (i, c) in host.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    c.copy_from_slice(&((i % 300) as f32 / 255.0 - 0.1).to_le_bytes());
                }
            }
            let bounds = Rect {
                x1: 3,
                y1: -2,
                x2: 3 + w as i32,
                y2: -2 + h as i32,
            };
            let image = Image {
                data: host.as_mut_ptr(),
                bounds,
                row_bytes: (w * bytes) as isize,
                depth,
                premultiplied: true,
            };
            let canvas = Rect {
                x1: 0,
                y1: 0,
                x2: 30,
                y2: 4,
            };
            for_both(&image, canvas);
        }

        fn for_both(image: &Image, canvas: Rect) {
            unsafe {
                assert_eq!(
                    decode::<u8>(image, canvas),
                    decode_reference::<u8>(image, canvas)
                );
                assert_eq!(
                    decode::<f32>(image, canvas),
                    decode_reference::<f32>(image, canvas)
                );
            }
            let pixels: Vec<Argb<f32>> = (0..canvas.width() * canvas.height())
                .map(|i| {
                    let v = (i % 290) as f32 / 255.0 - 0.05;
                    Argb {
                        alpha: v,
                        red: 1.0 - v,
                        green: v * 0.5,
                        blue: v + 0.01,
                    }
                })
                .collect();
            let window = canvas.intersect(&image.bounds);
            let size = image.row_bytes as usize * image.bounds.height();
            let read =
                |image: &Image| unsafe { std::slice::from_raw_parts(image.data, size).to_vec() };
            let before = read(image);
            unsafe { encode_reference(&pixels, canvas, image, window) };
            let expected = read(image);
            unsafe { std::ptr::copy_nonoverlapping(before.as_ptr(), image.data, size) };
            unsafe { encode(&pixels, canvas, image, window) };
            assert_eq!(read(image), expected);
        }
    }
}
