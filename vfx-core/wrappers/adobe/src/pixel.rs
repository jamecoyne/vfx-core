//! Conversions between After Effects pixel buffers and the packed RGBA f32 data
//! that vfx-core frames use. Pure functions, so they test without a host.

/// Bit depth of an After Effects world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    Bpc8,
    Bpc16,
    Bpc32,
}

impl Depth {
    pub fn from_bits(bits: i16) -> Option<Self> {
        match bits {
            8 => Some(Depth::Bpc8),
            16 => Some(Depth::Bpc16),
            32 => Some(Depth::Bpc32),
            _ => None,
        }
    }

    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Depth::Bpc8 => 4,
            Depth::Bpc16 => 8,
            Depth::Bpc32 => 16,
        }
    }
}

/// After Effects' 16-bit channel maximum (`PF_MAX_CHAN16`).
const MAX_CHAN16: f32 = 32768.0;

/// Decodes ARGB pixels (After Effects channel order) into packed RGBA f32.
/// `row_stride` is the byte distance between row starts, which may exceed the row's pixels.
pub fn decode(
    src: &[u8],
    row_stride: usize,
    width: usize,
    height: usize,
    depth: Depth,
) -> Vec<f32> {
    let bpp = depth.bytes_per_pixel();
    check_layout(src.len(), row_stride, width, height, bpp);
    let mut out = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        let row = &src[y * row_stride..y * row_stride + width * bpp];
        for px in row.chunks_exact(bpp) {
            // AE stores channels as alpha, red, green, blue.
            out.push(read_channel(px, 1, depth));
            out.push(read_channel(px, 2, depth));
            out.push(read_channel(px, 3, depth));
            out.push(read_channel(px, 0, depth));
        }
    }
    out
}

/// Encodes packed RGBA f32 back into ARGB pixels at the given depth.
/// Integer depths clamp to 0..=1 before quantising. 32-bit floats pass through unclamped.
pub fn encode(
    rgba: &[f32],
    dst: &mut [u8],
    row_stride: usize,
    width: usize,
    height: usize,
    depth: Depth,
) {
    let bpp = depth.bytes_per_pixel();
    check_layout(dst.len(), row_stride, width, height, bpp);
    assert_eq!(rgba.len(), width * height * 4, "rgba length");
    for y in 0..height {
        let row = &mut dst[y * row_stride..y * row_stride + width * bpp];
        for (x, px) in row.chunks_exact_mut(bpp).enumerate() {
            let p = &rgba[(y * width + x) * 4..][..4];
            write_channel(px, 0, p[3], depth);
            write_channel(px, 1, p[0], depth);
            write_channel(px, 2, p[1], depth);
            write_channel(px, 3, p[2], depth);
        }
    }
}

/// Size of an RGBA f32 canvas, and where a source buffer sits on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub canvas_width: usize,
    pub canvas_height: usize,
    /// Canvas position of the source's top-left pixel. May be negative.
    pub dx: isize,
    pub dy: isize,
}

/// Decodes `src` like [`decode`] and places it on a transparent canvas. Parts that fall
/// outside the canvas are dropped. After Effects buffers don't always share an origin,
/// for example an output grown for blur or a second layer of another size.
pub fn decode_onto(
    src: &[u8],
    row_stride: usize,
    width: usize,
    height: usize,
    depth: Depth,
    placement: Placement,
) -> Vec<f32> {
    let Placement {
        canvas_width: cw,
        canvas_height: ch,
        dx,
        dy,
    } = placement;
    let source = decode(src, row_stride, width, height, depth);
    if (cw, ch, dx, dy) == (width, height, 0, 0) {
        return source;
    }
    let mut canvas = vec![0.0f32; cw * ch * 4];
    // Columns of the source that land inside the canvas.
    let x0 = (-dx).max(0) as usize;
    let x1 = ((cw as isize - dx).min(width as isize)).max(0) as usize;
    if x0 >= x1 {
        return canvas;
    }
    for y in 0..height {
        let cy = y as isize + dy;
        if cy < 0 || cy >= ch as isize {
            continue;
        }
        let from = (y * width + x0) * 4..(y * width + x1) * 4;
        let to_start = (cy as usize * cw + (x0 as isize + dx) as usize) * 4;
        canvas[to_start..to_start + from.len()].copy_from_slice(&source[from]);
    }
    canvas
}

fn check_layout(len: usize, row_stride: usize, width: usize, height: usize, bpp: usize) {
    assert!(row_stride >= width * bpp, "row stride smaller than a row");
    if height > 0 {
        assert!(
            len >= row_stride * (height - 1) + width * bpp,
            "pixel buffer too small for {width}x{height}"
        );
    }
}

fn read_channel(px: &[u8], index: usize, depth: Depth) -> f32 {
    match depth {
        Depth::Bpc8 => f32::from(px[index]) / 255.0,
        Depth::Bpc16 => {
            let i = index * 2;
            f32::from(u16::from_ne_bytes([px[i], px[i + 1]])) / MAX_CHAN16
        }
        Depth::Bpc32 => {
            let i = index * 4;
            f32::from_ne_bytes(px[i..i + 4].try_into().unwrap())
        }
    }
}

fn write_channel(px: &mut [u8], index: usize, value: f32, depth: Depth) {
    match depth {
        Depth::Bpc8 => px[index] = (value.clamp(0.0, 1.0) * 255.0).round() as u8,
        Depth::Bpc16 => {
            let v = (value.clamp(0.0, 1.0) * MAX_CHAN16).round() as u16;
            let i = index * 2;
            px[i..i + 2].copy_from_slice(&v.to_ne_bytes());
        }
        Depth::Bpc32 => {
            let i = index * 4;
            px[i..i + 4].copy_from_slice(&value.to_ne_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2x2 ARGB8 source with distinct red values 10, 20, 30, 40 (opaque).
    fn two_by_two() -> Vec<u8> {
        [10u8, 20, 30, 40]
            .iter()
            .flat_map(|&r| [255, r, 0, 0])
            .collect()
    }

    fn red_at(canvas: &[f32], width: usize, x: usize, y: usize) -> u8 {
        (canvas[(y * width + x) * 4] * 255.0).round() as u8
    }

    #[test]
    fn decode_onto_offsets_into_a_larger_canvas() {
        let placement = Placement {
            canvas_width: 4,
            canvas_height: 4,
            dx: 1,
            dy: 2,
        };
        let canvas = decode_onto(&two_by_two(), 8, 2, 2, Depth::Bpc8, placement);
        assert_eq!(red_at(&canvas, 4, 1, 2), 10);
        assert_eq!(red_at(&canvas, 4, 2, 3), 40);
        assert_eq!(&canvas[0..4], &[0.0; 4], "uncovered canvas is transparent");
    }

    #[test]
    fn decode_onto_clips_negative_offsets() {
        let placement = Placement {
            canvas_width: 1,
            canvas_height: 1,
            dx: -1,
            dy: -1,
        };
        let canvas = decode_onto(&two_by_two(), 8, 2, 2, Depth::Bpc8, placement);
        assert_eq!(canvas.len(), 4);
        assert_eq!(
            red_at(&canvas, 1, 0, 0),
            40,
            "only the bottom-right pixel lands"
        );
    }

    #[test]
    fn argb8_decodes_in_ae_channel_order() {
        // alpha 255, red 10, green 20, blue 30
        let rgba = decode(&[255, 10, 20, 30], 4, 1, 1, Depth::Bpc8);
        assert_eq!(rgba, vec![10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 1.0]);
    }

    #[test]
    fn argb8_round_trips_exactly() {
        let src = [255, 10, 20, 30, 128, 0, 255, 77];
        let rgba = decode(&src, 8, 2, 1, Depth::Bpc8);
        let mut back = [0u8; 8];
        encode(&rgba, &mut back, 8, 2, 1, Depth::Bpc8);
        assert_eq!(back, src);
    }

    #[test]
    fn argb16_scales_by_32768() {
        let mut src = Vec::new();
        for v in [32768u16, 0, 16384, 32768] {
            src.extend_from_slice(&v.to_ne_bytes());
        }
        let rgba = decode(&src, 8, 1, 1, Depth::Bpc16);
        assert_eq!(rgba, vec![0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn float32_keeps_out_of_range_values() {
        let mut src = Vec::new();
        for v in [1.0f32, 2.5, -0.25, 0.5] {
            src.extend_from_slice(&v.to_ne_bytes());
        }
        let rgba = decode(&src, 16, 1, 1, Depth::Bpc32);
        assert_eq!(rgba, vec![2.5, -0.25, 0.5, 1.0]);

        let mut back = vec![0u8; 16];
        encode(&rgba, &mut back, 16, 1, 1, Depth::Bpc32);
        assert_eq!(back, src);
    }

    #[test]
    fn integer_encode_clamps() {
        let mut dst = [0u8; 4];
        encode(&[2.0, -1.0, 0.5, 1.5], &mut dst, 4, 1, 1, Depth::Bpc8);
        assert_eq!(dst, [255, 255, 0, 128]);
    }

    #[test]
    fn row_padding_is_skipped_on_decode_and_kept_on_encode() {
        // Two 1-pixel rows with 4 bytes of host padding after each.
        let src = [255, 1, 2, 3, 0xAA, 0xAA, 0xAA, 0xAA, 255, 4, 5, 6];
        let rgba = decode(&src, 8, 1, 2, Depth::Bpc8);
        assert_eq!(rgba.len(), 8);

        let mut dst = [0xEEu8; 12];
        encode(&rgba, &mut dst, 8, 1, 2, Depth::Bpc8);
        assert_eq!(&dst[0..4], &[255, 1, 2, 3]);
        assert_eq!(&dst[4..8], &[0xEE; 4], "padding untouched");
        assert_eq!(&dst[8..12], &[255, 4, 5, 6]);
    }

    #[test]
    fn depth_from_bits() {
        assert_eq!(Depth::from_bits(8), Some(Depth::Bpc8));
        assert_eq!(Depth::from_bits(16), Some(Depth::Bpc16));
        assert_eq!(Depth::from_bits(32), Some(Depth::Bpc32));
        assert_eq!(Depth::from_bits(12), None);
    }
}
