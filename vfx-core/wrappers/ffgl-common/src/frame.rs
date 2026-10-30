//! Repacks pixels between FFGL textures and the pipeline's canvases. FFGL textures are
//! premultiplied, rows bottom-up, and may be allocated larger than the image. Canvases
//! are straight alpha, rows top-down, exactly the image size.

/// Takes the `width` x `height` image out of a bottom-up texture `hw_width` wide,
/// flipping it top-down and unpremultiplying it.
pub fn texture_to_canvas(texels: &[f32], hw_width: u32, width: u32, height: u32) -> Vec<f32> {
    let (hw, w, h) = (hw_width as usize, width as usize, height as usize);
    let mut out = vec![0.0f32; w * h * 4];
    for row in 0..h {
        let src_row = h - 1 - row;
        for x in 0..w {
            let s = (src_row * hw + x) * 4;
            let d = (row * w + x) * 4;
            let a = texels[s + 3];
            for c in 0..3 {
                out[d + c] = if a > 0.0 { texels[s + c] / a } else { 0.0 };
            }
            out[d + 3] = a;
        }
    }
    out
}

/// The reverse: a top-down straight canvas to bottom-up premultiplied texels.
pub fn canvas_to_texture(canvas: &[f32], width: u32, height: u32) -> Vec<f32> {
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0.0f32; w * h * 4];
    for row in 0..h {
        let dst_row = h - 1 - row;
        for x in 0..w {
            let s = (row * w + x) * 4;
            let d = (dst_row * w + x) * 4;
            let a = canvas[s + 3];
            for c in 0..3 {
                out[d + c] = canvas[s + c] * a;
            }
            out[d + 3] = a;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flips_crops_and_unpremultiplies() {
        // A 3x2 allocation holding a 2x2 image. Bottom row red 0.1, 0.2; top row
        // 0.3 and a half-transparent 0.2 (premultiplied, so straight 0.4).
        #[rustfmt::skip]
        let texels = [
            0.1, 0.0, 0.0, 1.0,  0.2, 0.0, 0.0, 1.0,  9.0, 9.0, 9.0, 9.0,
            0.3, 0.0, 0.0, 1.0,  0.2, 0.0, 0.0, 0.5,  9.0, 9.0, 9.0, 9.0,
        ];
        let canvas = texture_to_canvas(&texels, 3, 2, 2);
        assert_eq!(canvas.len(), 16);
        assert_eq!(canvas[0], 0.3, "canvas row 0 is the top row");
        assert!((canvas[4] - 0.4).abs() < 1e-6, "unpremultiplied");
        assert_eq!(canvas[8], 0.1);
        assert!(!canvas.contains(&9.0), "padding column dropped");
    }

    #[test]
    fn round_trip() {
        #[rustfmt::skip]
        let texels = [
            0.1, 0.2, 0.3, 1.0,  0.05, 0.1, 0.0, 0.5,
            0.0, 0.0, 0.0, 0.0,  0.4, 0.4, 0.4, 0.8,
        ];
        let back = canvas_to_texture(&texture_to_canvas(&texels, 2, 2, 2), 2, 2);
        for (a, b) in back.iter().zip(texels) {
            assert!((a - b).abs() < 1e-6, "{back:?}");
        }
    }
}
