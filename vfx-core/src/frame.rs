use std::sync::Arc;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

use crate::Gpu;
use crate::cpu::{Argb, Channel, Image, ImageMut, View, ViewMut};
use crate::readback::{ReadbackError, map_read};

/// A GPU texture that effects read from or render into.
///
/// Every frame can be sampled, rendered to, and copied, so the same type serves
/// as input, output, scratch and feedback state.
pub struct Frame {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

impl Frame {
    pub fn new(gpu: &Gpu, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vfx frame"),
            size: extent(width, height),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
            format,
        }
    }

    /// Bytes per pixel for this frame's format.
    pub fn bytes_per_pixel(&self) -> u32 {
        self.format
            .block_copy_size(None)
            .expect("frame formats must be uncompressed")
    }

    /// Uploads tightly packed rows in this frame's format.
    pub fn write(&self, gpu: &Gpu, pixels: &[u8]) {
        self.write_strided(gpu, pixels, self.width * self.bytes_per_pixel());
    }

    /// Uploads rows whose start addresses are `row_stride` bytes apart, as host buffers often are.
    pub fn write_strided(&self, gpu: &Gpu, pixels: &[u8], row_stride: u32) {
        let row_bytes = (self.width * self.bytes_per_pixel()) as usize;
        let stride = row_stride as usize;
        assert!(stride >= row_bytes, "row stride smaller than a row");
        assert!(
            pixels.len() >= stride * (self.height as usize - 1) + row_bytes,
            "pixel buffer too small for {}x{} frame",
            self.width,
            self.height
        );
        // write_texture accepts any row stride, so no staging buffer or padding is needed.
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_stride),
                rows_per_image: Some(self.height),
            },
            extent(self.width, self.height),
        );
    }

    /// Reads back tightly packed rows. Blocks until the GPU finishes.
    pub fn read(&self, gpu: &Gpu) -> Vec<u8> {
        self.read_strided(gpu, self.width * self.bytes_per_pixel())
    }

    /// Reads back rows `row_stride` bytes apart. Padding bytes between rows are zeroed.
    pub fn read_strided(&self, gpu: &Gpu, row_stride: u32) -> Vec<u8> {
        let row_bytes = self.width * self.bytes_per_pixel();
        assert!(row_stride >= row_bytes, "row stride smaller than a row");
        let mut out = vec![0u8; (row_stride * self.height) as usize];
        Readback::new(gpu, self).read_into(gpu, self, &mut out, row_stride);
        out
    }

    /// Uploads tightly packed RGBA8 rows. Kept for the common 8-bit case.
    pub fn write_rgba8(&self, gpu: &Gpu, pixels: &[u8]) {
        assert_eq!(
            self.format,
            wgpu::TextureFormat::Rgba8Unorm,
            "write_rgba8 needs Rgba8Unorm"
        );
        self.write(gpu, pixels);
    }

    /// Reads back tightly packed RGBA8 rows. Kept for the common 8-bit case.
    pub fn read_rgba8(&self, gpu: &Gpu) -> Vec<u8> {
        assert_eq!(
            self.format,
            wgpu::TextureFormat::Rgba8Unorm,
            "read_rgba8 needs Rgba8Unorm"
        );
        self.read(gpu)
    }

    /// Records a full-frame copy into `dst`. Both frames must share size and format.
    pub(crate) fn copy_into(&self, encoder: &mut wgpu::CommandEncoder, dst: &Frame) {
        assert_eq!(
            (self.width, self.height, self.format),
            (dst.width, dst.height, dst.format)
        );
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &dst.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            extent(self.width, self.height),
        );
    }
}

/// A frame on the GPU device a host and its effect share.
#[derive(Clone, Copy)]
pub struct GpuFrame<'a> {
    pub gpu: &'a Gpu,
    pub frame: &'a Frame,
}

/// How many rows [`GpuFrame::write`] converts per upload. This limits its buffer to that many
/// rows of the frame.
const ROWS_PER_UPLOAD: usize = 32;

impl GpuFrame<'_> {
    /// Uploads `image`, in the frame's format. `image` is straight colour with the top row first.
    /// Fails with [`RenderError::DeviceLost`] if the device is already known to be lost. A loss
    /// during the call isn't reported, because wgpu drops that error.
    ///
    /// # Panics
    /// If `image` isn't the frame's size, or the frame is neither RGBA8 nor
    /// RGBA32F.
    pub fn write(&self, image: Image<'_>) -> Result<(), RenderError> {
        match image {
            Image::U8(view) => self.write_view(view),
            Image::U16(view) => self.write_view(view),
            Image::F32(view) => self.write_view(view),
        }
    }

    fn write_view<C: Channel>(&self, view: View<'_, C>) -> Result<(), RenderError> {
        if self.gpu.faults().lost().is_some() {
            return Err(RenderError::DeviceLost);
        }
        let frame = self.frame;
        assert_eq!(
            (view.width as u32, view.height as u32),
            (frame.width, frame.height),
            "image and frame sizes differ"
        );
        let row_bytes = frame.width as usize * frame.bytes_per_pixel() as usize;
        // The buffer holds `u32` words, so a row can be cast to `f32`s as well as to bytes.
        let mut words = vec![0u32; row_bytes / 4 * ROWS_PER_UPLOAD.min(frame.height as usize)];
        let rows = bytemuck::cast_slice_mut::<u32, u8>(&mut words);
        for top in (0..frame.height as usize).step_by(ROWS_PER_UPLOAD) {
            let count = ROWS_PER_UPLOAD.min(frame.height as usize - top);
            for (y, dst) in rows.chunks_exact_mut(row_bytes).take(count).enumerate() {
                let src = view.row((top + y) as i32);
                match frame.format {
                    wgpu::TextureFormat::Rgba8Unorm => {
                        for (dst, p) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                            *dst = [p.red, p.green, p.blue, p.alpha].map(to_u8);
                        }
                    }
                    wgpu::TextureFormat::Rgba32Float => {
                        let dst = bytemuck::cast_slice_mut::<u8, [f32; 4]>(dst);
                        for (dst, p) in dst.iter_mut().zip(src) {
                            *dst = [p.red, p.green, p.blue, p.alpha].map(C::to_unit);
                        }
                    }
                    format => panic!("no upload into {format:?}"),
                }
            }
            self.gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &frame.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: top as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &rows[..row_bytes * count],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes as u32),
                    rows_per_image: None,
                },
                extent(frame.width, count as u32),
            );
        }
        Ok(())
    }
}

/// Converts `c` to 8 bits. A `u8` converts exactly. Other types go through the unit range.
fn to_u8<C: Channel>(c: C) -> u8 {
    match bytemuck::try_cast::<C, u8>(c) {
        Ok(byte) => byte,
        Err(_) => u8::from_unit(c.to_unit()),
    }
}

/// Where a CPU render draws its frame: straight colour, top row first.
pub enum FrameOut<'a> {
    /// A frame the size of the render on the device the host shares. An effect
    /// that draws in RAM uploads into it.
    Texture(GpuFrame<'a>),
    Pixels(ImageMut<'a>),
}

impl FrameOut<'_> {
    /// Calls `draw` with pixels for the frame. For a CPU output these are the output's own pixels.
    /// For a texture they're 8-bit pixels, uploaded into the texture after `draw` returns. For a
    /// texture the pixels are allocated on every call, so an effect that renders often keeps its
    /// own pixels and calls [`GpuFrame::write`].
    pub fn with_pixels(self, draw: impl FnOnce(ImageMut<'_>)) -> Result<(), RenderError> {
        match self {
            FrameOut::Pixels(pixels) => {
                draw(pixels);
                Ok(())
            }
            FrameOut::Texture(texture) => {
                let (w, h) = (texture.frame.width as i32, texture.frame.height as i32);
                let mut pixels = vec![Argb::<u8>::default(); w as usize * h as usize];
                draw(ImageMut::U8(ViewMut::from_pixels(&mut pixels, w, h)));
                texture.write(Image::U8(View::from_pixels(&pixels, w, h)))
            }
        }
    }
}

/// Why a render drew nothing. The host shows its input or fails the frame.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RenderError {
    /// The device was lost; everything on it has to be made again on a new one.
    #[error("the GPU device was lost")]
    DeviceLost,
    #[error("the GPU device is still opening")]
    DeviceOpening,
    /// An allocation failed; the device is still alive.
    #[error("the GPU is out of memory")]
    OutOfMemory,
    /// A readback or wait ran out of time on a device that is still alive.
    #[error("the GPU didn't finish in time")]
    Timeout,
    /// Any other error wgpu reported for the render's work.
    #[error("GPU error: {0}")]
    Gpu(#[source] Arc<wgpu::Error>),
    /// The effect's GPU work failed, but wgpu reported no error.
    #[error("the effect's GPU work failed: {0}")]
    Internal(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl RenderError {
    /// Whether the effect had no device to draw on, because it was lost or the next one is still
    /// opening. An editor renders such a frame once more. In a render whose source isn't live, the
    /// effect waits for the next device.
    pub fn needs_device(&self) -> bool {
        matches!(self, Self::DeviceLost | Self::DeviceOpening)
    }
}

impl From<Arc<wgpu::Error>> for RenderError {
    fn from(error: Arc<wgpu::Error>) -> Self {
        match *error {
            wgpu::Error::OutOfMemory { .. } => RenderError::OutOfMemory,
            _ => RenderError::Gpu(error),
        }
    }
}

impl From<wgpu::Error> for RenderError {
    fn from(error: wgpu::Error) -> Self {
        Arc::new(error).into()
    }
}

/// How long a readback waits for the GPU before it fails, so a hung GPU can't freeze a host.
const READBACK_TIMEOUT: Duration = Duration::from_secs(5);

/// A reusable readback buffer for frames of one size and format on the same device.
pub struct Readback {
    buffer: wgpu::Buffer,
    size: (u32, u32, wgpu::TextureFormat),
    gpu_pitch: u32,
}

impl Readback {
    pub fn new(gpu: &Gpu, frame: &Frame) -> Self {
        let row_bytes = frame.width * frame.bytes_per_pixel();
        // The GPU copy needs a 256-byte-aligned row pitch, so read into an aligned
        // buffer and repack to the caller's stride.
        let gpu_pitch = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vfx readback"),
            size: u64::from(gpu_pitch) * u64::from(frame.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            size: (frame.width, frame.height, frame.format),
            gpu_pitch,
        }
    }

    /// Reads RGBA32F straight into float storage. Blocks until the GPU finishes, and panics if it
    /// doesn't within [`READBACK_TIMEOUT`]: hosts catch that and pass the input through, which an
    /// export must not mistake for a rendered frame.
    pub fn read_f32(&mut self, gpu: &Gpu, frame: &Frame) -> Vec<f32> {
        assert_eq!(frame.format, wgpu::TextureFormat::Rgba32Float);
        let mut out = vec![0.0f32; (frame.width * frame.height * 4) as usize];
        self.read_into(
            gpu,
            frame,
            bytemuck::cast_slice_mut(&mut out),
            frame.width * frame.bytes_per_pixel(),
        );
        out
    }

    fn read_into(&mut self, gpu: &Gpu, frame: &Frame, out: &mut [u8], row_stride: u32) {
        if let Err(err) = self.read_within(gpu, frame, out, row_stride, READBACK_TIMEOUT) {
            panic!("GPU readback failed: {err}");
        }
    }

    fn read_within(
        &mut self,
        gpu: &Gpu,
        frame: &Frame,
        out: &mut [u8],
        row_stride: u32,
        timeout: Duration,
    ) -> Result<(), ReadbackError> {
        assert_eq!(self.size, (frame.width, frame.height, frame.format));
        let row_bytes = frame.width * frame.bytes_per_pixel();
        let gpu_pitch = self.gpu_pitch;
        let buffer = &self.buffer;

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(gpu_pitch),
                    rows_per_image: Some(frame.height),
                },
            },
            extent(frame.width, frame.height),
        );
        let submission_index = gpu.queue.submit(Some(encoder.finish()));

        let mapped = map_read(buffer);
        let wait = wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(timeout),
        };
        // Another thread's poll may be the one that calls back, a moment later.
        let mapped = crate::poll_device(&gpu.device, wait)
            .map_err(ReadbackError::from)
            .and_then(|_| match mapped.recv_timeout(timeout) {
                Ok(Ok(())) => Ok(()),
                Err(RecvTimeoutError::Timeout) => Err(ReadbackError::Timeout),
                Ok(Err(_)) | Err(RecvTimeoutError::Disconnected) => Err(ReadbackError::DeviceLost),
            });
        if let Err(err) = mapped {
            // The map may still be pending on this buffer, so the next read takes a new one.
            *self = Self::new(gpu, frame);
            return Err(err);
        }

        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("readback range is mapped");
        for (src, dst) in data
            .chunks_exact(gpu_pitch as usize)
            .zip(out.chunks_exact_mut(row_stride as usize))
        {
            dst[..row_bytes as usize].copy_from_slice(&src[..row_bytes as usize]);
        }
        drop(data);
        buffer.unmap();
        Ok(())
    }
}

fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readback_recovers_after_timeout() {
        let gpu = crate::device::open(&crate::device::DeviceRequest::from_env())
            .expect("no GPU adapter available for headless tests");
        let frame = Frame::new(&gpu, 256, 256, wgpu::TextureFormat::Rgba32Float);
        let input: Vec<f32> = (0..256 * 256 * 4).map(|i| i as f32).collect();
        frame.write(&gpu, bytemuck::cast_slice(&input));
        let mut readback = Readback::new(&gpu, &frame);
        let row_stride = frame.width * frame.bytes_per_pixel();
        let mut out = vec![0u8; (row_stride * frame.height) as usize];
        // Whether a zero wait finishes first is the driver's; neither result may break the next.
        let _ = readback.read_within(&gpu, &frame, &mut out, row_stride, Duration::ZERO);
        assert_eq!(readback.read_f32(&gpu, &frame), input);
    }
}
