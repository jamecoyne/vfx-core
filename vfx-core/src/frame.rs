use crate::Gpu;

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

    /// Reads RGBA32F straight into float storage. Blocks until the GPU finishes.
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
        gpu.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed during readback");
        rx.recv()
            .expect("map callback dropped")
            .expect("failed to map readback buffer");

        let data = slice.get_mapped_range().expect("readback range is mapped");
        for (src, dst) in data
            .chunks_exact(gpu_pitch as usize)
            .zip(out.chunks_exact_mut(row_stride as usize))
        {
            dst[..row_bytes as usize].copy_from_slice(&src[..row_bytes as usize]);
        }
        drop(data);
        buffer.unmap();
    }
}

fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}
