//! Frames in RAM, for [`crate::CpuEffect`]s. Pixels are laid out as After Effects lays
//! them out: alpha, red, green, blue, of one channel type. `Argb<u8>`, `Argb<u16>` and
//! `Argb<f32>` are byte for byte `PF_Pixel8`, `PF_Pixel16` and `PF_PixelFloat`, so After
//! Effects' buffers are read and written in place, at their own row stride; other hosts
//! convert into these. Rows run top to bottom.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::Range;

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for f32 {}
}

/// One channel type: `u8`, `u16` or `f32`, each valid for any bit pattern. Integer channels
/// clamp when written; float channels keep any value.
pub trait Channel:
    sealed::Sealed + Copy + Default + PartialEq + Send + Sync + std::fmt::Debug + bytemuck::Pod
{
    /// The channel's white. After Effects' 16-bit white is 32768, not 65535.
    const MAX: f32;
    fn to_unit(self) -> f32;
    fn from_unit(f: f32) -> Self;
}

impl Channel for u8 {
    const MAX: f32 = 255.0;
    fn to_unit(self) -> f32 {
        f32::from(self) / <Self as Channel>::MAX
    }
    fn from_unit(f: f32) -> Self {
        (f.clamp(0.0, 1.0) * <Self as Channel>::MAX + 0.5) as u8
    }
}

impl Channel for u16 {
    const MAX: f32 = 32768.0;
    fn to_unit(self) -> f32 {
        f32::from(self) / <Self as Channel>::MAX
    }
    fn from_unit(f: f32) -> Self {
        (f.clamp(0.0, 1.0) * <Self as Channel>::MAX + 0.5) as u16
    }
}

impl Channel for f32 {
    const MAX: f32 = 1.0;
    fn to_unit(self) -> f32 {
        self
    }
    fn from_unit(f: f32) -> Self {
        f
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Argb<C> {
    pub alpha: C,
    pub red: C,
    pub green: C,
    pub blue: C,
}

// SAFETY: four channels of one plain type, laid out in order with no padding.
unsafe impl<C: bytemuck::Zeroable> bytemuck::Zeroable for Argb<C> {}
unsafe impl<C: bytemuck::Pod> bytemuck::Pod for Argb<C> {}

/// Converts between channel types through the unit range.
pub fn convert<A: Channel, B: Channel>(p: &Argb<A>) -> Argb<B> {
    Argb {
        alpha: B::from_unit(p.alpha.to_unit()),
        red: B::from_unit(p.red.to_unit()),
        green: B::from_unit(p.green.to_unit()),
        blue: B::from_unit(p.blue.to_unit()),
    }
}

/// A read-only picture: rows top to bottom, `stride` bytes apart.
#[derive(Clone, Copy)]
pub struct View<'a, C> {
    data: &'a [u8],
    pub width: i32,
    pub height: i32,
    stride: usize,
    _channel: PhantomData<C>,
}

impl<'a, C: Channel> View<'a, C> {
    /// Panics if `data` can't hold `height` rows of `width` pixels `stride` bytes apart, or
    /// isn't aligned for the channel type.
    pub fn new(data: &'a [u8], width: i32, height: i32, stride: usize) -> Self {
        check::<C>(data, width, height, stride);
        Self {
            data,
            width,
            height,
            stride,
            _channel: PhantomData,
        }
    }

    pub fn from_pixels(pixels: &'a [Argb<C>], width: i32, height: i32) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), size_of_val(pixels))
        };
        Self::new(bytes, width, height, row_bytes::<C>(width))
    }

    pub fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    pub fn row(&self, y: i32) -> &'a [Argb<C>] {
        row(self.data, y, self.width, self.stride)
    }

    /// Each row's pixels as bytes, top row first.
    pub fn byte_rows(&self) -> Vec<&'a [u8]> {
        if self.is_empty() {
            return Vec::new();
        }
        (0..self.height)
            .map(|y| bytemuck::cast_slice(self.row(y)))
            .collect()
    }
}

/// A picture being drawn into.
pub struct ViewMut<'a, C> {
    data: &'a mut [u8],
    pub width: i32,
    pub height: i32,
    stride: usize,
    /// The rows `data` holds: all of them, or one band.
    rows: Range<i32>,
    _channel: PhantomData<C>,
}

impl<'a, C: Channel> ViewMut<'a, C> {
    pub fn new(data: &'a mut [u8], width: i32, height: i32, stride: usize) -> Self {
        check::<C>(data, width, height, stride);
        Self {
            data,
            width,
            height,
            stride,
            rows: 0..height.max(0),
            _channel: PhantomData,
        }
    }

    pub fn from_pixels(pixels: &'a mut [Argb<C>], width: i32, height: i32) -> Self {
        let len = size_of_val(pixels);
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(pixels.as_mut_ptr().cast::<u8>(), len) };
        Self::new(bytes, width, height, row_bytes::<C>(width))
    }

    pub fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// The rows this view holds.
    pub fn rows(&self) -> Range<i32> {
        self.rows.clone()
    }

    /// Row `y` of the whole picture, which must be one of [`ViewMut::rows`].
    pub fn row(&self, y: i32) -> &[Argb<C>] {
        row(self.data, y - self.rows.start, self.width, self.stride)
    }

    /// Row `y` of the whole picture, which must be one of [`ViewMut::rows`].
    pub fn row_mut(&mut self, y: i32) -> &mut [Argb<C>] {
        let start = (y - self.rows.start) as usize * self.stride;
        let bytes = &mut self.data[start..start + row_bytes::<C>(self.width)];
        // `check` made sure every row start is aligned, and any bits are a valid channel.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), self.width as usize) }
    }

    /// Splits the picture into `count` bands of rows that can be drawn on separately, each
    /// still addressed by the whole picture's row numbers.
    pub fn bands(&mut self, count: usize) -> Vec<ViewMut<'_, C>> {
        debug_assert_eq!(self.rows, 0..self.height.max(0));
        let (width, height, stride) = (self.width, self.height.max(0), self.stride);
        let count = count.clamp(1, height.max(1) as usize) as i32;
        let mut rest: &mut [u8] = self.data;
        let mut bands = Vec::with_capacity(count as usize);
        for i in 0..count {
            let rows = i * height / count..(i + 1) * height / count;
            let len = if i + 1 == count {
                rest.len()
            } else {
                rows.len() * stride
            };
            let (data, tail) = std::mem::take(&mut rest).split_at_mut(len);
            rest = tail;
            bands.push(ViewMut {
                data,
                width,
                height,
                stride,
                rows,
                _channel: PhantomData,
            });
        }
        bands
    }
}

fn row_bytes<C>(width: i32) -> usize {
    width.max(0) as usize * size_of::<Argb<C>>()
}

fn row<C>(data: &[u8], y: i32, width: i32, stride: usize) -> &[Argb<C>] {
    let start = y as usize * stride;
    let bytes = &data[start..start + row_bytes::<C>(width)];
    // `check` made sure every row start is aligned, and any bits are a valid channel.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), width as usize) }
}

fn check<C>(data: &[u8], width: i32, height: i32, stride: usize) {
    let align = align_of::<Argb<C>>();
    assert!(
        (data.as_ptr() as usize).is_multiple_of(align) && stride.is_multiple_of(align),
        "pixel rows not aligned for their channel type"
    );
    if width <= 0 || height <= 0 {
        return;
    }
    let row = row_bytes::<C>(width);
    assert!(stride >= row, "row stride smaller than a row");
    assert!(
        data.len() >= stride * (height as usize - 1) + row,
        "pixel buffer too small for {width}x{height}"
    );
}

/// A frame at one of the three depths a host renders at.
#[derive(Clone, Copy)]
pub enum Image<'a> {
    U8(View<'a, u8>),
    U16(View<'a, u16>),
    F32(View<'a, f32>),
}

/// An output frame at one of the three depths.
pub enum ImageMut<'a> {
    U8(ViewMut<'a, u8>),
    U16(ViewMut<'a, u16>),
    F32(ViewMut<'a, f32>),
}

impl Image<'_> {
    pub fn size(&self) -> (i32, i32) {
        match self {
            Image::U8(v) => (v.width, v.height),
            Image::U16(v) => (v.width, v.height),
            Image::F32(v) => (v.width, v.height),
        }
    }

    /// Each row's pixels as bytes, top row first.
    pub fn byte_rows(&self) -> Vec<&[u8]> {
        match self {
            Image::U8(v) => v.byte_rows(),
            Image::U16(v) => v.byte_rows(),
            Image::F32(v) => v.byte_rows(),
        }
    }
}

impl ImageMut<'_> {
    pub fn size(&self) -> (i32, i32) {
        match self {
            ImageMut::U8(v) => (v.width, v.height),
            ImageMut::U16(v) => (v.width, v.height),
            ImageMut::F32(v) => (v.width, v.height),
        }
    }
}

/// Picks out the views of one depth, for effects written generically over [`Channel`].
pub trait Depth: Channel {
    fn view(image: Image<'_>) -> Option<View<'_, Self>>;
    fn image(view: View<'_, Self>) -> Image<'_>;
    fn image_mut(view: ViewMut<'_, Self>) -> ImageMut<'_>;
}

impl Depth for u8 {
    fn view(image: Image<'_>) -> Option<View<'_, Self>> {
        match image {
            Image::U8(v) => Some(v),
            _ => None,
        }
    }
    fn image(view: View<'_, Self>) -> Image<'_> {
        Image::U8(view)
    }
    fn image_mut(view: ViewMut<'_, Self>) -> ImageMut<'_> {
        ImageMut::U8(view)
    }
}

impl Depth for u16 {
    fn view(image: Image<'_>) -> Option<View<'_, Self>> {
        match image {
            Image::U16(v) => Some(v),
            _ => None,
        }
    }
    fn image(view: View<'_, Self>) -> Image<'_> {
        Image::U16(view)
    }
    fn image_mut(view: ViewMut<'_, Self>) -> ImageMut<'_> {
        ImageMut::U16(view)
    }
}

impl Depth for f32 {
    fn view(image: Image<'_>) -> Option<View<'_, Self>> {
        match image {
            Image::F32(v) => Some(v),
            _ => None,
        }
    }
    fn image(view: View<'_, Self>) -> Image<'_> {
        Image::F32(view)
    }
    fn image_mut(view: ViewMut<'_, Self>) -> ImageMut<'_> {
        ImageMut::F32(view)
    }
}

/// The frames a CPU render reads, numbered oldest first, `0..count`; the last is the frame
/// being rendered. A host can hand out one frame at a time and take it back as soon as `f`
/// returns. An effect may skip frames, or ask for one again.
pub trait FrameSource {
    /// Calls `f` with frame `index`, or with `None` if that frame isn't there.
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>));

    /// [`Self::with_frame`] for an effect that may find it doesn't need the pixels: `wanted` sees
    /// frame `index` as rows of the bytes this source holds it in, and `f` runs only if it returns
    /// true, so a source that converts frames needn't convert this one. A frame that isn't there
    /// skips `wanted` and calls `f` with `None`.
    ///
    /// Equal bytes mean equal pixels, so a hash of them stands for the frame. Equal pixels may
    /// still come as different bytes, and these bytes need not be the ones `with_frame` shows:
    /// compare them only with this method's, on the same source.
    fn with_frame_if_wanted(
        &mut self,
        index: usize,
        wanted: &mut dyn FnMut(&[&[u8]]) -> bool,
        f: &mut dyn FnMut(Option<Image<'_>>),
    ) {
        self.with_frame(index, &mut |image| match image {
            Some(frame) if !wanted(&frame.byte_rows()) => {}
            image => f(image),
        });
    }

    /// Whether frames arrive from a live stream, so an effect should draw what it has rather than
    /// hold up the output: true for [`History`], false where a render may be exported and has to
    /// be exact whatever it costs, unless the source sets a
    /// [fetch budget](Self::fetch_budget). A live source that is also
    /// [fixed](Self::fixed) numbers each frame once, and starts over only from a
    /// lower number.
    fn live(&self) -> bool {
        false
    }

    /// Whether a frame's pixels stay as they were handed over, so an effect that keeps work
    /// between renders needn't read an earlier frame again: true for a host's own copies of
    /// the frames it was given ([`History`]) and for OpenFX in Resolve while renders stream,
    /// false where the host renders earlier frames on request and an edit upstream can change
    /// them (other OpenFX, After Effects). A source that starts its numbering over hands out a
    /// different current frame, which an effect still checks.
    fn fixed(&self) -> bool {
        false
    }

    /// How many of the newest frames a [fixed](Self::fixed) source still has the pixels of, or
    /// will fetch: older ones come as `None`, and stay gone for as long as it says so. An effect
    /// that keeps work works those frames out again if the source later stops being fixed.
    fn held(&self) -> usize {
        usize::MAX
    }

    /// How many distinct earlier frames one render may ask for, the current frame
    /// not counted, or `None` for as many as it needs. For a host that holds every
    /// frame it hands out until the render returns; set it to at least 2. A render
    /// reads at most this many, plus up to as many again of the frames one analysis
    /// of the current frame needs and nothing was kept from, which it reads first.
    /// Only an effect that [keeps state](crate::CpuEffect::keeps_state) can keep to
    /// the budget: it reads the rest over the following renders and draws what it
    /// kept meanwhile, so a render may differ from a fresh one until it has read them.
    fn fetch_budget(&self) -> Option<usize> {
        None
    }
}

/// For hosts that hand over one frame at a time (Resolume, the preview): the last few
/// frames, oldest first, with their allocations recycled at the front. Only the newest are
/// held in full, as the effect says; a render still counts the rest, which come as `None`.
/// A host that may hand over the same picture again calls [`History::drop_repeat`] after filling
/// a frame, so that a repeat isn't counted as a new frame.
pub struct History<C> {
    frames: VecDeque<Vec<Argb<C>>>,
    /// The frame that made room for the newest, kept until the next one in case it's a repeat.
    left: Option<Vec<Argb<C>>>,
    /// A repeat's buffer, for the next frame.
    spare: Option<Vec<Argb<C>>>,
    /// `count` before the newest frame.
    count_before: usize,
    /// How many frames a render counts, `frames` being the newest of them.
    count: usize,
    width: i32,
    height: i32,
    /// How many frames have been pushed; the newest is number `pushed - 1`.
    pushed: i64,
}

impl<C> Default for History<C> {
    fn default() -> Self {
        Self {
            frames: VecDeque::new(),
            left: None,
            spare: None,
            count_before: 0,
            count: 0,
            width: 0,
            height: 0,
            pushed: 0,
        }
    }
}

impl<C: Depth> History<C> {
    pub fn clear(&mut self) {
        self.frames.clear();
        self.left = None;
        self.spare = None;
        self.count = 0;
        self.pushed = 0;
    }

    pub fn resize(&mut self, width: i32, height: i32) {
        self.clear();
        self.width = width;
        self.height = height;
    }

    pub fn size(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The newest frame's number: frames pushed since the last clear, less one.
    pub fn frame_number(&self) -> i64 {
        (self.pushed - 1).max(0)
    }

    /// Room for a new frame, the newest, to be filled by the caller. Counts `before` frames
    /// before it and holds the pixels of `held` of them.
    pub fn next(&mut self, before: usize, held: usize) -> &mut [Argb<C>] {
        self.count_before = self.count;
        self.count = (self.count + 1).min(before + 1);
        let keep = held.min(before) + 1;
        while self.frames.len() > keep {
            self.frames.pop_front();
        }
        let leaving = (self.frames.len() == keep).then(|| self.frames.pop_front());
        let leaving = leaving.flatten();
        let reused = self.spare.take().or_else(|| self.left.take());
        // Without earlier frames there's nothing a repeat could change.
        let (pixels, left) = match (before, reused, leaving) {
            (0, reused, leaving) => (reused.or(leaving), None),
            (_, reused, leaving) => (reused, leaving),
        };
        self.left = left;
        let pixels = pixels
            .unwrap_or_else(|| vec![Argb::default(); self.width as usize * self.height as usize]);
        self.frames.push_back(pixels);
        self.pushed += 1;
        self.frames.back_mut().expect("just pushed")
    }

    /// Takes the newest frame back if it's the same picture as the one before, and says if it did.
    /// New frames usually differ within the first rows, so this costs little unless it's a repeat.
    pub fn drop_repeat(&mut self) -> bool {
        let Some((newest, rest)) = self.frames.make_contiguous().split_last() else {
            return false;
        };
        let Some(previous) = rest.last().or(self.left.as_ref()) else {
            return false;
        };
        if pixel_bytes(newest) != pixel_bytes(previous) {
            return false;
        }
        self.spare = self.frames.pop_back();
        if let Some(left) = self.left.take() {
            self.frames.push_front(left);
        }
        self.count = self.count_before;
        self.pushed -= 1;
        true
    }
}

fn pixel_bytes<C: Depth>(pixels: &[Argb<C>]) -> &[u8] {
    // SAFETY: `Argb` is four channels of a plain number type, with no padding.
    unsafe { std::slice::from_raw_parts(pixels.as_ptr().cast(), size_of_val(pixels)) }
}

impl<C: Depth> FrameSource for History<C> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        let older = self.count - self.frames.len();
        let pixels = index.checked_sub(older).and_then(|i| self.frames.get(i));
        f(pixels.map(|p| C::image(View::from_pixels(p, self.width, self.height))));
    }

    fn live(&self) -> bool {
        true
    }

    /// Frames go in through [`History::next`] only, so earlier ones never change.
    fn fixed(&self) -> bool {
        true
    }

    fn held(&self) -> usize {
        self.frames.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_channels_clamp_and_round() {
        assert_eq!(u8::from_unit(1.5), 255);
        assert_eq!(u8::from_unit(-1.0), 0);
        assert_eq!(u8::from_unit(0.5), 128);
        assert_eq!(u16::from_unit(1.0), 32768);
        assert_eq!(f32::from_unit(1.5), 1.5);
    }

    #[test]
    fn history_recycles_and_bounds() {
        let grey = |v: u8| Argb {
            alpha: 255,
            red: v,
            green: v,
            blue: v,
        };
        let mut h = History::<u8>::default();
        h.resize(2, 1);
        h.next(1, 1).fill(grey(255));
        let first = h.frames[0].as_ptr();
        h.next(1, 1).fill(grey(0));
        h.next(1, 1).fill(grey(128));
        assert_eq!(h.len(), 2);
        assert_eq!((h.frames[0][0].red, h.frames[1][0].red), (0, 128));
        assert_eq!(h.frame_number(), 2);
        h.next(1, 1).fill(grey(64));
        assert_eq!(h.frames[1].as_ptr(), first, "the oldest buffer is reused");
        for _ in 0..5 {
            h.next(3, 3);
        }
        assert_eq!(h.len(), 4);
        h.next(0, 0);
        assert_eq!(h.len(), 1);
        h.clear();
        assert!(h.is_empty() && h.frame_number() == 0);
        h.resize(3, 2);
        assert_eq!(h.next(0, 0).len(), 6);
    }

    /// Frames not held still count, so numbering is what a full history's would be.
    #[test]
    fn history_counts_frames_it_does_not_hold() {
        let mut h = History::<u8>::default();
        h.resize(1, 1);
        for v in 0..6u8 {
            h.next(3, 1)[0].red = v;
        }
        assert_eq!((h.len(), h.held(), h.frame_number()), (4, 2, 5));
        let mut read = Vec::new();
        for index in 0..4 {
            h.with_frame(index, &mut |image| {
                read.push(image.map(|i| match i {
                    Image::U8(v) => v.row(0)[0].red,
                    _ => unreachable!(),
                }));
            });
        }
        assert_eq!(read, [None, None, Some(4), Some(5)]);
    }

    fn newest_values(h: &mut History<u8>) -> Vec<Option<u8>> {
        let mut read = Vec::new();
        for index in 0..h.len() {
            h.with_frame(index, &mut |image| {
                read.push(image.map(|i| match i {
                    Image::U8(v) => v.row(0)[0].red,
                    _ => unreachable!(),
                }));
            });
        }
        read
    }

    /// A repeat leaves the history as it was, whether or not the effect holds earlier frames.
    #[test]
    fn a_repeat_is_not_a_new_frame() {
        for held in [0, 1, 3] {
            let mut h = History::<u8>::default();
            h.resize(2, 1);
            let push = |h: &mut History<u8>, v: u8| {
                h.next(3, held)[0].red = v;
                h.frames.back_mut().unwrap()[1].red = 0;
                h.drop_repeat()
            };
            assert!(!push(&mut h, 1), "the first frame has nothing before it");
            for v in 2..7 {
                assert!(!push(&mut h, v));
            }
            let (number, read) = (h.frame_number(), newest_values(&mut h));
            for _ in 0..3 {
                assert!(push(&mut h, 6));
                assert_eq!(
                    (h.frame_number(), newest_values(&mut h)),
                    (number, read.clone())
                );
            }
            assert!(!push(&mut h, 7));
            assert_eq!(h.frame_number(), number + 1);
            assert_eq!(newest_values(&mut h).last(), Some(&Some(7)));
        }
    }

    /// The last pixel alone is enough to make a frame new.
    #[test]
    fn a_change_anywhere_is_a_new_frame() {
        for before in [0, 3] {
            let mut h = History::<f32>::default();
            h.resize(2, 2);
            h.next(before, 0).fill(Argb::default());
            h.next(before, 0).fill(Argb::default());
            assert_eq!(h.drop_repeat(), before > 0);
            let frame = h.next(before, 0);
            frame.fill(Argb::default());
            frame[3].blue = 0.001;
            assert!(!h.drop_repeat());
        }
    }

    #[test]
    fn strided_rows() {
        // Two pixels a row in room for three: the third is padding.
        let mut bytes = vec![0u8; 3 * 4 * 2];
        let mut v = ViewMut::<u8>::new(&mut bytes, 2, 2, 12);
        v.row_mut(1)[1].red = 9;
        assert_eq!(bytes[12 + 4 + 1], 9);
    }
}
