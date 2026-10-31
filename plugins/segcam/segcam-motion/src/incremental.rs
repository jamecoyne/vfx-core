//! segcam Motion worked out incrementally: each frame's grid, motion and tiles are kept, with
//! a hash of the pixels they came from. Where earlier frames can change upstream (OpenFX), a
//! render still reads every frame it depends on and works out again only what a changed frame
//! feeds, so an edit reaches it however far back it is. Where they can't (a host's own history
//! of frames, as in Resolume and the preview: [`FrameSource::fixed`]), it reads only the frames
//! it hasn't worked out, and the current one. Results come from each frame and the settings
//! alone, so render order doesn't change the output. Motion whose background is cut short by
//! the start of the frames, as at the start of a clip, is kept with the frame it starts at.
use std::collections::{BTreeMap, BTreeSet};
use std::ops::RangeInclusive;

use vfx_core::cpu::FrameSource;

use crate::gpu::{Cut, Drawer, Fault};
use crate::motion::{self, LumaGrid, Motion, MotionSettings, ThresholdSettings};
use crate::pixel::{Argb, Channel, Depth, View, ViewMut};
use crate::render::{
    Detect, FaceParts, OutputMode, Pile, PiledTile, PixelRect, RenderSettings, downsample,
    draw_output, fill_rect, frame_tiles, history_frames,
};
use crate::swarm::{self, SwarmMetrics, Tile};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Canvas {
    width: i32,
    height: i32,
    channel_bytes: usize,
    grid_width: i32,
}

/// What a frame's segments depend on besides the grids.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Detection {
    detect: Detect,
    motion: MotionSettings,
    background_frames: i32,
    threshold: ThresholdSettings,
    faces: FaceParts,
}

/// What a frame's motion and tiles were worked out from besides its own pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Basis {
    detection: Detection,
    /// The first frame of its background: later than usual near the start of the frames.
    from: i64,
}

#[derive(Default)]
pub struct Incremental {
    canvas: Option<Canvas>,
    grids: BTreeMap<i64, LumaGrid>,
    /// A hash of each frame's pixels, since edits upstream can leave the grid as it was.
    prints: BTreeMap<i64, u64>,
    motion: BTreeMap<i64, (Basis, Motion)>,
    /// Tiles by frame, with the basis and mirroring they were cut for.
    tiles: BTreeMap<i64, (Basis, bool, Vec<Tile>)>,
    /// Drawing on the GPU: a frame's tiles go into its atlas when a render first draws them
    /// there. The drawer is made by the first render that draws on the GPU, as hosts make
    /// instances often.
    gpu: Option<Box<Drawer>>,
    cuts: BTreeMap<i64, Cut>,
    gpu_wanted: bool,
    gpu_overlay: bool,
    /// From what share of the canvas in window pixels the swarm draws on the GPU.
    gpu_from: Option<f64>,
    window_share: f64,
}

/// A tile in the pile, and where it is kept: frame and index.
struct Placed<'a> {
    tile: &'a Tile,
    frame: i64,
    index: usize,
}

impl PiledTile for Placed<'_> {
    fn tile(&self) -> &Tile {
        self.tile
    }
}

/// Below these shares of the canvas in window pixels (each window with its shadow, overlaps
/// counted, as the CPU draws them) the CPU draws the swarm faster than the GPU's copy back. On an
/// RTX 4070 at 1080p the CPU took about 2.7 ms a canvas at 8 bits and 2.2 ms in float, the GPU
/// about 1.2 and 3.6 ms whatever the windows; these sit a little above where they cross.
const GPU_FROM_8_BIT: f64 = 0.5;
const GPU_FROM_DEEPER: f64 = 1.5;

impl Incremental {
    /// Draws Swarm on the GPU where there is one, or always on the CPU. Off by default, so the
    /// library's renders are the CPU's; the plugin and the CLI turn it on.
    pub fn use_gpu(&mut self, on: bool) {
        self.gpu_wanted = on;
        if !on {
            self.set_drawer(None);
        }
    }

    /// The drawer, once a render has drawn on the GPU.
    pub fn gpu(&self) -> Option<&Drawer> {
        self.gpu.as_deref()
    }

    /// From what share of the canvas in window pixels the swarm draws on the GPU; 0 draws every
    /// render there. By default, what was measured to be faster at the canvas's depth.
    pub fn gpu_threshold(&mut self, share: f64) {
        self.gpu_from = Some(share);
    }

    /// The share of the canvas the last swarm render's windows covered, overlaps counted.
    pub fn window_share(&self) -> f64 {
        self.window_share
    }

    /// Draws with `drawer` from now on.
    #[cfg(test)]
    pub(crate) fn use_drawer(&mut self, drawer: Drawer) {
        self.gpu_wanted = true;
        self.set_drawer(Some(Box::new(drawer)));
    }

    fn set_drawer(&mut self, drawer: Option<Box<Drawer>>) {
        self.gpu = drawer;
        self.cuts.clear();
    }

    /// Draws Overlay on the GPU too. Off by default: while frames go through host memory, the
    /// upload and readback cost more than the CPU's whole Overlay.
    pub fn gpu_overlay(&mut self, on: bool) {
        self.gpu_overlay = on;
    }

    /// Renders the last of `count` frames, frame `number`, into `out`. Reads the frames the
    /// render depends on (only those it lacks, from a fixed source) once each, oldest first,
    /// and works out again only what changed.
    pub fn advance<C: Depth>(
        &mut self,
        s: &RenderSettings,
        frames: &mut dyn FrameSource,
        count: usize,
        number: i64,
        out: &mut ViewMut<C>,
    ) {
        if out.is_empty() {
            return;
        }
        if count == 0 {
            let all = PixelRect::new(0, 0, out.width, out.height);
            fill_rect(out, all, Argb::default());
            return;
        }
        let on_gpu =
            s.mode == OutputMode::Swarm || s.mode == OutputMode::Overlay && self.gpu_overlay;
        if self.gpu_wanted && self.gpu.is_none() && on_gpu {
            let drawer = Drawer::new().map(Box::new);
            self.gpu_wanted = drawer.is_some();
            self.set_drawer(drawer);
        }
        if let Some(gpu) = &mut self.gpu {
            gpu.begin();
        }
        let (w, h) = (out.width, out.height);
        let canvas = Canvas {
            width: w,
            height: h,
            channel_bytes: size_of::<C>(),
            grid_width: s.grid_width,
        };
        if self.canvas != Some(canvas) {
            self.reset();
            self.canvas = Some(canvas);
        }
        let swarm = s.mode == OutputMode::Swarm;
        let background = i64::from(history_frames(s));
        let targets = targets(s);
        let first = number - count as i64 + 1;

        // Frames the host didn't hand over count as empty, as in a cold render, but nothing
        // worked out from one is kept, so it is read again next time.
        let mut missing = BTreeSet::new();
        let mut fault = None;
        let oldest = (number - targets + 1 - background).max(first);
        // A fixed source's earlier frames are as they were, so only the current frame is
        // checked: a different one means the source started over.
        let fixed = frames.fixed();
        let mut current = None;
        if fixed {
            frames.with_frame((number - first) as usize, &mut |image| {
                current = image.and_then(C::view).map(|v| fingerprint(&v));
            });
            if current.is_none_or(|print| self.prints.get(&number).is_some_and(|k| *k != print)) {
                self.reset();
            }
        }
        let visit = if fixed {
            self.lacking(s, first, number)
        } else {
            (oldest..=number).collect()
        };
        for frame in visit {
            frames.with_frame((frame - first) as usize, &mut |image| {
                let view = image.and_then(C::view);
                match &view {
                    None => {
                        self.forget(frame, background);
                        missing.insert(frame);
                    }
                    Some(_) if fixed && self.prints.contains_key(&frame) => {}
                    Some(v) => {
                        let print = current
                            .filter(|_| frame == number)
                            .unwrap_or_else(|| fingerprint(v));
                        if self.prints.get(&frame) != Some(&print) {
                            // A fixed source's frame read for the first time can't have
                            // changed what later frames worked out from it; elsewhere a
                            // frame that wasn't kept may have.
                            if !fixed {
                                self.forget(frame, background);
                            }
                            self.grids.insert(frame, downsample(v, s.grid_width));
                            self.prints.insert(frame, print);
                        }
                    }
                }
                let basis = basis(s, frame, first);
                if frame > number - targets && !self.complete(frame, &basis, s) {
                    let motion = match self.motion(frame, &basis) {
                        Some(motion) => motion.clone(),
                        None => {
                            let empty = LumaGrid::default();
                            let grid = self.grids.get(&frame).unwrap_or(&empty);
                            let mut motion = match s.detect {
                                Detect::Motion => {
                                    let history = (basis.from..frame)
                                        .map(|k| self.grids.get(&k).unwrap_or(&empty));
                                    motion::detect(history, grid, &s.motion)
                                }
                                Detect::Threshold => {
                                    motion::threshold(grid, &s.threshold, &s.motion)
                                }
                                Detect::FacesOnly => Motion {
                                    mask: vec![0; grid.count()],
                                    width: grid.width,
                                    height: grid.height,
                                    ..Motion::default()
                                },
                            };
                            if let Some(view) = view.as_ref().filter(|_| s.faces.any()) {
                                let found = crate::faces::find(view, s.faces, grid.count());
                                motion.blobs.extend(found);
                            }
                            self.motion.insert(frame, (basis, motion.clone()));
                            motion
                        }
                    };
                    if swarm {
                        let tiles = frame_tiles(&motion, view.as_ref(), frame, w, h, s.mirror);
                        self.tiles.insert(frame, (basis, s.mirror, tiles));
                        self.cuts.remove(&frame);
                    }
                }
                if frame == number && fault.is_none() {
                    fault = self.try_draw(s, count, number, view, out).err();
                }
            });
        }
        if let Some(fault) = fault {
            self.fall_back(&fault);
            return self.advance(s, frames, count, number, out);
        }
        for &frame in &missing {
            self.forget(frame, background);
        }
        let reach = background + targets;
        self.keep(number - reach..=number + reach);
    }

    /// The frames from `first` a render of `number` has to read: those whose motion or tiles
    /// aren't kept, the frames before them whose grids aren't, and `number`, which it draws.
    fn lacking(&self, s: &RenderSettings, first: i64, number: i64) -> BTreeSet<i64> {
        let mut wanted = BTreeSet::from([number]);
        for target in (number - targets(s) + 1).max(first)..=number {
            let basis = basis(s, target, first);
            if !self.complete(target, &basis, s) {
                wanted.insert(target);
                let before = basis.from..target;
                wanted.extend(before.filter(|k| !self.grids.contains_key(k)));
            }
        }
        wanted
    }

    /// Draws frame `number`, the last of `count`, from what is kept. `frame` is its pixels,
    /// which every output but the swarm draws over. If the GPU fails, the instance goes back
    /// to the CPU, whose tiles the next render cuts.
    pub fn draw<C: Channel>(
        &mut self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) {
        if let Err(fault) = self.try_draw(s, count, number, frame, out) {
            self.fall_back(&fault);
            self.draw_cpu(s, count, number, frame, out);
        }
    }

    fn try_draw<C: Channel>(
        &mut self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        let Some(mut gpu) = self.gpu.take() else {
            self.draw_cpu(s, count, number, frame, out);
            return Ok(());
        };
        let drawn = self.draw_gpu(&mut gpu, s, count, number, frame, out);
        self.gpu = Some(gpu);
        drawn
    }

    /// The GPU failed: this instance draws on the CPU from now on, from what is kept.
    fn fall_back(&mut self, fault: &Fault) {
        eprintln!(
            "segcam Motion: the GPU failed ({}); drawing on the CPU",
            fault.0
        );
        self.use_gpu(false);
    }

    fn draw_cpu<C: Channel>(
        &self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) {
        let first = number - count as i64 + 1;
        let pile = self.pile(s, count, number, out);
        let empty = Motion::default();
        let motion = self
            .motion(number, &basis(s, number, first))
            .unwrap_or(&empty);
        draw_output(out, s, frame, motion, number, pile.shown());
    }

    /// The pile of the tiles a swarm render draws from; empty for other outputs.
    fn pile<C: Channel>(
        &self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        out: &ViewMut<C>,
    ) -> Pile<Placed<'_>> {
        let first = number - count as i64 + 1;
        let mut pile = Pile::new(out.width, out.height, s.max_tiles);
        if s.mode == OutputMode::Swarm {
            for target in (number - targets(s) + 1).max(first)..=number {
                let tiles = self.tiles(target, &basis(s, target, first), s.mirror);
                for (index, tile) in tiles.into_iter().flatten().enumerate() {
                    pile.push(Placed {
                        tile,
                        frame: target,
                        index,
                    });
                }
            }
        }
        pile
    }

    fn draw_gpu<C: Channel>(
        &mut self,
        gpu: &mut Drawer,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        let first = number - count as i64 + 1;
        if let (OutputMode::Overlay, Some(view), true) = (s.mode, frame.as_ref(), self.gpu_overlay)
        {
            let empty = Motion::default();
            let motion = self
                .motion(number, &basis(s, number, first))
                .unwrap_or(&empty);
            return gpu.draw_overlay(out, view, &motion.blobs, number, s);
        }
        if s.mode != OutputMode::Swarm {
            self.draw_cpu(s, count, number, frame, out);
            return Ok(());
        }
        // The swarm draws where its windows draw faster: the choice depends on this render's
        // windows alone, so a kept render draws what a cold one does.
        let canvas = i64::from(out.width) * i64::from(out.height);
        let default = if size_of::<C>() == 1 {
            GPU_FROM_8_BIT
        } else {
            GPU_FROM_DEEPER
        };
        let from = (self.gpu_from.unwrap_or(default) * 1000.0).round() as i64;
        let (drawn, uploads) = {
            let pile = self.pile(s, count, number, out);
            let shown: Vec<&Placed> = pile.shown_entries().collect();
            let drawn = window_pixels(shown.iter().map(|p| p.tile), out.width, out.height);
            let mut uploads: Vec<i64> = shown.iter().map(|p| p.frame).collect();
            uploads.retain(|f| !self.cuts.contains_key(f));
            uploads.dedup();
            (drawn, uploads)
        };
        self.window_share = drawn as f64 / canvas as f64;
        if drawn * 1000 < canvas * from {
            self.draw_cpu(s, count, number, frame, out);
            return Ok(());
        }
        for f in uploads {
            if let Some((_, _, tiles)) = self.tiles.get(&f) {
                let cut = gpu.upload_tiles(tiles, (out.width, out.height))?;
                self.cuts.insert(f, cut);
            }
        }
        let pile = self.pile(s, count, number, out);
        let windows = pile
            .shown_entries()
            .map(|p| (p.tile, self.cuts[&p.frame].slots[p.index]));
        gpu.draw_swarm(out, windows, frame.as_ref(), number, s)
    }

    pub fn reset(&mut self) {
        self.grids.clear();
        self.prints.clear();
        self.motion.clear();
        self.tiles.clear();
        self.cuts.clear();
    }

    /// Whether `frame` has its motion and, for the swarm, its tiles.
    fn complete(&self, frame: i64, basis: &Basis, s: &RenderSettings) -> bool {
        self.motion(frame, basis).is_some()
            && (s.mode != OutputMode::Swarm || self.tiles(frame, basis, s.mirror).is_some())
    }

    fn motion(&self, frame: i64, basis: &Basis) -> Option<&Motion> {
        self.motion
            .get(&frame)
            .filter(|(b, _)| b == basis)
            .map(|(_, m)| m)
    }

    fn tiles(&self, frame: i64, basis: &Basis, mirror: bool) -> Option<&[Tile]> {
        let (b, m, tiles) = self.tiles.get(&frame)?;
        (b == basis && *m == mirror).then_some(tiles.as_slice())
    }

    /// Drops what was kept for `frame`, and the motion and tiles of the frames whose
    /// background it is part of.
    fn forget(&mut self, frame: i64, background: i64) {
        self.grids.remove(&frame);
        self.prints.remove(&frame);
        for later in frame..=frame + background {
            self.motion.remove(&later);
            self.tiles.remove(&later);
            self.cuts.remove(&later);
        }
    }

    fn keep(&mut self, frames: RangeInclusive<i64>) {
        self.grids.retain(|f, _| frames.contains(f));
        self.prints.retain(|f, _| frames.contains(f));
        self.motion.retain(|f, _| frames.contains(f));
        self.tiles.retain(|f, _| frames.contains(f));
        self.cuts.retain(|f, _| frames.contains(f));
    }
}

/// The pixels the swarm's windows draw: each with its shadow, clipped to the canvas, overlaps
/// counted.
fn window_pixels<'t>(tiles: impl Iterator<Item = &'t Tile>, width: i32, height: i32) -> i64 {
    let m = SwarmMetrics::new(height);
    let reach = swarm::shadow_kernel(m.scale).reach;
    let canvas = PixelRect::new(0, 0, width, height);
    tiles
        .map(|t| {
            let f = swarm::tile_extent(t.video, m.title_height);
            let area = PixelRect::new(f.x - reach, f.y - reach, f.w + 2 * reach, f.h + 2 * reach);
            let r = crate::render::intersect(area, canvas);
            i64::from(r.w) * i64::from(r.h)
        })
        .sum()
}

fn detection(s: &RenderSettings) -> Detection {
    Detection {
        detect: s.detect,
        motion: s.motion,
        background_frames: s.background_frames,
        threshold: s.threshold,
        faces: s.faces,
    }
}

/// What `frame` is worked out from in a render whose frames start at `first`.
fn basis(s: &RenderSettings, frame: i64, first: i64) -> Basis {
    Basis {
        detection: detection(s),
        from: (frame - i64::from(history_frames(s))).max(first),
    }
}

/// How many frames up to the current one are drawn from: the trail for the swarm.
fn targets(s: &RenderSettings) -> i64 {
    if s.mode == OutputMode::Swarm {
        i64::from(s.trail_frames.max(1))
    } else {
        1
    }
}

/// A hash of every pixel, read as u64 words in four lanes so it's limited by memory bandwidth.
fn fingerprint<C: Channel>(view: &View<C>) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes = [0u64; 4];
    for y in 0..view.height {
        let row = view.row(y);
        // SAFETY: Argb is four channels with no padding, and a channel is plain data.
        let bytes =
            unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), size_of_val(row)) };
        let (words, rest) = bytes.as_chunks::<8>();
        for chunk in words.chunks(4) {
            for (lane, word) in lanes.iter_mut().zip(chunk) {
                *lane = (*lane ^ u64::from_le_bytes(*word))
                    .wrapping_mul(K)
                    .rotate_left(29);
            }
        }
        for &b in rest {
            lanes[0] = (lanes[0] ^ u64::from(b)).wrapping_mul(K);
        }
    }
    lanes
        .iter()
        .fold(0, |h: u64, &l| (h ^ l).wrapping_mul(K).rotate_left(31))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vfx_core::cpu::Image;

    use super::*;
    use crate::gpu;

    const W: i32 = 160;
    const H: i32 = 90;

    struct Frames<'a>(&'a [Vec<Argb<u8>>]);

    impl FrameSource for Frames<'_> {
        fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
            f(self
                .0
                .get(index)
                .map(|p| Image::U8(View::from_pixels(p, W, H))));
        }
    }

    /// A bright square crossing a grey frame.
    fn crossing() -> Vec<Vec<Argb<u8>>> {
        (0..14)
            .map(|f| {
                (0..W * H)
                    .map(|i| {
                        let (x, y) = (i % W, i / W);
                        let inside = (6 + f * 9..30 + f * 9).contains(&x) && (30..54).contains(&y);
                        let v = if inside { 240 } else { 70 };
                        Argb {
                            alpha: 255,
                            red: v,
                            green: v / 2 + 20,
                            blue: v,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn render(state: &mut Incremental, frames: &[Vec<Argb<u8>>], n: usize) -> Vec<Argb<u8>> {
        let s = RenderSettings {
            mode: OutputMode::Swarm,
            background_frames: 4,
            trail_frames: 8,
            max_tiles: 20,
            ..RenderSettings::default()
        };
        let needed = crate::previous_frames_needed(&s) as usize;
        let clip = &frames[n.saturating_sub(needed)..=n];
        let mut out = vec![Argb::default(); (W * H) as usize];
        let view = &mut ViewMut::from_pixels(&mut out, W, H);
        state.advance(&s, &mut Frames(clip), clip.len(), n as i64, view);
        out
    }

    /// A device lost between renders, as when a driver resets, leaves the instance drawing
    /// what the CPU draws, and no new instance draws on that device. The device is a software
    /// one of the test's own.
    #[test]
    fn a_lost_gpu_falls_back_to_the_cpu() {
        let Ok(dev) = gpu::open("fallback") else {
            eprintln!("no software adapter: skipped");
            return;
        };
        let dev = Arc::new(dev);
        let frames = crossing();
        let mut cpu = Incremental::default();
        let mut on_gpu = Incremental::default();
        on_gpu.use_drawer(gpu::Drawer::on(dev.clone()).expect("a drawer on a working device"));
        on_gpu.gpu_threshold(0.0);
        for n in 0..frames.len() {
            if n == 7 {
                let drawn = on_gpu.gpu().map_or(0, |d| d.drawn);
                assert!(drawn > 0, "drawing on the GPU before the loss");
                dev.device.destroy();
            }
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(n < 7 || a == b, "frame {n} isn't the CPU's after the loss");
        }
        assert!(dev.failure().is_some(), "the loss went unnoticed");
        assert!(on_gpu.gpu().is_none(), "the instance is still on the GPU");
        assert!(
            gpu::Drawer::on(dev).is_none(),
            "a new instance draws on the lost device"
        );
    }
}
