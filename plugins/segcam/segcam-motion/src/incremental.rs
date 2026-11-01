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

use crate::gpu::{self, Cut, Drawer, Fault};
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
    /// For a frame that reuses the faces of the frame before, that frame's `from`, since the
    /// faces keep the IDs they had there.
    faces: Option<i64>,
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
    /// there. The drawer is made by the first render that draws on the GPU once the device is
    /// open, as hosts make instances often.
    gpu: Option<Box<Drawer>>,
    cuts: BTreeMap<i64, Cut>,
    gpu_wanted: bool,
    gpu_overlay: bool,
    /// The last fault told of, so a fault that repeats every render is told of once.
    told: Option<String>,
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
        // While the device opens, renders draw on the CPU.
        if self.gpu_wanted && self.gpu.is_none() && on_gpu && !gpu::opening() {
            let drawer = gpu::device().and_then(Drawer::on).map(Box::new);
            self.gpu_wanted = drawer.is_some();
            self.set_drawer(drawer);
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
        // What a changed frame feeds: the frames it is background to, and with faces the next.
        let feeds = background.max(i64::from(s.faces.any()));
        let targets = targets(s);
        let first = number - count as i64 + 1;
        let start = start(s, number, first);

        // Frames the host didn't hand over count as empty, as in a cold render, but nothing
        // worked out from one is kept, so it is read again next time.
        let mut missing = BTreeSet::new();
        let oldest = (start - background).max(first);
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
                        self.forget(frame, feeds);
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
                                self.forget(frame, feeds);
                            }
                            self.grids.insert(frame, downsample(v, s.grid_width));
                            self.prints.insert(frame, print);
                        }
                    }
                }
                let previous = basis(s, frame - 1, first);
                let basis = basis(s, frame, first);
                if frame >= start && !self.complete(frame, &basis, s) {
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
                            // The frame before was worked out first, unless it is missing.
                            let reused = basis
                                .faces
                                .filter(|_| !missing.contains(&(frame - 1)))
                                .and_then(|_| self.motion(frame - 1, &previous));
                            if let Some(before) = reused {
                                motion.blobs.extend(reused_faces(before, frame - 1));
                            } else if let Some(view) = view.as_ref().filter(|_| s.faces.any()) {
                                let found = crate::faces::find(view, s.faces, grid.count());
                                motion.blobs.extend(found);
                            }
                            self.motion.insert(frame, (basis, motion.clone()));
                            motion
                        }
                    };
                    if swarm {
                        // A trail of one draws no frame before, so reused faces need tiles here.
                        let reused = targets == 1;
                        let tiles =
                            frame_tiles(&motion, view.as_ref(), frame, w, h, s.mirror, reused);
                        self.tiles.insert(frame, (basis, s.mirror, tiles));
                        self.cuts.remove(&frame);
                    }
                }
                if frame == number {
                    self.draw(s, count, number, view, out);
                }
            });
        }
        for &frame in &missing {
            self.forget(frame, feeds);
        }
        let reach = background + targets;
        self.keep(number - reach..=number + reach);
    }

    /// The frames from `first` a render of `number` has to read: those whose motion or tiles
    /// aren't kept, the frames before them whose grids aren't, and `number`, which it draws.
    fn lacking(&self, s: &RenderSettings, first: i64, number: i64) -> BTreeSet<i64> {
        let mut wanted = BTreeSet::from([number]);
        for target in start(s, number, first)..=number {
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
    /// which every output but the swarm draws over. A render the GPU fails is drawn on the
    /// CPU from the same kept state.
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
        let Some(mut gpu) = self.gpu.take_if(|gpu| !gpu.resting()) else {
            self.draw_cpu(s, count, number, frame, out);
            return Ok(());
        };
        gpu.begin();
        let drawn = self.draw_gpu(&mut gpu, s, count, number, frame, out);
        self.gpu = Some(gpu);
        drawn
    }

    /// The GPU failed this render. A lost device is given up for good by this instance; after
    /// anything else the next render tries the GPU again, unless it failed a few lately.
    fn fall_back(&mut self, fault: &Fault) {
        if self.told.as_ref() != Some(&fault.0) {
            eprintln!(
                "segcam Motion: the GPU failed ({}); drawing on the CPU",
                fault.0
            );
            self.told = Some(fault.0.clone());
        }
        match &mut self.gpu {
            Some(gpu) if gpu.lost() => {
                self.gpu_wanted = false;
                self.set_drawer(None);
            }
            Some(gpu) => gpu.fault(),
            None => {}
        }
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
        let pile = pile(&self.tiles, s, count, number, out.width, out.height);
        let empty = Motion::default();
        let motion = self
            .motion(number, &basis(s, number, first))
            .unwrap_or(&empty);
        draw_output(out, s, frame, motion, number, pile.shown());
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
        let (w, h) = (out.width, out.height);
        let canvas = i64::from(w) * i64::from(h);
        let default = if size_of::<C>() == 1 {
            GPU_FROM_8_BIT
        } else {
            GPU_FROM_DEEPER
        };
        let from = (self.gpu_from.unwrap_or(default) * 1000.0).round() as i64;
        let m = SwarmMetrics::new(h);
        let shadow = swarm::shadow_kernel(m.scale);
        let pile = pile(&self.tiles, s, count, number, w, h);
        let shown: Vec<&Placed> = pile.shown_entries().collect();
        let drawn = window_pixels(shown.iter().map(|p| p.tile), &m, &shadow, w, h);
        self.window_share = drawn as f64 / canvas as f64;
        let empty = Motion::default();
        let motion = motion_of(&self.motion, number, &basis(s, number, first)).unwrap_or(&empty);
        if drawn * 1000 < canvas * from {
            draw_output(out, s, frame, motion, number, shown.iter().map(|p| p.tile));
            return Ok(());
        }
        // Only frames with a shown tile keep their place in the atlas. A full atlas draws this
        // render on the CPU; the frames the trail drops free it for the next.
        let wanted: BTreeSet<i64> = shown.iter().map(|p| p.frame).collect();
        self.cuts.retain(|f, _| wanted.contains(f));
        for f in wanted {
            if self.cuts.contains_key(&f) {
                continue;
            }
            let Some((_, _, tiles)) = self.tiles.get(&f) else {
                continue;
            };
            match gpu.upload_tiles(tiles, (w, h), &shadow)? {
                Some(cut) => self.cuts.insert(f, cut),
                None => {
                    draw_output(out, s, frame, motion, number, shown.iter().map(|p| p.tile));
                    return Ok(());
                }
            };
        }
        let windows = shown
            .iter()
            .map(|p| (p.tile, self.cuts[&p.frame].slots[p.index]));
        gpu.draw_swarm(out, windows, frame.as_ref(), number, s, &shadow)
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
        motion_of(&self.motion, frame, basis)
    }

    fn tiles(&self, frame: i64, basis: &Basis, mirror: bool) -> Option<&[Tile]> {
        tiles_of(&self.tiles, frame, basis, mirror)
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

/// The motion kept for `frame`, if it was worked out for `basis`.
fn motion_of<'a>(
    kept: &'a BTreeMap<i64, (Basis, Motion)>,
    frame: i64,
    basis: &Basis,
) -> Option<&'a Motion> {
    kept.get(&frame).filter(|(b, _)| b == basis).map(|(_, m)| m)
}

/// The tiles kept for `frame`, if they were cut for `basis` and `mirror`.
fn tiles_of<'a>(
    kept: &'a BTreeMap<i64, (Basis, bool, Vec<Tile>)>,
    frame: i64,
    basis: &Basis,
    mirror: bool,
) -> Option<&'a [Tile]> {
    let (b, m, tiles) = kept.get(&frame)?;
    (b == basis && *m == mirror).then_some(tiles.as_slice())
}

/// The pile of the `kept` tiles a swarm render of frame `number`, the last of `count`, draws
/// from on a `width` x `height` canvas; empty for other outputs.
fn pile<'a>(
    kept: &'a BTreeMap<i64, (Basis, bool, Vec<Tile>)>,
    s: &RenderSettings,
    count: usize,
    number: i64,
    width: i32,
    height: i32,
) -> Pile<Placed<'a>> {
    let first = number - count as i64 + 1;
    let mut pile = Pile::new(width, height, s.max_tiles);
    if s.mode == OutputMode::Swarm {
        for target in (number - targets(s) + 1).max(first)..=number {
            let tiles = tiles_of(kept, target, &basis(s, target, first), s.mirror);
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

/// The pixels the swarm's windows draw: each with its shadow, clipped to the canvas, overlaps
/// counted.
fn window_pixels<'t>(
    tiles: impl Iterator<Item = &'t Tile>,
    m: &SwarmMetrics,
    shadow: &swarm::ShadowKernel,
    width: i32,
    height: i32,
) -> i64 {
    let reach = shadow.reach;
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
    let from = |frame: i64| (frame - i64::from(history_frames(s))).max(first);
    Basis {
        detection: detection(s),
        from: from(frame),
        faces: (faces_from(s, frame, first) != frame).then(|| from(frame - 1)),
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

/// The frame whose faces `frame` shows in a render whose frames start at `first`. As in
/// segcam, faces run on every 2nd frame, the even ones; an odd frame reuses the frame before's.
fn faces_from(s: &RenderSettings, frame: i64, first: i64) -> i64 {
    if s.faces.any() && frame.rem_euclid(2) == 1 && frame > first {
        frame - 1
    } else {
        frame
    }
}

/// The first frame whose motion a render of `number` works out: the first it draws, or the
/// frame before if that one's faces are reused.
fn start(s: &RenderSettings, number: i64, first: i64) -> i64 {
    faces_from(s, (number - targets(s) + 1).max(first), first)
}

/// The faces in `motion`, frame `frame`'s, as the next frame reuses them: with their IDs.
fn reused_faces(motion: &Motion, frame: i64) -> impl Iterator<Item = motion::Blob> + '_ {
    use motion::SegmentKind as Kind;
    motion
        .blobs
        .iter()
        .enumerate()
        .filter(|(_, b)| !matches!(b.kind, Kind::Motion | Kind::Blob))
        .map(move |(i, b)| motion::Blob {
            reused: Some((frame, i)),
            ..*b
        })
}

/// A hash of every pixel, read as u64 words in four lanes so it's limited by memory bandwidth.
fn fingerprint<C: Channel>(view: &View<C>) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes = [0u64; 4];
    for y in 0..view.height {
        let bytes: &[u8] = bytemuck::cast_slice(view.row(y));
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
    use std::time::Duration;

    use vfx_core::cpu::Image;
    use vfx_core::wgpu;

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

    /// The GPU tests take turns: several devices open at once on WARP crash it now and then
    /// (an access violation in d3d10warp.dll) unless the D3D12 debug layer is on.
    fn turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A device of the test's own, on the adapter `SEGCAM_GPU_ADAPTER` picks or on a software
    /// one where there is none; `None` where there is no adapter at all.
    fn own_device() -> Option<Arc<gpu::Device>> {
        let choice = std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default();
        match gpu::open(&choice).or_else(|_| gpu::open("fallback")) {
            Ok(dev) => Some(Arc::new(dev)),
            Err(e) => {
                eprintln!("no adapter ({e}): skipped");
                None
            }
        }
    }

    /// An instance drawing every swarm render on `dev`.
    fn on_device(dev: &Arc<gpu::Device>) -> Incremental {
        let mut on_gpu = Incremental::default();
        on_gpu.use_drawer(gpu::Drawer::on(dev.clone()).expect("a drawer on a working device"));
        on_gpu.gpu_threshold(0.0);
        on_gpu
    }

    fn drawn(state: &Incremental) -> u64 {
        state.gpu().map_or(0, |d| d.drawn)
    }

    fn faults(state: &Incremental) -> u64 {
        state.gpu().map_or(0, |d| d.faults)
    }

    /// Whether every channel of `b` is within a level of `a`'s: what the GPU's drawing promises.
    fn within_a_level(a: &[Argb<u8>], b: &[Argb<u8>]) -> bool {
        a.iter().zip(b).all(|(p, q)| {
            [
                (p.alpha, q.alpha),
                (p.red, q.red),
                (p.green, q.green),
                (p.blue, q.blue),
            ]
            .iter()
            .all(|&(x, y)| x.abs_diff(y) <= 1)
        })
    }

    /// A device lost between renders, as when a driver resets, leaves the instance drawing
    /// what the CPU draws, and no new instance draws on that device.
    #[test]
    fn a_lost_gpu_falls_back_to_the_cpu() {
        let _turn = turn();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = Incremental::default();
        let mut on_gpu = on_device(&dev);
        for n in 0..frames.len() {
            if n == 7 {
                assert!(drawn(&on_gpu) > 0, "drawing on the GPU before the loss");
                dev.device.destroy();
            }
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(n < 7 || a == b, "frame {n} isn't the CPU's after the loss");
        }
        assert!(dev.lost().is_some(), "the loss went unnoticed");
        assert!(on_gpu.gpu().is_none(), "the instance is still on the GPU");
        assert!(!on_gpu.gpu_wanted, "the instance still looks for a GPU");
        assert!(
            gpu::Drawer::on(dev).is_none(),
            "a new instance draws on the lost device"
        );
    }

    /// A validation error leaves the device usable: it is counted, not kept as a failure, so
    /// new instances still draw on it and a render that didn't cause it isn't charged with it.
    #[test]
    fn an_error_leaves_the_device_usable() {
        let _turn = turn();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        dev.misuse();
        let _ = dev.device.poll(wgpu::PollType::Poll);
        assert_eq!(dev.errors().0, 1, "the error went unreported");
        assert!(dev.lost().is_none(), "an error lost the device");
        let mut on_gpu = on_device(&dev);
        render(&mut on_gpu, &frames, 0);
        assert_eq!(drawn(&on_gpu), 1, "the render didn't draw on the GPU");
        assert_eq!(
            faults(&on_gpu),
            0,
            "an earlier error was charged to the render"
        );
    }

    /// An instance taken off the GPU mid sequence draws from then on what a CPU only instance
    /// draws, byte for byte, from what it kept.
    #[test]
    fn leaving_the_gpu_keeps_the_cpu_output() {
        let _turn = turn();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = Incremental::default();
        let mut on_gpu = on_device(&dev);
        for n in 0..frames.len() {
            if n == 7 {
                assert!(drawn(&on_gpu) > 0, "drawing on the GPU before the switch");
                on_gpu.use_gpu(false);
            }
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_a_level(&a, &b), "frame {n} is off the CPU's");
            assert!(
                n < 7 || a == b,
                "frame {n} isn't the CPU's after the switch"
            );
        }
        assert!(on_gpu.gpu().is_none(), "the instance is still on the GPU");
        assert!(dev.lost().is_none(), "the device was given up");
    }

    /// A render the GPU can't take, here a canvas larger than the device binds, is drawn on
    /// the CPU byte for byte, and the next render is back on the GPU.
    #[test]
    fn a_fault_falls_back_for_one_render() {
        let _turn = turn();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = Incremental::default();
        let mut on_gpu = on_device(&dev);
        for n in 0..frames.len() {
            let limit = if n == 7 { 1024 } else { u64::MAX };
            on_gpu.gpu.as_mut().expect("on the GPU").set_limit(limit);
            let before = drawn(&on_gpu);
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_a_level(&a, &b), "frame {n} is off the CPU's");
            if n == 7 {
                assert_eq!(a, b, "the faulted frame isn't the CPU's");
                assert_eq!(drawn(&on_gpu), before, "the faulted frame counts as drawn");
            } else {
                assert_eq!(
                    drawn(&on_gpu),
                    before + 1,
                    "frame {n} wasn't drawn on the GPU"
                );
            }
        }
        assert_eq!(faults(&on_gpu), 1);
        assert!(on_gpu.gpu().is_some(), "the instance left the GPU");
        assert!(dev.lost().is_none(), "the device was given up");
    }

    /// A draw that isn't back in time is drawn on the CPU, and the next render is back on
    /// the GPU with a fresh readback buffer. Whether the first draw is in time depends on
    /// the GPU, so only what must hold either way is checked.
    #[test]
    fn a_slow_readback_falls_back_for_one_render() {
        let _turn = turn();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = Incremental::default();
        let mut on_gpu = on_device(&dev);
        for n in 0..frames.len() {
            let wait = if n == 7 {
                std::time::Duration::ZERO
            } else {
                std::time::Duration::from_secs(10)
            };
            on_gpu.gpu.as_mut().expect("on the GPU").set_wait(wait);
            let before = drawn(&on_gpu);
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_a_level(&a, &b), "frame {n} is off the CPU's");
            if n != 7 {
                assert_eq!(
                    drawn(&on_gpu),
                    before + 1,
                    "frame {n} wasn't drawn on the GPU"
                );
            }
        }
        assert!(faults(&on_gpu) <= 1);
        assert!(on_gpu.gpu().is_some(), "the instance left the GPU");
        assert!(dev.lost().is_none(), "the device was given up");
    }

    /// A GPU that fails every other render rests too: after a few faults among its last
    /// renders, the renders are the CPU's byte for byte and don't touch the GPU, and once the
    /// rest is over the GPU draws again.
    #[test]
    fn faults_close_together_rest_on_the_cpu() {
        let _turn = turn();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = Incremental::default();
        let mut on_gpu = on_device(&dev);
        // Frames 0, 2 and 4 fault; from 5 the instance rests, until the test ends it at 8.
        let resting_frames = 5..8;
        for n in 0..frames.len() {
            let drawer = on_gpu.gpu.as_mut().expect("on the GPU");
            let failing = n < 8 && n % 2 == 0 || resting_frames.contains(&n);
            drawer.set_limit(if failing { 1024 } else { u64::MAX });
            if n == resting_frames.end {
                assert!(drawer.resting(), "still trying the GPU at frame {n}");
                drawer.set_rest(Duration::ZERO);
            }
            let resting = drawer.resting();
            let (before, faulted) = (drawn(&on_gpu), faults(&on_gpu));
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_a_level(&a, &b), "frame {n} is off the CPU's");
            assert_eq!(resting, resting_frames.contains(&n), "frame {n}");
            if resting {
                assert_eq!(a, b, "resting frame {n} isn't the CPU's");
                assert_eq!(drawn(&on_gpu), before, "resting frame {n} drew on the GPU");
                assert_eq!(faults(&on_gpu), faulted, "resting frame {n} tried the GPU");
            } else if !failing {
                assert_eq!(drawn(&on_gpu), before + 1, "frame {n} wasn't on the GPU");
            }
        }
        assert_eq!(faults(&on_gpu), u64::from(gpu::TRIES));
        assert!(dev.lost().is_none(), "the device was given up");
    }

    /// An atlas too small for the trail draws the renders it can't hold on the CPU, and the
    /// GPU draws again once the trail moves on; the instance never leaves it.
    #[test]
    fn a_full_atlas_falls_back_for_one_render() {
        let _turn = turn();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = Incremental::default();
        let mut drawer = gpu::Drawer::on(dev.clone()).expect("a drawer on a working device");
        // Room for the canvas and a few frames' tiles, and no more.
        let bytes = (W * H * 4) as u64 + 4096;
        drawer.small_atlas(1024);
        drawer.set_limit(bytes);
        let mut on_gpu = Incremental::default();
        on_gpu.use_drawer(drawer);
        on_gpu.gpu_threshold(0.0);
        for n in 0..frames.len() {
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_a_level(&a, &b), "frame {n} is off the CPU's");
        }
        let drawn = drawn(&on_gpu);
        assert!(drawn > 0, "nothing was drawn on the GPU");
        assert!(
            drawn < frames.len() as u64,
            "every frame fit the atlas: the test proves nothing"
        );
        assert_eq!(faults(&on_gpu), 0, "a full atlas is not a fault");
        assert!(on_gpu.gpu().is_some(), "the instance left the GPU");
        assert!(dev.lost().is_none(), "the device was given up");
    }
}
