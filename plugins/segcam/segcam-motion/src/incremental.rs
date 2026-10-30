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

use crate::gpu::{Cut, Drawer, Slot};
use crate::motion::{self, LumaGrid, Motion, MotionSettings, ThresholdSettings};
use crate::pixel::{Argb, Channel, Depth, View, ViewMut};
use crate::render::{
    Detect, FaceParts, OutputMode, Pile, PiledTile, PixelRect, RenderSettings, downsample,
    draw_output, fill_rect, frame_tiles, history_frames,
};
use crate::swarm::Tile;

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
    /// Drawing on the GPU: then tiles keep their pixels in its atlas, cut by frame here.
    gpu: Option<Box<Drawer>>,
    cuts: BTreeMap<i64, Cut>,
}

/// A tile in the pile with its place in the GPU's atlas.
struct Placed<'a> {
    tile: &'a Tile,
    slot: Slot,
}

impl PiledTile for Placed<'_> {
    fn tile(&self) -> &Tile {
        self.tile
    }
}

impl Incremental {
    /// Draws Overlay and Swarm on the GPU, or back on the CPU. False if there is no GPU.
    pub fn use_gpu(&mut self, on: bool) -> bool {
        self.reset();
        self.gpu = on.then(Drawer::new).flatten().map(Box::new);
        self.gpu.is_some()
    }

    pub fn gpu(&self) -> Option<&Drawer> {
        self.gpu.as_deref()
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
        let mut atlas_full = false;
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
                        let pixels = view.as_ref().filter(|_| self.gpu.is_none());
                        let tiles = frame_tiles(&motion, pixels, frame, w, h, s.mirror);
                        if let Some(gpu) = &mut self.gpu {
                            let blobs = &motion.blobs;
                            match gpu.cut(view.as_ref(), frame, blobs, &tiles, h, s.mirror) {
                                Some(cut) => _ = self.cuts.insert(frame, cut),
                                None => atlas_full = true,
                            }
                        }
                        self.tiles.insert(frame, (basis, s.mirror, tiles));
                    }
                }
                if frame == number {
                    self.draw(s, count, number, view, out);
                }
            });
        }
        if atlas_full {
            eprintln!("segcam Motion: the GPU's tiles are full; drawing on the CPU");
            self.use_gpu(false);
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
    /// which every output but the swarm draws over.
    pub fn draw<C: Channel>(
        &mut self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) {
        if let Some(mut gpu) = self.gpu.take() {
            self.draw_gpu(&mut gpu, s, count, number, frame, out);
            self.gpu = Some(gpu);
            return;
        }
        let first = number - count as i64 + 1;
        let mut pile = Pile::<&Tile>::new(out.width, out.height, s.max_tiles);
        if s.mode == OutputMode::Swarm {
            for target in (number - targets(s) + 1).max(first)..=number {
                self.tiles(target, &basis(s, target, first), s.mirror)
                    .into_iter()
                    .flatten()
                    .for_each(|t| pile.push(t));
            }
        }
        let empty = Motion::default();
        let motion = self
            .motion(number, &basis(s, number, first))
            .unwrap_or(&empty);
        draw_output(out, s, frame, motion, number, pile.shown());
    }

    fn draw_gpu<C: Channel>(
        &self,
        gpu: &mut Drawer,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) {
        let first = number - count as i64 + 1;
        let empty = Motion::default();
        let motion = self
            .motion(number, &basis(s, number, first))
            .unwrap_or(&empty);
        match (s.mode, frame.as_ref()) {
            (OutputMode::Swarm, _) => {
                let mut pile = Pile::<Placed>::new(out.width, out.height, s.max_tiles);
                for target in (number - targets(s) + 1).max(first)..=number {
                    let tiles = self.tiles(target, &basis(s, target, first), s.mirror);
                    if let (Some(tiles), Some(cut)) = (tiles, self.cuts.get(&target)) {
                        for (tile, &slot) in tiles.iter().zip(&cut.slots) {
                            pile.push(Placed { tile, slot });
                        }
                    }
                }
                let windows = pile.shown_entries().map(|p| (p.tile, p.slot));
                gpu.draw_swarm(out, windows, frame.as_ref(), number, s);
            }
            (OutputMode::Overlay, Some(view)) => {
                gpu.draw_overlay(out, view, &motion.blobs, number, s);
            }
            _ => draw_output(out, s, frame, motion, number, std::iter::empty()),
        }
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
