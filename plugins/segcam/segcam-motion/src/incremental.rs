//! segcam Motion worked out incrementally: each frame's grid, motion and tiles are kept, with
//! a hash of the pixels they came from. Where earlier frames can change upstream (OpenFX), a
//! render still reads every frame it depends on and works out again only what a changed frame
//! feeds, so an edit reaches it however far back it is. Where they can't (a host's own history
//! of frames, as in Resolume and the preview: [`FrameSource::fixed`]), it reads only the frames
//! it hasn't worked out, and the current one. Results come from each frame and the settings
//! alone, so render order doesn't change the output.
use std::collections::{BTreeMap, BTreeSet};
use std::ops::RangeInclusive;

use vfx_core::cpu::FrameSource;

use crate::motion::{self, LumaGrid, Motion, MotionSettings, ThresholdSettings};
use crate::pixel::{Argb, Channel, Depth, View, ViewMut};
use crate::render::{
    Detect, FaceParts, OutputMode, Pile, PixelRect, RenderSettings, downsample, draw_output,
    fill_rect, frame_tiles, history_frames,
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

#[derive(Default)]
pub struct Incremental {
    canvas: Option<Canvas>,
    grids: BTreeMap<i64, LumaGrid>,
    /// A hash of each frame's pixels, since edits upstream can leave the grid as it was.
    prints: BTreeMap<i64, u64>,
    motion: BTreeMap<i64, (Detection, Motion)>,
    /// Tiles by frame, with the detection and mirroring they were cut for.
    tiles: BTreeMap<i64, (Detection, bool, Vec<Tile>)>,
}

impl Incremental {
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
        let detection = detection(s);
        let swarm = s.mode == OutputMode::Swarm;
        let background = i64::from(history_frames(s));
        let targets = targets(s);
        let first = number - count as i64 + 1;

        // Frames the host didn't hand over count as empty, as in a cold render, but nothing
        // worked out from one is kept, so it is read again next time. Nor is motion whose
        // background was cut short by the start of this render's frames.
        let mut missing = BTreeSet::new();
        let mut short = BTreeSet::new();
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
            self.lacking(s, oldest, number, background)
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
                if frame > number - targets && !self.complete(frame, &detection, s) {
                    if frame - background < first {
                        short.insert(frame);
                    }
                    let motion = match self.motion(frame, &detection) {
                        Some(motion) => motion.clone(),
                        None => {
                            let empty = LumaGrid::default();
                            let grid = self.grids.get(&frame).unwrap_or(&empty);
                            let mut motion = match s.detect {
                                Detect::Motion => {
                                    let history = ((frame - background).max(first)..frame)
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
                            self.motion.insert(frame, (detection, motion.clone()));
                            motion
                        }
                    };
                    if swarm {
                        let tiles = frame_tiles(&motion, view.as_ref(), frame, w, h, s.mirror);
                        self.tiles.insert(frame, (detection, s.mirror, tiles));
                    }
                }
                if frame == number {
                    self.draw(s, count, number, view, out);
                }
            });
        }
        for &frame in &missing {
            self.forget(frame, background);
        }
        for frame in short {
            self.motion.remove(&frame);
            self.tiles.remove(&frame);
        }
        let reach = background + targets;
        self.keep(number - reach..=number + reach);
    }

    /// The frames from `oldest` a render of `number` has to read: those whose motion or tiles
    /// aren't kept, the frames before them whose grids aren't, and `number`, which it draws.
    fn lacking(
        &self,
        s: &RenderSettings,
        oldest: i64,
        number: i64,
        background: i64,
    ) -> BTreeSet<i64> {
        let detection = detection(s);
        let mut wanted = BTreeSet::from([number]);
        for target in (number - targets(s) + 1).max(oldest)..=number {
            if !self.complete(target, &detection, s) {
                wanted.insert(target);
                let before = (target - background).max(oldest)..target;
                wanted.extend(before.filter(|k| !self.grids.contains_key(k)));
            }
        }
        wanted
    }

    /// Draws frame `number`, the last of `count`, from what is kept. `frame` is its pixels,
    /// which every output but the swarm draws over.
    pub fn draw<C: Channel>(
        &self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) {
        let detection = detection(s);
        let mut pile = Pile::<&Tile>::new(out.width, out.height, s.max_tiles);
        if s.mode == OutputMode::Swarm {
            let first = number - count as i64 + 1;
            for target in (number - targets(s) + 1).max(first)..=number {
                self.tiles(target, &detection, s.mirror)
                    .into_iter()
                    .flatten()
                    .for_each(|t| pile.push(t));
            }
        }
        let empty = Motion::default();
        let motion = self.motion(number, &detection).unwrap_or(&empty);
        draw_output(out, s, frame, motion, number, pile.shown());
    }

    pub fn reset(&mut self) {
        self.grids.clear();
        self.prints.clear();
        self.motion.clear();
        self.tiles.clear();
    }

    /// Whether `frame` has its motion and, for the swarm, its tiles.
    fn complete(&self, frame: i64, detection: &Detection, s: &RenderSettings) -> bool {
        self.motion(frame, detection).is_some()
            && (s.mode != OutputMode::Swarm || self.tiles(frame, detection, s.mirror).is_some())
    }

    fn motion(&self, frame: i64, detection: &Detection) -> Option<&Motion> {
        self.motion
            .get(&frame)
            .filter(|(d, _)| d == detection)
            .map(|(_, m)| m)
    }

    fn tiles(&self, frame: i64, detection: &Detection, mirror: bool) -> Option<&[Tile]> {
        let (d, m, tiles) = self.tiles.get(&frame)?;
        (d == detection && *m == mirror).then_some(tiles.as_slice())
    }

    /// Drops what was kept for `frame`, and the motion and tiles of the frames whose
    /// background it is part of.
    fn forget(&mut self, frame: i64, background: i64) {
        self.grids.remove(&frame);
        self.prints.remove(&frame);
        for later in frame..=frame + background {
            self.motion.remove(&later);
            self.tiles.remove(&later);
        }
    }

    fn keep(&mut self, frames: RangeInclusive<i64>) {
        self.grids.retain(|f, _| frames.contains(f));
        self.prints.retain(|f, _| frames.contains(f));
        self.motion.retain(|f, _| frames.contains(f));
        self.tiles.retain(|f, _| frames.contains(f));
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
