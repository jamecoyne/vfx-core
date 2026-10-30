//! segcam Motion worked out incrementally: each frame's grid, motion and tiles are kept, so a
//! render only reads the frames it doesn't have yet. Results come from each frame and the
//! settings alone, so render order doesn't change the output.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum Advance {
    Drawn,
    /// The source changed upstream: everything kept was dropped and nothing drawn. Render
    /// again from cold.
    Discontinuity,
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
    /// Renders the last of `count` frames, frame `number`, into `out`, reading only the frames
    /// [`Incremental::lacking`] names. The current frame is always read, last, since its grid
    /// shows whether the source changed.
    pub fn advance<C: Depth>(
        &mut self,
        s: &RenderSettings,
        frames: &mut dyn FrameSource,
        count: usize,
        number: i64,
        out: &mut ViewMut<C>,
    ) -> Advance {
        if out.is_empty() {
            return Advance::Drawn;
        }
        if count == 0 {
            let all = PixelRect::new(0, 0, out.width, out.height);
            fill_rect(out, all, Argb::default());
            return Advance::Drawn;
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

        let mut changed = false;
        for frame in self.lacking(s, count, number) {
            frames.with_frame((frame - first) as usize, &mut |image| {
                let view = image.and_then(C::view);
                let grid = view
                    .as_ref()
                    .map(|v| downsample(v, s.grid_width))
                    .unwrap_or_default();
                let print = view.as_ref().map_or(0, fingerprint);
                if frame == number
                    && (self.grids.get(&frame).is_some_and(|kept| *kept != grid)
                        || self.prints.get(&frame).is_some_and(|kept| *kept != print))
                {
                    changed = true;
                    return;
                }
                self.grids.insert(frame, grid);
                self.prints.insert(frame, print);
                if frame > number - targets && !self.complete(frame, &detection, s) {
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
        if changed {
            self.reset();
            return Advance::Discontinuity;
        }
        let reach = background + targets;
        self.keep(number - reach..=number + reach);
        Advance::Drawn
    }

    /// The frames [`Incremental::advance`] would read for this render.
    pub fn lacking(&self, s: &RenderSettings, count: usize, number: i64) -> BTreeSet<i64> {
        let detection = detection(s);
        let background = i64::from(history_frames(s));
        let first = number - count as i64 + 1;
        let mut wanted = BTreeSet::new();
        for target in (number - targets(s) + 1).max(first)..=number {
            if self.complete(target, &detection, s) {
                continue;
            }
            wanted.insert(target);
            let before = (target - background).max(first)..target;
            wanted.extend(before.filter(|k| !self.grids.contains_key(k)));
        }
        wanted.insert(number);
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
