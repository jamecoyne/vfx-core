//! Incremental segcam Motion render. Caches each frame's luma grid, analysis and tiles, keyed
//! by a hash of its pixels and an `AnalysisKey`. Where earlier frames can change upstream
//! (OpenFX), a render reads every frame it depends on and recomputes only what a changed hash
//! invalidates. Where they are immutable ([`FrameSource::fixed`]: a host's own frame history, as
//! in Resolume and the preview, or Resolve while it streams), it reads only uncached frames and
//! the current one, so an edit in Resolve appears when that frame renders or at the next check.
//! Output depends only on pixels and settings, never on render order, except while a live source
//! leaves frames out. The key includes the first background frame, so analysis computed near the
//! start of the frames stays valid.
//!
//! Sources holding only their newest frames ([`FrameSource::held`]) cannot supply older frames
//! again: their tiles are never recut, so new settings reach new windows only (as segcam 1.1),
//! and tiles the pile no longer draws release their pixels.
use std::collections::{BTreeMap, BTreeSet};
use std::ops::RangeInclusive;

use vfx_core::cpu::FrameSource;
use vfx_core::wgpu;
use vfx_core::{Reading, Report};
use vfx_plugin::vfx_host::by_rows;

use crate::gpu::{self, AtlasAllocation, Drawer, Fault};
use crate::motion::{self, LumaGrid, MotionSettings, Segmentation, ThresholdSettings};
use crate::pixel::{Argb, Channel, Depth, View, ViewMut};
use crate::render::{
    AnalysisStep, DetectionMode, FaceParts, OutputMode, Pile, PileEntry, PixelRect, RenderSettings,
    cut_frame_tiles, downsample, draw_output, fill_rect, history_frames,
};
use crate::swarm::{self, SwarmMetrics, Tile};

/// What every cached frame shares; a change clears the whole history.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HistoryFormat {
    width: i32,
    height: i32,
    channel_bytes: usize,
    grid_width: i32,
}

/// Settings a frame's segments depend on, besides the grids.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DetectionSettings {
    detection_mode: DetectionMode,
    motion: MotionSettings,
    background_frames: i32,
    every: i64,
    threshold: ThresholdSettings,
    faces: FaceParts,
}

/// Cache key of a frame's analysis and tiles, besides its pixel hash.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AnalysisKey {
    detection: DetectionSettings,
    /// First background frame: later than usual near the start of the frames.
    background_start: i64,
    /// For a frame reusing an earlier analysis's faces: that frame and its `background_start`,
    /// since the faces keep the IDs they had there.
    face_source: Option<(i64, i64)>,
}

/// What a render knows of a frame's pixels from having read them.
#[derive(Clone, Copy, PartialEq)]
enum PixelsRead {
    /// Their hash: an upstream edit can leave the luma grid unchanged.
    Hash(u64),
    /// A live source numbers each frame once, so it doesn't pay for a hash. Any later check that
    /// does hash the frame finds it changed, so it is worked out again.
    NotHashed,
}

/// A frame's analysis, with the key it was computed for.
struct KeptAnalysis {
    key: AnalysisKey,
    segments: Segmentation,
}

/// A frame's tiles, with the key and mirroring they were cut for.
struct KeptTiles {
    key: AnalysisKey,
    mirror: bool,
    tiles: Vec<Tile>,
}

/// What is kept of one frame. Each part comes and goes on its own: a change to a frame drops the
/// analysis of the frames that used it as background, and keeps their grids.
#[derive(Default)]
struct KeptFrame {
    grid: Option<LumaGrid>,
    read: Option<PixelsRead>,
    analysis: Option<KeptAnalysis>,
    tiles: Option<KeptTiles>,
}

#[derive(Default)]
pub struct IncrementalRenderer {
    history_format: Option<HistoryFormat>,
    kept: BTreeMap<i64, KeptFrame>,
    /// Created on the first GPU render after the device opens, not per instance, because hosts
    /// create instances often. A frame's tiles enter the atlas on first GPU draw.
    drawer: Option<Box<Drawer>>,
    atlas: AtlasPlacements,
    /// Whether the last render's source is live, so it must not stall to stay exact.
    source_is_live: bool,
    gpu_wanted: bool,
    gpu_overlay: bool,
    /// Suppresses repeat logs of the same message.
    last_reported_fault: Option<String>,
    /// See [`Self::set_gpu_min_coverage`].
    gpu_min_coverage: Option<f64>,
    /// The last render's draw hasn't been read back from the GPU yet (browsers).
    unfinished: bool,
    /// `None` before the first render.
    drew_on_gpu: Option<bool>,
    faces_wanted: bool,
    /// Face detection for live renders, which never wait for it.
    live_faces: crate::faces::LiveFaces,
    readings: Vec<Reading>,
    /// Live renders act as if the face models were still loading.
    #[cfg(test)]
    simulate_models_loading: bool,
    /// Face detection calls so far.
    #[cfg(test)]
    face_detection_calls: u32,
    /// A host's GPU frame the render may draw into instead of its output.
    target: Option<TargetOffer>,
}

/// A host's frame on `device`, offered to one render ([`vfx_core::CpuEffect::render_to`]), and
/// whether the render drew into it.
struct TargetOffer {
    device: wgpu::Device,
    frame: vfx_core::Frame,
    drawn_into: bool,
}

/// What stays the same through one render.
struct RenderPlan<'a> {
    s: &'a RenderSettings,
    number: i64,
    /// The first frame the source can reach in this render.
    first: i64,
    canvas: (i32, i32),
    to_analyse: BTreeSet<i64>,
    /// How many frames back a frame's background reaches.
    background: i64,
    /// Frames using a frame as background and, with faces, the next analysis.
    invalidation_reach: i64,
    trail_len: i64,
    fixed: bool,
    live: bool,
    /// A live source numbers each frame once, so a number says which frame it is.
    numbers_identify_frames: bool,
}

impl<'a> RenderPlan<'a> {
    fn new(
        s: &'a RenderSettings,
        source: &dyn FrameSource,
        count: usize,
        number: i64,
        canvas: (i32, i32),
    ) -> Self {
        let first = number - count as i64 + 1;
        let background = i64::from(history_frames(s));
        Self {
            s,
            number,
            first,
            canvas,
            to_analyse: frames_to_analyse(s, number, first),
            background,
            invalidation_reach: background.max(i64::from(s.faces.any()) * s.analysis_step.every),
            trail_len: trail_frame_count(s),
            fixed: source.fixed(),
            live: source.live(),
            numbers_identify_frames: source.fixed() && source.live(),
        }
    }

    /// What a render must read when earlier frames may have changed. Frames between analyses are
    /// no analysis's input.
    fn frames_feeding_analysis(&self) -> BTreeSet<i64> {
        let oldest_analysed = self
            .to_analyse
            .first()
            .map_or(self.number, |&f| f.min(self.number));
        let oldest = (oldest_analysed - self.background).max(self.first);
        (oldest..=self.number)
            .filter(|f| {
                f.rem_euclid(self.s.analysis_step.every) == 0
                    || *f == self.number
                    || self.to_analyse.contains(f)
            })
            .collect()
    }
}

/// Below these coverages, CPU drawing beats the GPU's readback cost. Coverage is window pixels
/// over canvas pixels, counted per window with shadow, overlaps included. See BENCHMARK.md.
#[cfg(not(target_arch = "wasm32"))]
const GPU_MIN_COVERAGE_8_BIT: f64 = 0.5;
#[cfg(not(target_arch = "wasm32"))]
const GPU_MIN_COVERAGE_DEEPER: f64 = 1.5;
// The browser CPU path runs on one thread, so the GPU draws every swarm there.
#[cfg(target_arch = "wasm32")]
const GPU_MIN_COVERAGE_8_BIT: f64 = 0.0;
#[cfg(target_arch = "wasm32")]
const GPU_MIN_COVERAGE_DEEPER: f64 = 0.0;

/// What a frame without a cached analysis draws from.
static NO_SEGMENTS: Segmentation = Segmentation {
    blobs: Vec::new(),
    mask: Vec::new(),
    width: 0,
    height: 0,
};

impl IncrementalRenderer {
    /// Draws Swarm on the GPU where there is one, or always on the CPU. Off by default, so the
    /// library's renders are the CPU's; the plugin and the CLI turn it on.
    pub fn use_gpu(&mut self, on: bool) {
        self.gpu_wanted = on;
        if !on {
            self.set_drawer(None);
        }
    }

    /// Whether Swarm is to draw on the GPU where there is one (see [`Self::use_gpu`]).
    pub fn gpu_wanted(&self) -> bool {
        self.gpu_wanted
    }

    /// The drawer, once a render has drawn on the GPU.
    pub fn drawer(&self) -> Option<&Drawer> {
        self.drawer.as_deref()
    }

    /// Minimum window coverage for drawing on the GPU; 0 always draws there. Defaults depend on
    /// canvas depth.
    pub fn set_gpu_min_coverage(&mut self, min_coverage: f64) {
        self.gpu_min_coverage = Some(min_coverage);
    }

    /// Whether the last render is still waiting on the GPU (see
    /// [`vfx_core::CpuEffect::unfinished`]).
    pub fn unfinished(&self) -> bool {
        self.unfinished
    }

    /// Whether the GPU drew the last render's output.
    pub fn drew_on_gpu(&self) -> bool {
        self.drew_on_gpu == Some(true)
    }

    /// Draws with `drawer` from now on.
    #[cfg(test)]
    pub(crate) fn use_drawer(&mut self, drawer: Drawer) {
        self.gpu_wanted = true;
        self.set_drawer(Some(Box::new(drawer)));
    }

    fn set_drawer(&mut self, drawer: Option<Box<Drawer>>) {
        self.drawer = drawer;
        self.atlas.frames.clear();
    }

    /// Draws Overlay on the GPU too. Off by default: while frames go through host memory, the
    /// upload and readback cost more than the CPU's whole Overlay.
    pub fn gpu_overlay(&mut self, on: bool) {
        self.gpu_overlay = on;
    }

    /// Offers the next render `target` to draw into instead of its output, which a swarm drawn
    /// on the GPU of the same device takes; [`IncrementalRenderer::took_target`] says whether it
    /// did.
    pub fn offer_target(&mut self, target: vfx_core::Target<'_>) {
        let f = target.frame;
        self.target = Some(TargetOffer {
            device: target.gpu.device.clone(),
            frame: vfx_core::Frame {
                texture: f.texture.clone(),
                view: f.view.clone(),
                width: f.width,
                height: f.height,
                format: f.format,
            },
            drawn_into: false,
        });
    }

    /// Whether the last render drew into the target offered for it. The offer ends.
    pub fn took_target(&mut self) -> bool {
        self.target.take().is_some_and(|offer| offer.drawn_into)
    }

    /// Whether this render can draw a swarm of `(w, h)` at `C` into the offered target: one
    /// on `drawer`'s device, of the canvas's size, for an 8-bit canvas.
    fn can_draw_into_target<C: Channel>(&self, drawer: &Drawer, (w, h): (i32, i32)) -> bool {
        let fits = self.target.as_ref().is_some_and(|offer| {
            (offer.frame.width, offer.frame.height) == (w as u32, h as u32)
                && drawer.draws_on(&offer.device)
        });
        size_of::<C>() == 1 && fits
    }

    /// Renders the last of `count` frames, frame `number`, into `out`. Reads each frame the
    /// render depends on once, oldest first (from a fixed source, only uncached ones), and
    /// recomputes only what changed.
    pub fn render_frame<C: Depth>(
        &mut self,
        s: &RenderSettings,
        source: &mut dyn FrameSource,
        count: usize,
        number: i64,
        out: &mut ViewMut<C>,
    ) {
        self.unfinished = false;
        self.drew_on_gpu = Some(false);
        self.source_is_live = source.live();
        if out.is_empty() {
            return;
        }
        // Live renders can't block on model load, so they render without faces until ready.
        // Non-live renders may be exported and must block.
        let without_faces = RenderSettings {
            faces: FaceParts::default(),
            ..*s
        };
        let s = if source.live() && s.faces.any() && !self.face_models_ready() {
            &without_faces
        } else {
            s
        };
        if count == 0 {
            let all = PixelRect::new(0, 0, out.width, out.height);
            fill_rect(out, all, Argb::default());
            return;
        }
        self.open_drawer_if_wanted(s);
        let (w, h) = (out.width, out.height);
        let history_format = HistoryFormat {
            width: w,
            height: h,
            channel_bytes: size_of::<C>(),
            grid_width: s.grid_width,
        };
        if self.history_format != Some(history_format) {
            self.clear_history();
            self.history_format = Some(history_format);
        }

        let plan = RenderPlan::new(s, source, count, number, (w, h));
        let current_hash = self.check_for_restart::<C>(&plan, source);
        // Frames older than this can't be read from this source again.
        let held = i64::try_from(source.held()).unwrap_or(i64::MAX);
        let oldest_held_frame = if plan.fixed {
            number.saturating_sub(held) + 1
        } else {
            i64::MIN
        };
        // A hash vouches for what a frame's pixels gave. Those of a frame out of the source's reach
        // get released, so a source that hands it over again later has it worked out again.
        if plan.fixed && held != i64::MAX {
            for (_, kept) in self.kept.range_mut(..oldest_held_frame) {
                kept.read = None;
            }
        }
        let swarm = s.mode == OutputMode::Swarm;
        if swarm {
            self.pin_unreadable_frames(s, plan.first, number, oldest_held_frame);
        }
        let frames_to_read = if plan.fixed {
            self.frames_to_read(s, plan.first, number, oldest_held_frame)
        } else {
            plan.frames_feeding_analysis()
        };
        if !plan.live || !s.faces.any() {
            self.live_faces.forget();
        }

        // Frames the host didn't hand over count as empty, as in a cold render, but nothing
        // computed from one is cached, so it is read again next time.
        let mut missing = BTreeSet::new();
        for frame in frames_to_read {
            source.with_frame((frame - plan.first) as usize, &mut |image| {
                let view = image.and_then(C::view);
                self.note_pixels(&plan, frame, view.as_ref(), current_hash, &mut missing);
                self.analyse_frame(&plan, frame, view.as_ref(), &missing);
                if frame == number {
                    self.draw(s, count, number, view, out);
                }
            });
        }
        for &frame in &missing {
            self.invalidate_frame(frame, plan.invalidation_reach);
        }
        let retention_radius = plan.background + plan.trail_len + 2 * AnalysisStep::MAX_EVERY;
        // Later frames are read when they come up. Keeping them from a seek would only compare
        // their hashes then with pixels that needn't match: Resolve hands over different pixels for
        // a frame in playback than for the same frame otherwise, which would clear the history.
        self.retain_frames(number - retention_radius..=number);
        if swarm {
            self.release_undrawn_data(s, count, number, (w, h), oldest_held_frame);
        }
    }

    /// While the device opens, renders draw on the CPU.
    fn open_drawer_if_wanted(&mut self, s: &RenderSettings) {
        let mode_uses_gpu =
            s.mode == OutputMode::Swarm || s.mode == OutputMode::Overlay && self.gpu_overlay;
        if self.gpu_wanted && self.drawer.is_none() && mode_uses_gpu && !gpu::opening() {
            let drawer = gpu::device().and_then(Drawer::new).map(Box::new);
            self.gpu_wanted = drawer.is_some();
            self.set_drawer(drawer);
        }
    }

    /// Clears the history if the source started over, and returns the current frame's hash if that
    /// took one. A fixed source's earlier frames never change, so only the current frame is hashed:
    /// a different hash means the source restarted. A live source numbers each frame once, so its
    /// frames aren't hashed at all, and a lower number means it restarted.
    fn check_for_restart<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        source: &mut dyn FrameSource,
    ) -> Option<u64> {
        let mut current_hash = None;
        if plan.numbers_identify_frames {
            let newest_read = self.kept.iter().rev().find(|(_, k)| k.read.is_some());
            if newest_read.is_some_and(|(&newest, _)| plan.number < newest) {
                self.clear_history();
            }
        } else if plan.fixed {
            source.with_frame((plan.number - plan.first) as usize, &mut |image| {
                current_hash = image.and_then(C::view).map(|v| frame_hash(&v));
            });
            let read = self.read_before(plan.number);
            if current_hash.is_none_or(|hash| read.is_some_and(|r| r != PixelsRead::Hash(hash))) {
                self.clear_history();
            }
        }
        current_hash
    }

    /// Records that `frame` was read, with its grid, and drops what a change to it invalidates.
    fn note_pixels<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        frame: i64,
        view: Option<&View<C>>,
        current_hash: Option<u64>,
        missing: &mut BTreeSet<i64>,
    ) {
        let Some(v) = view else {
            self.invalidate_frame(frame, plan.invalidation_reach);
            missing.insert(frame);
            return;
        };
        let before = self.read_before(frame);
        if plan.fixed && before.is_some() {
            return;
        }
        let read = if plan.numbers_identify_frames {
            PixelsRead::NotHashed
        } else if let Some(hash) = current_hash.filter(|_| frame == plan.number) {
            PixelsRead::Hash(hash)
        } else if before.is_some() {
            // Checking kept frames reads every one, so hash in parallel.
            PixelsRead::Hash(frame_hash_parallel(v))
        } else {
            PixelsRead::Hash(frame_hash(v))
        };
        if before != Some(read) {
            // A fixed source's frame read for the first time can't have changed what later frames
            // computed from it; in other sources an uncached frame may have.
            if !plan.fixed {
                self.invalidate_frame(frame, plan.invalidation_reach);
            }
            let kept = self.kept.entry(frame).or_default();
            kept.grid = Some(downsample(v, plan.s.grid_width));
            kept.read = Some(read);
        }
    }

    /// Works out `frame`'s analysis and tiles, if this render wants them and they aren't kept.
    fn analyse_frame<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        frame: i64,
        view: Option<&View<C>>,
        missing: &BTreeSet<i64>,
    ) {
        let s = plan.s;
        let swarm = s.mode == OutputMode::Swarm;
        let key = analysis_key(s, frame, plan.first);
        // Outside the swarm a missing analysis frame isn't analysed: the frame displaying it
        // analyses itself instead.
        let wanted = plan.to_analyse.contains(&frame) && (swarm || view.is_some())
            || !swarm
                && frame == plan.number
                && self.displayed_analysis_frame(s, plan.number, plan.first) == plan.number;
        if !wanted || self.is_cached(frame, &key, s) {
            return;
        }
        let motion = match self.cached_motion(frame, &key) {
            Some(motion) => motion.clone(),
            None => {
                let segments = self.find_segments(plan, frame, &key, view, missing);
                let motion = segments.clone();
                self.kept.entry(frame).or_default().analysis = Some(KeptAnalysis { key, segments });
                motion
            }
        };
        if swarm {
            self.cut_tiles(plan, frame, key, &motion, view);
        }
    }

    /// `frame`'s segments under `s`: what moved or passed the threshold, then its faces.
    fn find_segments<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        frame: i64,
        key: &AnalysisKey,
        view: Option<&View<C>>,
        missing: &BTreeSet<i64>,
    ) -> Segmentation {
        let s = plan.s;
        let empty = LumaGrid::default();
        let grid = self.grid(frame).unwrap_or(&empty);
        let mut motion = match s.detection_mode {
            DetectionMode::Motion => {
                let history = (key.background_start..frame)
                    .filter(|k| k.rem_euclid(s.analysis_step.every) == 0)
                    .map(|k| self.grid(k).unwrap_or(&empty));
                motion::detect_motion(history, grid, &s.motion)
            }
            DetectionMode::Threshold => motion::detect_threshold(grid, &s.threshold, &s.motion),
            DetectionMode::FacesOnly => Segmentation {
                mask: vec![0; grid.count()],
                width: grid.width,
                height: grid.height,
                ..Segmentation::default()
            },
        };
        let cells = grid.count();
        // The face source is analysed first, unless it is missing or a live source left it out,
        // which leaves it no grid and no faces.
        let source_frame = face_source_frame(s, frame, plan.first);
        let source_key = analysis_key(s, source_frame, plan.first);
        let source_motion = key
            .face_source
            .filter(|_| !missing.contains(&source_frame) && self.grid(source_frame).is_some())
            .and_then(|_| self.cached_motion(source_frame, &source_key));
        if let Some(before) = source_motion {
            motion.blobs.extend(reused_faces(before, source_frame));
        } else if let Some(view) = view.filter(|_| s.faces.any()) {
            #[cfg(test)]
            {
                self.face_detection_calls += 1;
            }
            if plan.live {
                // A live render takes the newest faces there are, and those already shown stay
                // until newer ones arrive.
                let first_face = motion.blobs.len();
                self.live_faces
                    .exchange(view, s.faces, cells, frame, first_face);
                motion.blobs.extend(self.live_faces.faces_on(frame));
            } else {
                // Any other render may be exported, so it has this frame's own faces, however long
                // they take.
                motion
                    .blobs
                    .extend(crate::faces::find(view, s.faces, cells));
            }
        }
        motion
    }

    /// Cuts `frame`'s windows out of its pixels, and frees its old place in the atlas.
    fn cut_tiles<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        frame: i64,
        key: AnalysisKey,
        motion: &Segmentation,
        view: Option<&View<C>>,
    ) {
        let s = plan.s;
        // Reused faces need tiles here if a trail no longer than the step draws no earlier
        // analysis, or if the windows of held faces would leave it before the next analysis, as a
        // slow detection's do on a short trail.
        let next_analysis = frame + s.analysis_step.every;
        let held_windows_leaving = self
            .live_faces
            .windows_leave_before(next_analysis, plan.trail_len);
        let include_reused_faces = plan.trail_len <= s.analysis_step.every || held_windows_leaving;
        let tiles = cut_frame_tiles(
            motion,
            view,
            self.live_faces.found_in::<C>().as_ref(),
            frame,
            plan.canvas,
            s.mirror,
            include_reused_faces,
        );
        self.kept.entry(frame).or_default().tiles = Some(KeptTiles {
            key,
            mirror: s.mirror,
            tiles,
        });
        self.atlas.frames.remove(&frame);
        if held_windows_leaving {
            self.live_faces.windows_cut_on(frame);
        }
    }

    /// Frames before `oldest_held_frame` can't be re-cut. Re-key their cached results to the
    /// current settings so they stay valid; frames with no result stay empty.
    fn pin_unreadable_frames(
        &mut self,
        s: &RenderSettings,
        first: i64,
        number: i64,
        oldest_held_frame: i64,
    ) {
        for frame in frames_to_analyse(s, number, first)
            .into_iter()
            .filter(|&f| f < oldest_held_frame)
        {
            let key = analysis_key(s, frame, first);
            let kept = self.kept.entry(frame).or_default();
            kept.analysis
                .get_or_insert_with(|| KeptAnalysis {
                    key,
                    segments: Segmentation::default(),
                })
                .key = key;
            let tiles = kept.tiles.get_or_insert_with(|| KeptTiles {
                key,
                mirror: s.mirror,
                tiles: Vec::new(),
            });
            (tiles.key, tiles.mirror) = (key, s.mirror);
        }
    }

    /// Releases data for unreadable frames that no later render reads. Invariant: a tile hidden
    /// by a newer one or evicted never becomes visible again, because the newer entries stay
    /// newer. Only the displayed frame's mask is read.
    fn release_undrawn_data(
        &mut self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        (width, height): (i32, i32),
        oldest_held_frame: i64,
    ) {
        let mut older = self.kept.range(..oldest_held_frame);
        if !older.any(|(_, kept)| kept.tiles.is_some()) {
            return;
        }
        let visible: BTreeSet<(i64, usize)> =
            build_pile(&self.kept, s, count, number, width, height)
                .visible_entries()
                .map(|p| (p.frame, p.index))
                .collect();
        for (&frame, kept) in self.kept.range_mut(..oldest_held_frame) {
            let tiles = kept.tiles.iter_mut().flat_map(|cut| cut.tiles.iter_mut());
            for (index, tile) in tiles.enumerate() {
                if !visible.contains(&(frame, index)) {
                    tile.hidden = true;
                    tile.pixels = Vec::new();
                    tile.title_bar = None;
                }
            }
        }
        let oldest_displayed_analysis = number - AnalysisStep::MAX_EVERY;
        for (_, kept) in self
            .kept
            .range_mut(..oldest_held_frame.min(oldest_displayed_analysis))
        {
            if let Some(analysis) = &mut kept.analysis {
                analysis.segments.mask = Vec::new();
            }
        }
    }

    /// Whether a live render can find faces.
    fn face_models_ready(&self) -> bool {
        #[cfg(test)]
        if self.simulate_models_loading {
            return false;
        }
        crate::faces::ready()
    }

    /// The frames from `first` a render of `number` must read: those without cached motion or
    /// tiles, the analyses before them without grids, and `number`, which it draws and, outside
    /// the swarm, may analyse. None before `oldest_held_frame`.
    fn frames_to_read(
        &self,
        s: &RenderSettings,
        first: i64,
        number: i64,
        oldest_held_frame: i64,
    ) -> BTreeSet<i64> {
        let mut wanted = BTreeSet::from([number]);
        let grids = |frame: i64| {
            (analysis_key(s, frame, first)
                .background_start
                .max(oldest_held_frame)..frame)
                .filter(|&k| k.rem_euclid(s.analysis_step.every) == 0 && self.grid(k).is_none())
        };
        for frame in frames_to_analyse(s, number, first)
            .into_iter()
            .filter(|&f| f >= oldest_held_frame)
        {
            if !self.is_cached(frame, &analysis_key(s, frame, first), s) {
                wanted.insert(frame);
                wanted.extend(grids(frame));
            }
        }
        if s.mode != OutputMode::Swarm {
            wanted.extend(grids(number));
        }
        wanted
    }

    /// Draws frame `number`, the last of `count`, from the cache. `frame` is its pixels, which
    /// every output but the swarm draws over. A render the GPU fails is drawn on the CPU from the
    /// same cached state.
    pub fn draw<C: Channel>(
        &mut self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) {
        self.drew_on_gpu = Some(false);
        self.update_face_readings(s, number, number - count as i64 + 1);
        if let Err(fault) = self.try_draw(s, count, number, frame, out) {
            self.fall_back(&fault);
            self.draw_cpu(s, count, number, frame, out);
        }
    }

    fn update_face_readings(&mut self, s: &RenderSettings, number: i64, first: i64) {
        self.faces_wanted = s.faces.any();
        self.readings = if s.faces.face {
            let (_, motion) = self.displayed_analysis(s, number, first);
            let models_loaded = crate::faces::runs_on().is_some();
            crate::faces::readings(models_loaded, &motion.blobs)
        } else {
            Vec::new()
        };
    }

    pub fn report(&self) -> Report {
        let mut report = Report::default();
        report.draw = self.drew_on_gpu.map(|on_gpu| match &self.drawer {
            // Browsers don't name their adapter.
            Some(drawer) if on_gpu && drawer.adapter().name.is_empty() => "GPU".to_owned(),
            Some(drawer) if on_gpu => format!("GPU ({})", drawer.adapter().name),
            _ => "CPU".to_owned(),
        });
        report.faces = crate::faces::runs_on()
            .filter(|_| self.faces_wanted)
            .map(str::to_owned);
        report.readings = self.readings.clone();
        report.gpu_fault = self.last_reported_fault.clone();
        report.out_of_gpu_memory = self.atlas.oldest_frame.is_some();
        report
    }

    fn try_draw<C: Channel>(
        &mut self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        let Some(mut drawer) = self.drawer.take_if(|drawer| !drawer.in_fault_cooldown()) else {
            self.draw_cpu(s, count, number, frame, out);
            return Ok(());
        };
        drawer.begin_render();
        let drawn = self.draw_gpu(&mut drawer, s, count, number, frame, out);
        self.unfinished = drawn.is_ok() && drawer.unfinished;
        self.drawer = Some(drawer);
        drawn
    }

    /// Device loss disables the GPU for this instance. Any other fault retries on the next
    /// render, unless [`gpu::MAX_RECENT_FAULTS`] is reached (cooldown).
    fn fall_back(&mut self, fault: &Fault) {
        if self.last_reported_fault.as_ref() != Some(&fault.0) {
            eprintln!(
                "segcam Motion: the GPU failed ({}); drawing on the CPU",
                fault.0
            );
            self.last_reported_fault = Some(fault.0.clone());
        }
        match &mut self.drawer {
            Some(drawer) if drawer.lost() => {
                self.gpu_wanted = false;
                self.set_drawer(None);
            }
            Some(drawer) => drawer.record_fault(),
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
        let pile = build_pile(&self.kept, s, count, number, out.width, out.height);
        let (analysis_frame, motion) = self.displayed_analysis(s, number, first);
        draw_output(out, s, frame, motion, analysis_frame, pile.visible());
    }

    /// The frame whose analysis frame `number` displays: the last analysed, or `number` itself
    /// when that analysis is missing.
    fn displayed_analysis_frame(&self, s: &RenderSettings, number: i64, first: i64) -> i64 {
        let analysis = s.analysis_step.analysis_frame(number, first);
        match self.cached_motion(analysis, &analysis_key(s, analysis, first)) {
            Some(_) => analysis,
            None => number,
        }
    }

    /// [`Self::displayed_analysis_frame`] and that frame's analysis; empty if it isn't cached.
    fn displayed_analysis(
        &self,
        s: &RenderSettings,
        number: i64,
        first: i64,
    ) -> (i64, &Segmentation) {
        let frame = self.displayed_analysis_frame(s, number, first);
        let motion = self
            .cached_motion(frame, &analysis_key(s, frame, first))
            .unwrap_or(&NO_SEGMENTS);
        (frame, motion)
    }

    fn draw_gpu<C: Channel>(
        &mut self,
        drawer: &mut Drawer,
        s: &RenderSettings,
        count: usize,
        number: i64,
        frame: Option<View<C>>,
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        let first = number - count as i64 + 1;
        if let (OutputMode::Overlay, Some(view), true) = (s.mode, frame.as_ref(), self.gpu_overlay)
        {
            let (analysis_frame, motion) = self.displayed_analysis(s, number, first);
            drawer.draw_overlay(out, view, &motion.blobs, number, analysis_frame, s)?;
            self.drew_on_gpu = Some(true);
            return Ok(());
        }
        if s.mode != OutputMode::Swarm {
            self.draw_cpu(s, count, number, frame, out);
            return Ok(());
        }
        // Choose by this render's windows alone, so cached and cold renders match. The thresholds
        // reflect readback cost, so drawing directly into a host frame always uses the GPU.
        let (w, h) = (out.width, out.height);
        let canvas_pixels = i64::from(w) * i64::from(h);
        let default = if size_of::<C>() == 1 {
            GPU_MIN_COVERAGE_8_BIT
        } else {
            GPU_MIN_COVERAGE_DEEPER
        };
        let into_target = self.can_draw_into_target::<C>(drawer, (w, h));
        let min_coverage_permille = match into_target {
            true => 0,
            false => (self.gpu_min_coverage.unwrap_or(default) * 1000.0).round() as i64,
        };
        let m = SwarmMetrics::new(h);
        let shadow = swarm::shadow_kernel(m.scale);
        let pile = build_pile(&self.kept, s, count, number, w, h);
        let mut visible: Vec<PileEntry> = pile.visible_entries().collect();
        self.atlas
            .leave_out_dropped(&mut visible, self.source_is_live);
        let window_pixel_count = window_pixels(visible.iter().map(|p| p.tile), &m, &shadow, w, h);
        let enough_windows = window_pixel_count * 1000 >= canvas_pixels * min_coverage_permille;
        let live = self.source_is_live;
        if !enough_windows
            || !self
                .atlas
                .place(&self.kept, &visible, drawer, (w, h), &shadow, live)?
        {
            let motion = cached_analysis(&self.kept, number, &analysis_key(s, number, first))
                .unwrap_or(&NO_SEGMENTS);
            let windows = visible.iter().map(|p| p.tile);
            draw_output(out, s, frame, motion, number, windows);
            return Ok(());
        }
        let windows = visible.iter().filter_map(|p| {
            let allocation = self.atlas.frames.get(&p.frame)?;
            Some((p.tile, allocation.slots[p.index]))
        });
        if into_target && let Some(offer) = self.target.as_mut() {
            drawer.draw_swarm_into(&offer.frame, windows, frame.as_ref(), number, s, &shadow)?;
            offer.drawn_into = true;
            self.drew_on_gpu = Some(true);
            return Ok(());
        }
        drawer.draw_swarm(out, windows, frame.as_ref(), number, s, &shadow)?;
        self.drew_on_gpu = Some(true);
        Ok(())
    }

    /// Clears every frame. Dropping the allocations frees their atlas space; the drawer keeps its
    /// buffers, its atlas at the size it grew to, and its fault and refusal history.
    pub fn clear_history(&mut self) {
        self.kept.clear();
        self.atlas = AtlasPlacements::default();
        self.live_faces.forget();
        self.unfinished = false;
        if let Some(drawer) = &mut self.drawer {
            drawer.drop_unfinished_draw();
        }
    }

    /// Whether `frame` has cached motion and, for the swarm, tiles.
    fn is_cached(&self, frame: i64, key: &AnalysisKey, s: &RenderSettings) -> bool {
        self.cached_motion(frame, key).is_some()
            && (s.mode != OutputMode::Swarm
                || cached_tiles(&self.kept, frame, key, s.mirror).is_some())
    }

    fn cached_motion(&self, frame: i64, key: &AnalysisKey) -> Option<&Segmentation> {
        cached_analysis(&self.kept, frame, key)
    }

    fn grid(&self, frame: i64) -> Option<&LumaGrid> {
        self.kept.get(&frame)?.grid.as_ref()
    }

    fn read_before(&self, frame: i64) -> Option<PixelsRead> {
        self.kept.get(&frame)?.read
    }

    /// Drops `frame`'s cached data, and the motion and tiles of the `dependents` frames after it.
    fn invalidate_frame(&mut self, frame: i64, dependents: i64) {
        if let Some(kept) = self.kept.get_mut(&frame) {
            kept.grid = None;
            kept.read = None;
        }
        for later in frame..=frame + dependents {
            if let Some(kept) = self.kept.get_mut(&later) {
                kept.analysis = None;
                kept.tiles = None;
            }
            self.atlas.frames.remove(&later);
        }
    }

    fn retain_frames(&mut self, frames: RangeInclusive<i64>) {
        self.kept.retain(|f, _| frames.contains(f));
        self.atlas.frames.retain(|f, _| frames.contains(f));
    }
}

/// Where kept frames' tiles sit in the atlas. It stays apart from [`KeptFrame`] because a draw
/// changes it while the windows it draws borrow their tiles.
#[derive(Default)]
struct AtlasPlacements {
    frames: BTreeMap<i64, AtlasAllocation>,
    /// A live render out of atlas stops drawing frames older than this.
    oldest_frame: Option<i64>,
}

impl AtlasPlacements {
    /// Takes out the windows a live render gave up for space, until the trail has moved past them.
    /// Only a live render goes without windows, and a live stream can end with some left out.
    fn leave_out_dropped(&mut self, visible: &mut Vec<PileEntry>, live: bool) {
        if !live {
            self.oldest_frame = None;
        }
        if let Some(oldest) = self.oldest_frame {
            let dropping = visible.iter().any(|p| p.frame < oldest);
            visible.retain(|p| p.frame >= oldest);
            if !dropping {
                self.oldest_frame = None;
            }
        }
    }

    /// Gives every frame with a `visible` window a place, newest first, and only those keep one.
    /// `false` when a render that may be exported doesn't fit: it has to be exact, so it draws on
    /// the CPU and frames leaving the trail free space for the next. A CPU draw would stall a
    /// `live` render, so that one gives up its oldest windows instead, or waits a render for a
    /// growing atlas.
    fn place(
        &mut self,
        kept: &BTreeMap<i64, KeptFrame>,
        visible: &[PileEntry],
        drawer: &mut Drawer,
        canvas: (i32, i32),
        shadow: &swarm::ShadowKernel,
        live: bool,
    ) -> Result<bool, Fault> {
        let wanted: BTreeSet<i64> = visible.iter().map(|p| p.frame).collect();
        self.frames.retain(|f, _| wanted.contains(f));
        for &f in wanted.iter().rev() {
            if self.frames.contains_key(&f) || self.oldest_frame.is_some_and(|oldest| f < oldest) {
                continue;
            }
            let Some(cut) = kept.get(&f).and_then(|k| k.tiles.as_ref()) else {
                continue;
            };
            loop {
                if let Some(allocation) = drawer.upload_tiles(&cut.tiles, canvas, shadow)? {
                    self.frames.insert(f, allocation);
                    break;
                }
                if !live {
                    return Ok(false);
                }
                if drawer.atlas_growing() {
                    break;
                }
                let evicted = self.frames.keys().next().copied();
                let Some(oldest) = evicted.filter(|&oldest| oldest < f) else {
                    self.oldest_frame = Some(f + 1);
                    break;
                };
                self.frames.remove(&oldest);
                self.oldest_frame = Some(oldest + 1);
            }
        }
        Ok(true)
    }
}

/// The analysis of `frame`, if computed for `key`.
fn cached_analysis<'a>(
    kept: &'a BTreeMap<i64, KeptFrame>,
    frame: i64,
    key: &AnalysisKey,
) -> Option<&'a Segmentation> {
    let analysis = kept.get(&frame)?.analysis.as_ref()?;
    (analysis.key == *key).then_some(&analysis.segments)
}

/// The tiles of `frame`, if cut for `key` and `mirror`.
fn cached_tiles<'a>(
    kept: &'a BTreeMap<i64, KeptFrame>,
    frame: i64,
    key: &AnalysisKey,
    mirror: bool,
) -> Option<&'a [Tile]> {
    let cut = kept.get(&frame)?.tiles.as_ref()?;
    (cut.key == *key && cut.mirror == mirror).then_some(cut.tiles.as_slice())
}

/// The pile of the `kept` tiles a swarm render of frame `number`, the last of `count`, draws
/// from on a `width` x `height` canvas; empty for other outputs.
fn build_pile<'a>(
    kept: &'a BTreeMap<i64, KeptFrame>,
    s: &RenderSettings,
    count: usize,
    number: i64,
    width: i32,
    height: i32,
) -> Pile<'a> {
    let first = number - count as i64 + 1;
    let mut pile = Pile::new(width, height, s.max_tiles);
    if s.mode == OutputMode::Swarm {
        for frame in analysis_frames_used(s, number, first) {
            let tiles = cached_tiles(kept, frame, &analysis_key(s, frame, first), s.mirror);
            for (index, tile) in tiles.into_iter().flatten().enumerate() {
                pile.push(PileEntry { tile, frame, index });
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
    let canvas = PixelRect::new(0, 0, width, height);
    tiles
        .map(|t| {
            let window = swarm::window_rect(t.content_rect, m.title_height);
            let r = crate::render::intersect(shadow.shadow_rect(window), canvas);
            i64::from(r.w) * i64::from(r.h)
        })
        .sum()
}

/// The cache key of `frame` in a render whose frames start at `first`.
fn analysis_key(s: &RenderSettings, frame: i64, first: i64) -> AnalysisKey {
    // The background's analyses end at the last one before the frame, on it or not.
    let every = s.analysis_step.every;
    let background_start = |frame: i64| {
        let last = frame - 1 - (frame - 1).rem_euclid(every);
        let oldest = last + every - i64::from(history_frames(s));
        oldest.min(frame).max(first)
    };
    AnalysisKey {
        detection: DetectionSettings {
            detection_mode: s.detection_mode,
            motion: s.motion,
            background_frames: s.background_frames,
            every: s.analysis_step.every,
            threshold: s.threshold,
            faces: s.faces,
        },
        background_start: background_start(frame),
        face_source: Some(face_source_frame(s, frame, first))
            .filter(|&source| source != frame)
            .map(|source| (source, background_start(source))),
    }
}

/// The frames whose motion a render of `number` uses: the analysed frames of the swarm's
/// trail, or the analysis the frame shows.
fn analysis_frames_used(s: &RenderSettings, number: i64, first: i64) -> impl Iterator<Item = i64> {
    // The trail's oldest frame is never analysed early, or its windows would come and go with it.
    let (frames, every) = if s.mode == OutputMode::Swarm {
        (
            (number - trail_frame_count(s) + 1).max(first)..=number,
            s.analysis_step.every,
        )
    } else {
        let visible = s.analysis_step.analysis_frame(number, first);
        (visible..=visible, 1)
    };
    frames.filter(move |f| f.rem_euclid(every) == 0)
}

/// Frames drawn from, current included: the trail for the swarm, otherwise 1.
fn trail_frame_count(s: &RenderSettings) -> i64 {
    if s.mode == OutputMode::Swarm {
        i64::from(s.trail_frames.max(1))
    } else {
        1
    }
}

/// The frame whose faces `frame` shows in a render whose frames start at `first`. As in
/// segcam, faces run on every 2nd analysis; the analysis between reuses the one before's.
fn face_source_frame(s: &RenderSettings, frame: i64, first: i64) -> i64 {
    let every = s.analysis_step.every;
    let before = frame - every;
    if s.faces.any() && frame.rem_euclid(2 * every) == every && before >= first {
        before
    } else {
        frame
    }
}

/// The frames whose motion a render of `number` computes: those it uses, and the frames whose
/// faces they reuse.
fn frames_to_analyse(s: &RenderSettings, number: i64, first: i64) -> BTreeSet<i64> {
    analysis_frames_used(s, number, first)
        .flat_map(|f| [face_source_frame(s, f, first), f])
        .collect()
}

/// The faces in `motion`, frame `frame`'s, for the next analysis to reuse with their IDs, which
/// faces it had reused already got from the frame that first found them.
fn reused_faces(motion: &Segmentation, frame: i64) -> impl Iterator<Item = motion::Blob> + '_ {
    use motion::SegmentKind as Kind;
    motion
        .blobs
        .iter()
        .enumerate()
        .filter(|(_, b)| !matches!(b.kind, Kind::Motion | Kind::ThresholdRegion))
        .map(move |(i, b)| motion::Blob {
            reused_from: b.reused_from.or(Some((frame, i))),
            ..*b
        })
}

const HASH_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

fn fold_hash(hashes: impl Iterator<Item = u64>) -> u64 {
    hashes.fold(0, |h, l| {
        (h ^ l).wrapping_mul(HASH_MULTIPLIER).rotate_left(31)
    })
}

/// A hash of a row, read as u64 words in four lanes so it's limited by memory bandwidth.
fn row_hash<C: Channel>(row: &[Argb<C>]) -> u64 {
    let mut lanes = [0u64; 4];
    let bytes: &[u8] = bytemuck::cast_slice(row);
    let (words, rest) = bytes.as_chunks::<8>();
    for chunk in words.chunks(4) {
        for (lane, word) in lanes.iter_mut().zip(chunk) {
            *lane = (*lane ^ u64::from_le_bytes(*word))
                .wrapping_mul(HASH_MULTIPLIER)
                .rotate_left(29);
        }
    }
    for &b in rest {
        lanes[0] = (lanes[0] ^ u64::from(b)).wrapping_mul(HASH_MULTIPLIER);
    }
    fold_hash(lanes.into_iter())
}

/// A hash of every pixel, from the rows' hashes in order, so any thread count gives the same one.
fn frame_hash<C: Channel>(view: &View<C>) -> u64 {
    fold_hash((0..view.height).map(|y| row_hash(view.row(y))))
}

/// [`frame_hash`] on the host worker threads.
fn frame_hash_parallel<C: Channel>(view: &View<C>) -> u64 {
    let mut rows = vec![0u64; view.height as usize];
    by_rows(&mut rows, 1, |y, hash| {
        hash[0] = row_hash(view.row(y as i32));
    });
    fold_hash(rows.into_iter())
}

#[cfg(test)]
mod tests {
    use super::gpu::gpu_test_lock;
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

    fn swarm_settings() -> RenderSettings {
        RenderSettings {
            mode: OutputMode::Swarm,
            background_frames: 4,
            trail_frames: 8,
            max_tiles: 20,
            ..RenderSettings::default()
        }
    }

    fn render(
        state: &mut IncrementalRenderer,
        frames: &[Vec<Argb<u8>>],
        n: usize,
    ) -> Vec<Argb<u8>> {
        let s = swarm_settings();
        let needed = crate::previous_frames_needed(&s) as usize;
        let clip = &frames[n.saturating_sub(needed)..=n];
        let mut out = vec![Argb::default(); (W * H) as usize];
        let view = &mut ViewMut::from_pixels(&mut out, W, H);
        state.render_frame(&s, &mut Frames(clip), clip.len(), n as i64, view);
        out
    }

    /// Playback that has dropped frames: only the current frame can be read.
    struct OnlyNewestFrame<'a> {
        frames: &'a [Vec<Argb<u8>>],
        reads: usize,
    }

    impl FrameSource for OnlyNewestFrame<'_> {
        fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
            assert_eq!(index, self.frames.len() - 1, "an older frame was asked for");
            self.reads += 1;
            f(Some(Image::U8(View::from_pixels(
                &self.frames[index],
                W,
                H,
            ))));
        }

        fn live(&self) -> bool {
            true
        }

        fn fixed(&self) -> bool {
            true
        }

        fn held(&self) -> usize {
            1
        }
    }

    /// A stream that dropped frames reads only what it renders, and the first check after it
    /// gives what a fresh effect gives, with nothing left over from those renders.
    #[test]
    fn render_after_dropped_frames_matches_fresh() {
        let frames = crossing();
        let s = swarm_settings();
        let needed = crate::previous_frames_needed(&s) as usize;
        let mut kept = IncrementalRenderer::default();
        for n in 0..6 {
            render(&mut kept, &frames, n);
        }
        for n in [8, 10, 13] {
            let clip = &frames[n - needed.min(n)..=n];
            let mut source = OnlyNewestFrame {
                frames: clip,
                reads: 0,
            };
            let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            kept.render_frame(&s, &mut source, clip.len(), n as i64, view);
            assert_eq!(source.reads, 1, "frame {n}");
        }
        let fresh = render(&mut IncrementalRenderer::default(), &frames, 13);
        assert!(render(&mut kept, &frames, 13) == fresh);
    }

    /// A frame whose faces would come from the one before looks for itself if that was dropped.
    #[test]
    fn dropped_face_source_leaves_frame_to_detect() {
        wait_for_models();
        let frames = crossing();
        let s = with_faces(ALL_PARTS);
        let needed = crate::previous_frames_needed(&s) as usize;
        let mut state = IncrementalRenderer::default();
        for n in [4, 7, 9] {
            let clip = &frames[n - needed.min(n)..=n];
            let mut source = OnlyNewestFrame {
                frames: clip,
                reads: 0,
            };
            let calls = face_detection_calls_in(&mut state, &mut source, clip.len(), n as i64);
            assert!(calls > 0, "frame {n} didn't look for faces");
        }
    }

    /// Going back leaves nothing kept ahead of the frame rendered to be compared later.
    #[test]
    fn later_frames_are_not_kept() {
        let frames = crossing();
        let mut state = IncrementalRenderer::default();
        render(&mut state, &frames, 10);
        render(&mut state, &frames, 2);
        assert!(state.kept.keys().all(|&frame| frame <= 2));
    }

    const ALL_PARTS: FaceParts = FaceParts {
        face: true,
        eyes: true,
        mouth: true,
        body: true,
    };

    fn with_faces(faces: FaceParts) -> RenderSettings {
        RenderSettings {
            mode: OutputMode::Swarm,
            background_frames: 4,
            trail_frames: 6,
            max_tiles: 20,
            faces,
            ..RenderSettings::default()
        }
    }

    /// Loads the models under the GPU tests' turn, since that opens devices too.
    fn wait_for_models() {
        let _lock = gpu_test_lock();
        crate::faces::wait();
    }

    /// How often a render of frame `n` looked for faces.
    fn face_detection_calls_in(
        state: &mut IncrementalRenderer,
        source: &mut dyn FrameSource,
        count: usize,
        n: i64,
    ) -> u32 {
        let s = with_faces(ALL_PARTS);
        let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
        let view = &mut ViewMut::from_pixels(&mut out, W, H);
        let before = state.face_detection_calls;
        state.render_frame(&s, source, count, n, view);
        state.face_detection_calls - before
    }

    /// While the models load a live render doesn't look for faces, and looks once they are up.
    #[test]
    fn live_render_skips_faces_until_models_load() {
        wait_for_models();
        let frames = crossing();
        let mut state = IncrementalRenderer {
            simulate_models_loading: true,
            ..IncrementalRenderer::default()
        };
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let before = crate::previous_frames_needed(&with_faces(ALL_PARTS)) as usize;
        for (n, frame) in frames.iter().enumerate() {
            state.simulate_models_loading = n < 7;
            history.next(before, before).copy_from_slice(frame);
            let count = history.len();
            let face_detection_calls =
                face_detection_calls_in(&mut state, &mut history, count, n as i64);
            assert_eq!(
                face_detection_calls > 0,
                n == 7 || n > 7 && n % 2 == 0,
                "frame {n}"
            );
        }
    }

    /// A live render returns while detection can't run, here since the models are locked, and an
    /// export would wait for them.
    #[test]
    fn live_render_does_not_wait_for_detection() {
        wait_for_models();
        let models = crate::faces::lock_models();
        let (done, finished) = std::sync::mpsc::channel();
        let renders = std::thread::spawn(move || {
            let frames = crossing();
            let mut state = IncrementalRenderer::default();
            let mut history = vfx_core::cpu::History::<u8>::default();
            history.resize(W, H);
            let before = crate::previous_frames_needed(&with_faces(ALL_PARTS)) as usize;
            for (n, frame) in frames.iter().enumerate() {
                history.next(before, before).copy_from_slice(frame);
                let count = history.len();
                face_detection_calls_in(&mut state, &mut history, count, n as i64);
            }
            done.send(()).unwrap();
        });
        assert!(
            finished.recv_timeout(Duration::from_secs(60)).is_ok(),
            "a live render waited for the models"
        );
        drop(models);
        renders.join().unwrap();
    }

    /// A live source whose numbering restarts drops what detection found in the frames before.
    #[test]
    fn live_restart_drops_found_faces() {
        wait_for_models();
        let frames = crossing();
        let mut state = IncrementalRenderer::default();
        let before = crate::previous_frames_needed(&with_faces(ALL_PARTS)) as usize;
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let started = std::time::Instant::now();
        let mut n = 0;
        while !state.live_faces.holds_faces() {
            assert!(started.elapsed() < Duration::from_secs(120), "no detection");
            history
                .next(before, before)
                .copy_from_slice(&frames[n as usize % frames.len()]);
            let count = history.len();
            face_detection_calls_in(&mut state, &mut history, count, n);
            n += 1;
            std::thread::sleep(Duration::from_millis(5));
        }
        // Two renders in, so the lower number is certain whichever frame the detection landed on.
        for _ in 0..2 {
            history
                .next(before, before)
                .copy_from_slice(&frames[n as usize % frames.len()]);
            let count = history.len();
            face_detection_calls_in(&mut state, &mut history, count, n);
            n += 1;
        }

        let mut restarted = vfx_core::cpu::History::<u8>::default();
        restarted.resize(W, H);
        restarted.next(before, before).copy_from_slice(&frames[0]);
        let count = restarted.len();
        face_detection_calls_in(&mut state, &mut restarted, count, 0);
        assert!(!state.live_faces.holds_faces());
    }

    /// Faces stay on screen however long the next detection takes, past the trail and what the
    /// frame cache keeps, in windows showing the frame they were found in.
    #[test]
    fn held_faces_outlast_the_frame_cache() {
        wait_for_models();
        let models = crate::faces::lock_models();
        let frames = crossing();
        let mut state = IncrementalRenderer::default();
        let face = motion::Blob {
            kind: motion::SegmentKind::Face,
            bounds: motion::NormRect {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            ..motion::Blob::default()
        };
        let found_in = Argb::<u8> {
            alpha: 255,
            red: 255,
            green: 0,
            blue: 0,
        };
        let settings = with_faces(ALL_PARTS);
        let before = crate::previous_frames_needed(&settings) as usize;
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let renders = 200;
        for n in 0..renders {
            history
                .next(before, before)
                .copy_from_slice(&frames[n as usize % frames.len()]);
            let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            let count = history.len();
            state.render_frame(&settings, &mut history, count, n, view);
            // The first render takes in the frame format, which clears everything.
            if n == 0 {
                state.live_faces.hold(ALL_PARTS, vec![face], 0, 0);
                let pixels = vec![found_in; (W * H) as usize];
                state.live_faces.hold_pixels(pixels, W, H);
            } else if n >= i64::from(settings.trail_frames) {
                assert!(out.contains(&found_in), "no window at {n}");
            }
        }
        drop(models);
    }

    /// A frame whose face source was left out asks for faces itself, and shows the held ones
    /// until they arrive.
    #[test]
    fn held_faces_stay_on_frames_whose_source_was_left_out() {
        wait_for_models();
        let models = crate::faces::lock_models();
        let frames = crossing();
        let mut state = IncrementalRenderer::default();
        let face = motion::Blob {
            kind: motion::SegmentKind::Face,
            ..motion::Blob::default()
        };
        let needed = crate::previous_frames_needed(&with_faces(ALL_PARTS)) as usize;
        for n in [2, 5, 7] {
            let clip = &frames[n - needed.min(n)..=n];
            let mut source = OnlyNewestFrame {
                frames: clip,
                reads: 0,
            };
            face_detection_calls_in(&mut state, &mut source, clip.len(), n as i64);
            // The first render takes in the frame format, which clears everything.
            if n == 2 {
                state.live_faces.hold(ALL_PARTS, vec![face], 0, 0);
                continue;
            }
            let motion = &state.kept[&(n as i64)].analysis.as_ref().unwrap().segments;
            let shown = motion
                .blobs
                .iter()
                .find(|b| b.kind == motion::SegmentKind::Face);
            assert_eq!(shown.and_then(|b| b.reused_from), Some((0, 0)), "frame {n}");
        }
        drop(models);
    }

    /// With all parts off and on again, a detection from before doesn't come back.
    #[test]
    fn turning_faces_off_drops_held_faces() {
        let frames = crossing();
        let mut state = IncrementalRenderer::default();
        let settings = with_faces(FaceParts::default());
        let before = crate::previous_frames_needed(&settings) as usize;
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
        for (n, frame) in frames.iter().enumerate().take(2) {
            history.next(before, before).copy_from_slice(frame);
            // Past the first render, which takes in the frame format and clears everything.
            if n == 1 {
                state
                    .live_faces
                    .hold(ALL_PARTS, vec![motion::Blob::default()], 0, 0);
            }
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            let count = history.len();
            state.render_frame(&settings, &mut history, count, n as i64, view);
        }
        assert!(!state.live_faces.holds_faces());
    }

    /// Live renders that leave frames out, then an export from there: nothing of the live stretch
    /// reaches it, neither faces held nor frames analysed.
    #[test]
    fn non_live_render_after_held_faces_matches_fresh() {
        wait_for_models();
        let frames = crossing();
        let s = with_faces(ALL_PARTS);
        let needed = crate::previous_frames_needed(&s) as usize;
        let clip_to = |n: usize| &frames[n - needed.min(n)..=n];
        let mut state = IncrementalRenderer::default();
        for n in [6, 8, 10] {
            let clip = clip_to(n);
            let mut source = OnlyNewestFrame {
                frames: clip,
                reads: 0,
            };
            face_detection_calls_in(&mut state, &mut source, clip.len(), n as i64);
            if n == 6 {
                let face = motion::Blob {
                    kind: motion::SegmentKind::Face,
                    bounds: motion::NormRect {
                        x: 0.25,
                        y: 0.25,
                        w: 0.5,
                        h: 0.5,
                    },
                    ..motion::Blob::default()
                };
                state.live_faces.hold(ALL_PARTS, vec![face], 6, 0);
            }
        }
        for n in [11, 13] {
            let clip = clip_to(n);
            let render = |state: &mut IncrementalRenderer| {
                let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
                let view = &mut ViewMut::from_pixels(&mut out, W, H);
                state.render_frame(&s, &mut Frames(clip), clip.len(), n as i64, view);
                out
            };
            let exported = render(&mut state);
            assert!(!state.live_faces.holds_faces());
            assert!(
                exported == render(&mut IncrementalRenderer::default()),
                "frame {n}"
            );
        }
    }

    /// New face windows show the frame the face was found in, and motion windows the current one.
    #[test]
    fn new_face_windows_cut_from_detection_frame() {
        let filled = |red| {
            vec![
                Argb::<u8> {
                    alpha: 255,
                    red,
                    green: 0,
                    blue: 0,
                };
                (W * H) as usize
            ]
        };
        let (current, found_in) = (filled(50), filled(200));
        let bounds = motion::NormRect {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        let blob = |kind| motion::Blob {
            bounds,
            kind,
            ..motion::Blob::default()
        };
        let motion = Segmentation {
            blobs: vec![
                blob(motion::SegmentKind::Motion),
                blob(motion::SegmentKind::Face),
            ],
            ..Segmentation::default()
        };
        let tiles = cut_frame_tiles(
            &motion,
            Some(&View::from_pixels(&current, W, H)),
            Some(&View::from_pixels(&found_in, W, H)),
            0,
            (W, H),
            false,
            false,
        );
        let reds: Vec<u8> = tiles
            .iter()
            .map(|t| bytemuck::cast_slice::<_, u8>(&t.pixels)[1])
            .collect();
        assert_eq!(reds, [50, 200]);
    }

    /// A render that may be exported doesn't draw without faces: it waits for the models.
    #[test]
    fn non_live_render_waits_for_face_models() {
        wait_for_models();
        let frames = crossing();
        let mut state = IncrementalRenderer {
            simulate_models_loading: true,
            ..IncrementalRenderer::default()
        };
        assert!(face_detection_calls_in(&mut state, &mut Frames(&frames[..1]), 1, 0) > 0);
    }

    /// A device of the test's own, on the adapter `SEGCAM_GPU_ADAPTER` picks or on a software
    /// one where there is none; `None` where there is no adapter at all.
    fn own_device() -> Option<Arc<gpu::GpuContext>> {
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
    fn on_device(dev: &Arc<gpu::GpuContext>) -> IncrementalRenderer {
        let mut on_gpu = IncrementalRenderer::default();
        on_gpu.use_drawer(gpu::Drawer::new(dev.clone()).expect("a drawer on a working device"));
        on_gpu.set_gpu_min_coverage(0.0);
        on_gpu
    }

    fn drawn(state: &IncrementalRenderer) -> u64 {
        state.drawer().map_or(0, |d| d.drawn)
    }

    fn faults(state: &IncrementalRenderer) -> u64 {
        state.drawer().map_or(0, |d| d.faults)
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
    fn lost_gpu_falls_back_to_the_cpu() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
        let mut on_gpu = on_device(&dev);
        for n in 0..frames.len() {
            if n == 7 {
                assert!(drawn(&on_gpu) > 0, "drawing on the GPU before the loss");
                let draw = on_gpu.report().draw.expect("a render");
                assert_eq!(draw, format!("GPU ({})", dev.info.name));
                dev.device.destroy();
            }
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(n < 7 || a == b, "frame {n} isn't the CPU's after the loss");
        }
        assert_eq!(cpu.report().draw.as_deref(), Some("CPU"));
        assert_eq!(on_gpu.report().draw.as_deref(), Some("CPU"));
        assert!(dev.lost().is_some(), "the loss went unnoticed");
        assert!(
            on_gpu.drawer().is_none(),
            "the instance is still on the GPU"
        );
        assert!(!on_gpu.gpu_wanted, "the instance still looks for a GPU");
        assert!(
            gpu::Drawer::new(dev).is_none(),
            "a new instance draws on the lost device"
        );
    }

    /// Hashes from both paths are compared when a host stops streaming.
    #[test]
    fn parallel_frame_hash_matches_serial() {
        // An odd width leaves a row's last pixel out of the words of 8-bit frames.
        let (w, h) = (201, 130);
        let mut pixels: Vec<Argb<u8>> = (0..w * h)
            .map(|i| {
                let v = (i * 7919 % 256) as u8;
                Argb {
                    alpha: 255,
                    red: v,
                    green: v.wrapping_add(1),
                    blue: v.wrapping_add(2),
                }
            })
            .collect();
        let hash = |pixels: &[Argb<u8>]| {
            let view = View::from_pixels(pixels, w, h);
            let serial = frame_hash(&view);
            assert_eq!(frame_hash_parallel(&view), serial);
            serial
        };
        let before = hash(&pixels);
        let last = pixels.len() - 1;
        for at in [0, w as usize - 1, last / 2, last] {
            pixels[at].green ^= 1;
            assert_ne!(hash(&pixels), before, "a change at pixel {at}");
            pixels[at].green ^= 1;
        }
        assert_eq!(hash(&pixels), before);
        let (top, bottom) = pixels.split_at_mut(w as usize);
        top.swap_with_slice(&mut bottom[..w as usize]);
        assert_ne!(hash(&pixels), before, "two rows swapped");

        let floats: Vec<Argb<f32>> = (0..w * h)
            .map(|i| Argb {
                alpha: 1.0,
                red: i as f32 / 7.0,
                green: 0.25,
                blue: 0.5,
            })
            .collect();
        let view = View::from_pixels(&floats, w, h);
        assert_eq!(frame_hash_parallel(&view), frame_hash(&view));
    }

    /// A validation error leaves the device usable: it is counted, not kept as a failure, so
    /// new instances still draw on it and a render that didn't cause it isn't charged with it.
    #[test]
    fn error_leaves_the_device_usable() {
        let _lock = gpu_test_lock();
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
    fn disabling_gpu_mid_sequence_matches_cpu() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
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
        assert!(
            on_gpu.drawer().is_none(),
            "the instance is still on the GPU"
        );
        assert!(dev.lost().is_none(), "the device was lost");
    }

    /// A render the GPU can't take, here a canvas larger than the device binds, is drawn on
    /// the CPU byte for byte, and the next render is back on the GPU.
    #[test]
    fn fault_falls_back_for_one_render() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
        let mut on_gpu = on_device(&dev);
        for n in 0..frames.len() {
            let limit = if n == 7 { 1024 } else { u64::MAX };
            on_gpu
                .drawer
                .as_mut()
                .expect("on the GPU")
                .set_max_binding_bytes(limit);
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
        assert!(on_gpu.drawer().is_some(), "the instance left the GPU");
        assert!(dev.lost().is_none(), "the device was lost");
    }

    /// A draw that isn't back in time is drawn on the CPU, and the next render is back on
    /// the GPU with a fresh readback buffer. Whether the first draw is in time depends on
    /// the GPU, so only what must hold either way is checked.
    #[test]
    fn slow_readback_falls_back_for_one_render() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
        let mut on_gpu = on_device(&dev);
        for n in 0..frames.len() {
            let timeout = if n == 7 {
                std::time::Duration::ZERO
            } else {
                std::time::Duration::from_secs(10)
            };
            on_gpu
                .drawer
                .as_mut()
                .expect("on the GPU")
                .set_readback_timeout(timeout);
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
        assert!(on_gpu.drawer().is_some(), "the instance left the GPU");
        assert!(dev.lost().is_none(), "the device was lost");
    }

    /// A GPU failing every other render enters cooldown too: after `MAX_RECENT_FAULTS` faults
    /// in the history, renders match the CPU byte for byte without touching the GPU, and the GPU
    /// draws again when the cooldown ends.
    #[test]
    fn repeated_gpu_faults_cool_down_then_recover() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
        let mut on_gpu = on_device(&dev);
        // Frames 0, 2 and 4 fault; the cooldown runs from 5 until the test ends it at 8.
        let cooldown_frames = 5..8;
        for n in 0..frames.len() {
            let drawer = on_gpu.drawer.as_mut().expect("on the GPU");
            let failing = n < 8 && n % 2 == 0 || cooldown_frames.contains(&n);
            drawer.set_max_binding_bytes(if failing { 1024 } else { u64::MAX });
            if n == cooldown_frames.end {
                assert!(
                    drawer.in_fault_cooldown(),
                    "still trying the GPU at frame {n}"
                );
                drawer.set_cooldown(Duration::ZERO);
            }
            let in_cooldown = drawer.in_fault_cooldown();
            let (before, faulted) = (drawn(&on_gpu), faults(&on_gpu));
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_a_level(&a, &b), "frame {n} is off the CPU's");
            assert_eq!(in_cooldown, cooldown_frames.contains(&n), "frame {n}");
            if in_cooldown {
                assert_eq!(a, b, "cooldown frame {n} isn't the CPU's");
                assert_eq!(drawn(&on_gpu), before, "cooldown frame {n} drew on the GPU");
                assert_eq!(faults(&on_gpu), faulted, "cooldown frame {n} tried the GPU");
            } else if !failing {
                assert_eq!(drawn(&on_gpu), before + 1, "frame {n} wasn't on the GPU");
            }
        }
        assert_eq!(faults(&on_gpu), u64::from(gpu::MAX_RECENT_FAULTS));
        assert!(dev.lost().is_none(), "the device was lost");
    }

    /// A host holding only the current frame: tiles the pile stopped drawing keep no pixels.
    #[test]
    fn undrawn_tiles_release_pixels() {
        let frames = crossing();
        let s = RenderSettings {
            mode: OutputMode::Swarm,
            background_frames: 4,
            trail_frames: 12,
            max_tiles: 2,
            ..RenderSettings::default()
        };
        let mut state = IncrementalRenderer::default();
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let before = crate::previous_frames_needed(&s) as usize;
        let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
        for (n, frame) in frames.iter().enumerate() {
            history.next(before, 0).copy_from_slice(frame);
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            let count = history.len();
            state.render_frame(&s, &mut history, count, n as i64, view);
        }
        // The current frame's are dropped by the next render.
        let last = frames.len() as i64 - 1;
        let tiles: Vec<&Tile> = state
            .kept
            .range(..last)
            .filter_map(|(_, kept)| kept.tiles.as_ref())
            .flat_map(|cut| &cut.tiles)
            .collect();
        let with_pixels = tiles.iter().filter(|t| !t.pixels.is_empty()).count();
        assert!(tiles.len() > 2, "the clip spawned too few tiles");
        assert!(with_pixels <= 2, "{with_pixels} tiles keep their pixels");
    }

    /// A clip loop must not cost the atlas it grew or the drawer's buffers.
    #[test]
    fn clear_history_keeps_drawer_and_frees_atlas() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut drawer = gpu::Drawer::new(dev.clone()).expect("a drawer on a working device");
        let initial_words = 1024;
        drawer.small_atlas(initial_words);
        let mut on_gpu = IncrementalRenderer::default();
        on_gpu.use_drawer(drawer);
        on_gpu.set_gpu_min_coverage(0.0);
        for n in 0..frames.len() {
            render(&mut on_gpu, &frames, n);
        }
        let drawer = on_gpu.drawer().expect("a drawer");
        let (drawer_address, grown_words) = (drawer as *const gpu::Drawer, drawer.atlas_words().0);
        assert!(grown_words > initial_words, "the atlas never grew");

        on_gpu.clear_history();
        let drawer = on_gpu.drawer().expect("a drawer");
        assert_eq!(drawer as *const gpu::Drawer, drawer_address);
        assert_eq!(drawer.atlas_words(), (grown_words, grown_words));

        let drawn_before = drawn(&on_gpu);
        let mut cpu = IncrementalRenderer::default();
        for n in 0..frames.len() {
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_a_level(&a, &b), "frame {n} is off the CPU's");
        }
        assert_eq!(drawn(&on_gpu), drawn_before + frames.len() as u64);
        assert_eq!(
            on_gpu.drawer().expect("a drawer").atlas_words().0,
            grown_words
        );
    }

    /// A CPU draw stalls a live render, so one out of atlas keeps drawing on the GPU, with fewer
    /// windows.
    #[test]
    fn live_full_atlas_drops_oldest_windows_on_gpu() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut drawer = gpu::Drawer::new(dev.clone()).expect("a drawer on a working device");
        // Room for a few frames' tiles, and no more.
        drawer.small_atlas(1024);
        drawer.set_max_binding_bytes((W * H * 4) as u64 * 6);
        let mut state = IncrementalRenderer::default();
        state.use_drawer(drawer);
        state.set_gpu_min_coverage(0.0);
        let s = RenderSettings {
            mode: OutputMode::Swarm,
            background_frames: 4,
            trail_frames: 8,
            max_tiles: 20,
            ..RenderSettings::default()
        };
        let needed = crate::previous_frames_needed(&s) as usize;
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
        for (n, frame) in frames.iter().enumerate() {
            history.next(needed, 1).copy_from_slice(frame);
            let count = history.len();
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            state.render_frame(&s, &mut history, count, n as i64, view);
        }
        assert!(state.atlas.oldest_frame.is_some(), "the atlas never filled");
        assert_eq!(drawn(&state), frames.len() as u64, "a render left the GPU");
        assert_eq!(faults(&state), 0);
        assert!(state.report().out_of_gpu_memory);
        assert!(
            out.iter().any(|p| p.alpha == 255),
            "the newest windows are gone"
        );
    }

    /// What a live render left out for want of atlas space comes back in the first render that
    /// may be exported.
    #[test]
    fn exact_render_after_full_atlas_has_every_window() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut drawer = gpu::Drawer::new(dev.clone()).expect("a drawer on a working device");
        drawer.small_atlas(1024);
        drawer.set_max_binding_bytes((W * H * 4) as u64 * 6);
        let mut state = IncrementalRenderer::default();
        state.use_drawer(drawer);
        state.set_gpu_min_coverage(0.0);
        let s = swarm_settings();
        let needed = crate::previous_frames_needed(&s) as usize;
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
        for (n, frame) in frames.iter().enumerate() {
            history.next(needed, 1).copy_from_slice(frame);
            let count = history.len();
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            state.render_frame(&s, &mut history, count, n as i64, view);
        }
        assert!(state.atlas.oldest_frame.is_some(), "the atlas never filled");
        let n = frames.len() - 1;
        let after = render(&mut state, &frames, n);
        let fresh = render(&mut IncrementalRenderer::default(), &frames, n);
        assert!(within_a_level(&fresh, &after));
        assert_eq!(state.atlas.oldest_frame, None);
    }

    /// An atlas too small for the trail draws the renders it can't hold on the CPU, and the
    /// GPU draws again once the trail moves on; the instance never leaves it.
    #[test]
    fn full_atlas_falls_back_for_one_render() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
        let mut drawer = gpu::Drawer::new(dev.clone()).expect("a drawer on a working device");
        // Room for the canvas and a few frames' tiles, and no more.
        let bytes = (W * H * 4) as u64 + 4096;
        drawer.small_atlas(1024);
        drawer.set_max_binding_bytes(bytes);
        let mut on_gpu = IncrementalRenderer::default();
        on_gpu.use_drawer(drawer);
        on_gpu.set_gpu_min_coverage(0.0);
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
        assert!(on_gpu.drawer().is_some(), "the instance left the GPU");
        assert!(dev.lost().is_none(), "the device was lost");
    }

    #[test]
    fn refused_growth_keeps_atlas() {
        let _lock = gpu_test_lock();
        let bytes = (W * H * 4) as u64 + 4096;
        let adapter_choice = std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default();
        let opened = gpu::device_with_buffer_limit(&adapter_choice, bytes)
            .or_else(|_| gpu::device_with_buffer_limit("fallback", bytes));
        let Ok(dev) = opened.map(Arc::new) else {
            eprintln!("no adapter: skipped");
            return;
        };
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
        let mut drawer = gpu::Drawer::new(dev.clone()).expect("a drawer on a working device");
        // The usual first atlas is over the limit too.
        drawer.small_atlas(1024);
        let errors = dev.errors().0;
        let mut on_gpu = IncrementalRenderer::default();
        on_gpu.use_drawer(drawer);
        on_gpu.set_gpu_min_coverage(0.0);
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
        assert_eq!(faults(&on_gpu), 0, "a refused growth is not a fault");
        assert_eq!(
            dev.errors().0,
            errors,
            "a refusal reached the device's errors"
        );
        assert!(on_gpu.drawer().is_some(), "the instance left the GPU");
        assert!(dev.lost().is_none(), "the device was lost");
    }
}
