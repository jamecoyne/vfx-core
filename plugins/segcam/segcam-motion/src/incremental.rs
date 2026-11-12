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
use std::ops::{RangeBounds, RangeInclusive};

use vfx_core::cpu::{FrameSource, Image};
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

/// A frame's segments, with the key they were found for.
struct Analysis {
    key: AnalysisKey,
    segments: Segmentation,
}

/// A frame's tiles, with the key and mirroring they were cut for.
struct CutTiles {
    key: AnalysisKey,
    mirror: bool,
    tiles: Vec<Tile>,
}

/// What is kept of one frame for later frames' analyses and for draws. Each
/// part comes and goes on its own: a change to a frame drops the analysis of
/// the frames that used it as background, and keeps their grids.
#[derive(Default)]
struct FrameRecord {
    /// The frame's luma, which later frames' backgrounds read. Never drawn.
    grid: Option<LumaGrid>,
    analysis: Option<Analysis>,
    /// Pixels cut from the frame, freed apart from the rest once no draw shows
    /// them.
    tiles: Option<CutTiles>,
}

/// A frame's record and what the renderer knows of it.
#[derive(Default)]
struct KeptFrame {
    record: FrameRecord,
    read: Option<PixelsRead>,
    /// Under a fetch budget, an earlier frame its analysis used has changed
    /// or wasn't read. The analysis and tiles are drawn until redone.
    stale: bool,
    /// The budgeted render that last verified it or found it missing, 0 for
    /// none. Checks take the oldest first.
    last_checked: u64,
}

#[derive(Default)]
pub struct IncrementalRenderer {
    history_format: Option<HistoryFormat>,
    kept: BTreeMap<i64, KeptFrame>,
    /// Renders under a fetch budget so far, which date
    /// [`KeptFrame::last_checked`].
    budgeted_render_count: u64,
    /// The last budgeted render from an unfixed source, which may follow an
    /// edit. A stream checks the frames checked before it.
    last_unfixed_render: u64,
    /// The budgeted render where the current run of changes was found.
    /// Frames checked before it are read oldest first with those to rebuild,
    /// so no frame is redone from an unchecked background frame.
    changes_found_at: u64,
    /// Created on the first GPU render after the device opens, not per instance, because hosts
    /// create instances often. A frame's tiles enter the atlas on first GPU draw.
    drawer: Option<Box<Drawer>>,
    atlas: AtlasPlacements,
    /// Whether the last render's source is live, so it must not stall to stay exact.
    source_is_live: bool,
    gpu_wanted: bool,
    /// Suppresses repeat logs of the same message.
    last_reported_fault: Option<String>,
    /// The last render's draw hasn't been read back from the GPU yet (browsers).
    unfinished: bool,
    /// `None` before the first render.
    drew_on_gpu: Option<bool>,
    faces_wanted: bool,
    analyser: Analyser,
    readings: Vec<Reading>,
    /// Live renders act as if the face models were still loading.
    #[cfg(test)]
    simulate_models_loading: bool,
    /// A host's GPU frame the render may draw into instead of its output.
    target: Option<TargetOffer>,
    /// The slot this instance draws from, where it isn't the process's.
    #[cfg(test)]
    test_slot: Option<&'static gpu::DeviceSlot>,
}

/// A host's frame, offered to one render ([`vfx_core::CpuEffect::render_to`]), the record of
/// the device it's on, and whether the render drew into it.
struct TargetOffer {
    faults: vfx_core::DeviceFaults,
    frame: vfx_core::Frame,
    drawn_into: bool,
}

/// What stays the same through one render.
struct RenderPlan<'a> {
    s: &'a RenderSettings,
    number: i64,
    /// The frame at index 0 of this render's count. A source that holds fewer frames than it
    /// counts ([`FrameSource::held`]) can't hand over the oldest of them.
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
    /// The earlier frames this render may read ([`FrameSource::fetch_budget`]).
    fetch_budget: Option<usize>,
    /// A live source numbers each frame once, so a number says which frame it is.
    numbers_identify_frames: bool,
    /// The current frame is analysed again to take what detection found: a picture that holds
    /// still stays one frame, and only an analysis takes faces.
    takes_faces_on_held_picture: bool,
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
            fetch_budget: source.fetch_budget(),
            numbers_identify_frames: source.fixed() && source.live(),
            takes_faces_on_held_picture: false,
        }
    }

    /// The frames the analysis of the newest frame this render analyses reads:
    /// that frame and its background. A render under a fetch budget reads those
    /// it has nothing kept for first, newest first and up to twice the budget,
    /// so a render after a jump shows the motion around its own frame.
    fn own_analysis_inputs(&self) -> BTreeSet<i64> {
        let Some(&frame) = self.to_analyse.range(..=self.number).next_back() else {
            return BTreeSet::new();
        };
        let every = self.s.analysis_step.every;
        (analysis_key(self.s, frame, self.first).background_start..frame)
            .filter(|f| f.rem_euclid(every) == 0)
            .chain([frame])
            .collect()
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

/// What a render under a fetch budget reads.
struct BudgetedReads {
    /// Every frame it reads, the current one included.
    read: BTreeSet<i64>,
    /// Those read to show them unchanged.
    checks: BTreeSet<i64>,
    /// Frames to rebuild left for later renders.
    deferred: Vec<i64>,
}

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

    fn device_slot(&self) -> &'static gpu::DeviceSlot {
        #[cfg(test)]
        if let Some(slot) = self.test_slot {
            return slot;
        }
        &gpu::SHARED_SLOT
    }

    /// Whether Swarm is to draw on the GPU where there is one (see [`Self::use_gpu`]).
    pub fn gpu_wanted(&self) -> bool {
        self.gpu_wanted
    }

    /// The drawer, once a render has drawn on the GPU.
    pub fn drawer(&self) -> Option<&Drawer> {
        self.drawer.as_deref()
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

    /// Offers the next render `target` to draw into instead of its output, which a swarm drawn
    /// on the GPU of the same device takes; [`IncrementalRenderer::took_target`] says whether it
    /// did.
    pub fn offer_target(&mut self, target: vfx_core::Target<'_>) {
        let f = target.frame;
        self.target = Some(TargetOffer {
            faults: target.gpu.faults(),
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
                && drawer.draws_on(&offer.faults)
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

        let mut plan = RenderPlan::new(s, source, count, number, (w, h));
        let current_hash = self.check_for_restart(&plan, source);
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
        plan.takes_faces_on_held_picture = self.drop_analysis_for_waiting_faces(&plan);
        let swarm = s.mode == OutputMode::Swarm;
        if swarm {
            self.pin_unreadable_frames(s, plan.first, number, oldest_held_frame);
        }
        let (mut checks, mut deferred) = (BTreeSet::new(), Vec::new());
        let (mut all_verified, mut changed) = (false, false);
        let frames_to_read = match plan.fetch_budget {
            None if plan.fixed => self.frames_to_read(s, plan.first, number, oldest_held_frame),
            None => plan.frames_feeding_analysis(),
            Some(budget) => {
                self.budgeted_render_count += 1;
                if !plan.fixed {
                    self.last_unfixed_render = self.budgeted_render_count;
                }
                let mut rebuild = self.frames_to_read(s, plan.first, number, oldest_held_frame);
                let checked = |f: &i64| self.kept.get(f).map_or(0, |kept| kept.last_checked);
                // A frame found missing is checked for its return with the
                // verified ones, or a missing frame that is never analysed would
                // take the budget every render.
                let known_missing: Vec<i64> = rebuild
                    .iter()
                    .copied()
                    .filter(|&f| f != number && self.is_known_missing(f))
                    .collect();
                for f in &known_missing {
                    rebuild.remove(f);
                }
                let verified_from = match plan.fixed {
                    true => self.last_unfixed_render.max(self.changes_found_at),
                    false => self.changes_found_at,
                };
                let (unverified, mut verified): (Vec<i64>, Vec<i64>) = plan
                    .frames_feeding_analysis()
                    .into_iter()
                    .filter(|f| *f >= oldest_held_frame && !rebuild.contains(f))
                    .partition(|f| checked(f) < verified_from);
                // A stream checks only what may have changed before it began.
                if plan.fixed {
                    verified.clear();
                }
                verified.extend(
                    known_missing
                        .into_iter()
                        .filter(|f| !unverified.contains(f)),
                );
                verified.sort_by_key(|f| (checked(f), *f));
                verified.dedup();
                // Frames still to rebuild belong to the run that made them stale.
                all_verified = unverified.is_empty()
                    && rebuild
                        .iter()
                        .all(|f| *f == number || checked(f) >= verified_from);
                // A stream reads the frames it went over, which have nothing
                // kept, before the rest, or a frame the host skipped would wait
                // behind the checks a render that wasn't a stream left.
                let ahead: BTreeSet<i64> = match plan.fixed {
                    true => rebuild
                        .iter()
                        .copied()
                        .filter(|f| *f != number && !self.kept.contains_key(f))
                        .collect(),
                    false => BTreeSet::new(),
                };
                let own: BTreeSet<i64> = plan
                    .own_analysis_inputs()
                    .into_iter()
                    .filter(|f| {
                        *f != number && *f >= oldest_held_frame && !self.kept.contains_key(f)
                    })
                    .collect();
                let unverified = BTreeSet::from_iter(unverified);
                let reads = Self::budgeted_reads(
                    rebuild,
                    &unverified,
                    &verified,
                    &own,
                    &ahead,
                    number,
                    budget,
                );
                (checks, deferred) = (reads.checks, reads.deferred);
                reads.read
            }
        };
        if !plan.live || !s.faces.any() {
            self.analyser.live_faces.forget();
        }

        // Frames the host didn't hand over count as empty, as in a cold render. Without
        // a fetch budget nothing computed from one is cached, so it is read again next
        // time; under one it is checked again with the frames verified longest ago.
        let mut missing = BTreeSet::new();
        for frame in frames_to_read {
            let index = (frame - plan.first) as usize;
            let before = self.read_before(frame);
            // Whether this read has to show the frame unchanged, rather than take it as it
            // was.
            let verify = !plan.fixed || checks.contains(&frame);
            let read_known = if !verify && before.is_some() {
                before
            } else if plan.numbers_identify_frames && !verify {
                Some(PixelsRead::NotHashed)
            } else if frame == number {
                current_hash.map(PixelsRead::Hash)
            } else {
                None
            };
            let read = std::cell::Cell::new(read_known);
            // A frame found unchanged with nothing left to work out needs no pixels, and
            // converting them is most of what a check costs a host that converts. A fixed
            // source lists only frames it needs, besides checks. Decided here, after
            // earlier frames' changes have dropped what they invalidate.
            let unchanged_is_enough = verify
                && frame != number
                && (!plan.to_analyse.contains(&frame)
                    || self.is_cached(frame, &analysis_key(s, frame, plan.first), s));
            let mut wanted = |rows: &[&[u8]]| {
                if read_known.is_some() {
                    return true;
                }
                let hashed = PixelsRead::Hash(frame_hash(rows));
                read.set(Some(hashed));
                !(unchanged_is_enough && before == Some(hashed))
            };
            let mut use_frame = |image: Option<Image<'_>>| {
                let view = image.and_then(C::view);
                let read = read.get();
                self.note_pixels(&plan, frame, view.as_ref(), read, verify, &mut missing);
                self.analyse_frame(&plan, frame, view.as_ref(), &missing);
                if frame == number {
                    self.draw(s, count, number, view, out);
                }
            };
            source.with_frame_if_wanted(index, &mut wanted, &mut use_frame);
            changed |= verify && before.is_some() && read.get().is_some_and(|r| Some(r) != before);
            // A frame the host didn't hand over goes to the back too, or it would take the
            // budget every render.
            if plan.fetch_budget.is_some()
                && (verify || before.is_none())
                && let Some(kept) = self.kept.get_mut(&frame)
            {
                kept.last_checked = self.budgeted_render_count;
            }
        }
        if changed && all_verified {
            self.changes_found_at = self.budgeted_render_count;
        }
        // Under a budget note_pixels has already made a missing frame's
        // dependents stale before they were read. Doing it again here would
        // have the budget redo the oldest of them every render.
        if plan.fetch_budget.is_none() {
            for &frame in &missing {
                self.invalidate_frame(&plan, frame);
            }
        }
        // Frames that use a deferred frame without a grid go stale, and so does
        // the current frame if its face source is deferred. The current frame's
        // own frames are read before older ones, so a deferred frame can be
        // older than frames this render analysed; the stale range after it
        // covers those too.
        if deferred.contains(&face_source_frame(s, number, plan.first)) {
            self.mark_stale(number..=number);
        }
        deferred.retain(|&f| kept_grid(&self.kept, f).is_none());
        for frame in deferred {
            self.mark_stale(frame..=frame + plan.invalidation_reach);
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

    /// While a device opens, or until a lost one's replaced, renders draw on the CPU.
    fn open_drawer_if_wanted(&mut self, s: &RenderSettings) {
        let mode_uses_gpu = s.mode == OutputMode::Swarm;
        // Read first, so a frame offered on a new device doesn't meet a drawer on the lost one.
        if let Some(why) = self.drawer.as_ref().and_then(|drawer| drawer.loss()) {
            self.fall_back(&Fault(why));
        }
        if self.gpu_wanted && self.drawer.is_none() && mode_uses_gpu {
            match self.device_slot().now() {
                gpu::DeviceNow::Ready(device) => {
                    if let Some(drawer) = Drawer::new(device) {
                        self.last_reported_fault = None;
                        self.set_drawer(Some(Box::new(drawer)));
                    }
                }
                gpu::DeviceNow::Waiting => {}
                gpu::DeviceNow::Never => self.gpu_wanted = false,
            }
        }
    }

    /// Clears the history if the source started over, or if a fixed source can't hand over the
    /// current frame, and returns the current frame's hash if that took one. A fixed source's
    /// earlier frames never change, so only the current frame is hashed: a different hash means the
    /// source restarted. A live source numbers each frame once, so its frames aren't hashed at all,
    /// and a lower number means it restarted.
    fn check_for_restart(
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
            source.with_frame_if_wanted(
                (plan.number - plan.first) as usize,
                &mut |rows| {
                    current_hash = Some(frame_hash(rows));
                    false
                },
                &mut |_| {},
            );
            let read = self.read_before(plan.number);
            if current_hash.is_none_or(|hash| read.is_some_and(|r| r != PixelsRead::Hash(hash))) {
                self.clear_history();
            }
        }
        current_hash
    }

    /// Whether a finished detection waits while the current frame's analysis is already kept, in
    /// which case that analysis goes so this render takes the faces. A frame between analyses
    /// has none to redo, so a picture holding still there keeps the faces it shows.
    fn drop_analysis_for_waiting_faces(&mut self, plan: &RenderPlan<'_>) -> bool {
        if !plan.numbers_identify_frames
            || !plan.s.faces.any()
            || !plan.to_analyse.contains(&plan.number)
            || !self.analyser.live_faces.answer_waiting()
        {
            return false;
        }
        match self.kept.get_mut(&plan.number) {
            Some(kept) if kept.record.analysis.is_some() => {
                kept.record.analysis = None;
                true
            }
            _ => false,
        }
    }

    /// Records that `frame` was read as `read`, with its grid, and drops what a
    /// change to it invalidates if the read was to `verify` it. A frame with no
    /// pixels or no `read` counts as missing.
    fn note_pixels<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        frame: i64,
        view: Option<&View<C>>,
        read: Option<PixelsRead>,
        verify: bool,
        missing: &mut BTreeSet<i64>,
    ) {
        let (Some(v), Some(read)) = (view, read) else {
            // A frame already known missing changed nothing since it was found so.
            if !self.is_known_missing(frame) {
                self.invalidate_frame(plan, frame);
            }
            missing.insert(frame);
            // Under a fetch budget it counts as empty, as in a fresh render, so later
            // renders don't spend the budget asking for it again. Without a hash a check
            // still does.
            if plan.fetch_budget.is_some() {
                self.kept.entry(frame).or_default().record.grid = Some(LumaGrid::default());
            }
            return;
        };
        if self.read_before(frame) != Some(read) {
            // A fixed source's frame read without a check can't have changed what later
            // frames computed from it; in other sources, and in a check, an uncached frame
            // may have.
            if verify {
                self.invalidate_frame(plan, frame);
            }
            let kept = self.kept.entry(frame).or_default();
            kept.record.grid = Some(downsample(v, plan.s.grid_width));
            kept.read = Some(read);
        }
    }

    /// Works out `frame`'s analysis and tiles if this render wants them and they
    /// aren't kept, and keeps them in its record.
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
        let analysed = self
            .analyser
            .analyse(plan, &self.kept, frame, key, view, missing);
        let kept = self.kept.entry(frame).or_default();
        if let Some(analysis) = analysed.analysis {
            kept.record.analysis = Some(analysis);
            kept.stale = false;
        }
        if let Some(tiles) = analysed.tiles {
            kept.record.tiles = Some(tiles);
            self.atlas.frames.remove(&frame);
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
            kept.record
                .analysis
                .get_or_insert_with(|| Analysis {
                    key,
                    segments: Segmentation::default(),
                })
                .key = key;
            let tiles = kept.record.tiles.get_or_insert_with(|| CutTiles {
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
        if !older.any(|(_, kept)| kept.record.tiles.is_some()) {
            return;
        }
        let visible: BTreeSet<(i64, usize)> =
            build_pile(&self.kept, s, count, number, width, height)
                .visible_entries()
                .map(|p| (p.frame, p.index))
                .collect();
        for (&frame, kept) in self.kept.range_mut(..oldest_held_frame) {
            let tiles = kept
                .record
                .tiles
                .iter_mut()
                .flat_map(|cut| cut.tiles.iter_mut());
            for (index, tile) in tiles.enumerate() {
                if !visible.contains(&(frame, index)) {
                    tile.hidden = true;
                    tile.pixels = Vec::new();
                }
            }
        }
        let oldest_displayed_analysis = number - AnalysisStep::MAX_EVERY;
        for (_, kept) in self
            .kept
            .range_mut(..oldest_held_frame.min(oldest_displayed_analysis))
        {
            if let Some(analysis) = &mut kept.record.analysis {
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
                .filter(|&k| {
                    k.rem_euclid(s.analysis_step.every) == 0 && kept_grid(&self.kept, k).is_none()
                })
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

    /// Draws frame `number`, the last of `count`, from its pixels, `frame`, and
    /// the records of the frames it shows, never other frames' pixels. Every
    /// output but the swarm draws over `frame`; a GPU fault draws on the CPU.
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

    /// Device loss leaves this instance on the CPU until the slot has another device, whose drawer
    /// starts with an empty atlas: tiles are in RAM, so the trail goes on. Any other fault retries
    /// on the next render, unless [`gpu::MAX_RECENT_FAULTS`] is reached (cooldown).
    fn fall_back(&mut self, fault: &Fault) {
        if self.last_reported_fault.as_ref() != Some(&fault.0) {
            eprintln!(
                "segcam Motion: the GPU failed ({}); drawing on the CPU",
                fault.0
            );
            self.last_reported_fault = Some(fault.0.clone());
        }
        match &mut self.drawer {
            Some(drawer) if drawer.lost() => self.set_drawer(None),
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
        match cached_analysis(&self.kept, analysis, &analysis_key(s, analysis, first)) {
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
        let motion = cached_analysis(&self.kept, frame, &analysis_key(s, frame, first))
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
        if s.mode != OutputMode::Swarm {
            self.draw_cpu(s, count, number, frame, out);
            return Ok(());
        }
        let (w, h) = (out.width, out.height);
        let into_target = self.can_draw_into_target::<C>(drawer, (w, h));
        let m = SwarmMetrics::new(h);
        let shadow = swarm::shadow_kernel(m.scale);
        let pile = build_pile(&self.kept, s, count, number, w, h);
        let mut visible: Vec<PileEntry> = pile.visible_entries().collect();
        self.atlas
            .leave_out_dropped(&mut visible, self.source_is_live);
        let live = self.source_is_live;
        // With no windows both paths draw the same picture, and the CPU's
        // costs nothing next to a readback.
        let no_windows = visible.is_empty() && !into_target;
        if no_windows
            || !self
                .atlas
                .place(&self.kept, &visible, drawer, (w, h), live)?
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
            drawer.draw_swarm_into(&offer.frame, windows, frame.as_ref(), s, &shadow)?;
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
        self.analyser.live_faces.forget();
        self.unfinished = false;
        if let Some(drawer) = &mut self.drawer {
            drawer.drop_unfinished_draw();
        }
    }

    /// Whether `frame` has cached motion and, for the swarm, tiles, none of it
    /// stale.
    fn is_cached(&self, frame: i64, key: &AnalysisKey, s: &RenderSettings) -> bool {
        fresh_analysis(&self.kept, frame, key).is_some()
            && (s.mode != OutputMode::Swarm
                || cached_tiles(&self.kept, frame, key, s.mirror).is_some())
    }

    /// Whether a render under a fetch budget found `frame` missing and it hasn't
    /// been handed over since: it has the empty grid such a render gives it.
    fn is_known_missing(&self, frame: i64) -> bool {
        kept_grid(&self.kept, frame).is_some_and(|g| g.count() == 0)
    }

    fn read_before(&self, frame: i64) -> Option<PixelsRead> {
        self.kept.get(&frame)?.read
    }

    /// Drops `frame`'s grid and hash, and the motion and tiles of the frames that
    /// used it. Under a fetch budget those are stale instead, and drawn until they
    /// are worked out again.
    fn invalidate_frame(&mut self, plan: &RenderPlan<'_>, frame: i64) {
        if let Some(kept) = self.kept.get_mut(&frame) {
            kept.record.grid = None;
            kept.read = None;
        }
        // The frame's own work is redone while its pixels are at hand, or, for a
        // missing frame outside the swarm, not done at all, as in a fresh render.
        let dependents = match plan.fetch_budget {
            Some(_) => {
                // Empty where nothing reads the frame's background.
                self.mark_stale(frame + 1..frame + plan.invalidation_reach + 1);
                frame..=frame
            }
            None => frame..=frame + plan.invalidation_reach,
        };
        for later in dependents {
            if let Some(kept) = self.kept.get_mut(&later) {
                kept.record.analysis = None;
                kept.record.tiles = None;
            }
            self.atlas.frames.remove(&later);
        }
    }

    fn mark_stale(&mut self, frames: impl RangeBounds<i64>) {
        for (_, kept) in self.kept.range_mut(frames) {
            kept.stale |= kept.record.analysis.is_some() || kept.record.tiles.is_some();
        }
    }

    /// The frames a render of `number` reads under a budget of `budget`
    /// earlier frames: the newest of `own`, up to twice the budget, then
    /// `ahead`, then the oldest of `rebuild` and `unverified` together, then
    /// the first of `verified`. The rest of `own` waits in `rebuild`, so the
    /// analysis that needs it is redone once it is read.
    fn budgeted_reads(
        rebuild: BTreeSet<i64>,
        unverified: &BTreeSet<i64>,
        verified: &[i64],
        own: &BTreeSet<i64>,
        ahead: &BTreeSet<i64>,
        number: i64,
        budget: usize,
    ) -> BudgetedReads {
        let own: BTreeSet<i64> = own
            .iter()
            .rev()
            .take(budget.saturating_mul(2))
            .copied()
            .collect();
        let budget = budget.max(own.len());
        let rest = rebuild
            .union(unverified)
            .filter(|f| !ahead.contains(f) && !own.contains(f));
        let ahead = ahead.iter().filter(|f| !own.contains(f));
        let mut oldest = own
            .iter()
            .chain(ahead)
            .chain(rest)
            .copied()
            .filter(|&f| f != number);
        let mut read: BTreeSet<i64> = oldest.by_ref().take(budget).collect();
        let deferred = oldest.filter(|f| rebuild.contains(f)).collect();
        let room = budget - read.len();
        read.extend(verified.iter().take(room));
        let checks = read
            .iter()
            .copied()
            .filter(|f| !rebuild.contains(f))
            .collect();
        read.insert(number);
        BudgetedReads {
            read,
            checks,
            deferred,
        }
    }

    fn retain_frames(&mut self, frames: RangeInclusive<i64>) {
        self.kept.retain(|f, _| frames.contains(f));
        self.atlas.frames.retain(|f, _| frames.contains(f));
    }
}

/// What analysing frames keeps that is no frame's record: live face detection,
/// whose answers arrive after the frame they were asked about.
#[derive(Default)]
struct Analyser {
    live_faces: crate::faces::LiveFaces,
    #[cfg(test)]
    face_detection_calls: u32,
}

/// The parts of a frame's record that [`Analyser::analyse`] made.
struct RecordParts {
    analysis: Option<Analysis>,
    tiles: Option<CutTiles>,
}

impl Analyser {
    /// Analyses `frame` under `key` from its pixels and these records in
    /// `kept`: the grids of `frame` and of `key`'s background frames, and of
    /// the face source `key` names its grid and, if that entry isn't `stale`,
    /// its analysis. `missing` frames have no grid. Gives an analysis unless a
    /// fresh one is kept, and the swarm's tiles.
    fn analyse<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        kept: &BTreeMap<i64, KeptFrame>,
        frame: i64,
        key: AnalysisKey,
        view: Option<&View<C>>,
        missing: &BTreeSet<i64>,
    ) -> RecordParts {
        let analysis = fresh_analysis(kept, frame, &key)
            .is_none()
            .then(|| Analysis {
                key,
                segments: self.find_segments(plan, kept, frame, &key, view, missing),
            });
        let segments = match &analysis {
            Some(found) => Some(&found.segments),
            None => fresh_analysis(kept, frame, &key),
        };
        let tiles = segments
            .filter(|_| plan.s.mode == OutputMode::Swarm)
            .map(|segments| self.cut_tiles(plan, frame, key, segments, view));
        RecordParts { analysis, tiles }
    }

    /// `frame`'s segments: what moved or passed the threshold, then its faces.
    fn find_segments<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        kept: &BTreeMap<i64, KeptFrame>,
        frame: i64,
        key: &AnalysisKey,
        view: Option<&View<C>>,
        missing: &BTreeSet<i64>,
    ) -> Segmentation {
        let s = plan.s;
        let empty = LumaGrid::default();
        let grid = kept_grid(kept, frame).unwrap_or(&empty);
        let mut motion = match s.detection_mode {
            DetectionMode::Motion => {
                let history = (key.background_start..frame)
                    .filter(|k| k.rem_euclid(s.analysis_step.every) == 0)
                    .map(|k| kept_grid(kept, k).unwrap_or(&empty));
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
        // The face source is analysed first, unless it is missing or a live
        // source left it out, which leaves it no grid, or under a fetch budget
        // an empty one, and no faces.
        let source_frame = face_source_frame(s, frame, plan.first);
        let source_key = analysis_key(s, source_frame, plan.first);
        let source_motion = key
            .face_source
            .filter(|_| !plan.takes_faces_on_held_picture)
            .filter(|_| {
                !missing.contains(&source_frame)
                    && kept_grid(kept, source_frame).is_some_and(|g| g.count() > 0)
            })
            .and_then(|_| fresh_analysis(kept, source_frame, &source_key));
        if let Some(before) = source_motion {
            motion.blobs.extend(reused_faces(before, source_frame));
        } else if let Some(view) = view.filter(|_| s.faces.any()) {
            #[cfg(test)]
            {
                self.face_detection_calls += 1;
            }
            if plan.live {
                // A live render takes the newest faces there are, and those
                // already shown stay until newer ones arrive.
                let first_face = motion.blobs.len();
                self.live_faces
                    .exchange(view, s.faces, cells, frame, first_face);
                motion.blobs.extend(self.live_faces.faces_on(frame));
            } else {
                // Any other render may be exported, so it has this frame's own
                // faces, however long they take.
                motion
                    .blobs
                    .extend(crate::faces::find(view, s.faces, cells));
            }
        }
        motion
    }

    /// Cuts `frame`'s windows out of its pixels.
    fn cut_tiles<C: Depth>(
        &mut self,
        plan: &RenderPlan<'_>,
        frame: i64,
        key: AnalysisKey,
        motion: &Segmentation,
        view: Option<&View<C>>,
    ) -> CutTiles {
        let s = plan.s;
        // Reused faces need tiles here if a trail no longer than the step draws
        // no earlier analysis, or if the windows of held faces would leave it
        // before the next analysis, as a slow detection's do on a short trail.
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
        if held_windows_leaving {
            self.live_faces.windows_cut_on(frame);
        }
        CutTiles {
            key,
            mirror: s.mirror,
            tiles,
        }
    }
}

/// Where kept frames' tiles sit in the atlas. A record holds no GPU handles,
/// and a draw changes this while the windows it draws borrow their tiles.
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
        live: bool,
    ) -> Result<bool, Fault> {
        let wanted: BTreeSet<i64> = visible.iter().map(|p| p.frame).collect();
        self.frames.retain(|f, _| wanted.contains(f));
        for &f in wanted.iter().rev() {
            if self.frames.contains_key(&f) || self.oldest_frame.is_some_and(|oldest| f < oldest) {
                continue;
            }
            let Some(cut) = kept.get(&f).and_then(|k| k.record.tiles.as_ref()) else {
                continue;
            };
            loop {
                if let Some(allocation) = drawer.upload_tiles(&cut.tiles, canvas)? {
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
    let analysis = kept.get(&frame)?.record.analysis.as_ref()?;
    (analysis.key == *key).then_some(&analysis.segments)
}

/// [`cached_analysis`] if `frame` isn't stale, which later work may build on.
fn fresh_analysis<'a>(
    kept: &'a BTreeMap<i64, KeptFrame>,
    frame: i64,
    key: &AnalysisKey,
) -> Option<&'a Segmentation> {
    kept.get(&frame).filter(|kept| !kept.stale)?;
    cached_analysis(kept, frame, key)
}

fn kept_grid(kept: &BTreeMap<i64, KeptFrame>, frame: i64) -> Option<&LumaGrid> {
    kept.get(&frame)?.record.grid.as_ref()
}

/// The tiles of `frame`, if cut for `key` and `mirror`.
fn cached_tiles<'a>(
    kept: &'a BTreeMap<i64, KeptFrame>,
    frame: i64,
    key: &AnalysisKey,
    mirror: bool,
) -> Option<&'a [Tile]> {
    let cut = kept.get(&frame)?.record.tiles.as_ref()?;
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
fn row_hash(bytes: &[u8]) -> u64 {
    let mut lanes = [0u64; 4];
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

/// A hash of every byte of a frame ([`FrameSource::with_frame_if_wanted`]), on the host worker
/// threads, from the rows' hashes in order, so any thread count gives the same one. A stream's
/// single frame hashes on threads too: one thread made a stream render slower at 1080p Float.
fn frame_hash(rows: &[&[u8]]) -> u64 {
    let mut hashes = vec![0u64; rows.len()];
    by_rows(&mut hashes, 1, |y, hash| {
        hash[0] = row_hash(rows[y]);
    });
    fold_hash(hashes.into_iter())
}

#[cfg(test)]
mod tests {
    use super::gpu::gpu_test_lock;
    use std::sync::Arc;
    use std::time::Duration;
    use vfx_core::wgpu;

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

    /// A host handing over a picture again makes no new windows: the picture after repeats is the
    /// one without them, and rendering a repeat shows what the frame before it showed.
    #[test]
    fn repeated_frames_add_nothing() {
        let frames = crossing();
        let s = swarm_settings();
        let needed = crate::previous_frames_needed(&s) as usize;
        let play = |repeats: usize| {
            let mut state = IncrementalRenderer::default();
            let mut history = vfx_core::cpu::History::<u8>::default();
            history.resize(W, H);
            let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
            for frame in &frames {
                let mut shown = None;
                for _ in 0..=repeats {
                    history.next(needed, 0).copy_from_slice(frame);
                    history.drop_repeat();
                    let count = history.len();
                    let number = history.frame_number();
                    let view = &mut ViewMut::from_pixels(&mut out, W, H);
                    state.render_frame(&s, &mut history, count, number, view);
                    assert!(
                        shown.is_none_or(|shown| shown == out),
                        "a repeat changed the picture"
                    );
                    shown = Some(out.clone());
                }
            }
            out
        };
        assert!(play(3) == play(0));
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

    /// Consecutive frames of a stream that holds only the current one look for
    /// faces on every second analysis, the others taking their face source's
    /// (SPEC F7).
    #[test]
    fn stream_holding_one_frame_reuses_faces() {
        wait_for_models();
        let frames = crossing();
        let s = with_faces(ALL_PARTS);
        let needed = crate::previous_frames_needed(&s) as usize;
        let mut state = IncrementalRenderer::default();
        let consecutive = 4..frames.len();
        let mut calls = 0;
        for n in consecutive.clone() {
            let clip = &frames[n - needed.min(n)..=n];
            let mut source = OnlyNewestFrame {
                frames: clip,
                reads: 0,
            };
            calls += face_detection_calls_in(&mut state, &mut source, clip.len(), n as i64);
        }
        let renders = consecutive.len() as u32;
        assert!(
            calls <= renders.div_ceil(2) + 1,
            "{calls} detections in {renders} renders"
        );
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
        let before = state.analyser.face_detection_calls;
        state.render_frame(&s, source, count, n, view);
        state.analyser.face_detection_calls - before
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

    /// A picture that holds still takes what detection found in it, on a frame with faces of its
    /// own and on one that reuses the frame before's, and doesn't change again until there's more.
    #[test]
    fn held_picture_takes_the_faces_found_in_it() {
        wait_for_models();
        // The worker waits behind this, so the only answers are the ones left here.
        let models = crate::faces::lock_models();
        let frames = crossing();
        let s = with_faces(ALL_PARTS);
        let before = crate::previous_frames_needed(&s) as usize;
        let mut state = IncrementalRenderer::default();
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let mut show = |state: &mut IncrementalRenderer, frame: &[Argb<u8>]| {
            history.next(before, before).copy_from_slice(frame);
            history.drop_repeat();
            let (count, number) = (history.len(), history.frame_number());
            let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            state.render_frame(&s, &mut history, count, number, view);
            out
        };
        let face = motion::Blob {
            bounds: crate::NormRect {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            kind: motion::SegmentKind::Face,
            ..motion::Blob::default()
        };
        for (at, red) in [(0, 201), (1, 202)] {
            let found_in = Argb {
                alpha: 255,
                red,
                green: 7,
                blue: 9,
            };
            let first = show(&mut state, &frames[at]);
            assert!(!first.contains(&found_in));
            assert!(state.analyser.live_faces.request_out());
            let requests = state.analyser.live_faces.requests_sent;
            state
                .analyser
                .live_faces
                .answer_with(ALL_PARTS, vec![face], found_in, (W, H));
            let with_faces = show(&mut state, &frames[at]);
            assert!(with_faces.contains(&found_in), "no window at frame {at}");
            assert!(!state.analyser.live_faces.answer_waiting());
            assert!(show(&mut state, &frames[at]) == with_faces);
            assert_eq!(state.analyser.live_faces.requests_sent, requests);
        }
        drop(models);
    }

    /// A picture that holds still is asked about once, however long it holds.
    #[test]
    fn held_picture_stops_asking_for_faces() {
        wait_for_models();
        let frames = crossing();
        let s = with_faces(ALL_PARTS);
        let before = crate::previous_frames_needed(&s) as usize;
        let mut state = IncrementalRenderer::default();
        let mut history = vfx_core::cpu::History::<u8>::default();
        history.resize(W, H);
        let started = std::time::Instant::now();
        while started.elapsed() < Duration::from_secs(2) {
            history.next(before, before).copy_from_slice(&frames[0]);
            history.drop_repeat();
            let (count, number) = (history.len(), history.frame_number());
            let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            state.render_frame(&s, &mut history, count, number, view);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(state.analyser.live_faces.requests_sent, 1);
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
        while !state.analyser.live_faces.holds_faces() {
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
        assert!(!state.analyser.live_faces.holds_faces());
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
                state.analyser.live_faces.hold(ALL_PARTS, vec![face], 0, 0);
                let pixels = vec![found_in; (W * H) as usize];
                state.analyser.live_faces.hold_pixels(pixels, W, H);
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
                state.analyser.live_faces.hold(ALL_PARTS, vec![face], 0, 0);
                continue;
            }
            let motion = &state.kept[&(n as i64)]
                .record
                .analysis
                .as_ref()
                .unwrap()
                .segments;
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
                    .analyser
                    .live_faces
                    .hold(ALL_PARTS, vec![motion::Blob::default()], 0, 0);
            }
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            let count = history.len();
            state.render_frame(&settings, &mut history, count, n as i64, view);
        }
        assert!(!state.analyser.live_faces.holds_faces());
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
                state.analyser.live_faces.hold(ALL_PARTS, vec![face], 6, 0);
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
            assert!(!state.analyser.live_faces.holds_faces());
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
        assert_eq!(
            reds,
            [200, 50],
            "the face first, cut from the frame it was found in"
        );
    }

    /// As segcam 1.1 stacks one analysis, bottom to top: faces and parts, bodies, then motion. The
    /// labels keep each blob's number, and a reused face has its window already.
    #[test]
    fn one_analysis_stacks_as_in_segcam() {
        use motion::SegmentKind::{Body, EyeLeft, Face, Motion, Mouth};
        let pixels = vec![Argb::<u8>::default(); (W * H) as usize];
        let view = View::from_pixels(&pixels, W, H);
        let blob = |kind, reused_from| motion::Blob {
            bounds: crate::NormRect {
                x: 0.1,
                y: 0.1,
                w: 0.2,
                h: 0.2,
            },
            kind,
            reused_from,
            ..motion::Blob::default()
        };
        let motion = Segmentation {
            blobs: vec![
                blob(Motion, None),
                blob(Motion, None),
                blob(Face, None),
                blob(EyeLeft, None),
                blob(Mouth, None),
                blob(Body, None),
                blob(Face, Some((3, 0))),
            ],
            ..Segmentation::default()
        };
        let titles = |reused| {
            cut_frame_tiles(&motion, Some(&view), None, 4, (W, H), false, reused)
                .iter()
                .map(|t| t.title.strip_prefix("LO.FI.SCI.FI ").unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        let stack = [
            "face#4.3",
            "eye.L#4.4",
            "mouth#4.5",
            "body#4.6",
            "motion#4.1",
            "motion#4.2",
        ];
        assert_eq!(titles(false), stack);
        assert_eq!(
            titles(true)[4],
            "face#3.1",
            "a reused face stays with the faces"
        );
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
        on_gpu
    }

    fn drawn(state: &IncrementalRenderer) -> u64 {
        state.drawer().map_or(0, |d| d.drawn)
    }

    fn faults(state: &IncrementalRenderer) -> u64 {
        state.drawer().map_or(0, |d| d.faults)
    }

    /// Whether `b` is as near `a`, the CPU's picture, as SPEC F8 holds an 8-bit
    /// swarm drawn on the GPU to: every channel within 6 levels, and of 10,000
    /// pixels at most 300 off at all and 30 off by over a level (at least 4 of
    /// each, for small canvases).
    fn within_gpu_bound(a: &[Argb<u8>], b: &[Argb<u8>]) -> bool {
        let apart = |(p, q): (&Argb<u8>, &Argb<u8>)| {
            let channels = [
                (p.alpha, q.alpha),
                (p.red, q.red),
                (p.green, q.green),
                (p.blue, q.blue),
            ];
            channels.iter().map(|&(x, y)| x.abs_diff(y)).max()
        };
        let levels: Vec<u8> = a.iter().zip(b).filter_map(apart).collect();
        let over = |level: u8| levels.iter().filter(|&&l| l > level).count();
        let cap = |in_10_000: usize| (levels.len() * in_10_000 / 10_000).max(4);
        over(6) == 0 && over(1) <= cap(30) && over(0) <= cap(300)
    }

    /// A device lost between renders, as when a driver resets: renders are what the CPU draws
    /// until the slot has another device, then the GPU's again, on a fresh atlas with the trail
    /// still there. This loses it by `destroy()`, which goes another way through wgpu than a real
    /// loss: it shows this code's states, not what a driver does.
    #[test]
    fn lost_gpu_draws_on_the_cpu_and_reopens() {
        let _lock = gpu_test_lock();
        let choice = std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default();
        if gpu::open(&choice).is_err() && gpu::open("fallback").is_err() {
            eprintln!("no adapter: skipped");
            return;
        }
        let slot: &'static gpu::DeviceSlot = Box::leak(Box::new(gpu::DeviceSlot::new(move |_| {
            gpu::open(&choice).or_else(|_| gpu::open("fallback"))
        })));
        let device = |slot: &'static gpu::DeviceSlot| loop {
            match slot.now() {
                gpu::DeviceNow::Ready(dev) => return dev,
                gpu::DeviceNow::Waiting => std::thread::sleep(Duration::from_millis(5)),
                gpu::DeviceNow::Never => panic!("the slot gave up"),
            }
        };
        let first = device(slot);
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
        let mut on_gpu = IncrementalRenderer {
            test_slot: Some(slot),
            ..Default::default()
        };
        on_gpu.use_gpu(true);
        for n in 0..7 {
            render(&mut cpu, &frames, n);
            render(&mut on_gpu, &frames, n);
        }
        assert!(drawn(&on_gpu) > 0, "drawing on the GPU before the loss");
        assert_eq!(
            on_gpu.report().draw,
            Some(format!("GPU ({})", first.info.name))
        );
        first.device.destroy();

        // Right after the loss, and for as long as the slot has no device.
        let (a, b) = (
            render(&mut cpu, &frames, 7),
            render(&mut on_gpu, &frames, 7),
        );
        assert!(a == b, "the render after the loss isn't the CPU's");
        assert_eq!(on_gpu.report().draw.as_deref(), Some("CPU"));
        assert!(first.lost().is_some(), "the loss went unnoticed");
        assert!(
            on_gpu.drawer().is_none(),
            "still drawing on the lost device"
        );
        assert!(gpu::Drawer::new(first.clone()).is_none(), "a drawer on it");
        assert!(on_gpu.gpu_wanted, "the instance gave up on the GPU");

        let second = device(slot);
        assert!(
            !Arc::ptr_eq(&first, &second),
            "the slot kept the lost device"
        );
        assert!(second.lost().is_none());
        // An instance that was on the GPU comes back to it, and draws the whole trail again.
        let (a, b) = (
            render(&mut cpu, &frames, 8),
            render(&mut on_gpu, &frames, 8),
        );
        assert!(drawn(&on_gpu) > 0, "the instance stayed on the CPU");
        assert_eq!(
            on_gpu.report().draw,
            Some(format!("GPU ({})", second.info.name))
        );
        assert!(
            on_gpu.report().gpu_fault.is_none(),
            "the old fault still shows"
        );
        assert!(within_gpu_bound(&a, &b), "the picture after the reopen");
    }

    /// A first open that keeps failing leaves the CPU for good once its retries are spent: with no
    /// adapter the CPU is the machine's path. Until then renders draw on the CPU and keep asking.
    #[test]
    fn failed_first_open_leaves_the_cpu_for_good() {
        let slot: &'static gpu::DeviceSlot = Box::leak(Box::new(gpu::DeviceSlot::new(|_| {
            Err(gpu::NoGpu::NoAdapter("none here".into()))
        })));
        let mut on_cpu = IncrementalRenderer {
            test_slot: Some(slot),
            ..Default::default()
        };
        on_cpu.use_gpu(true);
        assert!(slot.status().is_err());
        render(&mut on_cpu, &crossing(), 0);
        assert!(on_cpu.gpu_wanted, "the first failure was final");
        let deadline = std::time::Instant::now() + gpu::FIRST_OPEN_TEST_DEADLINE;
        while !matches!(slot.next_first_open_try(), gpu::DeviceNow::Never) {
            assert!(
                std::time::Instant::now() < deadline,
                "the slot never gave up"
            );
        }
        render(&mut on_cpu, &crossing(), 0);
        assert!(!on_cpu.gpu_wanted);
    }

    /// Hashes are compared across renders that may run on different numbers of threads.
    #[test]
    fn frame_hash_is_the_rows_hashes_in_order_and_sees_every_pixel() {
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
            let rows = View::from_pixels(pixels, w, h).byte_rows();
            let hash = frame_hash(&rows);
            assert_eq!(hash, fold_hash(rows.iter().map(|row| row_hash(row))));
            hash
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
        let rows = View::from_pixels(&floats, w, h).byte_rows();
        assert_eq!(
            frame_hash(&rows),
            fold_hash(rows.iter().map(|row| row_hash(row)))
        );
    }

    /// A Float frame hashes as its bits. Equal bits are the same frame, whatever the values, NaN
    /// included. Equal values in other bits, 0 and -0 or NaNs with other payloads, are a changed
    /// frame: that costs the work again, and never keeps work for pixels that changed.
    #[test]
    fn frame_hash_of_float_rows_follows_their_bits() {
        let (w, h) = (5, 3);
        let mut floats = vec![
            Argb {
                alpha: 1.0f32,
                red: 0.0,
                green: 0.5,
                blue: 0.25,
            };
            (w * h) as usize
        ];
        let hash = |floats: &[Argb<f32>]| frame_hash(&View::from_pixels(floats, w, h).byte_rows());
        let zero = hash(&floats);
        floats[7].red = -0.0;
        assert_ne!(hash(&floats), zero, "-0 in place of 0");
        floats[7].red = f32::NAN;
        let nan = hash(&floats);
        assert_eq!(hash(&floats), nan, "the same NaN");
        floats[7].red = f32::from_bits(f32::NAN.to_bits() | 1);
        assert!(floats[7].red.is_nan());
        assert_ne!(hash(&floats), nan, "a NaN with another payload");
        floats[7].red = 0.0;
        assert_eq!(hash(&floats), zero);
    }

    /// A host's padding after each row is no part of the frame: a view's rows end at its width.
    #[test]
    fn frame_hash_leaves_out_row_padding() {
        let (w, h) = (3, 4);
        let row = w as usize * size_of::<Argb<u8>>();
        let stride = row + 8;
        let packed: Vec<u8> = (0..h as usize * row)
            .map(|i| (i * 13 % 256) as u8)
            .collect();
        let expected = frame_hash(&View::<u8>::new(&packed, w, h, row).byte_rows());
        let mut padded = vec![0u8; h as usize * stride];
        for fill in [0x00, 0xff] {
            for (y, line) in padded.chunks_mut(stride).enumerate() {
                line[..row].copy_from_slice(&packed[y * row..(y + 1) * row]);
                line[row..].fill(fill);
            }
            let rows = View::<u8>::new(&padded, w, h, stride).byte_rows();
            assert_eq!(frame_hash(&rows), expected, "padding {fill:#x}");
        }
    }

    /// A source whose [`FrameSource::with_frame`] is costly, and which holds frames in other bytes
    /// than it shows, as a host does: reversed rows, counting the frames it converts.
    struct ConvertingFrames<'a> {
        frames: &'a [Vec<Argb<u8>>],
        converted: Vec<usize>,
    }

    impl FrameSource for ConvertingFrames<'_> {
        fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
            self.converted.push(index);
            f(self
                .frames
                .get(index)
                .map(|p| Image::U8(View::from_pixels(p, W, H))));
        }

        fn with_frame_if_wanted(
            &mut self,
            index: usize,
            wanted: &mut dyn FnMut(&[&[u8]]) -> bool,
            f: &mut dyn FnMut(Option<Image<'_>>),
        ) {
            if let Some(pixels) = self.frames.get(index) {
                let mut rows = View::from_pixels(pixels, W, H).byte_rows();
                rows.reverse();
                if !wanted(&rows) {
                    return;
                }
            }
            self.with_frame(index, f);
        }
    }

    fn render_converting(
        state: &mut IncrementalRenderer,
        frames: &[Vec<Argb<u8>>],
        n: usize,
    ) -> (Vec<Argb<u8>>, Vec<usize>) {
        let s = swarm_settings();
        let start = n.saturating_sub(crate::previous_frames_needed(&s) as usize);
        let clip = &frames[start..=n];
        let mut pixels = vec![Argb::default(); (W * H) as usize];
        let mut source = ConvertingFrames {
            frames: clip,
            converted: Vec::new(),
        };
        let out = &mut ViewMut::from_pixels(&mut pixels, W, H);
        state.render_frame(&s, &mut source, clip.len(), n as i64, out);
        let converted = source.converted.iter().map(|i| i + start).collect();
        (pixels, converted)
    }

    /// A check with nothing changed converts only the frame it draws; a change anywhere in an
    /// earlier frame is still seen, and what it invalidates comes out as a fresh render's.
    #[test]
    fn check_converts_only_what_changed() {
        let mut frames = crossing();
        let n = frames.len() - 1;
        let mut state = IncrementalRenderer::default();
        let (first, _) = render_converting(&mut state, &frames, n);
        let (same, converted) = render_converting(&mut state, &frames, n);
        assert_eq!(same, first);
        assert_eq!(converted, [n], "nothing changed");

        for pixel in [0, frames[4].len() - 1, frames[4].len() / 2] {
            frames[4][pixel].alpha ^= 1;
            let (_, converted) = render_converting(&mut state, &frames, n);
            assert!(converted.contains(&4), "a change at pixel {pixel}");
            assert!(!converted.contains(&3), "frames before it are unchanged");
            let (again, converted) = render_converting(&mut state, &frames, n);
            assert_eq!(converted, [n], "the change is taken in");
            let mut fresh = IncrementalRenderer::default();
            let (cold, _) = render_converting(&mut fresh, &frames, n);
            assert_eq!(again, cold);
        }
    }

    /// A frame that goes missing from a check counts as empty, as in a cold render, and one that
    /// comes back gives what it gave before.
    #[test]
    fn check_sees_a_frame_go_missing_and_come_back() {
        let frames = crossing();
        let n = frames.len() - 1;
        let mut state = IncrementalRenderer::default();
        let (first, _) = render_converting(&mut state, &frames, n);

        struct Gap<'a>(ConvertingFrames<'a>, usize);
        impl FrameSource for Gap<'_> {
            fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
                if index == self.1 {
                    f(None);
                } else {
                    self.0.with_frame(index, f);
                }
            }
            fn with_frame_if_wanted(
                &mut self,
                index: usize,
                wanted: &mut dyn FnMut(&[&[u8]]) -> bool,
                f: &mut dyn FnMut(Option<Image<'_>>),
            ) {
                if index == self.1 {
                    f(None);
                } else {
                    self.0.with_frame_if_wanted(index, wanted, f);
                }
            }
        }
        let s = swarm_settings();
        let start = n.saturating_sub(crate::previous_frames_needed(&s) as usize);
        let clip = &frames[start..=n];
        let with_gap = |state: &mut IncrementalRenderer| {
            let mut pixels = vec![Argb::<u8>::default(); (W * H) as usize];
            let converting = ConvertingFrames {
                frames: clip,
                converted: Vec::new(),
            };
            let mut source = Gap(converting, 4 - start);
            let out = &mut ViewMut::from_pixels(&mut pixels, W, H);
            state.render_frame(&s, &mut source, clip.len(), n as i64, out);
            pixels
        };
        let after_gap = with_gap(&mut state);
        let cold = with_gap(&mut IncrementalRenderer::default());
        assert_eq!(after_gap, cold, "the missing frame counts as empty");

        let (back, _) = render_converting(&mut state, &frames, n);
        assert_eq!(back, first);
    }

    /// [`ConvertingFrames`] under a fetch budget, recording the earlier
    /// frames it hands out.
    struct Budgeted<'a> {
        frames: ConvertingFrames<'a>,
        budget: usize,
        earlier: BTreeSet<usize>,
    }

    impl Budgeted<'_> {
        fn record_earlier(&mut self, index: usize) {
            if index + 1 != self.frames.frames.len() {
                self.earlier.insert(index);
            }
        }
    }

    impl FrameSource for Budgeted<'_> {
        fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
            self.record_earlier(index);
            self.frames.with_frame(index, f);
        }

        fn with_frame_if_wanted(
            &mut self,
            index: usize,
            wanted: &mut dyn FnMut(&[&[u8]]) -> bool,
            f: &mut dyn FnMut(Option<Image<'_>>),
        ) {
            self.record_earlier(index);
            self.frames.with_frame_if_wanted(index, wanted, f);
        }

        fn fetch_budget(&self) -> Option<usize> {
            Some(self.budget)
        }
    }

    struct BudgetedRender {
        pixels: Vec<Argb<u8>>,
        /// Indices of the earlier frames handed out.
        earlier: BTreeSet<usize>,
        /// Indices of the frames converted, the current one included.
        converted: Vec<usize>,
    }

    /// A render of frame `n` under `s` and a fetch budget of `budget`.
    fn render_budgeted(
        state: &mut IncrementalRenderer,
        s: &RenderSettings,
        frames: &[Vec<Argb<u8>>],
        n: usize,
        budget: usize,
    ) -> BudgetedRender {
        let start = n.saturating_sub(crate::previous_frames_needed(s) as usize);
        let clip = &frames[start..=n];
        let mut pixels = vec![Argb::default(); (W * H) as usize];
        let mut source = Budgeted {
            frames: ConvertingFrames {
                frames: clip,
                converted: Vec::new(),
            },
            budget,
            earlier: BTreeSet::new(),
        };
        let out = &mut ViewMut::from_pixels(&mut pixels, W, H);
        state.render_frame(s, &mut source, clip.len(), n as i64, out);
        BudgetedRender {
            pixels,
            earlier: source.earlier,
            converted: source.frames.converted,
        }
    }

    /// The most renders it may take to read every earlier frame under `s` and
    /// a budget of `budget`.
    fn catch_up_renders(s: &RenderSettings, budget: usize) -> usize {
        (crate::previous_frames_needed(s) as usize).div_ceil(budget) + 1
    }

    fn invert_red(frames: &mut [Vec<Argb<u8>>]) {
        for pixels in frames {
            for p in pixels.iter_mut() {
                p.red = 255 - p.red;
            }
        }
    }

    #[test]
    fn budget_catches_up_from_cold() {
        let s = swarm_settings();
        let frames = crossing();
        let n = frames.len() - 1;
        let (cold, _) = render_converting(&mut IncrementalRenderer::default(), &frames, n);
        let mut state = IncrementalRenderer::default();
        // Only the first render reads past the budget, for its frame's
        // background of 4; once that is kept every render keeps to 2.
        let renders = (1..=catch_up_renders(&s, 2)).find(|&r| {
            let render = render_budgeted(&mut state, &s, &frames, n, 2);
            let limit = if r == 1 { 4 } else { 2 };
            assert!(
                render.earlier.len() <= limit,
                "render {r} read {} earlier frames",
                render.earlier.len()
            );
            render.pixels == cold
        });
        assert!(
            renders.is_some_and(|r| r > 1),
            "caught up after {renders:?} renders"
        );
        let again = render_budgeted(&mut state, &s, &frames, n, 2);
        assert!(again.pixels == cold, "stays caught up");
        assert!(again.earlier.len() <= 2, "read {}", again.earlier.len());
    }

    /// A seek at 4K in Resolve: a budget of 8 earlier float frames under a
    /// background of 16. The first render reads its own frame's background past
    /// the budget and analyses that frame, since Resolve may not render a still
    /// frame again.
    #[test]
    fn budget_seek_analyses_its_own_frame() {
        let s = RenderSettings {
            background_frames: 16,
            trail_frames: 12,
            ..swarm_settings()
        };
        let frames: Vec<_> = (0..80).map(|f| crossing()[f % 14].clone()).collect();
        let n = 60;
        let start = n - crate::previous_frames_needed(&s) as usize;
        let mut state = IncrementalRenderer::default();
        let render = render_budgeted(&mut state, &s, &frames, n, 8);
        let background: BTreeSet<usize> = (n - 16..n).map(|f| f - start).collect();
        assert!(
            render.earlier.is_superset(&background),
            "read {:?}",
            render.earlier
        );
        assert_eq!(render.earlier.len(), 16, "nothing past the background");
        assert!(
            state
                .kept
                .get(&(n as i64))
                .is_some_and(|kept| kept.record.analysis.is_some()),
            "the seek's own frame is analysed"
        );
    }

    /// A background more than twice the budget, as Background 64 at 4K: the
    /// seek reads the newest 8 of its 16 background frames, keeps the partial
    /// analysis as stale, and catches up to a fresh render once it has read
    /// the rest.
    #[test]
    fn budget_seek_redoes_a_partial_analysis() {
        let s = RenderSettings {
            background_frames: 16,
            trail_frames: 12,
            ..swarm_settings()
        };
        let frames: Vec<_> = (0..80).map(|f| crossing()[f % 14].clone()).collect();
        let n = 60;
        let start = n - crate::previous_frames_needed(&s) as usize;
        let mut state = IncrementalRenderer::default();
        let seek = render_budgeted(&mut state, &s, &frames, n, 4);
        let newest: BTreeSet<usize> = (n - 8..n).map(|f| f - start).collect();
        assert_eq!(seek.earlier, newest, "the newest 8 background frames");
        let kept = &state.kept[&(n as i64)];
        assert!(kept.record.analysis.is_some(), "frame {n} analysed");
        assert!(
            kept.stale,
            "frame {n} kept as done from part of its background"
        );
        let cold = render_budgeted(
            &mut IncrementalRenderer::default(),
            &s,
            &frames,
            n,
            usize::MAX,
        );
        let renders = catch_up_renders(&s, 4);
        let caught_up = (1..=renders).any(|_| {
            let render = render_budgeted(&mut state, &s, &frames, n, 4);
            assert!(render.earlier.len() <= 8, "read {}", render.earlier.len());
            render.pixels == cold.pixels
        });
        assert!(caught_up, "not caught up in {renders} renders");
        assert!(!state.kept[&(n as i64)].stale, "frame {n} still stale");
    }

    /// With a background several times the budget, an edit to every frame is
    /// caught up reading each earlier frame once.
    #[test]
    fn budget_reads_each_frame_once_after_edit() {
        let s = RenderSettings {
            background_frames: 16,
            trail_frames: 12,
            ..swarm_settings()
        };
        let mut frames: Vec<_> = (0..40).map(|f| crossing()[f % 14].clone()).collect();
        let n = frames.len() - 1;
        let earlier = (crate::previous_frames_needed(&s) as usize).min(n);
        let mut state = IncrementalRenderer::default();
        let settled = (0..8).any(|_| {
            render_budgeted(&mut state, &s, &frames, n, usize::MAX).converted == [earlier]
        });
        assert!(settled, "renders before the edit never settled");
        invert_red(&mut frames[..n]);
        let cold = render_budgeted(
            &mut IncrementalRenderer::default(),
            &s,
            &frames,
            n,
            usize::MAX,
        );
        let renders = catch_up_renders(&s, 2);
        let mut converted = vec![0; earlier];
        let caught_up = (1..=renders).any(|_| {
            let render = render_budgeted(&mut state, &s, &frames, n, 2);
            for index in render.converted.into_iter().filter(|&i| i < earlier) {
                converted[index] += 1;
            }
            render.pixels == cold.pixels
        });
        assert!(caught_up, "not caught up in {renders} renders");
        assert!(
            converted.iter().all(|&c| c <= 1),
            "converted more than once: {converted:?}"
        );
    }

    /// An edit to earlier frames found by a budgeted check leaves the motion
    /// that used them drawn until it is redone, so the trail doesn't empty.
    #[test]
    fn budget_draws_stale_motion_after_edit() {
        let s = swarm_settings();
        let mut frames = crossing();
        let n = frames.len() - 1;
        let mut state = IncrementalRenderer::default();
        let (before, _) = render_converting(&mut state, &frames, n);
        invert_red(&mut frames[..n]);
        let first = render_budgeted(&mut state, &s, &frames, n, 2);
        assert_eq!(first.earlier.len(), 2);
        assert!(
            first.pixels == before,
            "the first check after the edit drew what was kept"
        );
        let (cold, _) = render_converting(&mut IncrementalRenderer::default(), &frames, n);
        assert!(cold != before, "the edit changes the picture");
        let renders = catch_up_renders(&s, 2);
        let caught_up = (1..=renders).any(|_| {
            let render = render_budgeted(&mut state, &s, &frames, n, 2);
            assert!(
                render.earlier.len() <= 2,
                "read {} earlier frames",
                render.earlier.len()
            );
            render.pixels == cold
        });
        assert!(caught_up, "not caught up in {renders} renders");
    }

    /// An edit to the frame a settled held frame's checks just read waits for the
    /// rotation to come back, and the frames checked before then are read again:
    /// 27 renders here, where a jump takes at most 15.
    #[test]
    fn budget_catches_up_after_one_frame_edit() {
        let s = RenderSettings {
            background_frames: 16,
            trail_frames: 12,
            ..swarm_settings()
        };
        let mut frames: Vec<_> = (0..40).map(|f| crossing()[f % 14].clone()).collect();
        let n = frames.len() - 1;
        let start = n - (crate::previous_frames_needed(&s) as usize).min(n);
        let mut state = IncrementalRenderer::default();
        for _ in 0..2 * catch_up_renders(&s, 2) {
            render_budgeted(&mut state, &s, &frames, n, 2);
        }
        let checked = render_budgeted(&mut state, &s, &frames, n, 2).earlier;
        let just_checked = start + checked.last().expect("a settled render checks");
        invert_red(&mut frames[just_checked..=just_checked]);
        let cold = render_budgeted(
            &mut IncrementalRenderer::default(),
            &s,
            &frames,
            n,
            usize::MAX,
        );
        let renders =
            (1..=40).find(|_| render_budgeted(&mut state, &s, &frames, n, 2).pixels == cold.pixels);
        assert_eq!(renders, Some(27), "frame {just_checked} edited");
        let again = render_budgeted(&mut state, &s, &frames, n, 2);
        assert!(again.pixels == cold.pixels, "stays caught up");
    }

    /// Threshold reads no background, so a frame's change reaches no later frame.
    /// Held renders under a budget still catch up.
    #[test]
    fn budget_threshold_catches_up() {
        let s = RenderSettings {
            detection_mode: crate::DetectionMode::Threshold,
            ..swarm_settings()
        };
        let frames = crossing();
        let n = frames.len() - 1;
        let cold = render_budgeted(
            &mut IncrementalRenderer::default(),
            &s,
            &frames,
            n,
            usize::MAX,
        );
        let mut state = IncrementalRenderer::default();
        let renders = catch_up_renders(&s, 2);
        let caught_up = (0..renders)
            .any(|_| render_budgeted(&mut state, &s, &frames, n, 2).pixels == cold.pixels);
        assert!(caught_up, "not caught up in {renders} renders");
    }

    /// Frames that never change, fixed or not as the host says.
    struct Steps<'a> {
        frames: &'a [Vec<Argb<u8>>],
        fixed: bool,
        budget: Option<usize>,
    }

    impl FrameSource for Steps<'_> {
        fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
            f(Some(Image::U8(View::from_pixels(
                &self.frames[index],
                W,
                H,
            ))))
        }

        fn fixed(&self) -> bool {
            self.fixed
        }

        fn fetch_budget(&self) -> Option<usize> {
            self.budget
        }
    }

    /// Of single steps through a clip, one of them not taken as a stream, then
    /// a step over a frame, the renders that differ from a fresh render.
    fn steps_over_a_frame(budget: Option<usize>) -> Vec<usize> {
        let s = RenderSettings {
            background_frames: 16,
            trail_frames: 12,
            ..swarm_settings()
        };
        let frames: Vec<_> = (0..60).map(|f| crossing()[f % 14].clone()).collect();
        let needed = crate::previous_frames_needed(&s) as usize;
        let render = |state: &mut IncrementalRenderer, n: usize, fixed: bool, budget| {
            let clip = &frames[n.saturating_sub(needed)..=n];
            let mut source = Steps {
                frames: clip,
                fixed,
                budget,
            };
            let mut pixels = vec![Argb::<u8>::default(); (W * H) as usize];
            let out = &mut ViewMut::from_pixels(&mut pixels, W, H);
            state.render_frame(&s, &mut source, clip.len(), n as i64, out);
            pixels
        };
        let mut state = IncrementalRenderer::default();
        // Settled single steps, one of which the host doesn't take as a stream.
        render(&mut state, 0, false, budget);
        for n in 1..=40 {
            render(&mut state, n, n != 40, budget);
        }
        let mut differing = Vec::new();
        for n in [42, 43, 44, 45] {
            let got = render(&mut state, n, true, budget);
            if got != render(&mut IncrementalRenderer::default(), n, false, None) {
                differing.push(n);
            }
        }
        differing
    }

    /// A frame a stream of single steps went over is read before the frames the
    /// stream checks again after a step it didn't take as a stream, so the next
    /// renders match a fresh render as they do without a budget.
    #[test]
    fn budget_reads_a_stepped_over_frame_first() {
        let without = steps_over_a_frame(None);
        assert_eq!(without, Vec::<usize>::new(), "without a budget");
        assert_eq!(steps_over_a_frame(Some(4)), without);
    }

    /// Renders of a stream, after a settled pause, an edit to earlier frames
    /// and the stream's first render, until its frames match fresh renders and
    /// keep matching.
    fn stream_renders_after_edit(edit_every_frame: bool) -> usize {
        let s = RenderSettings {
            background_frames: 16,
            trail_frames: 12,
            ..swarm_settings()
        };
        let mut frames: Vec<_> = (0..80).map(|f| crossing()[f % 14].clone()).collect();
        let needed = crate::previous_frames_needed(&s) as usize;
        let render = |state: &mut IncrementalRenderer,
                      frames: &[Vec<Argb<u8>>],
                      n: usize,
                      fixed: bool,
                      budget: Option<usize>| {
            let clip = &frames[n.saturating_sub(needed)..=n];
            let mut source = Steps {
                frames: clip,
                fixed,
                budget,
            };
            let mut pixels = vec![Argb::<u8>::default(); (W * H) as usize];
            let out = &mut ViewMut::from_pixels(&mut pixels, W, H);
            state.render_frame(&s, &mut source, clip.len(), n as i64, out);
            pixels
        };
        let mut state = IncrementalRenderer::default();
        for _ in 0..3 * catch_up_renders(&s, 4) {
            render(&mut state, &frames, 30, false, Some(4));
        }
        match edit_every_frame {
            true => invert_red(&mut frames[..30]),
            false => invert_red(&mut frames[22..=22]),
        }
        // Play starts with a render that isn't a stream, then streams.
        render(&mut state, &frames, 30, false, Some(4));
        let mut matched_from = None;
        for n in 31..70 {
            let got = render(&mut state, &frames, n, true, Some(4));
            let fresh = render(&mut IncrementalRenderer::default(), &frames, n, false, None);
            match (got == fresh, matched_from) {
                (true, None) => matched_from = Some(n - 30),
                (false, Some(_)) => matched_from = None,
                _ => {}
            }
        }
        matched_from.expect("the stream matches fresh renders by its end")
    }

    /// A stream after an edit reads its stale and unchecked frames oldest first
    /// as a held frame does, so frames aren't rebuilt from background frames not
    /// checked yet: 6 renders after an edit to every earlier frame, 8 after an
    /// edit to one, with a budget of 4 for 28 earlier frames.
    #[test]
    fn budget_stream_after_edit_catches_up() {
        assert_eq!(
            stream_renders_after_edit(true),
            6,
            "every earlier frame edited"
        );
        assert_eq!(
            stream_renders_after_edit(false),
            8,
            "one earlier frame edited"
        );
    }

    /// Playback that dropped frames keeps no hashes of earlier frames. A held
    /// render after it still catches up within the bound.
    #[test]
    fn budget_catches_up_after_dropped_frames() {
        let frames = crossing();
        let s = swarm_settings();
        let needed = crate::previous_frames_needed(&s) as usize;
        let n = frames.len() - 1;
        let mut state = IncrementalRenderer::default();
        render_budgeted(&mut state, &s, &frames, n - 4, usize::MAX);
        for t in n - 3..=n {
            let clip = &frames[t - needed.min(t)..=t];
            let mut source = OnlyNewestFrame {
                frames: clip,
                reads: 0,
            };
            let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
            let view = &mut ViewMut::from_pixels(&mut out, W, H);
            state.render_frame(&s, &mut source, clip.len(), t as i64, view);
        }
        let (cold, _) = render_converting(&mut IncrementalRenderer::default(), &frames, n);
        let renders = catch_up_renders(&s, 2);
        let caught_up = (1..=renders).find(|_| {
            let render = render_budgeted(&mut state, &s, &frames, n, 2);
            assert!(
                render.earlier.len() <= 2,
                "read {} earlier frames",
                render.earlier.len()
            );
            render.pixels == cold
        });
        assert!(caught_up.is_some(), "not caught up in {renders} renders");
    }

    /// After a detection change on a frame that reuses its face source's faces, a
    /// render that leaves the source for later leaves the frame stale, so it is
    /// redone with the source's faces once the source is.
    #[test]
    fn budget_redoes_frame_whose_face_source_waits() {
        wait_for_models();
        let frames = crossing();
        let n = frames.len() - 1;
        let s = with_faces(ALL_PARTS);
        assert_ne!(
            face_source_frame(&s, n as i64, 0),
            n as i64,
            "frame {n} reuses faces"
        );
        let mut state = IncrementalRenderer::default();
        render_budgeted(&mut state, &s, &frames, n, usize::MAX);
        let changed = RenderSettings {
            motion: motion::MotionSettings {
                sensitivity: s.motion.sensitivity / 2.0,
                ..s.motion
            },
            ..s
        };
        render_budgeted(&mut state, &changed, &frames, n, 2);
        assert!(state.kept[&(n as i64)].stale, "frame {n} kept as done");
        let cold = render_budgeted(
            &mut IncrementalRenderer::default(),
            &changed,
            &frames,
            n,
            usize::MAX,
        );
        let renders = catch_up_renders(&changed, 2);
        let caught_up = (0..renders)
            .any(|_| render_budgeted(&mut state, &changed, &frames, n, 2).pixels == cold.pixels);
        assert!(caught_up, "not caught up in {renders} renders");
        assert!(
            !state.kept[&(n as i64)].stale,
            "frame {n} still stale once caught up"
        );
    }

    /// A check that skips converting earlier frames can't hide what a lost device took with it:
    /// the tiles are in RAM, so the new device's empty atlas fills from them and the whole trail
    /// draws. A change to an earlier frame is still found on the new device.
    #[test]
    fn check_after_a_new_device_draws_the_whole_trail_without_converting() {
        let _lock = gpu_test_lock();
        let choice = std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default();
        if gpu::open(&choice).is_err() && gpu::open("fallback").is_err() {
            eprintln!("no adapter: skipped");
            return;
        }
        let slot: &'static gpu::DeviceSlot = Box::leak(Box::new(gpu::DeviceSlot::new(move |_| {
            gpu::open(&choice).or_else(|_| gpu::open("fallback"))
        })));
        let device = || loop {
            match slot.now() {
                gpu::DeviceNow::Ready(dev) => return dev,
                gpu::DeviceNow::Waiting => std::thread::sleep(Duration::from_millis(5)),
                gpu::DeviceNow::Never => panic!("the slot gave up"),
            }
        };
        let first_device = device();
        let mut frames = crossing();
        let n = frames.len() - 1;
        let cold = |frames: &[Vec<Argb<u8>>]| {
            render_converting(&mut IncrementalRenderer::default(), frames, n).0
        };
        let mut state = IncrementalRenderer {
            test_slot: Some(slot),
            ..Default::default()
        };
        state.use_gpu(true);
        render_converting(&mut state, &frames, n);
        let (on_first, converted) = render_converting(&mut state, &frames, n);
        assert_eq!(converted, [n], "nothing changed");
        assert!(drawn(&state) > 0, "drawing on the GPU before the loss");
        assert!(within_gpu_bound(&cold(&frames), &on_first));

        first_device.device.destroy();
        let (on_cpu, converted) = render_converting(&mut state, &frames, n);
        assert_eq!(converted, [n], "the loss changed no frame");
        assert_eq!(state.report().draw.as_deref(), Some("CPU"));
        assert_eq!(on_cpu, cold(&frames), "the render after the loss");

        let second_device = device();
        assert!(!Arc::ptr_eq(&first_device, &second_device));
        let (on_second, converted) = render_converting(&mut state, &frames, n);
        assert_eq!(converted, [n], "the new device needs no earlier frame");
        assert!(drawn(&state) > 0, "the instance stayed on the CPU");
        assert_eq!(
            state.report().draw,
            Some(format!("GPU ({})", second_device.info.name))
        );
        assert!(within_gpu_bound(&cold(&frames), &on_second));

        frames[4][0].alpha ^= 1;
        let (_, converted) = render_converting(&mut state, &frames, n);
        assert!(converted.contains(&4), "the change on the new device");
        assert!(!converted.contains(&3), "frames before it are unchanged");
        let (edited, converted) = render_converting(&mut state, &frames, n);
        assert_eq!(converted, [n], "the change is taken in");
        assert!(within_gpu_bound(&cold(&frames), &edited));
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
        // Frame 0 has no windows yet, so it draws on the CPU.
        render(&mut on_gpu, &frames, 1);
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
            assert!(within_gpu_bound(&a, &b), "frame {n} is off the CPU's");
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
                .set_max_buffer_bytes(limit);
            let before = drawn(&on_gpu);
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_gpu_bound(&a, &b), "frame {n} is off the CPU's");
            if n == 7 {
                assert_eq!(a, b, "the faulted frame isn't the CPU's");
                assert_eq!(drawn(&on_gpu), before, "the faulted frame counts as drawn");
            } else {
                // Frame 0 has no windows yet, so it draws on the CPU.
                assert_eq!(
                    drawn(&on_gpu),
                    before + u64::from(n > 0),
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
            assert!(within_gpu_bound(&a, &b), "frame {n} is off the CPU's");
            if n != 7 {
                // Frame 0 has no windows yet, so it draws on the CPU.
                assert_eq!(
                    drawn(&on_gpu),
                    before + u64::from(n > 0),
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
        // Frame 0 has no windows and draws on the CPU. Frames 1, 3 and 5 fault; the
        // cooldown runs from 6 until the test ends it at 9.
        let cooldown_frames = 6..9;
        for n in 0..frames.len() {
            let drawer = on_gpu.drawer.as_mut().expect("on the GPU");
            let failing = n < 9 && n % 2 == 1 || cooldown_frames.contains(&n);
            drawer.set_max_buffer_bytes(if failing { 1024 } else { u64::MAX });
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
            assert!(within_gpu_bound(&a, &b), "frame {n} is off the CPU's");
            assert_eq!(in_cooldown, cooldown_frames.contains(&n), "frame {n}");
            if in_cooldown {
                assert_eq!(a, b, "cooldown frame {n} isn't the CPU's");
                assert_eq!(drawn(&on_gpu), before, "cooldown frame {n} drew on the GPU");
                assert_eq!(faults(&on_gpu), faulted, "cooldown frame {n} tried the GPU");
            } else if !failing && n > 0 {
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
            .filter_map(|(_, kept)| kept.record.tiles.as_ref())
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
        drawer.limit_tile_atlas(64, 64);
        let initial_texels = drawer.atlas_texels().0;
        let mut on_gpu = IncrementalRenderer::default();
        on_gpu.use_drawer(drawer);
        for n in 0..frames.len() {
            render(&mut on_gpu, &frames, n);
        }
        let drawer = on_gpu.drawer().expect("a drawer");
        let (drawer_address, grown_texels) =
            (drawer as *const gpu::Drawer, drawer.atlas_texels().0);
        assert!(grown_texels > initial_texels, "the atlas never grew");

        on_gpu.clear_history();
        let drawer = on_gpu.drawer().expect("a drawer");
        assert_eq!(drawer as *const gpu::Drawer, drawer_address);
        assert_eq!(drawer.atlas_texels(), (grown_texels, grown_texels));

        let drawn_before = drawn(&on_gpu);
        let mut cpu = IncrementalRenderer::default();
        for n in 0..frames.len() {
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_gpu_bound(&a, &b), "frame {n} is off the CPU's");
        }
        // Frame 0 has no windows, so it draws on the CPU.
        assert_eq!(drawn(&on_gpu), drawn_before + frames.len() as u64 - 1);
        assert_eq!(
            on_gpu.drawer().expect("a drawer").atlas_texels().0,
            grown_texels
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
        drawer.limit_tile_atlas(64, 2);
        let mut state = IncrementalRenderer::default();
        state.use_drawer(drawer);
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
        // Frame 0 has no windows, so it draws on the CPU.
        let with_windows = frames.len() as u64 - 1;
        assert_eq!(drawn(&state), with_windows, "a render left the GPU");
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
        drawer.limit_tile_atlas(64, 2);
        let mut state = IncrementalRenderer::default();
        state.use_drawer(drawer);
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
        assert!(within_gpu_bound(&fresh, &after));
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
        // Room for a few frames' tiles, and no more.
        drawer.limit_tile_atlas(64, 2);
        let mut on_gpu = IncrementalRenderer::default();
        on_gpu.use_drawer(drawer);
        for n in 0..frames.len() {
            let (a, b) = (
                render(&mut cpu, &frames, n),
                render(&mut on_gpu, &frames, n),
            );
            assert!(within_gpu_bound(&a, &b), "frame {n} is off the CPU's");
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
}
