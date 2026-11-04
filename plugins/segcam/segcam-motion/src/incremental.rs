//! segcam Motion worked out incrementally: each frame's grid, motion and tiles are kept, with
//! a hash of the pixels they came from. Where earlier frames can change upstream (OpenFX), a
//! render reads every frame it depends on and works out again only what a changed frame feeds.
//! Where they are taken as unchanged ([`FrameSource::fixed`]: a host's own history of frames, as
//! in Resolume and the preview, or Resolve while it streams), it reads only the frames it hasn't
//! worked out, and the current one, so an edit in Resolve shows when that frame renders
//! or at the next check. Results come from each frame and the settings
//! alone, so render order doesn't change the output. Motion whose background is cut short by
//! the start of the frames, as at the start of a clip, is kept with the frame it starts at.
//!
//! A host that holds only its newest frames ([`FrameSource::held`]) can't hand the others over
//! again. What was cut from them stays as it was cut, so new settings reach new windows only,
//! as in segcam 1.1, and tiles the pile has stopped drawing drop their pixels.
use std::collections::{BTreeMap, BTreeSet};
use std::ops::RangeInclusive;

use vfx_core::cpu::FrameSource;
#[cfg(not(target_arch = "wasm32"))]
use vfx_core::wgpu;
use vfx_core::{Reading, Report};
use vfx_plugin::vfx_host::by_rows;

use crate::gpu::{self, AtlasAllocation, Drawer, Fault};
use crate::motion::{self, LumaGrid, Motion, MotionSettings, ThresholdSettings};
use crate::pixel::{Argb, Channel, Depth, View, ViewMut};
use crate::render::{
    AnalysisStep, Detect, FaceParts, OutputMode, Pile, PileEntry, PixelRect, RenderSettings,
    downsample, draw_output, fill_rect, frame_tiles, history_frames,
};
use crate::swarm::{self, SwarmMetrics, Tile};

#[derive(Clone, Copy, Debug, PartialEq)]
struct HistoryFormat {
    width: i32,
    height: i32,
    channel_bytes: usize,
    grid_width: i32,
}

/// What a frame's segments depend on besides the grids.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DetectionSettings {
    detect: Detect,
    motion: MotionSettings,
    background_frames: i32,
    every: i64,
    threshold: ThresholdSettings,
    faces: FaceParts,
}

/// What a frame's motion and tiles were worked out from besides its own pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AnalysisKey {
    detection: DetectionSettings,
    /// The first frame of its background: later than usual near the start of the frames.
    background_start: i64,
    /// For a frame that reuses the faces of an earlier analysis: that frame and its `from`,
    /// since the faces keep the IDs they had there.
    face_source: Option<(i64, i64)>,
}

#[derive(Default)]
pub struct IncrementalRenderer {
    history_format: Option<HistoryFormat>,
    grids: BTreeMap<i64, LumaGrid>,
    /// A hash of each frame's pixels, since edits upstream can leave the grid as it was.
    frame_hashes: BTreeMap<i64, u64>,
    motion: BTreeMap<i64, (AnalysisKey, Motion)>,
    /// Tiles by frame, with the basis and mirroring they were cut for.
    tiles: BTreeMap<i64, (AnalysisKey, bool, Vec<Tile>)>,
    /// Drawing on the GPU: a frame's tiles go into its atlas when a render first draws them
    /// there. The drawer is made by the first render that draws on the GPU once the device is
    /// open, as hosts make instances often.
    drawer: Option<Box<Drawer>>,
    atlas_allocations: BTreeMap<i64, AtlasAllocation>,
    gpu_wanted: bool,
    gpu_overlay: bool,
    /// The last fault told of, so a fault that repeats every render is told of once.
    last_reported_fault: Option<String>,
    /// From what share of the canvas in window pixels the swarm draws on the GPU.
    gpu_min_coverage: Option<f64>,
    /// The last render's draw hasn't come back from the GPU yet (browsers).
    unfinished: bool,
    /// `None` before the first render.
    drew_on_gpu: Option<bool>,
    faces_wanted: bool,
    readings: Vec<Reading>,
    /// Live renders act as if the face models were still loading.
    #[cfg(test)]
    simulate_models_loading: bool,
    /// How many times faces were looked for.
    #[cfg(test)]
    face_detection_calls: u32,
    /// A host's GPU frame the render may draw into instead of its output.
    #[cfg(not(target_arch = "wasm32"))]
    target: Option<TargetOffer>,
}

/// A host's frame on `device`, offered to one render ([`vfx_core::CpuEffect::render_to`]), and
/// whether the render drew into it.
#[cfg(not(target_arch = "wasm32"))]
struct TargetOffer {
    device: wgpu::Device,
    frame: vfx_core::Frame,
    drawn_into: bool,
}

/// Below these shares of the canvas in window pixels (each window with its shadow, overlaps
/// counted, as the CPU draws them) the CPU draws the swarm faster than the GPU's copy back. On an
/// RTX 4070 at 1080p the CPU took about 2.7 ms a canvas at 8 bits and 2.2 ms in float, the GPU
/// about 1.2 and 3.6 ms whatever the windows; these sit a little above where they cross.
#[cfg(not(target_arch = "wasm32"))]
const GPU_MIN_COVERAGE_8_BIT: f64 = 0.5;
#[cfg(not(target_arch = "wasm32"))]
const GPU_MIN_COVERAGE_DEEPER: f64 = 1.5;
// Browsers draw on the CPU on one thread: the GPU draws every swarm there.
#[cfg(target_arch = "wasm32")]
const GPU_MIN_COVERAGE_8_BIT: f64 = 0.0;
#[cfg(target_arch = "wasm32")]
const GPU_MIN_COVERAGE_DEEPER: f64 = 0.0;

impl IncrementalRenderer {
    /// Draws Swarm on the GPU where there is one, or always on the CPU. Off by default, so the
    /// library's renders are the CPU's; the plugin and the CLI turn it on.
    pub fn use_gpu(&mut self, on: bool) {
        self.gpu_wanted = on;
        if !on {
            self.set_drawer(None);
        }
    }

    /// The drawer, once a render has drawn on the GPU.
    pub fn drawer(&self) -> Option<&Drawer> {
        self.drawer.as_deref()
    }

    /// From what share of the canvas in window pixels the swarm draws on the GPU; 0 draws every
    /// render there. By default, what was measured to be faster at the canvas's depth.
    pub fn set_gpu_min_coverage(&mut self, min_coverage: f64) {
        self.gpu_min_coverage = Some(min_coverage);
    }

    /// Whether the last render is still waiting on the GPU (see [`vfx_core::CpuEffect::unfinished`]).
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
        self.atlas_allocations.clear();
    }

    /// Draws Overlay on the GPU too. Off by default: while frames go through host memory, the
    /// upload and readback cost more than the CPU's whole Overlay.
    pub fn gpu_overlay(&mut self, on: bool) {
        self.gpu_overlay = on;
    }

    /// Offers the next render `target` to draw into instead of its output, which a swarm drawn
    /// on the GPU of the same device takes; [`IncrementalRenderer::took_target`] says whether it did.
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn took_target(&mut self) -> bool {
        self.target.take().is_some_and(|offer| offer.drawn_into)
    }

    /// Whether this render can draw a swarm of `(w, h)` at `C` into the offered target: one
    /// on `drawer`'s device, of the canvas's size, for an 8-bit canvas.
    fn can_draw_into_target<C: Channel>(&self, drawer: &Drawer, (w, h): (i32, i32)) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        let fits = self.target.as_ref().is_some_and(|offer| {
            (offer.frame.width, offer.frame.height) == (w as u32, h as u32)
                && drawer.draws_on(&offer.device)
        });
        // No browser host offers a frame.
        #[cfg(target_arch = "wasm32")]
        let fits = {
            let _ = (drawer, w, h);
            false
        };
        size_of::<C>() == 1 && fits
    }

    /// Renders the last of `count` frames, frame `number`, into `out`. Reads the frames the
    /// render depends on (only those it lacks, from a fixed source) once each, oldest first,
    /// and works out again only what changed.
    pub fn render_frame<C: Depth>(
        &mut self,
        s: &RenderSettings,
        frames: &mut dyn FrameSource,
        count: usize,
        number: i64,
        out: &mut ViewMut<C>,
    ) {
        self.unfinished = false;
        self.drew_on_gpu = Some(false);
        if out.is_empty() {
            return;
        }
        // A live render can't wait for the face models, so it draws as with faces off until they
        // are up. Any other render may be exported, so finding faces waits for them.
        let without_faces = RenderSettings {
            faces: FaceParts::default(),
            ..*s
        };
        let s = if frames.live() && s.faces.any() && !self.face_models_ready() {
            &without_faces
        } else {
            s
        };
        if count == 0 {
            let all = PixelRect::new(0, 0, out.width, out.height);
            fill_rect(out, all, Argb::default());
            return;
        }
        let mode_uses_gpu =
            s.mode == OutputMode::Swarm || s.mode == OutputMode::Overlay && self.gpu_overlay;
        // While the device opens, renders draw on the CPU.
        if self.gpu_wanted && self.drawer.is_none() && mode_uses_gpu && !gpu::opening() {
            let drawer = gpu::device().and_then(Drawer::new).map(Box::new);
            self.gpu_wanted = drawer.is_some();
            self.set_drawer(drawer);
        }
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
        let swarm = s.mode == OutputMode::Swarm;
        let background = i64::from(history_frames(s));
        // What a changed frame feeds: the frames it is background to, and with faces the
        // analysis after.
        let invalidation_reach = background.max(i64::from(s.faces.any()) * s.analysis_step.every);
        let trail_len = trail_frame_count(s);
        let first = number - count as i64 + 1;
        let to_analyse = frames_to_analyse(s, number, first);

        // Frames the host didn't hand over count as empty, as in a cold render, but nothing
        // worked out from one is kept, so it is read again next time.
        let mut missing = BTreeSet::new();
        let oldest =
            (to_analyse.first().map_or(number, |&f| f.min(number)) - background).max(first);
        // A fixed source's earlier frames are as they were, so only the current frame is
        // checked: a different one means the source started over.
        let fixed = frames.fixed();
        let mut current_hash = None;
        if fixed {
            frames.with_frame((number - first) as usize, &mut |image| {
                current_hash = image.and_then(C::view).map(|v| frame_hash(&v));
            });
            if current_hash
                .is_none_or(|hash| self.frame_hashes.get(&number).is_some_and(|k| *k != hash))
            {
                self.clear_history();
            }
        }
        // The frames before this one whose pixels the source no longer has.
        let held = i64::try_from(frames.held()).unwrap_or(i64::MAX);
        let oldest_held_frame = if fixed {
            number.saturating_sub(held) + 1
        } else {
            i64::MIN
        };
        if swarm {
            self.pin_unreadable_frames(s, first, number, oldest_held_frame);
        }
        let frames_to_read = if fixed {
            self.frames_to_read(s, first, number, oldest_held_frame)
        } else {
            // Frames between analyses feed nothing.
            let feeding = |f: &i64| {
                f.rem_euclid(s.analysis_step.every) == 0 || *f == number || to_analyse.contains(f)
            };
            (oldest..=number).filter(feeding).collect()
        };
        for frame in frames_to_read {
            frames.with_frame((frame - first) as usize, &mut |image| {
                let view = image.and_then(C::view);
                match &view {
                    None => {
                        self.invalidate_frame(frame, invalidation_reach);
                        missing.insert(frame);
                    }
                    Some(_) if fixed && self.frame_hashes.contains_key(&frame) => {}
                    Some(v) => {
                        // After a pause every frame has a kept hash to check: those go on threads.
                        let hash = current_hash.filter(|_| frame == number).unwrap_or_else(|| {
                            if self.frame_hashes.contains_key(&frame) {
                                frame_hash_parallel(v)
                            } else {
                                frame_hash(v)
                            }
                        });
                        if self.frame_hashes.get(&frame) != Some(&hash) {
                            // A fixed source's frame read for the first time can't have
                            // changed what later frames worked out from it; elsewhere a
                            // frame that wasn't kept may have.
                            if !fixed {
                                self.invalidate_frame(frame, invalidation_reach);
                            }
                            self.grids.insert(frame, downsample(v, s.grid_width));
                            self.frame_hashes.insert(frame, hash);
                        }
                    }
                }
                let source_frame = face_source_frame(s, frame, first);
                let source_key = analysis_key(s, source_frame, first);
                let key = analysis_key(s, frame, first);
                // Outside the swarm a missing analysis isn't worked out: the frame showing it
                // works out its own instead.
                let wanted = to_analyse.contains(&frame) && (swarm || view.is_some())
                    || !swarm
                        && frame == number
                        && self.displayed_analysis_frame(s, number, first) == number;
                if wanted && !self.is_cached(frame, &key, s) {
                    let motion = match self.cached_motion(frame, &key) {
                        Some(motion) => motion.clone(),
                        None => {
                            let empty = LumaGrid::default();
                            let grid = self.grids.get(&frame).unwrap_or(&empty);
                            let mut motion = match s.detect {
                                Detect::Motion => {
                                    let history = (key.background_start..frame)
                                        .filter(|k| k.rem_euclid(s.analysis_step.every) == 0)
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
                            // The frame it reuses was worked out first, unless it is missing.
                            let source_motion = key
                                .face_source
                                .filter(|_| !missing.contains(&source_frame))
                                .and_then(|_| self.cached_motion(source_frame, &source_key));
                            if let Some(before) = source_motion {
                                motion.blobs.extend(reused_faces(before, source_frame));
                            } else if let Some(view) = view.as_ref().filter(|_| s.faces.any()) {
                                #[cfg(test)]
                                {
                                    self.face_detection_calls += 1;
                                }
                                let found = crate::faces::find(view, s.faces, grid.count());
                                motion.blobs.extend(found);
                            }
                            self.motion.insert(frame, (key, motion.clone()));
                            motion
                        }
                    };
                    if swarm {
                        // A trail no longer than the step draws no earlier analysis, so
                        // reused faces need tiles here.
                        let include_reused_faces = trail_len <= s.analysis_step.every;
                        let tiles = frame_tiles(
                            &motion,
                            view.as_ref(),
                            frame,
                            w,
                            h,
                            s.mirror,
                            include_reused_faces,
                        );
                        self.tiles.insert(frame, (key, s.mirror, tiles));
                        self.atlas_allocations.remove(&frame);
                    }
                }
                if frame == number {
                    self.draw(s, count, number, view, out);
                }
            });
        }
        for &frame in &missing {
            self.invalidate_frame(frame, invalidation_reach);
        }
        let retention_radius = background + trail_len + 2 * AnalysisStep::MAX_EVERY;
        self.retain_frames(number - retention_radius..=number + retention_radius);
        if swarm {
            self.release_undrawn_data(s, count, number, (w, h), oldest_held_frame);
        }
    }

    /// The frames before `oldest_held_frame` can't be cut again, so what they gave stays, whatever the
    /// settings are now; those that gave nothing yet stay empty.
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
            let motion = self.motion.entry(frame).or_insert((key, Motion::default()));
            motion.0 = key;
            let tiles = self
                .tiles
                .entry(frame)
                .or_insert((key, s.mirror, Vec::new()));
            (tiles.0, tiles.1) = (key, s.mirror);
        }
    }

    /// Drops what no later render reads from the frames before `oldest_held_frame`: a tile the pile no longer
    /// draws never comes back, as what buried it or pushed it out stays newer than it, and only
    /// the frame a render shows has its mask read.
    fn release_undrawn_data(
        &mut self,
        s: &RenderSettings,
        count: usize,
        number: i64,
        (width, height): (i32, i32),
        oldest_held_frame: i64,
    ) {
        if self.tiles.range(..oldest_held_frame).next().is_none() {
            return;
        }
        let shown: BTreeSet<(i64, usize)> =
            build_pile(&self.tiles, s, count, number, width, height)
                .shown_entries()
                .map(|p| (p.frame, p.index))
                .collect();
        for (&frame, (_, _, tiles)) in self.tiles.range_mut(..oldest_held_frame) {
            for (index, tile) in tiles.iter_mut().enumerate() {
                if !shown.contains(&(frame, index)) {
                    tile.hidden = true;
                    tile.pixels = Vec::new();
                    tile.bar = None;
                }
            }
        }
        let oldest_displayed_analysis = number - AnalysisStep::MAX_EVERY;
        for (_, (_, motion)) in self
            .motion
            .range_mut(..oldest_held_frame.min(oldest_displayed_analysis))
        {
            motion.mask = Vec::new();
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

    /// The frames from `first` a render of `number` has to read: those whose motion or tiles
    /// aren't kept, the analyses before them whose grids aren't, and `number`, which it draws
    /// and, outside the swarm, may have to analyse. None from before `oldest_held_frame`.
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
                .filter(|k| k.rem_euclid(s.analysis_step.every) == 0 && !self.grids.contains_key(k))
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
            let analysis_frame = self.displayed_analysis_frame(s, number, first);
            let motion =
                self.cached_motion(analysis_frame, &analysis_key(s, analysis_frame, first));
            let models_loaded = crate::faces::runs_on().is_some();
            crate::faces::readings(models_loaded, motion.map_or(&[], |m| &m.blobs))
        } else {
            Vec::new()
        };
    }

    pub fn report(&self) -> Report {
        let mut report = Report::default();
        report.draw = self.drew_on_gpu.map(|on_gpu| match &self.drawer {
            Some(drawer) if on_gpu => format!("GPU ({})", drawer.adapter().name),
            _ => "CPU".to_owned(),
        });
        report.faces = crate::faces::runs_on()
            .filter(|_| self.faces_wanted)
            .map(str::to_owned);
        report.readings = self.readings.clone();
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

    /// The GPU failed this render. A lost device is given up for good by this instance; after
    /// anything else the next render tries the GPU again, unless it failed a few lately.
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
        let pile = build_pile(&self.tiles, s, count, number, out.width, out.height);
        let empty = Motion::default();
        let analysis_frame = self.displayed_analysis_frame(s, number, first);
        let motion = self
            .cached_motion(analysis_frame, &analysis_key(s, analysis_frame, first))
            .unwrap_or(&empty);
        draw_output(out, s, frame, motion, analysis_frame, pile.shown());
    }

    /// The frame whose analysis frame `number` shows: the last analysed, or its own where that
    /// one is missing.
    fn displayed_analysis_frame(&self, s: &RenderSettings, number: i64, first: i64) -> i64 {
        let analysis = s.analysis_step.analysis_frame(number, first);
        match self.cached_motion(analysis, &analysis_key(s, analysis, first)) {
            Some(_) => analysis,
            None => number,
        }
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
            let empty = Motion::default();
            let analysis_frame = self.displayed_analysis_frame(s, number, first);
            let motion = self
                .cached_motion(analysis_frame, &analysis_key(s, analysis_frame, first))
                .unwrap_or(&empty);
            drawer.draw_overlay(out, view, &motion.blobs, number, analysis_frame, s)?;
            self.drew_on_gpu = Some(true);
            return Ok(());
        }
        if s.mode != OutputMode::Swarm {
            self.draw_cpu(s, count, number, frame, out);
            return Ok(());
        }
        // The swarm draws where its windows draw faster: the choice depends on this render's
        // windows alone, so a kept render draws what a cold one does. The shares below were
        // set by the cost of the copy back, which a host's frame to draw into takes away, so
        // then the GPU draws every swarm.
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
        let pile = build_pile(&self.tiles, s, count, number, w, h);
        let shown: Vec<PileEntry> = pile.shown_entries().collect();
        let window_pixel_count = window_pixels(shown.iter().map(|p| p.tile), &m, &shadow, w, h);
        let empty = Motion::default();
        let motion =
            cached_motion(&self.motion, number, &analysis_key(s, number, first)).unwrap_or(&empty);
        if window_pixel_count * 1000 < canvas_pixels * min_coverage_permille {
            draw_output(out, s, frame, motion, number, shown.iter().map(|p| p.tile));
            return Ok(());
        }
        // Only frames with a shown tile keep their place in the atlas. A full atlas draws this
        // render on the CPU; the frames the trail drops free it for the next.
        let wanted: BTreeSet<i64> = shown.iter().map(|p| p.frame).collect();
        self.atlas_allocations.retain(|f, _| wanted.contains(f));
        for f in wanted {
            if self.atlas_allocations.contains_key(&f) {
                continue;
            }
            let Some((_, _, tiles)) = self.tiles.get(&f) else {
                continue;
            };
            match drawer.upload_tiles(tiles, (w, h), &shadow)? {
                Some(allocation) => self.atlas_allocations.insert(f, allocation),
                None => {
                    draw_output(out, s, frame, motion, number, shown.iter().map(|p| p.tile));
                    return Ok(());
                }
            };
        }
        let windows = shown
            .iter()
            .map(|p| (p.tile, self.atlas_allocations[&p.frame].slots[p.index]));
        #[cfg(not(target_arch = "wasm32"))]
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

    /// Forgets every frame. Dropping the cuts returns their atlas space; the drawer stays, with
    /// its buffers, its atlas at the size it grew to, and its record of GPU faults and refusals.
    pub fn clear_history(&mut self) {
        self.grids.clear();
        self.frame_hashes.clear();
        self.motion.clear();
        self.tiles.clear();
        self.atlas_allocations.clear();
        self.unfinished = false;
        if let Some(drawer) = &mut self.drawer {
            drawer.drop_unfinished_draw();
        }
    }

    /// Whether `frame` has its motion and, for the swarm, its tiles.
    fn is_cached(&self, frame: i64, key: &AnalysisKey, s: &RenderSettings) -> bool {
        self.cached_motion(frame, key).is_some()
            && (s.mode != OutputMode::Swarm
                || cached_tiles(&self.tiles, frame, key, s.mirror).is_some())
    }

    fn cached_motion(&self, frame: i64, key: &AnalysisKey) -> Option<&Motion> {
        cached_motion(&self.motion, frame, key)
    }

    /// Drops what was kept for `frame`, and the motion and tiles of the frames whose
    /// background it is part of.
    fn invalidate_frame(&mut self, frame: i64, dependents: i64) {
        self.grids.remove(&frame);
        self.frame_hashes.remove(&frame);
        for later in frame..=frame + dependents {
            self.motion.remove(&later);
            self.tiles.remove(&later);
            self.atlas_allocations.remove(&later);
        }
    }

    fn retain_frames(&mut self, frames: RangeInclusive<i64>) {
        self.grids.retain(|f, _| frames.contains(f));
        self.frame_hashes.retain(|f, _| frames.contains(f));
        self.motion.retain(|f, _| frames.contains(f));
        self.tiles.retain(|f, _| frames.contains(f));
        self.atlas_allocations.retain(|f, _| frames.contains(f));
    }
}

/// The motion kept for `frame`, if it was worked out for `key`.
fn cached_motion<'a>(
    kept: &'a BTreeMap<i64, (AnalysisKey, Motion)>,
    frame: i64,
    key: &AnalysisKey,
) -> Option<&'a Motion> {
    kept.get(&frame).filter(|(b, _)| b == key).map(|(_, m)| m)
}

/// The tiles kept for `frame`, if they were cut for `key` and `mirror`.
fn cached_tiles<'a>(
    kept: &'a BTreeMap<i64, (AnalysisKey, bool, Vec<Tile>)>,
    frame: i64,
    key: &AnalysisKey,
    mirror: bool,
) -> Option<&'a [Tile]> {
    let (b, m, tiles) = kept.get(&frame)?;
    (b == key && *m == mirror).then_some(tiles.as_slice())
}

/// The pile of the `kept` tiles a swarm render of frame `number`, the last of `count`, draws
/// from on a `width` x `height` canvas; empty for other outputs.
fn build_pile<'a>(
    kept: &'a BTreeMap<i64, (AnalysisKey, bool, Vec<Tile>)>,
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

/// What `frame` is worked out from in a render whose frames start at `first`.
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
            detect: s.detect,
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
        let shown = s.analysis_step.analysis_frame(number, first);
        (shown..=shown, 1)
    };
    frames.filter(move |f| f.rem_euclid(every) == 0)
}

/// How many frames up to the current one are drawn from: the trail for the swarm.
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

/// The frames whose motion a render of `number` works out: those it uses, and the frames whose
/// faces they reuse.
fn frames_to_analyse(s: &RenderSettings, number: i64, first: i64) -> BTreeSet<i64> {
    analysis_frames_used(s, number, first)
        .flat_map(|f| [face_source_frame(s, f, first), f])
        .collect()
}

/// The faces in `motion`, frame `frame`'s, as the next analysis reuses them: with their IDs.
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

/// [`frame_hash`] on the threads hosts convert rows on.
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

    fn render(
        state: &mut IncrementalRenderer,
        frames: &[Vec<Argb<u8>>],
        n: usize,
    ) -> Vec<Argb<u8>> {
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
        state.render_frame(&s, &mut Frames(clip), clip.len(), n as i64, view);
        out
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
    fn a_live_render_finds_no_faces_during_the_load() {
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

    /// A render that may be exported doesn't draw without faces: it waits for the models.
    #[test]
    fn an_editing_render_looks_for_faces_during_the_load() {
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
    fn a_lost_gpu_falls_back_to_the_cpu() {
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

    /// Prints from both paths are compared when a host stops streaming.
    #[test]
    fn threaded_fingerprint_matches_serial() {
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
    fn an_error_leaves_the_device_usable() {
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
    fn leaving_the_gpu_keeps_the_cpu_output() {
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
        assert!(dev.lost().is_none(), "the device was given up");
    }

    /// A render the GPU can't take, here a canvas larger than the device binds, is drawn on
    /// the CPU byte for byte, and the next render is back on the GPU.
    #[test]
    fn a_fault_falls_back_for_one_render() {
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
        assert!(dev.lost().is_none(), "the device was given up");
    }

    /// A draw that isn't back in time is drawn on the CPU, and the next render is back on
    /// the GPU with a fresh readback buffer. Whether the first draw is in time depends on
    /// the GPU, so only what must hold either way is checked.
    #[test]
    fn a_slow_readback_falls_back_for_one_render() {
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
        assert!(dev.lost().is_none(), "the device was given up");
    }

    /// A GPU that fails every other render rests too: after a few faults among its last
    /// renders, the renders are the CPU's byte for byte and don't touch the GPU, and once the
    /// rest is over the GPU draws again.
    #[test]
    fn faults_close_together_rest_on_the_cpu() {
        let _lock = gpu_test_lock();
        let Some(dev) = own_device() else { return };
        let frames = crossing();
        let mut cpu = IncrementalRenderer::default();
        let mut on_gpu = on_device(&dev);
        // Frames 0, 2 and 4 fault; from 5 the instance rests, until the test ends it at 8.
        let resting_frames = 5..8;
        for n in 0..frames.len() {
            let drawer = on_gpu.drawer.as_mut().expect("on the GPU");
            let failing = n < 8 && n % 2 == 0 || resting_frames.contains(&n);
            drawer.set_max_binding_bytes(if failing { 1024 } else { u64::MAX });
            if n == resting_frames.end {
                assert!(
                    drawer.in_fault_cooldown(),
                    "still trying the GPU at frame {n}"
                );
                drawer.set_cooldown(Duration::ZERO);
            }
            let resting = drawer.in_fault_cooldown();
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
        assert_eq!(faults(&on_gpu), u64::from(gpu::MAX_RECENT_FAULTS));
        assert!(dev.lost().is_none(), "the device was given up");
    }

    /// A host holding only the current frame: tiles the pile stopped drawing keep no pixels.
    #[test]
    fn tiles_the_pile_stopped_drawing_drop_their_pixels() {
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
            .tiles
            .range(..last)
            .flat_map(|(_, (_, _, t))| t)
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

    /// An atlas too small for the trail draws the renders it can't hold on the CPU, and the
    /// GPU draws again once the trail moves on; the instance never leaves it.
    #[test]
    fn a_full_atlas_falls_back_for_one_render() {
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
        assert!(dev.lost().is_none(), "the device was given up");
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
        assert!(dev.lost().is_none(), "the device was given up");
    }
}
