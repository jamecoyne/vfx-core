//! Opt-in timing of CPU effect renders, for measuring a plugin inside a real host.
//! `VFX_OFX_TIMING=1` writes one line per render to `vfx-ofx-timing.log` in the temp
//! directory, splitting it into the host's image fetches and releases, hashing the host's rows,
//! converting them, the effect's work on earlier frames and on the current one, and writing the
//! output. Off, nothing is timed.
//!
//! The log is opened once, when the host loads the plugin, and emptied then; it stops growing at
//! [`CAP_BYTES`], since a show runs for hours. A render never waits for it: a line that finds
//! another render writing is dropped.

use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::image::{Depth, Image};

/// Where the log stops growing.
const CAP_BYTES: u64 = 8 << 20;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VFX_OFX_TIMING").is_some_and(|v| v == "1"))
}

struct Log {
    file: File,
    opened: Instant,
    written: u64,
}

/// `None` when timing is off or the file couldn't be created.
static LOG: OnceLock<Mutex<Option<Log>>> = OnceLock::new();

/// Creates the log, emptying one an earlier run left. Called on the host's load action, so that
/// no render thread opens a file.
pub fn open_log() {
    LOG.get_or_init(|| {
        let file = enabled()
            .then(|| File::create(std::env::temp_dir().join("vfx-ofx-timing.log")).ok())
            .flatten();
        Mutex::new(file.map(|file| Log {
            file,
            opened: Instant::now(),
            written: 0,
        }))
    });
}

/// The time now while timing, to measure a stage from.
pub fn start(times: &Option<RenderTimes>) -> Option<Instant> {
    times.is_some().then(Instant::now)
}

/// Adds the time since `started` to `stage`.
pub fn add(stage: &mut Duration, started: Option<Instant>) {
    if let Some(started) = started {
        *stage += started.elapsed();
    }
}

/// Counts and times of one render's stages.
#[derive(Default)]
pub struct RenderTimes {
    /// `clipGetImage` calls, including ones the host refused.
    pub fetches: u32,
    pub fetch: Duration,
    /// `clipReleaseImage` calls: a host may hold each image it gave out until it gets it back.
    pub releases: u32,
    pub release: Duration,
    /// Frames whose host rows the effect looked at before deciding whether to convert them.
    pub row_reads: u32,
    pub row_read: Duration,
    pub conversions: u32,
    pub conversion: Duration,
    /// The effect's work inside its frame callbacks, for earlier frames and for the current one.
    pub earlier_frames: Duration,
    pub current_frame: Duration,
    /// The first image the host handed over, which says what each fetch costs it in memory.
    pub fetched_image: Option<ImageShape>,
}

/// An image's depth, size and row stride.
#[derive(Clone, Copy)]
pub struct ImageShape {
    pub depth: Depth,
    pub width: usize,
    pub height: usize,
    pub row_bytes: isize,
}

impl ImageShape {
    pub fn of(image: &Image) -> Self {
        Self {
            depth: image.depth,
            width: image.bounds.width(),
            height: image.bounds.height(),
            row_bytes: image.row_bytes,
        }
    }
}

/// What the line says about the render besides its stages.
pub struct RenderLine {
    pub frame: i64,
    pub frames_asked: usize,
    /// The host's stream position: earlier frames taken as unchanged, frames dropped.
    pub earlier_unchanged: bool,
    pub frames_dropped: bool,
    /// The effect's render, including the release of the last image it fetched.
    pub render: Duration,
    pub output: Duration,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Writes one render's line, stamped with the seconds since the log opened. The time not in any
/// stage is the effect's own work between frames.
pub fn log(line: &RenderLine, t: &RenderTimes) {
    let Some(Ok(mut log)) = LOG.get().map(Mutex::try_lock) else {
        return;
    };
    let Some(log) = log.as_mut() else {
        return;
    };
    if log.written >= CAP_BYTES {
        return;
    }
    let staged =
        t.fetch + t.release + t.row_read + t.conversion + t.earlier_frames + t.current_frame;
    let between = line.render.saturating_sub(staged);
    let text = format!(
        "at {:.3} s frame {} asked {} unchanged {} dropped {}: fetch {} {:.1} ms, \
         release {} {:.1} ms, rows {} {:.1} ms, convert {} {:.1} ms, effect earlier {:.1} ms, \
         current {:.1} ms, between {:.1} ms, output {:.1} ms, total {:.1} ms, image {}\n",
        log.opened.elapsed().as_secs_f64(),
        line.frame,
        line.frames_asked,
        u8::from(line.earlier_unchanged),
        u8::from(line.frames_dropped),
        t.fetches,
        ms(t.fetch),
        t.releases,
        ms(t.release),
        t.row_reads,
        ms(t.row_read),
        t.conversions,
        ms(t.conversion),
        ms(t.earlier_frames),
        ms(t.current_frame),
        ms(between),
        ms(line.output),
        ms(line.render + line.output),
        t.fetched_image.map_or("none".to_string(), |i| format!(
            "{:?} {}x{} row {} B",
            i.depth, i.width, i.height, i.row_bytes
        )),
    );
    // One write per line, so lines from concurrent renders never mix. Timing must never disturb
    // a render, so a failed write is ignored.
    if log.file.write_all(text.as_bytes()).is_ok() {
        log.written += text.len() as u64;
    }
}
