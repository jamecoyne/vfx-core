//! Opt-in timing of CPU effect renders, for measuring a plugin inside a real host.
//! `VFX_OFX_TIMING=1` appends one line per render to `vfx-ofx-timing.log` in the temp
//! directory, splitting it into the host's image fetches, hashing the host's rows, converting
//! them, the effect's work on earlier frames and on the current one, and writing the output.
//! Off, nothing is timed.

use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::image::{Depth, Image};

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VFX_OFX_TIMING").is_some_and(|v| v != "0"))
}

/// Counts and times of one render's stages.
#[derive(Default)]
pub struct RenderTimes {
    /// `clipGetImage` calls, including ones the host refused.
    pub fetches: u32,
    pub fetch: Duration,
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

impl RenderTimes {
    /// Adds the time since `start` to `stage`.
    pub fn add(stage: &mut Duration, start: Instant) {
        *stage += start.elapsed();
    }
}

/// What the line says about the render besides its stages.
pub struct RenderLine {
    pub frame: i64,
    pub frames_asked: usize,
    /// The host's stream position: earlier frames taken as unchanged, frames dropped.
    pub earlier_unchanged: bool,
    pub frames_dropped: bool,
    pub render: Duration,
    pub output: Duration,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Appends one render's line. The time not in any stage is the effect's own work between frames.
pub fn log(line: &RenderLine, t: &RenderTimes) {
    static LOADED: OnceLock<Instant> = OnceLock::new();
    let at = LOADED.get_or_init(Instant::now).elapsed().as_secs_f64();
    let staged = t.fetch + t.row_read + t.conversion + t.earlier_frames + t.current_frame;
    let between = line.render.saturating_sub(staged);
    let text = format!(
        "at {at:.3} s frame {} asked {} unchanged {} dropped {}: fetch {} {:.1} ms, rows {} {:.1} ms, \
         convert {} {:.1} ms, effect earlier {:.1} ms, current {:.1} ms, between {:.1} ms, \
         output {:.1} ms, total {:.1} ms, image {}",
        line.frame,
        line.frames_asked,
        u8::from(line.earlier_unchanged),
        u8::from(line.frames_dropped),
        t.fetches,
        ms(t.fetch),
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
    let path = std::env::temp_dir().join("vfx-ofx-timing.log");
    // Timing must never disturb a render, so write errors are ignored.
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{text}");
    }
}
