//! Opt-in timing of CPU effect frames, for measuring a plugin inside a real host.
//! `VFX_FFGL_TIMING=1` waits for the host's queued GL work before each frame (so it isn't
//! counted) and for the upload after it, then every [`FRAMES_PER_LOG_LINE`] frames appends the median and
//! p95 of each stage, and the slowest frame, to `vfx-ffgl-timing.log` in the temp directory.
//! The waits cost the host some overlap, so leave it off outside measurements.

use std::io::Write;
use std::sync::OnceLock;
use std::time::Duration;

const FRAMES_PER_LOG_LINE: usize = 300;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VFX_FFGL_TIMING").is_some_and(|v| v != "0"))
}

/// Waits for the context's queued GL work.
///
/// # Safety
/// The host's GL context must be current.
pub unsafe fn finish_queued_gl() {
    // SAFETY: the caller has the context current.
    #[cfg(target_os = "windows")]
    unsafe {
        vfx_ffgl_common::gl::finish_shared()
    };
    #[cfg(target_os = "macos")]
    unsafe {
        vfx_ffgl_common::gl::finish()
    };
}

/// Per-frame milliseconds for each stage of a batch.
#[derive(Default)]
pub struct StageTimes {
    read_ms: Vec<f64>,
    render_ms: Vec<f64>,
    upload_ms: Vec<f64>,
}

impl StageTimes {
    /// Adds one frame's stages; at the end of a batch, writes it out with the frame size
    /// and the raw parameter values, so a log line says which mode it measured.
    pub fn record(
        &mut self,
        [read, render, upload]: [Duration; 3],
        size: (u32, u32),
        values: &[f32],
    ) {
        self.read_ms.push(read.as_secs_f64() * 1e3);
        self.render_ms.push(render.as_secs_f64() * 1e3);
        self.upload_ms.push(upload.as_secs_f64() * 1e3);
        if self.read_ms.len() < FRAMES_PER_LOG_LINE {
            return;
        }
        let total: Vec<f64> = (0..FRAMES_PER_LOG_LINE)
            .map(|i| self.read_ms[i] + self.render_ms[i] + self.upload_ms[i])
            .collect();
        let median_and_p95 = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            format!("{:.2}/{:.2}", v[v.len() / 2], v[v.len() * 95 / 100])
        };
        // A stall shows in neither the median nor the p95, so the worst frame and the count of
        // frames over 100 ms go in too.
        let max = total.iter().copied().fold(0.0, f64::max);
        let frames_over_100_ms = total.iter().filter(|&&t| t > 100.0).count();
        let values: Vec<String> = values.iter().map(|v| format!("{v:.2}")).collect();
        let line = format!(
            "{}x{} {FRAMES_PER_LOG_LINE} frames, ms median/p95: total {} (max {max:.2}, {frames_over_100_ms} over 100) read {} render {} upload {} values [{}]",
            size.0,
            size.1,
            median_and_p95(&total),
            median_and_p95(&self.read_ms),
            median_and_p95(&self.render_ms),
            median_and_p95(&self.upload_ms),
            values.join(" ")
        );
        self.read_ms.clear();
        self.render_ms.clear();
        self.upload_ms.clear();
        // Timing must never disturb a render, so write errors are ignored.
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(std::env::temp_dir().join("vfx-ffgl-timing.log"))
        {
            let _ = writeln!(file, "[pid {}] {line}", std::process::id());
        }
    }
}
