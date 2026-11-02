//! The bottom bar: frames in from the source, the effect's renders and what they cost, new
//! frames out to the screen, then this process's and the whole machine's CPU, memory and
//! GPU load. Sampled about once a second, so the bar costs no frames.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use egui::{RichText, Ui};
use sysinfo::{Pid, ProcessesToUpdate, System};

const INTERVAL: Duration = Duration::from_secs(1);

/// Running totals of what the preview did.
#[derive(Clone, Copy, Default)]
pub struct Counts {
    /// Renders the effect finished.
    pub rendered: u64,
    /// The time those renders took, for a CPU effect.
    pub render_time: Duration,
    /// Pictures presented in the window that differ from the one before.
    pub screen: u64,
}

impl Counts {
    pub fn since(self, earlier: Self) -> Self {
        Self {
            rendered: self.rendered - earlier.rendered,
            render_time: self.render_time - earlier.render_time,
            screen: self.screen - earlier.screen,
        }
    }

    /// The mean render in ms, when there were renders with a time.
    pub fn render_ms(self) -> Option<f64> {
        (self.rendered > 0 && !self.render_time.is_zero())
            .then(|| self.render_time.as_secs_f64() * 1000.0 / self.rendered as f64)
    }
}

/// Where the frames come from.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub enum SourceRate {
    /// A test pattern, new every loop.
    #[default]
    Pattern,
    /// A dropped image.
    Still,
    /// A video or camera, at its frame rate.
    Fps(f64),
}

/// The effect and source of the moment, which the loop knows and the bar names.
#[derive(Clone, Copy)]
pub struct What {
    pub effect: &'static str,
    /// The effect as `--effect` names it.
    pub slug: &'static str,
    pub on_cpu: bool,
    pub source: SourceRate,
    /// The effect output's size, in pixels.
    pub size: (u32, u32),
}

/// What the bar shows.
#[derive(Clone, Copy)]
pub struct Snapshot {
    pub counts: Counts,
    pub what: What,
    /// Over the last second: renders and new frames a second, and a render's mean and
    /// longest time in ms (a CPU effect's).
    pub rendered: f32,
    pub screen: f32,
    pub render_ms: Option<f32>,
    pub worst_ms: Option<f32>,
    /// The last loop's render time, if it finished one.
    pub last_render: Option<Duration>,
    /// This process, as a percentage of one core (so it can pass 100 on several).
    pub cpu: f32,
    pub memory_mb: u64,
    /// The whole machine, as a percentage of all its cores.
    pub system_cpu: f32,
    /// The GPU's busy percentage, where the OS reports it.
    pub gpu: Option<f32>,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            counts: Counts::default(),
            what: What {
                effect: "",
                slug: "",
                on_cpu: false,
                source: SourceRate::default(),
                size: (0, 0),
            },
            rendered: 0.0,
            screen: 0.0,
            render_ms: None,
            worst_ms: None,
            last_render: None,
            cpu: 0.0,
            memory_mb: 0,
            system_cpu: 0.0,
            gpu: None,
        }
    }
}

pub struct Sampler {
    sys: Option<System>,
    pid: Pid,
    last: Option<Instant>,
    last_counts: Counts,
    /// The longest render since the last sample.
    worst: Duration,
    snapshot: Snapshot,
    /// Written by the GPU thread (macOS only).
    gpu: Arc<Mutex<Option<f32>>>,
}

impl Sampler {
    pub fn new() -> Self {
        let gpu = Arc::new(Mutex::new(None));
        #[cfg(target_os = "macos")]
        {
            let shared = Arc::clone(&gpu);
            // Its own thread, since asking the OS for GPU load takes a few milliseconds.
            std::thread::spawn(move || {
                loop {
                    *shared.lock().unwrap() = gpu_percent();
                    std::thread::sleep(INTERVAL);
                }
            });
        }
        Self {
            // None if the OS won't say what's running; the bar then keeps its last numbers.
            sys: System::new().ok(),
            pid: Pid::from_u32(std::process::id()),
            last: None,
            last_counts: Counts::default(),
            worst: Duration::ZERO,
            snapshot: Snapshot::default(),
            gpu,
        }
    }

    /// The latest numbers, with `counts` as the render loop keeps them and `render` the
    /// time of the render this loop finished, if any. The rates, CPU and memory are only
    /// worked out again once a second.
    pub fn sample(&mut self, counts: Counts, render: Option<Duration>, what: What) -> Snapshot {
        if let Some(render) = render {
            self.worst = self.worst.max(render);
        }
        if self.last.is_none_or(|t| t.elapsed() >= INTERVAL) {
            if let Some(last) = self.last {
                let seconds = last.elapsed().as_secs_f32();
                let done = counts.since(self.last_counts);
                self.snapshot.rendered = done.rendered as f32 / seconds;
                self.snapshot.screen = done.screen as f32 / seconds;
                self.snapshot.render_ms = done.render_ms().map(|ms| ms as f32);
                self.snapshot.worst_ms =
                    (!self.worst.is_zero()).then_some(self.worst.as_secs_f32() * 1000.0);
                self.worst = Duration::ZERO;
            }
            self.last = Some(Instant::now());
            self.last_counts = counts;
            if let Some(sys) = self.sys.as_mut() {
                sys.refresh_cpu_usage();
                sys.refresh_memory();
                sys.refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);
                if let Some(process) = sys.process(self.pid) {
                    self.snapshot.cpu = process.cpu_usage();
                    self.snapshot.memory_mb = process.memory() / (1024 * 1024);
                }
                self.snapshot.system_cpu = sys.global_cpu_usage();
            }
            self.snapshot.gpu = *self.gpu.lock().unwrap();
        }
        self.snapshot.counts = counts;
        self.snapshot.what = what;
        self.snapshot.last_render = render;
        self.snapshot
    }
}

/// The GPU's busy percentage from the IOAccelerator's performance statistics.
#[cfg(target_os = "macos")]
fn gpu_percent() -> Option<f32> {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-w0", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let rest = text.split("\"Device Utilization %\"=").nth(1)?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// The source slot's text.
pub fn source_text(source: SourceRate) -> String {
    match source {
        SourceRate::Pattern => "source pattern".to_owned(),
        SourceRate::Still => "source still".to_owned(),
        SourceRate::Fps(fps) => format!("source {fps:.0} fps"),
    }
}

/// The effect slot's text: which effect, where it runs, and its renders a second with
/// their cost when it has one.
pub fn effect_text(s: &Snapshot) -> String {
    let where_ = if s.what.on_cpu { "CPU" } else { "GPU" };
    let mut text = format!(
        "{} on {where_}: rendered {:.0} fps",
        s.what.effect, s.rendered
    );
    if let (Some(ms), Some(worst)) = (s.render_ms, s.worst_ms) {
        text += &format!(", {ms:.0} ms, worst {worst:.0} ms");
    }
    text
}

/// Draws the bar, source to screen, then the machine.
pub fn bar(ui: &mut Ui, s: &Snapshot) {
    ui.horizontal_wrapped(|ui| {
        ui.monospace(format!("{} × {}", s.what.size.0, s.what.size.1));
        ui.separator();
        ui.monospace(source_text(s.what.source));
        ui.separator();
        ui.monospace(effect_text(s));
        ui.separator();
        ui.monospace(format!("screen {:.0} fps", s.screen));
        ui.separator();
        ui.monospace(format!("CPU {:.0}% (app {:.0}%)", s.system_cpu, s.cpu));
        ui.separator();
        ui.monospace(format!("Memory {} MB", s.memory_mb));
        ui.separator();
        match s.gpu {
            Some(gpu) => ui.monospace(format!("GPU {gpu:.0}%")),
            None => ui.label(RichText::new("GPU n/a").monospace().weak()),
        };
    });
}
