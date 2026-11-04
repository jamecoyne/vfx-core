//! The bottom bar: frames in from the source, the effect's renders and what they cost, new
//! frames out to the screen, then, on the desktop, this process's and the whole machine's CPU,
//! memory and GPU load, which browsers don't report. Sampled about once a second, so the bar
//! costs no frames.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex};
use std::time::Duration;

use egui::Ui;
#[cfg(not(target_arch = "wasm32"))]
use sysinfo::{Pid, ProcessesToUpdate, System};
use vfx_core::{DeviceFaults, Gpu, Report};
use web_time::Instant;

const INTERVAL: Duration = Duration::from_secs(1);

/// Running totals of what the preview did.
#[derive(Clone, Copy, Default)]
pub struct Counts {
    /// Renders the effect finished.
    pub rendered: u64,
    /// Those whose picture was drawn on the GPU rather than uploaded from RAM.
    pub drawn: u64,
    /// The time those renders took, for a CPU effect.
    pub render_time: Duration,
    /// Pictures presented in the window that differ from the one before.
    pub screen: u64,
    /// Frames that arrived from a video or camera, taken or not.
    pub arrived: u64,
}

impl Counts {
    pub fn since(self, earlier: Self) -> Self {
        Self {
            rendered: self.rendered - earlier.rendered,
            drawn: self.drawn - earlier.drawn,
            render_time: self.render_time - earlier.render_time,
            screen: self.screen - earlier.screen,
            arrived: self.arrived - earlier.arrived,
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
pub enum SourceKind {
    /// A test pattern, new every loop.
    #[default]
    Pattern,
    /// A dropped image.
    Still,
    /// A video or camera, at the rate its frames arrive.
    Video,
}

/// The effect and source of the moment, which the loop knows and the bar names.
#[derive(Clone, Copy)]
pub struct ShownEffect {
    pub effect: &'static str,
    /// The effect as `--effect` names it.
    pub slug: &'static str,
    pub on_cpu: bool,
    pub source: SourceKind,
    /// The effect output's size, in pixels.
    pub size: (u32, u32),
    /// The output is drawn on the GPU, so the screen rate counts drawn pictures, not ones
    /// compared with the last.
    pub screen_counts_drawn: bool,
}

/// What the bar shows, and on the desktop what `--stats` prints.
#[derive(Clone)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct Snapshot {
    pub counts: Counts,
    pub shown: ShownEffect,
    pub report: Report,
    /// A lost device or the errors the shared device has reported, in words.
    pub device_fault: Option<String>,
    /// Over the last second: frames in, renders and new frames a second, and a render's mean
    /// and longest time in ms (a CPU effect's).
    pub source: f32,
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
            shown: ShownEffect {
                effect: "",
                slug: "",
                on_cpu: false,
                source: SourceKind::default(),
                size: (0, 0),
                screen_counts_drawn: false,
            },
            report: Report::default(),
            device_fault: None,
            source: 0.0,
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
    last: Option<Instant>,
    last_counts: Counts,
    /// The longest render since the last sample.
    longest_render: Duration,
    snapshot: Snapshot,
    #[cfg(not(target_arch = "wasm32"))]
    load: MachineLoad,
}

/// The process's and machine's load, read on threads of their own.
#[cfg(not(target_arch = "wasm32"))]
struct MachineLoad {
    /// Written by the GPU thread (macOS only).
    gpu: Arc<Mutex<Option<f32>>>,
    /// Written by the usage thread. None until its first reading, and for good if the OS
    /// won't say what's running; the bar then keeps its last numbers.
    usage: Arc<Mutex<Option<Usage>>>,
}

/// This process's CPU and memory in MB, when the OS lists it, and the whole machine's CPU.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct Usage {
    process: Option<(f32, u64)>,
    system_cpu: f32,
}

#[cfg(not(target_arch = "wasm32"))]
impl MachineLoad {
    fn start() -> Self {
        let gpu = Arc::new(Mutex::new(None));
        #[cfg(target_os = "macos")]
        {
            let shared = Arc::clone(&gpu);
            // Its own thread, so querying the OS can't hold up the render loop.
            std::thread::spawn(move || {
                loop {
                    *shared.lock().unwrap() = gpu_percent();
                    std::thread::sleep(INTERVAL);
                }
            });
        }
        let usage = Arc::new(Mutex::new(None));
        let shared = Arc::clone(&usage);
        // Its own thread too: where Windows can't open its CPU counters, each CPU reading
        // reloads them first, which would stall the render loop.
        std::thread::spawn(move || {
            let Ok(mut sys) = System::new() else {
                return;
            };
            let pid = Pid::from_u32(std::process::id());
            loop {
                sys.refresh_cpu_usage();
                sys.refresh_memory();
                sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
                let process = sys
                    .process(pid)
                    .map(|process| (process.cpu_usage(), process.memory() / (1024 * 1024)));
                *shared.lock().unwrap() = Some(Usage {
                    process,
                    system_cpu: sys.global_cpu_usage(),
                });
                std::thread::sleep(INTERVAL);
            }
        });
        Self { gpu, usage }
    }

    fn read_into(&self, snapshot: &mut Snapshot) {
        if let Some(usage) = *self.usage.lock().unwrap() {
            if let Some((cpu, memory_mb)) = usage.process {
                snapshot.cpu = cpu;
                snapshot.memory_mb = memory_mb;
            }
            snapshot.system_cpu = usage.system_cpu;
        }
        snapshot.gpu = *self.gpu.lock().unwrap();
    }
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            last: None,
            last_counts: Counts::default(),
            longest_render: Duration::ZERO,
            snapshot: Snapshot::default(),
            #[cfg(not(target_arch = "wasm32"))]
            load: MachineLoad::start(),
        }
    }

    /// The latest numbers, with `counts` as the render loop keeps them and `render` the
    /// time of the render this loop finished, if any. The rates, CPU and memory are recomputed
    /// once a second.
    pub fn sample(
        &mut self,
        counts: Counts,
        render: Option<Duration>,
        shown: ShownEffect,
        report: &Report,
    ) -> Snapshot {
        if let Some(render) = render {
            self.longest_render = self.longest_render.max(render);
        }
        let second_passed = self.last.is_none_or(|t| t.elapsed() >= INTERVAL);
        if second_passed || shown.slug != self.snapshot.shown.slug {
            self.snapshot.report.clone_from(report);
            self.snapshot.report.readings.clear(); // the bar has no use for them
        }
        if second_passed {
            if let Some(last) = self.last {
                let seconds = last.elapsed().as_secs_f32();
                let delta = counts.since(self.last_counts);
                self.snapshot.source = delta.arrived as f32 / seconds;
                self.snapshot.rendered = delta.rendered as f32 / seconds;
                self.snapshot.screen = delta.screen as f32 / seconds;
                self.snapshot.render_ms = delta.render_ms().map(|ms| ms as f32);
                self.snapshot.worst_ms = (!self.longest_render.is_zero())
                    .then_some(self.longest_render.as_secs_f32() * 1000.0);
                self.longest_render = Duration::ZERO;
            }
            self.last = Some(Instant::now());
            self.snapshot.device_fault = device_fault_text(&Gpu::shared_faults());
            self.last_counts = counts;
            #[cfg(not(target_arch = "wasm32"))]
            self.load.read_into(&mut self.snapshot);
        }
        self.snapshot.counts = counts;
        self.snapshot.shown = shown;
        self.snapshot.last_render = render;
        self.snapshot.clone()
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

/// What a host's user should know about the device, if anything: its loss, or how many errors it
/// has reported and the first line of the last.
fn device_fault_text(faults: &DeviceFaults) -> Option<String> {
    if let Some(why) = faults.lost() {
        return Some(format!("GPU lost: {why}"));
    }
    let (count, last) = faults.errors();
    (count > 0).then(|| {
        format!(
            "GPU errors {count}: {}",
            last.lines().next().unwrap_or_default()
        )
    })
}

/// The source slot's text.
pub fn source_text(s: &Snapshot) -> String {
    match s.shown.source {
        SourceKind::Pattern => "source pattern".to_owned(),
        SourceKind::Still => "source still".to_owned(),
        SourceKind::Video => format!("source {:.0} fps", s.source),
    }
}

/// The effect's report as ", draw ..., faces ...", for the end of the effect slot.
pub fn report_text(report: &Report) -> String {
    let mut text = String::new();
    if let Some(draw) = &report.draw {
        text += &format!(", draw {draw}");
    }
    if let Some(faces) = &report.faces {
        text += &format!(", faces {faces}");
    }
    if report.out_of_gpu_memory {
        text += ", out of GPU memory";
    }
    text
}

/// The effect slot's text: which effect, where it runs, and its renders a second with
/// their cost when it has one.
pub fn effect_text(s: &Snapshot) -> String {
    let device = if s.shown.on_cpu { "CPU" } else { "GPU" };
    let mut text = format!(
        "{} on {device}{}: rendered {:.0} fps",
        s.shown.effect,
        report_text(&s.report),
        s.rendered
    );
    if let (Some(ms), Some(worst)) = (s.render_ms, s.worst_ms) {
        text += &format!(", {ms:.0} ms, worst {worst:.0} ms");
    }
    text
}

/// Draws the bar, source to screen, then the machine.
pub fn bar(ui: &mut Ui, s: &Snapshot) {
    ui.horizontal_wrapped(|ui| {
        if let Some(fault) = &s.device_fault {
            ui.colored_label(egui::Color32::from_rgb(230, 120, 90), fault);
            ui.separator();
        }
        ui.monospace(format!("{} × {}", s.shown.size.0, s.shown.size.1));
        ui.separator();
        ui.monospace(source_text(s));
        ui.separator();
        ui.monospace(effect_text(s));
        ui.separator();
        let drawn = if s.shown.screen_counts_drawn {
            " drawn"
        } else {
            ""
        };
        ui.monospace(format!("screen {:.0} fps{drawn}", s.screen));
        #[cfg(not(target_arch = "wasm32"))]
        machine_load(ui, s);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn machine_load(ui: &mut Ui, s: &Snapshot) {
    ui.separator();
    ui.monospace(format!("CPU {:.0}% (app {:.0}%)", s.system_cpu, s.cpu));
    ui.separator();
    ui.monospace(format!("Memory {} MB", s.memory_mb));
    ui.separator();
    match s.gpu {
        Some(gpu) => ui.monospace(format!("GPU {gpu:.0}%")),
        None => ui.label(egui::RichText::new("GPU n/a").monospace().weak()),
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_text_says_when_out_of_gpu_memory() {
        let mut report = Report::default();
        assert_eq!(report_text(&report), "");
        report.draw = Some("GPU".to_owned());
        report.out_of_gpu_memory = true;
        assert_eq!(report_text(&report), ", draw GPU, out of GPU memory");
    }

    #[test]
    fn device_error_reaches_the_fault_text() {
        let gpu = pollster::block_on(Gpu::headless()).unwrap();
        let faults = DeviceFaults::watch(&gpu.device);
        assert_eq!(device_fault_text(&faults), None);
        let zero_sized = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 0,
                height: 0,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let _ = gpu.device.create_texture(&zero_sized);
        let text = device_fault_text(&faults).expect("an error");
        assert!(text.starts_with("GPU errors 1: "), "{text}");
        assert_eq!(text.lines().count(), 1);
    }
}
