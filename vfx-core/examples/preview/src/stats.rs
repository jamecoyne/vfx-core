//! The bottom bar: frame rate, this process's and the whole machine's CPU, memory, and GPU
//! load. Sampled about once a second, so the bar costs no frames.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use egui::{RichText, Ui};
use sysinfo::{Pid, ProcessesToUpdate, System};

const INTERVAL: Duration = Duration::from_secs(1);

/// What the bar shows.
#[derive(Clone, Copy, Default)]
pub struct Snapshot {
    pub fps: f32,
    /// The effects output's size, in pixels.
    pub size: (u32, u32),
    /// This process, as a percentage of one core (so it can pass 100 on several).
    pub cpu: f32,
    pub memory_mb: u64,
    /// The whole machine, as a percentage of all its cores.
    pub system_cpu: f32,
    /// The GPU's busy percentage, where the OS reports it.
    pub gpu: Option<f32>,
}

pub struct Sampler {
    sys: Option<System>,
    pid: Pid,
    last: Option<Instant>,
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
            snapshot: Snapshot::default(),
            gpu,
        }
    }

    /// The latest numbers, with `fps` as the render loop measures it. CPU and memory are
    /// only read again once a second.
    pub fn sample(&mut self, fps: f32, size: (u32, u32)) -> Snapshot {
        if self.last.is_none_or(|t| t.elapsed() >= INTERVAL) {
            self.last = Some(Instant::now());
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
        self.snapshot.fps = fps;
        self.snapshot.size = size;
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

/// Draws the bar.
pub fn bar(ui: &mut Ui, s: &Snapshot) {
    ui.horizontal(|ui| {
        ui.monospace(format!("{} × {}", s.size.0, s.size.1));
        ui.separator();
        ui.monospace(format!("{:.0} fps", s.fps));
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
