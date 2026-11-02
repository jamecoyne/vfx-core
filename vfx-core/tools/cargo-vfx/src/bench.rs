//! `cargo vfx bench`: the offline rows of `plugins/segcam/BENCHMARK.md`, on any clip. Rows go to
//! stdout, ready to paste; progress goes to stderr.

use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use crate::{env_or, run as run_cmd};

const USAGE: &str = "usage: cargo vfx bench CLIP [--preset NAME]... [--depth 8|float] [--runs N] [--cpu] [--reference DIR]
  --preset     overlay, overlay-faces, show, show-faces, 1.1 (the first four)
  --depth      one depth (both)
  --runs       runs pooled per row (2)
  --cpu        the CLI's --cpu path: every frame rendered fresh on the CPU
  --reference  an earlier run's folder of frame hashes (target/bench/COMMIT), for the Matches column";

const FRAMES: u32 = 300;
/// The first frames are the clip start, the rest steady state.
const SPLIT: usize = 40;

const PRESETS: [(&str, &[&str]); 5] = [
    ("overlay", &["--mode", "overlay", "--background", "16"]),
    (
        "overlay faces",
        &[
            "--mode",
            "overlay",
            "--background",
            "16",
            "--face",
            "--eyes",
            "--mouth",
            "--body",
        ],
    ),
    (
        "show",
        &[
            "--mode",
            "swarm",
            "--background",
            "16",
            "--trail",
            "24",
            "--max-tiles",
            "100",
            "--swarm-background",
            "transparent",
        ],
    ),
    (
        "show faces",
        &[
            "--mode",
            "swarm",
            "--background",
            "16",
            "--trail",
            "24",
            "--max-tiles",
            "100",
            "--swarm-background",
            "transparent",
            "--face",
            "--eyes",
            "--mouth",
            "--body",
        ],
    ),
    ("1.1 look", &["--preset", "1.1", "--keep"]),
];
const DEPTHS: [(&str, &[&str]); 2] = [("8 bit", &[]), ("float", &["--float"])];

struct Options {
    clip: String,
    presets: Vec<usize>,
    depths: Vec<usize>,
    runs: usize,
    cpu: bool,
    reference: Option<String>,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        clip: String::new(),
        presets: Vec::new(),
        depths: vec![0, 1],
        runs: 2,
        cpu: false,
        reference: None,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or(format!("{arg} needs a value\n{USAGE}"));
        match arg.as_str() {
            "--preset" => {
                let name = value()?.replace('-', " ");
                let found = PRESETS
                    .iter()
                    .position(|(p, _)| *p == name || p.strip_suffix(" look") == Some(&name));
                o.presets
                    .push(found.ok_or(format!("unknown preset `{name}`\n{USAGE}"))?);
            }
            "--depth" => {
                o.depths = match value()?.as_str() {
                    "8" => vec![0],
                    "float" => vec![1],
                    other => return Err(format!("unknown depth `{other}`\n{USAGE}")),
                }
            }
            "--runs" => {
                o.runs = value()?
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or(format!("--runs needs a count\n{USAGE}"))?
            }
            "--cpu" => o.cpu = true,
            "--reference" => o.reference = Some(value()?.clone()),
            clip if !clip.starts_with('-') && o.clip.is_empty() => o.clip = clip.to_owned(),
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    if o.clip.is_empty() {
        return Err(USAGE.into());
    }
    if o.presets.is_empty() {
        o.presets = vec![0, 1, 2, 3];
    }
    Ok(o)
}

pub fn run(args: &[String], root: &Path, target: &Path) -> Result<(), String> {
    let o = parse(args)?;
    let (width, height, rate) = probe(&o.clip)?;
    run_cmd(
        Command::new(env_or("CARGO", "cargo"))
            .current_dir(root)
            .args([
                "build",
                "--release",
                "--quiet",
                "--package",
                "segcam-motion",
                "--bin",
                "segcam-motion",
            ]),
    )?;
    let cli = target
        .join("release")
        .join(format!("segcam-motion{}", std::env::consts::EXE_SUFFIX));
    let bench = target.join("bench");
    let commit = commit(root) + if o.cpu { "-cpu" } else { "" };
    let hashes = bench.join(&commit);
    fs::create_dir_all(&hashes).map_err(|e| format!("{}: {e}", hashes.display()))?;

    let raw = bench.join("frames.argb");
    let result = decode(&o.clip, &raw).and_then(|()| {
        let mut machine = Some(machine());
        for &preset in &o.presets {
            for &depth in &o.depths {
                let row = Row {
                    preset,
                    depth,
                    machine: machine.take().unwrap_or_else(|| "same".to_owned()),
                };
                row.measure(&o, &cli, &raw, (width, height, &rate), &hashes)?;
            }
        }
        Ok(())
    });
    let _ = fs::remove_file(&raw);
    result?;
    eprintln!("frame hashes in {}", hashes.display());
    Ok(())
}

struct Row {
    preset: usize,
    depth: usize,
    machine: String,
}

impl Row {
    fn measure(
        &self,
        o: &Options,
        cli: &Path,
        raw: &Path,
        (width, height, rate): (u32, u32, &str),
        hashes: &Path,
    ) -> Result<(), String> {
        let (preset, preset_args) = PRESETS[self.preset];
        let (depth, depth_args) = DEPTHS[self.depth];
        let file = format!(
            "{}-{}.txt",
            preset.replace(' ', "-"),
            depth.replace(' ', "")
        );
        let reference = o.reference.as_ref().and_then(|dir| {
            let text = fs::read_to_string(Path::new(dir).join(&file)).ok()?;
            let name = Path::new(dir).file_name()?.to_string_lossy().into_owned();
            Some((name, text.lines().map(str::to_owned).collect::<Vec<_>>()))
        });

        let mut cmd = Command::new(cli);
        cmd.args([width.to_string(), height.to_string()])
            .args(preset_args)
            .args(depth_args);
        if preset_args.contains(&"--keep") {
            cmd.args(["--fps", rate]);
        }
        if o.cpu {
            cmd.arg("--cpu");
        }

        let (mut first, mut rest) = (Vec::new(), Vec::new());
        let mut matched = usize::MAX;
        let mut path = String::new();
        for n in 0..o.runs {
            eprintln!("{preset}, {depth}, run {} of {}", n + 1, o.runs);
            let timed = timed_run(&mut cmd, raw, width as usize * height as usize * 4)?;
            let split = timed.ms.len().min(SPLIT);
            first.extend_from_slice(&timed.ms[..split]);
            rest.extend_from_slice(&timed.ms[split..]);
            path = timed.path;
            if let Some((_, want)) = &reference {
                let same = timed.hashes.iter().zip(want).filter(|(a, b)| a == b);
                matched = matched.min(same.count());
            }
            if n == 0 {
                let path = hashes.join(&file);
                fs::write(&path, timed.hashes.join("\n") + "\n")
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        let frames = (first.len() + rest.len()) / o.runs;
        let matches = match reference {
            Some((name, _)) => format!("{name}, {matched} of {frames}"),
            None => "no reference".to_owned(),
        };
        let median_rest = median(&mut rest);
        println!(
            "| {} | CLI | {preset} | {depth} | {path} | {} | {} | {} | {} | {matches} |",
            self.machine,
            ms_cell(median(&mut first)),
            ms_cell(median_rest),
            ms_cell(p95(&mut rest)),
            median_rest.map_or("no frames".to_owned(), |m| format!("{:.1} fps", 1000.0 / m)),
        );
        Ok(())
    }
}

struct Timed {
    /// Each frame's time out of the CLI, frame 0's from the start of the process.
    ms: Vec<f64>,
    hashes: Vec<String>,
    path: String,
}

fn timed_run(cmd: &mut Command, raw: &Path, frame_bytes: usize) -> Result<Timed, String> {
    let input = File::open(raw).map_err(|e| format!("{}: {e}", raw.display()))?;
    let start = Instant::now();
    let mut child = cmd
        .stdin(input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{cmd:?}: {e}"))?;
    let mut out = child.stdout.take().ok_or("the CLI has no stdout")?;
    let mut err = child.stderr.take().ok_or("the CLI has no stderr")?;
    let log = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = err.read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    });

    let mut frame = vec![0u8; frame_bytes];
    let (mut ms, mut hashes) = (Vec::new(), Vec::new());
    let mut last = start;
    loop {
        match out.read_exact(&mut frame) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("reading the CLI's frames: {e}")),
        }
        let now = Instant::now();
        ms.push((now - last).as_secs_f64() * 1000.0);
        last = now;
        hashes.push(format!("{:016x}", hash(&frame)));
    }
    let status = child.wait().map_err(|e| format!("{cmd:?}: {e}"))?;
    let log = log.join().unwrap_or_default();
    if !status.success() || ms.is_empty() {
        return Err(format!("{cmd:?} failed ({status}):\n{log}"));
    }
    Ok(Timed {
        ms,
        hashes,
        path: path_from(&log),
    })
}

/// The path from the CLI's last line, and where the face models ran from segcam's log line.
fn path_from(log: &str) -> String {
    let mut path = log
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("segcam-motion: ")?.split_once(", "))
        .map_or("unknown", |(_, path)| path)
        .to_owned();
    let faces = log.lines().find_map(|l| {
        let (_, device) = l.split_once("face models on ")?;
        device.split(", ready").next()
    });
    if let Some(device) = faces {
        path += &format!(", faces on {device}");
    }
    path
}

fn hash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let (words, tail) = bytes.as_chunks::<8>();
    let tail = tail.iter().map(|&b| u64::from(b));
    for word in words.iter().map(|w| u64::from_le_bytes(*w)).chain(tail) {
        h = (h ^ word).wrapping_mul(0x0100_0000_01b3).rotate_left(23);
    }
    h
}

fn median(ms: &mut [f64]) -> Option<f64> {
    ms.sort_by(f64::total_cmp);
    let mid = ms.len() / 2;
    match ms.len() {
        0 => None,
        n if n % 2 == 1 => Some(ms[mid]),
        _ => Some((ms[mid - 1] + ms[mid]) / 2.0),
    }
}

fn p95(ms: &mut [f64]) -> Option<f64> {
    ms.sort_by(f64::total_cmp);
    let rank = (ms.len() * 95).div_ceil(100);
    ms.get(rank.checked_sub(1)?).copied()
}

fn ms_cell(ms: Option<f64>) -> String {
    ms.map_or("no frames".to_owned(), |m| format!("{m:.1} ms"))
}

/// Width, height and frame rate of the clip's first video stream.
fn probe(clip: &str) -> Result<(u32, u32, String), String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-of", "json"])
        .args(["-show_entries", "stream=width,height,r_frame_rate", clip])
        .output()
        .map_err(|e| format!("ffprobe: {e} (the bench needs ffmpeg on the PATH)"))?;
    if !out.status.success() {
        return Err(format!(
            "ffprobe {clip}: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("ffprobe: {e}"))?;
    let stream = &json["streams"][0];
    let size = |key: &str| stream[key].as_u64().map(|n| n as u32);
    let (Some(width), Some(height)) = (size("width"), size("height")) else {
        return Err(format!("{clip}: no video stream"));
    };
    let rate = stream["r_frame_rate"].as_str().unwrap_or_default();
    let rate = match rate.split_once('/') {
        Some((a, b)) => {
            let (a, b): (f64, f64) = (a.parse().unwrap_or(0.0), b.parse().unwrap_or(1.0));
            (a / b).to_string()
        }
        None => rate.to_owned(),
    };
    Ok((width, height, rate))
}

/// The clip's first frames as the raw ARGB the CLI reads.
fn decode(clip: &str, raw: &Path) -> Result<(), String> {
    eprintln!("decoding the first {FRAMES} frames of {clip}");
    run_cmd(
        Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i", clip, "-map", "0:v:0"])
            .args(["-fps_mode", "passthrough", "-frames:v", &FRAMES.to_string()])
            .args(["-f", "rawvideo", "-pix_fmt", "argb"])
            .arg(raw),
    )
}

fn commit(root: &Path) -> String {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|sha| !sha.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default()
}

/// CPU, GPUs with their drivers and which drives the display, and OS, from the system.
fn machine() -> String {
    let parts = if cfg!(windows) {
        windows_machine()
    } else if cfg!(target_os = "macos") {
        macos_machine()
    } else {
        linux_machine()
    };
    let text = parts.join(", ").replace("(R)", "").replace("(TM)", "");
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn windows_machine() -> Vec<String> {
    let script = "@{cpu=(Get-CimInstance Win32_Processor).Name;\
        os=(Get-CimInstance Win32_OperatingSystem | ForEach-Object {$_.Caption + ' ' + $_.BuildNumber});\
        gpus=@(Get-CimInstance Win32_VideoController | ForEach-Object {@{name=$_.Name;driver=$_.DriverVersion;\
        display=[bool]$_.CurrentHorizontalResolution}})} | ConvertTo-Json -Depth 3 -Compress";
    let text = output("powershell", &["-NoProfile", "-Command", script]);
    let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    let text_of = |v: &serde_json::Value| v.as_str().unwrap_or("unknown").to_owned();
    let mut parts = vec![text_of(&json["cpu"])];
    for gpu in json["gpus"].as_array().into_iter().flatten() {
        let name = text_of(&gpu["name"]);
        let driver = driver_number(&name, &text_of(&gpu["driver"]));
        parts.push(if gpu["display"].as_bool() == Some(true) {
            format!("{name} (display, {driver})")
        } else {
            format!("{name} ({driver})")
        });
    }
    parts.push(text_of(&json["os"]).replace("Microsoft ", ""));
    parts
}

/// NVIDIA's number is the last five digits of the Windows one: 32.0.16.2101 is 621.01.
fn driver_number(gpu: &str, driver: &str) -> String {
    let digits: String = driver.chars().filter(char::is_ascii_digit).collect();
    match digits.len().checked_sub(5) {
        Some(from) if gpu.contains("NVIDIA") => {
            format!("{}.{}", &digits[from..from + 3], &digits[from + 3..])
        }
        _ => driver.to_owned(),
    }
}

fn macos_machine() -> Vec<String> {
    let text = output("system_profiler", &["SPDisplaysDataType", "-json"]);
    let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    let mut parts = vec![output("sysctl", &["-n", "machdep.cpu.brand_string"])];
    for gpu in json["SPDisplaysDataType"].as_array().into_iter().flatten() {
        let name = gpu["sppci_model"].as_str().unwrap_or("unknown");
        let display = gpu["spdisplays_ndrvs"]
            .as_array()
            .is_some_and(|d| !d.is_empty());
        parts.push(if display {
            format!("{name} (display)")
        } else {
            name.to_owned()
        });
    }
    parts.push(format!(
        "macOS {} {}",
        output("sw_vers", &["-productVersion"]),
        output("sw_vers", &["-buildVersion"])
    ));
    parts
}

fn linux_machine() -> Vec<String> {
    let line_of = |file: &str, key: &str| {
        let text = fs::read_to_string(file).unwrap_or_default();
        text.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_once([':', '=']))
            .map_or("unknown".to_owned(), |(_, v)| {
                v.trim().trim_matches('"').to_owned()
            })
    };
    let mut parts = vec![line_of("/proc/cpuinfo", "model name")];
    let pci = output("lspci", &[]);
    parts.extend(
        pci.lines()
            .filter(|l| l.contains("VGA") || l.contains("3D controller"))
            .filter_map(|l| Some(l.splitn(3, ':').nth(2)?.trim().to_owned())),
    );
    parts.push(line_of("/etc/os-release", "PRETTY_NAME"));
    parts
}
