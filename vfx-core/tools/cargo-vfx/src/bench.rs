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
  --cpu        the CLI's --cpu path: on the CPU, every frame fresh except in the kept 1.1 look
  --reference  an earlier run's folder of frame hashes on the same clip (target/bench/COMMIT), for
               the Matches column; it has to hold every row this run prints";

const FRAMES: u32 = 300;
const FIRST_FRAMES: usize = 40;

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
    let hashes = bench.join(hash_folder(&head_commit(root), tree_is_dirty(root), o.cpu));
    fs::create_dir_all(&hashes).map_err(|e| format!("{}: {e}", hashes.display()))?;

    let raw = bench.join("frames.argb");
    let frame_bytes = width as usize * height as usize * 4;
    let result = decode(&o.clip, &raw).and_then(|()| {
        // References are checked before anything is timed.
        let clip = decoded_frames_hash(&raw, frame_bytes)?;
        let mut rows = Vec::new();
        for &preset in &o.presets {
            for &depth in &o.depths {
                rows.push(Row::new(&o, preset, depth, clip)?);
            }
        }
        let mut machine = Some(machine());
        for row in &rows {
            let machine = machine.take().unwrap_or_else(|| "same".to_owned());
            row.measure(&o, &cli, &raw, (width, height, &rate), &hashes, &machine)?;
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
    file: String,
    header: String,
    /// The reference folder's name and this row's hashes in it.
    reference: Option<(String, Vec<String>)>,
}

impl Row {
    fn new(o: &Options, preset: usize, depth: usize, clip: u64) -> Result<Self, String> {
        let (preset_name, depth_name) = (PRESETS[preset].0, DEPTHS[depth].0);
        let file = format!(
            "{}-{}.txt",
            preset_name.replace(' ', "-"),
            depth_name.replace(' ', "")
        );
        let header = hash_file_header(clip, preset_name, depth_name);
        let reference = match &o.reference {
            Some(dir) => Some(load_reference(Path::new(dir), &file, &header)?),
            None => None,
        };
        Ok(Self {
            preset,
            depth,
            file,
            header,
            reference,
        })
    }

    fn measure(
        &self,
        o: &Options,
        cli: &Path,
        raw: &Path,
        (width, height, rate): (u32, u32, &str),
        hashes: &Path,
        machine: &str,
    ) -> Result<(), String> {
        let (preset, preset_args) = PRESETS[self.preset];
        let (depth, depth_args) = DEPTHS[self.depth];

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
        let (mut matched, mut of) = (usize::MAX, 0);
        let (mut drawn, mut faces) = (DrawnOn::default(), None);
        for n in 0..o.runs {
            eprintln!("{preset}, {depth}, run {} of {}", n + 1, o.runs);
            let timed = timed_run(&mut cmd, raw, width as usize * height as usize * 4)?;
            // The 1.1 look has faces too.
            if (preset_args.contains(&"--face") || preset_args.contains(&"--preset"))
                && faces_device(&timed.log).is_none()
            {
                return Err(format!(
                    "{preset}: no face models, so this row would time no faces:\n{}",
                    timed.log
                ));
            }
            let split = timed.ms.len().min(FIRST_FRAMES);
            first.extend_from_slice(&timed.ms[..split]);
            rest.extend_from_slice(&timed.ms[split..]);
            drawn.pool(
                drawn_on(&timed.log)
                    .ok_or_else(|| format!("{preset}: the CLI named no path:\n{}", timed.log))?,
            );
            faces = faces_device(&timed.log).map(str::to_owned);
            if let Some((_, want)) = &self.reference {
                let (same, total) = count_matches(&timed.hashes, want);
                matched = matched.min(same);
                of = of.max(total);
            }
            if n == 0 {
                let path = hashes.join(&self.file);
                fs::write(
                    &path,
                    format!("{}\n{}\n", self.header, timed.hashes.join("\n")),
                )
                .map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        let matches = match &self.reference {
            Some((name, _)) => format!("{name}, {matched} of {of}"),
            None => "no reference".to_owned(),
        };
        let path = path_cell(&drawn, faces.as_deref());
        let median_rest = median(&mut rest);
        println!(
            "| {machine} | CLI | {preset} | {depth} | {path} | {} | {} | {} | {} | {matches} |",
            ms_cell(median(&mut first)),
            ms_cell(median_rest),
            ms_cell(p95(&mut rest)),
            median_rest.map_or("no frames".to_owned(), |m| format!("{:.1} fps", 1000.0 / m)),
        );
        Ok(())
    }
}

struct TimedRun {
    /// Each frame's time out of the CLI, frame 0's from the start of the process.
    ms: Vec<f64>,
    hashes: Vec<String>,
    log: String,
}

fn timed_run(cmd: &mut Command, raw: &Path, frame_bytes: usize) -> Result<TimedRun, String> {
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
        hashes.push(format!("{:016x}", frame_hash(&frame)));
    }
    let status = child.wait().map_err(|e| format!("{cmd:?}: {e}"))?;
    let log = log.join().unwrap_or_default();
    if !status.success() || ms.is_empty() {
        return Err(format!("{cmd:?} failed ({status}):\n{log}"));
    }
    Ok(TimedRun { ms, hashes, log })
}

fn hash_file_header(clip: u64, preset: &str, depth: &str) -> String {
    format!("clip {clip:016x} preset {preset} depth {depth}")
}

fn load_reference(dir: &Path, file: &str, header: &str) -> Result<(String, Vec<String>), String> {
    let path = dir.join(file);
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let hashes = parse_reference(&text, header).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = dir.file_name().unwrap_or(dir.as_os_str());
    Ok((name.to_string_lossy().into_owned(), hashes))
}

fn parse_reference(text: &str, header: &str) -> Result<Vec<String>, String> {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or_default();
    if first != header {
        return Err(format!(
            "not a reference for this row: it holds `{first}`, this row is `{header}`"
        ));
    }
    Ok(lines.map(str::to_owned).collect())
}

/// Out of the longer of the two, so a short run or reference doesn't match.
fn count_matches(run: &[String], reference: &[String]) -> (usize, usize) {
    let same = run.iter().zip(reference).filter(|(a, b)| a == b).count();
    (same, run.len().max(reference.len()))
}

/// Names the clip without its file name.
fn decoded_frames_hash(raw: &Path, frame_bytes: usize) -> Result<u64, String> {
    let mut file = File::open(raw).map_err(|e| format!("{}: {e}", raw.display()))?;
    let mut frame = vec![0u8; frame_bytes];
    let mut h = 0u64;
    loop {
        match file.read_exact(&mut frame) {
            Ok(()) => {
                h = (h ^ frame_hash(&frame))
                    .wrapping_mul(0x0100_0000_01b3)
                    .rotate_left(23)
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(h),
            Err(e) => return Err(format!("{}: {e}", raw.display())),
        }
    }
}

#[derive(Debug, Default, PartialEq)]
struct DrawnOn {
    frames: u64,
    gpu_frames: u64,
    adapter: String,
    kept: bool,
}

impl DrawnOn {
    fn pool(&mut self, run: DrawnOn) {
        self.frames += run.frames;
        self.gpu_frames += run.gpu_frames;
        self.kept = run.kept;
        if !run.adapter.is_empty() {
            self.adapter = run.adapter;
        }
    }
}

fn drawn_on(log: &str) -> Option<DrawnOn> {
    let line = log
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("segcam-motion: "))?;
    let mut parts = line.splitn(3, ", ");
    let frames = parts.next()?.strip_suffix(" frames")?.parse().ok()?;
    let kept = match parts.next()? {
        "kept" => true,
        "fresh" => false,
        _ => return None,
    };
    let (gpu_frames, adapter) = parts.next()?.split_once(" on the GPU")?;
    Some(DrawnOn {
        frames,
        gpu_frames: gpu_frames.parse().ok()?,
        adapter: adapter
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .to_owned(),
        kept,
    })
}

fn path_cell(drawn: &DrawnOn, faces: Option<&str>) -> String {
    let DrawnOn {
        frames,
        gpu_frames,
        adapter,
        kept,
    } = drawn;
    let cpu_frames = frames - gpu_frames;
    let path = if *gpu_frames == 0 {
        let how = if *kept { "kept" } else { "fresh" };
        format!("CPU, {how}, {cpu_frames} of {frames} frames")
    } else if cpu_frames == 0 {
        format!("GPU ({adapter}), {gpu_frames} of {frames} frames")
    } else {
        format!("GPU ({adapter}), {gpu_frames} of {frames} frames, CPU {cpu_frames} frames")
    };
    match faces {
        Some(device) => format!("{path}, faces on {device}"),
        None => path,
    }
}

fn faces_device(log: &str) -> Option<&str> {
    log.lines().find_map(|l| {
        let (_, device) = l.split_once("face models on ")?;
        device.split(", ready").next()
    })
}

fn frame_hash(bytes: &[u8]) -> u64 {
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

fn decode(clip: &str, raw: &Path) -> Result<(), String> {
    eprintln!("decoding the first {FRAMES} frames");
    // ffmpeg before 5.1 has only -vsync. The size is the coded one, so frames stay unrotated.
    let help = command_stdout("ffmpeg", &["-hide_banner", "-h", "full"]);
    let passthrough_flag = if help.contains("-fps_mode") {
        "-fps_mode"
    } else {
        "-vsync"
    };
    run_cmd(
        Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-noautorotate", "-i", clip])
            .args(["-map", "0:v:0", passthrough_flag, "passthrough"])
            .args(["-frames:v", &FRAMES.to_string()])
            .args(["-f", "rawvideo", "-pix_fmt", "argb"])
            .arg(raw),
    )
}

fn head_commit(root: &Path) -> String {
    let sha = git(root, &["rev-parse", "--short=7", "HEAD"]);
    if sha.is_empty() {
        "unknown".to_owned()
    } else {
        sha
    }
}

fn tree_is_dirty(root: &Path) -> bool {
    !git(root, &["status", "--porcelain"]).is_empty()
}

fn git(root: &Path, args: &[&str]) -> String {
    let root = root.to_string_lossy();
    command_stdout("git", &[&["-C", &root], args].concat())
}

/// A dirty tree must not overwrite its commit's hashes.
fn hash_folder(commit: &str, dirty: bool, cpu: bool) -> String {
    let dirty = if dirty { "-dirty" } else { "" };
    let cpu = if cpu { "-cpu" } else { "" };
    format!("{commit}{dirty}{cpu}")
}

fn command_stdout(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default()
}

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
    let text = command_stdout("powershell", &["-NoProfile", "-Command", script]);
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
    let sysctl = |key: &str| command_stdout("sysctl", &["-n", key]);
    macos_parts(
        &sysctl("machdep.cpu.brand_string"),
        &sysctl("hw.physicalcpu"),
        &sysctl("hw.memsize"),
        &command_stdout("system_profiler", &["SPDisplaysDataType", "-json"]),
        &format!(
            "macOS {} {}",
            command_stdout("sw_vers", &["-productVersion"]),
            command_stdout("sw_vers", &["-buildVersion"])
        ),
    )
}

/// Apple silicon's CPU and GPU are one chip, so it is named once.
fn macos_parts(
    cpu: &str,
    cpu_cores: &str,
    memory_bytes: &str,
    displays: &str,
    os: &str,
) -> Vec<String> {
    let json: serde_json::Value = serde_json::from_str(displays).unwrap_or_default();
    let gpus = json["SPDisplaysDataType"].as_array();
    let chip = || {
        let [gpu] = gpus?.as_slice() else {
            return None;
        };
        if gpu["sppci_model"].as_str()? != cpu {
            return None;
        }
        let gpu_cores: u32 = gpu["sppci_cores"].as_str()?.parse().ok()?;
        let cpu_cores: u32 = cpu_cores.parse().ok()?;
        let gb = memory_bytes.parse::<u64>().ok()? >> 30;
        Some(format!(
            "{cpu}, {cpu_cores} CPU cores, {gpu_cores} GPU cores, {gb} GB"
        ))
    };
    if let Some(chip) = chip() {
        return vec![chip, os.to_owned()];
    }
    let mut parts = vec![cpu.to_owned()];
    for gpu in gpus.into_iter().flatten() {
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
    parts.push(os.to_owned());
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
    let pci = command_stdout("lspci", &[]);
    parts.extend(
        pci.lines()
            .filter(|l| l.contains("VGA") || l.contains("3D controller"))
            .filter_map(|l| Some(l.splitn(3, ':').nth(2)?.trim().to_owned())),
    );
    if parts.len() == 1 {
        parts.push("GPU unknown".to_owned());
    }
    parts.push(line_of("/etc/os-release", "PRETTY_NAME"));
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn short_hashes_do_not_match() {
        let full = strings(&["a", "b", "c"]);
        assert_eq!(count_matches(&full, &full), (3, 3));
        assert_eq!(count_matches(&full[..2], &full), (2, 3));
        assert_eq!(count_matches(&full, &full[..2]), (2, 3));
        assert_eq!(count_matches(&full, &strings(&["a", "x", "c"])), (2, 3));
        assert_eq!(count_matches(&[], &full), (0, 3));
    }

    #[test]
    fn dirty_tree_gets_own_folder() {
        assert_eq!(hash_folder("abc1234", false, false), "abc1234");
        assert_eq!(hash_folder("abc1234", true, false), "abc1234-dirty");
        assert_eq!(hash_folder("abc1234", false, true), "abc1234-cpu");
        assert_eq!(hash_folder("abc1234", true, true), "abc1234-dirty-cpu");
    }

    #[test]
    fn other_row_reference_refused() {
        let this = hash_file_header(1, "overlay", "8 bit");
        let text = format!("{this}\na\nb\n");
        assert_eq!(parse_reference(&text, &this).unwrap(), strings(&["a", "b"]));
        for other in [
            hash_file_header(2, "overlay", "8 bit"),
            hash_file_header(1, "show", "8 bit"),
            hash_file_header(1, "overlay", "float"),
        ] {
            assert!(parse_reference(&text, &other).is_err());
        }
        assert!(parse_reference("a\nb\n", &this).is_err());
        assert!(parse_reference("", &this).is_err());
    }

    const M4_DISPLAYS: &str = r#"{"SPDisplaysDataType": [{
        "sppci_model": "Apple M4", "sppci_cores": "10",
        "spdisplays_ndrvs": [{"spdisplays_main": "spdisplays_yes"}]
    }]}"#;
    const MACOS: &str = "macOS 27.2 26B123";

    #[test]
    fn apple_silicon_is_one_chip() {
        assert_eq!(
            macos_parts("Apple M4", "10", "25769803776", M4_DISPLAYS, MACOS),
            ["Apple M4, 10 CPU cores, 10 GPU cores, 24 GB", MACOS]
        );
    }

    #[test]
    fn missing_numbers_list_cpu_gpu() {
        let today = ["Apple M4", "Apple M4 (display)", MACOS];
        assert_eq!(
            macos_parts("Apple M4", "", "25769803776", M4_DISPLAYS, MACOS),
            today
        );
        assert_eq!(macos_parts("Apple M4", "10", "", M4_DISPLAYS, MACOS), today);
        let no_gpu_cores = M4_DISPLAYS.replace(r#""sppci_cores": "10","#, "");
        assert_eq!(
            macos_parts("Apple M4", "10", "25769803776", &no_gpu_cores, MACOS),
            today
        );
        assert_eq!(
            macos_parts("Apple M4", "10", "25769803776", "", MACOS),
            ["Apple M4", MACOS]
        );
    }

    #[test]
    fn intel_mac_lists_cpu_gpus() {
        let displays = r#"{"SPDisplaysDataType": [
            {"sppci_model": "Intel UHD Graphics 630"},
            {"sppci_model": "AMD Radeon Pro 5500M", "sppci_cores": "8",
             "spdisplays_ndrvs": [{"spdisplays_main": "spdisplays_yes"}]}
        ]}"#;
        assert_eq!(
            macos_parts("Intel Core i9-9980HK", "8", "34359738368", displays, MACOS),
            [
                "Intel Core i9-9980HK",
                "Intel UHD Graphics 630",
                "AMD Radeon Pro 5500M (display)",
                MACOS
            ]
        );
    }

    #[test]
    fn gpu_cpu_counts_pool_across_runs() {
        let mut drawn = DrawnOn::default();
        for log in [
            "segcam-motion: 300 frames, kept, 297 on the GPU (RTX 4070)\n",
            "segcam-motion: 300 frames, kept, 300 on the GPU (RTX 4070)\n",
        ] {
            drawn.pool(drawn_on(log).unwrap());
        }
        assert_eq!(
            path_cell(&drawn, None),
            "GPU (RTX 4070), 597 of 600 frames, CPU 3 frames"
        );
    }

    #[test]
    fn path_cell_names_one_path() {
        let all_gpu = drawn_on("segcam-motion: 300 frames, kept, 300 on the GPU (RTX 4070)");
        assert_eq!(
            path_cell(&all_gpu.unwrap(), Some("DirectML 0")),
            "GPU (RTX 4070), 300 of 300 frames, faces on DirectML 0"
        );
        let kept = drawn_on("segcam-motion: 300 frames, kept, 0 on the GPU").unwrap();
        assert_eq!(path_cell(&kept, None), "CPU, kept, 300 of 300 frames");
        let fresh = drawn_on("segcam-motion: 300 frames, fresh, 0 on the GPU").unwrap();
        assert_eq!(path_cell(&fresh, None), "CPU, fresh, 300 of 300 frames");
    }

    #[test]
    fn adapter_name_keeps_commas() {
        let drawn = drawn_on("segcam-motion: 1 frames, kept, 1 on the GPU (a, b)").unwrap();
        assert_eq!(drawn.adapter, "a, b");
    }

    #[test]
    fn unknown_cli_line_names_no_path() {
        assert_eq!(drawn_on("segcam-motion: 300 frames"), None);
        assert_eq!(drawn_on(""), None);
    }

    #[test]
    fn face_device_read_from_log() {
        let up = "segcam Motion: face models on DirectML 0 (GPU), ready in 900 ms\n";
        assert_eq!(faces_device(up), Some("DirectML 0 (GPU)"));
        assert_eq!(faces_device("segcam Motion: no face models\n"), None);
        assert_eq!(faces_device(""), None);
    }
}
