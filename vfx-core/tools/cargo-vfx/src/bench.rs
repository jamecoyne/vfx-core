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
    let hashes = bench.join(folder(&commit(root), dirty(root), o.cpu));
    fs::create_dir_all(&hashes).map_err(|e| format!("{}: {e}", hashes.display()))?;

    let raw = bench.join("frames.argb");
    let frame_bytes = width as usize * height as usize * 4;
    let result = decode(&o.clip, &raw).and_then(|()| {
        // References are checked before anything is timed.
        let clip = frames_hash(&raw, frame_bytes)?;
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
        let header = header(clip, preset_name, depth_name);
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
        let mut path = String::new();
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
            let split = timed.ms.len().min(SPLIT);
            first.extend_from_slice(&timed.ms[..split]);
            rest.extend_from_slice(&timed.ms[split..]);
            path = path_from(&timed.log);
            if let Some((_, want)) = &self.reference {
                let (same, total) = matching(&timed.hashes, want);
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

struct Timed {
    /// Each frame's time out of the CLI, frame 0's from the start of the process.
    ms: Vec<f64>,
    hashes: Vec<String>,
    log: String,
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
    Ok(Timed { ms, hashes, log })
}

/// A hash file's first line: what its hashes are of.
fn header(clip: u64, preset: &str, depth: &str) -> String {
    format!("clip {clip:016x} preset {preset} depth {depth}")
}

fn load_reference(dir: &Path, file: &str, header: &str) -> Result<(String, Vec<String>), String> {
    let path = dir.join(file);
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let hashes = reference_hashes(&text, header).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = dir.file_name().unwrap_or(dir.as_os_str());
    Ok((name.to_string_lossy().into_owned(), hashes))
}

fn reference_hashes(text: &str, header: &str) -> Result<Vec<String>, String> {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or_default();
    if first != header {
        return Err(format!(
            "not a reference for this row: it holds `{first}`, this row is `{header}`"
        ));
    }
    Ok(lines.map(str::to_owned).collect())
}

/// The frames that match, and out of how many: a run or a reference that stops short doesn't match.
fn matching(run: &[String], reference: &[String]) -> (usize, usize) {
    let same = run.iter().zip(reference).filter(|(a, b)| a == b).count();
    (same, run.len().max(reference.len()))
}

/// The hash of the decoded frames, which names the clip without its file name.
fn frames_hash(raw: &Path, frame_bytes: usize) -> Result<u64, String> {
    let mut file = File::open(raw).map_err(|e| format!("{}: {e}", raw.display()))?;
    let mut frame = vec![0u8; frame_bytes];
    let mut h = 0u64;
    loop {
        match file.read_exact(&mut frame) {
            Ok(()) => {
                h = (h ^ hash(&frame))
                    .wrapping_mul(0x0100_0000_01b3)
                    .rotate_left(23)
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(h),
            Err(e) => return Err(format!("{}: {e}", raw.display())),
        }
    }
}

/// The path from the CLI's last line, and where the face models ran.
fn path_from(log: &str) -> String {
    let path = log
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("segcam-motion: ")?.split_once(", "))
        .map_or("unknown", |(_, path)| path);
    match faces_device(log) {
        Some(device) => format!("{path}, faces on {device}"),
        None => path.to_owned(),
    }
}

/// The device from segcam's log line, which also says when no models loaded.
fn faces_device(log: &str) -> Option<&str> {
    log.lines().find_map(|l| {
        let (_, device) = l.split_once("face models on ")?;
        device.split(", ready").next()
    })
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
    eprintln!("decoding the first {FRAMES} frames");
    // ffmpeg before 5.1 has only -vsync. The size is the coded one, so frames stay unrotated.
    let help = output("ffmpeg", &["-hide_banner", "-h", "full"]);
    let every_frame = if help.contains("-fps_mode") {
        "-fps_mode"
    } else {
        "-vsync"
    };
    run_cmd(
        Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-noautorotate", "-i", clip])
            .args(["-map", "0:v:0", every_frame, "passthrough"])
            .args(["-frames:v", &FRAMES.to_string()])
            .args(["-f", "rawvideo", "-pix_fmt", "argb"])
            .arg(raw),
    )
}

fn commit(root: &Path) -> String {
    let sha = git(root, &["rev-parse", "--short=7", "HEAD"]);
    if sha.is_empty() {
        "unknown".to_owned()
    } else {
        sha
    }
}

fn dirty(root: &Path) -> bool {
    !git(root, &["status", "--porcelain"]).is_empty()
}

fn git(root: &Path, args: &[&str]) -> String {
    let root = root.to_string_lossy();
    output("git", &[&["-C", &root], args].concat())
}

/// The hash folder's name: a dirty tree must not overwrite its commit's hashes.
fn folder(commit: &str, dirty: bool, cpu: bool) -> String {
    let dirty = if dirty { "-dirty" } else { "" };
    let cpu = if cpu { "-cpu" } else { "" };
    format!("{commit}{dirty}{cpu}")
}

fn output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default()
}

/// CPU, GPUs and OS from the system: Windows adds each GPU's driver and which drives the display,
/// macOS which drives the display.
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
    if parts.len() == 1 {
        parts.push("GPU unknown".to_owned());
    }
    parts.push(line_of("/etc/os-release", "PRETTY_NAME"));
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn a_short_run_or_reference_does_not_match() {
        let full = lines(&["a", "b", "c"]);
        assert_eq!(matching(&full, &full), (3, 3));
        assert_eq!(matching(&full[..2], &full), (2, 3));
        assert_eq!(matching(&full, &full[..2]), (2, 3));
        assert_eq!(matching(&full, &lines(&["a", "x", "c"])), (2, 3));
        assert_eq!(matching(&[], &full), (0, 3));
    }

    #[test]
    fn a_dirty_tree_has_its_own_folder() {
        assert_eq!(folder("abc1234", false, false), "abc1234");
        assert_eq!(folder("abc1234", true, false), "abc1234-dirty");
        assert_eq!(folder("abc1234", false, true), "abc1234-cpu");
        assert_eq!(folder("abc1234", true, true), "abc1234-dirty-cpu");
    }

    #[test]
    fn a_reference_of_another_row_is_refused() {
        let this = header(1, "overlay", "8 bit");
        let text = format!("{this}\na\nb\n");
        assert_eq!(reference_hashes(&text, &this).unwrap(), lines(&["a", "b"]));
        for other in [
            header(2, "overlay", "8 bit"),
            header(1, "show", "8 bit"),
            header(1, "overlay", "float"),
        ] {
            assert!(reference_hashes(&text, &other).is_err());
        }
        assert!(reference_hashes("a\nb\n", &this).is_err());
        assert!(reference_hashes("", &this).is_err());
    }

    #[test]
    fn the_face_device_comes_from_the_log() {
        let up = "segcam Motion: face models on DirectML 0 (GPU), ready in 900 ms\n";
        assert_eq!(faces_device(up), Some("DirectML 0 (GPU)"));
        assert_eq!(faces_device("segcam Motion: no face models\n"), None);
        assert_eq!(faces_device(""), None);
    }
}
