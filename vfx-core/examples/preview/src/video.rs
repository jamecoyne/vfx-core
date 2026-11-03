//! Video files as a source, decoded by `ffmpeg` / `ffprobe` processes on PATH (native
//! only).
//!
//! Live playback: ffmpeg decodes to raw 8 bit ARGB frames on a pipe, the layout a CPU effect's
//! history takes as is. A reader thread takes a file's frames from the pipe at its frame rate
//! and queues them with their times; the loop takes them in order, and a loop that falls
//! behind loses the oldest, so the picture never trails the clock. Recording writes the output
//! through [`Encoder`].

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use vfx_core::cpu::{Argb, Channel};

/// Decoded frames waiting for the loop, at most: enough for a render that runs long, few
/// enough that a loop that stays behind shows the picture no more than this many frames late.
const MAX_QUEUED: usize = 2;

/// File extensions treated as video when dropped on the window.
const EXTENSIONS: &[&str] = &["mp4", "mov", "m4v", "mkv", "webm", "avi", "gif", "wmv"];

pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Where frames come from.
#[derive(Clone)]
pub enum Input {
    File(PathBuf),
    /// A camera by the name `cameras` lists, in a mode it supports (`None` leaves ffmpeg's
    /// default).
    Camera(String, Option<Mode>),
}

/// A frame size and rate a camera can deliver.
#[derive(Clone, Copy)]
pub struct Mode {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

/// The camera modes to try, in order. ffmpeg's default is often a different shape than the
/// camera's usual feed (a portrait size for a landscape camera, which stretches the image),
/// and each camera takes only some sizes and rates.
const CAMERA_MODES: [Option<Mode>; 4] = [
    Some(Mode {
        width: 1280,
        height: 720,
        fps: 30,
    }),
    Some(Mode {
        width: 640,
        height: 480,
        fps: 30,
    }),
    Some(Mode {
        width: 1920,
        height: 1080,
        fps: 60,
    }),
    None,
];

impl Input {
    /// The ffmpeg / ffprobe arguments that open this input.
    fn args(&self) -> Vec<OsString> {
        let mut args: Vec<OsString> = Vec::new();
        match self {
            Input::File(path) => args.extend(["-i".into(), path.into()]),
            Input::Camera(name, mode) => {
                let (format, device) = if cfg!(target_os = "windows") {
                    ("dshow", format!("video={name}"))
                } else if cfg!(target_os = "macos") {
                    ("avfoundation", name.clone())
                } else {
                    ("v4l2", name.clone())
                };
                args.extend(["-f".into(), format.into()]);
                if let Some(m) = mode {
                    args.extend([
                        "-video_size".into(),
                        format!("{}x{}", m.width, m.height).into(),
                        "-framerate".into(),
                        m.fps.to_string().into(),
                    ]);
                }
                args.extend(["-i".into(), device.into()]);
            }
        }
        args
    }

    pub fn name(&self) -> String {
        match self {
            Input::File(path) => path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            Input::Camera(name, _) => format!("Webcam: {name}"),
        }
    }
}

/// The cameras ffmpeg can open, by name.
pub fn cameras() -> Vec<String> {
    if cfg!(target_os = "linux") {
        let mut devices: Vec<String> = std::fs::read_dir("/dev")
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|p| p.starts_with("/dev/video"))
            .collect();
        devices.sort();
        return devices;
    }
    let format = if cfg!(target_os = "windows") {
        "dshow"
    } else {
        "avfoundation"
    };
    let Ok(out) = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            format,
            "-i",
            "dummy",
        ])
        .stdin(Stdio::null())
        .output()
    else {
        return Vec::new();
    };
    // ffmpeg lists devices on stderr.
    let text = String::from_utf8_lossy(&out.stderr);
    if cfg!(target_os = "windows") {
        // [dshow @ ...] "USB Camera" (video)
        text.lines()
            .filter(|l| l.ends_with("(video)"))
            .filter_map(|l| l.split('"').nth(1).map(str::to_string))
            .collect()
    } else {
        // [AVFoundation indev @ ...] [0] FaceTime HD Camera, up to the audio devices.
        text.lines()
            .skip_while(|l| !l.contains("video devices"))
            .skip(1)
            .take_while(|l| !l.contains("audio devices"))
            .filter_map(|l| l.split("] ").nth(2).map(str::to_string))
            .collect()
    }
}

/// What ffprobe says about an input, plus the size frames are decoded at.
#[derive(Clone)]
pub struct Info {
    pub input: Input,
    /// Decoded size: the input's, scaled to fit the preview's limit, even.
    pub size: (u32, u32),
    /// As ffmpeg writes it ("30000/1001"), so exports keep the exact rate.
    pub rate: String,
    pub fps: f64,
    /// Infinite for cameras.
    pub duration: f64,
    /// Fitted into `size` with transparent padding, not scaled to it (see [`Info::fitted`]).
    pub fit: bool,
}

impl Info {
    /// Decodes into `size`, keeping the input's shape and padding the rest with transparent
    /// pixels, so a layer lines up with a frame of that size.
    pub fn fitted(mut self, size: (u32, u32)) -> Self {
        self.size = size;
        self.fit = true;
        self
    }

    /// The file, unless this is a camera.
    pub fn file(&self) -> Option<&Path> {
        match &self.input {
            Input::File(path) => Some(path),
            Input::Camera(..) => None,
        }
    }

    pub fn probe(input: Input, max_side: u32) -> Result<Self, String> {
        let candidates = match &input {
            Input::File(_) => vec![input.clone()],
            Input::Camera(name, _) => CAMERA_MODES
                .iter()
                .map(|&mode| Input::Camera(name.clone(), mode))
                .collect(),
        };
        for input in candidates {
            let out = Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate:format=duration",
                    "-of",
                    "default=noprint_wrappers=1",
                ])
                .args(input.args())
                .stdin(Stdio::null())
                .output()
                .map_err(|e| format!("couldn't start ffprobe (is it on PATH?): {e}"))?;
            let text = String::from_utf8_lossy(&out.stdout);
            let field = |key: &str| {
                text.lines()
                    .find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
                    .map(str::trim)
            };
            let (w, h): (u32, u32) = match (
                field("width").and_then(|v| v.parse().ok()),
                field("height").and_then(|v| v.parse().ok()),
            ) {
                (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
                // A camera refuses a mode it doesn't support; try the next one.
                _ if matches!(input, Input::Camera(..)) => continue,
                _ => return Err("no video stream found".into()),
            };
            let rate = field("avg_frame_rate")
                .filter(|r| parse_rate(r).is_some())
                .unwrap_or("30")
                .to_string();
            let fps = parse_rate(&rate).unwrap_or(30.0);
            // Still images and some streams have no duration; treat them as one frame.
            let duration = match input {
                Input::Camera(..) => f64::INFINITY,
                Input::File(_) => field("duration")
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|d| *d > 0.0)
                    .unwrap_or(1.0 / fps),
            };

            let scale = (max_side as f32 / w.max(h) as f32).min(1.0);
            let even = |v: u32| ((v as f32 * scale) as u32 & !1).max(2);
            return Ok(Self {
                input,
                size: (even(w), even(h)),
                rate,
                fps,
                duration,
                fit: false,
            });
        }
        Err("no frame rate the camera supports worked".into())
    }

    fn frame_pixels(&self) -> usize {
        (self.size.0 * self.size.1) as usize
    }

    /// ffmpeg arguments that decode this input to raw 8 bit ARGB at `size` on stdout, from
    /// `start` seconds (files only). `Pace::OneFrame` stops after the first frame. Not `-re`:
    /// it hands over the first half second at once, which the queue would lose at every loop.
    fn decode_command(&self, start: f64, pace: Pace) -> Command {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error"]);
        if self.file().is_some() && start > 0.0 {
            cmd.args(["-ss", &format!("{start:.3}")]);
        }
        let (w, h) = self.size;
        // Through RGBA first: swscale rounds YUV to ARGB differently, up to 3 levels off.
        let filter = if self.fit {
            format!(
                "format=rgba,scale={w}:{h}:force_original_aspect_ratio=decrease,\
                 pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black@0,fps={}",
                self.rate
            )
        } else {
            format!("scale={w}:{h},format=rgba,fps={}", self.rate)
        };
        cmd.args(self.input.args()).args(["-an", "-vf"]).arg(filter);
        if pace == Pace::OneFrame {
            cmd.args(["-frames:v", "1"]);
        }
        cmd.args(["-pix_fmt", "argb", "-f", "rawvideo", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        cmd
    }
}

fn parse_rate(rate: &str) -> Option<f64> {
    let (n, d) = rate.split_once('/').unwrap_or((rate, "1"));
    let fps = n.parse::<f64>().ok()? / d.parse::<f64>().ok()?;
    (fps.is_finite() && fps > 0.0).then_some(fps)
}

#[derive(Clone, Copy, PartialEq)]
enum Pace {
    /// Real time, to the end.
    Live,
    /// Just the first frame (a paused seek).
    OneFrame,
}

/// Live playback that can pause and seek. Each seek replaces the ffmpeg process with
/// one started at the new position; a reader thread per process queues its frames, and
/// frames from a replaced process are ignored.
pub struct Video {
    pub info: Info,
    shared: Arc<Mutex<Playback>>,
}

struct Decoded {
    pixels: Vec<Argb<u8>>,
    time: f64,
}

struct Playback {
    /// Bumped on every seek, so old reader threads know to stop.
    generation: u64,
    child: Option<Child>,
    /// Oldest first, at most [`MAX_QUEUED`].
    queue: VecDeque<Decoded>,
    /// Buffers to decode into again, so a frame allocates nothing.
    spare: Vec<Vec<Argb<u8>>>,
    /// The time of the frame last taken, or where the last seek landed.
    position: f64,
    /// Where the current decode started, in seconds.
    start: f64,
    /// Frames the current decode has delivered.
    frames: u64,
    /// Frames that arrived since the last [`Video::take_arrived_count`], dropped ones too.
    arrived: u64,
    playing: bool,
    /// A playing file reached its end (the caller loops it with `seek`).
    ended: bool,
}

impl Video {
    /// Starts playing `input`, scaled to fit `max_side`.
    pub fn open(input: Input, max_side: u32) -> Result<Self, String> {
        let info = Info::probe(input, max_side)?;
        Self::start(info)
    }

    /// Starts playing `input` fitted into `size` (see [`Info::fitted`]).
    pub fn open_fitted(input: Input, size: (u32, u32)) -> Result<Self, String> {
        let info = Info::probe(input, size.0.max(size.1))?.fitted(size);
        Self::start(info)
    }

    fn start(info: Info) -> Result<Self, String> {
        let video = Self {
            info,
            shared: Arc::new(Mutex::new(Playback {
                generation: 0,
                child: None,
                queue: VecDeque::new(),
                spare: Vec::new(),
                position: 0.0,
                start: 0.0,
                frames: 0,
                arrived: 0,
                playing: true,
                ended: false,
            })),
        };
        video.seek(0.0, true)?;
        Ok(video)
    }

    /// Restarts decoding at `position` seconds, playing on or showing just that
    /// frame. Cameras always play live from now.
    pub fn seek(&self, position: f64, play: bool) -> Result<(), String> {
        let camera = self.info.file().is_none();
        let position = if camera {
            0.0
        } else {
            position.clamp(0.0, (self.info.duration - 1.0 / self.info.fps).max(0.0))
        };
        let pace = if play || camera {
            Pace::Live
        } else {
            Pace::OneFrame
        };

        let mut s = self.shared.lock().unwrap();
        if let Some(mut old) = s.child.take() {
            let _ = old.kill();
            let _ = old.wait();
        }
        let mut child = self
            .info
            .decode_command(position, pace)
            .spawn()
            .map_err(|e| format!("couldn't start ffmpeg (is it on PATH?): {e}"))?;
        let mut stdout = child.stdout.take().expect("stdout is piped");
        s.generation += 1;
        s.child = Some(child);
        s.position = position;
        s.start = position;
        s.frames = 0;
        s.playing = play || camera;
        s.ended = false;
        while let Some(old) = s.queue.pop_front() {
            s.spare.push(old.pixels);
        }

        let generation = s.generation;
        let shared = Arc::clone(&self.shared);
        let size = self.info.frame_pixels();
        let (fps, duration) = (self.info.fps, self.info.duration);
        let mut pixels = s.spare.pop().unwrap_or_else(|| bytemuck::zeroed_vec(size));
        let mut first_frame_at = None;
        let mut frames_read = 0u32;
        std::thread::spawn(move || {
            // Ends at the end of the file, or when a seek or drop kills this ffmpeg.
            while stdout
                .read_exact(bytemuck::cast_slice_mut(&mut pixels))
                .is_ok()
            {
                if !camera {
                    // Due by the clock from the first frame; a late one goes straight on.
                    let now = Instant::now();
                    let since_first = Duration::from_secs_f64(f64::from(frames_read) / fps);
                    let due = *first_frame_at.get_or_insert(now) + since_first;
                    std::thread::sleep(due.saturating_duration_since(now));
                    frames_read += 1;
                }
                let mut s = shared.lock().unwrap();
                if s.generation != generation {
                    return;
                }
                let time = (s.start + s.frames as f64 / fps).min(duration);
                s.frames += 1;
                s.arrived += 1;
                let next = if s.queue.len() == MAX_QUEUED {
                    s.queue.pop_front().map(|oldest| oldest.pixels)
                } else {
                    s.spare.pop()
                };
                let next = next.unwrap_or_else(|| bytemuck::zeroed_vec(size));
                let pixels = std::mem::replace(&mut pixels, next);
                s.queue.push_back(Decoded { pixels, time });
            }
            let mut s = shared.lock().unwrap();
            if s.generation == generation && s.playing {
                s.ended = true;
            }
        });
        Ok(())
    }

    /// Swaps the oldest waiting frame into `pixels`, whose old buffer is decoded into again,
    /// and returns its time. None when nothing arrived since the last call.
    pub fn take_frame(&self, pixels: &mut Vec<Argb<u8>>) -> Option<f64> {
        let mut s = self.shared.lock().unwrap();
        let frame = s.queue.pop_front()?;
        let old = std::mem::replace(pixels, frame.pixels);
        // A buffer from another video, of another size, would put the pipe out of step.
        if old.len() == self.info.frame_pixels() {
            s.spare.push(old);
        }
        s.position = frame.time;
        Some(frame.time)
    }

    /// As [`Video::take_frame`], skipping to the newest waiting frame, for a layer that
    /// has no history to keep whole.
    pub fn take_newest(&self, pixels: &mut Vec<Argb<u8>>) -> Option<f64> {
        {
            let mut s = self.shared.lock().unwrap();
            while s.queue.len() > 1 {
                let old = s.queue.pop_front().expect("more than one");
                s.spare.push(old.pixels);
            }
        }
        self.take_frame(pixels)
    }

    /// The frames that arrived from ffmpeg since the last call.
    pub fn take_arrived_count(&self) -> u64 {
        std::mem::take(&mut self.shared.lock().unwrap().arrived)
    }

    /// Where playback is in the file (or how long a camera has run), in seconds: the frame
    /// last taken.
    pub fn position(&self) -> f64 {
        self.shared.lock().unwrap().position
    }

    pub fn playing(&self) -> bool {
        self.shared.lock().unwrap().playing
    }

    /// Whether a playing file has reached its end and its last frame was taken.
    pub fn ended(&self) -> bool {
        let s = self.shared.lock().unwrap();
        s.ended && s.queue.is_empty()
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        let mut s = self.shared.lock().unwrap();
        s.generation += 1;
        if let Some(mut child) = s.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Puts this process in a Windows job object that kills everything in it when the
/// process ends, however it ends, so ffmpeg and ffplay never outlive the app (they
/// inherit the job). Elsewhere this does nothing; the `Drop`s still clean up on a
/// normal exit.
pub fn kill_children_on_exit() {
    #[cfg(windows)]
    // SAFETY: plain Win32 calls with a zeroed, correctly sized limits struct. The job
    // handle is deliberately never closed: Windows closes it when the process ends,
    // which is what kills the job.
    unsafe {
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return;
        }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&raw const limits).cast(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        AssignProcessToJobObject(job, GetCurrentProcess());
    }
}

/// Frames per second in a recording. Frames arrive at the live rate and are timed by the
/// wall clock, so a slow frame repeats rather than speeding the recording up.
const RECORD_FPS: u32 = 30;

/// Records frames to an H.264 MP4 as they arrive. The sound isn't recorded.
pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
    /// A float frame as 8 bit, kept for the next.
    raw: Vec<Argb<u8>>,
}

impl Encoder {
    pub fn record(size: (u32, u32), out: &Path) -> Result<Self, String> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "argb", "-s",
            ])
            .arg(format!("{}x{}", size.0, size.1))
            .args(["-use_wallclock_as_timestamps", "1", "-i", "-"])
            .args(["-vf", &format!("fps={RECORD_FPS}"), "-c:v", "libx264"])
            .args(["-pix_fmt", "yuv420p", "-crf", "18"])
            .arg(out)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("couldn't start ffmpeg: {e}"))?;
        let stdin = child.stdin.take();
        Ok(Self {
            child,
            stdin,
            raw: Vec::new(),
        })
    }

    /// Writes one output frame as it is.
    pub fn write(&mut self, pixels: &[Argb<u8>]) -> Result<(), String> {
        self.stdin
            .as_mut()
            .expect("open until finish")
            .write_all(bytemuck::cast_slice(pixels))
            .map_err(|_| "ffmpeg stopped accepting frames".to_string())
    }

    /// Writes one output frame of a GPU effect (float RGBA).
    pub fn write_float(&mut self, pixels: &[f32]) -> Result<(), String> {
        let mut raw = std::mem::take(&mut self.raw);
        raw.clear();
        raw.extend(pixels.as_chunks::<4>().0.iter().map(|&[r, g, b, a]| Argb {
            alpha: u8::from_unit(a),
            red: u8::from_unit(r),
            green: u8::from_unit(g),
            blue: u8::from_unit(b),
        }));
        let written = self.write(&raw);
        self.raw = raw;
        written
    }

    /// Closes the input and waits for ffmpeg to finish the file.
    pub fn finish(mut self) -> Result<(), String> {
        drop(self.stdin.take());
        match self.child.wait() {
            Ok(status) if status.success() => Ok(()),
            Ok(status) => Err(format!("ffmpeg failed ({status})")),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // Only reached without `finish` on cancel or error: stop and leave the partial file.
        if self.stdin.take().is_some() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Plays a file's sound through ffplay (no window), from `start` seconds. None
/// for cameras.
pub struct Sound(Child);

impl Sound {
    pub fn play(info: &Info, start: f64) -> Option<Self> {
        let path = info.file()?;
        Command::new("ffplay")
            .args(["-v", "quiet", "-nodisp", "-autoexit", "-vn", "-ss"])
            .arg(format!("{start:.3}"))
            .arg(path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .ok()
            .map(Self)
    }
}

impl Drop for Sound {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_for(what: &str, mut ready: impl FnMut() -> bool) {
        let until = Instant::now() + Duration::from_secs(10);
        while !ready() {
            assert!(Instant::now() < until, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// A 3 s, 25 fps clip for one test to play.
    fn test_clip(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vfx-preview-{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("clip.mp4");
        let made = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc2=size=160x120:rate=25:duration=3")
            .args(["-pix_fmt", "yuv420p"])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(made.success());
        clip
    }

    /// Seeks paused and playing, and reports the end. Needs ffmpeg:
    /// `cargo test -p vfx-preview -- --ignored`.
    #[test]
    #[ignore]
    fn seeks_pauses_and_ends() {
        let video = Video::open(Input::File(test_clip("seek-test")), 1920).unwrap();
        assert!(video.playing());
        let mut pixels = Vec::new();

        // Paused seek: one frame, and the position stays put.
        video.seek(2.0, false).unwrap();
        assert!(!video.playing());
        wait_for("the paused frame", || {
            video.take_frame(&mut pixels).is_some()
        });
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            (video.position() - 2.0).abs() < 0.05,
            "{}",
            video.position()
        );
        assert!(!video.ended());

        // Playing on from 2 s reaches the end of the 3 s clip in about a second, with its
        // last frames taken.
        video.seek(2.0, true).unwrap();
        wait_for("the end", || {
            video.take_frame(&mut pixels);
            video.ended()
        });
        assert!(video.position() > 2.8, "{}", video.position());
    }

    /// A loop that falls behind gets the newest frames in order, each with its own time, and
    /// the clock stays with the frame it took.
    #[test]
    #[ignore]
    fn slow_loop_drops_oldest_frames() {
        let video = Video::open(Input::File(test_clip("queue-test")), 1920).unwrap();
        wait_for("the first frame", || video.take_arrived_count() > 0);
        std::thread::sleep(Duration::from_millis(400));
        let arrived = video.take_arrived_count();
        assert!(arrived <= 13, "{arrived} frames in 0.4 s at 25 fps");

        let mut pixels = Vec::new();
        let first = video.take_frame(&mut pixels).expect("a waiting frame");
        assert!(first > 0.2, "{first} is an old frame");
        assert_eq!(video.position(), first);
        let second = video.take_frame(&mut pixels).expect("a second one");
        assert_eq!(video.position(), second);
        assert!(
            (second - first - 1.0 / 25.0).abs() < 1e-3,
            "{first} {second}"
        );
    }

    /// One buffer taken in turn from videos of two sizes, as a layer's is when its video
    /// changes: a video only ever decodes into buffers of its own size.
    #[test]
    fn wrong_size_buffer_not_decoded() {
        let clip = test_clip("sizes-test");
        let large = Video::open(Input::File(clip.clone()), 160).unwrap();
        let small = Video::open(Input::File(clip), 80).unwrap();
        let mut pixels = Vec::new();
        for video in [&large, &small, &large, &small] {
            let size = video.info.frame_pixels();
            wait_for("a frame", || video.take_newest(&mut pixels).is_some());
            assert_eq!(pixels.len(), size);
            let spare = &video.shared.lock().unwrap().spare;
            assert!(spare.iter().all(|buffer| buffer.len() == size));
        }
    }

    /// The bytes of a frame are ffmpeg's plain RGBA decode, in ARGB order: the frame that
    /// reaches a history is the one the preview always decoded.
    #[test]
    fn decoding_keeps_rgba_bytes() {
        let clip = test_clip("bytes-test");
        for side in [160, 80] {
            let info = Info::probe(Input::File(clip.clone()), side).unwrap();
            let (w, h) = info.size;
            let filter = format!("scale={w}:{h},fps={}", info.rate);
            let plain = Command::new("ffmpeg")
                .args(["-v", "error", "-i"])
                .arg(&clip)
                .args(["-an", "-vf", &filter, "-frames:v", "1"])
                .args(["-pix_fmt", "rgba", "-f", "rawvideo", "-"])
                .output()
                .unwrap()
                .stdout;

            let video = Video::start(info).unwrap();
            let mut pixels = Vec::new();
            wait_for("a frame", || video.take_frame(&mut pixels).is_some());
            assert_eq!(pixels.len() * 4, plain.len());
            for (p, rgba) in pixels.iter().zip(plain.as_chunks::<4>().0) {
                assert_eq!([p.red, p.green, p.blue, p.alpha], *rgba);
            }
        }
    }
}

#[cfg(test)]
mod record_tests {
    use super::Encoder;

    #[test]
    fn records_frames_to_an_mp4() {
        let out = std::env::temp_dir().join("vfx-preview-record-test.mp4");
        let mut encoder = Encoder::record((64, 48), &out).expect("ffmpeg starts");
        let frame = vec![vfx_core::cpu::Argb::<u8>::default(); 64 * 48];
        let float = vec![0.5f32; 64 * 48 * 4];
        for _ in 0..5 {
            encoder.write(&frame).expect("frame written");
            encoder.write_float(&float).expect("frame written");
            std::thread::sleep(std::time::Duration::from_millis(40));
        }
        encoder.finish().expect("ffmpeg finishes");
        let len = std::fs::metadata(&out).expect("file written").len();
        assert!(len > 0, "{len}");
        let _ = std::fs::remove_file(&out);
    }
}
