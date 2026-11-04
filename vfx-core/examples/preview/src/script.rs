//! Scripted runs: command line flags that set the preview up and measure it, so a
//! measurement is a command and needs no clicks. With no flags the preview starts as usual.

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use vfx_core::{ParamKind, ParamSpec, ParamValue};

use crate::catalog;
use crate::effects::PATTERNS;
use crate::stats::{Counts, Snapshot, SourceRate, What};

pub const USAGE: &str = "\
vfx-preview [FLAGS] [FILE]

  FILE                 a video or image to start with (same as --input FILE)
  --effect SLUG        the effect to show, as --list names it (segcam-motion)
  --mode NAME          the effect's mode: an option of its menu (Swarm, \"Motion matte\")
  --param NAME=VALUE   a parameter by its name; repeatable. Numbers, true/false, an
                       option's name or index, r,g,b,a for colours, x,y for points
  --input SOURCE       pattern[:NAME|N], camera[:NAME|N], or a video or image path
  --size WxH           the frame size (patterns, and videos, webcams and stills fitted into it)
  --detect-faces       tick the Expressions panel's Detect faces
  --stats              print the stats bar to stdout once a second
  --frames N           exit after N frames
  --seconds N          exit after N seconds
  --quiet              no window on screen (frames unpaced, not shown)
  --list               print the effects and their parameters, then exit
  --help               print this, then exit
";

/// What the command line asked for.
#[derive(Default, Debug, PartialEq)]
pub struct Options {
    pub effect: Option<String>,
    pub mode: Option<String>,
    /// `NAME=VALUE` pairs, checked against the effect's parameters once it's picked.
    pub params: Vec<(String, String)>,
    pub input: Option<Source>,
    pub size: Option<(u32, u32)>,
    pub detect_faces: bool,
    pub stats: bool,
    pub stop: Option<Stop>,
    pub quiet: bool,
    pub list: bool,
    pub help: bool,
}

#[derive(Debug, PartialEq)]
pub enum Source {
    /// A pattern by name or index; empty for the first.
    Pattern(String),
    /// A webcam by part of its name or its index; empty for the first.
    Camera(String),
    File(PathBuf),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stop {
    Frames(u64),
    Seconds(f64),
}

impl Options {
    /// Whether anything beyond the old `--effect SLUG [FILE]` was asked for. Mistakes in
    /// those flags end the run instead of showing in the panel.
    pub fn scripted(&self) -> bool {
        self.mode.is_some()
            || !self.params.is_empty()
            || self.size.is_some()
            || self.detect_faces
            || self.stats
            || self.stop.is_some()
            || self.quiet
            || matches!(self.input, Some(Source::Pattern(_) | Source::Camera(_)))
    }
}

pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Options, String> {
    let mut options = Options::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let flag = arg.to_string_lossy().into_owned();
        let mut value = || {
            args.next()
                .map(|a| a.to_string_lossy().into_owned())
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--effect" => options.effect = Some(value()?),
            "--mode" => options.mode = Some(value()?),
            "--param" => {
                let pair = value()?;
                let (name, v) = pair
                    .split_once('=')
                    .ok_or_else(|| format!("--param {pair}: expected NAME=VALUE"))?;
                options
                    .params
                    .push((name.trim().to_owned(), v.trim().to_owned()));
            }
            "--input" => options.input = Some(source(&value()?)),
            "--size" => options.size = Some(size(&value()?)?),
            "--detect-faces" => options.detect_faces = true,
            "--stats" => options.stats = true,
            "--frames" => {
                let v = value()?;
                let n = v
                    .parse()
                    .map_err(|_| format!("--frames {v}: not a count"))?;
                options.stop = Some(Stop::Frames(n));
            }
            "--seconds" => {
                let v = value()?;
                let s: f64 = v
                    .parse()
                    .map_err(|_| format!("--seconds {v}: not a number"))?;
                if !s.is_finite() || s <= 0.0 {
                    return Err(format!("--seconds {v}: must be above 0"));
                }
                options.stop = Some(Stop::Seconds(s));
            }
            "--quiet" => options.quiet = true,
            "--list" => options.list = true,
            "--help" | "-h" => options.help = true,
            _ if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ => options.input = Some(Source::File(arg.into())),
        }
    }
    Ok(options)
}

fn source(text: &str) -> Source {
    let (kind, rest) = text.split_once(':').unwrap_or((text, ""));
    match kind {
        "pattern" => Source::Pattern(rest.to_owned()),
        "camera" => Source::Camera(rest.to_owned()),
        _ => Source::File(text.into()),
    }
}

fn size(text: &str) -> Result<(u32, u32), String> {
    let parsed = text
        .split_once(['x', 'X'])
        .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)));
    match parsed {
        Some((w, h)) if w > 0 && h > 0 => Ok((w, h)),
        _ => Err(format!("--size {text}: expected WxH, as 1920x1080")),
    }
}

/// The index of `wanted` among `names`, by name (any case and spacing) or by index.
pub fn pick(names: &[&str], wanted: &str) -> Option<usize> {
    if let Ok(i) = wanted.parse::<usize>() {
        return (i < names.len()).then_some(i);
    }
    let wanted = catalog::slug(wanted);
    names.iter().position(|n| catalog::slug(n) == wanted)
}

pub fn pattern(wanted: &str) -> Result<usize, String> {
    if wanted.is_empty() {
        return Ok(0);
    }
    pick(PATTERNS, wanted).ok_or_else(|| {
        format!(
            "no pattern called {wanted}; there are {}",
            PATTERNS.join(", ")
        )
    })
}

/// The webcam `wanted` names: part of its name, its index, or the first.
pub fn camera(cameras: &[String], wanted: &str) -> Result<String, String> {
    let found = if wanted.is_empty() {
        cameras.first()
    } else if let Ok(i) = wanted.parse::<usize>() {
        cameras.get(i)
    } else {
        let wanted = wanted.to_lowercase();
        cameras.iter().find(|c| c.to_lowercase().contains(&wanted))
    };
    found.cloned().ok_or_else(|| match cameras.len() {
        0 => "no webcams found".to_owned(),
        _ => format!("no webcam {wanted}; there are {}", cameras.join(", ")),
    })
}

/// `text` as a value for `spec`.
pub fn value(spec: &ParamSpec, text: &str) -> Result<ParamValue, String> {
    let bad = || format!("--param {}={text}: {}", spec.name, expected(spec));
    let numbers =
        || -> Option<Vec<f32>> { text.split(',').map(|n| n.trim().parse().ok()).collect() };
    match spec.kind {
        ParamKind::Float { .. } => text.parse().map(ParamValue::Float).map_err(|_| bad()),
        ParamKind::Int { options, .. } if !options.is_empty() => pick(options, text)
            .map(|i| ParamValue::Int(i as i32))
            .ok_or_else(bad),
        ParamKind::Int { .. } => text.parse().map(ParamValue::Int).map_err(|_| bad()),
        ParamKind::Bool { .. } => match text.to_ascii_lowercase().as_str() {
            "true" | "on" | "yes" | "1" => Ok(ParamValue::Bool(true)),
            "false" | "off" | "no" | "0" => Ok(ParamValue::Bool(false)),
            _ => Err(bad()),
        },
        ParamKind::Color { .. } => match numbers().as_deref() {
            Some(&[r, g, b]) => Ok(ParamValue::Color([r, g, b, 1.0])),
            Some(&[r, g, b, a]) => Ok(ParamValue::Color([r, g, b, a])),
            _ => Err(bad()),
        },
        ParamKind::Point { .. } => match numbers().as_deref() {
            Some(&[x, y]) => Ok(ParamValue::Point([x, y])),
            _ => Err(bad()),
        },
    }
}

fn expected(spec: &ParamSpec) -> String {
    match spec.kind {
        ParamKind::Float { min, max, .. } => format!("a number from {min} to {max}"),
        ParamKind::Int { options, .. } if !options.is_empty() => {
            format!("one of {}", options.join(", "))
        }
        ParamKind::Int { min, max, .. } => format!("a whole number from {min} to {max}"),
        ParamKind::Bool { .. } => "true or false".to_owned(),
        ParamKind::Color { .. } => "r,g,b or r,g,b,a".to_owned(),
        ParamKind::Point { .. } => "x,y".to_owned(),
    }
}

/// The effect's menu parameter that has an option called `mode`, and that option's value.
pub fn mode(specs: &[ParamSpec], mode: &str) -> Result<(&'static str, ParamValue), String> {
    specs
        .iter()
        .find_map(|s| match s.kind {
            ParamKind::Int { options, .. } if !options.is_empty() => {
                pick(options, mode).map(|i| (s.name, ParamValue::Int(i as i32)))
            }
            _ => None,
        })
        .ok_or_else(|| format!("--mode {mode}: the effect has no mode by that name"))
}

/// Every effect and its parameters, for `--list`.
pub fn list() -> String {
    let mut out = String::new();
    for entry in catalog::entries() {
        out.push_str(&format!("{}  ({})\n", entry.slug, entry.name));
        for spec in (entry.make)().params() {
            out.push_str(&format!("  {}  {}\n", spec.name, expected(spec)));
        }
    }
    out
}

/// Counts frames for `--stats`, `--frames` and `--seconds`, from the first frame on. A
/// frame here is a pass of the app's loop, which is what `--frames` counts.
pub struct Meter {
    print: bool,
    stop: Option<Stop>,
    start: Instant,
    started: bool,
    frames: u64,
    /// The longest render of the run.
    worst: Duration,
    /// The second being counted: its start and longest render.
    window: Instant,
    window_worst: Duration,
    /// The totals at the first frame and at the start of the second being counted.
    first_counts: Counts,
    window_counts: Counts,
    done: bool,
}

impl Meter {
    pub fn new(print: bool, stop: Option<Stop>) -> Self {
        let now = Instant::now();
        Self {
            print,
            stop,
            start: now,
            started: false,
            frames: 0,
            worst: Duration::ZERO,
            window: now,
            window_worst: Duration::ZERO,
            first_counts: Counts::default(),
            window_counts: Counts::default(),
            done: false,
        }
    }

    /// Counts a frame. True once the run should end.
    pub fn frame(&mut self, stats: &Snapshot) -> bool {
        if self.done {
            return true;
        }
        let now = Instant::now();
        if !self.started {
            self.started = true;
            (self.start, self.window) = (now, now);
            (self.first_counts, self.window_counts) = (stats.counts, stats.counts);
        }
        self.frames += 1;
        if let Some(render) = stats.last_render {
            self.worst = self.worst.max(render);
            self.window_worst = self.window_worst.max(render);
        }
        let elapsed = now.duration_since(self.start).as_secs_f64();
        let span = now.duration_since(self.window).as_secs_f64();
        if span >= 1.0 {
            if self.print {
                let done = stats.counts.since(self.window_counts);
                println!("{}", line(elapsed, span, self.window_worst, stats, done));
            }
            (self.window, self.window_worst) = (now, Duration::ZERO);
            self.window_counts = stats.counts;
        }
        self.done = match self.stop {
            Some(Stop::Frames(n)) => self.frames >= n,
            Some(Stop::Seconds(s)) => elapsed >= s,
            None => false,
        };
        if self.done && self.print {
            let seconds = elapsed.max(f64::EPSILON);
            let done = stats.counts.since(self.first_counts);
            println!(
                "done frames={} seconds={elapsed:.2} {}",
                self.frames,
                counts_text(seconds, self.worst, &stats.what, done)
            );
        }
        self.done
    }
}

/// The counts as keys: the effect and where it runs, the source's rate, renders a second
/// with their mean and longest time in ms when the effect has one, and new frames a second.
fn counts_text(seconds: f64, worst: Duration, what: &What, done: Counts) -> String {
    let source = match what.source {
        SourceRate::Pattern => "pattern".to_owned(),
        SourceRate::Still => "still".to_owned(),
        SourceRate::Video => format!("{:.1}", done.arrived as f64 / seconds),
    };
    let mut text = format!(
        "effect={} on={} source={source} rendered={:.1}",
        what.slug,
        if what.on_cpu { "cpu" } else { "gpu" },
        done.rendered as f64 / seconds,
    );
    if let Some(ms) = done.render_ms() {
        text += &format!(
            " rendered_ms={ms:.1} worst_ms={:.1}",
            worst.as_secs_f64() * 1000.0
        );
    }
    text += &format!(" screen={:.1}", done.screen as f64 / seconds);
    text
}

/// One `--stats` line: the counts over the last `span` seconds, then the machine.
fn line(elapsed: f64, span: f64, worst: Duration, s: &Snapshot, done: Counts) -> String {
    let gpu = s.gpu.map_or("n/a".to_owned(), |g| format!("{g:.1}"));
    let mut text = format!(
        "stats t={elapsed:.1} {} size={}x{} cpu={:.1} app_cpu={:.1} mem_mb={} gpu={gpu}",
        counts_text(span, worst, &s.what, done),
        s.what.size.0,
        s.what.size.1,
        s.system_cpu,
        s.cpu,
        s.memory_mb,
    );
    if let Some(draw) = &s.report.draw {
        text += &format!(" draw={draw:?}");
    }
    if let Some(faces) = &s.report.faces {
        text += &format!(" faces={faces:?}");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(text: &[&str]) -> Result<Options, String> {
        parse(text.iter().map(OsString::from))
    }

    #[test]
    fn no_flags_is_the_old_start() {
        let o = args(&["--effect", "segcam-motion", "clip.mp4"]).unwrap();
        assert_eq!(o.effect.as_deref(), Some("segcam-motion"));
        assert_eq!(o.input, Some(Source::File("clip.mp4".into())));
        assert!(!o.scripted());
        assert_eq!(args(&[]).unwrap(), Options::default());
    }

    #[test]
    fn flags_parse() {
        let o = args(&[
            "--mode",
            "Swarm",
            "--param",
            "grid=96",
            "--input",
            "pattern:checker",
            "--size",
            "640x360",
            "--detect-faces",
            "--stats",
            "--seconds",
            "5",
            "--quiet",
        ])
        .unwrap();
        assert!(o.detect_faces);
        assert_eq!(o.mode.as_deref(), Some("Swarm"));
        assert_eq!(o.params, [("grid".to_owned(), "96".to_owned())]);
        assert_eq!(o.input, Some(Source::Pattern("checker".into())));
        assert_eq!(o.size, Some((640, 360)));
        assert_eq!(o.stop, Some(Stop::Seconds(5.0)));
        assert!(o.stats && o.quiet && o.scripted());
        assert_eq!(
            args(&["--input", "camera"]).unwrap().input,
            Some(Source::Camera("".into()))
        );
    }

    #[test]
    fn mistakes_are_errors() {
        assert!(args(&["--size", "640"]).is_err());
        assert!(args(&["--size", "0x10"]).is_err());
        assert!(args(&["--frames", "many"]).is_err());
        assert!(args(&["--seconds", "0"]).is_err());
        assert!(args(&["--param", "grid"]).is_err());
        assert!(args(&["--effect"]).is_err());
        assert!(args(&["--fast"]).is_err());
    }

    #[test]
    fn values_follow_the_spec() {
        let menu = ParamSpec::choice("output", &["Overlay", "Motion matte", "Swarm"], 0);
        assert_eq!(value(&menu, "motion-matte"), Ok(ParamValue::Int(1)));
        assert_eq!(value(&menu, "Motion matte"), Ok(ParamValue::Int(1)));
        assert_eq!(value(&menu, "2"), Ok(ParamValue::Int(2)));
        assert!(value(&menu, "3").is_err());
        let toggle = ParamSpec::toggle("mirror", false);
        assert_eq!(value(&toggle, "on"), Ok(ParamValue::Bool(true)));
        assert!(value(&toggle, "maybe").is_err());
        let colour = ParamSpec::color("box_color", [0.0; 4]);
        assert_eq!(
            value(&colour, "1, 0.5, 0"),
            Ok(ParamValue::Color([1.0, 0.5, 0.0, 1.0]))
        );
        assert!(value(&ParamSpec::float("sensitivity", 0.0, 1.0, 0.5), "x").is_err());
    }

    #[test]
    fn modes_are_found_by_option() {
        let specs = [
            ParamSpec::float("sensitivity", 0.0, 1.0, 0.5),
            ParamSpec::choice("output", &["Overlay", "Motion matte", "Swarm"], 0),
        ];
        assert_eq!(mode(&specs, "swarm"), Ok(("output", ParamValue::Int(2))));
        assert!(mode(&specs, "Collage").is_err());
    }

    #[test]
    fn segcam_has_its_modes() {
        let segcam = catalog::find_slug("segcam-motion").unwrap();
        let specs = (catalog::entries()[segcam].make)().params().to_vec();
        for m in ["Overlay", "Motion matte", "Box matte", "Swarm"] {
            assert_eq!(mode(&specs, m).map(|(name, _)| name), Ok("output"), "{m}");
        }
    }

    #[test]
    fn the_meter_stops_after_its_frames() {
        let mut meter = Meter::new(false, Some(Stop::Frames(3)));
        let stats = Snapshot::default();
        assert!(!meter.frame(&stats));
        assert!(!meter.frame(&stats));
        assert!(meter.frame(&stats));
        assert!(meter.frame(&stats));
        assert_eq!(meter.frames, 3);
    }

    /// The keys name what they count; the render times are there for a CPU effect only.
    #[test]
    fn a_stats_line_names_what_it_counts() {
        let s = Snapshot {
            what: What {
                effect: "segcam Motion",
                slug: "segcam-motion",
                on_cpu: true,
                source: SourceRate::Video,
                size: (1920, 1080),
            },
            ..Default::default()
        };
        let done = Counts {
            rendered: 60,
            render_time: Duration::from_millis(600),
            screen: 30,
            arrived: 59,
        };
        assert_eq!(
            line(1.0, 1.0, Duration::from_millis(20), &s, done),
            "stats t=1.0 effect=segcam-motion on=cpu source=59.0 rendered=60.0 rendered_ms=10.0 \
             worst_ms=20.0 screen=30.0 size=1920x1080 cpu=0.0 app_cpu=0.0 mem_mb=0 gpu=n/a"
        );
        let s = Snapshot {
            what: What {
                effect: "Gaussian blur",
                slug: "gaussian-blur",
                on_cpu: false,
                source: SourceRate::Pattern,
                size: (0, 0),
            },
            ..Default::default()
        };
        let done = Counts {
            rendered: 120,
            render_time: Duration::ZERO,
            screen: 120,
            arrived: 0,
        };
        assert_eq!(
            line(2.0, 1.0, Duration::ZERO, &s, done),
            "stats t=2.0 effect=gaussian-blur on=gpu source=pattern rendered=120.0 screen=120.0 \
             size=0x0 cpu=0.0 app_cpu=0.0 mem_mb=0 gpu=n/a"
        );
    }

    #[test]
    fn inputs_pick_patterns_and_cameras() {
        assert_eq!(pattern(""), Ok(0));
        assert_eq!(pattern("colour-bars"), Ok(1));
        assert!(pattern("stripes").is_err());
        let cams = [
            "Integrated Webcam".to_owned(),
            "OBS Virtual Camera".to_owned(),
        ];
        assert_eq!(camera(&cams, "obs").as_deref(), Ok("OBS Virtual Camera"));
        assert_eq!(camera(&cams, "").as_deref(), Ok("Integrated Webcam"));
        assert!(camera(&cams, "5").is_err());
        assert!(camera(&[], "").is_err());
    }
}
