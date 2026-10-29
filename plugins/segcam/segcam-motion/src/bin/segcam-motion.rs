//! The effect's render path without After Effects: raw ARGB frames in on stdin, rendered
//! frames out on stdout, through exactly the code the effect runs. Pipe it between two
//! ffmpegs to see what the effect will do to a clip:
//!
//! ```text
//! ffmpeg -i in.mov -f rawvideo -pix_fmt argb - |
//!   segcam-motion 1920 1080 --mode swarm |
//!   ffmpeg -f rawvideo -pix_fmt argb -s 1920x1080 -r 30 -i - -pix_fmt yuv420p out.mp4
//! ```
//!
//! Options and defaults match segcam's own `segcam-motion`, so the two can be compared.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use segcam_motion::pixel::Depth;
use segcam_motion::{Argb, Detect, OutputMode, RenderSettings, SwarmBackground, View, ViewMut};
use vfx_core::cpu::{FrameSource, Image};

const USAGE: &str = "usage: segcam-motion WIDTH HEIGHT [options] < argb-frames > argb-frames
  --mode overlay|matte|boxmatte|swarm   (overlay)
  --sensitivity 0..100                  (86.4)
  --adaptation percent                  (6)
  --background frames                   (16)
  --min-size percent                    (0.25)
  --max-segments n                      (16)
  --grid columns                        (192)
  --trail frames  --max-tiles n         (24, 100)
  --detect motion|threshold|faces       (motion)
  --face  --eyes  --mouth  --body       (off; segcam-face's models)
  --level 0..255  --auto-level  --invert   (170; threshold)
  --swarm-background transparent|source|black (transparent)
  --live-windows                       (swarm windows show the current frame)
  --mirror  --no-boxes  --no-labels
  --float                              (render in float, still raw 8-bit I/O)
  --straight-alpha                     (input and output use straight alpha)";

/// The last few frames, oldest first: ffmpeg's argb is `Argb<u8>` byte for byte.
struct Ring<C> {
    frames: VecDeque<Vec<Argb<C>>>,
    width: i32,
    height: i32,
}

impl<C: Depth> FrameSource for Ring<C> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        f(self
            .frames
            .get(index)
            .map(|p| C::image(View::from_pixels(p, self.width, self.height))));
    }
}

fn parse(args: &[String]) -> Option<(i32, i32, RenderSettings)> {
    let width: i32 = args.first()?.parse().ok().filter(|&w| w > 0)?;
    let height: i32 = args.get(1)?.parse().ok().filter(|&h| h > 0)?;
    let mut s = RenderSettings::default();
    let mut rest = args[2..].iter();
    while let Some(arg) = rest.next() {
        let mut value = || rest.next().map(String::as_str);
        match arg.as_str() {
            "--mode" => {
                s.mode = match value()? {
                    "overlay" => OutputMode::Overlay,
                    "matte" => OutputMode::Matte,
                    "boxmatte" => OutputMode::BoxMatte,
                    "swarm" => OutputMode::Swarm,
                    _ => return None,
                }
            }
            "--detect" => {
                s.detect = match value()? {
                    "motion" => Detect::Motion,
                    "threshold" => Detect::Threshold,
                    "faces" => Detect::FacesOnly,
                    _ => return None,
                }
            }
            "--level" => s.threshold.level = value()?.parse().ok()?,
            "--auto-level" => s.threshold.auto = true,
            "--invert" => s.threshold.invert = true,
            "--face" => s.faces.face = true,
            "--eyes" => s.faces.eyes = true,
            "--mouth" => s.faces.mouth = true,
            "--body" => s.faces.body = true,
            "--swarm-background" => {
                s.swarm_background = match value()? {
                    "transparent" => SwarmBackground::Transparent,
                    "source" => SwarmBackground::Source,
                    "black" => SwarmBackground::Black,
                    _ => return None,
                }
            }
            "--sensitivity" => s.motion.sensitivity = value()?.parse::<f64>().ok()? / 100.0,
            "--adaptation" => s.motion.adaptation = (value()?.parse::<f64>().ok()? / 100.0) as f32,
            "--background" => s.background_frames = value()?.parse().ok()?,
            "--min-size" => s.motion.min_area_fraction = value()?.parse::<f64>().ok()? / 100.0,
            "--max-segments" => s.motion.max_segments = value()?.parse().ok()?,
            "--grid" => s.grid_width = value()?.parse().ok()?,
            "--trail" => s.trail_frames = value()?.parse().ok()?,
            "--max-tiles" => s.max_tiles = value()?.parse().ok()?,
            "--straight-alpha" => s.straight_alpha = true,
            "--mirror" => s.mirror = true,
            "--live-windows" => s.live_windows = true,
            "--no-boxes" => s.boxes = false,
            "--no-labels" => s.labels = false,
            _ => return None,
        }
    }
    Some((width, height, s))
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let float = args.iter().any(|a| a == "--float");
    args.retain(|a| a != "--float");
    let Some((width, height, settings)) = parse(&args) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };

    if float {
        run::<f32>(width, height, settings)
    } else {
        run::<u8>(width, height, settings)
    }
}

fn run<C: Depth>(width: i32, height: i32, settings: RenderSettings) -> ExitCode {
    let pixels = width as usize * height as usize;
    let keep = segcam_motion::previous_frames_needed(&settings) as usize + 1;
    let mut ring = Ring {
        frames: VecDeque::new(),
        width,
        height,
    };
    let mut out = vec![Argb::<C>::default(); pixels];
    let (mut stdin, mut stdout) = (io::stdin().lock(), io::stdout().lock());
    let mut bytes = vec![0u8; pixels * 4];
    let mut frame = 0i64;

    while stdin.read_exact(&mut bytes).is_ok() {
        let argb = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&[alpha, red, green, blue]| Argb {
                alpha: C::from_unit(f32::from(alpha) / 255.0),
                red: C::from_unit(f32::from(red) / 255.0),
                green: C::from_unit(f32::from(green) / 255.0),
                blue: C::from_unit(f32::from(blue) / 255.0),
            })
            .collect();
        ring.frames.push_back(argb);
        if ring.frames.len() > keep {
            ring.frames.pop_front();
        }

        // Like After Effects: every frame rendered from scratch out of the frames before it.
        let count = ring.frames.len();
        let mut view = ViewMut::from_pixels(&mut out, width, height);
        segcam_motion::render(&settings, &mut ring, count, frame, &mut view);

        for (dst, p) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(&out) {
            *dst = [p.alpha, p.red, p.green, p.blue]
                .map(|c| (c.to_unit().clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        if stdout.write_all(&bytes).is_err() {
            return ExitCode::FAILURE;
        }
        frame += 1;
    }
    eprintln!("segcam-motion: {frame} frames");
    ExitCode::SUCCESS
}
