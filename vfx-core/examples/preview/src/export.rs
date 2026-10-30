//! Rendering a video through the effects and expressions to an MP4 next to the original.
//!
//! Every source frame is processed once, in order, at the file's own frame rate, so
//! time-based expressions and audio bands land exactly where they do in the file; it runs a
//! few frames per redraw so the window keeps updating (and shows the frames going by).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use vfx_core::Gpu;

use crate::audio::Audio;
use crate::controls::Controls;
use crate::effects::Effects;
use crate::face::Detector;
use crate::modulation::{self, Expressions};
use crate::video::{Decoder, Encoder, Info};

pub struct Export {
    info: Info,
    decoder: Decoder,
    encoder: Option<Encoder>,
    frame: u64,
    /// Finds the face in every frame, when face tracking is on.
    detector: Option<Detector>,
    /// Face values, held while no face is found, as live tracking does.
    face: modulation::Face,
    pub out: PathBuf,
}

impl Export {
    pub fn start(info: &Info, track_faces: bool) -> Result<Self, String> {
        let file = info.file().ok_or("only video files can be rendered")?;
        let out = output_path(file);
        Ok(Self {
            info: info.clone(),
            detector: track_faces.then(Detector::new).transpose()?,
            decoder: Decoder::open(info)?,
            encoder: Some(Encoder::create(info, &out)?),
            frame: 0,
            face: modulation::FACE_NONE,
            out,
        })
    }

    pub fn progress(&self) -> f32 {
        let total = (self.info.duration * self.info.fps).ceil().max(1.0);
        (self.frame as f64 / total).min(1.0) as f32
    }

    /// Renders frames for about `budget`. Ok(true) once the file is written.
    pub fn step(
        &mut self,
        gpu: &Gpu,
        effects: &mut Effects,
        base: &Controls,
        exprs: &mut Expressions,
        audio: &Audio,
        budget: Duration,
    ) -> Result<bool, String> {
        // Audio expressions need the whole track; it decodes in the background on load.
        if !audio.ready() {
            return Ok(false);
        }
        let until = Instant::now() + budget;
        let dt = (1.0 / self.info.fps) as f32;
        while Instant::now() < until {
            let Some(pixels) = self.decoder.next_frame() else {
                self.encoder.take().expect("finished once").finish()?;
                return Ok(true);
            };
            let t = self.frame as f64 / self.info.fps;
            if let Some(detector) = &mut self.detector {
                match detector.detect(&pixels, self.info.size) {
                    Some(face) => self.face = face,
                    None => self.face[0] = 0.0,
                }
            }
            let sources =
                modulation::sources(t, Some((t, self.info.duration)), audio.levels(t), self.face);
            let values = exprs.apply(base.effect, base.values(), &sources, dt);
            effects.write_source(gpu, &pixels);
            let out = effects
                .render(gpu, base, &values, t, true)
                .expect("capture returns the frame");
            self.encoder
                .as_mut()
                .expect("open until finished")
                .write(&out)?;
            self.frame += 1;
        }
        Ok(false)
    }
}

/// `clip_fx.mp4` beside `clip.mov`, or `clip_fx2.mp4` and so on if that exists.
fn output_path(source: &Path) -> PathBuf {
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "video".into());
    let dir = source.parent().unwrap_or(Path::new("."));
    (1..)
        .map(|n| {
            let suffix = if n == 1 { String::new() } else { n.to_string() };
            dir.join(format!("{stem}_fx{suffix}.mp4"))
        })
        .find(|p| !p.exists())
        .expect("some name is free")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulation::Target;
    use std::process::Command;

    /// Renders a generated clip with sound through a bass-driven expression. Needs ffmpeg and
    /// a GPU, so it only runs when asked: `cargo test -p vfx-preview -- --ignored`.
    #[test]
    #[ignore]
    fn renders_a_clip_with_its_sound() {
        let dir = std::env::temp_dir().join("vfx-preview-export-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("clip.mp4");
        let made = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc2=size=320x240:rate=25:duration=2")
            .args(["-f", "lavfi", "-i", "sine=frequency=100:duration=2"])
            .args(["-pix_fmt", "yuv420p", "-shortest"])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(made.success());

        let gpu = pollster::block_on(Gpu::headless()).unwrap();
        let info = Info::probe(crate::video::Input::File(clip.clone()), 1920).unwrap();
        assert_eq!(info.size, (320, 240));
        let mut effects = Effects::new(&gpu);
        effects.start_stream(&gpu, "clip".into(), info.size);
        let audio = Audio::load(&clip);
        let mut exprs = Expressions::default();
        let colour = crate::catalog::find("Colour Adjust").unwrap();
        exprs.set(Target::new(colour, "brightness"), "bass * 2 - 1");
        let mut export = Export::start(&info, true).unwrap();
        while !export
            .step(
                &gpu,
                &mut effects,
                &Controls::default(),
                &mut exprs,
                &audio,
                Duration::from_millis(50),
            )
            .unwrap()
        {}
        assert!(audio.levels(1.0)[1] > 0.5, "the 100 Hz tone reads as bass");

        let probe = Command::new("ffprobe")
            .args(["-v", "error", "-count_frames", "-show_entries"])
            .args(["stream=codec_type,nb_read_frames", "-of", "csv=p=0"])
            .arg(&export.out)
            .output()
            .unwrap();
        let streams = String::from_utf8_lossy(&probe.stdout);
        assert!(streams.contains("video,50"), "{streams}");
        assert!(streams.contains("audio"), "{streams}");
    }
}
