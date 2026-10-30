//! A video's sound as numbers for control rules: overall level and four frequency
//! bands, from an FFT around any point in time.
//!
//! The whole track is decoded once (mono, 22 kHz) in the background, so the live
//! preview and exports analyse exactly the same samples at the same times.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, OnceLock};

const RATE: f32 = 22_050.0;
const FFT_SIZE: usize = 1024;
/// Loudness mapped to 0–1 between these (dBFS).
const FLOOR_DB: f32 = -60.0;

/// Band edges in Hz: bass, low mid, high mid, treble.
const BANDS: [(f32, f32); 4] = [
    (20.0, 250.0),
    (250.0, 1000.0),
    (1000.0, 4000.0),
    (4000.0, RATE / 2.0),
];

/// Level, then the four bands, each roughly 0–1.
pub type Levels = [f32; 5];

pub struct Audio {
    samples: Arc<OnceLock<Vec<f32>>>,
}

impl Audio {
    /// Starts decoding `path`'s first audio track. Until it's done (or if the file has
    /// no sound) every level reads 0.
    pub fn load(path: &Path) -> Self {
        let samples = Arc::new(OnceLock::new());
        let slot = Arc::clone(&samples);
        let path = path.to_path_buf();
        std::thread::spawn(move || {
            let decoded = Command::new("ffmpeg")
                .args(["-v", "error", "-i"])
                .arg(&path)
                .args(["-vn", "-ac", "1", "-ar", "22050", "-f", "f32le", "-"])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .ok()
                .and_then(|mut child| {
                    let mut bytes = Vec::new();
                    child.stdout.take()?.read_to_end(&mut bytes).ok()?;
                    let _ = child.wait();
                    Some(
                        bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| f32::from_le_bytes(*b))
                            .collect(),
                    )
                });
            let _ = slot.set(decoded.unwrap_or_default());
        });
        Self { samples }
    }

    /// Whether decoding has finished (with or without any sound).
    pub fn ready(&self) -> bool {
        self.samples.get().is_some()
    }

    /// The levels in the FFT window centred on `time` seconds.
    pub fn levels(&self, time: f64) -> Levels {
        let Some(samples) = self.samples.get().filter(|s| !s.is_empty()) else {
            return [0.0; 5];
        };
        let centre = (time * RATE as f64) as isize;
        let start = centre - FFT_SIZE as isize / 2;
        let mut re = [0.0f32; FFT_SIZE];
        let mut im = [0.0f32; FFT_SIZE];
        let mut window_sum = 0.0;
        let mut square_sum = 0.0;
        for (i, slot) in re.iter_mut().enumerate() {
            let s = usize::try_from(start + i as isize)
                .ok()
                .and_then(|j| samples.get(j))
                .copied()
                .unwrap_or(0.0);
            // Hann window.
            let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT_SIZE as f32).cos();
            *slot = s * w;
            window_sum += w;
            square_sum += s * s;
        }
        fft(&mut re, &mut im);

        let mut levels = [0.0; 5];
        levels[0] = to_unit((square_sum / FFT_SIZE as f32).sqrt());
        let hz_per_bin = RATE / FFT_SIZE as f32;
        for (band, (lo, hi)) in BANDS.iter().enumerate() {
            let bins =
                (lo / hz_per_bin).ceil() as usize..=((hi / hz_per_bin) as usize).min(FFT_SIZE / 2);
            let count = bins.clone().count().max(1) as f32;
            // Power mean of the bins' amplitudes, scaled so a full-scale sine reads 1.
            let power: f32 = bins
                .map(|k| {
                    let a = 2.0 * (re[k] * re[k] + im[k] * im[k]).sqrt() / window_sum;
                    a * a
                })
                .sum();
            levels[band + 1] = to_unit((power / count).sqrt());
        }
        levels
    }
}

/// An amplitude as 0–1 on a decibel scale from FLOOR_DB to 0 dBFS.
fn to_unit(amplitude: f32) -> f32 {
    let db = 20.0 * amplitude.max(1e-9).log10();
    ((db - FLOOR_DB) / -FLOOR_DB).clamp(0.0, 1.0)
}

/// In-place radix-2 FFT; the length must be a power of two.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -std::f32::consts::TAU / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (angle * k as f32).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_of(samples: Vec<f32>) -> Audio {
        let lock = OnceLock::new();
        let _ = lock.set(samples);
        Audio {
            samples: Arc::new(lock),
        }
    }

    fn sine(hz: f32) -> Vec<f32> {
        (0..22_050)
            .map(|i| (std::f32::consts::TAU * hz * i as f32 / RATE).sin())
            .collect()
    }

    #[test]
    fn a_tone_lands_in_its_band() {
        let bass = audio_of(sine(100.0)).levels(0.5);
        assert!(bass[1] > 0.8, "{bass:?}");
        assert!(bass[4] < 0.3, "{bass:?}");
        let treble = audio_of(sine(6000.0)).levels(0.5);
        assert!(treble[4] > treble[1], "{treble:?}");
    }

    #[test]
    fn silence_and_no_audio_read_zero() {
        assert_eq!(audio_of(vec![0.0; 4096]).levels(0.1), [0.0; 5]);
        assert_eq!(audio_of(Vec::new()).levels(0.1), [0.0; 5]);
    }
}
