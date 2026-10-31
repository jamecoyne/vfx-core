//! Rows of a CPU frame shared out over a few threads, for the hosts that convert frames row
//! by row. Six threads at most: a conversion is bound by memory, and a hybrid CPU's slow
//! cores would otherwise set the pace. Short bands from a queue let a slow or busy core
//! hand rows to the others.

use std::sync::{LazyLock, Mutex};

const THREADS: usize = 6;
const BANDS_PER_THREAD: usize = 4;
/// Rows that keep a thread worth starting; smaller frames use fewer threads.
const ROWS_PER_THREAD: usize = 32;

static CORES: LazyLock<usize> =
    LazyLock::new(|| std::thread::available_parallelism().map_or(1, usize::from));

/// Runs `f` on each row of `rows` (`width` items each) with its row number, a band of rows
/// at a time on up to six threads. On one core, and in browsers, on the calling thread.
pub fn by_rows<T: Send>(rows: &mut [T], width: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if width == 0 {
        return;
    }
    let height = rows.len() / width;
    let threads = (*CORES)
        .min(THREADS)
        .min(height.div_ceil(ROWS_PER_THREAD))
        .max(1);
    on_threads(threads, rows, width, f);
}

fn on_threads<T: Send>(
    threads: usize,
    rows: &mut [T],
    width: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    if threads < 2 {
        for (i, row) in rows.chunks_mut(width).enumerate() {
            f(i, row);
        }
        return;
    }
    let height = rows.len() / width;
    let band = height.div_ceil(threads * BANDS_PER_THREAD).max(1);
    let bands = Mutex::new(rows.chunks_mut(band * width).enumerate());
    let next = || bands.lock().ok().and_then(|mut bands| bands.next());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                while let Some((i, chunk)) = next() {
                    for (j, row) in chunk.chunks_mut(width).enumerate() {
                        f(i * band + j, row);
                    }
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Converts a `width` wide frame from source rows `pitch` apart on `threads` threads,
    /// tagging each pixel with its row.
    fn convert(threads: usize, width: usize, height: usize, pitch: usize) -> Vec<u32> {
        let src: Vec<u32> = (0..pitch * height).map(|i| i as u32 * 7).collect();
        let mut dst = vec![u32::MAX; width * height];
        on_threads(threads, &mut dst, width, |y, row| {
            for (d, s) in row.iter_mut().zip(&src[y * pitch..][..width]) {
                *d = s ^ ((y as u32) << 24);
            }
        });
        dst
    }

    #[test]
    fn threads_match_a_single_thread() {
        // (width, height, pitch): a padded pitch, an odd height, fewer rows than threads,
        // one row, and more bands than rows.
        for (w, h, p) in [(5, 64, 8), (7, 37, 7), (3, 3, 4), (9, 1, 16), (2, 10, 2)] {
            let serial = convert(1, w, h, p);
            assert_eq!(convert(6, w, h, p), serial, "{w}x{h}, pitch {p}");
            assert_eq!(convert(2, w, h, p), serial, "{w}x{h}, pitch {p}");
            assert!(serial.iter().all(|&v| v != u32::MAX));
        }
    }

    #[test]
    fn empty_frames_are_left_alone() {
        let mut none: [u32; 0] = [];
        by_rows(&mut none, 4, |_, _| panic!("no rows"));
        let mut some = [1u32; 4];
        by_rows(&mut some, 0, |_, _| panic!("no width"));
        assert_eq!(some, [1; 4]);
    }
}
