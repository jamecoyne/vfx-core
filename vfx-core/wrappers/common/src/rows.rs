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

/// How many threads CPU frame work uses at most: all the cores, up to six.
pub fn worker_threads() -> usize {
    (*CORES).min(THREADS)
}

/// Threads kept for the life of the process, so that frame work starts none. A thread started per
/// frame costs a call into every loaded DLL, and NVIDIA's D3D driver keeps a record of each thread
/// ever started, which adds up over a show. The caller takes a share of the work, so one fewer is
/// enough. They outlive an effect instance, which pins its plugin binary before its first render.
#[cfg(not(target_arch = "wasm32"))]
static WORKERS: LazyLock<Option<rayon::ThreadPool>> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(worker_threads().saturating_sub(1).max(1))
        .thread_name(|index| format!("vfx-host frame worker {index}"))
        .build()
        .ok()
});

/// Runs `work` on `threads` threads at once, the caller's among them, and returns when all have
/// finished. `work` takes what's left from a queue of its own, which makes it safe to run in any
/// number of copies: a busy worker's share falls to the others, and if the workers can't start, the
/// caller does it all. A band of pixels at a time is the size of work meant here, since instances
/// share the workers and a caller that's done waits for the copies it queued. On one core, and in
/// browsers, it runs once on the calling thread.
pub fn on_workers(threads: usize, work: impl Fn() + Sync) {
    #[cfg(not(target_arch = "wasm32"))]
    if threads > 1
        && let Some(workers) = WORKERS.as_ref()
    {
        workers.in_place_scope(|scope| {
            for _ in 1..threads {
                scope.spawn(|_| work());
            }
            work();
        });
        return;
    }
    let _ = threads;
    work();
}

/// Runs `f` on each row of `rows` (`width` items each) with its row number, a band of rows
/// at a time on up to six threads. On one core, and in browsers, on the calling thread.
pub fn by_rows<T: Send>(rows: &mut [T], width: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if width == 0 {
        return;
    }
    let height = rows.len() / width;
    let threads = worker_threads()
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
    on_workers(threads, || {
        while let Some((i, chunk)) = next() {
            for (j, row) in chunk.chunks_mut(width).enumerate() {
                f(i * band + j, row);
            }
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
