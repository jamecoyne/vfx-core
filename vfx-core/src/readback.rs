//! Reads small GPU results back through a ring of mapped buffers. A live caller gets the
//! newest finished result without waiting.

use std::sync::mpsc::{self, Receiver, TryRecvError};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use crate::{PollFailure, poll_device};

/// The number of slots in a [`ReadbackRing`]: the result being made and the two before
/// it.
const SLOTS: usize = 3;

/// How many submissions a result may be behind the newest before
/// [`ReadbackRing::newest_finished`] drops it.
const MAX_AGE: u64 = 2;

/// Why a readback failed.
#[derive(Debug, thiserror::Error)]
pub enum ReadbackError {
    /// [`ReadbackRing::wait`] found no submitted result newer than the last
    /// one returned.
    #[error("nothing submitted to wait for")]
    NoSubmission,
    /// The GPU didn't finish within the timeout, as seen by the poll or by the map's
    /// callback.
    #[error("the GPU didn't finish the readback in time")]
    Timeout,
    /// The device is lost. Every later call on the ring returns this. The caller drops
    /// the ring and makes a new one on the next device.
    #[error("device lost during readback")]
    DeviceLost,
    /// A poll failed for another reason, such as a submission that belongs to another
    /// device.
    #[error("couldn't poll for the readback: {0}")]
    Poll(wgpu::PollError),
    #[error("couldn't read the mapped buffer: {0}")]
    MappedRange(wgpu::MapRangeError),
}

impl From<PollFailure> for ReadbackError {
    fn from(failure: PollFailure) -> Self {
        match failure {
            PollFailure::DeviceLost => Self::DeviceLost,
            PollFailure::Other(wgpu::PollError::Timeout) => Self::Timeout,
            PollFailure::Other(err) => Self::Poll(err),
        }
    }
}

/// Receives the result of a [`map_read`].
pub(crate) type MapOutcome = Receiver<Result<(), wgpu::BufferAsyncError>>;

/// Starts mapping all of `buffer` for reading. Call it after the submit that fills the
/// buffer, because a buffer with a map pending can't be the destination of a copy.
pub(crate) fn map_read(buffer: &wgpu::Buffer) -> MapOutcome {
    let (sender, outcome) = mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    outcome
}

/// A result read back, with the index [`ReadbackRing::submit`] returned.
#[derive(Debug)]
pub struct Finished<R> {
    pub index: u64,
    pub value: R,
}

/// A slot's mapped buffer and the caller's state for it. The next submit into the slot
/// reuses both.
struct Kept<T> {
    buffer: wgpu::Buffer,
    target: T,
}

enum Slot<T> {
    /// `Some` holds the last buffer and target so the next submit can reuse them.
    Free(Option<Kept<T>>),
    InFlight {
        index: u64,
        kept: Kept<T>,
        /// What [`ReadbackRing::wait`] waits for. Browsers can't wait.
        #[cfg_attr(target_arch = "wasm32", expect(dead_code))]
        submission: wgpu::SubmissionIndex,
        mapped: MapOutcome,
    },
    Mapped {
        index: u64,
        kept: Kept<T>,
    },
}

impl<T> Slot<T> {
    fn index(&self) -> Option<u64> {
        match self {
            Self::InFlight { index, .. } | Self::Mapped { index, .. } => Some(*index),
            Self::Free(_) => None,
        }
    }
}

/// A ring of [`SLOTS`] buffers that the caller's commands copy results into. Each slot
/// also keeps a `T` for the caller, such as the texture the result is drawn into.
pub struct ReadbackRing<T> {
    device: wgpu::Device,
    slots: [Slot<T>; SLOTS],
    /// The index the next [`Self::submit`] returns.
    next_index: u64,
    /// The index of the newest result returned. No older result is returned after it.
    returned: Option<u64>,
    /// Set once a loss is seen. Every later call returns [`ReadbackError::DeviceLost`].
    lost: bool,
}

impl<T> ReadbackRing<T> {
    pub fn new(device: wgpu::Device) -> Self {
        Self {
            device,
            slots: std::array::from_fn(|_| Slot::Free(None)),
            next_index: 0,
            returned: None,
            lost: false,
        }
    }

    /// Submits the commands `record` returns and starts reading the slot's buffer back.
    /// The commands fill that buffer of `size` bytes. `record` gets the slot's kept
    /// target, if any, and returns the one to keep.
    ///
    /// Returns the result's index, or `None` without calling `record` when every slot is
    /// in flight. A skipped submit still takes an index.
    pub fn submit(
        &mut self,
        queue: &wgpu::Queue,
        size: u64,
        record: impl FnOnce(Option<T>, &wgpu::Buffer) -> (T, wgpu::CommandBuffer),
    ) -> Result<Option<u64>, ReadbackError> {
        self.receive_maps()?;
        let index = self.next_index;
        self.next_index += 1;
        let Some(slot) = self.slots.iter_mut().find(|s| matches!(s, Slot::Free(_))) else {
            return Ok(None);
        };
        let Slot::Free(kept) = slot else {
            unreachable!()
        };
        let (buffer, target) = match kept.take() {
            Some(kept) if kept.buffer.size() == size => (kept.buffer, Some(kept.target)),
            kept => (
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vfx readback ring"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                kept.map(|kept| kept.target),
            ),
        };
        let (target, commands) = record(target, &buffer);
        let submission = queue.submit(Some(commands));
        let mapped = map_read(&buffer);
        *slot = Slot::InFlight {
            index,
            kept: Kept { buffer, target },
            submission,
            mapped,
        };
        Ok(Some(index))
    }

    /// Returns `read` of the newest result whose readback has finished, if it's at most
    /// [`MAX_AGE`] submissions old and newer than any result returned before. Otherwise
    /// returns `None`. Never blocks. Older finished results are dropped.
    pub fn newest_finished<R>(
        &mut self,
        read: impl FnOnce(&T, &[u8]) -> R,
    ) -> Result<Option<Finished<R>>, ReadbackError> {
        self.receive_maps()?;
        let polled = poll_device(&self.device, wgpu::PollType::Poll);
        self.note_loss(polled)?;
        self.receive_maps()?;
        let finished = self.slots.iter().filter_map(|slot| match slot {
            Slot::Mapped { index, .. } => Some(*index),
            _ => None,
        });
        match newest_within_age(finished, self.next_index, self.returned) {
            Some(index) => self.take(index, read).map(Some),
            None => {
                self.release_older(self.next_index);
                Ok(None)
            }
        }
    }

    /// Waits up to `timeout` for the newest submitted result not yet
    /// returned, and returns `read` of it; older results are dropped.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait<R>(
        &mut self,
        timeout: Duration,
        read: impl FnOnce(&T, &[u8]) -> R,
    ) -> Result<Finished<R>, ReadbackError> {
        let deadline = Instant::now() + timeout;
        self.receive_maps()?;
        let index = self
            .slots
            .iter()
            .filter_map(Slot::index)
            .filter(|&index| self.returned.is_none_or(|returned| index > returned))
            .max()
            .ok_or(ReadbackError::NoSubmission)?;
        if let Some(Slot::InFlight { submission, .. }) = self.slot_with_index(index) {
            let wait = wgpu::PollType::Wait {
                submission_index: Some(submission.clone()),
                timeout: Some(timeout),
            };
            let polled = poll_device(&self.device, wait);
            self.note_loss(polled)?;
        }
        // Another poll of this device may hold the callback for a moment.
        // The maps of the older results called back too, and `take` drops those results.
        loop {
            self.receive_maps()?;
            if let Some(Slot::Mapped { .. }) = self.slot_with_index(index) {
                return self.take(index, read);
            }
            if Instant::now() >= deadline {
                return Err(ReadbackError::Timeout);
            }
            std::thread::yield_now();
        }
    }

    fn slot_with_index(&self, index: u64) -> Option<&Slot<T>> {
        self.slots.iter().find(|slot| slot.index() == Some(index))
    }

    /// Moves the slots whose map has called back to mapped. A map of the ring's buffers
    /// fails only when the device is lost, and that frees every slot.
    fn receive_maps(&mut self) -> Result<(), ReadbackError> {
        for slot in &mut self.slots {
            let Slot::InFlight { index, mapped, .. } = slot else {
                continue;
            };
            let index = *index;
            match mapped.try_recv() {
                Ok(Ok(())) => {
                    let Slot::InFlight { kept, .. } = std::mem::replace(slot, Slot::Free(None))
                    else {
                        unreachable!()
                    };
                    *slot = Slot::Mapped { index, kept };
                }
                Ok(Err(_)) | Err(TryRecvError::Disconnected) => self.lost = true,
                Err(TryRecvError::Empty) => {}
            }
        }
        if self.lost {
            self.slots = std::array::from_fn(|_| Slot::Free(None));
            return Err(ReadbackError::DeviceLost);
        }
        Ok(())
    }

    /// Returns a poll's result, and remembers if it reported a loss.
    fn note_loss<P>(&mut self, polled: Result<P, PollFailure>) -> Result<P, ReadbackError> {
        polled.map_err(|failure| {
            let err = ReadbackError::from(failure);
            if matches!(err, ReadbackError::DeviceLost) {
                self.lost = true;
            }
            err
        })
    }

    /// Reads the mapped result `index`, then frees its slot and every older mapped slot.
    fn take<R>(
        &mut self,
        index: u64,
        read: impl FnOnce(&T, &[u8]) -> R,
    ) -> Result<Finished<R>, ReadbackError> {
        let value = match self.slot_with_index(index) {
            Some(Slot::Mapped { kept, .. }) => kept
                .buffer
                .slice(..)
                .get_mapped_range()
                .map(|bytes| read(&kept.target, &bytes)),
            _ => unreachable!("take is called with a mapped index"),
        };
        self.returned = self.returned.max(Some(index));
        self.release_older(index + 1);
        let value = value.map_err(ReadbackError::MappedRange)?;
        Ok(Finished { index, value })
    }

    /// Unmaps and frees the mapped slots older than `index`.
    fn release_older(&mut self, index: u64) {
        for slot in &mut self.slots {
            if matches!(slot, Slot::Mapped { index: i, .. } if *i < index) {
                let Slot::Mapped { kept, .. } = std::mem::replace(slot, Slot::Free(None)) else {
                    unreachable!()
                };
                kept.buffer.unmap();
                *slot = Slot::Free(Some(kept));
            }
        }
    }
}

/// Returns the newest of the `finished` indices that is at most [`MAX_AGE`] behind the
/// newest submitted one (`next_index - 1`) and newer than `returned`.
fn newest_within_age(
    finished: impl Iterator<Item = u64>,
    next_index: u64,
    returned: Option<u64>,
) -> Option<u64> {
    let oldest = next_index.saturating_sub(MAX_AGE + 1);
    let oldest = returned.map_or(oldest, |returned| oldest.max(returned + 1));
    finished.filter(|&index| index >= oldest).max()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::LazyLock;

    use super::*;
    use crate::Gpu;
    use crate::device::DeviceRequest;

    const TIMEOUT: Duration = Duration::from_secs(30);

    /// The device every test shares, on `WGPU_BACKEND`'s backends and
    /// `WGPU_ADAPTER_NAME`'s adapter. The tests share one because making a D3D12 device
    /// while another thread releases one can crash the driver.
    fn gpu() -> &'static Gpu {
        static GPU: LazyLock<Gpu> = LazyLock::new(|| {
            crate::device::open(&DeviceRequest::from_env()).expect("no GPU device")
        });
        &GPU
    }

    /// Source buffers that hold the words 0, 1, ... Result `n` copies buffer `n`.
    fn sources(gpu: &Gpu, count: u32) -> Vec<wgpu::Buffer> {
        (0..count)
            .map(|n| {
                let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 4,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue.write_buffer(&buffer, 0, &n.to_le_bytes());
                buffer
            })
            .collect()
    }

    fn ring(gpu: &Gpu) -> ReadbackRing<()> {
        ReadbackRing::new(gpu.device.clone())
    }

    fn submit(gpu: &Gpu, ring: &mut ReadbackRing<()>, source: &wgpu::Buffer) -> Option<u64> {
        ring.submit(&gpu.queue, 4, |_, buffer| {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(source, 0, buffer, 0, 4);
            ((), encoder.finish())
        })
        .unwrap()
    }

    fn word(_: &(), bytes: &[u8]) -> u32 {
        u32::from_le_bytes(bytes.try_into().unwrap())
    }

    /// Waits until every map in flight has called back. Another test's poll
    /// can hold a callback after the queue is idle, so this loops.
    fn wait_mapped(gpu: &Gpu, ring: &mut ReadbackRing<()>) {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            ring.receive_maps().unwrap();
            let in_flight = |s: &Slot<()>| matches!(s, Slot::InFlight { .. });
            if !ring.slots.iter().any(in_flight) {
                return;
            }
            assert!(Instant::now() < deadline, "maps didn't call back");
            std::thread::yield_now();
        }
    }

    mod newest_finished {
        use super::*;

        mod returns {
            use super::*;

            #[test]
            fn newest_result() {
                let gpu = gpu();
                let sources = sources(gpu, 3);
                let mut ring = ring(gpu);
                for source in &sources {
                    submit(gpu, &mut ring, source);
                }
                wait_mapped(gpu, &mut ring);
                let finished = ring.newest_finished(word).unwrap().unwrap();
                assert_eq!((finished.index, finished.value), (2, 2));
            }

            #[test]
            fn each_result_once() {
                let gpu = gpu();
                let sources = sources(gpu, 1);
                let mut ring = ring(gpu);
                submit(gpu, &mut ring, &sources[0]);
                wait_mapped(gpu, &mut ring);
                assert!(ring.newest_finished(word).unwrap().is_some());
                assert!(ring.newest_finished(word).unwrap().is_none());
            }
        }

        #[test]
        fn read_frees_slots() {
            let gpu = gpu();
            let sources = sources(gpu, 1);
            let mut ring = ring(gpu);
            for _ in 0..3 {
                submit(gpu, &mut ring, &sources[0]);
            }
            wait_mapped(gpu, &mut ring);
            ring.newest_finished(word).unwrap();
            let indices: Vec<_> = (0..3)
                .map(|_| submit(gpu, &mut ring, &sources[0]))
                .collect();
            assert_eq!(indices, [Some(3), Some(4), Some(5)]);
        }

        #[test]
        fn drops_results_past_max_age() {
            let gpu = gpu();
            let sources = sources(gpu, 1);
            let mut ring = ring(gpu);
            // Three results, then three skipped submits, so the newest result is 3
            // behind.
            for _ in 0..6 {
                submit(gpu, &mut ring, &sources[0]);
            }
            wait_mapped(gpu, &mut ring);
            assert!(ring.newest_finished(word).unwrap().is_none());
            assert!(ring.slots.iter().all(|s| matches!(s, Slot::Free(_))));
        }

        #[test]
        fn skips_map_after_wait() {
            let gpu = gpu();
            let sources = sources(gpu, 2);
            let mut ring = ring(gpu);
            submit(gpu, &mut ring, &sources[0]);
            submit(gpu, &mut ring, &sources[1]);
            // Result 0's map calls back only after `wait` returned result 1.
            let (late, pending) = mpsc::channel();
            let Slot::InFlight { mapped, .. } = &mut ring.slots[0] else {
                unreachable!("the first submit takes the first slot")
            };
            *mapped = pending;
            assert_eq!(ring.wait(TIMEOUT, word).unwrap().index, 1);
            late.send(Ok(())).unwrap();
            assert!(ring.newest_finished(word).unwrap().is_none());
        }
    }

    mod submit {
        use super::*;

        #[test]
        fn gives_none_when_slots_busy() {
            let gpu = gpu();
            let sources = sources(gpu, 1);
            let mut ring = ring(gpu);
            let indices: Vec<_> = (0..4)
                .map(|_| submit(gpu, &mut ring, &sources[0]))
                .collect();
            // A slot stays taken after its map calls back, until a result is read.
            assert_eq!(indices, [Some(0), Some(1), Some(2), None]);
        }

        #[test]
        fn passes_kept_target() {
            let gpu = gpu();
            let sources = sources(gpu, 1);
            let mut ring = ReadbackRing::new(gpu.device.clone());
            let mut kept_targets = Vec::new();
            for _ in 0..2 {
                ring.submit(&gpu.queue, 4, |kept: Option<u32>, buffer| {
                    kept_targets.push(kept);
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(&sources[0], 0, buffer, 0, 4);
                    (7, encoder.finish())
                })
                .unwrap();
                ring.wait(TIMEOUT, |_, _| ()).unwrap();
            }
            assert_eq!(kept_targets, [None, Some(7)]);
        }
    }

    mod wait {
        use super::*;

        #[test]
        fn returns_newest_result() {
            let gpu = gpu();
            let sources = sources(gpu, 2);
            let mut ring = ring(gpu);
            submit(gpu, &mut ring, &sources[0]);
            submit(gpu, &mut ring, &sources[1]);
            let finished = ring.wait(TIMEOUT, word).unwrap();
            assert_eq!((finished.index, finished.value), (1, 1));
        }

        #[test]
        fn fails_without_submit() {
            let err = ring(gpu()).wait(TIMEOUT, word).unwrap_err();
            assert!(matches!(err, ReadbackError::NoSubmission), "{err}");
        }

        #[test]
        fn times_out_on_pending_map() {
            let gpu = gpu();
            let sources = sources(gpu, 1);
            let mut ring = ring(gpu);
            submit(gpu, &mut ring, &sources[0]);
            // A map that never calls back.
            let (_sender, pending) = mpsc::channel();
            let Slot::InFlight { mapped, .. } = &mut ring.slots[0] else {
                unreachable!("the first submit takes the first slot")
            };
            *mapped = pending;
            let err = ring.wait(Duration::from_millis(50), word).unwrap_err();
            assert!(matches!(err, ReadbackError::Timeout), "{err}");
        }
    }

    mod device_loss {
        use super::*;

        /// A ring whose one map in flight has failed, as maps do on a lost
        /// device.
        fn failed_map(gpu: &Gpu) -> ReadbackRing<()> {
            let sources = sources(gpu, 1);
            let mut ring = ring(gpu);
            submit(gpu, &mut ring, &sources[0]);
            let (sender, failed) = mpsc::channel();
            sender.send(Err(wgpu::BufferAsyncError)).unwrap();
            let Slot::InFlight { mapped, .. } = &mut ring.slots[0] else {
                unreachable!("the first submit takes the first slot")
            };
            *mapped = failed;
            ring
        }

        #[test]
        fn newest_finished_reports_loss() {
            let err = failed_map(gpu()).newest_finished(word).unwrap_err();
            assert!(matches!(err, ReadbackError::DeviceLost), "{err}");
        }

        #[test]
        fn wait_reports_loss() {
            let err = failed_map(gpu()).wait(TIMEOUT, word).unwrap_err();
            assert!(matches!(err, ReadbackError::DeviceLost), "{err}");
        }

        #[test]
        fn submit_after_loss_fails() {
            let gpu = gpu();
            let mut ring = failed_map(gpu);
            let _ = ring.newest_finished(word);
            let err = ring
                .submit(&gpu.queue, 4, |_, _| {
                    unreachable!("a lost ring records nothing")
                })
                .unwrap_err();
            assert!(matches!(err, ReadbackError::DeviceLost), "{err}");
        }
    }
}
