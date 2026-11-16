//! What an effect builds once for each device of a [`DeviceSlot`].
//!
//! Two cases are left unhandled on purpose. A build that fails on a working device isn't retried
//! on that device. A GPU that hangs without the system resetting it keeps making renders time
//! out. In both cases the user has to restart the app. A reset or a new device builds again.

use std::error::Error;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, TryLockError, Weak};

use super::{Device, DeviceSlot, LossSubscription, NoDevice, panic_message};
use crate::wgpu;

type Build<T> = dyn Fn(&Arc<Device>) -> Result<T, NoDevice> + Send + Sync;

/// Holds one `T` per device of a slot, for the device's lifetime, such as an effect's pipelines
/// and shaders. State that varies per render or per size goes in vfx-host's pipeline pool
/// instead.
///
/// Each build runs on its own thread. [`Self::try_wait`] returns [`NoDevice::Opening`] while the
/// slot is opening a device or the build is running. While the slot has no device, it returns
/// the slot's reason. A call that finds the slot without a device drops the value. A failed
/// build isn't retried on its device.
///
/// When the slot loses a device it opened, the build drops its value on that device from the
/// slot's loss thread, without waiting for a render. Each user does the same for what it keeps
/// there ([`Self::user_with`]). D3D12 won't open a new device on an adapter until the removed
/// device is fully released.
pub struct DeviceBuild<T> {
    slot: &'static DeviceSlot,
    build: Box<Build<T>>,
    state: Mutex<State<T>>,
    build_done: Condvar,
    declines_software: bool,
    /// The live [`BuildUser`]s. When the last one is dropped, a value on a host's device is dropped
    /// too, so nothing holds the host's device once there are no users.
    users: AtomicUsize,
    /// What each user keeps, for the loss thread to release.
    kept: Mutex<Vec<Weak<dyn Release>>>,
    /// Taken by the first call that may hand out a value or a user.
    loss: OnceLock<LossSubscription>,
}

/// A user's state, which the loss thread releases.
trait Release: Send + Sync {
    /// Lets go of what's kept on the lost device `id`, unless a render holds the state. In that
    /// case the render finds out about the loss itself.
    fn release(&self, id: u64);
}

/// What [`DeviceBuild::user_with`] keeps for a user.
struct UserState<S> {
    state: Mutex<S>,
    release: fn(&mut S, u64),
}

impl<S: Send> Release for UserState<S> {
    fn release(&self, id: u64) {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };
        (self.release)(&mut state, id);
    }
}

enum State<T> {
    Empty,
    /// For the slot's device with this id.
    Building {
        id: u64,
        host_shared: bool,
    },
    // `value` drops before `device`, which goes last, on the slot's retire thread.
    Ready {
        value: Arc<T>,
        device: Arc<Device>,
    },
    /// Holds the device, so a host's device keeps its id and doesn't get a new build.
    Failed {
        device: Arc<Device>,
        why: NoDevice,
    },
}

/// The device reported an error during a build but didn't keep the error.
#[derive(Debug, thiserror::Error)]
#[error("the device reported an error it didn't keep")]
struct UnkeptError;

impl<T: Send + Sync + 'static> DeviceBuild<T> {
    /// Builds with `build` on each device of `slot`. `build` gets the device so it can keep it, if
    /// what it makes has to be dropped before the device is. A build fails with
    /// [`NoDevice::Unusable`] holding `build`'s error, or the last error the device reported
    /// meanwhile. It fails with [`NoDevice::Lost`] if the device was lost meanwhile, and with
    /// [`NoDevice::Panicked`] if it panics.
    pub fn new<E>(
        slot: &'static DeviceSlot,
        build: impl Fn(&Arc<Device>) -> Result<T, E> + Send + Sync + 'static,
    ) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        let checked = move |device: &Arc<Device>| {
            let faults = device.gpu().faults();
            // The host may have reported errors before this point.
            let errors_before = faults.error_count();
            let value = build(device).map_err(|e| NoDevice::Unusable(Arc::new(e)))?;
            if let Some(why) = device.lost() {
                return Err(NoDevice::Lost(why));
            }
            if faults.error_count() != errors_before {
                let why: Arc<dyn Error + Send + Sync> = match faults.last_error() {
                    Some(error) => error,
                    None => Arc::new(UnkeptError),
                };
                return Err(NoDevice::Unusable(why));
            }
            Ok(value)
        };
        Self {
            slot,
            build: Box::new(checked),
            state: Mutex::new(State::Empty),
            build_done: Condvar::new(),
            declines_software: false,
            users: AtomicUsize::new(0),
            kept: Mutex::new(Vec::new()),
            loss: OnceLock::new(),
        }
    }

    /// With `declines`, a software adapter's device doesn't get a build, and
    /// [`NoDevice::SoftwareOnly`] is returned instead.
    pub fn declining_software(self, declines: bool) -> Self {
        Self {
            declines_software: declines,
            ..self
        }
    }

    fn state(&self) -> MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Registers this build with the slot so it's told when a device the slot opened is lost.
    fn watch_losses(&'static self) {
        self.loss
            .get_or_init(|| self.slot.on_loss(move |id| self.release_device(id)));
    }

    /// Drops what was built on the lost device `id`, then has each user that isn't held by a render
    /// let go of what it keeps there. The slot's loss thread calls it.
    fn release_device(&self, id: u64) {
        let gone = {
            let mut state = self.state();
            let on_lost = match &*state {
                State::Empty => false,
                State::Building { id: building, .. } => *building == id,
                State::Ready { device, .. } | State::Failed { device, .. } => device.id() == id,
            };
            on_lost.then(|| std::mem::replace(&mut *state, State::Empty))
        };
        self.build_done.notify_all();
        drop(gone);
        let kept: Vec<_> = {
            let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            kept.iter().filter_map(Weak::upgrade).collect()
        };
        for user in kept {
            user.release(id);
        }
    }

    /// Replaces the state with `next`, dropping the old state after releasing the lock.
    fn replace(&self, next: State<T>) {
        let gone = std::mem::replace(&mut *self.state(), next);
        self.build_done.notify_all();
        drop(gone);
    }

    /// Returns the value on the slot's device, or the reason there isn't one. It never blocks. It
    /// starts a build on a device it hasn't seen yet.
    pub fn try_wait(&'static self) -> Result<Arc<T>, NoDevice> {
        self.watch_losses();
        let device = match self.slot.try_wait() {
            Ok(device) => device,
            Err(why) => {
                self.replace(State::Empty);
                return Err(why);
            }
        };
        let adapter = device.adapter_info();
        if self.declines_software && adapter.device_type == wgpu::DeviceType::Cpu {
            return Err(NoDevice::SoftwareOnly(adapter.name));
        }
        let (id, host_shared) = (device.id(), device.host_shared());
        let mut state = self.state();
        match &*state {
            State::Ready { value, device } if device.id() == id => return Ok(value.clone()),
            State::Building { id: building, .. } if *building == id => {
                return Err(NoDevice::Opening);
            }
            State::Failed { device, why } if device.id() == id => return Err(why.clone()),
            _ => {}
        }
        let building = State::Building { id, host_shared };
        let gone = std::mem::replace(&mut *state, building);
        drop(state);
        drop(gone);
        let spawned = std::thread::Builder::new()
            .name("vfx device build".into())
            .spawn(move || self.build_on(device));
        if let Err(e) = spawned {
            // The next call tries again.
            self.replace(State::Empty);
            return Err(NoDevice::NoThread(Arc::new(e)));
        }
        Err(NoDevice::Opening)
    }

    /// Builds on `device` and keeps the outcome, unless the state moved on meanwhile.
    fn build_on(&self, device: Arc<Device>) {
        let built =
            catch_unwind(AssertUnwindSafe(|| (self.build)(&device))).unwrap_or_else(|panic| {
                let message = panic_message(&*panic);
                Err(NoDevice::Panicked(format!(
                    "building on the device: {message}"
                )))
            });
        let mut state = self.state();
        if !matches!(&*state, State::Building { id, .. } if *id == device.id()) {
            // The state was replaced during the build, so discard this outcome.
            drop(state);
            drop(built);
            self.build_done.notify_all();
            return;
        }
        *state = match built {
            Ok(value) => State::Ready {
                value: Arc::new(value),
                device,
            },
            Err(why) => State::Failed { device, why },
        };
        drop(state);
        self.build_done.notify_all();
    }

    /// Works like [`Self::try_wait`], but first waits for the slot's first open or replacement and
    /// for the build on its device.
    pub fn wait(&'static self) -> Result<Arc<T>, NoDevice> {
        // A value on a lost device holds the device, and the slot's reopen waits for it to be
        // dropped.
        let lost = matches!(&*self.state(), State::Ready { device, .. } if device.lost().is_some());
        if lost {
            self.replace(State::Empty);
        }
        self.slot.wait()?;
        loop {
            let answer = self.try_wait();
            if !matches!(answer, Err(NoDevice::Opening)) {
                return answer;
            }
            let state = self.state();
            if matches!(*state, State::Building { .. }) {
                let building = |state: &mut State<T>| matches!(state, State::Building { .. });
                drop(self.build_done.wait_while(state, building));
            } else {
                // The slot itself is opening again.
                drop(state);
                self.slot.wait()?;
            }
        }
    }

    /// Drops a value on a host's device on the caller's thread, so the host can open its next
    /// device. A build under way on a host's device discards its value when it finishes.
    pub fn release_host_shared(&self) {
        let host_shared = match &*self.state() {
            State::Empty => false,
            State::Building { host_shared, .. } => *host_shared,
            State::Ready { device, .. } | State::Failed { device, .. } => device.host_shared(),
        };
        if host_shared {
            self.replace(State::Empty);
        }
    }

    /// Counts one user until the returned guard is dropped. The last user to drop releases a value
    /// on a host's device ([`Self::release_host_shared`]) and then runs `on_last`.
    pub fn user(&'static self, on_last: fn()) -> BuildUser<T> {
        self.user_with((), |_, _| {}, on_last)
    }

    /// Like [`Self::user`], for a user that keeps `state`. The state may hold things made on the
    /// slot's devices, such as an effect instance's drawer. [`BuildUser::lock`] hands it out. When
    /// the slot loses a device it opened, the slot's loss thread calls `release` on the state with
    /// the device's id, unless a render holds the lock. In that case the render finds out about the
    /// loss itself. `release` drops what the state keeps on that device and doesn't block.
    pub fn user_with<S: Send + 'static>(
        &'static self,
        state: S,
        release: fn(&mut S, u64),
        on_last: fn(),
    ) -> BuildUser<T, S> {
        self.watch_losses();
        self.users.fetch_add(1, Ordering::AcqRel);
        let state = Arc::new(UserState {
            state: Mutex::new(state),
            release,
        });
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        kept.retain(|user| user.strong_count() > 0);
        kept.push(Arc::downgrade(&state) as Weak<dyn Release>);
        BuildUser {
            build: self,
            on_last,
            state,
        }
    }
}

/// One user of a [`DeviceBuild`], from [`DeviceBuild::user`] or [`DeviceBuild::user_with`].
pub struct BuildUser<T: Send + Sync + 'static, S: Send + 'static = ()> {
    build: &'static DeviceBuild<T>,
    on_last: fn(),
    state: Arc<UserState<S>>,
}

impl<T: Send + Sync + 'static, S: Send + 'static> BuildUser<T, S> {
    /// The user's state, which the loss thread can't release while this is held.
    pub fn lock(&self) -> MutexGuard<'_, S> {
        self.state.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<T: Send + Sync + 'static, S: Send + 'static> Drop for BuildUser<T, S> {
    fn drop(&mut self) {
        if self.build.users.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.build.release_host_shared();
            (self.on_last)();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::time::{Duration, Instant};

    use super::super::tests::{HostShare, gpu_test_lock, lose, test_gpu};
    use super::*;

    /// What a test builds: the device, held the way an effect's pipelines hold it.
    struct Held(#[expect(dead_code, reason = "held for its drop")] Arc<Device>);

    fn test_slot() -> &'static DeviceSlot {
        Box::leak(Box::new(DeviceSlot::new(|_| Ok(test_gpu()))))
    }

    fn held_build(slot: &'static DeviceSlot) -> &'static DeviceBuild<Held> {
        Box::leak(Box::new(DeviceBuild::new(slot, |device| {
            Ok::<_, Infallible>(Held(device.clone()))
        })))
    }

    #[derive(Debug, thiserror::Error)]
    #[error("a build the test failed")]
    struct FailedBuild;

    /// A build on a separate slot that always fails.
    fn failing_build() -> &'static DeviceBuild<Held> {
        Box::leak(Box::new(DeviceBuild::new(
            test_slot(),
            |_| -> Result<Held, FailedBuild> { Err(FailedBuild) },
        )))
    }

    mod release {
        use super::*;

        #[test]
        fn host_value_drops() {
            let _lock = gpu_test_lock();
            let _host = HostShare::new();
            let build = held_build(test_slot());
            let held = Arc::downgrade(&build.wait().expect("a build"));
            assert!(held.upgrade().is_some(), "the build holds it");
            build.release_host_shared();
            assert!(held.upgrade().is_none(), "kept the host's device");
        }

        /// A value on a device the slot opened outlives its last user. Only a value on a host's
        /// device is dropped with the last user.
        #[test]
        fn own_device_value_stays() {
            let _lock = gpu_test_lock();
            let build = held_build(test_slot());
            let user = build.user(|| {});
            build.slot.wait().expect("an adapter");
            assert!(matches!(build.try_wait(), Err(NoDevice::Opening)));
            drop(user);
            assert!(
                matches!(*build.state(), State::Building { .. } | State::Ready { .. }),
                "the last user dropped a build on the slot's own device"
            );
        }

        #[test]
        fn last_user_drops_host_value() {
            let _lock = gpu_test_lock();
            let _host = HostShare::new();
            let build = held_build(test_slot());
            let first = build.user(|| {});
            let second = build.user(|| {});
            let held = Arc::downgrade(&build.wait().expect("a build"));
            drop(first);
            assert!(held.upgrade().is_some(), "went with a user still live");
            drop(second);
            assert!(held.upgrade().is_none(), "kept the host's device");
        }
    }

    mod failure {
        use super::*;

        #[test]
        fn build_error_is_source() {
            let _lock = gpu_test_lock();
            let build = failing_build();
            let failed = build.wait();
            let source = match &failed {
                Err(NoDevice::Unusable(source)) => source,
                other => panic!("{:?}", other.as_ref().map(|_| "a value")),
            };
            assert!(source.is::<FailedBuild>(), "{source}");
        }

        #[test]
        fn build_stays_failed() {
            let _lock = gpu_test_lock();
            let build = failing_build();
            assert!(build.wait().is_err());
            assert!(matches!(build.try_wait(), Err(NoDevice::Unusable(_))));
        }

        #[test]
        fn panic_becomes_panicked() {
            let _lock = gpu_test_lock();
            let build: &'static DeviceBuild<Held> = Box::leak(Box::new(DeviceBuild::new(
                test_slot(),
                |_| -> Result<Held, Infallible> { panic!("a build the test panicked") },
            )));
            assert!(matches!(build.wait(), Err(NoDevice::Panicked(_))));
        }
    }

    /// A lost device's value is dropped, and the slot's next device gets a new build.
    #[test]
    fn lost_device_value_drops() {
        let _lock = gpu_test_lock();
        let build = held_build(test_slot());
        let first = build.wait().expect("a build");
        let first_device = build.slot.wait().expect("a device");
        lose(&first_device);
        drop(first_device);
        let held = Arc::downgrade(&first);
        drop(first);
        build.wait().expect("a build on the next device");
        assert!(held.upgrade().is_none(), "kept the lost device's value");
    }

    /// After a loss with no render running, what a build and its users hold on the lost device is
    /// dropped.
    mod loss {
        use super::*;

        /// Loses the device `build`'s value is on, without calling into the build.
        fn lose_device(build: &'static DeviceBuild<Held>) {
            let device = build.slot.wait().expect("a device");
            lose(&device);
        }

        /// Waits up to 10 s for `held` to go.
        fn released<V>(held: &std::sync::Weak<V>) -> bool {
            let deadline = Instant::now() + Duration::from_secs(10);
            while held.strong_count() > 0 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            held.strong_count() == 0
        }

        #[test]
        fn value_drops_without_call() {
            let _lock = gpu_test_lock();
            let build = held_build(test_slot());
            let held = Arc::downgrade(&build.wait().expect("a build"));
            lose_device(build);
            assert!(
                released(&held),
                "the build kept its value on the lost device"
            );
        }

        #[test]
        fn user_state_drops_without_call() {
            let _lock = gpu_test_lock();
            let build = held_build(test_slot());
            let value = build.wait().expect("a build");
            let held = Arc::downgrade(&value);
            let release = |kept: &mut Option<Arc<Held>>, _: u64| *kept = None;
            let _user = build.user_with(Some(value), release, || {});
            lose_device(build);
            assert!(
                released(&held),
                "the user kept its value on the lost device"
            );
        }

        /// A user whose state a render holds keeps that state. The render finds out about the loss
        /// itself.
        #[test]
        fn locked_user_state_stays() {
            let _lock = gpu_test_lock();
            let build = held_build(test_slot());
            let value = build.wait().expect("a build");
            let release = |kept: &mut Option<Arc<Held>>, _: u64| *kept = None;
            let user = build.user_with(Some(value), release, || {});
            // This user is released after `user`, so once its count changes, the loss thread is
            // past `user`.
            let count = |calls: &mut Arc<AtomicUsize>, _: u64| {
                calls.fetch_add(1, Ordering::AcqRel);
            };
            let counter = build.user_with(Arc::new(AtomicUsize::new(0)), count, || {});
            let calls = counter.lock().clone();
            let rendering = user.lock();
            lose_device(build);
            let deadline = Instant::now() + Duration::from_secs(10);
            while calls.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(
                calls.load(Ordering::Acquire),
                1,
                "the loss thread didn't get past"
            );
            assert!(rendering.is_some(), "released under a render");
        }
    }

    #[test]
    #[cfg(windows)]
    fn software_adapter_is_declined() {
        use super::super::{AdapterChoice, DeviceRequest, open};

        let _lock = gpu_test_lock();
        let slot: &'static DeviceSlot = Box::leak(Box::new(DeviceSlot::new(|_| {
            open(&DeviceRequest {
                adapter: AdapterChoice::Name("Microsoft Basic Render Driver".into()),
                ..DeviceRequest::default()
            })
        })));
        let build = Box::leak(Box::new(
            DeviceBuild::new(slot, |device| Ok::<_, Infallible>(Held(device.clone())))
                .declining_software(true),
        ));
        slot.wait().expect("WARP");
        assert!(matches!(build.try_wait(), Err(NoDevice::SoftwareOnly(_))));
    }

    #[test]
    fn build_in_progress_is_opening() {
        let _lock = gpu_test_lock();
        let release: &'static (Mutex<bool>, Condvar) =
            Box::leak(Box::new((Mutex::new(false), Condvar::new())));
        let build: &'static DeviceBuild<Held> =
            Box::leak(Box::new(DeviceBuild::new(test_slot(), move |device| {
                let (released, signal) = release;
                let held = released.lock().unwrap();
                drop(signal.wait_timeout_while(held, Duration::from_secs(10), |r| !*r));
                Ok::<_, Infallible>(Held(device.clone()))
            })));
        build.slot.wait().expect("an adapter");
        assert!(matches!(build.try_wait(), Err(NoDevice::Opening)));
        assert!(matches!(build.try_wait(), Err(NoDevice::Opening)));
        *release.0.lock().unwrap() = true;
        release.1.notify_all();
        assert!(build.wait().is_ok());
    }
}
