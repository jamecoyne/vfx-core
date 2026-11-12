use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What a device reported through its error and loss callbacks. wgpu holds one of each per device
/// and its default error handler panics, so one place installs them and everyone reads this.
#[derive(Clone, Default)]
pub struct DeviceFaults {
    errors: Arc<Mutex<(u64, String)>>,
    lost: Arc<Mutex<Option<String>>>,
}

impl DeviceFaults {
    /// Takes over `device`'s callbacks. Each new error message goes to stderr once.
    pub fn watch(device: &wgpu::Device) -> Self {
        let faults = Self::default();
        faults.install(device);
        faults
    }

    fn install(&self, device: &wgpu::Device) {
        let errors = self.errors.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            if let Ok(mut errors) = errors.lock() {
                let message = e.to_string();
                if errors.1 != message {
                    eprintln!("GPU error: {message}");
                }
                *errors = (errors.0 + 1, message);
            }
        }));
        let lost = self.lost.clone();
        device.set_device_lost_callback(move |reason, e| {
            if let Ok(mut lost) = lost.lock() {
                lost.get_or_insert(format!("{reason:?}: {e}"));
            }
        });
    }

    /// How many errors the device has reported, and the last one.
    pub fn errors(&self) -> (u64, String) {
        self.errors.lock().map_or((0, String::new()), |e| e.clone())
    }

    /// Why the device was lost, once it is.
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().ok().and_then(|lost| lost.clone())
    }

    /// Whether both are the record of one device. `wgpu::Device`'s own equality holds only within
    /// one `Instance`, so it can call a device reopened from a new instance the old one.
    pub fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.lost, &other.lost)
    }
}

/// When to try for a new device after a loss: 1, 2, 4, 8 and 16 s, then every 30 s for as long as
/// it takes, since a show lasts hours and a device may come back late. A device that lived
/// [`Self::STABLE_AFTER`] starts the schedule over; one that dies sooner goes on from where it
/// was, so a GPU that opens and dies isn't reopened every second all night.
#[derive(Clone, Debug, Default)]
pub struct ReopenBackoff {
    tries: u32,
}

impl ReopenBackoff {
    const LONGEST: Duration = Duration::from_secs(30);
    /// How long a device has to last to count as back.
    pub const STABLE_AFTER: Duration = Duration::from_secs(60);

    /// The wait before the next try, which this counts.
    pub fn next_delay(&mut self) -> Duration {
        let delay = Duration::from_secs(1 << self.tries.min(5)).min(Self::LONGEST);
        self.tries += 1;
        delay
    }

    /// Whether the next try should wait for the adapter that was lost, for about 30 s. A reset can
    /// hide an adapter for seconds, and a machine with another GPU would otherwise land there and
    /// stay for the night. After that any adapter beats none.
    pub fn same_adapter_only(&self) -> bool {
        self.tries <= 4
    }

    /// A device was lost after `lived`.
    pub fn lost(&mut self, lived: Duration) {
        if lived >= Self::STABLE_AFTER {
            self.tries = 0;
        }
    }
}

/// Why [`poll_device`] failed.
#[derive(Debug)]
pub enum PollFailure {
    DeviceLost,
    Other(wgpu::PollError),
}

impl fmt::Display for PollFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceLost => write!(f, "the device was lost during a poll"),
            Self::Other(e) => write!(f, "{e}"),
        }
    }
}

/// [`wgpu::Device::poll`], for anything that has to go on after the device is lost: wgpu panics
/// there rather than return an error.
pub fn poll_device(
    device: &wgpu::Device,
    poll: wgpu::PollType,
) -> Result<wgpu::PollStatus, PollFailure> {
    catch_unwind(AssertUnwindSafe(|| device.poll(poll)))
        .map_err(|_| PollFailure::DeviceLost)?
        .map_err(PollFailure::Other)
}

/// The device and queue every effect renders with.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Lives as long as this device's callbacks, so what one device reports can't reach another.
    faults: DeviceFaults,
}

impl Gpu {
    /// Wraps a device the host already owns, so effects run on the host's own GPU context.
    pub fn from_parts(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            faults: DeviceFaults::default(),
        }
    }

    /// Requests a GPU with no window surface. Async, so the caller picks the executor.
    pub async fn headless() -> Result<Self, GpuError> {
        // Plugins run inside hosts that own OpenGL contexts, and wgpu's GL backend would
        // take over the host's current one.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            backend_options: Self::backend_options(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| GpuError(e.to_string()))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                // Effects blend in 32-bit float targets where the adapter can.
                required_features: adapter.features() & wgpu::Features::FLOAT32_BLENDABLE,
                memory_hints: Self::memory_hints(),
                ..wgpu::DeviceDescriptor::default()
            })
            .await
            .map_err(|e| GpuError(e.to_string()))?;
        Ok(Self::from_parts(device, queue))
    }

    /// Backend options for every wgpu instance a host or effect makes. On Windows
    /// they select the DXC the Setup shares, else the one beside this module (the
    /// zips), else the slower FXC; `WGPU_DX12_COMPILER` overrides.
    pub fn backend_options() -> wgpu::BackendOptions {
        wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: dx12_shader_compiler(),
                ..wgpu::Dx12BackendOptions::from_env_or_default()
            },
            ..wgpu::BackendOptions::from_env_or_default()
        }
    }

    /// Allocator block sizes for a device that stays open for a whole show, in place of wgpu's
    /// default of 128 to 256 MiB. Vulkan and D3D12 reserve memory in blocks and place a new
    /// allocation in the newest block with room, so one small buffer made beside a large one holds
    /// the whole block once the large one has gone, and on integrated GPUs a reserved block is
    /// system memory. A buffer larger than the current block gets memory of its own, which it
    /// returns when it goes.
    ///
    /// Host blocks are half these sizes, and staging comes from them. A fixed 64 MiB would make
    /// those 32 MiB, which a 4K frame's staging all but fills: blocks were then made and freed
    /// about twice a second. The range lets the host blocks reach 64 MiB.
    pub fn memory_hints() -> wgpu::MemoryHints {
        const MIB: u64 = 1 << 20;
        wgpu::MemoryHints::Manual {
            suballocated_device_memory_block_size: 64 * MIB..128 * MIB,
        }
    }

    /// Lets CPU effects draw on this device: in browsers, where they can't open their own, and
    /// in hosts that offer their output a GPU frame on it ([`crate::CpuEffect::render_to`]).
    /// Effects pick their device on first use, so a host shares it before they render. This takes
    /// over the device's error and loss callbacks; read them with [`Gpu::faults`]. Sharing the same
    /// `Gpu` again keeps its record. Sharing another replaces it for later readers, and a clone
    /// taken earlier goes on following its own device.
    pub fn share(&self) {
        self.watch_faults();
        let parts = Some(self.parts());
        #[cfg(target_arch = "wasm32")]
        SHARED.with(|s| *s.borrow_mut() = parts);
        #[cfg(not(target_arch = "wasm32"))]
        {
            *SHARED.lock().unwrap_or_else(|e| e.into_inner()) = parts;
        }
    }

    /// Takes over the device's error and loss callbacks, for a host that doesn't share it; read
    /// them with [`Gpu::faults`].
    pub fn watch_faults(&self) {
        self.faults.install(&self.device);
    }

    fn parts(&self) -> GpuParts {
        (self.device.clone(), self.queue.clone(), self.faults.clone())
    }

    /// The device a host shared with [`Gpu::share`].
    pub fn shared() -> Option<Self> {
        #[cfg(target_arch = "wasm32")]
        let parts = SHARED.with(|s| s.borrow().clone());
        #[cfg(not(target_arch = "wasm32"))]
        let parts = SHARED.lock().unwrap_or_else(|e| e.into_inner()).clone();
        parts.map(|(device, queue, faults)| Self {
            device,
            queue,
            faults,
        })
    }

    /// What this device has reported since it was first shared.
    pub fn faults(&self) -> DeviceFaults {
        self.faults.clone()
    }

    /// What the device last shared with [`Gpu::share`] has reported; nothing if none was.
    pub fn shared_faults() -> DeviceFaults {
        Self::shared().map(|gpu| gpu.faults).unwrap_or_default()
    }
}

type GpuParts = (wgpu::Device, wgpu::Queue, DeviceFaults);

/// The D3D12 shader compiler for this process. A `DynamicDxc` path that doesn't
/// load fails wgpu's whole D3D12 backend, so a DXC that is missing or doesn't
/// load falls back to FXC here.
fn dx12_shader_compiler() -> wgpu::Dx12Compiler {
    static COMPILER: std::sync::OnceLock<wgpu::Dx12Compiler> = std::sync::OnceLock::new();
    COMPILER
        .get_or_init(|| {
            if let Some(compiler) = wgpu::Dx12Compiler::from_env() {
                return compiler;
            }
            #[cfg(windows)]
            match dxc::bundled() {
                Ok(dxc_path) => wgpu::Dx12Compiler::DynamicDxc { dxc_path },
                Err(err) => {
                    eprintln!("vfx-core: D3D12 shaders compile with FXC, which is slow: {err}");
                    wgpu::Dx12Compiler::Fxc
                }
            }
            #[cfg(not(windows))]
            wgpu::Dx12Compiler::Auto
        })
        .clone()
}

/// The DXC that the Windows Setup shares in Common Files, or that the zips
/// put beside each module.
#[cfg(windows)]
mod dxc {
    use std::error::Error;
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use std::path::{Path, PathBuf};
    use std::{io, mem, ptr, slice};

    use libloading::Library;
    use windows_sys::Win32::System::Com::CoTaskMemFree;
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleFileNameW, GetModuleHandleExW,
    };
    use windows_sys::Win32::UI::Shell::{
        FOLDERID_ProgramFilesCommon, KF_FLAG_DEFAULT, SHGetKnownFolderPath,
    };

    /// The DXC release that packaging ships, from `wrappers/dxc-tag`.
    const TAG: &str = include_str!("../wrappers/dxc-tag");

    /// Loads the `dxcompiler.dll` in [`shared_dir`], or else the one beside
    /// this module, and returns its path.
    ///
    /// The shared folder is this build's DXC release; a DLL beside a plugin in
    /// a shared folder can be older or another vendor's. wgpu needs the full
    /// path: a host doesn't search a plugin's folder.
    pub fn bundled() -> io::Result<String> {
        load_first(shared_dir().as_deref(), &module_dir()?)
    }

    /// Loads the `dxcompiler.dll` in `shared`, or else the one in `beside`,
    /// and returns its path. A DLL that is there but won't load counts as
    /// missing; the error names every place tried.
    fn load_first(shared: Option<&Path>, beside: &Path) -> io::Result<String> {
        let mut tried = Vec::new();
        for dir in shared.into_iter().chain([beside]) {
            match load(&dir.join("dxcompiler.dll")) {
                Ok(path) => return Ok(path),
                Err(err) => tried.push(err.to_string()),
            }
        }
        Err(io::Error::new(io::ErrorKind::NotFound, tried.join("; ")))
    }

    /// Where the Windows Setup puts one DXC for all its plugins, a folder per
    /// release: `Common Files/vfx-core/dxc-<tag>`. `None` if Windows can't
    /// name its Common Files folder.
    fn shared_dir() -> Option<PathBuf> {
        let mut path = ptr::null_mut();
        // SAFETY: the folder id and the out pointer are valid; on success
        // `path` is a NUL terminated string that Windows allocated.
        let status = unsafe {
            SHGetKnownFolderPath(
                &FOLDERID_ProgramFilesCommon,
                KF_FLAG_DEFAULT as u32,
                ptr::null_mut(),
                &mut path,
            )
        };
        let common = (status >= 0).then(|| {
            // SAFETY: `path` is NUL terminated, so every index before the NUL
            // is inside the string.
            let len = (0..).take_while(|&i| unsafe { *path.add(i) } != 0).count();
            // SAFETY: the first `len` units are the string, read above.
            OsString::from_wide(unsafe { slice::from_raw_parts(path, len) })
        });
        // SAFETY: Windows asks for the string to be freed whether the call
        // succeeded or not; freeing null does nothing.
        unsafe { CoTaskMemFree(path as *const _) };
        Some(
            PathBuf::from(common?)
                .join("vfx-core")
                .join(format!("dxc-{}", TAG.trim())),
        )
    }

    fn load(path: &Path) -> io::Result<String> {
        let shown = path.display();
        let Some(utf8) = path.to_str() else {
            return Err(io::Error::other(format!("{shown} is not Unicode")));
        };
        // SAFETY: loading runs the `DllMain` of the file at that path, which is
        // trusted to be the DXC that packaging put there.
        let library = unsafe { Library::new(path) }.map_err(|err| {
            match err.source().and_then(|os| os.downcast_ref::<io::Error>()) {
                Some(os) => io::Error::new(os.kind(), format!("{shown}: {os}")),
                None => io::Error::other(format!("{shown}: {err}")),
            }
        })?;
        // It stays loaded, so every instance wgpu opens finds it in memory.
        mem::forget(library);
        Ok(utf8.to_owned())
    }

    /// The folder of the module this code is linked into: a plugin's own file,
    /// not its host's.
    fn module_dir() -> io::Result<PathBuf> {
        let flags =
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        let mut module = ptr::null_mut();
        // SAFETY: with FROM_ADDRESS the name is read as an address, here a
        // function in this module.
        if unsafe { GetModuleHandleExW(flags, module_dir as *const u16, &mut module) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // The longest path Windows allows, so the name is never cut short.
        let mut name = vec![0u16; 1 << 15];
        // SAFETY: `name` is as long as the size passed.
        let len = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) };
        if len == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut path = PathBuf::from(OsString::from_wide(&name[..len as usize]));
        path.pop();
        Ok(path)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn module_dir_matches_current_exe() {
            let exe = std::env::current_exe().unwrap();
            assert_eq!(module_dir().unwrap(), exe.parent().unwrap());
        }

        /// A DLL that loads, copied into `dir` as `dxcompiler.dll`.
        fn put_loadable_dll(dir: &Path) {
            let system = std::env::var_os("SystemRoot").unwrap();
            let source = Path::new(&system).join("System32").join("version.dll");
            std::fs::copy(source, dir.join("dxcompiler.dll")).unwrap();
        }

        #[test]
        fn load_first_prefers_shared() {
            let (shared, beside) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            put_loadable_dll(shared.path());
            put_loadable_dll(beside.path());
            let path = load_first(Some(shared.path()), beside.path()).unwrap();
            assert_eq!(Path::new(&path), shared.path().join("dxcompiler.dll"));
        }

        #[test]
        fn load_first_falls_back_past_a_bad_shared_dll() {
            let (shared, beside) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            std::fs::write(shared.path().join("dxcompiler.dll"), b"not a DLL").unwrap();
            put_loadable_dll(beside.path());
            let path = load_first(Some(shared.path()), beside.path()).unwrap();
            assert_eq!(Path::new(&path), beside.path().join("dxcompiler.dll"));
        }

        #[test]
        fn load_first_without_shared_dir_uses_beside() {
            let beside = tempfile::tempdir().unwrap();
            put_loadable_dll(beside.path());
            let path = load_first(None, beside.path()).unwrap();
            assert_eq!(Path::new(&path), beside.path().join("dxcompiler.dll"));
        }

        #[test]
        fn load_first_names_both_folders() {
            let (shared, beside) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            let err = load_first(Some(shared.path()), beside.path()).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::NotFound);
            let shown = err.to_string();
            assert!(shown.contains(&*shared.path().to_string_lossy()), "{shown}");
            assert!(shown.contains(&*beside.path().to_string_lossy()), "{shown}");
        }

        #[test]
        fn load_names_missing_dll() {
            let dir = tempfile::tempdir().unwrap();
            let err = load(&dir.path().join("dxcompiler.dll")).unwrap_err();
            assert!(err.to_string().contains("dxcompiler.dll"), "{err}");
        }

        #[test]
        fn shared_dir_is_in_an_existing_common_files() {
            let dir = shared_dir().unwrap();
            let common = dir.parent().and_then(Path::parent).unwrap();
            assert!(common.is_dir(), "{}", common.display());
            let name = dir.file_name().unwrap().to_string_lossy();
            assert!(name.starts_with("dxc-"), "{name}");
        }
    }
}

// Browsers' wgpu objects stay on their thread.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static SHARED: std::cell::RefCell<Option<GpuParts>> = const { std::cell::RefCell::new(None) };
}

#[cfg(not(target_arch = "wasm32"))]
static SHARED: std::sync::Mutex<Option<GpuParts>> = std::sync::Mutex::new(None);

#[derive(Debug)]
pub struct GpuError(String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU setup failed: {}", self.0)
    }
}

impl std::error::Error for GpuError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_up_to_the_longest_wait() {
        let mut backoff = ReopenBackoff::default();
        let delays: Vec<u64> = (0..8).map(|_| backoff.next_delay().as_secs()).collect();
        assert_eq!(delays, [1, 2, 4, 8, 16, 30, 30, 30]);
    }

    #[test]
    fn backoff_waits_for_the_same_adapter_through_the_short_delays() {
        let mut backoff = ReopenBackoff::default();
        let same: Vec<bool> = (0..7)
            .map(|_| {
                backoff.next_delay();
                backoff.same_adapter_only()
            })
            .collect();
        assert_eq!(same, [true, true, true, true, false, false, false]);
    }

    #[test]
    fn backoff_starts_over_only_after_a_device_that_lasted() {
        let mut backoff = ReopenBackoff::default();
        backoff.next_delay();
        backoff.next_delay();
        backoff.lost(Duration::from_secs(5));
        assert_eq!(backoff.next_delay().as_secs(), 4, "a device that died soon");
        backoff.lost(ReopenBackoff::STABLE_AFTER);
        assert_eq!(backoff.next_delay().as_secs(), 1, "a device that lasted");
    }

    #[test]
    fn same_device_follows_the_record_and_not_the_clone() {
        let (one, other) = (DeviceFaults::default(), DeviceFaults::default());
        assert!(one.same_device(&one.clone()));
        assert!(!one.same_device(&other));
    }
}
