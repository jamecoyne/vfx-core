//! The models on ONNX Runtime: its CPU kernels everywhere, and DirectML on a GPU on Windows.

use std::fmt;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorRef;

use crate::{
    Body, BodyDetector, Detector, FACE_MESH_MODEL, Face, FaceSegmenter, Image, Landmarker,
    PICODET_MODEL, Pixels, Rect, YUNET_MODEL, mesh, picodet, yunet,
};

/// Where the models run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// DirectML on the DXGI adapter with this index; 0 is the display adapter.
    #[cfg(windows)]
    DirectMl(u32),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            #[cfg(windows)]
            Self::DirectMl(adapter) => match adapter_name(*adapter) {
                Some(name) => write!(f, "DirectML {adapter} ({name})"),
                None => write!(f, "DirectML {adapter}"),
            },
        }
    }
}

#[cfg(windows)]
fn adapter_name(adapter: u32) -> Option<String> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
    let desc = unsafe { factory.EnumAdapters1(adapter).and_then(|a| a.GetDesc1()) }.ok()?;
    Some(
        String::from_utf16_lossy(&desc.Description)
            .trim_end_matches('\0')
            .to_owned(),
    )
}

/// The bundled models on ONNX Runtime.
pub type OrtSegmenter = FaceSegmenter<OrtYuNet, OrtFaceMesh, OrtPicoDet>;

/// The bundled models on the fastest device, with the mesh if `mesh` and bodies if `bodies`, or
/// `None` if no device loads the detector. The first call times each device; later calls reuse
/// its choice.
pub fn fastest(mesh: bool, bodies: bool) -> Option<(OrtSegmenter, Device)> {
    static CHOSEN: OnceLock<Device> = OnceLock::new();
    let (detector, face_mesh, device) = match CHOSEN.get() {
        Some(&device) => {
            let detector = OrtYuNet::new(YUNET_MODEL, device).ok()?;
            let face_mesh = mesh.then(|| OrtFaceMesh::new(FACE_MESH_MODEL, device).ok());
            (detector, face_mesh.flatten(), device)
        }
        None => {
            let grey = vec![128; 640 * 360 * 4];
            let image = Image {
                pixels: Pixels::Rgba8(&grey),
                width: 640,
                height: 360,
                stride: 640,
            };
            let best = trials(&image, mesh).into_iter().next()?;
            (
                best.detector,
                best.mesh,
                *CHOSEN.get_or_init(|| best.device),
            )
        }
    };
    let bodies = bodies
        .then(|| OrtPicoDet::new(PICODET_MODEL, device).ok())
        .flatten();
    Some((FaceSegmenter::new(detector, face_mesh, bodies), device))
}

/// The face models loaded on one device, and how long they took there.
pub struct Trial {
    pub device: Device,
    pub detector: OrtYuNet,
    pub mesh: Option<OrtFaceMesh>,
    /// Median time of a detection, plus a mesh run when the mesh is loaded.
    pub time: Duration,
}

const TIMED_RUNS: usize = 9;
#[cfg(windows)]
const MAX_ADAPTERS: u32 = 8;

/// Loads the face models, with the mesh if `mesh`, on the CPU and each DirectML adapter, times a
/// few runs on `image`, and returns the devices that worked, fastest first.
pub fn trials(image: &Image, mesh: bool) -> Vec<Trial> {
    let devices = std::iter::once(Device::Cpu);
    #[cfg(windows)]
    let devices = devices.chain((0..MAX_ADAPTERS).map(Device::DirectMl));
    let mut trials = Vec::new();
    for device in devices {
        let loaded = OrtYuNet::new(YUNET_MODEL, device).and_then(|detector| {
            let mesh = mesh
                .then(|| OrtFaceMesh::new(FACE_MESH_MODEL, device))
                .transpose()?;
            Ok((detector, mesh))
        });
        match loaded {
            Ok((mut detector, mut mesh)) => {
                let time = time_runs(&mut detector, mesh.as_mut(), image);
                trials.push(Trial {
                    device,
                    detector,
                    mesh,
                    time,
                });
            }
            // Adapters are numbered from 0, so the first that fails is past the last.
            Err(_) if device != Device::Cpu => break,
            Err(_) => {}
        }
    }
    trials.sort_by_key(|trial| trial.time);
    trials
}

fn time_runs(
    detector: &mut OrtYuNet,
    mut mesh: Option<&mut OrtFaceMesh>,
    image: &Image,
) -> Duration {
    let face = Face {
        rect: Rect {
            x: 0.4,
            y: 0.3,
            w: 0.2,
            h: 0.4,
        },
        landmarks: [
            (0.45, 0.4),
            (0.55, 0.4),
            (0.5, 0.5),
            (0.46, 0.6),
            (0.54, 0.6),
        ],
        score: 1.0,
    };
    let mut run = || {
        detector.detect(image);
        if let Some(mesh) = mesh.as_deref_mut() {
            mesh.landmarks(image, &face);
        }
    };
    run();
    run();
    // The median, since a GPU's first runs can stall.
    let mut times: Vec<Duration> = (0..TIMED_RUNS)
        .map(|_| {
            let started = Instant::now();
            run();
            started.elapsed()
        })
        .collect();
    times.sort();
    times[TIMED_RUNS / 2]
}

static CPU_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Caps the CPU threads for models loaded from here on; 0 leaves it to ONNX Runtime.
pub fn set_cpu_threads(threads: usize) {
    CPU_THREADS.store(threads, Ordering::Relaxed);
}

fn session(model: &[u8], device: Device) -> ort::Result<Session> {
    // ONNX Runtime allows one environment per process.
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    let environment = match ENVIRONMENT.get() {
        Some(environment) => environment,
        None => {
            let environment = ort::init().with_name("segcam").build()?;
            ENVIRONMENT.get_or_init(|| environment)
        }
    };
    let builder = Session::builder(environment)?;
    let mut builder = match device {
        Device::Cpu => match CPU_THREADS.load(Ordering::Relaxed) {
            0 => builder,
            threads => builder.with_intra_threads(threads)?,
        },
        // DirectML can't use memory patterns or run nodes in parallel.
        #[cfg(windows)]
        Device::DirectMl(adapter) => builder
            .with_memory_pattern(false)?
            .with_parallel_execution(false)?
            .with_execution_providers([ort::ep::DirectML::default()
                .with_device_id(adapter as i32)
                .build()
                .error_on_failure()])?,
    };
    builder.commit_from_memory(model)
}

/// YuNet: faces and their five points, at most four, best first.
pub struct OrtYuNet {
    session: Session,
    input: Vec<f32>,
    /// Output indices of the class, objectness, box and landmark maps for each stride.
    maps: [usize; 12],
    pub score_threshold: f32,
    pub nms_threshold: f32,
    pub max_faces: usize,
}

const YUNET_MAPS: [&str; 12] = [
    "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
];

impl OrtYuNet {
    /// Loads `model` and runs it once to warm up the device.
    pub fn new(model: &[u8], device: Device) -> ort::Result<Self> {
        let session = session(model, device)?;
        let mut maps = [0; 12];
        for (index, name) in maps.iter_mut().zip(YUNET_MAPS) {
            *index = session
                .outputs()
                .iter()
                .position(|o| o.name() == name)
                .ok_or_else(|| ort::Error::new(format!("YuNet has no output {name}")))?;
        }
        let mut detector = Self {
            session,
            input: vec![0.0; 3 * yunet::INPUT * yunet::INPUT],
            maps,
            score_threshold: 0.6,
            nms_threshold: 0.3,
            max_faces: 4,
        };
        detector.run()?;
        Ok(detector)
    }

    fn run(&mut self) -> ort::Result<Vec<Vec<f32>>> {
        let shape = [1, 3, yunet::INPUT, yunet::INPUT];
        let outputs = self.session.run(ort::inputs![TensorRef::from_array_view((
            shape,
            self.input.as_slice()
        ))?])?;
        let map = |&i: &usize| Ok(outputs[i].try_extract_tensor::<f32>()?.1.to_vec());
        self.maps.iter().map(map).collect()
    }
}

impl Detector for OrtYuNet {
    /// Faces in `image`, best first. An unusable image or a failed run gives none.
    fn detect(&mut self, image: &Image) -> Vec<Face> {
        let Some(scale) = yunet::fill_input(image, &mut self.input) else {
            return Vec::new();
        };
        let Ok(outputs) = self.run() else {
            return Vec::new();
        };
        let maps: [&[f32]; 12] = std::array::from_fn(|i| outputs[i].as_slice());
        let (score, nms, max) = (self.score_threshold, self.nms_threshold, self.max_faces);
        yunet::faces(&maps, image, scale, score, nms, max)
    }
}

/// PicoDet-S: people, best first.
pub struct OrtPicoDet {
    session: Session,
    input: Vec<f32>,
    pub score_threshold: f32,
    pub nms_threshold: f32,
    /// People covering more of the frame than this are left out as close-ups.
    pub max_area: f64,
}

impl OrtPicoDet {
    /// Loads `model` and runs it once to warm up the device.
    pub fn new(model: &[u8], device: Device) -> ort::Result<Self> {
        let mut detector = Self {
            session: session(model, device)?,
            input: vec![0.0; 3 * picodet::INPUT * picodet::INPUT],
            score_threshold: 0.4,
            nms_threshold: 0.5,
            max_area: 0.6,
        };
        detector.run()?;
        Ok(detector)
    }

    fn run(&mut self) -> ort::Result<Vec<Vec<f32>>> {
        let shape = [1, 3, picodet::INPUT, picodet::INPUT];
        let outputs = self.session.run(ort::inputs![TensorRef::from_array_view((
            shape,
            self.input.as_slice()
        ))?])?;
        (0..8)
            .map(|i| Ok(outputs[i].try_extract_tensor::<f32>()?.1.to_vec()))
            .collect()
    }
}

impl BodyDetector for OrtPicoDet {
    /// People in `image`, best first. An unusable image or a failed run gives none.
    fn detect(&mut self, image: &Image) -> Vec<Body> {
        if picodet::fill_input(image, &mut self.input).is_none() {
            return Vec::new();
        }
        let Ok(outputs) = self.run() else {
            return Vec::new();
        };
        let maps: [&[f32]; 8] = std::array::from_fn(|i| outputs[i].as_slice());
        picodet::bodies(
            &maps,
            self.score_threshold,
            self.nms_threshold,
            self.max_area,
        )
    }
}

/// MediaPipe Face Mesh V2: 478 landmarks of a face.
pub struct OrtFaceMesh {
    session: Session,
    input: Vec<f32>,
}

impl OrtFaceMesh {
    /// Loads `model` and runs it once to warm up the device.
    pub fn new(model: &[u8], device: Device) -> ort::Result<Self> {
        let input = vec![0.0; mesh::INPUT * mesh::INPUT * 3];
        let mut mesh = Self {
            session: session(model, device)?,
            input,
        };
        mesh.run()?;
        Ok(mesh)
    }

    /// The landmarks (x, y, z in crop pixels) and the face presence score.
    fn run(&mut self) -> ort::Result<(Vec<f32>, f32)> {
        let shape = [1, mesh::INPUT, mesh::INPUT, 3];
        let outputs = self.session.run(ort::inputs![TensorRef::from_array_view((
            shape,
            self.input.as_slice()
        ))?])?;
        let points = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
        let presence = outputs[1].try_extract_tensor::<f32>()?.1.first().copied();
        Ok((points, presence.unwrap_or(f32::MIN)))
    }
}

impl Landmarker for OrtFaceMesh {
    fn landmarks(&mut self, image: &Image, face: &Face) -> Option<Vec<(f64, f64)>> {
        let crop = mesh::crop(image, face);
        mesh::fill_input(image, &crop, &mut self.input)?;
        let (points, presence) = self.run().ok()?;
        mesh::landmarks_from(&points, presence, &crop, image)
    }
}
