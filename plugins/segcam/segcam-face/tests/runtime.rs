#![cfg(feature = "onnx")]
use segcam_face::runtime::{self, Device, OrtFaceMesh, OrtPicoDet, OrtYuNet};
use segcam_face::{
    BodyDetector, Detector, FACE_MESH_MODEL, FaceSegmenter, Image, PICODET_MODEL, Parts, Pixels,
    YUNET_MODEL,
};

/// Cargo copies DirectML.dll to target/<profile>, but tests run from its deps folder, where Windows
/// would fall back to system32's copy, too old on Windows 10. Loads the build's copy first.
#[cfg(windows)]
fn load_directml() {
    use std::os::windows::ffi::OsStrExt;
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let exe = std::env::current_exe().expect("the test binary's path");
        let Some(dll) = exe
            .ancestors()
            .skip(1)
            .take(2)
            .map(|dir| dir.join("DirectML.dll"))
            .find(|dll| dll.exists())
        else {
            return;
        };
        let name: Vec<u16> = dll.as_os_str().encode_wide().chain([0]).collect();
        // SAFETY: a null-terminated path; the library stays loaded for the whole test run.
        unsafe { LoadLibraryW(name.as_ptr()) };
    });
}

#[cfg(not(windows))]
fn load_directml() {}

fn devices() -> Vec<Device> {
    load_directml();
    #[allow(unused_mut)]
    let mut devices = vec![Device::Cpu];
    // DirectML where it loads. Not every Windows machine has a GPU it can use (CI's don't),
    // so SEGCAM_REQUIRE_DIRECTML=1 makes a failure to load an error, on a machine that has.
    #[cfg(windows)]
    match OrtYuNet::new(YUNET_MODEL, Device::DirectMl(0)) {
        Ok(_) => devices.push(Device::DirectMl(0)),
        Err(e) if std::env::var_os("SEGCAM_REQUIRE_DIRECTML").is_some() => {
            panic!("DirectML didn't load ({e})")
        }
        Err(e) => eprintln!("DirectML didn't load ({e}); testing the CPU only"),
    }
    eprintln!("devices: {devices:?}");
    devices
}

const ALL: Parts = Parts {
    face: true,
    eyes: true,
    mouth: true,
    body: true,
};

#[test]
fn a_flat_frame_has_no_faces_or_bodies() {
    let bytes = vec![90u8; 320 * 180 * 4];
    let floats = vec![0.35f32; 320 * 180 * 4];
    for device in devices() {
        let mut faces = OrtYuNet::new(YUNET_MODEL, device).unwrap();
        let mut bodies = OrtPicoDet::new(PICODET_MODEL, device).unwrap();
        for pixels in [
            Pixels::Rgba8(&bytes),
            Pixels::Bgra8(&bytes),
            Pixels::RgbaF32(&floats),
        ] {
            let image = Image {
                pixels,
                width: 320,
                height: 180,
                stride: 320,
            };
            assert!(faces.detect(&image).is_empty(), "{device}");
            assert!(bodies.detect(&image).is_empty(), "{device}");
        }
    }
}

#[test]
fn odd_frames_find_nothing_and_do_not_panic() {
    for device in devices() {
        let mut segmenter = FaceSegmenter::new(
            OrtYuNet::new(YUNET_MODEL, device).unwrap(),
            OrtFaceMesh::new(FACE_MESH_MODEL, device).ok(),
            OrtPicoDet::new(PICODET_MODEL, device).ok(),
        );
        // Empty, tiny, too short for its size, and a stride shorter than a row.
        for (w, h, stride, len) in [(0, 0, 0, 0), (1, 1, 1, 4), (7, 3, 7, 10), (4, 2, 3, 32)] {
            let data = vec![200u8; len];
            let image = Image {
                pixels: Pixels::Rgba8(&data),
                width: w,
                height: h,
                stride,
            };
            assert!(segmenter.find(&image, ALL).is_empty(), "{device}: {w}x{h}");
        }
    }
}

#[test]
fn the_fastest_device_loads_everything() {
    load_directml();
    let (mut segmenter, device) = runtime::fastest(true, true).expect("a device");
    assert!(
        segmenter.mesh.is_some() && segmenter.bodies.is_some(),
        "{device}"
    );
    let data = vec![128u8; 64 * 64 * 4];
    let image = Image {
        pixels: Pixels::Rgba8(&data),
        width: 64,
        height: 64,
        stride: 64,
    };
    assert!(segmenter.find(&image, ALL).is_empty());
}

#[test]
fn the_best_segmenter_finds_nothing_in_a_flat_frame() {
    load_directml();
    let mut segmenter = segcam_face::Segmenter::best().expect("a segmenter");
    let data = vec![128u8; 320 * 180 * 4];
    let image = Image {
        pixels: Pixels::Bgra8(&data),
        width: 320,
        height: 180,
        stride: 320,
    };
    assert!(segmenter.find(&image, ALL).is_empty());
}
