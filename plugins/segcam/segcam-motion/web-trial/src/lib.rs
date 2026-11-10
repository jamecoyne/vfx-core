//! A page that draws segcam's swarm with [`segcam_motion::swarm_pass`] on
//! WebGL2 and compares it with the CPU picture: the browser half of the
//! render pass trial. `build.sh` builds and serves it.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use segcam_motion::motion::Segmentation;
use segcam_motion::render::{OutputMode, draw_output};
use segcam_motion::swarm::{SwarmMetrics, Tile, shadow_kernel};
use segcam_motion::swarm_pass::scene::{self, Difference};
use segcam_motion::swarm_pass::{Occlusion, Placement, SwarmPass};
use segcam_motion::{Argb, Channel, RenderSettings, View, ViewMut};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_time::Instant;

const CANVAS: (i32, i32) = (1920, 1080);

/// Resolves after the browser has run its other tasks, GPU work among them.
async fn yield_to_browser() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let window = web_sys::window().expect("a page has a window");
        window
            .set_timeout_with_callback(&resolve)
            .expect("a timeout can be set");
    });
    let _ = JsFuture::from(promise).await;
}

/// Draws `windows` over `frame` and waits for the pixels in `out`. Returns
/// the milliseconds from the submit to the last pixel copied.
async fn draw<C: Channel>(
    device: &wgpu::Device,
    pass: &mut SwarmPass,
    out: &mut [Argb<C>],
    windows: &[(&Tile, Placement)],
    frame: &[Argb<C>],
    s: &RenderSettings,
    occlusion: Occlusion,
) -> f64 {
    let started = Instant::now();
    let (w, h) = CANVAS;
    let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
    let frame = View::from_pixels(frame, w, h);
    pass.submit::<C>(
        CANVAS,
        windows.iter().copied(),
        Some(&frame),
        s,
        &shadow,
        occlusion,
    );
    let mapped = Arc::new(AtomicBool::new(false));
    let done = mapped.clone();
    pass.readback()
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            result.expect("the readback maps");
            done.store(true, Ordering::Release);
        });
    while !mapped.load(Ordering::Acquire) {
        let _ = device.poll(wgpu::PollType::Poll);
        if !mapped.load(Ordering::Acquire) {
            yield_to_browser().await;
        }
    }
    pass.copy_mapped(&mut ViewMut::from_pixels(out, w, h));
    started.elapsed().as_secs_f64() * 1000.0
}

/// Adds a table row per swarm case at depth `C` to `report`, and returns the
/// last picture drawn.
async fn compare<C: Channel>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    report: &mut String,
) -> Vec<Argb<C>> {
    let bits = 8 * size_of::<C>();
    let (w, h) = CANVAS;
    let mut drawn = vec![Argb::<C>::default(); (w * h) as usize];
    for (name, s) in scene::cases() {
        let tiles = scene::trail_tiles::<C>(CANVAS, 12, 3, s.mirror);
        let frame = scene::frame_pixels::<C>(w, h, 12);
        let mut pass = SwarmPass::new(device, queue, 8, 2);
        let placed = pass.upload(&tiles, h).expect("the atlas holds the tiles");
        let visible = scene::visible(&tiles, CANVAS, s.max_tiles);
        let mut expected = vec![Argb::<C>::default(); (w * h) as usize];
        draw_output(
            &mut ViewMut::from_pixels(&mut expected, w, h),
            &s,
            Some(View::from_pixels(&frame, w, h)),
            &Segmentation::default(),
            12,
            visible.iter().map(|(_, tile)| *tile),
        );
        let windows: Vec<_> = visible.iter().map(|&(i, tile)| (tile, placed[i])).collect();
        for occlusion in [Occlusion::None, Occlusion::Depth] {
            draw(
                device, &mut pass, &mut drawn, &windows, &frame, &s, occlusion,
            )
            .await;
            let Difference {
                max,
                over_half,
                over_one,
                ..
            } = scene::difference(&expected, &drawn);
            let _ = writeln!(
                report,
                "| {bits} bit | {w} x {h} | {name} | {occlusion:?} | {max:.2} levels | \
                 {over_half} pixels | {over_one} pixels |"
            );
        }
    }
    drawn
}

/// Adds the median and worst time of 100 draws of a 120 frame trail, of
/// which at most `max_tiles` windows stay, to `report`.
async fn time(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    (max_tiles, per_frame, layers): (i32, i32, u32),
    report: &mut String,
) {
    let (w, h) = CANVAS;
    let s = RenderSettings {
        mode: OutputMode::Swarm,
        max_tiles,
        ..RenderSettings::default()
    };
    let tiles = scene::trail_tiles::<u8>(CANVAS, 120, per_frame, false);
    let frame = scene::frame_pixels::<u8>(w, h, 120);
    let visible: Vec<usize> = scene::visible(&tiles, CANVAS, max_tiles)
        .into_iter()
        .map(|(index, _)| index)
        .collect();
    let shown: Vec<Tile> = (0..)
        .zip(tiles)
        .filter(|(index, _)| visible.binary_search(index).is_ok())
        .map(|(_, tile)| tile)
        .collect();
    let mut pass = SwarmPass::new(device, queue, layers, 2);
    let placed = pass.upload(&shown, h).expect("the atlas holds the tiles");
    let windows: Vec<_> = shown.iter().zip(placed).collect();
    let mut out = vec![Argb::<u8>::default(); (w * h) as usize];
    for occlusion in [Occlusion::None, Occlusion::Depth] {
        let mut times = Vec::new();
        for _ in 0..120 {
            times.push(draw(device, &mut pass, &mut out, &windows, &frame, &s, occlusion).await);
        }
        let times = &mut times[20..];
        times.sort_by(f64::total_cmp);
        let _ = writeln!(
            report,
            "| 8 bit | {} | render pass, {occlusion:?} | {:.2} ms | {:.2} ms |",
            windows.len(),
            times[times.len() / 2],
            times[times.len() - 1]
        );
    }
}

fn document() -> Result<web_sys::Document, JsValue> {
    web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| "the page has no document".into())
}

fn show(report: &str) {
    if let Some(log) = document().ok().and_then(|d| d.get_element_by_id("report")) {
        log.set_text_content(Some(report));
    }
}

/// Paints an 8-bit ARGB picture of the canvas size on the page's 2D canvas.
fn paint(pixels: &[Argb<u8>]) -> Result<(), JsValue> {
    let canvas = document()?
        .get_element_by_id("picture")
        .ok_or("the page has no picture canvas")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    canvas.set_width(CANVAS.0 as u32);
    canvas.set_height(CANVAS.1 as u32);
    let context = canvas
        .get_context("2d")?
        .ok_or("the canvas has no 2D context")?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;
    let rgba: Vec<u8> = pixels
        .iter()
        .flat_map(|p| [p.red, p.green, p.blue, p.alpha])
        .collect();
    let image = web_sys::ImageData::new_with_u8_clamped_array(
        wasm_bindgen::Clamped(&rgba),
        CANVAS.0 as u32,
    )?;
    context.put_image_data(&image, 0.0, 0.0)
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let canvas = document()?
        .create_element("canvas")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(|e| e.to_string())?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|e| e.to_string())?;
    let limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("segcam swarm pass page"),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .map_err(|e| e.to_string())?;

    let info = adapter.get_info();
    let mut report = format!("adapter: {} on {:?}\n", info.name, info.backend);
    use wgpu::TextureFormat as F;
    let renders = |format| {
        let usages = adapter.get_texture_format_features(format).allowed_usages;
        usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
    };
    for format in [F::Rgba8Unorm, F::Rgba16Float, F::Rgba16Uint, F::Rgba32Float] {
        let _ = writeln!(report, "{format:?} is a render target: {}", renders(format));
    }
    report += "| depth | canvas | case | occlusion | largest difference | over half a level | \
               over one level |\n";
    show(&report);

    let picture = compare::<u8>(&device, &queue, &mut report).await;
    paint(&picture)?;
    show(&report);
    if renders(F::Rgba16Float) && renders(F::Rgba16Uint) {
        compare::<u16>(&device, &queue, &mut report).await;
        show(&report);
    }
    if renders(F::Rgba16Float) && renders(F::Rgba32Float) {
        compare::<f32>(&device, &queue, &mut report).await;
        show(&report);
    }
    report += "| depth | windows | path | median | worst |\n";
    for scene in [(60, 3, 4), (1000, 9, 25)] {
        time(&device, &queue, scene, &mut report).await;
        show(&report);
    }
    report += "done\n";
    show(&report);
    let request = web_sys::RequestInit::new();
    request.set_method("POST");
    request.set_body(&JsValue::from_str(&report));
    let window = web_sys::window().ok_or("the page has no window")?;
    let _ = JsFuture::from(window.fetch_with_str_and_init("report", &request)).await;
    Ok(())
}
