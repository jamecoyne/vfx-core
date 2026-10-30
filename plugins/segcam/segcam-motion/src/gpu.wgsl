// segcam Motion's drawing on the GPU. One invocation per output pixel replays what the CPU
// path does to that pixel (render.rs, swarm.rs), rounding to the canvas depth where it does.
// Pixels are After Effects' ARGB at the host's depth, packed into u32 words.

struct Globals {
    width: i32,
    height: i32,
    // 0: 8 bit, 1: 16 bit (white is 32768), 2: float.
    depth: u32,
    flags: u32,
    frame_w: i32,
    frame_h: i32,
    // 0: clear, 1: black, 2: the frame.
    backdrop: u32,
    count: u32,
    items: u32,
    corner: u32,
    corner_lo: i32,
    corner_size: i32,
    title_height: i32,
    bar_h: i32,
    scale: f32,
    radius: f32,
    full: f32,
    opacity: f32,
    edge_shade: f32,
    line: i32,
    text_scale: i32,
    // Tables in `data`: 8-bit levels as units, and where title bars are sampled.
    units: u32,
    bar_x: u32,
    bar_y: u32,
    green: vec4<f32>,
}

const STRAIGHT: u32 = 1u;
const MIRRORED: u32 = 2u;
const LIVE: u32 = 4u;
const BOXES: u32 = 8u;
const LABELS: u32 = 16u;

const WINDOW_WORDS: u32 = 20u;
const BLOB_WORDS: u32 = 12u;
const JOB_WORDS: u32 = 16u;

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> frame: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage, read> data: array<u32>;
@group(0) @binding(4) var<storage, read_write> atlas: array<u32>;

fn has(flag: u32) -> bool {
    return (g.flags & flag) != 0u;
}

fn words() -> u32 {
    return 1u << g.depth;
}

fn white() -> f32 {
    return select(select(1.0, 32768.0, g.depth == 1u), 255.0, g.depth == 0u);
}

/// 8-bit levels as the CPU reads them, n / 255 rounded once.
fn units8(n: vec4<u32>) -> vec4<f32> {
    let o = g.units;
    return bitcast<vec4<f32>>(vec4<u32>(data[o + n.x], data[o + n.y], data[o + n.z],
        data[o + n.w]));
}

/// A frame pixel as (alpha, red, green, blue) in units.
fn load_frame(i: u32) -> vec4<f32> {
    if g.depth == 0u {
        let w = frame[i];
        return units8(vec4<u32>(w, w >> 8u, w >> 16u, w >> 24u) & vec4<u32>(255u));
    }
    if g.depth == 1u {
        let a = frame[2u * i];
        let b = frame[2u * i + 1u];
        return vec4<f32>(vec4<u32>(a, a >> 16u, b, b >> 16u) & vec4<u32>(65535u)) / 32768.0;
    }
    let o = 4u * i;
    return bitcast<vec4<f32>>(vec4<u32>(frame[o], frame[o + 1u], frame[o + 2u], frame[o + 3u]));
}

fn copy_raw(src: u32, dst: u32) {
    let n = words();
    for (var k = 0u; k < n; k++) {
        out[dst * n + k] = frame[src * n + k];
    }
}

/// `Channel::from_unit` at an integer depth.
fn level(v: vec4<f32>) -> vec4<u32> {
    return vec4<u32>(clamp(v, vec4<f32>(0.0), vec4<f32>(1.0)) * white() + 0.5);
}

/// What storing `v` at the canvas depth and reading it back gives.
fn quantize(v: vec4<f32>) -> vec4<f32> {
    if g.depth == 2u {
        return v;
    }
    if g.depth == 0u {
        return units8(level(v));
    }
    return vec4<f32>(level(v)) / white();
}

fn store(i: u32, v: vec4<f32>) {
    if g.depth == 0u {
        let n = level(v);
        out[i] = n.x | (n.y << 8u) | (n.z << 16u) | (n.w << 24u);
    } else if g.depth == 1u {
        let n = level(v);
        out[2u * i] = n.x | (n.y << 16u);
        out[2u * i + 1u] = n.z | (n.w << 16u);
    } else {
        let w = bitcast<vec4<u32>>(v);
        let o = 4u * i;
        out[o] = w.x;
        out[o + 1u] = w.y;
        out[o + 2u] = w.z;
        out[o + 3u] = w.w;
    }
}

struct Rect {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

fn rect(o: u32) -> Rect {
    return Rect(bitcast<i32>(data[o]), bitcast<i32>(data[o + 1u]), bitcast<i32>(data[o + 2u]),
        bitcast<i32>(data[o + 3u]));
}

fn inside(r: Rect, x: i32, y: i32) -> bool {
    return x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h;
}

fn float_at(o: u32) -> f32 {
    return bitcast<f32>(data[o]);
}

/// Where `copy_scaled` and live windows read the frame under canvas pixel (`x`, `y`).
fn frame_index(x: i32, y: i32) -> u32 {
    let dx = select(x, g.width - 1 - x, has(MIRRORED));
    let sx = dx * g.frame_w / g.width;
    let sy = y * g.frame_h / g.height;
    return u32(sy * g.frame_w + sx);
}

// ---------------------------------------------------------------------------------------
// Tiles: each cut from the frame into the atlas at 8 bits, as `make_tile` cuts it.

/// Which of a crop's `c` pixels tile pixel `t` of `l` shows: floor((t + 0.5) * c / l).
fn sample(t: i32, c: i32, l: i32) -> i32 {
    return min(c - 1, (2 * t + 1) * c / (2 * l));
}

@compute @workgroup_size(16, 16)
fn cut(@builtin(global_invocation_id) id: vec3<u32>) {
    let job = g.items + id.z * JOB_WORDS;
    let video = rect(job);
    let stored = rect(job + 4u);
    let crop = rect(job + 8u);
    let x = i32(id.x);
    let y = i32(id.y);
    if x >= stored.w || y >= stored.h {
        return;
    }
    var tx = stored.x + x - video.x;
    if has(MIRRORED) {
        tx = video.w - 1 - tx;
    }
    let sx = crop.x + sample(tx, crop.w, video.w);
    let sy = crop.y + sample(stored.y + y - video.y, crop.h, video.h);
    let p = load_frame(u32(sy * g.frame_w + sx));
    let n = vec4<u32>(clamp(p, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + 0.5);
    atlas[data[job + 12u] + u32(y * stored.w + x)] = n.y | (n.z << 8u) | (n.w << 16u);
}

// ---------------------------------------------------------------------------------------
// The overlay: the frame, each segment's box, its ID chip and the label in it.

@compute @workgroup_size(16, 16)
fn overlay(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    if x >= g.width || y >= g.height {
        return;
    }
    let black = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    var color = vec4<f32>(0.0);
    var painted = false;
    let line = g.line;
    let s = g.text_scale;
    for (var b = 0u; b < g.count; b++) {
        let o = g.items + b * BLOB_WORDS;
        let r = rect(o);
        if has(BOXES) && (inside(Rect(r.x, r.y, r.w, line), x, y)
            || inside(Rect(r.x, r.y + r.h - line, r.w, line), x, y)
            || inside(Rect(r.x, r.y, line, r.h), x, y)
            || inside(Rect(r.x + r.w - line, r.y, line, r.h), x, y)) {
            color = g.green;
            painted = true;
        }
        if !has(LABELS) {
            continue;
        }
        if inside(rect(o + 4u), x, y) {
            color = g.green;
            painted = true;
        }
        let rx = x - bitcast<i32>(data[o + 8u]);
        let ry = y - bitcast<i32>(data[o + 9u]);
        let chars = data[o + 10u];
        if rx < 0 || ry < 0 || ry >= 7 * s {
            continue;
        }
        let c = u32(rx / (6 * s));
        let col = rx % (6 * s) / s;
        if c >= chars || col >= 5 {
            continue;
        }
        let row = u32(ry / s);
        let glyph = data[o + 11u] + 2u * c;
        let bits = (data[glyph + row / 4u] >> (8u * (row % 4u))) & 255u;
        if (bits & (1u << u32(4 - col))) != 0u {
            color = black;
            painted = true;
        }
    }
    let i = u32(y * g.width + x);
    if painted {
        store(i, quantize(color));
    } else {
        copy_raw(frame_index(x, y), i);
    }
}

// ---------------------------------------------------------------------------------------
// The swarm: each window in pile order, as `draw_window_with` draws it.

fn corner_loss(i: i32, j: i32) -> f32 {
    let n = g.corner_size;
    if i < 0 || j < 0 || i >= n || j >= n {
        return 0.0;
    }
    return float_at(g.corner + u32(j * n + i));
}

fn bar_at(bar: u32, width: i32, x: i32, y: i32) -> vec3<f32> {
    let o = bar + 3u * u32(y * width + x);
    return vec3<f32>(bitcast<f32>(atlas[o]), bitcast<f32>(atlas[o + 1u]),
        bitcast<f32>(atlas[o + 2u]));
}

fn lerp(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

/// Where a title bar is sampled for window pixel `i`, from the CPU's positions in f64 (whole
/// part and fraction), held to the bar's pixels `0..=last`.
fn bar_coord(table: u32, i: i32, last: i32) -> vec2<f32> {
    let o = table + 2u * u32(i);
    let whole = bitcast<i32>(data[o]);
    let t = float_at(o + 1u);
    if whole < 0 {
        return vec2<f32>(0.0);
    }
    if whole > last || (whole == last && t > 0.0) {
        return vec2<f32>(f32(last), 0.0);
    }
    return vec2<f32>(f32(whole), t);
}

/// The title bar at 1x, sampled for window pixel (`lx`, `ly`).
fn sample_bar(bar: u32, width: i32, lx: i32, ly: i32) -> vec3<f32> {
    let cx = bar_coord(g.bar_x, lx, width - 1);
    let cy = bar_coord(g.bar_y, ly, g.bar_h - 1);
    let x0 = i32(cx.x);
    let y0 = i32(cy.x);
    let x1 = min(x0 + 1, width - 1);
    let y1 = min(y0 + 1, g.bar_h - 1);
    let tx = cx.y;
    let ty = cy.y;
    let top = lerp(bar_at(bar, width, x0, y0), bar_at(bar, width, x1, y0), tx);
    let bottom = lerp(bar_at(bar, width, x0, y1), bar_at(bar, width, x1, y1), tx);
    return lerp(top, bottom, ty);
}

/// Draws window `w` over `px`; false if the window doesn't reach the pixel.
fn draw_window(w: u32, x: i32, y: i32, px: ptr<function, vec4<f32>>) -> bool {
    let o = g.items + w * WINDOW_WORDS;
    let area = rect(o + 4u);
    if !inside(area, x, y) {
        return false;
    }
    let frame_r = rect(o);
    let stored = rect(o + 8u);
    let plan = data[o + 12u];
    let tile = data[o + 13u];
    let pixels = data[o + 14u] != 0u;
    let bar = data[o + 15u];
    let bar_w = bitcast<i32>(data[o + 16u]);

    let aw = u32(area.w);
    let ah = u32(area.h);
    let j = u32(x - area.x);
    let i = u32(y - area.y);
    let profile_x = bitcast<f32>(atlas[plan + j]);
    let corner_x = bitcast<i32>(atlas[plan + aw + j]);
    let dx = bitcast<f32>(atlas[plan + 2u * aw + j]);
    let rows = plan + 3u * aw;
    let profile_y = bitcast<f32>(atlas[rows + i]);
    let corner_y = bitcast<i32>(atlas[rows + ah + i]);
    let dy = bitcast<f32>(atlas[rows + 2u * ah + i]);

    let lx = x - frame_r.x;
    let ly = y - frame_r.y;
    let vx = x - stored.x;
    let vy = y - stored.y;
    let in_stored = vx >= 0 && vx < stored.w && vy >= 0 && vy < stored.h;
    var cov = g.full;
    if !(dx == 0.0 && dy == 0.0) {
        cov = clamp(g.radius + 0.5 - sqrt(dx * dx + dy * dy), 0.0, 1.0);
    }
    var tile_color = vec3<f32>(0.0);
    if pixels && in_stored {
        let t = atlas[tile + u32(vy * stored.w + vx)];
        tile_color = units8(vec4<u32>(0u, t, t >> 8u, t >> 16u) & vec4<u32>(255u)).yzw;
    }
    // At 8 bits a frozen tile's pixel under the whole mask is the tile's own level.
    if g.depth == 0u && !has(LIVE) && pixels && ly > g.title_height && ly < frame_r.h
        && lx >= 0 && lx < frame_r.w && in_stored && cov == 1.0 {
        *px = vec4<f32>(1.0, tile_color);
        return true;
    }

    var a = (*px).x;
    var p = (*px).yzw;
    if has(STRAIGHT) {
        p = p * a;
    }
    let shadow = max(0.0, g.opacity * (profile_x * profile_y - corner_loss(corner_x, corner_y)));
    p = p * (1.0 - shadow);
    a += shadow * (1.0 - a);

    if lx >= 0 && lx < frame_r.w && ly >= 0 && ly < frame_r.h {
        var c: vec3<f32>;
        if ly < g.title_height {
            c = sample_bar(bar, bar_w, lx, ly);
        } else {
            c = tile_color;
            if has(LIVE) {
                c = load_frame(frame_index(x, y)).yzw;
            }
            if ly - g.title_height < 1 {
                c = c * g.edge_shade;
            }
        }
        p = lerp(p, c, cov);
        a += cov * (1.0 - a);
    }
    if has(STRAIGHT) && a > 0.0 && a < 1.0 {
        p = p / a;
    }
    *px = quantize(vec4<f32>(a, p));
    return true;
}

/// Whether window `w` covers the pixel whole, so nothing under it shows: exactly at 8 and 16
/// bits, within a rounding step at float.
fn covers(w: u32, x: i32, y: i32) -> bool {
    let o = g.items + w * WINDOW_WORDS;
    if !inside(rect(o), x, y) {
        return false;
    }
    let area = rect(o + 4u);
    let plan = data[o + 12u];
    let aw = u32(area.w);
    let ah = u32(area.h);
    let dx = bitcast<f32>(atlas[plan + 2u * aw + u32(x - area.x)]);
    let dy = bitcast<f32>(atlas[plan + 3u * aw + 2u * ah + u32(y - area.y)]);
    if dx == 0.0 && dy == 0.0 {
        return g.full == 1.0;
    }
    return clamp(g.radius + 0.5 - sqrt(dx * dx + dy * dy), 0.0, 1.0) == 1.0;
}

// Which windows of the current batch reach this workgroup's 16 x 16 pixels.
var<workgroup> reaches: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn swarm(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wg: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    let on_canvas = x < g.width && y < g.height;
    let block = Rect(i32(wg.x) * 16, i32(wg.y) * 16, 16, 16);
    var px = vec4<f32>(0.0);
    if g.backdrop == 1u {
        px = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    } else if g.backdrop == 2u && on_canvas {
        px = load_frame(frame_index(x, y));
    }
    var touched = false;
    for (var first = 0u; first < g.count; first += 256u) {
        var hit = 0u;
        if first + li < g.count {
            let area = rect(g.items + (first + li) * WINDOW_WORDS + 4u);
            hit = u32(area.x < block.x + block.w && block.x < area.x + area.w
                && area.y < block.y + block.h && block.y < area.y + area.h);
        }
        workgroupBarrier();
        reaches[li] = hit;
        workgroupBarrier();
        if on_canvas {
            let n = min(256u, g.count - first);
            // Drawing starts at the newest window that covers the pixel whole.
            var start = 0u;
            for (var k = n; k > 0u; k--) {
                if reaches[k - 1u] != 0u && covers(first + k - 1u, x, y) {
                    start = k - 1u;
                    break;
                }
            }
            for (var k = start; k < n; k++) {
                if reaches[k] != 0u && draw_window(first + k, x, y, &px) {
                    touched = true;
                }
            }
        }
    }
    if !on_canvas {
        return;
    }
    let i = u32(y * g.width + x);
    if g.backdrop == 2u && !touched {
        copy_raw(frame_index(x, y), i);
    } else {
        store(i, quantize(px));
    }
}
