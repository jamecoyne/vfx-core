// The swarm drawn with render passes (swarm_pass.rs): one instanced quad per
// window composited premultiplied, then a pass that writes the host's ARGB.

struct Globals {
    canvas: vec2<i32>,
    frame_size: vec2<i32>,
    flags: u32,
    // 0: 8 bit, 1: 16 bit (white is 32768), 2: float.
    depth: u32,
    window_count: u32,
    title_height: i32,
    bar_height: i32,
    corner_min_offset: i32,
    corner_size: i32,
    reach: i32,
    scale: f32,
    radius: f32,
    inner_coverage: f32,
    sigma: f32,
    offset_y: f32,
    opacity: f32,
    edge_shade: f32,
}

const STRAIGHT: u32 = 1u;
const MIRRORED: u32 = 2u;
const LIVE: u32 = 4u;

@group(0) @binding(0) var<uniform> g: Globals;
// Tile and frame texels hold ARGB bytes: `.gba` is the colour, `.r` alpha.
@group(0) @binding(1) var tiles: texture_2d_array<f32>;
@group(0) @binding(2) var bars: texture_2d_array<f32>;
@group(0) @binding(3) var corner: texture_2d<f32>;
@group(0) @binding(4) var frame: texture_2d<f32>;
@group(0) @binding(5) var frame_16: texture_2d<u32>;
@group(1) @binding(0) var composed: texture_2d<f32>;

struct Instance {
    @location(0) window: vec4<i32>,
    @location(1) stored: vec4<i32>,
    // Atlas x, y and layer, then whether the tile has pixels.
    @location(2) tile: vec4<i32>,
    // Atlas x, y and layer, then the bar's width.
    @location(3) bar: vec4<i32>,
    @location(4) order: u32,
}

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) window: vec4<i32>,
    @location(1) @interpolate(flat) stored: vec4<i32>,
    @location(2) @interpolate(flat) tile: vec4<i32>,
    @location(3) @interpolate(flat) bar: vec4<i32>,
}

fn has(flag: u32) -> bool {
    return (g.flags & flag) != 0u;
}

fn unit_corner(vertex: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0));
    return corners[vertex % 6u];
}

// Newer windows are nearer, so the depth test rejects what they cover.
fn varyings(pixel: vec2<f32>, i: Instance) -> Varyings {
    let ndc = pixel / vec2<f32>(g.canvas) * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let z = 1.0 - f32(i.order + 1u) / f32(g.window_count + 1u);
    return Varyings(vec4(ndc, z, 1.0), i.window, i.stored, i.tile, i.bar);
}

// The window and as far as its shadow reaches.
@vertex
fn vs_window(@builtin(vertex_index) vertex: u32, i: Instance) -> Varyings {
    let reach = f32(g.reach);
    let origin = vec2<f32>(i.window.xy) - reach;
    let size = vec2<f32>(i.window.zw) + 2.0 * reach;
    return varyings(origin + unit_corner(vertex) * size, i);
}

// The pixels of a side of `len` whose centres lie between the rounded corners.
fn between_corners(len: i32) -> vec2<f32> {
    let first = ceil(g.radius - 0.5);
    let end = floor(f32(len) - g.radius - 0.5) + 1.0;
    return vec2(first, max(first, end));
}

// The pixels the mask covers whole: two bands that cross, leaving the corners.
@vertex
fn vs_body(@builtin(vertex_index) vertex: u32, i: Instance) -> Varyings {
    let size = vec2<f32>(i.window.zw);
    var low = vec2(0.0);
    var high = size;
    if g.inner_coverage < 1.0 {
        high = low;
    } else if vertex < 6u {
        let span = between_corners(i.window.z);
        low.x = span.x;
        high.x = span.y;
    } else {
        let span = between_corners(i.window.w);
        low.y = span.x;
        high.y = span.y;
    }
    let pixel = vec2<f32>(i.window.xy) + mix(low, high, unit_corner(vertex));
    return varyings(pixel, i);
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let at = vec2<f32>(vec2(vertex & 1u, vertex >> 1u)) * 4.0 - 1.0;
    return vec4(at, 0.0, 1.0);
}

// The frame under canvas pixel `p` as (red, green, blue, alpha), laid on the
// canvas as `copy_scaled` lays it.
fn frame_at(p: vec2<i32>) -> vec4<f32> {
    let x = select(p.x, g.canvas.x - 1 - p.x, has(MIRRORED));
    let at = vec2(x, p.y) * g.frame_size / g.canvas;
    if g.depth == 1u {
        return vec4<f32>(textureLoad(frame_16, at, 0).gbar) / 32768.0;
    }
    return textureLoad(frame, at, 0).gbar;
}

fn bar_texel(bar: vec4<i32>, p: vec2<i32>) -> vec3<f32> {
    return textureLoad(bars, bar.xy + p, bar.z, 0).rgb;
}

// The title bar, kept at 1x, under window pixel `local`.
fn bar_at(bar: vec4<i32>, local: vec2<i32>) -> vec3<f32> {
    let last = vec2<f32>(vec2(bar.w, g.bar_height) - 1);
    let step = vec2(1.0 / g.scale, f32(g.bar_height) / f32(g.title_height));
    let at = clamp((vec2<f32>(local) + 0.5) * step - 0.5, vec2(0.0), last);
    let p0 = vec2<i32>(at);
    let p1 = min(p0 + 1, vec2<i32>(last));
    let t = at - vec2<f32>(p0);
    let top = mix(bar_texel(bar, p0), bar_texel(bar, vec2(p1.x, p0.y)), t.x);
    let bottom = mix(bar_texel(bar, vec2(p0.x, p1.y)), bar_texel(bar, p1), t.x);
    return mix(top, bottom, t.y);
}

// What the window shows at canvas pixel `p`, which is inside it.
fn window_color(v: Varyings, p: vec2<i32>) -> vec3<f32> {
    let local = p - v.window.xy;
    if local.y < g.title_height {
        return bar_at(v.bar, local);
    }
    var color = vec3(0.0);
    let at = p - v.stored.xy;
    if has(LIVE) {
        color = frame_at(p).rgb;
    } else if v.tile.w != 0 && all(at >= vec2(0)) && all(at < v.stored.zw) {
        color = textureLoad(tiles, v.tile.xy + at, v.tile.z, 0).gba;
    }
    if local.y == g.title_height {
        color *= g.edge_shade;
    }
    return color;
}

// Abramowitz and Stegun 7.1.26, within 1.5e-7.
fn erf(x: vec2<f32>) -> vec2<f32> {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - poly * exp(-a * a));
}

fn normal_cdf(z: vec2<f32>) -> vec2<f32> {
    return 0.5 + 0.5 * erf(z * 0.70710678);
}

// How much of the light the window's shadow takes at pixel centre `centre`:
// a blurred box less what the rounded corners take from it (`ShadowKernel`).
fn shadow(window: vec4<i32>, centre: vec2<f32>) -> f32 {
    let size = vec2<f32>(window.zw);
    let near = centre - vec2<f32>(window.xy) - vec2(0.0, g.offset_y);
    let profile = normal_cdf(near / g.sigma) - normal_cdf((near - size) / g.sigma);
    let inward = select(size - near, near, near < size * 0.5);
    let index = vec2<i32>(floor(inward)) - g.corner_min_offset;
    var loss = 0.0;
    if all(index >= vec2(0)) && all(index < vec2(g.corner_size)) {
        loss = textureLoad(corner, index, 0).r;
    }
    return max(0.0, g.opacity * (profile.x * profile.y - loss));
}

// A window's shadow and masked picture as one premultiplied source.
@fragment
fn fs_window(v: Varyings) -> @location(0) vec4<f32> {
    let p = vec2<i32>(v.position.xy);
    let shadowed = shadow(v.window, vec2<f32>(p) + 0.5);
    let local = p - v.window.xy;
    var coverage = 0.0;
    var color = vec3(0.0);
    if all(local >= vec2(0)) && all(local < v.window.zw) {
        let size = vec2<f32>(v.window.zw);
        let c = vec2<f32>(local) + 0.5;
        let outside = c - min(max(c, vec2(g.radius)), size - g.radius);
        coverage = g.inner_coverage;
        if any(outside != vec2(0.0)) {
            coverage = clamp(g.radius + 0.5 - length(outside), 0.0, 1.0);
        }
        color = window_color(v, p);
    }
    return vec4(color * coverage, 1.0 - (1.0 - shadowed) * (1.0 - coverage));
}

@fragment
fn fs_body(v: Varyings) -> @location(0) vec4<f32> {
    return vec4(window_color(v, vec2<i32>(v.position.xy)), 1.0);
}

@fragment
fn fs_backdrop(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let c = frame_at(vec2<i32>(position.xy));
    if has(STRAIGHT) {
        return vec4(c.rgb * c.a, c.a);
    }
    return c;
}

// The composited pixel as the host stores it: (alpha, red, green, blue).
fn host_pixel(position: vec4<f32>) -> vec4<f32> {
    let c = textureLoad(composed, vec2<i32>(position.xy), 0);
    if has(STRAIGHT) && c.a > 0.0 && c.a < 1.0 {
        return vec4(c.a, c.rgb / c.a);
    }
    return c.argb;
}

@fragment
fn fs_pack(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return host_pixel(position);
}

@fragment
fn fs_pack_16(@builtin(position) position: vec4<f32>) -> @location(0) vec4<u32> {
    let v = clamp(host_pixel(position), vec4(0.0), vec4(1.0));
    return vec4<u32>(v * 32768.0 + 0.5);
}
