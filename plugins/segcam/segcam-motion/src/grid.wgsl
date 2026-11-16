// Computes the luma grid of an RGBA frame, one fragment per cell. Each cell is within 1
// of the one `render::downsample` makes. A NaN or infinite channel makes the cell's value
// unspecified.

struct Params {
    frame_width: u32,
    frame_height: u32,
    grid_width: u32,
    grid_height: u32,
}

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// These are the four pixels `downsample` reads for a cell.
fn cell_samples(position: vec4<f32>) -> array<vec2<i32>, 4> {
    let gx = u32(position.x);
    let gy = u32(position.y);
    let y0 = gy * params.frame_height / params.grid_height;
    let y1 = max(y0 + 1u, (gy + 1u) * params.frame_height / params.grid_height);
    let yb = min(params.frame_height - 1u, (y0 + y1) / 2u);
    let x0 = gx * params.frame_width / params.grid_width;
    let x1 = max(x0 + 1u, (gx + 1u) * params.frame_width / params.grid_width);
    let xb = min(params.frame_width - 1u, (x0 + x1) / 2u);
    return array<vec2<i32>, 4>(
        vec2<i32>(i32(x0), i32(y0)),
        vec2<i32>(i32(xb), i32(y0)),
        vec2<i32>(i32(x0), i32(yb)),
        vec2<i32>(i32(xb), i32(yb)),
    );
}

@fragment
fn grid(@builtin(position) position: vec4<f32>) -> @location(0) u32 {
    let at = cell_samples(position);
    var sum = 0.0;
    for (var i = 0; i < 4; i++) {
        sum += dot(textureLoad(frame, at[i], 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    }
    // `downsample` truncates the mean and clamps it to a byte.
    return u32(clamp(sum * (255.0 / 4.0), 0.0, 255.0));
}
