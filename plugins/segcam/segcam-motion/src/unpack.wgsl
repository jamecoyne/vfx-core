// The swarm's 8-bit output into a host's GPU frame (`Drawer::draw_swarm_into`): each word is
// one pixel, alpha, red, green and blue a byte each from the lowest, straight colour, top row
// first. The frame gets the same colour, in its own format.

@group(0) @binding(0) var<storage, read> words: array<u32>;
// x: the canvas width in pixels.
@group(0) @binding(1) var<uniform> canvas: vec4<u32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // One triangle over the whole frame.
    let corner = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let w = words[u32(at.y) * canvas.x + u32(at.x)];
    let a = f32(w & 0xffu);
    let r = f32((w >> 8u) & 0xffu);
    let g = f32((w >> 16u) & 0xffu);
    let b = f32(w >> 24u);
    return vec4<f32>(r, g, b, a) / 255.0;
}
