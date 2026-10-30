#!/usr/bin/env python3
"""After Effects test harness for the VFX Preview plugin.

Builds a fixture project per variant (bit depth x GPU/CPU) in the running After
Effects, renders every case with aerender, and checks the frames:

  reference   After Effects matches the plugin's pipeline run directly on the same pixels
  gpu-cpu     the GPU-native path matches the CPU path
  mfr         Multi-Frame Rendering on and off give identical frames
  half        Half resolution matches Full, downsampled
  path        the plugin log shows the expected render path
  checks      per-case expectations from cases.py

Usage: python3 vfx-core/wrappers/adobe/ae-tests/run.py [--cases blur,hdr] [--variants 32bpc-gpu]

Needs the plugin installed, After Effects 2026 (it is started if needed), ffmpeg, and
cargo. Results go to target/ae-tests/; the exit status is non-zero on any failure.
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from cases import CASES, PATTERN_SIZE, POPUPS, params
from checks import TOLERANCE, Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AE_APP = "Adobe After Effects 2026"
AERENDER = Path(f"/Applications/{AE_APP}/aerender")


def plugin_env():
    """VFX Preview's names and IDs, from vfx-core/effects/vfx-chain's [package.metadata.vfx]."""
    out = subprocess.check_output(["cargo", "vfx", "env", "-p", "vfx-chain"], cwd=ROOT, text=True)
    values = {}
    for line in out.splitlines():
        key, rest = line.split("=", 1)
        values[key] = rest.strip('"')
    return values


PLUGIN = plugin_env()
PLUGIN_LOG = Path.home() / f"Library/Logs/{PLUGIN['PLUGIN_FILE']}-AfterEffects.log"
FPS = 30


@dataclass
class Variant:
    name: str
    depth: int
    accel: str  # GpuAccelType in the project: METAL or SOFTWARE
    mfr_off: bool = False  # also render with Multi-Frame Rendering off

    @property
    def render_path(self):
        # The plugin only offers GPU-native rendering for 32 bpc float.
        if self.depth == 32 and self.accel == "METAL":
            return "GPU-native"
        return "CPU (frames through RAM)"


VARIANTS = [
    Variant("8bpc-cpu", 8, "SOFTWARE"),
    Variant("16bpc-cpu", 16, "SOFTWARE"),
    Variant("32bpc-cpu", 32, "SOFTWARE"),
    Variant("32bpc-gpu", 32, "METAL", mfr_off=True),
]


# --- Test images ---------------------------------------------------------------------


def pattern_a(width, height):
    """Gradients, a checker, an alpha ramp and a soft hole. Colour under transparent
    pixels is bright magenta, so any colour bleed from them shows up."""
    out = bytearray()
    for y in range(height):
        for x in range(width):
            r = round(255 * x / (width - 1))
            g = round(255 * y / (height - 1))
            b = 200 if (x // 8 + y // 8) % 2 else 50
            a = 255
            ramp_start = int(width * 0.6)
            if x >= ramp_start:
                a = round(255 * (width - 1 - x) / (width - 1 - ramp_start))
            d = ((x - 20) ** 2 + (y - 20) ** 2) ** 0.5
            if d < 10:
                a = min(a, round(255 * max(0.0, (d - 6) / 4)))
            if a == 0:
                r, g, b = 255, 0, 255
            out += bytes((r, g, b, a))
    return bytes(out)


def pattern_b(width, height):
    """Colour bands, half transparent along the top, for the blend layer."""
    bands = [(230, 60, 40), (40, 200, 90), (50, 80, 230), (240, 220, 60)]
    out = bytearray()
    for y in range(height):
        for x in range(width):
            r, g, b = bands[(x * len(bands)) // width]
            out += bytes((r, g, b, 128 if y < height // 4 else 255))
    return bytes(out)


def write_png(path, width, height, rgba):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(
        b"\0" + rgba[y * width * 4 : (y + 1) * width * 4] for y in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows, 9))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def to_output(pixels, depth):
    """Straight f32 RGBA as After Effects writes it to a premultiplied 8-bit TIFF.

    Integer depths clip each channel to 0..1 in the plugin's encode, before After
    Effects premultiplies. At 32 bpc the plugin keeps values above 1.0 and the clip
    only happens after premultiplying, so a bright, half-transparent pixel can still
    reach full white.
    """
    clip = lambda v: min(max(v, 0.0), 1.0)
    out = bytearray(len(pixels))
    for i in range(0, len(pixels), 4):
        a = clip(pixels[i + 3])
        a8 = round(a * 255)
        for c in range(3):
            v = pixels[i + c]
            if depth == 32:
                out[i + c] = round(clip(v * a) * 255)
            else:
                out[i + c] = round(round(clip(v) * 255) * a8 / 255)
        out[i + 3] = a8
    return bytes(out)


def read_tiff(path):
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)]
    ).decode()
    width, height = (int(v) for v in probe.strip().split(","))
    data = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    )
    return Image(width, height, data)


# --- Fixture project -------------------------------------------------------------------


def case_layout(case):
    """Comp size, and the source layer's size and centre in comp pixels."""
    source = case["source"]
    if source == "pattern":
        layer_size = PATTERN_SIZE
    else:
        layer_size = source[2]
    size = case.get("size", PATTERN_SIZE if source == "pattern" else layer_size)
    position = case.get("position", (size[0] / 2, size[1] / 2))
    return size, layer_size, position


def jsx_case(case):
    size, layer_size, position = case_layout(case)
    effects = []
    for effect in case["effects"]:
        values = {}
        for key, value in effect.items():
            if key in POPUPS:
                value += 1  # After Effects popups are 1-based
            elif isinstance(value, bool):
                value = 1 if value else 0
            values[key] = value
        effects.append(values)
    source = case["source"]
    return {
        "name": case["name"],
        "w": size[0],
        "h": size[1],
        "frames": case.get("frames", 1),
        "pattern": source == "pattern",
        "solid": None if source == "pattern" else {
            "color": list(source[1]), "w": layer_size[0], "h": layer_size[1]},
        "position": list(position) if "position" in case else None,
        "mask": list(case["mask"]) if case.get("mask") else None,
        "blend": case.get("blend"),
        "effects": effects,
        "keyframes": case.get("keyframes", {}),
        "resolutions": case.get("resolutions", ["Full"]),
    }


JSX = r"""
var SPEC = %(spec)s;

function setParam(fx, name, value) {
    fx.property(name).setValue(value);
}

function build(v) {
    app.newProject();
    var p = app.project;
    p.bitsPerChannel = v.depth;
    p.gpuAccelType = v.accel == "METAL" ? GpuAccelType.METAL : GpuAccelType.SOFTWARE;
    var footage = {};
    for (var key in SPEC.footage) {
        var item = p.importFile(new ImportOptions(new File(SPEC.footage[key])));
        item.mainSource.alphaMode = AlphaMode.STRAIGHT;
        footage[key] = item;
    }
    for (var i = 0; i < SPEC.cases.length; i++) {
        var c = SPEC.cases[i];
        var comp = p.items.addComp(c.name, c.w, c.h, 1, c.frames / %(fps)d, %(fps)d);
        comp.workAreaStart = 0;
        comp.workAreaDuration = c.frames / %(fps)d;
        if (c.blend) {
            comp.layers.add(footage[c.blend]).enabled = false;
        }
        var layer = c.pattern
            ? comp.layers.add(footage.pattern)
            : comp.layers.addSolid(c.solid.color, "source", c.solid.w, c.solid.h, 1);
        if (c.position) {
            layer.property("ADBE Transform Group").property("ADBE Position").setValue(c.position);
        }
        if (c.mask) {
            var m = layer.property("ADBE Mask Parade").addProperty("ADBE Mask Atom");
            var s = new Shape();
            s.vertices = [[c.mask[0], c.mask[1]], [c.mask[2], c.mask[1]],
                          [c.mask[2], c.mask[3]], [c.mask[0], c.mask[3]]];
            s.closed = true;
            m.property("ADBE Mask Shape").setValue(s);
        }
        for (var e = 0; e < c.effects.length; e++) {
            var fx = layer.property("ADBE Effect Parade").addProperty("%(match_name)s");
            for (var name in c.effects[e]) {
                setParam(fx, name, c.effects[e][name]);
            }
            if (c.blend) {
                setParam(fx, "Blend Layer", 2);
            }
            if (e == 0) {
                for (var k in c.keyframes) {
                    for (var t = 0; t < c.keyframes[k].length; t++) {
                        fx.property(k).setValueAtTime(c.keyframes[k][t][0], c.keyframes[k][t][1]);
                    }
                }
            }
        }
        for (var r = 0; r < c.resolutions.length; r++) {
            var res = c.resolutions[r];
            var rq = p.renderQueue.items.add(comp);
            if (res != "Full") {
                rq.setSettings({"Resolution": res});
            }
            var om = rq.outputModule(1);
            om.applyTemplate("TIFF Sequence with Alpha");
            om.file = new File(v.out + "/" + c.name + "_" + res + "_[#####].tif");
        }
    }
    p.save(new File(v.aep));
}

// Never discard someone's work: an unsaved project asks first, and Cancel stops here.
if (app.project.close(CloseOptions.PROMPT_TO_SAVE_CHANGES)) {
    for (var n = 0; n < SPEC.variants.length; n++) {
        build(SPEC.variants[n]);
    }
    app.project.close(CloseOptions.DO_NOT_SAVE_CHANGES);
    app.newProject();
}
"""


def build_fixtures(out, cases, variants, footage):
    spec = {
        "footage": {k: str(v) for k, v in footage.items()},
        "cases": [jsx_case(c) for c in cases],
        "variants": [
            {"depth": v.depth, "accel": v.accel,
             "aep": str(out / f"{v.name}.aep"), "out": str(out / v.name)}
            for v in variants
        ],
    }
    for v in variants:
        (out / v.name).mkdir(parents=True, exist_ok=True)
    script = out / "fixtures.jsx"
    script.write_text(JSX % {"spec": json.dumps(spec), "fps": FPS, "match_name": PLUGIN["AE_MATCH_NAME"]})
    subprocess.run(
        ["osascript", "-e",
         f'with timeout of 900 seconds\n'
         f'tell application "{AE_APP}" to DoScriptFile "{script}"\n'
         f'end timeout'],
        check=True, capture_output=True,
    )
    missing = [v.name for v in variants if not (out / f"{v.name}.aep").exists()]
    if missing:
        sys.exit(f"After Effects didn't save fixtures for {', '.join(missing)} "
                 "(was the save prompt cancelled, or did the script fail?)")


# --- Rendering -------------------------------------------------------------------------


def aerender(variant, out, mfr):
    """Renders the variant's whole render queue. Returns the plugin log lines."""
    frames = out / variant.name
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    PLUGIN_LOG.unlink(missing_ok=True)
    log = out / f"{variant.name}-mfr-{mfr.lower()}.log"
    result = subprocess.run(
        [str(AERENDER), "-project", str(out / f"{variant.name}.aep"),
         "-mfr", mfr, "100", "-v", "ERRORS_AND_PROGRESS", "-log", str(log)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"aerender failed for {variant.name} (MFR {mfr}); see {log}")
    frames.rename(out / f"{variant.name}-mfr-{mfr.lower()}")
    return PLUGIN_LOG.read_text().splitlines() if PLUGIN_LOG.exists() else []


def load_frames(directory, case):
    frames = {}
    for res in case.get("resolutions", ["Full"]):
        frames[res] = [
            read_tiff(directory / f"{case['name']}_{res}_{f:05d}.tif")
            for f in range(case.get("frames", 1))
        ]
    return frames


# --- Reference -------------------------------------------------------------------------


def value_at(keys, t):
    """Linear interpolation, as After Effects does between linear keyframes."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def reference_block(case, frame, files, output):
    effects = case["effects"]
    if any(e != effects[0] for e in effects):
        raise ValueError(f"{case['name']}: reference needs identical stacked instances")
    p = params(effects[0])
    t = frame / FPS
    for key, keys in case.get("keyframes", {}).items():
        p[key] = value_at(keys, t)
    (width, height), _, _ = case_layout(case)
    lines = [f"size {width} {height}", f"input {files['pattern']}", f"output {output}",
             f"time {t}", f"repeat {len(effects)}"]
    if case.get("blend"):
        lines.append(f"layer {files[case['blend']]}")
        lines.append(f"blend {p['Blend Mode']} {p['Blend Amount']}")
    if p["Linear Light"]:
        lines.append("linear 1")
    if p["Color Adjust"]:
        lines.append(f"color {p['Brightness']} {p['Contrast']} {p['Saturation']}")
    if p["Vignette"]:
        cx, cy = p["Vignette Center"] or (width / 2, height / 2)
        r, g, b, a = p["Vignette Color"]
        lines.append(f"vignette {cx} {cy} {p['Vignette Radius']} {p['Vignette Soft']} "
                     f"{r} {g} {b} {a} {p['Vignette Amount']}")
    if p["Blur"]:
        lines.append(f"blur {p['Blur Radius']} 1 {p['Blur Direction']} "
                     f"{1 if p['Blur Edges'] else 0}")
    return "\n".join(lines)


def render_references(out, cases, files):
    ref = out / "reference"
    ref.mkdir(exist_ok=True)
    blocks, outputs = [], {}
    for case in cases:
        if not case.get("reference"):
            continue
        for f in range(case.get("frames", 1)):
            path = ref / f"{case['name']}_{f:05d}.rgbaf"
            blocks.append(reference_block(case, f, files, path))
            outputs[(case["name"], f)] = path
    if not blocks:
        return {}
    job = ref / "job.txt"
    job.write_text("\n\n".join(blocks) + "\n")
    subprocess.run([str(ROOT / "target/release/examples/ae_reference"), str(job)], check=True)
    # (case, frame) -> {depth: Image}, quantised the way each depth writes its output.
    images = {}
    for (name, f), path in outputs.items():
        (width, height), _, _ = case_layout(next(c for c in cases if c["name"] == name))
        data = path.read_bytes()
        pixels = struct.unpack(f"<{len(data) // 4}f", data)
        images[(name, f)] = {
            depth: Image(width, height, to_output(pixels, depth)) for depth in (8, 16, 32)
        }
    return images


# --- Comparisons -----------------------------------------------------------------------


def difference(a, b):
    """Largest and mean absolute channel difference between two same-size images."""
    if (a.width, a.height) != (b.width, b.height):
        return None
    diffs = [abs(x - y) for x, y in zip(a.data, b.data)]
    return max(diffs), sum(diffs) / len(diffs)


def downsample(img):
    """2x2 box filter on premultiplied pixels, rounding odd sizes up like After Effects."""
    w, h = (img.width + 1) // 2, (img.height + 1) // 2
    out = bytearray()
    for y in range(h):
        for x in range(w):
            for c in range(4):
                vals = [img.data[(sy * img.width + sx) * 4 + c]
                        for sy in (2 * y, 2 * y + 1) if sy < img.height
                        for sx in (2 * x, 2 * x + 1) if sx < img.width]
                out.append(round(sum(vals) / len(vals)))
    return Image(w, h, bytes(out))


def save_failure(out, variant, case, frame, ae, ref):
    failures = out / "failures"
    failures.mkdir(exist_ok=True)
    stem = f"{variant}_{case}_{frame:05d}"
    write_png(failures / f"{stem}_ae.png", ae.width, ae.height, ae.data)
    if ref is not None and (ref.width, ref.height) == (ae.width, ae.height):
        write_png(failures / f"{stem}_ref.png", ref.width, ref.height, ref.data)
        diff = bytes(min(255, abs(x - y) * 16) if i % 4 != 3 else 255
                     for i, (x, y) in enumerate(zip(ae.data, ref.data)))
        write_png(failures / f"{stem}_diff.png", ae.width, ae.height, diff)


# Downsampling resamples the layer before the effect, so Half can't match Full pixel
# for pixel. A blur radius that isn't scaled doubles the blur at Half, which moves the
# mean by far more than this.
HALF_MEAN_LIMIT = 4.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", help="comma-separated case names (default: all)")
    parser.add_argument("--variants", help="comma-separated variant names (default: all)")
    parser.add_argument("--out", type=Path, default=ROOT / "target/ae-tests")
    args = parser.parse_args()

    cases = CASES
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in CASES if c["name"] in wanted]
        unknown = wanted - {c["name"] for c in cases}
        if unknown:
            sys.exit(f"unknown cases: {', '.join(sorted(unknown))}")
    variants = VARIANTS
    if args.variants:
        wanted = set(args.variants.split(","))
        variants = [v for v in VARIANTS if v.name in wanted]

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    started = time.time()

    print("building the reference renderer")
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "vfx-chain",
                    "--example", "ae_reference"], cwd=ROOT, check=True)

    files = {}
    for key, make in (("pattern", pattern_a), ("pattern_b", pattern_b)):
        rgba = make(*PATTERN_SIZE)
        write_png(out / f"{key}.png", *PATTERN_SIZE, rgba)
        (out / f"{key}.rgba").write_bytes(rgba)
        files[key] = out / f"{key}.rgba"

    print(f"building fixtures in After Effects ({len(cases)} cases, {len(variants)} variants)")
    build_fixtures(out, cases, variants, {k: out / f"{k}.png" for k in files})

    references = render_references(out, cases, files)

    results = {}  # (case, check) -> {variant: "ok" | message}
    def record(case, check, variant, failures):
        results.setdefault((case, check), {})[variant] = "; ".join(failures) or "ok"

    rendered = {}
    for v in variants:
        runs = ["ON", "OFF"] if v.mfr_off else ["ON"]
        for mfr in runs:
            print(f"rendering {v.name} with MFR {mfr}")
            log = aerender(v, out, mfr)
            ok = any("render path" in line and v.render_path in line for line in log)
            record("(all)", f"path MFR {mfr}", v.name,
                   [] if ok else [f"log has no '{v.render_path}' render: {log}"])
            rendered[(v.name, mfr)] = out / f"{v.name}-mfr-{mfr.lower()}"

    for case in cases:
        name = case["name"]
        for v in variants:
            frames = load_frames(rendered[(v.name, "ON")], case)
            if case.get("reference"):
                failures = []
                for f, ae in enumerate(frames["Full"]):
                    ref = references[(name, f)][v.depth]
                    d = difference(ae, ref)
                    if d is None:
                        failures.append(f"frame {f} is {ae.width}x{ae.height}, "
                                        f"expected {ref.width}x{ref.height}")
                    elif d[0] > TOLERANCE:
                        failures.append(f"frame {f} max diff {d[0]} (mean {d[1]:.2f})")
                        save_failure(out, v.name, name, f, ae, ref)
                record(name, "reference", v.name, failures)
            if "Half" in frames:
                half, full = frames["Half"][0], downsample(frames["Full"][0])
                d = difference(half, full)
                failures = []
                if d is None:
                    failures.append(f"Half is {half.width}x{half.height}, "
                                    f"expected {full.width}x{full.height}")
                elif d[1] > HALF_MEAN_LIMIT:
                    failures.append(f"mean diff {d[1]:.2f} vs Full downsampled (max {d[0]})")
                    save_failure(out, v.name, name + "_half", 0, half, full)
                record(name, "half", v.name, failures)
            for check in case.get("checks", []):
                try:
                    failures = check(case, v, frames)
                except Exception as e:  # a broken check is a failure, not a crash
                    failures = [f"check raised {e!r}"]
                if failures:
                    save_failure(out, v.name, name, 0, frames["Full"][0], None)
                record(name, check.__qualname__.split(".")[0], v.name, failures)
            if v.mfr_off:
                off = load_frames(rendered[(v.name, "OFF")], case)
                failures = []
                for res in frames:
                    for f, (a, b) in enumerate(zip(frames[res], off[res])):
                        d = difference(a, b)
                        if d is None or d[0] > 0:
                            failures.append(f"{res} frame {f} differs (max {d and d[0]})")
                record(name, "mfr on vs off", v.name, failures)
        gpu = [v for v in variants if v.accel == "METAL" and v.depth == 32]
        cpu = [v for v in variants if v.accel == "SOFTWARE" and v.depth == 32]
        if gpu and cpu:
            a = load_frames(rendered[(gpu[0].name, "ON")], case)
            b = load_frames(rendered[(cpu[0].name, "ON")], case)
            failures = []
            for res in a:
                for f, (x, y) in enumerate(zip(a[res], b[res])):
                    d = difference(x, y)
                    if d is None or d[0] > 1:
                        failures.append(f"{res} frame {f} differs (max {d and d[0]})")
            record(name, "gpu vs cpu", "32bpc", failures)

    columns = [v.name for v in variants] + ["32bpc"]
    width = max(len(f"{c} {k}") for c, k in results) + 2
    lines = [f"{'case / check':<{width}}" + "".join(f"{c:<12}" for c in columns)]
    failed = []
    for (case, check), cells in results.items():
        row = f"{case + ' ' + check:<{width}}"
        for c in columns:
            cell = cells.get(c)
            row += f"{'' if cell is None else ('ok' if cell == 'ok' else 'FAIL'):<12}"
            if cell not in (None, "ok"):
                failed.append(f"{case} / {check} / {c}: {cell}")
        lines.append(row)
    total = sum(len(cells) for cells in results.values())
    lines += ["", f"{total - len(failed)} of {total} passed in {time.time() - started:.0f}s"]
    lines += [""] + failed
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    (out / "report.json").write_text(json.dumps(
        [{"case": c, "check": k, "results": r} for (c, k), r in results.items()], indent=1))
    print(report)
    print(f"\nreport: {out / 'report.txt'}")
    if failed:
        print(f"failure images: {out / 'failures'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
