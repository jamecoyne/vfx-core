#!/usr/bin/env python3
"""After Effects check for the Rust segcam Motion effect.

Builds one project per bit depth (8, 16, 32 bpc) in the running After Effects, with a comp
per output mode for this effect and, when it is installed, for segcam's own C++ effect, all
on the same footage. Renders them with aerender and checks:

  c++       the Rust effect matches the C++ effect, within a level: clang fuses
            multiply-adds that Rust keeps separate, which moves the odd soft edge by one,
            and at 16 and 32 bpc After Effects dithers as it writes the 8-bit TIFFs, a
            little differently for each render item, so even an untouched frame moves
  mfr       Multi-Frame Rendering on and off give identical frames
  motion    the footage's movers were found: the matte isn't empty, the overlay isn't the
            input, so a pass-through can't pass

Usage: python3 plugins/segcam/segcam-motion/ae-tests/run.py [--clip footage.mp4] [--frames 40]

Default footage is generated: squares crossing a gradient, as a PNG sequence, so every
input pixel is known. Needs both effects installed, After Effects 2026 (it is started if
needed), ffmpeg. Results go to target/segcam-ae-tests/; exit status is non-zero on failure.
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AE_APP = "Adobe After Effects 2026"
AERENDER = Path(f"/Applications/{AE_APP}/aerender")
MEDIACORE = Path("/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore")
FPS = 30
DEPTHS = [8, 16, 32]
MODES = ["overlay", "matte", "boxmatte", "swarm"]  # Output popup values 1..4
EFFECTS = {
    "rust": ("LOFISCIFI segcam Motion Rust", "SegcamMotionRust.plugin"),
    "cpp": ("LOFISCIFI segcam Motion", "SegcamMotion.plugin"),
}
# 8-bit levels the Rust and C++ effects may differ by.
TOLERANCE = 1


# --- Footage -------------------------------------------------------------------------


def write_png(path, width, height, rgb):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    stride = width * 3
    rows = b"".join(b"\0" + rgb[y * stride : (y + 1) * stride] for y in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows, 6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def generate_footage(directory, width, height, frames):
    """Squares of several sizes crossing a still gradient, at different speeds; a fixed
    square that never moves. Returns the first frame's path."""
    directory.mkdir(parents=True)
    background = bytearray()
    for y in range(height):
        for x in range(width):
            background += bytes((40 + 60 * x // width, 50 + 40 * y // height, 70))
    movers = [  # x0, y, size, dx per frame, colour
        (10, 60, 60, 9, (240, 240, 240)),
        (width - 90, 200, 40, -6, (230, 200, 60)),
        (200, 20, 24, 4, (60, 220, 120)),
    ]
    still = (width // 2, height - 70, 50, (200, 60, 60))
    for f in range(frames):
        img = bytearray(background)

        def square(x0, y0, size, colour):
            for y in range(max(0, y0), min(height, y0 + size)):
                for x in range(max(0, x0), min(width, x0 + size)):
                    img[(y * width + x) * 3 : (y * width + x) * 3 + 3] = bytes(colour)

        square(*still[:3], still[3])
        for x0, y, size, dx, colour in movers:
            square(x0 + dx * f, y, size, colour)
        write_png(directory / f"frame_{f:05d}.png", width, height, bytes(img))
    return directory / "frame_00000.png"


def probe(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)]
    ).decode()
    return tuple(int(v) for v in out.strip().splitlines()[0].split(","))


def read_tiff(path):
    width, height = probe(path)
    data = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    )
    return width, height, data


# --- Projects ------------------------------------------------------------------------


JSX = r"""
var SPEC = %(spec)s;

function build(depth) {
    app.newProject();
    var p = app.project;
    p.bitsPerChannel = depth;
    p.gpuAccelType = GpuAccelType.SOFTWARE;
    var options = new ImportOptions(new File(SPEC.footage));
    options.sequence = SPEC.sequence;
    var footage = p.importFile(options);
    if (SPEC.sequence) {
        footage.mainSource.conformFrameRate = %(fps)d;
    }
    var duration = SPEC.frames / %(fps)d;
    for (var e = 0; e < SPEC.effects.length; e++) {
        for (var m = 0; m < SPEC.modes.length; m++) {
            var name = SPEC.effects[e].key + "_" + SPEC.modes[m];
            var comp = p.items.addComp(name, SPEC.w, SPEC.h, 1, duration, %(fps)d);
            var layer = comp.layers.add(footage);
            var fx = layer.property("ADBE Effect Parade").addProperty(SPEC.effects[e].match);
            fx.property("Output").setValue(m + 1);
            var rq = p.renderQueue.items.add(comp);
            rq.timeSpanStart = 0;
            rq.timeSpanDuration = duration;
            var om = rq.outputModule(1);
            om.applyTemplate("TIFF Sequence with Alpha");
            om.file = new File(SPEC.out + "/" + depth + "/" + name + "_[#####].tif");
        }
    }
    p.save(new File(SPEC.out + "/" + depth + ".aep"));
}

// Never discard someone's work: an unsaved project asks first, and Cancel stops here.
if (app.project.close(CloseOptions.PROMPT_TO_SAVE_CHANGES)) {
    for (var d = 0; d < SPEC.depths.length; d++) {
        build(SPEC.depths[d]);
    }
    app.project.close(CloseOptions.DO_NOT_SAVE_CHANGES);
    app.newProject();
}
"""


def build_projects(out, footage, sequence, size, frames, effects):
    spec = {
        "footage": str(footage),
        "sequence": sequence,
        "w": size[0],
        "h": size[1],
        "frames": frames,
        "modes": MODES,
        "depths": DEPTHS,
        "effects": [{"key": k, "match": EFFECTS[k][0]} for k in effects],
        "out": str(out),
    }
    for depth in DEPTHS:
        (out / str(depth)).mkdir(parents=True, exist_ok=True)
    script = out / "projects.jsx"
    script.write_text(JSX % {"spec": json.dumps(spec), "fps": FPS})
    subprocess.run(
        ["osascript", "-e",
         f'with timeout of 900 seconds\n'
         f'tell application "{AE_APP}" to DoScriptFile "{script}"\n'
         f'end timeout'],
        check=True, capture_output=True,
    )
    missing = [d for d in DEPTHS if not (out / f"{d}.aep").exists()]
    if missing:
        sys.exit(f"After Effects didn't save projects for {missing} bpc "
                 "(was the save prompt cancelled, or did the script fail?)")


def aerender(out, depth, mfr):
    frames = out / str(depth)
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    log = out / f"{depth}-mfr-{mfr.lower()}.log"
    result = subprocess.run(
        [str(AERENDER), "-project", str(out / f"{depth}.aep"),
         "-mfr", mfr, "100", "-v", "ERRORS_AND_PROGRESS", "-log", str(log)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"aerender failed for {depth} bpc (MFR {mfr}); see {log}")
    done = out / f"{depth}-mfr-{mfr.lower()}"
    frames.rename(done)
    return done


# --- Checks --------------------------------------------------------------------------


def compare(a, b):
    """(max channel difference, count of differing channels), or None if sizes differ."""
    if a[:2] != b[:2]:
        return None
    diffs = [abs(x - y) for x, y in zip(a[2], b[2]) if x != y]
    return (max(diffs) if diffs else 0), len(diffs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", type=Path, help="footage to use instead of the generated squares")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--out", type=Path, default=ROOT / "target/segcam-ae-tests")
    args = parser.parse_args()

    effects = [k for k, (_, bundle) in EFFECTS.items() if (MEDIACORE / bundle).exists()]
    if "rust" not in effects:
        sys.exit(f"{EFFECTS['rust'][1]} isn't installed in {MEDIACORE}")
    if "cpp" not in effects:
        print("segcam's C++ effect isn't installed: skipping the c++ check")

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.clip:
        footage, sequence = args.clip.resolve(), False
        size = probe(footage)
    else:
        size = (640, 360)
        footage, sequence = generate_footage(out / "footage", *size, args.frames), True
        # The last frame, which the motion checks look at (PNG reads the same way as TIFF).
        last_input = read_tiff(out / "footage" / f"frame_{args.frames - 1:05d}.png")

    print(f"building {len(DEPTHS)} projects in After Effects "
          f"({len(effects)} effects x {len(MODES)} modes, {args.frames} frames)")
    build_projects(out, footage, sequence, size, args.frames, effects)

    rendered = {}
    for depth in DEPTHS:
        for mfr in ("ON", "OFF"):
            print(f"rendering {depth} bpc with MFR {mfr}")
            rendered[(depth, mfr)] = aerender(out, depth, mfr)

    failures = 0

    def report(check, what, problems):
        nonlocal failures
        status = "ok" if not problems else "FAIL " + "; ".join(problems[:4])
        failures += bool(problems)
        print(f"  {check:7} {what:22} {status}")

    print("\nchecks")
    for depth in DEPTHS:
        on, off = rendered[(depth, "ON")], rendered[(depth, "OFF")]
        for mode in MODES:
            what = f"{depth} bpc {mode}"
            mfr, cpp, motion = [], [], []
            worst = 0
            for f in range(args.frames):
                frame = f"{mode}_{f:05d}.tif"
                rust = read_tiff(on / f"rust_{frame}")
                d = compare(rust, read_tiff(off / f"rust_{frame}"))
                if d != (0, 0):
                    mfr.append(f"frame {f}: {d}")
                if "cpp" in effects:
                    d = compare(rust, read_tiff(on / f"cpp_{frame}"))
                    if d is None or d[0] > TOLERANCE:
                        cpp.append(f"frame {f}: max {d[0] if d else 'size'}")
                    elif d:
                        worst = max(worst, d[0])
                if f == args.frames - 1:
                    alpha = rust[2][3::4]
                    if mode in ("matte", "boxmatte", "swarm") and not any(alpha):
                        motion.append("output is empty")
                    if mode == "overlay" and not args.clip and rust[2] == last_input[2]:
                        motion.append("output is the input")
                    if mode in ("matte", "boxmatte") and all(a == 255 for a in alpha):
                        motion.append("matte is solid")
            report("mfr", what, mfr)
            if "cpp" in effects:
                report("c++", what + (f" (≤{worst})" if worst else ""), cpp)
            report("motion", what, motion)

    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) failed'}; "
          f"frames in {out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
