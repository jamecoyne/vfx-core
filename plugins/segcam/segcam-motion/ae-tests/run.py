#!/usr/bin/env python3
"""After Effects check for the Rust segcam Motion effect.

Builds one project per bit depth (8, 16, 32 bpc) in the running After Effects, with a comp
per output mode for this effect and, when it is installed, for segcam's own C++ effect, all
on the same footage. Renders them with aerender and checks:

  c++       the Rust effect matches the C++ effect, within a level: clang fuses
            multiply-adds that Rust keeps separate, which moves the odd soft edge by one,
            and at 16 and 32 bpc After Effects dithers as it writes the 8-bit frames, a
            little differently for each render item, so even an untouched frame moves
  mfr       Multi-Frame Rendering on and off give identical frames
  motion    the footage's movers were found: the matte isn't empty, the overlay isn't the
            input, so a pass-through can't pass

Usage: python3 plugins/segcam/segcam-motion/ae-tests/run.py [--clip footage.mp4] [--frames 40]
(`python` or `py` on Windows)

Default footage is generated: squares crossing a gradient, as a PNG sequence, so every
input pixel is known. Needs the Rust effect installed (the C++ one is compared when it is),
After Effects 2026 (it is started if needed), ffmpeg. Runs on macOS and Windows. Exit status
is non-zero on failure.

Frames are 8-bit PNGs at 8 and 16 bpc and TIFFs at 32, and only frames that fail a check are
kept. They go to $AE_TEST_OUT or --out. The harness refuses a folder on the system disk: a run
at 1080p writes gigabytes.
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

AE_APP = "Adobe After Effects 2026"
WINDOWS = sys.platform == "win32"
if WINDOWS:
    AE_SUPPORT = Path(rf"C:\Program Files\Adobe\{AE_APP}\Support Files")
    AERENDER = AE_SUPPORT / "aerender.exe"
    MEDIACORE = Path(r"C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore")
    # A Windows plugin is a folder holding the .aex and the DLLs it loads beside it.
    BUNDLES = ("segcam/segcam.aex", "SegcamMotion/SegcamMotion.aex")
else:
    AERENDER = Path(f"/Applications/{AE_APP}/aerender")
    MEDIACORE = Path("/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore")
    BUNDLES = ("segcam.plugin", "SegcamMotion.plugin")
FPS = 30
DEPTHS = [8, 16, 32]
MODES = ["overlay", "matte", "boxmatte", "swarm"]  # Output popup values 1..4
EFFECTS = {
    "rust": ("LOFISCIFI segcam", BUNDLES[0]),
    "cpp": ("LOFISCIFI segcam Motion", BUNDLES[1]),
}
# 8-bit levels the Rust and C++ effects may differ by.
TOLERANCE = 1
# After Effects ships this template hidden: 8-bit RGBA PNG, compressed. A script can't set
# an output module's format or depth. PNG holds straight colour, which clips a 32 bpc value
# above 1.0 before alpha, so 32 bpc keeps the TIFF template.
PNG_TEMPLATE = "_HIDDEN X-Factor 8 Premul"
TIFF_TEMPLATE = "TIFF Sequence with Alpha"
TRANSPARENT = re.compile(rb"\x00+")
PARTIAL = re.compile(rb"[\x01-\xfe]")


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


def premultiplied(data):
    """Straight 8-bit RGBA premultiplied, rounding to nearest. After Effects' PNGs hold
    straight colour whatever the template says, and this gives back its TIFF's bytes."""
    alpha = data[3::4]
    out = bytearray(data)
    for run in TRANSPARENT.finditer(alpha):
        out[run.start() * 4 : run.end() * 4] = bytes((run.end() - run.start()) * 4)
    for pixel in PARTIAL.finditer(alpha):
        a = alpha[pixel.start()]
        i = pixel.start() * 4
        out[i : i + 3] = bytes((v * a + 127) // 255 for v in data[i : i + 3])
    return bytes(out)


def read_frame(path):
    """A frame as premultiplied 8-bit RGBA, as the TIFF template writes it."""
    width, height = probe(path)
    data = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    )
    if path.suffix == ".png":
        data = premultiplied(data)
    return width, height, data


# --- Projects ------------------------------------------------------------------------


JSX = r"""
var SPEC = %(spec)s;

// The PNG template if this After Effects has it, else the TIFF one; saved for the harness.
function chooseTemplate() {
    var probe = app.project.items.addComp("template probe", 4, 4, 1, 1, %(fps)d);
    var templates = app.project.renderQueue.items.add(probe).outputModule(1).templates;
    var chosen = SPEC.tiff;
    for (var t = 0; t < templates.length; t++) {
        if (templates[t] == SPEC.png) {
            chosen = SPEC.png;
        }
    }
    var record = new File(SPEC.out + "/template.txt");
    record.open("w");
    record.write(chosen);
    record.close();
    return chosen;
}

function build(depth, template, extension) {
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
            fx.property("Detect").setValue(1);
            fx.property("Eyes").setValue(0);
            fx.property("Mouth").setValue(0);
            var rq = p.renderQueue.items.add(comp);
            rq.timeSpanStart = 0;
            rq.timeSpanDuration = duration;
            var om = rq.outputModule(1);
            om.applyTemplate(template);
            om.file = new File(SPEC.out + "/" + depth + "/" + name + "_[#####]." + extension);
        }
    }
    p.save(new File(SPEC.out + "/" + depth + ".aep"));
}

// Never discard someone's work: an unsaved project asks first, and Cancel stops here.
if (app.project.close(CloseOptions.PROMPT_TO_SAVE_CHANGES)) {
    app.newProject();
    var png = chooseTemplate() == SPEC.png;
    app.project.close(CloseOptions.DO_NOT_SAVE_CHANGES);
    for (var d = 0; d < SPEC.depths.length; d++) {
        var depth = SPEC.depths[d];
        var usePng = png && depth < 32;
        build(depth, usePng ? SPEC.png : SPEC.tiff, usePng ? "png" : "tif");
    }
    app.project.close(CloseOptions.DO_NOT_SAVE_CHANGES);
    app.newProject();
}
"""


def build_projects(out, footage, sequence, size, frames, effects):
    spec = {
        "footage": footage.as_posix(),
        "sequence": sequence,
        "w": size[0],
        "h": size[1],
        "frames": frames,
        "modes": MODES,
        "depths": DEPTHS,
        "effects": [{"key": k, "match": EFFECTS[k][0]} for k in effects],
        "out": out.as_posix(),
        "png": PNG_TEMPLATE,
        "tiff": TIFF_TEMPLATE,
    }
    for depth in DEPTHS:
        (out / str(depth)).mkdir(parents=True, exist_ok=True)
    script = out / "projects.jsx"
    script.write_text(JSX % {"spec": json.dumps(spec), "fps": FPS})
    if WINDOWS:
        run_script_windows(script, [out / f"{d}.aep" for d in DEPTHS])
    else:
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
    png = (out / "template.txt").read_text() == PNG_TEMPLATE
    if not png:
        print(f"After Effects has no '{PNG_TEMPLATE}' template: writing uncompressed TIFFs")
    return {depth: "png" if png and depth < 32 else "tif" for depth in DEPTHS}


def run_script_windows(script, projects, timeout=900):
    """Runs a script in After Effects on Windows, starting it if needed. `AfterFX.exe -r`
    hands the script over and returns before it has run, so this waits for the projects the
    script saves: done when every one exists and none has grown for two seconds."""
    subprocess.Popen([str(AE_SUPPORT / "AfterFX.exe"), "-r", str(script)])
    deadline = time.monotonic() + timeout
    sizes = None
    while True:
        current = [p.stat().st_size if p.exists() else None for p in projects]
        if None not in current and current == sizes:
            return
        if time.monotonic() > deadline:
            # The caller names the missing projects.
            print(f"After Effects didn't finish the project script within {timeout} s")
            return
        sizes = current
        time.sleep(2)


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
    if a[2] == b[2]:
        return 0, 0
    diffs = [abs(x - y) for x, y in zip(a[2], b[2]) if x != y]
    return (max(diffs) if diffs else 0), len(diffs)


def default_out():
    if "AE_TEST_OUT" in os.environ:
        return Path(os.environ["AE_TEST_OUT"])
    sys.exit("set AE_TEST_OUT or --out to a folder that is not on the system disk")


def on_system_disk(path):
    existing = next(p for p in [path, *path.parents] if p.exists())
    system = Path(os.environ.get("SystemDrive", "C:") + "\\") if WINDOWS else Path("/")
    return existing.stat().st_dev == system.stat().st_dev


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clip", type=Path, help="footage to use instead of the generated squares")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--out", type=Path, help="folder for frames (default: $AE_TEST_OUT)")
    args = parser.parse_args()

    effects = [k for k, (_, bundle) in EFFECTS.items() if (MEDIACORE / bundle).exists()]
    if "rust" not in effects:
        sys.exit(f"{EFFECTS['rust'][1]} isn't installed in {MEDIACORE}")
    if "cpp" not in effects:
        print("segcam's C++ effect isn't installed: skipping the c++ check")

    out = (args.out or default_out()).resolve()
    if on_system_disk(out):
        sys.exit(f"{out} is on the system disk; set AE_TEST_OUT or --out to another drive")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.clip:
        footage, sequence = args.clip.resolve(), False
        size = probe(footage)
    else:
        size = (640, 360)
        footage, sequence = generate_footage(out / "footage", *size, args.frames), True
        # The last frame, which the motion checks look at.
        last_input = read_frame(out / "footage" / f"frame_{args.frames - 1:05d}.png")

    print(f"building {len(DEPTHS)} projects in After Effects "
          f"({len(effects)} effects x {len(MODES)} modes, {args.frames} frames)")
    extensions = build_projects(out, footage, sequence, size, args.frames, effects)

    failures = 0

    def report(check, what, problems):
        nonlocal failures
        status = "ok" if not problems else "FAIL " + "; ".join(problems[:4])
        failures += bool(problems)
        print(f"  {check:7} {what:22} {status}")

    # Each depth is checked as soon as it has rendered, and its frames that pass every check
    # are deleted, so a run holds at most one depth's frames.
    for depth in DEPTHS:
        for mfr in ("ON", "OFF"):
            print(f"rendering {depth} bpc with MFR {mfr}")
            aerender(out, depth, mfr)
        on, off = out / f"{depth}-mfr-on", out / f"{depth}-mfr-off"
        keep = set()
        print(f"checks at {depth} bpc")
        for mode in MODES:
            what = f"{depth} bpc {mode}"
            mfr, cpp, motion = [], [], []
            worst = 0
            for f in range(args.frames):
                frame = f"{mode}_{f:05d}.{extensions[depth]}"
                rust = read_frame(on / f"rust_{frame}")
                d = compare(rust, read_frame(off / f"rust_{frame}"))
                if d != (0, 0):
                    mfr.append(f"frame {f}: {d}")
                    keep |= {on / f"rust_{frame}", off / f"rust_{frame}"}
                if "cpp" in effects:
                    d = compare(rust, read_frame(on / f"cpp_{frame}"))
                    if d is None or d[0] > TOLERANCE:
                        cpp.append(f"frame {f}: max {d[0] if d else 'size'}")
                        keep |= {on / f"rust_{frame}", on / f"cpp_{frame}"}
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
                    if motion:
                        keep.add(on / f"rust_{frame}")
            report("mfr", what, mfr)
            if "cpp" in effects:
                report("c++", what + (f" (≤{worst})" if worst else ""), cpp)
            report("motion", what, motion)
        for path in [*on.iterdir(), *off.iterdir()]:
            if path not in keep:
                path.unlink()

    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) failed'}; "
          f"failing frames kept in {out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
