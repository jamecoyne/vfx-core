#!/usr/bin/env python3
"""Compare the FFGL texture output against the standalone CLI, byte for byte."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def sequence(width, height, count):
    frames = bytearray()
    for frame in range(count):
        image = bytearray(width * height * 4)
        left = 16 + (frame * 7) % (width - 80)
        top = 20 + (frame * 3) % (height - 70)
        for y in range(height):
            for x in range(width):
                moving = left <= x < left + 40 and top <= y < top + 35
                alpha = 0 if x < 5 else (128 if y > height * 2 // 3 else 255)
                color = (232, 201, 169) if moving else (19 + x % 11, 29 + y % 9, 43)
                i = (y * width + x) * 4
                image[i:i + 4] = bytes((alpha, *(color if alpha else (0, 0, 0))))
        frames.extend(image)
    return bytes(frames)


def premultiply(argb):
    result = bytearray(argb)
    table = [bytes((v * a + 127) // 255 for v in range(256)) for a in range(256)]
    for i in range(0, len(result), 4):
        lut = table[result[i]]
        result[i + 1] = lut[result[i + 1]]
        result[i + 2] = lut[result[i + 2]]
        result[i + 3] = lut[result[i + 3]]
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", type=Path)
    parser.add_argument("plugin", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("RUNNER_TEMP", "target/ffgl-minihost")))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    width, height, count = 320, 180, 48
    raw = sequence(width, height, count)
    size = width * height * 4
    cases = 0
    with tempfile.TemporaryDirectory(prefix="segcam-", dir=args.work_dir) as directory:
        directory = Path(directory)
        source = directory / "input.argb"
        source.write_bytes(raw)
        for mode, name in enumerate(("overlay", "matte", "boxmatte", "swarm")):
            base = [str(args.cli.resolve()), str(width), str(height), "--mode", name, "--straight-alpha"]
            for variant in ("default", "custom", "rewind", "resize"):
                command = base.copy()
                if variant == "custom":
                    command += ["--sensitivity", "90", "--adaptation", "12.5", "--background", "4",
                                "--min-size", "0.125", "--max-segments", "8", "--grid", "96",
                                "--mirror", "--no-labels", "--trail", "5", "--max-tiles", "12"]
                segments = (raw[:24 * size], raw[24 * size:]) if variant == "rewind" else (raw,)
                reference = b"".join(subprocess.run(command, input=segment, capture_output=True, check=True).stdout
                                     for segment in segments)
                assert len(reference) == len(raw), "CLI returned the wrong number of frames"
                expected = premultiply(reference)
                # Both padded and exact allocations exercise the default settings.
                for padding in ((0, 1) if variant == "default" else (1,)):
                    output = directory / "plugin.argb"
                    subprocess.run([str(args.host.resolve()), str(args.plugin.resolve()), "segcam-stream",
                                    str(width), str(height), str(mode), str(source), str(output),
                                    str(padding), variant], check=True)
                    actual = output.read_bytes()
                    if actual != expected:
                        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
                        raise AssertionError(f"{name}/{variant}/pad={padding}: mismatch at byte {first}, "
                                             f"frame {first // size}, got {actual[first:first+4]!r}, expected {expected[first:first+4]!r}")
                    cases += 1
                    print(f"IDENTICAL {name:8} {variant:7} padded={padding}: {count} frames", flush=True)
    print(f"{cases} sequences, {cases * count} frames: identical 8-bit premultiplied ARGB")


if __name__ == "__main__":
    main()
