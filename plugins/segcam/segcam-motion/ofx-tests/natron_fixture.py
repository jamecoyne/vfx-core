"""Generate or check the Natron PNG fixture (requires Pillow)."""
import sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[2])
if sys.argv[1] == "generate":
    root.mkdir(parents=True, exist_ok=True)
    for t in range(32):
        image = Image.new("RGBA", (320, 180), (24, 24, 24, 255))
        pixels = image.load()
        for y in range(60, 105):
            for x in range(20 + t * 6, 60 + t * 6):
                pixels[x, y] = (220, 220, 220, 255)
        image.save(root / f"in{t:04}.png")
elif sys.argv[1] == "check":
    for t in range(20, 24):
        image = Image.open(root / f"out{t:04}.png").convert("RGB")
        assert image.size == (320, 180), image.size
        greens = [(x, y) for y in range(180) for x in range(320)
                  if (lambda p: p[1] > p[0] + 25 and p[1] > p[2] + 25)(image.getpixel((x, y)))]
        assert len(greens) > 100, (t, len(greens))
        assert any(15 + t*6 <= x <= 65 + t*6 and 55 <= y <= 110 for x, y in greens)
        print(f"frame {t}: {len(greens)} green overlay pixels near the moving object")
else:
    raise SystemExit("usage: natron_fixture.py generate|check SCRATCH_DIRECTORY")
