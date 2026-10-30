"""Frame checks for cases that have a known answer without a reference render.

Each factory returns a check: check(case, variant, frames) -> list of failure messages,
where `frames` maps a render resolution to its list of frames (`Image`s) and
`variant.depth` is the project's bits per channel. Output frames are After Effects'
TIFFs, which store premultiplied 8-bit RGBA.
"""

TOLERANCE = 2


class Image:
    def __init__(self, width, height, data):
        self.width, self.height, self.data = width, height, data

    def pixel(self, x, y):
        i = (y * self.width + x) * 4
        return tuple(self.data[i : i + 4])

    def alpha(self, x, y):
        return self.data[(y * self.width + x) * 4 + 3]


def alpha_spreads_past(x, y, expect=True):
    """Alpha at (x, y), outside the source's edge, is non-zero (or zero if not expect)."""

    def check(case, variant, frames):
        a = frames["Full"][0].alpha(x, y)
        if (a > 0) != expect:
            want = "> 0" if expect else "0"
            return [f"alpha at ({x}, {y}) is {a}, expected {want}"]
        return []

    return check


def edges_symmetric(axis="x", centre=None):
    """Alpha mirrors across the centre: blur reaches equally far past both edges.

    `centre` is the source's geometric centre in comp pixels, default the comp centre.
    """

    def check(case, variant, frames):
        img = frames["Full"][0]
        cx, cy = centre or (img.width / 2, img.height / 2)
        worst = (0, None)
        if axis == "x":
            y = int(cy)
            for x in range(img.width):
                m = round(2 * cx - 1 - x)
                if 0 <= m < img.width:
                    d = abs(img.alpha(x, y) - img.alpha(m, y))
                    worst = max(worst, (d, (x, m)), key=lambda w: w[0])
        else:
            x = int(cx)
            for y in range(img.height):
                m = round(2 * cy - 1 - y)
                if 0 <= m < img.height:
                    d = abs(img.alpha(x, y) - img.alpha(x, m))
                    worst = max(worst, (d, (y, m)), key=lambda w: w[0])
        if worst[0] > TOLERANCE:
            a, b = worst[1]
            return [f"alpha not symmetric along {axis}: {a} vs {b} differ by {worst[0]}"]
        return []

    return check


def flat_value(expected_by_depth):
    """The centre pixel's red channel equals the expected 8-bit value for this depth."""

    def check(case, variant, frames):
        img = frames["Full"][0]
        r = img.pixel(img.width // 2, img.height // 2)[0]
        want = expected_by_depth[variant.depth]
        if abs(r - want) > TOLERANCE:
            return [f"centre red is {r}, expected {want} at {variant.depth} bpc"]
        return []

    return check
