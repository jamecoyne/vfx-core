"""Test cases for the After Effects harness.

Each case becomes one composition holding one layer with one or more VFX Preview
instances. `run.py` renders every case in every project variant, then runs its checks.

Case fields:
  name        composition name, also the output file prefix
  size        (width, height) of the composition
  source      "pattern" (the generated test image), or ("solid", (r, g, b), (w, h))
  position    layer position in the comp; defaults to the centre
  mask        (left, top, right, bottom) rectangle mask on the source, in layer pixels
  blend       "pattern_b" to add the second test image as a hidden layer and pick it
              in Blend Layer
  effects     one dict per stacked instance: AE parameter name -> value. Popups are
              0-based here. Unlisted parameters keep their defaults.
  keyframes   {parameter: [(seconds, value), ...]} on the first instance
  frames      frames to render (default 1)
  resolutions render resolutions (default ["Full"]); "Half" adds a downsampling check
  reference   True to compare against vfx-core run on the same pixels
  checks      extra checks: callables (case, variant, frames) -> list of failures
"""

from checks import alpha_spreads_past, edges_symmetric, flat_value

# Parameter defaults, as the plugin sets them up (popups 0-based).
DEFAULTS = {
    "Linear Light": False,
    "Color Adjust": True,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "Saturation": 1.0,
    "Vignette": False,
    "Vignette Center": None,  # layer centre
    "Vignette Radius": 0.75,
    "Vignette Soft": 0.5,
    "Vignette Color": (0.0, 0.0, 0.0, 1.0),
    "Vignette Amount": 1.0,
    "Blend Mode": 0,
    "Blend Amount": 1.0,
    "Blur": True,
    "Blur Radius": 0.0,
    "Blur Direction": 0,
    "Blur Edges": False,
}

POPUPS = {"Blend Mode", "Blur Direction"}

PATTERN_SIZE = (96, 64)

CASES = [
    # Defaults must leave the image untouched.
    dict(name="passthrough", source="pattern", effects=[{}], reference=True),
    dict(
        name="color",
        source="pattern",
        effects=[{"Brightness": 0.1, "Contrast": 1.4, "Saturation": 0.4}],
        reference=True,
    ),
    dict(
        name="color_off",
        source="pattern",
        effects=[{"Color Adjust": False, "Brightness": 0.5}],
        reference=True,
    ),
    # Semi-transparent edges: catches a wrong alpha convention or colour bleed.
    dict(name="blur", source="pattern", effects=[{"Blur Radius": 6.0}], reference=True),
    dict(
        name="blur_horizontal",
        source="pattern",
        effects=[{"Blur Radius": 6.0, "Blur Direction": 1}],
        reference=True,
    ),
    dict(
        name="blur_vertical",
        source="pattern",
        effects=[{"Blur Radius": 6.0, "Blur Direction": 2}],
        reference=True,
    ),
    dict(
        name="blur_large",
        source="pattern",
        effects=[{"Blur Radius": 40.0}],
        reference=True,
    ),
    # Downsampled previews must blur by the same amount in full-resolution terms.
    dict(
        name="blur_half",
        source="pattern",
        effects=[{"Blur Radius": 8.0}],
        resolutions=["Full", "Half"],
    ),
    # A mask trims the input; the blur still has to spread past it.
    dict(
        name="blur_mask",
        source=("solid", (1.0, 1.0, 1.0), PATTERN_SIZE),
        mask=(30, 0, 66, 64),
        effects=[{"Blur Radius": 6.0, "Blur Direction": 1}],
        checks=[alpha_spreads_past(27, 32), edges_symmetric(axis="x")],
    ),
    # Blur Edges off: clipped at the layer. On: grows the same on every side.
    dict(
        name="grow_off",
        size=PATTERN_SIZE,
        source=("solid", (1.0, 1.0, 1.0), (40, 24)),
        position=(40, 30),
        effects=[{"Blur Radius": 5.0}],
        checks=[alpha_spreads_past(18, 30, expect=False)],
    ),
    dict(
        name="grow_on",
        size=PATTERN_SIZE,
        source=("solid", (1.0, 1.0, 1.0), (40, 24)),
        position=(40, 30),
        effects=[{"Blur Radius": 5.0, "Blur Edges": True}],
        checks=[alpha_spreads_past(18, 30), edges_symmetric(axis="x", centre=(40, 30))],
    ),
    dict(
        name="vignette",
        source="pattern",
        effects=[{"Vignette": True}],
        reference=True,
    ),
    dict(
        name="vignette_offset",
        source="pattern",
        effects=[
            {
                "Vignette": True,
                "Vignette Center": (30.0, 20.0),
                "Vignette Radius": 0.4,
                "Vignette Soft": 0.3,
                "Vignette Color": (1.0, 0.0, 0.0, 1.0),
                "Vignette Amount": 0.6,
            }
        ],
        reference=True,
    ),
    dict(
        name="vignette_half",
        source="pattern",
        effects=[{"Vignette": True, "Vignette Center": (30.0, 20.0)}],
        resolutions=["Full", "Half"],
    ),
    *[
        dict(
            name=f"blend_{mode}",
            source="pattern",
            blend="pattern_b",
            effects=[{"Blend Mode": index, "Blend Amount": 0.6}],
            reference=True,
        )
        for index, mode in enumerate(["normal", "add", "multiply", "screen"])
    ],
    dict(
        name="linear",
        source="pattern",
        effects=[{"Linear Light": True, "Brightness": 0.1, "Blur Radius": 4.0}],
        reference=True,
    ),
    dict(
        name="linear_blend",
        source="pattern",
        blend="pattern_b",
        effects=[{"Linear Light": True, "Blend Mode": 1, "Blend Amount": 0.5}],
        reference=True,
    ),
    dict(
        name="everything",
        source="pattern",
        blend="pattern_b",
        effects=[
            {
                "Linear Light": True,
                "Brightness": -0.05,
                "Contrast": 1.2,
                "Saturation": 1.3,
                "Vignette": True,
                "Blend Mode": 3,
                "Blend Amount": 0.4,
                "Blur Radius": 3.0,
            }
        ],
        reference=True,
    ),
    # Two stacked instances, as two copies of the effect on one layer.
    dict(
        name="stacked",
        source="pattern",
        effects=[{"Brightness": 0.05, "Blur Radius": 3.0}] * 2,
        reference=True,
    ),
    # 32 bpc: 0.5 + 1 = 1.5 must reach the second instance unclipped and come back to
    # 0.5. Integer depths clip at 1.0 and end at 0.
    dict(
        name="hdr",
        source=("solid", (0.5, 0.5, 0.5), (32, 32)),
        effects=[{"Brightness": 1.0}, {"Brightness": -1.0}],
        checks=[flat_value({8: 0, 16: 0, 32: 128})],
    ),
    # Animated parameter: each frame needs its own time and value, also under MFR.
    dict(
        name="animated",
        source="pattern",
        effects=[{"Blur Radius": 2.0}],
        keyframes={"Brightness": [(0.0, -0.3), (8 / 30, 0.3)]},
        frames=9,
        reference=True,
    ),
]


def params(effect):
    """An instance's full parameter set: defaults overlaid with the case's values."""
    merged = dict(DEFAULTS)
    merged.update(effect)
    return merged
