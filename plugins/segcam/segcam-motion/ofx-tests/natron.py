"""Run with NatronRenderer --clear-cache --clear-openfx-cache -t natron.py.
Set SEGCAM_NATRON_DIR to the folder made by natron_fixture.py generate.
"""
import os
from pathlib import Path
import NatronEngine

root = Path(os.environ["SEGCAM_NATRON_DIR"]).as_posix()
app.getProjectParam("outputFormat").set(app.addFormat("320 180 0 0 320 180 1 segcam"))
reader = app.createReader(root + "/in####.png")
# Let the reader establish the project format before creating the effect.
warmup = app.createWriter(root + "/warmup.png")
warmup.connectInput(0, reader)
app.render(warmup, 0, 0)
effect = app.createNode("com.lofiscifi.SegcamMotionRust")
assert effect is not None, "segcam OpenFX plugin missing"
effect.connectInput(0, reader)
writer = app.createWriter(root + "/out####.png")
writer.connectInput(0, effect)
app.render(writer, 20, 23)
print("SEGCAM_RENDER_COMPLETE", flush=True)
