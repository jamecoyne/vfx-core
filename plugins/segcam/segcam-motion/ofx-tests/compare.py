"""Render a moving sequence through the DLL and the reference CLI."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument("host"); p.add_argument("plugin"); p.add_argument("cli"); p.add_argument("scratch")
p.add_argument("--quick",action="store_true",help="8 bit only: every mode, row order, alpha and render order")
a=p.parse_args(); scratch=Path(a.scratch); scratch.mkdir(parents=True,exist_ok=True)
cases=0
for premult in (0,1):
    raw=bytearray()
    for t in range(32):
        for y in range(180):
            for x in range(320):
                moving=20+t*6 <= x < 60+t*6 and 60 <= y < 105
                alpha=128 if premult else 255
                v=(110 if moving else 12) if premult else (220 if moving else 24)
                raw.extend((alpha,v,v,v))
    source=scratch/f"input-{premult}.argb"; source.write_bytes(raw)
    for mode,index in (("overlay",0),("matte",1),("swarm",3)):
        ref=subprocess.run([a.cli,"320","180","--mode",mode]+(["--straight-alpha"] if not premult else []),input=raw,stdout=subprocess.PIPE,check=True).stdout
        float_ref=None if a.quick else subprocess.run([a.cli,"320","180","--mode",mode,"--float"]+(["--straight-alpha"] if not premult else []),input=raw,stdout=subprocess.PIPE,check=True).stdout
        for depth in ((8,) if a.quick else (8,16,32)):
            for top in (0,1):
                # Forwards, backwards, shuffled, and every frame cold on a fresh instance.
                for order in (0,1,2,3):
                    out=scratch/"output.argb"
                    subprocess.run([a.host,a.plugin,str(source),str(out),str(index),str(depth),str(top),str(premult),str(order)],check=True)
                    actual=out.read_bytes(); assert len(actual)==len(ref)
                    expected=ref if depth==8 else float_ref
                    error=0 if actual==expected else max(abs(x-y) for x,y in zip(actual,expected))
                    assert error <= (0 if depth==8 else 1),(mode,depth,top,premult,order,error)
                    cases+=1
                    print(f"{mode} {depth} bit top={top} premult={premult} order={order}: max error {error}",flush=True)
print(f"{cases} DLL/CLI comparisons passed")
