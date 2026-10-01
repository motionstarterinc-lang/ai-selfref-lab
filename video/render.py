"""Render video/index.html to MP4, frame by frame (deterministic).   python video/render.py [--fps 24] [--start 0 --end 90]"""
import argparse
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--start", type=float, default=0)
ap.add_argument("--end", type=float, default=90)
ap.add_argument("--out", default=str(HERE / "mimicry_study.mp4"))
ap.add_argument("--stills", action="store_true", help="save one PNG per scene instead of a video")
a = ap.parse_args()

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--allow-file-access-from-files"])
    pg = b.new_page(viewport={"width": 1080, "height": 1920})
    pg.goto((HERE / "index.html").as_uri() + "?capture")
    pg.wait_for_function("window.READY === true", timeout=60000)
    if a.stills:
        for i, t in enumerate([4, 10, 16, 25, 32, 40, 50, 59, 68, 81, 88]):
            pg.evaluate(f"window.render({t})")
            pg.screenshot(path=str(HERE / f"still_{i + 1:02d}.png"))
        print("stills written")
    else:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", a.out],
                              stdin=subprocess.PIPE)
        n = int((a.end - a.start) * a.fps)
        for i in range(n):
            pg.evaluate(f"window.render({a.start + i / a.fps})")
            ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
            if i % (a.fps * 5) == 0:
                print(f"{i / a.fps:5.1f}s / {a.end - a.start:.0f}s", flush=True)
        ff.stdin.close()
        ff.wait()
        print("wrote", a.out)
    b.close()
