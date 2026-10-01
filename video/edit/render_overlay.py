"""Render the HTML overlay (captions, cards, flashes) to a transparent ProRes 4444 layer -> layers/overlay_alpha.mov"""
import json, subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
H = Path(__file__).resolve().parent; FPS = 30
(H / "layers").mkdir(exist_ok=True)
T = json.load(open(H / "cues.json"))[-1]["t1"] + 0.3; N = int(T * FPS)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-",
                       "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", str(H / "layers" / "overlay_alpha.mov")], stdin=subprocess.PIPE)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--allow-file-access-from-files"]); pg = b.new_page(viewport={"width": 1080, "height": 1920})
    pg.goto((H / "overlay.html").as_uri()); pg.wait_for_function("window.READY===true")
    for n in range(N):
        pg.evaluate(f"window.render({n / FPS})"); ff.stdin.write(pg.screenshot(type="png", omit_background=True))
        if n % (FPS * 15) == 0: print(f"{n / FPS:.0f}/{T:.0f}s", flush=True)
    b.close()
ff.stdin.close(); ff.wait(); print("overlay done")
