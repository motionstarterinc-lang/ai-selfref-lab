import subprocess, sys
from pathlib import Path
from playwright.sync_api import sync_playwright
H = Path(__file__).resolve().parent; FPS = 30
LINES = Path("/home/claude/rec/lines")
with sync_playwright() as p:
    b = p.chromium.launch(args=["--allow-file-access-from-files"]); pg = b.new_page(viewport={"width": 1080, "height": 1920})
    pg.goto((H / "term.html").as_uri()); pg.wait_for_function("window.READY===true")
    for line in map(int, sys.argv[1:]):
        D = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(LINES / f"L{line:02d}.mp4")], capture_output=True, text=True).stdout)
        out = H / "term" / f"L{line:02d}.mp4"; out.parent.mkdir(exist_ok=True)
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
        for n in range(int(D * FPS) + 2):
            pg.evaluate(f"window.render({line}, {n / FPS})"); ff.stdin.write(pg.screenshot(type="jpeg", quality=95))
        ff.stdin.close(); ff.wait(); print(out)
    b.close()
