"""Real sound effects (Mixkit free license) placed on the cut/transition timeline -> sfx2.wav"""
import json, subprocess, numpy as np, wave
from pathlib import Path
from playwright.sync_api import sync_playwright
H = Path(__file__).resolve().parent; SFX = H.parent / "sfx"; SR = 48000
def load(name):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(SFX / f"{name}.mp3"), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768
KIT = {"line": ("whoosh_1491", .55), "cut": ("swoosh_166", .30), "flip": ("paper_1104", .80), "glitch": ("glitch_2595", .45), "card": ("click_2568", .40)}
S = {k: load(v[0]) for k, v in KIT.items()}
with sync_playwright() as p:
    b = p.chromium.launch(args=["--allow-file-access-from-files"]); pg = b.new_page()
    pg.goto((H / "overlay.html").as_uri()); pg.wait_for_function("window.READY===true"); cards = pg.evaluate("window.CARD_TIMES"); b.close()
ev = json.load(open(H / "events.json")) + [{"t": t, "kind": "card"} for t in cards]
T = json.load(open(H / "cues.json"))[-1]["t1"] + 1.5; out = np.zeros(int(T * SR))
for e in ev:
    k = e["kind"]; sig = S[k] * KIT[k][1]
    at = e["t"] - (.25 if k in ("line", "cut") else 0)   # whooshes lead into the cut
    i = max(0, int(at * SR)); n = min(len(sig), len(out) - i); out[i:i + n] += sig[:n]
out = np.clip(out, -1, 1)
with wave.open(str(H / "sfx2.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype(np.int16).tobytes())
print({k: sum(e["kind"] == k for e in ev) for k in KIT})
