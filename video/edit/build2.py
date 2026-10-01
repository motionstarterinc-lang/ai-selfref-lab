"""Split-screen edit v5: fast-cut visuals on top (1080x1180), your synced face clip on the bottom (1080x740).
python build2.py  -> edit/top/Lxx.mp4, edit/split/Lxx.mp4, edit/events.json (cut/transition times for SFX + glitches)"""
import json, subprocess, sys
from pathlib import Path

H = Path(__file__).resolve().parent
LINES = Path("/home/claude/rec/lines"); REC = Path("/home/claude/rec/screen")
HF2, HF, BLF, STK, TERM, FIG = H.parent / "hf2", H.parent / "hf", H.parent / "blender/frames", H.parent / "stock", H / "term", H.parent.parent / "docs/findings"
import os
FULL = os.environ.get("FULL") == "1"   # full-screen visuals, no face
W, TH, FH, FPS = 1080, (1920 if FULL else 1180), 740, 30
cues = {c["line"]: c for c in json.load(open(H / "cues.json"))}
(H / "top").mkdir(exist_ok=True); (H / "split").mkdir(exist_ok=True)


def at(line, word, k=0):
    ws = [w for w in cues[line]["words"] if w["w"].lower().strip('.,:?!“”"') == word]
    return round(ws[k]["s"] - cues[line]["t0"] - .08, 3)


# shot kinds: vid (Higgsfield clip), still (photo, push-in), rec (screen recording window), term (terminal render),
#             blend (Blender frames), stock (Mixkit clip), card (chart that page-flips in)
# each shot = (kind, source, start-in-line seconds, extra); a shot runs until the next one starts
SHOTS = {
    1: [("vid", HF2 / "v21.mp4", 0, {}), ("term", "L01", at(1, "it"), {"ss": 1.4}), ("vid", HF2 / "v01.mp4", at(1, "and"), {})],
    2: [("vid", HF2 / "v02.mp4", 0, {}), ("stock", "h20961", at(2, "close"), {}), ("blend", "mirror", at(2, "mimicking"), {"ss": 3})],
    3: [("rec", ("Recording_for_git_hub.mp4", 8, 0, 150, 955), 0, {}), ("rec", ("Recording_for_git_hub.mp4", 46, 8, 280, 640), at(3, "every"), {})],
    4: [("term", "L04", 0, {}), ("rec", ("Recording_for_git_hub_pt_2.mp4", 12, 958, 176, 760), at(4, "grok"), {"speed": 3})],
    5: [("rec", ("Recording_for_git_hub.mp4", 48, 8, 280, 640), 0, {}), ("term", "L05", at(5, "focus"), {"ss": 2.8})],
    6: [("vid", HF2 / "v06.mp4", 0, {}), ("blend", "bars", at(6, "before", 1), {}), ("card", FIG / "q2_base_instruct.png", at(6, "say"), {}),
        ("vid", HF2 / "v06.mp4", at(6, "denial"), {"ss": 2.5})],
    7: [("vid", HF2 / "v07.mp4", 0, {}), ("blend", "grid", at(7, "the"), {}), ("card", FIG / "q1_detect.png", at(7, "even"), {}),
        ("blend", "grid", at(7, "but"), {"ss": 6})],
    8: [("stock", "h43527", 0, {}), ("stock", "h41642", at(8, "one"), {}), ("rec", ("Recording_for_git_hub.mp4", 52, 8, 280, 640), at(8, "the", 0), {})],
    9: [("stock", "h23282", 0, {}), ("vid", HF2 / "v09.mp4", at(9, "they"), {}), ("vid", HF2 / "v29.mp4", at(9, "a", 1), {})],
    10: [("blend", "mirror", 0, {}), ("card", FIG / "q5_reasoning.png", at(10, "until"), {}), ("vid", HF2 / "v30.mp4", at(10, "it", 3), {})],
    11: [("rec", ("Recording_for_git_hub_pt_2.mp4", 30, 0, 200, 955), 0, {})],
}
GLITCH = [(4, "grok"), (1, "potato"), (6, "denial"), (10, "can't"), (7, "nothing")]   # RGB-split hits on reveals
PUNCH = f"scale=w='trunc({W}*(1+0.08*max(0,1-t/0.3))/2)*2':h='trunc({TH}*(1+0.08*max(0,1-t/0.3))/2)*2':eval=frame,crop={W}:{TH}"
GRADE = "eq=contrast=1.06:saturation=1.08,vignette=PI/5"
FILL = f"scale={W}:{TH}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{TH}"
BL = f"nlmeans=s=5:p=5:r=9,hqdn3d=2:2:6:6,{FILL},minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:vsbmc=1,unsharp=5:5:0.4"


def run(args):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-600:]); sys.exit(1)


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout)


def shot(kind, src, d, ex, out):
    enc = ["-t", f"{d:.3f}", "-an", "-c:v", "libx264", "-crf", "17", "-preset", "veryfast", "-r", str(FPS), "-pix_fmt", "yuv420p", str(out)]
    if kind == "vid":   # play at natural speed, slow down only if the shot is longer than the clip
        sp = max(1.0, d / (dur(src) - .05 - ex.get("ss", 0)))
        run(["-ss", str(ex.get("ss", 0)), "-i", str(src), "-vf", f"setpts=PTS*{sp:.3f},fps={FPS},{FILL},{PUNCH},{GRADE}", *enc])
    elif kind == "still":
        n = int(d * FPS) + 2
        run(["-loop", "1", "-framerate", str(FPS), "-i", str(src), "-vf",
             f"scale=2160:-2:flags=lanczos,zoompan=z='1.0+0.10*on/{n}':x='iw/2-iw/zoom/2':y='ih*0.42-ih/zoom/2':d=1:s={W}x{TH}:fps={FPS},{PUNCH},{GRADE}", *enc])
    elif kind == "rec":
        f, a, x, y, w = src; h = int(w * .75); sp = ex.get("speed", 1.0)
        fc = (f"[0:v]setpts=PTS/{sp},fps={FPS},crop={w}:{h}:{x}:{y},split[f][g];[g]{FILL},boxblur=30:3,colorchannelmixer=rr=.16:gg=.17:bb=.22[bg];"
              f"[f]scale=1010:-2:flags=lanczos,pad=iw+6:ih+6:3:3:color=0x2a2f3a[win];[bg][win]overlay=(W-w)/2:(H-h)/2+40,{PUNCH}[v]")
        run(["-ss", str(a), "-i", str(REC / f), "-filter_complex", fc, "-map", "[v]", *enc])
    elif kind == "term":
        run(["-ss", str(ex.get("ss", 0)), "-i", str(TERM / f"{src}.mp4"), "-vf", f"crop={W}:{TH}:0:{0 if FULL else 150},{PUNCH}", *enc])
    elif kind == "blend":
        run(["-framerate", "6", "-i", str(BLF / src / "f_%04d.png"), "-vf", f"trim=start={ex.get('ss', 0)},setpts=PTS-STARTPTS,{BL},{PUNCH}", *enc])
    elif kind == "stock":
        run(["-stream_loop", "2", "-i", str(STK / f"{src}.mp4"), "-vf", f"fps={FPS},{FILL},unsharp=5:5:0.6,{PUNCH},{GRADE}", *enc])
    elif kind == "card":   # page flip: the chart slides in from the right with a slight turn, over a dark blurred backdrop
        fc = (f"color=c=0x0d1017:s={W}x{TH}:r={FPS}[bg];[0:v]scale=1000:-2:flags=lanczos,pad=iw+16:ih+16:8:8:color=white,format=rgba[card];"
              f"[bg][card]overlay=x='(W-w)/2+1100*pow(max(0,1-t/0.3),3)':y='(H-h)/2':shortest=1,"
              f"scale=w='trunc({W}*(1+0.04*min(t,3)/3)/2)*2':h='trunc({TH}*(1+0.04*min(t,3)/3)/2)*2':eval=frame,crop={W}:{TH}[v]")
        run(["-loop", "1", "-framerate", str(FPS), "-i", str(src), "-filter_complex", fc, "-map", "[v]", *enc])


events = []
lines = [int(a) for a in sys.argv[1:]] or list(range(1, 12))
for L in lines:
    D = dur(LINES / f"L{L:02d}.mp4"); t0 = cues[L]["t0"]
    parts = []
    for i, (kind, src, s, ex) in enumerate(SHOTS[L]):
        e = SHOTS[L][i + 1][2] if i + 1 < len(SHOTS[L]) else D
        p = H / "top" / f"L{L:02d}_{i}.mp4"; shot(kind, src, e - s, ex, p); parts.append(p)
        events.append({"t": round(t0 + s, 3), "kind": "line" if i == 0 else ("flip" if kind == "card" else "cut")})
    lst = H / "top" / f"L{L:02d}.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(H / "top" / f"L{L:02d}.mp4")])
    if FULL:
        __import__("shutil").copy(H / "top" / f"L{L:02d}.mp4", H / "split" / f"L{L:02d}.mp4")
        print(f"L{L:02d} {D:.2f}s {len(parts)} shots (full screen)", flush=True); continue
    # bottom: your synced clip, framed on the face
    face = "crop=1080:740:0:330,eq=contrast=1.1:brightness=0.01:saturation=1.06,colorbalance=rs=.02:bs=-.02,vignette=PI/5"
    run(["-i", str(H / "top" / f"L{L:02d}.mp4"), "-i", str(LINES / f"L{L:02d}.mp4"), "-filter_complex",
         f"[1:v]{face},setsar=1[f];[0:v]setsar=1[t];[t][f]vstack=2[v]", "-map", "[v]", "-t", f"{D:.3f}", "-an",
         "-c:v", "libx264", "-crf", "17", "-preset", "veryfast", "-r", str(FPS), str(H / "split" / f"L{L:02d}.mp4")])
    print(f"L{L:02d} {D:.2f}s {len(parts)} shots", flush=True)
for L, w in GLITCH:
    events.append({"t": round(cues[L]["t0"] + at(L, w) + .08, 3), "kind": "glitch"})
old = json.load(open(H / "events.json")) if (H / "events.json").exists() and len(lines) < 11 else []
keep = [e for e in old if not any(cues[L]["t0"] <= e["t"] < cues[L]["t1"] + .2 for L in lines)]
json.dump(sorted(keep + events, key=lambda e: e["t"]), open(H / "events.json", "w"), indent=1)
