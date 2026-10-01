"""v5 composite: split-screen clips + caption/card layer + RGB-glitch hits + voice + SFX + beat (ducked under voice).
python render_v5.py [out.mp4]"""
import json, os, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; LINES = Path("/home/claude/rec/lines"); FPS = 30
dur = lambda p: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout)
durs = [dur(LINES / f"L{i:02d}.mp4") for i in range(1, 12)]
glitches = [e["t"] for e in json.load(open(H / "events.json")) if e["kind"] == "glitch"]
ins, fc = [], []
for i in range(11):
    ins += ["-i", str(H / "split" / f"L{i+1:02d}.mp4")]
    fc.append(f"[{i}:v]tpad=stop_mode=clone:stop_duration=1,trim=duration={durs[i]:.3f},setpts=PTS-STARTPTS,fps={FPS},setsar=1[b{i}]")
for i in range(11):
    ins += ["-i", str(LINES / f"L{i+1:02d}.mp4")]
import os
NOMUSIC = os.environ.get("NOMUSIC") == "1"   # public version: voice + SFX only (no third-party beat)
ins += ["-i", str(H / "layers" / "overlay_alpha.mov"), "-i", str(H / "sfx2.wav")] + ([] if NOMUSIC else ["-i", str(H / "music" / "music.mp3")])
g = "+".join(f"between(t,{t:.2f},{t + .22:.2f})" for t in glitches)
fc.append("".join(f"[b{i}]" for i in range(11)) + "concat=n=11:v=1:a=0[v0]")
fc.append(f"[v0]rgbashift=rh=-14:bh=14:enable='{g}',noise=alls=40:allf=t:enable='{g}',noise=alls=4:allf=t[v1]")
fc.append("[22:v]format=yuva444p10le[ov];[v1][ov]overlay=0:0:shortest=1,format=yuv420p[v]")
fc.append("".join(f"[{11+i}:a]" for i in range(11)) + "concat=n=11:v=0:a=1,loudnorm=I=-15:TP=-2:LRA=9,aresample=48000,asplit=2[voice][key]")
fc.append("[23:a]aresample=48000,volume=0.9[sfx]")
if NOMUSIC:
    fc.append("[key]anullsink;[voice][sfx]amix=inputs=2:duration=first:normalize=0,afade=t=out:st=%.2f:d=1.2,alimiter=limit=0.95[a]" % (sum(durs) - 1.2))
else:
    fc.append("[24:a]aresample=48000,volume=0.32,afade=t=in:d=1.0[m0];[m0][key]sidechaincompress=threshold=0.02:ratio=5:attack=15:release=350[mus]")
    fc.append("[voice][sfx][mus]amix=inputs=3:duration=first:normalize=0,afade=t=out:st=%.2f:d=1.2,alimiter=limit=0.95[a]" % (sum(durs) - 1.2))
out = H.parent / (sys.argv[1] if len(sys.argv) > 1 else "mimicry_v5.mp4")
r = subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-b:v", os.environ.get("VBR", "2400k"), "-maxrate", "3000k", "-bufsize", "6000k", "-preset", "slow",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
print(r.stderr[-800:] or f"wrote {out}")
