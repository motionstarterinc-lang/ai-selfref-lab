# How the overview video was made

Everything is scripted and free except a few Higgsfield generations (~60 credits total).

| Layer | Tool | Files |
|---|---|---|
| Voiceover | My phone, cut to the script with Whisper word timings, cleaned with ffmpeg (high-pass, denoise, compressor, loudnorm) | `edit/captions.py`, `SCRIPT.md` |
| Real footage | Screen recordings of the app + terminal | (not in repo) |
| 3D shots | Blender 5 (Cycles, CPU) driven from Python | `blender/scenes.py` |
| Cinematic inserts | Higgsfield (GPT Image 2.5 stills → Kling 3.0 animations), prompted like a film shoot: one hard key light, one rim, real-film texture details, lens, and a list of things to avoid | prompts below |
| Stock | Mixkit (free license) | (not in repo) |
| Terminal scenes, captions, number cards | HTML rendered frame-by-frame with Playwright | `edit/term.html`, `edit/overlay.html`, `edit/render_term.py`, `edit/render_overlay.py` |
| Cuts, transitions, glitch hits | ffmpeg | `edit/build2.py`, `edit/render_v5.py` |
| Sound effects | Mixkit SFX on the cut timeline | `edit/sfx2.py` |

Run order: `captions.py` → `render_term.py 1 4 5` → `FULL=1 build2.py` → `render_overlay.py` → `sfx2.py` → `NOMUSIC=1 render_v5.py`.
Paths to the raw recordings are hard-coded at the top of the scripts; point them at your own footage.

## Example Higgsfield prompt (the style that stopped looking "AI")
> Macro close-up on an old wooden-handled rubber stamp pressed down hard onto a tall stack of identical white index cards on a scratched metal desk… One hard caged work lamp hangs directly above and is the only key light… A single cold cyan fluorescent tube far behind gives a thin rim… Photographed with a real camera on real film: worn lacquer, dried ink crust, paper fibers, scuffs and rust spots… 50mm macro… No hands, no people, no readable text, no logos, no watermark.
