"""HTML rendering of token streams for the GUI (kept out of app.py so it's testable)."""
import html
import math

from .metrics import entropy_from_top

# Diverging scale: confident = blue, uncertain = orange, no data = grey. CVD-safe pair (no red/green).
BLUE, ORANGE, GREY = (42, 120, 214), (235, 104, 52), (137, 135, 129)


def token_color(logprob: float | None) -> str:
    if logprob is None:
        return f"rgba({GREY[0]},{GREY[1]},{GREY[2]},0.18)"
    p = math.exp(logprob)
    if p >= 0.5:  # 0.5..1 -> faint..strong blue
        a, (r, g, b) = 0.12 + (p - 0.5) * 0.9, BLUE
    else:         # 0.5..0 -> faint..strong orange
        a, (r, g, b) = 0.12 + (0.5 - p) * 1.3, ORANGE
    return f"rgba({r},{g},{b},{a:.2f})"


def token_tooltip(ev: dict) -> str:
    if ev.get("logprob") is None:
        return "no logprobs from this model"
    lines = [f"p = {math.exp(ev['logprob']):.1%}"]
    ent = entropy_from_top(ev.get("top") or [])
    if ent is not None:
        lines.append(f"entropy = {ent:.2f} bits")
    for alt in ev.get("top") or []:
        lines.append(f"{alt['token']!r}: {math.exp(alt['logprob']):.1%}")
    return "\n".join(lines)


def tokens_html(events: list[dict]) -> str:
    spans = []
    for ev in events:
        txt = html.escape(ev["text"]).replace("\n", "↵<br>")
        spans.append(f'<span title="{html.escape(token_tooltip(ev))}" style="background:{token_color(ev.get("logprob"))};'
                     f'border-radius:3px;padding:1px 0;margin:0 1px 0 0;white-space:pre-wrap">{txt}</span>')
    return ('<div style="font-family:ui-monospace,Menlo,monospace;font-size:14px;line-height:1.9;'
            'max-height:420px;overflow-y:auto">' + "".join(spans) + "</div>")


LEGEND = ('<div style="font-size:12px;opacity:.8">'
          f'<span style="background:rgba({BLUE[0]},{BLUE[1]},{BLUE[2]},.55);padding:0 6px;border-radius:3px">confident</span> '
          f'<span style="background:rgba({ORANGE[0]},{ORANGE[1]},{ORANGE[2]},.6);padding:0 6px;border-radius:3px">uncertain</span> '
          f'<span style="background:rgba({GREY[0]},{GREY[1]},{GREY[2]},.25);padding:0 6px;border-radius:3px">no logprobs</span>'
          ' · hover a token for its top-5 alternatives</div>')
