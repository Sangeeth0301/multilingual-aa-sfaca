"""Highlighted-document export (self-contained HTML)."""
import datetime
import html

TIERS = [(0.8, "#e5484d"), (0.65, "#f08a8d"), (0.5, "#f8c9ca")]


def _style(seg):
    if seg["type"] == "notation":
        return "background:#4b5563;color:#f3f4f6;font-family:monospace"
    p = seg.get("ai_prob")
    if p is None or not seg.get("scored"):
        return ""
    for thr, col in TIERS:
        if p >= thr:
            return "background:%s;color:#111" % col
    return ""


def _title(seg):
    p = seg.get("ai_prob")
    return "" if p is None else "%.0f%% AI" % (p * 100)


def to_html(result: dict, filename: str = "document") -> str:
    body = "".join(
        '<span style="%s" title="%s">%s</span>' % (_style(s), _title(s), html.escape(s["text"]))
        for s in result["segments"])
    scored = [s for s in result["segments"] if s.get("scored")]
    flagged = sum(1 for s in scored if s["ai_prob"] >= 0.5)
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(filename)} - SFA-CA report</title>
<body style="font:16px/1.7 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem">
<h1>AI-attribution report</h1>
<p><b>{result["ai_pct"]:.1f}% likely AI</b> / {result["human_pct"]:.1f}% likely human.
Most likely generator (if AI): <b>{html.escape(str(result["top_generator"]))}</b>.
{flagged} of {len(scored)} passages flagged. Generated {datetime.date.today()}.</p>
<p><span style="background:#f8c9ca">50-65%</span> <span style="background:#f08a8d">65-80%</span>
<span style="background:#e5484d">80%+</span> <span style="background:#4b5563;color:#fff">math/code (not scored)</span></p>
<p style="color:#555;font-size:14px">This is a statistical likelihood, not proof of authorship. Short, paraphrased or edited text is unreliable.</p>
<hr><div style="white-space:pre-wrap">{body}</div></body></html>"""
