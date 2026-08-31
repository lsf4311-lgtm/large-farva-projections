"""
Renders the recommendation as an email. Sending is intentionally NOT wired
up yet -- per the project's core design principle, this agent proposes and
stops; a human reads the email and acts on it manually in Ottoneu. When
you're ready to actually send (vs. just generate), fill in SMTP_CONFIG in
config.py, flip config.SEND_EMAIL, and use the send_via_smtp() stub below.
"""

from __future__ import annotations
import datetime as dt
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config
from reasoning import LineupSlateInput


def _row(name: str, reason: str) -> str:
    return f"<li><strong>{name}</strong> — {reason}</li>"


def _advanced_stats_rows(slate: LineupSlateInput | None) -> str:
    """Built directly from the raw computed numbers, not from Claude's
    response -- keeps these figures exact rather than model-transcribed."""
    if not slate or not slate.available_batters:
        return ""
    rows = []
    for b in sorted(slate.available_batters, key=lambda x: x.season_ops or 0, reverse=True):
        if b.season_ops is None:
            continue
        rows.append(
            f"<tr><td>{b.name}</td>"
            f"<td>{b.season_avg:.3f}</td><td>{b.season_obp:.3f}</td>"
            f"<td>{b.season_slg:.3f}</td><td>{b.season_ops:.3f}</td>"
            f"<td>{b.season_woba:.3f}</td></tr>"
        )
    if not rows:
        return ""
    return f"""
  <h3>Season Advanced Stats</h3>
  <table style="border-collapse: collapse; font-size: 13px;">
    <tr style="text-align:left; color:#666;">
      <th style="padding-right:16px;">Player</th><th style="padding-right:16px;">AVG</th>
      <th style="padding-right:16px;">OBP</th><th style="padding-right:16px;">SLG</th>
      <th style="padding-right:16px;">OPS</th><th>wOBA</th>
    </tr>
    {''.join(rows)}
  </table>
"""


def render_html(recommendation: dict, slate: LineupSlateInput | None = None) -> str:
    date = recommendation.get("date", "")
    start = recommendation.get("start", [])
    sit = recommendation.get("sit", [])
    close_calls = recommendation.get("close_calls", [])
    flags = recommendation.get("low_confidence_flags", [])

    start_html = "\n".join(_row(p["name"], p["reason"]) for p in start)
    sit_html = "\n".join(_row(p["name"], p["reason"]) for p in sit)

    close_html = "".join(
        f"<li>{' vs '.join(c['players'])}: {c['note']}</li>" for c in close_calls
    ) or "<li>None today.</li>"

    flags_html = "".join(
        f"<li>{f['name']}: {f['reason']}</li>" for f in flags
    ) or "<li>None -- all recommendations rest on reasonable samples.</li>"

    return f"""\
<html>
<body style="font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 640px; margin: auto; color: #1a1a1a;">
  <h2>{config.EMAIL_SUBJECT_PREFIX} — {date}</h2>

  <h3>Start</h3>
  <ul>{start_html or '<li>No changes recommended.</li>'}</ul>

  <h3>Sit</h3>
  <ul>{sit_html or '<li>No changes recommended.</li>'}</ul>

  <h3>Close calls</h3>
  <ul>{close_html}</ul>

  <h3>Low-confidence flags (small sample)</h3>
  <ul>{flags_html}</ul>
{_advanced_stats_rows(slate)}
  <p style="color:#888; font-size: 12px;">
    Generated automatically from splits + matchup data. Nothing has been
    submitted to Ottoneu -- review and set your lineup manually.
  </p>
</body>
</html>
"""


def render_text(recommendation: dict) -> str:
    lines = [f"{config.EMAIL_SUBJECT_PREFIX} — {recommendation.get('date', '')}", ""]
    lines.append("START")
    for p in recommendation.get("start", []):
        lines.append(f"  - {p['name']}: {p['reason']}")
    lines.append("")
    lines.append("SIT")
    for p in recommendation.get("sit", []):
        lines.append(f"  - {p['name']}: {p['reason']}")
    lines.append("")
    lines.append("CLOSE CALLS")
    for c in recommendation.get("close_calls", []):
        lines.append(f"  - {' vs '.join(c['players'])}: {c['note']}")
    lines.append("")
    lines.append("LOW-CONFIDENCE FLAGS")
    for f in recommendation.get("low_confidence_flags", []):
        lines.append(f"  - {f['name']}: {f['reason']}")
    return "\n".join(lines)


def write_local(recommendation: dict, slate: LineupSlateInput | None = None) -> tuple[str, str]:
    """Writes .html and .txt versions to config.OUTPUT_DIR, returns paths."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    stamp = dt.date.today().isoformat()
    html_path = os.path.join(config.OUTPUT_DIR, f"lineup_recommendation_{stamp}.html")
    txt_path = os.path.join(config.OUTPUT_DIR, f"lineup_recommendation_{stamp}.txt")

    with open(html_path, "w") as f:
        f.write(render_html(recommendation, slate))
    with open(txt_path, "w") as f:
        f.write(render_text(recommendation))

    return html_path, txt_path


def send_via_smtp(recommendation: dict) -> None:
    """Not called anywhere by default (config.SEND_EMAIL gates this).
    Wire up once you've decided you actually want unattended sending --
    revisit the guardrails checklist in the steering doc first."""
    cfg = config.SMTP_CONFIG
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{config.EMAIL_SUBJECT_PREFIX} — {recommendation.get('date', '')}"
    msg["From"] = cfg["from_addr"]
    msg["To"] = cfg["to_addr"]
    msg.attach(MIMEText(render_text(recommendation), "plain"))
    msg.attach(MIMEText(render_html(recommendation), "html"))

    with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
        server.starttls()
        server.login(cfg["username"], cfg["app_password"])
        server.send_message(msg)
