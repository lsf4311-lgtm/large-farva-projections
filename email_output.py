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


_GRADE_COLORS = {"A": "#1a7f37", "B": "#4a9c4a", "C": "#8a8a2a", "D": "#c07a1e", "F": "#c0392b"}


def _pitcher_matchup_rows(slate: LineupSlateInput | None) -> str:
    """Same principle as the batter table above: every number here comes
    straight from PitcherInput, not from Claude's prose, so the person can
    check the model's rationale against the actual inputs it was given
    rather than trusting a one-sentence summary of four separate signals."""
    if not slate or not slate.available_pitchers:
        return ""
    rows = []
    for p in sorted(slate.available_pitchers,
                     key=lambda x: (x.matchup_grade or "Z", -x.season_pts_per_out)):
        grade_display = p.matchup_grade or "—"
        color = _GRADE_COLORS.get(p.matchup_grade, "#888")
        rank_display = p.opponent_ops_rank or "no data"
        confirmed_display = (
            "Confirmed" if p.rotation_confirmed else
            '<span style="color:#c07a1e;">Unconfirmed</span>'
        )
        rows.append(
            f"<tr><td>{p.name}</td><td>{p.opponent_team}</td>"
            f"<td style='color:{color}; font-weight:600;'>{grade_display}</td>"
            f"<td>{rank_display}</td>"
            f"<td>{p.season_pts_per_out:.3f}</td>"
            f"<td>{p.recent_pts_per_out:.3f} <span style='color:#888;'>({p.recent_outs_sample} outs)</span></td>"
            f"<td>{confirmed_display}</td></tr>"
        )
    if not rows:
        return ""
    return f"""
  <h3>Pitcher Matchup Data</h3>
  <table style="border-collapse: collapse; font-size: 13px;">
    <tr style="text-align:left; color:#666;">
      <th style="padding-right:16px;">Player</th><th style="padding-right:16px;">Opp</th>
      <th style="padding-right:16px;">Grade</th><th style="padding-right:16px;">Opp OPS Rank</th>
      <th style="padding-right:16px;">Season Pts/Out</th><th style="padding-right:16px;">Recent Pts/Out</th>
      <th>Rotation</th>
    </tr>
    {''.join(rows)}
  </table>
  <p style="color:#888; font-size: 11px;">Opp OPS Rank: 1/30 = toughest lineup in MLB, 30/30 = weakest.</p>
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
{_pitcher_matchup_rows(slate)}
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


def write_json(recommendation: dict, slate: LineupSlateInput,
               path: str = "data/pitching_recommendation_latest.json") -> str:
    """Writes a combined JSON (Claude's recommendation + the raw pitcher
    data table) to a TRACKED location, not the gitignored output/ dir --
    this file needs to survive a git commit + push to actually show up on
    the deployed app.py Streamlit page. Overwrites the previous day's file
    each run rather than keeping history; add date-stamped archiving later
    if trend-over-time ever becomes something worth showing."""
    import json as _json
    from dataclasses import asdict

    payload = {
        "date": recommendation.get("date", slate.date),
        "generated_at": dt.datetime.now().isoformat(),
        "start": recommendation.get("start", []),
        "sit": recommendation.get("sit", []),
        "close_calls": recommendation.get("close_calls", []),
        "low_confidence_flags": recommendation.get("low_confidence_flags", []),
        "pitchers": [asdict(p) for p in slate.available_pitchers],
        "batters": [asdict(b) for b in slate.available_batters],
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        _json.dump(payload, f, indent=2)
    return path


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
