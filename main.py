"""
Entry point. Run once (cron, or by hand) -- pulls the same setlineups
page four times (season / vsR / vsL / last15) to get real Ottoneu-native
stat lines for each split, blends them into a matchup signal, asks Claude
for a recommendation, writes the email locally, and stops. No auto-
actions, per the steering doc's core design principle.

Usage:
    python main.py

Schedule with cron for your actual decision cadence, e.g. nightly at 6pm
before next day's lock:
    0 18 * * * cd /path/to/fantasy_agent && /path/to/venv/bin/python main.py
"""

from __future__ import annotations
import datetime as dt
import sys

import config
import ottoneu_roster
import matchup_signal
import pitcher_matchup_report
import reasoning
import email_output


def run() -> None:
    target_date = dt.date.today() + dt.timedelta(days=config.LOOKAHEAD_DAYS)
    print(f"[{dt.datetime.now()}] Running for slate date {target_date}")

    if config.EVALUATE_HITTERS:
        print("Fetching Ottoneu roster (season / vsR / vsL / last15)...")
        season_players = ottoneu_roster.fetch_roster_and_matchups(target_date, "season")
        vsr_players = ottoneu_roster.fetch_roster_and_matchups(target_date, "vsR")
        vsl_players = ottoneu_roster.fetch_roster_and_matchups(target_date, "vsL")
        last15_players = ottoneu_roster.fetch_roster_and_matchups(target_date, "last15")
    else:
        # Pitching-only mode: pitcher_signals only needs season + last15,
        # so skip the vsR/vsL fetches entirely -- halves the page-fetch
        # work per run. See config.EVALUATE_HITTERS.
        print("Fetching Ottoneu roster (season / last15 -- vsR/vsL skipped, hitters disabled)...")
        season_players = ottoneu_roster.fetch_roster_and_matchups(target_date, "season")
        last15_players = ottoneu_roster.fetch_roster_and_matchups(target_date, "last15")
    print(f"  {len(season_players)} rows on the season view")

    if config.EVALUATE_HITTERS:
        batter_signals = matchup_signal.build_batter_signals(
            season_players, vsr_players, vsl_players, last15_players
        )
    else:
        batter_signals = {}
    pitcher_signals = matchup_signal.build_pitcher_signals(season_players, last15_players)

    sp_candidates = [
        p.name for p in season_players
        if p.is_pitcher and p.roster_slot in ("SP", "Bench")
        and "SP" in p.position_eligibility and p.opponent_team_abbrev
    ]
    print(f"Checking rotation predictions for {len(sp_candidates)} SP-eligible candidates "
          f"(active SP slots + bench starters -- an empty SP slot or a benched starter with "
          f"a great matchup are exactly the calls this agent should be able to make) "
          f"(a rostered SP's team having a game tomorrow doesn't mean HE'S pitching "
          f"it -- using real rotation-gap prediction instead of the roster page for this)...")
    confirmed_starters, uncertain_starters = pitcher_matchup_report.filter_to_predicted_starters(
        sp_candidates, target_date
    )
    if uncertain_starters:
        print(f"  No rotation prediction available for: {', '.join(uncertain_starters)} "
              f"(including anyway, flagged low-confidence)")
    probable_starters = confirmed_starters + uncertain_starters
    print(f"  {len(confirmed_starters)} confirmed by rotation prediction, "
          f"{len(uncertain_starters)} uncertain, {len(sp_candidates)} total SP candidates")

    print(f"Fetching matchup grades for {len(probable_starters)} probable starters "
          f"(reuses league_analysis_final's existing pitching-report pipeline)...")
    matchup_grades = pitcher_matchup_report.get_matchups_for_date(probable_starters, target_date)

    batter_inputs: list[reasoning.BatterInput] = []
    pitcher_inputs: list[reasoning.PitcherInput] = []

    for player in season_players:
        if not player.opponent_team_abbrev:
            continue  # no game today for this player's MLB team

        if player.is_pitcher:
            if player.name not in probable_starters:
                continue  # not predicted to start on target_date -- see README on rotation logic
            signal = pitcher_signals.get(player.ottoneu_id)
            if signal is None:
                continue
            matchup = matchup_grades.get(player.name)
            grade = matchup["grade"] if matchup else None
            ops_rank = None
            ops_value = None
            if matchup and matchup.get("rankings", {}).get("OPS"):
                r = matchup["rankings"]["OPS"]
                ops_rank = f"{r['rank']}/{r['total']}"
                ops_value = r.get("value")
            pitcher_inputs.append(
                reasoning.PitcherInput(
                    name=player.name,
                    opponent_team=player.opponent_team_abbrev,
                    season_pts_per_out=signal.season_pts_per_out,
                    recent_pts_per_out=signal.recent_pts_per_out,
                    recent_outs_sample=signal.recent_outs,
                    matchup_grade=grade,
                    opponent_ops_rank=ops_rank,
                    opponent_ops=ops_value,
                    rotation_confirmed=player.name in confirmed_starters,
                )
            )
        elif config.EVALUATE_HITTERS:
            signal = batter_signals.get(player.ottoneu_id)
            if signal is None:
                continue
            adv = signal.season_advanced
            batter_inputs.append(
                reasoning.BatterInput(
                    name=player.name,
                    position_eligibility=player.position_eligibility,
                    opponent_pitcher_name=player.opposing_pitcher_name or "TBD",
                    opponent_pitcher_hand=player.opposing_pitcher_hand or "?",
                    matchup_pts_per_pa=signal.expected_pts_per_pa,
                    season_pts_per_pa=signal.season_pts_per_pa,
                    platoon_pa_sample=signal.platoon_pa,
                    recent_pts_per_pa=signal.recent_pts_per_pa,
                    season_avg=adv.avg if adv else None,
                    season_obp=adv.obp if adv else None,
                    season_slg=adv.slg if adv else None,
                    season_ops=adv.ops if adv else None,
                    season_woba=adv.woba if adv else None,
                )
            )
        # else: hitter evaluation disabled (config.EVALUATE_HITTERS=False) -- skip batters entirely

    if not batter_inputs and not pitcher_inputs:
        print("No players with usable data for this slate -- nothing to recommend. Exiting.")
        sys.exit(0)

    slate = reasoning.LineupSlateInput(
        date=target_date.isoformat(),
        available_batters=batter_inputs,
        available_pitchers=pitcher_inputs,
        # Pitching-only for now (config.EVALUATE_HITTERS=False) -- SP only.
        # RP is deliberately left out here even though the real roster has
        # 4 RP slots: this codebase doesn't evaluate relief pitchers at
        # all yet (documented gap in the README), so asking Claude to fill
        # RP slots with zero RP candidates in the data just confused it
        # into commenting on the shortfall instead of doing the SP part
        # cleanly -- confirmed happening in practice, not just a guess.
        # Restore both SP and RP here once RP evaluation actually exists.
        roster_slots=({"SP": 5} if not config.EVALUATE_HITTERS
                      else {"C": 2, "1B": 1, "2B": 1, "SS": 1, "MI": 1, "3B": 1,
                            "OF": 5, "Util": 1, "SP": 5, "RP": 4}),
    )

    print(f"Calling Claude with {len(batter_inputs)} batters and {len(pitcher_inputs)} pitchers...")
    recommendation = reasoning.get_recommendation(slate)

    html_path, txt_path = email_output.write_local(recommendation, slate)
    print(f"Wrote {html_path}")
    print(f"Wrote {txt_path}")

    if config.SEND_EMAIL:
        email_output.send_via_smtp(recommendation)
        print("Sent email.")


if __name__ == "__main__":
    run()
