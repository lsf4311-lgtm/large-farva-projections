import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Farva Operations Center",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.stApp { background-color: #0a0e1a; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #f0f4ff; }

.metric-card {
    background: #131929;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #38bdf8;
    margin-top: 4px;
}

.rank-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    background: #1e2d4a;
    color: #38bdf8;
}

.starter-tag {
    background: #0f3460;
    color: #38bdf8;
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2px 6px;
    border-radius: 3px;
}
.bench-tag {
    background: #1a1a2e;
    color: #64748b;
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2px 6px;
    border-radius: 3px;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 24px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #1e2d4a;
    border-radius: 8px;
}

.stSelectbox label, .stTextInput label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

section[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e2d4a;
}

.last-updated {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# ── Import pipeline from league_analysis_final.py ───────────────────────────────────
sys.path.append(os.path.dirname(__file__))

from league_analysis_final import (
    get_league_rosters, fuzzy_match_players, optimize_lineup,
    make_api_request, get_fa_positions, fetch_projections,
    get_pitching_report, get_all_team_stats, DATA_DIR, LEAGUE_ID
)

# ── Data Loading (cached weekly) ──────────────────────────────────────────────
PROJECTION_FILES = {
    'OOPSY': {
        'hitting': 'fangraphs-leaderboard-projections_oopsy hitting 2026.csv',
        'pitching': 'fangraphs-leaderboard-projections_oopsy pitching 2026.csv',
    },
    'ATC': {
        'hitting': 'fangraphs-leaderboard-projections_2026 hitting_atc.csv',
        'pitching': 'fangraphs-leaderboard-projections_2026 pitching_atc.csv',
    },
    'OOPSY DC': {
        'hitting': 'fangraphs-leaderboard-projections_oopsydc hitting 2026.csv',
        'pitching': 'fangraphs-leaderboard-projections_oopsydc pitching 2026.csv',
    },
    'ATC DC': {
        'hitting': 'fangraphs-leaderboard-projections_atcdc hitting 2026.csv',
        'pitching': 'fangraphs-leaderboard-projections_atcdc pitching 2026.csv',
    },
    'THE BAT X': {
        'hitting': 'fangraphs-leaderboard-projections_thebatx hitting 2026.csv',
        'pitching': 'fangraphs-leaderboard-projections_thebatx pitching 2026.csv',
    },
    'THE BAT X DC': {
        'hitting': 'fangraphs-leaderboard-projections_thebatxdc hitting 2026.csv',
        'pitching': 'fangraphs-leaderboard-projections_thebatxdc pitching 2026.csv',
    },
}

@st.cache_data(ttl=604800)
def load_all_data(projection_system='OOPSY'):
    """Run the full pipeline and return projected players + standings."""
    # Rosters - with fallback to cached CSV
    try:
        rosters = get_league_rosters()
        rosters.to_csv(os.path.join(DATA_DIR, 'league_rosters.csv'), index=False)
        roster_source = 'live'
    except Exception as e:
        print(f"Live scrape failed: {e}. Falling back to cached rosters.")
        rosters = pd.read_csv(os.path.join(DATA_DIR, 'league_rosters.csv'))
        roster_source = 'cached'

    # Projections — try live fetch first, fall back to CSV
    proj_files = PROJECTION_FILES[projection_system]
    proj_source = 'live'
    try:
        fg_user = st.secrets.get('FG_USER') or os.environ.get('FG_USER')
        fg_pass = st.secrets.get('FG_PASS') or os.environ.get('FG_PASS')
        if not fg_user or not fg_pass:
            raise Exception("FanGraphs credentials not configured")
        print(f"Fetching {projection_system} projections from FanGraphs...")
        atc_hitting, atc_pitching = fetch_projections(projection_system, fg_user, fg_pass)
        # Save fresh copies to disk as updated fallback
        atc_hitting.to_csv(os.path.join(DATA_DIR, proj_files['hitting']), index=False)
        atc_pitching.to_csv(os.path.join(DATA_DIR, proj_files['pitching']), index=False)
        print(f"  Projections saved to disk")
    except Exception as e:
        print(f"Live projection fetch failed: {e}. Falling back to CSV.")
        proj_source = 'cached'
        atc_hitting = pd.read_csv(os.path.join(DATA_DIR, proj_files['hitting']))
        atc_pitching = pd.read_csv(os.path.join(DATA_DIR, proj_files['pitching']))

    atc_hitting = atc_hitting.rename(columns={'PlayerId': 'fg_id'}) if 'PlayerId' in atc_hitting.columns else atc_hitting
    atc_pitching = atc_pitching.rename(columns={'PlayerId': 'fg_id'}) if 'PlayerId' in atc_pitching.columns else atc_pitching

    # Crosswalk
    crosswalk = pd.read_csv(os.path.join(DATA_DIR, 'sfbb_crosswalk.csv'))
    crosswalk = crosswalk.drop_duplicates(subset='OTTONEUID')
    crosswalk['OTTONEUID'] = crosswalk['OTTONEUID'].fillna('').apply(
        lambda x: str(int(float(x))) if x != '' else '').str.strip()
    crosswalk['IDFANGRAPHS'] = crosswalk['IDFANGRAPHS'].astype(str).str.strip()

    for df in [rosters, atc_hitting, atc_pitching]:
        df['fg_id'] = df['fg_id'].astype(str).str.strip()

    # Merge
    rosters_with_fgid = rosters.merge(
        crosswalk[['OTTONEUID', 'IDFANGRAPHS']],
        left_on='fg_id', right_on='OTTONEUID', how='left'
    )

    roster_hitters = rosters_with_fgid[rosters_with_fgid['player_type'] == 'hitters']
    roster_pitchers = rosters_with_fgid[rosters_with_fgid['player_type'] == 'pitchers']

    hitters_merged = roster_hitters.merge(
        atc_hitting[['fg_id', 'Name', 'FPTS']], left_on='IDFANGRAPHS', right_on='fg_id', how='left')
    pitchers_merged = roster_pitchers.merge(
        atc_pitching[['fg_id', 'Name', 'FPTS']], left_on='IDFANGRAPHS', right_on='fg_id', how='left')

    all_players = pd.concat([hitters_merged, pitchers_merged], ignore_index=True)
    all_players['FPTS'] = all_players['FPTS'].fillna(0)

    # Ohtani fix
    ohtani_hitting_fpts = atc_hitting[
        atc_hitting['Name'].str.contains('Ohtani', case=False)]['FPTS'].values[0]
    ohtani_mask = all_players['player_name'].str.contains('Ohtani', case=False)
    all_players.loc[ohtani_mask, 'FPTS'] += ohtani_hitting_fpts

    # Fuzzy fallback
    missing_hitters = all_players[(all_players['FPTS'] == 0) & (all_players['player_type'] == 'hitters')]
    missing_pitchers = all_players[(all_players['FPTS'] == 0) & (all_players['player_type'] == 'pitchers')]

    fuzzy_h = fuzzy_match_players(missing_hitters, atc_hitting)
    fuzzy_p = fuzzy_match_players(missing_pitchers, atc_pitching)

    for _, row in pd.concat([fuzzy_h, fuzzy_p]).iterrows():
        if row['FPTS'] > 0:
            mask = ((all_players['player_name'] == row['player_name']) &
                    (all_players['team_name'] == row['team_name']))
            all_players.loc[mask, 'FPTS'] = row['FPTS']

    # Optimize lineups
    team_results = []
    lineup_assignments = {}
    slot_lookup = {}

    for team_name, team_df in all_players.groupby('team_name'):
        total_fpts, lineup = optimize_lineup(team_df)
        lineup_assignments[team_name] = set(lineup['player_name'].tolist()) if not lineup.empty else set()
        # Store slot assignments for positional breakdown
        if not lineup.empty:
            for _, row in lineup.iterrows():
                slot_lookup[(team_name, row['player_name'])] = row['slot']
        team_results.append({
            'Rank': 0,
            'Team': team_name,
            'Projected FPTS': round(total_fpts, 1),
            'Salary': team_df['salary'].sum(),
            'Players': len(team_df),
        })

    standings = pd.DataFrame(team_results)
    standings = standings.sort_values('Projected FPTS', ascending=False).reset_index(drop=True)
    standings['Rank'] = standings.index + 1
    standings = standings[['Rank', 'Team', 'Projected FPTS', 'Salary', 'Players']]

    # Tag starters/bench in player df
    all_players['starter'] = all_players.apply(
        lambda r: r['player_name'] in lineup_assignments.get(r['team_name'], set()), axis=1)

    # Tag slot assignments
    all_players['slot'] = all_players.apply(
        lambda r: slot_lookup.get((r['team_name'], r['player_name']), 'Bench'), axis=1)
    # Identify free agents - players in projections but not on any roster
    rostered_fg_ids = set(rosters_with_fgid['IDFANGRAPHS'].dropna().tolist())
    rostered_names = set(all_players['player_name'].str.lower().tolist())

    # Scrape accurate position eligibility from FanGraphs API
    print("Fetching FA positions from FanGraphs API...")
    fa_pos_map = get_fa_positions()

    # Hitting free agents — filter by fg_id AND name to catch crosswalk gaps
    fa_hitting = atc_hitting[
        (~atc_hitting['fg_id'].isin(rostered_fg_ids)) &
        (~atc_hitting['Name'].str.lower().isin(rostered_names))
    ].copy()
    fa_hitting['player_type'] = 'hitters'
    fa_hitting['position'] = fa_hitting['fg_id'].map(fa_pos_map)

    # Crosswalk fallback for any hitter missing from FanGraphs API (e.g. Burger)
    crosswalk_pos_map = crosswalk.set_index('IDFANGRAPHS')['POS'].to_dict()
    missing_pos = fa_hitting['position'].isna()
    fa_hitting.loc[missing_pos, 'position'] = fa_hitting.loc[missing_pos, 'fg_id'].map(crosswalk_pos_map)
    fa_hitting['position'] = fa_hitting['position'].fillna('')

    # Pitching free agents — filter by fg_id AND name to catch crosswalk gaps
    fa_pitching = atc_pitching[
        (~atc_pitching['fg_id'].isin(rostered_fg_ids)) &
        (~atc_pitching['Name'].str.lower().isin(rostered_names))
    ].copy()
    fa_pitching['player_type'] = 'pitchers'
    fa_pitching['position'] = fa_pitching['fg_id'].map(fa_pos_map).fillna('SP/RP')

    # Combine and clean
    free_agents = pd.concat([fa_hitting, fa_pitching], ignore_index=True)
    free_agents = free_agents[free_agents['FPTS'] > 0].copy()
    free_agents = free_agents.rename(columns={'Name': 'player_name'})
    free_agents = free_agents[['player_name', 'fg_id', 'FPTS', 'player_type', 'position']]
    free_agents = free_agents.sort_values('FPTS', ascending=False).reset_index(drop=True)

    return all_players, standings, free_agents, rosters_with_fgid, crosswalk, atc_hitting, atc_pitching, roster_source, proj_source, datetime.now(), projection_system



# ── Sidebar (projection toggle must be declared BEFORE data load) ─────────────
with st.sidebar:
    st.markdown("## ⚾ Farva Operations Center")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Standings",
        "Team Detail",
        "Positional Breakdown",
        "Free Agent Targets",
        "Player Search",
        "Head to Head",
        "Pitching Report",
    ])
    st.markdown("---")

    # Projection system toggle
    proj_system = st.selectbox(
        "Projection System",
        ["OOPSY", "ATC", "THE BAT X", "OOPSY DC", "ATC DC", "THE BAT X DC"],
        index=0,
        help="OOPSY/ATC/THE BAT X = preseason. DC variants = in-season (unlocks Opening Day)."
    )

    st.markdown("---")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown('<p class="metric-label">Data Freshness</p>', unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading league data..."):
    all_players, standings, free_agents, rosters_with_fgid, crosswalk, atc_hitting, atc_pitching, roster_source, proj_source, last_updated, active_proj_system = load_all_data(proj_system)

# ── Sidebar (data-dependent widgets rendered after load) ──────────────────────
with st.sidebar:
    # Roster freshness
    timestamp_path = os.path.join(DATA_DIR, 'roster_scrape_timestamp.txt')
    try:
        with open(timestamp_path, 'r') as f:
            roster_ts = f.read().strip()
        if roster_source == 'cached':
            st.markdown(f'<p style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #f59e0b;">⚠️ Rosters (cached)<br>{roster_ts}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #64748b;">✓ Rosters (live)<br>{roster_ts}</p>', unsafe_allow_html=True)
    except:
        st.markdown('<p style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #64748b;">Rosters: unknown</p>', unsafe_allow_html=True)

    # Projection freshness — reflects whichever system is active
    try:
        proj_files = PROJECTION_FILES[active_proj_system]
        proj_path = os.path.join(DATA_DIR, proj_files['hitting'])
        proj_ts = datetime.fromtimestamp(os.path.getmtime(proj_path)).strftime('%Y-%m-%d')
        if proj_source == 'cached':
            st.markdown(f'<p style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #f59e0b;">⚠️ Projections ({active_proj_system}, cached)<br>{proj_ts}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #64748b;">✓ Projections ({active_proj_system}, live)<br>{proj_ts}</p>', unsafe_allow_html=True)
    except:
        st.markdown('<p style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #64748b;">Projections: unknown</p>', unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

# ── 1. Standings ──────────────────────────────────────────────────────────────
if page == "Standings":
    st.markdown("# Projected Standings")
    st.markdown(f"""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #64748b; margin-bottom: 16px;">
    All data based on 2026 {active_proj_system} preseason projections, with team-level totals optimized across starters vs. bench. Don't take data at face value — may be lurking inaccuracies.
    </p>
    """, unsafe_allow_html=True)
    st.markdown(f'<p class="last-updated">Last updated: {last_updated.strftime("%b %d, %Y %I:%M %p")}</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    top_team = standings.iloc[0]
    avg_fpts = standings['Projected FPTS'].mean()
    spread = standings['Projected FPTS'].iloc[0] - standings['Projected FPTS'].iloc[-1]

    with col1:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Projected Leader</div>
            <div class="metric-value" style="font-size:18px">{top_team["Team"]}</div>
        </div>''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">League Avg FPTS</div>
            <div class="metric-value">{avg_fpts:,.0f}</div>
        </div>''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">1st to Last Gap</div>
            <div class="metric-value">{spread:,.0f}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Full Standings</p>', unsafe_allow_html=True)
    st.dataframe(standings, width='stretch', hide_index=True, height=458)


# ── 2. Team Detail ────────────────────────────────────────────────────────────
elif page == "Team Detail":
    st.markdown("# Team Detail")

    team_names = sorted(all_players['team_name'].unique())
    default_team_idx = team_names.index('Large Farva') if 'Large Farva' in team_names else 0
    selected_team = st.selectbox("Select Team", team_names, index=default_team_idx)

    team_data = all_players[all_players['team_name'] == selected_team].copy()
    team_standing = standings[standings['Team'] == selected_team].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Projected Rank</div>
            <div class="metric-value">#{int(team_standing["Rank"])}</div>
        </div>''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Projected FPTS</div>
            <div class="metric-value">{team_standing["Projected FPTS"]:,.0f}</div>
        </div>''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Total Salary</div>
            <div class="metric-value">${int(team_standing["Salary"])}</div>
        </div>''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Roster Size</div>
            <div class="metric-value">{int(team_standing["Players"])}</div>
        </div>''', unsafe_allow_html=True)

    # Starters
    st.markdown('<p class="section-header">Starting Lineup</p>', unsafe_allow_html=True)
    starters = team_data[team_data['starter'] == True][
        ['player_name', 'position', 'salary', 'FPTS']
    ].sort_values('FPTS', ascending=False).rename(columns={
        'player_name': 'Player', 'position': 'POS', 'salary': 'Salary', 'FPTS': 'Proj FPTS'
    })
    starters['Proj FPTS'] = starters['Proj FPTS'].round(1)
    st.dataframe(starters, width='stretch', hide_index=True)

    # Bench
    st.markdown('<p class="section-header">Bench</p>', unsafe_allow_html=True)
    bench = team_data[team_data['starter'] == False][
        ['player_name', 'position', 'salary', 'FPTS']
    ].sort_values('FPTS', ascending=False).rename(columns={
        'player_name': 'Player', 'position': 'POS', 'salary': 'Salary', 'FPTS': 'Proj FPTS'
    })
    bench['Proj FPTS'] = bench['Proj FPTS'].round(1)
    st.dataframe(bench, width='stretch', hide_index=True)

# ── 3. Positional Breakdown ───────────────────────────────────────────────────
elif page == "Positional Breakdown":
    st.markdown("# Positional Breakdown")
    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #64748b; margin-bottom: 16px;">
    ⚠️ Beware Ohtani, who masquerades as all SP points — his hitting projection is baked into his pitching row.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("Projected FPTS by position slot for each team's optimized lineup.")

    # Define slot groups and display order
    slot_order = ['C', '1B', '2B', 'SS', '3B', 'MI', 
                  'OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'Util',
                  'SP1', 'SP2', 'SP3', 'SP4', 'SP5',
                  'RP1', 'RP2', 'RP3', 'RP4', 'RP5',
                  'Bench']

    # Build breakdown table
    rows = []
    for team_name, team_df in all_players.groupby('team_name'):
        team_standing = standings[standings['Team'] == team_name].iloc[0]
        row = {'Team': team_name, 'Rank': int(team_standing['Rank'])}
        
        # Starters by slot
        for slot in slot_order[:-1]:  # exclude Bench
            slot_players = team_df[team_df['slot'] == slot]
            row[slot] = round(slot_players['FPTS'].sum(), 0) if len(slot_players) > 0 else 0
        
        # Bench total
        bench_players = team_df[team_df['slot'] == 'Bench']
        row['Bench'] = round(bench_players['FPTS'].sum(), 0)
        
        rows.append(row)

    breakdown_df = pd.DataFrame(rows)
    breakdown_df = breakdown_df.sort_values('Rank').reset_index(drop=True)
    
    # Reorder columns
    cols = ['Rank', 'Team'] + slot_order
    breakdown_df = breakdown_df[cols]

    st.dataframe(breakdown_df, width='stretch', hide_index=True, height=458)

    st.markdown('<p class="section-header">Position Group Totals</p>', unsafe_allow_html=True)

    # Summarized version grouping by position type
    group_rows = []
    for team_name, team_df in all_players.groupby('team_name'):
        team_standing = standings[standings['Team'] == team_name].iloc[0]
        
        catcher = team_df[team_df['slot'] == 'C']['FPTS'].sum()
        infield = team_df[team_df['slot'].isin(['1B','2B','SS','3B','MI'])]['FPTS'].sum()
        util = team_df[team_df['slot'] == 'Util']['FPTS'].sum()
        outfield = team_df[team_df['slot'].isin(['OF1','OF2','OF3','OF4','OF5'])]['FPTS'].sum()
        sp = team_df[team_df['slot'].isin(['SP1','SP2','SP3','SP4','SP5'])]['FPTS'].sum()
        rp = team_df[team_df['slot'].isin(['RP1','RP2','RP3','RP4','RP5'])]['FPTS'].sum()
        bench = team_df[team_df['slot'] == 'Bench']['FPTS'].sum()

        util_players = team_df[team_df['slot'] == 'Util']
        util_to_infield = util_players[
            util_players['position'].str.contains('1B|2B|SS|3B', na=False)]['FPTS'].sum()
        util_to_c = util_players[
            util_players['position'].str.contains('C', na=False) &
            ~util_players['position'].str.contains('1B|2B|SS|3B|OF', na=False)]['FPTS'].sum()
        util_to_of = util_players[
            util_players['position'].str.contains('OF', na=False) &
            ~util_players['position'].str.contains('1B|2B|SS|3B', na=False)]['FPTS'].sum()

        group_rows.append({
            'Rank': int(team_standing['Rank']),
            'Team': team_name,
            'C': round(catcher + util_to_c, 0),
            'Infield': round(infield + util_to_infield, 0),
            'Outfield': round(outfield + util_to_of, 0),
            'SP': round(sp, 0),
            'RP': round(rp, 0),
            'Bench': round(bench, 0),
        })

    group_df = pd.DataFrame(group_rows).sort_values('Rank').reset_index(drop=True)
    st.dataframe(group_df, width='stretch', hide_index=True, height=458)

# ── 4. Free Agent Targets ─────────────────────────────────────────────────────
elif page == "Free Agent Targets":
    st.markdown("# Free Agent Targets")
    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #64748b; margin-bottom: 16px;">
    Unrostered players ranked by projected FPTS. Section 1 shows the best available upgrade at each position vs your current roster depth.
    </p>
    """, unsafe_allow_html=True)

    # Team selector
    team_names = sorted(all_players['team_name'].unique())
    default_idx = team_names.index('Large Farva') if 'Large Farva' in team_names else 0
    selected_team = st.selectbox("Compare against team", team_names, index=default_idx)

    # Get selected team's roster
    my_roster = all_players[all_players['team_name'] == selected_team].copy()

    # Position slot definitions for comparison
    hitting_positions = ['C', '1B', '2B', 'SS', '3B', 'OF']
    pitching_positions = ['SP', 'RP']

    def get_weakest_at_position(roster_df, pos):
        """Get weakest rostered player at a given position."""
        if pos == 'OF':
            eligible = roster_df[roster_df['position'].str.contains('OF', na=False)]
        elif pos in ['SP', 'RP']:
            eligible = roster_df[roster_df['position'].str.contains(pos, na=False)]
        else:
            eligible = roster_df[roster_df['position'].str.contains(f'(?<![A-Z]){pos}(?![A-Z])', na=False, regex=True)]
        
        if len(eligible) == 0:
            return None, 0
        
        weakest = eligible.nsmallest(1, 'FPTS').iloc[0]
        return weakest['player_name'], weakest['FPTS']

    def get_best_fa_at_position(fa_df, pos):
        """Get best free agent at a given position."""
        if pos == 'OF':
            eligible = fa_df[fa_df['position'].str.contains('OF', na=False)]
        elif pos in ['SP', 'RP']:
            eligible = fa_df[fa_df['position'].str.contains(pos, na=False)]
        else:
            eligible = fa_df[fa_df['position'].str.contains(f'(?<![A-Z]){pos}(?![A-Z])', na=False, regex=True)]
        
        if len(eligible) == 0:
            return None, 0
        
        best = eligible.nlargest(1, 'FPTS').iloc[0]
        return best['player_name'], best['FPTS']

    # free_agents already has accurate positions scraped from FanGraphs
    free_agents_full = free_agents.copy()

    # ── Section 1: Best Available by Position ─────────────────────────────────
    st.markdown('<p class="section-header">Best Available by Position</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #64748b;">Comparing against full roster (starters + bench) for {selected_team}</p>', unsafe_allow_html=True)

    upgrade_rows = []
    for pos in hitting_positions + pitching_positions:
        best_fa_name, best_fa_fpts = get_best_fa_at_position(free_agents_full, pos)
        weakest_name, weakest_fpts = get_weakest_at_position(my_roster, pos)

        if best_fa_name is None:
            continue

        gain = round(best_fa_fpts - weakest_fpts, 1)
        upgrade_rows.append({
            'POS': pos,
            'Best Available FA': best_fa_name,
            'FA FPTS': round(best_fa_fpts, 1),
            'Weakest Rostered': weakest_name or 'None',
            'Rostered FPTS': round(weakest_fpts, 1),
            'Gain': gain
        })

    upgrade_df = pd.DataFrame(upgrade_rows)
    upgrade_df['FA FPTS'] = upgrade_df['FA FPTS'].apply(lambda x: f"{float(x):.1f}")
    upgrade_df['Rostered FPTS'] = upgrade_df['Rostered FPTS'].apply(lambda x: f"{float(x):.1f}")
    upgrade_df['Gain'] = upgrade_df['Gain'].apply(lambda x: f"{float(x):.1f}")
    
    def color_gain(val):
        try:
            if float(val) > 0:
                return 'color: #22c55e'
            elif float(val) < 0:
                return 'color: #ef4444'
        except:
            pass
        return ''

    st.dataframe(
        upgrade_df.style.applymap(color_gain, subset=['Gain']),
        width='stretch',
        hide_index=True
    )

    # ── Section 2: Full Free Agent List ──────────────────────────────────────
    st.markdown('<p class="section-header">Full Free Agent List</p>', unsafe_allow_html=True)

    all_positions_filter = ['All', 'C', '1B', '2B', 'SS', '3B', 'OF', 'SP', 'RP']
    selected_pos = st.selectbox("Filter by Position", all_positions_filter)

    fa_display = free_agents_full.copy()

    if selected_pos != 'All':
        if selected_pos in ['SP', 'RP']:
            fa_display = fa_display[fa_display['player_type'] == 'pitchers']
            fa_display = fa_display[fa_display['position'].str.contains(selected_pos, na=False)]
        else:
            fa_display = fa_display[fa_display['player_type'] == 'hitters']
            fa_display = fa_display[fa_display['position'].str.contains(
                f'(?<![A-Z]){selected_pos}(?![A-Z])', na=False, regex=True)]

    fa_display = fa_display.sort_values('FPTS', ascending=False).head(50)[[
        'player_name', 'position', 'FPTS'
    ]].rename(columns={
        'player_name': 'Player',
        'position': 'POS',
        'FPTS': 'Proj FPTS'
    })
    fa_display['Proj FPTS'] = fa_display['Proj FPTS'].round(1)

    st.dataframe(fa_display, width='stretch', hide_index=True)

# ── 5. Player Search ──────────────────────────────────────────────────────────
elif page == "Player Search":
    st.markdown("# Player Search")

    search = st.text_input("Search by player name")

    if search:
        results = all_players[
            all_players['player_name'].str.contains(search, case=False, na=False)
        ][['player_name', 'team_name', 'position', 'salary', 'FPTS', 'starter']].copy()

        results['starter'] = results['starter'].map({True: 'Starter', False: 'Bench'})
        results['FPTS'] = results['FPTS'].round(1)
        results = results.rename(columns={
            'player_name': 'Player', 'team_name': 'Team', 'position': 'POS',
            'salary': 'Salary', 'FPTS': 'Proj FPTS', 'starter': 'Status'
        })

        st.dataframe(results, width='stretch', hide_index=True)
    else:
        st.markdown('<p style="color:#64748b; font-family: IBM Plex Mono, monospace; font-size:13px;">Type a player name to search...</p>',
                    unsafe_allow_html=True)


# ── 6. Head to Head ───────────────────────────────────────────────────────────
elif page == "Head to Head":
    st.markdown("# Head to Head")

    team_names = sorted(all_players['team_name'].unique())
    col1, col2 = st.columns(2)
    with col1:
        default_idx = team_names.index('Large Farva') if 'Large Farva' in team_names else 0
        team_a = st.selectbox("Team A", team_names, index=default_idx)
    with col2:
        team_b = st.selectbox("Team B", team_names, index=1)

    if team_a == team_b:
        st.warning("Select two different teams.")
    else:
        # FPTS comparison
        fpts_a = standings[standings['Team'] == team_a]['Projected FPTS'].values[0]
        fpts_b = standings[standings['Team'] == team_b]['Projected FPTS'].values[0]
        rank_a = standings[standings['Team'] == team_a]['Rank'].values[0]
        rank_b = standings[standings['Team'] == team_b]['Rank'].values[0]
        sal_a = standings[standings['Team'] == team_a]['Salary'].values[0]
        sal_b = standings[standings['Team'] == team_b]['Salary'].values[0]

        st.markdown('<p class="section-header">Comparison</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-label">{team_a}</div>
                <div class="metric-value">{fpts_a:,.0f}</div>
                <div class="metric-label">Rank #{int(rank_a)} · ${int(sal_a)}</div>
            </div>''', unsafe_allow_html=True)
        with c2:
            diff = abs(fpts_a - fpts_b)
            leader = team_a if fpts_a > fpts_b else team_b
            st.markdown(f'''<div class="metric-card">
                <div class="metric-label">Gap</div>
                <div class="metric-value" style="font-size:20px">{diff:,.0f}</div>
                <div class="metric-label">Edge: {leader.split()[0]}</div>
            </div>''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-label">{team_b}</div>
                <div class="metric-value">{fpts_b:,.0f}</div>
                <div class="metric-label">Rank #{int(rank_b)} · ${int(sal_b)}</div>
            </div>''', unsafe_allow_html=True)

        # Side by side rosters
        st.markdown('<p class="section-header">Rosters Side by Side</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        for col, team in [(col_a, team_a), (col_b, team_b)]:
            with col:
                st.markdown(f"**{team}**")
                roster = all_players[all_players['team_name'] == team][
                    ['player_name', 'position', 'salary', 'FPTS', 'starter']
                ].copy()
                roster['starter'] = roster['starter'].map({True: '▶', False: '—'})
                roster['FPTS'] = roster['FPTS'].round(1)
                roster = roster.sort_values('FPTS', ascending=False).rename(columns={
                    'player_name': 'Player', 'position': 'POS',
                    'salary': '$', 'FPTS': 'FPTS', 'starter': 'S'
                })
                st.dataframe(roster, width='stretch', hide_index=True)

# ── 7. Pitching Report ────────────────────────────────────────────────────────
elif page == "Pitching Report":
    st.markdown("# Pitching Report")
    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #64748b; margin-bottom: 16px;">
    Weekly schedule, matchup grades, and season stats for your SP-eligible pitchers.
    Pitcher stats refresh daily. Opponent team rankings refresh weekly (slow — ~2 min on first load).
    </p>
    """, unsafe_allow_html=True)

    # ── Cached data loaders ───────────────────────────────────────────────────
    @st.cache_data(ttl=604800, show_spinner="Loading team stats (this takes ~2 min on first load)...")
    def load_team_stats():
        return get_all_team_stats()

    @st.cache_data(ttl=86400, show_spinner="Loading pitcher data...")
    def load_pitching_report(pitcher_names_tuple):
        team_stats = load_team_stats()
        return get_pitching_report(list(pitcher_names_tuple), team_stats)

    # ── Get Large Farva's SP-eligible pitchers from all_players ───────────────
    my_pitchers = all_players[
        (all_players['team_name'] == 'Large Farva') &
        (all_players['player_type'] == 'pitchers') &
        (all_players['position'].str.contains('SP', na=False))
    ]['player_name'].tolist()

    if not my_pitchers:
        st.warning("No SP-eligible pitchers found on Large Farva's roster.")
    else:
        # Pass as tuple so it's hashable for cache key
        report_data = load_pitching_report(tuple(sorted(my_pitchers)))

        GRADE_COLORS = {'A': '#22c55e', 'B': '#86efac', 'C': '#94a3b8', 'D': '#f97316', 'F': '#ef4444'}
        GRADE_ORDER = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4, '—': 5}

        # ── Summary ranking table ─────────────────────────────────────────────
        st.markdown('<p class="section-header">This Week\'s Starter Rankings</p>', unsafe_allow_html=True)
        summary_rows = []
        for pitcher in report_data:
            matchups = pitcher['matchups']
            games = len(matchups)
            if matchups:
                best_grade = min([m['grade'] for m in matchups], key=lambda g: GRADE_ORDER.get(g, 5))
                opponents = ', '.join([f"{m['opponent']} ({m['home_away'][0]})" for m in matchups])
            else:
                best_grade = '—'
                opponents = 'No games'
            summary_rows.append({
                'Pitcher': pitcher['name'],
                'Games': games,
                'Best Grade': best_grade,
                'Matchups': opponents,
            })

        summary_rows.sort(key=lambda r: (GRADE_ORDER.get(r['Best Grade'], 5), -r['Games']))
        summary_df = pd.DataFrame(summary_rows)

        # Color-code the Best Grade column
        def style_grade(val):
            color = GRADE_COLORS.get(val, '#94a3b8')
            return f'color: {color}; font-weight: 600;'

        st.dataframe(
            summary_df.style.applymap(style_grade, subset=['Best Grade']),
            hide_index=True,
            width='stretch'
        )
        st.markdown("---")

        for pitcher in report_data:
            name = pitcher['name']
            stats = pitcher['stats']
            schedule = pitcher['schedule']
            rotation = pitcher['rotation']
            matchups = pitcher['matchups']

            st.markdown(f'<p class="section-header">{name}</p>', unsafe_allow_html=True)

            # ── Top metrics row ───────────────────────────────────────────────
            col1, col2, col3, col4, col5 = st.columns(5)
            metrics = [
                (col1, "IP", f"{stats.get('IP', '—')}"),
                (col2, "K%", f"{stats.get('K_percent', '—')}%"),
                (col3, "BB%", f"{stats.get('BB_percent', '—')}%"),
                (col4, "wOBA Against", f"{stats.get('wOBA_against', '—')}"),
                (col5, "Hard Hit%", f"{stats.get('hard_hit_percent', '—')}%"),
            ]
            for col, label, val in metrics:
                with col:
                    st.markdown(f'''<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="font-size:20px">{val}</div>
                    </div>''', unsafe_allow_html=True)

            st.markdown("")

            # ── Rotation info ─────────────────────────────────────────────────
            if rotation.get('is_starter'):
                last = rotation.get('last_start')
                nxt = rotation.get('next_predicted_start')
                gap = rotation.get('avg_days_between_starts')
                parts = []
                if last:
                    parts.append(f"Last start: **{last.strftime('%b %d')}**")
                if gap:
                    parts.append(f"Avg rest: **{gap} days**")
                if nxt:
                    parts.append(f"Next predicted: **{nxt.strftime('%b %d')}**")
                if parts:
                    st.markdown(f'<p style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #64748b;">{" · ".join(parts)}</p>', unsafe_allow_html=True)

            # ── Home/Away splits ──────────────────────────────────────────────
            home_s = stats.get('home_splits', {})
            away_s = stats.get('away_splits', {})
            if home_s or away_s:
                st.markdown('<p class="section-header">Home / Away Splits</p>', unsafe_allow_html=True)
                splits_data = {
                    'Split': ['Home', 'Away'],
                    'IP': [home_s.get('innings', '—'), away_s.get('innings', '—')],
                    'K%': [f"{home_s.get('K_percent', '—')}%", f"{away_s.get('K_percent', '—')}%"],
                    'BB%': [f"{home_s.get('BB_percent', '—')}%", f"{away_s.get('BB_percent', '—')}%"],
                    'wOBA Against': [home_s.get('wOBA_against', '—'), away_s.get('wOBA_against', '—')],
                }
                st.dataframe(pd.DataFrame(splits_data), hide_index=True, width='stretch')

            # ── This week's matchups ──────────────────────────────────────────
            if matchups:
                st.markdown('<p class="section-header">This Week\'s Matchups</p>', unsafe_allow_html=True)
                for m in matchups:
                    grade = m['grade']
                    grade_color = GRADE_COLORS.get(grade, '#94a3b8')
                    opp_stats = m['opp_stats']
                    rankings = m['rankings']

                    mc1, mc2 = st.columns([1, 3])
                    with mc1:
                        st.markdown(f'''<div class="metric-card" style="text-align:center">
                            <div class="metric-label">{m["date"]} · {m["home_away"]}</div>
                            <div style="font-family: IBM Plex Mono, monospace; font-size: 32px; font-weight: 600; color: {grade_color};">{grade}</div>
                            <div class="metric-label">vs {m["opponent"]}</div>
                        </div>''', unsafe_allow_html=True)
                    with mc2:
                        if opp_stats:
                            opp_rows = []
                            stat_labels = {'OPS': 'OPS', 'wOBA': 'wOBA', 'K_percent': 'K%', 'HR_rate': 'HR%'}
                            for stat_key, stat_label in stat_labels.items():
                                val = opp_stats.get(stat_key, '—')
                                rank_info = rankings.get(stat_key, {})
                                rank_str = f"#{rank_info['rank']}/{rank_info['total']}" if rank_info else '—'
                                opp_rows.append({'Stat': stat_label, f'{m["opponent"]} ({m["home_away"]})': val, 'Rank': rank_str})
                            st.dataframe(pd.DataFrame(opp_rows), hide_index=True, width='stretch')
                        else:
                            st.markdown('<p style="color:#64748b; font-family: IBM Plex Mono, monospace; font-size: 12px;">No opponent stats available yet (season not started)</p>', unsafe_allow_html=True)

                    st.markdown("")
            elif schedule:
                st.markdown(f'<p style="color:#64748b; font-family: IBM Plex Mono, monospace; font-size: 12px;">No games this week for {schedule.get("team_name", name)}\'s team.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#64748b; font-family: IBM Plex Mono, monospace; font-size: 12px;">Could not load schedule.</p>', unsafe_allow_html=True)

            st.markdown("---")