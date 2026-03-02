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
st.write("App loaded")

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
    make_api_request, DATA_DIR, LEAGUE_ID
)

# ── Data Loading (cached weekly) ──────────────────────────────────────────────
@st.cache_data(ttl=604800)
def load_all_data():
    """Run the full pipeline and return projected players + standings."""
    # Rosters
    rosters = get_league_rosters()

    # Projections
    atc_hitting = pd.read_csv(os.path.join(DATA_DIR, 'fangraphs-leaderboard-projections_oopsy hitting 2026.csv'))
    atc_pitching = pd.read_csv(os.path.join(DATA_DIR, 'fangraphs-leaderboard-projections_oopsy pitching 2026.csv'))
    atc_hitting = atc_hitting.rename(columns={'PlayerId': 'fg_id'})
    atc_pitching = atc_pitching.rename(columns={'PlayerId': 'fg_id'})

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

    return all_players, standings, datetime.now()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ Farva Operations Center")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Standings",
        "Team Detail",
        "Positional Breakdown",
        "Player Search",
        "Head to Head"
    ])
    st.markdown("---")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading league data..."):
    all_players, standings, last_updated = load_all_data()

# ── Pages ─────────────────────────────────────────────────────────────────────

# ── 1. Standings ──────────────────────────────────────────────────────────────
if page == "Standings":
    st.markdown("# Projected Standings")
    st.markdown("""
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #64748b; margin-bottom: 16px;">
    All data based on 2026 OOPSY preseason, and team-level projections optimized to consider starters vs. bench players. Though, don't take data at face value - may be lurking inaccuracies.
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

# ── 4. Player Search ──────────────────────────────────────────────────────────
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


# ── 5. Head to Head ───────────────────────────────────────────────────────────
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