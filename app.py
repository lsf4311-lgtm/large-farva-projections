import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ottoneu War Room",
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

    for team_name, team_df in all_players.groupby('team_name'):
        total_fpts, lineup = optimize_lineup(team_df)
        lineup_assignments[team_name] = set(lineup['player_name'].tolist()) if not lineup.empty else set()
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

    return all_players, standings, datetime.now()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ War Room")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Standings",
        "Team Detail",
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
    st.dataframe(standings, width='stretch', hide_index=True)


# ── 2. Team Detail ────────────────────────────────────────────────────────────
elif page == "Team Detail":
    st.markdown("# Team Detail")

    team_names = sorted(all_players['team_name'].unique())
    selected_team = st.selectbox("Select Team", team_names)

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


# ── 3. Player Search ──────────────────────────────────────────────────────────
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


# ── 4. Head to Head ───────────────────────────────────────────────────────────
elif page == "Head to Head":
    st.markdown("# Head to Head")

    team_names = sorted(all_players['team_name'].unique())
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", team_names, index=0)
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