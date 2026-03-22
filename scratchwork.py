from league_analysis_final import fetch_projections
hitting, pitching = fetch_projections('OOPSY', 'matthewchoman@gmail.com', 'brettbotchomanager')
print(hitting.columns.tolist())
print(hitting[['Name', 'fg_id', 'FPTS']].head(10))