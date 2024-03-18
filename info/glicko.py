import pandas as pd
import numpy as np
import json

# Simplified Glicko update function for demonstration
def glicko_update(R_winner, R_loser, RD_winner, RD_loser, K=38):
    """
    Actualiza las clasificaciones Glicko para el ganador y el perdedor de un encuentro, teniendo en cuenta RD.
    """
    q = np.log(10) / 400
    g_RD = lambda RD: 1 / np.sqrt(1 + 3 * (q**2) * (RD**2) / (np.pi**2))
    
    E_winner = 1 / (1 + 10 ** (g_RD(RD_loser) * (R_loser - R_winner) / 400))
    E_loser = 1 / (1 + 10 ** (g_RD(RD_winner) * (R_winner - R_loser) / 400))
    
    R_winner_updated = R_winner + K * (1 - E_winner)
    R_loser_updated = R_loser - K * E_loser  # Ajuste para la coherencia con la pérdida
    
    return R_winner_updated, R_loser_updated


def calculate_all_glicko_ratings(datasheet_path):
    df = pd.read_csv(datasheet_path)
    
    # Initialize Glicko ratings and RD
    team_glicko = {team_id: 500 for team_id in pd.concat([df['Team1ID'], df['Team2ID']]).unique()}
    team_RD = {team_id: 350 for team_id in pd.concat([df['Team1ID'], df['Team2ID']]).unique()}
    
    for index, row in df.iterrows():
        winner_id = row['Team1ID'] if row['TeamWinner'] == 1 else row['Team2ID']
        loser_id = row['Team2ID'] if row['TeamWinner'] == 1 else row['Team1ID']
        
        R_winner, R_loser = team_glicko[winner_id], team_glicko[loser_id]
        RD_winner, RD_loser = team_RD[winner_id], team_RD[loser_id]
        
        R_winner_updated, R_loser_updated = glicko_update(R_winner, R_loser, RD_winner, RD_loser)
        
        team_glicko[winner_id] = R_winner_updated
        team_glicko[loser_id] = R_loser_updated
    
    return team_glicko


# Assuming datasheet_path is defined and points to your match data CSV
team_glicko_ratings = calculate_all_glicko_ratings('data/datasheetv2.csv')
team_glicko_ratings_str_keys = {str(key): value for key, value in team_glicko_ratings.items()}

# Write Glicko ratings to a JSON file
with open('team_glicko_ratings.json', 'w') as f:
    json.dump(team_glicko_ratings_str_keys, f, indent=4)
