import pandas as pd
import numpy as np
import json

df = pd.read_csv('data/datasheetv3.csv')

def glicko_update(R_winner, R_loser, RD_winner, RD_loser, K=64, RD_reduction_factor=0.97, q=np.log(10)/400):
    def g_RD(RD):
        return 1 / np.sqrt(1 + 3 * q**2 * RD**2 / np.pi**2)
    
    g_RD_winner = g_RD(RD_winner)
    g_RD_loser = g_RD(RD_loser)
    
    E_winner = 1 / (1 + 10 ** (g_RD_loser * (R_loser - R_winner) / 400))
    E_loser = 1 / (1 + 10 ** (g_RD_winner * (R_winner - R_loser) / 400))
    
    R_winner_updated = R_winner + K * (1 - E_winner)
    R_loser_updated = R_loser - K * E_loser
    
    RD_winner_updated = max(RD_winner * RD_reduction_factor, 30)
    RD_loser_updated = max(RD_loser * RD_reduction_factor, 30)
    
    return R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated

def calculate_player_glicko_ratings(df):
    unique_player_ids = set(df[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID', 'Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']].melt()['value'].dropna().unique())
    
    player_glicko = {int(player_id): 1500 for player_id in unique_player_ids}
    player_RD = {int(player_id): 350 for player_id in unique_player_ids}
    
    for _, row in df.iterrows():
        winner_team = str(row['TeamWinner'])
        loser_team = '2' if winner_team == '1' else '1'
        
        for pos in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            winner_id = row[pos + winner_team + 'ID']
            loser_id = row[pos + loser_team + 'ID']

            if pd.notnull(winner_id) and pd.notnull(loser_id):
                winner_id = int(winner_id)
                loser_id = int(loser_id)

                R_winner, RD_winner = player_glicko[winner_id], player_RD[winner_id]
                R_loser, RD_loser = player_glicko[loser_id], player_RD[loser_id]

                R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated = glicko_update(
                    R_winner, R_loser, RD_winner, RD_loser)

                player_glicko[winner_id], player_glicko[loser_id] = R_winner_updated, R_loser_updated
                player_RD[winner_id], player_RD[loser_id] = RD_winner_updated, RD_loser_updated

    with open('player_glicko_ratings.json', 'w') as f:
        json.dump({"player_glicko": player_glicko, "player_RD": player_RD}, f, indent=4)

    return player_glicko, player_RD

player_glicko_ratings, player_RD = calculate_player_glicko_ratings(df)
