import pandas as pd
import numpy as np
import json

# Carga y preparación de los datos
df = pd.read_csv('data/datasheetv3.csv')

def glicko_update(R_winner, R_loser, RD_winner, RD_loser, K=64, RD_reduction_factor=0.97, q=np.log(10)/400):
    def g_RD(RD):
        return 1 / np.sqrt(1 + 3 * q**2 * RD**2 / np.pi**2)
    
    # Calculate the glicko parameters
    g_RD_winner = g_RD(RD_winner)
    g_RD_loser = g_RD(RD_loser)
    
    E_winner = 1 / (1 + 10 ** (g_RD_loser * (R_loser - R_winner) / 400))
    E_loser = 1 / (1 + 10 ** (g_RD_winner * (R_winner - R_loser) / 400))
    
    # Update the ratings based on the match outcome
    R_winner_updated = R_winner + K * (1 - E_winner)
    R_loser_updated = R_loser - K * E_loser
    
    # Update the rating deviations
    RD_winner_updated = max(RD_winner * RD_reduction_factor, 30)
    RD_loser_updated = max(RD_loser * RD_reduction_factor, 30)
    
    return R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated

def calculate_player_glicko_ratings(df):
    # Extract unique player IDs
    unique_player_ids = set(df[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID', 'Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']].melt()['value'].dropna().unique())
    
    # Initialize player ratings and rating deviations
    player_glicko = {int(player_id): 1500 for player_id in unique_player_ids}
    player_RD = {int(player_id): 350 for player_id in unique_player_ids}
    
    # Iterate over each match in the dataset
    for _, row in df.iterrows():
        equipo_ganador = str(row['TeamWinner'])
        equipo_perdedor = '2' if equipo_ganador == '1' else '1'
        
        # Update ratings for each position
        for pos in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            id_ganador = row[pos + equipo_ganador + 'ID']
            id_perdedor = row[pos + equipo_perdedor + 'ID']

            if pd.notnull(id_ganador) and pd.notnull(id_perdedor):
                id_ganador = int(id_ganador)
                id_perdedor = int(id_perdedor)

                # Obtain current ratings and RD
                R_winner, RD_winner = player_glicko[id_ganador], player_RD[id_ganador]
                R_loser, RD_loser = player_glicko[id_perdedor], player_RD[id_perdedor]

                # Update ratings and RD using the glicko_update function
                R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated = glicko_update(
                    R_winner, R_loser, RD_winner, RD_loser)

                # Update Glicko ratings and RD in the dictionaries
                player_glicko[id_ganador], player_glicko[id_perdedor] = R_winner_updated, R_loser_updated
                player_RD[id_ganador], player_RD[id_perdedor] = RD_winner_updated, RD_loser_updated

    # Save the results to a JSON file
    with open('player_glicko_ratings.json', 'w') as f:
        json.dump({"player_glicko": player_glicko, "player_RD": player_RD}, f, indent=4)

    return player_glicko, player_RD

# Example usage of the function
player_glicko_ratings, player_RD = calculate_player_glicko_ratings(df)
