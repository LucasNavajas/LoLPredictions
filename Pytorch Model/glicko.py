import pandas as pd
import numpy as np
import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR,'data/datasheet.csv'))

def glicko_update(R_winner, R_loser, RD_winner, RD_loser, K=64, RD_reduction_factor=0.97, q=np.log(10)/400):
    """
    Performs a single Glicko-like rating update for a match between two players:
    one winner and one loser.

    This function applies some elements from Glicko (notably the g(RD) factor) 
    and combines them with an Elo-like update. The ratings (R_winner, R_loser) 
    and rating deviations (RD_winner, RD_loser) are each updated in place for 
    exactly one match result.

    Args:
        R_winner (float): The winner's current rating.
        R_loser (float): The loser's current rating.
        RD_winner (float): The winner's current rating deviation (RD).
        RD_loser (float): The loser's current rating deviation (RD).
        K (float, optional): The K-factor controlling the update magnitude. 
            Defaults to 64.
        RD_reduction_factor (float, optional): Factor by which to reduce the 
            rating deviation after each match, simulating that playing a match 
            reduces the uncertainty. Defaults to 0.97.
        q (float, optional): The Glicko scale constant, typically ln(10)/400. 
            Defaults to ln(10)/400.

    Returns:
        tuple: A tuple containing the updated ratings and rating deviations:
               (R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated).
    """

    # Adjust how effective the rating difference is based on players rating deviations RD. Higher RD, match less reliable
    def g_RD(RD):
        return 1 / np.sqrt(1 + 3 * q**2 * RD**2 / np.pi**2)
    
    g_RD_winner = g_RD(RD_winner)
    g_RD_loser = g_RD(RD_loser)
    
    # E is the probability that the winner and loser have each other of winning according to this Glicko implementation
    E_winner = 1 / (1 + 10 ** (g_RD_loser * (R_loser - R_winner) / 400))
    E_loser = 1 / (1 + 10 ** (g_RD_winner * (R_winner - R_loser) / 400))
    
    # Apply an Elo-style update with a K-factor but using the Glicko based expected score
    R_winner_updated = R_winner + K * (1 - E_winner)
    R_loser_updated = R_loser - K * E_loser
    
    # Reduce the rating deviation after each match and enforce a minimum of 30
    RD_winner_updated = max(RD_winner * RD_reduction_factor, 30)
    RD_loser_updated = max(RD_loser * RD_reduction_factor, 30)
    
    return R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated

def calculate_player_glicko_ratings(df):
    """
    Calculates and updates Glicko-like ratings for all players in the given DataFrame.

    The function extracts unique player IDs from each positional column
    (Top1ID, Jg1ID, Mid1ID, Adc1ID, Supp1ID, Top2ID, Jg2ID, Mid2ID, Adc2ID, Supp2ID),
    initializes their ratings (Glicko) and rating deviations (RD), iterates through
    each match (row), and applies the 'glicko_update' function to update ratings
    for winners and losers in each lane.

    Finally, it writes the resulting dictionaries of player ratings and RDs
    to a JSON file and returns them.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing match data. Must include:
            - 'Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID',
            - 'Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID',
            - 'TeamWinner' (indicating which team won the match).
    
    Returns:
        tuple: (player_glicko, player_RD)
            - player_glicko (dict): Maps player_id -> updated Glicko rating
            - player_RD (dict): Maps player_id -> updated rating deviation
    """
    # Collect all unique player IDs without picking NaNs and only unique values
    unique_player_ids = set(
        df[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID', 
            'Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']]
        .melt()['value']
        .dropna()
        .unique()
    )
    
    # Initialize each player rating as 1500 and RD as 350
    player_glicko = {int(player_id): 1500 for player_id in unique_player_ids}
    player_RD = {int(player_id): 350 for player_id in unique_player_ids}
    
    # Iterate over each match in the DataFrame
    for _, row in df.iterrows():
        # Determine the winner, where "1" is Blue team and "2" is Red team
        winner_team = str(row['TeamWinner'])
        loser_team = '2' if winner_team == '1' else '1'
        
        # Get the winners' and losers' IDs per role and update their Glicko rating
        for pos in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            winner_id = row[pos + winner_team + 'ID']
            loser_id = row[pos + loser_team + 'ID']

            # Only update if both IDs are valid and not NaN
            if pd.notnull(winner_id) and pd.notnull(loser_id):
                winner_id = int(winner_id)
                loser_id = int(loser_id)

                # Retrieve current ratings and RDs for the two players in the position
                R_winner, RD_winner = player_glicko[winner_id], player_RD[winner_id]
                R_loser, RD_loser = player_glicko[loser_id], player_RD[loser_id]

                # Apply the Glicko update function to get the new ratings and RDs
                R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated = glicko_update(
                    R_winner, R_loser, RD_winner, RD_loser)

                # Store the updated values back in the dictionaries
                player_glicko[winner_id], player_glicko[loser_id] = R_winner_updated, R_loser_updated
                player_RD[winner_id], player_RD[loser_id] = RD_winner_updated, RD_loser_updated

    # Once every match has been processed, write the results in a JSON file.
    with open(os.path.join(BASE_DIR,'./info/player_glicko_ratings.json'), 'w') as f:
        json.dump({"player_glicko": player_glicko, "player_RD": player_RD}, f, indent=4)

    return

calculate_player_glicko_ratings(df)
