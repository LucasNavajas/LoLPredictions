import pandas as pd
from itertools import combinations

def calculate_synergy(matches_df):
    synergy_data = {}
    
    # Iterate over all matches
    for index, row in matches_df.iterrows():
        # Assume function get_champion_ids() retrieves the champion IDs for the team
        # This function needs to be defined by you based on your dataframe structure
        team1_champions = get_champion_ids(row, team=1)
        team2_champions = get_champion_ids(row, team=2)
        
        # Record the synergy for each unique champion pair within each team
        for team_champions in [team1_champions, team2_champions]:
            for champ1, champ2 in combinations(team_champions, 2):
                pair_key = tuple(sorted([champ1, champ2]))
                
                # Initialize the record for this champion pair if it doesn't exist
                if pair_key not in synergy_data:
                    synergy_data[pair_key] = {'wins': 0, 'total': 0}
                
                # Increment win and total counters based on the match outcome
                if team_champions == team1_champions and row['TeamWinner'] == 1:
                    synergy_data[pair_key]['wins'] += 1
                elif team_champions == team2_champions and row['TeamWinner'] == 2:
                    synergy_data[pair_key]['wins'] += 1
                synergy_data[pair_key]['total'] += 1
    
    # Calculate the win rate for each champion pair
    synergy_win_rates = {}
    for key, value in synergy_data.items():
        try:
            # Attempt to create a string key with integer champion IDs
            string_key = f"{int(key[0])}-{int(key[1])}"
            synergy_win_rates[string_key] = value['wins'] / value['total']
        except ValueError:
            # Skip this pair if either champion ID is NaN
            continue
    return synergy_win_rates

def get_champion_ids(row, team):
    # You will need to replace the below line with the actual code that extracts the
    # champion IDs from the match row, based on your dataset's structure.
    return [row[f'{position}{team}Champion'] for position in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']]

# Load your match data
matches_df = pd.read_csv('data/datasheetv2.csv')

# Calculate the synergy data
team_synergy_data = calculate_synergy(matches_df)

# Save the synergy data to a JSON file
import json
with open('team_synergies.json', 'w') as f:
    json.dump(team_synergy_data, f)
