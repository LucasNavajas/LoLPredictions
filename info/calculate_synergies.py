import pandas as pd
from itertools import combinations
import json

def calculate_synergy_common(matches_df):
    synergy_data = {}
    
    for index, row in matches_df.iterrows():
        # Suponiendo la existencia de una función get_champion_ids que hace lo mismo que antes
        team1_champions = get_champion_ids(row, team=1)
        team2_champions = get_champion_ids(row, team=2)
        
        for team_champions in [team1_champions, team2_champions]:
            for champ1, champ2 in combinations(team_champions, 2):
                pair_key = tuple(sorted([champ1, champ2]))
                
                if pair_key not in synergy_data:
                    synergy_data[pair_key] = {'wins': 0, 'total': 0}
                
                if team_champions == team1_champions and row['TeamWinner'] == 1:
                    synergy_data[pair_key]['wins'] += 1
                elif team_champions == team2_champions and row['TeamWinner'] == 2:
                    synergy_data[pair_key]['wins'] += 1
                synergy_data[pair_key]['total'] += 1

    return synergy_data

def get_champion_ids(row, team):
    # champion IDs from the match row, based on your dataset's structure.
    return [row[f'{position}{team}Champion'] for position in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']]

def convert_to_json(synergy_data):
    synergy_win_rates = {f"{key[0]}-{key[1]}": value['wins'] / (value['total']) for key, value in synergy_data.items()}
    with open('team_synergies.json', 'w') as f:
        json.dump(synergy_win_rates, f)


# Load your match data
matches_df = pd.read_csv('data/datasheetv2.csv')
matches_df

# Calculate the synergy data
team_synergy_data = calculate_synergy_common(matches_df)
convert_to_json(team_synergy_data)

