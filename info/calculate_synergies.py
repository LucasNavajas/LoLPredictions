import pandas as pd
from itertools import combinations
import json

def calculate_accumulative_winrates(matches_df):
    champion_winrates = {}
    for index, row in matches_df.iterrows():
        region = row['RegionID']
        champions = get_champion_ids(row, 1) + get_champion_ids(row, 2)
        winner = row['TeamWinner']
        
        for i, champ in enumerate(champions):
            key = (champ, region)
            
            if key not in champion_winrates:
                champion_winrates[key] = {'wins': 0, 'total': 0}
            
            champion_winrates[key]['total'] += 1
            if ((i < 5 and winner == 1) or (i >= 5 and winner == 2)):
                champion_winrates[key]['wins'] += 1
                
    # Calculating win rates
    for key, stats in champion_winrates.items():
        champion_winrates[key]['winrate'] = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
    
    return champion_winrates


def calculate_synergy_with_winrates(matches_df, champion_winrates):
    synergy_data = {}
    
    for index, row in matches_df.iterrows():
        region = row['RegionID']
        team1_champions = get_champion_ids(row, team=1)
        team2_champions = get_champion_ids(row, team=2)
        
        for team_champions in [team1_champions, team2_champions]:
            for champ1, champ2 in combinations(team_champions, 2):
                # Incluye la región en la clave
                pair_key = (tuple(sorted([champ1, champ2])), region)
                
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
    synergy_win_rates = {}
    for key, value in synergy_data.items():
        # Descomponer la clave compuesta
        champion_pair, region = key
        pair_key = f"{champion_pair[0]}-{champion_pair[1]}-{region}"
        synergy_win_rates[pair_key] = value['wins'] / value['total'] if value['total'] > 0 else 0

    with open('team_synergies_by_region.json', 'w') as f:
        json.dump(synergy_win_rates, f, indent=4)



# Load your match data
matches_df = pd.read_csv('data/datasheetv2.csv')
matches_df

# Calculate the synergy data
winrates_df = calculate_accumulative_winrates(matches_df)
team_synergy_data = calculate_synergy_with_winrates(matches_df, winrates_df)
convert_to_json(team_synergy_data)

