import json
import pandas as pd

def calculate_h2h_win_rates(datasheet):
    # Calculate win/loss count for each match-up
    h2h_records = {}

    for index, row in datasheet.iterrows():
        # Sort team IDs and create a tuple
        teams = tuple(sorted([row['Team1ID'], row['Team2ID']]))
        winner = row['TeamWinner']

        if teams not in h2h_records:
            h2h_records[teams] = {'wins': 0, 'losses': 0}
        
        if winner == 1:
            h2h_records[teams]['wins'] += 1
        else:
            h2h_records[teams]['losses'] += 1

    # Convert to win rates and ensure keys are strings suitable for JSON
    h2h_win_rates_str_keys = {'-'.join(map(str, teams)): record['wins'] / (record['wins'] + record['losses']) for teams, record in h2h_records.items()}

    return h2h_win_rates_str_keys

# Assuming 'datasheet_path' points to your dataset
datasheet = pd.read_csv('data/datasheetv2.csv')
h2h_win_rates = calculate_h2h_win_rates(datasheet)

# Save the h2h_win_rates with string keys to a JSON file
with open('h2h_win_rates.json', 'w') as file:
    json.dump(h2h_win_rates, file)
