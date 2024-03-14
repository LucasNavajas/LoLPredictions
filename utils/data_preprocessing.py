from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class MatchDataset(Dataset):
    def __init__(self, features, labels):  # Add an optional regions parameter
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_tensor = self.features[idx]
        label_tensor = self.labels[idx]
        return feature_tensor, label_tensor
    

def apply_feature_weights(df):
    # Define weights for the prioritized features
    weights = {
        'H2H_Win_Diff': 1.1,  
        'Team1WinRate': 1.75,  
        'Team2WinRate': 1.75,  
        'Team1_Synergy': 1.5,  
        'Team2_Synergy': 1.5,
        'Top1ChampionWinrate' : 0.5,
        'Jg1ChampionWinrate' : 0.5,
        'Mid1ChampionWinrate' : 0.5,
        'Adc1ChampionWinrate' : 0.5,
        'Supp1ChampionWinrate' : 0.5,
        'Top2ChampionWinrate' : 0.5,
        'Jg2ChampionWinrate' : 0.5,
        'Mid2ChampionWinrate' : 0.5,
        'Adc2ChampionWinrate' : 0.5,
        'Supp2ChampionWinrate' : 0.5,

    }
    
    # Apply the weights to the DataFrame
    for feature, weight in weights.items():
        if feature in df.columns:
            df[feature] *= weight
    
    return df

def add_synergy_to_matches(df, synergy_df):
    # Function to retrieve the synergy score for a given pair of champions
    def get_synergy(pair):
        if pair in synergy_dict:
            return synergy_dict[pair]
        else:
            # If no synergy information is present, you could use a default value
            return 0.5  # or some other default value you determine
    
    # Convert the synergy DataFrame to a dictionary for faster look-up
    synergy_dict = pd.Series(synergy_df.win_rate.values, index=synergy_df.pair).to_dict()

    # List to store synergy scores for each match
    synergy_scores_team1 = []
    synergy_scores_team2 = []

    # Calculate the synergy score for each match
    for index, row in df.iterrows():
        champions_team1 = [row[f'Top1Champion'], row[f'Jg1Champion'], row[f'Mid1Champion'], row[f'Adc1Champion'], row[f'Supp1Champion']]
        champions_team2 = [row[f'Top2Champion'], row[f'Jg2Champion'], row[f'Mid2Champion'], row[f'Adc2Champion'], row[f'Supp2Champion']]

        # Calculate the average synergy for the team's champion pairs
        synergy_score_team1 = np.mean([get_synergy(tuple(sorted([champions_team1[i], champions_team1[j]])))
                                       for i in range(len(champions_team1)) for j in range(i + 1, len(champions_team1))])
        synergy_score_team2 = np.mean([get_synergy(tuple(sorted([champions_team2[i], champions_team2[j]])))
                                       for i in range(len(champions_team2)) for j in range(i + 1, len(champions_team2))])

        synergy_scores_team1.append(synergy_score_team1)
        synergy_scores_team2.append(synergy_score_team2)

    # Add the synergy scores as new columns in the DataFrame
    df['Team1_Synergy'] = synergy_scores_team1
    df['Team2_Synergy'] = synergy_scores_team2

    return df

def calculate_champion_synergy(datasheet):
    # Initialize a record for champion pair win rates
    champion_pairs = {}

    for index, row in datasheet.iterrows():
        champions_team1 = [row[f'Top1Champion'], row[f'Jg1Champion'], row[f'Mid1Champion'], row[f'Adc1Champion'], row[f'Supp1Champion']]
        champions_team2 = [row[f'Top2Champion'], row[f'Jg2Champion'], row[f'Mid2Champion'], row[f'Adc2Champion'], row[f'Supp2Champion']]

        # Create unique pairs of champions within the team and update win/loss
        for team_champions in [champions_team1, champions_team2]:
            for i in range(len(team_champions)):
                for j in range(i + 1, len(team_champions)):
                    pair = tuple(sorted([team_champions[i], team_champions[j]]))
                    winner = row['TeamWinner']

                    if pair not in champion_pairs:
                        champion_pairs[pair] = {'wins': 0, 'total_games': 0}
                    
                    if ((winner == 1 and team_champions == champions_team1) or 
                        (winner == 2 and team_champions == champions_team2)):
                        champion_pairs[pair]['wins'] += 1
                    
                    champion_pairs[pair]['total_games'] += 1

    # Convert the pair records to a DataFrame
    champion_synergy_records = [{'pair': key, 'win_rate': value['wins'] / value['total_games']} for key, value in champion_pairs.items()]
    champion_synergy_df = pd.DataFrame(champion_synergy_records)
    
    # Sort by the best synergy
    champion_synergy_df.sort_values(by='win_rate', ascending=False, inplace=True)

    return champion_synergy_df

def calculate_head_to_head_record(datasheet):
    # Initialize a dictionary to keep track of win-loss records
    head_to_head = {}
    
    # Sort the datasheet by date if it's not already
    datasheet.sort_values(by='Date', ascending=True, inplace=True)

    # List to store the head-to-head score for each match
    h2h_scores = []

    for index, row in datasheet.iterrows():
        teams_tuple = tuple(sorted([row['Team1ID'], row['Team2ID']]))
        winner = row['TeamWinner']

        # Initialize the record for these teams if not already present
        if teams_tuple not in head_to_head:
            head_to_head[teams_tuple] = {'Team1Wins': 0, 'Team2Wins': 0}
        
        # Current record before this match
        current_record = head_to_head[teams_tuple].copy()

        # Update the win count based on the winner
        if winner == 1:
            head_to_head[teams_tuple]['Team1Wins'] += 1
        else:
            head_to_head[teams_tuple]['Team2Wins'] += 1
        
        # Add the current record to the list
        h2h_scores.append(current_record)

    return pd.DataFrame(h2h_scores)

def add_head_to_head_feature(df, h2h_scores):
    # Assuming your main dataframe is 'df' and h2h_scores is returned from the function above
    df['Team1_H2H_Wins'] = h2h_scores['Team1Wins']
    df['Team2_H2H_Wins'] = h2h_scores['Team2Wins']
    
    # Optionally, calculate win rates or differences in wins for each pair
    df['H2H_Win_Diff'] = df['Team1_H2H_Wins'] - df['Team2_H2H_Wins']

    return df


def calculate_team_win_rates(datasheet_path):
    datasheet = pd.read_csv(datasheet_path)

    # Initialize records for win/loss tally
    records = []
    for index, row in datasheet.iterrows():
        winner_id = 'Team1ID' if row['TeamWinner'] == 1 else 'Team2ID'
        loser_id = 'Team2ID' if row['TeamWinner'] == 1 else 'Team1ID'
        
        # Record for winning team
        records.append({'team_id': row[winner_id], 'win': 1, 'loss': 0})
        # Record for losing team
        records.append({'team_id': row[loser_id], 'win': 0, 'loss': 1})

    # Convert records to a DataFrame
    team_records = pd.DataFrame(records)

    # Aggregate wins and losses by team
    team_performance = team_records.groupby('team_id').agg({'win': 'sum', 'loss': 'sum'}).reset_index()

    # Calculate win rates
    team_performance['win_rate'] = team_performance['win'] / (team_performance['win'] + team_performance['loss'])

    return team_performance[['team_id', 'win_rate']]

def calculate_player_champion_win_rates(datasheet_path):
    datasheet = pd.read_csv(datasheet_path)

    # Initialize records for win/loss tally
    records = []
    for index, row in datasheet.iterrows():
        team_winner = row['TeamWinner']
        for team in ['1', '2']:
            win = int(team == str(team_winner))
            loss = 1 - win
            for role in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
                player_id_col = f"{role}{team}ID"
                champion_id_col = f"{role}{team}Champion"
                records.append({'player_id': row[player_id_col], 'champion_id': row[champion_id_col], 'win': win, 'loss': loss})

    player_champion_records = pd.DataFrame(records)
    player_champion_performance = player_champion_records.groupby(['player_id', 'champion_id']).agg({'win': 'sum', 'loss': 'sum'}).reset_index()
    player_champion_performance['win_rate'] = player_champion_performance['win'] / (player_champion_performance['win'] + player_champion_performance['loss'])

    return player_champion_performance


def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    team_win_rates = calculate_team_win_rates(filepath)
    player_champion_win_rates = calculate_player_champion_win_rates(filepath)

    # Merge team win rates
    df = df.merge(team_win_rates, how='left', left_on='Team1ID', right_on='team_id').rename(columns={'win_rate': 'Team1WinRate'}).drop('team_id', axis=1)
    df = df.merge(team_win_rates, how='left', left_on='Team2ID', right_on='team_id').rename(columns={'win_rate': 'Team2WinRate'}).drop('team_id', axis=1)

    df['TeamWinner'] = df['TeamWinner'] - 1
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['DaysSinceLatest'] = (df['Date'].max() - df['Date']).dt.days
    h2h_scores = calculate_head_to_head_record(df)
    df = add_head_to_head_feature(df, h2h_scores)
    df.drop('Date', axis=1, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    champion_synergy_df = calculate_champion_synergy(df)
    df = add_synergy_to_matches(df, champion_synergy_df)
    df = apply_feature_weights(df)

    numerical_features = ['DaysSinceLatest','Team1WinRate', 'Team2WinRate','H2H_Win_Diff', 'Team1_Synergy',  'Team2_Synergy']

    
    for role in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            for team in ['1', '2']:
                new_column_name = f'{role}{team}ChampionWinRate'
                merge_df = player_champion_win_rates[['player_id', 'champion_id', 'win_rate']].copy()
                merge_df.rename(columns={'win_rate': new_column_name}, inplace=True)

                # Merge and drop unnecessary columns
                df = df.merge(merge_df, how='left', left_on=[f'{role}{team}ID', f'{role}{team}Champion'], right_on=['player_id', 'champion_id']).drop(['player_id', 'champion_id'], axis=1)
                
                # Fill NaN values with 0.5 for player-champion combinations without data
                df[new_column_name].fillna(0.50, inplace=True)

                # Append new win rate column name to 'numerical_features'
                numerical_features.append(new_column_name)
    
    print(df.head)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ], remainder='passthrough')
    

    X = df.drop('TeamWinner', axis=1)
    y = df['TeamWinner']
    X_processed = preprocessor.fit_transform(X)

    # Split the processed data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=42)

    class_sample_counts = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    smoothing_factor = 0.03
    max_sample_count = np.max(class_sample_counts)
    weight = 1. / (class_sample_counts + (smoothing_factor * max_sample_count))
    samples_weight = np.array([weight[t] for t in y_train])

    # Create a WeightedRandomSampler
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Convert split data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    train_data = MatchDataset(X_train_tensor, y_train_tensor)
    test_data = MatchDataset(X_test_tensor, y_test_tensor)

    # Return processed and split tensors
    return train_data, test_data, sampler

# Then, use the returned tensors for training and evaluating your model.

