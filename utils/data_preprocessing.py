from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class MatchDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_tensor = self.features[idx]
        label_tensor = self.labels[idx]

        return feature_tensor, label_tensor

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
    df.drop('Date', axis=1, inplace=True)
    df.fillna(0, inplace=True)

    numerical_features = ['DaysSinceLatest', 'Team1WinRate', 'Team2WinRate']

    
    for role in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
        for team in ['1', '2']:
            # Define the new win rate column name for clarity
            new_column_name = f'{role}{team}ChampionWinRate'
            
            # Perform the merge operation with an explicit selection of columns to avoid unwanted columns
            # Assuming 'player_id' and 'champion_id' are in the player_champion_win_rates DataFrame
            merge_df = player_champion_win_rates[['player_id', 'champion_id', 'win_rate']].copy()
            merge_df.rename(columns={'win_rate': new_column_name}, inplace=True)
            
            df = df.merge(merge_df, how='left', left_on=[f'{role}{team}ID', f'{role}{team}Champion'], right_on=['player_id', 'champion_id']).drop(['player_id', 'champion_id'], axis=1)
            
            # Since we're careful with the merge, we directly add the intended win rate column name
            numerical_features.append(new_column_name)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ], remainder='passthrough')

    X = df.drop('TeamWinner', axis=1)
    y = df['TeamWinner']
    X_processed = preprocessor.fit_transform(X)

    # Split the processed data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=42)

    # Convert split data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    train_data = MatchDataset(X_train_tensor, y_train_tensor)
    test_data = MatchDataset(X_test_tensor, y_test_tensor)

    # Return processed and split tensors
    return train_data, test_data

# Then, use the returned tensors for training and evaluating your model.

