from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import numpy as np

class MatchDataset(Dataset):
    def __init__(self, team1_ids, team2_ids, champions_team1, champions_team2, players_team1, players_team2, labels):
        self.team1_ids = team1_ids
        self.team2_ids = team2_ids
        self.champions_team1 = champions_team1
        self.champions_team2 = champions_team2
        self.players_team1 = players_team1
        self.players_team2 = players_team2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): 
        team1_id_tensor = self.team1_ids[idx]
        team2_id_tensor = self.team2_ids[idx]
        champions_team1_tensor = self.champions_team1[idx]
        champions_team2_tensor = self.champions_team2[idx]
        players_team1_tensor = self.players_team1[idx]
        players_team2_tensor = self.players_team2[idx]
        label_tensor = self.labels[idx]

        return team1_id_tensor, team2_id_tensor, champions_team1_tensor, champions_team2_tensor, players_team1_tensor, players_team2_tensor, label_tensor

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['TeamWinner'] = df['TeamWinner'] - 1
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    max_date = df['Date'].max()
    df['DaysSinceLatest'] = (max_date - df['Date']).dt.days
    df = df.drop('Date', axis=1)
    df = df.fillna(0)

    # List of categorical features for OneHotEncoding
    categorical_features = ['Team1ID', 'Team2ID', 'RegionID', 'Top1Champion', 'Jg1Champion', 'Mid1Champion', 'Adc1Champion', 'Supp1Champion', 'Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion']

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
    remainder='passthrough')
    
    X = df.drop(['TeamWinner'], axis=1)
    y = df['TeamWinner']

    # Apply the preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Since X_processed is a sparse matrix, we need to convert it to a dense format before making a PyTorch tensor
    X_processed = X_processed.toarray()  # Convert to dense

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.1, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Assuming `X_train` and `X_test` are pandas DataFrames at this point
    # Convert DataFrame columns to tensors for the training set
    team1_ids_train_tensor = torch.tensor(X_train['Team1ID'].values, dtype=torch.long)
    team2_ids_train_tensor = torch.tensor(X_train['Team2ID'].values, dtype=torch.long)
    champions_team1_train_tensor = torch.tensor(X_train[['Top1Champion', 'Jg1Champion', 'Mid1Champion', 'Adc1Champion', 'Supp1Champion']].values, dtype=torch.long)
    champions_team2_train_tensor = torch.tensor(X_train[['Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion']].values, dtype=torch.long)
    players_team1_train_tensor = torch.tensor(X_train[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID']].values, dtype=torch.long)
    players_team2_train_tensor = torch.tensor(X_train[['Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']].values, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) # Or dtype=torch.long for classification labels

    # Repeat for the test set
    team1_ids_test_tensor = torch.tensor(X_test['Team1ID'].values, dtype=torch.long)
    team2_ids_test_tensor = torch.tensor(X_test['Team2ID'].values, dtype=torch.long)
    champions_team1_test_tensor = torch.tensor(X_test[['Top1Champion', 'Jg1Champion', 'Mid1Champion', 'Adc1Champion', 'Supp1Champion']].values, dtype=torch.long)
    champions_team2_test_tensor = torch.tensor(X_test[['Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion']].values, dtype=torch.long)
    players_team1_test_tensor = torch.tensor(X_test[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID']].values, dtype=torch.long)
    players_team2_test_tensor = torch.tensor(X_test[['Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']].values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32) # Or dtype=torch.long for classification labels

    # Now instantiate MatchDataset objects with tensors
    train_data = MatchDataset(team1_ids_train_tensor, team2_ids_train_tensor, champions_team1_train_tensor, champions_team2_train_tensor, players_team1_train_tensor, players_team2_train_tensor, y_train_tensor)
    test_data = MatchDataset(team1_ids_test_tensor, team2_ids_test_tensor, champions_team1_test_tensor, champions_team2_test_tensor, players_team1_test_tensor, players_team2_test_tensor, y_test_tensor)

    # Assuming `weights_tensor` is already defined and corresponds to the training data

    return train_data, test_data
