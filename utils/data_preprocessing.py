from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from joblib import dump


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
    

# Glicko rating update function
def glicko_update(R_winner, R_loser, RD_winner, RD_loser, K=64, RD_reduction_factor=0.97):
    
    q = np.log(10) / 400

    g_RD = lambda RD: 1 / np.sqrt(1 + 3 * q**2 * RD**2 / np.pi**2)

    E_winner = 1 / (1 + 10 ** (g_RD(RD_loser) * (R_loser - R_winner) / 400))

    E_loser = 1 / (1 + 10 ** (g_RD(RD_winner) * (R_winner - R_loser) / 400))

    R_winner_updated = R_winner + K * (1 - E_winner)

    R_loser_updated = R_loser - K * E_loser

    RD_winner_updated = max(RD_winner * RD_reduction_factor, 30) 

    RD_loser_updated = max(RD_loser * RD_reduction_factor, 30)

    return R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated

# Function to calculate Glicko ratings for all players based on match data
def calculate_player_glicko_ratings(df):
    unique_player_ids = set(df[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID', 
                                'Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']].melt()['value'].dropna().unique())
    
    player_glicko = {int(player_id): 1500 for player_id in unique_player_ids}
    player_RD = {int(player_id): 350 for player_id in unique_player_ids}

    team1_glicko_updates, team2_glicko_updates = [], []
    team1_rd_updates, team2_rd_updates = [], []

    # Iterate over each match and update Glicko ratings
    for index, row in df.iterrows():
        winner_team = str(row['TeamWinner'])
        loser_team = '2' if winner_team == '1' else '1'

        glicko_team1, glicko_team2 = [], []
        rd_team1, rd_team2 = [], []

        # Iterate through positions (Top, Jg, Mid, Adc, Supp) to update player ratings
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

                if winner_team == '1':
                    glicko_team1.append(R_winner_updated)
                    rd_team1.append(RD_winner_updated)
                    glicko_team2.append(R_loser_updated)
                    rd_team2.append(RD_loser_updated)
                else:
                    glicko_team2.append(R_winner_updated)
                    rd_team2.append(RD_winner_updated)
                    glicko_team1.append(R_loser_updated)
                    rd_team1.append(RD_loser_updated)

        team1_glicko_updates.append(np.mean(glicko_team1))
        team2_glicko_updates.append(np.mean(glicko_team2))
        team1_rd_updates.append(np.mean(rd_team1))
        team2_rd_updates.append(np.mean(rd_team2))

    df['Team1Glicko'] = team1_glicko_updates
    df['Team2Glicko'] = team2_glicko_updates
    df['Team1RD'] = team1_rd_updates
    df['Team2RD'] = team2_rd_updates
    return df


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    df = calculate_player_glicko_ratings(df)

    # Convert TeamWinner to 0 and 1
    df['TeamWinner'] = df['TeamWinner'] - 1

    numerical_features = ['Team1Glicko', 'Team2Glicko']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('synergy', 'passthrough', [
                'Top1Champion', 'Jg1Champion', 'Mid1Champion', 'Adc1Champion', 'Supp1Champion',
                'Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion'])
        ], remainder='drop')

    X = df.drop('TeamWinner', axis=1)
    y = df['TeamWinner']

    X_processed = preprocessor.fit_transform(X)

    dump(preprocessor, 'preprocessor.joblib')

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_processed, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.50, random_state=0)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    train_data = MatchDataset(X_train_tensor, y_train_tensor)
    val_data = MatchDataset(X_val_tensor, y_val_tensor)
    test_data = MatchDataset(X_test_tensor, y_test_tensor)

    return train_data, test_data, val_data
