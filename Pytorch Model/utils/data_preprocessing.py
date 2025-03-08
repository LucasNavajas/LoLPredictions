from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from joblib import dump
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MatchDataset(Dataset):
    """
    A PyTorch Dataset that holds the features and labels for each match.

    - features: Tensor containing the preprocessed features (e.g., Glicko ratings, champion IDs, etc.).
    - labels:   Tensor containing the corresponding match outcome or label (e.g., which team won).
    """
    def __init__(self, features, labels):
        """
        Stores the dataset features and labels.

        Args:
            features (torch.Tensor): Preprocessed features of shape (num_samples, num_features).
            labels (torch.Tensor):   Labels of shape (num_samples,).
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a single sample and its label.

        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            (torch.Tensor, torch.Tensor): A tuple (feature_tensor, label_tensor)
                                          representing one sample's data.
        """
        feature_tensor = self.features[idx]
        label_tensor = self.labels[idx]
        return feature_tensor, label_tensor
    
def glicko_update(R_winner, R_loser, RD_winner, RD_loser, K=64, RD_reduction_factor=0.97):
    """
    Performs a single Glicko-like rating update given a match between two players.

    Args:
        R_winner (float):         Current rating of the winner.
        R_loser (float):          Current rating of the loser.
        RD_winner (float):        Current rating deviation (RD) for the winner.
        RD_loser (float):         Current rating deviation (RD) for the loser.
        K (float, optional):      K-factor controlling update magnitude (default=64).
        RD_reduction_factor (float, optional): Factor by which to reduce RD after each match (default=0.97).

    Returns:
        tuple: (R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated)
    """
    
    # Glicko uses a constant q and a function g_RD(RD) to scale rating differences by rating uncertainty
    q = np.log(10) / 400

    # Function to scale based on rating deviation
    g_RD = lambda RD: 1 / np.sqrt(1 + 3 * q**2 * RD**2 / np.pi**2)

    # Expected win probabilities for each player winner or loser. Lower confidence, higher RD
    E_winner = 1 / (1 + 10 ** (g_RD(RD_loser) * (R_loser - R_winner) / 400))
    E_loser = 1 / (1 + 10 ** (g_RD(RD_winner) * (R_winner - R_loser) / 400))

    # Elo-style update using Glicko's expected value, adjusting the winner's rating up, loser's rating down
    R_winner_updated = R_winner + K * (1 - E_winner)
    R_loser_updated = R_loser - K * E_loser

    # Reduce rating deviation after a match, ensuring a minimum RD of 30 to avoid overconfidence
    RD_winner_updated = max(RD_winner * RD_reduction_factor, 30) 
    RD_loser_updated = max(RD_loser * RD_reduction_factor, 30)

    return R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated


def calculate_player_glicko_ratings(df):
    """
    Calculates Glicko-like ratings for all players in a DataFrame containing match data.
    It also appends average team Glicko ratings and RDs to the DataFrame.

    Args:
        df (pd.DataFrame): Must include columns for each role on each team
                           (e.g., 'Top1ID', 'Jg1ID', ..., 'Top2ID', 'Jg2ID', ..., 'TeamWinner').

    Returns:
        pd.DataFrame: The original DataFrame with added columns:
                      'Team1Glicko', 'Team2Glicko', 'Team1RD', 'Team2RD'.
    """
    # Get all unique players from the dataframe
    unique_player_ids = set(df[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID', 
                                'Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']].melt()['value'].dropna().unique())
    
    # Initialize each player's rating to 1500 and RD to 350
    player_glicko = {int(player_id): 1500 for player_id in unique_player_ids}
    player_RD = {int(player_id): 350 for player_id in unique_player_ids}

    # Lists to store the average Glicko ratings and RDs for both teams per match
    team1_glicko_updates, team2_glicko_updates = [], []
    team1_rd_updates, team2_rd_updates = [], []

    # Process each match in the DataFrame
    for index, row in df.iterrows():
        # Determine winner and loser
        winner_team = str(row['TeamWinner'])
        loser_team = '2' if winner_team == '1' else '1'

        # Temporary containers for the updated teams' ratings in this match
        glicko_team1, glicko_team2 = [], []
        rd_team1, rd_team2 = [], []

        # For each role, update the winner's and loser's ratings
        for pos in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            winner_id = row[pos + winner_team + 'ID']
            loser_id = row[pos + loser_team + 'ID']

            # Only proceed if both IDs are not null
            if pd.notnull(winner_id) and pd.notnull(loser_id):
                winner_id = int(winner_id)
                loser_id = int(loser_id)

                R_winner, RD_winner = player_glicko[winner_id], player_RD[winner_id]
                R_loser, RD_loser = player_glicko[loser_id], player_RD[loser_id]

                # Apply the Glicko update for these two players
                R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated = glicko_update(
                    R_winner, R_loser, RD_winner, RD_loser)

                # Store the updated values back in our dictionaries
                player_glicko[winner_id], player_glicko[loser_id] = R_winner_updated, R_loser_updated
                player_RD[winner_id], player_RD[loser_id] = RD_winner_updated, RD_loser_updated

                # Keep track of these updated ratings for team-wise averaging
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

        # Compute the average rating and RD for each team in this match
        team1_glicko_updates.append(np.mean(glicko_team1))
        team2_glicko_updates.append(np.mean(glicko_team2))
        team1_rd_updates.append(np.mean(rd_team1))
        team2_rd_updates.append(np.mean(rd_team2))

    # Add the new columns to the DataFrame to keep track of average Glicko and RD
    df['Team1Glicko'] = team1_glicko_updates
    df['Team2Glicko'] = team2_glicko_updates
    df['Team1RD'] = team1_rd_updates
    df['Team2RD'] = team2_rd_updates
    return df


def load_and_preprocess_data(filepath):
    """
    Loads match data from a CSV, calculates Glicko ratings for all players,
    preprocesses select columns (scaling numeric features, passing champion IDs),
    splits the data into train/validation/test sets, and returns them as PyTorch Datasets.

    Args:
        filepath (str): Path to the CSV file containing match data.

    Returns:
        tuple of (train_data, test_data, val_data):
            - train_data (MatchDataset): Training dataset with preprocessed features and labels.
            - test_data (MatchDataset):  Testing dataset with preprocessed features and labels.
            - val_data (MatchDataset):   Validation dataset with preprocessed features and labels.
    """

    df = pd.read_csv(os.path.join(BASE_DIR,filepath))
    
    # Calculate Glicko ratings and attach average team-level Glicko columns to df
    df = calculate_player_glicko_ratings(df)

    # Convert 'TeamWinner' from {1,2} to {0,1} by subtracting 1 (Team 1 => 0, Team 2 => 1)
    df['TeamWinner'] = df['TeamWinner'] - 1

    # Scale numerical features (Team1Glicko, Team2Glicko) and pass champion columns through unchanged (since they're IDs)
    numerical_features = ['Team1Glicko', 'Team2Glicko']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('synergy', 'passthrough', [
                'Top1Champion', 'Jg1Champion', 'Mid1Champion', 'Adc1Champion', 'Supp1Champion',
                'Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion'])
        ], remainder='drop')

    # Separate target column from feature columns
    X = df.drop('TeamWinner', axis=1)
    y = df['TeamWinner']

    # Fit and transform the preprocessor on X
    X_processed = preprocessor.fit_transform(X)

    # Save the preprocessor for inference or future use
    dump(preprocessor, os.path.join(BASE_DIR,'../preprocessor.joblib'))

    # Split the data into train/validation/test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_processed, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.50, random_state=0)

    # Convert NumPy arrays to PyTorch tensors (float32 for features, long for labels)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create custom MatchDataset instances
    train_data = MatchDataset(X_train_tensor, y_train_tensor)
    val_data = MatchDataset(X_val_tensor, y_val_tensor)
    test_data = MatchDataset(X_test_tensor, y_test_tensor)

    # Return the train, test, and validation datasets
    return train_data, test_data, val_data
