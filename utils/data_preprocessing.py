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
    df = df.drop('MatchID', axis=1, errors="ignore")
    df['TeamWinner'] = df['TeamWinner'] - 1
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    max_date = df['Date'].max()
    df['DaysSinceLatest'] = (max_date - df['Date']).dt.days
    df = df.drop('Date', axis=1)
    df = df.fillna(0)

    percentage_columns = ['Top1KP', 'Top2KP', 'Jg1KP', 'Jg2KP', 'Mid1KP', 'Mid2KP', 'Adc1KP', 'Adc2KP', 'Supp1KP', 'Supp2KP']  # Añade aquí todas tus columnas de porcentaje

    for col in percentage_columns:
        df[col] = df[col].str.replace('%', '').astype(float) / 100


    # Lista de todas las características numéricas que deseas escalar
    numerical_features = ['Top1Kills', 'Top1Deaths', 'Top1Assists', 'Top1CS', 'Top1CSM', 'Top1GPM', 'Top1VisionScore', 'Top1VSPM', 'Top1DMGTotal', 'Top1DPM', 'Top1K+A/Min', 'Top1KP', 'Top1GD@15', 'Top1CSD@15', 'Top1XPD@15', 'Top1LVLD@15', 'Top1ObjStolen', 'Top1DMGTurrets', 'Top1TimeSpentDead', 'Top1ItemsPurchased', 'Top1BountyCollected', 'Top1BountyLost',
                             'Jg1Kills', 'Jg1Deaths', 'Jg1Assists', 'Jg1CS', 'Jg1CSM', 'Jg1GPM', 'Jg1VisionScore', 'Jg1VSPM', 'Jg1DMGTotal', 'Jg1DPM', 'Jg1K+A/Min', 'Jg1KP', 'Jg1GD@15', 'Jg1CSD@15', 'Jg1XPD@15', 'Jg1LVLD@15', 'Jg1ObjStolen', 'Jg1DMGTurrets', 'Jg1TimeSpentDead', 'Jg1ItemsPurchased', 'Jg1BountyCollected', 'Jg1BountyLost',
                             'Mid1Kills', 'Mid1Deaths', 'Mid1Assists', 'Mid1CS', 'Mid1CSM', 'Mid1GPM', 'Mid1VisionScore', 'Mid1VSPM', 'Mid1DMGTotal', 'Mid1DPM', 'Mid1K+A/Min', 'Mid1KP', 'Mid1GD@15', 'Mid1CSD@15', 'Mid1XPD@15', 'Mid1LVLD@15', 'Mid1ObjStolen', 'Mid1DMGTurrets', 'Mid1TimeSpentDead', 'Mid1ItemsPurchased', 'Mid1BountyCollected', 'Mid1BountyLost',
                             'Adc1Kills', 'Adc1Deaths', 'Adc1Assists', 'Adc1CS', 'Adc1CSM', 'Adc1GPM', 'Adc1VisionScore', 'Adc1VSPM', 'Adc1DMGTotal', 'Adc1DPM', 'Adc1K+A/Min', 'Adc1KP', 'Adc1GD@15', 'Adc1CSD@15', 'Adc1XPD@15', 'Adc1LVLD@15', 'Adc1ObjStolen', 'Adc1DMGTurrets', 'Adc1TimeSpentDead', 'Adc1ItemsPurchased', 'Adc1BountyCollected', 'Adc1BountyLost',
                             'Supp1Kills', 'Supp1Deaths', 'Supp1Assists', 'Supp1CS', 'Supp1CSM', 'Supp1GPM', 'Supp1VisionScore', 'Supp1VSPM', 'Supp1DMGTotal', 'Supp1DPM', 'Supp1K+A/Min', 'Supp1KP', 'Supp1GD@15', 'Supp1CSD@15', 'Supp1XPD@15', 'Supp1LVLD@15', 'Supp1ObjStolen', 'Supp1DMGTurrets', 'Supp1TimeSpentDead', 'Supp1ItemsPurchased', 'Supp1BountyCollected', 'Supp1BountyLost',
                            'Top2Kills', 'Top2Deaths', 'Top2Assists', 'Top2CS', 'Top2CSM', 'Top2GPM', 'Top2VisionScore', 'Top2VSPM', 'Top2DMGTotal', 'Top2DPM', 'Top2K+A/Min', 'Top2KP', 'Top2GD@15', 'Top2CSD@15', 'Top2XPD@15', 'Top2LVLD@15', 'Top2ObjStolen', 'Top2DMGTurrets', 'Top2TimeSpentDead', 'Top2ItemsPurchased', 'Top2BountyCollected', 'Top2BountyLost',
                             'Jg2Kills', 'Jg2Deaths', 'Jg2Assists', 'Jg2CS', 'Jg2CSM', 'Jg2GPM', 'Jg2VisionScore', 'Jg2VSPM', 'Jg2DMGTotal', 'Jg2DPM', 'Jg2K+A/Min', 'Jg2KP', 'Jg2GD@15', 'Jg2CSD@15', 'Jg2XPD@15', 'Jg2LVLD@15', 'Jg2ObjStolen', 'Jg2DMGTurrets', 'Jg2TimeSpentDead', 'Jg2ItemsPurchased', 'Jg2BountyCollected', 'Jg2BountyLost',
                             'Mid2Kills', 'Mid2Deaths', 'Mid2Assists', 'Mid2CS', 'Mid2CSM', 'Mid2GPM', 'Mid2VisionScore', 'Mid2VSPM', 'Mid2DMGTotal', 'Mid2DPM', 'Mid2K+A/Min', 'Mid2KP', 'Mid2GD@15', 'Mid2CSD@15', 'Mid2XPD@15', 'Mid2LVLD@15', 'Mid2ObjStolen', 'Mid2DMGTurrets', 'Mid2TimeSpentDead', 'Mid2ItemsPurchased', 'Mid2BountyCollected', 'Mid2BountyLost',
                             'Adc2Kills', 'Adc2Deaths', 'Adc2Assists', 'Adc2CS', 'Adc2CSM', 'Adc2GPM', 'Adc2VisionScore', 'Adc2VSPM', 'Adc2DMGTotal', 'Adc2DPM', 'Adc2K+A/Min', 'Adc2KP', 'Adc2GD@15', 'Adc2CSD@15', 'Adc2XPD@15', 'Adc2LVLD@15', 'Adc2ObjStolen', 'Adc2DMGTurrets', 'Adc2TimeSpentDead', 'Adc2ItemsPurchased', 'Adc2BountyCollected', 'Adc2BountyLost',
                             'Supp2Kills', 'Supp2Deaths', 'Supp2Assists', 'Supp2CS', 'Supp2CSM', 'Supp2GPM', 'Supp2VisionScore', 'Supp2VSPM', 'Supp2DMGTotal', 'Supp2DPM', 'Supp2K+A/Min', 'Supp2KP', 'Supp2GD@15', 'Supp2CSD@15', 'Supp2XPD@15', 'Supp2LVLD@15', 'Supp2ObjStolen', 'Supp2DMGTurrets', 'Supp2TimeSpentDead', 'Supp2ItemsPurchased', 'Supp2BountyCollected', 'Supp2BountyLost'
                             ]
    # Lista de variables categóricas para OneHotEncoding
    categorical_features = ['Team1ID', 'Team2ID', 'RegionID', 'Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion', 'Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion']

    # Crear el transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
    remainder='passthrough')
    X = df.values.astype('float32')
    # Separar la variable objetivo y las características
    y = df['TeamWinner']
    X = df.drop(['TeamWinner'], axis=1)  # Omitir MatchID y Date si no son relevantes

    # Aplicar el preprocesamiento
    X_processed = preprocessor.fit_transform(X)

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Convertir a tensores de PyTorch
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
