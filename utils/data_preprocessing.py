from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import numpy as np
from joblib import dump, load
import json

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
    

def glicko_update(R_winner, R_loser, RD_winner, RD_loser, K=64, RD_reduction_factor=0.95):
    q = np.log(10) / 400
    g_RD = lambda RD: 1 / np.sqrt(1 + 3 * q**2 * RD**2 / np.pi**2)
    E_winner = 1 / (1 + 10 ** (g_RD(RD_loser) * (R_loser - R_winner) / 400))
    E_loser = 1 / (1 + 10 ** (g_RD(RD_winner) * (R_winner - R_loser) / 400))
    R_winner_updated = R_winner + K * (1 - E_winner)
    R_loser_updated = R_loser - K * E_loser
    # Aplicar un simple factor de reducción al RD para ambos jugadores
    RD_winner_updated = max(RD_winner * RD_reduction_factor, 30)  # Asumir 30 como mínimo RD sugerido por Glicko
    RD_loser_updated = max(RD_loser * RD_reduction_factor, 30)
    return R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated



def calculate_player_glicko_ratings(df):
    unique_player_ids = set(df[['Top1ID', 'Jg1ID', 'Mid1ID', 'Adc1ID', 'Supp1ID', 'Top2ID', 'Jg2ID', 'Mid2ID', 'Adc2ID', 'Supp2ID']].melt()['value'].dropna().unique())
    player_glicko = {int(player_id): 1500 for player_id in unique_player_ids}
    player_RD = {int(player_id): 350 for player_id in unique_player_ids}

    # Listas para almacenar las clasificaciones Glicko y RD actualizadas de los equipos para cada partido
    team1_glicko_updates, team2_glicko_updates = [], []
    team1_rd_updates, team2_rd_updates = [], []

    for index, row in df.iterrows():
        equipo_ganador = str(row['TeamWinner'])
        equipo_perdedor = '2' if equipo_ganador == '1' else '1'

        # Listas para acumular los Glicko y RD de cada jugador por equipo en este partido
        glicko_team1, glicko_team2 = [], []
        rd_team1, rd_team2 = [], []

        for pos in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            id_ganador = row[pos + equipo_ganador + 'ID']
            id_perdedor = row[pos + equipo_perdedor + 'ID']

            if pd.notnull(id_ganador) and pd.notnull(id_perdedor):
                id_ganador = int(id_ganador)
                id_perdedor = int(id_perdedor)

                R_winner, RD_winner = player_glicko[id_ganador], player_RD[id_ganador]
                R_loser, RD_loser = player_glicko[id_perdedor], player_RD[id_perdedor]

                R_winner_updated, R_loser_updated, RD_winner_updated, RD_loser_updated = glicko_update(
                    R_winner, R_loser, RD_winner, RD_loser)

                player_glicko[id_ganador], player_glicko[id_perdedor] = R_winner_updated, R_loser_updated
                player_RD[id_ganador], player_RD[id_perdedor] = RD_winner_updated, RD_loser_updated

                # Acumular las clasificaciones Glicko y RD para los equipos basado en este partido
                if equipo_ganador == '1':
                    glicko_team1.append(R_winner_updated)
                    rd_team1.append(RD_winner_updated)
                    glicko_team2.append(R_loser_updated)
                    rd_team2.append(RD_loser_updated)
                else:
                    glicko_team2.append(R_winner_updated)
                    rd_team2.append(RD_winner_updated)
                    glicko_team1.append(R_loser_updated)
                    rd_team1.append(RD_loser_updated)

        # Calcular el promedio de Glicko y RD para cada equipo en este partido y agregarlo a las listas
        team1_glicko_updates.append(np.mean(glicko_team1))
        team2_glicko_updates.append(np.mean(glicko_team2))
        team1_rd_updates.append(np.mean(rd_team1))
        team2_rd_updates.append(np.mean(rd_team2))

    # Añadir las clasificaciones Glicko y RD actualizadas al DataFrame original
    df['Team1Glicko'] = team1_glicko_updates
    df['Team2Glicko'] = team2_glicko_updates
    df['Team1RD'] = team1_rd_updates
    df['Team2RD'] = team2_rd_updates
    return df

def calculate_accumulative_regional_winrates(datasheet):
    champion_regional_stats = {}
    
    for index, row in datasheet.iterrows():
        region = row['RegionID']
        champions = [row[f'Top1Champion'], row[f'Jg1Champion'], row[f'Mid1Champion'], row[f'Adc1Champion'], row[f'Supp1Champion'],
                     row[f'Top2Champion'], row[f'Jg2Champion'], row[f'Mid2Champion'], row[f'Adc2Champion'], row[f'Supp2Champion']]
        winner = row['TeamWinner']
        
        for i, champ in enumerate(champions):
            key = (champ, region)
            
            if key not in champion_regional_stats:
                champion_regional_stats[key] = {'wins': 0, 'games': 0}
            
            champion_regional_stats[key]['games'] += 1
            if ((winner == 1 and i < 5) or (winner == 2 and i >= 5)):
                champion_regional_stats[key]['wins'] += 1
                
    # Crear un diccionario de winrates
    winrates_dict = {key: stats['wins'] / stats['games'] for key, stats in champion_regional_stats.items() if stats['games'] > 0}
    
    return winrates_dict

def add_synergy_to_matches(df, synergies_dict):
    synergy_scores_team1 = []
    synergy_scores_team2 = []

    for index, row in df.iterrows():
        champions_team1 = [row[f'Top1Champion'], row[f'Jg1Champion'], row[f'Mid1Champion'], row[f'Adc1Champion'], row[f'Supp1Champion']]
        champions_team2 = [row[f'Top2Champion'], row[f'Jg2Champion'], row[f'Mid2Champion'], row[f'Adc2Champion'], row[f'Supp2Champion']]

        synergy_score_team1 = np.mean([synergies_dict.get(tuple(sorted([champions_team1[i], champions_team1[j]])), 0.5) 
                                       for i in range(len(champions_team1)) for j in range(i + 1, len(champions_team1))])
        synergy_score_team2 = np.mean([synergies_dict.get(tuple(sorted([champions_team2[i], champions_team2[j]])), 0.5) 
                                       for i in range(len(champions_team2)) for j in range(i + 1, len(champions_team2))])

        synergy_scores_team1.append(synergy_score_team1)
        synergy_scores_team2.append(synergy_score_team2)

    df['Team1_Synergy'] = synergy_scores_team1
    df['Team2_Synergy'] = synergy_scores_team2

    return df


def calculate_champion_synergy(datasheet, winrates_dict):
    champion_pairs = {}
    
    for index, row in datasheet.iterrows():
        region = row['RegionID']
        champions_team1 = [row[f'Top1Champion'], row[f'Jg1Champion'], row[f'Mid1Champion'], row[f'Adc1Champion'], row[f'Supp1Champion']]
        champions_team2 = [row[f'Top2Champion'], row[f'Jg2Champion'], row[f'Mid2Champion'], row[f'Adc2Champion'], row[f'Supp2Champion']]
        winner = row['TeamWinner']

        for team_champions in [champions_team1, champions_team2]:
            for i in range(len(team_champions)):
                for j in range(i + 1, len(team_champions)):
                    pair = tuple(sorted([team_champions[i], team_champions[j]]))
                    if pair not in champion_pairs:
                        champion_pairs[pair] = {'wins': 0, 'total_games': 0}
                    
                    if ((winner == 1 and team_champions == champions_team1) or (winner == 2 and team_champions == champions_team2)):
                        champion_pairs[pair]['wins'] += 1
                    
                    champion_pairs[pair]['total_games'] += 1

    # Convertir a diccionario de sinergias
    synergies_dict = {key: value['wins'] / value['total_games'] for key, value in champion_pairs.items() if value['total_games'] > 0}
    
    return synergies_dict


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
    df['H2H_Win_Diff'] = df['Team1_H2H_Wins'] / df['Team2_H2H_Wins'].replace(0, 1)

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

def load_champion_themes(filepath):
    with open(filepath, 'r') as file:
        champion_themes = json.load(file)
    return champion_themes

def calcular_puntajes_tematicos_para_equipo(champions_ids, champion_themes):
    
    # Inicializa el contador de temas para calcular la sinergia
    theme_counts = {}
    
    for champ_id in champions_ids:
        # Obtiene las temáticas para el campeón actual si existen
        themes = champion_themes.get(str(champ_id), [])
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    # Inicializa el puntaje de sinergia
    puntaje_sinergia = 0
    
    for theme, count in theme_counts.items():
        if theme in [5, 6]:
            puntaje_sinergia += count ** 3  # Aumenta el peso de estos temas de manera más significativa
        else:
            puntaje_sinergia += max(count - 1, 0)  # Solo suma sinergia si hay más de un campeón con el mismo tema
    
    return puntaje_sinergia


def calcular_puntajes_tematicos(df, champion_themes):
    puntajes_sinergia_equipo1 = []
    puntajes_sinergia_equipo2 = []
    tema_equipo1 = []
    tema_equipo2 = []

    for _, row in df.iterrows():
        # Inicializa diccionarios vacíos para almacenar los campeones y sus temáticas para cada equipo
        champions_team1 = {}
        champions_team2 = {}

        # Extrae los campeones y sus temáticas para el equipo 1
        for pos in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            champ_id_team1 = str(row[f'{pos}1Champion'])
            if champ_id_team1 in champion_themes:
                champions_team1[champ_id_team1] = champion_themes[champ_id_team1]

        # Realiza lo mismo para el equipo 2
        for pos in ['Top', 'Jg', 'Mid', 'Adc', 'Supp']:
            champ_id_team2 = str(row[f'{pos}2Champion'])
            if champ_id_team2 in champion_themes:
                champions_team2[champ_id_team2] = champion_themes[champ_id_team2]

        # Calcula los puntajes de variedad y sinergia para cada equipo
        puntajes_sinergia_equipo1.append(calcular_puntajes_tematicos_para_equipo(champions_team1, champion_themes))
        puntajes_sinergia_equipo2.append(calcular_puntajes_tematicos_para_equipo(champions_team2, champion_themes))
        tema_equipo1.append(calcular_tema_principal_equipo(champions_team1, champion_themes))
        tema_equipo2.append(calcular_tema_principal_equipo(champions_team2, champion_themes))

    df['PuntajeTemaEquipo1'] = puntajes_sinergia_equipo1
    df['PuntajeTemaEquipo2'] = puntajes_sinergia_equipo2
    df['TemaEquipo1'] = tema_equipo1
    df['TemaEquipo2'] = tema_equipo2

    return df

def calcular_tema_principal_equipo(champions_ids, champion_themes):
    """
    Determina el tema principal de un equipo basado en las temáticas más frecuentes de sus campeones.
    
    Args:
    - champions_ids: Lista de IDs de los campeones en el equipo.
    - champion_themes: Diccionario que mapea IDs de campeones a listas de sus temáticas.
    
    Returns:
    - ID del tema principal del equipo.
    """
    # Contador para las ocurrencias de cada tema en el equipo
    tema_contador = {}
    
    for champ_id in champions_ids:
        # Obtiene las temáticas para el campeón actual
        themes = champion_themes.get(str(champ_id), [])
        for tema in themes:
            if tema in tema_contador:
                tema_contador[tema] += 1
            else:
                tema_contador[tema] = 1

    # Encuentra el tema con la mayor cantidad de ocurrencias
    tema_principal = max(tema_contador, key=tema_contador.get, default=None)

    return tema_principal


def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    df = calculate_player_glicko_ratings(df)
    player_champion_win_rates = calculate_player_champion_win_rates(filepath)
    champion_themes = load_champion_themes('info/color_themes.json')
    calcular_puntajes_tematicos(df, champion_themes)
    df['TeamWinner'] = df['TeamWinner'] - 1
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['DaysSinceLatest'] = (df['Date'].max() - df['Date']).dt.days
    df.drop('Date', axis=1, inplace=True)
    winrate_df = calculate_accumulative_regional_winrates(df)
    champion_synergy_df = calculate_champion_synergy(df, winrate_df)
    df = add_synergy_to_matches(df, champion_synergy_df)
    numerical_features = ['PuntajeTemaEquipo1', 'PuntajeTemaEquipo2', 'Team1Glicko', 'Team2Glicko']
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('synergy', 'passthrough', ['Team1_Synergy', 'Team2_Synergy'])
    ], remainder='drop')

    

    X = df.drop('TeamWinner', axis=1)
    y = df['TeamWinner']
    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    elif hasattr(X_processed, "todense"):
        X_processed = X_processed.todense()

    dump(preprocessor, 'preprocessor.joblib')
    # Split the processed data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_processed, y, test_size=0.20, random_state=0)

    # Luego dividir entrenamiento+validación en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=0)  # 0.25 x 0.8 = 0.2
    # Convert split data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    train_data = MatchDataset(X_train_tensor, y_train_tensor)
    val_data = MatchDataset(X_val_tensor, y_val_tensor)
    test_data = MatchDataset(X_test_tensor, y_test_tensor)

    weights = len(y_train) / (2. * np.bincount(y_train))
    # Return processed and split tensors
    return train_data, test_data, val_data, weights

# Then, use the returned tensors for training and evaluating your model.

