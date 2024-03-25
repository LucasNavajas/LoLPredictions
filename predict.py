import torch
import json
import pandas as pd
from models.match_predictor_model import MatchPredictor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from utils.data_preprocessing import calculate_team_win_rates
from joblib import dump, load
import torch.nn.functional as F


def load_ids_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        ids = json.load(file)
        ids = {key.lower(): value for key, value in ids.items()}
    return ids

def load_h2h_win_rates(filepath):
    with open(filepath, 'r') as file:
        h2h_win_rates = json.load(file)
    return h2h_win_rates

def get_h2h_win_rate(team1_id, team2_id, h2h_win_rates):
    teams = str(team1_id)+"-"+str(team2_id)
    teams2 = str(team2_id)+"-"+str(team1_id)
    if h2h_win_rates.get(str(teams), 0.5)!=0.5:
        return h2h_win_rates.get(str(teams), 0.5)
    elif h2h_win_rates.get(str(teams2), 0.5)!=0.5:
        return h2h_win_rates.get(str(teams2), 0.5)
    else:
        return 0.5
def load_champion_themes(filepath):
    with open(filepath, 'r') as file:
        champion_themes = json.load(file)
    return champion_themes

def get_id(name, ids):
    # Normalize the region name to lower case to ensure case-insensitivity
    normalized_name = name.lower()
    # Get the region ID, return None or a custom message if not found
    return ids.get(normalized_name, f"{name} not found")

def calculate_specific_player_champion_win_rate(datasheet_path, player_id, champion_id):
    """
    Calculates the win rate for a given player-champion combination across all matches, 
    independent of whether the player was on team 1 or team 2.
    
    Args:
    - datasheet_path: Path to the dataset CSV file.
    - player_id: The player ID.
    - champion_id: The champion ID.
    
    Returns:
    - The win rate as a float.
    """
    datasheet = pd.read_csv(datasheet_path)

    # Filter records for matches where the player played the specified champion on either team
    player_champion_matches = datasheet[((datasheet['Top1ID'] == player_id) & (datasheet['Top1Champion'] == champion_id)) |
                                        ((datasheet['Jg1ID'] == player_id) & (datasheet['Jg1Champion'] == champion_id)) |
                                        ((datasheet['Mid1ID'] == player_id) & (datasheet['Mid1Champion'] == champion_id)) |
                                        ((datasheet['Adc1ID'] == player_id) & (datasheet['Adc1Champion'] == champion_id)) |
                                        ((datasheet['Supp1ID'] == player_id) & (datasheet['Supp1Champion'] == champion_id)) |
                                        ((datasheet['Top2ID'] == player_id) & (datasheet['Top2Champion'] == champion_id)) |
                                        ((datasheet['Jg2ID'] == player_id) & (datasheet['Jg2Champion'] == champion_id)) |
                                        ((datasheet['Mid2ID'] == player_id) & (datasheet['Mid2Champion'] == champion_id)) |
                                        ((datasheet['Adc2ID'] == player_id) & (datasheet['Adc2Champion'] == champion_id)) |
                                        ((datasheet['Supp2ID'] == player_id) & (datasheet['Supp2Champion'] == champion_id))]

    # Calculate wins. A win occurs when the player's team (team 1 or team 2) is the winner.
    wins = player_champion_matches[((player_champion_matches['TeamWinner'] == 1) & 
                                    ((player_champion_matches['Top1ID'] == player_id) | 
                                     (player_champion_matches['Jg1ID'] == player_id) | 
                                     (player_champion_matches['Mid1ID'] == player_id) | 
                                     (player_champion_matches['Adc1ID'] == player_id) | 
                                     (player_champion_matches['Supp1ID'] == player_id))) |
                                   ((player_champion_matches['TeamWinner'] == 2) & 
                                    ((player_champion_matches['Top2ID'] == player_id) | 
                                     (player_champion_matches['Jg2ID'] == player_id) | 
                                     (player_champion_matches['Mid2ID'] == player_id) | 
                                     (player_champion_matches['Adc2ID'] == player_id) | 
                                     (player_champion_matches['Supp2ID'] == player_id)))].shape[0]

    total_games = player_champion_matches.shape[0]

    # Calculate win rate, defaulting to 0.5 if there are no matches
    return 0.5 if total_games == 0 else wins / total_games

def load_champion_synergies(filepath):
    with open(filepath, 'r') as file:
        champion_synergies = json.load(file)
    return champion_synergies

def calculate_team_synergy(champions_ids, champion_synergies, region):
    champions_ids = champions_ids.squeeze(0).numpy()
    synergy_score = 0
    num_pairs = 0

    # Asume que 'region' es una variable que contiene el identificador de la región actual (como 'NA', 'EUW', etc.)
    for i in range(len(champions_ids)):
        for j in range(i + 1, len(champions_ids)):
            pair_key = f"{champions_ids[i]}-{champions_ids[j]}-{region}" if champions_ids[i] < champions_ids[j] else f"{champions_ids[j]}-{champions_ids[i]}-{region}"
            synergy = champion_synergies.get(pair_key, 0.5)  # Default to 0.5 if no data available
            synergy_score += synergy
            num_pairs += 1

    return synergy_score / num_pairs if num_pairs > 0 else 0.5

def load_glicko_ratings(filepath):
    with open(filepath, 'r') as file:
        glicko_ratings = json.load(file)
    return glicko_ratings

def calculate_average(player_ids, player_glicko_ratings, player_RD):
    ratings_sum = 0
    RD_sum = 0
    num_players = 0
    for player_id in player_ids:
        if str(player_id) in player_glicko_ratings:
            ratings_sum += player_glicko_ratings[str(player_id)]
            RD_sum += player_RD[str(player_id)]
            num_players += 1
    if num_players == 0:
        return 0, 0  # Return 0 for both if no players found
    return ratings_sum / num_players, RD_sum / num_players 

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

def predict_model(model, device, all_features):
    # Convierte los tensores de PyTorch a un DataFrame de Pandas
    if all_features.is_cuda:
        all_features_np = all_features.cpu().numpy()
    else:
        all_features_np = all_features.numpy()
    df = pd.DataFrame(all_features_np, columns=['Team1_Synergy',  'Team2_Synergy','PuntajeTemaEquipo1', 'PuntajeTemaEquipo2', 'Team1Glicko', 'Team2Glicko'])
    preprocessor = load('preprocessor.joblib')

    # Aplica el preprocessor cargado para transformar los datos
    df_preprocessed = preprocessor.transform(df)
    # Convierte los datos transformados de nuevo a un tensor PyTorch
    data_tensor = torch.tensor(df_preprocessed, dtype=torch.float32).to(device)
    # Asegura que el modelo esté en modo evaluación y desactiva el cálculo de gradientes
    model.eval()
    with torch.no_grad():
        outputs = model(data_tensor)
        probabilidades = F.softmax(outputs, dim=1)
        print(f"Probabilidad de Blue: {probabilidades[0][0].numpy()*100}%")
        print(f"Probabilidad de Red: {probabilidades[0][1].numpy()*100}%")
        _, predicted = torch.max(outputs, 1)

    return predicted


if __name__ == "__main__":
    # Configuration and model parameters
    num_teams = 283
    num_champions = 168
    num_players = 1554
    num_regions = 31
    embedding_dim = 10
    num_numerical_features = 6
    output_dim = 2  # Assuming binary classification for win/lose
    num_themes = 7

    # Load the trained model
    model_path = 'model.pth'
    model = MatchPredictor(num_numerical_features, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cpu')
    model.to(device)

    region_ids = load_ids_from_json('info/region_ids.json')
    players_ids = load_ids_from_json("info/players_ids.json")
    champions_ids = load_ids_from_json("info/champions_ids.json")
    teams_ids = load_ids_from_json("info/teams_ids.json")
    glicko_ratings = load_glicko_ratings('info/player_glicko_ratings.json')
    player_glicko_ratings = glicko_ratings['player_glicko']
    player_RD = glicko_ratings["player_RD"]

    region = "lpl"
    region_id = get_id(region, region_ids)

    team1_name = "thundertalk gaming"
    team1 = get_id(team1_name, teams_ids)

    team2_name = "funplus phoenix"
    team2 = get_id(team2_name, teams_ids)

    players1 = "hoya,beichuan,ucal,1xn,qiuqiu"
    players1 = players1.split(",")
    players1_ids = [get_id(name, players_ids) for name in players1]

    players2 = "xiaolaohu,milkyway,care,deokdam,life"
    players2 = players2.split(",")
    players2_ids = [get_id(name, players_ids) for name in players2]

    champions1 = "renekton,poppy,aurelion sol,varus,renata glasc"
    champions1 = champions1.split(",")
    champions1_ids = [get_id(name, champions_ids) for name in champions1]
    
    champions2 = "jayce,graves,veigar,senna,nautilus"
    champions2 = champions2.split(",")
    champions2_ids = [get_id(name, champions_ids) for name in champions2]


    team_win_rates = calculate_team_win_rates('data/datasheetv2.csv')
    # Example inputs for prediction
    # Note: These values should be properly preprocessed to match your training data
    team1_id = torch.tensor([[team1]], dtype=torch.long)
    team2_id = torch.tensor([[team2]], dtype=torch.long)
    champions_team1 = torch.tensor([champions1_ids], dtype=torch.long)
    champions_team2 = torch.tensor([champions2_ids], dtype=torch.long)
    players_team1 = torch.tensor([players1_ids], dtype=torch.long)	
    players_team2 = torch.tensor([players2_ids], dtype=torch.long)
    champion_synergies = load_champion_synergies('info/team_synergies_by_region.json')

    # Calculate team synergies
    team1_synergy = calculate_team_synergy(champions_team1, champion_synergies, region_id)
    team2_synergy = calculate_team_synergy(champions_team2, champion_synergies, region_id)
    # Convert to tensor and add to numerical_features for prediction
    team1_synergy_tensor = torch.tensor([[team1_synergy]], dtype=torch.float32)
    team2_synergy_tensor = torch.tensor([[team2_synergy]], dtype=torch.float32)

    team1_glicko_rating, team1_RD = calculate_average(players1_ids, player_glicko_ratings, player_RD)
    team2_glicko_rating, team2_RD = calculate_average(players2_ids, player_glicko_ratings, player_RD)
    team1_glicko_rating_tensor = torch.tensor([[team1_glicko_rating]], dtype=torch.float32)
    team2_glicko_rating_tensor = torch.tensor([[team2_glicko_rating]], dtype=torch.float32)

    # Asumiendo que `champion_themes` ya está cargado
    champion_themes = load_champion_themes('info/color_themes.json')

    # Calcula los puntajes de tema para cada equipo
    puntaje_tema_equipo1 = calcular_puntajes_tematicos_para_equipo(champions1_ids, champion_themes)
    puntaje_tema_equipo2 = calcular_puntajes_tematicos_para_equipo(champions2_ids, champion_themes)

    # Convertir los puntajes de tema a tensores
    puntaje_tema_equipo1_tensor = torch.tensor([[puntaje_tema_equipo1]], dtype=torch.float32)
    puntaje_tema_equipo2_tensor = torch.tensor([[puntaje_tema_equipo2]], dtype=torch.float32)

    tema_equipo1 = calcular_tema_principal_equipo(champions1_ids, champion_themes)
    tema_equipo1_tensor = torch.tensor([tema_equipo1], dtype=torch.long)	
    tema_equipo2 = calcular_tema_principal_equipo(champions2_ids, champion_themes)
    tema_equipo2_tensor = torch.tensor([tema_equipo2], dtype=torch.long)


    print("-------------------------------------------------------------------------------------------------------------------------")
    print(f"Blue Team: {team1_name} id: {team1}")
    print(f"Blue Team Players: {players1} ids: {players1_ids}")
    print(f"Blue Team Champions: {champions1} ids: {champions1_ids}")
    print(f"Blue Team Synergy: {team1_synergy}")
    print(f"Blue Team Glicko: {team1_glicko_rating}")
    print(f"Blue Team Theme Points: {puntaje_tema_equipo1}")
    print(f"Blue Theme: {tema_equipo1}")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print(f"Red Team: {team2_name} id: {team2}")
    print(f"Red Team Players: {players2} ids: {players2_ids}")
    print(f"Red Team Champions: {champions2} ids: {champions2_ids}")
    print(f"Red Team Synergy: {team2_synergy}")
    print(f"Red Team Glicko: {team2_glicko_rating}")
    print(f"Red Team Theme Points: {puntaje_tema_equipo2}")
    print(f"Red Team Theme: {tema_equipo2}")
    
    # Concatenate the tensors to form the complete numerical_features tensor
    all_features = torch.cat([
                                    #additional_numerical_features_tensor, 
                                    team1_synergy_tensor,team2_synergy_tensor,
                                    puntaje_tema_equipo1_tensor, puntaje_tema_equipo2_tensor,
                                    team1_glicko_rating_tensor, team2_glicko_rating_tensor], dim=1)
    # Call the prediction function
    predicted_outcome = predict_model(model, device,all_features)
    outcome = f"{team1_name} (Blue Team) Wins" if predicted_outcome.item() == 0 else f"{team2_name} (Red Team) Wins"
    print(f"Predicted outcome: {outcome}")
