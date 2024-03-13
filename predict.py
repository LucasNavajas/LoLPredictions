import torch
import json
import pandas as pd
from models.match_predictor_model import MatchPredictor
from utils.data_preprocessing import calculate_team_win_rates


def load_ids_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        ids = json.load(file)
        ids = {key.lower(): value for key, value in ids.items()}
    return ids


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
    return 0.6 if total_games == 0 else wins / total_games



def predict_model(model, device, team1_id, team2_id, region_id, champions_team1, champions_team2, players_team1, players_team2, bans_team1, bans_team2, numerical_features):

    # Ensure model is in evaluation mode
    model.eval()

    # Combine all inputs into a single features tensor
    # Note: Adjust the concatenation based on the exact structure your model expects
    features = torch.cat([numerical_features, team1_id, team2_id, region_id, champions_team1, champions_team2, players_team1, players_team2, bans_team1, bans_team2], dim=1)

    # Transfer input data to the specified device
    features = features.to(device)

    # No gradient computation needed
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

    return predicted


if __name__ == "__main__":
    # Configuration and model parameters
    num_teams = 283
    num_champions = 168
    num_players = 1543
    num_regions = 31
    embedding_dim = 10
    num_numerical_features = 12
    output_dim = 2  # Assuming binary classification for win/lose

    # Load the trained model
    model_path = 'model.pth'
    model = MatchPredictor(num_teams, num_champions, num_players, num_regions, embedding_dim, num_numerical_features, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cpu')
    model.to(device)

    region_ids = load_ids_from_json('info/region_ids.json')
    players_ids = load_ids_from_json("info/players_ids.json")
    champions_ids = load_ids_from_json("info/champions_ids.json")
    teams_ids = load_ids_from_json("info/teams_ids.json")

    region_name = "lpl"
    region = get_id(region_name, region_ids)

    team1_name = "weibo gaming"
    team1 = get_id(team1_name, teams_ids)

    team2_name = "edward gaming"
    team2 = get_id(team2_name, teams_ids)

    players1 = "zdz,xiaohao,xiaohu,light,crisp"
    players1 = players1.split(",")
    players1_ids = [get_id(name, players_ids) for name in players1]

    players2 = "ale,monki,fisher,thesnake,vampire"
    players2 = players2.split(",")
    players2_ids = [get_id(name, players_ids) for name in players2]

    champions1 = "camille,xin zhao,ahri,twisted fate,alistar"
    champions1 = champions1.split(",")
    champions1_ids = [get_id(name, champions_ids) for name in champions1]
    
    champions2 = "jayce,volibear,orianna,senna,nautilus"
    champions2 = champions2.split(",")
    champions2_ids = [get_id(name, champions_ids) for name in champions2]

    bans1 = "varus,rakan,karma,neeko,jax"
    bans1 = bans1.split(",")
    bans1_ids = [get_id(name, champions_ids) for name in bans1]

    bans2 = "kalista,lucian,smolder,zeri,rell"
    bans2 = bans2.split(",")
    bans2_ids = [get_id(name, champions_ids) for name in bans2] 

    team_win_rates = calculate_team_win_rates('data/datasheetv2.csv')
    # Example inputs for prediction
    # Note: These values should be properly preprocessed to match your training data
    team1_id = torch.tensor([[team1]], dtype=torch.long)
    team2_id = torch.tensor([[team2]], dtype=torch.long)
    region_id = torch.tensor([[region]], dtype=torch.long)
    champions_team1 = torch.tensor([champions1_ids], dtype=torch.long)
    champions_team2 = torch.tensor([champions2_ids], dtype=torch.long)
    players_team1 = torch.tensor([players1_ids], dtype=torch.long)	
    players_team2 = torch.tensor([players2_ids], dtype=torch.long)
    bans_team1 = torch.tensor([bans1_ids], dtype=torch.long)
    bans_team2 = torch.tensor([bans2_ids], dtype=torch.long)

    # Win rates calculated as per your code snippet
    team1_win_rate = torch.tensor([[team_win_rates.loc[team_win_rates['team_id'] == team1, 'win_rate'].iloc[0]]], dtype=torch.float32)
    team2_win_rate = torch.tensor([[team_win_rates.loc[team_win_rates['team_id'] == team2, 'win_rate'].iloc[0]]], dtype=torch.float32)

    print(f"Region: {region_name} id: {region}")
    print(f"Blue Team: {team1_name} id: {team1}")
    print(f"Blue Team Players: {players1} ids: {players1_ids}")
    print(f"Blue Team Champions: {champions1} ids: {champions1_ids}")
    print(f"Blue Team Bans: {bans1} ids: {bans1_ids}")
    print(f"Blue Team Winrate: {team1_win_rate}")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print(f"Red Team: {team2_name} id: {team2}")
    print(f"Red Team Players: {players2} ids: {players2_ids}")
    print(f"Red Team Champions: {champions2} ids: {champions2_ids}")
    print(f"Blue Team Bans: {bans2} ids: {bans2_ids}")
    print(f"Red Team Winrate: {team2_win_rate}")
    

    datasheet_path = 'data/datasheetv2.csv'
    roles = ['Top', 'Jg', 'Mid', 'Adc', 'Supp']
    additional_numerical_features = []
    # Ensure champions_team1 and players_team1 are tensors with shape [1, 5] or similar
    for (player_id, champion_id, role) in zip(players_team1.squeeze().tolist(), champions_team1.squeeze().tolist(), roles):
        win_rate = calculate_specific_player_champion_win_rate(datasheet_path, player_id=player_id, champion_id=champion_id)
        additional_numerical_features.append(win_rate)

    # Assuming a similar structure for team 2 and repeating the process
    for (player_id, champion_id, role) in zip(players_team2.squeeze().tolist(), champions_team2.squeeze().tolist(), roles):
        win_rate = calculate_specific_player_champion_win_rate(datasheet_path, player_id=player_id, champion_id=champion_id)
        additional_numerical_features.append(win_rate)

    # Convert the list of win rates to a tensor and ensure it has the correct shape
    additional_numerical_features_tensor = torch.tensor(additional_numerical_features, dtype=torch.float32).view(1, 10)  # Adjust the shape as necessary

    # Concatenate the tensors to form the complete numerical_features tensor
    numerical_features = torch.cat([team1_win_rate, team2_win_rate, additional_numerical_features_tensor], dim=1)
    # Call the prediction function
    predicted_outcome = predict_model(model, device, team1_id, team2_id, region_id, champions_team1, champions_team2, bans_team1, bans_team2, players_team1, players_team2, numerical_features)
    outcome = f"{team1_name} (Blue Team) Wins" if predicted_outcome.item() == 0 else f"{team2_name} (Red Team) Wins"
    print(f"Predicted outcome: {outcome}")
