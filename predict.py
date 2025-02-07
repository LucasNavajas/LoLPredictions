import torch
import json
import pandas as pd
from models.match_predictor_model import MatchPredictor
from joblib import load
import torch.nn.functional as F

def load_ids_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        ids = json.load(file)
        ids = {key.lower(): value for key, value in ids.items()}
    return ids

def predict_probabilities(model, device, all_features):
    if all_features.is_cuda:
        all_features_np = all_features.cpu().numpy()
    else:
        all_features_np = all_features.numpy()
    df = pd.DataFrame(all_features_np, columns=[
        'Team1Glicko', 'Team2Glicko',
        'Top1Champion', 'Jg1Champion', 'Mid1Champion', 'Adc1Champion', 'Supp1Champion',
        'Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion'
    ])
    preprocessor = load('preprocessor.joblib')
    df_preprocessed = preprocessor.transform(df)
    data_tensor = torch.tensor(df_preprocessed, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(data_tensor)
        probabilities = torch.sigmoid(outputs) 
    return probabilities.item()

def get_id(name, ids):
    normalized_name = name.lower()
    return ids.get(normalized_name, f"{name} not found")


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
        return 0, 0
    return ratings_sum / num_players, RD_sum / num_players 

if __name__ == "__main__":
    num_champions = 170
    embedding_dim = 10
    output_dim = 1 

    model_path = 'model.pth'
    model = MatchPredictor(output_dim, num_champions, embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    players_ids = load_ids_from_json("info/players_ids.json")
    champions_ids = load_ids_from_json("info/champions_ids.json")
    glicko_ratings = load_glicko_ratings('info/player_glicko_ratings.json')
    player_glicko_ratings = glicko_ratings['player_glicko']
    player_RD = glicko_ratings["player_RD"]

    players1 = "oscarinin,razork,humanoid,upset,mikyx"
    players1 = players1.split(",")
    players1_ids = [get_id(name, players_ids) for name in players1]

    players2 = "canna,yike,vladi,caliste,targamas"
    players2 = players2.split(",")
    players2_ids = [get_id(name, players_ids) for name in players2]

    champions1 = "ambessa,vi,aurora,kaisa,rakan"
    champions1 = champions1.split(",")
    champions1_ids = [get_id(name, champions_ids) for name in champions1]
    
    champions2 = "poppy,xin zhao,azir,ezreal,leona"
    champions2 = champions2.split(",")
    champions2_ids = [get_id(name, champions_ids) for name in champions2]

    champions_team1 = torch.tensor([champions1_ids], dtype=torch.long)
    champions_team2 = torch.tensor([champions2_ids], dtype=torch.long)

    team1_glicko_rating, team1_RD = calculate_average(players1_ids, player_glicko_ratings, player_RD)
    team2_glicko_rating, team2_RD = calculate_average(players2_ids, player_glicko_ratings, player_RD)
    team1_glicko_rating_tensor = torch.tensor([[team1_glicko_rating]], dtype=torch.float32)
    team2_glicko_rating_tensor = torch.tensor([[team2_glicko_rating]], dtype=torch.float32)

    all_features = torch.cat([
        team1_glicko_rating_tensor, team2_glicko_rating_tensor, 
        champions_team1, champions_team2
    ], dim=1)

    original_probability = predict_probabilities(model, device, all_features)
    predicted_outcome_original = "Blue Team Wins" if original_probability <= 0.5 else "Red Team Wins"

    # Swap the champions between teams
    swapped_features = torch.cat([
        team1_glicko_rating_tensor, team2_glicko_rating_tensor,
        champions_team2, champions_team1 
    ], dim=1)

    # Predict the outcome with swapped team compositions
    swapped_probability = predict_probabilities(model, device, swapped_features)
    predicted_outcome_swapped = "Blue Team Wins" if swapped_probability <= 0.5 else "Red Team Wins"

    if predicted_outcome_original == "Blue Team Wins":
        probability_difference = swapped_probability - original_probability
    else:
        probability_difference = original_probability - swapped_probability

    print(f"MODEL: {model_path}")
    # Determine if the winner is "safe"
    if predicted_outcome_original == "Blue Team Wins":
        if original_probability < swapped_probability or (original_probability < 0.5 and swapped_probability > 0.5):
            print(f"Predicted outcome: {predicted_outcome_original} (safe)")
        else:
            print(f"Predicted outcome: {predicted_outcome_original} (not safe)")
    else: 
        if original_probability > swapped_probability or (original_probability > 0.5 and swapped_probability < 0.5):
            print(f"Predicted outcome: {predicted_outcome_original} (safe)")
        else:
            print(f"Predicted outcome: {predicted_outcome_original} (not safe)")
            
    print(f"Original probability: {original_probability}")
    print(f"Swapped probability: {swapped_probability}")

    print(f"Difference between original and swapped probabilities: {probability_difference:.4f}")