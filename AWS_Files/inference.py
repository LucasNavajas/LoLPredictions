import os
import torch
import json
import pandas as pd
from models.match_predictor_model import MatchPredictor
from joblib import load

def model_fn(model_dir):
    num_champions = 169
    embedding_dim = 10
    output_dim = 1 
    model = MatchPredictor(output_dim, num_champions, embedding_dim)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=torch.device('cpu')))
    return model

# Function to handle input data
def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        data = json.loads(request_body)
        
        # Extract necessary information from the input data
        players1 = data["players1"]
        players2 = data["players2"]
        champions1 = data["champions1"]
        champions2 = data["champions2"]
        
        # Load necessary resources
        players_ids = load_ids_from_json("info/players_ids.json")
        champions_ids = load_ids_from_json("info/champions_ids.json")
        glicko_ratings = load_glicko_ratings('info/player_glicko_ratings.json')
        
        # Get Glicko ratings and champion IDs
        player_glicko_ratings = glicko_ratings['player_glicko']
        player_RD = glicko_ratings["player_RD"]

        players1_ids = [get_id(name, players_ids) for name in players1]
        players2_ids = [get_id(name, players_ids) for name in players2]
        champions1_ids = [get_id(name, champions_ids) for name in champions1]
        champions2_ids = [get_id(name, champions_ids) for name in champions2]

        # Calculate average Glicko ratings for teams
        team1_glicko_rating, team1_RD = calculate_average(players1_ids, player_glicko_ratings, player_RD)
        team2_glicko_rating, team2_RD = calculate_average(players2_ids, player_glicko_ratings, player_RD)

        # Convert data to tensors
        champions_team1 = torch.tensor([champions1_ids], dtype=torch.long)
        champions_team2 = torch.tensor([champions2_ids], dtype=torch.long)
        team1_glicko_rating_tensor = torch.tensor([[team1_glicko_rating]], dtype=torch.float32)
        team2_glicko_rating_tensor = torch.tensor([[team2_glicko_rating]], dtype=torch.float32)

        # Concatenate features
        all_features = torch.cat([
            team1_glicko_rating_tensor, team2_glicko_rating_tensor,
            champions_team1, champions_team2
        ], dim=1)

        return all_features
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Function to perform prediction
def predict_fn(input_data, model):
    device = torch.device('cpu')
    model.to(device)
    preprocessor = load('preprocessor.joblib')
    
    # Preprocess input data
    input_data_np = input_data.numpy()
    df = pd.DataFrame(input_data_np, columns=[
        'Team1Glicko', 'Team2Glicko',
        'Top1Champion', 'Jg1Champion', 'Mid1Champion', 'Adc1Champion', 'Supp1Champion',
        'Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion'
    ])
    df_preprocessed = preprocessor.transform(df)
    data_tensor = torch.tensor(df_preprocessed, dtype=torch.float32).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(data_tensor)
        probabilities = torch.sigmoid(outputs)
    return probabilities.item()

# Function to format the output
def output_fn(prediction, content_type='application/json'):
    if content_type == 'application/json':
        return json.dumps({"prediction": prediction})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Utility functions
def load_ids_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        ids = json.load(file)
        ids = {key.lower(): value for key, value in ids.items()}
    return ids

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