import torch
import json
import pandas as pd
from models.match_predictor_model import MatchPredictor
from joblib import load
import torch.nn.functional as F
import sympy as sp
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------
def load_ids_from_json(filepath):
    """
    Loads a JSON file containing mappings (e.g., names -> IDs) and converts all keys to lowercase.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: A dictionary mapping lowercase strings to IDs.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        ids = json.load(file)
        # Convert all keys to lowercase for case-insensitive lookups
        ids = {key.lower(): value for key, value in ids.items()}
    return ids

def predict_probabilities(model, device, all_features):
    """
    Given a trained model, device, and feature tensor, this function:
      1. Converts the tensor to a NumPy array if needed.
      2. Builds a DataFrame with the appropriate columns.
      3. Applies the preprocessor to transform the data.
      4. Runs a forward pass through the model.
      5. Returns the predicted probability (sigmoid of the logits).

    Args:
        model (nn.Module): The PyTorch model (MatchPredictor).
        device (torch.device): CPU or CUDA device.
        all_features (torch.Tensor): A single sample or batch of samples containing
                                     [Team1Glicko, Team2Glicko, 10 champion IDs].

    Returns:
        float: The predicted probability for the input sample (range [0, 1]).
    """
    # Move data back to CPU if needed, then convert to NumPy
    if all_features.is_cuda:
        all_features_np = all_features.cpu().numpy()
    else:
        all_features_np = all_features.numpy()

    # Build a DataFrame for consistent column naming
    df = pd.DataFrame(all_features_np, columns=[
        'Team1Glicko', 'Team2Glicko',
        'Top1Champion', 'Jg1Champion', 'Mid1Champion', 'Adc1Champion', 'Supp1Champion',
        'Top2Champion', 'Jg2Champion', 'Mid2Champion', 'Adc2Champion', 'Supp2Champion'
    ])

    # Load the saved preprocessor to transform the data accordingly
    preprocessor = load(os.path.join(BASE_DIR,'preprocessor.joblib'))
    df_preprocessed = preprocessor.transform(df)

    # Convert the preprocessed data back into a PyTorch tensor
    data_tensor = torch.tensor(df_preprocessed, dtype=torch.float32).to(device)

    model.eval() # evaluation mode
    with torch.no_grad():
        outputs = model(data_tensor) # raw logits
        probabilities = torch.sigmoid(outputs) # convert logits -> probabilities (in [0,1])

    return probabilities.item()

def get_id(name, ids):
    """
    Utility function to fetch the ID given a name, ensuring the name is lowercased
    and validating if the key exists in the 'ids' dict.

    Args:
        name (str): Name to look up.
        ids (dict): Mapping from name -> ID.

    Returns:
        ID (int): The mapped ID for the name.

    Raises:
        ValueError: If the name doesn't exist in the dict.
    """
    normalized_name = name.lower()
    if not ids.get(normalized_name):
        raise ValueError(f"{normalized_name} does not exist")
    return ids.get(normalized_name, f"{name} not found")


def load_glicko_ratings(filepath):
    """
    Load a JSON file containing player_glicko and player_RD dictionaries.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: The loaded Glicko ratings, containing:
              {
                'player_glicko': {...},
                'player_RD': {...}
              }
    """
    with open(filepath, 'r') as file:
        glicko_ratings = json.load(file)
    return glicko_ratings

def calculate_average(player_ids, player_glicko_ratings, player_RD):
    """
    Calculates the average Glicko rating and average RD for a list of player IDs.

    Args:
        player_ids (list[int]): List of player IDs.
        player_glicko_ratings (dict): Mapping from str(ID) -> rating.
        player_RD (dict): Mapping from str(ID) -> rating deviation.

    Returns:
        tuple(float, float): (average_rating, average_RD)
    """
    ratings_sum = 0
    RD_sum = 0
    num_players = 0
    for player_id in player_ids:
        # Ensure player_id is in the dictionary keys
        if str(player_id) in player_glicko_ratings:
            ratings_sum += player_glicko_ratings[str(player_id)]
            RD_sum += player_RD[str(player_id)]
            num_players += 1
            
    return ratings_sum / num_players, RD_sum / num_players 

def calculate_composition_winrates(original_prob, swapped_prob):
    """
    Uses symbolic math (Sympy) to compute the win rates of compositions
    given two probabilities: 
      - 'original_prob' (Team Red's chance of winning with original compositions),
      - 'swapped_prob'  (Team Red's chance after swapping champion picks).

    The logic is:
      - If original_prob > 0.5, ratio_1 = swapped_prob / original_prob
        else, ratio_1 = (1 - swapped_prob) / (1 - original_prob).

      - W_C_R + W_C_B = 1
      - W_C_B = ratio_1 * W_C_R
      => solve for (W_C_R, W_C_B).

    Returns:
        (float, float): The computed (Win rate for Composition C_R, Win rate for Composition C_B).
    """
    # Define symbolic variables
    W_C_R, W_C_B = sp.symbols('W_C_R W_C_B')
    
    # Determine ratio based on which team is favored initially
    if original_prob > 0.5:
        ratio_1 = swapped_prob / original_prob
    else:
        ratio_1 = (1 - swapped_prob) / (1 - original_prob)
    
    # System of equations:
    eq1 = W_C_R + W_C_B - 1  
    eq2 = W_C_B - ratio_1 * W_C_R  
    
    # Solve the system
    solution = sp.solve((eq1, eq2), (W_C_R, W_C_B))
    
    return solution[W_C_R], solution[W_C_B]

def calculate_confidence(probability):
    """
    Computes a 'confidence' measure for the predicted probability:
      - If probability < 0.5 => prediction is "Blue Team" (0),
        confidence = 1 - prob  (the closer to 1, the more confident).
      - If probability >= 0.5 => prediction is "Red Team" (1),
        confidence = prob      (the closer to 1, the more confident).
      - Scaled to a percentage.

    Args:
        probability (float): Probability that Red Team wins.

    Returns:
        float: Confidence level, as a percentage (0-100).
    """
    # Confidence is the distance from 0.5, but more simply pick the side (prob vs. 1-prob) that's bigger and multiply by 100
    confidence = max(probability, 1 - probability) * 100
    return confidence

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    # ----------------------------
    # Model and Device Setup
    # ----------------------------
    num_champions = 171
    embedding_dim = 10
    output_dim = 1 

    model_path = os.path.join(BASE_DIR, "model.pth")
    model = MatchPredictor(output_dim, num_champions, embedding_dim)
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    device = torch.device('cuda')
    model.to(device)

    # ----------------------------
    # Load IDs and Glicko Ratings
    # ----------------------------
    players_ids = load_ids_from_json(os.path.join(BASE_DIR,"info/players_ids.json"))
    champions_ids = load_ids_from_json(os.path.join(BASE_DIR,"info/champions_ids.json"))
    glicko_ratings = load_glicko_ratings(os.path.join(BASE_DIR,'info/player_glicko_ratings.json'))
    player_glicko_ratings = glicko_ratings['player_glicko']
    player_RD = glicko_ratings["player_RD"]

    # ----------------------------
    # Example Inputs
    # ----------------------------
    players1 = "siwoo,lucid,showmaker,aiming,beryl"
    players1 = players1.split(",")
    players1_ids = [get_id(name, players_ids) for name in players1]

    players2 = "kiin,canyon,chovy,ruler,duro"
    players2 = players2.split(",")
    players2_ids = [get_id(name, players_ids) for name in players2]

    champions1 = "jayce,xin zhao,galio,zeri,rakan"
    champions1 = champions1.split(",")
    champions1_ids = [get_id(name, champions_ids) for name in champions1]
    
    champions2 = "rumble,vi,sylas,ezreal,alistar"
    champions2 = champions2.split(",")
    champions2_ids = [get_id(name, champions_ids) for name in champions2]

    # Convert champion IDs to tensors
    champions_team1 = torch.tensor([champions1_ids], dtype=torch.long)
    champions_team2 = torch.tensor([champions2_ids], dtype=torch.long)

    # ----------------------------
    # Compute Team-Average Glicko
    # ----------------------------
    team1_glicko_rating, team1_RD = calculate_average(players1_ids, player_glicko_ratings, player_RD)
    team2_glicko_rating, team2_RD = calculate_average(players2_ids, player_glicko_ratings, player_RD)

    # Create tensors for Glicko ratings
    team1_glicko_rating_tensor = torch.tensor([[team1_glicko_rating]], dtype=torch.float32)
    team2_glicko_rating_tensor = torch.tensor([[team2_glicko_rating]], dtype=torch.float32)

    # ----------------------------
    # Combine into a single input feature tensor (original scenario)
    # ----------------------------
    all_features = torch.cat([
        team1_glicko_rating_tensor, team2_glicko_rating_tensor, 
        champions_team1, champions_team2
    ], dim=1)

    # ----------------------------
    # Predict Original Probability
    # ----------------------------
    original_probability = predict_probabilities(model, device, all_features)
    predicted_outcome_original = "Blue Team Wins" if original_probability <= 0.5 else "Red Team Wins"

    # ----------------------------
    # Swapped Scenario
    # Swap champion picks between teams
    # ----------------------------
    swapped_features = torch.cat([
        team1_glicko_rating_tensor, team2_glicko_rating_tensor,
        champions_team2, champions_team1 
    ], dim=1)

    # Predict the new probability with swapped compositions
    swapped_probability = predict_probabilities(model, device, swapped_features)

    # Compute the adjusted win rates for Composition C_R and Composition C_B, where C_R is the original winner and C_B the original loser
    winrate_C_R, winrate_C_B = calculate_composition_winrates(original_probability, swapped_probability)

    # Compute the confidence of the original probability
    confidence_original = calculate_confidence(original_probability)

    # ----------------------------
    # Print Results
    # ----------------------------
    print(original_probability)
    print(swapped_probability)
    print(f"Predicted outcome: {predicted_outcome_original}" )
    print(f"Confidence Level for Original Probability: {confidence_original:.2f}%")
    print(f"Win rate of Composition C_R: {winrate_C_R:.2%}")
    print(f"Win rate of Composition C_B: {winrate_C_B:.2%}")