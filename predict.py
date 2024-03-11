import torch
from models.match_predictor_model import MatchPredictor
from utils.data_preprocessing import calculate_team_win_rates

def predict_model(model, device, team1_id, team2_id, region_id, champions_team1, champions_team2, players_team1, players_team2, bans_team1, bans_team2, numerical_features):
    """
    Predicts the outcome of matches using the trained model.

    Parameters:
    - model: The trained MatchPredictor model.
    - device: The device to perform computation on ('cpu' or 'cuda').
    - team1_id, team2_id, region_id, champions_team1, champions_team2, players_team1, players_team2, bans_team1, bans_team2, team1_win_rate, team2_win_rate, numerical_features: Input tensors for the model.

    Returns:
    - predictions: The predicted outcomes.
    """
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
    num_numerical_features = 3
    output_dim = 2  # Assuming binary classification for win/lose

    # Load the trained model
    model_path = 'model.pth'
    model = MatchPredictor(num_teams, num_champions, num_players, num_regions, embedding_dim, num_numerical_features, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cpu')
    model.to(device)

    team_win_rates = calculate_team_win_rates('data/datasheetv2.csv')
    # Example inputs for prediction
    # Note: These values should be properly preprocessed to match your training data
    team1_id = torch.tensor([[26]], dtype=torch.long)
    team2_id = torch.tensor([[96]], dtype=torch.long)
    region_id = torch.tensor([[10]], dtype=torch.long)  # Example Region ID
    champions_team1 = torch.tensor([[20,91,94,163,10				]], dtype=torch.long)
    champions_team2 = torch.tensor([[1,49,2,123,44				]], dtype=torch.long)
    players_team1 = torch.tensor([[134,135,136,137,138				]], dtype=torch.long)	
    players_team2 = torch.tensor([[459,460,461,462,463				]], dtype=torch.long)
    bans_team1 = torch.tensor([[113,56,69,109,106				]], dtype=torch.long)
    bans_team2 = torch.tensor([[87,130,135,55,142				]], dtype=torch.long)
    # Example starting numerical_features tensor
    days_since_latest = torch.tensor([[0]], dtype=torch.long)  # Assume batch_size=1 for simplicity

    # Win rates calculated as per your code snippet
    team1_win_rate = torch.tensor([[team_win_rates.loc[team_win_rates['team_id'] == 26, 'win_rate'].iloc[0]]], dtype=torch.float32)
    team2_win_rate = torch.tensor([[team_win_rates.loc[team_win_rates['team_id'] == 96, 'win_rate'].iloc[0]]], dtype=torch.float32)

    # Concatenate the tensors to form the complete numerical_features tensor
    numerical_features = torch.cat([days_since_latest, team1_win_rate, team2_win_rate], dim=1)
    # Call the prediction function
    predicted_outcome = predict_model(model, device, team1_id, team2_id, region_id, champions_team1, champions_team2, bans_team1, bans_team2, players_team1, players_team2, numerical_features)
    outcome = "Blue Team Wins" if predicted_outcome.item() == 0 else "Red Team Wins"
    print(f"Predicted outcome: {outcome}")
