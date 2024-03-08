import torch
from models.match_predictor_model import MatchPredictor
def predict_model(model, device, team1_id, team2_id, champions_team1, champions_team2, players_team1, players_team2):
    """
    Predicts the outcome of matches using the trained model.

    Parameters:
    - model: The trained MatchPredictor model.
    - device: The device to perform computation on ('cpu' or 'cuda').
    - team1_id, team2_id, champions_team1, champions_team2, players_team1, players_team2: Input tensors for the model.

    Returns:
    - predictions: The predicted outcomes.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Transfer input data to the specified device
    team1_id = team1_id.to(device)
    team2_id = team2_id.to(device)
    champions_team1 = champions_team1.to(device)
    champions_team2 = champions_team2.to(device)
    players_team1 = players_team1.to(device)
    players_team2 = players_team2.to(device)

    # No gradient computation needed
    with torch.no_grad():
        outputs = model(team1_id, team2_id, champions_team1, champions_team2, players_team1, players_team2)
        _, predicted = torch.max(outputs, 1)

    return predicted


if __name__ == "__main__":
    # Load the trained model
    num_teams = 272
    num_champions = 167
    num_players = 1520
    embedding_dim = 10

    model = MatchPredictor(num_teams=num_teams, num_champions=num_champions, num_players=num_players, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    team1_id = torch.tensor([72], dtype=torch.long)  # Example tensor
    team2_id = torch.tensor([144], dtype=torch.long)
    champions_team1 = torch.tensor([[109,112,138,142,87]], dtype=torch.long)
    champions_team2 = torch.tensor([[140,104,130,73,80]], dtype=torch.long)
    players_team1 = torch.tensor([[1384,582,1401,1113,23]], dtype=torch.long)
    players_team2 = torch.tensor([[560,429,220,1268,527	]], dtype=torch.long)
    
    # Call the prediction function
    predicted_outcome = predict_model(model, device, team1_id, team2_id, champions_team1, champions_team2, players_team1, players_team2)
    if(predicted_outcome.item()==0):
        print("Predicted outcome: Blue Win")
    else:
        print("Predicted outcome: Red Win")
    