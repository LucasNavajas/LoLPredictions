import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchPredictor(nn.Module):
    def __init__(self, output_dim, num_champions, embedding_dim):
        super(MatchPredictor, self).__init__()
        
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)
        
        # Layers for team 1 champions (first 5)
        self.team1_fc1 = nn.Linear(embedding_dim + 1, 50)
        self.team1_fc2 = nn.Linear(50, 25)
        self.team1_fc3 = nn.Linear(25, 10) 
        
        # Layers for team 2 champions (last 5)
        self.team2_fc1 = nn.Linear(embedding_dim + 1, 50)
        self.team2_fc2 = nn.Linear(50, 25)
        self.team2_fc3 = nn.Linear(25, 10) 
        
        self.final_fc = nn.Linear(10 * 10, output_dim)

    def forward(self, features):
        # Extract Glicko ratings (first 2 elements)
        glicko_team1 = features[:, 0].unsqueeze(1)  # Glicko rating for team 1
        glicko_team2 = features[:, 1].unsqueeze(1)  # Glicko rating for team 2
        
        # Extract champion indices (last 10 elements)
        champion_indices = features[:, -10:].long() 

        # Separate champion indices into two teams
        team1_indices = champion_indices[:, :5]  # First 5 champions for team 1
        team2_indices = champion_indices[:, 5:]  # Last 5 champions for team 2
        
        team1_embeddings = [self.champion_embedding(team1_indices[:, i]) for i in range(5)]
        processed_team1_embeddings = []
        for embedding in team1_embeddings:
            x = torch.cat((embedding, glicko_team1), dim=1)
            x = F.relu(self.team1_fc1(x))
            x = F.relu(self.team1_fc2(x))
            x = F.relu(self.team1_fc3(x)) 
            processed_team1_embeddings.append(x)
        processed_team1_embeddings = torch.cat(processed_team1_embeddings, dim=1)
        
        team2_embeddings = [self.champion_embedding(team2_indices[:, i]) for i in range(5)]
        processed_team2_embeddings = []
        for embedding in team2_embeddings:
            x = torch.cat((embedding, glicko_team2), dim=1)
            x = F.relu(self.team2_fc1(x))
            x = F.relu(self.team2_fc2(x))
            x = F.relu(self.team2_fc3(x))
            processed_team2_embeddings.append(x)
        processed_team2_embeddings = torch.cat(processed_team2_embeddings, dim=1)
        
        concatenated = torch.cat((processed_team1_embeddings, processed_team2_embeddings), dim=1)
        
        output = self.final_fc(concatenated)
        
        return output
