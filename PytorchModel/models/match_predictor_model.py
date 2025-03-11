import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchPredictor(nn.Module):
    """
    A PyTorch model that predicts match outcomes given:
      1) Glicko ratings for each team,
      2) The champion picks for both teams.

    The model:
      - Embeds champion IDs into a dense representation (embedding layer).
      - Processes each champion embedding individually with its respective team's Glicko rating
        through a small feed-forward network (three layers).
      - Concatenates the processed champion representations for both teams.
      - Passes the concatenated vector through a final linear layer to produce the final output.
    """
    def __init__(self, output_dim, num_champions, embedding_dim):
        """
        Args:
            output_dim (int): The dimension of the model's output (e.g., 1 for binary classification).
            num_champions (int): Total number of possible champions (for embedding index range).
            embedding_dim (int): Dimensionality of the champion embeddings.
        """
        super(MatchPredictor, self).__init__()
        
        # Embedding layer to transform each champion ID into a dense vector of size embedding_dim
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)
        
        # Feed-forward layers for Team 1 champion embeddings
        # Append the team's Glicko rating to the champion embedding, so the input is (embedding_dim + 1)
        self.team1_fc1 = nn.Linear(embedding_dim + 1, 50)
        self.team1_fc2 = nn.Linear(50, 25)
        self.team1_fc3 = nn.Linear(25, 10) 
        
        # Feed-forward layers for Team 2 champion embeddings
        # Same dimensionality as for Team 1
        self.team2_fc1 = nn.Linear(embedding_dim + 1, 50)
        self.team2_fc2 = nn.Linear(50, 25)
        self.team2_fc3 = nn.Linear(25, 10) 
        
        # Input size = 10 units for each of the 5 champions of each team (i.e., 5*10) for Team 1
        # plus 10 units for each of the 5 champions of Team 2 -> total 100 units.
        # output_dim is 1 since this is a binary classification model
        self.final_fc = nn.Linear(10 * 10, output_dim)

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): A batch of input features of shape (batch_size, 12).
                                     Specifically:
                                     - features[:, 0]  -> Glicko rating for Team 1
                                     - features[:, 1]  -> Glicko rating for Team 2
                                     - features[:, 2:] -> Champion IDs for Team 1 and Team 2 (5 each, total 10).
        
        Returns:
            torch.Tensor: Model output of shape (batch_size, output_dim).
        """

        # Extract Glicko ratings for both teams
        glicko_team1 = features[:, 0].unsqueeze(1)  # Glicko rating for team 1
        glicko_team2 = features[:, 1].unsqueeze(1)  # Glicko rating for team 2
        
        # Extract champion indices (last 10 elements)
        champion_indices = features[:, -10:].long() 

        # Separate champion indices into two teams
        team1_indices = champion_indices[:, :5]  # First 5 champions for team 1
        team2_indices = champion_indices[:, 5:]  # Last 5 champions for team 2
        
        # Embed each Team 1 champion, then concatenate with Team 1's Glicko rating and pass through the feed-forward network
        team1_embeddings = [self.champion_embedding(team1_indices[:, i]) for i in range(5)]
        processed_team1_embeddings = []
        for embedding in team1_embeddings:
             # Concatenate champion embedding with Glicko rating
            x = torch.cat((embedding, glicko_team1), dim=1)
            # Pass through 3-layer MLP, with ReLU activation
            x = F.relu(self.team1_fc1(x))
            x = F.relu(self.team1_fc2(x))
            x = F.relu(self.team1_fc3(x)) 
            processed_team1_embeddings.append(x)
        # Combine the processed embeddings for all 5 champions into a single tensor
        processed_team1_embeddings = torch.cat(processed_team1_embeddings, dim=1)
        
        # Similarly, embed Team 2 champions, then concatenate with Team 2's Glicko rating, and pass through the feed-forward network
        team2_embeddings = [self.champion_embedding(team2_indices[:, i]) for i in range(5)]
        processed_team2_embeddings = []
        for embedding in team2_embeddings:
            x = torch.cat((embedding, glicko_team2), dim=1)
            x = F.relu(self.team2_fc1(x))
            x = F.relu(self.team2_fc2(x))
            x = F.relu(self.team2_fc3(x))
            processed_team2_embeddings.append(x)
        processed_team2_embeddings = torch.cat(processed_team2_embeddings, dim=1)
        
        # Concatenate Team 1 and Team 2 processed embeddings into a single vector
        concatenated = torch.cat((processed_team1_embeddings, processed_team2_embeddings), dim=1)
        
        # Final linear layer to produce the output
        output = self.final_fc(concatenated)
        
        return output
