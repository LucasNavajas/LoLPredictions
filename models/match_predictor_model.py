import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchPredictor(nn.Module):
    def __init__(self, num_teams, num_champions, num_players, num_regions, embedding_dim, num_numerical_features, output_dim, dropout_rate=0.9):
        super(MatchPredictor, self).__init__()

        # Define embedding sizes for teams, champions, players, regions, and bans
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)  # Used for champions and bans
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the total input size for the linear layer
        total_input_size = num_numerical_features + (4 * embedding_dim) + (2 * 20 * embedding_dim / 5) -30 # Adjusted formula based on your structure

        # Linear layer to produce output
        self.fc = nn.Linear(int(total_input_size), output_dim)

    def forward(self, features):
        numerical_features = features[:, :8].float()

        # Adjust indices for ID features based on the new order, including regionID now
        team1_id = features[:, 8].long()
        team2_id = features[:, 9].long()
        region_id = features[:, 10].long()  # Handling for regionID

        # Adjusting indices due to addition of regionID
        champions_start = 11  # Starting index for champion IDs, adjusted due to addition of regionID
        champions_end = champions_start + 10
        champions_team1 = features[:, champions_start:champions_start+5].long()
        champions_team2 = features[:, champions_start+5:champions_end].long()

         # Adjusting for bans following players
        bans_start = champions_end
        bans_end = bans_start + 10
        bans_team1 = features[:, bans_start:bans_start+5].long()
        bans_team2 = features[:, bans_start+5:bans_end].long()

        # Adjusting for player IDs following champions
        players_start = bans_end
        players_end = players_start + 10
        players_team1 = features[:, players_start:players_start+5].long()
        players_team2 = features[:, players_start+5:players_end].long()

        # Embedding layers for all features
        team1_embed = self.team_embedding(team1_id).squeeze(1)
        team2_embed = self.team_embedding(team2_id).squeeze(1)
        region_embed = self.region_embedding(region_id).squeeze(1)  # Embedding for regionID

        champions_team1_embed = self.champion_embedding(champions_team1).mean(dim=1)
        champions_team2_embed = self.champion_embedding(champions_team2).mean(dim=1)

        players_team1_embed = self.player_embedding(players_team1).mean(dim=1)
        players_team2_embed = self.player_embedding(players_team2).mean(dim=1)

        bans_team1_embed = self.champion_embedding(bans_team1).mean(dim=1)
        bans_team2_embed = self.champion_embedding(bans_team2).mean(dim=1)

        # Combine all embeddings and numerical features
        combined_features = torch.cat([
            numerical_features,
            team1_embed, team2_embed, region_embed,  # Include region_embed here
            champions_team1_embed, champions_team2_embed,
            players_team1_embed, players_team2_embed,
            bans_team1_embed, bans_team2_embed
        ], dim=1)
        
        combined_features = self.dropout(combined_features)
        # Pass through the linear layer
        output = self.fc(combined_features)
        return output
