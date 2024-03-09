import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchPredictor(nn.Module):
    def __init__(self, num_teams, num_champions, num_players, num_regions, embedding_dim, num_numerical_features, output_dim):
        super(MatchPredictor, self).__init__()

        # Define embedding sizes for teams, champions, players, regions, and bans
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)  # Used for champions and bans
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)  # Embedding layer for regionID
        # Linear layer to produce output
        self.fc = nn.Linear(93, output_dim)

    def forward(self, features):

        # Extract numerical features
        numerical_features = features[:, :3].float()  # First 3 are numerical features

        # Adjust indices for ID features based on the new order, including regionID now
        team1_id = features[:, 3].long()
        team2_id = features[:, 4].long()
        region_id = features[:, 5].long()  # Handling for regionID

        # Adjusting indices due to addition of regionID
        champions_start = 6  # Starting index for champion IDs, adjusted due to addition of regionID
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

        # Pass through the linear layer
        output = self.fc(combined_features)
        return output
