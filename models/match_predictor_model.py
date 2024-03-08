import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchPredictor(nn.Module):
    def __init__(self, num_teams, num_champions, num_players, embedding_dim):
        super(MatchPredictor, self).__init__()
        # Embeddings
        
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        
        # Ejemplo de capa lineal, ajusta según tu arquitectura deseada
        self.fc = nn.Linear(embedding_dim * 6, 2)  # Asume 5 campeones por equipo + team ID * embedding_dim como tamaño de entrada, 2 para la salida (ganar/perder)
        

    def forward(self, team1_id, team2_id, champions_team1, champions_team2, players_team1, players_team2):
        # Embedding para equipos
        team1_embed = self.team_embedding(team1_id)
        team2_embed = self.team_embedding(team2_id)
        
        # Embedding para campeones
        champions_team1_embed = self.champion_embedding(champions_team1).mean(dim=1)  # Promedio de los embeddings si múltiples campeones
        champions_team2_embed = self.champion_embedding(champions_team2).mean(dim=1)
        
        # Embedding para jugadores
        players_team1_embed = self.player_embedding(players_team1).mean(dim=1)
        players_team2_embed = self.player_embedding(players_team2).mean(dim=1)
        
        # Concatenar todos los embeddings
        combined = torch.cat([team1_embed, team2_embed, champions_team1_embed, champions_team2_embed, players_team1_embed, players_team2_embed], dim=1)
        
        # Pasar a través de una capa lineal (o más, según tu diseño)
        output = self.fc(combined)
        return output
