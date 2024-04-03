import torch
import torch.nn as nn
import torch.nn.functional as F
class MatchPredictor(nn.Module):
    def __init__(self, num_features, output_dim, num_champions, embedding_dim):
        super(MatchPredictor, self).__init__()
        
        # Suponiendo que num_champions es el total de campeones únicos y embedding_dim es un hiperparámetro elegido
        self.champion_embedding = nn.Embedding(num_champions, 5)
        
        # Capas anteriores
        self.group1_fc = nn.Linear(2, 25)
        self.group2_fc = nn.Linear(2, 25) 
        self.group3_fc = nn.Linear(2, 25)
        
        # Ajusta la entrada de la capa final para incluir todos los embeddings
        self.final_fc = nn.Linear(75 + 5 * 10, output_dim) # Ajusta según sea necesario

    def forward(self, features):
        # Extrae los índices de los campeones (últimos 10 elementos)
        champion_indices = features[:, -10:].long()  # Asume que los últimos 10 son índices de campeones
        
        # Resto de las características
        other_features = features[:, :-10]  # Todo excepto los últimos 10 elementos
        
        # Procesa los índices de los campeones con embeddings
        champion_embeddings = [self.champion_embedding(champion_indices[:, i]) for i in range(10)]
        champion_embeddings = torch.cat(champion_embeddings, dim=1)
        
        group1_features = F.relu(self.group1_fc(other_features[:, 0:2]))
        group2_features = F.relu(self.group2_fc(other_features[:, 2:4]))
        group3_features = F.relu(self.group3_fc(other_features[:, 4:6]))
        
        # Concatena y pasa a través de la capa final
        concatenated = torch.cat((group1_features, group2_features, group3_features, champion_embeddings), dim=1)
        output = self.final_fc(concatenated)
        
        return output


