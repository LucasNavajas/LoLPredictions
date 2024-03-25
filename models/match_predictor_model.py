import torch
import torch.nn as nn
import torch.nn.functional as F
class MatchPredictor(nn.Module):
    def __init__(self, num_features, output_dim):
        super(MatchPredictor, self).__init__()
        
        # Define capas dedicadas para cada grupo de características
        self.group1_fc = nn.Linear(2, 200)  # Alta prioridad
        self.group2_fc = nn.Linear(2, 2)  # Media prioridad
        self.group3_fc = nn.Linear(2, 1)  # Baja prioridad
        
        # Capa final para clasificación
        self.final_fc = nn.Linear(203, output_dim)  # 7 = 4 + 2 + 1, salida de cada grupo

    def forward(self, features):
        # Procesa cada grupo de características
        group1_features = F.relu(self.group1_fc(features[:, 0:2]))
        group2_features = F.relu(self.group2_fc(features[:, 2:4]))
        group3_features = F.relu(self.group3_fc(features[:, 4:6]))
        
        # Concatena las salidas de las capas
        concatenated = torch.cat((group1_features, group2_features, group3_features), dim=1)
        
        # Pasa las características concatenadas a través de la capa final
        output = self.final_fc(concatenated)
        
        return output


