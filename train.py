# train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.data_preprocessing import load_and_preprocess_data
from models.match_predictor_model import MatchPredictor

# Cargar y preprocesar los datos
train_data, test_data = load_and_preprocess_data('data/datasheet.csv')

# Crear DataLoaders para los conjuntos de entrenamiento y prueba
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Estos números son solo ejemplos, debes reemplazarlos por los valores reales de tu conjunto de datos
num_teams = 272
num_champions = 167
num_players = 1520
embedding_dim = 10

model = MatchPredictor(num_teams=num_teams, num_champions=num_champions, num_players=num_players, embedding_dim=embedding_dim)


# Definir la función de pérdida y el optimizador
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Configuración del dispositivo para entrenamiento (CPU o GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10  # Por ejemplo, definimos 10 épocas para el entrenamiento
for epoch in range(num_epochs):
    model.train()  # Poner el modelo en modo entrenamiento
    for batch in train_loader:
        # Desempaqueta los inputs y targets del batch
        team1_id, team2_id, champions_team1, champions_team2, players_team1, players_team2, targets = batch

        # Mueve los inputs y targets al dispositivo (GPU o CPU)
        team1_id, team2_id = team1_id.to(device), team2_id.to(device)
        champions_team1, champions_team2 = champions_team1.to(device), champions_team2.to(device)
        players_team1, players_team2 = players_team1.to(device), players_team2.to(device)
        targets = targets.to(device)

        # Paso hacia adelante
        predictions = model(team1_id, team2_id, champions_team1, champions_team2, players_team1, players_team2)
        targets = targets.long()  # Convert targets to long if they're in float

        # Calculate loss
        loss = criterion(predictions, targets)

        # Retropropagación y un paso de optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Aquí también puedes agregar código para la evaluación del modelo usando test_loader
    # ...

model.eval()  # Poner el modelo en modo de evaluación
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        # Unpack all items returned by the dataset
        team1_id_tensor, team2_id_tensor, champions_team1_tensor, champions_team2_tensor, players_team1_tensor, players_team2_tensor, labels = data
        
        # Transfer tensors to the appropriate device (e.g., GPU)
        team1_id_tensor = team1_id_tensor.to(device)
        team2_id_tensor = team2_id_tensor.to(device)
        champions_team1_tensor = champions_team1_tensor.to(device)
        champions_team2_tensor = champions_team2_tensor.to(device)
        players_team1_tensor = players_team1_tensor.to(device)
        players_team2_tensor = players_team2_tensor.to(device)
        labels = labels.to(device)
        
        # Assuming your model's forward method can take these tensors directly
        outputs = model(team1_id_tensor, team2_id_tensor, champions_team1_tensor, champions_team2_tensor, players_team1_tensor, players_team2_tensor)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(predicted)):
            print(f"Sample {i}: Predicted class index = {predicted[i].item()}, Actual class index = {labels[i].item()}")


accuracy = correct / total
print(f'Accuracy del modelo en el conjunto de prueba: {accuracy * 100:.2f}%')
# Guardar el modelo entrenado
torch.save(model.state_dict(), 'model.pth')
print('Entrenamiento completado.')
