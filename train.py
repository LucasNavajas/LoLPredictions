import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.data_preprocessing import load_and_preprocess_data
from models.match_predictor_model import MatchPredictor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# Fijar la semilla para la reproducibilidad
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Además, si estás utilizando CUDA (PyTorch con GPU), también debes fijar la semilla para CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU.
    # Asegurarse de que PyTorch pueda reproducir resultados en GPU con la misma semilla
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_wrapper(x):
    # Convert the input from a NumPy array to a PyTorch tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Move the tensor to the same device as the model
    x_tensor = x_tensor.to(device)
    
    # Get the model's output and detach it from the computation graph
    with torch.no_grad():
        output = model(x_tensor).detach().cpu().numpy()
    
    return output


# Load and preprocess data
train_dataset, test_dataset, val_dataset, weights = load_and_preprocess_data('data/datasheetv2.csv')

# Create DataLoaders for training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

num_teams = 283
num_champions = 168
num_players = 1554
num_regions = 31
embedding_dim = 10
num_themes = 7
num_numerical_features = 6
output_dim = 2  # Assuming binary classification for win/lose

model = MatchPredictor(num_numerical_features, output_dim, num_champions, embedding_dim)


weights = torch.tensor([1.0, 1.1])  # Aumenta el peso de la clase 0
class_weights = torch.FloatTensor(weights).cpu()

# Cuando inicialices tu criterio de pérdida, pasa los pesos
criterion = nn.CrossEntropyLoss(class_weights)
optimizer = Adam(model.parameters(), lr=0.001)

# Setup device for training (GPU or CPU)
device = torch.device('cpu')
model.to(device)

# Number of epochs
num_epochs = 5

best_val_loss = np.inf
patience = 100  # Number of epochs to wait for improvement before stopping
patience_counter = 0
accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for features, labels in train_loader:
        # Move inputs and targets to the device (GPU or CPU)
        features = features.to(device)
        labels = labels.to(device)
        # Forward pass
        predictions = model(features)
        labels = labels.long()  # Ensure targets are of type long

        # Calculate loss
        loss = criterion(predictions, labels)

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation loop inside the epoch loop to calculate accuracy per epoch
    model.eval()  # Set model to evaluation mode
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in valid_loader:
            # Tu ciclo de validación aquí, similar al de prueba
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    # Calcula la precisión y la pérdida promedio de validación
    valid_accuracy = correct / total
    valid_loss /= len(valid_loader)
    print(f'Epoch {epoch}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}')
    epoch_accuracy = correct / total
    accuracies.append(epoch_accuracy)

fig, ax = plt.subplots()
ax.plot(range(num_epochs), accuracies)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy per Epoch')
plt.show()
accuracy = correct / total
print(f'Model accuracy on the test set: {accuracy * 100:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
print('Training completed.')
