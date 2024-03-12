import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.data_preprocessing import load_and_preprocess_data
from models.match_predictor_model import MatchPredictor

# Load and preprocess data
train_dataset, test_dataset, sampler = load_and_preprocess_data('data/datasheetv2.csv')

# Create DataLoaders for training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

num_teams = 283
num_champions = 168
num_players = 1543
num_regions = 31
embedding_dim = 10
num_numerical_features = 12
output_dim = 2  # Assuming binary classification for win/lose

model = MatchPredictor(num_teams, num_champions, num_players, num_regions, embedding_dim, num_numerical_features, output_dim)

# Define loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Setup device for training (GPU or CPU)
device = torch.device('cpu')
model.to(device)

# Number of epochs
num_epochs = 1000

best_val_loss = np.inf
patience = 100  # Number of epochs to wait for improvement before stopping
patience_counter = 0

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

# Evaluation loop
model.eval()  # Set model to evaluation mode
correct = 0
val_loss = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:
        # Move tensors to the appropriate device
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter
            # Save model checkpoint if desired
        else:
            patience_counter += 1  # No improvement
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch+1}")
            break

accuracy = correct / total
print(f'Model accuracy on the test set: {accuracy * 100:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
print('Training completed.')
