import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.data_preprocessing import load_and_preprocess_data
from models.match_predictor_model import MatchPredictor
import matplotlib.pyplot as plt


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
train_dataset, test_dataset, sampler = load_and_preprocess_data('data/datasheetv2.csv')

# Create DataLoaders for training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

num_teams = 283
num_champions = 168
num_players = 1554
num_regions = 31
embedding_dim = 10
num_numerical_features = 6
output_dim = 2  # Assuming binary classification for win/lose

model = MatchPredictor(num_teams, num_champions, num_players, num_regions, embedding_dim, num_numerical_features, output_dim)

# Define loss function and optimizer
criterion = CrossEntropyLoss()
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
    correct = 0
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
    
    # Calculate accuracy for the current epoch and append to the list
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
