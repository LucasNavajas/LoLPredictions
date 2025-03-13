import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from utils.data_preprocessing import load_and_preprocess_data
from models.match_predictor_model import MatchPredictor
import matplotlib.pyplot as plt
import random
import os

# ----------------------------------------------------------------------
# Setup and Seeding
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set random seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# If GPU is available, set the seed for CUDA as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
     # These two lines help ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------
# model_wrapper function (used for inference / external calls)
# ----------------------------------------------------------------------
def model_wrapper(x):
    """
    Converts raw input array 'x' into a FloatTensor,
    performs a forward pass, and returns a binary output (0 or 1).

    Args:
        x (array-like): Input features of shape (N, num_features).
    
    Returns:
        np.ndarray: Binary predictions (0 or 1) for each sample.
    """
    # Move input to device and cast to float
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    
    # Put model in eval mode
    model.eval()
    
    with torch.no_grad(): 
        # Forward pass (logits) and convert tensor to numpy
        output = model(x_tensor).detach().cpu().numpy() 
    
    # Since output is a logit, threshold at 0.5 after sigmoid
    binary_output = (output >= 0.5).astype(int)
    
    return binary_output

# ----------------------------------------------------------------------
# Load and preprocess dataset
# ----------------------------------------------------------------------
train_dataset, test_dataset, val_dataset = load_and_preprocess_data(os.path.join(BASE_DIR,'data/datasheet.csv'))


# Read the CSV, compute Glicko ratings, preprocess features, and split into (train, val, test) set as PyTorch Datasets
# Wrap the Datasets in DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ----------------------------------------------------------------------
# Model Configuration
# ----------------------------------------------------------------------
num_champions = 171 # The size of the champion 'vocabulary' for embeddings
embedding_dim = 10  # Dimensionality of each champion's embedding
output_dim = 1  # Output a single logit for a binary classification 

# Instantiate the model
model = MatchPredictor(output_dim, num_champions, embedding_dim)

# Define the loss (BCEWithLogitsLoss expects raw logits, not probabilities)
criterion = BCEWithLogitsLoss() 

# Define the optimizer (Adam) with a small learning rate
optimizer = Adam(model.parameters(), lr=0.0001) 

device = torch.device('cuda')
model.to(device)

# ----------------------------------------------------------------------
# Training Configuration
# ----------------------------------------------------------------------
num_epochs = 120

# Lists to track metrics over epochs
train_accuracies = []
val_accuracies = []
val_losses = []

# ----------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------
for epoch in range(num_epochs):
    model.train()  # enable training mode

    train_correct = 0
    train_total = 0

    # -------- Train over each mini-batch --------
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).float()

        # Forward pass: model outputs logits of shape (batch_size, 1)
        logits = model(features).squeeze(dim=1)  # shape: (batch_size,)

        # Compute the loss using raw logits
        loss = criterion(logits, labels)

        # Clear gradients, backprop, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        probs = torch.sigmoid(logits)           # convert logits -> probabilities
        predicted = (probs > 0.5).float()       # threshold at 0.5
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total

    # -------- Validation (held-out) evaluation --------
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in valid_loader:
            features, labels = features.to(device), labels.to(device).float()

            # Forward pass in evaluation mode (raw logits)
            logits = model(features).squeeze(dim=1)

            # Validation loss with raw logits
            loss = criterion(logits, labels)
            valid_loss += loss.item()

            # Calculate validation accuracy
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_accuracy = correct / total
    valid_loss /= len(valid_loader)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(valid_accuracy)
    val_losses.append(valid_loss)

    print(f'Epoch {epoch}, '
          f'Train Accuracy: {train_accuracy:.4f}, '
          f'Valid Loss: {valid_loss:.4f}, '
          f'Valid Accuracy: {valid_accuracy:.4f}')

# ----------------------------------------------------------------------
# Plotting Accuracy and Validation Loss
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# 1) Training vs Validation Accuracy
ax[0].plot(range(num_epochs), train_accuracies, label='Train Accuracy')
ax[0].plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Training and Validation Accuracy per Epoch')
ax[0].legend()

# 2) Validation Loss
ax[1].plot(range(num_epochs), val_losses, label='Validation Loss', color='red')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_title('Validation Loss per Epoch')
ax[1].legend()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Final Test Evaluation
# ----------------------------------------------------------------------
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device).float()

        # Get raw logits from the model
        logits = model(features).squeeze(dim=1)
        # Convert to probabilities to threshold
        probs = torch.sigmoid(logits)
        predicted = (probs > 0.5).float()

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = test_correct / test_total
print(f'Model accuracy on the test set: {test_accuracy * 100:.2f}%')

# ----------------------------------------------------------------------
# Save Model Weights
# ----------------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(BASE_DIR, 'model.pth'))
print('Training completed.')