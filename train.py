import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from utils.data_preprocessing import load_and_preprocess_data
from models.match_predictor_model import MatchPredictor
import matplotlib.pyplot as plt
import random

# Set random seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_wrapper(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    
    model.eval()
    
    with torch.no_grad(): 
        output = model(x_tensor).detach().cpu().numpy() 
    
    binary_output = (output >= 0.5).astype(int)
    
    return binary_output


train_dataset, test_dataset, val_dataset = load_and_preprocess_data('data/datasheetv3.csv')


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


num_champions = 170  
embedding_dim = 10 
output_dim = 1  

model = MatchPredictor(output_dim, num_champions, embedding_dim)

criterion = BCEWithLogitsLoss() 
optimizer = Adam(model.parameters(), lr=0.0001) 

device = torch.device('cpu')
model.to(device)

num_epochs = 120 

train_accuracies = []
val_accuracies = []
val_losses = []

for epoch in range(num_epochs):
    model.train()  
    train_correct = 0
    train_total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).float()

        predictions = model(features).squeeze()  

        loss = criterion(predictions, labels)  

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        outputs = torch.sigmoid(predictions)
        predicted = (outputs > 0.5).float()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    model.eval() 
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in valid_loader:
            features, labels = features.to(device), labels.to(device).float()
            outputs = model(features).squeeze()
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    valid_accuracy = correct / total
    valid_loss /= len(valid_loader)
    train_accuracy = train_correct / train_total
    train_accuracies.append(train_accuracy)
    val_accuracies.append(valid_accuracy)
    val_losses.append(valid_loss)

    print(f'Epoch {epoch}, Train Accuracy: {train_accuracy}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}')

# Plot training and validation accuracy/loss
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(range(num_epochs), train_accuracies, label='Train Accuracy')
ax[0].plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Training and Validation Accuracy per Epoch')
ax[0].legend()

ax[1].plot(range(num_epochs), val_losses, label='Validation Loss', color='red')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_title('Validation Loss per Epoch')
ax[1].legend()

plt.tight_layout()
plt.show()


model.eval() 
test_correct = 0
test_total = 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device).float()
        outputs = model(features).squeeze()
        outputs = torch.sigmoid(outputs)
        predicted = (outputs > 0.5).float()
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()


accuracy = test_correct / test_total
print(f'Model accuracy on the test set: {accuracy * 100:.2f}%')

torch.save(model.state_dict(), 'model.pth')
print('Training completed.')
