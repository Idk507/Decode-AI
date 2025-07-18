import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(1000, 2) * 10
y = (X[:, 0] + X[:, 1] > 10).astype(np.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Batch Size = 32

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize model, loss, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Learning Rate = 0.01

# Training loop
num_epochs = 50  # Number of Epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        total_loss += loss.item()
    
    # Print average loss per epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_tensor)
    predictions = (outputs >= 0.5).float()
    accuracy = (predictions == y_tensor).float().mean()
    print(f"Final Accuracy: {accuracy:.4f}")
