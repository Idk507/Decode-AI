import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')
#hyperparameters
num_epochs = 5
learning_rate = 0.001
batch_size = 100
# MNIST dataset
transform = transforms.Compose([
    transforms.Pad(2),  # Pad 28x28 to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # C1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # C3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)  # C5
        self.fc1 = nn.Linear(120, 84)  # F6
        self.fc2 = nn.Linear(84, 10)   # F7

    def forward(self, x):
        x = torch.relu(self.conv1(x))      # C1: 32x32x1 -> 28x28x6
        x = self.pool1(x)                  # S2: 28x28x6 -> 14x14x6
        x = torch.relu(self.conv2(x))      # C3: 14x14x6 -> 10x10x16
        x = self.pool2(x)                  # S4: 10x10x16 -> 5x5x16
        x = torch.relu(self.conv3(x))      # C5: 5x5x16 -> 1x1x120
        x = x.view(-1, 120)                # Flatten
        x = torch.relu(self.fc1(x))        # F6: 120 -> 84
        x = self.fc2(x)                    # F7: 84 -> 10
        return x
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

print('Training Finished')
# Testing the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total}%')
