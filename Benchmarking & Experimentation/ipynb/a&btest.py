import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import ttest_rel
import random

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001
seeds = [42, 43, 44, 45, 46]  # Multiple seeds for robustness

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self, use_dropout=True):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# A/B Testing
model_a_accuracies = []  # CNN with dropout
model_b_accuracies = []  # CNN without dropout

for seed in seeds:
    print(f"\nRunning experiment with seed {seed}")
    set_seed(seed)
    
    # Model A: CNN with dropout
    model_a = CNN(use_dropout=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_a = optim.Adam(model_a.parameters(), lr=learning_rate)
    train_model(model_a, train_loader, criterion, optimizer_a)
    acc_a = evaluate_model(model_a, test_loader)
    model_a_accuracies.append(acc_a)
    print(f"Model A (with dropout) - Seed {seed} - Test Accuracy: {acc_a:.2f}%")
    
    # Model B: CNN without dropout
    model_b = CNN(use_dropout=False).to(device)
    optimizer_b = optim.Adam(model_b.parameters(), lr=learning_rate)
    train_model(model_b, train_loader, criterion, optimizer_b)
    acc_b = evaluate_model(model_b, test_loader)
    model_b_accuracies.append(acc_b)
    print(f"Model B (without dropout) - Seed {seed} - Test Accuracy: {acc_b:.2f}%")

# Statistical analysis
mean_a = np.mean(model_a_accuracies)
std_a = np.std(model_a_accuracies)
mean_b = np.mean(model_b_accuracies)
std_b = np.std(model_b_accuracies)

print(f"\nModel A (with dropout) - Mean Accuracy: {mean_a:.2f}% ± {std_a:.2f}%")
print(f"Model B (without dropout) - Mean Accuracy: {mean_b:.2f}% ± {std_b:.2f}%")

# Paired t-test
t_stat, p_value = ttest_rel(model_a_accuracies, model_b_accuracies)
print(f"\nPaired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
if p_value < 0.05:
    print("Result: Statistically significant difference (p < 0.05)")
else:
    print("Result: No statistically significant difference (p >= 0.05)")

# Environment details
print("\nEnvironment Details:")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {device}")
