import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import time
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cpu')  # Use CPU for consistent latency measurement

# Hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001
seeds = [42, 43, 44]  # Multiple seeds for robustness

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

# Define CNN model with configurable layers
class CNN(nn.Module):
    def __init__(self, num_conv_layers=2):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3
        out_channels = 32
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            out_channels *= 2
        self.conv_layers = nn.Sequential(*layers)
        self.flatten_size = 32 * 2**(num_conv_layers-1) * (32 // 2**num_conv_layers)**2
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flatten_size)
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

# Evaluation function for accuracy
def evaluate_accuracy(model, test_loader):
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

# Measure latency (average inference time per sample)
def measure_latency(model, test_loader):
    model.eval()
    total_time = 0.0
    num_samples = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            start_time = time.time()
            model(images)
            total_time += time.time() - start_time
            num_samples += images.size(0)
    latency = (total_time / num_samples) * 1000  # Convert to milliseconds per sample
    return latency

# Multi-objective optimization
configs = [
    {'num_conv_layers': 2, 'name': 'Shallow CNN'},
    {'num_conv_layers': 4, 'name': 'Deep CNN'}
]
results = {config['name']: {'accuracies': [], 'latencies': []} for config in configs}

for config in configs:
    print(f"\nEvaluating {config['name']}")
    for seed in seeds:
        set_seed(seed)
        model = CNN(num_conv_layers=config['num_conv_layers']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train
        train_model(model, train_loader, criterion, optimizer)
        
        # Evaluate accuracy
        accuracy = evaluate_accuracy(model, test_loader)
        results[config['name']]['accuracies'].append(accuracy)
        
        # Measure latency
        latency = measure_latency(model, test_loader)
        results[config['name']]['latencies'].append(latency)
        
        print(f"Seed {seed} - Accuracy: {accuracy:.2f}%, Latency: {latency:.2f}ms")

# Summarize results
for config_name in results:
    mean_acc = np.mean(results[config_name]['accuracies'])
    std_acc = np.std(results[config_name]['accuracies'])
    mean_lat = np.mean(results[config_name]['latencies'])
    std_lat = np.std(results[config_name]['latencies'])
    print(f"\n{config_name}:")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Mean Latency: {mean_lat:.2f}ms ± {std_lat:.2f}ms")

# Statistical comparison (A/B testing)
acc_shallow = results['Shallow CNN']['accuracies']
acc_deep = results['Deep CNN']['accuracies']
lat_shallow = results['Shallow CNN']['latencies']
lat_deep = results['Deep CNN']['latencies']

t_stat_acc, p_value_acc = ttest_rel(acc_shallow, acc_deep)
t_stat_lat, p_value_lat = ttest_rel(lat_shallow, lat_deep)

print("\nStatistical Comparison:")
print(f"Accuracy t-test: t-statistic = {t_stat_acc:.4f}, p-value = {p_value_acc:.4f}")
print(f"Latency t-test: t-statistic = {t_stat_lat:.4f}, p-value = {p_value_lat:.4f}")

# Plot Pareto front
plt.figure(figsize=(8, 6))
for config_name in results:
    mean_acc = np.mean(results[config_name]['accuracies'])
    mean_lat = np.mean(results[config_name]['latencies'])
    plt.scatter(mean_lat, mean_acc, label=config_name, s=100)
plt.xlabel('Latency (ms)')
plt.ylabel('Accuracy (%)')
plt.title('Pareto Front: Accuracy vs. Latency')
plt.legend()
plt.grid(True)
plt.savefig('pareto_front.png')
plt.close()

# Environment details
print("\nEnvironment Details:")
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {device}")
