import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Target network: Simple MLP
class TargetNetwork(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=5):
        super(TargetNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Weights will be set by the hypernetwork
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None

    def set_weights(self, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        self.fc1_weight = fc1_weight
        self.fc1_bias = fc1_bias
        self.fc2_weight = fc2_weight
        self.fc2_bias = fc2_bias

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.linear(x, self.fc1_weight, self.fc1_bias)
        x = F.relu(x)
        x = F.linear(x, self.fc2_weight, self.fc2_bias)
        return x

# Hypernetwork: Generates weights for the target network
class HyperNetwork(nn.Module):
    def __init__(self, embedding_dim=32, hidden_dim=128, target_input_dim=784, target_hidden_dim=64, target_output_dim=5):
        super(HyperNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output sizes for target network weights
        self.fc1_weight_out = nn.Linear(hidden_dim, target_input_dim * target_hidden_dim)
        self.fc1_bias_out = nn.Linear(hidden_dim, target_hidden_dim)
        self.fc2_weight_out = nn.Linear(hidden_dim, target_hidden_dim * target_output_dim)
        self.fc2_bias_out = nn.Linear(hidden_dim, target_output_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        fc1_weight = self.fc1_weight_out(x).view(-1, 64, 784)  # [batch, hidden_dim, input_dim]
        fc1_bias = self.fc1_bias_out(x)  # [batch, hidden_dim]
        fc2_weight = self.fc2_weight_out(x).view(-1, 5, 64)  # [batch, output_dim, hidden_dim]
        fc2_bias = self.fc2_bias_out(x)  # [batch, output_dim]
        return fc1_weight, fc1_bias, fc2_weight, fc2_bias

# Synthetic dataset for few-shot learning
class FewShotDataset(Dataset):
    def __init__(self, num_tasks, n_way=5, k_shot=1, k_query=15):
        self.num_tasks = num_tasks
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.data = []
        self.labels = []
        for _ in range(num_tasks):
            task_data = torch.randn((n_way * (k_shot + k_query), 1, 28, 28))
            task_labels = torch.cat([torch.full((k_shot + k_query,), i, dtype=torch.long) for i in range(n_way)])
            self.data.append(task_data)
            self.labels.append(task_labels)

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        for i in range(self.n_way):
            indices = (labels == i).nonzero(as_tuple=True)[0]
            support_data.append(data[indices[:self.k_shot]])
            support_labels.append(labels[indices[:self.k_shot]])
            query_data.append(data[indices[self.k_shot:self.k_shot + self.k_query]])
            query_labels.append(labels[indices[self.k_shot:self.k_shot + self.k_query]])
        support_data = torch.cat(support_data)
        support_labels = torch.cat(support_labels)
        query_data = torch.cat(query_data)
        query_labels = torch.cat(query_labels)
        # Compute task embedding (e.g., mean of support set)
        task_embedding = support_data.view(-1, 784).mean(dim=0, keepdim=True)
        return support_data, support_labels, query_data, query_labels, task_embedding

# Training loop
def train_hypernetwork(hypernet, target_net, dataset, num_iterations=1000, lr=0.001):
    optimizer = optim.Adam(hypernet.parameters(), lr=lr)
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        total_loss = 0
        for support_data, support_labels, query_data, query_labels, task_embedding in dataset:
            # Generate weights for the target network
            fc1_weight, fc1_bias, fc2_weight, fc2_bias = hypernet(task_embedding)
            target_net.set_weights(fc1_weight, fc1_bias, fc2_weight, fc2_bias)
            # Forward pass on query set
            logits = target_net(query_data)
            loss = F.cross_entropy(logits, query_labels)
            total_loss += loss
        total_loss.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")

# Main function
def main():
    dataset = FewShotDataset(num_tasks=1000, n_way=5, k_shot=1, k_query=15)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    hypernet = HyperNetwork(embedding_dim=784)  # Input is flattened 28x28 image
    target_net = TargetNetwork()
    train_hypernetwork(hypernet, target_net, dataloader)

if __name__ == "__main__":
    main()
