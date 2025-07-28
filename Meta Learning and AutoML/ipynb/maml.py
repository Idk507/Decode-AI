import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import uuid

# Define a simple CNN for few-shot learning
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
            # Generate synthetic 28x28 grayscale images
            task_data = torch.randn((n_way * (k_shot + k_query), 1, 28, 28))
            task_labels = torch.cat([torch.full((k_shot + k_query,), i, dtype=torch.long) for i in range(n_way)])
            self.data.append(task_data)
            self.labels.append(task_labels)

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        # Split into support and query sets
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
        return support_data, support_labels, query_data, query_labels

# MAML implementation
class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, n_way=5, k_shot=1, k_query=15):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)

    def inner_loop(self, support_data, support_labels):
        # Create a copy of the model parameters
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        # Inner loop: gradient descent on support set
        for _ in range(5):  # Number of inner loop updates
            logits = self.model(support_data)
            loss = F.cross_entropy(logits, support_labels)
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # Update fast weights
            fast_weights = {name: fast_weights[name] - self.inner_lr * grad for (name, _), grad in zip(self.model.named_parameters(), grads)}
            # Update model parameters temporarily
            for name, param in self.model.named_parameters():
                param.data = fast_weights[name].data
        return fast_weights

    def outer_loop(self, dataset, num_iterations=1000):
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            meta_loss = 0
            for support_data, support_labels, query_data, query_labels in dataset:
                # Inner loop
                fast_weights = self.inner_loop(support_data, support_labels)
                # Query loss with fast weights
                original_params = {name: param.clone() for name, param in self.model.named_parameters()}
                for name, param in self.model.named_parameters():
                    param.data = fast_weights[name].data
                logits = self.model(query_data)
                loss = F.cross_entropy(logits, query_labels)
                meta_loss += loss
                # Restore original parameters
                for name, param in self.model.named_parameters():
                    param.data = original_params[name].data
            # Outer loop update
            meta_loss.backward()
            self.optimizer.step()
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Meta Loss: {meta_loss.item():.4f}")

# Training
def main():
    # Initialize dataset and model
    dataset = FewShotDataset(num_tasks=1000, n_way=5, k_shot=1, k_query=15)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SimpleCNN(num_classes=5)
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, n_way=5, k_shot=1, k_query=15)
    
    # Train
    maml.outer_loop(dataloader, num_iterations=1000)

if __name__ == "__main__":
    main()
