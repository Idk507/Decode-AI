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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Learned optimizer: LSTM-based
class LearnedOptimizer(nn.Module):
    def __init__(self, hidden_dim=20):
        super(LearnedOptimizer, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output update for each parameter

    def forward(self, gradients, prev_state):
        # Input: [batch, num_params, 2] (gradient, loss)
        lstm_out, new_state = self.lstm(gradients, prev_state)
        updates = self.fc(lstm_out)  # [batch, num_params, 1]
        return updates.squeeze(-1), new_state

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
        return torch.cat(support_data), torch.cat(support_labels), torch.cat(query_data), torch.cat(query_labels)

# Training loop
class MetaOptimizer:
    def __init__(self, opt_net, target_net, lr=0.001, inner_steps=5):
        self.opt_net = opt_net
        self.target_net = target_net
        self.inner_steps = inner_steps
        self.optimizer = optim.Adam(opt_net.parameters(), lr=lr)

    def optimize(self, support_data, support_labels, query_data, query_labels):
        # Initialize target network parameters
        target_params = [p.clone().requires_grad_(True) for p in self.target_net.parameters()]
        num_params = sum(p.numel() for p in target_params)
        h, c = (torch.zeros(2, 1, 20), torch.zeros(2, 1, 20))  # LSTM initial state

        # Inner loop: Optimize target network
        for _ in range(self.inner_steps):
            logits = self.target_net(support_data)
            loss = F.cross_entropy(logits, support_labels)
            grads = torch.autograd.grad(loss, target_params, create_graph=True)
            flat_grads = torch.cat([g.flatten() for g in grads]).unsqueeze(0).unsqueeze(-1)
            loss_input = loss.detach().reshape(1, 1, 1).expand(1, num_params, 1)
            opt_input = torch.cat([flat_grads, loss_input], dim=-1)  # [1, num_params, 2]
            updates, (h, c) = self.opt_net(opt_input, (h, c))
            # Apply updates
            offset = 0
            for i, param in enumerate(target_params):
                param_size = param.numel()
                param.add_(updates[:, offset:offset + param_size].view_as(param))
                offset += param_size
            target_params = [p.detach().requires_grad_(True) for p in target_params]

        # Query loss
        logits = self.target_net(query_data)
        query_loss = F.cross_entropy(logits, query_labels)
        return query_loss

    def train(self, dataset, num_iterations=1000):
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            total_loss = 0
            for support_data, support_labels, query_data, query_labels in dataset:
                loss = self.optimize(support_data, support_labels, query_data, query_labels)
                total_loss += loss
            total_loss.backward()
            self.optimizer.step()
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")

# Main function
def main():
    dataset = FewShotDataset(num_tasks=1000, n_way=5, k_shot=1, k_query=15)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    target_net = TargetNetwork()
    opt_net = LearnedOptimizer()
    meta_opt = MetaOptimizer(opt_net, target_net)
    meta_opt.train(dataloader)

if __name__ == "__main__":
    main()
