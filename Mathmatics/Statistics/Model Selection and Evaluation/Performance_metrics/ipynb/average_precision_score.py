import torch
import numpy as np
from sklearn.metrics import average_precision_score

# Sample data: predicted probabilities and true labels
y_true = torch.tensor([0, 1, 2, 0])  # True labels (4 samples, 3 classes)
y_pred = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])  # Predicted probabilities

# Top-K Accuracy (k=2)
def top_k_accuracy(y_true, y_pred, k=2):
    _, top_k_preds = y_pred.topk(k, dim=1)
    correct = top_k_preds.eq(y_true.view(-1, 1).expand_as(top_k_preds))
    return correct.any(dim=1).float().mean().item()

print(f"Top-2 Accuracy: {top_k_accuracy(y_true, y_pred, k=2):.4f}")

# Mean Average Precision (mAP)
y_true_one_hot = torch.zeros_like(y_pred).scatter_(1, y_true.unsqueeze(1), 1).numpy()
y_pred_np = y_pred.numpy()
mAP = np.mean([average_precision_score(y_true_one_hot[:, i], y_pred_np[:, i]) for i in range(y_true_one_hot.shape[1])])
print(f"mAP: {mAP:.4f}")
