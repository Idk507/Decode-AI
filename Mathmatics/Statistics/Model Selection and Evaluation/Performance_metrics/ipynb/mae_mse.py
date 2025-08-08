import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

# Sample data
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

# MSE, RMSE, MAE
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Huber Loss (PyTorch)
def huber_loss(y_true, y_pred, delta=1.0):
    error = torch.tensor(y_true) - torch.tensor(y_pred)
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * torch.abs(error) - 0.5 * delta ** 2
    return torch.where(is_small_error, squared_loss, linear_loss).mean()

huber = huber_loss(y_true, y_pred, delta=1.0)
print(f"Huber Loss: {huber:.4f}")
