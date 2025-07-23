import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
n_samples, n_features = 100, 50  # 100 samples, 50 features
X = np.random.randn(n_samples, n_features)

# Create correlated features (e.g., feature 1 and 2 are correlated)
X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n_samples)

# True coefficients: only 10 features are relevant
true_coef = np.zeros(n_features)
true_coef[:10] = np.random.uniform(0.5, 2.0, 10)  # Non-zero coefficients
y = X @ true_coef + 0.1 * np.random.randn(n_samples)  # Target with noise

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
lasso = Lasso(alpha=0.1)  # L1 regularization
ridge = Ridge(alpha=0.1)  # L2 regularization
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Elastic Net (alpha=0.1, l1_ratio=0.5 means equal L1 and L2)

# Fit models
lasso.fit(X_scaled, y)
ridge.fit(X_scaled, y)
elastic_net.fit(X_scaled, y)

# Predictions
y_pred_lasso = lasso.predict(X_scaled)
y_pred_ridge = ridge.predict(X_scaled)
y_pred_elastic = elastic_net.predict(X_scaled)

# Evaluate models
mse_lasso = mean_squared_error(y, y_pred_lasso)
mse_ridge = mean_squared_error(y, y_pred_ridge)
mse_elastic = mean_squared_error(y, y_pred_elastic)

# Print MSE
print(f"Lasso MSE: {mse_lasso:.4f}")
print(f"Ridge MSE: {mse_ridge:.4f}")
print(f"Elastic Net MSE: {mse_elastic:.4f}")

# Plot coefficients
plt.figure(figsize=(12, 6))
x = np.arange(n_features)
plt.plot(x, true_coef, 'k-', label='True Coefficients', linewidth=2)
plt.plot(x, lasso.coef_, 'r--', label='Lasso Coefficients')
plt.plot(x, ridge.coef_, 'b--', label='Ridge Coefficients')
plt.plot(x, elastic_net.coef_, 'g--', label='Elastic Net Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Coefficients: Lasso, Ridge, and Elastic Net')
plt.legend()
plt.grid(True)
plt.show()
