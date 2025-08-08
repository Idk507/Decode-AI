
---

### **Foundational Regression Metrics**
These are the most commonly used metrics, suitable for general regression tasks.

1. **Mean Squared Error (MSE)**:
   - **Definition**: Average of the squared differences between predicted and actual values.
   - **Formula**:
     <img width="454" height="119" alt="image" src="https://github.com/user-attachments/assets/c675016c-f0fb-467e-b105-f535de84c639" />

     where $\( y_i \)$ is the actual value, $\( \hat{y}_i \)$ is the predicted value, and $\( n \)$ is the number of samples.
   - **Use Case**: Standard metric for regression; penalizes larger errors more heavily due to squaring.
   - **Limitations**: Sensitive to outliers; not in the same units as the target variable.
   - **Deep Learning Context**: Common loss function in neural networks (e.g., linear regression, forecasting).

2. **Root Mean Squared Error (RMSE)**:
   - **Definition**: Square root of MSE, providing error in the same units as the target variable.
   - **Formula**:
     \[
     \text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
     \]
   - **Use Case**: Interpretable metric for regression; used in tasks like house price prediction or time-series forecasting.
   - **Limitations**: Still sensitive to outliers; may overemphasize large errors.
   - **Deep Learning Context**: Used for evaluation and comparison in regression tasks (e.g., CNNs for age prediction).

3. **R-Squared (Coefficient of Determination)**:
   - **Definition**: Proportion of variance in the dependent variable explained by the model.
   - **Formula**:
     <img width="516" height="94" alt="image" src="https://github.com/user-attachments/assets/8d47d207-2b5c-4add-847d-9f359adcee00" />

     where $\( \text{SS}_{\text{res}} \)$ is the residual sum of squares, \( \text{SS}_{\text{tot}} \) is the total sum of squares, and \( \bar{y} \) is the mean of actual values.
   - **Range**: 0 to 1 (1 = perfect fit, 0 = no explanatory power; negative if model is worse than mean prediction).
   - **Use Case**: Measures goodness of fit; widely used in linear regression and forecasting.
   - **Limitations**: Can be misleading for nonlinear models; increases with more predictors.
   - **Deep Learning Context**: Used to evaluate complex models but less common in deep learning due to nonlinearities.

---

### **Intermediate Regression Metrics**
These metrics provide additional insights or address specific limitations of foundational metrics.

4. **Mean Absolute Error (MAE)**:
   - **Definition**: Average of absolute differences between predicted and actual values.
   - **Formula**:
    <img width="331" height="117" alt="image" src="https://github.com/user-attachments/assets/07e6f77b-c973-46c8-8025-31583226c38e" />

   - **Use Case**: Robust to outliers; interpretable in target units (e.g., financial forecasting, energy consumption).
   - **Limitations**: Less sensitive to large errors; not differentiable at zero (less common as a loss function).
   - **Deep Learning Context**: Used as an alternative loss function or evaluation metric in robust regression tasks.

5. **Mean Absolute Percentage Error (MAPE)**:
   - **Definition**: Average of absolute percentage errors, expressing error relative to actual values.
   - **Formula**:
   <img width="386" height="99" alt="image" src="https://github.com/user-attachments/assets/36658570-078d-4675-955e-decaebe9039b" />

   - **Use Case**: Useful when relative errors matter (e.g., sales forecasting, demand prediction).
   - **Limitations**: Undefined or unstable when \( y_i = 0 \); biased toward underpredictions.
   - **Deep Learning Context**: Less common in deep learning due to numerical instability but used in business applications.

6. **Median Absolute Error (MedAE)**:
   - **Definition**: Median of absolute differences between predicted and actual values.
   - **Formula**:
     <img width="588" height="74" alt="image" src="https://github.com/user-attachments/assets/bb10a674-3f87-43c7-b448-fa549fae450b" />

   - **Use Case**: Highly robust to outliers; used in noisy datasets (e.g., real estate price prediction).
   - **Limitations**: Ignores distribution of errors; less informative about overall error magnitude.
   - **Deep Learning Context**: Useful for evaluating models on skewed or noisy data.

7. **Adjusted R-Squared**:
   - **Definition**: R-squared adjusted for the number of predictors to penalize model complexity.
   - **Formula**:
    <img width="461" height="99" alt="image" src="https://github.com/user-attachments/assets/489160fe-0141-471b-a99f-ada6c0b872ae" />

     where $\( p \)$ is the number of predictors.
   - **Use Case**: Compares models with different numbers of features; prevents overfitting in feature-rich models.
   - **Limitations**: Still assumes linear relationships; less relevant for deep learning.
   - **Deep Learning Context**: Rarely used due to complex architectures but applicable in simpler neural networks.

8. **Mean Squared Logarithmic Error (MSLE)**:
   - **Definition**: Average of squared differences between the logarithms of predicted and actual values.
   - **Formula**:
    <img width="573" height="150" alt="image" src="https://github.com/user-attachments/assets/b5092ece-9f9c-4774-8285-3f2b9c6c5606" />

   - **Use Case**: Suitable for positive-valued targets with exponential growth (e.g., population growth, stock prices).
   - **Limitations**: Requires positive values; less interpretable than MSE.
   - **Deep Learning Context**: Used as a loss function in tasks with skewed distributions.

---

### **Advanced Regression Metrics**
These metrics are used in specialized deep learning or statistical applications, addressing complex scenarios or robustness.

9. **Huber Loss**:
   - **Definition**: Combines MSE for small errors and MAE for large errors, using a threshold \( \delta \).
   - **Formula**:
   <img width="663" height="114" alt="image" src="https://github.com/user-attachments/assets/49ab018e-e945-4fa6-a7b5-b9eb0c0bf12b" />

   - **Use Case**: Robust to outliers; used in regression tasks with noisy data (e.g., sensor data prediction).
   - **Deep Learning Context**: Common loss function in neural networks for robust regression.
   - **Limitations**: Requires tuning \( \delta \); less interpretable than MAE or MSE.

10. **Quantile Loss**:
    - **Definition**: Evaluates predictions for specific quantiles (e.g., median, 90th percentile), useful for uncertainty estimation.
    - **Formula** (for quantile \( \tau \)):
     <img width="621" height="86" alt="image" src="https://github.com/user-attachments/assets/4f109088-fcc8-4de5-a6fc-0359b4f819b8" />

    - **Use Case**: Quantile regression (e.g., predicting confidence intervals in finance or weather forecasting).
    - **Deep Learning Context**: Used in probabilistic neural networks for uncertainty quantification.
    - **Limitations**: Requires specifying quantile; less common for point predictions.

11. **Explained Variance Score**:
    - **Definition**: Measures the proportion of variance in the target explained by the model, accounting for bias.
    - **Formula**:
    <img width="476" height="106" alt="image" src="https://github.com/user-attachments/assets/f15fd68f-73d0-487d-9bb8-73ffa0f9f87a" />

    - **Use Case**: Alternative to R-squared; robust to bias in predictions.
    - **Limitations**: Similar to R-squared but less widely used.
    - **Deep Learning Context**: Occasionally used to assess variance explained by complex models.

12. **Weighted Mean Absolute Error (WMAE)**:
    - **Definition**: MAE with weights applied to errors, emphasizing certain samples or regions.
    - **Formula**:
      <img width="354" height="103" alt="image" src="https://github.com/user-attachments/assets/88e1777e-6794-43fc-aa4b-0b6a7b46c279" />

      where $\( w_i \)$ is the weight for sample $\( i \)$.
    - **Use Case**: Used when some errors are more critical (e.g., time-series forecasting with higher weights for recent data).
    - **Deep Learning Context**: Implemented in custom loss functions for weighted regression.
    - **Limitations**: Requires defining weights; subjective in some cases.

13. **Log-Cosh Loss**:
    - **Definition**: Smooth approximation of absolute error, using the logarithm of the hyperbolic cosine.
    - **Formula**:
      <img width="499" height="104" alt="image" src="https://github.com/user-attachments/assets/2364b381-faac-44eb-bda3-ab6a988b9548" />

    - **Use Case**: Robust to outliers; smoother gradients than MAE for optimization.
    - **Deep Learning Context**: Used as a loss function in neural networks for regression tasks.
    - **Limitations**: Less interpretable; computationally heavier than MSE.

14. **Kullback-Leibler Divergence (KL Divergence)** (for Probabilistic Regression):
    - **Definition**: Measures divergence between predicted and true probability distributions (e.g., for probabilistic regression outputs).
    - **Formula**:
     <img width="453" height="100" alt="image" src="https://github.com/user-attachments/assets/f79bd120-5673-4db7-b179-dd354b159dd1" />

    - **Use Case**: Evaluates probabilistic regression models (e.g., Gaussian process regression, Bayesian neural networks).
    - **Deep Learning Context**: Common in generative models or uncertainty quantification.
    - **Limitations**: Requires distributional assumptions; sensitive to zero probabilities.

15. **Jensen-Shannon Divergence (JS Divergence)**:
    - **Definition**: Symmetric version of KL divergence for comparing probability distributions.
    - **Formula**:
      <img width="668" height="86" alt="image" src="https://github.com/user-attachments/assets/4cd393aa-3510-49e4-a0a8-d7a47fafee06" />

    - **Use Case**: Probabilistic regression tasks (e.g., density estimation in VAEs).
    - **Deep Learning Context**: Used in generative modeling or distribution alignment.
    - **Limitations**: Computationally intensive; less common for point predictions.

16. **Pinball Loss**:
    - **Definition**: Generalization of quantile loss, used for quantile regression.
    - **Formula**:
     <img width="636" height="109" alt="image" src="https://github.com/user-attachments/assets/e8eebb0e-ad64-4126-8bcf-3b06d951f332" />

    - **Use Case**: Similar to quantile loss; used in forecasting with uncertainty (e.g., energy demand).
    - **Deep Learning Context**: Implemented in neural networks for quantile regression.
    - **Limitations**: Requires specifying quantile; not for point predictions.

17. **Symmetric Mean Absolute Percentage Error (SMAPE)**:
    - **Definition**: Modified MAPE to address issues with zero values and asymmetry.
    - **Formula**:
      <img width="399" height="127" alt="image" src="https://github.com/user-attachments/assets/c06dddbc-6dba-41e4-af30-ae8fff28098c" />

    - **Use Case**: Relative error metric for forecasting (e.g., retail sales).
    - **Deep Learning Context**: Used in time-series models to handle varying scales.
    - **Limitations**: Can still be unstable for small values; less common in deep learning.

18. **Continuous Ranked Probability Score (CRPS)**:
    - **Definition**: Measures the difference between predicted and actual cumulative distribution functions for probabilistic regression.
    - **Formula**:
     <img width="448" height="86" alt="image" src="https://github.com/user-attachments/assets/a66e0b3e-e8ae-4b25-94f1-ac1305e86cee" />

      where $\( \hat{F} \)$ is the predicted CDF, $\( y \)$ is the true value.
    - **Use Case**: Probabilistic forecasting (e.g., weather, energy).
    - **Deep Learning Context**: Used in Bayesian neural networks or ensemble forecasting.
    - **Limitations**: Complex to compute; requires distributional outputs.

---

### **Statistical and Probabilistic Foundations**
- **Error Decomposition**: MSE and RMSE relate to the bias-variance decomposition, where total error = Bias² + Variance + Irreducible Error.
- **Central Limit Theorem**: Assumes large-sample errors are normally distributed, justifying MSE’s use in optimization.
- **Bayesian Perspective**: Metrics like KL divergence and CRPS evaluate distributional predictions, aligning with Bayesian methods.
- **Robustness**: MAE, MedAE, and Huber loss are grounded in robust statistics, minimizing outlier impact.
- **Probabilistic Outputs**: Deep learning models often predict distributions (e.g., in Bayesian networks), making metrics like KL divergence and CRPS critical.

---

### **Deep Learning Contexts**
- **Time-Series Forecasting**: MAPE, SMAPE, and CRPS are common in RNNs, LSTMs, or transformers (e.g., stock price prediction).
- **Computer Vision**: MSE, MAE, and Huber loss are used in regression tasks like depth estimation or keypoint prediction.
- **Probabilistic Models**: KL divergence, JS divergence, and CRPS evaluate uncertainty in Bayesian neural networks or VAEs.
- **Robust Regression**: Huber loss and quantile loss handle noisy data in tasks like autonomous driving or medical imaging.
- **Financial Applications**: MAPE, WMAE, and quantile loss address relative errors and weighted importance.

---

### **Implementation Example (Python with Scikit-learn/PyTorch)**
Here’s an example computing MSE, RMSE, MAE, and Huber loss for a regression model:

```python
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
```

---

### **Visualizing Metrics**
To compare regression metrics across models, a bar chart is effective. Here’s an example comparing MSE, RMSE, MAE, and Huber loss for two models (e.g., linear regression vs. neural network):

<img width="972" height="544" alt="image" src="https://github.com/user-attachments/assets/2714113e-4265-4a3f-bef3-ec9a58495823" />



---

### **Best Practices**
- **Match Metric to Task**: Use MSE/RMSE for general regression, MAE/MedAE for robustness, MAPE/SMAPE for relative errors, and CRPS for probabilistic predictions.
- **Combine Metrics**: Use multiple metrics (e.g., RMSE + MAE + R²) to capture different aspects of performance.
- **Cross-Validation**: Compute metrics via k-fold CV for robust estimates.
- **Domain Knowledge**: Align metrics with task requirements (e.g., MAPE for business forecasting, Huber for noisy data).
- **Scale Sensitivity**: Normalize data or use scale-invariant metrics (e.g., MAPE, SMAPE) for varying target ranges.

---

### **Conclusion**
This list covers regression metrics from foundational (MSE, RMSE, R²) to advanced (CRPS, quantile loss), addressing general and deep learning-specific regression tasks. Each metric serves unique purposes, from robustness (Huber, MedAE) to probabilistic evaluation (KL divergence, CRPS). 
