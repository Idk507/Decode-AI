
---

### 1. Mean Absolute Error (MAE)

**Formula:**
<img width="329" height="114" alt="image" src="https://github.com/user-attachments/assets/fa41abfb-117d-49d5-b20b-dd1887fd4e2e" />

where:
- $\( y_i \)$: Actual value
- $\( \hat{y}_i \)$ : Predicted value
- $\( n \)$ : Number of observations

**Explanation:**
MAE measures the average magnitude of errors in a set of predictions, without considering their direction (i.e., it takes the absolute value of errors). It represents the average absolute difference between predicted and actual values.

**When, Where, and Why It Is Used:**
- **When**: MAE is used when you want a straightforward, interpretable metric that gives equal weight to all errors, regardless of their size.
- **Where**: Commonly used in regression tasks, such as forecasting (e.g., predicting house prices, stock prices, or demand), where understanding the average error in the same units as the target variable is valuable.
- **Why**: MAE is robust to outliers compared to squared error metrics because it does not square the errors, making it less sensitive to extreme values. It’s intuitive since it directly reflects the average error magnitude.
- **Role**: MAE provides a clear measure of prediction accuracy in the original units of the data, making it easy to communicate to stakeholders.

**Advantages:**
- Easy to interpret (error in the same units as the target).
- Robust to outliers compared to MSE.
- Computationally simple.

**Limitations:**
- Does not penalize larger errors more heavily, which may be undesirable in cases where large deviations are critical.
- Not differentiable at zero, which can complicate optimization in some machine learning algorithms.

**Example Use Case:**
In weather forecasting, MAE can be used to evaluate the average error in predicted temperatures (e.g., an MAE of 2°C means predictions are off by 2°C on average).

---

### 2. Mean Squared Error (MSE)

**Formula:**
<img width="313" height="99" alt="image" src="https://github.com/user-attachments/assets/ac91b036-3e3a-47b0-b352-1edb77ac1439" />


**Explanation:**
MSE measures the average of the squared differences between predicted and actual values. By squaring the errors, it penalizes larger errors more heavily than smaller ones.

**When, Where, and Why It Is Used:**
- **When**: MSE is used when large errors are particularly undesirable, and you want to penalize them quadratically.
- **Where**: Common in regression tasks, such as linear regression, neural networks, and time-series forecasting, where differentiable loss functions are needed for optimization.
- **Why**: Squaring errors emphasizes larger deviations, making MSE sensitive to outliers. It’s also mathematically convenient for optimization (e.g., in gradient descent) because it’s differentiable.
- **Role**: MSE is a standard loss function in many machine learning algorithms, as it balances model fit and complexity.

**Advantages:**
- Differentiable, making it suitable for optimization in gradient-based methods.
- Emphasizes larger errors, which can be useful in high-stakes applications.

**Limitations:**
- Sensitive to outliers due to squaring, which can distort the overall error metric.
- Units are squared (e.g., if predicting dollars, MSE is in dollars²), making interpretation less intuitive than MAE.

**Example Use Case:**
In a machine learning model predicting house prices, MSE is used to ensure that large prediction errors (e.g., predicting $300,000 for a $500,000 house) are heavily penalized.

---

### 3. Root Mean Squared Error (RMSE)

**Formula:**
<img width="576" height="127" alt="image" src="https://github.com/user-attachments/assets/6c28573d-6ec2-4c9a-9b61-5e42a1c9e12f" />

**Explanation:**
RMSE is the square root of MSE, bringing the error metric back to the original units of the target variable. It measures the standard deviation of the residuals (prediction errors).

**When, Where, and Why It Is Used:**
- **When**: RMSE is used when you want a metric that penalizes large errors but is interpretable in the original units.
- **Where**: Widely used in regression tasks, such as time-series forecasting, financial modeling, and engineering applications.
- **Why**: RMSE combines the benefits of MSE (penalizing large errors) with interpretability (same units as the target). It’s often preferred over MSE for reporting model performance.
- **Role**: RMSE is a popular evaluation metric because it balances sensitivity to large errors with interpretability.

**Advantages:**
- Interpretable in the same units as the target variable.
- Sensitive to large errors, making it suitable for applications where large deviations are critical.

**Limitations:**
- Like MSE, RMSE is sensitive to outliers, which can skew results.
- May overemphasize large errors in cases where small errors are more important.

**Example Use Case:**
In energy consumption forecasting, RMSE is used to evaluate model accuracy, as large errors in predicting energy demand could lead to significant operational issues.

---

### 4. R² (R-Squared, Coefficient of Determination)

**Formula:**
<img width="329" height="90" alt="image" src="https://github.com/user-attachments/assets/d42b17f3-40ae-46ec-a3c2-c724b0050aa7" />

where:
- $\( \bar{y} \)$ : Mean of the actual values
- Numerator: Sum of squared residuals (errors)
- Denominator: Total sum of squares (variance of the actual values)

**Explanation:**
R² measures the proportion of variance in the dependent variable (target) explained by the independent variables (features) in the model. It ranges from 0 to 1 (or negative for very poor models), where 1 indicates perfect prediction and 0 indicates the model explains none of the variance.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate how well a regression model explains the variability of the target variable.
- **Where**: Common in linear regression, ANOVA, and other regression models, especially in research and statistical modeling.
- **Why**: R² provides a normalized measure of model fit, making it easy to compare models. It’s particularly useful when assessing the explanatory power of a model.
- **Role**: R² is a standard metric for assessing the goodness-of-fit of regression models.

**Advantages:**
- Easy to interpret as a proportion of explained variance.
- Normalized, allowing comparison across datasets.

**Limitations:**
- Can be misleading for non-linear models or when overfitting occurs.
- Does not indicate whether the model is biased or whether predictions are accurate in absolute terms.
- Can be artificially high with more predictors, even if they add little value.

**Example Use Case:**
In a study predicting student test scores based on study hours, R² of 0.85 means 85% of the variance in test scores is explained by study hours.

---

### 5. Adjusted R²

**Formula:**
<img width="485" height="135" alt="image" src="https://github.com/user-attachments/assets/549b29b4-f8fb-4e89-8729-b67c796abfd5" />


where:
- $\( k \)$ : Number of predictors (features) in the model
- $\( n \)$ : Number of observations

**Explanation:**
Adjusted R² modifies R² to account for the number of predictors in the model, penalizing models that include unnecessary variables. It adjusts for the degrees of freedom, making it a more reliable measure for models with multiple predictors.

**When, Where, and Why It Is Used:**
- **When**: Used when comparing models with different numbers of predictors or when overfitting is a concern.
- **Where**: Common in multiple regression, feature selection, and model comparison in statistical analysis.
- **Why**: Unlike R², Adjusted R² does not automatically increase with more predictors, making it better for assessing model complexity versus explanatory power.
- **Role**: Helps prevent overfitting by penalizing models with excessive predictors that don’t improve fit significantly.

**Advantages:**
- Accounts for model complexity, making it more robust than R² for model comparison.
- Useful for feature selection and avoiding overfitting.

**Limitations:**
- Still assumes a linear relationship between predictors and the target.
- Less intuitive than R² for non-technical audiences.

**Example Use Case:**
In a regression model predicting car prices with features like mileage, age, and brand, Adjusted R² helps determine if adding more features (e.g., color) improves the model or just adds complexity.

---

### 6. Mean Absolute Percentage Error (MAPE)

**Formula:**
<img width="358" height="113" alt="image" src="https://github.com/user-attachments/assets/ca19231d-8b93-4ff6-913e-802983ad5ccf" />


**Explanation:**
MAPE measures the average percentage error between predicted and actual values, expressed as a percentage. It normalizes errors by the actual values, making it scale-independent.

**When, Where, and Why It Is Used:**
- **When**: Used when relative errors are more important than absolute errors, especially for datasets with varying scales.
- **Where**: Common in business forecasting (e.g., sales, demand, or revenue prediction) and time-series analysis.
- **Why**: MAPE is intuitive for stakeholders as it expresses errors in percentage terms, making it easier to understand relative performance across different scales.
- **Role**: MAPE is used to evaluate model accuracy in relative terms, especially when comparing performance across datasets with different units or magnitudes.

**Advantages:**
- Scale-independent, allowing comparison across datasets.
- Intuitive for non-technical audiences (percentage errors).

**Limitations:**
- Undefined or problematic when actual values $(\( y_i \))$ are zero or close to zero.
- Can be biased toward underpredictions, as percentage errors are asymmetric for over- and underpredictions.
- Sensitive to small actual values, which can inflate errors.

**Example Use Case:**
In retail, MAPE is used to evaluate demand forecasting models, where a MAPE of 10% means predictions are off by 10% on average relative to actual sales.

---

### 7. Median Absolute Error

**Formula:**
<img width="586" height="87" alt="image" src="https://github.com/user-attachments/assets/026e0498-6c3f-475b-b550-034c644d3012" />


**Explanation:**
Median Absolute Error is the median of the absolute differences between predicted and actual values. It represents the central tendency of errors, ignoring their direction.

**When, Where, and Why It Is Used:**
- **When**: Used when robustness to outliers is critical, as the median is less affected by extreme values than the mean.
- **Where**: Common in datasets with outliers or skewed error distributions, such as real estate or financial forecasting.
- **Why**: MedAE provides a robust alternative to MAE, focusing on the typical error rather than the average, which can be skewed by outliers.
- **Role**: Useful for evaluating model performance in the presence of outliers or non-normal error distributions.

**Advantages:**
- Highly robust to outliers, as it focuses on the median error.
- Easy to interpret in the original units of the target.

**Limitations:**
- Less sensitive to the full distribution of errors compared to MAE or MSE.
- Not differentiable, limiting its use in optimization.

**Example Use Case:**
In predicting home prices in a market with extreme outliers (e.g., luxury homes), MedAE provides a better sense of typical prediction error than MAE.

---

### 8. Explained Variance Score

**Formula:**
<img width="335" height="117" alt="image" src="https://github.com/user-attachments/assets/30d01062-119a-4c47-ab17-e58a63a5e0a0" />

where:
- $\( \text{Var}(y - \hat{y}) \)$ : Variance of the residuals
- $\( \text{Var}(y) \)$ : Variance of the actual values

**Explanation:**
Explained Variance Score measures the proportion of the variance in the target variable that is explained by the model. It is similar to R² but focuses on variance rather than squared errors.

**When, Where, and Why It Is Used:**
- **When**: Used to assess how well a model captures the variance of the target variable, especially in regression tasks.
- **Where**: Common in machine learning and statistical modeling, particularly when comparing models.
- **Why**: It provides a normalized measure of model performance, similar to R², but can be more robust in certain cases (e.g., when residuals are not centered around zero).
- **Role**: Used to evaluate how much of the target’s variability is captured by the model.

**Advantages:**
- Normalized, allowing comparison across datasets.
- Robust to certain types of bias in residuals compared to R².

**Limitations:**
- Less commonly used than R², so less familiar to some audiences.
- May not fully capture model bias or absolute error magnitude.

**Example Use Case:**
In a climate model predicting temperature variations, the Explained Variance Score indicates how much of the temperature variability is captured by the model.

---

### 9. Mean Squared Logarithmic Error (MSLE)

**Formula:**
<img width="469" height="103" alt="image" src="https://github.com/user-attachments/assets/fb461b6f-83c1-44d2-9df8-17d058179657" />


**Explanation:**
MSLE measures the average squared difference between the logarithms of the actual and predicted values (with a small constant added to handle zeros). It focuses on relative errors, as the logarithmic transformation reduces the impact of large absolute differences.

**When, Where, and Why It Is Used:**
- **When**: Used when relative errors are more important, and the target variable spans several orders of magnitude.
- **Where**: Common in datasets with exponential growth or skewed distributions, such as population growth, revenue, or biological measurements.
- **Why**: MSLE penalizes relative errors, making it suitable for positive-valued data with large ranges. It reduces the impact of large absolute errors.
- **Role**: MSLE is used as a loss function or evaluation metric in regression tasks with skewed or exponential data.

**Advantages:**
- Robust to large absolute errors, focusing on relative differences.
- Suitable for positive-valued data with large ranges.

**Limitations:**
- Requires positive values (or small constant to handle zeros).
- Less intuitive than MAE or RMSE due to logarithmic transformation.

**Example Use Case:**
In predicting the number of website visits (which can range from tens to millions), MSLE ensures that relative errors are prioritized over absolute errors.

---

### 10. Root Mean Squared Logarithmic Error (RMSLE)

**Formula:**
<img width="666" height="100" alt="image" src="https://github.com/user-attachments/assets/824a80a1-a0d9-4cbc-a690-1bac6a61e380" />


**Explanation:**
RMSLE is the square root of MSLE, bringing the error metric back to a scale closer to the original logarithmic errors. It measures the standard deviation of logarithmic errors.

**When, Where, and Why It Is Used:**
- **When**: Used when relative errors are important, and interpretability in logarithmic terms is desired.
- **Where**: Common in machine learning competitions (e.g., Kaggle) and applications with skewed data, such as sales forecasting or traffic prediction.
- **Why**: Like MSLE, RMSLE focuses on relative errors but is more interpretable due to the square root transformation.
- **Role**: RMSLE is used to evaluate model performance in tasks where relative accuracy matters more than absolute accuracy.

**Advantages:**
- Focuses on relative errors, robust to large absolute differences.
- More interpretable than MSLE due to the square root.

**Limitations:**
- Requires positive values (or handling of zeros).
- Less intuitive for non-technical audiences.

**Example Use Case:**
In predicting bike rental demand, RMSLE ensures that errors in predicting low versus high demand are evaluated relatively, avoiding over-penalizing large absolute errors.

---

### 11. Huber Loss

**Formula:**
<img width="565" height="220" alt="image" src="https://github.com/user-attachments/assets/a26e2ed5-70e0-422c-b758-f0037176863f" />

where $\( \delta \)$ is a threshold parameter.

**Explanation:**
Huber Loss combines the properties of MSE (for small errors) and MAE (for large errors). For errors smaller than $\( \delta \)$, it behaves like MSE (quadratic loss); for errors larger than $\( \delta \)$, it behaves like MAE (linear loss).

**When, Where, and Why It Is Used:**
- **When**: Used when you want a loss function that is robust to outliers but still differentiable for optimization.
- **Where**: Common in robust regression, machine learning models (e.g., gradient boosting), and applications with noisy data.
- **Why**: Huber Loss balances sensitivity to small errors (like MSE) with robustness to outliers (like MAE). The $\( \delta \)$ parameter controls the transition point.
- **Role**: Used as a loss function during model training to handle datasets with outliers effectively.

**Advantages:**
- Robust to outliers compared to MSE.
- Differentiable, making it suitable for gradient-based optimization.
- Flexible due to the tunable $\( \delta \)$ parameter.

**Limitations:**
- Requires tuning of $\( \delta \)$ , which can be dataset-dependent.
- Less interpretable as an evaluation metric compared to MAE or RMSE.

**Example Use Case:**
In financial modeling with noisy data (e.g., stock price prediction), Huber Loss is used to train models that are robust to occasional extreme price movements.

---

### 12. Quantile Loss

**Formula:**
<img width="504" height="184" alt="image" src="https://github.com/user-attachments/assets/05e9742f-0dab-4497-9e85-0ff81e1dca73" />

where $\( q \)$ is the quantile (e.g., 0.5 for median).

**Explanation:**
Quantile Loss is used to predict specific quantiles of the target distribution (e.g., median for $\( q = 0.5 \)$). It penalizes underpredictions and overpredictions differently based on the quantile $\( q \)$.

**When, Where, and Why It Is Used:**
- **When**: Used when predicting specific quantiles (e.g., 10th, 50th, or 90th percentile) is important, such as in uncertainty estimation.
- **Where**: Common in quantile regression, forecasting (e.g., demand forecasting with uncertainty), and risk analysis.
- **Why**: Quantile Loss allows models to focus on specific parts of the target distribution, providing insights into uncertainty or tail behavior.
- **Role**: Used in training quantile regression models to predict conditional quantiles rather than the mean.

**Advantages:**
- Provides predictions for specific quantiles, useful for uncertainty quantification.
- Robust to outliers when targeting lower or higher quantiles.

**Limitations:**
- Requires specifying the quantile $\( q \)$ , which may not always be straightforward.
- Less intuitive as a general evaluation metric compared to MAE or RMSE.

**Example Use Case:**
In supply chain management, Quantile Loss is used to predict the 90th percentile of demand to ensure sufficient inventory without overstocking.

---

### 13. Tweedie Deviance

**Formula:**
The Tweedie Deviance depends on the power parameter $\( p \)$ :
<img width="718" height="244" alt="image" src="https://github.com/user-attachments/assets/fa2bcfbe-d3f2-449c-bb64-1055550248d9" />

**Explanation:**
Tweedie Deviance is a loss function for Tweedie distributions, which are used for modeling data with a mix of zeros and continuous positive values (e.g., insurance claims, rainfall). The power parameter \( p \) determines the specific distribution (e.g., \( p=1 \) for Poisson, \( p=2 \) for Gamma).

**When, Where, and Why It Is Used:**
- **When**: Used for datasets with mixed distributions (zeros and positive values) or skewed data.
- **Where**: Common in insurance (claim amount prediction), environmental modeling, and gradient boosting frameworks like XGBoost or LightGBM.
- **Why**: Tweedie Deviance is designed for Tweedie distributions, which handle a wide range of data types, including count data, continuous data, or zero-inflated data.
- **Role**: Used as a loss function in training models for datasets with specific distributional properties.

**Advantages:**
- Flexible for modeling various types of data (e.g., zero-inflated, skewed).
- Widely supported in modern machine learning frameworks.

**Limitations:**
- Requires tuning the power parameter $\( p \)$ .
- Less interpretable as a standalone evaluation metric.

**Example Use Case:**
In insurance, Tweedie Deviance is used to predict claim amounts, where many policyholders have zero claims, and a few have large claims.

---

### 14. Akaike Information Criterion (AIC) / Bayesian Information Criterion (BIC)

**Formulas:**
<img width="361" height="123" alt="image" src="https://github.com/user-attachments/assets/1bf70e75-c0b9-464a-8621-a48d77d53054" />

where:
- $\( k \)$: Number of parameters in the model
- $\( n \)$: Number of observations
- $\( L \)$ : Maximum likelihood of the model

**Explanation:**
AIC and BIC are information criteria used for model selection. They balance model fit (via likelihood) with model complexity (via the number of parameters). Lower AIC/BIC values indicate better models.

**When, Where, and Why It Is Used:**
- **When**: Used for comparing multiple models to select the best one, considering both fit and complexity.
- **Where**: Common in statistical modeling, time-series analysis (e.g., ARIMA), and regression model selection.
- **Why**: AIC and BIC penalize overly complex models, helping to avoid overfitting. BIC penalizes complexity more heavily than AIC due to the \( \ln(n) \) term.
- **Role**: Used in model selection to choose the model that best balances goodness-of-fit and simplicity.

**Advantages:**
- Accounts for both model fit and complexity.
- Widely used in statistical modeling for fair comparisons.

**Limitations:**
- Assumes the model is correctly specified and data follows the assumed distribution.
- BIC may overly penalize complex models in large datasets.

**Example Use Case:**
In time-series forecasting with ARIMA models, AIC/BIC is used to select the optimal order of the model (e.g., ARIMA(p,d,q)).

---

### Summary Table

| Metric                  | Focus                            | Robust to Outliers | Units                    | Use Case Example                     |
|-------------------------|----------------------------------|--------------------|--------------------------|-------------------------------------|
| MAE                     | Absolute errors                 | Yes                | Same as target           | Weather forecasting                 |
| MSE                     | Squared errors                  | No                 | Squared units            | House price prediction              |
| RMSE                    | Squared errors (rooted)         | No                 | Same as target           | Energy demand forecasting           |
| R²                      | Explained variance              | No                 | Unitless (0 to 1)        | Student performance analysis        |
| Adjusted R²             | Explained variance (adjusted)   | No                 | Unitless (0 to 1)        | Feature selection in regression     |
| MAPE                    | Relative errors                 | Yes (but issues with zeros) | Percentage             | Sales forecasting                   |
| Median Absolute Error   | Median absolute errors          | Yes                | Same as target           | Real estate with outliers           |
| Explained Variance Score| Variance explained              | Yes                | Unitless (0 to 1)        | Climate modeling                    |
| MSLE                    | Logarithmic relative errors     | Yes                | Log scale                | Website traffic prediction          |
| RMSLE                   | Logarithmic relative errors     | Yes                | Log scale                | Bike rental demand                  |
| Huber Loss              | Hybrid (MSE/MAE)                | Yes                | Depends on \( \delta \)  | Financial modeling with noise       |
| Quantile Loss           | Specific quantiles              | Yes                | Same as target           | Supply chain demand forecasting     |
| Tweedie Deviance        | Mixed distributions             | Varies             | Depends on \( p \)       | Insurance claim prediction          |
| AIC/BIC                 | Model fit vs. complexity        | N/A                | Unitless                 | ARIMA model selection               |

---

### Conclusion
Each of these metrics and loss functions serves a specific purpose in regression tasks, depending on the data characteristics, model goals, and application context. Choosing the right metric involves understanding the trade-offs between interpretability, robustness to outliers, and sensitivity to error magnitude or distribution. For example, MAE and MedAE are robust and interpretable, while MSE and RMSE emphasize larger errors. R² and Adjusted R² are useful for assessing explanatory power, while specialized metrics like MSLE, RMSLE, Huber Loss, and Tweedie Deviance cater to specific data types or modeling needs. AIC/BIC are critical for model selection, balancing fit and complexity.
