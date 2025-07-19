
---

## **3. Random Search**

### **Definition**

Random Search samples random combinations of hyperparameters from predefined ranges or distributions. Instead of testing every combination, it evaluates a fixed number of randomly selected configurations.

### **Mathematical Foundation**

* **Parameter Space**:
  Similar to Grid Search, $\Theta$ is the hyperparameter space, but parameters can be sampled from continuous distributions (e.g., uniform or log-uniform) rather than discrete values.

* **Sampling**:
  For a fixed budget of $n$ evaluations, randomly sample:

  $$
  \theta_1, \theta_2, \dots, \theta_n \sim \Theta
  $$

  For example:

  * Learning rate:
    $\alpha \sim \text{Uniform}(0.001, 1.0)$
  * Number of trees:
    $n \sim \text{DiscreteUniform}(50, 200)$

* **Objective Function**:
  Evaluate $f(\theta_i)$ for each sampled configuration, typically using cross-validation.

* **Optimization**:
  Select:

  $$
  \theta^* = \arg\max_{\theta_i} f(\theta_i)
  $$

### **Process**

1. Define the hyperparameter space (ranges or distributions).
2. Specify the number of iterations (budget).
3. Randomly sample hyperparameter combinations.
4. Train and evaluate the model for each sampled combination.
5. Select the best-performing configuration.

### **Advantages**

* **Efficient**: Explores a diverse set of configurations with fewer evaluations than Grid Search.
* **Flexible**: Can handle continuous parameter ranges and non-uniform distributions.
* **Scalable**: Better suited for high-dimensional spaces.
* **Surprisingly Effective**: Research (Bergstra & Bengio, 2012) shows Random Search often finds near-optimal solutions faster than Grid Search.

### **Disadvantages**

* **Non-Exhaustive**: May miss the global optimum since it relies on random sampling.
* **Unpredictable**: Results depend on the random seed.
* **Requires Tuning Budget**: Need to set the number of iterations.

### **Use Cases**

* High-dimensional parameter spaces.
* Exploratory tuning to find promising parameter ranges.
* Resource-constrained environments.
* Continuous parameters where grids are limiting.

### **Example Code: Scikit-learn RandomizedSearchCV**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.datasets import load_iris
from scipy.stats import randint

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = RandomForestClassifier(random_state=42)

# Define hyperparameter distributions
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10)
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Output results
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)

# Evaluate on test set
test_score = random_search.best_estimator_.score(X_test, y_test)
print("Test Set Score:", test_score)
```

**Example Output:**

```
Best Parameters: {'max_depth': None, 'min_samples_split': 4, 'n_estimators': 150}
Best Cross-Validation Score: 0.970
Test Set Score: 0.966
```

---

