Here’s a professional, cleanly formatted version of your text for academic or technical documentation:

---

## **2. Grid Search**

### **Definition**

Grid Search is an exhaustive search method that evaluates every possible combination of hyperparameters within predefined ranges. It systematically tests all combinations to identify the configuration that yields the best performance.

### **Mathematical Foundation**

* **Parameter Space**:
  Let $\Theta$ be the hyperparameter space, defined as a Cartesian product of discrete or continuous ranges for each hyperparameter. For example, if you have two hyperparameters:

  * Learning rate: $\alpha \in \{0.01, 0.1, 1.0\}$
  * Number of trees: $n \in \{50, 100, 200\}$

  Then,

  
  $ \Theta = \{(\alpha, n) \mid \alpha \in \{0.01, 0.1, 1.0\},\ n \in \{50, 100, 200\}\} $

  with $3 \times 3 = 9$ possible combinations.

* **Objective Function**:
  For each $\theta \in \Theta$, evaluate $f(\theta)$, typically the average performance (e.g., accuracy) across $k$-fold cross-validation:

  $ f(\theta) = \frac{1}{k} \sum_{i=1}^k \text{Score}(\theta, D_{\text{val}_i}) $

  where $D_{\text{val}_i}$ is the validation set for the $i$-th fold.

* **Optimization**:
  Find:

  $
  \theta^* = \arg\max_{\theta \in \Theta} f(\theta)
  $

  (or $\arg\min$ for loss functions).

### **Process**

1. Define the hyperparameter grid (e.g., ranges for learning rate, number of estimators).
2. For each combination of hyperparameters in the grid:

   * Train the model using the training set.
   * Evaluate performance using cross-validation.
3. Select the combination with the best performance.

### **Advantages**

* **Exhaustive**: Guarantees finding the optimal configuration within the specified grid.
* **Simple**: Easy to implement and understand.
* **Reproducible**: Consistent results for the same grid and dataset.

### **Disadvantages**

* **Computationally Expensive**:
  The number of combinations grows exponentially with the number of hyperparameters and their possible values (curse of dimensionality).
  For $n$ hyperparameters with $m$ values each:

  $
  \text{Total evaluations} = m^n
  $

* **Inefficient**: Tests many unpromising combinations, wasting resources.

* **Limited by Grid**: Only explores predefined discrete values, potentially missing optimal values between grid points.

### **Use Cases**

* Small parameter spaces (e.g., tuning a simple model like logistic regression with 2–3 parameters).
* High-precision requirements where exhaustive search is necessary.
* Prototyping: exploring hyperparameter ranges in small-scale experiments.

### **Example Code: Scikit-learn GridSearchCV**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = RandomForestClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Output results
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate on test set
test_score = grid_search.best_estimator_.score(X_test, y_test)
print("Test Set Score:", test_score)
```

**Example Output:**

```
Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
Best Cross-Validation Score: 0.975
Test Set Score: 0.966
```

---

Let me know if you’d like this in a specific format (Markdown, PDF, HTML, etc.).
