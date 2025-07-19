
---

## **4. Other Hyperparameter Optimization Methods**

### **4.1 Bayesian Optimization**

**Definition**
Bayesian Optimization models the objective function $f(\theta)$ using a probabilistic surrogate (e.g., Gaussian Process or Tree-structured Parzen Estimator) and selects promising hyperparameter configurations by optimizing an acquisition function.

**Mathematical Foundation**

* **Surrogate Model**
  Assume $f(\theta)$ is expensive to evaluate. Model it with a probabilistic function:

  $f(\theta) \sim \text{GP}(m(\theta), k(\theta, \theta'))$

  where $m(\theta)$ is the mean function and $k(\theta, \theta')$ is the covariance (kernel) function.

* **Acquisition Function**
  Select the next $\theta$ by optimizing an acquisition function, e.g., Expected Improvement (EI):

  
  $\text{EI}(\theta) = \mathbb{E}[\max(f(\theta) - f(\theta^*), 0)]$

  where $\theta^*$ is the current best configuration.

* **Optimization Process**

  1. Initialize with a few random evaluations.
  2. Fit the surrogate model to observed $(\theta, f(\theta))$.
  3. Optimize the acquisition function to select the next $\theta$.
  4. Evaluate $f(\theta)$, update the surrogate, repeat.

**Advantages**

* Sample-efficient.
* Handles continuous hyperparameters.
* Prioritizes promising regions automatically.

**Disadvantages**

* Complex setup.
* Surrogate model fitting can be computationally expensive.
* Risk of local optima.

**Use Cases**

* Expensive model training.
* Continuous hyperparameters (e.g., learning rate, dropout).
* Complex models like neural networks.

**Popular Tools**: Optuna, Hyperopt, Scikit-Optimize, GPyOpt.

**Example Code (Optuna)**

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define objective function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Output results
print("Best Parameters:", study.best_params)
print("Best Cross-Validation Score:", study.best_value)

# Evaluate on test set
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
print("Test Set Score:", best_model.score(X_test, y_test))
```

**Example Output:**

```
Best Parameters: {'n_estimators': 120, 'max_depth': None, 'min_samples_split': 3}
Best Cross-Validation Score: 0.975
Test Set Score: 0.966
```

---

### **4.2 Genetic Algorithms**

**Definition**
Genetic Algorithms (GAs) simulate natural selection to evolve hyperparameter configurations. Configurations ("individuals") evolve through mutation, crossover, and selection over generations.

**Mathematical Foundation**

* **Population**:
  A set of configurations:

  
  $\{\theta_1, \theta_2, \dots, \theta_N\}$

* **Fitness Function**:
  Evaluate each configuration using $f(\theta_i)$.

* **Evolution Steps**:

  * **Selection**: Prefer high-performing configurations.
  * **Crossover**: Combine parameters of two "parents."
  * **Mutation**: Randomly alter parameters.
  * **Iteration**: Repeat until stopping criteria.

**Advantages**

* Global search; can avoid local optima.
* Handles both discrete and continuous parameters.
* Naturally parallelizable.

**Disadvantages**

* Requires tuning of GA-specific parameters.
* Potentially slow convergence.
* Results may vary between runs.

**Use Cases**

* Complex or high-dimensional models.
* Neural architecture search.
* Non-differentiable parameter spaces.

**Popular Tools**: DEAP, PyGAD, TPOT.

---

### **4.3 Gradient-Based Optimization**

**Definition**
Gradient-based methods optimize continuous hyperparameters using gradients, often integrated into training loops (e.g., learning rate scheduling).

**Mathematical Foundation**

* Use gradients of the objective:

  $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$

**Advantages**

* Fast optimization for continuous, differentiable parameters.
* Can be integrated directly into training.

**Disadvantages**

* Only applicable to differentiable hyperparameters.
* Risk of getting trapped in local minima.

**Use Cases**

* Learning rate tuning.
* Training neural networks where hyperparameters influence gradients.

**Popular Tools**: PyTorch, TensorFlow (with custom optimization).

---


