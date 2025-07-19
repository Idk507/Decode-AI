

---

# 🎰 Bandit-Based Methods: Hyperband & BOHB

Bandit-based methods optimize hyperparameters by treating hyperparameter optimization as a **multi-armed bandit problem**, balancing **exploration vs. exploitation** efficiently. They are especially useful when model evaluations are **expensive**.

---

## ⚡ Why Bandit-Based Methods?

* Training full models for all hyperparameter configurations is expensive.
* Early stopping can eliminate poor configurations faster.
* Focus resources on promising candidates.

---

# 1️⃣ **Hyperband**

## 🚀 What is Hyperband?

Hyperband is a **resource allocation algorithm** designed for hyperparameter optimization. It builds upon the **Successive Halving (SH)** method but handles the trade-off between exploring many configurations and exploiting few high-performing ones efficiently.

---

## 📐 Mathematical Foundation

### Key Terms:

* **R**: Maximum total resources allocated to any single configuration (e.g., total epochs).
* **η**: Halving factor (controls how aggressively bad configurations are eliminated).
* **n**: Number of configurations evaluated in each round.

---

### 🎯 Successive Halving (SH)

1. Evaluate **n** configurations.
2. Train each configuration for **r** resources (e.g., epochs).
3. Select top **1/η** configurations.
4. Allocate **η×r** resources to remaining configurations.
5. Repeat until one configuration remains.

---

### 🔁 Hyperband Algorithm

Instead of running SH just once, Hyperband runs multiple SH iterations with different starting configurations vs. resource allocation trade-offs.

**Number of brackets:**

$$
s_{max} = \lfloor \log_\eta(R) \rfloor
$$

Each bracket **s** varies between:

* More configurations with fewer resources (**exploration**).
* Fewer configurations with more resources (**exploitation**).

For bracket **s**, compute:

* Number of configurations:

$$
n = \left\lceil \frac{s_{max} + 1}{s + 1} \cdot \eta^s \right\rceil
$$

* Initial resources:

$$
r = \frac{R}{\eta^s}
$$

Hyperband runs SH with these parameters in each bracket.

---

## 🔄 Hyperband Workflow

1. For each bracket **s**:

   * Run Successive Halving with **n** configurations and **r** resources.
2. Keep track of best-performing configuration across all brackets.

---

## ✅ Advantages of Hyperband

* Efficient resource allocation.
* Automatic early stopping.
* Scales well to large search spaces.

## ⚠️ Limitations

* Random sampling of configurations (no modeling of search space).
* May waste resources if search space is highly structured.

---

## 🛠️ Hyperband Python Example (Using Scikit-Optimize)

Hyperband is available in libraries like **Ray Tune** and **Keras Tuner**. Example using **Keras Tuner**:

```python
import keras_tuner as kt
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential([
        keras.layers.Dense(
            units=hp.Int('units', min_value=32, max_value=512, step=32),
            activation='relu'
        ),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='my_dir',
    project_name='hyperband_tuning'
)

tuner.search(x_train, y_train, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models(1)[0]
```

---

# 2️⃣ **BOHB (Bayesian Optimization + Hyperband)**

## 🚀 What is BOHB?

BOHB combines **Bayesian Optimization (BO)** with **Hyperband** to model the hyperparameter space, using prior knowledge to guide sampling while retaining Hyperband's efficient resource allocation.

* **Bayesian Optimization** explores the space more intelligently using a surrogate model (usually a Tree Parzen Estimator or Gaussian Process).
* **Hyperband** handles resource allocation via early stopping.

---

## 📐 BOHB Mathematical Steps

1. **Surrogate Model:**
   Builds a probabilistic model (e.g., TPE) of the objective function:

   $$
   p(y|x)
   $$

   Where:

   * \$x\$ = hyperparameter configuration
   * \$y\$ = observed loss or validation error.

2. **Acquisition Function:**
   Selects the next configuration using:

   $ x_{next} = \arg\max_x \ \alpha(x)$

   Where \$\alpha(x)\$ could be Expected Improvement (EI), Upper Confidence Bound (UCB), etc.

3. **Resource Allocation:**
   Once configurations are sampled, Hyperband manages how much resource to assign using successive halving.

---

## ✅ Advantages of BOHB

* Smarter sampling (guided by Bayesian models).
* Efficient resource allocation (using Hyperband).
* Balances exploration and exploitation better than Hyperband alone.

## ⚠️ Disadvantages

* Computational overhead due to surrogate model training.
* Requires more memory to track historical evaluations.

---

## 🛠️ BOHB Python Example (Using HpBandSter)

```python
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
import numpy as np

# Define search space
cs = CS.ConfigurationSpace()
cs.add_hyperparameter(CS.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-1, log=True))
cs.add_hyperparameter(CS.UniformIntegerHyperparameter('batch_size', lower=16, upper=128))

# Define objective function
def my_worker(config, budget):
    lr = config['lr']
    batch_size = config['batch_size']
    # Simulate training a model
    loss = (0.2 + lr) + (100. / batch_size)  # Dummy loss
    return {'loss': loss, 'info': {}}

# Start NameServer
NS = hpns.NameServer(run_id='bohb_example', host='127.0.0.1', port=None)
NS.start()

# Run BOHB
bohb = BOHB(
    configspace=cs,
    run_id='bohb_example',
    min_budget=1,
    max_budget=9
)
res = bohb.run(n_iterations=20)

# Fetch and print best configuration
id2config = res.get_id2config_mapping()
inc_id = res.get_incumbent_id()
print('Best found configuration:', id2config[inc_id]['config'])
```

---

# 📊 Summary Comparison

| Method    | Sampling Strategy           | Resource Allocation | Strengths                |
| --------- | --------------------------- | ------------------- | ------------------------ |
| Hyperband | Random Search               | Successive Halving  | Fast, early stopping     |
| BOHB      | Bayesian Optimization (TPE) | Successive Halving  | Smarter search, adaptive |

---

# 💡 Conclusion

* Use **Hyperband** when you want fast, early-stopping-driven optimization without needing prior knowledge.
* Use **BOHB** when you want to model the search space intelligently while retaining Hyperband’s efficiency.

---

