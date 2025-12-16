

---

# 1. What Is Dynamic Programming in Reinforcement Learning?

**Dynamic Programming (DP)** is a family of methods for **solving Markov Decision Processes (MDPs)** when:

✅ The **environment model is known**

* Transition probabilities ( P(s'|s,a) )
* Reward function ( R(s,a) )

❌ Not suitable when the model is unknown (that’s where Monte Carlo & TD come in).

---

### Core DP Idea (Bellman’s Principle of Optimality)

> *An optimal policy has the property that whatever the initial state, the remaining decisions must themselves be optimal.*

This leads directly to the **Bellman equations**.

---

# 2. MDP Formal Definition (Foundation)

An MDP is defined as:
[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
]

* ( \mathcal{S} ): states
* ( \mathcal{A} ): actions
* ( P(s'|s,a) ): transition probability
* ( R(s,a) ): expected reward
* ( \gamma \in [0,1) ): discount factor

---

# 3. Value Functions (What DP Computes)

### State-Value Function

[
V^\pi(s) = \mathbb{E}*\pi \left[ \sum*{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right]
]

### Action-Value Function

[
Q^\pi(s,a) = \mathbb{E}*\pi \left[ \sum*{t=0}^{\infty} \gamma^t r_t \mid s_0=s, a_0=a \right]
]

DP computes these **exactly**, given the model.

---

# 4. Bellman Expectation Equation (Policy Evaluation)

### Intuition

The value of a state =
**immediate reward + discounted value of next state**

---

### Bellman Expectation Equation (State Value)

[
V^\pi(s) = \sum_{a} \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma V^\pi(s') \right]
]

🔹 This is a **system of linear equations**
🔹 One equation per state

---

### Bellman Expectation Equation (Action Value)

[
Q^\pi(s,a) =
\sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a') \right]
]

---

# 5. Bellman Optimality Equation (Control)

Now we want the **optimal policy** ( \pi^* ).

---

### Optimal State-Value Function

[
V^*(s) = \max_\pi V^\pi(s)
]

### Bellman Optimality Equation

[
V^*(s) =
\max_{a}
\sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma V^*(s') \right]
]

This replaces **expectation over policy** with **max over actions**.

---

### Optimal Action-Value Equation

[
Q^*(s,a) =
\sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma \max_{a'} Q^*(s',a') \right]
]

➡️ This is the **foundation of Q-learning**.

---

# 6. Why Bellman Equations Matter

Bellman equations:

* Decompose long-term return into **one-step lookahead**
* Enable **recursive computation**
* Turn RL into **fixed-point problems**

Mathematically:
[
V = \mathcal{T} V
]

Where ( \mathcal{T} ) is the **Bellman operator**.

---

# 7. Bellman Operator & Contraction Mapping

### Bellman Optimality Operator

[
(\mathcal{T}V)(s) =
\max_a \sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma V(s') \right]
]

### Contraction Property

[
|\mathcal{T}V_1 - \mathcal{T}V_2|*\infty
\le \gamma |V_1 - V_2|*\infty
]

✅ Guarantees **unique fixed point**
✅ Guarantees **convergence**

---

# 8. Core DP Algorithms in RL

## 1️⃣ Policy Evaluation (Prediction)

Compute ( V^\pi ) for a fixed policy.

### Iterative Update

[
V_{k+1}(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma V_k(s') \right]
]

Stop when:
[
\max_s |V_{k+1}(s) - V_k(s)| < \epsilon
]

---

## 2️⃣ Policy Improvement

Make policy greedy w.r.t value function:

[
\pi'(s) =
\arg\max_a
\sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma V^\pi(s') \right]
]

### Policy Improvement Theorem

[
V^{\pi'}(s) \ge V^\pi(s)
]

---

## 3️⃣ Policy Iteration (PI)

Alternate:

1. **Policy Evaluation**
2. **Policy Improvement**

Until policy converges.

### Guarantee

Converges to ( \pi^* ) in **finite steps** (for finite MDPs).

---

## 4️⃣ Value Iteration (VI)

Combine evaluation + improvement.

### Update Rule

[
V_{k+1}(s) =
\max_a
\sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma V_k(s') \right]
]

Stop when:
[
\max_s |V_{k+1}(s) - V_k(s)| < \epsilon
]

Then extract policy:
[
\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a)
\left[ R(s,a) + \gamma V(s') \right]
]

---

# 9. Policy Iteration vs Value Iteration

| Aspect        | Policy Iteration | Value Iteration |
| ------------- | ---------------- | --------------- |
| Evaluation    | Full             | Partial         |
| Updates       | More expensive   | Cheaper         |
| Convergence   | Fewer iterations | More iterations |
| Practical use | Small MDPs       | Larger MDPs     |

---

# 10. Example (Small Intuition)

If:

* Reward now = 1
* Next state value = 10
* ( \gamma = 0.9 )

Then:
[
V(s) = 1 + 0.9 \times 10 = 10
]

DP repeatedly applies this logic **until self-consistent**.

---

# 11. Limitations of Dynamic Programming

❌ Requires full model
❌ State space must be small
❌ Computationally expensive
❌ Not suitable for continuous spaces

➡️ Leads to:

* **Monte Carlo methods**
* **Temporal Difference learning**
* **Deep RL**

---

# 12. How DP Connects to Modern RL

| DP Concept         | Modern RL Equivalent |
| ------------------ | -------------------- |
| Bellman equation   | TD target            |
| Value iteration    | Q-learning           |
| Policy evaluation  | Critic               |
| Policy improvement | Actor                |

---

# 13. One-Line Summary

> **Dynamic Programming in Reinforcement Learning solves MDPs by using Bellman equations to recursively compute optimal value functions through policy evaluation and improvement, assuming a known environment model.**

---

