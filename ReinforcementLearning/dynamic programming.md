

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
<img width="682" height="247" alt="image" src="https://github.com/user-attachments/assets/303e8a45-03a5-40cb-8e08-6c8d86694d06" />

---

# 3. Value Functions (What DP Computes)

### State-Value Function
<img width="365" height="123" alt="image" src="https://github.com/user-attachments/assets/9486e2d3-342a-4c9c-bebe-d45f838ba04e" />


### Action-Value Function

<img width="463" height="111" alt="image" src="https://github.com/user-attachments/assets/0531650d-5824-466a-815f-4608104a1ca2" />

DP computes these **exactly**, given the model.

---

# 4. Bellman Expectation Equation (Policy Evaluation)

### Intuition

The value of a state =
**immediate reward + discounted value of next state**

---

### Bellman Expectation Equation (State Value)

<img width="596" height="101" alt="image" src="https://github.com/user-attachments/assets/182d2fd4-1aad-4ab0-9d8d-9e6d27db8a58" />


🔹 This is a **system of linear equations**
🔹 One equation per state

---

### Bellman Expectation Equation (Action Value)

<img width="674" height="93" alt="image" src="https://github.com/user-attachments/assets/e9f24b32-2ec1-49c1-9587-8c908732268f" />


---

# 5. Bellman Optimality Equation (Control)

Now we want the **optimal policy** ( \pi^* ).

---

### Optimal State-Value Function

<img width="273" height="88" alt="image" src="https://github.com/user-attachments/assets/4a7e3119-4ce8-4d8b-8a5f-1ae0e10f5f4e" />


### Bellman Optimality Equation

<img width="541" height="77" alt="image" src="https://github.com/user-attachments/assets/1768fda5-85a2-4462-8a22-322a304a32c7" />


This replaces **expectation over policy** with **max over actions**.

---

### Optimal Action-Value Equation

<img width="581" height="88" alt="image" src="https://github.com/user-attachments/assets/fa160847-f0c1-482b-8ed3-5f3b5fc1ec93" />


➡️ This is the **foundation of Q-learning**.

---

# 6. Why Bellman Equations Matter

Bellman equations:

* Decompose long-term return into **one-step lookahead**
* Enable **recursive computation**
* Turn RL into **fixed-point problems**

Mathematically:
<img width="156" height="66" alt="image" src="https://github.com/user-attachments/assets/cac3a90c-dd03-4a91-a9ff-a364f838fa33" />


Where $( \mathcal{T} )$ is the **Bellman operator**.

---

# 7. Bellman Operator & Contraction Mapping

### Bellman Optimality Operator

<img width="567" height="55" alt="image" src="https://github.com/user-attachments/assets/4dc25e38-c645-4dd1-b7dd-b3dc4d6144eb" />

### Contraction Property

<img width="378" height="69" alt="image" src="https://github.com/user-attachments/assets/fa08f9af-23cb-453c-945a-be9eb6421a80" />


✅ Guarantees **unique fixed point**
✅ Guarantees **convergence**

---

# 8. Core DP Algorithms in RL

## 1️⃣ Policy Evaluation (Prediction)

Compute ( V^\pi ) for a fixed policy.

### Iterative Update

<img width="613" height="67" alt="image" src="https://github.com/user-attachments/assets/d243d196-c6fb-4281-a2a3-66d990613152" />


Stop when:
[

<img width="216" height="56" alt="image" src="https://github.com/user-attachments/assets/4bbb912a-3251-47fe-809d-af8c66217a97" />

---

## 2️⃣ Policy Improvement

Make policy greedy w.r.t value function:

<img width="544" height="77" alt="image" src="https://github.com/user-attachments/assets/5b942583-763b-4acd-b910-02272fc44e65" />


### Policy Improvement Theorem

<img width="194" height="50" alt="image" src="https://github.com/user-attachments/assets/7ed3c7b1-52b0-472a-950b-0bd2aee110d7" />

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

<img width="782" height="384" alt="image" src="https://github.com/user-attachments/assets/d6054953-0d31-48b0-b9ea-1854d14e83cd" />

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
<img width="370" height="65" alt="image" src="https://github.com/user-attachments/assets/cd271b58-3788-4e31-8cc5-12a3232b4fab" />


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

