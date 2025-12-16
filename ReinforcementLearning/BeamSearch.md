Below is a **from-scratch, end-to-end explanation of Beam Search in the context of Reinforcement Learning (RL)**, including the **mathematical foundations**, **algorithmic steps**, and **how it differs from classical RL action selection**.

I’ll structure this so you can build intuition first, then formal understanding.

---

# 1. What Is Beam Search (Core Idea)

**Beam Search** is a **heuristic search algorithm** that keeps the **top-k best partial solutions** at each step instead of exploring all possibilities.

* It is a **compromise** between:

  * **Greedy search** (k = 1 → fast but myopic)
  * **Exhaustive search** (k = ∞ → optimal but infeasible)

In RL, Beam Search is typically used for:

* **Planning**
* **Trajectory optimization**
* **Policy rollouts**
* **Sequence decision problems** (e.g., language models, robotics, game planning)

> ⚠️ Important: Beam Search is **not a learning algorithm** itself — it is a **decision-time planning / inference strategy**.

---

# 2. Where Beam Search Fits in Reinforcement Learning

### Classical RL Loop

[
(s_t, a_t, r_t, s_{t+1})
]

Normally, RL chooses actions via:

* ε-greedy
* Softmax
* Policy sampling

### Beam Search Instead:

* Considers **multiple future action sequences**
* Scores entire trajectories
* Chooses the **best first action** from the best trajectory

This is closer to **model-based RL** or **planning-augmented RL**.

---

# 3. Problem Setup (Formal)

### Markov Decision Process (MDP)

[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
]

Where:

* ( \mathcal{S} ): states
* ( \mathcal{A} ): actions
* ( P(s'|s,a) ): transition model
* ( R(s,a) ): reward
* ( \gamma \in [0,1] ): discount factor

### Objective

Maximize expected return:
[
G_t = \sum_{k=0}^{T} \gamma^k r_{t+k}
]

---

# 4. Beam Search in RL (High-Level View)

At time step ( t ):

1. Start from current state ( s_t )
2. Expand **possible action sequences** up to depth ( H )
3. Keep only the **top-k trajectories** at each depth
4. Choose the action sequence with the **highest total return**
5. Execute **only the first action**
6. Repeat at next timestep

This is **receding horizon planning**.

---

# 5. Mathematical Scoring of a Beam

Each beam represents a **partial trajectory**:
[
\tau = (s_t, a_t, s_{t+1}, a_{t+1}, ..., s_{t+h})
]

### Trajectory Score

[
J(\tau) = \sum_{k=0}^{h-1} \gamma^k R(s_{t+k}, a_{t+k})
]

If using a **value function** to approximate the tail:
[
J(\tau) = \sum_{k=0}^{h-1} \gamma^k R(s_{t+k}, a_{t+k}) + \gamma^h V(s_{t+h})
]

This makes Beam Search **scalable**.

---

# 6. Beam Search Algorithm (Step-by-Step)

### Inputs

* Beam width ( k )
* Planning horizon ( H )
* Policy ( \pi(a|s) ) or action generator
* Transition model ( P ) (learned or known)

---

### Initialization

[
\mathcal{B}_0 = { (s_t, J=0) }
]

---

### Iterative Expansion (for depth ( h = 1 \dots H ))

For each beam ( b \in \mathcal{B}_{h-1} ):

1. Generate candidate actions:
   [
   a \sim \pi(a|s)
   ]
2. Predict next state:
   [
   s' \sim P(s'|s,a)
   ]
3. Compute incremental reward:
   [
   r = R(s,a)
   ]
4. Update score:
   [
   J' = J + \gamma^{h-1} r
   ]

Collect all candidates:
[
\mathcal{C}*h = \bigcup*{b \in \mathcal{B}_{h-1}} \text{Expand}(b)
]

---

### Pruning Step (Key Idea)

Keep top-k beams:
[
\mathcal{B}*h = \text{Top-k}*{J}(\mathcal{C}_h)
]

---

### Action Selection

Choose best beam:
[
\tau^* = \arg\max_{\tau \in \mathcal{B}_H} J(\tau)
]

Execute:
[
a_t = \tau^*_0
]

---

# 7. Beam Search vs Value-Based RL (Key Difference)

| Aspect         | Value-Based RL | Beam Search RL  |
| -------------- | -------------- | --------------- |
| Decision       | Single action  | Action sequence |
| Lookahead      | Implicit       | Explicit        |
| Planning       | No             | Yes             |
| Requires model | No             | Yes             |
| Optimality     | Asymptotic     | Approximate     |

Beam Search **explicitly reasons about future** instead of relying purely on learned Q-values.

---

# 8. Relation to Q-Learning (Mathematical Link)

Q-learning:
[
Q(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q(s',a')]
]

Beam Search:
[
\max_{a_{0:H}} \sum_{t=0}^{H} \gamma^t r_t
]

➡️ Beam Search approximates the **argmax over action sequences**, whereas Q-learning approximates the **argmax over actions recursively**.

---

# 9. Using Policy Probabilities (Log-Space Form)

Common in sequence models (e.g., RLHF, language models):

Score:
[
J(\tau) = \sum_{t=0}^{H} \log \pi(a_t|s_t)
]

Or reward-augmented:
[
J(\tau) = \sum_{t=0}^{H} \left( \log \pi(a_t|s_t) + \lambda R(s_t,a_t) \right)
]

This blends **RL + probabilistic inference**.

---

# 10. Complexity Analysis

Let:

* ( A = |\mathcal{A}| )
* ( H ) = horizon
* ( k ) = beam width

### Time Complexity

[
O(H \cdot k \cdot A)
]

### Space Complexity

[
O(k \cdot H)
]

Compare to exhaustive search:
[
O(A^H)
]

Beam Search is **tractable but approximate**.

---

# 11. Strengths and Weaknesses

### ✅ Advantages

* Strong lookahead
* Handles sparse rewards
* Works well with learned world models
* Reduces catastrophic short-term decisions

### ❌ Limitations

* Not guaranteed optimal
* Sensitive to beam width
* Requires accurate model or value estimates
* Can collapse to similar trajectories

---

# 12. When Beam Search Is Used in RL

* **Model-Based RL**
* **Trajectory optimization**
* **MPC (Model Predictive Control)**
* **RLHF decoding**
* **Game AI**
* **Robotics planning**
* **Language-action models**

---

# 13. Conceptual Summary (One Sentence)

> **Beam Search in Reinforcement Learning is a planning algorithm that approximates the optimal action sequence by maintaining the top-k highest-reward trajectories over a finite horizon using a learned or known environment model.**

---
.
