
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

<img width="175" height="63" alt="image" src="https://github.com/user-attachments/assets/2e3dc871-f661-4743-8219-0d4b898c61fd" />


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

<img width="253" height="61" alt="image" src="https://github.com/user-attachments/assets/49396d4b-acc6-4a4b-9ba6-edc93f35f5d8" />


Where:
<img width="316" height="184" alt="image" src="https://github.com/user-attachments/assets/942bf522-1606-49af-a309-db71c646cc99" />

### Objective

Maximize expected return:

<img width="227" height="121" alt="image" src="https://github.com/user-attachments/assets/e89513fc-1049-478d-9324-7c572d1294d7" />


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

<img width="339" height="44" alt="image" src="https://github.com/user-attachments/assets/9d4317c6-4df7-4325-a19b-ce3eb6737392" />


### Trajectory Score

<img width="348" height="100" alt="image" src="https://github.com/user-attachments/assets/4f4cee1b-497e-4816-9b53-d33853cc102d" />


If using a **value function** to approximate the tail:
<img width="476" height="112" alt="image" src="https://github.com/user-attachments/assets/0f1c1609-98a3-4690-9002-6463792ea2cd" />


This makes Beam Search **scalable**.

---

# 6. Beam Search Algorithm (Step-by-Step)

### Inputs

* Beam width ( k )
* Planning horizon ( H )
* Policy $( \pi(a|s) )$ or action generator
* Transition model ( P ) (learned or known)

---

### Initialization

<img width="273" height="78" alt="image" src="https://github.com/user-attachments/assets/bb1c64a5-13b7-4d93-af48-584ae162979f" />


---

### Iterative Expansion (for depth ( h = 1 \dots H ))

<img width="669" height="530" alt="image" src="https://github.com/user-attachments/assets/4db3327d-289b-4cbc-9b0e-2ccc9185820c" />


Collect all candidates:
<img width="271" height="92" alt="image" src="https://github.com/user-attachments/assets/1dd7aa33-596f-4024-bcfc-084b761f3d6b" />

---

### Pruning Step (Key Idea)

Keep top-k beams:
<img width="262" height="59" alt="image" src="https://github.com/user-attachments/assets/42a604f0-7027-4e4b-841a-d4d77ac8cf4d" />


---

### Action Selection

Choose best beam:
<img width="117" height="82" alt="image" src="https://github.com/user-attachments/assets/0a9f20c3-e371-4e8a-9c07-0b9999d32c43" />

Execute:

<img width="351" height="68" alt="image" src="https://github.com/user-attachments/assets/3bfe2ffc-b499-4a0a-9761-4ce62caae7df" />

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
<img width="202" height="139" alt="image" src="https://github.com/user-attachments/assets/974403b5-c956-4b40-9634-83b744e853db" />


Beam Search:
<img width="306" height="112" alt="image" src="https://github.com/user-attachments/assets/18fc9652-f133-4925-ae40-d2cb7be3181a" />


➡️ Beam Search approximates the **argmax over action sequences**, whereas Q-learning approximates the **argmax over actions recursively**.

---

# 9. Using Policy Probabilities (Log-Space Form)

Common in sequence models (e.g., RLHF, language models):

Score:
<img width="423" height="111" alt="image" src="https://github.com/user-attachments/assets/33e4fa9e-d913-40a2-8d14-71baaa356cd0" />


Or reward-augmented:
<img width="196" height="75" alt="image" src="https://github.com/user-attachments/assets/f331bf79-eb9e-48b8-b57c-425d93bfb77c" />


This blends **RL + probabilistic inference**.

---

# 10. Complexity Analysis

Let:

* $( A = |\mathcal{A}| )$
* ( H ) = horizon
* ( k ) = beam width

### Time Complexity

<img width="130" height="67" alt="image" src="https://github.com/user-attachments/assets/8ccd73d5-bb51-4ee7-9ff2-523d75a27165" />


### Space Complexity

<img width="100" height="48" alt="image" src="https://github.com/user-attachments/assets/c3ee4ad1-387d-43d9-af4c-5b9bcff7213b" />


Compare to exhaustive search:
![Uploading image.png…]()


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
