# 1. What Is a Markov Reward Process? 

A **Markov Reward Process (MRP)** is:

> A **Markov Chain** where **each state gives you a reward**.

That’s it.

There are:

* **States**
* **Probabilities of moving between states**
* **Rewards for being in states**

❌ No actions
❌ No decisions

The system just **runs by itself**.

---

# 2. Why Do We Need an MRP?

Let’s look at the progression:

```
Markov Chain → Markov Reward Process → Markov Decision Process
```

| Model        | States | Rewards | Actions |
| ------------ | ------ | ------- | ------- |
| Markov Chain | ✅      | ❌       | ❌       |
| MRP          | ✅      | ✅       | ❌       |
| MDP          | ✅      | ✅       | ✅       |

MRP lets us:

* Talk about **long-term reward**
* Define **value functions**
* Introduce the **Bellman equation**

Without MRP, RL math would feel like it appears from nowhere.

---

# 3. Components of a Markov Reward Process

An MRP is defined as:

<img width="220" height="92" alt="image" src="https://github.com/user-attachments/assets/13513a9e-2e9e-45c9-bf90-16e7d1ce7f6e" />


Let’s explain **every symbol** in simple terms.

---

## 3.1 State Space (S)

A **state** is a snapshot of the system right now.

Examples:

* Square in a grid
* Stage of a process
* Page you are on

<img width="217" height="63" alt="image" src="https://github.com/user-attachments/assets/df19f486-4241-422b-8013-9a8a53f86e34" />

---

## 3.2 Transition Probability Matrix (P)

<img width="133" height="52" alt="image" src="https://github.com/user-attachments/assets/a0fdadaa-abc2-4047-82b2-0788d8b37e2a" />


Means:

> Probability of moving to state ( s' ) from state ( s )

Properties:

<img width="567" height="168" alt="image" src="https://github.com/user-attachments/assets/9fe310cd-c46a-4b64-b6f7-90c688fcdb57" />

This is the **same as in Markov Chains**.

---

## 3.3 Reward Function (R)

The **reward** tells how good a state is.

### State-based reward:

<img width="258" height="74" alt="image" src="https://github.com/user-attachments/assets/7aead913-e5cc-4600-bffe-4984390a5ed8" />


Example:

* +1 for goal state
* −1 for bad state
* 0 for neutral state

Sometimes rewards depend on transitions:

<img width="117" height="48" alt="image" src="https://github.com/user-attachments/assets/e8f3b436-3ef2-40b1-b6d2-55b76efea91d" />


---

## 3.4 Discount Factor (γ)

<img width="160" height="53" alt="image" src="https://github.com/user-attachments/assets/2e77b42c-85f4-4516-8db3-c8ee6bfa5b5a" />

Controls how much we care about the future.

| γ value | Meaning             |
| ------- | ------------------- |
| 0       | Only care about now |
| 0.9     | Care about future   |
| 1       | No discount (rare)  |

---

# 4. The Markov Property (Why It Works)

MRP follows the **Markov property**:

> The future depends only on the current state.

Mathematically:

<img width="457" height="73" alt="image" src="https://github.com/user-attachments/assets/469697c4-9265-4220-8ae4-e2761ba9f980" />


This allows **recursive value computation**.

---

# 5. Return (Total Reward Over Time)

The **return** is the total reward you collect.

<img width="390" height="45" alt="image" src="https://github.com/user-attachments/assets/73d985d9-d726-4750-8f79-217082675ea1" />


Or:

<img width="287" height="155" alt="image" src="https://github.com/user-attachments/assets/957881ce-7a7e-45fd-8ab7-678b93edbbfa" />

---

# 6. Value Function (Main Quantity in MRP)

### State Value Function

<img width="324" height="95" alt="image" src="https://github.com/user-attachments/assets/92458525-1a83-4c99-9879-e3dd8ec2df9f" />

Meaning:

> Expected total future reward if you start in state s.

This is **the main thing we want to compute** in an MRP.

---

# 7. Bellman Equation for MRP (Core Equation)

This is where everything connects.

---

## 7.1 Intuition

> Value now = reward now + value of what comes next

---

## 7.2 Bellman Expectation Equation (MRP)

<img width="389" height="71" alt="image" src="https://github.com/user-attachments/assets/9214c287-c060-4688-8389-e710e8dd42e9" />

---

## 7.3 What Each Term Means

| Term       | Meaning             |                           |
| ---------- | ------------------- | ------------------------- |
| ( V(s) )   | Value of state s    |                           |
| ( R(s) )   | Immediate reward    |                           |
| ( P(s'     | s) )                | Probability of next state |
| ( V(s') )  | Value of next state |                           |
| ( \gamma ) | Discount factor     |                           |

---

# 8. Bellman Equation (Matrix Form)

For all states together:

<img width="214" height="60" alt="image" src="https://github.com/user-attachments/assets/6e4b2f3e-aa38-489e-aa22-d23f5c7212b5" />


Rearranging:

<img width="219" height="87" alt="image" src="https://github.com/user-attachments/assets/23232db7-c11d-4cc2-9107-ba3d5fb35700" />


Closed-form solution:

<img width="255" height="54" alt="image" src="https://github.com/user-attachments/assets/803f372b-1938-4265-beef-b37261d52870" />

This shows MRP can be **solved exactly**.

---

# 9. How to Compute Value Function

### 1️⃣ Direct Solution

* Solve linear equations
* Works for small state spaces

### 2️⃣ Iterative Solution

<img width="455" height="97" alt="image" src="https://github.com/user-attachments/assets/4f817fdd-d363-4dfe-8508-fef68c0bee17" />


Converges due to contraction property.

---

# 10. Why Convergence Is Guaranteed

The Bellman operator is a **contraction**:

<img width="359" height="62" alt="image" src="https://github.com/user-attachments/assets/799ac900-aee3-4e44-b2e1-c5dd3bfeebef" />


Guarantees:

* Unique solution
* Stable convergence

---

# 11. Types of MRPs

### Continuing MRP

* No terminal state
* Infinite horizon

### Episodic MRP

* Ends in terminal state
* Terminal state has value 0

---

# 12. Terminal States

A **terminal state**:

* Ends the process
* No future reward

<img width="199" height="84" alt="image" src="https://github.com/user-attachments/assets/3ff00451-14ff-443d-bd35-c527c0b067ca" />

---

# 13. Real-World Examples

* Student progressing through grades
* Machine degrading over time
* Customer journey
* Board game without choices

---

# 14. MRP vs Markov Chain vs MDP

| Model        | Reward | Action |
| ------------ | ------ | ------ |
| Markov Chain | ❌      | ❌      |
| MRP          | ✅      | ❌      |
| MDP          | ✅      | ✅      |

---

# 15. How MRP Leads to Reinforcement Learning

MRP introduces:

* Returns
* Value functions
* Bellman equations

RL builds on this by:

* Adding **actions**
* Learning **policies**

---

# 16. Common Confusions

❌ MRP makes decisions
✅ MRP only evaluates outcomes

❌ MRP learns from experience
✅ MRP assumes known probabilities

---

# 17. One-Line Summary

> **A Markov Reward Process is a Markov Chain with rewards, used to compute the long-term value of states using the Bellman equation.**

---

# 18. Big Picture

```
Markov Chain → MRP → MDP → RL Algorithms
```

MRP is where **value functions are born**.

---


