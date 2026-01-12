---

# 1. What Is Monte Carlo in Reinforcement Learning? (Very Simple)

**Monte Carlo (MC)** methods learn by:

> **Watching what actually happens and averaging the results**

In Reinforcement Learning terms:

> **Run the agent → finish the episode → look at the total reward → average over many episodes**

That’s it.

No environment model
No transition probabilities
Just **experience and averaging**

---

# 2. Why Is It Called “Monte Carlo”?

The name comes from **random sampling** (like casino games in Monte Carlo).

Monte Carlo methods:

* Use **random samples**
* Estimate quantities using **averages**
* Get better with **more data**

---

# 3. What Problem Monte Carlo Solves

In RL, we want to know:

* How good is a state?
* How good is an action?

Monte Carlo answers this by:

* Trying it many times
* Observing total reward
* Averaging the results

---

# 4. Where Monte Carlo Fits in RL

Recall the RL family:

```
Dynamic Programming → Monte Carlo → Temporal Difference
```

| Method      | Model Needed? | Learns From       |
| ----------- | ------------- | ----------------- |
| DP          | Yes           | Full model        |
| Monte Carlo | No            | Complete episodes |
| TD          | No            | Partial episodes  |

Monte Carlo is **model-free**.

---

# 5. Core Assumptions of Monte Carlo Methods

Monte Carlo methods assume:

1. **Episodic tasks** (episodes must end)
2. **Finite return**
3. Ability to **sample complete episodes**

---

# 6. Important Terms (Explained Simply)

---

## 6.1 Episode

An **episode** is one complete run:

* Start state → terminal state

Example:

* One game of chess
* One maze run

---

## 6.2 Trajectory

A sequence of states, actions, and rewards:

<img width="268" height="59" alt="image" src="https://github.com/user-attachments/assets/64682e72-13a8-4a98-ac95-0bea81ac0ebf" />


---

## 6.3 Return (G)

The **return** is the total reward collected.

<img width="368" height="69" alt="image" src="https://github.com/user-attachments/assets/00830b92-15c4-4157-bc98-395301d3fd71" />


This is what Monte Carlo estimates.

---

## 6.4 Discount Factor (γ)

Controls how much future rewards matter.

* γ close to 1 → long-term thinking
* γ close to 0 → short-term thinking

---

# 7. Monte Carlo Value Estimation (State Value)

### Goal:

Estimate:

<img width="79" height="58" alt="image" src="https://github.com/user-attachments/assets/0357bf89-668b-49e1-a86e-d18eb3fa2853" />

Meaning:

> Average return after visiting state s

---

## 7.1 Basic Idea

1. Visit state s
2. Finish episode
3. Compute return
4. Average returns

---

## 7.2 Monte Carlo Estimate

<img width="287" height="110" alt="image" src="https://github.com/user-attachments/assets/44ee1889-a745-46b4-81d8-18edb01b48d1" />

Where:

* ( N(s) ) = number of times state s was visited
* ( G_i(s) ) = return after i-th visit

---

# 8. First-Visit vs Every-Visit Monte Carlo

---

## 8.1 First-Visit MC

Only use the **first time** a state appears in an episode.

✔️ Lower correlation
✔️ Simpler theory

---

## 8.2 Every-Visit MC

Use **every occurrence** of a state.

✔️ More data
✔️ Faster learning

Both converge to the same value.

---

# 9. Monte Carlo for Action Values (Q-Values)

We can estimate:

<img width="128" height="72" alt="image" src="https://github.com/user-attachments/assets/2059af18-9e2c-4d08-90d1-e840d05f582d" />


Same idea:

* Look at returns after taking action a in state s
* Average them

<img width="340" height="94" alt="image" src="https://github.com/user-attachments/assets/72146890-0777-474c-ac33-a107e265fa6e" />

---

# 10. Incremental Monte Carlo Update (Practical Form)

Instead of storing all returns:

<img width="355" height="68" alt="image" src="https://github.com/user-attachments/assets/27fc5531-cd18-4f49-8975-57cde75c8c4d" />

Where:
<img width="221" height="71" alt="image" src="https://github.com/user-attachments/assets/71233fd3-72ce-4214-b10b-cfa44dc34f38" />


This is just **running average**.

---

# 11. Monte Carlo Control (Learning the Best Policy)

So far we only **evaluate** a policy.

Now we want to **improve** it.

---

## 11.1 Monte Carlo Control Loop

1. Start with any policy
2. Generate episode
3. Estimate Q(s,a)
4. Make policy greedy
5. Repeat

---

## 11.2 ε-Greedy Policy

To ensure exploration:

<img width="407" height="128" alt="image" src="https://github.com/user-attachments/assets/1ee4fe50-5beb-4484-95d0-1acf5ca72876" />


This ensures:

* All actions are tried
* Convergence is guaranteed

---

# 12. On-Policy vs Off-Policy Monte Carlo

---

## 12.1 On-Policy MC

* Learn **the same policy you follow**
* Uses ε-greedy

Example:

> SARSA-like behavior

---

## 12.2 Off-Policy MC

* Learn one policy
* Follow another

Uses **importance sampling**.

---

# 13. Importance Sampling (Simple Explanation)

If:

* Behavior policy = b
* Target policy = π

Correction factor:

<img width="206" height="82" alt="image" src="https://github.com/user-attachments/assets/327e152d-1860-4fb3-8a34-f8f1c5415ca4" />

This adjusts returns so they look like they came from π.

---

# 14. Monte Carlo vs Bellman Equation

| Aspect           | Monte Carlo |
| ---------------- | ----------- |
| Bootstrapping    | ❌ No        |
| Uses Bellman     | ❌ No        |
| Uses full return | ✅ Yes       |
| Bias             | ❌           |
| Variance         | ✅ High      |

Monte Carlo uses **true returns**, not estimates.

---

# 15. Strengths of Monte Carlo Methods

✅ Simple
✅ Unbiased estimates
✅ Model-free
✅ Works well when episodes are short

---

# 16. Weaknesses of Monte Carlo Methods

❌ Requires episode to finish
❌ High variance
❌ Slow learning
❌ Not suitable for continuing tasks

---

# 17. When to Use Monte Carlo

* Games
* Simulations
* Episodic tasks
* When environment model is unknown

---

# 18. Monte Carlo vs TD vs DP (Big Picture)

| Feature          | DP  | MC   | TD     |
| ---------------- | --- | ---- | ------ |
| Model needed     | Yes | No   | No     |
| Bootstrapping    | Yes | No   | Yes    |
| Episode required | No  | Yes  | No     |
| Variance         | Low | High | Medium |

---

# 19. Simple Example (Intuition)

If you play a game 100 times:

* Track total score
* Average score after each position

That’s Monte Carlo.

---

# 20. One-Sentence Summary

> **Monte Carlo methods in Reinforcement Learning estimate value functions by averaging actual returns obtained from complete episodes, without using a model of the environment.**

---

# 21. Big Picture Flow

```
MDP
 ↓
Policy
 ↓
Episodes
 ↓
Returns
 ↓
Averaging (Monte Carlo)
 ↓
Value Estimates
```

---


