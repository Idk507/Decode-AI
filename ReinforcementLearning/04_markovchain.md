<img width="138" height="39" alt="image" src="https://github.com/user-attachments/assets/b8669d8b-6d7e-4010-9504-096cb61f3d20" />

---

# 1. What Is a Markov Chain? (In Very Simple Terms)

A **Markov Chain** is a way to describe a system that:

1. Moves **step by step**
2. Has a **fixed set of states**
3. Chooses the **next state based only on the current state**
4. Uses **probabilities** to decide the next move

👉 The **past does NOT matter**, only the **present**.

---

### Simple Example (Weather)

States:

* ☀️ Sunny
* ☁️ Cloudy
* 🌧️ Rainy

Rules:

* If today is **Sunny**, tomorrow might be:

  * Sunny: 70%
  * Cloudy: 20%
  * Rainy: 10%

This is a **Markov Chain**.

---

# 2. The Markov Property (Most Important Rule)

### Markov Property (Plain English)

> *The future depends only on the present, not on the past.*

Mathematically:
<img width="644" height="52" alt="image" src="https://github.com/user-attachments/assets/b026210a-9f23-4fba-baa7-f0098fb3d5ff" />

Meaning:

* Knowing yesterday and last week **adds no extra information**
* Only **today’s state matters**

---

# 3. What Is a “State”?

A **state** is a snapshot of the system at a moment in time.

Examples:

* Weather today
* Location of a robot
* Page you are currently viewing
* Mood of a person

### State Space

<img width="246" height="49" alt="image" src="https://github.com/user-attachments/assets/3e007af9-746f-49a5-a909-38ca6d1c51b1" />


It is the **set of all possible states**.

---

# 4. Transitions (Moving Between States)

A **transition** means moving from one state to another.

### Transition Probability

<img width="112" height="54" alt="image" src="https://github.com/user-attachments/assets/0230da9c-678b-4f59-acf9-d677c0126f2d" />


Meaning:

> Probability of going to state ( s' ) **given** you are currently in state ( s )

Example:
<img width="341" height="86" alt="image" src="https://github.com/user-attachments/assets/739f3ecb-e9b2-4e9d-8dce-4d2f9d03b333" />


---

# 5. Transition Matrix (Core Representation)

All transition probabilities are stored in a **matrix**.

### Transition Matrix ( P )

<img width="493" height="136" alt="image" src="https://github.com/user-attachments/assets/feb4f849-4bb6-4a1d-a5df-e42a688275f0" />


### Properties

* Each **row sums to 1**
* Values are between **0 and 1**

Example:
<img width="264" height="143" alt="image" src="https://github.com/user-attachments/assets/35390198-d8e5-4d88-94e0-c977654bc253" />


---

# 6. Discrete-Time vs Continuous-Time Markov Chains

### Discrete-Time Markov Chain (DTMC)

* System moves in **fixed steps**
* Example: daily weather

### Continuous-Time Markov Chain (CTMC)

* State can change **at any time**
* Uses **rates**, not probabilities
* Example: call arrivals in a call center

(For RL, DTMC is most common.)

---

# 7. Initial State Distribution

We need to know where we start.

### Initial Distribution

<img width="410" height="58" alt="image" src="https://github.com/user-attachments/assets/be8c9168-cc04-48d7-bf89-39e5fbbac0dd" />

Example:
<img width="222" height="70" alt="image" src="https://github.com/user-attachments/assets/9ff5ef7c-7e8a-4701-98e2-1ed6fef85051" />

---

# 8. State Distribution Over Time

After one step:
<img width="138" height="39" alt="image" src="https://github.com/user-attachments/assets/fae38d2a-7b92-4032-bbeb-cc4061cc0e89" />


After ( t ) steps:
<img width="129" height="56" alt="image" src="https://github.com/user-attachments/assets/06145ba2-6b95-4717-a36f-a8664fef614d" />

This tells us:

> Probability of being in each state after ( t ) steps

---

# 9. Long-Term Behavior (Stationary Distribution)

### Stationary Distribution

<img width="599" height="94" alt="image" src="https://github.com/user-attachments/assets/6809a5b7-ed9f-45e7-ac15-8d13b500c979" />

Meaning:

* Once reached, **it does not change**
* Long-run behavior of the chain

---

### Example Meaning

> In the long run, 60% of days are sunny, 25% cloudy, 15% rainy

---

# 10. Important Types of States

### 1️⃣ Absorbing State

Once entered, cannot leave.

<img width="201" height="57" alt="image" src="https://github.com/user-attachments/assets/89374f29-b26b-4724-abf9-44aafdd4f237" />


Example:

* Game over
* Terminal failure

---

### 2️⃣ Transient State

* Might leave forever
* Not guaranteed to return

---

### 3️⃣ Recurrent State

* Will return with probability 1

---

# 11. Communicating States

States **communicate** if:
<img width="267" height="79" alt="image" src="https://github.com/user-attachments/assets/7c7bc90e-5f67-44ca-89b5-76cdf74359fb" />


A chain is:

* **Irreducible** if all states communicate

---

# 12. Periodicity

A state is **periodic** if it can be revisited only at fixed intervals.

Example:

* Period 2 → only at even times

If **aperiodic**, returns can happen at any time.

---

# 13. Ergodic Markov Chain (Very Important)

A chain is **ergodic** if it is:

* Irreducible
* Aperiodic
* Positive recurrent

✅ Guarantees a **unique stationary distribution**

---

# 14. Markov Chain vs Markov Process vs MDP

| Concept                 | Action? | Reward? |
| ----------------------- | ------- | ------- |
| Markov Chain            | ❌ No    | ❌ No    |
| Markov Reward Process   | ❌ No    | ✅ Yes   |
| Markov Decision Process | ✅ Yes   | ✅ Yes   |

---

# 15. How Markov Chains Relate to Reinforcement Learning

Reinforcement Learning builds on Markov Chains:

<img width="395" height="74" alt="image" src="https://github.com/user-attachments/assets/532d970c-fd62-40ee-93f2-edd5ddf264c4" />


RL adds:

* Actions
* Rewards
* Policies

---

# 16. Simple Real-World Applications

* Weather prediction
* PageRank (Google)
* Board games
* Text generation
* Biology (DNA sequences)
* Finance
* Queue systems

---

# 17. Simple Intuition Summary

> A **Markov Chain** is a system that moves between states using probabilities, where the future depends only on the current state and not on how you got there.

---

# 18. One-Line Mathematical Definition

<img width="424" height="48" alt="image" src="https://github.com/user-attachments/assets/ee9bf789-7079-4800-a701-a84bee6cd865" />


---

# 19. Visual Way to Think

* **Nodes** = states
* **Arrows** = transition probabilities
* **Matrix** = numeric representation
* **Long-run distribution** = steady behavior

---

# 20. Final Big Picture

Markov Chains are:

* The **foundation of probabilistic modeling**
* The **core assumption behind RL**
* The **bridge between probability and decision-making**

---

