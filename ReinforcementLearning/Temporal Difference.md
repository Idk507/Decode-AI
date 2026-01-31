Temporal Difference learning, usually called **TD learning**, is one of the core ideas in Reinforcement Learning. I will explain it slowly, from the very beginning, using simple intuition first, then gradually introduce the math, and finally connect everything end-to-end so that nothing feels magical or skipped. Think of this as learning how a child figures out whether a choice was good or bad while playing a game, but explained with enough depth that it still holds up mathematically.

---

### 1. The basic Reinforcement Learning setting (the world TD lives in)

Imagine a **player** (called an *agent*) playing a game. The game has **situations** (called *states*), and in each situation the player can choose an **action**. After choosing an action, three things happen:

1. The game moves to a new situation (next state).
2. The player gets a number called a **reward** (like +1 for good, −1 for bad).
3. The game continues.

The goal of the player is **not** just to get a good reward right now, but to get as much reward as possible **over time**.

To do this, the player needs to answer a very important question:

> “How good is it to be in this situation?”

That “how good” number is called the **value** of a state.

---

### 2. What is a value function?

The **value function**, written as ( V(s) ), means:

> “If I start in state ( s ) and keep playing sensibly, how much reward will I get in the future?”

Future rewards matter less than immediate rewards, because rewards far away in time are uncertain. To handle this, we use a **discount factor**.

---

### 3. Discount factor (γ) – why future rewards shrink

The discount factor is written as 𝛾 (gamma), and it is a number between 0 and 1.

* ( 𝛾 = 0 ): only care about the immediate reward.
* ( 𝛾 = 0.9 ): care a lot about the future, but not infinitely.
* ( 𝛾 = 1 ): care equally about now and the far future (rare in practice).

So if you expect a reward of 10 in the future, its value *now* is:

<img width="345" height="48" alt="image" src="https://github.com/user-attachments/assets/ee42b45c-8e3a-4f70-bfc1-25a48e3f512f" />


This matches how humans think: getting candy **now** feels better than getting the same candy next year.

---

### 4. The true definition of value (the Bellman idea)

In theory, the value of a state is:

[
<img width="486" height="76" alt="image" src="https://github.com/user-attachments/assets/039071c6-1f50-4c9f-a42a-e45bc53f3dcc" />
]

This just means:

* Add the reward now
* Add the next reward, but smaller
* Add the next one, even smaller
* Keep going

But this formula is **impossible to compute directly** in real problems because:

* You don’t know the future.
* The world may be random.
* Episodes may be very long.

So we need a way to **learn** values step by step from experience.

That is where **Temporal Difference learning** comes in.

---

### 5. The key idea of Temporal Difference learning (intuition first)

Temporal Difference learning is based on a very simple human habit:

> “I had an expectation. I observed what actually happened. I adjust my expectation.”

Example for a 12-year-old:

* You think a math test will be easy.
* You get the paper and realize it is harder than expected.
* Next time, you lower your expectation.

TD learning does **exactly this**, but with numbers.

---

### 6. What does “Temporal Difference” mean?

* **Temporal** means “across time” (now vs next moment).
* **Difference** means “error between what I expected and what I observed”.

So TD learning updates values using the **difference between**:

* what you *thought* the value was, and
* what the next step *suggests* the value should be.

This difference is called the **TD error**.

---

### 7. One-step prediction: the TD target

Suppose:

* You are in state ( s )
* You move to state ( s' )
* You receive reward ( r )

<img width="617" height="219" alt="image" src="https://github.com/user-attachments/assets/f0afc125-70bb-4b10-8c37-f930cbd53384" />


This is called the **TD target**.

Why this makes sense:

* You got reward ( r ) immediately.
* From the next state ( s' ), you expect ( V(s') ) more reward.
* But that future reward is discounted by ( γ ).

---

### 8. Temporal Difference error (the heart of TD learning)

The **TD error**, written as ( δ), is:

<img width="259" height="33" alt="image" src="https://github.com/user-attachments/assets/c0e9fa15-b913-4eeb-8878-f295a3110576" />

Read it in plain language:

> “What I observed minus what I expected.”

<img width="490" height="113" alt="image" src="https://github.com/user-attachments/assets/9ac28d81-912a-4ef2-a9b1-445e92551826" />

This single number tells you **how wrong you were**.

---

### 9. Updating the value function (learning step)

You do not fully replace your old belief. You **move it slightly** in the right direction.

<img width="233" height="43" alt="image" src="https://github.com/user-attachments/assets/2bcf25c7-c10f-464d-86f4-ff6774b42d61" />


Here, α (alpha) is the **learning rate**, a number between 0 and 1.

* Small α: learn slowly, but stably.
* Largeα: learn fast, but may be unstable.

<img width="715" height="106" alt="image" src="https://github.com/user-attachments/assets/e8717cd1-fa4c-4be1-a966-327afd661622" />

This is the **core TD learning equation**.

---

### 10. Why TD learning is powerful

TD learning has two huge advantages:

**First**, it does not wait until the end of the game.
Unlike Monte Carlo methods, TD learning updates values **after every step**.

**Second**, it learns from incomplete information.
It does not need to know how the game works. It only needs:

* current state
* reward
* next state

This makes TD learning ideal for real-world problems.

---

### 11. TD(0): the simplest Temporal Difference algorithm

TD(0) means:

* TD learning
* using **0 future steps beyond the next state**

Algorithm in words:

1. Start with random values for all states.
2. Observe a transition ( s --> s' ) with reward ( r ).
3. Compute TD error.
4. Update ( V(s) ).
5. Repeat forever.

Over time, the values become accurate.

---

### 12. Relationship to other methods (big picture)

TD learning sits **between** two extremes:

* **Dynamic Programming**: needs a full model of the environment.
* **Monte Carlo**: waits until the end of an episode.

TD learning:

* needs no model
* does not wait until the end
* learns online, step by step

This balance is why TD learning is foundational.

---

### 13. From TD learning to Q-learning (important connection)

So far we talked about **state values** ( V(s) ).

In practice, we often want **action values**:

<img width="593" height="54" alt="image" src="https://github.com/user-attachments/assets/f4b0bb47-5850-4563-81e4-0ba2382a946e" />

Q-learning uses the same TD idea:

<img width="370" height="71" alt="image" src="https://github.com/user-attachments/assets/d44ad701-93a9-4cfc-bc42-596ef2c80878" />


This is still Temporal Difference learning — just applied to actions.

---

### 14. Why TD learning converges (simple intuition)

Each update:

* pushes values toward reality
* corrects past mistakes using new evidence

If:

* learning rate decreases slowly
* all states are visited often
* rewards are bounded

Then TD learning converges to the correct value function.

This has strong mathematical guarantees.

---

### 15. Summary (end-to-end understanding)

Temporal Difference learning is about **learning from mistakes over time**.

* It predicts how good a state is.
* It compares prediction with reality one step later.
* It computes an error (TD error).
* It nudges the value in the right direction.
* It repeats this process thousands of times.

In one sentence:

> **TD learning learns by comparing “what I thought would happen” with “what actually happened next”.**

