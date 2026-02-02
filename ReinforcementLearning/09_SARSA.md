
---

## 1. The big picture: What problem SARSA is solving

In Reinforcement Learning, we do not just want to know:

> “How good is this state?”

We want to know:

> “How good is it to take *this specific action* in *this specific state*?”

This is more powerful, because if you know this, you can **directly choose the best action**.

So instead of a **value function**:

V(s)

we use an **action-value function**:

Q(s, a)


This means:

> “If I am in state ( s ), and I take action ( a ), and then continue behaving according to my policy, how much total future reward will I get?”

SARSA is a method to **learn this Q-function step by step from experience**, without knowing how the environment works internally.

---

## 2. Why the name “SARSA”

The name SARSA comes from the **five things it uses in every update**:


(S, A, R, S', A')


That means:

* ( S ): current state
* ( A ): current action
* ( R ): reward you receive
* ( S' ): next state
* ( A' ): next action you choose

So SARSA is a learning rule that says:

> “I update what I believe about the value of my current state and action using what reward I got, what state I moved to, and what action I will take next.”

This is very important: SARSA learns based on **what it actually does**, not what it *could* do.

---

## 3. The idea of a policy (how the agent behaves)

A **policy**, written as ( π(a | s) ), means:

> “The probability that I choose action ( a ) when I am in state ( s ).”

SARSA is an **on-policy** method.
That means:

> It learns the value of the *same policy it is using to act.*

This is different from methods like Q-learning, which learn the value of a greedy (best possible) policy even while behaving randomly.

---

## 4. Why randomness is needed (exploration vs exploitation)

If the agent always chooses what it thinks is best, it may **never discover better options**.

So we often use **ε-greedy behavior**:

* With probability ( 𝜀 ): choose a random action (explore)
* With probability ( 1 -  𝜀 ): choose the best-known action (exploit)

This makes SARSA realistic, because its learning reflects this *imperfect, exploratory behavior*.

---

## 5. The true meaning of Q(s, a)

Formally, the action-value function is defined as:

<img width="652" height="61" alt="image" src="https://github.com/user-attachments/assets/810ab5c1-d37f-4803-85e7-995041581bd7" />


In words:

> “If I start in state ( s ), take action ( a ), and then follow policy ( \pi ), how much discounted reward will I get on average?”

Here:

* E means “expected value” (average over randomness)
* 𝛾 is the discount factor (0 to 1)

---

## 6. The Bellman equation for Q-values

The recursive truth about Q-values is:

<img width="355" height="66" alt="image" src="https://github.com/user-attachments/assets/28e3f584-d293-411e-9599-4ea5eeeec657" />


This says:

> The value of doing action ( a ) in state ( s ) is the reward you get now, plus the discounted value of what you do next.

But in real life:

* You do not know the expectation
* You only see **one sample at a time**

So SARSA learns using **sample-based updates**.

---

## 7. The SARSA learning target

When the agent experiences:

(s, a, r, s', a')


It forms a **target** for what ( Q(s, a) ) should be:

<img width="273" height="64" alt="image" src="https://github.com/user-attachments/assets/40e1dea2-e4d3-4425-bc7d-9c3abc9d8cc3" />


This target is based on:

* The reward you just got
* The value of the next state-action pair you actually chose

---

## 8. Temporal Difference error in SARSA

The **TD error** is the difference between:

* What you observed
* What you believed

<img width="327" height="71" alt="image" src="https://github.com/user-attachments/assets/0fb7e50f-fbf4-448a-a66d-3898c162fa4c" />


This single number tells you:

> “How wrong was my estimate of this action in this state?”

---

## 9. The SARSA update rule (core equation)


<img width="807" height="227" alt="image" src="https://github.com/user-attachments/assets/fb719689-6544-4b5d-bc96-c9487e3b1c7f" />

This is the **full SARSA learning equation**.

---

## 10. Why this makes mathematical sense

Let’s rewrite it in another form:

<img width="383" height="62" alt="image" src="https://github.com/user-attachments/assets/c65c3579-90bc-4c09-b69b-b17903f9d119" />


So:

> The new value is mostly the old belief, plus a small piece of the new evidence.

This is a **stochastic approximation** of the Bellman equation. Over many updates, this converges to the true Q-function for the policy being followed.

---

## 11. Step-by-step numerical example

Imagine:

<img width="437" height="230" alt="image" src="https://github.com/user-attachments/assets/37a2c9c9-a9b4-4186-8dd4-2cdf440b9589" />


<img width="672" height="309" alt="image" src="https://github.com/user-attachments/assets/563f00b5-cc0a-4de8-90e7-b7e008074e4d" />

So the agent slightly increases its belief that this action is good.

---

## 12. Why SARSA is called “on-policy”

Notice this part:
<img width="194" height="82" alt="image" src="https://github.com/user-attachments/assets/5c67b3b7-2001-4c70-9adb-aa30fb696c1b" />


This uses the **actual next action chosen by the policy**, including exploration.

So if the policy sometimes chooses risky or random actions, SARSA learns that those actions can lead to bad outcomes. This often makes SARSA **more cautious and safer** in real environments.

---

## 13. Comparison with Q-learning (deep insight)
<img width="558" height="217" alt="image" src="https://github.com/user-attachments/assets/47038fce-9cad-4ffa-ba77-60a5b3876c06" />


This means:

* Q-learning assumes perfect behavior in the future
* SARSA assumes realistic behavior in the future

This difference is critical in environments where mistakes are costly.

---

## 14. Full algorithm (conceptual flow)

At the system level, SARSA works like this:

The agent initializes a table or neural network for Q-values. It enters the environment and selects an action using its policy. It observes the reward and next state. It selects the next action using the same policy. Then it updates the Q-value for the original state-action pair using the SARSA equation. This loop continues until learning stabilizes.

---

## 15. Mathematical convergence conditions

SARSA converges to the correct Q-function for the policy if:

* All state-action pairs are visited infinitely often
* The learning rate decreases properly over time
* Rewards are bounded
* The policy becomes stable

These conditions come from stochastic approximation theory.

---

## 16. Function approximation version (deep learning view)

Instead of a table, you can use a neural network:
<img width="643" height="286" alt="image" src="https://github.com/user-attachments/assets/846ea9b8-c83d-45c3-9b7c-8ad937f5059d" />

This is the foundation of **Deep SARSA**.

---

## 17. Production-grade Python implementation (tabular)

```python
import numpy as np
import random

class SARSAAgent:
    """
    Tabular SARSA Agent for discrete state and action spaces.
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, a_next):
        """
        SARSA update rule.
        """
        td_target = r + self.gamma * self.Q[s_next, a_next]
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
```

---

## 18. When to use SARSA in real systems

SARSA is ideal when:

* Safety matters
* Exploration can cause harm
* You want learning to reflect actual behavior

Examples include:

* Robotics
* Traffic control
* Finance systems
* Healthcare decision support

---

## 19. Deep intuition summary

In one sentence:

> SARSA learns by asking: “I was in this situation, I took this action, I got this reward, I ended up here, and I plan to do this next — so how should I adjust my belief about my original choice?”

This makes SARSA one of the most **honest and realistic learning algorithms in Reinforcement Learning**, because it learns from what it *actually does*, not what it *wishes it would do*.

---

