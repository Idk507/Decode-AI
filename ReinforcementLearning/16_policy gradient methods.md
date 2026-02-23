Policy Gradient methods are a major family of Reinforcement Learning algorithms that take a different approach from value-based methods like Q-learning. Instead of first learning how good states or actions are and then choosing the best action, policy gradient methods **directly learn how to act**. I will build the explanation step by step in very simple language first, then move into the mathematics carefully so that every term is clear.

---

## 1. Intuition: learning behavior directly

Imagine teaching a robot to play a game. Instead of asking:

“How good is this state?”
or
“How good is this action?”

we directly teach:

> “In this situation, what action should I take?”

The robot keeps a **policy**, which is simply a rule for choosing actions. Policy gradient methods adjust this rule little by little so that actions leading to higher rewards become more likely.

---

## 2. What is a policy?

A **policy** is written as:

<img width="140" height="45" alt="image" src="https://github.com/user-attachments/assets/c0384087-3b92-4815-96e1-b92a30830718" />


This means:

> The probability of taking action ( a ) when the agent is in state ( s ).

Unlike deterministic rules (“always move left”), policy gradient methods often use **stochastic policies**, meaning the agent chooses actions with certain probabilities. This helps exploration.

Example:

* In state ( s ):

  * move left with probability 0.7
  * move right with probability 0.3

These probabilities are controlled by parameters ( \theta ), so we write:

<img width="100" height="51" alt="image" src="https://github.com/user-attachments/assets/9c911f30-350f-4ee4-bcfd-1347dc6c1141" />


Learning means **changing θ so that good actions become more probable**.

---

## 3. Goal of learning (objective function)

The agent wants to maximize the expected total future reward:

<img width="840" height="168" alt="image" src="https://github.com/user-attachments/assets/94e09bf5-28c3-4093-b9a5-1d6bef626475" />


This is the discounted total reward (return).

So the learning problem becomes:

> Find parameters θ that maximize θ.

---

## 4. The key idea: gradient ascent

Instead of guessing randomly, we move the parameters in the direction that **increases expected reward**:

<img width="212" height="48" alt="image" src="https://github.com/user-attachments/assets/3e5b9cbf-bbf6-40eb-9069-7c55feca03a0" />


This is called **policy gradient** because we compute the gradient of expected reward with respect to the policy parameters.

---

## 5. The Policy Gradient Theorem (core formula)

The fundamental result in policy gradient methods is:

<img width="431" height="51" alt="image" src="https://github.com/user-attachments/assets/82e8822e-2215-400b-889c-6a55fdf494f2" />


This looks complicated, but each term has a simple meaning:

<img width="821" height="74" alt="image" src="https://github.com/user-attachments/assets/7b138142-e89b-45e2-a256-19301935563a" />

So the update rule becomes:

<img width="386" height="51" alt="image" src="https://github.com/user-attachments/assets/45439590-7c8b-4f67-9407-086507569cac" />

In simple words:

> If an action produced high reward, increase the probability of choosing it.
> If it produced low reward, decrease its probability.

---

## 6. REINFORCE algorithm (basic policy gradient)

The simplest policy gradient method is **REINFORCE**.

After an episode ends, for each step ( t ):

<img width="332" height="41" alt="image" src="https://github.com/user-attachments/assets/51d351e2-b7c0-485e-a672-8b54562a961d" />


where:

* ( G_t ) = total reward from time ( t ) onward

So actions followed by large future rewards get strengthened.

---

## 7. Why the log term appears (intuition)

The term:

<img width="200" height="69" alt="image" src="https://github.com/user-attachments/assets/6df6625d-0f62-4070-b95f-6dbf95015fc6" />


comes from probability theory and allows us to compute gradients even when actions are sampled randomly. It tells us:

> “How should I change the parameters to make this chosen action more or less likely?”

Multiplying it by reward connects **probability change** with **performance**.

---

## 8. Reducing variance using a baseline

Direct policy gradient updates can be noisy. So we subtract a **baseline** ( b(s) ), usually the state value:

<img width="448" height="53" alt="image" src="https://github.com/user-attachments/assets/888f82c9-c232-4d2a-a5e9-bf3bf43e84be" />


This does not change the expected gradient but makes learning more stable.

When ( b(s)=V(s) ), we get the **advantage function**:

A(s,a) = Q(s,a) - V(s)


which measures how much better an action is compared to average.

---

## 9. Actor–Critic methods

Policy gradient methods often use two networks:

**Actor**

* represents the policy ( \pi_\theta )
* decides actions

**Critic**

* estimates value ( V(s) )
* helps compute advantage

<img width="699" height="215" alt="image" src="https://github.com/user-attachments/assets/41b97b35-3147-451e-894b-ff1b2f9c2b35" />


acts as an estimate of advantage.

This combination forms **Actor-Critic algorithms**, widely used in modern RL.

---

## 10. Neural network policy representation

Policies are usually neural networks:

<img width="306" height="41" alt="image" src="https://github.com/user-attachments/assets/03642d82-161e-4330-994b-d51ce83c5196" />


The network outputs probabilities for each action, and gradient ascent adjusts weights so that high-reward actions become more likely.

---

## 11. Advantages of policy gradient methods

They:

* Work naturally with continuous action spaces
* Learn stochastic policies directly
* Optimize behavior rather than intermediate value estimates
* Are the foundation of advanced algorithms (PPO, TRPO, A3C)

---

## 12. Simple conceptual example

Suppose a robot chooses:

* left (probability 0.5)
* right (probability 0.5)

If choosing left leads to high reward, the gradient update increases the probability of left slightly. After many episodes, the robot automatically learns to favor actions producing better outcomes.

---

## 13. End-to-end learning pipeline

The agent observes the state, samples an action from the policy, executes it, receives reward and next state, computes return or advantage, calculates the gradient of log probability of the taken action, updates policy parameters using gradient ascent, and repeats the process continuously. Over time the policy becomes better at choosing reward-maximizing actions.

---

## 14. Deep intuitive summary

Policy gradient methods teach an agent by **rewarding the decisions that worked well and discouraging the ones that did not**, directly adjusting the probabilities of choosing actions. Instead of learning a map of the world first and then acting, the agent learns the behavior itself through gradual probability adjustments guided by reward signals.

---

## 15. Connection to modern deep reinforcement learning

Most modern algorithms—PPO, A2C, SAC, TRPO—are built on policy gradient foundations. Understanding policy gradients means understanding the core mechanism behind many large-scale decision-making AI systems used in robotics, games, and autonomous control.

---

