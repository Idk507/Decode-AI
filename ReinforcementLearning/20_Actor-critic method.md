
# Actor–Critic Methods in Reinforcement Learning  
*A Very Detailed, Intuitive, and Mathematical Explanation*

Actor–Critic methods are one of the most important families of algorithms in Reinforcement Learning (RL). They combine the strengths of **value-based learning** and **policy-based learning** to create powerful and stable learning algorithms used in many modern AI systems such as robotics, game playing, and autonomous control.

To understand Actor–Critic methods properly, we will build the explanation step by step:

1. Intuition of Reinforcement Learning  
2. Why Actor–Critic is needed  
3. Core components: Actor and Critic  
4. Mathematical formulation  
5. Learning updates  
6. Full algorithm workflow  
7. Neural network implementation  
8. Advantages and limitations  

Every concept will be explained simply while also showing the math behind it.

---

# 1. Quick Recap: Reinforcement Learning

In Reinforcement Learning, an **agent** interacts with an **environment**.

At each step:

1. The agent observes a **state** \( s_t \)
2. The agent selects an **action** \( a_t \)
3. The environment gives a **reward** \( r_{t+1} \)
4. The environment moves to a **new state** \( s_{t+1} \)

The goal of the agent is to maximize the **total future reward**.

---

## Return (Total Future Reward)

The total reward the agent wants to maximize is called the **return**:

\[
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ...
\]

Where:

- \( r \) = reward  
- \( \gamma \) = discount factor  

The discount factor ensures future rewards count slightly less than immediate ones.

---

# 2. Two Classical RL Approaches

Before Actor–Critic, there were two major approaches:

## Value-Based Methods

Example: Q-learning.

The agent learns:

\[
Q(s,a)
\]

which measures how good it is to take action \(a\) in state \(s\).

Then it chooses the best action.

Problem:  
Hard to use when actions are continuous.

---

## Policy-Based Methods

Example: Policy Gradient.

The agent directly learns the **policy**:

\[
\pi(a|s)
\]

which is the probability of taking action \(a\) in state \(s\).

Problem:  
High variance learning (very unstable updates).

---

# 3. Actor–Critic Idea

Actor–Critic combines both approaches.

Two components exist:

### Actor

The **Actor** decides actions.

It represents the **policy**:

\[
\pi_\theta(a|s)
\]

θ = policy parameters.

---

### Critic

The **Critic** evaluates actions.

It estimates the **value function**:

\[
V(s)
\]

or sometimes

\[
Q(s,a)
\]

The critic tells the actor whether its actions were good or bad.

---

# 4. Simple Intuition

Imagine a student and a teacher.

Student = Actor  
Teacher = Critic  

The student answers questions.

The teacher says:

- "Good answer"
- "Bad answer"

The student improves based on feedback.

This is exactly how Actor–Critic works.

---

# 5. Value Function Used by the Critic

The critic usually learns the **state value function**:

\[
V(s)
\]

Which means:

Expected return starting from state \(s\).

Mathematically:

\[
V(s) = E[G_t | S_t = s]
\]

---

# 6. Temporal Difference Error

The critic learns using **Temporal Difference (TD) learning**.

TD error:

\[
\delta = r + \gamma V(s') - V(s)
\]

This error measures:

> Difference between predicted value and observed outcome.

If δ is positive:

The outcome was better than expected.

If δ is negative:

The outcome was worse than expected.

---

# 7. Critic Update Rule

The critic updates its value estimate:

\[
V(s) \leftarrow V(s) + \alpha \delta
\]

Where:

- \( \alpha \) = learning rate

The critic slowly improves its predictions.

---

# 8. Actor Update Rule

The actor adjusts its policy using the TD error as feedback.

Policy gradient update:

\[
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \delta
\]

Meaning:

- If δ positive → increase probability of the action
- If δ negative → decrease probability

The actor learns which actions are better.

---

# 9. Advantage Function

Sometimes we use the **advantage function**:

\[
A(s,a) = Q(s,a) - V(s)
\]

Advantage measures:

> How much better an action is compared to the average action in that state.

In Actor–Critic, the TD error approximates the advantage.

---

# 10. Full Actor–Critic Algorithm

Step-by-step process:

1. Initialize actor parameters \(θ\)
2. Initialize critic parameters \(w\)

Repeat:

Observe state \(s\)

Actor selects action:

\[
a \sim \pi_\theta(a|s)
\]

Environment returns:

reward \(r\) and next state \(s'\)

Critic computes TD error:

\[
\delta = r + \gamma V_w(s') - V_w(s)
\]

Update critic:

\[
w \leftarrow w + \alpha_c \delta \nabla_w V_w(s)
\]

Update actor:

\[
\theta \leftarrow \theta + \alpha_a \delta \nabla_\theta \log \pi_\theta(a|s)
\]

Move to next state.

Repeat.

---

# 11. Neural Network Implementation

In modern RL both actor and critic are neural networks.

## Actor Network

Input: state

Output: action probabilities

Example architecture:

```

State → Dense Layer → ReLU → Dense Layer → Softmax → Action Probabilities

```

---

## Critic Network

Input: state

Output: value estimate

```

State → Dense Layer → ReLU → Dense Layer → Value Estimate

```

---

# 12. Why Actor–Critic Works Better

Actor–Critic solves two major problems.

### Reduces Variance

Pure policy gradients have unstable updates.

The critic stabilizes learning.

---

### Works with Continuous Actions

Actor outputs continuous values directly.

This is useful for robotics and control.

---

# 13. Major Actor–Critic Algorithms

Many modern algorithms are based on Actor–Critic:

- A2C (Advantage Actor Critic)
- A3C (Asynchronous Advantage Actor Critic)
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor Critic)

These are widely used in modern deep reinforcement learning.

---

# 14. Real World Example

Robot arm picking objects.

Actor decides:

- Move left
- Move right
- Grab object

Critic evaluates:

- Did the action improve the chance of success?

Actor updates behavior based on critic feedback.

Over time the robot learns to pick objects efficiently.

---

# 15. End-to-End Learning Process

The complete Actor–Critic learning cycle:

1. Observe state
2. Actor chooses action
3. Environment returns reward
4. Critic evaluates the outcome
5. Compute TD error
6. Update critic value estimate
7. Update actor policy
8. Repeat millions of times

Eventually the policy becomes optimal.

---

# 16. Advantages

Actor–Critic methods:

- Reduce variance of policy gradients
- Work in continuous action spaces
- Learn faster than pure policy methods
- Scale well with neural networks

---

# 17. Limitations

Actor–Critic methods can suffer from:

- instability if critic is inaccurate
- sensitivity to hyperparameters
- exploration challenges

Advanced algorithms address these issues.

---

# 18. One Sentence Summary

Actor–Critic methods train two models simultaneously: an **actor that decides actions** and a **critic that evaluates those actions**, allowing the agent to improve its behavior efficiently by combining policy learning with value estimation.
