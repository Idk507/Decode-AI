# REINFORCE (Policy Gradient) Algorithm in Reinforcement Learning  
## Complete End-to-End Explanation (Very Detailed + Simple Language)

The **REINFORCE algorithm** is one of the most important Reinforcement Learning algorithms because it teaches an agent **how to directly learn behavior (policy)** using rewards. It is the simplest and purest example of a **Policy Gradient method**.

This explanation will start from very basic intuition and slowly move toward full mathematics and implementation understanding. Every term will be explained carefully.

---

# 1. Big Picture First (Simple Intuition)

Imagine teaching a child to play a video game.

The child:
- sees a situation (state)
- chooses an action
- gets points (reward)

After finishing the game, the child thinks:

> “The actions I took before winning were good. I should do them more often.”

REINFORCE works exactly like this.

Instead of memorizing states or building value tables, the agent learns:

> “Increase probability of actions that gave good future rewards.”

---

# 2. Reinforcement Learning Setup

Everything happens inside an environment.

At each step:

Agent sees:

State → \( s_t \)

Agent chooses:

Action → \( a_t \)

Environment gives:

Reward → \( r_{t+1} \)

and moves to:

Next state → \( s_{t+1} \)

This continues until episode ends.

Example:
- Chess game
- Robot walking
- Video game episode

---

# 3. What is a Policy?

A **policy** decides what action to take.

We write:

\[
\pi(a|s)
\]

Meaning:

> Probability of taking action \( a \) in state \( s \).

Example:

Robot sees obstacle.

Policy:

- Go left → 0.7
- Go right → 0.3

Policy is stochastic (random).

---

## Parameterized Policy

We control policy using parameters:

\[
\pi_\theta(a|s)
\]

θ = neural network weights.

Learning means:

> Change θ so better actions happen more often.

---

# 4. Goal of Learning

Agent wants maximum future reward.

Return:

\[
G_t =
r_{t+1}
+
\gamma r_{t+2}
+
\gamma^2 r_{t+3}
+ ...
\]

Where:

Reward = feedback.

Discount factor γ:

0 ≤ γ ≤ 1.

Future rewards become slightly smaller.

Example:

γ = 0.9

Reward after 1 step → 90%.

After 2 → 81%.

---

## Objective Function

We maximize:

\[
J(\theta)
\]

Meaning:

Expected return when following policy.

Learning problem:

Maximize:

\[
J(\theta).
\]

---

# 5. Why Policy Gradient?

Instead of asking:

“How good is state?”

or

“What is best action value?”

We directly learn:

> “How should action probabilities change?”

We climb toward better reward.

---

# 6. Gradient Ascent

Update:

\[
\theta
\leftarrow
\theta + \alpha \nabla_\theta J(\theta)
\]

Where:

α = learning rate.

Gradient = direction increasing reward.

---

# 7. Policy Gradient Theorem

Key result:

\[
\nabla_\theta J(\theta)
=
E[
\nabla_\theta \log \pi_\theta(a|s)
Q^\pi(s,a)
]
\]

Meaning:

Increase probability of good actions.

---

# 8. REINFORCE Main Idea

We usually do not know:

\[
Q^\pi(s,a)
\]

So REINFORCE replaces it with:

Actual return observed.

\[
G_t
\]

Monte Carlo estimate.

---

# 9. REINFORCE Update Rule

For each timestep:

\[
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta
\log \pi_\theta(a_t|s_t)
G_t
\]

Simple meaning:

If big reward followed action:

increase probability.

Small reward:

reduce probability.

---

# 10. Understanding Log Gradient

Why:

\[
\log \pi_\theta(a|s)
\]

?

Mathematical trick called:

Log derivative trick:

\[
\nabla_\theta \pi
=
\pi \nabla_\theta \log \pi
\]

Makes gradient computation possible even when sampling actions randomly.

---

# 11. Step by Step Example (Kid Friendly)

Robot action probabilities:

Left = 0.5

Right = 0.5.

Robot chooses Left.

Gets big reward.

Gradient says:

Increase probability Left.

Next episode:

Left = 0.6.

Later:

Left = 0.9.

Learning happens naturally.

---

# 12. Complete Algorithm Steps

Initialize θ.

Repeat:

Generate episode.

For each timestep:

Store:
- state
- action
- reward.

Compute returns.

Update θ.

Repeat thousands of times.

---

# 13. Computing Return

Return from timestep t:

\[
G_t =
r_{t+1}
+
\gamma r_{t+2}
+
\gamma^2 r_{t+3}
...
\]

Example:

Rewards:

[1,0,2].

γ=0.9.

Return at start:

1 + 0.9(0) + 0.81(2)=2.62.

---

# 14. Loss Function View

We minimize:

\[
L(\theta)
=
- E[
\log \pi_\theta(a_t|s_t) G_t
]
\]

Negative sign because optimizers minimize loss.

---

# 15. Neural Network Policy

Policy network:

Input:

state.

Output:

action probabilities.

Example:

Network outputs:

[2.0,1.0].

Softmax:

Action1=0.73.

Action2=0.27.

Sample action.

---

# 16. Gradient Computation

Backpropagation computes:

\[
\nabla_\theta \log \pi_\theta(a|s)
\]

Automatically.

Multiply with return.

Update weights.

---

# 17. Big Problem — High Variance

Returns vary a lot.

One lucky episode may give huge reward.

Learning unstable.

---

# 18. Baseline Trick

Subtract baseline:

\[
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta
\log \pi_\theta(a|s)
(G_t - b(s))
\]

Usually:

\[
b(s)=V(s)
\]

Value estimate.

---

# 19. Advantage Function

Advantage:

\[
A(s,a)=Q(s,a)-V(s)
\]

Meaning:

How much better than average.

More stable learning.

---

# 20. REINFORCE With Baseline

Modern version:

\[
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta \log \pi(a|s)
(G_t-V(s))
\]

Used widely.

---

# 21. Why REINFORCE Works

Law of large numbers.

Good actions consistently give high returns.

Gradient slowly increases probability.

Eventually policy improves.

---

# 22. Python Example (Simple)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):

    def __init__(self,state_dim,action_dim):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,action_dim)
        )

    def forward(self,x):
        return torch.softmax(self.net(x),dim=-1)

```
# Training:
```
def reinforce_update(policy,optimizer,log_probs,returns):

    loss=0

    for logp,G in zip(log_probs,returns):

        loss+= -logp*G

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

```
---

# 23. Advantages

Simple.

Works with continuous actions.

Direct policy optimization.

Foundation of modern RL.

---

# 24. Limitations

High variance.

Slow learning.

Needs many samples.

---

# 25. Modern Algorithms Built on REINFORCE

Actor Critic.

A2C.

PPO.

TRPO.

SAC.

All extend REINFORCE idea.

---

# 26. Full End-to-End Learning Pipeline

Observe state.

Policy outputs probabilities.

Sample action.

Environment responds.

Store rewards.

Compute returns.

Compute gradient.

Backpropagation.

Update parameters.

Repeat.

---

# 27. One Sentence Summary

REINFORCE teaches an agent by increasing the probability of actions that led to high future rewards using gradients of log action probabilities multiplied by observed returns.

---

# 28. Deep Insight

Value-based RL learns:

“How good is world?”

Policy gradient learns:

“How should I behave?”

REINFORCE is the first algorithm showing how behavior itself can be learned using mathematics.

```
```
