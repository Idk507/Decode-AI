Policy Gradient Algorithms are a class of Reinforcement Learning methods where the agent **directly learns how to behave (the policy itself)** instead of first learning value tables or Q-values. Below is a complete end-to-end explanation starting from intuition and then moving into mathematical formulation, algorithm steps, and implementation understanding.

---

# Reinforcement Learning Policy Gradient Algorithm

## 1. Core Idea (Simple Intuition)

Imagine teaching someone to play basketball.

Instead of saying:

* “Standing here gives 5 points.”
* “Throwing like this gives 3 points.”

you directly teach:

> “Do more of the actions that helped you score.”

Policy Gradient works exactly like this.

The agent has a **policy** that decides actions.

It keeps adjusting this policy so that:

* good actions become more likely,
* bad actions become less likely.

---

# 2. What is a Policy?

A policy is a rule for choosing actions.

Mathematically:

[
\pi(a|s)
]

means:

Probability of taking action (a) when in state (s).

Example:

State = enemy nearby.

Policy:

* attack → 0.8 probability
* run → 0.2 probability

---

## Parameterized Policy

The policy depends on parameters:

[
\pi_\theta(a|s)
]

where:

* ( \theta ) = neural network weights.

Learning means:

> Change θ so rewards increase.

---

# 3. Objective of Policy Gradient

Agent wants maximum future reward.

Total reward (Return):

[
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ...
]

where:

* (r) = reward,
* ( \gamma ) = discount factor.

Goal:

[
J(\theta) = Expected\ Return
]

We want:

[
\max_\theta J(\theta)
]

---

# 4. Learning Using Gradient Ascent

We improve policy using gradient ascent:

[
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
]

Where:

* α = learning rate.
* gradient = direction increasing reward.

---

# 5. Policy Gradient Theorem (Core Formula)

The theorem gives gradient:

[
\nabla_\theta J(\theta)
=======================

E[ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]
]

Meaning:

Change policy based on:

* how sensitive probability is to parameters,
* multiplied by how good the action was.

Simple interpretation:

> Increase probability of actions that produced high reward.

---

# 6. Why Log Probability?

We sample actions randomly.

Direct differentiation is hard.

Using:

[
\nabla_\theta log(\pi)
]

makes gradients easy to compute using probability theory.

It tells:

> How to change weights to increase likelihood of chosen action.

---

# 7. Basic Policy Gradient Algorithm (REINFORCE)

Simplest algorithm.

Steps:

1. Run policy in environment.
2. Collect episode.
3. Compute returns.
4. Update policy.

Update rule:

[
\theta =
\theta +
\alpha
\nabla_\theta log\pi_\theta(a_t|s_t) G_t
]

Meaning:

Actions followed by large rewards get reinforced.

---

# 8. Algorithm Step-by-Step

Episode begins.

Agent observes state.

Policy network outputs action probabilities.

Action sampled.

Environment returns reward and next state.

Episode continues.

After episode ends:

Compute future rewards.

Update parameters.

Repeat many episodes.

---

# 9. Variance Problem

Monte Carlo returns fluctuate heavily.

Example:

One lucky episode gives huge reward.

Learning becomes unstable.

Solution:

Use baseline.

---

# 10. Baseline (Variance Reduction)

Subtract baseline:

[
(Q(s,a) - b(s))
]

Usually:

[
b(s)=V(s)
]

This gives:

Advantage:

[
A(s,a)=Q(s,a)-V(s)
]

Meaning:

> Was this action better than average?

---

# 11. Advantage Policy Gradient Update

Now update:

[
\theta =
\theta +
\alpha
\nabla_\theta log\pi(a|s)A(s,a)
]

Much more stable.

---

# 12. Actor-Critic Policy Gradient

Modern systems use:

Actor:

policy network.

Critic:

value estimator.

Critic computes TD error:

[
\delta=r+\gamma V(s')-V(s)
]

Actor update:

[
\theta=\theta+\alpha\nabla_\theta log\pi(a|s)\delta
]

Most modern RL algorithms use this.

---

# 13. Neural Network Policy

Policy network:

Input:

state.

Output:

action probabilities.

Softmax example:

Network output:

[3.0 , 1.0].

Softmax:

Action1 = 0.88.

Action2 = 0.12.

Sample action.

---

# 14. Advantages of Policy Gradient

Works with continuous actions.

Learns stochastic strategies.

Stable optimization.

Directly optimizes performance.

Used in robotics and LLM RLHF systems.

---

# 15. Simple Numerical Example

Robot:

Left = 0.5.

Right = 0.5.

Robot chooses Left.

Gets big reward.

Gradient update increases Left probability.

Next:

Left = 0.6.

Eventually:

Left dominates.

Learning happens gradually.

---

# 16. Python Conceptual Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):

    def __init__(self,state_dim,action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        return self.net(x)


policy = PolicyNet(4,2)
optimizer = optim.Adam(policy.parameters(),lr=1e-3)

def update(state,action,return_value):

    probs = policy(state)

    log_prob = torch.log(probs[action])

    loss = -log_prob * return_value

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Loss is negative because PyTorch minimizes loss while we maximize reward.

---

# 17. Limitations

High variance.

Needs many samples.

Sensitive learning rates.

Can converge slowly.

---

# 18. Modern Algorithms Built on Policy Gradient

PPO.

TRPO.

A2C.

SAC.

RLHF in large language models.

---

# 19. One Sentence Summary

Policy Gradient algorithms directly learn behavior by increasing the probability of actions that produced higher long-term rewards using gradient ascent on expected return.

---

